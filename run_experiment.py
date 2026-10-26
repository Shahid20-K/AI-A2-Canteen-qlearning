"""Train, validate, evaluate, and plot the campus-canteen Q-learning project.

Run from the project root: python run_experiment.py
All demand values are synthetic assumptions, not actual canteen records.
"""
from __future__ import annotations

import csv
from dataclasses import replace
from functools import partial
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from canteen_rl.agent import (
    TrainingConfig, train_q_learning, greedy_policy,
    fixed_batch_policy, time_of_day_policy,
    evaluate_policy_episodes, summarize_episode_metrics,
)
from canteen_rl.environment import CanteenEnv, EnvironmentConfig

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

TRAINING_SEEDS = [2026, 2027, 2028, 2029, 2030]
TRAINING_EPISODES = 6000
VALIDATION_EPISODES = 400
VALIDATION_SEED_START = 300_000
Q_TUNING_SEEDS = (8101, 8102)
Q_TUNING_EPISODES = 4000
Q_TUNING_VALIDATION_EPISODES = 600
Q_TUNING_VALIDATION_SEED_START = 360_000
EVALUATION_EPISODES = 2000
EVALUATION_SEED_START = 500_000
TARGET_FRACTIONS = (0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20)
HISTORY_ADJUSTMENTS = (0.00, 0.10, 0.20)
Q_LEARNING_RATES = (0.08, 0.18)
Q_EPSILON_DECAYS = (0.9992, 0.9996)


def paired_difference_summary(agent_values: np.ndarray, baseline_values: np.ndarray) -> dict:
    """Approximate 95% CI for paired per-episode agent-minus-baseline differences."""
    agent_values = np.asarray(agent_values, dtype=float)
    baseline_values = np.asarray(baseline_values, dtype=float)
    if agent_values.shape != baseline_values.shape or agent_values.ndim != 1 or len(agent_values) == 0:
        raise ValueError("paired metrics must be non-empty 1D arrays of equal length")
    if not np.isfinite(agent_values).all() or not np.isfinite(baseline_values).all():
        raise ValueError("paired metrics must contain finite values")
    diffs = agent_values - baseline_values
    mean = float(diffs.mean())
    std = float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0
    margin = 1.96 * std / np.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
    return {
        "mean_difference": mean,
        "std_difference": std,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
        "episodes": int(len(diffs)),
    }


def scaled_demand_config(base: EnvironmentConfig, scale: float) -> EnvironmentConfig:
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("demand scale must be positive and finite")
    means = tuple(max(1, int(round(value * scale))) for value in base.demand_means)
    return replace(base, demand_means=means)


def tune_forecast_baseline(env_config: EnvironmentConfig) -> tuple[float, float, list[dict]]:
    """Tune target fraction and sales-history adjustment on validation days only."""
    records = []
    for fraction in TARGET_FRACTIONS:
        for adjustment in HISTORY_ADJUSTMENTS:
            policy = partial(time_of_day_policy, target_fraction=fraction,
                             history_adjustment=adjustment,
                             planning_demand_means=env_config.demand_means)
            metrics = summarize_episode_metrics(evaluate_policy_episodes(
                policy, VALIDATION_EPISODES, env_config, VALIDATION_SEED_START
            ))
            records.append({
                "target_fraction": fraction,
                "history_adjustment": adjustment,
                "reward_mean": metrics["reward_mean"],
                "reward_ci95_low": metrics["reward_ci95_low"],
                "reward_ci95_high": metrics["reward_ci95_high"],
                "waste_mean": metrics["waste_mean"],
                "stockout_mean": metrics["stockout_mean"],
            })
    # Stable tie-break: reward, then smaller history adjustment, then smaller target.
    best = max(records, key=lambda row: (row["reward_mean"], -row["history_adjustment"], -row["target_fraction"]))
    return float(best["target_fraction"]), float(best["history_adjustment"]), records


def tune_q_learning_hyperparameters(env_config: EnvironmentConfig) -> tuple[dict, list[dict]]:
    """Choose learning rate and epsilon decay on independent validation episodes.

    Every candidate receives the same validation scenarios. A small two-seed tuning
    sweep limits computation while reducing dependence on one training initialization.
    Final test episodes are reserved for the selected configuration only.
    """
    records = []
    for learning_rate in Q_LEARNING_RATES:
        for epsilon_decay in Q_EPSILON_DECAYS:
            seed_means = {}
            for seed in Q_TUNING_SEEDS:
                config = TrainingConfig(
                    episodes=Q_TUNING_EPISODES,
                    seed=seed,
                    learning_rate=learning_rate,
                    discount_factor=0.92,
                    epsilon_decay=epsilon_decay,
                )
                q_table, _returns, _info = train_q_learning(config, env_config)
                evaluated = evaluate_policy_episodes(
                    greedy_policy(q_table), Q_TUNING_VALIDATION_EPISODES,
                    env_config, Q_TUNING_VALIDATION_SEED_START,
                )
                seed_means[str(seed)] = summarize_episode_metrics(evaluated)["reward_mean"]
            records.append({
                "learning_rate": learning_rate,
                "discount_factor": 0.92,
                "epsilon_decay": epsilon_decay,
                "training_episodes_per_seed": Q_TUNING_EPISODES,
                "training_seeds": list(Q_TUNING_SEEDS),
                "validation_reward_by_seed": seed_means,
                "validation_reward_mean": float(np.mean(list(seed_means.values()))),
                "validation_reward_sd_across_tuning_seeds": float(np.std(list(seed_means.values()), ddof=1)),
            })
    best = max(records, key=lambda row: row["validation_reward_mean"])
    selected = {
        "learning_rate": float(best["learning_rate"]),
        "discount_factor": float(best["discount_factor"]),
        "epsilon_decay": float(best["epsilon_decay"]),
    }
    return selected, records


def write_sample_trace(q_table: np.ndarray, baseline_policy, env_config: EnvironmentConfig,
                       seed: int = 1_000_026) -> None:
    """Save a readable one-day side-by-side trace using the same demand draws."""
    q_env = CanteenEnv(env_config)
    baseline_env = CanteenEnv(env_config)
    q_state, q_reset_info = q_env.reset(seed=seed)
    b_state, b_reset_info = baseline_env.reset(seed=seed)
    if q_reset_info["demand_outlook_index"] != b_reset_info["demand_outlook_index"]:
        raise RuntimeError("Matched trace did not produce the same demand outlook")

    q_policy = greedy_policy(q_table)
    fieldnames = [
        "time_slot", "demand_outlook", "sales_pace_before", "prior_stockout_before", "demand", "q_start_inventory",
        "q_requested_batch", "q_actual_prepared", "q_sold", "q_unmet_demand",
        "q_waste", "q_reward", "baseline_start_inventory", "baseline_requested_batch",
        "baseline_actual_prepared", "baseline_sold", "baseline_unmet_demand",
        "baseline_waste", "baseline_reward",
    ]
    rows = []
    done = False
    while not done:
        q_start = q_env.inventory
        b_start = baseline_env.inventory
        q_action = q_policy(q_state, q_env)
        b_action = baseline_policy(b_state, baseline_env)
        q_next, q_reward, q_done, q_info = q_env.step(q_action)
        b_next, b_reward, b_done, b_info = baseline_env.step(b_action)
        if q_info["demand"] != b_info["demand"]:
            raise RuntimeError("Matched sample trace produced different demands across policies")
        rows.append({
            "time_slot": q_info["time_slot"],
            "demand_outlook": q_reset_info["demand_outlook"],
            "sales_pace_before": q_info["sales_pace_category"],
            "prior_stockout_before": q_info["prior_stockout_occurred"],
            "demand": q_info["demand"],
            "q_start_inventory": q_start,
            "q_requested_batch": env_config.actions[q_action],
            "q_actual_prepared": q_info["prepared"],
            "q_sold": q_info["sold"],
            "q_unmet_demand": q_info["stockout"],
            "q_waste": q_info["waste"],
            "q_reward": round(q_reward, 4),
            "baseline_start_inventory": b_start,
            "baseline_requested_batch": env_config.actions[b_action],
            "baseline_actual_prepared": b_info["prepared"],
            "baseline_sold": b_info["sold"],
            "baseline_unmet_demand": b_info["stockout"],
            "baseline_waste": b_info["waste"],
            "baseline_reward": round(b_reward, 4),
        })
        q_state, b_state, done = q_next, b_next, q_done
        if q_done != b_done:
            raise RuntimeError("Matched trace policies ended at different time steps")

    with (RESULTS / "sample_episode_trace.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_policy_heatmaps(q_table: np.ndarray, env_config: EnvironmentConfig) -> None:
    """Show policy slices across sales pace and forecast (no prior stockout slice)."""
    env = CanteenEnv(env_config)
    inventory_values = np.arange(env_config.max_inventory + 1)
    time_values = np.arange(env.horizon)
    fig, axes = plt.subplots(3, 3, figsize=(10.8, 8.2), sharex=True, sharey=True, constrained_layout=True)
    outlook_labels = ("Low forecast", "Normal forecast", "High forecast")
    pace_labels = ("Sales below expected", "Sales on track", "Sales above expected")
    for pace in range(env.sales_pace_levels):
        for outlook in range(env.outlook_levels):
            ax = axes[pace, outlook]
            action_sizes = np.zeros((env.horizon, env_config.max_inventory + 1), dtype=int)
            for t in time_values:
                for inventory in inventory_values:
                    state = env.encode_state(int(t), int(inventory), outlook, pace, False)
                    action_sizes[t, inventory] = env_config.actions[int(np.argmax(q_table[state]))]
            img = ax.imshow(action_sizes.T, origin="lower", aspect="auto", interpolation="nearest",
                            extent=(-0.5, env.horizon - 0.5, -0.5, env_config.max_inventory + 0.5),
                            vmin=min(env_config.actions), vmax=max(env_config.actions))
            ax.set_title(f"{outlook_labels[outlook]}\n{pace_labels[pace]}", fontsize=7.5)
            ax.set_xticks(time_values)
            ax.grid(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("Time slot")
    for ax in axes[:, 0]:
        ax.set_ylabel("Inventory (portions)")
    fig.colorbar(img, ax=axes.ravel().tolist(), shrink=0.78, label="Chosen preparation batch (portions)")
    fig.suptitle("Policy slices by forecast and sales pace (prior-stockout flag = false)", fontsize=10)
    fig.savefig(RESULTS / "policy_heatmap.png", dpi=220)
    plt.close(fig)


def main():
    env_config = EnvironmentConfig()

    # Tune the static baseline and Q-learning hyperparameters on validation scenarios.
    print("[1/6] Tuning history-aware baseline on validation episodes...", flush=True)
    tuned_fraction, tuned_history_adjustment, baseline_tuning_records = tune_forecast_baseline(env_config)
    print("[2/6] Tuning Q-learning hyperparameters on validation episodes...", flush=True)
    tuned_q_config, q_tuning_records = tune_q_learning_hyperparameters(env_config)
    baseline_policies = {
        "Fixed 15-portion batch": fixed_batch_policy,
        "Forecast-aware history rule": partial(
            time_of_day_policy,
            target_fraction=tuned_fraction,
            history_adjustment=tuned_history_adjustment,
            planning_demand_means=env_config.demand_means,
        ),
    }

    baseline_episode_metrics = {
        name: evaluate_policy_episodes(
            policy, EVALUATION_EPISODES, env_config, EVALUATION_SEED_START
        )
        for name, policy in baseline_policies.items()
    }
    baseline_summaries = {
        name: summarize_episode_metrics(values)
        for name, values in baseline_episode_metrics.items()
    }
    print("[3/6] Training final Q-learning agents on five seeds...", flush=True)

    training_results = []
    q_episode_metrics_by_seed = {}
    returns_by_seed = []
    primary_q_table = None
    primary_returns = None
    primary_train_info = None
    primary_q_summary = None

    for seed in TRAINING_SEEDS:
        print(f"  training seed {seed}...", flush=True)
        train_cfg = TrainingConfig(episodes=TRAINING_EPISODES, seed=seed,
                                   learning_rate=tuned_q_config["learning_rate"],
                                   discount_factor=tuned_q_config["discount_factor"],
                                   epsilon_decay=tuned_q_config["epsilon_decay"])
        q_table, returns, training_info = train_q_learning(train_cfg, env_config)
        print(f"    training completed for seed {seed}; evaluating greedy policy...", flush=True)
        q_episode_metrics = evaluate_policy_episodes(
            greedy_policy(q_table), EVALUATION_EPISODES, env_config, EVALUATION_SEED_START
        )
        q_summary = summarize_episode_metrics(q_episode_metrics)
        print(f"    evaluation completed for seed {seed}: reward={q_summary['reward_mean']:.2f}", flush=True)
        q_episode_metrics_by_seed[str(seed)] = q_episode_metrics
        returns_by_seed.append(returns.copy())
        training_results.append({
            "seed": seed,
            "training": training_info,
            "test_metrics": q_summary,
        })
        if seed == TRAINING_SEEDS[0]:
            primary_q_table = q_table.copy()
            primary_returns = returns.copy()
            primary_train_info = training_info
            primary_q_summary = q_summary

    assert primary_q_table is not None and primary_returns is not None and primary_q_summary is not None
    returns_matrix = np.vstack(returns_by_seed)
    np.save(RESULTS / "q_table.npy", primary_q_table)
    np.save(RESULTS / "training_returns.npy", primary_returns)
    np.save(RESULTS / "training_returns_by_seed.npy", returns_matrix)

    primary_metrics = {"Q-learning (seed 2026)": primary_q_summary, **baseline_summaries}

    paired_results = {}
    for baseline_name, baseline_values in baseline_episode_metrics.items():
        metric_comparisons = {}
        for metric in ("reward", "sold", "stockout", "waste", "prepared"):
            per_seed = {
                seed: paired_difference_summary(
                    q_episode_metrics_by_seed[seed][metric], baseline_values[metric]
                )
                for seed in q_episode_metrics_by_seed
            }
            means = np.array([v["mean_difference"] for v in per_seed.values()], dtype=float)
            metric_comparisons[metric] = {
                "mean_difference_across_training_seeds": float(means.mean()),
                "std_of_seed_level_differences": float(means.std(ddof=1)),
                "per_training_seed": per_seed,
            }
        paired_results[baseline_name] = metric_comparisons

    # Generalization stress test: same trained Q-table, altered demand means.
    print("[4/6] Running demand-shift stress tests...", flush=True)
    stress_results = {}
    q_primary_policy = greedy_policy(primary_q_table)
    stress_scenarios = {"low_demand_0.75x": 0.75, "nominal_1.00x": 1.00, "high_demand_1.25x": 1.25}
    for scenario_name, factor in stress_scenarios.items():
        scenario_config = env_config if factor == 1.0 else scaled_demand_config(env_config, factor)
        scenario_policies = {"Q-learning (trained at nominal demand)": q_primary_policy, **baseline_policies}
        stress_results[scenario_name] = {
            name: summarize_episode_metrics(evaluate_policy_episodes(
                policy, 1000, scenario_config, EVALUATION_SEED_START + 100_000
            ))
            for name, policy in scenario_policies.items()
        }

    robust_metrics = {}
    for key in training_results[0]["test_metrics"]:
        values = np.array([row["test_metrics"][key] for row in training_results], dtype=float)
        robust_metrics[key] = {
            "mean_across_training_seeds": float(values.mean()),
            "std_across_training_seeds": float(values.std(ddof=1)),
            "min_across_training_seeds": float(values.min()),
            "max_across_training_seeds": float(values.max()),
        }

    results = {
        "project": "Adaptive Meal Preparation in a Campus Canteen using Q-Learning",
        "synthetic_simulation_notice": "Demand profiles, outlook probabilities, forecast accuracy, carryover fraction, and reward weights are illustrative assumptions, not measured campus data.",
        "state_representation": "time slot, exact inventory, noisy observed forecast, cumulative sales pace category (below/on-track/above expected), and whether any earlier slot had a stockout; true daily regime is hidden",
        "sales_history_features": {
            "sales_pace_categories": ["below_expected", "on_track", "above_expected"],
            "classification": "Cumulative sales in completed slots divided by cumulative expected demand under the Bayesian forecast estimate; below 0.85 is below expected, above 1.15 is above expected, otherwise on track. Before the first slot it is on track.",
            "prior_stockout_flag": "True if any earlier slot had unmet demand; this helps distinguish censored sales from a low-demand signal.",
            "limitation": "These are compact observation summaries, not a full Bayesian belief state; the process remains partially observable.",
        },
        "forecast_model": {
            "accuracy_assumption": env_config.forecast_accuracy,
            "wrong_forecast_rule": "If the forecast is wrong, one of the other outlook categories is selected uniformly.",
            "true_regime_hidden_from_policy": True,
            "baseline_demand_estimate": "Posterior expected demand multiplier given the observed forecast, using Bayes' rule.",
        },
        "state_count": CanteenEnv(env_config).n_states,
        "forecast_accuracy_assumption": env_config.forecast_accuracy,
        "forecast_observation_model": "True regime is sampled from the configured prior. The forecast is correct with the configured probability; on an error, one of the other categories is chosen uniformly.",
        "baseline_forecast_interpretation": "The rule-based baseline uses the Bayesian posterior forecast estimate and the same sales-pace/prior-stockout signals as Q-learning, with its target and correction tuned on a separate validation set.",
        "training": primary_train_info,
        "q_learning_hyperparameter_tuning": {
            "purpose": "Screen a four-candidate learning-rate/epsilon-decay grid on a two-seed validation sweep before final test evaluation.",
            "tuning_training_seeds": list(Q_TUNING_SEEDS),
            "training_episodes_per_candidate_seed": Q_TUNING_EPISODES,
            "validation_episodes_per_candidate_seed": Q_TUNING_VALIDATION_EPISODES,
            "validation_seed_start": Q_TUNING_VALIDATION_SEED_START,
            "candidate_learning_rates": list(Q_LEARNING_RATES),
            "candidate_epsilon_decays": list(Q_EPSILON_DECAYS),
            "selected_parameters": tuned_q_config,
            "candidate_results": q_tuning_records,
        },
        "robustness_training_seeds": TRAINING_SEEDS,
        "training_episodes_per_seed": TRAINING_EPISODES,
        "validation_tuning": {
            "purpose": "Select forecast-aware rule target and sales-history correction on validation scenarios separate from final test episodes.",
            "target_fractions_considered": list(TARGET_FRACTIONS),
            "history_adjustments_considered": list(HISTORY_ADJUSTMENTS),
            "validation_episodes_per_candidate": VALIDATION_EPISODES,
            "validation_seed_start": VALIDATION_SEED_START,
            "selected_target_fraction": tuned_fraction,
            "selected_history_adjustment": tuned_history_adjustment,
            "candidate_results": baseline_tuning_records,
        },
        "evaluation_episodes_per_policy": EVALUATION_EPISODES,
        "evaluation_seed_start": EVALUATION_SEED_START,
        "test_metrics": primary_metrics,
        "multi_seed_robustness": {
            "description": "Mean and SD across independently trained Q-learning agents; each was tested on the same held-out synthetic days.",
            "q_learning_metrics": robust_metrics,
            "per_training_seed": training_results,
        },
        "paired_comparisons": paired_results,
        "demand_shift_stress_test": {
            "description": "Seed-2026 Q-table and rule-based baselines evaluated on 1,000 episodes per scenario. Demand means scale in the environment, while both the Q-table and the tuned schedule's planning demand means remain nominal; this tests deployment shift without giving the baseline the changed demand means.",
            "results": stress_results,
        },
    }
    (RESULTS / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("[5/6] Saving metrics and generating figures...", flush=True)

    # Learning curve averaged across all five training seeds, with +/- 1 SD band.
    bin_size = 200
    usable = returns_matrix.shape[1] // bin_size * bin_size
    binned_by_seed = returns_matrix[:, :usable].reshape(len(TRAINING_SEEDS), -1, bin_size).mean(axis=2)
    x = np.arange(1, binned_by_seed.shape[1] + 1) * bin_size
    curve_mean = binned_by_seed.mean(axis=0)
    curve_std = binned_by_seed.std(axis=0, ddof=1)
    plt.figure(figsize=(7.2, 3.8))
    plt.plot(x, curve_mean, linewidth=1.7, label="Mean across 5 seeds")
    plt.fill_between(x, curve_mean - curve_std, curve_mean + curve_std, alpha=0.18,
                     label="±1 SD across seeds")
    plt.xlabel("Training episode")
    plt.ylabel("Mean return per simulated day")
    plt.title("Q-learning training curves (200-episode bins)")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS / "learning_curve.png", dpi=220)
    plt.close()

    # Operational comparison plot.
    plot_names = list(primary_metrics.keys())
    metric_specs = [
        ("reward_mean", "Average return per day"),
        ("waste_mean", "Wasted portions per day"),
        ("stockout_mean", "Unmet demand per day"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.6))
    short_labels = ["Q-learning\n(seed 2026)", "Fixed\nbatch", f"Tuned rule\n({tuned_fraction:.0%}, adj {tuned_history_adjustment:.0%})"]
    for ax, (metric, title) in zip(axes, metric_specs):
        values = [primary_metrics[name][metric] for name in plot_names]
        bars = ax.bar(range(len(plot_names)), values)
        ax.set_title(title, fontsize=9)
        if metric == "reward_mean" and min(values) < 0:
            ax.set_ylim(bottom=min(values) - 6.0)
        ax.set_xticks(range(len(plot_names)))
        ax.set_xticklabels(short_labels, fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, values):
            y_text = value + (0.4 if value >= 0 else -0.8)
            ax.text(bar.get_x() + bar.get_width() / 2, y_text, f"{value:.1f}",
                    ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
    fig.suptitle("Held-out evaluation: 2,000 matched synthetic days per policy", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(RESULTS / "policy_comparison.png", dpi=220)
    plt.close(fig)

    plot_policy_heatmaps(primary_q_table, env_config)
    write_sample_trace(primary_q_table, baseline_policies["Forecast-aware history rule"], env_config)

    print("[6/6] Figure generation complete.", flush=True)
    print("Experiment complete.")
    print(f"Q-table shape: {primary_q_table.shape}; state count: {CanteenEnv(env_config).n_states}")
    print(f"Training: {TRAINING_EPISODES} episodes x {len(TRAINING_SEEDS)} independent seeds")
    print(f"State count: {CanteenEnv(env_config).n_states} (includes sales pace and prior-stockout history)")
    print(f"Observed forecast accuracy assumption: {env_config.forecast_accuracy:.0%}")
    print(f"Q-learning validation tuning selected alpha={tuned_q_config['learning_rate']:.3f}, epsilon decay={tuned_q_config['epsilon_decay']:.4f}")
    print(f"Baseline validation tuning selected target={tuned_fraction:.0%}, history adjustment={tuned_history_adjustment:.0%}")
    print(f"Evaluation: {EVALUATION_EPISODES} matched held-out days per policy and training seed")
    print("\nSeed-2026 primary model and baselines:")
    for label, values in primary_metrics.items():
        print(f"{label}: reward={values['reward_mean']:.2f} "
              f"[95% CI {values['reward_ci95_low']:.2f}, {values['reward_ci95_high']:.2f}], "
              f"waste={values['waste_mean']:.2f}, unmet_demand={values['stockout_mean']:.2f}, "
              f"sold={values['sold_mean']:.2f}")
    print("\nQ-learning reward mean across training seeds: "
          f"{robust_metrics['reward_mean']['mean_across_training_seeds']:.2f} "
          f"+/- {robust_metrics['reward_mean']['std_across_training_seeds']:.2f} SD")
    print("\nPrimary-seed paired reward differences (Q-learning minus baseline):")
    for baseline_name, metrics_by_metric in paired_results.items():
        value = metrics_by_metric["reward"]["per_training_seed"][str(TRAINING_SEEDS[0])]
        print(f"vs {baseline_name}: mean={value['mean_difference']:.2f}; "
              f"95% CI [{value['ci95_low']:.2f}, {value['ci95_high']:.2f}]")
    print(f"Results saved in: {RESULTS}")


if __name__ == "__main__":
    main()
