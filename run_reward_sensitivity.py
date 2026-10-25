"""Train Q-learning agents under different waste penalties and compare them fairly.

All policies are evaluated in the *same nominal environment and reward function*
so the reported reward means are directly comparable. The experiment is a
sensitivity diagnostic, not a search for a post-hoc best policy.

Run from the project root: python run_reward_sensitivity.py
"""
from __future__ import annotations

import json
from pathlib import Path
from dataclasses import replace
import numpy as np
import matplotlib.pyplot as plt

from canteen_rl.agent import (
    TrainingConfig,
    train_q_learning,
    greedy_policy,
    evaluate_policy_episodes,
    summarize_episode_metrics,
)
from canteen_rl.environment import EnvironmentConfig

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# Multipliers are deliberately spread around the baseline coefficient 1.50.
WASTE_PENALTIES = (0.75, 1.50, 2.25)
TRAINING_SEEDS = (6201, 6202, 6203)
TRAINING_EPISODES = 20_000
EVALUATION_EPISODES = 2_000
EVALUATION_SEED_START = 700_000
METRICS_TO_REPORT = ("reward", "sold", "stockout", "waste", "prepared", "demand")


def mean_and_seed_sd(per_seed_summaries: list[dict[str, float]]) -> tuple[dict, dict]:
    """Aggregate evaluation episode means across independently trained agents."""
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for metric in METRICS_TO_REPORT:
        key = f"{metric}_mean"
        values = np.asarray([summary[key] for summary in per_seed_summaries], dtype=float)
        means[key] = float(values.mean())
        stds[key] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return means, stds


def main() -> None:
    evaluation_config = EnvironmentConfig()
    all_results = []

    print("Reward sensitivity experiment")
    print("Training reward varies by waste penalty; all test policies use the same original reward weights.")
    print(f"Training: {TRAINING_EPISODES} episodes x {len(TRAINING_SEEDS)} seeds x {len(WASTE_PENALTIES)} penalties")
    print(f"Evaluation: {EVALUATION_EPISODES} matched held-out days per trained agent")

    for penalty in WASTE_PENALTIES:
        training_env_config = replace(evaluation_config, waste_penalty=penalty)
        seed_summaries: dict[str, dict[str, float]] = {}
        for seed in TRAINING_SEEDS:
            q_table, _returns, _training_info = train_q_learning(
                TrainingConfig(episodes=TRAINING_EPISODES, seed=seed),
                training_env_config,
            )
            # Evaluate every learned policy with the *same fixed base reward*.
            evaluated = evaluate_policy_episodes(
                greedy_policy(q_table),
                episodes=EVALUATION_EPISODES,
                env_config=evaluation_config,
                seed_start=EVALUATION_SEED_START,
            )
            seed_summaries[str(seed)] = summarize_episode_metrics(evaluated)
            print(
                f"  waste penalty {penalty:.2f}, seed {seed}: "
                f"base-reward={seed_summaries[str(seed)]['reward_mean']:.2f}, "
                f"waste={seed_summaries[str(seed)]['waste_mean']:.2f}, "
                f"unmet={seed_summaries[str(seed)]['stockout_mean']:.2f}"
            )

        per_seed_values = list(seed_summaries.values())
        means, seed_sds = mean_and_seed_sd(per_seed_values)
        all_results.append({
            "training_waste_penalty": penalty,
            "training_seeds": list(TRAINING_SEEDS),
            "training_episodes_per_seed": TRAINING_EPISODES,
            "evaluation_episodes_per_seed": EVALUATION_EPISODES,
            "mean_metrics_under_common_base_reward": means,
            "sd_of_seed_level_metrics": seed_sds,
            "per_seed_metrics": seed_summaries,
        })

    output = {
        "purpose": "Sensitivity diagnostic: measure how the waste penalty used during training changes the learned policy's operational behavior.",
        "important_comparison_rule": "All learned policies are evaluated on the same held-out demand scenarios using the original EnvironmentConfig reward weights (waste_penalty=1.50). Thus base-reward values are comparable across rows. Training penalties are not selected using these test outcomes.",
        "base_evaluation_reward_weights": {
            "sold_reward": evaluation_config.sold_reward,
            "preparation_cost": evaluation_config.preparation_cost,
            "stockout_penalty": evaluation_config.stockout_penalty,
            "waste_penalty": evaluation_config.waste_penalty,
        },
        "forecast_model": {
            "accuracy_assumption": evaluation_config.forecast_accuracy,
            "true_regime_hidden_from_policy": True,
            "wrong_forecast_rule": "If wrong, one of the other outlook categories is selected uniformly.",
        },
        "training_waste_penalties": list(WASTE_PENALTIES),
        "training_seeds": list(TRAINING_SEEDS),
        "training_episodes_per_seed": TRAINING_EPISODES,
        "evaluation_episodes_per_seed": EVALUATION_EPISODES,
        "evaluation_seed_start": EVALUATION_SEED_START,
        "uncertainty_note": "SD is calculated across three independently trained policies, not across individual test days; this is a small exploratory sensitivity study, not a formal claim of statistical superiority.",
        "results": all_results,
    }
    output_path = RESULTS / "reward_sensitivity.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    x = np.arange(len(all_results))
    labels = [f"{row['training_waste_penalty']:.2f}" for row in all_results]
    metrics = [
        ("reward_mean", "Mean reward under base weights", "Mean reward/day"),
        ("waste_mean", "Discarded food", "Wasted portions/day"),
        ("stockout_mean", "Unmet demand", "Unmet portions/day"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    for ax, (metric, title, ylabel) in zip(axes, metrics):
        means = [row["mean_metrics_under_common_base_reward"][metric] for row in all_results]
        errors = [row["sd_of_seed_level_metrics"][metric] for row in all_results]
        ax.errorbar(x, means, yerr=errors, fmt="o-", capsize=3, linewidth=1.4)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Waste penalty used in training")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xticks(x, labels)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Reward sensitivity: three trained seeds per setting (error bars = seed-level SD)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(RESULTS / "reward_sensitivity.png", dpi=220)
    plt.close(fig)

    print(f"\nSaved: {output_path}")
    print(f"Saved: {RESULTS / 'reward_sensitivity.png'}")
    print("Remember: all reported test rewards use the same base reward weights, so policies can be compared on the same scale.")


if __name__ == "__main__":
    main()
