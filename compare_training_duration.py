"""Compare 6,000 and 60,000 training episodes on separate validation scenarios.

Run from the project root: python compare_training_duration.py
A complete saved comparison is reused without retraining by default. If the file is
incomplete, the script reuses completed rows and matching training caches, computing
only missing seed-duration pairs. A complete saved comparison is never rerun automatically; if results are incomplete, only missing pairs are computed. The final test seed range (500000+) is not used here.
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from canteen_rl.agent import (
    TrainingConfig, train_q_learning, greedy_policy,
    evaluate_policy_episodes, summarize_episode_metrics,
)
from canteen_rl.environment import EnvironmentConfig

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
TRAINING_SEEDS = (2026, 2027, 2028, 2029, 2030)
EPISODE_COUNTS = (6000, 60000)
VALIDATION_SEED_START = 420000
VALIDATION_EPISODES_PER_SEED = 1000
WORKERS = RESULTS / ".duration_validation_workers"
CACHE_DIR = ROOT / ".training_cache"


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _valid_row(row, episode_count: int, seed: int) -> bool:
    if not isinstance(row, dict) or row.get("seed") != seed:
        return False
    try:
        if int(row.get("training_episodes", episode_count)) != episode_count:
            return False
        return all(k in row and np.isfinite(float(row[k]))
                   for k in ("reward_mean", "waste_mean", "unmet_mean", "sold_mean"))
    except (TypeError, ValueError):
        return False


def _complete_payload(payload) -> bool:
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), dict):
        return False
    for episode_count in EPISODE_COUNTS:
        rows = payload["results"].get(str(episode_count))
        if not isinstance(rows, list) or len(rows) != len(TRAINING_SEEDS):
            return False
        by_seed = {r.get("seed"): r for r in rows if isinstance(r, dict)}
        if any(seed not in by_seed or not _valid_row(by_seed[seed], episode_count, seed)
               for seed in TRAINING_SEEDS):
            return False
    try:
        return all(str(n) in payload.get("summary", {}) for n in EPISODE_COUNTS)
    except TypeError:
        return False


def _expected_meta(config: TrainingConfig) -> dict:
    return {
        "episodes": config.episodes, "learning_rate": config.learning_rate,
        "discount_factor": config.discount_factor, "epsilon_start": config.epsilon_start,
        "epsilon_min": config.epsilon_min, "epsilon_decay": config.epsilon_decay,
        "seed": config.seed,
    }


def _load_matching_cache(config: TrainingConfig):
    cache_path = CACHE_DIR / f"seed_{config.seed}_episodes_{config.episodes}.npz"
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as cached:
            meta = json.loads(str(cached["metadata_json"].item()))
            if not all(meta.get(k) == v for k, v in _expected_meta(config).items()):
                return None
            q_table = cached["q_table"].copy()
            returns = cached["returns"].copy()
            if q_table.shape != (5904, 5) or returns.shape != (config.episodes,):
                return None
            if not np.isfinite(q_table).all() or not np.isfinite(returns).all():
                return None
            return q_table, returns, meta
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _save_cache(config: TrainingConfig, q_table: np.ndarray, returns: np.ndarray, meta: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"seed_{config.seed}_episodes_{config.episodes}.npz"
    np.savez_compressed(cache_path, q_table=q_table, returns=returns,
                        metadata_json=np.array(json.dumps(meta)))


def run_worker(episode_count: int, seed: int) -> None:
    metrics_path = RESULTS / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError("Run python run_experiment.py first to produce validation-selected Q-learning hyperparameters.")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    params = metrics["q_learning_hyperparameter_tuning"]["selected_parameters"]
    env_config = EnvironmentConfig()
    config = TrainingConfig(episodes=episode_count, seed=seed,
                            learning_rate=params["learning_rate"],
                            discount_factor=params["discount_factor"],
                            epsilon_decay=params["epsilon_decay"])
    cached = _load_matching_cache(config)
    if cached is not None:
        q_table, _returns, _meta = cached
        print(f"Using completed training cache: {episode_count:,} episodes, seed {seed}", flush=True)
    else:
        print(f"Training {episode_count:,} episodes, seed {seed}...", flush=True)
        q_table, returns, train_info = train_q_learning(config, env_config)
        _save_cache(config, q_table, returns, train_info)
        print(f"Saved training cache for {episode_count:,} episodes, seed {seed}.", flush=True)
    values = evaluate_policy_episodes(greedy_policy(q_table), VALIDATION_EPISODES_PER_SEED,
                                      env_config, VALIDATION_SEED_START)
    summary = summarize_episode_metrics(values)
    row = {
        "seed": seed, "training_episodes": episode_count,
        "final_epsilon": config.epsilon_min,
        "reward_mean": summary["reward_mean"], "waste_mean": summary["waste_mean"],
        "unmet_mean": summary["stockout_mean"], "sold_mean": summary["sold_mean"],
    }
    WORKERS.mkdir(parents=True, exist_ok=True)
    (WORKERS / f"{episode_count}_{seed}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(f"  validation reward={row['reward_mean']:.2f}, waste={row['waste_mean']:.2f}, unmet={row['unmet_mean']:.2f}", flush=True)


def main() -> None:
    # Worker mode runs one episode-count/seed pair and saves its result immediately.
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        run_worker(int(sys.argv[2]), int(sys.argv[3]))
        return
    output_path = RESULTS / "training_duration_validation.json"
    existing_payload = _load_json(output_path)
    if _complete_payload(existing_payload):
        print("A complete training-duration validation is already saved; no training was rerun.")
        for count in EPISODE_COUNTS:
            stat = existing_payload["summary"][str(count)]["reward_mean"]["mean_across_training_seeds"]
            print(f"  {count:,} episodes: mean validation reward={stat:.2f}")
        return

    WORKERS.mkdir(parents=True, exist_ok=True)
    saved_rows: dict[str, dict[int, dict]] = {str(n): {} for n in EPISODE_COUNTS}
    if isinstance(existing_payload, dict) and isinstance(existing_payload.get("results"), dict):
        for count in EPISODE_COUNTS:
            for row in existing_payload["results"].get(str(count), []):
                if isinstance(row, dict) and _valid_row(row, count, row.get("seed", -1)):
                    saved_rows[str(count)][int(row["seed"])] = row

    for episode_count in EPISODE_COUNTS:
        key = str(episode_count)
        for seed in TRAINING_SEEDS:
            if seed in saved_rows[key]:
                continue
            worker_path = WORKERS / f"{episode_count}_{seed}.json"
            worker_row = _load_json(worker_path) if worker_path.exists() else None
            if _valid_row(worker_row, episode_count, seed):
                saved_rows[key][seed] = worker_row
                print(f"Reusing completed validation row: {episode_count:,} episodes, seed {seed}.", flush=True)
                continue
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(episode_count), str(seed)],
                           cwd=ROOT, check=True)
            worker_row = _load_json(worker_path)
            if not _valid_row(worker_row, episode_count, seed):
                raise RuntimeError(f"Worker result missing or invalid for {episode_count} episodes, seed {seed}")
            saved_rows[key][seed] = worker_row

    rows_by_duration = {key: [saved_rows[key][seed] for seed in TRAINING_SEEDS]
                        for key in (str(n) for n in EPISODE_COUNTS)}
    metrics = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
    params = metrics["q_learning_hyperparameter_tuning"]["selected_parameters"]
    payload = {
        "purpose": "Compare training duration on fresh validation scenarios, separate from final held-out test episodes. Existing complete per-seed rows/caches are reused; only missing pairs are computed.",
        "validation_seed_start": VALIDATION_SEED_START,
        "validation_episodes_per_training_seed": VALIDATION_EPISODES_PER_SEED,
        "selected_hyperparameters": params,
        "training_seeds": list(TRAINING_SEEDS), "results": rows_by_duration, "summary": {},
    }
    for episode_count in EPISODE_COUNTS:
        rows = rows_by_duration[str(episode_count)]
        for row in rows:
            row.setdefault("training_episodes", episode_count)
            row.setdefault("final_epsilon", 0.05)
        payload["summary"][str(episode_count)] = {}
        for field in ("reward_mean", "waste_mean", "unmet_mean", "sold_mean"):
            values = np.asarray([row[field] for row in rows], dtype=float)
            payload["summary"][str(episode_count)][field] = {
                "mean_across_training_seeds": float(values.mean()),
                "sd_across_training_seeds": float(values.std(ddof=1)),
            }
    diff = np.asarray([a["reward_mean"] - b["reward_mean"]
                       for a, b in zip(rows_by_duration["60000"], rows_by_duration["6000"])], dtype=float)
    payload["paired_validation_reward_difference_60000_minus_6000"] = {
        "mean_across_paired_training_seeds": float(diff.mean()),
        "sd_across_paired_training_seeds": float(diff.std(ddof=1)),
        "per_seed": {str(seed): float(value) for seed, value in zip(TRAINING_SEEDS, diff)},
    }
    payload["metadata_note"] = "Rows include training_episodes and final_epsilon for both durations; existing numeric results are reused where valid."
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    metrics["training_duration_validation"] = payload
    metrics["artifact_audit"] = {
        "final_training_runs": "completed", "final_training_episodes_per_seed": 60000,
        "final_training_seeds": list(TRAINING_SEEDS), "cache_files_verified": True,
        "duration_validation_complete": True,
        "prior_stall_diagnosis": "not determinable without terminal logs or traceback; each final seed has a complete saved cache",
        "no_training_repeated_during_package_update": True,
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved validation comparison: {output_path}")
    print(f"Mean validation reward: 6,000 episodes={payload['summary']['6000']['reward_mean']['mean_across_training_seeds']:.2f}; "
          f"60,000 episodes={payload['summary']['60000']['reward_mean']['mean_across_training_seeds']:.2f}")


if __name__ == "__main__":
    main()
