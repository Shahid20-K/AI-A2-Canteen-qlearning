"""Tabular Q-learning and evaluation policies for the canteen simulation."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Dict
import numpy as np

from .environment import CanteenEnv, EnvironmentConfig


@dataclass
class TrainingConfig:
    episodes: int = 20000
    learning_rate: float = 0.12
    discount_factor: float = 0.92
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.9992
    seed: int = 2026

    def __post_init__(self) -> None:
        if isinstance(self.episodes, bool) or not isinstance(self.episodes, int) or self.episodes <= 0:
            raise ValueError("episodes must be a positive integer")
        if not math.isfinite(self.learning_rate) or not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        if not math.isfinite(self.discount_factor) or not 0.0 <= self.discount_factor <= 1.0:
            raise ValueError("discount_factor must be in [0, 1]")
        for name, value in (("epsilon_start", self.epsilon_start), ("epsilon_min", self.epsilon_min)):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.epsilon_min > self.epsilon_start:
            raise ValueError("epsilon_min cannot exceed epsilon_start")
        if not math.isfinite(self.epsilon_decay) or not 0.0 < self.epsilon_decay <= 1.0:
            raise ValueError("epsilon_decay must be in (0, 1]")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")


def choose_action(q_table: np.ndarray, state: int, epsilon: float,
                  rng: np.random.Generator) -> int:
    """Epsilon-greedy action selection with random tie-breaking."""
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")
    if q_table.ndim != 2 or not 0 <= state < q_table.shape[0]:
        raise ValueError("state is outside the Q-table")
    if rng.random() < epsilon:
        return int(rng.integers(q_table.shape[1]))
    best = np.flatnonzero(q_table[state] == np.max(q_table[state]))
    return int(rng.choice(best))


def train_q_learning(config: TrainingConfig | None = None,
                     env_config: EnvironmentConfig | None = None):
    cfg = config or TrainingConfig()
    env_cfg = env_config or EnvironmentConfig()
    env = CanteenEnv(env_cfg, seed=cfg.seed)
    rng = np.random.default_rng(cfg.seed + 1)
    q_table = np.zeros((env.n_states, env.n_actions), dtype=np.float64)
    episode_returns = np.empty(cfg.episodes, dtype=np.float64)
    epsilon = cfg.epsilon_start

    for episode in range(cfg.episodes):
        state, _ = env.reset(seed=cfg.seed + 1000 + episode)
        done = False
        total_reward = 0.0
        while not done:
            action = choose_action(q_table, state, epsilon, rng)
            next_state, reward, done, _ = env.step(action)
            # Correct finite-horizon Q-learning target: no bootstrap after terminal step.
            best_next = 0.0 if done else float(np.max(q_table[next_state]))
            target = reward + cfg.discount_factor * best_next
            q_table[state, action] += cfg.learning_rate * (target - q_table[state, action])
            state = next_state
            total_reward += reward
        episode_returns[episode] = total_reward
        epsilon = max(cfg.epsilon_min, epsilon * cfg.epsilon_decay)

    return q_table, episode_returns, {
        "episodes": cfg.episodes,
        "learning_rate": cfg.learning_rate,
        "discount_factor": cfg.discount_factor,
        "epsilon_start": cfg.epsilon_start,
        "epsilon_min": cfg.epsilon_min,
        "epsilon_decay": cfg.epsilon_decay,
        "final_epsilon": epsilon,
        "seed": cfg.seed,
    }


def greedy_policy(q_table: np.ndarray) -> Callable[[int, CanteenEnv], int]:
    if q_table.ndim != 2:
        raise ValueError("q_table must be a 2D array")

    def policy(state: int, env: CanteenEnv) -> int:
        if not 0 <= state < q_table.shape[0]:
            raise ValueError("state is outside the Q-table")
        # Deterministic tie-breaking makes evaluation fully reproducible.
        return int(np.argmax(q_table[state]))
    return policy


def fixed_batch_policy(state: int, env: CanteenEnv) -> int:
    """Baseline: request the same 15 additional portions in every time slot."""
    return int(np.argmin(np.abs(np.asarray(env.config.actions) - 15)))


def time_of_day_policy(state: int, env: CanteenEnv,
                       target_fraction: float = 0.80,
                       planning_demand_means: tuple[float, ...] | None = None) -> int:
    """Forecast-aware non-learning policy with a fixed target fraction.

    The target fraction should be selected on a separate validation set, not the
    final test scenarios. ``planning_demand_means`` lets a stress test hold the
    planning model fixed while the environment's actual demand distribution shifts.
    """
    if not math.isfinite(target_fraction) or target_fraction < 0.0:
        raise ValueError("target_fraction must be finite and non-negative")
    means = planning_demand_means if planning_demand_means is not None else env.config.demand_means
    if len(means) != env.horizon:
        raise ValueError("planning_demand_means must have one mean per time slot")
    # The observed category is noisy; use the posterior expected regime multiplier
    # rather than treating the forecast as the hidden true regime.
    nominal = means[env.time] * env.expected_regime_multiplier()
    target_total = int(round((target_fraction * nominal) / 5) * 5)
    desired_addition = max(0, target_total - env.inventory)
    distances = np.abs(np.asarray(env.config.actions) - desired_addition)
    return int(np.argmin(distances))


def evaluate_policy_episodes(policy: Callable[[int, CanteenEnv], int], episodes: int = 2000,
                             env_config: EnvironmentConfig | None = None,
                             seed_start: int = 500_000) -> Dict[str, np.ndarray]:
    """Return one row of operational metrics per episode for paired comparisons."""
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    if isinstance(seed_start, bool) or not isinstance(seed_start, int):
        raise ValueError("seed_start must be an integer")
    env_cfg = env_config or EnvironmentConfig()
    env = CanteenEnv(env_cfg)
    metric_names = ("reward", "sold", "prepared", "stockout", "waste", "demand")
    episodes_data = {key: np.zeros(episodes, dtype=np.float64) for key in metric_names}
    for ep in range(episodes):
        state, _ = env.reset(seed=seed_start + ep)
        done = False
        while not done:
            action = policy(state, env)
            state, reward, done, info = env.step(action)
            episodes_data["reward"][ep] += reward
            episodes_data["sold"][ep] += info["sold"]
            episodes_data["prepared"][ep] += info["prepared"]
            episodes_data["stockout"][ep] += info["stockout"]
            episodes_data["waste"][ep] += info["waste"]
            episodes_data["demand"][ep] += info["demand"]
    return episodes_data


def summarize_episode_metrics(episodes_data: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Summarize episode metrics with mean, SD, and an approximate 95% normal CI."""
    summary: Dict[str, float] = {}
    for key, raw_values in episodes_data.items():
        values = np.asarray(raw_values, dtype=np.float64)
        if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
            raise ValueError(f"{key} must contain a non-empty finite 1D array")
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        margin = 1.96 * std / np.sqrt(len(values)) if len(values) > 1 else 0.0
        summary[f"{key}_mean"] = mean
        summary[f"{key}_std"] = std
        summary[f"{key}_ci95_low"] = mean - margin
        summary[f"{key}_ci95_high"] = mean + margin
    return summary


def evaluate_policy(policy: Callable[[int, CanteenEnv], int], episodes: int = 2000,
                    env_config: EnvironmentConfig | None = None,
                    seed_start: int = 500_000) -> Dict[str, float]:
    """Compatibility wrapper returning summary metrics for a policy."""
    return summarize_episode_metrics(evaluate_policy_episodes(policy, episodes, env_config, seed_start))
