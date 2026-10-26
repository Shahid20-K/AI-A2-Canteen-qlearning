"""Custom simulated environment for perishable meal preparation in a campus canteen."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Dict, Tuple
import math
import numpy as np


@dataclass(frozen=True)
class EnvironmentConfig:
    """Illustrative simulation parameters; these are not measured campus data."""
    demand_means: Tuple[int, ...] = (8, 12, 22, 30, 24, 18, 27, 14)
    # Assumed allowable discrete prep batches: at most 20 additional portions per slot.
    actions: Tuple[int, ...] = (0, 5, 10, 15, 20)
    max_inventory: int = 40
    carryover_fraction: float = 0.30
    regime_multipliers: Tuple[float, ...] = (0.70, 1.00, 1.30)
    regime_probabilities: Tuple[float, ...] = (0.25, 0.50, 0.25)
    # The displayed outlook is a noisy category forecast, not the true regime.
    forecast_accuracy: float = 0.70
    # Sales pace is categorized against expected cumulative demand using these ratio cutoffs.
    sales_pace_below_threshold: float = 0.85
    sales_pace_above_threshold: float = 1.15
    sold_reward: float = 1.0
    preparation_cost: float = 0.35
    stockout_penalty: float = 1.0
    waste_penalty: float = 1.50


class CanteenEnv:
    """Finite-horizon simulation with integer inventory and stochastic demand.

    Observation/state: (time slot, exact inventory, observed daily demand-outlook
    category, cumulative sales pace, whether a stockout occurred earlier in the day).
    The actual daily regime is hidden. The forecast category matches it with the
    configured probability (70% by default); when wrong, another category is
    selected uniformly. Individual slot demand remains stochastic. Sales pace is
    computed only from completed slots and is categorized relative to expected
    cumulative demand under the posterior forecast. A prior-stockout flag helps
    distinguish weak sales caused by inventory shortages from a low-demand signal.
    The problem remains partially observable; this feature is a compact history
    summary, not a full Bayesian belief state.

    Action: index into ``config.actions``, representing additional portions to
    prepare. The discrete action set (0, 5, 10, 15, 20) is an explicit assumed
    per-slot batch-size limit, not a measured kitchen capacity. Total inventory
    has a separate capacity limit. Of unsold portions, a configured fraction (rounded down) carries
    into the next slot; all remaining unsold portions are discarded. Nothing
    carries past the final slot. These are pedagogical assumptions, not a
    production-ready model.
    """

    def __init__(self, config: EnvironmentConfig | None = None, seed: int | None = None):
        self.config = config or EnvironmentConfig()
        self._validate_config(self.config)

        self.rng = np.random.default_rng(seed)
        self.horizon = len(self.config.demand_means)
        self.n_actions = len(self.config.actions)
        self.outlook_levels = len(self.config.regime_multipliers)
        # Extra state factors: three sales-pace categories and a prior-stockout flag.
        self.sales_pace_levels = 3
        self.stockout_history_levels = 2
        self.n_states = (self.horizon * (self.config.max_inventory + 1)
                         * self.outlook_levels * self.sales_pace_levels * self.stockout_history_levels)
        self.time = 0
        self.inventory = 0
        self.demand_outlook_index = int(np.argmax(self.config.regime_probabilities))
        self._actual_regime_index = self.demand_outlook_index
        self._actual_regime_multiplier = self.config.regime_multipliers[self._actual_regime_index]
        self._forecast_expected_multiplier = 1.0
        self.cumulative_sold = 0
        self.cumulative_expected_demand = 0.0
        self.prior_stockout_occurred = False

    @staticmethod
    def _validate_config(config: EnvironmentConfig) -> None:
        if not config.demand_means:
            raise ValueError("demand_means must contain at least one time slot")
        if any(not math.isfinite(float(x)) or float(x) < 0 for x in config.demand_means):
            raise ValueError("demand_means must contain finite, non-negative values")
        if not config.actions:
            raise ValueError("actions must contain at least one preparation quantity")
        if any(isinstance(x, bool) or not isinstance(x, Integral) or x < 0 for x in config.actions):
            raise ValueError("actions must be non-negative integers")
        if len(set(config.actions)) != len(config.actions):
            raise ValueError("actions must be unique")
        if 0 not in config.actions:
            raise ValueError("actions must include 0 so the agent can choose not to prepare")
        if isinstance(config.max_inventory, bool) or not isinstance(config.max_inventory, Integral) or config.max_inventory < 0:
            raise ValueError("max_inventory must be a non-negative integer")
        if not math.isfinite(float(config.carryover_fraction)) or not 0.0 <= config.carryover_fraction <= 1.0:
            raise ValueError("carryover_fraction must be finite and in [0, 1]")
        if not config.regime_multipliers:
            raise ValueError("At least one demand regime is required")
        if len(config.regime_multipliers) != len(config.regime_probabilities):
            raise ValueError("regime_multipliers and regime_probabilities must have equal length")
        if any(not math.isfinite(float(x)) or float(x) < 0 for x in config.regime_multipliers):
            raise ValueError("regime_multipliers must be finite and non-negative")
        if any(not math.isfinite(float(p)) or float(p) < 0 for p in config.regime_probabilities):
            raise ValueError("regime_probabilities must be finite and non-negative")
        if not math.isfinite(float(config.forecast_accuracy)) or not 0.0 <= config.forecast_accuracy <= 1.0:
            raise ValueError("forecast_accuracy must be finite and in [0, 1]")
        if (not math.isfinite(float(config.sales_pace_below_threshold))
                or not math.isfinite(float(config.sales_pace_above_threshold))
                or not 0.0 < config.sales_pace_below_threshold < config.sales_pace_above_threshold):
            raise ValueError("sales pace thresholds must be finite, positive, and strictly increasing")
        if not np.isclose(sum(config.regime_probabilities), 1.0):
            raise ValueError("regime_probabilities must be non-negative and sum to 1")
        if not any(p > 0 for p in config.regime_probabilities):
            raise ValueError("at least one regime probability must be positive")
        reward_weights = (config.sold_reward, config.preparation_cost,
                          config.stockout_penalty, config.waste_penalty)
        if any(not math.isfinite(float(w)) or float(w) < 0 for w in reward_weights):
            raise ValueError("reward coefficients must be finite and non-negative")

    def reset(self, seed: int | None = None) -> Tuple[int, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.time = 0
        self.inventory = 0
        self.cumulative_sold = 0
        self.cumulative_expected_demand = 0.0
        self.prior_stockout_occurred = False
        # Sample the hidden true regime first. The agent receives only a noisy
        # category forecast; the actual regime is kept private to the dynamics.
        self._actual_regime_index = int(self.rng.choice(
            self.outlook_levels, p=self.config.regime_probabilities
        ))
        self._actual_regime_multiplier = self.config.regime_multipliers[self._actual_regime_index]
        if self.outlook_levels == 1 or self.rng.random() < self.config.forecast_accuracy:
            self.demand_outlook_index = self._actual_regime_index
        else:
            alternatives = [i for i in range(self.outlook_levels) if i != self._actual_regime_index]
            self.demand_outlook_index = int(self.rng.choice(alternatives))

        self._forecast_expected_multiplier = self.expected_regime_multiplier(self.demand_outlook_index)
        labels = ("low", "normal", "high")
        label = labels[self.demand_outlook_index] if self.outlook_levels == 3 else str(self.demand_outlook_index)
        return self._state_index(), {
            "demand_outlook": label,
            "demand_outlook_index": self.demand_outlook_index,
            "forecast_accuracy_assumption": self.config.forecast_accuracy,
            "sales_pace_category": self.sales_pace_label(self.sales_pace_category_index()),
            "prior_stockout_occurred": False,
        }

    @staticmethod
    def sales_pace_label(index: int) -> str:
        """Human-readable label for below/on-track/above expected cumulative sales."""
        labels = ("below_expected", "on_track", "above_expected")
        if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < len(labels):
            raise ValueError("sales pace index must be 0, 1, or 2")
        return labels[int(index)]

    def sales_pace_category_index(self) -> int:
        """Categorize prior sales against expected cumulative demand.

        Only completed slots contribute. When a prior stockout occurred, a separate
        state flag lets the policy avoid interpreting low sales as pure low demand.
        """
        if self.cumulative_expected_demand <= 0.0:
            return 1  # At the beginning of the day there is no sales history.
        ratio = self.cumulative_sold / self.cumulative_expected_demand
        if ratio < self.config.sales_pace_below_threshold:
            return 0
        if ratio > self.config.sales_pace_above_threshold:
            return 2
        return 1

    def encode_state(self, time_slot: int, inventory: int, outlook_index: int,
                     sales_pace_index: int, prior_stockout: bool) -> int:
        """Encode all five observable state features as one integer state ID."""
        integer_values = (time_slot, inventory, outlook_index, sales_pace_index)
        if any(isinstance(v, bool) or not isinstance(v, Integral) for v in integer_values):
            raise ValueError("time, inventory, outlook, and sales pace must be integers")
        if not 0 <= time_slot < self.horizon:
            raise ValueError("time_slot is outside the episode horizon")
        if not 0 <= inventory <= self.config.max_inventory:
            raise ValueError("inventory is outside configured bounds")
        if not 0 <= outlook_index < self.outlook_levels:
            raise ValueError("outlook_index is outside configured bounds")
        if not 0 <= sales_pace_index < self.sales_pace_levels:
            raise ValueError("sales_pace_index must be 0, 1, or 2")
        if not isinstance(prior_stockout, (bool, np.bool_)):
            raise ValueError("prior_stockout must be boolean")
        packed = ((int(time_slot) * (self.config.max_inventory + 1) + int(inventory))
                  * self.outlook_levels + int(outlook_index))
        packed = packed * self.sales_pace_levels + int(sales_pace_index)
        return packed * self.stockout_history_levels + int(bool(prior_stockout))

    def _state_index(self) -> int:
        """Encode current observation and compact sales history as one integer."""
        if not 0 <= self.time < self.horizon:
            raise RuntimeError("No state is defined outside an active episode")
        return self.encode_state(self.time, self.inventory, self.demand_outlook_index,
                                 self.sales_pace_category_index(), self.prior_stockout_occurred)

    def expected_regime_multiplier(self, observed_outlook_index: int | None = None) -> float:
        """Return E[regime multiplier | noisy forecast] using Bayes' rule.

        This is used by the rule-based baseline. It sees the same forecast as the
        RL agent but does not get access to the hidden true regime.
        """
        observed = self.demand_outlook_index if observed_outlook_index is None else observed_outlook_index
        if isinstance(observed, bool) or not isinstance(observed, Integral):
            raise ValueError("observed_outlook_index must be an integer")
        if not 0 <= observed < self.outlook_levels:
            raise ValueError(f"observed_outlook_index must be in [0, {self.outlook_levels - 1}]")
        if self.outlook_levels == 1:
            return float(self.config.regime_multipliers[0])

        accuracy = self.config.forecast_accuracy
        likelihood = np.full(self.outlook_levels, (1.0 - accuracy) / (self.outlook_levels - 1))
        likelihood[int(observed)] = accuracy
        unnormalized = np.asarray(self.config.regime_probabilities, dtype=float) * likelihood
        total = float(unnormalized.sum())
        if total <= 0.0:
            # This can only occur for an observation impossible under the configured
            # prior/confusion model (e.g., manually forcing a state in a unit test).
            return float(self.config.regime_multipliers[int(observed)])
        posterior = unnormalized / total
        return float(np.dot(posterior, np.asarray(self.config.regime_multipliers, dtype=float)))

    def decode_state(self, state_index: int) -> tuple[int, int, int, int, bool]:
        """Return (time, inventory, outlook, sales-pace category, prior-stockout flag)."""
        if isinstance(state_index, bool) or not isinstance(state_index, Integral):
            raise ValueError("state_index must be an integer")
        if not 0 <= state_index < self.n_states:
            raise ValueError(f"state_index must be in [0, {self.n_states - 1}]")
        prior_stockout = bool(int(state_index) % self.stockout_history_levels)
        packed = int(state_index) // self.stockout_history_levels
        sales_pace_index = packed % self.sales_pace_levels
        packed //= self.sales_pace_levels
        outlook_index = packed % self.outlook_levels
        packed //= self.outlook_levels
        inventory = packed % (self.config.max_inventory + 1)
        time_slot = packed // (self.config.max_inventory + 1)
        return time_slot, inventory, outlook_index, sales_pace_index, prior_stockout

    def step(self, action_index: int) -> Tuple[int, float, bool, dict]:
        if isinstance(action_index, bool) or not isinstance(action_index, Integral):
            raise ValueError("action_index must be an integer action index")
        if not 0 <= action_index < self.n_actions:
            raise ValueError(f"action_index must be in [0, {self.n_actions - 1}]")
        if self.time >= self.horizon:
            raise RuntimeError("Episode is finished; call reset() before step().")

        pace_before = self.sales_pace_category_index()
        stockout_history_before = self.prior_stockout_occurred
        cumulative_sold_before = self.cumulative_sold
        cumulative_expected_before = self.cumulative_expected_demand
        requested_prep = int(self.config.actions[action_index])
        prep = min(requested_prep, max(0, self.config.max_inventory - self.inventory))
        available = self.inventory + prep
        mean_demand = self.config.demand_means[self.time] * self._actual_regime_multiplier
        demand = int(self.rng.poisson(mean_demand))
        sold = min(available, demand)
        stockout = max(0, demand - available)
        unsold = max(0, available - sold)

        is_final_slot = self.time == self.horizon - 1
        carryover = 0 if is_final_slot else int(math.floor(unsold * self.config.carryover_fraction))
        wasted = unsold - carryover
        reward = (
            self.config.sold_reward * sold
            - self.config.preparation_cost * prep
            - self.config.stockout_penalty * stockout
            - self.config.waste_penalty * wasted
        )

        info: Dict[str, float | int | str] = {
            "time_slot": self.time,
            "demand": demand,
            "prepared": prep,
            "requested_preparation": requested_prep,
            "starting_inventory": self.inventory,
            "available": available,
            "sold": sold,
            "stockout": stockout,
            "waste": wasted,
            "carryover": carryover,
            "reward": float(reward),
            "demand_outlook_index": self.demand_outlook_index,
            "sales_pace_category_index": pace_before,
            "sales_pace_category": self.sales_pace_label(pace_before),
            "prior_stockout_occurred": stockout_history_before,
            "cumulative_sold_before": cumulative_sold_before,
            "cumulative_expected_demand_before": cumulative_expected_before,
        }

        self.cumulative_sold += sold
        self.cumulative_expected_demand += self.config.demand_means[self.time] * self._forecast_expected_multiplier
        if stockout > 0:
            self.prior_stockout_occurred = True
        self.inventory = carryover
        self.time += 1
        done = self.time >= self.horizon
        next_state = 0 if done else self._state_index()  # terminal value is not bootstrapped
        return next_state, float(reward), done, info
