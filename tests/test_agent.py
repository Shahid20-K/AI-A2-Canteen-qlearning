import unittest
import numpy as np

from canteen_rl.agent import (
    TrainingConfig,
    train_q_learning,
    greedy_policy,
    fixed_batch_policy,
    time_of_day_policy,
    evaluate_policy_episodes,
)
from canteen_rl.environment import CanteenEnv, EnvironmentConfig
from run_experiment import scaled_demand_config


class TestQLearningAndEvaluation(unittest.TestCase):
    def test_training_returns_finite_q_table_with_expected_shape(self):
        env_config = EnvironmentConfig()
        training_config = TrainingConfig(episodes=5, seed=19)
        q_table, returns, info = train_q_learning(training_config, env_config)
        env = CanteenEnv(env_config)
        self.assertEqual(q_table.shape, (env.n_states, env.n_actions))
        self.assertEqual(returns.shape, (5,))
        self.assertTrue(np.isfinite(q_table).all())
        self.assertTrue(np.isfinite(returns).all())
        self.assertGreaterEqual(info["final_epsilon"], training_config.epsilon_min)

        policy = greedy_policy(q_table)
        state, _ = env.reset(seed=44)
        action = policy(state, env)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, env.n_actions)

    def test_policy_evaluation_is_reproducible(self):
        first = evaluate_policy_episodes(fixed_batch_policy, episodes=20, seed_start=900)
        second = evaluate_policy_episodes(fixed_batch_policy, episodes=20, seed_start=900)
        for metric in first:
            np.testing.assert_array_equal(first[metric], second[metric])

    def test_policies_share_same_demand_scenarios_when_seed_matched(self):
        cfg = EnvironmentConfig()
        fixed = evaluate_policy_episodes(fixed_batch_policy, episodes=25, env_config=cfg, seed_start=1200)
        schedule = evaluate_policy_episodes(time_of_day_policy, episodes=25, env_config=cfg, seed_start=1200)
        # Since policy actions do not draw from the environment RNG, equal reset seeds
        # produce the same daily demand sequence. This supports paired policy comparisons.
        np.testing.assert_array_equal(fixed["demand"], schedule["demand"])

    def test_training_config_rejects_invalid_parameters(self):
        with self.assertRaises(ValueError):
            TrainingConfig(episodes=0)
        with self.assertRaises(ValueError):
            TrainingConfig(learning_rate=1.2)
        with self.assertRaises(ValueError):
            TrainingConfig(epsilon_min=0.2, epsilon_start=0.1)

    def test_history_aware_baseline_uses_sales_pace_and_stockout_flag(self):
        env = CanteenEnv()
        env.reset(seed=88)
        env.time = 3
        env.inventory = 0
        env.demand_outlook_index = 1  # normal forecast; posterior mean multiplier is 1.0
        env.cumulative_expected_demand = 100.0
        env.cumulative_sold = 50  # below-expected sales pace
        env.prior_stockout_occurred = False
        under_pace_action = time_of_day_policy(
            0, env, target_fraction=0.70, history_adjustment=0.20
        )
        env.prior_stockout_occurred = True
        stockout_history_action = time_of_day_policy(
            0, env, target_fraction=0.70, history_adjustment=0.20
        )
        # With the same observed low sales pace, a known past stockout shifts the
        # baseline's preparation decision upward rather than interpreting sales as low demand.
        self.assertGreater(stockout_history_action, under_pace_action)

    def test_forecast_baseline_does_not_get_shifted_means_in_stress_test(self):
        nominal = EnvironmentConfig()
        shifted = scaled_demand_config(nominal, 0.75)
        env = CanteenEnv(shifted)
        env.reset(seed=88)
        env.time = 2
        env.inventory = 0
        env.demand_outlook_index = 1
        # The shifted environment alone would select a smaller batch, while a baseline
        # retaining its nominal planning model continues to use the nominal mean.
        shifted_aware_action = time_of_day_policy(0, env, target_fraction=1.0)
        nominal_model_action = time_of_day_policy(
            0, env, target_fraction=1.0, planning_demand_means=nominal.demand_means
        )
        self.assertNotEqual(shifted_aware_action, nominal_model_action)


if __name__ == "__main__":
    unittest.main()
