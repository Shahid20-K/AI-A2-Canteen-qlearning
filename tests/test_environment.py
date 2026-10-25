import unittest

from canteen_rl.environment import CanteenEnv, EnvironmentConfig


class TestCanteenEnv(unittest.TestCase):
    def test_episode_horizon_and_transition_invariants(self):
        env = CanteenEnv(seed=10)
        state, _ = env.reset(seed=10)
        self.assertGreaterEqual(state, 0)
        total_steps = 0
        done = False
        while not done:
            state, reward, done, info = env.step(0)
            self.assertGreaterEqual(info["demand"], 0)
            self.assertGreaterEqual(info["sold"], 0)
            self.assertLessEqual(info["sold"], info["available"])
            self.assertEqual(info["sold"] + info["stockout"], info["demand"])
            self.assertEqual(info["waste"] + info["carryover"], info["available"] - info["sold"])
            self.assertGreaterEqual(info["carryover"], 0)
            self.assertLessEqual(info["carryover"], env.config.max_inventory)
            if not done:
                self.assertGreaterEqual(state, 0)
                self.assertLess(state, env.n_states)
            total_steps += 1
        self.assertEqual(total_steps, len(EnvironmentConfig().demand_means))

    def test_exact_inventory_state_and_dimensions(self):
        env = CanteenEnv(seed=4)
        state, info = env.reset(seed=4)
        self.assertEqual(env.n_states, 8 * 41 * 3)
        self.assertEqual(env.n_actions, 5)
        self.assertIn(info["demand_outlook"], {"low", "normal", "high"})
        self.assertGreaterEqual(state, 0)
        self.assertLess(state, env.n_states)

        # Same time and outlook, but different exact inventories must produce different states.
        env.inventory = 12
        state_12 = env._state_index()
        env.inventory = 13
        state_13 = env._state_index()
        self.assertNotEqual(state_12, state_13)

        # For a fixed time and outlook, every allowed inventory maps to a unique state.
        env.inventory = 0
        states = []
        for inventory in range(env.config.max_inventory + 1):
            env.inventory = inventory
            states.append(env._state_index())
        self.assertEqual(len(states), len(set(states)))

    def test_invalid_action_fails(self):
        env = CanteenEnv(seed=1)
        env.reset(seed=1)
        with self.assertRaises(ValueError):
            env.step(100)


    def test_forecast_accuracy_is_approximately_configured_probability(self):
        env = CanteenEnv(EnvironmentConfig(forecast_accuracy=0.70))
        matches = 0
        episodes = 5000
        for seed in range(episodes):
            env.reset(seed=10_000 + seed)
            matches += int(env.demand_outlook_index == env._actual_regime_index)
        observed_accuracy = matches / episodes
        self.assertAlmostEqual(observed_accuracy, 0.70, delta=0.025)

    def test_perfect_forecast_reveals_true_category(self):
        env = CanteenEnv(EnvironmentConfig(forecast_accuracy=1.0))
        for seed in range(50):
            _state, info = env.reset(seed=seed)
            self.assertEqual(env.demand_outlook_index, env._actual_regime_index)
            self.assertEqual(info["demand_outlook_index"], env.demand_outlook_index)

    def test_expected_multiplier_uses_posterior_for_noisy_forecast(self):
        config = EnvironmentConfig(forecast_accuracy=0.70)
        env = CanteenEnv(config)
        # With symmetric priors/multipliers, observing the normal category has an
        # expected multiplier of 1.0 under this confusion model.
        self.assertAlmostEqual(env.expected_regime_multiplier(1), 1.0, places=12)
        self.assertLess(env.expected_regime_multiplier(0), 1.0)
        self.assertGreater(env.expected_regime_multiplier(2), 1.0)

    def test_invalid_forecast_accuracy_is_rejected(self):
        for accuracy in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(accuracy=accuracy):
                with self.assertRaises(ValueError):
                    CanteenEnv(EnvironmentConfig(forecast_accuracy=accuracy))

    def test_same_seed_is_reproducible(self):
        records = []
        for _ in range(2):
            env = CanteenEnv()
            env.reset(seed=77)
            episode = []
            done = False
            while not done:
                _, _, done, info = env.step(2)
                episode.append((info["demand"], info["sold"], info["waste"]))
            records.append(episode)
        self.assertEqual(records[0], records[1])

    def test_reward_calculation(self):
        # Zero demand makes the reward arithmetic deterministic for this test.
        cfg = EnvironmentConfig(demand_means=(0,) * 8)
        env = CanteenEnv(cfg, seed=123)
        env.reset(seed=123)
        # Action index 1 prepares 5 portions; 30% carryover floors to 1, waste is 4.
        _, reward, _, info = env.step(1)
        expected = 0 - 0.35 * 5 - 1.0 * 0 - 1.5 * 4
        self.assertEqual(info["prepared"], 5)
        self.assertEqual(info["waste"], 4)
        self.assertEqual(info["carryover"], 1)
        self.assertAlmostEqual(reward, expected)

    def test_state_decode_round_trip(self):
        env = CanteenEnv()
        for time_slot in range(env.horizon):
            env.time = time_slot
            for inventory in range(env.config.max_inventory + 1):
                env.inventory = inventory
                for outlook in range(env.outlook_levels):
                    env.demand_outlook_index = outlook
                    state = env._state_index()
                    self.assertEqual(env.decode_state(state), (time_slot, inventory, outlook))

    def test_non_integer_action_is_rejected(self):
        env = CanteenEnv()
        env.reset(seed=17)
        with self.assertRaises(ValueError):
            env.step(1.5)

    def test_invalid_environment_configurations_are_rejected(self):
        invalid_configs = [
            EnvironmentConfig(demand_means=()),
            EnvironmentConfig(actions=(5, 10)),
            EnvironmentConfig(actions=(0, 5, 5)),
            EnvironmentConfig(regime_probabilities=(0.25, 0.50, 0.20)),
            EnvironmentConfig(regime_probabilities=(-0.1, 0.6, 0.5)),
            EnvironmentConfig(carryover_fraction=1.1),
        ]
        for config in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    CanteenEnv(config)


if __name__ == "__main__":
    unittest.main()
