import unittest

import numpy as np

from envs.train_env import is_expected_initial_observation
from utils.initialize import generate_initial_state


class TrainEnvResetTests(unittest.TestCase):
    def test_accepts_observation_matching_generated_initial_state(self):
        initial = generate_initial_state()
        my_state = np.zeros(13, dtype=np.float64)
        enemy_state = np.zeros(13, dtype=np.float64)
        my_state[:12] = initial[:12]
        enemy_state[:12] = initial[12:]
        my_state[12] = 1000.0
        enemy_state[12] = 1000.0

        self.assertTrue(is_expected_initial_observation(my_state, enemy_state, initial))

    def test_rejects_stale_round_observation(self):
        initial = generate_initial_state()
        my_state = np.zeros(13, dtype=np.float64)
        enemy_state = np.zeros(13, dtype=np.float64)
        my_state[:3] = np.array([2860.0, 0.0, 30.0])
        enemy_state[:3] = np.array([120.0, 0.0, 30.0])
        my_state[12] = 370.0
        enemy_state[12] = 210.0

        self.assertFalse(is_expected_initial_observation(my_state, enemy_state, initial))


if __name__ == "__main__":
    unittest.main()
