import unittest

import numpy as np

from utils import action, observation


def make_state(pos, angles=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), health=1.0):
    state = np.zeros(13, dtype=np.float64)
    state[0:3] = np.array(pos, dtype=np.float64)
    state[3:6] = np.array(angles, dtype=np.float64)
    state[6:9] = np.array(vel, dtype=np.float64)
    state[12] = health
    return state


class ActionTests(unittest.TestCase):
    def test_marshal_action_uses_agent_output_with_throttle_mapping(self):
        agent_action = np.array([-1.0, -0.5, 0.25, 2.0])

        real_action = action.marshal_action(agent_action)

        np.testing.assert_allclose(real_action, np.array([0.0, -0.5, 0.25, 1.0]))

    def test_marshal_action_clips_out_of_range_values(self):
        agent_action = np.array([2.0, -2.0, 0.0, 0.5])

        real_action = action.marshal_action(agent_action)

        np.testing.assert_allclose(real_action, np.array([1.0, -1.0, 0.0, 0.5]))


class ObservationTests(unittest.TestCase):
    def test_zero_enemy_position_uses_fixed_target_fallback(self):
        my = make_state((0.0, 0.0, 30.0), angles=(0.0, 0.0, 0.0), vel=(30.0, 0.0, 0.0))
        enemy = make_state((0.0, 0.0, 0.0))

        obs = observation.marshal_observation(my, enemy)

        self.assertEqual(obs.shape, (16,))
        self.assertGreater(obs[0], 0.0)
        self.assertAlmostEqual(obs[1], 0.0)
        self.assertAlmostEqual(obs[4], 1.0)
        self.assertTrue(np.all(obs <= 1.0))
        self.assertTrue(np.all(obs >= -1.0))


if __name__ == "__main__":
    unittest.main()
