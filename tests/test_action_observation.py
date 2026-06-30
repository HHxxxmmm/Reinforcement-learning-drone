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

    def test_marshal_action_can_lock_throttle(self):
        agent_action = np.array([1.0, 0.2, -0.2, 0.4])

        real_action = action.marshal_action(agent_action, lock_throttle=True, fixed_throttle=0.0)

        np.testing.assert_allclose(real_action, np.array([0.0, 0.2, -0.2, 0.4]))

    def test_marshal_action_can_lock_roll_and_scale(self):
        agent_action = np.array([0.0, 1.0, 1.0, 1.0])

        real_action = action.marshal_action(agent_action, lock_roll=True, action_scale=0.5)

        np.testing.assert_allclose(real_action, np.array([0.5, 0.5, 0.0, 0.5]))

    def test_marshal_action_can_lock_pitch(self):
        agent_action = np.array([0.0, 1.0, 0.0, 0.5])

        real_action = action.marshal_action(agent_action, lock_pitch=True, fixed_pitch=0.0)

        np.testing.assert_allclose(real_action, np.array([0.5, 0.0, 0.0, 0.5]))

    def test_marshal_action_can_lock_yaw(self):
        agent_action = np.array([0.0, 0.0, 0.0, 1.0])

        real_action = action.marshal_action(agent_action, lock_yaw=True, fixed_yaw=0.0, action_scale=0.4)

        np.testing.assert_allclose(real_action, np.array([0.5, 0.0, 0.0, 0.0]))

    def test_marshal_action_caps_throttle_with_max_throttle(self):
        agent_action = np.array([-1.0, 0.0, 0.0, 0.0])

        low = action.marshal_action(agent_action, max_throttle=0.35)
        high = action.marshal_action(np.array([1.0, 0.0, 0.0, 0.0]), max_throttle=0.35)

        self.assertAlmostEqual(low[0], 0.0)
        self.assertAlmostEqual(high[0], 0.35)

    def test_marshal_action_caps_yaw_with_max_yaw(self):
        agent_action = np.array([0.0, 0.0, 0.0, -1.0])

        low = action.marshal_action(agent_action, action_scale=0.4, max_yaw=0.15)
        high = action.marshal_action(np.array([0.0, 0.0, 0.0, 1.0]), action_scale=0.4, max_yaw=0.15)

        self.assertAlmostEqual(low[3], -0.15)
        self.assertAlmostEqual(high[3], 0.15)

    def test_marshal_action_caps_pitch_with_max_pitch(self):
        agent_action = np.array([0.0, -1.0, 0.0, 0.0])

        low = action.marshal_action(agent_action, action_scale=0.4, max_pitch=0.15)
        high = action.marshal_action(np.array([0.0, 1.0, 0.0, 0.0]), action_scale=0.4, max_pitch=0.15)

        self.assertAlmostEqual(low[1], -0.15)
        self.assertAlmostEqual(high[1], 0.15)


class ObservationTests(unittest.TestCase):
    def test_zero_enemy_position_uses_fixed_target_fallback(self):
        observation.set_enemy_fallback_position(np.array([80.0, 0.0, 30.0]))
        my = make_state((0.0, 0.0, 30.0), angles=(0.0, 0.0, 0.0), vel=(30.0, 0.0, 0.0))
        enemy = make_state((0.0, 0.0, 0.0))

        obs = observation.marshal_observation(my, enemy)

        self.assertEqual(obs.shape, (20,))
        self.assertGreater(obs[0], 0.0)
        self.assertAlmostEqual(obs[1], 0.0)
        self.assertAlmostEqual(obs[4], 1.0)
        self.assertGreater(obs[16], 0.0)
        self.assertGreater(obs[17], 0.0)
        self.assertTrue(np.all(obs <= 1.0))
        self.assertTrue(np.all(obs >= -1.0))

    def test_attack_geometry_features_reward_centerline_over_side_offset(self):
        my = make_state((0.0, 0.0, 30.0), angles=(0.0, 0.0, 0.0), vel=(30.0, 0.0, 0.0))
        centered_enemy = make_state((80.0, 0.0, 30.0))
        side_enemy = make_state((80.0, 20.0, 30.0))

        centered = observation.marshal_observation(my, centered_enemy)
        side = observation.marshal_observation(my, side_enemy)

        self.assertGreater(centered[17], side[17])
        self.assertAlmostEqual(centered[18], 0.0)

    def test_hp_features_support_thousand_point_scale(self):
        my = make_state((0.0, 0.0, 30.0), health=1000.0)
        enemy = make_state((80.0, 0.0, 30.0), health=500.0)

        obs = observation.marshal_observation(my, enemy)

        self.assertAlmostEqual(obs[9], 0.5)
        self.assertAlmostEqual(obs[10], 1.0)
        self.assertAlmostEqual(obs[11], 0.0)

    def test_hp_features_keep_normalized_scale(self):
        my = make_state((0.0, 0.0, 30.0), health=1.0)
        enemy = make_state((80.0, 0.0, 30.0), health=0.5)

        obs = observation.marshal_observation(my, enemy)

        self.assertAlmostEqual(obs[9], 0.5)
        self.assertAlmostEqual(obs[10], 1.0)
        self.assertAlmostEqual(obs[11], 0.0)


if __name__ == "__main__":
    unittest.main()
