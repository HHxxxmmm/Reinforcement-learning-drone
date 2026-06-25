import unittest

import numpy as np

from utils import initialize, reward, truncate


def make_state(pos, angles=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), health=1000.0):
    state = np.zeros(13, dtype=np.float64)
    state[0:3] = np.array(pos, dtype=np.float64)
    state[3:6] = np.array(angles, dtype=np.float64)
    state[6:9] = np.array(vel, dtype=np.float64)
    state[12] = health
    return state


class RewardTests(unittest.TestCase):
    def test_reward_reports_named_components_and_total(self):
        prev_my = make_state((0.0, 0.0, 20.0))
        prev_enemy = make_state((120.0, 0.0, 20.0))
        my = make_state((5.0, 0.0, 20.0))
        enemy = make_state((120.0, 0.0, 20.0), health=990.0)

        comps = reward.reward_components(prev_my, prev_enemy, my, enemy)

        for key in (
            "distance_progress",
            "proximity",
            "alignment",
            "corridor",
            "enemy_damage",
            "self_damage",
            "survival",
            "total",
        ):
            self.assertIn(key, comps)
            self.assertTrue(np.isfinite(comps[key]))

        subtotal = sum(value for key, value in comps.items() if key != "total")
        self.assertAlmostEqual(comps["total"], subtotal)

    def test_closing_aligned_attack_with_damage_is_rewarded(self):
        prev_my = make_state((0.0, 0.0, 20.0), angles=(0.0, 0.0, 0.0))
        prev_enemy = make_state((120.0, 0.0, 20.0), health=1000.0)
        my = make_state((10.0, 0.0, 20.0), angles=(0.0, 0.0, 0.0))
        enemy = make_state((120.0, 0.0, 20.0), health=990.0)

        comps = reward.reward_components(prev_my, prev_enemy, my, enemy)

        self.assertGreater(comps["distance_progress"], 0.0)
        self.assertGreater(comps["alignment"], 0.0)
        self.assertGreater(comps["corridor"], 0.0)
        self.assertGreater(comps["enemy_damage"], 0.0)
        self.assertGreater(comps["total"], 0.0)

    def test_single_hit_damage_uses_documented_ten_hp_scale(self):
        prev_my = make_state((0.0, 0.0, 20.0), angles=(0.0, 0.0, 0.0))
        prev_enemy = make_state((12.0, 0.0, 20.0), health=1000.0)
        my = make_state((0.0, 0.0, 20.0), angles=(0.0, 0.0, 0.0))
        enemy = make_state((12.0, 0.0, 20.0), health=990.0)

        comps = reward.reward_components(prev_my, prev_enemy, my, enemy)

        self.assertAlmostEqual(comps["enemy_damage"], 20.0)

    def test_attack_box_reward_requires_documented_front_range(self):
        prev_my = make_state((0.0, 0.0, 20.0), angles=(0.0, 0.0, 0.0))
        prev_enemy = make_state((7.0, 0.0, 20.0))
        my = make_state((0.0, 0.0, 20.0), angles=(0.0, 0.0, 0.0))
        inside_enemy = make_state((7.0, 0.0, 20.0))
        too_close_enemy = make_state((5.0, 0.0, 20.0))
        too_far_enemy = make_state((70.0, 0.0, 20.0))
        lateral_enemy = make_state((7.0, 2.0, 20.0))

        inside = reward.reward_components(prev_my, prev_enemy, my, inside_enemy)
        too_close = reward.reward_components(prev_my, prev_enemy, my, too_close_enemy)
        too_far = reward.reward_components(prev_my, prev_enemy, my, too_far_enemy)
        lateral = reward.reward_components(prev_my, prev_enemy, my, lateral_enemy)

        self.assertGreater(inside["attack_box"], 0.0)
        self.assertLessEqual(too_close["attack_box"], 0.0)
        self.assertLessEqual(too_far["attack_box"], 0.0)
        self.assertLess(lateral["attack_box"], inside["attack_box"])

    def test_moving_away_misaligned_and_taking_damage_is_penalized(self):
        prev_my = make_state((0.0, 0.0, 20.0), angles=(0.0, 0.0, np.pi))
        prev_enemy = make_state((120.0, 0.0, 20.0), health=1000.0)
        my = make_state((-10.0, 0.0, 20.0), angles=(0.0, 0.0, np.pi), health=990.0)
        enemy = make_state((120.0, 0.0, 20.0), health=1000.0)

        comps = reward.reward_components(prev_my, prev_enemy, my, enemy)

        self.assertLess(comps["distance_progress"], 0.0)
        self.assertLess(comps["alignment"], 0.0)
        self.assertLess(comps["self_damage"], 0.0)
        self.assertLess(comps["total"], 0.0)


class InitializeTests(unittest.TestCase):
    def test_initial_state_has_two_aircraft_and_safe_start_distance(self):
        initial = initialize.generate_initial_state()

        self.assertEqual(initial.shape, (24,))
        my_pos = initial[0:3].astype(np.float64) * 10.0
        enemy_pos = initial[12:15].astype(np.float64) * 10.0
        self.assertGreaterEqual(np.linalg.norm(enemy_pos - my_pos), 1000.0)
        self.assertGreater(initial[2], 0)
        self.assertGreaterEqual(initial[6], 20)
        self.assertLessEqual(initial[6], 40)
        self.assertGreaterEqual(initial[18], 0)


class TruncateTests(unittest.TestCase):
    def test_truncates_low_altitude_and_extreme_separation(self):
        enemy = make_state((120.0, 0.0, 20.0))

        self.assertTrue(truncate.check_truncation(make_state((0.0, 0.0, -1.0)), enemy))
        self.assertTrue(truncate.check_truncation(make_state((-600.0, 0.0, 20.0)), enemy))

    def test_keeps_valid_engagement_running(self):
        my = make_state((0.0, 0.0, 20.0))
        enemy = make_state((120.0, 0.0, 20.0))

        self.assertFalse(truncate.check_truncation(my, enemy))


if __name__ == "__main__":
    unittest.main()
