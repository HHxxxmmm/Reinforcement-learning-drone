import unittest

import numpy as np

from utils import initialize, reward, truncate


def make_state(pos, angles=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), health=1.0):
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
        enemy = make_state((120.0, 0.0, 20.0), health=0.99)

        comps = reward.reward_components(prev_my, prev_enemy, my, enemy)

        for key in (
            "distance_progress",
            "proximity",
            "alignment",
            "attack_box",
            "corridor",
            "centerline",
            "turn_penalty",
            "enemy_damage",
            "enemy_hp_shaping",
            "overshoot",
            "finish_centerline",
            "finish_speed_penalty",
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
        prev_enemy = make_state((65.0, 0.0, 20.0), health=1.0)
        my = make_state((10.0, 0.0, 20.0), angles=(0.0, 0.0, 0.0))
        enemy = make_state((65.0, 0.0, 20.0), health=0.99)

        comps = reward.reward_components(prev_my, prev_enemy, my, enemy)

        self.assertGreater(comps["distance_progress"], 0.0)
        self.assertGreater(comps["alignment"], 0.0)
        self.assertGreater(comps["corridor"], 0.0)
        self.assertGreater(comps["enemy_damage"], 0.0)
        self.assertGreater(comps["total"], 0.0)

    def test_single_hit_damage_uses_simple_normalized_hp_scale(self):
        prev_my = make_state((0.0, 0.0, 20.0), angles=(0.0, 0.0, 0.0))
        prev_enemy = make_state((12.0, 0.0, 20.0), health=1.0)
        my = make_state((0.0, 0.0, 20.0), angles=(0.0, 0.0, 0.0))
        enemy = make_state((12.0, 0.0, 20.0), health=0.99)

        comps = reward.reward_components(prev_my, prev_enemy, my, enemy)

        self.assertAlmostEqual(comps["enemy_damage"], 8.0)

    def test_damage_reward_also_accepts_thousand_point_hp_scale(self):
        prev_my = make_state((0.0, 0.0, 20.0), angles=(0.0, 0.0, 0.0))
        prev_enemy = make_state((12.0, 0.0, 20.0), health=1000.0)
        my = make_state((0.0, 0.0, 20.0), angles=(0.0, 0.0, 0.0))
        enemy = make_state((12.0, 0.0, 20.0), health=990.0)

        comps = reward.reward_components(prev_my, prev_enemy, my, enemy)

        self.assertAlmostEqual(comps["enemy_damage"], 8.0)

    def test_enemy_damage_only_scores_inside_attack_box(self):
        prev_my = make_state((0.0, 0.0, 20.0), angles=(0.0, 0.0, 0.0))
        prev_enemy = make_state((120.0, 0.0, 20.0), health=1.0)
        my = make_state((0.0, 0.0, 20.0), angles=(0.0, 0.0, 0.0))
        enemy = make_state((120.0, 0.0, 20.0), health=0.99)

        comps = reward.reward_components(prev_my, prev_enemy, my, enemy)

        self.assertLessEqual(comps["attack_box"], 0.0)
        self.assertEqual(comps["enemy_damage"], 0.0)

    def test_enemy_hp_shaping_rewards_hp_reduction_not_static_low_hp(self):
        my = make_state((60.0, 0.0, 30.0), angles=(0.0, 0.0, 0.0))
        prev_enemy = make_state((120.0, 0.0, 30.0), health=1000.0)
        damaged_enemy = make_state((120.0, 0.0, 30.0), health=990.0)
        static_low_enemy = make_state((120.0, 0.0, 30.0), health=50.0)

        damaged = reward.reward_components(my, prev_enemy, my, damaged_enemy)
        static_low = reward.reward_components(my, static_low_enemy, my, static_low_enemy)

        self.assertGreater(damaged["enemy_hp_shaping"], 0.0)
        self.assertEqual(static_low["enemy_hp_shaping"], 0.0)

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

    def test_zero_enemy_position_uses_fixed_target_fallback(self):
        prev_my = make_state((0.0, 0.0, 30.0), angles=(0.0, 0.0, 0.0))
        prev_enemy = make_state((0.0, 0.0, 0.0))
        my = make_state((5.0, 0.0, 30.0), angles=(0.0, 0.0, 0.0))
        enemy = make_state((0.0, 0.0, 0.0))

        comps = reward.reward_components(prev_my, prev_enemy, my, enemy)

        self.assertGreater(comps["distance_progress"], 0.0)
        self.assertGreater(comps["alignment"], 0.0)

    def test_moving_away_misaligned_and_taking_damage_is_penalized(self):
        prev_my = make_state((0.0, 0.0, 20.0), angles=(0.0, 0.0, np.pi))
        prev_enemy = make_state((120.0, 0.0, 20.0), health=1.0)
        my = make_state((-10.0, 0.0, 20.0), angles=(0.0, 0.0, np.pi), health=0.99)
        enemy = make_state((120.0, 0.0, 20.0), health=1.0)

        comps = reward.reward_components(prev_my, prev_enemy, my, enemy)

        self.assertLess(comps["distance_progress"], 0.0)
        self.assertLess(comps["alignment"], 0.0)
        self.assertLess(comps["self_damage"], 0.0)
        self.assertLess(comps["total"], 0.0)

    def test_centerline_reward_prefers_small_lateral_error_before_attack_box(self):
        prev_my = make_state((0.0, 0.0, 30.0), angles=(0.0, 0.0, 0.0))
        prev_enemy = make_state((80.0, 0.0, 30.0))
        my = make_state((0.0, 0.0, 30.0), angles=(0.0, 0.0, 0.0))
        centered_enemy = make_state((80.0, 0.0, 30.0))
        side_enemy = make_state((80.0, 20.0, 30.0))

        centered = reward.reward_components(prev_my, prev_enemy, my, centered_enemy)
        side = reward.reward_components(prev_my, prev_enemy, my, side_enemy)

        self.assertGreater(centered["centerline"], side["centerline"])

    def test_turn_penalty_discourages_large_angular_rates(self):
        prev_my = make_state((0.0, 0.0, 30.0), angles=(0.0, 0.0, 0.0))
        prev_enemy = make_state((80.0, 0.0, 30.0))
        steady = make_state((0.0, 0.0, 30.0), angles=(0.0, 0.0, 0.0))
        turning = make_state((0.0, 0.0, 30.0), angles=(0.0, 0.0, 0.0))
        turning[9:12] = np.array([0.0, 0.0, 2.0])
        enemy = make_state((80.0, 0.0, 30.0))

        steady_comps = reward.reward_components(prev_my, prev_enemy, steady, enemy)
        turning_comps = reward.reward_components(prev_my, prev_enemy, turning, enemy)

        self.assertEqual(steady_comps["turn_penalty"], 0.0)
        self.assertLess(turning_comps["turn_penalty"], steady_comps["turn_penalty"])

    def test_overshoot_penalizes_passing_alive_target(self):
        prev_my = make_state((0.0, 0.0, 30.0), angles=(0.0, 0.0, 0.0))
        prev_enemy = make_state((80.0, 0.0, 30.0), health=50.0)
        passed_my = make_state((82.0, 0.0, 30.0), angles=(0.0, 0.0, 0.0))
        enemy = make_state((80.0, 0.0, 30.0), health=50.0)

        comps = reward.reward_components(prev_my, prev_enemy, passed_my, enemy)

        self.assertLess(comps["overshoot"], 0.0)

    def test_finish_centerline_penalizes_lateral_error_near_alive_target(self):
        prev_my = make_state((60.0, 0.0, 30.0), angles=(0.0, 0.0, 0.0))
        prev_enemy = make_state((80.0, 0.0, 30.0), health=50.0)
        centered_my = make_state((60.0, 0.0, 30.0), angles=(0.0, 0.0, 0.0))
        side_my = make_state((60.0, 0.0, 30.0), angles=(0.0, 0.0, 0.0))
        centered_enemy = make_state((80.0, 0.0, 30.0), health=50.0)
        side_enemy = make_state((80.0, 2.0, 30.0), health=50.0)

        centered = reward.reward_components(prev_my, prev_enemy, centered_my, centered_enemy)
        side = reward.reward_components(prev_my, prev_enemy, side_my, side_enemy)

        self.assertGreater(centered["finish_centerline"], side["finish_centerline"])
        self.assertGreater(centered["finish_centerline"], 3.5)

    def test_finish_speed_penalty_discourages_high_speed_near_alive_target(self):
        prev_my = make_state((60.0, 0.0, 30.0), angles=(0.0, 0.0, 0.0), vel=(12.0, 0.0, 0.0))
        prev_enemy = make_state((80.0, 0.0, 30.0), health=50.0)
        slow_my = make_state((60.0, 0.0, 30.0), angles=(0.0, 0.0, 0.0), vel=(12.0, 0.0, 0.0))
        fast_my = make_state((60.0, 0.0, 30.0), angles=(0.0, 0.0, 0.0), vel=(30.0, 0.0, 0.0))
        enemy = make_state((80.0, 0.0, 30.0), health=50.0)

        slow = reward.reward_components(prev_my, prev_enemy, slow_my, enemy)
        fast = reward.reward_components(prev_my, prev_enemy, fast_my, enemy)

        self.assertEqual(slow["finish_speed_penalty"], 0.0)
        self.assertLess(fast["finish_speed_penalty"], slow["finish_speed_penalty"])
        self.assertAlmostEqual(fast["finish_speed_penalty"], -1.2)

    def test_finish_speed_penalty_ignores_far_target(self):
        prev_my = make_state((0.0, 0.0, 30.0), angles=(0.0, 0.0, 0.0), vel=(30.0, 0.0, 0.0))
        prev_enemy = make_state((80.0, 0.0, 30.0), health=50.0)
        fast_my = make_state((0.0, 0.0, 30.0), angles=(0.0, 0.0, 0.0), vel=(30.0, 0.0, 0.0))
        enemy = make_state((80.0, 0.0, 30.0), health=50.0)

        comps = reward.reward_components(prev_my, prev_enemy, fast_my, enemy)

        self.assertEqual(comps["finish_speed_penalty"], 0.0)


class InitializeTests(unittest.TestCase):
    def test_initial_state_has_two_aircraft_and_safe_start_distance(self):
        initial = initialize.generate_initial_state()

        self.assertEqual(initial.shape, (24,))
        my_pos = initial[0:3].astype(np.float64) * 10.0
        enemy_pos = initial[12:15].astype(np.float64) * 10.0
        self.assertGreaterEqual(np.linalg.norm(enemy_pos - my_pos), 700.0)
        np.testing.assert_array_equal(initial[12:15], np.array([80, 0, 30]))
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

    def test_zero_enemy_position_uses_fixed_target_for_separation(self):
        my = make_state((500.0, 0.0, 30.0))
        enemy = make_state((0.0, 0.0, 0.0))

        self.assertFalse(truncate.check_truncation(my, enemy))


if __name__ == "__main__":
    unittest.main()
