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
            "yaw_misalign_penalty",
            "attack_box",
            "corridor",
            "centerline",
            "turn_penalty",
            "enemy_damage",
            "enemy_hp_shaping",
            "overshoot",
            "finish_centerline",
            "speed_cruise_penalty",
            "speed_approach_penalty",
            "speed_penalty",
            "attack_speed_penalty",
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

    def test_enemy_damage_rewards_any_enemy_hp_drop(self):
        prev_my = make_state((0.0, 0.0, 20.0), angles=(0.0, 0.0, 0.0))
        prev_enemy = make_state((120.0, 0.0, 20.0), health=1.0)
        my = make_state((0.0, 0.0, 20.0), angles=(0.0, 0.0, 0.0))
        enemy = make_state((120.0, 0.0, 20.0), health=0.99)

        comps = reward.reward_components(prev_my, prev_enemy, my, enemy)

        self.assertAlmostEqual(comps["enemy_damage"], 8.0)

    def test_enemy_damage_rewards_hp_drop_even_when_too_close_for_attack_box(self):
        prev_my = make_state((0.0, 0.0, 20.0), angles=(0.0, 0.0, 0.0))
        prev_enemy = make_state((5.0, 0.0, 20.0), health=1.0)
        my = make_state((0.0, 0.0, 20.0), angles=(0.0, 0.0, 0.0))
        enemy = make_state((5.0, 0.0, 20.0), health=0.99)

        comps = reward.reward_components(prev_my, prev_enemy, my, enemy)

        self.assertLessEqual(comps["attack_box"], 0.0)
        self.assertAlmostEqual(comps["enemy_damage"], 8.0)

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

    def test_yaw_misalign_penalty_scales_with_one_minus_cos(self):
        prev_my = make_state((0.0, 0.0, 30.0), angles=(0.0, 0.0, 0.0))
        prev_enemy = make_state((80.0, 0.0, 30.0))
        aligned = make_state((0.0, 0.0, 30.0), angles=(0.0, 0.0, 0.0))
        misaligned = make_state((0.0, 0.0, 30.0), angles=(0.0, 0.0, 0.35))
        enemy = make_state((80.0, 0.0, 30.0))

        aligned_comps = reward.reward_components(
            prev_my, prev_enemy, aligned, enemy, yaw_misalign_weight=5.0
        )
        misaligned_comps = reward.reward_components(
            prev_my, prev_enemy, misaligned, enemy, yaw_misalign_weight=5.0
        )

        self.assertAlmostEqual(aligned_comps["yaw_misalign_penalty"], 0.0, places=5)
        self.assertLess(misaligned_comps["yaw_misalign_penalty"], aligned_comps["yaw_misalign_penalty"])
        self.assertLess(misaligned_comps["total"], aligned_comps["total"])

    def test_damage_reward_per_hit_override(self):
        prev_my = make_state((0.0, 0.0, 30.0))
        prev_enemy = make_state((80.0, 0.0, 30.0), health=1.0)
        my = make_state((1.0, 0.0, 30.0))
        enemy = make_state((80.0, 0.0, 30.0), health=0.99)

        default_comps = reward.reward_components(prev_my, prev_enemy, my, enemy)
        boosted_comps = reward.reward_components(
            prev_my, prev_enemy, my, enemy, damage_reward_per_hit=12.0
        )

        self.assertAlmostEqual(boosted_comps["enemy_damage"], default_comps["enemy_damage"] * 1.5)

    def test_alignment_reward_weight_override(self):
        prev_my = make_state((0.0, 0.0, 30.0), angles=(0.0, 0.0, 0.0))
        prev_enemy = make_state((80.0, 0.0, 30.0))
        my = make_state((0.0, 0.0, 30.0), angles=(0.0, 0.0, 0.0))
        enemy = make_state((80.0, 0.0, 30.0))

        default_comps = reward.reward_components(prev_my, prev_enemy, my, enemy)
        reduced_comps = reward.reward_components(
            prev_my, prev_enemy, my, enemy, alignment_weight=1.0
        )

        self.assertAlmostEqual(reduced_comps["alignment"], default_comps["alignment"] * 0.5)
        self.assertEqual(reduced_comps["yaw_misalign_penalty"], default_comps["yaw_misalign_penalty"])

    def test_altitude_match_rewards_same_height_over_offset(self):
        prev_my = make_state((0.0, 0.0, 100.0))
        prev_enemy = make_state((80.0, 0.0, 100.0))
        level_my = make_state((1.0, 0.0, 100.0))
        low_my = make_state((1.0, 0.0, 90.0))
        enemy = make_state((80.0, 0.0, 100.0))

        level_comps = reward.reward_components(
            prev_my, prev_enemy, level_my, enemy, altitude_match_weight=4.0
        )
        low_comps = reward.reward_components(
            prev_my, prev_enemy, low_my, enemy, altitude_match_weight=4.0
        )

        self.assertGreater(level_comps["altitude_match"], low_comps["altitude_match"])
        self.assertAlmostEqual(level_comps["altitude_match"], 4.0, places=5)

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

    def test_speed_penalty_discourages_excess_speed_on_approach(self):
        prev_my = make_state((60.0, 0.0, 30.0), angles=(0.0, 0.0, 0.0), vel=(5.0, 0.0, 0.0))
        prev_enemy = make_state((80.0, 0.0, 30.0), health=50.0)
        slow_my = make_state((60.0, 0.0, 30.0), angles=(0.0, 0.0, 0.0), vel=(5.0, 0.0, 0.0))
        fast_my = make_state((60.0, 0.0, 30.0), angles=(0.0, 0.0, 0.0), vel=(30.0, 0.0, 0.0))
        enemy = make_state((80.0, 0.0, 30.0), health=50.0)

        slow = reward.reward_components(prev_my, prev_enemy, slow_my, enemy)
        fast = reward.reward_components(prev_my, prev_enemy, fast_my, enemy)

        self.assertEqual(slow["speed_penalty"], 0.0)
        self.assertLess(fast["speed_penalty"], slow["speed_penalty"])

    def test_attack_speed_penalty_in_attack_box_when_too_fast(self):
        prev_my = make_state((50.0, 0.0, 30.0), angles=(0.0, 0.0, 0.0), vel=(20.0, 0.0, 0.0))
        prev_enemy = make_state((80.0, 0.0, 30.0), health=50.0)
        slow_my = make_state((50.0, 0.0, 30.0), angles=(0.0, 0.0, 0.0), vel=(20.0, 0.0, 0.0))
        fast_my = make_state((50.0, 0.0, 30.0), angles=(0.0, 0.0, 0.0), vel=(35.0, 0.0, 0.0))
        enemy = make_state((80.0, 0.0, 30.0), health=50.0)

        slow = reward.reward_components(prev_my, prev_enemy, slow_my, enemy)
        fast = reward.reward_components(prev_my, prev_enemy, fast_my, enemy)

        self.assertEqual(slow["attack_speed_penalty"], 0.0)
        self.assertLess(fast["attack_speed_penalty"], 0.0)

    def test_speed_penalty_ignores_far_target(self):
        prev_my = make_state((0.0, 0.0, 30.0), angles=(0.0, 0.0, 0.0), vel=(30.0, 0.0, 0.0))
        prev_enemy = make_state((200.0, 0.0, 30.0), health=50.0)
        fast_my = make_state((0.0, 0.0, 30.0), angles=(0.0, 0.0, 0.0), vel=(30.0, 0.0, 0.0))
        enemy = make_state((200.0, 0.0, 30.0), health=50.0)

        comps = reward.reward_components(prev_my, prev_enemy, fast_my, enemy)

        self.assertEqual(comps["speed_approach_penalty"], 0.0)
        self.assertEqual(comps["speed_penalty"], 0.0)

    def test_speed_cruise_penalty_caps_far_overspeed(self):
        prev_my = make_state((0.0, 0.0, 30.0), angles=(0.0, 0.0, 0.0), vel=(30.0, 0.0, 0.0))
        prev_enemy = make_state((200.0, 0.0, 30.0), health=50.0)
        fast_my = make_state((0.0, 0.0, 30.0), angles=(0.0, 0.0, 0.0), vel=(45.0, 0.0, 0.0))
        enemy = make_state((200.0, 0.0, 30.0), health=50.0)

        comps = reward.reward_components(
            prev_my,
            prev_enemy,
            fast_my,
            enemy,
            speed_approach_range_m=900.0,
            speed_cruise_cap_mps=165.0,
            speed_cruise_penalty_weight=0.018,
        )

        self.assertLess(comps["speed_cruise_penalty"], 0.0)
        self.assertEqual(comps["speed_approach_penalty"], 0.0)

    def test_speed_penalties_disabled_when_altitude_mismatch(self):
        prev_my = make_state((60.0, 0.0, 30.0), angles=(0.0, 0.0, 0.0), vel=(30.0, 0.0, 0.0))
        prev_enemy = make_state((80.0, 0.0, 30.0), health=50.0)
        low_my = make_state((60.0, 0.0, 20.0), angles=(0.0, 0.0, 0.0), vel=(40.0, 0.0, 0.0))
        enemy = make_state((80.0, 0.0, 30.0), health=50.0)

        comps = reward.reward_components(
            prev_my,
            prev_enemy,
            low_my,
            enemy,
            speed_penalty_weight=0.01,
            attack_speed_penalty_weight=6.0,
            attack_speed_limit_mps=130.0,
            speed_penalty_altitude_gate_m=30.0,
        )

        self.assertEqual(comps["speed_penalty"], 0.0)
        self.assertLess(comps["attack_speed_penalty"], 0.0)

    def test_lift_speed_penalty_when_below_enemy(self):
        prev_my = make_state((60.0, 0.0, 20.0), angles=(0.0, 0.0, 0.0), vel=(5.0, 0.0, 0.0))
        prev_enemy = make_state((80.0, 0.0, 30.0), health=50.0)
        slow_my = make_state((60.0, 0.0, 20.0), angles=(0.0, 0.0, 0.0), vel=(5.0, 0.0, 0.0))
        fast_my = make_state((60.0, 0.0, 20.0), angles=(0.0, 0.0, 0.0), vel=(12.0, 0.0, 0.0))
        enemy = make_state((80.0, 0.0, 30.0), health=50.0)

        slow = reward.reward_components(
            prev_my, prev_enemy, slow_my, enemy,
            min_lift_speed_mps=110.0, lift_speed_penalty_weight=0.02,
        )
        fast = reward.reward_components(
            prev_my, prev_enemy, fast_my, enemy,
            min_lift_speed_mps=110.0, lift_speed_penalty_weight=0.02,
        )

        self.assertLess(slow["lift_speed_penalty"], 0.0)
        self.assertEqual(fast["lift_speed_penalty"], 0.0)

    def test_overshoot_respects_margin_before_penalty(self):
        prev_my = make_state((0.0, 0.0, 30.0), angles=(0.0, 0.0, 0.0))
        prev_enemy = make_state((80.0, 0.0, 30.0), health=50.0)
        passed_my = make_state((82.0, 0.0, 30.0), angles=(0.0, 0.0, 0.0))
        enemy = make_state((80.0, 0.0, 30.0), health=50.0)

        within_margin = reward.reward_components(
            prev_my, prev_enemy, passed_my, enemy, overshoot_margin_m=-150.0
        )
        at_default_margin = reward.reward_components(prev_my, prev_enemy, passed_my, enemy)

        self.assertEqual(within_margin["overshoot"], 0.0)
        self.assertLess(at_default_margin["overshoot"], 0.0)


class InitializeTests(unittest.TestCase):
    def test_initial_state_has_two_aircraft_and_safe_start_distance(self):
        initial = initialize.generate_initial_state()

        self.assertEqual(initial.shape, (24,))
        my_pos = initial[0:3].astype(np.float64) * 10.0
        enemy_pos = initial[12:15].astype(np.float64) * 10.0
        self.assertGreaterEqual(np.linalg.norm(enemy_pos - my_pos), 1000.0)
        np.testing.assert_array_equal(initial[0:3], np.array([0, 0, 100]))
        np.testing.assert_array_equal(initial[12:15], np.array([120, 0, 100]))
        self.assertEqual(initial[6], 10)

    def test_enemy_y_range(self):
        ys = {
            int(initialize.generate_initial_state(combat_enemy_y_range=[-5, 5])[13])
            for _ in range(40)
        }
        self.assertTrue(ys.issubset(set(range(-5, 6))))
        self.assertGreater(len(ys), 1)

    def test_enemy_y_positive_prob_skews_toward_positive_side(self):
        pos = 0
        neg = 0
        for _ in range(400):
            y = int(
                initialize.generate_initial_state(
                    combat_enemy_y_range=[-5, 5],
                    combat_enemy_y_positive_prob=0.75,
                )[13]
            )
            if y > 0:
                pos += 1
            elif y < 0:
                neg += 1
        self.assertGreater(pos, neg)
        self.assertGreater(pos, 200)


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
