import unittest

import numpy as np

from utils import initialize_align, reward_align


def make_state(pos, angles=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), health=1.0):
    state = np.zeros(13, dtype=np.float64)
    state[0:3] = np.array(pos, dtype=np.float64)
    state[3:6] = np.array(angles, dtype=np.float64)
    state[6:9] = np.array(vel, dtype=np.float64)
    state[12] = health
    return state


class AlignInitializeTests(unittest.TestCase):
    def test_random_pm1_yaw_is_plus_or_minus_one(self):
        yaws = {
            int(initialize_align.generate_initial_state(initial_yaw="random_pm1")[5])
            for _ in range(30)
        }
        self.assertEqual(yaws, {-1, 1})

    def test_resolve_initial_yaw_fixed(self):
        self.assertEqual(initialize_align.resolve_initial_yaw(1), 1)
        self.assertEqual(initialize_align.resolve_initial_yaw(-1), -1)

    def test_y_lateral_offset_gives_small_misalignment_with_yaw_zero(self):
        my = make_state((0.0, 0.0, 100.0), angles=(0.0, 0.0, 0.0))
        enemy = make_state((120.0, 8.0, 100.0))
        mis = initialize_align.nose_misalignment_deg_from_state(
            my, np.array([120.0, 8.0, 100.0])
        )
        self.assertAlmostEqual(mis, initialize_align.los_misalignment_deg_from_y_offset(120, 8), delta=0.5)
        self.assertGreater(mis, 2.0)
        self.assertLess(mis, 6.0)

    def test_enemy_y_range_eight(self):
        ys = {
            int(initialize_align.generate_initial_state(
                initial_yaw=0, enemy_y_range=[-8, 8]
            )[13])
            for _ in range(50)
        }
        self.assertGreater(len(ys), 1)
        self.assertTrue(ys.issubset(set(range(-8, 9))))


class AlignRewardTests(unittest.TestCase):
    def test_higher_cosine_gets_higher_reward(self):
        enemy = make_state((120.0, 0.0, 100.0))
        my_low = make_state((0.0, 0.0, 100.0), angles=(0.0, 0.0, 0.3))
        my_high = make_state((0.0, 0.0, 100.0), angles=(0.0, 0.0, 0.05))

        low = reward_align.reward_components(my_low, enemy, my_low, enemy)
        high = reward_align.reward_components(my_high, enemy, my_high, enemy)

        self.assertGreater(high["cosine"], low["cosine"])
        self.assertGreater(high["total"], low["total"])

    def test_tight_bonus_when_within_one_degree(self):
        enemy = make_state((120.0, 0.0, 100.0))
        loose = make_state((0.0, 0.0, 100.0), angles=(0.0, 0.0, np.deg2rad(3.8)))
        tight = make_state((0.0, 0.0, 100.0), angles=(0.0, 0.0, np.deg2rad(0.5)))
        threshold = reward_align.align_cos_threshold_from_deg(1.0)

        loose_c = reward_align.reward_components(loose, enemy, loose, enemy, align_cos_threshold=threshold)
        tight_c = reward_align.reward_components(tight, enemy, tight, enemy, align_cos_threshold=threshold)

        self.assertLess(reward_align.alignment_cos(loose, enemy), threshold)
        self.assertGreaterEqual(reward_align.alignment_cos(tight, enemy), threshold)
        self.assertEqual(loose_c["tight"], 0.0)
        self.assertGreater(tight_c["tight"], 0.0)

    def test_success_bonus_only_on_success_flag(self):
        my = make_state((0.0, 0.0, 100.0), angles=(0.0, 0.0, 0.0))
        enemy = make_state((120.0, 0.0, 100.0))
        success = reward_align.reward_components(
            my, enemy, my, enemy, episode_success=True, required_hold_steps=8
        )
        plain = reward_align.reward_components(my, enemy, my, enemy, episode_success=False)

        self.assertGreater(success["success_bonus"], 0.0)
        self.assertEqual(plain["success_bonus"], 0.0)
        self.assertAlmostEqual(success["success_bonus"], 80.0)

    def test_success_bonus_scales_with_required_hold(self):
        my = make_state((0.0, 0.0, 100.0), angles=(0.0, 0.0, 0.0))
        enemy = make_state((120.0, 0.0, 100.0))
        short = reward_align.reward_components(
            my, enemy, my, enemy, episode_success=True, required_hold_steps=8
        )
        long = reward_align.reward_components(
            my, enemy, my, enemy, episode_success=True, required_hold_steps=20
        )
        self.assertAlmostEqual(short["success_bonus"], 80.0)
        self.assertAlmostEqual(long["success_bonus"], 200.0)

    def test_hold_curriculum_no_cap(self):
        self.assertEqual(reward_align.required_hold_steps(0), 8)
        self.assertEqual(reward_align.required_hold_steps(4), 8)
        self.assertEqual(reward_align.required_hold_steps(5), 10)
        self.assertEqual(reward_align.required_hold_steps(10), 12)
        self.assertEqual(reward_align.required_hold_steps(25), 18)
        capped = reward_align.required_hold_steps(100, max_hold=40)
        self.assertEqual(capped, 40)

    def test_timeout_penalty_on_flag(self):
        my = make_state((0.0, 0.0, 100.0), angles=(0.0, 0.0, 0.0))
        enemy = make_state((120.0, 0.0, 100.0))
        timed_out = reward_align.reward_components(my, enemy, my, enemy, episode_timeout=True)
        plain = reward_align.reward_components(my, enemy, my, enemy, episode_timeout=False)
        self.assertLess(timed_out["timeout_penalty"], 0.0)
        self.assertEqual(plain["timeout_penalty"], 0.0)


if __name__ == "__main__":
    unittest.main()
