import unittest
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from utils import observation
from utils.policy_reset import reset_policy_pitch_head, reset_policy_yaw_head


def make_state(pos, angles=(0.0, 0.0, 0.0), vel=(10.0, 0.0, 0.0), health=1.0):
    state = np.zeros(13, dtype=np.float64)
    state[0:3] = np.array(pos, dtype=np.float64)
    state[3:6] = np.array(angles, dtype=np.float64)
    state[6:9] = np.array(vel, dtype=np.float64)
    state[12] = health
    return state


class PolicyResetTests(unittest.TestCase):
    def test_reset_yaw_head_zeros_bias_and_preserves_throttle_row(self):
        ckpt = Path("model/stage2_phase2_v2/ppo_combat_p2_v2_5000_steps.zip")
        if not ckpt.is_file():
            self.skipTest(f"missing checkpoint: {ckpt}")

        model = PPO.load(str(ckpt), device="cpu")
        throttle_row = model.policy.action_net.weight[0].detach().clone()
        throttle_bias = float(model.policy.action_net.bias[0].item())

        info = reset_policy_yaw_head(model, yaw_action_idx=3, log_std_init=-0.5)

        self.assertAlmostEqual(info["yaw_bias"], 0.0, places=5)
        self.assertAlmostEqual(info["yaw_log_std"], -0.5, places=5)
        self.assertTrue(torch.allclose(model.policy.action_net.weight[0], throttle_row))
        self.assertAlmostEqual(float(model.policy.action_net.bias[0].item()), throttle_bias, places=5)
        self.assertAlmostEqual(float(model.policy.action_net.bias[3].item()), 0.0, places=5)

    def test_reset_yaw_head_clears_p2_left_bias_on_p2_checkpoint(self):
        ckpt = Path("model/stage2_phase2_v2/ppo_combat_p2_v2_5000_steps.zip")
        if not ckpt.is_file():
            self.skipTest(f"missing checkpoint: {ckpt}")

        observation.set_enemy_fallback_position(np.array([120.0, 0.0, 100.0]))
        model = PPO.load(str(ckpt), device="cpu")
        before, _ = model.predict(
            observation.marshal_observation(make_state([0, 0, 100]), make_state([120, 0, 100])),
            deterministic=True,
        )
        reset_policy_yaw_head(model, log_std_init=-0.5)
        after_vals = []
        for ey in (-5, 0, 5):
            obs = observation.marshal_observation(
                make_state([0, 0, 100]),
                make_state([120, ey, 100]),
            )
            action, _ = model.predict(obs, deterministic=True)
            after_vals.append(float(action[3]))

        self.assertLess(float(before[3]), -0.05)
        self.assertLess(max(abs(y) for y in after_vals), 0.05)

    def test_reset_pitch_head_zeros_bias_and_preserves_throttle_row(self):
        ckpt = Path("model/stage2_phase2_v3_nocap/ppo_combat_p2_v3_nocap_1000_steps.zip")
        if not ckpt.is_file():
            self.skipTest(f"missing checkpoint: {ckpt}")

        model = PPO.load(str(ckpt), device="cpu")
        throttle_row = model.policy.action_net.weight[0].detach().clone()
        yaw_row = model.policy.action_net.weight[3].detach().clone()

        info = reset_policy_pitch_head(model, pitch_action_idx=1, log_std_init=-0.5)

        self.assertAlmostEqual(info["pitch_bias"], 0.0, places=5)
        self.assertAlmostEqual(info["pitch_log_std"], -0.5, places=5)
        self.assertTrue(torch.allclose(model.policy.action_net.weight[0], throttle_row))
        self.assertTrue(torch.allclose(model.policy.action_net.weight[3], yaw_row))
        self.assertAlmostEqual(float(model.policy.action_net.bias[1].item()), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
