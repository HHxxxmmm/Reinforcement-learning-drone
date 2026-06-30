"""Partial policy reset helpers for curriculum hand-offs (e.g. P2 → P3 yaw fix)."""
import torch
from torch import nn


"""Partial policy reset helpers for curriculum hand-offs (e.g. P2 → P3 yaw fix)."""
import torch
from torch import nn


def reset_policy_junior_p1_head(
    model,
    throttle_action_idx=0,
    pitch_action_idx=1,
    max_throttle=0.46,
    action_scale=0.50,
    target_throttle_frac=0.88,
    target_pitch_cmd=0.12,
):
    """
    Bias throttle/pitch means toward junior trim (probes: thr~0.50, pitch~+0.12).

    Untrained PPO mean≈0 → throttle≈max/2 (~0.23), far below level-flight needs;
    aircraft dives with speed stuck near initial_speed.
    """
    policy = model.policy
    action_net = policy.action_net
    thr_idx = int(throttle_action_idx)
    pit_idx = int(pitch_action_idx)
    cap = max(0.0, min(1.0, float(max_throttle)))
    scale = max(0.05, min(1.0, float(action_scale)))
    target_thr = max(0.0, min(1.0, float(target_throttle_frac))) * cap
    thr_action = 2.0 * target_thr / cap - 1.0 if cap > 0 else -1.0
    thr_action = max(-1.0, min(1.0, thr_action))
    pit_action = target_pitch_cmd / scale if scale > 0 else 0.0
    pit_action = max(-1.0, min(1.0, pit_action))

    with torch.no_grad():
        action_net.bias[thr_idx] = thr_action
        action_net.bias[pit_idx] = pit_action

    return {
        "throttle_action_idx": thr_idx,
        "pitch_action_idx": pit_idx,
        "throttle_bias": float(thr_action),
        "pitch_bias": float(pit_action),
        "target_throttle": float(target_thr),
        "target_pitch_cmd": float(target_pitch_cmd),
    }


def reset_policy_yaw_head(
    model,
    yaw_action_idx=3,
    log_std_init=None,
):
    """
    Re-initialize only the policy mean head row for yaw while keeping
    throttle / pitch / roll rows and all value / shared layers intact.

    Use when resuming from a checkpoint that locked yaw (P1) so action[yaw]
    never received meaningful gradients but other dims are worth keeping.
    """
    policy = model.policy
    action_net = policy.action_net
    idx = int(yaw_action_idx)
    if idx < 0 or idx >= action_net.out_features:
        raise ValueError(f"yaw_action_idx={idx} out of range for action_dim={action_net.out_features}")

    with torch.no_grad():
        nn.init.orthogonal_(action_net.weight[idx : idx + 1], gain=0.01)
        action_net.bias[idx].zero_()
        if log_std_init is not None and hasattr(policy, "log_std"):
            policy.log_std[idx] = float(log_std_init)

    bias = float(action_net.bias[idx].item())
    log_std = float(policy.log_std[idx].item()) if hasattr(policy, "log_std") else None
    return {"yaw_bias": bias, "yaw_log_std": log_std, "yaw_action_idx": idx}
