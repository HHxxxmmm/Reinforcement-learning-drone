"""Partial policy reset helpers for curriculum hand-offs (e.g. P2 → P3 yaw fix)."""
import torch
from torch import nn


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
