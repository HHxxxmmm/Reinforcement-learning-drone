"""Partial policy reset helpers for curriculum hand-offs (e.g. P2 → P3 yaw fix)."""
import torch
from torch import nn


def reset_policy_action_head(
    model,
    action_idx,
    log_std_init=None,
):
    """Re-initialize one policy mean head row; keep other action dims and value net."""
    policy = model.policy
    action_net = policy.action_net
    idx = int(action_idx)
    if idx < 0 or idx >= action_net.out_features:
        raise ValueError(f"action_idx={idx} out of range for action_dim={action_net.out_features}")

    with torch.no_grad():
        nn.init.orthogonal_(action_net.weight[idx : idx + 1], gain=0.01)
        action_net.bias[idx].zero_()
        if log_std_init is not None and hasattr(policy, "log_std"):
            policy.log_std[idx] = float(log_std_init)

    bias = float(action_net.bias[idx].item())
    log_std = float(policy.log_std[idx].item()) if hasattr(policy, "log_std") else None
    return {"bias": bias, "log_std": log_std, "action_idx": idx}


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
    info = reset_policy_action_head(model, yaw_action_idx, log_std_init=log_std_init)
    return {
        "yaw_bias": info["bias"],
        "yaw_log_std": info["log_std"],
        "yaw_action_idx": info["action_idx"],
    }


def reset_policy_pitch_head(
    model,
    pitch_action_idx=1,
    log_std_init=None,
):
    """
    Re-initialize only the policy mean head row for pitch while keeping
    throttle / roll / yaw rows and all value / shared layers intact.

    Use when unlocking pitch after it was locked (P1/P2) so action[pitch]
    never received meaningful gradients.
    """
    info = reset_policy_action_head(model, pitch_action_idx, log_std_init=log_std_init)
    return {
        "pitch_bias": info["bias"],
        "pitch_log_std": info["log_std"],
        "pitch_action_idx": info["action_idx"],
    }
