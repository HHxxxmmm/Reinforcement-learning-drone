"""
Stage-1 reward: cosine monotonic — larger alignment cos → larger reward, then hold.
"""
import numpy as np

from utils.reward import _forward_vector, _position_m

EPS = 1e-8
DEFAULT_ALIGN_TARGET_DEG = 1.0
DEFAULT_ALIGN_COS_THRESHOLD = float(np.cos(np.deg2rad(DEFAULT_ALIGN_TARGET_DEG)))
COSINE_REWARD_WEIGHT = 5.0
DEFAULT_HOLD_STEPS_START = 8
DEFAULT_HOLD_INCREMENT = 2
DEFAULT_HOLD_EVERY_SUCCESSES = 5
DEFAULT_SUCCESS_BONUS_PER_HOLD_STEP = 10.0


def required_hold_steps(
    success_count,
    start=DEFAULT_HOLD_STEPS_START,
    increment=DEFAULT_HOLD_INCREMENT,
    every=DEFAULT_HOLD_EVERY_SUCCESSES,
    max_hold=None,
):
    """每成功 every 次，required hold +increment；max_hold 为 None 或 ≤0 时不设上限。"""
    level = int(success_count) // max(int(every), 1)
    required = int(start) + level * int(increment)
    if max_hold is not None:
        cap = int(max_hold)
        if cap > 0:
            required = min(cap, required)
    return required


def success_bonus_amount(required_hold_steps, per_hold_step=DEFAULT_SUCCESS_BONUS_PER_HOLD_STEP):
    return float(per_hold_step) * max(int(required_hold_steps), 1)


def align_cos_threshold_from_deg(deg):
    return float(np.cos(np.deg2rad(float(deg))))


def _los_to_enemy(my_state, enemy_state):
    rel_m = _position_m(enemy_state, enemy=True) - _position_m(my_state)
    distance = np.linalg.norm(rel_m)
    los = rel_m / (distance + EPS)
    return los, distance


def _alignment_cos(my_state, enemy_state):
    los, _ = _los_to_enemy(my_state, enemy_state)
    forward = _forward_vector(my_state)
    return float(np.clip(np.dot(forward, los), -1.0, 1.0))


def _clamp(value, low, high):
    return float(np.clip(value, low, high))


def reward_components(
    prev_my_state,
    prev_enemy_state,
    my_state,
    enemy_state,
    hold_steps=0,
    required_hold_steps=8,
    align_cos_threshold=DEFAULT_ALIGN_COS_THRESHOLD,
    cosine_weight=COSINE_REWARD_WEIGHT,
    episode_success=False,
    episode_timeout=False,
    success_bonus_per_hold_step=DEFAULT_SUCCESS_BONUS_PER_HOLD_STEP,
    timeout_penalty=-1.5,
    **_ignored,
):
    prev_cos = _alignment_cos(prev_my_state, prev_enemy_state)
    cos = _alignment_cos(my_state, enemy_state)
    angular_rate = float(np.linalg.norm(np.asarray(my_state[9:12], dtype=np.float64)))
    aligned = cos >= align_cos_threshold

    comps = {}
    # cos ∈ [-1, 1] → [0, weight]；严格随 cos 增大而增大
    comps["cosine"] = float(cosine_weight) * (cos + 1.0) * 0.5
    comps["cosine_progress"] = 3.0 * _clamp(cos - prev_cos, -0.15, 0.15)

    if aligned:
        comps["hold"] = 1.5 * min(1.0, hold_steps / max(required_hold_steps, 1))
        comps["tight"] = 1.0
        comps["stability"] = -0.03 * min(angular_rate, 3.0)
    else:
        comps["hold"] = 0.0
        comps["tight"] = 0.0
        comps["stability"] = -0.01 * min(angular_rate, 5.0)

    comps["step"] = -0.003
    comps["success_bonus"] = (
        success_bonus_amount(required_hold_steps, success_bonus_per_hold_step)
        if episode_success
        else 0.0
    )
    comps["timeout_penalty"] = float(timeout_penalty) if episode_timeout else 0.0
    comps["total"] = float(sum(comps.values()))
    return comps


def calculate_reward(
    prev_my_state,
    prev_enemy_state,
    my_state,
    enemy_state,
    **kwargs,
):
    return reward_components(
        prev_my_state, prev_enemy_state, my_state, enemy_state, **kwargs
    )["total"]


def alignment_cos(my_state, enemy_state):
    return _alignment_cos(my_state, enemy_state)
