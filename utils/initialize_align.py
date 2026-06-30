"""
Stage-1 initialization.

Platform: roll/pitch/yaw in initial packet are int32.
Typical stage-1: yaw=0, enemy lateral offset via random y → small LOS angle error.
"""
import numpy as np

DEFAULT_COMBAT_MY_ALT_UNIT = 100
DEFAULT_COMBAT_ENEMY_POS = (120, 0, 100)
DEFAULT_ENEMY_Y_RANGE = (-8, 8)
DEFAULT_START_X = 0
YAW_PM1_CHOICES = (-1, 1)


def _rng(rng):
    return np.random.default_rng() if rng is None else rng


def _normalize_enemy_pos(enemy_pos):
    if enemy_pos is None:
        return np.array(DEFAULT_COMBAT_ENEMY_POS, dtype=np.int32)
    pos = np.asarray(enemy_pos, dtype=np.int32).reshape(3)
    if pos.shape != (3,):
        raise ValueError(f"enemy_pos must have 3 elements, got {enemy_pos}")
    return pos


def _sample_enemy_y(enemy_y_range, rng):
    if enemy_y_range is None:
        return 0
    lo, hi = int(enemy_y_range[0]), int(enemy_y_range[1])
    if lo > hi:
        lo, hi = hi, lo
    return int(_rng(rng).integers(lo, hi + 1))


def _enemy_pos_with_y(enemy_pos, enemy_y_range, rng):
    out = _normalize_enemy_pos(enemy_pos).copy()
    out[1] = _sample_enemy_y(enemy_y_range, rng)
    return out


def resolve_initial_yaw(initial_yaw, rng=None):
    """initial_yaw: int, or 'random_pm1' / 'pm1' → uniform in {-1, +1}."""
    if initial_yaw in ("random_pm1", "pm1", "random"):
        return int(_rng(rng).choice(YAW_PM1_CHOICES))
    return int(initial_yaw)


def align_ta_v2_initial_state(
    enemy_pos=None,
    altitude_unit=None,
    enemy_y_range=None,
    rng=None,
    initial_yaw=0,
):
    alt = DEFAULT_COMBAT_MY_ALT_UNIT if altitude_unit is None else int(altitude_unit)
    enemy = _enemy_pos_with_y(enemy_pos, enemy_y_range, rng)
    yaw = resolve_initial_yaw(initial_yaw, rng)
    my = np.zeros(12, dtype=np.int32)
    enemy_state = np.zeros(12, dtype=np.int32)
    my[0:3] = np.array([0, 0, alt], dtype=np.int32)
    my[5] = yaw
    enemy_state[0:3] = enemy
    return my, enemy_state


def align_stage1_initial_state(
    enemy_pos=None,
    altitude_unit=None,
    enemy_y_range=None,
    rng=None,
    initial_yaw=0,
    start_x=DEFAULT_START_X,
    initial_pitch=0,
    initial_roll=0,
):
    alt = DEFAULT_COMBAT_MY_ALT_UNIT if altitude_unit is None else int(altitude_unit)
    enemy = _enemy_pos_with_y(enemy_pos, enemy_y_range, rng)
    yaw = resolve_initial_yaw(initial_yaw, rng)
    my = np.zeros(12, dtype=np.int32)
    enemy_state = np.zeros(12, dtype=np.int32)
    my[0:3] = np.array([int(start_x), 0, alt], dtype=np.int32)
    my[3] = int(initial_roll)
    my[4] = int(initial_pitch)
    my[5] = yaw
    enemy_state[0:3] = enemy
    return my, enemy_state


def align_v5_c1_initial_state(**kwargs):
    return align_stage1_initial_state(initial_yaw=0, start_x=0, **kwargs)


def align_v5_c2_initial_state(**kwargs):
    return align_stage1_initial_state(initial_yaw=2, start_x=0, **kwargs)


def los_misalignment_deg_from_y_offset(enemy_x_unit, enemy_y_unit):
    """yaw=0, nose +x: horizontal LOS error ≈ atan(|y|/x) degrees."""
    x = max(float(enemy_x_unit), 1e-8)
    return float(np.degrees(np.arctan(abs(float(enemy_y_unit)) / x)))


def generate_initial_state(
    init_mode="align_ta_v2",
    altitude_unit=DEFAULT_COMBAT_MY_ALT_UNIT,
    enemy_pos=None,
    enemy_y_range=DEFAULT_ENEMY_Y_RANGE,
    initial_yaw=0,
    start_x=DEFAULT_START_X,
    initial_pitch=0,
    initial_roll=0,
    rng=None,
    **_ignored,
):
    common = dict(
        enemy_pos=enemy_pos,
        altitude_unit=altitude_unit,
        enemy_y_range=enemy_y_range,
        rng=rng,
    )
    if init_mode == "align_ta_v2":
        my, enemy = align_ta_v2_initial_state(initial_yaw=initial_yaw, **common)
    elif init_mode == "align_stage1":
        my, enemy = align_stage1_initial_state(
            initial_yaw=initial_yaw,
            start_x=start_x,
            initial_pitch=initial_pitch,
            initial_roll=initial_roll,
            **common,
        )
    elif init_mode == "align_v5_c1":
        my, enemy = align_v5_c1_initial_state(**common)
    elif init_mode == "align_v5_c2":
        my, enemy = align_v5_c2_initial_state(**common)
    else:
        raise ValueError(f"unknown init_mode: {init_mode}")
    return np.append(my, enemy)


def nose_misalignment_deg_from_state(my_state, enemy_pos_unit):
    from utils.reward import _forward_vector

    my_pos = np.asarray(my_state[0:3], dtype=np.float64)
    enemy_pos_unit = np.asarray(enemy_pos_unit, dtype=np.float64)
    rel = enemy_pos_unit - my_pos
    los = rel / (np.linalg.norm(rel) + 1e-8)
    forward = _forward_vector(my_state)
    cos_angle = float(np.clip(np.dot(forward, los), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))
