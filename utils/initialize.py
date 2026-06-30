"""
Combat / shared initialization (ported from project root initialize.py).

Stage-2 default: my (0,0,100) speed 10 → enemy (120, y, 100).
"""
import numpy as np

DEFAULT_COMBAT_INITIAL_SPEED = 10
DEFAULT_COMBAT_MY_ALT_UNIT = 100
DEFAULT_COMBAT_ENEMY_POS = (120, 0, 100)


def _normalize_enemy_pos(combat_enemy_pos):
    if combat_enemy_pos is None:
        return np.array(DEFAULT_COMBAT_ENEMY_POS, dtype=np.int32)
    pos = np.asarray(combat_enemy_pos, dtype=np.int32).reshape(3)
    if pos.shape != (3,):
        raise ValueError(f"combat_enemy_pos must have 3 elements, got {combat_enemy_pos}")
    return pos


def _sample_enemy_y(combat_enemy_y_range, rng, combat_enemy_y_positive_prob=None):
    if combat_enemy_y_range is None:
        return 0
    lo, hi = int(combat_enemy_y_range[0]), int(combat_enemy_y_range[1])
    if lo > hi:
        lo, hi = hi, lo
    if lo == hi:
        return lo
    rng = np.random.default_rng() if rng is None else rng
    if combat_enemy_y_positive_prob is not None:
        p_pos = float(np.clip(combat_enemy_y_positive_prob, 0.0, 1.0))
        pos_lo = max(1, lo) if hi >= 1 else lo
        pos_hi = hi
        neg_lo = lo
        neg_hi = min(-1, hi) if lo <= -1 else hi
        has_pos = pos_lo <= pos_hi
        has_neg = neg_lo <= neg_hi
        if has_pos and has_neg:
            if rng.random() < p_pos:
                return int(rng.integers(pos_lo, pos_hi + 1))
            return int(rng.integers(neg_lo, neg_hi + 1))
        if has_pos and rng.random() < p_pos:
            return int(rng.integers(pos_lo, pos_hi + 1))
        if has_neg:
            return int(rng.integers(neg_lo, neg_hi + 1))
    return int(rng.integers(lo, hi + 1))


def _enemy_pos_with_y(combat_enemy_pos, combat_enemy_y_range, rng, combat_enemy_y_positive_prob=None):
    enemy_pos = _normalize_enemy_pos(combat_enemy_pos)
    enemy_pos = enemy_pos.copy()
    enemy_pos[1] = _sample_enemy_y(
        combat_enemy_y_range, rng, combat_enemy_y_positive_prob=combat_enemy_y_positive_prob
    )
    return enemy_pos


def _combat_initial_state(
    combat_initial_speed=None,
    combat_enemy_pos=None,
    combat_my_alt_unit=None,
    combat_enemy_y_range=None,
    combat_enemy_y_positive_prob=None,
    rng=None,
):
    speed = DEFAULT_COMBAT_INITIAL_SPEED if combat_initial_speed is None else int(combat_initial_speed)
    alt = DEFAULT_COMBAT_MY_ALT_UNIT if combat_my_alt_unit is None else int(combat_my_alt_unit)
    enemy_pos = _enemy_pos_with_y(
        combat_enemy_pos,
        combat_enemy_y_range,
        rng,
        combat_enemy_y_positive_prob=combat_enemy_y_positive_prob,
    )
    my_initial_state = np.zeros(12, dtype=np.int32)
    enemy_initial_state = np.zeros(12, dtype=np.int32)

    my_initial_state[0:3] = np.array([0, 0, alt], dtype=np.int32)
    my_initial_state[6:9] = np.array([speed, 0, 0], dtype=np.int32)
    enemy_initial_state[0:3] = enemy_pos
    return my_initial_state, enemy_initial_state


def generate_initial_state(
    mode="combat",
    combat_initial_speed=None,
    combat_enemy_pos=None,
    combat_my_alt_unit=None,
    combat_enemy_y_range=None,
    combat_enemy_y_positive_prob=None,
    rng=None,
    **_ignored,
):
    if mode != "combat":
        raise ValueError(f"utils.initialize only supports mode=combat, got {mode!r}")
    my_initial_state, enemy_initial_state = _combat_initial_state(
        combat_initial_speed,
        combat_enemy_pos,
        combat_my_alt_unit,
        combat_enemy_y_range,
        combat_enemy_y_positive_prob,
        rng,
    )
    return np.append(my_initial_state, enemy_initial_state)
