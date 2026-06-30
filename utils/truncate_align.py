"""Stage-1 truncation: only safety checks, no separation penalty."""
import numpy as np

MIN_ALTITUDE_UNIT = 1.0
EPS = 1e-8


def check_truncation(my_state, enemy_state):
    my_pos = np.asarray(my_state[0:3], dtype=np.float64)
    if not np.all(np.isfinite(my_pos)):
        return True
    if my_pos[2] < MIN_ALTITUDE_UNIT:
        return True
    return False
