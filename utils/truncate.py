import numpy as np


POSITION_SCALE_M = 10.0
MIN_ALTITUDE_UNIT = 1.0
MAX_SEPARATION_M = 5000.0
MAX_ABS_POSITION_UNIT = 1000.0


def check_truncation(my_state, enemy_state):
    my_pos = np.asarray(my_state[0:3], dtype=np.float64)
    enemy_pos = np.asarray(enemy_state[0:3], dtype=np.float64)
    separation_m = np.linalg.norm((enemy_pos - my_pos) * POSITION_SCALE_M)

    if not np.all(np.isfinite(my_pos)) or not np.all(np.isfinite(enemy_pos)):
        return True
    if my_pos[2] < MIN_ALTITUDE_UNIT:
        return True
    if separation_m > MAX_SEPARATION_M:
        return True
    if np.max(np.abs(my_pos)) > MAX_ABS_POSITION_UNIT:
        return True
    return False
