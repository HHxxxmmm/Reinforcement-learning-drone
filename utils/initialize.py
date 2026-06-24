import numpy as np


def generate_initial_state():
    # 与 MATLAB get_my_initial.m 默认一致：双方全 0
    my_initial_state = np.zeros(12, dtype=np.int32)
    enemy_initial_state = np.zeros(12, dtype=np.int32)
    return np.concatenate([my_initial_state, enemy_initial_state])