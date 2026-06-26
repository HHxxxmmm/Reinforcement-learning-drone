import numpy as np


def generate_initial_state():
    my_initial_state = np.zeros(12, dtype=np.int32)
    enemy_initial_state = np.zeros(12, dtype=np.int32)

    my_initial_state[0:3] = np.array([0, 0, 30], dtype=np.int32)
    my_initial_state[6:9] = np.array([30, 0, 0], dtype=np.int32)

    enemy_initial_state[0:3] = np.array([120, 0, 30], dtype=np.int32)
    initial_state = np.append(my_initial_state, enemy_initial_state)
    return initial_state
