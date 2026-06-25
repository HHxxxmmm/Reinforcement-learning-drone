import numpy as np


def generate_initial_state():
    rng = np.random.default_rng()

    my_initial_state = np.zeros(12, dtype=np.int32)
    enemy_initial_state = np.zeros(12, dtype=np.int32)

    my_initial_state[0:3] = np.array([0, 0, 30], dtype=np.int32)
    my_initial_state[6:9] = np.array([30, 0, 0], dtype=np.int32)

    enemy_initial_state[0] = rng.integers(120, 181)
    enemy_initial_state[1] = rng.integers(-25, 26)
    enemy_initial_state[2] = rng.integers(24, 37)
    initial_state = np.append(my_initial_state, enemy_initial_state)
    return initial_state
