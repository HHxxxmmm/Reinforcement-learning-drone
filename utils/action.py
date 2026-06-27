import numpy as np


def marshal_action(action):
    """
    Map network output (4 values in [-1, 1]) to real control commands.

    Network output order  │  real control  │  real range
    ───────────────────────┼─────────────────┼──────────────
    action[0]  throttle    │  throttle        │  [0, 1]
    action[1]  pitch       │  pitch           │  [-1, 1]
    action[2]  roll        │  roll            │  [-1, 1]
    action[3]  yaw         │  yaw             │  [-1, 1]

    The returned array is passed to pack_action() → adaptor.send_action_packet().
    """
    action = np.asarray(action, dtype=np.float64)

    real_action = np.zeros(4, dtype=np.float64)

    # throttle:  [-1, 1]  →  [0, 1]
    real_action[0] = np.clip((action[0] + 1.0) / 2.0, 0.0, 1.0)

    # pitch, roll, yaw:  [-1, 1]  →  [-1, 1]  (identity with safety clip)
    real_action[1] = np.clip(action[1], -1.0, 1.0)
    real_action[2] = np.clip(action[2], -1.0, 1.0)
    real_action[3] = np.clip(action[3], -1.0, 1.0)

    return real_action
