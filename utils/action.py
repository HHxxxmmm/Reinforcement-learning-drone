import numpy as np


def marshal_action(
    action,
    lock_throttle=False,
    fixed_throttle=0.0,
    max_throttle=1.0,
    lock_roll=False,
    lock_pitch=False,
    fixed_pitch=0.0,
    max_pitch=1.0,
    lock_yaw=False,
    fixed_yaw=0.0,
    max_yaw=1.0,
    action_scale=1.0,
):
    """
    Map network output (4 values in [-1, 1]) to real control commands.

    Network output order  │  real control  │  real range
    ───────────────────────┼─────────────────┼──────────────
    action[0]  throttle    │  throttle        │  [0, max_throttle] or fixed
    action[1]  pitch       │  pitch           │  [-1, 1]
    action[2]  roll        │  roll            │  [-1, 1]
    action[3]  yaw         │  yaw             │  [-max_yaw, max_yaw] or fixed

    The returned array is passed to pack_action() → adaptor.send_action_packet().
    """
    action = np.asarray(action, dtype=np.float64)

    real_action = np.zeros(4, dtype=np.float64)

    throttle_cap = float(np.clip(max_throttle, 0.0, 1.0))
    pitch_cap = float(np.clip(max_pitch, 0.0, 1.0))
    yaw_cap = float(np.clip(max_yaw, 0.0, 1.0))

    if lock_throttle:
        real_action[0] = np.clip(float(fixed_throttle), 0.0, throttle_cap)
    else:
        # throttle:  [-1, 1]  →  [0, max_throttle]
        real_action[0] = np.clip((action[0] + 1.0) / 2.0 * throttle_cap, 0.0, throttle_cap)

    # pitch, roll, yaw:  [-1, 1]  →  [-1, 1]  (identity with safety clip)
    scale = float(np.clip(action_scale, 0.05, 1.0))
    if lock_pitch:
        real_action[1] = float(np.clip(fixed_pitch, -pitch_cap, pitch_cap))
    else:
        real_action[1] = np.clip(action[1] * scale, -pitch_cap, pitch_cap)
    real_action[2] = 0.0 if lock_roll else np.clip(action[2] * scale, -1.0, 1.0)
    if lock_yaw:
        real_action[3] = float(np.clip(fixed_yaw, -yaw_cap, yaw_cap))
    else:
        real_action[3] = np.clip(action[3] * scale, -yaw_cap, yaw_cap)

    return real_action
