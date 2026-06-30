"""Heuristic orientation setup for stage-1 when the platform ignores initial euler angles."""
import numpy as np

from utils.reward import _forward_vector


def _los_unit(my_state, enemy_pos_unit):
    my_pos = np.asarray(my_state[0:3], dtype=np.float64)
    rel = np.asarray(enemy_pos_unit, dtype=np.float64) - my_pos
    norm = np.linalg.norm(rel)
    return rel / (norm + 1e-8)


def misalignment_deg(my_state, enemy_pos_unit):
    from utils.initialize_align import nose_misalignment_deg_from_state

    return nose_misalignment_deg_from_state(my_state, enemy_pos_unit)


def steer_action(my_state, enemy_pos_unit, toward_los=True, gain=1.2):
    los = _los_unit(my_state, enemy_pos_unit)
    fwd = _forward_vector(my_state)
    axis = np.cross(fwd, los) if toward_los else np.cross(los, fwd)
    norm = np.linalg.norm(axis)
    if norm < 1e-6:
        return 0.0, 0.0
    axis = axis / norm
    pitch = float(np.clip(-axis[1] * gain, -1.0, 1.0))
    yaw = float(np.clip(axis[2] * gain, -1.0, 1.0))
    return pitch, yaw


def setup_misalignment_deg(
    my_state,
    enemy_pos_unit,
    target_deg=30.0,
    tolerance_deg=5.0,
    max_steps=200,
    send_action_fn=None,
    recv_obs_fn=None,
    marshal_action_fn=None,
):
    """
    Send heuristic pitch/yaw until nose-LOS angle is near target_deg.
    Returns final (my_state, enemy_state, steps_used).
    """
    if send_action_fn is None or recv_obs_fn is None or marshal_action_fn is None:
        raise ValueError("setup requires send_action_fn, recv_obs_fn, marshal_action_fn")

    from envs.train_env import pack_action, split_observation

    enemy_pos_unit = np.asarray(enemy_pos_unit, dtype=np.float64)
    enemy_state = np.zeros(13, dtype=np.float64)
    steps_used = 0

    for step in range(max_steps):
        angle = misalignment_deg(my_state, enemy_pos_unit)
        if abs(angle - target_deg) <= tolerance_deg:
            steps_used = step
            break

        if angle > target_deg:
            pitch, yaw = steer_action(my_state, enemy_pos_unit, toward_los=True, gain=1.4)
        else:
            pitch, yaw = steer_action(my_state, enemy_pos_unit, toward_los=False, gain=1.0)

        agent_action = np.array([0.0, pitch, 0.0, yaw], dtype=np.float64)
        real_action = marshal_action_fn(agent_action)
        send_action_fn(pack_action(real_action, False))
        obs = recv_obs_fn()
        my_state, enemy_state, _ = split_observation(obs)
        steps_used = step + 1

    return my_state, enemy_state, steps_used
