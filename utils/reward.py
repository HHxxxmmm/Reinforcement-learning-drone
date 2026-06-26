import numpy as np


POSITION_SCALE_M = 10.0
MAX_HP = 1000.0
HIT_DAMAGE_HP = 10.0
ATTACK_MIN_RANGE_M = 60.0
ATTACK_MAX_RANGE_M = 660.0
ATTACK_HALF_WIDTH_M = 10.0
EPS = 1e-8


def _position_m(state):
    return np.asarray(state[0:3], dtype=np.float64) * POSITION_SCALE_M


def _health(state):
    return float(state[12]) if len(state) > 12 else MAX_HP


def _forward_vector(state):
    # State angles are roll, pitch, yaw. The aircraft nose direction only
    # needs pitch/yaw for this simple shaping term.
    pitch = float(state[4])
    yaw = float(state[5])
    cp = np.cos(pitch)
    forward = np.array([cp * np.cos(yaw), cp * np.sin(yaw), np.sin(pitch)])
    return forward / (np.linalg.norm(forward) + EPS)


def _range_and_los(my_state, enemy_state):
    rel = _position_m(enemy_state) - _position_m(my_state)
    distance = np.linalg.norm(rel)
    los = rel / (distance + EPS)
    return distance, los, rel


def _clamp(value, low, high):
    return float(np.clip(value, low, high))


# This is the reward calculation function. We provide current state and previous state for you.
def reward_components(prev_my_state, prev_enemy_state, my_state, enemy_state):
    prev_distance, _, _ = _range_and_los(prev_my_state, prev_enemy_state)
    distance, los, rel = _range_and_los(my_state, enemy_state)
    forward = _forward_vector(my_state)

    closing_m = prev_distance - distance
    alignment_cos = float(np.dot(forward, los))
    forward_distance = float(np.dot(rel, forward))
    lateral_error = np.linalg.norm(rel - forward_distance * forward)
    enemy_damage = max(0.0, _health(prev_enemy_state) - _health(enemy_state))
    self_damage = max(0.0, _health(prev_my_state) - _health(my_state))

    comps = {}
    comps["distance_progress"] = _clamp(closing_m / 50.0, -2.0, 2.0)
    comps["proximity"] = 0.6 * np.exp(-distance / 1200.0)
    comps["alignment"] = 1.2 * alignment_cos
    if ATTACK_MIN_RANGE_M <= forward_distance <= ATTACK_MAX_RANGE_M:
        lateral_score = _clamp(1.0 - lateral_error / ATTACK_HALF_WIDTH_M, -1.0, 1.0)
        comps["attack_box"] = 2.0 * max(0.0, alignment_cos) * lateral_score
    else:
        comps["attack_box"] = -0.2 * max(0.0, alignment_cos)
    if forward_distance > 0.0:
        comps["corridor"] = 1.6 * max(0.0, alignment_cos) * np.exp(-lateral_error / 20.0)
    else:
        comps["corridor"] = -0.4
    comps["enemy_damage"] = 20.0 * (enemy_damage / HIT_DAMAGE_HP)
    comps["self_damage"] = -20.0 * (self_damage / HIT_DAMAGE_HP)
    comps["survival"] = -0.02
    comps["kill_bonus"] = 300.0 if _health(prev_enemy_state) > 0.0 and _health(enemy_state) <= 0.0 else 0.0
    comps["death_penalty"] = -300.0 if _health(prev_my_state) > 0.0 and _health(my_state) <= 0.0 else 0.0

    comps["total"] = float(sum(comps.values()))
    return comps


def calculate_reward(prev_my_state, prev_enemy_state, my_state, enemy_state):
    return reward_components(prev_my_state, prev_enemy_state, my_state, enemy_state)["total"]