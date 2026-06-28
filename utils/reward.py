import numpy as np


POSITION_SCALE_M = 10.0
DEFAULT_HP = 1.0
NORMALIZED_HP_THRESHOLD = 1.5
NORMALIZED_HIT_DAMAGE_HP = 0.01
THOUSAND_POINT_HIT_DAMAGE_HP = 10.0
FIXED_TARGET_POS_UNIT = np.array([120.0, 0.0, 30.0])
ATTACK_MIN_RANGE_M = 60.0
ATTACK_MAX_RANGE_M = 660.0
ATTACK_HALF_WIDTH_M = 10.0
DAMAGE_REWARD_PER_HIT = 8.0
SELF_DAMAGE_PENALTY_PER_HIT = 14.0
ENEMY_HP_SHAPING_WEIGHT = 4.0
CENTERLINE_SCALE_M = 120.0
TURN_RATE_PENALTY_WEIGHT = 0.04
OVERSHOOT_PENALTY = 8.0
FINISH_RANGE_M = 250.0
FINISH_CENTERLINE_SCALE_M = 25.0
FINISH_SPEED_TARGET_MPS = 180.0
FINISH_SPEED_PENALTY_WEIGHT = 0.01
MAX_FINISH_SPEED_PENALTY = 4.0
EPS = 1e-8


def _position_m(state, enemy=False):
    pos = np.asarray(state[0:3], dtype=np.float64)
    if enemy and np.linalg.norm(pos) < EPS:
        pos = FIXED_TARGET_POS_UNIT
    return pos * POSITION_SCALE_M


def _health(state):
    return float(state[12]) if len(state) > 12 else DEFAULT_HP


def _hit_damage_scale(*states):
    max_hp = max(_health(state) for state in states)
    if max_hp <= NORMALIZED_HP_THRESHOLD:
        return NORMALIZED_HIT_DAMAGE_HP
    return THOUSAND_POINT_HIT_DAMAGE_HP


def _hp_fraction(state):
    hp = _health(state)
    if hp <= NORMALIZED_HP_THRESHOLD:
        return _clamp(hp, 0.0, 1.0)
    return _clamp(hp / 1000.0, 0.0, 1.0)


def _forward_vector(state):
    # State angles are roll, pitch, yaw. The aircraft nose direction only
    # needs pitch/yaw for this simple shaping term.
    pitch = float(state[4])
    yaw = float(state[5])
    cp = np.cos(pitch)
    forward = np.array([cp * np.cos(yaw), cp * np.sin(yaw), np.sin(pitch)])
    return forward / (np.linalg.norm(forward) + EPS)


def _range_and_los(my_state, enemy_state):
    rel = _position_m(enemy_state, enemy=True) - _position_m(my_state)
    distance = np.linalg.norm(rel)
    los = rel / (distance + EPS)
    return distance, los, rel


def _clamp(value, low, high):
    return float(np.clip(value, low, high))


def _attack_box_score(forward_distance, lateral_error, alignment_cos):
    if not (ATTACK_MIN_RANGE_M <= forward_distance <= ATTACK_MAX_RANGE_M):
        return -0.2 * max(0.0, alignment_cos), False
    lateral_score = _clamp(1.0 - lateral_error / ATTACK_HALF_WIDTH_M, -1.0, 1.0)
    score = 3.0 * max(0.0, alignment_cos) * lateral_score
    return score, lateral_error <= ATTACK_HALF_WIDTH_M and alignment_cos > 0.0


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
    enemy_hit_damage = _hit_damage_scale(prev_enemy_state, enemy_state)
    self_hit_damage = _hit_damage_scale(prev_my_state, my_state)
    angular_rate = np.linalg.norm(np.asarray(my_state[9:12], dtype=np.float64))
    speed_mps = np.linalg.norm(np.asarray(my_state[6:9], dtype=np.float64)) * POSITION_SCALE_M
    enemy_alive = _health(enemy_state) > 0.0

    comps = {}
    comps["distance_progress"] = _clamp(closing_m / 50.0, -2.0, 2.0)
    comps["proximity"] = 0.15 * np.exp(-distance / 1200.0)
    comps["alignment"] = 2.0 * alignment_cos
    attack_box_score, in_attack_box = _attack_box_score(forward_distance, lateral_error, alignment_cos)
    comps["attack_box"] = attack_box_score
    if forward_distance > 0.0:
        comps["corridor"] = 1.2 * max(0.0, alignment_cos) * np.exp(-lateral_error / 20.0)
    else:
        comps["corridor"] = -0.4
    if forward_distance > 0.0:
        comps["centerline"] = 1.8 * max(0.0, alignment_cos) * np.exp(-lateral_error / CENTERLINE_SCALE_M)
    else:
        comps["centerline"] = -0.4
    comps["turn_penalty"] = -TURN_RATE_PENALTY_WEIGHT * min(angular_rate, 5.0)
    comps["enemy_damage"] = DAMAGE_REWARD_PER_HIT * (enemy_damage / enemy_hit_damage) if in_attack_box else 0.0
    comps["self_damage"] = -SELF_DAMAGE_PENALTY_PER_HIT * (self_damage / self_hit_damage)
    comps["enemy_hp_shaping"] = ENEMY_HP_SHAPING_WEIGHT * max(0.0, _hp_fraction(prev_enemy_state) - _hp_fraction(enemy_state))
    comps["overshoot"] = -OVERSHOOT_PENALTY if enemy_alive and forward_distance < 0.0 else 0.0
    if enemy_alive and 0.0 < forward_distance < FINISH_RANGE_M:
        comps["finish_centerline"] = 4.0 * max(0.0, alignment_cos) * np.exp(-lateral_error / FINISH_CENTERLINE_SCALE_M)
        speed_excess = max(0.0, speed_mps - FINISH_SPEED_TARGET_MPS)
        comps["finish_speed_penalty"] = -min(MAX_FINISH_SPEED_PENALTY, FINISH_SPEED_PENALTY_WEIGHT * speed_excess)
    else:
        comps["finish_centerline"] = 0.0
        comps["finish_speed_penalty"] = 0.0
    comps["survival"] = -0.02
    comps["kill_bonus"] = 300.0 if _health(prev_enemy_state) > 0.0 and _health(enemy_state) <= 0.0 else 0.0
    comps["death_penalty"] = -300.0 if _health(prev_my_state) > 0.0 and _health(my_state) <= 0.0 else 0.0

    comps["total"] = float(sum(comps.values()))
    return comps


def calculate_reward(prev_my_state, prev_enemy_state, my_state, enemy_state):
    return reward_components(prev_my_state, prev_enemy_state, my_state, enemy_state)["total"]
