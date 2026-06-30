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
YAW_MISALIGN_WEIGHT = 0.0
ALIGNMENT_REWARD_WEIGHT = 2.0
ATTACK_BOX_REWARD_WEIGHT = 3.0
CORRIDOR_REWARD_WEIGHT = 1.2
CENTERLINE_REWARD_WEIGHT = 1.8
FINISH_CENTERLINE_REWARD_WEIGHT = 4.0
KILL_BONUS = 300.0
SELF_DAMAGE_PENALTY_PER_HIT = 14.0
ENEMY_HP_SHAPING_WEIGHT = 4.0
CENTERLINE_SCALE_M = 120.0
DISTANCE_PROGRESS_SCALE_M = 150.0
DISTANCE_PROGRESS_CLAMP = 0.5
TURN_RATE_PENALTY_WEIGHT = 0.04
OVERSHOOT_PENALTY = 8.0
FINISH_RANGE_M = 1200.0
FINISH_CENTERLINE_SCALE_M = 25.0
SPEED_APPROACH_RANGE_M = 1200.0
SPEED_DESIRED_MIN_MPS = 50.0
SPEED_DESIRED_DIST_SCALE = 4.0
SPEED_CRUISE_CAP_MPS = 180.0
SPEED_PENALTY_WEIGHT = 0.025
ATTACK_SPEED_LIMIT_MPS = 250.0
ATTACK_SPEED_PENALTY_WEIGHT = 8.0
ATTACK_SPEED_PENALTY_SCALE_MPS = 200.0
ALTITUDE_MATCH_WEIGHT = 0.0
ALTITUDE_MATCH_SCALE_M = 30.0
SPEED_PENALTY_ALTITUDE_GATE_M = 35.0
MIN_LIFT_SPEED_MPS = 110.0
ATTACK_SPEED_BONUS_WEIGHT = 0.0
ATTACK_SPEED_TARGET_MPS = 100.0
ATTACK_SPEED_BONUS_WIDTH_MPS = 45.0
LIFT_SPEED_PENALTY_WEIGHT = 0.0
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


def _attack_box_score(forward_distance, lateral_error, alignment_cos, attack_box_weight=ATTACK_BOX_REWARD_WEIGHT):
    if not (ATTACK_MIN_RANGE_M <= forward_distance <= ATTACK_MAX_RANGE_M):
        return -0.2 * max(0.0, alignment_cos), False
    lateral_score = _clamp(1.0 - lateral_error / ATTACK_HALF_WIDTH_M, -1.0, 1.0)
    score = float(attack_box_weight) * max(0.0, alignment_cos) * lateral_score
    return score, lateral_error <= ATTACK_HALF_WIDTH_M and alignment_cos > 0.0


# This is the reward calculation function. We provide current state and previous state for you.
def reward_components(
    prev_my_state,
    prev_enemy_state,
    my_state,
    enemy_state,
    overshoot_margin_m=0.0,
    yaw_misalign_weight=None,
    damage_reward_per_hit=None,
    alignment_weight=None,
    attack_box_weight=None,
    corridor_weight=None,
    centerline_weight=None,
    finish_centerline_weight=None,
    enemy_hp_shaping_weight=None,
    kill_bonus=None,
    altitude_match_weight=None,
    altitude_match_scale_m=None,
    altitude_below_penalty_weight=None,
    speed_penalty_weight=None,
    attack_speed_penalty_weight=None,
    speed_approach_range_m=None,
    speed_desired_min_mps=None,
    speed_desired_dist_scale=None,
    speed_cruise_cap_mps=None,
    speed_cruise_penalty_weight=None,
    attack_speed_limit_mps=None,
    attack_speed_penalty_scale_mps=None,
    speed_penalty_altitude_gate_m=None,
    min_lift_speed_mps=None,
    lift_speed_penalty_weight=None,
    attack_dwell_reward_weight=None,
    attack_speed_bonus_weight=None,
    attack_speed_target_mps=None,
    attack_speed_bonus_width_mps=None,
    low_hp_attack_bonus_weight=None,
    low_hp_attack_threshold=None,
    approach_slow_penalty_weight=None,
    approach_slow_min_mps=None,
    approach_slow_range_m=None,
):
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
    misalign_weight = float(YAW_MISALIGN_WEIGHT if yaw_misalign_weight is None else yaw_misalign_weight)
    hit_reward = float(DAMAGE_REWARD_PER_HIT if damage_reward_per_hit is None else damage_reward_per_hit)
    align_w = float(ALIGNMENT_REWARD_WEIGHT if alignment_weight is None else alignment_weight)
    atk_box_w = float(ATTACK_BOX_REWARD_WEIGHT if attack_box_weight is None else attack_box_weight)
    corridor_w = float(CORRIDOR_REWARD_WEIGHT if corridor_weight is None else corridor_weight)
    centerline_w = float(CENTERLINE_REWARD_WEIGHT if centerline_weight is None else centerline_weight)
    finish_cl_w = float(FINISH_CENTERLINE_REWARD_WEIGHT if finish_centerline_weight is None else finish_centerline_weight)
    hp_shape_w = float(ENEMY_HP_SHAPING_WEIGHT if enemy_hp_shaping_weight is None else enemy_hp_shaping_weight)
    kill_bonus_val = float(KILL_BONUS if kill_bonus is None else kill_bonus)
    alt_match_w = float(ALTITUDE_MATCH_WEIGHT if altitude_match_weight is None else altitude_match_weight)
    alt_match_scale = float(ALTITUDE_MATCH_SCALE_M if altitude_match_scale_m is None else altitude_match_scale_m)
    alt_below_w = float(0.0 if altitude_below_penalty_weight is None else altitude_below_penalty_weight)
    speed_pen_w = float(SPEED_PENALTY_WEIGHT if speed_penalty_weight is None else speed_penalty_weight)
    atk_speed_pen_w = float(
        ATTACK_SPEED_PENALTY_WEIGHT if attack_speed_penalty_weight is None else attack_speed_penalty_weight
    )
    approach_range_m = float(
        SPEED_APPROACH_RANGE_M if speed_approach_range_m is None else speed_approach_range_m
    )
    desired_min_mps = float(SPEED_DESIRED_MIN_MPS if speed_desired_min_mps is None else speed_desired_min_mps)
    desired_dist_scale = float(
        SPEED_DESIRED_DIST_SCALE if speed_desired_dist_scale is None else speed_desired_dist_scale
    )
    cruise_cap_mps = float(SPEED_CRUISE_CAP_MPS if speed_cruise_cap_mps is None else speed_cruise_cap_mps)
    cruise_pen_w = float(0.0 if speed_cruise_penalty_weight is None else speed_cruise_penalty_weight)
    atk_speed_limit_mps = float(
        ATTACK_SPEED_LIMIT_MPS if attack_speed_limit_mps is None else attack_speed_limit_mps
    )
    atk_speed_scale_mps = float(
        ATTACK_SPEED_PENALTY_SCALE_MPS
        if attack_speed_penalty_scale_mps is None
        else attack_speed_penalty_scale_mps
    )
    speed_alt_gate_m = float(
        SPEED_PENALTY_ALTITUDE_GATE_M if speed_penalty_altitude_gate_m is None else speed_penalty_altitude_gate_m
    )
    min_lift_mps = float(MIN_LIFT_SPEED_MPS if min_lift_speed_mps is None else min_lift_speed_mps)
    lift_speed_pen_w = float(0.0 if lift_speed_penalty_weight is None else lift_speed_penalty_weight)
    attack_dwell_w = float(0.0 if attack_dwell_reward_weight is None else attack_dwell_reward_weight)
    atk_speed_bonus_w = float(
        ATTACK_SPEED_BONUS_WEIGHT if attack_speed_bonus_weight is None else attack_speed_bonus_weight
    )
    atk_speed_target_mps = float(
        ATTACK_SPEED_TARGET_MPS if attack_speed_target_mps is None else attack_speed_target_mps
    )
    atk_speed_bonus_width = float(
        ATTACK_SPEED_BONUS_WIDTH_MPS
        if attack_speed_bonus_width_mps is None
        else attack_speed_bonus_width_mps
    )
    low_hp_bonus_w = float(0.0 if low_hp_attack_bonus_weight is None else low_hp_attack_bonus_weight)
    low_hp_threshold = float(0.25 if low_hp_attack_threshold is None else low_hp_attack_threshold)
    approach_slow_pen_w = float(
        0.0 if approach_slow_penalty_weight is None else approach_slow_penalty_weight
    )
    approach_slow_min = float(170.0 if approach_slow_min_mps is None else approach_slow_min_mps)
    approach_slow_range = float(900.0 if approach_slow_range_m is None else approach_slow_range_m)

    my_pos_unit = np.asarray(my_state[0:3], dtype=np.float64)
    enemy_pos_unit = np.asarray(enemy_state[0:3], dtype=np.float64)
    if np.linalg.norm(enemy_pos_unit) < EPS:
        enemy_pos_unit = FIXED_TARGET_POS_UNIT
    alt_diff_m = abs(float(my_pos_unit[2] - enemy_pos_unit[2])) * POSITION_SCALE_M
    below_m = max(0.0, float(enemy_pos_unit[2] - my_pos_unit[2])) * POSITION_SCALE_M
    altitude_ok_for_speed = alt_diff_m <= speed_alt_gate_m

    comps = {}
    comps["distance_progress"] = _clamp(
        closing_m / DISTANCE_PROGRESS_SCALE_M,
        -DISTANCE_PROGRESS_CLAMP,
        DISTANCE_PROGRESS_CLAMP,
    )
    comps["proximity"] = 0.15 * np.exp(-distance / 1200.0)
    comps["alignment"] = align_w * alignment_cos
    if misalign_weight > 0.0:
        comps["yaw_misalign_penalty"] = -misalign_weight * (1.0 - alignment_cos)
    else:
        comps["yaw_misalign_penalty"] = 0.0
    attack_box_score, in_attack_box = _attack_box_score(
        forward_distance, lateral_error, alignment_cos, attack_box_weight=atk_box_w
    )
    comps["attack_box"] = attack_box_score
    speed_band = max(atk_speed_bonus_width, EPS)
    speed_shape = np.exp(-((speed_mps - atk_speed_target_mps) / speed_band) ** 2)
    if attack_dwell_w > 0.0 and in_attack_box and enemy_alive:
        comps["attack_dwell"] = attack_dwell_w * max(0.0, alignment_cos) * float(speed_shape)
    else:
        comps["attack_dwell"] = 0.0
    if atk_speed_bonus_w > 0.0 and in_attack_box and enemy_alive:
        comps["attack_speed_bonus"] = (
            atk_speed_bonus_w * max(0.0, alignment_cos) * float(speed_shape)
        )
    else:
        comps["attack_speed_bonus"] = 0.0
    enemy_hp_frac = _hp_fraction(enemy_state)
    if (
        low_hp_bonus_w > 0.0
        and in_attack_box
        and enemy_alive
        and 0.0 < enemy_hp_frac < low_hp_threshold
    ):
        comps["low_hp_attack_bonus"] = (
            low_hp_bonus_w * max(0.0, alignment_cos) * (1.0 - enemy_hp_frac / low_hp_threshold)
        )
    else:
        comps["low_hp_attack_bonus"] = 0.0
    if forward_distance > 0.0:
        comps["corridor"] = corridor_w * max(0.0, alignment_cos) * np.exp(-lateral_error / 20.0)
    else:
        comps["corridor"] = -0.4
    if forward_distance > 0.0:
        comps["centerline"] = centerline_w * max(0.0, alignment_cos) * np.exp(-lateral_error / CENTERLINE_SCALE_M)
    else:
        comps["centerline"] = -0.4
    comps["turn_penalty"] = -TURN_RATE_PENALTY_WEIGHT * min(angular_rate, 5.0)
    if alt_match_w > 0.0:
        comps["altitude_match"] = alt_match_w * np.exp(-alt_diff_m / max(alt_match_scale, EPS))
        if alt_below_w > 0.0:
            comps["altitude_below_penalty"] = -alt_below_w * below_m
        else:
            comps["altitude_below_penalty"] = 0.0
    else:
        comps["altitude_match"] = 0.0
        comps["altitude_below_penalty"] = 0.0
    if lift_speed_pen_w > 0.0 and below_m > 5.0:
        comps["lift_speed_penalty"] = -lift_speed_pen_w * max(0.0, min_lift_mps - speed_mps)
    else:
        comps["lift_speed_penalty"] = 0.0
    # 敌机 HP 下降即给分，与攻击盒/距离无关（平台扣血即反馈）
    comps["enemy_damage"] = (
        hit_reward * (enemy_damage / enemy_hit_damage) if enemy_damage > 0.0 else 0.0
    )
    comps["self_damage"] = -SELF_DAMAGE_PENALTY_PER_HIT * (self_damage / self_hit_damage)
    comps["enemy_hp_shaping"] = hp_shape_w * max(0.0, _hp_fraction(prev_enemy_state) - _hp_fraction(enemy_state))
    comps["overshoot"] = (
        -OVERSHOOT_PENALTY if enemy_alive and forward_distance < float(overshoot_margin_m) else 0.0
    )
    if enemy_alive and 0.0 < forward_distance < FINISH_RANGE_M:
        comps["finish_centerline"] = finish_cl_w * max(0.0, alignment_cos) * np.exp(-lateral_error / FINISH_CENTERLINE_SCALE_M)
    else:
        comps["finish_centerline"] = 0.0

    # 速度 shaping：仅高度对齐时惩罚超速（junior 掉高时收油会进一步恶化）
    comps["speed_cruise_penalty"] = 0.0
    comps["speed_approach_penalty"] = 0.0
    comps["speed_penalty"] = 0.0
    if enemy_alive and forward_distance > 0.0 and altitude_ok_for_speed:
        if cruise_pen_w > 0.0 and forward_distance >= approach_range_m:
            comps["speed_cruise_penalty"] = -cruise_pen_w * max(0.0, speed_mps - cruise_cap_mps)
        if speed_pen_w > 0.0 and forward_distance < approach_range_m:
            desired_speed = max(desired_min_mps, forward_distance / max(desired_dist_scale, EPS))
            comps["speed_approach_penalty"] = -speed_pen_w * max(0.0, speed_mps - desired_speed)
        comps["speed_penalty"] = comps["speed_cruise_penalty"] + comps["speed_approach_penalty"]

    if (
        approach_slow_pen_w > 0.0
        and enemy_alive
        and ATTACK_MAX_RANGE_M < forward_distance < approach_slow_range
    ):
        comps["approach_slow_penalty"] = -approach_slow_pen_w * max(0.0, approach_slow_min - speed_mps)
    else:
        comps["approach_slow_penalty"] = 0.0

    if (
        enemy_alive
        and ATTACK_MIN_RANGE_M < forward_distance < ATTACK_MAX_RANGE_M
        and speed_mps > atk_speed_limit_mps
    ):
        comps["attack_speed_penalty"] = (
            -atk_speed_pen_w
            * (speed_mps - atk_speed_limit_mps)
            / max(atk_speed_scale_mps, EPS)
        )
    else:
        comps["attack_speed_penalty"] = 0.0

    comps["survival"] = -0.02
    comps["kill_bonus"] = kill_bonus_val if _health(prev_enemy_state) > 0.0 and _health(enemy_state) <= 0.0 else 0.0
    comps["death_penalty"] = -300.0 if _health(prev_my_state) > 0.0 and _health(my_state) <= 0.0 else 0.0

    comps["total"] = float(
        sum(value for key, value in comps.items() if key not in ("total", "speed_penalty"))
    )
    return comps


def calculate_reward(prev_my_state, prev_enemy_state, my_state, enemy_state):
    return reward_components(prev_my_state, prev_enemy_state, my_state, enemy_state)["total"]
