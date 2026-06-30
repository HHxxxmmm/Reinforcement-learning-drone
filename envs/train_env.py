import gymnasium
from gymnasium import spaces
from gymnasium.utils import env_checker
import numpy as np
from utils import adaptor, action, observation, reward, truncate, initialize
from utils import initialize_align, reward_align, truncate_align, orient_setup
import yaml
import os


def _load_stage_modules(stage):
    if stage == "align1":
        return initialize_align, reward_align, truncate_align
    return initialize, reward, truncate


def float_to_bool(f):
    if f == 0.0:
        return False
    if f == 1.0:
        return True
    raise ValueError(f"Unexpected termination flag: {f}")


def pack_initial(initial_observation, room_id=0, unit_id=0, state=2, sync_step=1):
    integer_observation = initial_observation.astype(np.int32)
    if integer_observation.shape[0] != 24:
        raise ValueError(f"initial body must be 24 int32, got {integer_observation.shape[0]}")

    packet = np.zeros(100, dtype=np.int32)
    packet[0] = room_id
    packet[1] = unit_id
    packet[2:14] = integer_observation[:12]
    packet[14:26] = integer_observation[12:24]
    packet[26] = state
    packet[27] = sync_step
    return packet


def split_observation(observation):
    my_state = observation[0:13].astype(np.float64).copy()
    enemy_state = observation[13:26].astype(np.float64).copy()
    terminated = float_to_bool(observation[26])
    return my_state, enemy_state, terminated


def pack_action(action, truncated):
    if truncated:
        truncation = 1.0
    else:
        truncation = 0.0
    full_pack = np.append(action, truncation)
    return full_pack


def is_expected_initial_observation(
    my_state,
    enemy_state,
    initial_body,
    position_tolerance_unit=5.0,
    allow_zero_enemy=False,
):
    expected_my = initial_body[:3].astype(np.float64)
    expected_enemy = initial_body[12:15].astype(np.float64)
    my_pos = np.asarray(my_state[:3], dtype=np.float64)
    enemy_pos = np.asarray(enemy_state[:3], dtype=np.float64)
    my_hp = float(my_state[12]) if len(my_state) > 12 else 0.0
    enemy_hp = float(enemy_state[12]) if len(enemy_state) > 12 else 0.0

    enemy_ok = np.linalg.norm(enemy_pos - expected_enemy) <= position_tolerance_unit
    if allow_zero_enemy and np.linalg.norm(enemy_pos) < 1e-6:
        enemy_ok = True

    hp_ok = (my_hp >= 0.9 and enemy_hp >= 0.9) if my_hp <= 1.5 else (my_hp >= 900.0 and enemy_hp >= 900.0)

    return (
        np.linalg.norm(my_pos - expected_my) <= position_tolerance_unit
        and enemy_ok
        and hp_ok
    )


class TrainEnv(gymnasium.Env):
    def __init__(self, config_path):
        super().__init__()

        # Agent space bounds
        action_upper_bound = np.ones(shape=[4], dtype=np.float64)
        action_lower_bound = np.negative(np.ones(shape=[4], dtype=np.float64))
        self.action_space = spaces.Box(shape=[4], dtype=np.float64, low=action_lower_bound, high=action_upper_bound)

        observation_upper_bound = np.ones(shape=[20], dtype=np.float64)
        observation_lower_bound = np.negative(np.ones(shape=[20], dtype=np.float64))
        self.observation_space = spaces.Box(shape=[20], dtype=np.float64, low=observation_lower_bound, high=observation_upper_bound)

        # 仅创建 adaptor，reset() 时再连接（避免 __init__ 空连一次干扰握手）
        self.adaptor = adaptor.NetworkAdaptor(config_path)

        self.my_state, self.enemy_state = None, None
        self.state = None
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.save_path = self.config["save_path"]
        self.room_id = self.config.get("room_id", 0)
        self.unit_id = self.config.get("unit_id", 0)
        # 联机训练 state=2，本地训练 state=1（见 MATLAB/get_my_initial.m）
        self.initial_state = 2 if self.config.get("train_mode", "online") == "online" else 1
        self.sync_step = int(self.config.get("sync_step", 1))
        self.verbose = bool(self.config.get("logger", False))

        self.stage = self.config.get("stage", "combat")
        self.stage_cfg = self.config.get("stage_params", {}) or {}
        self.init_mod, self.reward_mod, self.truncate_mod = _load_stage_modules(self.stage)

        action_cfg = self.config.get("action", {}) or {}
        self.lock_throttle = bool(action_cfg.get("lock_throttle", False))
        self.fixed_throttle = float(action_cfg.get("fixed_throttle", 0.0))
        self.max_throttle = float(action_cfg.get("max_throttle", 1.0))
        self.lock_roll = bool(action_cfg.get("lock_roll", False))
        self.lock_pitch = bool(action_cfg.get("lock_pitch", False))
        self.fixed_pitch = float(action_cfg.get("fixed_pitch", 0.0))
        self.max_pitch = float(action_cfg.get("max_pitch", 1.0))
        self.lock_yaw = bool(action_cfg.get("lock_yaw", False))
        self.fixed_yaw = float(action_cfg.get("fixed_yaw", 0.0))
        self.max_yaw = float(action_cfg.get("max_yaw", 1.0))
        self.action_scale = float(action_cfg.get("action_scale", 1.0))

        self.step_count = 0
        self.hold_count = 0
        self._hold_curriculum_successes = 0
        self._episode_min_enemy_hp = 1.0
        self._episode_start_enemy_hp = 1.0
        self._init_kwargs = self._build_init_kwargs()

    def _build_init_kwargs(self):
        if self.stage == "align1":
            keys = (
                "init_mode",
                "altitude_unit",
                "enemy_pos",
                "enemy_y_range",
                "initial_yaw",
                "start_x",
                "initial_pitch",
                "initial_roll",
            )
            kwargs = {key: self.stage_cfg[key] for key in keys if key in self.stage_cfg}
            if "enemy_pos" not in kwargs and "enemy_x_unit" in self.stage_cfg:
                alt = int(self.stage_cfg.get("altitude_unit", 100))
                x = int(self.stage_cfg["enemy_x_unit"])
                kwargs["enemy_pos"] = [x, 0, alt]
            if "init_mode" not in kwargs:
                kwargs["init_mode"] = "align_ta_v2"
            return kwargs

        if self.stage == "combat":
            kwargs = {"mode": "combat"}
            mapping = (
                ("initial_speed", "combat_initial_speed"),
                ("altitude_unit", "combat_my_alt_unit"),
                ("enemy_pos", "combat_enemy_pos"),
                ("enemy_y_range", "combat_enemy_y_range"),
                ("enemy_y_positive_prob", "combat_enemy_y_positive_prob"),
            )
            for cfg_key, init_key in mapping:
                if cfg_key in self.stage_cfg:
                    kwargs[init_key] = self.stage_cfg[cfg_key]
            return kwargs

        return {}

    def _align_cos_threshold(self):
        if "align_target_deg" in self.stage_cfg:
            return reward_align.align_cos_threshold_from_deg(self.stage_cfg["align_target_deg"])
        return float(self.stage_cfg.get("align_cos_threshold", reward_align.DEFAULT_ALIGN_COS_THRESHOLD))

    def _required_hold_steps(self):
        max_hold = self.stage_cfg.get("hold_steps_max")
        return reward_align.required_hold_steps(
            self._hold_curriculum_successes,
            start=int(self.stage_cfg.get("hold_steps", reward_align.DEFAULT_HOLD_STEPS_START)),
            increment=int(self.stage_cfg.get("hold_steps_increment", reward_align.DEFAULT_HOLD_INCREMENT)),
            every=int(self.stage_cfg.get("hold_steps_every_successes", reward_align.DEFAULT_HOLD_EVERY_SUCCESSES)),
            max_hold=max_hold,
        )

    def _timeout_extra_steps(self):
        return int(self.stage_cfg.get("timeout_extra_steps", 100))

    def _episode_step_limit(self):
        """动态步数上限：当前 hold 数 + extra（默认 100）；超出该值即超时。"""
        return self.hold_count + self._timeout_extra_steps()

    def _send_finish_round(self):
        finish = pack_action(self._marshal_action(np.zeros(4, dtype=np.float64)), True)
        self.adaptor.send_action_packet(finish)

    def _hard_max_steps(self):
        cap = self.stage_cfg.get("max_steps_per_episode")
        if cap is None:
            return None
        cap = int(cap)
        return cap if cap > 0 else None

    def _marshal_action(self, agent_action):
        return action.marshal_action(
            agent_action,
            lock_throttle=self.lock_throttle,
            fixed_throttle=self.fixed_throttle,
            max_throttle=self.max_throttle,
            lock_roll=self.lock_roll,
            lock_pitch=self.lock_pitch,
            fixed_pitch=self.fixed_pitch,
            max_pitch=self.max_pitch,
            lock_yaw=self.lock_yaw,
            fixed_yaw=self.fixed_yaw,
            max_yaw=self.max_yaw,
            action_scale=self.action_scale,
        )

    def _update_align_counters(self):
        cos = self.reward_mod.alignment_cos(self.my_state, self.enemy_state)
        if cos >= self._align_cos_threshold():
            self.hold_count += 1
        else:
            self.hold_count = 0

    def _episode_success(self):
        if self.stage != "align1":
            return False
        if bool(self.stage_cfg.get("disable_hold_termination", False)):
            return False
        return self.hold_count >= self._required_hold_steps()

    def _episode_timeout(self):
        if self.stage == "combat":
            hard = self._hard_max_steps()
            return hard is not None and self.step_count >= hard
        if self.stage != "align1":
            return False
        if bool(self.stage_cfg.get("disable_dynamic_timeout", False)):
            hard = self._hard_max_steps()
            return hard is not None and self.step_count >= hard
        hard = self._hard_max_steps()
        if hard is not None and self.step_count >= hard:
            return True
        return self.step_count > self._episode_step_limit()

    def _combat_overshoot(self):
        if self.stage != "combat":
            return False
        if not bool(self.stage_cfg.get("terminate_on_overshoot", True)):
            return False
        if float(self.enemy_state[12]) <= 0.0:
            return False
        _, _, rel = reward._range_and_los(self.my_state, self.enemy_state)
        forward = reward._forward_vector(self.my_state)
        forward_distance_m = float(np.dot(rel, forward))
        margin = float(self.stage_cfg.get("overshoot_margin_m", 0.0))
        return forward_distance_m < margin

    def _alignment_cos(self):
        if hasattr(self.reward_mod, "alignment_cos"):
            return float(self.reward_mod.alignment_cos(self.my_state, self.enemy_state))
        _, los, _ = reward._range_and_los(self.my_state, self.enemy_state)
        forward = reward._forward_vector(self.my_state)
        return float(np.clip(np.dot(forward, los), -1.0, 1.0))

    def step(self, agent_action):
        # Check for truncation first
        truncated = self.truncate_mod.check_truncation(self.my_state, self.enemy_state)

        # Marshal agent actions into real actions and send
        real_action = self._marshal_action(agent_action)
        send_pack = pack_action(real_action, truncated)
        self.adaptor.send_action_packet(send_pack)

        # Save previous state for reward calculation
        prev_my_state = self.my_state.copy()
        prev_enemy_state = self.enemy_state.copy()

        # Get new observations and unmarshal
        original_observation = self.adaptor.get_observation_packet()
        self.my_state, self.enemy_state, terminated = split_observation(original_observation)
        if self.verbose:
            print(self.my_state)

        self.step_count += 1
        if self.stage == "align1":
            self._update_align_counters()
        elif self.stage == "combat":
            enemy_hp = float(self.enemy_state[12]) if len(self.enemy_state) > 12 else 1.0
            self._episode_min_enemy_hp = min(self._episode_min_enemy_hp, enemy_hp)

        self.state = observation.marshal_observation(self.my_state, self.enemy_state)

        episode_success = self._episode_success()
        episode_overshoot = self._combat_overshoot()
        episode_timeout = self._episode_timeout() and not episode_success and not episode_overshoot
        required_hold_this_ep = self._required_hold_steps() if self.stage == "align1" else 0

        if episode_success:
            terminated = True
            truncated = False
            if self.stage == "align1":
                bonus = reward_align.success_bonus_amount(
                    required_hold_this_ep,
                    float(self.stage_cfg.get(
                        "success_bonus_per_hold_step",
                        reward_align.DEFAULT_SUCCESS_BONUS_PER_HOLD_STEP,
                    )),
                )
                self._hold_curriculum_successes += 1
                if self.verbose or bool(self.stage_cfg.get("log_hold_curriculum", False)):
                    print(
                        "[align1 success] "
                        f"held={self.hold_count} required={required_hold_this_ep} "
                        f"bonus={bonus:.0f} "
                        f"total_successes={self._hold_curriculum_successes} "
                        f"next_required={self._required_hold_steps()}"
                    )
                self._send_finish_round()
        elif episode_overshoot:
            truncated = True
            terminated = False
            if self.verbose or bool(self.stage_cfg.get("log_init", False)):
                speed_mps = float(np.linalg.norm(self.my_state[6:9])) * reward.POSITION_SCALE_M
                my_x_m = float(self.my_state[0]) * reward.POSITION_SCALE_M
                enemy_hp = float(self.enemy_state[12]) if len(self.enemy_state) > 12 else 1.0
                _, _, rel = reward._range_and_los(self.my_state, self.enemy_state)
                forward = reward._forward_vector(self.my_state)
                fwd_m = float(np.dot(rel, forward))
                margin_m = float(self.stage_cfg.get("overshoot_margin_m", 0.0))
                print(
                    f"[combat overshoot] step={self.step_count} "
                    f"speed={speed_mps:.1f}m/s my_x={my_x_m:.0f}m "
                    f"enemy_hp={enemy_hp:.3f} min_hp={self._episode_min_enemy_hp:.3f} "
                    f"fwd={fwd_m:.0f}m margin={margin_m:.0f}m"
                )
            self._send_finish_round()
        elif episode_timeout:
            truncated = True
            terminated = False
            if self.stage == "align1" and (
                self.verbose or bool(self.stage_cfg.get("log_hold_curriculum", False))
            ):
                print(
                    "[align1 timeout] "
                    f"step={self.step_count} hold={self.hold_count} "
                    f"limit={self._episode_step_limit()} required={required_hold_this_ep}"
                )
            elif self.stage == "combat" and (
                self.verbose or bool(self.stage_cfg.get("log_init", False))
            ):
                print(f"[combat max_steps] step={self.step_count}")
            self._send_finish_round()
        elif terminated and self.stage == "combat":
            truncated = False
            self._send_finish_round()
        elif truncated:
            terminated = False
            if self.stage == "combat":
                self._send_finish_round()

        reward_kwargs = {}
        if self.stage == "align1":
            reward_kwargs = dict(
                hold_steps=self.hold_count,
                required_hold_steps=required_hold_this_ep,
                align_cos_threshold=self._align_cos_threshold(),
                cosine_weight=float(self.stage_cfg.get("cosine_weight", reward_align.COSINE_REWARD_WEIGHT)),
                success_bonus_per_hold_step=float(self.stage_cfg.get(
                    "success_bonus_per_hold_step",
                    reward_align.DEFAULT_SUCCESS_BONUS_PER_HOLD_STEP,
                )),
                timeout_penalty=float(self.stage_cfg.get("timeout_penalty", -1.5)),
                episode_success=episode_success,
                episode_timeout=episode_timeout,
            )
        elif self.stage == "combat":
            reward_kwargs = {}
            combat_reward_keys = (
                "overshoot_margin_m",
                "yaw_misalign_weight",
                "pitch_up_weight",
                "damage_reward_per_hit",
                "alignment_weight",
                "attack_box_weight",
                "corridor_weight",
                "centerline_weight",
                "finish_centerline_weight",
                "enemy_hp_shaping_weight",
                "kill_bonus",
            )
            for key in combat_reward_keys:
                if key in self.stage_cfg:
                    reward_kwargs[key] = float(self.stage_cfg[key])

        comps = self.reward_mod.reward_components(
            prev_my_state,
            prev_enemy_state,
            self.my_state,
            self.enemy_state,
            **reward_kwargs,
        )
        step_reward = comps["total"]
        info = {
            "reward_comps": comps,
            "align_cos": self._alignment_cos(),
            "hold_count": int(self.hold_count),
            "required_hold_steps": int(required_hold_this_ep if self.stage == "align1" else 0),
            "hold_curriculum_successes": int(self._hold_curriculum_successes),
            "episode_step_limit": int(self._episode_step_limit() if self.stage == "align1" else 0),
        }
        for k, v in comps.items():
            if k != "total":
                info[f"r/{k}"] = float(v)
        return self.state, step_reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.hold_count = 0
        self._episode_min_enemy_hp = 1.0
        self._episode_start_enemy_hp = 1.0
        self.adaptor.connect()
        initial_body = self.init_mod.generate_initial_state(**self._init_kwargs)
        enemy_fallback = initial_body[12:15].astype(np.float64)
        observation.set_enemy_fallback_position(enemy_fallback)
        reward.FIXED_TARGET_POS_UNIT = enemy_fallback.copy()

        new_initial_packet = pack_initial(
            initial_body,
            room_id=self.room_id,
            unit_id=self.unit_id,
            state=self.initial_state,
            sync_step=self.sync_step,
        )
        noop = pack_action(self._marshal_action(np.zeros(4, dtype=np.float64)), False)
        finish_round = pack_action(self._marshal_action(np.zeros(4, dtype=np.float64)), True)

        for attempt in range(6):
            self.adaptor.send_initial_packet(new_initial_packet)
            try:
                original_observation = self.adaptor.get_observation_packet()
            except TimeoutError:
                if self.verbose:
                    print("No observation after initial packet; sending a noop action.")
                self.adaptor.send_action_packet(noop)
                original_observation = self.adaptor.get_observation_packet()

            self.my_state, self.enemy_state, termination = split_observation(original_observation)
            if is_expected_initial_observation(
                self.my_state,
                self.enemy_state,
                initial_body,
                allow_zero_enemy=(self.stage == "align1"),
            ):
                break

            if self.verbose:
                print("discard stale reset observation:", self.my_state, self.enemy_state)
            self.adaptor.send_action_packet(finish_round)

        enemy_pos = initial_body[12:15].astype(np.float64)
        setup_steps = 0
        if self.stage == "align1" and bool(self.stage_cfg.get("setup_orientation", False)):
            target_deg = float(self.stage_cfg.get("cone_half_angle_deg", 30.0))
            tol_deg = float(self.stage_cfg.get("setup_tolerance_deg", 5.0))
            max_setup = int(self.stage_cfg.get("setup_max_steps", 200))
            self.my_state, self.enemy_state, setup_steps = orient_setup.setup_misalignment_deg(
                self.my_state,
                enemy_pos,
                target_deg=target_deg,
                tolerance_deg=tol_deg,
                max_steps=max_setup,
                send_action_fn=self.adaptor.send_action_packet,
                recv_obs_fn=self.adaptor.get_observation_packet,
                marshal_action_fn=self._marshal_action,
            )

        if self.stage == "align1":
            mis_deg = self.init_mod.nose_misalignment_deg_from_state(self.my_state, enemy_pos)
            reset_info = {
                "init_sent_yaw_int": int(initial_body[5]),
                "init_enemy_y": float(enemy_pos[1]),
                "init_obs_rpy_rad": [float(self.my_state[3]), float(self.my_state[4]), float(self.my_state[5])],
                "init_misalignment_deg": mis_deg,
                "setup_steps": setup_steps,
            }
            if self.verbose or bool(self.stage_cfg.get("log_init_alignment", False)):
                print(
                    "[align1 reset] "
                    f"sent_yaw={reset_info['init_sent_yaw_int']} "
                    f"enemy_y={reset_info['init_enemy_y']:.0f} "
                    f"obs_yaw={reset_info['init_obs_rpy_rad'][2]:.3f} "
                    f"misalignment={mis_deg:.1f}° "
                    f"setup_steps={setup_steps}"
                )
        else:
            enemy_pos = initial_body[12:15].astype(np.float64)
            reset_info = {
                "init_enemy_y": float(enemy_pos[1]),
                "init_speed": float(initial_body[6]),
                "init_altitude": float(initial_body[2]),
            }
            self._episode_start_enemy_hp = float(
                self.enemy_state[12] if len(self.enemy_state) > 12 else 1.0
            )
            self._episode_min_enemy_hp = self._episode_start_enemy_hp
            if self.verbose or bool(self.stage_cfg.get("log_init", False)):
                print(
                    "[combat reset] "
                    f"enemy_y={reset_info['init_enemy_y']:.0f} "
                    f"speed={reset_info['init_speed']:.0f} "
                    f"alt={reset_info['init_altitude']:.0f} "
                    f"enemy_hp={self._episode_start_enemy_hp:.3f}"
                )

        self.state = observation.marshal_observation(self.my_state, self.enemy_state)
        return self.state, reset_info


if __name__ == '__main__':
    env = TrainEnv('../config/envs.yaml')
    env_checker.check_env(env)
