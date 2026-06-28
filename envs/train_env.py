import gymnasium
from gymnasium import spaces
from gymnasium.utils import env_checker
import numpy as np
from utils import adaptor, action, observation, reward, truncate, initialize
import yaml
import os

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


def is_expected_initial_observation(my_state, enemy_state, initial_body, position_tolerance_unit=5.0):
    expected_my = initial_body[:3].astype(np.float64)
    expected_enemy = initial_body[12:15].astype(np.float64)
    my_pos = np.asarray(my_state[:3], dtype=np.float64)
    enemy_pos = np.asarray(enemy_state[:3], dtype=np.float64)
    my_hp = float(my_state[12]) if len(my_state) > 12 else 0.0
    enemy_hp = float(enemy_state[12]) if len(enemy_state) > 12 else 0.0

    return (
        np.linalg.norm(my_pos - expected_my) <= position_tolerance_unit
        and np.linalg.norm(enemy_pos - expected_enemy) <= position_tolerance_unit
        and my_hp >= 900.0
        and enemy_hp >= 900.0
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


    def step(self, agent_action):
        # Check for truncation first
        truncated = truncate.check_truncation(self.my_state, self.enemy_state)

        # Marshal agent actions into real actions and send
        # First marshal unified actions into platform actions
        real_action = action.marshal_action(agent_action)
        # Then append truncation flag to the packet and send
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
        # Process whole state into agent state
        self.state = observation.marshal_observation(self.my_state, self.enemy_state)

        # Check for termination
        # But truncation has a higher priority
        if truncated:
            terminated = False

        comps = reward.reward_components(prev_my_state, prev_enemy_state, self.my_state, self.enemy_state)
        step_reward = comps["total"]
        info = {
            "reward_comps": comps,
        }
        for k, v in comps.items():
            if k != "total":
                info[f"r/{k}"] = float(v)
        return self.state, step_reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.adaptor.connect()
        initial_body = initialize.generate_initial_state()
        new_initial_packet = pack_initial(
            initial_body,
            room_id=self.room_id,
            unit_id=self.unit_id,
            state=self.initial_state,
            sync_step=self.sync_step,
        )
        noop = pack_action(action.marshal_action(np.zeros(4, dtype=np.float64)), False)
        finish_round = pack_action(action.marshal_action(np.zeros(4, dtype=np.float64)), True)

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
            if is_expected_initial_observation(self.my_state, self.enemy_state, initial_body):
                break

            if self.verbose:
                print("discard stale reset observation:", self.my_state, self.enemy_state)
            self.adaptor.send_action_packet(finish_round)

        self.state = observation.marshal_observation(self.my_state, self.enemy_state)
        return self.state, {}


if __name__ == '__main__':
    env = TrainEnv('../config/envs.yaml')
    env_checker.check_env(env)
