# This is the reward calculation function. We provide current state and previous state for you.
def reward_components(prev_my_state, prev_enemy_state, my_state, enemy_state):
    comps = {}

    comps["total"] = 0
    return comps

def calculate_reward(prev_my_state, prev_enemy_state, my_state, enemy_state):
    return reward_components(prev_my_state, prev_enemy_state, my_state, enemy_state)["total"]