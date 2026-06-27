import numpy as np

# ——————————————————————————————
# Normalisation constants
# ——————————————————————————————
POSITION_SCALE_M = 10.0        # 1 position unit = 10 m
MAX_DIST_UNIT    = 500.0       # 500 units = 5000 m  (matches truncate.MAX_SEPARATION_M)
MAX_ALT_UNIT     = 100.0       # 100 units = 1000 m  (reasonable combat ceiling)
MAX_SPEED_UNIT   = 30.0        #  30 units =  300 m/s
MAX_HP           = 1.0           # Server sends hp as [0, 1] float; 0.01 per hit, ~100 hits to kill
EPS              = 1e-8


def _forward_vector(state: np.ndarray) -> np.ndarray:
    """Aircraft nose direction from pitch (idx 4) and yaw (idx 5)."""
    pitch = float(state[4])
    yaw   = float(state[5])
    cp = np.cos(pitch)
    fwd = np.array([cp * np.cos(yaw), cp * np.sin(yaw), np.sin(pitch)])
    return fwd / (np.linalg.norm(fwd) + EPS)


# ——————————————————————————————
# Observation processing
# ——————————————————————————————
def marshal_observation(my_state, enemy_state):
    """
    Convert raw 13-d per‑side states into a compact, normalised observation
    vector for the agent.  Every element lies in [-1, 1].

    Raw layout (per side, indices 0‑12)
    ────────────────────────────────────
    idx  │ meaning          │ unit
    ─────┼───────────────────┼───────────
    0‑2  │ position x,y,z    │ 10 m
    3‑5  │ roll, pitch, yaw  │ rad
    6‑8  │ velocity u,v,w    │ 10 m/s
    9‑11 │ angular vel.      │ rad/s
    12   │ hp                │ 0–1 float
    ─────┴───────────────────┴───────────

    Returns
    -------
    obs : np.ndarray  shape (16,)  dtype float64  range [-1, 1]
    """
    my_state    = np.asarray(my_state,    dtype=np.float64)
    enemy_state = np.asarray(enemy_state, dtype=np.float64)

    # ---- positions ----------------------------------------------------
    my_pos    = my_state[0:3]
    enemy_pos = enemy_state[0:3]
    rel_pos   = enemy_pos - my_pos                     # 3-d

    # ---- distance & line‑of‑sight -------------------------------------
    distance = np.linalg.norm(rel_pos) + EPS
    los      = rel_pos / distance                      # unit vector

    # ---- forward vectors & aspect‑angle cosines -----------------------
    my_fwd    = _forward_vector(my_state)
    enemy_fwd = _forward_vector(enemy_state)

    my_aa_cos    = np.dot(my_fwd,    los)              # my nose → enemy
    enemy_aa_cos = np.dot(enemy_fwd, -los)              # enemy nose → me

    # ---- velocities ---------------------------------------------------
    my_vel    = my_state[6:9]
    enemy_vel = enemy_state[6:9]
    rel_vel   = enemy_vel - my_vel

    my_speed    = np.linalg.norm(my_vel)    + EPS
    enemy_speed = np.linalg.norm(enemy_vel) + EPS
    closing     = -np.dot(rel_vel, los)               # >0 ⇒ approaching

    # ---- hp -----------------------------------------------------------
    my_hp    = float(my_state[12])
    enemy_hp = float(enemy_state[12])

    # ---- orientation --------------------------------------------------
    my_pitch = float(my_state[4])
    my_roll  = float(my_state[3])

    # === assemble observation vector ===================================
    obs = np.zeros(16, dtype=np.float64)

    #  0‑ 2   relative position              [-1, 1]
    obs[0:3] = np.clip(rel_pos / MAX_DIST_UNIT, -1.0, 1.0)

    #  3      distance                        0→+1  mapped to [-1, 1]
    obs[3]   = np.clip(2.0 * distance / MAX_DIST_UNIT - 1.0, -1.0, 1.0)

    #  4      my AA cosine                    already [-1, 1]
    obs[4]   = my_aa_cos

    #  5      enemy AA cosine                 already [-1, 1]
    obs[5]   = enemy_aa_cos

    #  6      speed difference   (mine − theirs)   [-1, 1]
    obs[6]   = np.clip((my_speed - enemy_speed) / MAX_SPEED_UNIT, -1.0, 1.0)

    #  7      my altitude                     0→+1  mapped to [-1, 1]
    obs[7]   = np.clip(2.0 * my_pos[2] / MAX_ALT_UNIT - 1.0, -1.0, 1.0)

    #  8      altitude difference  (mine − theirs)  [-1, 1]
    obs[8]   = np.clip((my_pos[2] - enemy_pos[2]) / MAX_ALT_UNIT, -1.0, 1.0)

    #  9      hp difference  (mine − theirs)        [-1, 1]
    obs[9]   = np.clip((my_hp - enemy_hp) / MAX_HP, -1.0, 1.0)

    # 10      my hp ratio                      0→+1  mapped to [-1, 1]
    obs[10]  = np.clip(2.0 * my_hp / MAX_HP - 1.0, -1.0, 1.0)

    # 11      enemy hp ratio                   0→+1  mapped to [-1, 1]
    obs[11]  = np.clip(2.0 * enemy_hp / MAX_HP - 1.0, -1.0, 1.0)

    # 12      closing speed                      [-1, 1]
    obs[12]  = np.clip(closing / MAX_SPEED_UNIT, -1.0, 1.0)

    # 13      my pitch / π                      [-1, 1]
    obs[13]  = np.clip(my_pitch / np.pi, -1.0, 1.0)

    # 14      my roll / π                       [-1, 1]
    obs[14]  = np.clip(my_roll / np.pi, -1.0, 1.0)

    # 15      my speed                          0→+1  mapped to [-1, 1]
    obs[15]  = np.clip(2.0 * my_speed / MAX_SPEED_UNIT - 1.0, -1.0, 1.0)

    return obs
