"""
Sweep throttle+pitch for junior_dynamics altitude trim (pitch unlocked).

Usage:
    python scripts/probe_junior_trim.py --config ./config/envs.junior.phase1.yaml
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.train_env import TrainEnv


def control_to_action(throttle: float, pitch: float, env: TrainEnv) -> np.ndarray:
    thr = float(np.clip(throttle, 0.0, env.max_throttle))
    pit = float(np.clip(pitch, -env.max_pitch, env.max_pitch))
    scale = float(env.action_scale)
    thr_a = 2.0 * thr / env.max_throttle - 1.0 if env.max_throttle > 0 else -1.0
    pit_a = pit / scale if scale > 0 else 0.0
    return np.array([thr_a, pit_a, 0.0, 0.0], dtype=np.float32)


def run_case(env: TrainEnv, label: str, throttle: float, pitch: float, steps: int):
    obs, _ = env.reset()
    z0 = float(env.my_state[2])
    act = control_to_action(throttle, pitch, env)
    zs = []
    for n in range(steps):
        obs, _, term, trunc, _ = env.step(act)
        zs.append(float(env.my_state[2]))
        if term or trunc:
            break
    z1 = float(env.my_state[2])
    trend = float(np.polyfit(np.arange(len(zs)), zs, 1)[0]) if len(zs) >= 2 else 0.0
    spd = float(np.linalg.norm(env.my_state[6:9]))
    print(
        f"{label:<22} thr={throttle:.2f} pitch={pitch:+.2f}  "
        f"steps={len(zs):3d}  z {z0:.1f}->{z1:.1f}  dz={z1-z0:+.2f}  "
        f"trend={trend:+.4f}/step  spd={spd:.2f}"
    )
    return z1 - z0, trend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "envs.junior.phase1.yaml"))
    parser.add_argument("--steps", type=int, default=150)
    args = parser.parse_args()

    env = TrainEnv(args.config)
    print(
        f"room={env.room_id} port={env.adaptor.port}  "
        f"init_speed={env._init_kwargs.get('combat_initial_speed')}  "
        f"max_throttle={env.max_throttle} max_pitch={env.max_pitch}"
    )
    print(f"{'case':<22} {'controls':<22} {'result'}")
    cases = [
        ("flat_low_thr", 0.35, 0.0),
        ("flat_mid_thr", 0.50, 0.0),
        ("flat_high_thr", 0.65, 0.0),
        ("nose_up_mid", 0.50, 0.12),
        ("nose_up_high", 0.65, 0.15),
        ("nose_up_more", 0.65, 0.25),
        ("nose_up_max", 0.65, 0.28),
    ]
    for label, thr, pit in cases:
        run_case(env, label, thr, pit, args.steps)


if __name__ == "__main__":
    main()
