"""
Fixed-throttle probe for junior_dynamics altitude drift.

Usage (UE junior room open):
    python scripts/probe_junior_altitude.py --config ./config/envs.junior.probe.yaml
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.train_env import TrainEnv


def throttle_to_action(throttle: float, max_throttle: float = 0.35) -> float:
    """Map desired throttle in [0, max_throttle] to network action[0] in [-1, 1]."""
    if max_throttle <= 0.0:
        return -1.0
    t = float(np.clip(throttle, 0.0, max_throttle))
    return float(2.0 * t / max_throttle - 1.0)


def fixed_action(throttle: float, max_throttle: float = 0.35) -> np.ndarray:
    return np.array(
        [throttle_to_action(throttle, max_throttle), 0.0, 0.0, 0.0],
        dtype=np.float32,
    )


def run_probe(env: TrainEnv, label: str, throttle: float, steps: int, log_every: int):
    print(f"\n=== {label} (throttle={throttle:.3f}) ===")
    obs, info = env.reset()
    my0 = env.my_state.copy()
    en0 = env.enemy_state.copy()
    print(
        f"init my=({my0[0]:.1f},{my0[1]:.1f},{my0[2]:.1f}) "
        f"spd={np.linalg.norm(my0[6:9]):.1f} "
        f"enemy_z={en0[2]:.1f} alt_diff={my0[2]-en0[2]:.1f}"
    )

    act = fixed_action(throttle, max_throttle=env.max_throttle)
    z_samples = []
    spd_samples = []
    terminated = truncated = False
    total_reward = 0.0
    n = 0

    while not (terminated or truncated) and n < steps:
        obs, reward, terminated, truncated, info = env.step(act)
        total_reward += reward
        n += 1
        z = float(env.my_state[2])
        spd = float(np.linalg.norm(env.my_state[6:9]))
        z_samples.append(z)
        spd_samples.append(spd)
        if n % log_every == 0 or n == 1:
            print(
                f"  step={n:4d}  z={z:6.2f}  dz={z-my0[2]:+6.2f}  "
                f"spd={spd:5.2f}  enemy_z={env.enemy_state[2]:.2f}  "
                f"reward={reward:.3f}"
            )

    my1 = env.my_state
    dz = float(my1[2] - my0[2])
    trend = float(np.polyfit(np.arange(len(z_samples)), z_samples, 1)[0]) if len(z_samples) >= 2 else 0.0
    print(
        f"done steps={n} dz_total={dz:+.2f} unit ({dz*10:+.0f} m)  "
        f"z_end={my1[2]:.2f}  trend={trend:+.4f} unit/step  "
        f"reason={'truncated' if truncated else ('terminated' if terminated else 'max_steps')}  "
        f"reward_sum={total_reward:.1f}"
    )
    return {
        "label": label,
        "throttle": throttle,
        "steps": n,
        "z_start": float(my0[2]),
        "z_end": float(my1[2]),
        "dz": dz,
        "trend": trend,
        "spd_end": float(np.linalg.norm(my1[6:9])),
    }


def main():
    parser = argparse.ArgumentParser(description="Junior dynamics fixed-throttle altitude probe")
    parser.add_argument("--config", default=str(ROOT / "config" / "envs.junior.probe.yaml"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    env = TrainEnv(args.config)
    print(
        f"connect {env.adaptor.host}:{env.adaptor.port} "
        f"room={env.room_id} pitch_locked={env.lock_pitch} yaw_locked={env.lock_yaw}"
    )

    cases = [
        ("A_cruise_0.12", 0.12),
        ("B_idle_0", 0.0),
        ("C_high_0.35", 0.35),
    ]
    results = []
    for label, thr in cases:
        results.append(run_probe(env, label, thr, args.steps, args.log_every))

    print("\n=== Summary ===")
    print(f"{'case':<16} {'throttle':>8} {'dz(unit)':>10} {'dz(m)':>8} {'trend/step':>12}")
    for r in results:
        print(
            f"{r['label']:<16} {r['throttle']:8.2f} {r['dz']:+10.2f} "
            f"{r['dz']*10:+8.0f} {r['trend']:+12.4f}"
        )


if __name__ == "__main__":
    main()
