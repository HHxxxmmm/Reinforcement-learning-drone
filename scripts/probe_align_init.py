"""
Probe stage-1 initial alignment via initialize packet (yaw int radians).

Usage (UE room open):
    python scripts/probe_align_init.py --config ./config/envs.stage1.yaml
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.train_env import TrainEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config" / "envs.stage1.yaml"))
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    env = TrainEnv(args.config)
    mode = env._init_kwargs.get("init_mode", "align_ta_v2")

    print(f"init_mode={mode}, kwargs={env._init_kwargs}")
    for ep in range(args.episodes):
        _, info = env.reset()
        print(
            f"ep={ep + 1} sent_yaw={info.get('init_sent_yaw_int')} "
            f"obs_yaw={info.get('init_obs_rpy_rad', [0, 0, 0])[2]:.3f} "
            f"misalignment={info.get('init_misalignment_deg', float('nan')):.1f}° "
            f"setup_steps={info.get('setup_steps', 0)}"
        )


if __name__ == "__main__":
    main()
