"""Batch-evaluate Stage-1 align checkpoints in UE.

Eval mode:
  - No hold-success termination (only measure hold).
  - No dynamic timeout; each episode runs exactly max_steps (default 512).
  - Sends finish packet to UE after each episode so the next reset is clean.
  - Random enemy y from env config each reset.

Example:
    python scripts/eval_align_ckpts.py \\
        --env-config ./config/envs.stage1.yaml \\
        --model-dir ./model/stage1_align_v3/ \\
        --episodes 3
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO

from envs.train_env import TrainEnv


def _ckpt_sort_key(path: Path):
    if "final" in path.stem.lower():
        return (1, 10**9, path.name)
    m = re.search(r"_(\d+)_steps$", path.stem)
    steps = int(m.group(1)) if m else -1
    return (0, steps, path.name)


def discover_checkpoints(model_dir: Path, include_final: bool):
    paths = sorted(model_dir.glob("*.zip"), key=_ckpt_sort_key)
    if not include_final:
        paths = [p for p in paths if "final" not in p.stem.lower()]
    return paths


def configure_eval_env(env: TrainEnv, max_steps: int):
    env.stage_cfg["disable_hold_termination"] = True
    env.stage_cfg["disable_dynamic_timeout"] = True
    env.stage_cfg["max_steps_per_episode"] = int(max_steps)
    env.stage_cfg["log_hold_curriculum"] = False
    env._hold_curriculum_successes = 0


def finish_episode(env: TrainEnv):
    """Tell UE this round is over so the next reset gets a clean initial pose."""
    env._send_finish_round()
    try:
        env.adaptor.get_observation_packet()
    except Exception:
        pass


def run_episode(env, model, deterministic: bool, max_steps: int):
    obs, reset_info = env.reset()
    init_mis = reset_info.get("init_misalignment_deg")
    if init_mis is not None and float(init_mis) > 8.0:
        print(f"    WARN: large init misalignment {init_mis:.1f}° — UE may not have reset cleanly")

    total_reward = 0.0
    steps = 0
    max_hold = 0
    max_align_cos = -1.0
    terminated = False
    truncated = False
    last_info = {}

    while steps < max_steps and not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, last_info = env.step(action)
        total_reward += float(reward)
        steps += 1
        max_hold = max(max_hold, int(last_info.get("hold_count", 0)))
        max_align_cos = max(max_align_cos, float(last_info.get("align_cos", -1.0)))

    if terminated:
        reason = "terminated"
    elif truncated:
        reason = "truncated"
    elif steps >= max_steps:
        reason = "max_steps"
        finish_episode(env)
    else:
        reason = "stopped"
        finish_episode(env)

    best_misalignment_deg = float(
        np.rad2deg(np.arccos(np.clip(max_align_cos, -1.0, 1.0)))
        if max_align_cos >= 0
        else float("nan")
    )

    return {
        "steps": steps,
        "max_hold": max_hold,
        "final_hold": int(last_info.get("hold_count", 0)),
        "max_align_cos": max_align_cos,
        "max_misalignment_deg": best_misalignment_deg,
        "total_reward": total_reward,
        "reason": reason,
        "enemy_y": reset_info.get("init_enemy_y"),
        "init_misalignment_deg": init_mis,
    }


def eval_checkpoint(
    model_path: Path,
    env_config: str,
    episodes: int,
    deterministic: bool,
    max_steps: int,
):
    env = TrainEnv(config_path=env_config)
    configure_eval_env(env, max_steps)

    model = PPO.load(str(model_path), env=env, device="auto")
    rows = []
    for ep in range(1, episodes + 1):
        row = run_episode(env, model, deterministic=deterministic, max_steps=max_steps)
        row["checkpoint"] = model_path.name
        row["episode"] = ep
        rows.append(row)
        print(
            f"  ep {ep}: steps={row['steps']} max_hold={row['max_hold']} "
            f"init_mis={row.get('init_misalignment_deg', float('nan')):.1f}° "
            f"enemy_y={row.get('enemy_y')} reason={row['reason']}"
        )
    env.adaptor.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description="Batch eval Stage-1 align checkpoints")
    parser.add_argument(
        "--env-config",
        default=str(ROOT / "config" / "envs.stage1.yaml"),
    )
    parser.add_argument(
        "--model-dir",
        default=str(ROOT / "model" / "stage1_align_v3"),
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--no-final", action="store_true", help="Skip ppo_simple_final.zip")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--output",
        default=str(ROOT / "logs" / "stage1_align_v3" / "eval_ckpts_512.csv"),
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Evaluate a single checkpoint only (filename or path)",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        if not ckpt.is_file():
            ckpt = model_dir / args.checkpoint
        checkpoints = [ckpt]
    else:
        checkpoints = discover_checkpoints(model_dir, include_final=not args.no_final)

    if not checkpoints:
        raise FileNotFoundError(f"No .zip checkpoints in {model_dir}")

    deterministic = not args.stochastic
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "checkpoint",
        "episode",
        "enemy_y",
        "init_misalignment_deg",
        "steps",
        "max_hold",
        "final_hold",
        "max_align_cos",
        "max_misalignment_deg",
        "total_reward",
        "reason",
    ]

    all_rows = []
    print(
        f"Eval: {len(checkpoints)} checkpoints × {args.episodes} episodes, "
        f"max_steps={args.max_steps}, deterministic={deterministic}"
    )
    print(f"Env: {args.env_config}")
    print(f"Output: {out_path}")

    for i, ckpt in enumerate(checkpoints, 1):
        print(f"\n[{i}/{len(checkpoints)}] {ckpt.name}")
        try:
            rows = eval_checkpoint(
                ckpt,
                args.env_config,
                args.episodes,
                deterministic,
                args.max_steps,
            )
            all_rows.extend(rows)
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_rows)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            raise

    print("\n=== Summary (mean max_hold per checkpoint) ===")
    by_ckpt = {}
    init_mis_by_ckpt = {}
    for row in all_rows:
        by_ckpt.setdefault(row["checkpoint"], []).append(row["max_hold"])
        init_mis_by_ckpt.setdefault(row["checkpoint"], []).append(
            float(row.get("init_misalignment_deg") or 0.0)
        )
    for name in sorted(by_ckpt.keys(), key=lambda n: _ckpt_sort_key(Path(n))):
        holds = by_ckpt[name]
        inits = init_mis_by_ckpt[name]
        mean_hold = sum(holds) / len(holds)
        mean_init = sum(inits) / len(inits)
        print(
            f"  {name}: mean_max_hold={mean_hold:.1f} mean_init_mis={mean_init:.1f}°  ({holds})"
        )
    print(f"\nWrote {len(all_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
