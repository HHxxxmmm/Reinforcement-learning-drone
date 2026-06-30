"""Eval Phase-3 v3 pitch nocap checkpoints at fixed enemy y (-5..5).

Example:
    python scripts/eval_stage2_phase3_v3_pitch_nocap_by_y.py
"""
import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO

from envs.train_env import TrainEnv
from scripts.eval_policy import run_episode

MODEL_DIR = ROOT / "model/stage2_phase3_v3_pitch_nocap"
ENV_CONFIG = ROOT / "config/envs.stage2.phase3.v3_pitch_nocap.eval.yaml"
OUTPUT_DIR = ROOT / "logs/stage2_phase3_v3_pitch_nocap/eval"
SUMMARY_CSV = OUTPUT_DIR / "phase3_v3_pitch_nocap_by_y_summary.csv"
DEFAULT_Y_VALUES = list(range(-5, 6))


def _ckpt_sort_key(name):
    m = re.search(r"_(\d+)_steps", str(name))
    return int(m.group(1)) if m else -1


def set_fixed_enemy_y(env: TrainEnv, y: int):
    env.stage_cfg["enemy_y_range"] = [int(y), int(y)]
    env._init_kwargs = env._build_init_kwargs()


def eval_checkpoint(ckpt: Path, y_values, env_config: str, log_interval: int):
    env = TrainEnv(config_path=env_config)
    model = PPO.load(str(ckpt), env=env, device="auto")
    rows = []
    for y in y_values:
        set_fixed_enemy_y(env, y)
        print(f"  y={y:+d}", end=" ", flush=True)
        row = run_episode(env, model, deterministic=True, log_interval=log_interval, max_steps=0)
        row["checkpoint"] = ckpt.stem
        row["enemy_y"] = int(y)
        row["model"] = ckpt.name
        rows.append(row)
        kill_tag = "KILL" if row["killed"] else "----"
        print(
            f"-> pitch={row['final_pitch_deg']:+5.1f}deg "
            f"yaw={row['final_yaw_deg']:+5.1f}deg hp={row['enemy_hp_final']:.3f} [{kill_tag}]"
        )
    env.adaptor.close()
    return rows


def print_checkpoint_matrix(rows, y_values):
    by_y = {int(r["enemy_y"]): r for r in rows}
    hp_cells = " ".join(f"{y:+3d}:{by_y[y]['enemy_hp_final']:.2f}" for y in y_values if y in by_y)
    pitch_cells = " ".join(
        f"{y:+3d}:{by_y[y]['final_pitch_deg']:+4.1f}" for y in y_values if y in by_y
    )
    yaw_cells = " ".join(f"{y:+3d}:{by_y[y]['final_yaw_deg']:+5.1f}" for y in y_values if y in by_y)
    kills = sum(1 for r in rows if r["killed"])
    print(f"    enemy_hp  [{hp_cells}]  kills={kills}/{len(rows)}")
    print(f"    pitch_deg [{pitch_cells}]")
    print(f"    yaw_deg   [{yaw_cells}]")


def main():
    parser = argparse.ArgumentParser(description="Eval P3 v3 pitch nocap ckpts at fixed enemy y")
    parser.add_argument("--env-config", default=str(ENV_CONFIG))
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument("--y-values", default="", help="Comma-separated y list (default: -5..5)")
    parser.add_argument("--log-interval", type=int, default=0)
    parser.add_argument("--checkpoint", default="", help="Single ckpt filename or path")
    parser.add_argument("--output", default=str(SUMMARY_CSV))
    args = parser.parse_args()

    if args.y_values.strip():
        y_values = [int(x.strip()) for x in args.y_values.split(",") if x.strip()]
    else:
        y_values = DEFAULT_Y_VALUES

    model_dir = Path(args.model_dir)
    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        if not ckpt.is_file():
            ckpt = model_dir / args.checkpoint
        checkpoints = [ckpt]
    else:
        checkpoints = sorted(model_dir.glob("*.zip"), key=lambda p: _ckpt_sort_key(p.name))

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints in {model_dir}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_rows = []

    print(
        f"Evaluating {len(checkpoints)} checkpoints × {len(y_values)} y values "
        f"(1 ep/y, deterministic)"
    )
    print(f"Env: {args.env_config}")
    print(f"Y grid: {y_values}")

    for ckpt in checkpoints:
        print(f"\n>>> {ckpt.name}")
        rows = eval_checkpoint(ckpt, y_values, args.env_config, args.log_interval)
        all_rows.extend(rows)
        print_checkpoint_matrix(rows, y_values)

    fieldnames = [
        "checkpoint", "model", "enemy_y", "steps", "reason",
        "final_pitch_rad", "final_pitch_deg", "final_pitch_cmd",
        "final_yaw_rad", "final_yaw_deg", "enemy_hp_final", "enemy_hp_min",
        "damage_dealt", "killed", "total_reward", "init_enemy_y",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n=== Kills by checkpoint ===")
    by_ckpt = {}
    for row in all_rows:
        by_ckpt.setdefault(row["checkpoint"], []).append(row)
    for name in sorted(by_ckpt.keys(), key=_ckpt_sort_key):
        rows = by_ckpt[name]
        kills = sum(1 for r in rows if r["killed"])
        print(f"  {name:<45} K={kills}/{len(rows)}")

    print(f"\nWrote {len(all_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
