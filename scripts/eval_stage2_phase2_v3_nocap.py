"""Batch eval Stage-2 Phase-2 v3 nocap checkpoints (no max_yaw).

Example:
    python scripts/eval_stage2_phase2_v3_nocap.py --episodes 3
"""
import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model/stage2_phase2_v3_nocap"
ENV_CONFIG = ROOT / "config/envs.stage2.phase2.v3.nocap.eval.yaml"
OUTPUT_DIR = ROOT / "logs/stage2_phase2_v3_nocap/eval"
SUMMARY_CSV = OUTPUT_DIR / "phase2_v3_nocap_summary.csv"


def _ckpt_sort_key(name):
    m = re.search(r"_(\d+)_steps", str(name))
    return int(m.group(1)) if m else -1


def main():
    parser = argparse.ArgumentParser(description="Batch eval Phase-2 v3 nocap combat checkpoints")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--log-interval", type=int, default=0, help="0 = quiet per-step logs")
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Single checkpoint filename or path (default: all in model/stage2_phase2_v3_nocap/)",
    )
    args = parser.parse_args()

    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        if not ckpt.is_file():
            ckpt = MODEL_DIR / args.checkpoint
        checkpoints = [ckpt]
    else:
        checkpoints = sorted(MODEL_DIR.glob("*.zip"), key=lambda p: _ckpt_sort_key(p.name))

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints in {MODEL_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []

    print(f"Evaluating {len(checkpoints)} Phase-2 v3 nocap checkpoints × {args.episodes} episodes")
    print(f"Env: {ENV_CONFIG}")

    for ckpt in checkpoints:
        out = OUTPUT_DIR / f"{ckpt.stem}.csv"
        cmd = [
            sys.executable,
            str(ROOT / "scripts/eval_policy.py"),
            "--model", str(ckpt),
            "--env-config", str(ENV_CONFIG),
            "--episodes", str(args.episodes),
            "--log-interval", str(args.log_interval),
            "--output", str(out),
        ]
        print(f"\n>>> {ckpt.name}")
        subprocess.run(cmd, cwd=str(ROOT), check=True)
        with open(out, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            row["checkpoint"] = ckpt.stem
        all_rows.extend(rows)
        if rows:
            yaws = [float(r["final_yaw_deg"]) for r in rows]
            hps = [float(r["enemy_hp_final"]) for r in rows]
            kills = sum(1 for r in rows if str(r["killed"]).lower() == "true")
            print(
                f"    yaw_mean={sum(yaws)/len(yaws):+.2f}deg  "
                f"hp_mean={sum(hps)/len(hps):.3f}  kills={kills}/{len(rows)}"
            )

    fieldnames = [
        "checkpoint", "model", "episode", "steps", "reason",
        "final_yaw_rad", "final_yaw_deg", "enemy_hp_final", "enemy_hp_min",
        "damage_dealt", "killed", "total_reward", "init_enemy_y",
    ]
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n=== Phase-2 v3 nocap summary (final_yaw_deg | enemy_hp | kills) ===")
    print(f"{'checkpoint':<40} {'yaw_deg':>8} {'enemy_hp':>9} {'kills':>6}")
    by_ckpt = {}
    for row in all_rows:
        by_ckpt.setdefault(row["checkpoint"], []).append(row)
    for name in sorted(by_ckpt.keys(), key=_ckpt_sort_key):
        rows = by_ckpt[name]
        yaw = sum(float(r["final_yaw_deg"]) for r in rows) / len(rows)
        hp = sum(float(r["enemy_hp_final"]) for r in rows) / len(rows)
        kills = sum(1 for r in rows if str(r["killed"]).lower() == "true")
        print(f"  {name:<40} {yaw:+8.2f} {hp:9.3f} {kills}/{len(rows)}")
    print(f"\nSummary CSV: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
