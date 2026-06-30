"""Batch eval Junior Q2 Phase-2 v2 checkpoints (pitch/yaw/dy/dz + enemy HP).

Example:
    python scripts/eval_junior_phase2_v2.py --episodes 3
    python scripts/eval_junior_phase2_v2.py --checkpoint ppo_junior_p2_v2_1000_steps.zip
"""
import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model/junior_q2_phase2_v2"
ENV_CONFIG = ROOT / "config/envs.junior.phase2.v2.eval.yaml"
OUTPUT_DIR = ROOT / "logs/junior_q2_phase2_v2/eval"
SUMMARY_CSV = OUTPUT_DIR / "phase2_v2_summary.csv"


def _ckpt_sort_key(name):
    m = re.search(r"_(\d+)_steps", str(name))
    return int(m.group(1)) if m else -1


def main():
    parser = argparse.ArgumentParser(description="Batch eval Junior Phase-2 v2 checkpoints")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--log-interval", type=int, default=0, help="0 = quiet per-step logs")
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Also write per-step pitch/yaw/dy/dz trace CSV for each checkpoint",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Single checkpoint filename or path (default: all in model/junior_q2_phase2_v2/)",
    )
    parser.add_argument(
        "--model-dir",
        default=str(MODEL_DIR),
        help="Directory containing .zip checkpoints",
    )
    args = parser.parse_args()

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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []

    print(f"Evaluating {len(checkpoints)} checkpoints × {args.episodes} episodes")
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
        if args.trace:
            cmd.extend(["--trace-output", str(OUTPUT_DIR / f"{ckpt.stem}_trace.csv")])

        print(f"\n>>> {ckpt.name}")
        subprocess.run(cmd, cwd=str(ROOT), check=True)
        with open(out, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            row["checkpoint"] = ckpt.stem
        all_rows.extend(rows)
        if rows:
            pitches = [float(r["final_pitch_deg"]) for r in rows]
            yaws = [float(r["final_yaw_deg"]) for r in rows]
            dys = [float(r["final_dy_m"]) for r in rows]
            dzs = [float(r["final_dz_m"]) for r in rows]
            hps = [float(r["enemy_hp_final"]) for r in rows]
            kills = sum(1 for r in rows if str(r["killed"]).lower() == "true")
            print(
                f"    pitch={sum(pitches)/len(pitches):+.2f}deg  "
                f"yaw={sum(yaws)/len(yaws):+.2f}deg  "
                f"dy={sum(dys)/len(dys):+.1f}m  dz={sum(dzs)/len(dzs):+.1f}m  "
                f"hp={sum(hps)/len(hps):.3f}  kills={kills}/{len(rows)}"
            )

    fieldnames = [
        "checkpoint",
        "model",
        "episode",
        "steps",
        "reason",
        "init_my_y",
        "init_my_z",
        "final_pitch_rad",
        "final_pitch_deg",
        "final_yaw_rad",
        "final_yaw_deg",
        "final_dy_m",
        "final_dz_m",
        "final_cmd_pitch",
        "final_cmd_yaw",
        "enemy_hp_final",
        "enemy_hp_min",
        "damage_dealt",
        "killed",
        "total_reward",
        "init_enemy_y",
    ]
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print("\n=== Junior Phase-2 v2 summary ===")
    print(f"{'checkpoint':<32} {'pitch':>7} {'yaw':>7} {'dy_m':>7} {'dz_m':>7} {'hp':>6} {'kills':>6}")
    by_ckpt = {}
    for row in all_rows:
        by_ckpt.setdefault(row["checkpoint"], []).append(row)
    for name in sorted(by_ckpt.keys(), key=_ckpt_sort_key):
        rows = by_ckpt[name]
        pitch = sum(float(r["final_pitch_deg"]) for r in rows) / len(rows)
        yaw = sum(float(r["final_yaw_deg"]) for r in rows) / len(rows)
        dy = sum(float(r["final_dy_m"]) for r in rows) / len(rows)
        dz = sum(float(r["final_dz_m"]) for r in rows) / len(rows)
        hp = sum(float(r["enemy_hp_final"]) for r in rows) / len(rows)
        kills = sum(1 for r in rows if str(r["killed"]).lower() == "true")
        print(
            f"  {name:<32} {pitch:+7.2f} {yaw:+7.2f} {dy:+7.1f} {dz:+7.1f} {hp:6.3f} {kills}/{len(rows)}"
        )
    print(f"\nSummary CSV: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
