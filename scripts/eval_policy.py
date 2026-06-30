"""Run a trained PPO checkpoint for visual evaluation in UE.

Start the UE room first, update config/envs.yaml, then run for example:

    python scripts/eval_policy.py \\
        --model ./model/stage2_phase2/ppo_combat_p2_50000_steps.zip \\
        --env-config ./config/envs.stage2.phase2.eval.yaml \\
        --episodes 3
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO

from envs.train_env import TrainEnv
from utils import reward as reward_utils

POSITION_SCALE_M = reward_utils.POSITION_SCALE_M


def attack_geometry(my_state, enemy_state):
    distance, _, rel = reward_utils._range_and_los(my_state, enemy_state)
    forward = reward_utils._forward_vector(my_state)
    forward_distance = float((rel * forward).sum())
    lateral_error = float(((rel - forward_distance * forward) ** 2).sum() ** 0.5)
    return distance, forward_distance, lateral_error


def run_episode(env, model, deterministic, log_interval, max_steps, trace_rows=None):
    obs, reset_info = env.reset()
    init_my_y = float(env.my_state[1])
    init_my_z = float(env.my_state[2])
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    min_enemy_hp = 1.0
    last_info = {}
    last_real_action = np.zeros(4, dtype=np.float64)

    while not (terminated or truncated):
        agent_action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, last_info = env.step(agent_action)
        total_reward += float(reward)
        steps += 1
        min_enemy_hp = min(min_enemy_hp, float(env.enemy_state[12]))
        last_real_action = env._marshal_action(agent_action)
        pitch_rad = float(env.my_state[4])
        yaw_rad = float(env.my_state[5])
        dy_m = (float(env.my_state[1]) - init_my_y) * POSITION_SCALE_M
        dz_m = (float(env.my_state[2]) - init_my_z) * POSITION_SCALE_M

        if trace_rows is not None:
            trace_rows.append(
                {
                    "step": steps,
                    "pitch_rad": pitch_rad,
                    "pitch_deg": float(np.degrees(pitch_rad)),
                    "yaw_rad": yaw_rad,
                    "yaw_deg": float(np.degrees(yaw_rad)),
                    "dy_m": dy_m,
                    "dz_m": dz_m,
                    "cmd_pitch": float(last_real_action[1]),
                    "cmd_yaw": float(last_real_action[3]),
                    "enemy_hp": float(env.enemy_state[12]),
                    "reward": float(reward),
                }
            )

        if log_interval and steps % log_interval == 0:
            comps = last_info.get("reward_comps", {})
            distance, forward_distance, lateral_error = attack_geometry(env.my_state, env.enemy_state)
            print(
                f"step={steps:5d} reward={reward:8.3f} total={total_reward:9.3f} "
                f"dist={distance:7.1f} fwd={forward_distance:7.1f} lat={lateral_error:7.1f} "
                f"my_hp={env.my_state[12]:.3f} enemy_hp={env.enemy_state[12]:.3f} "
                f"pitch={pitch_rad:+.3f}rad yaw={yaw_rad:+.3f}rad "
                f"dy={dy_m:+.1f}m dz={dz_m:+.1f}m "
                f"act=[{last_real_action[0]:.2f},{last_real_action[1]:.2f},{last_real_action[2]:.2f},{last_real_action[3]:.2f}] "
                f"attack_box={comps.get('attack_box', 0.0):.3f} "
                f"centerline={comps.get('centerline', 0.0):.3f} "
                f"enemy_damage={comps.get('enemy_damage', 0.0):.3f}"
            )

        if max_steps and steps >= max_steps:
            print(f"Reached local max steps: {max_steps}")
            env._send_finish_round()
            break

    if truncated:
        reason = "truncated"
    elif terminated:
        reason = "terminated"
    elif max_steps and steps >= max_steps:
        reason = "max_steps"
    else:
        reason = "stopped"

    final_pitch_rad = float(env.my_state[4])
    final_yaw_rad = float(env.my_state[5])
    final_dy_m = (float(env.my_state[1]) - init_my_y) * POSITION_SCALE_M
    final_dz_m = (float(env.my_state[2]) - init_my_z) * POSITION_SCALE_M
    enemy_hp_final = float(env.enemy_state[12])

    return {
        "steps": steps,
        "reason": reason,
        "total_reward": total_reward,
        "init_my_y": init_my_y,
        "init_my_z": init_my_z,
        "final_pitch_rad": final_pitch_rad,
        "final_pitch_deg": float(np.degrees(final_pitch_rad)),
        "final_yaw_rad": final_yaw_rad,
        "final_yaw_deg": float(np.degrees(final_yaw_rad)),
        "final_dy_m": final_dy_m,
        "final_dz_m": final_dz_m,
        "final_cmd_pitch": float(last_real_action[1]),
        "final_cmd_yaw": float(last_real_action[3]),
        "enemy_hp_final": enemy_hp_final,
        "enemy_hp_min": min_enemy_hp,
        "damage_dealt": 1.0 - min_enemy_hp,
        "killed": enemy_hp_final <= 0.01,
        "init_enemy_y": reset_info.get("init_enemy_y"),
    }


def print_episode_result(episode, row):
    kill_tag = "KILL" if row["killed"] else "no_kill"
    print(
        f"Episode {episode}: steps={row['steps']} reason={row['reason']} "
        f"pitch={row['final_pitch_deg']:+.2f}deg ({row['final_pitch_rad']:+.4f} rad) "
        f"yaw={row['final_yaw_deg']:+.2f}deg ({row['final_yaw_rad']:+.4f} rad) "
        f"dy={row['final_dy_m']:+.1f}m dz={row['final_dz_m']:+.1f}m "
        f"cmd_pitch={row['final_cmd_pitch']:+.3f} cmd_yaw={row['final_cmd_yaw']:+.3f} "
        f"enemy_hp={row['enemy_hp_final']:.3f} min_hp={row['enemy_hp_min']:.3f} "
        f"damage={row['damage_dealt']:.3f} [{kill_tag}] reward={row['total_reward']:.1f}"
    )


def print_summary(rows):
    n = len(rows)
    if n == 0:
        return

    yaws = [row["final_yaw_deg"] for row in rows]
    hps = [row["enemy_hp_final"] for row in rows]
    kills = sum(1 for row in rows if row["killed"])

    pitches = [row["final_pitch_deg"] for row in rows]
    dys = [row["final_dy_m"] for row in rows]
    dzs = [row["final_dz_m"] for row in rows]

    print("\n=== Eval summary ===")
    print(f"  episodes={n}  kills={kills}/{n}")
    print(
        f"  final_pitch_deg: mean={np.mean(pitches):+.2f}  std={np.std(pitches):.2f}  "
        f"min={min(pitches):+.2f}  max={max(pitches):+.2f}"
    )
    print(
        f"  final_yaw_deg:   mean={np.mean(yaws):+.2f}  std={np.std(yaws):.2f}  "
        f"min={min(yaws):+.2f}  max={max(yaws):+.2f}"
    )
    print(
        f"  final_dy_m:      mean={np.mean(dys):+.1f}  std={np.std(dys):.1f}  "
        f"min={min(dys):+.1f}  max={max(dys):+.1f}"
    )
    print(
        f"  final_dz_m:      mean={np.mean(dzs):+.1f}  std={np.std(dzs):.1f}  "
        f"min={min(dzs):+.1f}  max={max(dzs):+.1f}"
    )
    print(f"  enemy_hp_final: mean={np.mean(hps):.3f}  min={min(hps):.3f}  max={max(hps):.3f}")
    print(f"  damage_dealt: mean={np.mean([row['damage_dealt'] for row in rows]):.3f}")
    print("\n  ep | pitch_deg | yaw_deg | dy_m | dz_m | enemy_hp | killed | reason")
    for i, row in enumerate(rows, 1):
        print(
            f"  {i:2d} | {row['final_pitch_deg']:+7.2f} | {row['final_yaw_deg']:+6.2f} | "
            f"{row['final_dy_m']:+5.1f} | {row['final_dz_m']:+5.1f} | "
            f"{row['enemy_hp_final']:6.3f} | {'yes' if row['killed'] else ' no '} | {row['reason']}"
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO policy in UE")
    parser.add_argument(
        "--model",
        default=str(ROOT / "model" / "ppo_simple_25000_steps.zip"),
        help="Path to a trained PPO .zip checkpoint",
    )
    parser.add_argument(
        "--env-config",
        default=str(ROOT / "config" / "envs.yaml"),
        help="Environment config path",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Optional local safety limit. 0 means wait for UE termination/truncation.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=60,
        help="Print state every N steps",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy actions instead of deterministic actions",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional CSV path for per-episode metrics",
    )
    parser.add_argument(
        "--trace-output",
        default="",
        help="Optional CSV path for per-step pitch/yaw/dy/dz trace",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    env = TrainEnv(config_path=args.env_config)
    model = PPO.load(str(model_path), env=env, device="auto")
    deterministic = not args.stochastic

    print(f"Loaded model: {model_path}")
    print(f"Environment: {env.adaptor.host}:{env.adaptor.port} room={env.room_id} unit={env.unit_id}")
    print("Keep the UE battle window visible if you want to record the flight.")
    print("Primary metrics: pitch, yaw, dy/dz offset, enemy_hp_final")

    rows = []
    trace_rows = []
    for episode in range(1, args.episodes + 1):
        print(f"\n=== Eval episode {episode} ===")
        ep_trace = [] if args.trace_output else None
        row = run_episode(
            env,
            model,
            deterministic=deterministic,
            log_interval=args.log_interval,
            max_steps=args.max_steps,
            trace_rows=ep_trace,
        )
        row["episode"] = episode
        row["model"] = model_path.name
        rows.append(row)
        if ep_trace is not None:
            for tr in ep_trace:
                tr["episode"] = episode
                tr["model"] = model_path.name
            trace_rows.extend(ep_trace)
        print_episode_result(episode, row)

    print_summary(rows)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
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
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} episode rows to {out_path}")

    if args.trace_output:
        trace_path = Path(args.trace_output)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_fields = [
            "model",
            "episode",
            "step",
            "pitch_rad",
            "pitch_deg",
            "yaw_rad",
            "yaw_deg",
            "dy_m",
            "dz_m",
            "cmd_pitch",
            "cmd_yaw",
            "enemy_hp",
            "reward",
        ]
        with open(trace_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=trace_fields)
            writer.writeheader()
            writer.writerows(trace_rows)
        print(f"Wrote {len(trace_rows)} trace rows to {trace_path}")

    env.adaptor.close()


if __name__ == "__main__":
    main()
