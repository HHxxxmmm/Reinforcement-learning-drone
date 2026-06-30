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


def attack_geometry(my_state, enemy_state):
    distance, _, rel = reward_utils._range_and_los(my_state, enemy_state)
    forward = reward_utils._forward_vector(my_state)
    forward_distance = float((rel * forward).sum())
    lateral_error = float(((rel - forward_distance * forward) ** 2).sum() ** 0.5)
    return distance, forward_distance, lateral_error


def run_episode(env, model, deterministic, log_interval, max_steps):
    obs, reset_info = env.reset()
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    min_enemy_hp = 1.0
    last_info = {}

    while not (terminated or truncated):
        agent_action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, last_info = env.step(agent_action)
        total_reward += float(reward)
        steps += 1
        min_enemy_hp = min(min_enemy_hp, float(env.enemy_state[12]))

        if log_interval and steps % log_interval == 0:
            comps = last_info.get("reward_comps", {})
            real_action = env._marshal_action(agent_action)
            distance, forward_distance, lateral_error = attack_geometry(env.my_state, env.enemy_state)
            print(
                f"step={steps:5d} reward={reward:8.3f} total={total_reward:9.3f} "
                f"dist={distance:7.1f} fwd={forward_distance:7.1f} lat={lateral_error:7.1f} "
                f"my_hp={env.my_state[12]:.3f} enemy_hp={env.enemy_state[12]:.3f} "
                f"yaw={env.my_state[5]:.3f} act=[{real_action[0]:.2f},{real_action[1]:.2f},{real_action[2]:.2f},{real_action[3]:.2f}] "
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
    enemy_hp_final = float(env.enemy_state[12])
    last_real = getattr(env, "_last_real_action", None)
    final_pitch_cmd = float(last_real[1]) if last_real is not None else float("nan")

    return {
        "steps": steps,
        "reason": reason,
        "total_reward": total_reward,
        "final_pitch_rad": final_pitch_rad,
        "final_pitch_deg": float(np.degrees(final_pitch_rad)),
        "final_pitch_cmd": final_pitch_cmd,
        "final_yaw_rad": final_yaw_rad,
        "final_yaw_deg": float(np.degrees(final_yaw_rad)),
        "enemy_hp_final": enemy_hp_final,
        "enemy_hp_min": min_enemy_hp,
        "damage_dealt": 1.0 - min_enemy_hp,
        "killed": enemy_hp_final <= 0.01,
        "init_enemy_y": reset_info.get("init_enemy_y"),
    }


def print_episode_result(episode, row):
    kill_tag = "KILL" if row["killed"] else "no_kill"
    pitch_cmd = row.get("final_pitch_cmd", float("nan"))
    pitch_cmd_s = f" pitch_cmd={pitch_cmd:+.3f}" if np.isfinite(pitch_cmd) else ""
    print(
        f"Episode {episode}: steps={row['steps']} reason={row['reason']} "
        f"final_pitch={row['final_pitch_deg']:+.2f}deg ({row['final_pitch_rad']:+.4f} rad){pitch_cmd_s} "
        f"final_yaw={row['final_yaw_deg']:+.2f}deg ({row['final_yaw_rad']:+.4f} rad) "
        f"enemy_hp={row['enemy_hp_final']:.3f} min_hp={row['enemy_hp_min']:.3f} "
        f"damage={row['damage_dealt']:.3f} [{kill_tag}] reward={row['total_reward']:.1f}"
    )


def print_summary(rows):
    n = len(rows)
    if n == 0:
        return

    yaws = [row["final_yaw_deg"] for row in rows]
    pitches = [row["final_pitch_deg"] for row in rows]
    hps = [row["enemy_hp_final"] for row in rows]
    kills = sum(1 for row in rows if row["killed"])

    print("\n=== Eval summary (final pitch/yaw + enemy HP) ===")
    print(f"  episodes={n}  kills={kills}/{n}")
    print(f"  final_pitch_deg: mean={np.mean(pitches):+.2f}deg  std={np.std(pitches):.2f}deg  "
          f"min={min(pitches):+.2f}deg  max={max(pitches):+.2f}deg")
    print(f"  final_yaw_deg: mean={np.mean(yaws):+.2f}deg  std={np.std(yaws):.2f}deg  "
          f"min={min(yaws):+.2f}deg  max={max(yaws):+.2f}deg")
    print(f"  enemy_hp_final: mean={np.mean(hps):.3f}  min={min(hps):.3f}  max={max(hps):.3f}")
    print(f"  damage_dealt: mean={np.mean([row['damage_dealt'] for row in rows]):.3f}")
    print("\n  ep | final_pitch_deg | final_yaw_deg | enemy_hp | killed | reason")
    for i, row in enumerate(rows, 1):
        print(
            f"  {i:2d} | {row['final_pitch_deg']:+8.2f}deg | {row['final_yaw_deg']:+8.2f}deg | "
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
        help="Optional CSV path for per-episode metrics (final_yaw_deg, enemy_hp_final, ...)",
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
    print("Primary metrics: final_pitch_deg, final_yaw_deg, enemy_hp_final")

    rows = []
    for episode in range(1, args.episodes + 1):
        print(f"\n=== Eval episode {episode} ===")
        row = run_episode(
            env,
            model,
            deterministic=deterministic,
            log_interval=args.log_interval,
            max_steps=args.max_steps,
        )
        row["episode"] = episode
        row["model"] = model_path.name
        rows.append(row)
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
            "final_pitch_rad",
            "final_pitch_deg",
            "final_pitch_cmd",
            "final_yaw_rad",
            "final_yaw_deg",
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
        print(f"\nWrote {len(rows)} rows to {out_path}")

    env.adaptor.close()


if __name__ == "__main__":
    main()
