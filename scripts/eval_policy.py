"""Run a trained PPO checkpoint for visual evaluation in UE.

Start the UE room first, update config/envs.yaml, then run for example:

    python scripts/eval_policy.py --model ./model/ppo_simple_25000_steps.zip
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO

from envs.train_env import TrainEnv


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

    for episode in range(1, args.episodes + 1):
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False
        print(f"\n=== Eval episode {episode} ===")

        while not (terminated or truncated):
            agent_action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(agent_action)
            total_reward += reward
            steps += 1

            if steps % args.log_interval == 0:
                comps = info.get("reward_comps", {})
                print(
                    f"step={steps:5d} reward={reward:8.3f} total={total_reward:9.3f} "
                    f"my_hp={env.my_state[12]:.3f} enemy_hp={env.enemy_state[12]:.3f} "
                    f"attack_box={comps.get('attack_box', 0.0):.3f} "
                    f"enemy_damage={comps.get('enemy_damage', 0.0):.3f}"
                )

            if args.max_steps and steps >= args.max_steps:
                print(f"Reached local max steps: {args.max_steps}")
                break

        reason = "truncated" if truncated else ("terminated" if terminated else "max_steps")
        print(
            f"Episode {episode} finished: steps={steps}, reason={reason}, "
            f"total_reward={total_reward:.3f}, enemy_hp={env.enemy_state[12]:.3f}"
        )

    env.adaptor.close()


if __name__ == "__main__":
    main()
