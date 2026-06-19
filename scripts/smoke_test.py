"""
阶段 1 冒烟测试：随机动作跑完一整局，验证 reset/step 通信链路。

用法（在 Python/ 目录下，且 UE 房间已开启）:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --episodes 3 --log-interval 200
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.train_env import TrainEnv


def format_state(label, state):
    return (
        f"{label}: pos=({state[0]:.1f},{state[1]:.1f},{state[2]:.1f}) "
        f"hp={state[12]:.3f}"
    )


def run_episode(env, episode_idx, max_steps, log_interval):
    print(f"\n=== Episode {episode_idx} ===")
    print(
        f"reset: room_id={env.room_id}, unit_id={env.unit_id}, "
        f"state={env.initial_state}, sync_step={env.sync_step}"
    )
    print("发送初始包，等待首帧观测...")
    obs, _ = env.reset()
    print(f"reset OK, obs shape={obs.shape}")
    print(format_state("我方", env.my_state))
    print(format_state("敌方", env.enemy_state))
    if all(abs(v) < 1e-6 for v in env.my_state[:3]) and all(abs(v) < 1e-6 for v in env.enemy_state[:3]):
        print(
            "警告: 双方位置全为 0，可能 room_id/端口不对，或尚未进入对战画面。"
        )

    total_reward = 0.0
    steps = 0
    terminated = truncated = False

    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if steps % log_interval == 0:
            print(
                f"  step={steps:5d}  "
                f"my_hp={env.my_state[12]:.3f}  "
                f"enemy_hp={env.enemy_state[12]:.3f}  "
                f"my_z={env.my_state[2]:.1f}  "
                f"enemy_pos=({env.enemy_state[0]:.1f},{env.enemy_state[1]:.1f},{env.enemy_state[2]:.1f})  "
                f"reward={reward:.3f}"
            )

        if max_steps and steps >= max_steps:
            print(f"  达到本地安全步数上限 {max_steps}，停止本局")
            break

    end_reason = "truncated" if truncated else ("terminated" if terminated else "max_steps")
    print(
        f"Episode {episode_idx} 结束: steps={steps}, "
        f"total_reward={total_reward:.3f}, reason={end_reason}"
    )
    print(format_state("我方", env.my_state))
    print(format_state("敌方", env.enemy_state))
    return steps, terminated, truncated, total_reward


def main():
    parser = argparse.ArgumentParser(description="Simple 场景 TCP 冒烟测试")
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "envs.yaml"),
        help="环境配置文件路径",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="连续跑几局（对应 UE 房间 maxEpisodes，建议先 1）",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="单局本地安全步数上限，0 表示不限制（交给平台终止）",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="每隔多少步打印一次状态",
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="跑完后执行 gymnasium env_checker.check_env",
    )
    args = parser.parse_args()

    print(f"连接配置: {args.config}")
    env = TrainEnv(config_path=args.config)
    print(
        f"配置就绪: {env.adaptor.host}:{env.adaptor.port} "
        f"(unit_id={env.unit_id}, room_id={env.room_id})"
    )
    if env.room_id == 0 and env.initial_state == 2:
        print(
            "警告: 联机模式 room_id=0，若 reset 超时，请在 envs.yaml 填入 UE 房间的「加入房间uid」"
        )

    if args.check_env:
        from gymnasium.utils import env_checker
        print("\n执行 env_checker.check_env（在跑局之前，避免房间结束后无法 reset）...")
        env_checker.check_env(env)
        print("env_checker 通过")

    results = []
    for ep in range(1, args.episodes + 1):
        results.append(run_episode(env, ep, args.max_steps, args.log_interval))

    print("\n=== 汇总 ===")
    for i, (steps, terminated, truncated, total_reward) in enumerate(results, 1):
        print(
            f"  Episode {i}: steps={steps}, terminated={terminated}, "
            f"truncated={truncated}, reward={total_reward:.3f}"
        )

    print("\n冒烟测试完成。")


if __name__ == "__main__":
    main()
