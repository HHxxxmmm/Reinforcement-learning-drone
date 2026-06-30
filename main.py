"""
训练入口：PPO + Checkpoint + 奖励分项回调 + TensorBoard。

用法（UE 房间已开、envs.yaml 已配对）:
    python main.py
    python main.py --config ./config/algs.yaml
"""
import argparse
import os
from pathlib import Path

import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from torch import nn

from envs.train_env import TrainEnv
from utils.callback import RewardComponentsCallback
from utils.policy_reset import reset_policy_pitch_head, reset_policy_yaw_head

ROOT = Path(__file__).resolve().parent


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(device_cfg):
    if device_cfg is None or device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg


def build_policy_kwargs(raw):
    raw = raw or {}
    net_arch = raw.get("net_arch", {"pi": [128, 128], "vf": [128, 128]})
    kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=net_arch,
        ortho_init=bool(raw.get("ortho_init", False)),
    )
    if "log_std_init" in raw:
        kwargs["log_std_init"] = float(raw["log_std_init"])
    return kwargs


def main():
    parser = argparse.ArgumentParser(description="PPO 训练")
    parser.add_argument(
        "--env-config",
        default=str(ROOT / "config" / "envs.yaml"),
        help="仿真环境配置",
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "algs.yaml"),
        help="算法与训练超参配置",
    )
    parser.add_argument(
        "--load",
        default="",
        help="从已有 PPO .zip 续训（覆盖 config 中的 load_path）",
    )
    args = parser.parse_args()

    alg_cfg = load_yaml(args.config)
    env_cfg = load_yaml(args.env_config)
    load_path = args.load or alg_cfg.get("load_path") or ""

    log_dir = alg_cfg.get("log_dir", "./logs/")
    model_dir = alg_cfg.get("model_dir", "./model/")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    device = resolve_device(alg_cfg.get("device", "auto"))
    ppo_cfg = alg_cfg.get("ppo", {})

    print(f"环境: {env_cfg.get('host')}:{env_cfg.get('port')} room={env_cfg.get('room_id')}")
    print(f"训练: {alg_cfg.get('algorithm', 'PPO')} device={device} steps={alg_cfg.get('total_timesteps')}")
    if load_path:
        print(f"续训: {load_path}  reset_num_timesteps={alg_cfg.get('reset_num_timesteps', True)}")
    print(f"日志: {log_dir}  模型: {model_dir}")
    print("请确认 UE 已进入对战画面后再开始训练。")

    logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=int(alg_cfg.get("checkpoint", {}).get("save_freq", 5000)),
            save_path=model_dir,
            name_prefix=alg_cfg.get("checkpoint", {}).get("name_prefix", "ppo_simple"),
        ),
        RewardComponentsCallback(
            csv_path=alg_cfg.get("reward_components_csv"),
        ),
    ])

    base_env = TrainEnv(config_path=args.env_config)
    env = Monitor(base_env, filename=os.path.join(log_dir, "monitor.csv"))

    ppo_kwargs = dict(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=float(ppo_cfg.get("learning_rate", 2e-4)),
        n_steps=int(ppo_cfg.get("n_steps", 512)),
        batch_size=int(ppo_cfg.get("batch_size", 256)),
        n_epochs=int(ppo_cfg.get("n_epochs", 5)),
        gamma=float(ppo_cfg.get("gamma", 0.99)),
        gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
        clip_range=float(ppo_cfg.get("clip_range", 0.2)),
        ent_coef=float(ppo_cfg.get("ent_coef", 0.003)),
        vf_coef=float(ppo_cfg.get("vf_coef", 0.6)),
        max_grad_norm=float(ppo_cfg.get("max_grad_norm", 0.5)),
        target_kl=float(ppo_cfg.get("target_kl", 0.02)),
        policy_kwargs=build_policy_kwargs(ppo_cfg.get("policy_kwargs")),
        device=device,
    )

    if load_path:
        if not os.path.isfile(load_path):
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")
        model = PPO.load(load_path, env=env, device=device)
        model.set_env(env)
        if alg_cfg.get("reset_yaw_head_on_load", False):
            info = reset_policy_yaw_head(
                model,
                yaw_action_idx=int(alg_cfg.get("yaw_action_idx", 3)),
                log_std_init=alg_cfg.get("yaw_log_std_init_on_load"),
            )
            print(
                "已重置 yaw 输出头（保留油门等其它维度）: "
                f"action_idx={info['yaw_action_idx']} bias={info['yaw_bias']:.4f} "
                f"log_std={info['yaw_log_std']}"
            )
        if alg_cfg.get("reset_pitch_head_on_load", False):
            info = reset_policy_pitch_head(
                model,
                pitch_action_idx=int(alg_cfg.get("pitch_action_idx", 1)),
                log_std_init=alg_cfg.get("pitch_log_std_init_on_load"),
            )
            print(
                "已重置 pitch 输出头（保留油门/yaw 等其它维度）: "
                f"action_idx={info['pitch_action_idx']} bias={info['pitch_bias']:.4f} "
                f"log_std={info['pitch_log_std']}"
            )
    else:
        model = PPO(**ppo_kwargs)
    model.set_logger(logger)
    model.learn(
        total_timesteps=int(alg_cfg.get("total_timesteps", 20000)),
        progress_bar=bool(alg_cfg.get("progress_bar", True)),
        reset_num_timesteps=bool(alg_cfg.get("reset_num_timesteps", True)),
        log_interval=int(alg_cfg.get("log_interval", 1)),
        callback=callbacks,
    )
    name_prefix = alg_cfg.get("checkpoint", {}).get("name_prefix", "ppo_simple")
    final_path = os.path.join(model_dir, f"{name_prefix}_final")
    model.save(final_path)
    print(f"训练完成，最终模型已保存: {final_path}.zip")


if __name__ == "__main__":
    main()
