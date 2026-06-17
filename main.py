"""
main函数，用于训练模型并保存
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
from envs.train_env import TrainEnv
from stable_baselines3 import PPO, SAC, TD3
from torch import nn
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback

def main():
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)
    logger = configure(log_dir, ["stdout", "csv"])
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path="./model/",
        name_prefix="model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    base_env = TrainEnv(config_path='./config/envs.yaml')
    env = Monitor(base_env)

    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        ortho_init=False,
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=2e-4,  # 先 3e-4；KL 高就降至 2e-4
        n_steps=512,  # 512（每env）
        batch_size=256,  # e.g. 1024
        n_epochs=5,  # 3-5 都行
        gamma=0.99,  # 你的任务短回合 → 0.98更灵敏
        gae_lambda=0.95,
        clip_range=0.2,  # 0.15~0.2 之间
        ent_coef=0.003,  # 探索；看entropy曲线微调
        vf_coef=0.6,  # 值函数比重略高些
        max_grad_norm=0.5,
        target_kl=0.02,  # 早停阈：0.015~0.03
        policy_kwargs=policy_kwargs,
        device="cpu",
    )
    model.set_logger(logger)
    model.learn(total_timesteps=20000,progress_bar=True, reset_num_timesteps=False,log_interval=1,
                callback=checkpoint_callback)

if __name__ == "__main__":
    main()