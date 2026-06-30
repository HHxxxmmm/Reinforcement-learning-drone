# Stage-1 对准训练 — 会话总结（交接下个会话）

> 课程大作业第一题子阶段：Simple 固定靶、油门锁定，只学 pitch/yaw 对准并保持。  
> 代码目录：`Reinforcement-learning-drone/`  
> 最后更新：2026-06-30

---

## 1. 本阶段目标

- **任务**：从略带偏差的初始姿态，把机头对准靶机视线（LOS），并连续保持一段时间即算成功。
- **不做**：追击、开火、油门控制（油门恒为 0）。
- **成功后**：可接 Stage-2（初速 10、完整 combat `reward.py`、击落靶机）。

---

## 2. 平台关键结论（踩坑记录）

| 现象 | 结论 |
|------|------|
| 初始包 `roll/pitch/yaw` 用 degrees / milliradians 发 | **无效或严重偏差**；不要用 |
| 初始包 `yaw` 填整数 **3** | 作业里「背对」用法；本阶段已不用 |
| 初始包 `yaw=0` + 敌方 **y 横向偏移** | **推荐**：小幅视线角差，无需 setup |
| `setup_orientation` 启发式摆姿态 | **默认关闭**；曾出现跑满 200 步、摆到 0° 等问题 |
| 固定靶 `enemy_pos` 观测常为 `(0,0,0)` | `observation` / `reward` 用 `set_enemy_fallback_position()` 跟每局 `enemy_pos` 对齐 |
| 位置单位 | **1 unit = 10 m**；速度 **1 unit/s = 10 m/s** |
| HP（Simple） | 观测为 **0~1** 浮点，满血约 `1.0` |

**参考实现**：项目根目录 `initialize.py` 中 `align_stage1` / `align_ta_v2`（`yaw` 为 int32）。已迁入 `utils/initialize_align.py`。

---

## 3. 当前初始化方案（最终版）

```
我方: (0, 0, 100)   → 高度 1000 m
敌方: (120, y, 100) → 前方 1200 m，y 每局随机
yaw:  0（int32，机头朝 +x）
油门/速度: 0
```

**y 随机范围：`[-8, 8]` unit（±80 m）**

水平视线角差近似：`atan(|y| / 120)`：

| y | 角差 |
|---|------|
| ±2 | ~1.0° |
| ±5 | ~2.4° |
| ±8 | ~3.8°（最大初始偏差） |

配置见 `config/envs.stage1.yaml`：

```yaml
stage: align1
stage_params:
  init_mode: align_ta_v2
  initial_yaw: 0
  enemy_y_range: [-8, 8]
  setup_orientation: false
```

reset 日志示例：

```
[align1 reset] sent_yaw=0 enemy_y=5 obs_yaw=... misalignment=2.4° setup_steps=0
```

---

## 4. 对准成功条件（严格）

| 参数 | 值 | 含义 |
|------|-----|------|
| `align_target_deg` | **1.0** | 视线角 ≤ 1° 才算「对准」 |
| `hold_steps` | **8** | 起始需连续保持 8 步（~0.13s @60Hz） |
| `hold_steps_max` | 40 | 课程上限 |
| `hold_steps_increment` | 4 | 每档 +4 步 |
| `hold_steps_every_successes` | 3 | 每成功 3 局升一档 |

**必须比初始更准**：初始最大 ~3.8°，成功要求 ≤1°。

实现：`train_env.py` 中 `_align_cos_threshold()` 读 `align_target_deg`；`_hold_curriculum_successes` 累计成功局数递进 `hold_steps`。

---

## 5. 奖励函数（`utils/reward_align.py`）

| 分项 | 公式/逻辑 |
|------|-----------|
| `cosine` | `5.0 × (cos + 1) / 2`，**cos 越大奖励越高** |
| `cosine_progress` | 本步比上步 cos 提高则加分 |
| `tight` | 仅当 cos ≥ cos(1°) 时每步 +1.0 |
| `hold` | 对准区间内，随连续保持步数增加 |
| `stability` | 对过大角速度轻罚 |
| `step` | -0.003 |
| `success_bonus` | 达成 hold 课程则 +80 |
| `timeout_penalty` | 512 步未成功 -1.5 |

TensorBoard 可看 `reward/r/cosine_mean`、`reward/r/hold_mean`、`reward/r/tight_mean`。

---

## 6. 动作与 PPO

**动作（`config/envs.stage1.yaml` → `action`）**

| 项 | 值 |
|----|-----|
| `lock_throttle` | true（恒 0） |
| `lock_roll` | true |
| `action_scale` | 0.35（限制舵量，防甩头） |

**PPO（`config/algs.stage1.yaml`）**

- `ent_coef: 0.001`，`clip_range: 0.1`，`vf_coef: 0.3`，`log_std_init: -1.0`
- `total_timesteps: 80000`
- 日志/模型：`./logs/stage1_align_v2/`，`./model/stage1_align_v2/`

---

## 7. 怎么跑

```powershell
cd D:\Term6\无人系统设计\project\Reinforcement-learning-drone
..\venv\Scripts\Activate.ps1

# UE 建房后进对战画面，改 envs.stage1.yaml 的 port / room_id
python main.py --env-config ./config/envs.stage1.yaml --config ./config/algs.stage1.yaml

tensorboard --logdir ./logs/stage1_align_v2/
```

**冒烟 / 查初始姿态：**

```powershell
python scripts/probe_align_init.py --config ./config/envs.stage1.yaml
```

**单元测试：**

```powershell
python -m unittest tests.test_align_stage1 -v
```

---

## 8. 训练健康指标

| 指标 | 正常趋势 |
|------|----------|
| `rollout/ep_rew_mean` | 上升，成功局出现明显正值 |
| `reward/r/cosine_mean` | 接近 5（cos→1） |
| `reward/r/tight_mean` | > 0 且逐渐稳定 |
| `reward/r/hold_mean` | 开始出现正值 |
| `reward/r/success_bonus_mean` | 偶发尖峰 |
| `rollout/ep_len_mean` | 成功局变短（提前 terminated） |
| reset `misalignment` | 约 0°~3.8°，`setup_steps=0` |

**异常**

- `alignment` 长期为负 / `setup_steps=200` → 检查是否误开 `setup_orientation` 或 yaw 编码错误
- `hold_mean` 一直 0 → 1° 过严或 `action_scale` 太小；可试 `align_target_deg: 1.5` 或 `action_scale: 0.5`
- 第二局断连 → UE **比赛轮数**不够（建议 ≥200）

---

## 9. 相关文件一览

| 文件 | 作用 |
|------|------|
| `config/envs.stage1.yaml` | 房间、stage 参数、动作约束 |
| `config/algs.stage1.yaml` | PPO 超参 |
| `utils/initialize_align.py` | 初始化（y 随机、yaw=0） |
| `utils/reward_align.py` | Stage-1 奖励 |
| `utils/truncate_align.py` | 仅坠地截断 |
| `envs/train_env.py` | `stage: align1` 分支、hold 课程、reset 日志 |
| `utils/action.py` | `lock_throttle` / `lock_roll` / `action_scale` |
| `main.py` | 训练入口 |
| `scripts/probe_align_init.py` | 在线查 reset 姿态 |
| `tests/test_align_stage1.py` | 单测 |
| 项目根 `initialize.py` | 原始 align 模式参考（`align_stage1`、`align_ta_v2` 等） |

**可忽略 / 勿默认开启**

- `utils/orient_setup.py`（仅 `setup_orientation: true` 时用）

---

## 10. 课程递进建议（下个会话）

1. **Stage-1 练稳**：`success_bonus` 稳定后，hold 会自动升到 40 步。
2. **加大泛化**：`enemy_y_range` 改为 `[-12, 12]`（~5.7°）或恢复 `initial_yaw: random_pm1`（与 y 组合慎用）。
3. **收紧对准**：`align_target_deg: 0.5`。
4. **Stage-2 接近战**：
   - `stage` 改 `combat` 或新 config
   - `utils/initialize.py`：初速 **10**，敌方 `(120, y, 100)`
   - `utils/reward.py` + 油门放开
   - 可加载 `model/stage1_align_v2/*.zip` 续训（需在 `main.py` 加 `--load` 或手动 `model.learn(reset_num_timesteps=False)`）

---

## 11. 当前房间配置（训练前务必改）

`config/envs.stage1.yaml` 中（会话末）：

```yaml
port: 1002
room_id: 19943
```

每次 UE 新建房间后更新 `port`、`room_id`；**最大回合数** 建议与 `max_steps_per_episode: 512` 一致。

---

## 12. 决策时间线（便于写报告）

1. 初版：30° 圆锥 + setup → 平台不吃错误角度编码  
2. 改用根目录 `yaw=3` 背对 → 用户改为不要背对  
3. `yaw=±1` 随机 → 用户改为 `yaw=0` + **y 偏移**  
4. 奖励：cos 单调 + hold 课程；对准从 15° 收紧到 **1°**  
5. 去掉 setup，reset 应 `setup_steps=0`

---

*下个会话可直接说：「继续 Stage-1」或「按 STAGE1_会话总结 接 Stage-2」。*
