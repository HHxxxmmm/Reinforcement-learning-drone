# Stage-2 Phase-2 v2 / Phase-3 交接文档

> 课程大作业：Simple vs 固定靶机，PPO 在线 combat。  
> 代码目录：`Reinforcement-learning-drone/`  
> 最后更新：2026-06-30  
> **新开对话可说：「按 STAGE2_P3_会话总结 继续」**

---

## 1. 总体路线（当前采用）

```mermaid
flowchart LR
    P1[Phase-1 锁 yaw<br/>只学油门] -->|30000 ckpt| P2v2[Phase-2 v2<br/>y=0 解锁 yaw]
    P2v2 -->|5000 基线| P3[Phase-3<br/>y∈[-5,5] + +y偏置]
```

| 阶段 | 配置 | 加载 | 状态 |
|------|------|------|------|
| **P1** | `envs.stage2.phase1.yaml` | 从零 | ✅ 30000 ckpt 稳定击杀 |
| **P2 v2** | `envs.stage2.phase2.v2.yaml` | P1@30000 | ✅ y=0；**5000 步基线表现最好** |
| **P3** | `envs.stage2.phase3.yaml` | **p2_v2@5000** | ⚠️ 训练中；+y 侧弱于 -y |

旧路线 `stage2_phase2/`（无 v2）、Stage-1 续训已弃用，详见 `STAGE2_会话总结.md`。

---

## 2. UE 房间（训练 / eval 分开）

| 用途 | port | room_id | 配置 |
|------|------|---------|------|
| **P3 训练（当前）** | 1005 | 20195 | `envs.stage2.phase3.yaml` |
| **P3 / 基线 y 扫点 eval** | 1000 | 20204 | `envs.stage2.phase3.eval.yaml` |
| P2 v2 eval（y=0） | 1000 | 20177 | `envs.stage2.phase2.v2.eval.yaml` |

host 均为 `10.119.14.141`。**train 与 eval 不能同占一 room。**

---

## 3. 动作（三阶段共用）

```yaml
action:
  lock_throttle: false
  max_throttle: 0.35
  lock_roll: true
  lock_pitch: true
  lock_yaw: false          # P1 为 true
  max_yaw: 0.15            # 限制 yaw 幅度，防 P2 左拐过大
  action_scale: 0.4
```

实现：`utils/action.py`（`max_throttle` / `max_yaw` 类比上限，非 lock）。

---

## 4. 奖励（当前数值 + 可配置机制）

**yaml 可覆盖项**（未写则用 `utils/reward.py` 默认）：

| 参数 | 当前 P2v2 / P3 yaml | 代码默认 |
|------|---------------------|----------|
| `yaw_misalign_weight` | **10.0** | 0（惩罚：`-weight×(1-cos)`） |
| `damage_reward_per_hit` | **12.0** | 8 |
| `alignment_weight` | （未写） | 2.0 |
| `attack_box_weight` | （未写） | 3.0 |
| `corridor_weight` | （未写） | 1.2 |
| `centerline_weight` | （未写） | 1.8 |
| `finish_centerline_weight` | （未写） | 4.0 |
| `enemy_hp_shaping_weight` | （未写） | 4.0 |
| `kill_bonus` | （未写） | 300 |

曾尝试：减半对准奖励 + 加强伤害 + 降 yaw 惩罚到 6 → 已**回退**到上表（仅保留 yaml 权重**机制**）。

---

## 5. PPO 超参（P2 v2 / P3 稳定版）

```yaml
# algs.stage2.phase2.v2.yaml / algs.stage2.phase3.yaml
learning_rate: 5.0e-5    # 原 1e-4 易过优化崩溃
n_steps: 768             # 曾用 512/2048，现 768
n_epochs: 3              # 原 5
ent_coef: 0.005
clip_range: 0.1
```

**教训**：P2 v2 在 iter8 `enemy_damage≈4.4` 后 iter9 崩溃 → 采样偏小 + 更新过激进，非纯 reward 问题。

---

## 6. Phase-2 v2（y=0）

### 6.1 启动（从零）

```powershell
python main.py --env-config ./config/envs.stage2.phase2.v2.yaml --config ./config/algs.stage2.phase2.v2.yaml
tensorboard --logdir ./logs/stage2_phase2_v2/
```

- 加载：`model/stage2_phase1/ppo_combat_p1_30000_steps.zip`
- `reset_num_timesteps: true`
- ckpt：`model/stage2_phase2_v2/ppo_combat_p2_v2_*`

### 6.2 现象

- P1 锁 yaw 遗留 **action[3] 负偏置** → 解锁后**持续左拐**
- `max_yaw: 0.15` + `yaw_misalign_weight` 有改善但 y=0 仍难稳定击杀
- **5000 步 ckpt** 在 y=0 eval 上相对后续步数更稳（用户选定作 P3 基线）

### 6.3 y=0 eval

```powershell
python scripts/eval_stage2_phase2_v2.py --episodes 3 --log-interval 0
```

指标：**final_yaw_deg**、**enemy_hp_final**。

---

## 7. Phase-3（y 随机）

### 7.1 配置要点

```yaml
# envs.stage2.phase3.yaml
enemy_y_range: [-5, 5]
enemy_y_positive_prob: 0.65   # 65% +y∈[1,5]，35% -y∈[-5,-1]（不含0）
```

实现：`utils/initialize.py` → `combat_enemy_y_positive_prob`；`train_env.py` 透传。

### 7.2 启动（续训 5000 基线）

```powershell
python main.py --env-config ./config/envs.stage2.phase3.yaml --config ./config/algs.stage2.phase3.yaml
tensorboard --logdir ./logs/stage2_phase3/
```

```yaml
# algs.stage2.phase3.yaml
load_path: ./model/stage2_phase2_v2/ppo_combat_p2_v2_5000_steps.zip
reset_num_timesteps: false
total_timesteps: 80000
```

- ckpt：`model/stage2_phase3/ppo_combat_p3_*`（当前至 **30000**）
- UE 轮数建议 ≥ **200**

### 7.3 左拐 → -y 易、+y 难（核心发现）

**基线 p2_v2@5000** 在 **20204:1000** 固定 y 扫点 eval（11 值各 1 局）：

| y | -5 | -4 | -3 | -2 | -1 | 0 | +1 | +2 | +3 | +4 | +5 |
|---|----|----|----|----|----|---|----|----|----|----|-----|
| enemy_hp | **0** | **0** | **0** | **0** | **0** | **0** | 0.03 | 0.33 | 0.52 | 0.69 | 0.86 |
| 击杀 | ✓×6 | | | | | | | | | | |

- **-y / y≤0：6/6 击杀**
- **+y：0 击杀**，hp 随 |y| 增大而升高
- final_yaw 全程约 **-6.4° ~ -7.4°**（左拐）

CSV：`logs/stage2_phase3/eval/p2_v2_5000_baseline_by_y.csv`

→ P3 加 `enemy_y_positive_prob: 0.65` 就是为补 +y 泛化。

---

## 8. Eval：按 y 扫点（P3 / 任意 ckpt）

```powershell
# 全部 P3 ckpt × y∈{-5..+5}
python scripts/eval_stage2_phase3_by_y.py --log-interval 0

# 基线 5000 复测
python scripts/eval_stage2_phase3_by_y.py `
  --checkpoint ./model/stage2_phase2_v2/ppo_combat_p2_v2_5000_steps.zip `
  --output ./logs/stage2_phase3/eval/p2_v2_5000_baseline_by_y.csv `
  --log-interval 0

# 只测 +y 侧
python scripts/eval_stage2_phase3_by_y.py --y-values "1,2,3,4,5"
```

汇总：`logs/stage2_phase3/eval/phase3_by_y_summary.csv`  
脚本：`scripts/eval_stage2_phase3_by_y.py`  
配置：`config/envs.stage2.phase3.eval.yaml`

---

## 9. 代码改动清单（相对 STAGE2_会话总结）

| 文件 | 改动 |
|------|------|
| `utils/action.py` | `max_yaw` |
| `utils/reward.py` | `yaw_misalign_penalty`；reward 权重 yaml 可配 |
| `utils/initialize.py` | `combat_enemy_y_positive_prob`（+y 偏置采样） |
| `envs/train_env.py` | 透传 reward / y 偏置参数 |
| `config/envs.stage2.phase2.v2.yaml` | P2 v2 训练 |
| `config/algs.stage2.phase2.v2.yaml` | 稳定 PPO |
| `config/envs.stage2.phase3.yaml` | P3 y 随机 + +y 偏置 |
| `config/algs.stage2.phase3.yaml` | 续训 p2_v2@5000 |
| `config/envs.stage2.phase3.eval.yaml` | P3 eval 房间 |
| `scripts/eval_policy.py` | final_yaw + enemy_hp 评判 |
| `scripts/eval_stage2_phase2_v2.py` | P2 v2 批量 eval |
| `scripts/eval_stage2_phase3_by_y.py` | **P3 按 y 扫点 eval** |

---

## 10. Checkpoint 索引

| 路径 | 说明 |
|------|------|
| `model/stage2_phase1/ppo_combat_p1_30000_steps.zip` | P1 推荐 |
| `model/stage2_phase2_v2/ppo_combat_p2_v2_5000_steps.zip` | **P3 基线 / +y 问题参照** |
| `model/stage2_phase2_v2/ppo_combat_p2_v2_*` | P2 v2 全序列 5k~40k |
| `model/stage2_phase3/ppo_combat_p3_*` | P3 当前 10k~30k |

---

## 11. 待办 / 建议（下一会话）

1. **对 P3 全 ckpt 跑 `eval_stage2_phase3_by_y.py`**，对比 5000 基线是否 +y 改善。
2. 若 +y 仍弱：提高 `enemy_y_positive_prob`（0.7~0.75）或短期 **固定 +y 课程**（仅 [1,5] 训几 k 步）。
3. 若 P3 ckpt 整体优于 5000：换 **P3 最佳 ckpt** 作报告/eval 主力。
4. 左拐根因：可考虑 P2 **分步解锁 yaw**、或 eval 时微调 `max_yaw` / 略降 `yaw_misalign_weight`（曾用 6）。
5. eval 与 train **换 room**；比赛轮数 ≥ 总步数/512。

---

## 12. 常用命令速查

```powershell
cd Reinforcement-learning-drone
..\venv\Scripts\Activate.ps1

# P3 训练
python main.py --env-config ./config/envs.stage2.phase3.yaml --config ./config/algs.stage2.phase3.yaml

# P3 y 扫点 eval
python scripts/eval_stage2_phase3_by_y.py --log-interval 0

# 单 ckpt y=0 eval（P2 v2）
python scripts/eval_policy.py --model ./model/stage2_phase2_v2/ppo_combat_p2_v2_5000_steps.zip --env-config ./config/envs.stage2.phase2.v2.eval.yaml --episodes 3

# 单测
python -m unittest tests.test_reward_initialize_truncate tests.test_action_observation -v
```

---

## 13. 相关文档

| 文件 | 内容 |
|------|------|
| `STAGE2_会话总结.md` | P1/P2 早期交接、左拐分析 |
| `TRAINING_总结.md` | 全局训练与踩坑 |
| **本文** | P2 v2 + P3 + y 扫点 eval + 5000 基线结论 |

---

## 14. 一句话状态

**P2 v2 从 P1@30000 用 max_yaw + yaw 惩罚 + 稳定 PPO 在 y=0 上练到 5000 步作基线；P3 从该 ckpt 解锁 y 随机并加 +y 采样偏置；基线 eval 证实左拐导致 -y 全击杀、+y 几乎无伤害，P3 训练进行中（p3@30k），下步应用 y 扫点 eval 验证 +y 是否改善。**
