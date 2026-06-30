# dominic/simple-reward-final-tuning 分支说明

本文档说明 `dominic/simple-reward-final-tuning` 相对 `main` 分支的具体修改内容，方便组内同伴 review、继续训练和撰写报告。

## 一、分支目标

本分支主要针对 Simple 场景下“战机能接近并部分命中靶机，但最后阶段容易横向偏出、飞过目标，无法稳定击落”的问题，做了以下几类调整：

- 强化攻击几何相关奖励：对准、中心线、攻击盒、近距离补刀。
- 修正 HP 尺度兼容：同时支持 `0-1` 归一化血量和 `0-1000` 血量。
- 扩展 observation：让 agent 能直接看到前向距离、横向误差、垂直误差和攻击距离窗口。
- 修复 reset 时可能拿到上一局残留观测的问题。
- 增强 eval 输出，方便判断失败原因。
- 增加本地训练/冒烟配置和单元测试。

## 二、初始化参数修改

文件：`utils/initialize.py`

### 修改内容

敌机初始位置从：

```python
enemy_initial_state[0:3] = np.array([120, 0, 30], dtype=np.int32)
```

调整为：

```python
enemy_initial_state[0:3] = np.array([80, 0, 30], dtype=np.int32)
```

### 含义

- 位置单位按 `1 unit = 10 m` 处理。
- 敌机初始距离由约 `1200 m` 调整为约 `800 m`。
- 这是 Simple 阶段的课程化训练设置，用于先让 agent 在较短距离内学会命中和补刀。
- 最终提交课程作业时，如果题目要求初始距离不少于 `1000 m`，需要再恢复为 `[100, 0, 30]`、`[120, 0, 30]` 或更远距离重新训练/展示。

当前我方初始状态保持不变：

```python
my_initial_state[0:3] = [0, 0, 30]
my_initial_state[6:9] = [30, 0, 0]
```

即高度约 `300 m`，初速度约 `300 m/s`。

## 三、Observation 接口修改

文件：`utils/observation.py`、`envs/train_env.py`

### 观测维度

`observation_space` 从 `16` 维改为 `20` 维：

```python
spaces.Box(shape=[20], ...)
```

因此如果后续同伴修改 observation，需要同步检查：

- `utils/observation.py`
- `envs/train_env.py` 中的 `observation_space`
- 训练出来的旧模型是否还能兼容新维度

旧的 0-15 号特征基本保留，新加 16-19 号攻击几何特征。

### HP 归一化兼容

新增 `_hp_fraction()`：

```python
if hp <= 1.5:
    return clip(hp, 0, 1)
else:
    return clip(hp / 1000, 0, 1)
```

作用：

- Simple 实测可能返回 `0-1000` HP，也兼容早期 `0-1` HP。
- 避免 `enemy_hp`、`my_hp`、`hp_diff` 因单位不同导致 observation 尺度错误。

### 新增 observation 特征

新增常量：

```python
ATTACK_MIN_RANGE_M = 60.0
ATTACK_MAX_RANGE_M = 660.0
ATTACK_CENTERLINE_SCALE_M = 200.0
VERTICAL_ERROR_SCALE_M = 200.0
```

新增输出：

| 索引 | 含义 | 归一化方式 |
|---|---|---|
| `16` | 目标在我方机头方向上的前向距离 | `2 * forward_distance / 660 - 1`，裁剪到 `[-1,1]` |
| `17` | 横向中心线得分 | `1 - lateral_error / 200`，裁剪到 `[-1,1]` |
| `18` | 垂直高度误差 | `vertical_error / 200`，裁剪到 `[-1,1]` |
| `19` | 攻击距离窗口得分 | 在 `60-660m` 内为高分，离窗口越远越低 |

设计意图：

- 原始相对位置和视线角不够直接，agent 不容易知道“我是否在攻击盒内”。
- 新特征直接告诉 agent 是否在目标前方、是否横向贴近中心线、是否处在有效攻击距离段。

## 四、Reward 参数和逻辑修改

文件：`utils/reward.py`

### 新增/调整参数

```python
CENTERLINE_SCALE_M = 120.0
TURN_RATE_PENALTY_WEIGHT = 0.04
OVERSHOOT_PENALTY = 8.0
FINISH_RANGE_M = 250.0
FINISH_CENTERLINE_SCALE_M = 25.0
FINISH_SPEED_TARGET_MPS = 180.0
FINISH_SPEED_PENALTY_WEIGHT = 0.01
MAX_FINISH_SPEED_PENALTY = 4.0
```

已有关键参数：

```python
ATTACK_MIN_RANGE_M = 60.0
ATTACK_MAX_RANGE_M = 660.0
ATTACK_HALF_WIDTH_M = 10.0
DAMAGE_REWARD_PER_HIT = 8.0
SELF_DAMAGE_PENALTY_PER_HIT = 14.0
ENEMY_HP_SHAPING_WEIGHT = 4.0
```

### 新增 reward 分项

本分支新增以下 `comps` 字段，训练日志中会出现对应曲线：

| 分项 | 作用 |
|---|---|
| `centerline` | 鼓励目标在机头前方且横向误差小 |
| `turn_penalty` | 抑制过大的角速度，避免乱转刷局部奖励 |
| `overshoot` | 敌机还活着但已经飞过目标时给惩罚 |
| `finish_centerline` | 最后 `250m` 内更强地鼓励贴近中心线补刀 |
| `finish_speed_penalty` | 最后 `250m` 内速度超过 `180m/s` 时惩罚，避免一头冲过目标 |

### HP shaping 修改

旧逻辑倾向于按“敌机血越低，每步奖励越高”给分，容易产生低血量附近的静态刷分倾向。

本分支改为势能差分：

```python
enemy_hp_shaping = ENEMY_HP_SHAPING_WEIGHT * max(
    0.0,
    hp_fraction(prev_enemy) - hp_fraction(enemy)
)
```

含义：

- 只有敌机这一帧真的掉血，才给 `enemy_hp_shaping`。
- 敌机低血但没有继续受伤，不再持续白给奖励。
- 更符合“持续输出直到击落”的目标。

### 命中奖励限制

`enemy_damage` 仍然只在 `attack_box` 内给分：

```python
enemy_damage = DAMAGE_REWARD_PER_HIT * hit_count if in_attack_box else 0.0
```

目的：

- 避免远距离或几何不对时偶发掉血造成误导。
- 让奖励更集中到“机头对准 + 进入攻击盒 + 持续命中”。

### 近距离补刀逻辑

当敌机仍存活，且目标在机头前方 `0-250m` 内：

```python
finish_centerline = 4.0 * alignment_pos * exp(-lateral_error / 25)
finish_speed_penalty = -min(4, 0.01 * max(0, speed_mps - 180))
```

目的：

- 解决训练中经常出现的“已打到剩 80-120 HP，但最后横向偏出、飞过目标”问题。
- 鼓励最后阶段保持中心线，不要高速掠过。

### 飞过目标惩罚

```python
overshoot = -8.0 if enemy_alive and forward_distance < 0 else 0.0
```

含义：

- 如果敌机还活着，但已经从我方机头前方变到身后，说明攻击没有完成。
- 该项会持续负奖励，迫使策略避免“一次冲过不回头”。

## 五、TrainEnv reset 逻辑修改

文件：`envs/train_env.py`

### 新增接口

新增函数：

```python
is_expected_initial_observation(my_state, enemy_state, initial_body, position_tolerance_unit=5.0)
```

判断 reset 后拿到的观测是否真的是本局初始状态：

- 我方位置接近初始化位置。
- 敌方位置接近初始化位置。
- 双方 HP 接近满血，当前阈值为 `>=900`。

### reset 流程调整

旧 reset 在发送 initial packet 后直接拿观测，实际运行时可能拿到上一局残留状态。

新 reset 流程：

1. 生成 `initial_body`。
2. 发送 initial packet。
3. 尝试获取观测。
4. 如果超时，发送一次 noop action 再获取。
5. 用 `is_expected_initial_observation()` 检查是否为新局初始观测。
6. 如果发现是旧局残留观测，发送 `finish_round`，丢弃旧观测并重试。
7. 最多尝试 6 次。

目的：

- 避免训练 episode 开始时状态已经在上一局末尾，导致 reward、truncate、训练曲线全部失真。

## 六、评估脚本修改

文件：`scripts/eval_policy.py`

新增 `attack_geometry()`，评估时打印更多几何信息：

```text
dist, fwd, lat, yaw, act, attack_box, centerline, enemy_damage
```

示例输出字段含义：

| 字段 | 含义 |
|---|---|
| `dist` | 我方到敌方距离 |
| `fwd` | 敌机在我方机头方向上的前向距离 |
| `lat` | 敌机偏离机头中心线的横向误差 |
| `yaw` | 我方偏航角 |
| `act` | 实际发给服务器的动作，含 throttle/pitch/roll/yaw |
| `attack_box` | 当前攻击盒奖励分项 |
| `centerline` | 当前中心线奖励分项 |
| `enemy_damage` | 当前命中奖励分项 |

推荐评估命令：

```powershell
python scripts\eval_policy.py --env-config config\envs.local.yaml --model model\ppo_simple_15000_steps.zip --episodes 1 --log-interval 20
```

评估时不要只看 `ep_rew_mean`，重点看：

- `enemy_hp` 是否降到 0。
- 接近目标时 `lat` 是否小于约 `10m`。
- 飞过目标时 `fwd` 是否变成负数。
- `enemy_damage` 是否持续出现。

## 七、新增本地配置

### `config/envs.local.yaml`

用于本地 TinyEngine：

```yaml
host: 127.0.0.1
port: 1000
unit_id: 0
room_id: 0
train_mode: local
sync_step: 1
socket_timeout: 30
logger: false
save_path: "./logs/single_training/"
```

适用场景：

- 本地打开 `TinyEngine.exe`。
- 点击 Start 后控制台出现 `Tcp server started on port 1000`。
- 用本地端口 `1000` 训练或评估。

### `config/algs.local_smoke.yaml`

用于快速冒烟训练：

```yaml
total_timesteps: 2048
checkpoint.save_freq: 1024
ppo.n_steps: 512
ppo.batch_size: 256
ppo.ent_coef: 0.003
```

作用：

- 不用于正式训练。
- 只用于确认通信、reset、reward logging、checkpoint 保存链路正常。

## 八、新增和更新测试

### 更新 `tests/test_action_observation.py`

覆盖：

- observation 维度从 `16` 改为 `20`。
- 新增攻击几何特征。
- HP 特征同时支持 `1000` 血和 `1.0` 血。

### 更新 `tests/test_reward_initialize_truncate.py`

覆盖：

- 新增 reward 分项是否存在。
- `enemy_hp_shaping` 只奖励真实掉血，不奖励静态低血。
- `centerline` 对横向误差敏感。
- `turn_penalty` 对角速度敏感。
- `overshoot` 对飞过活目标给负奖励。
- `finish_centerline` 鼓励近距离中心线补刀。
- `finish_speed_penalty` 在近距离高速时生效，远距离时不生效。
- 初始化距离按当前 800m 训练配置检查。

### 新增 `tests/test_train_env_reset.py`

覆盖：

- reset 初始观测匹配时接受。
- 上一局残留观测或低血状态会被拒绝。

## 九、当前训练现象和后续建议

这版代码用于 Simple 800m 课程化训练时，模型已经能打掉大部分敌机 HP。最近一轮 eval 中：

- `ppo_simple_15000_steps.zip` 最好，敌机剩约 `80 HP`。
- `ppo_simple_20000_steps.zip` 和 `ppo_simple_final.zip` 反而略退化，敌机剩约 `120 HP`。
- 失败主要发生在最后阶段：`lat` 约 `35-45m`，大于攻击盒横向约 `±10m`，随后 `fwd` 变负，说明飞过目标。

下一步优先建议：

1. 先评估 `5000`、`10000`、`15000`、`20000` checkpoint，不要默认用 final。
2. 如果都不能击落 800m 靶机，建议把我方初始速度从 `[30,0,0]` 降到 `[20,0,0]`，延长攻击窗口。
3. 800m 打通后，再把敌机初始位置恢复到 `[100,0,30]` 或 `[120,0,30]`，满足课程最终初始距离不少于 `1000m` 的要求。

## 十、验证命令

本分支提交前使用过以下检查：

```powershell
python -m unittest discover tests
python -m compileall envs utils scripts tests
```

如果同伴拉取分支后要复现，请先确认 TinyEngine 已经进入 Simple 场景并 Start 成功，然后运行：

```powershell
python main.py --env-config config\envs.local.yaml --config config\algs.yaml
```

评估 checkpoint：

```powershell
python scripts\eval_policy.py --env-config config\envs.local.yaml --model model\ppo_simple_15000_steps.zip --episodes 1 --log-interval 20
```
