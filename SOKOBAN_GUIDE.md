# Sokoban 训练指南

本指南帮助你使用 RAGEN 框架训练 Sokoban（推箱子）强化学习模型。

## 🎮 Sokoban 任务说明

**Sokoban（推箱子）** 是一个经典的规划问题：
- **环境**: 6x6 网格世界，1个箱子，1个目标位置
- **目标**: 将箱子推到目标位置
- **动作**: `up`, `down`, `left`, `right`
- **奖励**: 成功推到目标获得正奖励，无效动作/超时获得负奖励
- **最大步数**: 100 步

## 📋 前置要求

### 1. 环境安装
```bash
bash scripts/setup_ragen.sh
```

### 2. 准备模型
下载或指定 Qwen2.5-7B-Instruct 模型：

```bash
# 方法 1: 使用 HuggingFace CLI 下载
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./models/Qwen2.5-7B-Instruct

# 方法 2: 使用已有模型
# 修改 test_sokoban.sh 中的 BASE_MODEL 路径
```

### 3. 检查数据
确认数据文件存在：
```bash
ls -lh data/sokoban/
# 应该看到: train.parquet, test.parquet
```

## 🚀 快速开始

### 方法 1: 使用交互式测试脚本（推荐）

```bash
bash test_sokoban.sh
```

**包含 3 种模式：**
1. **快速测试** (5 steps, ~5分钟) - 验证环境配置
2. **小规模训练** (50 steps, ~30分钟) - 调试超参数
3. **完整训练** (200 steps, 数小时) - 复现论文结果

### 方法 2: 使用 train.sh 命令行

**快速测试：**
```bash
bash train.sh sokoban \
    model.base_model=./models/Qwen2.5-7B-Instruct \
    model.experiment_name=sokoban_quick_test \
    training.train_batch_size=4 \
    training.ppo_batch_size=4 \
    training.n_rollout=2 \
    training.total_training_steps=5
```

**标准训练：**
```bash
bash train.sh sokoban \
    model.base_model=./models/Qwen2.5-7B-Instruct \
    model.experiment_name=sokoban_grpo \
    training.train_batch_size=16 \
    training.ppo_batch_size=64 \
    training.n_rollout=16 \
    training.max_turns=5 \
    training.temperature=0.7 \
    optimization.adv_estimator=grpo \
    optimization.actor_lr=1e-6
```

## 🔧 关键参数说明

### 训练规模参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `training.train_batch_size` | 每次更新的 rollout 数量 | 快速测试: 4, 标准: 16 |
| `training.ppo_batch_size` | PPO 更新的 batch size | 快速测试: 4, 标准: 64 |
| `training.n_rollout` | 每个 prompt 生成的响应数 (GRPO) | 快速测试: 2, 标准: 16 |
| `training.micro_batch_size` | 梯度累积的 micro batch | 1-4 (根据 GPU 内存) |
| `training.total_training_steps` | 总训练步数 | 快速测试: 5, 标准: 200 |

### 环境参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `env.env_kwargs.dim_x` | 网格宽度 | 6 |
| `env.env_kwargs.dim_y` | 网格高度 | 6 |
| `env.env_kwargs.num_boxes` | 箱子数量 | 1 |
| `env.env_kwargs.max_steps` | 最大步数 | 100 |
| `training.max_turns` | 对话轮数 | 5 |

### 优化参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `optimization.adv_estimator` | 优势估计算法 | `grpo` (推荐), `brpo`, `gae` |
| `optimization.actor_lr` | Actor 学习率 | 1e-6 |
| `optimization.kl_coef` | KL 散度系数 | 0.01 |
| `training.temperature` | 采样温度 | 0.7 |

## 📊 监控训练

### 1. WandB 集成

```bash
# 设置 API Key
export WANDB_API_KEY=your_api_key
wandb login $WANDB_API_KEY

# 然后运行训练
bash test_sokoban.sh
```

### 2. 本地日志

训练日志保存在：
```
outputs/exp_configs/logs/<date>/<time>/
```

### 3. Checkpoint

模型 checkpoint 保存在：
```
checkpoints/RAGEN-Sokoban/<experiment_name>/
```

## 🎯 预期结果

**成功训练的指标：**
- ✅ 奖励（reward）逐步上升
- ✅ 成功率（success rate）提高
- ✅ 平均步数减少（说明学会了更优路径）
- ✅ KL 散度保持稳定（不会偏离太远）

**训练曲线示例：**
```
Step 0:   avg_reward = -5.2, success_rate = 0.05
Step 50:  avg_reward = 2.1,  success_rate = 0.42
Step 100: avg_reward = 5.8,  success_rate = 0.68
Step 200: avg_reward = 8.3,  success_rate = 0.85
```

## 🐛 常见问题

### 1. GPU 内存不足 (CUDA OOM)

**解决方案：**
```bash
# 减小 micro_batch_size
training.micro_batch_size=1

# 减小 batch size
training.train_batch_size=4
training.ppo_batch_size=4

# 减小 rollout 数量
training.n_rollout=2
```

### 2. 模型路径错误

**错误信息：**
```
FileNotFoundError: [Errno 2] No such file or directory: '/opt/tiger/Qwen2.5-7B-Instruct'
```

**解决方案：**
```bash
# 修改 test_sokoban.sh 中的 BASE_MODEL 变量
BASE_MODEL="./models/Qwen2.5-7B-Instruct"  # 改为你的实际路径
```

### 3. 数据文件缺失

**错误信息：**
```
FileNotFoundError: data/sokoban/train.parquet not found
```

**解决方案：**
```bash
# 检查数据文件
ls data/sokoban/

# 如果缺失，需要生成数据（参考 ragen/env/sokoban/create_dataset.py）
python ragen/env/sokoban/create_dataset.py
```

### 4. Ray 集群启动失败

**解决方案：**
```bash
# 清理旧的 Ray 进程
ray stop

# 然后重新运行训练
bash test_sokoban.sh
```

## 📖 进阶实验

### 1. 超参数搜索

参考 `ragen_cmd.md` 中的超参数搜索脚本：

```bash
# 搜索 PPO batch size
bash scripts/hyperparam_search.sh \
    --env_name=sokoban \
    --exp_base_name="hyperparam_searching" \
    --search_group 1 \
    --n_gpus 4 \
    --micro_batch_size 2
```

### 2. 对比不同 RL 算法

**GRPO (默认):**
```bash
bash train.sh sokoban optimization.adv_estimator=grpo
```

**BRPO:**
```bash
bash train.sh sokoban optimization.adv_estimator=brpo
```

**PPO (GAE):**
```bash
bash train.sh sokoban optimization.adv_estimator=gae
```

### 3. 测试思考能力的影响

**有思考过程 (默认):**
```bash
bash train.sh sokoban training.no_think_rl=false
```

**无思考过程:**
```bash
bash train.sh sokoban training.no_think_rl=true
```

## 📚 相关资源

- **RAGEN 论文**: https://ragen-tutorial.readthedocs.io/
- **veRL 文档**: https://verl.readthedocs.io/
- **Sokoban 环境实现**: `ragen/env/sokoban/env.py`
- **超参数搜索指南**: `ragen_cmd.md`

## 🎓 学习路径

1. **第一步**: 运行快速测试（模式 1），验证环境
   ```bash
   bash test_sokoban.sh  # 选择模式 1
   ```

2. **第二步**: 理解训练输出和指标
   - 查看 WandB 或本地日志
   - 理解 reward, success_rate, kl_divergence

3. **第三步**: 小规模训练（模式 2），调试超参数
   ```bash
   bash test_sokoban.sh  # 选择模式 2
   ```

4. **第四步**: 完整训练（模式 3），复现论文
   ```bash
   bash test_sokoban.sh  # 选择模式 3
   ```

5. **第五步**: 尝试高级实验
   - 超参数搜索
   - 不同 RL 算法对比
   - 消融实验（thinking vs no-thinking）

---

**祝训练顺利！** 🚀

如有问题，请查看：
- `CLAUDE.md` - 项目总体说明
- `ragen_cmd.md` - 实验命令集合
- GitHub Issues: https://github.com/JarvisUSTC/DoctorAgent-RL/issues