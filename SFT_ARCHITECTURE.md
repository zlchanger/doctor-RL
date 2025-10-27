# SFT 架构说明 - DoctorAgent-RL 项目

## 🏗️ 两种 SFT 系统概览

这个项目中存在两种 SFT 系统，但它们**底层使用的是同一个训练器**：

### 1️⃣ **直接调用 RAGEN SFT Trainer**（DoctorAgent-RL 医疗对话使用）

```bash
sft/finetune_lora_med.sh
  └─> torchrun -m ragen.trainer.fsdp_sft_trainer
        └─> FSDP + LoRA 训练
```

**特点：**
- ✅ 直接使用 RAGEN 框架的 SFT trainer
- ✅ 数据已经预处理好（`MTMedDialog_sft_train.parquet`）
- ✅ 适合医疗对话等复杂任务
- ✅ 手动控制所有参数

### 2️⃣ **SFT Pipeline 自动化流程**（Sokoban/FrozenLake 使用）

```bash
python -m sft.sft_pipeline
  └─> 1. generate_data()        # 生成 SFT 数据
  └─> 2. finetune_model()       # 调用 ragen.trainer.fsdp_sft_trainer
  └─> 3. merge_model()          # 合并 LoRA 权重
  └─> 4. validate_model()       # 使用 RL 脚本验证
```

**特点：**
- ✅ 全自动流程（数据生成 → 训练 → 合并 → 验证）
- ✅ 适合规则环境（如 Sokoban，可以自动生成最优解）
- ✅ 底层同样使用 `ragen.trainer.fsdp_sft_trainer`

---

## 📊 架构对比图

```
┌─────────────────────────────────────────────────────────────┐
│                      SFT 系统架构                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  医疗对话 (DoctorAgent-RL)          游戏环境 (Sokoban/FL)   │
│         │                                    │               │
│         ▼                                    ▼               │
│  sft/finetune_lora_med.sh         sft/sft_pipeline.py       │
│         │                                    │               │
│         │                           ┌────────┴────────┐     │
│         │                           │                 │     │
│         │                      生成数据          合并+验证   │
│         │                           │                 │     │
│         └───────────┬───────────────┘                 │     │
│                     ▼                                 ▼     │
│         ┌──────────────────────────────────────────────┐   │
│         │   ragen.trainer.fsdp_sft_trainer (核心)      │   │
│         │   - FSDP 分布式训练                          │   │
│         │   - LoRA 参数高效微调                        │   │
│         │   - Hydra 配置管理                           │   │
│         └──────────────────────────────────────────────┘   │
│                     ▼                                       │
│         ┌──────────────────────────────────────────────┐   │
│         │   veRL SFT 底层基础设施                       │   │
│         │   - FSDP wrapper                              │   │
│         │   - Mixed precision training                  │   │
│         │   - Checkpoint saving                         │   │
│         └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔍 详细分析

### **DoctorAgent-RL 的 SFT 使用方式**

#### 脚本位置
```bash
sft/finetune_lora_med.sh
```

#### 调用方式
```bash
torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m ragen.trainer.fsdp_sft_trainer \
    data.train_files=data/MTMedDialog_sft_train.parquet \
    data.val_files=data/MTMedDialog_sft_val.parquet \
    data.prompt_key=prompt \
    data.response_key=response \
    model.partial_pretrain=Qwen2.5-7B-Instruct \
    model.lora_rank=64 \
    model.lora_alpha=32 \
    ...
```

#### 数据格式
```python
# MTMedDialog_sft_train.parquet
{
    "prompt": "患者初始描述",
    "response": "<think>医生的思考过程</think><answer>医生的问题或诊断</answer>"
}
```

#### 训练后处理
```bash
# 合并 LoRA 权重
python sft/utils/merge_lora.py \
    --base_model_name Qwen2.5-7B-Instruct \
    --lora_model_path $save_path \
    --output_path DoctorLLM-7B-SFT-1000-thinking
```

---

### **sft_pipeline.py 的自动化流程**

#### 脚本位置
```bash
sft/sft_pipeline.py
```

#### 使用方式
```bash
python -m sft.sft_pipeline \
    --config config/base.yaml \
    --env_type sokoban
```

#### 流程步骤

**步骤 1: 生成 SFT 数据**
```python
def generate_data(self):
    # 使用 BFS/DFS 等算法生成最优解
    # 例如: Sokoban 用 BFS 找到推箱子的最优路径
    # 输出: data/sft/sokoban/train.parquet
```

**步骤 2: 微调模型（调用同样的 trainer）**
```python
def finetune_model(self):
    cmd = [
        "torchrun -m ragen.trainer.fsdp_sft_trainer",
        f"data.train_files=data/sft/{self.env_type}/train.parquet",
        ...
    ]
    # 底层还是用 ragen.trainer.fsdp_sft_trainer
```

**步骤 3: 合并 LoRA**
```python
def merge_model(self, lora_path):
    # 找到最佳 checkpoint（基于验证集 loss）
    # 调用 sft/utils/merge_lora.py
```

**步骤 4: 验证模型**
```python
def validate_model(self, merged_model_path):
    # 使用 RL 脚本在验证集上测试性能
    # 调用 ragen.trainer.main_ppo (val_only=True)
```

---

## 🎯 核心训练器：`ragen.trainer.fsdp_sft_trainer`

### 特点
- **FSDP 分布式训练**: 支持多 GPU 并行
- **LoRA 高效微调**: 参数量少，训练快
- **Hydra 配置**: 灵活的参数覆盖
- **veRL 集成**: 使用 veRL 的底层工具

### 关键参数
```python
# 数据相关
data.train_files         # 训练数据路径
data.prompt_key          # prompt 列名
data.response_key        # response 列名
data.max_length          # 最大序列长度

# 模型相关
model.partial_pretrain   # 基础模型路径
model.lora_rank          # LoRA rank (64)
model.lora_alpha         # LoRA alpha (32)
model.target_modules     # LoRA 应用的模块 (all-linear)

# 训练相关
data.train_batch_size    # 总 batch size
data.micro_batch_size    # 每个 GPU 的 micro batch
optim.lr                 # 学习率 (1e-4)
trainer.total_epochs     # 训练轮数
```

---

## 🤔 为什么有两种方式？

### **医疗对话为什么不用 pipeline？**

1. **数据已预处理**:
   - `MTMedDialog_sft_train.parquet` 已经由 DeepSeek-V3 生成好
   - 包含思考过程（`<think>...</think>`）
   - 无需自动生成

2. **复杂任务**:
   - 医疗对话无法用算法生成"正确答案"
   - 需要人工标注或大模型生成

3. **灵活性**:
   - 手动控制更精确
   - 便于调试和实验

### **Sokoban/FrozenLake 为什么用 pipeline？**

1. **自动生成数据**:
   - 可以用 BFS 算法找到最优解
   - 不需要人工标注

2. **端到端流程**:
   - 从数据生成到验证一步完成
   - 适合快速实验

3. **论文需要**:
   - 需要对比 SFT vs RL 的效果
   - Pipeline 简化了实验流程

---

## 📋 使用建议

### **如果你的任务是医疗对话类**

**推荐使用**: 直接调用 `ragen.trainer.fsdp_sft_trainer`

```bash
# 使用现有脚本
bash sft/finetune_lora_med.sh 8 ./sft_output

# 或自定义参数
torchrun --nproc_per_node=4 \
    -m ragen.trainer.fsdp_sft_trainer \
    data.train_files=your_data.parquet \
    model.partial_pretrain=your_model \
    ...
```

### **如果你的任务是规则环境（如游戏）**

**推荐使用**: `sft_pipeline.py`

```bash
# 通过 train.sh
bash train.sh sokoban rl_or_sft=sft

# 或直接调用 pipeline
python -m sft.sft_pipeline \
    --config config/base.yaml \
    --env_type sokoban
```

### **如果你想自定义流程**

**直接使用底层 trainer**:

```bash
torchrun -m ragen.trainer.fsdp_sft_trainer \
    data.train_files=my_custom_data.parquet \
    data.prompt_key=input \
    data.response_key=output \
    model.partial_pretrain=Qwen2.5-7B-Instruct \
    model.lora_rank=64 \
    trainer.experiment_name=my_exp
```

---

## 🔗 文件关系总结

```
DoctorAgent-RL 项目 SFT 文件结构
├── sft/
│   ├── finetune_lora_med.sh          # 医疗对话 SFT 脚本 ⭐
│   ├── sft_pipeline.py                # 自动化 SFT 流程
│   └── utils/
│       ├── merge_lora.py              # LoRA 权重合并工具
│       └── generate_sft_verl_*.py     # 各环境数据生成
│
├── ragen/
│   ├── trainer/
│   │   ├── fsdp_sft_trainer.py        # 核心 SFT 训练器 ⭐⭐⭐
│   │   └── main_ppo.py                # RL 训练器
│   └── utils/
│       ├── dataset.py                 # SFT 数据集类
│       └── fsdp_utils.py              # FSDP 工具
│
└── data/
    ├── MTMedDialog_sft_train.parquet  # 医疗对话 SFT 数据 ⭐
    └── sokoban/
        ├── train.parquet              # Sokoban 训练数据
        └── test.parquet               # Sokoban 测试数据
```

---

## 💡 关键结论

1. **底层统一**: 无论哪种方式，都使用 `ragen.trainer.fsdp_sft_trainer`
2. **医疗对话**: 使用 `sft/finetune_lora_med.sh` 直接调用 trainer
3. **游戏环境**: 使用 `sft/sft_pipeline.py` 自动化流程
4. **都是 RAGEN**: 两种方式都是 RAGEN 框架的一部分，不是外部工具

---

**结论**: DoctorAgent-RL 的 SFT 是通过 **RAGEN 框架的 `fsdp_sft_trainer`** 构建的，
`sft/finetune_lora_med.sh` 直接调用这个 trainer，而 `sft_pipeline.py` 是一个
更高层的封装（用于其他环境），但底层同样使用这个 trainer。

🎯 **简单来说**: 都是用的 RAGEN SFT，只是调用方式不同！