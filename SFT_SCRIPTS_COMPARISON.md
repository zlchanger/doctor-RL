# SFT 脚本对比：finetune_lora.sh vs finetune_lora_med.sh

## 📋 快速对比表

| 特性 | `finetune_lora.sh` | `finetune_lora_med.sh` |
|------|-------------------|------------------------|
| **用途** | 通用环境 (Sokoban, FrozenLake) | 医疗对话 (DoctorAgent-RL) |
| **输入参数** | `[env_type] <nproc> <save_path>` | `<nproc> <save_path>` |
| **默认环境** | sokoban (可配置) | medical_consultation (固定) |
| **训练数据** | `data/sft/${env_type}/train.parquet` | `data/MTMedDialog_sft_train.parquet` |
| **验证数据** | `data/sft/${env_type}/test.parquet` | `data/MTMedDialog_sft_val.parquet` |
| **基础模型** | Qwen2.5-0.5B | Qwen2.5-7B-Instruct |
| **max_length** | 2048 | 6784 |
| **micro_batch_size** | 4 | 8 |
| **total_epochs** | 5 | 3 |
| **gradient_checkpointing** | False | True |
| **with_thinking** | 未设置 | False (可配置) |
| **自动合并 LoRA** | ❌ 否 | ✅ 是 |
| **项目名称** | 未设置 | Medical-Dialogue |

---

## 🔍 逐行对比

### **1. 参数解析**

#### `finetune_lora.sh`
```bash
env_type=${1:-sokoban}  # 第一个参数是环境类型，默认 sokoban
shift 1
nproc_per_node=$1       # 第二个参数是 GPU 数量
save_path=$2            # 第三个参数是保存路径
```

**使用示例：**
```bash
bash sft/finetune_lora.sh sokoban 4 ./sft_output
bash sft/finetune_lora.sh frozenlake 8 ./sft_output
```

#### `finetune_lora_med.sh`
```bash
nproc_per_node=$1       # 第一个参数是 GPU 数量
save_path=$2            # 第二个参数是保存路径
# 环境固定为医疗对话，无需指定
```

**使用示例：**
```bash
bash sft/finetune_lora_med.sh 8 ./sft_output
```

---

### **2. 数据配置**

#### `finetune_lora.sh` (通用环境)
```bash
data.train_files=data/sft/${env_type}/train.parquet
data.val_files=data/sft/${env_type}/test.parquet
data.max_length=2048
```

**数据路径示例：**
- Sokoban: `data/sft/sokoban/train.parquet`
- FrozenLake: `data/sft/frozenlake/train.parquet`

#### `finetune_lora_med.sh` (医疗对话)
```bash
data.train_files=data/MTMedDialog_sft_train.parquet
data.val_files=data/MTMedDialog_sft_val.parquet
data.max_length=6784
```

**为什么 max_length 不同？**
- **通用环境 (2048)**: 游戏环境的对话较短，2048 足够
- **医疗对话 (6784)**: 医疗咨询包含长对话历史 + 思考过程，需要更长的上下文

---

### **3. 模型配置**

#### `finetune_lora.sh` (轻量模型)
```bash
model.partial_pretrain=Qwen/Qwen2.5-0.5B
model.enable_gradient_checkpointing=False
data.micro_batch_size=4
```

**特点：**
- ✅ 0.5B 小模型，训练快，显存占用少
- ✅ 不用 gradient checkpointing（速度优先）
- ✅ 适合快速实验和调试

#### `finetune_lora_med.sh` (生产级模型)
```bash
model.partial_pretrain=Qwen2.5-7B-Instruct
model.enable_gradient_checkpointing=True
data.micro_batch_size=8
```

**特点：**
- ✅ 7B 大模型，性能更强
- ✅ 启用 gradient checkpointing（节省显存）
- ✅ 更大的 micro_batch_size（更稳定的梯度）
- ✅ 适合生产环境部署

---

### **4. 训练超参数**

#### `finetune_lora.sh`
```bash
data.train_batch_size=128
data.micro_batch_size=4
trainer.total_epochs=5
optim.lr=1e-4
```

**梯度累积步数：**
```
gradient_accumulation_steps = train_batch_size / (micro_batch_size × n_gpus)
                            = 128 / (4 × 4)  # 假设 4 GPUs
                            = 8
```

#### `finetune_lora_med.sh`
```bash
data.train_batch_size=128
data.micro_batch_size=8
trainer.total_epochs=3
optim.lr=1e-4
```

**梯度累积步数：**
```
gradient_accumulation_steps = 128 / (8 × 8)  # 假设 8 GPUs
                            = 2
```

**为什么 epochs 不同？**
- **通用环境 (5 epochs)**: 数据量较小，需要更多轮次
- **医疗对话 (3 epochs)**: 数据集较大，3 轮足够，防止过拟合

---

### **5. 特殊功能**

#### `finetune_lora.sh`
```bash
# 训练完成后不自动合并 LoRA
# 需要手动运行：
python sft/utils/merge_lora.py \
    --base_model_name Qwen/Qwen2.5-0.5B \
    --lora_model_path ./sft_output \
    --output_path merged_model
```

#### `finetune_lora_med.sh`
```bash
# 训练完成后自动合并 LoRA
python sft/utils/merge_lora.py \
    --base_model_name Qwen2.5-7B-Instruct \
    --lora_model_path $save_path \
    --output_path DoctorLLM-7B-SFT-1000-thinking
```

**区别：**
- ✅ `finetune_lora_med.sh` 自动合并，直接得到可用模型
- ❌ `finetune_lora.sh` 需要手动合并（更灵活）

---

### **6. 医疗对话特有参数**

#### `finetune_lora_med.sh` 独有
```bash
+data.with_thinking=False
trainer.project_name=Medical-Dialogue
```

**`with_thinking` 参数说明：**
- `True`: 训练时保留 `<think>...</think>` 标签
- `False`: 训练时去除思考标签，只训练最终回答

**数据格式示例：**
```python
# with_thinking=True (保留思考过程)
response = "<think>患者可能有消化问题，需要询问饮食</think><answer>请问您最近的饮食习惯如何？</answer>"

# with_thinking=False (只保留回答)
response = "请问您最近的饮食习惯如何？"
```

---

## 🎯 使用场景选择

### **使用 `finetune_lora.sh`** 的情况

✅ **适合：**
- 训练 Sokoban / FrozenLake 等游戏环境
- 快速原型开发和实验
- 显存受限（使用小模型）
- 需要灵活配置环境类型

❌ **不适合：**
- 医疗对话等复杂任务
- 需要长上下文的任务
- 生产环境部署

**使用示例：**
```bash
# Sokoban SFT
bash sft/finetune_lora.sh sokoban 4 ./sft_sokoban

# FrozenLake SFT
bash sft/finetune_lora.sh frozenlake 4 ./sft_frozenlake
```

---

### **使用 `finetune_lora_med.sh`** 的情况

✅ **适合：**
- DoctorAgent-RL 项目的 SFT cold start
- 医疗对话等复杂任务
- 需要大模型性能
- 需要自动化流程（训练+合并）

❌ **不适合：**
- 快速实验（7B 模型训练较慢）
- 显存受限的机器
- 非医疗对话任务

**使用示例：**
```bash
# DoctorAgent-RL SFT
bash sft/finetune_lora_med.sh 8 ./sft_checkpoints

# 训练完成后自动生成：
# - ./sft_checkpoints/global_step_*/  (LoRA 权重)
# - DoctorLLM-7B-SFT-1000-thinking/   (合并后的模型)
```

---

## 📊 资源需求对比

### **GPU 显存需求（估算）**

| 配置 | `finetune_lora.sh` | `finetune_lora_med.sh` |
|------|-------------------|------------------------|
| **模型大小** | 0.5B | 7B |
| **单 GPU 显存** | ~8 GB | ~24 GB |
| **推荐 GPU** | RTX 3090 (24GB) | A100 (40GB/80GB) |
| **最少 GPU 数** | 1 | 4 |
| **推荐 GPU 数** | 4 | 8 |

### **训练时间（估算）**

| 配置 | `finetune_lora.sh` | `finetune_lora_med.sh` |
|------|-------------------|------------------------|
| **数据量** | ~1000 样本 | ~1000 样本 |
| **Epochs** | 5 | 3 |
| **单 epoch 时间** | ~5-10 分钟 | ~30-60 分钟 |
| **总训练时间** | ~30-50 分钟 | ~2-3 小时 |

---

## 🔧 修改建议

### **如果你想用小模型测试医疗对话**

修改 `finetune_lora_med.sh`:
```bash
# 改为小模型
model.partial_pretrain=Qwen/Qwen2.5-0.5B

# 减小 max_length (避免显存溢出)
data.max_length=2048

# 可以关闭 gradient checkpointing (加速)
model.enable_gradient_checkpointing=False
```

### **如果你想用大模型训练 Sokoban**

修改 `finetune_lora.sh`:
```bash
# 改为大模型
model.partial_pretrain=Qwen/Qwen2.5-7B-Instruct

# 启用 gradient checkpointing
model.enable_gradient_checkpointing=True

# 调整 batch size
data.micro_batch_size=8
```

---

## 💡 关键总结

### **核心区别**
1. **目标任务不同**:
   - `finetune_lora.sh`: 游戏环境（规则简单）
   - `finetune_lora_med.sh`: 医疗对话（复杂推理）

2. **模型规模不同**:
   - `finetune_lora.sh`: 0.5B 小模型（快速实验）
   - `finetune_lora_med.sh`: 7B 大模型（生产级）

3. **上下文长度不同**:
   - `finetune_lora.sh`: 2048 tokens（游戏状态简短）
   - `finetune_lora_med.sh`: 6784 tokens（医疗对话长）

4. **自动化程度不同**:
   - `finetune_lora.sh`: 只训练，手动合并
   - `finetune_lora_med.sh`: 训练 + 自动合并

### **底层一致**
- ✅ 都使用 `ragen.trainer.fsdp_sft_trainer`
- ✅ 都使用 LoRA (rank=64, alpha=32)
- ✅ 都使用 FSDP 分布式训练
- ✅ 都支持 Hydra 配置覆盖

---

## 📚 相关命令参考

### **完整训练流程对比**

#### 使用 `finetune_lora.sh`
```bash
# 1. 生成 SFT 数据（如果需要）
python -m ragen.env.sokoban.create_dataset

# 2. 训练
bash sft/finetune_lora.sh sokoban 4 ./sft_output

# 3. 手动合并 LoRA
python sft/utils/merge_lora.py \
    --base_model_name Qwen/Qwen2.5-0.5B \
    --lora_model_path ./sft_output/global_step_XXX \
    --output_path ./merged_model

# 4. 使用合并后的模型进行 RL 训练
bash train.sh sokoban \
    model.base_model=./merged_model \
    ...
```

#### 使用 `finetune_lora_med.sh`
```bash
# 1. 数据已预处理好 (data/MTMedDialog_sft_train.parquet)

# 2. 一键训练+合并
bash sft/finetune_lora_med.sh 8 ./sft_checkpoints

# 3. 直接使用合并后的模型
# DoctorLLM-7B-SFT-1000-thinking/ 已经可以用于 RL

# 4. RL 训练
bash scripts_exp/doctor-agent-rl-dynamic.sh
```

---

**结论**: `finetune_lora_med.sh` 是针对 DoctorAgent-RL 优化的专用脚本，
而 `finetune_lora.sh` 是通用脚本，可用于多种环境的快速实验。