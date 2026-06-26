# slime 框架 8卡4090 完整复现计划

基于实际硬件资源的可行性评估与分阶段执行方案

---

## 目录

1. [资源评估与限制分析](#1-资源评估与限制分析)
2. [复现目标与阶段规划](#2-复现目标与阶段规划)
3. [第一阶段：环境搭建与基础验证](#3-第一阶段环境搭建与基础验证)
4. [第二阶段：小模型全流程复现](#4-第二阶段小模型全流程复现)
5. [第三阶段：中等模型训练](#5-第三阶段中等模型训练)
6. [第四阶段：Agent训练复现](#6-第四阶段agent训练复现)
7. [第五阶段：性能优化与调试](#7-第五阶段性能优化与调试)
8. [关键问题与解决方案](#8-关键问题与解决方案)
9. [学习路线图](#9-学习路线图)
10. [附录：配置模板](#10-附录配置模板)

---

## 1. 资源评估与限制分析

### 1.1 硬件资源清单

| 资源 | 规格 | 说明 |
|------|------|------|
| GPU | 8x RTX 4090 | 每卡 24GB 显存 |
| 互联 | PCIe 4.0 | 无 NVLink，带宽约 32GB/s |
| 总显存 | 192GB | 8 * 24GB |
| 适用模型 | 0.5B - 7B | 受限于显存和互联 |

### 1.2 4090 的关键限制

**显存限制**：
- 24GB 显存限制了模型大小
- 训练 + 推理共置模式需要精细的显存管理
- 大 batch size 可能导致 OOM

**互联限制**：
- 无 NVLink，PCIe 带宽较低
- 张量并行（TP）效率较低
- 建议 TP size ≤ 2

**计算能力**：
- 不支持原生 FP8（Ada 架构才支持）
- BF16 训练是可行的
- 推理性能相对较低

### 1.3 可行的模型规模

| 模型 | 参数量 | 4090 可行性 | 配置建议 |
|------|--------|-------------|----------|
| Qwen2.5-0.5B | 500M | ✅ 完全可行 | TP=1, 共置模式 |
| Qwen2.5-1.5B | 1.5B | ✅ 可行 | TP=1, 共置模式 |
| Qwen2.5-3B | 3B | ✅ 可行 | TP=2, 共置模式 |
| Qwen3-4B | 4B | ✅ 可行 | TP=2, 共置模式 |
| Qwen2.5-7B | 7B | ⚠️ 需优化 | TP=2, 分离模式 |
| Qwen2.5-14B | 14B | ❌ 困难 | 需要更多显存 |

---

## 2. 复现目标与阶段规划

### 2.1 总体目标

**主要目标**：
1. 完整复现 slime 的训练流程
2. 理解 RL 后训练的核心机制
3. 掌握 Agent 训练的方法
4. 积累工程实践经验

**阶段目标**：
- 阶段 1：环境搭建，验证基础功能
- 阶段 2：小模型（0.5B）全流程训练
- 阶段 3：中等模型（3B-4B）训练
- 阶段 4：Agent 训练（多轮交互）
- 阶段 5：性能优化与调试

### 2.2 时间规划

| 阶段 | 时间 | 主要任务 | 产出 |
|------|------|----------|------|
| 阶段 1 | 1-2 天 | 环境搭建 | 可运行的 slime 环境 |
| 阶段 2 | 2-3 天 | 0.5B 训练 | 训练曲线、模型 checkpoint |
| 阶段 3 | 3-5 天 | 3B-4B 训练 | 训练曲线、性能指标 |
| 阶段 4 | 3-5 天 | Agent 训练 | 多轮交互模型 |
| 阶段 5 | 2-3 天 | 优化调试 | 性能报告、问题总结 |

**总计**：约 2-3 周

---

## 3. 第一阶段：环境搭建与基础验证

### 3.1 Docker 环境准备

**步骤 1：拉取 Docker 镜像**
```bash
docker pull slimerl/slime:latest
```

**步骤 2：启动容器**
```bash
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -it slimerl/slime:latest /bin/bash
```

**步骤 3：验证 GPU**
```bash
nvidia-smi
# 应该看到 8 张 4090
```

### 3.2 模型下载

**下载 Qwen2.5-0.5B 模型**：
```bash
# Hugging Face
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir /root/Qwen2.5-0.5B-Instruct

# 或使用 ModelScope（国内更快）
# modelscope download --model Qwen/Qwen2.5-0.5B-Instruct --local_dir /root/Qwen2.5-0.5B-Instruct
```

**下载训练数据**：
```bash
hf download --repo-type dataset zhuzilin/gsm8k --local-dir /root/gsm8k
```

### 3.3 模型权重转换

**步骤 1：加载模型配置**
```bash
cd /root/slime
source scripts/models/qwen2.5-0.5B.sh
```

**步骤 2：转换为 Megatron 格式**
```bash
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen2.5-0.5B-Instruct \
    --save /root/Qwen2.5-0.5B-Instruct_torch_dist
```

**步骤 3：验证转换结果**
```bash
ls /root/Qwen2.5-0.5B-Instruct_torch_dist/
# 应该看到 latest_checkpointed_iteration.txt 和迭代目录
```

### 3.4 基础功能验证

**验证 Ray 启动**：
```bash
ray start --head --node-ip-address 127.0.0.1 --num-gpus 8 --disable-usage-stats
ray status
```

**验证训练脚本**：
```bash
# 使用最小配置测试
python train.py --help
```

### 3.5 阶段 1 检查点

- [ ] Docker 环境正常运行
- [ ] 8 张 4090 可见且可用
- [ ] Qwen2.5-0.5B 模型下载完成
- [ ] 模型权重转换成功
- [ ] Ray 集群正常启动
- [ ] slime 命令行工具可用

---

## 4. 第二阶段：小模型全流程复现

### 4.1 配置文件准备

**创建训练脚本** `/root/slime/run-4090-0.5B.sh`：
```bash
#!/bin/bash

# 清理之前的进程
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3

set -ex

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/scripts/models/qwen2.5-0.5B.sh"

# 针对 4090 优化的配置
CKPT_ARGS=(
   --hf-checkpoint /root/Qwen2.5-0.5B-Instruct/
   --ref-load /root/Qwen2.5-0.5B-Instruct_torch_dist/
   --load /root/Qwen2.5-0.5B_slime/
   --save /root/Qwen2.5-0.5B_slime/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data /root/gsm8k/train.parquet
   --input-key messages
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout 100
   --rollout-batch-size 16          # 减小 batch size
   --n-samples-per-prompt 4         # 减少采样数
   --rollout-max-response-len 512   # 减少最大长度
   --rollout-temperature 1
   --global-batch-size 64
)

EVAL_ARGS=(
   --eval-interval 10
   --eval-prompt-data gsm8k /root/gsm8k/test.parquet
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 512
   --eval-top-k 1
)

# 针对 4090 的性能配置
PERF_ARGS=(
   --tensor-model-parallel-size 1   # TP=1 避免 PCIe 通信
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096        # 减小以适应 24GB 显存
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# 4090 特定配置
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1  # 单卡推理
   --sglang-mem-fraction-static 0.6 # 降低显存占用
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# 启动 Ray
ray start --head --node-ip-address 127.0.0.1 --num-gpus 8 --disable-usage-stats

# 提交训练任务
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1"
     }
   }' \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   --calculate-per-token-loss \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
```

### 4.2 4090 特定优化

**显存优化**：
```bash
# 降低 SGLang 显存占用
--sglang-mem-fraction-static 0.6

# 减小 batch size
--rollout-batch-size 16
--n-samples-per-prompt 4
--max-tokens-per-gpu 4096
```

**通信优化**：
```bash
# 避免 PCIe 通信瓶颈
--tensor-model-parallel-size 1
```

**训练优化**：
```bash
# 启用重计算节省显存
--recompute-granularity full
--recompute-method uniform
--recompute-num-layers 1
```

### 4.3 训练监控

**启动 wandb 监控**（可选）：
```bash
export WANDB_KEY=your_wandb_key

# 在脚本中添加
WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-4090
   --wandb-group qwen2.5-0.5B
   --wandb-key ${WANDB_KEY}
)
```

**查看训练日志**：
```bash
# 实时查看日志
tail -f /root/slime/runs/*/train.log
```

### 4.4 阶段 2 检查点

- [ ] 训练脚本创建完成
- [ ] 训练成功启动
- [ ] 奖励曲线开始上升
- [ ] 无 OOM 错误
- [ ] 评估指标正常
- [ ] checkpoint 保存成功

---

## 5. 第三阶段：中等模型训练

### 5.1 Qwen2.5-3B 训练

**下载模型**：
```bash
hf download Qwen/Qwen2.5-3B --local-dir /root/Qwen2.5-3B
```

**转换权重**：
```bash
cd /root/slime
source scripts/models/qwen2.5-3B.sh

PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen2.5-3B \
    --save /root/Qwen2.5-3B_torch_dist
```

**创建训练脚本** `/root/slime/run-4090-3B.sh`：
```bash
#!/bin/bash

# 清理进程
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3

set -ex

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/scripts/models/qwen2.5-3B.sh"

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen2.5-3B/
   --ref-load /root/Qwen2.5-3B_torch_dist/
   --load /root/Qwen2.5-3B_slime/
   --save /root/Qwen2.5-3B_slime/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data /root/gsm8k/train.parquet
   --input-key messages
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout 100
   --rollout-batch-size 8           # 进一步减小
   --n-samples-per-prompt 4
   --rollout-max-response-len 512
   --rollout-temperature 1
   --global-batch-size 32
)

EVAL_ARGS=(
   --eval-interval 10
   --eval-prompt-data gsm8k /root/gsm8k/test.parquet
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 512
   --eval-top-k 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 2   # 3B 模型需要 TP=2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2  # TP=2 推理
   --sglang-mem-fraction-static 0.6
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# 启动 Ray
ray start --head --node-ip-address 127.0.0.1 --num-gpus 8 --disable-usage-stats

# 使用共置模式，4卡训练+推理
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1"
     }
   }' \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
```

### 5.2 Qwen3-4B 训练（可选）

**下载模型**：
```bash
hf download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B
```

**转换权重**：
```bash
cd /root/slime
source scripts/models/qwen3-4B.sh

PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-4B \
    --save /root/Qwen3-4B_torch_dist
```

**配置要点**：
- 使用与 3B 类似的配置
- TP=2
- 减小 batch size

### 5.3 阶段 3 检查点

- [ ] 3B 模型下载和转换完成
- [ ] 训练脚本创建完成
- [ ] 训练成功启动
- [ ] 无 OOM 错误
- [ ] 奖励曲线正常
- [ ] 与 0.5B 模型对比效果

---

## 6. 第四阶段：Agent 训练复现

### 6.1 Search-R1 复现

**准备环境**：
```bash
cd /root/
git clone https://github.com/PeterGriffinJin/Search-R1.git
cd Search-R1/
pip install -e . --no-deps
pip install tensordict chardet
```

**准备数据**：
```bash
WORK_DIR=/root/Search-R1
LOCAL_DIR=$WORK_DIR/data/nq_hotpotqa_train

DATA=nq,hotpotqa
python $WORK_DIR/scripts/data_process/qa_search_train_merge.py \
    --local_dir $LOCAL_DIR \
    --data_sources $DATA
```

**创建训练脚本** `/root/slime/run-4090-search-r1.sh`：
```bash
#!/bin/bash

# 清理进程
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3

set -ex

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/scripts/models/qwen2.5-3B.sh"

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen2.5-3B/
   --ref-load /root/Qwen2.5-3B_torch_dist/
   --load /root/Qwen2.5-3B_search_r1/
   --save /root/Qwen2.5-3B_search_r1/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data /root/Search-R1/data/nq_hotpotqa_train/train.parquet
   --input-key prompt
   --label-key reward_model
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 100
   --rollout-batch-size 8
   --n-samples-per-prompt 4
   --rollout-max-response-len 512
   --rollout-temperature 1
   --global-batch-size 32
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.6
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# 自定义生成和奖励函数
CUSTOM_ARGS=(
   --custom-generate-function-path /root/slime/examples/search-r1/generate_with_search.generate
   --custom-rm-path /root/slime/examples/search-r1/generate_with_search.reward_func
)

# 启动 Ray
ray start --head --node-ip-address 127.0.0.1 --num-gpus 8 --disable-usage-stats

# 提交训练任务
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM:/root/slime/examples/search-r1",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1"
     }
   }' \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
```

### 6.2 本地搜索服务器配置

**启动本地检索服务器**（可选，需要单独环境）：
```bash
# 创建 conda 环境
conda create -n retriever python=3.10 -y
conda activate retriever

# 安装依赖
pip install transformers datasets pyserini huggingface_hub
conda install faiss-gpu=1.8.0 -c pytorch -c nvidia -y
pip install uvicorn fastapi

# 下载索引
python /root/slime/examples/search-r1/local_dense_retriever/download.py --save_path /root/Index

# 启动服务器
python /root/slime/examples/search-r1/local_dense_retriever/retrieval_server.py \
    --index_path /root/Index/e5_Flat.index \
    --corpus_path /root/Index/wiki-18.jsonl \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model intfloat/e5-base-v2 \
    --faiss_gpu
```

### 6.3 阶段 4 检查点

- [ ] Search-R1 环境准备完成
- [ ] 训练数据准备完成
- [ ] 训练脚本创建完成
- [ ] 训练成功启动
- [ ] 多轮交互正常工作
- [ ] 奖励函数正常返回

---

## 7. 第五阶段：性能优化与调试

### 7.1 性能分析工具

**使用 profiling 工具**：
```bash
# 启动 rollout 进程等待
python train.py \
    --rollout-function-path slime.rollout.sleep_rollout.sleep \
    ... (其他参数)

# 在另一个终端启动 profiling
python tools/profile_rollout.py --router-url http://127.0.0.1:3000 --action start --num-steps 3
```

**使用 trace viewer**：
```bash
# 保存 trace 数据
--save-trace-data /path/to/trace.json

# 在 Chrome 中查看
# chrome://tracing
```

### 7.2 常见问题排查

**OOM 问题**：
```bash
# 1. 降低显存占用
--sglang-mem-fraction-static 0.5
--max-tokens-per-gpu 2048

# 2. 启用重计算
--recompute-granularity full
--recompute-method uniform

# 3. 减小 batch size
--rollout-batch-size 4
--n-samples-per-prompt 2
```

**训练速度慢**：
```bash
# 1. 检查 PCIe 带宽
nvidia-smi topo -m

# 2. 使用 TP=1 避免通信
--tensor-model-parallel-size 1

# 3. 启用异步训练
python train_async.py
```

**权重同步问题**：
```bash
# 检查权重同步
--check-weight-update-equal

# 单独测试推理
--debug-rollout-only
```

### 7.3 性能优化建议

**针对 4090 的优化**：
1. **TP 策略**：小模型用 TP=1，中等模型用 TP=2
2. **显存管理**：降低 `sglang-mem-fraction-static`
3. **Batch Size**：根据显存动态调整
4. **重计算**：启用节省显存
5. **通信**：避免跨 PCIe 的 TP

### 7.4 阶段 5 检查点

- [ ] 性能分析工具使用熟练
- [ ] 常见问题排查方法掌握
- [ ] 性能优化建议实施
- [ ] 训练效率提升明显
- [ ] 系统稳定性提高

---

## 8. 关键问题与解决方案

### 8.1 显存不足

**问题**：OOM 错误

**解决方案**：
```bash
# 1. 降低 SGLang 显存占用
--sglang-mem-fraction-static 0.5

# 2. 减小 batch size
--rollout-batch-size 4
--n-samples-per-prompt 2
--max-tokens-per-gpu 2048

# 3. 启用重计算
--recompute-granularity full

# 4. 使用分离模式
--actor-num-gpus-per-node 4
--rollout-num-gpus 4
```

### 8.2 PCIe 通信瓶颈

**问题**：TP 效率低

**解决方案**：
```bash
# 1. 使用 TP=1
--tensor-model-parallel-size 1

# 2. 使用共置模式
--colocate

# 3. 减少通信频率
--update-weight-mode delta
```

### 8.3 训练不稳定

**问题**：梯度爆炸、loss 震荡

**解决方案**：
```bash
# 1. 降低学习率
--lr 1e-7

# 2. 增加 KL 惩罚
--kl-loss-coef 0.01

# 3. 调整 clip ratio
--eps-clip 0.1
```

### 8.4 推理速度慢

**问题**：生成速度慢

**解决方案**：
```bash
# 1. 减小生成长度
--rollout-max-response-len 256

# 2. 使用 FP8 推理（如果支持）
--hf-checkpoint /root/model-fp8

# 3. 减少采样数
--n-samples-per-prompt 2
```

---

## 9. 学习路线图

### 9.1 第一周：基础掌握

**目标**：理解 slime 架构和基本概念

**学习内容**：
1. 阅读 slime README 和文档
2. 理解 Training/Rollout/Data Buffer 架构
3. 学习 Megatron 和 SGLang 基础
4. 完成环境搭建

**产出**：
- 环境搭建完成
- 能够运行示例脚本
- 理解基本概念

### 9.2 第二周：流程复现

**目标**：完成小模型全流程训练

**学习内容**：
1. 准备训练数据
2. 转换模型权重
3. 配置训练参数
4. 监控训练过程

**产出**：
- 0.5B 模型训练完成
- 训练曲线分析
- 问题排查经验

### 9.3 第三周：深入理解

**目标**：理解 RL 算法和 Agent 训练

**学习内容**：
1. 学习 GRPO/PPO 算法
2. 理解 Loss Masking 机制
3. 学习自定义接口
4. 完成 Agent 训练

**产出**：
- 3B-4B 模型训练完成
- Agent 训练经验
- 技术总结文档

### 9.4 第四周：优化提升

**目标**：掌握性能优化和调试技巧

**学习内容**：
1. 学习性能分析工具
2. 掌握问题排查方法
3. 学习优化技巧
4. 总结最佳实践

**产出**：
- 性能优化报告
- 最佳实践文档
- 面试准备完成

---

## 10. 附录：配置模板

### 10.1 最小配置（0.5B，单卡测试）

```bash
PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 2048
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.5
)
```

### 10.2 标准配置（0.5B，8卡）

```bash
PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.6
)
```

### 10.3 中等模型配置（3B，4卡）

```bash
PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.6
)
```

### 10.4 Agent 训练配置

```bash
CUSTOM_ARGS=(
   --custom-generate-function-path my_module.generate
   --custom-rm-path my_module.reward_func
)

ROLLOUT_ARGS=(
   --rollout-batch-size 8
   --n-samples-per-prompt 4
   --rollout-max-response-len 512
)
```

---

## 附录 A：关键命令速查

### 环境管理
```bash
# 启动 Docker
docker run --rm --gpus all --ipc=host --shm-size=16g -it slimerl/slime:latest /bin/bash

# 启动 Ray
ray start --head --node-ip-address 127.0.0.1 --num-gpus 8

# 提交任务
ray job submit --address="http://127.0.0.1:8265" -- python3 train.py ...
```

### 模型管理
```bash
# 下载模型
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir /root/Qwen2.5-0.5B-Instruct

# 转换权重
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py ...

# 验证权重
--check-weight-update-equal
```

### 训练监控
```bash
# 查看 GPU 使用
nvidia-smi

# 查看训练日志
tail -f /root/slime/runs/*/train.log

# 启动 profiling
python tools/profile_rollout.py --router-url http://127.0.0.1:3000 --action start
```

---

## 附录 B：参考资料

1. **slime 官方文档**：https://thudm.github.io/slime/
2. **GitHub 仓库**：https://github.com/THUDM/slime
3. **Quick Start 指南**：docs/en/get_started/quick_start.md
4. **示例代码**：examples/ 目录

---

**最后更新**：2025-06-23

**适用环境**：8x RTX 4090，24GB 显存，PCIe 互联
