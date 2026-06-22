# LLM & Agent 应用/后训练面试准备文档

基于 slime 框架的深度学习与面试准备指南

---

## 目录

1. [slime 框架概述](#1-slime-框架概述)
2. [核心架构与设计思想](#2-核心架构与设计思想)
3. [关键技术深度解析](#3-关键技术深度解析)
4. [RL 后训练核心概念](#4-rl-后训练核心概念)
5. [Agent 与多轮交互](#5-agent-与多轮交互)
6. [系统工程与分布式训练](#6-系统工程与分布式训练)
7. [面试考察点与深挖方向](#7-面试考察点与深挖方向)
8. [项目经验介绍模板](#8-项目经验介绍模板)
9. [常见面试问题与参考答案](#9-常见面试问题与参考答案)
10. [进阶话题](#10-进阶话题)

---

## 1. slime 框架概述

### 1.1 项目定位

slime 是一个用于 **RL scaling 的 LLM 后训练框架**，提供两大核心能力：

1. **高性能训练**：通过连接 Megatron 与 SGLang 实现高效训练
2. **灵活的数据生成**：通过自定义数据生成接口和基于服务器的引擎实现任意训练数据生成工作流

**核心设计理念**：训练、rollout、数据缓冲、奖励计算、验证器反馈和环境交互都通过相同的训练/rollout/数据缓冲路径流动，而不是将系统变成一堆断开连接的训练器、rollout 服务和 Agent 框架。

### 1.2 生产验证

slime 已经在以下模型的完整训练流程中得到验证：
- GLM 系列：GLM-5.2, GLM-5.1, GLM-5, GLM-4.7, GLM-4.6, GLM-4.5
- Qwen 系列：Qwen3.6, Qwen3.5, Qwen3Next, Qwen3MoE, Qwen3, Qwen2.5
- DeepSeek V3 系列：DeepSeek V3, V3.1, DeepSeek R1
- Llama 3

### 1.3 技术栈

- **训练后端**：Megatron-LM
- **推理后端**：SGLang
- **调度框架**：Ray
- **通信**：NCCL, HTTP
- **数据格式**：JSONL, SafeTensors

---

## 2. 核心架构与设计思想

### 2.1 三大核心模块

```
┌─────────────────────────────────────────────────────────────┐
│                      slime 架构                              │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Training   │  │   Rollout    │  │ Data Buffer  │      │
│  │  (Megatron)  │  │(SGLang+Router)│  │              │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                 │               │
│         └─────────────────┴─────────────────┘               │
│                           │                                 │
│                    ┌──────┴───────┐                         │
│                    │   Ray 调度   │                         │
│                    └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

#### Training (Megatron)
- 负责主训练过程
- 从 Data Buffer 读取数据
- 训练后将参数同步到 rollout 模块

#### Rollout (SGLang + Router)
- 生成新数据（包括奖励/验证器输出）
- 存储到 Data Buffer
- 自定义 generate 函数可以包装多轮循环、工具调用、环境/沙箱交互和基于验证器的奖励

#### Data Buffer
- 桥接模块
- 管理提示初始化、自定义数据和 rollout 生成方法
- 支持 Agentic 工作流通过相同接口生成样本

### 2.2 设计原则

1. **经过实战验证**：不是 demo，而是生产级框架
2. **正确性优先**：RL bug 通常是静默的，slime 保持数据流显式
3. **原生设计**：直接传递 Megatron 和 SGLang 参数，不添加抽象层
4. **最大数据生成自由度**：数学、代码、搜索、工具、沙箱、验证器、环境、多 Agent 系统都可以作为数据生成或奖励工作流插入
5. **轻量且有主见**：专注于 Megatron + SGLang 路径，深度优化

---

## 3. 关键技术深度解析

### 3.1 权重同步机制

#### 全量同步（默认）
```python
# 每一步都广播所有参数
# 成本与模型大小线性相关
actor_model.update_weights()  # 触发全量权重同步
```

#### Delta Weight Sync（增量同步）
- **核心思想**：只传输变化的权重（~3% 密度）
- **使用场景**：训练/推理分离架构，跨数据中心
- **实现方式**：
  - **Diff**：逐字节比较当前权重与快照
  - **Encode**：编码变化的 (position, value) 对
  - **传输**：通过 NCCL 或磁盘
  - **应用**：接收端 NaN-masked overwrite

```bash
# Delta sync 配置示例
--update-weight-mode delta
--update-weight-transport disk
--update-weight-encoding deltas_zstd
```

**面试深挖点**：
- Q: 为什么需要 Delta Weight Sync？
- A: 全量同步成本与模型大小线性相关，但 RL 步之间只有少量权重变化。Delta sync 只传输差异部分，大幅降低通信成本。

### 3.2 训练-推理协同

#### 分离模式（Disaggregated）
```bash
--actor-num-nodes 1
--actor-num-gpus-per-node 4
--rollout-num-gpus 4
```
- 训练和推理使用不同 GPU
- 适合大规模训练

#### 共置模式（Colocated）
```bash
--colocate
```
- 训练和推理共享 GPU
- 需要调整 `--sglang-mem-fraction-static` 避免显存不足
- 适合资源受限场景

**面试深挖点**：
- Q: 共置模式下如何解决显存竞争？
- A: 通过 `--offload-train` 和 `--offload-rollout` 在不同阶段将模型卸载到 CPU，配合 `--sglang-mem-fraction-static` 调整 SGLang 显存占用。

### 3.3 动态采样（Dynamic Sampling）

DAPO 风格的采样策略：

```bash
--over-sampling-batch-size 64
--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
```

**工作流程**：
1. 采样 `over_sampling_batch_size` 个 prompts
2. 每个 prompt 生成 `n_samples_per_prompt` 个响应
3. 应用过滤函数（如检查奖励标准差 > 0）
4. 如果过滤太严格，自动触发新一轮过采样

**面试深挖点**：
- Q: 为什么需要动态采样？
- A: 避免同质化数据，提高数据多样性。如果一组样本奖励都相同，对训练没有帮助。

### 3.4 Partial Rollout

```bash
--partial-rollout
--buffer-filter-path slime.rollout.filter_hub.buffer_filters.pop_first
```

**作用**：缓存动态采样中被中断的样本，避免计算资源浪费

**面试深挖点**：
- Q: Partial Rollout 如何与动态采样配合？
- A: 动态采样会丢弃不满足条件的样本，但这些样本可能已经生成了一部分。Partial Rollout 缓存这些部分生成的样本，在下一轮 rollout 中继续生成。

### 3.5 数据打包与动态批处理

```bash
--use-dynamic-batch-size
--max-tokens-per-gpu 4608
```

**核心思想**：
- slime 始终通过数据打包方式训练模型
- 动态批处理智能打包不同长度的样本
- 严格确保 per-sample loss 或 per-token loss 正确

**面试深挖点**：
- Q: 动态批处理如何保证 loss 计算正确？
- A: 通过 loss_mask 机制，每个 token 有独立的 loss 权重，打包时正确处理边界。

---

## 4. RL 后训练核心概念

### 4.1 训练流程闭环

```
Data Sampling → Weight Update → Data Sampling → ...
```

**关键参数约束**：
```
(rollout-batch-size × n-samples-per-prompt) = (global-batch-size × num-steps-per-rollout)
```

- `rollout-batch-size`：采样的 prompt 数量
- `n-samples-per-prompt`：每个 prompt 生成的响应数量
- `global-batch-size`：一次参数更新的样本大小
- `num-steps-per-rollout`：使用当前采样数据执行的参数更新次数

### 4.2 支持的 RL 算法

1. **GRPO**（Group Relative Policy Optimization）
   ```bash
   --advantage-estimator grpo
   --eps-clip 0.2
   --eps-clip-high 0.28
   ```

2. **PPO**（Proximal Policy Optimization）
   ```bash
   --advantage-estimator ppo
   ```

3. **Reinforce++**
   ```bash
   --advantage-estimator reinforce_plus_plus
   ```

4. **GSPO**
   ```bash
   --advantage-estimator gspo
   ```

### 4.3 KL 散度控制

```bash
--use-kl-loss
--kl-loss-coef 0.001
--kl-loss-type low_var_kl
```

**面试深挖点**：
- Q: KL 散度在 RL 训练中的作用是什么？
- A: 防止策略偏离参考模型太远，稳定训练。`kl-loss-coef` 为 0 时只作为监控指标，不参与 loss 计算。

### 4.4 Loss 计算方式

```bash
--calculate-per-token-loss  # per-token loss
# 默认：per-sample loss = mean(sum(sample_i) / len(sample_i))
# per-token loss = sum(sum(sample_i)) / sum(len(sample_i))
```

### 4.5 重要性采样（TIS）

```bash
--use-tis
```

**作用**：处理 off-policy 数据，通过重要性采样修正策略偏差

**面试深挖点**：
- Q: 什么时候需要 TIS？
- A: 当使用旧策略生成的数据训练新策略时，需要通过重要性采样修正分布差异。

---

## 5. Agent 与多轮交互

### 5.1 自定义生成函数

```python
async def generate(args, sample: Sample, sampling_params) -> Sample:
    # 实现多轮交互逻辑
    for turn in range(max_turns):
        # 1. 模型生成动作
        model_output = await call_sglang(prompt + full_response, ...)
        
        # 2. 解析并执行动作
        action, content = parse_action(model_output)
        
        # 3. 获取观察结果
        if action == "search":
            tool_output = await search(content)
            # 4. 设置 loss_mask
            loss_masks += [0] * len(tool_tokens)  # 工具输出不参与 loss
        
        # 5. 终止条件
        if action == "answer":
            break
    
    sample.loss_mask = loss_masks
    return sample
```

### 5.2 Loss Masking 机制

**核心思想**：
- 模型生成的 token → `loss_mask = 1`（参与 loss 计算）
- 工具/环境返回的 token → `loss_mask = 0`（不参与 loss 计算）

**面试深挖点**：
- Q: 为什么 Agent 训练需要 Loss Masking？
- A: 工具返回的内容不是模型生成的，如果参与 loss 计算，会让模型学习"复制"工具输出，而不是学习如何正确使用工具。

### 5.3 Search-R1 示例解析

**实现要点**：
1. 使用 `--custom-generate-function-path` 指定自定义生成函数
2. 使用 `--custom-rm-path` 指定自定义奖励函数
3. 在生成函数中实现搜索-生成循环
4. 正确处理 loss_mask

**代码结构**：
```python
# generate_with_search.py
SEARCH_R1_CONFIGS = {
    "max_turns": 2,
    "topk": 3,
    "search_backend": "local",  # 或 "google"
    "return_logprob": True,  # 用于 TIS
}

async def generate(args, sample: Sample, sampling_params) -> Sample:
    # 实现搜索增强生成
    ...

async def reward_func(args, sample, **kwargs):
    # 实现奖励计算
    ...
```

### 5.4 多 Agent 系统

```bash
--rollout-function-path examples/multi_agent/rollout_with_multi_agents.py
```

**支持模式**：
- 多个 Agent 协作
- Agent 间通信
- 复杂工作流编排

---

## 6. 系统工程与分布式训练

### 6.1 Ray 调度架构

```python
# placement_group.py
def _create_placement_group(num_gpus):
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    pg = placement_group(bundles, strategy="PACK")
    ...
```

**关键概念**：
- **Placement Group**：GPU 资源分配
- **Ray Actor**：分布式计算单元
- **Remote Function**：远程函数调用

### 6.2 并行策略

```bash
--tensor-model-parallel-size 2      # 张量并行
--sequence-parallel                 # 序列并行
--pipeline-model-parallel-size 1    # 流水线并行
--context-parallel-size 2           # 上下文并行
--expert-model-parallel-size 1      # 专家并行（MoE）
--expert-tensor-parallel-size 1     # 专家张量并行
```

**面试深挖点**：
- Q: 不同并行策略的适用场景？
- A: 
  - 张量并行：单层内部分割
  - 流水线并行：层间分割
  - 序列并行：长序列分割
  - 专家并行：MoE 模型专家分割

### 6.3 显存优化

```bash
--recompute-granularity full    # 重计算粒度
--recompute-method uniform      # 重计算方法
--recompute-num-layers 1        # 重计算层数
--offload-train                 # 训练时卸载到 CPU
--offload-rollout               # 推理时卸载到 CPU
```

### 6.4 异步训练

**同步训练**（train.py）：
```python
for rollout_id in range(args.num_rollout):
    rollout_data_ref = rollout_manager.generate.remote(rollout_id)
    actor_model.async_train(rollout_id, rollout_data_ref)
    actor_model.update_weights()
```

**异步训练**（train_async.py）：
```python
rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
for rollout_id in range(args.start_rollout_id, args.num_rollout):
    # 提前开始下一轮 rollout
    rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)
    # 训练当前数据
    actor_model.async_train(rollout_id, rollout_data_curr_ref)
```

**面试深挖点**：
- Q: 异步训练的优势和挑战？
- A: 
  - 优势：重叠训练和推理，提高吞吐量
  - 挑战：权重同步时机、数据一致性

---

## 7. 面试考察点与深挖方向

### 7.1 基础概念考察

#### 考察点 1：RL 后训练流程
**问题**：请描述 LLM RL 后训练的基本流程
**深挖方向**：
- 数据采样（Rollout）
- 奖励计算
- 策略更新
- 权重同步

**参考答案**：
RL 后训练是一个闭环流程：
1. 使用当前策略采样生成数据（Rollout）
2. 计算奖励（可以是规则、模型或环境反馈）
3. 使用 RL 算法（如 GRPO/PPO）更新策略
4. 将新权重同步到推理引擎
5. 重复上述过程

#### 考察点 2：GRPO vs PPO
**问题**：GRPO 和 PPO 有什么区别？
**深挖方向**：
- 优势计算方式
- 是否需要 Critic 模型
- 计算效率

**参考答案**：
- GRPO：组内相对优势，不需要 Critic 模型，计算效率高
- PPO：需要 Critic 模型估计价值函数，更稳定但计算成本高

### 7.2 系统设计考察

#### 考察点 3：训练-推理协同
**问题**：如何设计一个高效的训练-推理协同系统？
**深挖方向**：
- 资源分配策略
- 权重同步机制
- 显存管理
- 故障恢复

**参考答案**：
需要考虑：
1. 分离 vs 共置部署的权衡
2. 全量 vs 增量权重同步
3. 显存优化（卸载、重计算）
4. 检查点和容错机制

#### 考察点 4：大规模分布式训练
**问题**：如何扩展到数百 GPU 的训练？
**深挖方向**：
- 并行策略选择
- 通信优化
- 负载均衡

### 7.3 Agent 系统考察

#### 考察点 5：Agent 训练的挑战
**问题**：Agent RL 训练面临哪些独特挑战？
**深挖方向**：
- 多轮交互的信用分配
- 工具调用的奖励设计
- 长序列处理
- 部分可观测性

**参考答案**：
1. **信用分配**：多轮交互中，如何分配奖励到每一步
2. **Loss Masking**：工具输出不应参与 loss 计算
3. **长序列**：Agent 轨迹可能很长，需要高效处理
4. **稀疏奖励**：只有最终结果有奖励，中间步骤没有

#### 考察点 6：多轮交互的 Loss 设计
**问题**：在多轮 Agent 交互中，如何设计 loss？
**深挖方向**：
- 哪些 token 应该参与 loss
- 如何处理工具输出
- 奖励分配策略

### 7.4 工程实现考察

#### 考察点 7：数据流设计
**问题**：如何设计一个灵活的数据流系统？
**深挖方向**：
- 数据格式设计
- 自定义接口
- 扩展性

**参考答案**：
slime 的设计：
1. 使用 `Sample` 数据类统一数据格式
2. 通过函数路径参数实现自定义（如 `--custom-generate-function-path`）
3. Data Buffer 作为桥接模块

#### 考察点 8：正确性保证
**问题**：如何保证 RL 训练的正确性？
**深挖方向**：
- 数值稳定性
- 数据一致性
- 调试工具

---

## 8. 项目经验介绍模板

### 8.1 项目背景（1分钟）

"我研究/使用了 slime 框架，这是一个用于 RL scaling 的 LLM 后训练框架。它被用于训练 GLM、Qwen、DeepSeek 等知名模型。框架的核心设计是将训练（Megatron）、推理（SGLang）和数据管理（Data Buffer）统一在一个系统中。"

### 8.2 技术亮点（2分钟）

"slime 有几个关键技术亮点：

1. **原生引擎透传**：直接使用 Megatron 和 SGLang 的参数，不需要额外抽象层
2. **灵活的自定义接口**：通过函数路径参数实现自定义生成、奖励计算等
3. **高效的权重同步**：支持全量和增量两种模式
4. **动态采样**：支持 DAPO 风格的采样策略"

### 8.3 个人贡献/学习（1分钟）

"通过学习 slime，我深入理解了：
1. RL 后训练的完整流程
2. 大规模分布式训练的工程挑战
3. Agent 系统的训练方法
4. 系统设计的权衡（如分离 vs 共置）"

### 8.4 技术细节准备

**如果被问到具体实现**：
- 权重同步：可以解释 Delta Weight Sync 的原理
- 数据流：可以解释 Sample 数据类和 Data Buffer
- 自定义接口：可以解释如何通过函数路径实现扩展

---

## 9. 常见面试问题与参考答案

### Q1: 什么是 RL 后训练？为什么需要它？

**参考答案**：
RL 后训练是在预训练/微调之后，使用强化学习进一步优化模型的过程。它可以让模型：
1. 学习人类偏好（如 RLHF）
2. 提高推理能力（如 DeepSeek-R1）
3. 学习使用工具（如 Agent 训练）
4. 适应特定任务（如代码生成、数学推理）

### Q2: 解释 GRPO 算法

**参考答案**：
GRPO（Group Relative Policy Optimization）是一种高效的 RL 算法：
1. 对每个 prompt 生成一组响应
2. 计算组内相对优势（而不是绝对优势）
3. 使用 clipping 机制限制策略更新幅度
4. 不需要 Critic 模型，计算效率高

**优势计算**：
```
A_i = (r_i - mean(r_group)) / std(r_group)
```

### Q3: 如何处理 Agent 训练中的信用分配问题？

**参考答案**：
信用分配是 Agent 训练的核心挑战，有几种方法：
1. **稀疏奖励**：只在最终结果给予奖励，依赖 RL 算法自动分配信用
2. **密集奖励**：每一步都给予奖励（如工具调用成功/失败）
3. **优势函数**：使用 Critic 模型估计每一步的价值
4. **TIS**：通过重要性采样修正 off-policy 数据

### Q4: 解释训练-推理分离 vs 共置的权衡

**参考答案**：

| 方面 | 分离模式 | 共置模式 |
|------|----------|----------|
| 资源利用率 | 较低（需要两套 GPU） | 较高（共享 GPU） |
| 通信成本 | 高（需要权重同步） | 低（本地访问） |
| 显存管理 | 简单 | 复杂（需要卸载） |
| 适用场景 | 大规模训练 | 资源受限 |

### Q5: 如何设计一个可扩展的 RL 训练框架？

**参考答案**：
关键设计原则：
1. **模块化**：训练、推理、数据管理分离
2. **可扩展性**：通过自定义接口支持新功能
3. **正确性优先**：显式数据流，易于调试
4. **性能优化**：异步执行、权重同步优化

### Q6: 解释 Loss Masking 在 Agent 训练中的作用

**参考答案**：
在 Agent 训练中，轨迹包含模型生成和工具返回两部分：
- 模型生成的 token：应该参与 loss 计算（loss_mask = 1）
- 工具返回的 token：不应该参与 loss 计算（loss_mask = 0）

原因：工具返回的内容不是模型生成的，如果参与 loss，会让模型学习"复制"工具输出，而不是学习如何正确使用工具。

### Q7: 如何处理长序列 Agent 轨迹？

**参考答案**：
几种方法：
1. **上下文压缩**：定期压缩历史上下文
2. **分段训练**：将长轨迹分成多个段分别训练
3. **滑动窗口**：只保留最近的 N 轮交互
4. **高效注意力**：使用 Flash Attention 等优化

### Q8: 解释动态采样的作用和实现

**参考答案**：
**作用**：提高数据多样性，避免同质化数据

**实现**：
1. 采样比实际需要更多的 prompts
2. 每个 prompt 生成多个响应
3. 应用过滤函数（如检查奖励标准差 > 0）
4. 丢弃不满足条件的样本
5. 如果过滤太严格，自动触发新一轮采样

### Q9: 如何保证分布式训练的正确性？

**参考答案**：
1. **数值检查**：定期比较权重，确保同步正确
2. **检查点**：支持训练恢复
3. **调试工具**：trace viewer、profiling
4. **单元测试**：CPU 单元测试、GPU 端到端测试

### Q10: 解释 slime 的自定义接口设计

**参考答案**：
slime 使用函数路径参数实现自定义：
```bash
--custom-generate-function-path my_module.generate
--custom-rm-path my_module.reward_func
```

**优点**：
1. 不需要修改核心代码
2. 易于扩展和测试
3. 支持复杂工作流（如多轮交互、工具调用）

---

## 10. 进阶话题

### 10.1 On-Policy vs Off-Policy

**On-Policy**：
- 使用当前策略生成的数据训练
- 数据新鲜，但计算成本高
- slime 默认模式

**Off-Policy**：
- 使用旧策略生成的数据训练
- 数据可复用，但需要修正
- 通过 TIS 实现

### 10.2 异步 RL 训练

**挑战**：
- 权重同步时机
- 数据一致性
- 训练稳定性

**slime 的实现**：
- 提前开始下一轮 rollout
- 使用 `update_weights_interval` 控制同步频率
- 确保权重更新不在生成中间进行

### 10.3 多模态 Agent

**支持**：
- 图像输入
- 视频输入
- 多模态工具调用

**实现**：
- 使用 processor 处理多模态输入
- 在 Sample 中存储多模态数据

### 10.4 可验证环境（Verifiable Environments）

**概念**：
- 程序化生成问题
- 算法可验证的奖励
- 支持大规模 RL 训练

**示例**：
- 数学问题
- 代码生成
- 逻辑推理

### 10.5 推测解码（Speculative Decoding）

**原理**：
- 使用小模型生成草稿
- 大模型并行验证
- 提高推理速度

**slime 支持**：
```bash
--speculative-decoding
```

---

## 附录 A：关键代码位置

| 模块 | 文件路径 | 说明 |
|------|----------|------|
| 参数解析 | `slime/utils/arguments.py` | 所有参数定义 |
| Sample 数据类 | `slime/utils/types.py` | 核心数据结构 |
| SGLang Rollout | `slime/rollout/sglang_rollout.py` | 默认 rollout 实现 |
| 训练循环 | `train.py` | 同步训练入口 |
| 异步训练 | `train_async.py` | 异步训练入口 |
| 权重同步 | `slime/backends/sglang_utils/` | 权重同步实现 |
| 自定义接口 | `docs/en/get_started/customization.md` | 自定义接口文档 |

## 附录 B：关键配置示例

### 基础 GRPO 训练
```bash
--advantage-estimator grpo
--use-kl-loss
--kl-loss-coef 0.001
--eps-clip 0.2
--eps-clip-high 0.28
```

### Agent 训练
```bash
--custom-generate-function-path my_agent.generate
--custom-rm-path my_agent.reward_func
--rollout-max-response-len 8192
```

### 大规模训练
```bash
--tensor-model-parallel-size 4
--pipeline-model-parallel-size 2
--context-parallel-size 2
--expert-model-parallel-size 4
```

## 附录 C：学习资源

1. **官方文档**：https://thudm.github.io/slime/
2. **GitHub 仓库**：https://github.com/THUDM/slime
3. **博客文章**：
   - [slime: An SGLang-Native Post-Training Framework for RL Scaling](https://lmsys.org/blog/2025-07-09-slime/)
   - [Agent-Oriented Design: An Asynchronous and Decoupled Framework for Agentic RL](https://www.notion.so/Agent-Oriented-Design-An-Asynchronous-and-Decoupled-Framework-for-Agentic-RL-2278e692d081802cbdd5d37cef76a547)
4. **示例代码**：`examples/` 目录下的各种示例

---

**最后更新**：2025-06-22

**适用场景**：LLM 算法实习面试准备，重点关注 RL 后训练、Agent 系统、分布式训练等方向
