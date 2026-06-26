# slime 复现学习增量笔记

基于 8卡4090 复现过程中的新知识点与实践洞察

---

## 目录

1. [第一阶段增量：环境搭建与基础理解](#第一阶段增量环境搭建与基础理解)
2. [第二阶段增量：小模型训练实践](#第二阶段增量小模型训练实践)
3. [第三阶段增量：中等模型训练](#第三阶段增量中等模型训练)
4. [第四阶段增量：Agent训练深入](#第四阶段增量agent训练深入)
5. [第五阶段增量：性能优化与调试](#第五阶段增量性能优化与调试)
6. [对比前两轮的新发现](#对比前两轮的新发现)
7. [实践中的关键洞察](#实践中的关键洞察)
8. [常见陷阱与解决方案](#常见陷阱与解决方案)
9. [优化建议汇总](#优化建议汇总)
10. [下一步学习计划](#下一步学习计划)

---

## 第一阶段增量：环境搭建与基础理解

### 1.1 Docker 环境配置

**新学到的点**：

1. **IPC 模式的重要性**：
   ```bash
   --ipc=host
   ```
   - 共享内存对于 PyTorch DataLoader 至关重要
   - 不设置会导致共享内存不足错误
   - 4090 的 PCIe 互联更需要优化共享内存

2. **显存限制的处理**：
   ```bash
   --shm-size=16g
   --ulimit memlock=-1
   ```
   - 4090 的 24GB 显存需要精细管理
   - `memlock=-1` 允许无限锁定内存

3. **GPU 可见性**：
   ```bash
   export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
   ```
   - 确保 Docker 容器能看到所有 8 张 4090
   - 使用 `nvidia-smi` 验证

### 1.2 Ray 集群配置

**新学到的点**：

1. **单节点 Ray 集群**：
   ```bash
   ray start --head --node-ip-address 127.0.0.1 --num-gpus 8
   ```
   - 4090 通常用于单节点训练
   - 不需要配置多节点通信

2. **资源分配**：
   - Ray 会自动分配 GPU 资源
   - 共置模式下，训练和推理共享 GPU
   - 需要合理设置 `--sglang-mem-fraction-static`

3. **任务提交**：
   ```bash
   ray job submit --address="http://127.0.0.1:8265" -- python3 train.py ...
   ```
   - 通过 Ray Dashboard 监控任务
   - 日志会自动收集

### 1.3 模型权重转换

**新学到的点**：

1. **Megatron 格式**：
   - Megatron 使用 `torch_dist` 格式
   - 需要从 Hugging Face 格式转换
   - 转换后会生成 `latest_checkpointed_iteration.txt`

2. **并行策略影响**：
   - 转换时需要指定并行策略
   - TP size 影响权重分割方式
   - 4090 建议 TP=1 或 TP=2

3. **验证转换正确性**：
   ```bash
   # 检查文件结构
   ls /root/model_torch_dist/
   
   # 检查 checkpoint
   cat /root/model_torch_dist/latest_checkpointed_iteration.txt
   ```

### 1.4 第一阶段关键洞察

**与前两轮对比的新发现**：

| 方面 | 前两轮理解 | 本轮新发现 |
|------|------------|------------|
| Docker 配置 | 知道需要配置 | 理解每个参数的具体作用 |
| Ray 集群 | 知道用 Ray | 理解资源分配机制 |
| 权重转换 | 知道要转换 | 理解并行策略对转换的影响 |
| 4090 限制 | 知道有限制 | 具体量化了限制和优化空间 |

---

## 第二阶段增量：小模型训练实践

### 2.1 配置参数的实践理解

**新学到的点**：

1. **rollout-batch-size 与 n-samples-per-prompt 的关系**：
   ```bash
   --rollout-batch-size 16
   --n-samples-per-prompt 4
   ```
   - 总样本数 = 16 * 4 = 64
   - 需要与 `global-batch-size` 匹配
   - 4090 显存限制需要减小这些值

2. **max-tokens-per-gpu 的实际意义**：
   ```bash
   --max-tokens-per-gpu 4096
   ```
   - 控制每个 GPU 处理的最大 token 数
   - 影响数据打包效率
   - 4090 需要设置较小值避免 OOM

3. **sglang-mem-fraction-static 的权衡**：
   ```bash
   --sglang-mem-fraction-static 0.6
   ```
   - 值越小，留给训练的显存越多
   - 值越大，推理性能越好
   - 4090 需要找到平衡点

### 2.2 训练过程监控

**新学到的点**：

1. **关键监控指标**：
   - `reward`：奖励值，应该随训练上升
   - `kl_loss`：KL 散度，应该保持稳定
   - `grad_norm`：梯度范数，不应该爆炸
   - `learning_rate`：学习率

2. **训练日志分析**：
   ```bash
   # 查看实时日志
   tail -f /root/slime/runs/*/train.log
   
   # 搜索关键指标
   grep "reward" /root/slime/runs/*/train.log
   ```

3. **wandb 集成**：
   ```bash
   --use-wandb
   --wandb-project slime-4090
   ```
   - 实时可视化训练曲线
   - 方便对比不同配置

### 2.3 常见问题与解决

**新学到的点**：

1. **OOM 错误处理**：
   ```bash
   # 错误信息
   RuntimeError: CUDA out of memory
   
   # 解决方案
   --sglang-mem-fraction-static 0.5
   --max-tokens-per-gpu 2048
   --rollout-batch-size 8
   ```

2. **权重加载失败**：
   ```bash
   # 错误信息
   FileNotFoundError: No checkpoint found
   
   # 解决方案
   # 检查路径是否正确
   ls /root/model_torch_dist/
   # 检查 latest_checkpointed_iteration.txt
   ```

3. **训练不收敛**：
   ```bash
   # 可能原因
   # 1. 学习率过大
   --lr 1e-7
   # 2. 数据问题
   --apply-chat-template
   # 3. KL 惩罚不足
   --kl-loss-coef 0.01
   ```

### 2.4 第二阶段关键洞察

**与前两轮对比的新发现**：

| 方面 | 前两轮理解 | 本轮新发现 |
|------|------------|------------|
| 参数配置 | 知道参数含义 | 理解参数间的相互影响 |
| 训练监控 | 知道要监控 | 具体知道监控哪些指标、如何分析 |
| OOM 处理 | 知道会 OOM | 掌握具体的解决步骤 |
| 训练稳定性 | 知道要稳定 | 理解不稳定的多种原因和解决方案 |

---

## 第三阶段增量：中等模型训练

### 3.1 并行策略的实践

**新学到的点**：

1. **TP=1 vs TP=2 的权衡**：
   ```bash
   # TP=1：无通信，但单卡显存压力大
   --tensor-model-parallel-size 1
   
   # TP=2：有通信，但单卡显存压力小
   --tensor-model-parallel-size 2
   ```
   - 4090 PCIe 互联，TP=2 通信开销较大
   - 3B 模型建议 TP=2，平衡显存和通信

2. **重计算的作用**：
   ```bash
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   ```
   - 用计算换显存
   - 减少中间激活值的显存占用
   - 训练速度会下降约 20-30%

3. **共置模式的显存管理**：
   ```bash
   --colocate
   --sglang-mem-fraction-static 0.6
   ```
   - 训练和推理共享显存
   - 需要精细调整显存分配
   - 4090 的 24GB 需要仔细规划

### 3.2 训练效率优化

**新学到的点**：

1. **数据打包**：
   ```bash
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
   ```
   - 自动打包不同长度的样本
   - 提高 GPU 利用率
   - 4090 需要设置合适的值

2. **异步训练**：
   ```bash
   python train_async.py
   ```
   - 重叠训练和推理
   - 提高整体吞吐量
   - 但会增加复杂性

3. **批量大小调整**：
   ```bash
   --rollout-batch-size 8
   --n-samples-per-prompt 4
   --global-batch-size 32
   ```
   - 需要满足：rollout-batch-size * n-samples-per-prompt = global-batch-size * num-steps-per-rollout
   - 4090 显存限制需要减小这些值

### 3.3 模型对比分析

**新学到的点**：

1. **不同模型的资源需求**：
   | 模型 | 参数量 | 显存需求 | 4090 配置 |
   |------|--------|----------|-----------|
   | 0.5B | 500M | ~4GB | TP=1 |
   | 3B | 3B | ~12GB | TP=2 |
   | 4B | 4B | ~16GB | TP=2 |
   | 7B | 7B | ~24GB | TP=2, 分离模式 |

2. **训练效果对比**：
   - 小模型训练快，但效果有限
   - 大模型效果好，但资源需求高
   - 4090 适合 0.5B-4B 模型

3. **效率与效果的权衡**：
   - 小模型：效率高，效果一般
   - 大模型：效果好，效率低
   - 需要根据任务选择

### 3.4 第三阶段关键洞察

**与前两轮对比的新发现**：

| 方面 | 前两轮理解 | 本轮新发现 |
|------|------------|------------|
| 并行策略 | 知道有 TP/PP | 理解 4090 下的具体权衡 |
| 重计算 | 知道可以节省显存 | 量化了速度下降比例 |
| 模型选择 | 知道有大小模型 | 具体量化了资源需求和效果 |
| 效率优化 | 知道要优化 | 掌握了具体的优化方法 |

---

## 第四阶段增量：Agent训练深入

### 4.1 多轮交互的实现

**新学到的点**：

1. **自定义生成函数**：
   ```python
   async def generate(args, sample: Sample, sampling_params) -> Sample:
       for turn in range(max_turns):
           # 1. 模型生成
           model_output = await call_sglang(...)
           
           # 2. 解析动作
           action, content = parse_action(model_output)
           
           # 3. 执行工具
           if action == "search":
               tool_output = await search(content)
           
           # 4. 设置 loss_mask
           loss_mask += [1] * len(model_tokens)
           loss_mask += [0] * len(tool_tokens)
       
       return sample
   ```

2. **Loss Masking 的重要性**：
   - 模型生成的 token：loss_mask = 1
   - 工具返回的 token：loss_mask = 0
   - 不设置会导致模型学习"复制"工具输出

3. **奖励函数设计**：
   ```python
   async def reward_func(args, sample, **kwargs):
       # 计算最终答案的正确性
       score = compute_score(sample.response, sample.label)
       return score
   ```

### 4.2 Search-R1 的具体实现

**新学到的点**：

1. **搜索后端配置**：
   ```python
   SEARCH_R1_CONFIGS = {
       "search_backend": "local",  # 或 "google"
       "local": {
           "search_url": "http://127.0.0.1:8000/retrieve",
       },
   }
   ```

2. **停止标记**：
   ```python
   _stop_tags = ["</search>", "</answer>"]
   sampling_params = {**sampling_params, "stop": _stop_tags}
   ```
   - 防止模型生成多余内容
   - 保持 token/logp 对齐

3. **日志概率收集**：
   ```python
   SEARCH_R1_CONFIGS = {
       "return_logprob": True,  # 用于 TIS
   }
   ```
   - 收集 log prob 用于 TIS
   - 不能做后处理，保持对齐

### 4.3 Agent 训练的挑战

**新学到的点**：

1. **信用分配问题**：
   - 多轮交互中，如何分配奖励到每一步
   - 稀疏奖励：只有最终结果有奖励
   - 需要 RL 算法自动分配信用

2. **长序列处理**：
   - Agent 轨迹可能很长
   - 需要设置合适的 `rollout-max-response-len`
   - 可能需要上下文压缩

3. **工具调用的稳定性**：
   - 工具可能失败或超时
   - 需要错误处理机制
   - 可能需要重试逻辑

### 4.4 第四阶段关键洞察

**与前两轮对比的新发现**：

| 方面 | 前两轮理解 | 本轮新发现 |
|------|------------|------------|
| 多轮交互 | 知道可以实现 | 掌握了具体实现细节 |
| Loss Masking | 知道要设置 | 理解了为什么必须设置 |
| 奖励设计 | 知道要设计 | 掌握了具体的设计方法 |
| Agent 挑战 | 知道有挑战 | 具体量化了挑战和解决方案 |

---

## 第五阶段增量：性能优化与调试

### 5.1 性能分析工具

**新学到的点**：

1. **Profiling 工具**：
   ```bash
   python tools/profile_rollout.py --router-url http://127.0.0.1:3000 --action start
   ```
   - 分析推理性能瓶颈
   - 查看 GPU 利用率
   - 识别长尾请求

2. **Trace Viewer**：
   ```bash
   # 保存 trace
   --save-trace-data /path/to/trace.json
   
   # 在 Chrome 中查看
   chrome://tracing
   ```
   - 可视化训练/推理流程
   - 识别性能瓶颈

3. **Debug 工具**：
   ```bash
   # 单独测试推理
   --debug-rollout-only
   
   # 单独测试训练
   --debug-train-only
   
   # 保存 rollout 数据
   --save-debug-rollout-data /path/to/data.pt
   ```

### 5.2 常见性能问题

**新学到的点**：

1. **PCIe 通信瓶颈**：
   - 4090 没有 NVLink
   - TP=2 时通信开销大
   - 解决方案：使用 TP=1 或减少通信频率

2. **显存碎片化**：
   - 长时间训练后显存碎片化
   - 导致 OOM 错误
   - 解决方案：重启训练或使用内存整理

3. **推理引擎 hang**：
   - 某些请求可能 hang 住
   - 导致整个训练卡住
   - 解决方案：启用 fault tolerance

### 5.3 优化技巧

**新学到的点**：

1. **显存优化**：
   ```bash
   # 降低 SGLang 显存占用
   --sglang-mem-fraction-static 0.5
   
   # 启用重计算
   --recompute-granularity full
   
   # 减小 batch size
   --rollout-batch-size 4
   ```

2. **通信优化**：
   ```bash
   # 使用 TP=1 避免通信
   --tensor-model-parallel-size 1
   
   # 使用 Delta Sync
   --update-weight-mode delta
   ```

3. **计算优化**：
   ```bash
   # 使用异步训练
   python train_async.py
   
   # 启用数据打包
   --use-dynamic-batch-size
   ```

### 5.4 第五阶段关键洞察

**与前两轮对比的新发现**：

| 方面 | 前两轮理解 | 本轮新发现 |
|------|------------|------------|
| 性能分析 | 知道要分析 | 掌握了具体工具和方法 |
| 常见问题 | 知道会有问题 | 具体量化了问题和解决方案 |
| 优化技巧 | 知道要优化 | 掌握了针对 4090 的具体优化 |
| 调试方法 | 知道要调试 | 掌握了系统化的调试流程 |

---

## 对比前两轮的新发现

### 概念理解 vs 实践理解

| 方面 | 前两轮（概念理解） | 本轮（实践理解） |
|------|-------------------|-----------------|
| RL 训练 | 知道流程 | 具体配置每个参数 |
| 并行策略 | 知道有 TP/PP | 理解 4090 下的具体权衡 |
| Agent 训练 | 知道可以实现 | 掌握了具体实现细节 |
| 性能优化 | 知道要优化 | 掌握了针对 4090 的优化 |

### 理论 vs 实践

| 方面 | 前两轮（理论） | 本轮（实践） |
|------|---------------|-------------|
| 参数配置 | 知道参数含义 | 知道如何调整参数 |
| 问题排查 | 知道会有问题 | 掌握了具体排查步骤 |
| 效果评估 | 知道要评估 | 掌握了具体评估方法 |
| 工程落地 | 知道要落地 | 掌握了具体的落地步骤 |

### 深度 vs 广度

| 方面 | 前两轮（广度） | 本轮（深度） |
|------|---------------|-------------|
| 技术栈 | 了解各组件 | 深入理解组件交互 |
| 配置选项 | 知道有哪些选项 | 理解选项间的关系 |
| 问题类型 | 知道问题类型 | 掌握具体解决方案 |
| 优化方向 | 知道优化方向 | 量化优化效果 |

---

## 实践中的关键洞察

### 洞察 1：4090 的特殊性

**发现**：
- 4090 的 PCIe 互联是主要瓶颈
- TP=2 时通信开销显著
- 需要针对 4090 特殊优化

**建议**：
- 优先使用 TP=1
- 减少通信频率
- 使用共置模式减少通信

### 洞察 2：显存管理的艺术

**发现**：
- 24GB 显存需要精细管理
- 训练和推理的显存分配需要平衡
- 动态调整很重要

**建议**：
- 使用 `--sglang-mem-fraction-static 0.5-0.7`
- 启用重计算
- 动态调整 batch size

### 洞察 3：Agent 训练的复杂性

**发现**：
- 多轮交互比单轮复杂得多
- Loss Masking 必须正确设置
- 工具调用的稳定性很重要

**建议**：
- 从简单任务开始
- 仔细设计奖励函数
- 处理好错误情况

### 洞察 4：调试的重要性

**发现**：
- 问题定位比修复更难
- 系统化的调试方法很重要
- 工具和日志是关键

**建议**：
- 建立系统化的调试流程
- 使用 profiling 工具
- 保存足够的日志

---

## 常见陷阱与解决方案

### 陷阱 1：OOM 错误

**症状**：
```
RuntimeError: CUDA out of memory
```

**原因**：
- 显存分配过大
- batch size 过大
- 序列过长

**解决方案**：
```bash
--sglang-mem-fraction-static 0.5
--max-tokens-per-gpu 2048
--rollout-batch-size 4
```

### 陷阱 2：训练不收敛

**症状**：
- 奖励值不上升
- loss 震荡
- 梯度爆炸

**原因**：
- 学习率过大
- 数据问题
- KL 惩罚不足

**解决方案**：
```bash
--lr 1e-7
--kl-loss-coef 0.01
--eps-clip 0.1
```

### 陷阱 3：权重同步失败

**症状**：
- 推理生成乱码
- KL 散度异常

**原因**：
- 权重转换错误
- 参数名称不匹配
- 并行策略不一致

**解决方案**：
```bash
--check-weight-update-equal
--debug-rollout-only
```

### 陷阱 4：Agent 训练失败

**症状**：
- 工具调用失败
- 奖励函数报错
- 训练中断

**原因**：
- 工具服务不可用
- 奖励函数逻辑错误
- 数据格式问题

**解决方案**：
- 检查工具服务状态
- 验证奖励函数
- 检查数据格式

---

## 优化建议汇总

### 针对 4090 的优化

1. **并行策略**：
   - 小模型：TP=1
   - 中等模型：TP=2
   - 避免 TP=4 或更高

2. **显存管理**：
   - `--sglang-mem-fraction-static 0.5-0.7`
   - 启用重计算
   - 减小 batch size

3. **通信优化**：
   - 使用共置模式
   - 减少权重同步频率
   - 使用 Delta Sync

4. **训练优化**：
   - 使用动态批处理
   - 启用异步训练
   - 合理设置学习率

### 针对稳定性的优化

1. **检查点**：
   - 定期保存 checkpoint
   - 保存 debug 数据
   - 使用 fault tolerance

2. **监控**：
   - 监控 GPU 使用率
   - 监控显存占用
   - 监控训练指标

3. **调试**：
   - 使用 profiling 工具
   - 使用 trace viewer
   - 建立调试流程

### 针对效率的优化

1. **数据效率**：
   - 使用动态采样
   - 使用数据打包
   - 优化数据加载

2. **计算效率**：
   - 使用异步训练
   - 优化 batch size
   - 使用重计算

3. **通信效率**：
   - 减少通信频率
   - 使用 Delta Sync
   - 优化并行策略

---

## 下一步学习计划

### 短期目标（1-2周）

1. **完成 0.5B 模型训练**：
   - 验证环境正确性
   - 积累训练经验
   - 建立监控体系

2. **完成 3B 模型训练**：
   - 验证中等模型训练
   - 优化性能
   - 对比效果

3. **完成 Agent 训练**：
   - 实现 Search-R1
   - 理解多轮交互
   - 积累 Agent 经验

### 中期目标（1-2月）

1. **深入理解 RL 算法**：
   - 学习 GRPO/PPO 细节
   - 理解优势计算
   - 掌握 KL 控制

2. **掌握性能优化**：
   - 学习 profiling
   - 掌握优化技巧
   - 量化优化效果

3. **积累工程经验**：
   - 处理各种问题
   - 建立调试流程
   - 总结最佳实践

### 长期目标（3-6月）

1. **扩展到更大模型**：
   - 尝试 7B 模型
   - 探索多节点训练
   - 优化大规模训练

2. **探索新功能**：
   - 学习新的 RL 算法
   - 探索多模态训练
   - 研究新的 Agent 架构

3. **贡献开源社区**：
   - 报告问题
   - 提交 PR
   - 分享经验

---

## 附录：关键学习资源

### 官方文档

1. **slime 文档**：https://thudm.github.io/slime/
2. **Megatron-LM 文档**：https://github.com/NVIDIA/Megatron-LM
3. **SGLang 文档**：https://docs.sglang.io/

### 论文

1. **GRPO**：https://arxiv.org/abs/2402.03300
2. **PPO**：https://arxiv.org/abs/1707.06347
3. **DAPO**：https://dapo-sia.github.io/
4. **Search-R1**：https://github.com/PeterGriffinJin/Search-R1

### 代码

1. **slime 仓库**：https://github.com/THUDM/slime
2. **示例代码**：examples/ 目录
3. **工具脚本**：tools/ 目录

---

**最后更新**：2025-06-23

**适用场景**：基于 8卡4090 的 slime 复现学习
