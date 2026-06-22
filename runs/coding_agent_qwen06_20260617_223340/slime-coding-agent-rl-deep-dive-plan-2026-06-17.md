# slime `examples/coding_agent_rl` 深度解读、复现与改进计划

> 源码基准：远端 `/home/chenyizhou/slime`，commit `243773c`，`[coding-agent-rl] Refactor coding-agent RL: turn-node TrajectoryManager + pluggable harness layer (#2005)`。  
> 关联前文：`/Users/sskchen/Documents/New project/slime-framework-interview-prep-2026-06-14.md`。前文讲 slime 总体后训练框架；本文聚焦 coding-agent RL example。  
> 目标：让你能真正读懂这个 example，能分阶段复现，能提出有代码落点的改进方案，也能在 LLM/Agent 后训练面试里把它讲出深度。

---

## 0. 先给面试官的一分钟版本

`examples/coding_agent_rl` 是 slime 里最接近真实 Agent 后训练的 example：它不是让模型单轮回答，而是让一个真实 coding agent CLI 在 sandbox 里读文件、改代码、跑命令，最后产出 `git diff`；再把这个 diff 放到另一个干净 sandbox 中用测试 harness 打分，得到 reward；训练样本则由 adapter 把 agent 的多轮 string/message 对话还原成精确 token trajectory。

新版重构后的核心价值是分层清楚：

- `examples/coding_agent_rl/generate.py` 只是 per-sample orchestrator。
- `examples/coding_agent_rl/swe.py` 只负责 SWE 任务：workspace、diff、eval。
- `slime.agent.sandbox` 抽象 sandbox 后端，当前有 E2B。
- `slime.agent.harness` 抽象 coding agent CLI，当前有 Claude Code 和 Codex。
- `slime.agent.adapters` 把 Anthropic/OpenAI 协议转成 SGLang `/generate`，并捕获真实 token/logprob。
- `slime.agent.trajectory.TrajectoryManager` 解决最关键的问题：把多轮 agent 的 string history 转回 token-level trainable `Sample`，处理 sub-agent、auto-compaction、prompt drift、fan-out 和 loss mask。

面试时最有含金量的表达：

> 这个 example 的关键不是“用 Claude Code 跑一遍 SWE-bench”，而是如何在真实 agent 运行时保持 RL 训练的 token 语义正确：模型采样出来的 token 才能 `loss_mask=1`，工具观察、模板、重写、compaction 后无法证明来源的 token 必须 mask 掉；reward 来自干净 eval sandbox 的 diff 验证，避免 test-cheating；如果一个 session 分叉成多个 leaf chain，还要通过 `rollout_id`、reward split 和 per-rollout loss denominator 控制样本权重。

---

## 1. 当前版本为什么值得重读

远端当前最新相关提交是：

```text
243773c [coding-agent-rl] Refactor coding-agent RL: turn-node TrajectoryManager + pluggable harness layer (#2005)
```

这个提交对 coding-agent RL 做了大重构，变化非常重要：

- 删除了旧的 example-local sandbox/adapter 逻辑。
- 新增 `slime/agent` 通用包。
- 新增 `AnthropicAdapter` 和 `OpenAIAdapter`。
- 新增 `ClaudeCodeHarness` 和 `CodexHarness`。
- 新增 `TrajectoryManager` 的 turn-node 树结构。
- 新增大量 CPU-only agent tests。
- `examples/coding_agent_rl` 本身变薄，只保留 SWE 任务 recipe 和 launcher。

这意味着你准备面试时不应该只讲“这个 example 怎么跑”，而要讲它如何被抽象成一个可复用 Agent RL 基础设施。

---

## 2. 文件地图

### 2.1 Example 层

| 文件 | 职责 |
| --- | --- |
| `/home/chenyizhou/slime/examples/coding_agent_rl/README.md` | 解释 coding-agent RL 的总体 loop、环境变量、dataset schema、string-in token-out、fan-out |
| `/home/chenyizhou/slime/examples/coding_agent_rl/generate.py` | slime `--custom-generate-function-path` 入口；每个 sample 启动 sandbox、跑 harness、抓 diff、eval、导出 Sample |
| `/home/chenyizhou/slime/examples/coding_agent_rl/swe.py` | SWE 任务层：metadata 解析、workspace 准备、diff capture、干净 sandbox eval |
| `/home/chenyizhou/slime/examples/coding_agent_rl/run_qwen36_35b_a3b_swe_8nodes.sh` | 8 节点 Qwen3.6-35B-A3B SWE RL recipe，包含模型并行、SGLang、E2B、Claude Code 参数 |

### 2.2 通用 Agent 层

| 文件 | 职责 |
| --- | --- |
| `/home/chenyizhou/slime/slime/agent/adapters/common.py` | `BaseAdapter`：session lifecycle、HTTP turn pipeline、SGLang `/generate` 调用、logprob 捕获 |
| `/home/chenyizhou/slime/slime/agent/adapters/anthropic.py` | Anthropic Messages 协议适配，Claude Code 使用 |
| `/home/chenyizhou/slime/slime/agent/adapters/openai.py` | OpenAI Chat Completions 协议适配，Codex CLI 使用 |
| `/home/chenyizhou/slime/slime/agent/harness/common.py` | `BaseHarness`、detached launch + marker polling、Node/npm CLI 安装 |
| `/home/chenyizhou/slime/slime/agent/harness/claude_code.py` | Claude Code harness：配置、环境变量、CLI 调用 |
| `/home/chenyizhou/slime/slime/agent/harness/codex.py` | Codex harness：TOML 配置、OpenAI-compatible endpoint、CLI 调用 |
| `/home/chenyizhou/slime/slime/agent/sandbox.py` | `Sandbox` Protocol 和 `E2BSandbox` 实现 |
| `/home/chenyizhou/slime/slime/agent/trajectory.py` | `TrajectoryManager`，把多轮消息树线性化成 `Sample` |
| `/home/chenyizhou/slime/slime/agent/parsing.py` | reasoning/tool-call parser 包装，优先用 SGLang parser，XML fallback |
| `/home/chenyizhou/slime/slime/agent/aiohttp_threaded.py` | 在后台线程启动 adapter HTTP app |

### 2.3 测试层

| 文件 | 覆盖点 |
| --- | --- |
| `/home/chenyizhou/slime/tests/test_agent/test_agent_rollout_cpu.py` | CPU-only 端到端：generate -> sandbox fake -> harness -> adapter HTTP -> fake SGLang -> trajectory -> eval |
| `/home/chenyizhou/slime/tests/test_agent/test_trajectory_manager_branching.py` | TrajectoryManager 分叉、漂移、rewrite merge、reward split、dedup 的矩阵测试 |
| `/home/chenyizhou/slime/tests/test_agent/test_adapters.py` | Anthropic/OpenAI adapter 协议转换、streaming、tool call、token capture |
| `/home/chenyizhou/slime/tests/test_agent/test_harness.py` | harness 配置、detached launch、marker polling、agent 用户创建 |
| `/home/chenyizhou/slime/tests/test_agent/_fakes.py` | FakeTokenizer、FakeSGLangServer、FakeSandbox |

---

## 3. 全链路数据流

### 3.1 大图

```text
slime RolloutManager
  -> custom_generate: examples.coding_agent_rl.generate.generate(args, sample, sampling_params)
       -> parse dataset metadata
       -> open adapter session
       -> boot work sandbox
       -> prepare SWE workspace
       -> run coding harness in sandbox
            Claude Code CLI / Codex CLI
              -> calls adapter HTTP endpoint
                   AnthropicAdapter / OpenAIAdapter
                     -> render messages with model chat template
                     -> SGLang /generate(input_ids, return_logprob=True)
                     -> parse reasoning/tool calls
                     -> record TurnRecord into TrajectoryManager
              -> agent reads/edits/runs commands in sandbox
       -> capture git diff
       -> boot clean eval sandbox
       -> apply diff
       -> run swepro/eval_cmd/f2p_script
       -> reward
       -> adapter.finish_session()
            TrajectoryManager linearizes root-to-leaf chains
            -> list[Sample]
              tokens: prompt + response trajectory
              loss_mask: model output 1, context/tool/unsafe drift 0
              rollout_log_probs: exact SGLang sampled token logprobs
              reward: split across emitted samples
              rollout_id: shared for same base trajectory
  -> RolloutManager flattens and converts to train_data
  -> DP schedule groups by rollout_id
  -> Megatron loss uses masks/logprobs/rewards
```

### 3.2 为什么是 string-in, token-out

Coding agent CLI 是 string/message 世界：

- Claude Code 发 Anthropic Messages 请求。
- Codex 发 OpenAI Chat Completions 请求。
- 工具返回、shell output、file content 都是字符串。
- CLI 可能重放历史、压缩历史、派发 sub-agent。

RL 训练是 token 世界：

- policy gradient 优化的是模型实际采样 token 的 logprob。
- PPO/GSPO/TIS 需要生成时的 behavior logprob。
- `loss_mask` 必须精确区分哪些 token 是模型 action。

所以 adapter 的核心 contract 是：

```text
进入 adapter 的是 message/string history；
离开 adapter 进入训练的必须是模型当时实际采样的 token ids + logprobs。
```

不能把 agent 最后的完整文本重新 tokenize 后训练。这样会破坏 token/logprob 对齐。

---

## 4. `generate.py` 深挖

`generate.py` 是 slime 的 custom generate 入口：

```bash
--custom-generate-function-path examples.coding_agent_rl.generate.generate
```

它做四件事：

1. 启动 agent sandbox 并安装 harness CLI。
2. 准备 SWE workspace。
3. 运行 coding agent，得到 diff。
4. 在干净 eval sandbox 中打分，最后导出训练 samples。

### 4.1 `SweConfig`

`SweConfig.from_env()` 从环境变量读取：

- `ADAPTER_PUBLIC_HOST`
- `ADAPTER_BIND_HOST`
- `ADAPTER_PORT`
- `SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS`
- `SWE_AGENT_TIME_BUDGET_SEC`
- `SWE_EVAL_TIMEOUT_SEC`
- `SWE_ROLLOUT_GUARD_SEC`
- `SWE_BOOT_CONCURRENCY`
- `SWE_BOOT_RETRIES`

设计点：

- `SWE_AGENT_TIME_BUDGET_SEC` 是 agent CLI 的运行预算。
- `SWE_EVAL_TIMEOUT_SEC` 是 eval sandbox 的测试预算。
- `SWE_ROLLOUT_GUARD_SEC` 是外层总保险，默认约等于 agent + eval + 180 秒。
- `SWE_BOOT_CONCURRENCY` 限制同时启动 sandbox 数，避免 E2B/gateway 长尾抖动。
- `SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS` 会传给 `TrajectoryManager`，影响短 rewrite 是 merge 还是 fork。

面试可讲：

> Agent rollout 的耗时不只在模型 decode，也在 sandbox boot、CLI install、repo setup、tool call 和 eval。`generate.py` 把 agent/eval/outer guard 拆开，是为了知道到底哪里超时。

### 4.2 `boot_agent_sandbox`

`boot_agent_sandbox(image, instance_id)`：

- 用 `E2BSandbox(image)` 启动 sandbox。
- 受 `_BOOT_SEM` 限制并发。
- 调 `ClaudeCodeHarness().install_cli(cand)` 安装 Node 22 + Claude Code CLI。
- 对 transient boot/install failure 重试。
- 离开上下文时 kill sandbox。

这里的边界：

- work sandbox 只用于 agent 实际工作。
- eval sandbox 在 `swe.evaluate()` 内另起，保持干净。
- agent 工作 sandbox 的测试结果不是最终 reward，最终 reward 必须来自 clean eval。

### 4.3 `_AdapterService`

`_AdapterService` 是 `SingletonMeta`：

- 每个 Ray worker 进程内只启动一个 adapter HTTP server。
- 加载 tokenizer。
- 读取 `rollout_max_context_len`。
- 获取 `sglang_tool_call_parser` 和 `sglang_reasoning_parser`。
- 创建 `AnthropicAdapter`。
- 用 `run_app_in_thread()` 在后台线程启动 aiohttp app。

关键细节：

- `ADAPTER_PUBLIC_HOST` 必须是 sandbox 能访问的地址，不能是 `127.0.0.1`。
- adapter 对外暴露的是 Anthropic-compatible endpoint。
- adapter 内部连接的是 SGLang router：`http://{args.sglang_router_ip}:{args.sglang_router_port}`。
- `handler_cancellation=True` 很重要：客户端断开会取消 handler，adapter 会向 SGLang `/abort_request` 释放 KV/slot。

面试可讲：

> 这个 adapter 是“host side reverse bridge”：sandbox 里的 CLI 以为自己在调 Anthropic API，其实请求回到训练节点的 adapter；adapter 再把消息转成 SGLang `/generate` 的 token 请求。

### 4.4 `generate()` 主流程

伪代码：

```python
state = _AdapterService(args)
md = swe.get_metadata(base_sample)
if missing image/workdir:
    return aborted sample

session_id = base_sample.session_id = _session_id(...)
state.adapter.open_session(session_id, sampling_defaults=sampling_params, max_context_tokens=...)

try:
    with rollout_guard:
        with boot_agent_sandbox(md["image"]):
            prepare_workspace(sb, workdir, md)
            agent_exit_code = ClaudeCodeHarness().run(...)
            diff_text = git_diff(sb, workdir)

        reward, applied_cleanly = swe.evaluate(clean sandbox, diff_text, tests...)

        samples = adapter.finish_session(session_id, base_sample, reward)
        attach agent_exit_code metadata
        return samples
except timeout/exception:
    return aborted sample
finally:
    adapter.finish_session(session_id)  # idempotent cleanup
```

注意点：

- `generate()` 总是返回 `list[Sample]`，即使 abort 也是 `[sample]`。
- abort sample 设置 `status=ABORTED`、`reward=0.0`、`loss_mask=[0]`。
- 如果 agent 从未联系 adapter，`finish_session()` 返回空，标记 `adapter_session_empty`。
- `agent_exit_code != 0` 不一定意味着没有 reward，因为 diff 可能已经产生并 eval 通过；代码只是记录 warning。
- `applied_cleanly` 目前只进日志，没有写入 sample metadata，这是一个可改进点。

---

## 5. `swe.py` 深挖

`swe.py` 是任务层，故意不依赖 Claude Code。它只回答三件事：

1. 数据集行怎么变成 SWE metadata。
2. sandbox workspace 怎么准备。
3. diff 怎么评估。

### 5.1 Dataset schema

标准 JSONL：

```json
{
  "prompt": "fallback problem statement",
  "label": "instance_id",
  "metadata": {
    "image": "your-registry/swe-image:tag",
    "workdir": "/workspace/repo",
    "problem_statement": "issue body",
    "swepro": {},
    "eval_cmd": "pytest -x tests/...",
    "pre_commands": ["git checkout ..."]
  }
}
```

也兼容 sweb-style：

```json
{
  "metadata": {
    "remote_env_info": {
      "instance_id": "...",
      "image_url": "...",
      "workdir": "...",
      "f2p_script": "...",
      "pre_commands": "..."
    }
  }
}
```

命令行要配：

```bash
--input-key prompt
--label-key label
--metadata-key metadata
```

### 5.2 `get_metadata`

输出统一 md：

- `instance_id`
- `image`
- `workdir`
- `problem_statement`
- `swepro`
- `eval_cmd`
- `f2p_script`
- `pre_commands`

面试可能问：

> 为什么 label 只在是短字符串时才当 instance_id fallback？

回答：

> label 可能是复杂对象或长答案，不一定适合作为 ID。代码只接受短字符串，避免把巨大 label 塞进日志/session id。

### 5.3 `prepare_workspace`

流程：

- `ensure_agent_user(sb, workdir)`：创建非 root 的 `agent` 用户并 chown workdir。
- 如果有 `swepro.before_repo_set_cmd`，运行 setup。
- 如果有 `pre_commands`，运行它们。
- 写入 `PROBLEM_STATEMENT.md`。

注意：

- `pre_commands` 同时用于 work sandbox 和 eval sandbox。
- 注释里说得很清楚：很多 sweb 数据会用 `git checkout <base_sha> -f`，如果 work sandbox 不跑 pre_commands，模型看到的 repo base 和 eval base 不一致，会导致 diff apply 大量失败。

### 5.4 `git_diff`

命令：

```bash
cd {workdir} && git add -N . && git diff -- . ':(exclude)PROBLEM_STATEMENT.md' ':(exclude).harness/'
```

设计点：

- `git add -N .` 让新文件也进入 diff。
- 排除 `PROBLEM_STATEMENT.md`，防止模型改题目。
- 排除 `.harness/`，防止 harness 日志污染 diff。

可以改进：

- 统计 diff 文件数、增删行、是否触碰 tests。
- 对 tests 修改做强过滤或 reward penalty。
- 对超大 diff 做截断/拒绝，避免 eval sandbox apply 慢或污染训练。

### 5.5 `evaluate`

关键保证：

```text
eval sandbox 是 fresh sandbox；
只把模型产生的 diff apply 进去；
然后运行 dataset 提供的 verifier。
```

评分路径优先级：

1. `swepro`
2. `eval_cmd`
3. `f2p_script`

如果没有任何 grader：

- reward = 0
- applied_cleanly = True

如果 diff apply 失败：

- reward = 0
- applied_cleanly = False

`_apply_diff` 尝试三种方式：

1. `git apply --3way --whitespace=nowarn`
2. `git apply --whitespace=nowarn`
3. `patch -p1 --no-backup-if-mismatch`

面试可讲：

> Clean eval sandbox 是防 reward hacking 的核心。Agent 可以在 work sandbox 里跑测试帮助自己，但它不能通过修改测试、污染环境状态或依赖工作目录副作用来拿 reward；最终 reward 只看 clean repo + diff + hidden/selected tests。

---

## 6. Adapter 深挖：Anthropic/OpenAI 协议到 SGLang token

### 6.1 `BaseAdapter`

`BaseAdapter` 负责所有协议共享逻辑：

- `open_session(sid, sampling_defaults, max_context_tokens)`
- `shutdown_session(sid)`
- `finish_session(sid, base_sample, reward)`
- `_run_turn(request)`
- `call_sglang_generate(...)`
- `TrajectoryManager.record_turn(...)`

一个 turn 的 pipeline：

```text
HTTP request body
  -> protocol-specific preprocess
  -> resolve session id
  -> turn cap check
  -> translate wire messages to tokenizer chat-template messages
  -> tokenizer.apply_chat_template(..., add_generation_prompt=True)
  -> call SGLang /generate with input_ids and return_logprob=True
  -> decode output_ids
  -> parse reasoning/tool_calls
  -> build protocol reply
  -> record TurnRecord into TrajectoryManager
  -> return Anthropic/OpenAI response
```

### 6.2 `call_sglang_generate`

请求：

```json
{
  "rid": "...",
  "input_ids": [...],
  "sampling_params": {
    "skip_special_tokens": false,
    "spaces_between_special_tokens": false,
    "no_stop_trim": true,
    "max_new_tokens": ...
  },
  "return_logprob": true
}
```

关键点：

- `skip_special_tokens=False`：训练要保留真实 token 序列。
- `no_stop_trim=True`：不要把 stop token 悄悄裁掉，避免 response/token/logprob 不一致。
- `return_logprob=True`：后续 PPO/GSPO/TIS 才能用 behavior logprob。
- `X-SMG-Routing-Key=session_id`：让同一个 agent session 尽量路由到同一 SGLang engine，提高 prefix cache 命中。
- 如果 request cancel/timeout，会调用 `/abort_request` 释放 SGLang 资源。

### 6.3 AnthropicAdapter

Claude Code 走 Anthropic Messages。

它做的协议转换：

- `system` -> chat-template system。
- Anthropic `user` text block -> user。
- Anthropic `tool_result` -> chat-template tool。
- Anthropic assistant `thinking` -> `reasoning_content`。
- Anthropic assistant `tool_use` -> chat-template `tool_calls`。
- 去掉 wire-only tool id，因为下轮 replay 时 id 会变，保留会导致 message dict equality 失败。

输出：

- 如果有 reasoning，返回 Anthropic `thinking` block。
- 如果有 text，返回 text block。
- 如果有 tool call，返回 `tool_use` block，并生成新的 wire id。
- stop reason 映射为 `tool_use`、`max_tokens`、`end_turn`。

### 6.4 OpenAIAdapter

Codex CLI 走 OpenAI Chat Completions。

关键差异：

- `developer` role 被归一成 `system`。
- `tool_calls[].function.arguments` 从 JSON string 转成 dict，便于 chat template 和 tree matching。
- wire-only `tool_call_id` 和 `tool_calls[].id` 被丢弃。
- response 里 tool call 的 arguments 要重新 JSON dump，因为 OpenAI wire spec 需要字符串。
- manager 只保留第一个 tool call，因为某些 client 会丢 parallel tool calls，保留多个会导致 replay 不稳定。

面试可讲：

> Adapter 里很多看似“协议小修补”的逻辑，其实都是为了 TrajectoryManager 的 tree matching：下一个请求 replay 的 assistant/tool history 必须和上一个模型输出的 manager_message 在 dict equality 上一致，否则每一轮都会无意义 fork。

### 6.5 parser

`slime.agent.parsing.parse_model_output()`：

- reasoning parser：调用 SGLang `ReasoningParser`，例如 `qwen3`。
- tool parser：调用 SGLang `FunctionCallParser`，例如 `qwen3_coder`。
- fallback：Anthropic-style XML tool call。

launcher 里必须配：

```bash
--sglang-tool-call-parser qwen3_coder
--sglang-reasoning-parser qwen3
```

否则 Claude Code/Codex 看到的 tool call 结构可能错，agent 无法执行工具。

---

## 7. `TrajectoryManager` 是核心中的核心

如果只能深挖一个文件，就深挖：

```text
/home/chenyizhou/slime/slime/agent/trajectory.py
```

### 7.1 为什么需要它

Agent CLI 每轮发来的 prompt 是字符串消息历史，而训练要的是 token trajectory。

困难点：

- 后续 prompt 会 replay 前面 assistant output。
- tool observation 是字符串，不是模型 action。
- CLI 可能 auto-compact，把历史压缩改写。
- Claude Code 可能 dispatch sub-agent，产生多条分支。
- 某些客户端会改写 assistant message 格式、空格、tool id。
- tokenizer/chat template 重新渲染后不一定和原始 sampled tokens 完全一致。

`TrajectoryManager` 用一个 per-session message tree 解决这些问题。

### 7.2 两层逻辑：message tree + token linearization

第一层：`record_turn`

- 输入：
  - `sid`
  - `TurnRecord(prompt_ids, output_ids, output_log_probs, finish_reason)`
  - `prompt_messages`
  - `response_message`
- 按 message dict equality 找挂载点。
- 如果 prompt history 分叉，就在 tree 上新建 branch。
- 每个生成的 assistant response 是一个 node，带 `turn`。

第二层：`get_trajectory`

- 遍历 root-to-leaf chains。
- 把每条 chain 线性化成一个或多个 `_SampleBuilder`。
- 处理 token drift：clean、realign、fork。
- 对 shared response 做 dedup，避免同一生成 turn 在多个 leaf 里重复训练。
- 输出 `list[Sample]`。

### 7.3 `TurnRecord`

```python
TurnRecord(
    prompt_ids=[...],
    output_ids=[...],
    finish_reason="stop/tool_calls/length",
    output_log_probs=[...],
)
```

这是 adapter 和 trajectory manager 的契约。

关键不变量：

- `len(output_log_probs) == len(output_ids)`，除非没有 logprobs。
- `prompt_ids` 是当前 turn 输入给 SGLang 的真实 token。
- `output_ids` 是 SGLang 实际 sampled token。

### 7.4 Drift 分类

`_SampleBuilder.classify_token_drift()` 把当前 builder 已有 token 和新 turn 的 `prompt_ids` 做 common prefix。

三种结果：

#### CLEAN

```text
held tokens 是 prompt_ids 的精确前缀
```

处理：

- 只追加 prompt tail，`loss_mask=0`。
- 追加当前 output，`loss_mask=1`。

#### REALIGN

```text
drift 发生在最近一次 response span 内，
并且当前 output 长度小于 fork_threshold
```

处理：

- 从 drift 所在 response 起，用新 prompt 覆盖旧 tail。
- 覆盖部分全部 `loss_mask=0`。
- 再追加当前 output。

含义：

> 如果客户端轻微改写了上一轮 assistant 文本，例如空格或格式差异，我们保留上下文连续性，但不再对被改写的那段 backprop。

#### FORK

```text
drift 太早、太大，或者发生在更早 turn
```

处理：

- 关闭当前 builder。
- 新开一个 builder。
- 当前 turn 成为新 segment 的第一 turn。

含义：

> 当 token provenance 已经无法安全证明时，宁可切段，不要把错配 token 当作模型 sampled token 训练。

### 7.5 rewrite merge

`_try_merge_assistant_rewrite()` 处理短 assistant rewrite。

场景：

- 客户端 replay 了上一轮 assistant，但内容被轻微重写。
- 如果直接 fork，会产生一个 dead-end leaf，可能训练一段其实已经被客户端放弃的 response。

策略：

- 只有当 mount point 下恰好一个 assistant child。
- 这个 child 是 leaf。
- 它是 generated。
- response 长度小于 `fork_threshold`。

才把它 demote 成 routing-only，停止训练它。

长 response 不 merge，选择 fork，因为长 abandoned response 可能仍有真实训练信号。

### 7.6 cross-leaf dedup

`MessageNode.response_trained` 确保：

- 同一个 generated assistant turn 如果出现在多个 sibling leaf 的 shared prefix 中。
- 第一个 leaf 训练它。
- 后续 leaf 只把它作为 context 重新发出，`loss_mask=0`。

否则 sub-agent/fan-out 会把 shared prefix 重复训练多次。

### 7.7 输出 Sample

`_SampleBuilder.to_sample()` 输出：

- `tokens`：包含 first-turn prompt + response trajectory。
- `response_length`：`len(loss_mask) - leading_prompt_len`。
- `loss_mask`：去掉 first prompt，只覆盖 response region。
- `rollout_log_probs`：与 response region 对齐。
- `rollout_id`：优先用 `base_sample.rollout_id`，否则用 `base_sample.index`。
- `reward`：先是 0，之后 `get_trajectory()` 统一赋值。

当前 reward 策略：

```python
per_sample_reward = reward / len(samples)
```

也就是说，一个 agent session 如果导出 K 个 `Sample`，每个 sample 拿 `reward / K`。同时它们共享 `rollout_id`，后续 slime 的 DP schedule/loss reducer 会把它们视作同一 rollout。

需要注意：

- reward split 和 per-rollout denominator 是两个层面的设计。
- 当前代码选择“reward scalar 守恒”：所有 sample reward 之和等于原始 reward。
- 这是否是最优 credit assignment，需要实验；源码里也明确留下了 custom reward/credit assignment 的扩展空间。

这是非常好的改进点。

---

## 8. Harness 与 Sandbox

### 8.1 `Sandbox` Protocol

接口很小：

```python
async with Sandbox(...) as sb:
    await sb.exec(cmd, user="agent", timeout=..., check=False)
    await sb.write_file(path, content_or_host_path, user="agent")
    text = await sb.read_file(path, user="agent")
```

当前实现：

- `E2BSandbox`
- 使用 `AsyncSandbox.create(timeout=..., metadata={image_key: image})`
- RPC 有 transient retry。
- `read_file` 失败返回空字符串。

环境变量：

- `SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY`
- legacy: `SWE_SANDBOX_IMAGE_METADATA_KEY`
- `SLIME_AGENT_SANDBOX_LIFETIME_SEC`
- `SLIME_AGENT_SANDBOX_RPC_RETRIES`

### 8.2 `ensure_agent_user`

创建非 root 用户：

```bash
id agent || useradd -m -s /bin/bash agent
chown -R agent:agent /home/agent {workdir}
git config --system --add safe.directory '*'
```

意义：

- agent 不以 root 运行。
- workdir 可写。
- git diff 不受 safe.directory 限制。

### 8.3 `BaseHarness`

抽象三步：

```python
install_cli(sb)
write_config(sb, ctx)
launch_and_wait(sb, ctx, prompt, time_budget_sec)
```

`run()` 的顺序：

```text
ensure_agent_user
write_config
launch_and_wait
```

### 8.4 `run_command`

这是处理长命令的关键：

- 不用长连接一直等 CLI。
- 写 `.harness/run.sh`。
- `setsid ... &` detached 启动。
- stdout/stderr tee 到 `.harness/trajectory.jsonl`。
- CLI 退出码写到 `.harness/done`。
- host 每 5 秒短 RPC poll marker。
- 超时返回 `EXIT_TIME_BUDGET_EXCEEDED = -1`。

为什么这样做：

> 很多 sandbox gateway 对长连接、空闲连接、stdout stream 有限制。detached + marker polling 更稳，短 RPC 也能保持 sandbox 活性。

### 8.5 Claude Code harness

环境变量：

- `SLIME_AGENT_NODE_TARBALL`
- `SLIME_AGENT_CC_TARBALL`
- `SLIME_AGENT_CC_EXTRA_ARGS`
- `SLIME_AGENT_CC_EXTRA_ENVS`

配置：

- 预先写 `.claude.json` 和 `.claude/settings.json`。
- 设置 `hasCompletedOnboarding` 和 `bypassPermissionsModeAccepted`。

运行：

```bash
/usr/local/bin/claude -p '<prompt>' \
  --permission-mode bypassPermissions \
  --output-format stream-json \
  --include-partial-messages \
  --include-hook-events \
  --verbose
```

通过 env 把请求导向 adapter：

```bash
ANTHROPIC_BASE_URL=http://host:18001
ANTHROPIC_AUTH_TOKEN=<session_id>
ANTHROPIC_MODEL=slime-actor
```

### 8.6 Codex harness

虽然 `generate.py` 当前生产路径硬编码 Claude Code + AnthropicAdapter，通用层已经支持：

- `CodexHarness`
- `OpenAIAdapter`

Codex 特殊点：

- `base_url` 必须写在 `config.toml` 里。
- 通过 base64 round-trip 写 TOML，避免 shell quoting 问题。
- `OPENAI_API_KEY=session_id`，adapter 从 Bearer token 解析 sid。

这给拓展提供了直接落点：让 `generate.py` 支持 harness/adapter 可配置。

---

## 9. Launcher recipe 解读

脚本：

```text
/home/chenyizhou/slime/examples/coding_agent_rl/run_qwen36_35b_a3b_swe_8nodes.sh
```

### 9.1 模型与并行

默认：

```bash
TP_SIZE=2
PP_SIZE=1
CP_SIZE=8
EP_SIZE=8
ETP_SIZE=1
```

rollout:

```bash
ROLLOUT_TP_SIZE=8
ROLLOUT_DP_SIZE=8
ROLLOUT_EP_SIZE=8
ROLLOUT_MEM_UTILIZATION=0.75
```

解释：

- 这是 Qwen3.6-35B-A3B MoE 级别 recipe。
- agent trajectory 很长，`CP_SIZE=8` 配合 96k context。
- rollout 侧用 SGLang DP attention、EP、DP LM head。
- 训练和 rollout colocate。

### 9.2 Context length

```bash
MAX_CONTEXT_LEN=96000
MAX_GEN_LEN=32768
```

rollout args：

```bash
--rollout-max-context-len 96000
--rollout-max-response-len 32768
```

README 里解释：

- `rollout-max-response-len` 是每个 SGLang `/generate` turn 的 `max_new_tokens` 上限。
- `rollout-max-context-len` 是多轮 prompt+response 总预算，adapter 每 turn 会 clamp 剩余可生成 token。
- trajectory export 不因为长度再丢 segment。

### 9.3 Rollout batch

```bash
--rollout-batch-size 8
--n-samples-per-prompt 8
--global-batch-size 64
--micro-batch-size 1
```

注意：

- `generate()` 本身可能 fan-out，多 leaf/sample。
- 实际 flatten 后 sample 数可能大于 `8 * 8`。
- DP schedule 按 `rollout_id` 组织，而不是简单按 flat sample 数。

### 9.4 算法

```bash
--advantage-estimator gspo
--kl-loss-coef 0.00
--kl-coef 0.00
--entropy-coef 0.00
--eps-clip 1e-4
--eps-clip-high 2e-4
```

可以理解为：

- 用 GSPO 风格策略优化。
- 初始 recipe 里 KL/entropy 都关掉。
- clip 非常小，说明希望更新极保守。

面试可以说：

> Coding-agent reward 非常稀疏且昂贵，trajectory 长、off-policy 风险和 reward hacking 风险都高，所以 recipe 里采用很保守的 clip。是否加 KL、过程 reward、格式 reward，是后续实验轴。

### 9.5 性能相关

```bash
--max-tokens-per-gpu $((MAX_CONTEXT_LEN / CP_SIZE))
--log-probs-chunk-size 1024
--use-dynamic-batch-size
--recompute-granularity full
--optimizer-cpu-offload
```

关键点：

- 长单条 trajectory 会让 logprob forward OOM，所以 chunk logprobs。
- `max-tokens-per-gpu` 按 CP rank 的 slice 设置。
- dynamic batch 应对不同 coding task 长度差异。
- optimizer CPU offload 降显存。

### 9.6 SGLang parser

```bash
--sglang-tool-call-parser qwen3_coder
--sglang-reasoning-parser qwen3
```

这不是可有可无的参数。没有它，adapter 可能无法把 Qwen 输出解析成工具调用，Claude Code/Codex 就拿不到可执行 action。

### 9.7 Claude Code policy

脚本里：

- 注册 read-only `investigator` sub-agent。
- 禁用 `WebFetch`/`WebSearch`。
- 禁用 slash commands。
- `autoCompactEnabled=true`
- `autoCompactWindow=80000`

设计点：

- sub-agent 提供 fan-out 训练样本。
- 禁外网减少数据泄漏和 reward hacking。
- auto compact window 小于 96k context，让 CLI 在训练 cap 前压缩。

---

## 10. 分阶段复现计划

这里的“复现”不要一上来跑 8 节点训练。正确路线是先复现 correctness，再复现 sandbox，再复现 rollout，再复现训练。

### 阶段 A：源码与轻量测试环境

目标：

- 不需要 GPU。
- 不需要 E2B。
- 不需要 SGLang。
- 先跑 CPU-only tests，证明核心 agent dataflow 正确。

建议在远端 slime 环境里用 `uv`，因为当前非交互 SSH 下默认 `python3` 可能很旧，而 `uv` 已在 `~/.local/bin/uv`：

```bash
cd /home/chenyizhou/slime

# 轻量 CPU-only correctness 环境；不安装完整训练栈
~/.local/bin/uv venv --python 3.11 .venv-agent-cpu
~/.local/bin/uv pip install --python .venv-agent-cpu/bin/python pytest aiohttp httpx pillow
```

这些 agent tests 不执行 tensor 运算，但 `slime.utils.types` 顶层 import 了 `torch`。如果只想快速验证 adapter/harness/trajectory/generate CPU 路径，可以在测试进程里注入一个最小 `torch` stub；如果要验证训练侧或任何 tensor 逻辑，则必须安装真实 torch。

```bash
export RUN_DIR=/home/chenyizhou/slime/runs/agent_cpu_tests_manual
mkdir -p "$RUN_DIR"
cat > "$RUN_DIR/pytest.ini" <<'INI'
[pytest]
addopts =
testpaths =
INI

.venv-agent-cpu/bin/python - <<'PY'
import os, sys, types
from pathlib import Path

repo = Path("/home/chenyizhou/slime")
sys.path.insert(0, str(repo))

torch = types.ModuleType("torch")
class Tensor: pass
class dtype: pass
class Size(tuple): pass
torch.Tensor = Tensor
torch.dtype = dtype
torch.Size = Size
sys.modules.setdefault("torch", torch)

import pytest
files = [
    "tests/test_agent/test_harness.py",
    "tests/test_agent/test_trajectory_manager_branching.py",
    "tests/test_agent/test_adapters.py",
    "tests/test_agent/test_agent_rollout_cpu.py",
]
pytest_ini = str(Path(os.environ["RUN_DIR"]) / "pytest.ini")
raise SystemExit(pytest.main(["-q", "-c", pytest_ini, *files]))
PY
```

实际远端验证记录：

```text
repo: /home/chenyizhou/slime
commit: 243773c
venv: .venv-agent-cpu, Python 3.11.13 via uv
log: /home/chenyizhou/slime/runs/agent_cpu_tests_20260617_133322/run.log
result: 62 passed, 1 skipped in 0.50s
```

可视化 trajectory：

```bash
TRAJ_DUMP=1 PYTHONPATH=. python tests/test_agent/test_trajectory_manager_branching.py
```

你要观察：

- `loss_mask` 与 `response_length` 对齐。
- drift case 怎么 fork/realign。
- reward 如何按 sample 数 split。
- cross-leaf dedup 如何避免重复训练 shared assistant response。

### 阶段 B：读 debug dump，不训练

目标：

- 用 slime 的 rollout debug 能力看 `Sample`。
- 先确认 custom generate 的返回 shape、metadata、loss mask、reward。

推荐启用：

```bash
--debug-rollout-only
--save-debug-rollout-data /tmp/cagent_rollout_{rollout_id}.pt
```

检查 dump：

```python
import torch
d = torch.load("/tmp/cagent_rollout_0.pt", map_location="cpu")
samples = d["samples"]
for s in samples[:3]:
    print(s.status, s.reward, s.response_length, sum(s.loss_mask), s.metadata)
    print(len(s.loss_mask), len(s.rollout_log_probs), s.response_length)
```

重点检查：

- abort_reason 是否多。
- `agent_exit_code` 分布。
- reward 是否全 0。
- `response_length` 是否极端大。
- `sum(loss_mask)` 是否为 0。
- fan-out K 是否比预期大。

### 阶段 C：E2B/Sandbox smoke

目标：

- 只验证 sandbox boot、CLI install、adapter 可达、eval clean sandbox。
- 不关心训练效果。

准备：

- 一个最小 sandbox image，里面有一个 git repo。
- `metadata.image` 指向 image。
- `metadata.workdir` 指向 repo。
- `metadata.eval_cmd` 可以先设成 `true`，验证空 diff reward=1。
- `SLIME_AGENT_NODE_TARBALL` 和 `SLIME_AGENT_CC_TARBALL` 路径有效。
- `ADAPTER_PUBLIC_HOST` 必须是 sandbox 可访问的 host IP。

最小 JSONL：

```json
{"prompt":"Fix the issue in PROBLEM_STATEMENT.md.","label":"toy-1","metadata":{"instance_id":"toy-1","image":"your-image","workdir":"/workspace/repo","problem_statement":"Make the existing test pass.","eval_cmd":"true"}}
```

如果 agent 从未打回 adapter，会得到：

```text
abort_reason = adapter_session_empty
```

如果 sandbox 无法打回 host adapter，优先查：

- `ADAPTER_PUBLIC_HOST`
- `ADAPTER_PORT`
- sandbox 到 host 的网络策略
- `no_proxy`/`NO_PROXY`
- adapter `/healthz`

### 阶段 D：真实 SGLang + 单机/小批 rollout

目标：

- 真实模型生成工具调用。
- 真实 adapter -> SGLang -> Claude Code loop。
- 保存 debug rollout data。

建议先不要训练，使用极小 rollout：

```bash
--rollout-batch-size 1
--n-samples-per-prompt 1
--num-rollout 1
--save-debug-rollout-data "${RUN_ROOT}/rollout_dumps/rollout_{rollout_id}.pt"
```

观察：

- Claude Code 是否能正常调用工具。
- SGLang parser 是否正确解析 tool calls。
- `trajectory.jsonl` 是否有 agent 过程。
- `git_diff` 是否非空。
- eval sandbox 是否能 apply diff。
- reward 分布。

### 阶段 E：完整 8 节点 recipe

目标：

- 跑 `run_qwen36_35b_a3b_swe_8nodes.sh`。
- 观察系统瓶颈和训练指标。

启动前必须确认：

- `HOSTFILE` 存在，每行 worker IP，root passwordless ssh。
- `HF_CHECKPOINT`、`REF_MODEL_PATH` 是匹配的 Qwen3.6-35B-A3B。
- `PROMPT_DATA` schema 符合 README。
- E2B gateway/image routing 可用。
- Node/Claude Code tarball 可被 Ray workers 访问。
- `ADAPTER_PUBLIC_HOST` 是 sandbox 可达 IP。
- `sglang_tool_call_parser=qwen3_coder` 和 model 匹配。

运行：

```bash
cd /home/chenyizhou/slime
tmux new -s cagent-rl

export HF_CHECKPOINT=/path/to/Qwen3.6-35B-A3B
export REF_MODEL_PATH=/path/to/Qwen3.6-35B-A3B_torch_dist
export PROMPT_DATA=/path/to/swe_train.jsonl
export SLIME_AGENT_NODE_TARBALL=/path/to/node-v22.x-linux-x64.tar.xz
export SLIME_AGENT_CC_TARBALL=/path/to/anthropic-ai-claude-code-local-linux-x64.tgz
export E2B_API_KEY=e2b_xxx
export SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY=image

bash examples/coding_agent_rl/run_qwen36_35b_a3b_swe_8nodes.sh
```

观察文件：

```text
runs/${EXP_TAG}_${STAMP}/run.log
runs/${EXP_TAG}_${STAMP}/rollout_dumps/rollout_*.pt
```

### 阶段 F：端到端指标分析

至少记录：

- rollout wall time。
- sandbox boot time。
- agent CLI time。
- eval time。
- samples per rollout。
- segments per session。
- trained tokens per sample。
- abort rate。
- diff apply success rate。
- reward/pass rate。
- SGLang queue/prefill/decode time。
- actor train TFLOPS/token/s。
- `wait_time_ratio`。

判断瓶颈：

- 如果 `adapter_session_empty` 多：CLI 没打回 adapter，查 harness/env/network。
- 如果 reward 全 0：先查 diff apply，再查 eval command，再查 problem/workdir/pre_commands。
- 如果 rollout 很慢：分 sandbox boot、agent、SGLang decode、eval。
- 如果 train 等 rollout：考虑 fully async、提高 rollout engine、降低 eval timeout、缓存 CLI 安装。
- 如果 SGLang prefill 高：看 prefix cache、routing key、context length、auto compaction。

---

## 11. 实际改进拓展 Plan

下面不是空泛方向，而是有明确文件落点、测试落点和面试价值的改进计划。

### 改进 1：让 `generate.py` 支持 harness/adapter 可配置

现状：

- 生产 `generate.py` 硬编码 `ClaudeCodeHarness + AnthropicAdapter`。
- 但通用层已经有 `CodexHarness + OpenAIAdapter`。
- 测试 `test_codex_openai_rollout_closes_loop` 只是 hand-wired，没接到 `generate.py`。

目标：

- 通过环境变量选择：
  - `SLIME_AGENT_HARNESS=claude_code|codex`
  - `SLIME_AGENT_ADAPTER=anthropic|openai`
- 默认保持 Claude Code。
- Codex path 也能走完整 `generate()`。

修改点：

- `examples/coding_agent_rl/generate.py`
  - `SweConfig` 增加 `harness_name` 和 `adapter_name`。
  - `_AdapterService` 根据 adapter_name 创建 `AnthropicAdapter` 或 `OpenAIAdapter`。
  - `boot_agent_sandbox` 根据 harness_name 安装对应 CLI。
  - `generate()` 根据 harness_name 调用对应 harness。
- `examples/coding_agent_rl/README.md`
  - 增加 Codex env vars：`SLIME_AGENT_CODEX_TARBALL`、`SLIME_AGENT_CODEX_EXTRA_ARGS`。
- `tests/test_agent/test_agent_rollout_cpu.py`
  - 把 hand-wired Codex 测试升级成 `generate()` 配置测试。
- `tests/test_agent/test_harness.py`
  - 已有 Codex harness 测试可复用。

验收：

```bash
PYTHONPATH=. python -m pytest -q \
  tests/test_agent/test_agent_rollout_cpu.py \
  tests/test_agent/test_harness.py
```

面试价值：

> 这说明你能看懂重构后的插件边界，不只是跑 Claude Code，而是把 harness/adapter 真正产品化成可配置组件。

### 改进 2：可插拔 credit assignment / reward split

现状：

- `TrajectoryManager.get_trajectory()` 把 session reward 均分到所有 emitted samples。
- 代码里有 custom reward/credit assignment 的扩展空间。
- 这对 sub-agent/fork/compaction 后的 credit assignment 很关键。

目标：

支持几种策略：

- `uniform_sample`：当前行为，`reward / K`。
- `uniform_rollout`：每个 sample 都给完整 reward，但依赖 per-rollout reducer 控制权重。
- `trained_token_weighted`：按每个 sample 的 trained token 数分配 reward。
- `last_sample_only`：只给最后一个 leaf/segment reward，前面为 0。
- `custom_function_path`：用户自定义。

修改点：

- `slime/agent/trajectory.py`
  - `get_trajectory(..., reward_split="uniform_sample", reward_split_fn=None)`。
  - 或让 `TrajectoryManager` 初始化时持有 splitter。
- `slime/agent/adapters/common.py`
  - `finish_session()` 接收 reward split 配置并传给 manager。
- `examples/coding_agent_rl/generate.py`
  - 从 env 读 `SLIME_AGENT_REWARD_SPLIT`。
- `tests/test_agent/test_trajectory_manager_branching.py`
  - 新增 token-weighted、last-only 的 golden tests。

验收：

```bash
PYTHONPATH=. python -m pytest -q tests/test_agent/test_trajectory_manager_branching.py
```

面试价值：

> Coding-agent RL 的 reward 是 outcome-level，但训练样本可能被 tree/fork 拆成多个 `Sample`。credit assignment 是 Agent 后训练核心问题之一，能改这里说明你不只会接环境，也懂 reward 到 token loss 的路径。

### 改进 3：把 `applied_cleanly`、diff stats、eval path 写入 metadata 和 rollout metrics

现状：

- `generate.py` 日志里有 reward、applied、agent_exit_code、elapsed、segments。
- 但 sample metadata 只写了 `agent_exit_code`。
- Debug dump 后不容易批量分析 diff apply failure、eval path、diff 大小。

目标：

每个 emitted sample metadata 增加：

- `instance_id`
- `agent_exit_code`
- `applied_cleanly`
- `eval_path`: `swepro|eval_cmd|f2p_script|none`
- `diff_chars`
- `diff_files`
- `diff_added_lines`
- `diff_deleted_lines`
- `segments`
- `session_elapsed_sec`
- `sandbox_image`

修改点：

- `examples/coding_agent_rl/swe.py`
  - 新增 `diff_stats(diff_text)`。
  - `evaluate()` 返回 richer object 或额外 metadata。
- `examples/coding_agent_rl/generate.py`
  - 把 metadata 合并到每个 sample。
- `tests/test_agent/test_agent_rollout_cpu.py`
  - 断言 metadata 字段存在。

验收：

```bash
PYTHONPATH=. python -m pytest -q tests/test_agent/test_agent_rollout_cpu.py
```

面试价值：

> Agent RL debug 最大难点是 reward 失败原因不可见。把 eval/diff/apply 结构化进 metadata，可以把“训练不涨”拆成模型没改、diff apply 失败、测试失败、sandbox 失败、adapter 空轨迹等不同问题。

### 改进 4：新增 DockerSandbox，降低本地复现门槛

现状：

- 生产后端是 E2B-compatible sandbox。
- 这对大规模训练合理，但学习/调试门槛较高。

目标：

- 实现 `DockerSandbox` 符合 `Sandbox` Protocol。
- 支持：
  - `exec`
  - `write_file`
  - `read_file`
  - `async with`
  - timeout
  - volume/workdir
  - 可选 network disabled
- 通过 env 选择：
  - `SLIME_AGENT_SANDBOX_BACKEND=e2b|docker`

修改点：

- 新建 `slime/agent/sandbox_docker.py` 或放入 `sandbox.py`。
- `examples/coding_agent_rl/generate.py` 和 `swe.py` 用 factory 创建 sandbox，不直接写死 `E2BSandbox`。
- `tests/test_agent/test_agent_rollout_cpu.py` 保持 FakeSandbox。
- 新增 integration test 可选跑 Docker。

验收：

```bash
PYTHONPATH=. python -m pytest -q tests/test_agent/test_harness.py tests/test_agent/test_agent_rollout_cpu.py
```

面试价值：

> 这是把研究 recipe 变成可复现工程的改进。小团队/学生没有 E2B cluster，也可以用 Docker 做 correctness 和小规模数据验证。

### 改进 5：过程奖励与 safety filter

现状：

- reward 是 binary outcome：eval pass -> 1，否则 0。
- Coding task reward 极稀疏。

目标：

加可选 reward components：

- diff apply clean：`+0.1`
- 不修改 tests：`+0.1`
- 有非空 diff：`+0.05`
- eval pass：`+1.0`
- invalid/no trajectory：`0`
- 超时 penalty。
- 过大 diff penalty。

但最终训练 reward 是否用这些 components 要可配置，避免污染 benchmark pass rate。

修改点：

- `swe.py` 返回 reward components。
- `generate.py` 根据 `SWE_REWARD_MODE=binary|shaped` 选择 reward。
- metadata 永远记录 components。
- tests 覆盖 binary/shaped 两种。

面试价值：

> 稀疏 outcome reward 对 coding agent 学习效率很差。过程 reward 可以提高信号密度，但也可能诱导 reward hacking，所以要把 components 记录下来，并保持 binary eval 作为主指标。

### 改进 6：Agent rollout tracing

目标：

- 给以下环节打 trace span：
  - sandbox boot
  - install CLI
  - prepare workspace
  - agent run
  - SGLang generate turn
  - git diff
  - eval boot
  - apply diff
  - run tests
  - finish_session

修改点：

- 参考 slime `trace_span`/`trace_event` 机制。
- 在 `generate.py`、`swe.py`、adapter turn pipeline 加 spans。

面试价值：

> Agent RL 的慢点经常不在模型，而在环境、sandbox、eval 或网络。没有 trace 只能看端到端 latency，无法优化 MFU/吞吐。

### 改进 7：接入 fully async rollout

现状：

- coding tasks 长短差异大。
- 同步 rollout 会被慢 task 拖住。

目标：

- 让 coding-agent RL 可以使用 `slime.rollout.fully_async_rollout`。
- 后台维持 in-flight coding tasks pool。
- 训练消费完成的 trajectories。

挑战：

- off-policy 更明显。
- weight version 记录要清晰。
- 超长任务可能跨多个 update。
- reward/debug dump 顺序不再直观。

面试价值：

> Coding-agent RL 是 fully async rollout 的典型场景，因为环境耗时 heavy-tail。这个方向体现你能把算法和系统吞吐结合起来，而不是只谈 loss。

---

## 12. 面试深挖题与回答

### Q1：这个 example 和普通 RLHF/GRPO 最大区别是什么？

普通 RLHF/GRPO 多是单轮 prompt -> response -> reward。Coding-agent RL 是真实 agent loop：

- agent CLI 多轮调用模型。
- 模型输出可能是 tool call，不是最终答案。
- 工具观察、shell output、file content 都进入上下文但不参与 loss。
- 最终 reward 来自 diff 在 clean sandbox 上的测试结果。
- 一个 session 可能因 sub-agent/compaction 导出多个 `Sample`。

所以难点从“算 reward”变成“保持 trajectory token provenance 正确”。

### Q2：为什么不能把 Claude Code 的 trajectory.jsonl 直接拿来训练？

因为 trajectory.jsonl 是 string/log 世界，不一定保留：

- 每个模型输出的 exact token ids。
- 每个 token 的 rollout logprob。
- chat template 渲染后的 token 边界。
- 哪些 token 是模型生成，哪些是工具/环境观察。
- compaction/rewrite 后哪些 token 来源还可信。

训练要用 SGLang 实际 sampled token，否则 PPO/GSPO/TIS 的 ratio/KL 都错。

### Q3：为什么需要 clean eval sandbox？

防止 test-cheating 和环境污染。

如果在 agent 工作 sandbox 里直接看测试结果：

- 模型可能改测试。
- 模型可能留下缓存/状态。
- 模型可能依赖运行顺序副作用。
- 模型可能污染 evaluator。

clean eval sandbox 只接收 diff，然后跑 grader，reward 更可信。

### Q4：为什么 tool observation 的 loss mask 是 0？

工具观察不是模型 action。训练它会让模型学习预测环境输出，而不是优化自己的决策。

但是 observation 必须进上下文，否则后续 action 条件不完整。

### Q5：sub-agent 和 auto-compaction 怎么处理？

TrajectoryManager 用 message tree：

- 每个 turn 的 prompt_messages 按 dict equality 找路径。
- 如果 sub-agent 或 compaction 让 prompt history 分叉，就产生新 leaf。
- 每个 leaf 线性化成 sample。
- shared generated response 只训练一次，其他 leaf 作为 context mask=0。

### Q6：token drift 怎么处理？

三种：

- clean：继续追加。
- realign：短 drift 在最近 response 内，覆盖 drift tail 并 mask=0。
- fork：漂移太早/太大/在更早 response 内，新开 segment。

原则：

> 能证明是模型 sampled token 的才训练；证明不了就作为 context 或切段。

### Q7：为什么 adapter 要丢掉 tool call id？

Anthropic/OpenAI 的 tool call id 是 wire-only correlation id。下一轮 client replay history 时 id 可能变。

如果把 id 放进 manager_message，TrajectoryManager 用 dict equality 匹配历史时会失败，导致每轮都 fork。

### Q8：`X-SMG-Routing-Key` 有什么用？

同一个 session 的多轮请求带同一个 routing key，SGLang router 可以把它们路由到同一 engine。

好处：

- prefix cache 命中率更高。
- 多轮 agent 长上下文 prefill 成本下降。

### Q9：为什么 `skip_special_tokens=False`、`no_stop_trim=True`？

训练需要 exact sampled token sequence。

如果 decode/generate 时跳过 special token 或 trim stop token：

- `output_ids` 与 response 文本不一致。
- stop token 的 logprob 可能丢失。
- loss mask 和 logprob 对齐可能破坏。

### Q10：reward split 和 `rollout_id` 有什么关系？

一个 agent session 可能导出多个 samples。

当前代码：

- reward 按 sample 数均分。
- samples 共享 `rollout_id`。

后续 RolloutManager 会计算 `rollout_mask_sums`，loss reducer 可按 rollout 级别计权。

这是一个设计选择，不是唯一答案。可以拓展成 token-weighted、last-turn-only、custom credit assignment。

### Q11：如果 rollout reward 全 0，怎么排查？

顺序：

1. 看 `abort_reason`。
2. 看 agent 是否打回 adapter。
3. 看 `agent_exit_code`。
4. 看 diff 是否为空。
5. 看 diff 是否 apply cleanly。
6. 看 eval path 是 swepro/eval_cmd/f2p。
7. 看 pre_commands 是否让 work/eval base 一致。
8. 看测试超时还是失败。
9. 看是否修改 tests 被过滤。
10. 看 parser 是否导致工具调用失败。

### Q12：怎么提高这个 recipe 的吞吐？

先拆瓶颈：

- sandbox boot 慢：调 `SWE_BOOT_CONCURRENCY`，预热 image，缓存 CLI 安装，或改 Docker/long-lived sandbox pool。
- agent 慢：缩短 prompt、限制 tool、降低 time budget、要求 investigator 子代理。
- SGLang prefill 慢：routing key、prefix cache、PD、context 压缩。
- SGLang decode 慢：更多 rollout engine、spec decoding、FP8 KV、合适 TP/DP。
- eval 慢：减少 selected tests、并发 eval、缓存依赖、设置 timeout。
- train 等 rollout：fully async rollout。

### Q13：怎么把 Codex 接进这个 example？

通用层已经有：

- `CodexHarness`
- `OpenAIAdapter`

要做的是让 `generate.py` 不硬编码 Claude Code：

- env 选择 harness。
- env 选择 adapter。
- boot 时安装对应 CLI。
- run 时调用对应 harness。
- adapter URL 对应 `/v1` 或 Anthropic root。
- 增加 CPU-only tests。

### Q14：怎么防 reward hacking？

- clean eval sandbox。
- hidden tests / selected tests。
- diff 排除 problem statement 和 harness logs。
- 禁止 WebFetch/WebSearch。
- 不允许修改 tests，或修改 tests 给 penalty。
- eval assets 不暴露给 agent。
- 记录 high-reward bad cases。
- 对异常 diff/异常工具调用/超短轨迹做审计。

---

## 13. 你可以怎样把它讲成自己的项目准备

不要说“我已经大规模训过这个 recipe”。可以诚实地说：

> 我重点读了 slime 最新 coding-agent RL 重构，把它拆成 example task layer、sandbox、harness、adapter、trajectory manager 和 Megatron 训练接口。  
>  
> 我认为这个 example 最核心的问题是 string/message agent 环境如何变成 token-level RL training sample。新版通过 `AnthropicAdapter/OpenAIAdapter` 捕获 SGLang 真实 output ids/logprobs，通过 `TrajectoryManager` 处理 sub-agent、compaction、token drift 和 fan-out，通过 clean eval sandbox 生成 reward。  
>  
> 我准备复现时会先跑 CPU-only agent tests 验证 correctness，再做 E2B smoke，再跑 debug rollout dump，最后才上训练。改进上我会优先做 harness/adapter 可配置、credit assignment 可插拔、metadata/trace 增强和 DockerSandbox，这些都能直接降低复现门槛或提高 Agent RL 训练信号质量。

---

## 14. 面试前复习清单

你应该能不用看文档回答：

- `generate.py` 的四阶段 orchestrator 是什么。
- `swe.py` 为什么要分 work sandbox 和 eval sandbox。
- dataset schema 中 `image/workdir/problem_statement/swepro/eval_cmd/f2p_script/pre_commands` 的作用。
- `ADAPTER_PUBLIC_HOST` 为什么不能是 `127.0.0.1`。
- Claude Code 怎么通过 Anthropic env vars 打到 adapter。
- Codex 怎么通过 OpenAI-compatible endpoint 打到 adapter。
- Adapter 为什么必须保存 exact output ids/logprobs。
- `TrajectoryManager` 的 message tree 和 token linearization 分别解决什么。
- clean/realign/fork 三种 drift case。
- rewrite merge 为什么只吸收短 assistant rewrite。
- cross-leaf dedup 为什么必要。
- reward split 当前怎么做，有什么可改。
- `rollout_id` 对 fan-out sample 的意义。
- SGLang tool/reasoning parser 配错会发生什么。
- clean eval sandbox 如何防 reward hacking。
- CPU-only tests 覆盖了哪些真实代码路径。
- 复现顺序为什么应该从 CPU-only correctness 开始，而不是直接 8 节点训练。
- 吞吐优化如何拆 sandbox、agent、SGLang prefill/decode、eval、train wait。

---

## 15. 最值得你真的动手的三件事

### 第一件：跑通 CPU-only tests 并读 TRAJ_DUMP

这是最小但最有价值的 hands-on。

你会真正看到：

- 一条 agent chain 怎么变成 Sample。
- 分叉后 reward 怎么 split。
- drift 发生时哪些 token 被 mask。
- shared response 为什么不重复训练。

### 第二件：做 metadata/trace 增强

这是最容易合入、最实用的改进：

- 不改变算法。
- 不动训练核心。
- 直接提升 debug 能力。
- 面试也好讲。

### 第三件：做 harness selector 或 credit assignment

二选一：

- 如果你想偏系统工程：做 harness selector，把 Codex 接到 `generate.py`。
- 如果你想偏算法/Agent RL：做 credit assignment，把 reward split 做成可配置。

这两件都能体现你理解了新版重构的边界。

---

## 16. 这份 example 和前一份 slime 总文档的连接

前一份文档里说过的 slime 总体主线：

```text
custom_generate -> Sample -> RolloutManager -> train_data -> DP schedule -> Megatron loss
```

在 coding-agent RL 里具体化为：

```text
Claude/Codex CLI
  -> Adapter 捕获 TurnRecord
  -> TrajectoryManager 输出 Sample
  -> Sample.loss_mask 只训练模型 action token
  -> Sample.rollout_log_probs 支持 ratio/TIS
  -> Sample.rollout_id 处理 fan-out
  -> reward 来自 clean eval sandbox
  -> Megatron GSPO/PPO loss
```

你可以把它当成 Agent 后训练的标准案例来讲：

- 普通 RLHF：回答文本是 action。
- Tool RL：tool call 是 action，observation 是 context。
- Coding-agent RL：整个 edit/diff/test loop 是 environment，diff eval 是 reward，token provenance 是训练正确性的生命线。

---

## 17. 最后一句判断

这个 example 的面试价值很高，因为它把 Agent 后训练里最容易停留在概念层的东西落到了代码：

- sandbox isolation
- tool protocol adapter
- exact token/logprob capture
- multi-turn loss mask
- compaction/sub-agent fan-out
- clean verifier reward
- long-context rollout performance
- reward credit assignment
- debug/test strategy

如果你能把这些讲清楚，再拿一个小改进做 hands-on，就已经不是“没用过 slime”的状态，而是“没大规模训过，但能读懂和改动真实 Agent RL 系统”的状态。

---

## 18. 2026-06-17 远端 Qwen3-0.6B hands-on 记录

这一节记录一次实际上 GPU 跑通的 coding-agent RL 最小闭环。它不是伪装成官方 SWE recipe 全量复现，而是在当前远端机器缺少 E2B/Claude Code/SGLang/Ray 完整环境的情况下，用本地隔离 workspace 替代外部 sandbox/harness，保留 agent RL 最核心的数据闭环：

```text
Qwen3-0.6B 真实生成 action
  -> local harness 应用 patch
  -> clean pytest verifier 给 reward
  -> TrajectoryManager 记录 TurnRecord
  -> 输出 slime Sample / RolloutFnTrainOutput dump
```

### 18.1 环境与依赖边界

远端路径：

```text
/home/chenyizhou/slime
/home/chenyizhou/models/Qwen3-0.6B
```

已经验证：

- `.venv` 可用，Python 3.11.13。
- `torch 2.12.0+cu130` 可见 8 张 RTX 3090。
- Qwen3-0.6B 能在 3090 上加载和生成。
- `tests/test_agent` 之前已通过。
- 8 卡 NCCL all-reduce 之前已通过。

当前缺口：

- `.venv` 缺 `ray`、`sglang`、`sglang_router`，所以不能直接启动 slime 原生 `train.py` 的 SGLang rollout/Ray job。
- Megatron-LM 源码在 `/home/chenyizhou/OpenSearch-VL/RL/Megatron-LM`，但训练侧还缺 `mbridge`、`torch_memory_saver`、`six`、`einops` 等依赖；这不是补一个小包就能稳定跑 full Megatron step 的状态，更接近需要官方 Docker/build_conda 环境。
- 没找到官方 coding-agent SWE recipe 需要的 SWE 数据、E2B sandbox、Node tarball、Claude Code CLI tarball。

所以本次 hands-on 的目标被明确限定为：真实模型 + 真实 GPU + 真实 verifier reward + 真实 slime agent trajectory/sample contract，而不是官方 E2B/Claude 全链路。

### 18.2 实验设计

脚本：

```text
/home/chenyizhou/slime/runs/coding_agent_qwen06_20260617_223340/run_qwen06_coding_agent_flow.py
```

本地文档工作区也保留了一份同名脚本。

实验构造 8 个小型 Python repo，每个 repo 有一个 bug、两个候选 patch、一个 pytest verifier。Qwen3-0.6B 的任务不是自由生成 diff，而是在 A/B patch 中选择。这是为了让 0.6B 模型也能稳定完成闭环，同时仍然保留 coding-agent RL 的关键动作：

- 读 issue/file/test。
- 选择 patch action。
- harness 修改 workspace。
- clean verifier 运行 pytest。
- 通过测试给 reward=1，否则 reward=0。
- 把模型生成 token、logprobs、loss mask、reward 写成 slime `Sample`。

启动时使用 8 个 worker，每个 worker 绑定一张 GPU：

```bash
cd /home/chenyizhou/slime
. /home/chenyizhou/slime/.venv/bin/activate
export PYTHONPATH=/home/chenyizhou/slime
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python /home/chenyizhou/slime/runs/coding_agent_qwen06_20260617_223340/run_qwen06_coding_agent_flow.py \
  --run-dir /home/chenyizhou/slime/runs/coding_agent_qwen06_20260617_223340 \
  --model-path /home/chenyizhou/models/Qwen3-0.6B \
  --python-bin /home/chenyizhou/slime/.venv/bin/python \
  --num-workers 8 \
  --max-new-tokens 64 \
  --temperature 0.0
```

一个实际踩坑：

- 第一次运行脚本放在 `runs/` 目录下，Python 的 `sys.path[0]` 是脚本目录，不是 repo 根目录，因此 `import slime` 失败。
- 修复方式是显式设置 `PYTHONPATH=/home/chenyizhou/slime`。

另一个小点：

- Qwen3 默认会先输出 `<think>`。
- 对这种短 action 选择，应在 `apply_chat_template(..., enable_thinking=False)` 中关闭 thinking，否则模型会先解释，action parsing 和 token loss 都会变脏。

### 18.3 产物

主 run 目录：

```text
/home/chenyizhou/slime/runs/coding_agent_qwen06_20260617_223340
```

关键文件：

```text
run.log
summary.json
records.json
prompt_data.jsonl
samples.jsonl
rollout_fn_train_output.pt
rollout_debug_dump.pt
rollout_debug_0.pt
gpu_snapshot_before.txt
gpu_snapshot_after.txt
task_records/*.json
repo_templates/*
workspaces/*
```

其中：

- `prompt_data.jsonl` 是任务/prompt 数据。
- `records.json` 保存每个 task 的模型输出、选择、测试输出、reward。
- `samples.jsonl` 是可读版 `Sample.to_dict()`。
- `rollout_fn_train_output.pt` 是 `RolloutFnTrainOutput(samples=..., metrics=...)`。
- `rollout_debug_dump.pt` 是便于离线检查的 dict dump。
- `rollout_debug_0.pt` 是按 slime `--load-debug-rollout-data` 读取逻辑整理的平铺格式：`{"rollout_id": 0, "samples": [sample_dict, ...]}`。

### 18.4 结果

运行日志 summary：

```json
{
  "num_tasks": 8,
  "num_samples": 8,
  "pass_rate": 1.0,
  "total_response_tokens": 16,
  "avg_response_tokens": 2.0,
  "wall_time_sec": 11.067,
  "num_gpus_visible": 8,
  "num_workers": 8
}
```

每个任务的模型选择和 verifier 结果：

| task | gpu | action | reward | verifier |
| --- | ---: | --- | ---: | --- |
| add_returns_sum | 0 | A | 1.0 | 2 passed |
| is_even_uses_zero_remainder | 1 | A | 1.0 | 2 passed |
| reverse_text_slices_backwards | 2 | A | 1.0 | 1 passed |
| first_item_not_last | 3 | A | 1.0 | 1 passed |
| clamp_bounds_value | 4 | A | 1.0 | 2 passed |
| safe_divide_divides | 5 | A | 1.0 | 2 passed |
| max_of_two_not_min | 6 | A | 1.0 | 1 passed |
| factorial_product | 7 | A | 1.0 | 2 passed |

`.pt` dump 反序列化检查通过：

```text
rollout_type RolloutFnTrainOutput
metrics {"avg_response_tokens": 2.0, "num_gpus_visible": 8, "num_samples": 8, "num_tasks": 8, "num_workers": 8, "pass_rate": 1.0, "total_response_tokens": 16, "wall_time_sec": 11.067}
PT_CHECK_OK
```

额外生成的 debug replay 形状检查也通过：

```text
WROTE /home/chenyizhou/slime/runs/coding_agent_qwen06_20260617_223340/rollout_debug_0.pt
DEBUG_REPLAY_SHAPE_OK 8
```

逐 sample 检查过：

- `reward == 1.0`
- `status == completed`
- `response_length == len(loss_mask) == len(rollout_log_probs)`
- `sum(loss_mask) == response_length`
- 每个 sample 都带 task/action/workspace/eval metadata

所有 workspace 重新跑 pytest 也通过。

### 18.5 这个实验证明了什么

它证明的不是“大规模 SWE RL 训练已经完成”，而是以下关键链路已经实际跑通：

- 真实 Qwen3-0.6B 在 8 张 3090 上并行生成。
- agent action 能被 harness 解释为代码修改。
- verifier reward 来自 clean workspace 的真实测试结果。
- `TrajectoryManager.record_turn()` 能用真实 prompt/output token ids 和 output logprobs 生成训练轨迹。
- 输出的 `Sample` 满足 slime rollout/train 数据面的基本字段约束。
- `rollout_debug_0.pt` 已整理成 `--load-debug-rollout-data` 兼容的 sample dump 形状，可作为后续 train-only 调试输入。

面试里可以这样说：

> 我没有把缺失的 E2B/Claude/SGLang 环境假装成已经配好，而是先把 coding-agent RL 最小闭环在真实 GPU 和真实 Qwen3-0.6B 上跑通。这个闭环覆盖了 action generation、environment mutation、test reward、TurnRecord、loss mask、rollout logprob、Sample dump。官方 recipe 的差别主要在 harness/sandbox 和 rollout engine：本地 pytest harness 要替换成 E2B + Claude/Codex harness，HF generate 要替换成 SGLang router，debug dump 再接 Megatron train-only。

### 18.6 距离官方 SWE recipe 还差什么

要把这次 local harness 升级到官方 `examples/coding_agent_rl` recipe，需要补齐：

1. SGLang/Ray 运行环境

   - 安装或使用 slime Docker。
   - 确认 `ray`、`sglang`、`sglang_router` 能 import。
   - 能启动 `--debug-rollout-only`。

2. Megatron 训练侧

   - 补齐 Megatron 依赖。
   - 用 `scripts/models/qwen3-0.6B.sh` 的 `MODEL_ARGS` 转换 HF checkpoint 到 `torch_dist`。
   - 用 `--load-debug-rollout-data` 跑 train-only 一步，先不碰 SGLang。

3. 官方 coding-agent 外部环境

   - 准备 SWE-style JSONL 数据。
   - 准备 sandbox image/workdir/eval metadata。
   - 准备 `SLIME_AGENT_NODE_TARBALL`。
   - 准备 `SLIME_AGENT_CC_TARBALL` 或改造成 Codex harness。
   - 配置 `ADAPTER_PUBLIC_HOST`，确保 sandbox 能打回 adapter。

4. 把 local harness 改造成正式 extension

   - 把候选 patch 选择改成自由 diff 生成或 tool call。
   - 记录完整 command transcript。
   - reward 不只 pass/fail，可以拆成 compile/test/hidden-test/style 多维。
   - 给长 trajectory 做 credit assignment，而不是所有响应 token 同一个 terminal reward。

### 18.7 下一步最实际的升级路线

优先级建议：

1. 安装/使用官方 Docker，先跑 `tests/test_qwen2.5_0.5B_debug_rollout_then_train.py` 这种两段式 debug。
2. 用 Qwen3-0.6B 做一次 `debug_rollout_only -> save_debug_rollout_data -> load_debug_rollout_data`，确认 Megatron train-only 一步。
3. 把这次 local harness 改成 `--custom-generate-function-path` 插件，而不是独立脚本。
4. 把 A/B patch 改成模型输出 unified diff，再用 `git apply --check` 和 pytest verifier 评估。
5. 最后再接 E2B/Claude/Codex，解决 sandbox 网络、CLI 包、token provenance 和 long-tail rollout 并发。

这一组升级能从“小闭环已经真实跑过”自然推进到“官方 recipe 可复现并能改造”，面试叙事很完整。
