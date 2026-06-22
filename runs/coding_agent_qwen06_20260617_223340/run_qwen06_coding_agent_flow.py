#!/usr/bin/env python3
"""Run a tiny local coding-agent RL rollout with Qwen3-0.6B.

This is intentionally self-contained: it uses a local clean-room workspace as
the environment/harness, while exercising slime's Sample and TrajectoryManager
contracts with real GPU model generation, test-based rewards, token ids,
loss masks, and rollout logprobs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from slime.agent.trajectory import TrajectoryManager, TurnRecord
from slime.rollout.base_types import RolloutFnTrainOutput
from slime.utils.types import Sample


TASKS: list[dict[str, Any]] = [
    {
        "task_id": "add_returns_sum",
        "issue": "calculator.add(a, b) should return the sum, but it currently subtracts.",
        "module": "calculator.py",
        "before": "def add(a, b):\n    return a - b\n",
        "patch_a": "def add(a, b):\n    return a + b\n",
        "patch_b": "def add(a, b):\n    return a - b\n",
        "correct": "A",
        "tests": "from calculator import add\n\n\ndef test_add_positive_numbers():\n    assert add(2, 3) == 5\n\n\ndef test_add_negative_number():\n    assert add(-2, 5) == 3\n",
    },
    {
        "task_id": "is_even_uses_zero_remainder",
        "issue": "numbers.is_even(n) should be True for even integers, but the current predicate is inverted.",
        "module": "numbers_local.py",
        "before": "def is_even(n):\n    return n % 2 == 1\n",
        "patch_a": "def is_even(n):\n    return n % 2 == 0\n",
        "patch_b": "def is_even(n):\n    return n % 2 == 1\n",
        "correct": "A",
        "tests": "from numbers_local import is_even\n\n\ndef test_even_values():\n    assert is_even(0)\n    assert is_even(12)\n\n\ndef test_odd_values():\n    assert not is_even(7)\n",
    },
    {
        "task_id": "reverse_text_slices_backwards",
        "issue": "strings.reverse_text(text) should return the input string reversed.",
        "module": "strings.py",
        "before": "def reverse_text(text):\n    return text\n",
        "patch_a": "def reverse_text(text):\n    return text[::-1]\n",
        "patch_b": "def reverse_text(text):\n    return text\n",
        "correct": "A",
        "tests": "from strings import reverse_text\n\n\ndef test_reverse_text():\n    assert reverse_text('abc') == 'cba'\n    assert reverse_text('racecar') == 'racecar'\n",
    },
    {
        "task_id": "first_item_not_last",
        "issue": "lists.first_item(seq) should return the first element, but it currently returns the last.",
        "module": "lists.py",
        "before": "def first_item(seq):\n    return seq[-1]\n",
        "patch_a": "def first_item(seq):\n    return seq[0]\n",
        "patch_b": "def first_item(seq):\n    return seq[-1]\n",
        "correct": "A",
        "tests": "from lists import first_item\n\n\ndef test_first_item():\n    assert first_item([10, 20, 30]) == 10\n    assert first_item(['x', 'y']) == 'x'\n",
    },
    {
        "task_id": "clamp_bounds_value",
        "issue": "math_utils.clamp(x, lo, hi) should bound x into [lo, hi], but it currently returns x unchanged.",
        "module": "math_utils.py",
        "before": "def clamp(x, lo, hi):\n    return x\n",
        "patch_a": "def clamp(x, lo, hi):\n    return min(max(x, lo), hi)\n",
        "patch_b": "def clamp(x, lo, hi):\n    return x\n",
        "correct": "A",
        "tests": "from math_utils import clamp\n\n\ndef test_clamp_inside():\n    assert clamp(5, 1, 10) == 5\n\n\ndef test_clamp_low_high():\n    assert clamp(-3, 1, 10) == 1\n    assert clamp(30, 1, 10) == 10\n",
    },
    {
        "task_id": "safe_divide_divides",
        "issue": "arithmetic.safe_divide(a, b) should divide a by b and return None on zero division.",
        "module": "arithmetic.py",
        "before": "def safe_divide(a, b):\n    if b == 0:\n        return None\n    return a * b\n",
        "patch_a": "def safe_divide(a, b):\n    if b == 0:\n        return None\n    return a / b\n",
        "patch_b": "def safe_divide(a, b):\n    if b == 0:\n        return None\n    return a * b\n",
        "correct": "A",
        "tests": "from arithmetic import safe_divide\n\n\ndef test_safe_divide_regular():\n    assert safe_divide(8, 2) == 4\n\n\ndef test_safe_divide_zero():\n    assert safe_divide(8, 0) is None\n",
    },
    {
        "task_id": "max_of_two_not_min",
        "issue": "compare.max_of_two(a, b) should return the larger value, but it currently returns the smaller one.",
        "module": "compare.py",
        "before": "def max_of_two(a, b):\n    return a if a < b else b\n",
        "patch_a": "def max_of_two(a, b):\n    return a if a >= b else b\n",
        "patch_b": "def max_of_two(a, b):\n    return a if a < b else b\n",
        "correct": "A",
        "tests": "from compare import max_of_two\n\n\ndef test_max_of_two():\n    assert max_of_two(1, 9) == 9\n    assert max_of_two(10, 3) == 10\n    assert max_of_two(4, 4) == 4\n",
    },
    {
        "task_id": "factorial_product",
        "issue": "factorial.factorial(n) should compute n!, but it currently returns n.",
        "module": "factorial.py",
        "before": "def factorial(n):\n    return n\n",
        "patch_a": "def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n",
        "patch_b": "def factorial(n):\n    return n\n",
        "correct": "A",
        "tests": "from factorial import factorial\n\n\ndef test_factorial_base_cases():\n    assert factorial(0) == 1\n    assert factorial(1) == 1\n\n\ndef test_factorial_values():\n    assert factorial(5) == 120\n",
    },
]


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_prompt(task: dict[str, Any]) -> list[dict[str, str]]:
    user = f"""You are a coding agent inside a small Python repository.
Pick the patch that makes the tests pass. Reply with exactly one letter: A or B.

Issue:
{task["issue"]}

File {task["module"]} currently contains:
```python
{task["before"]}```

Tests:
```python
{task["tests"]}```

Patch A:
```python
{task["patch_a"]}```

Patch B:
```python
{task["patch_b"]}```

Answer with A or B only."""
    return [
        {
            "role": "system",
            "content": "You are a careful coding agent. Do not explain. Choose the test-passing patch.",
        },
        {"role": "user", "content": user},
    ]


def parse_action(text: str) -> str | None:
    stripped = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL)
    match = re.search(r"\b([AB])\b", stripped)
    if match:
        return match.group(1)
    stripped = stripped.strip().upper()
    if stripped.startswith("A"):
        return "A"
    if stripped.startswith("B"):
        return "B"
    return None


def run_pytest(workspace: Path, python_bin: str, timeout: int) -> dict[str, Any]:
    started = time.time()
    proc = subprocess.run(
        [python_bin, "-m", "pytest", "-q"],
        cwd=workspace,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    return {
        "returncode": proc.returncode,
        "duration_sec": round(time.time() - started, 3),
        "output": proc.stdout[-4000:],
    }


def run_one(task: dict[str, Any], args_dict: dict[str, Any], worker_index: int) -> dict[str, Any]:
    run_dir = Path(args_dict["run_dir"])
    model_path = args_dict["model_path"]
    python_bin = args_dict["python_bin"]
    gpu_id = worker_index % max(1, args_dict["num_gpus"])
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(gpu_id)

    repo_template = run_dir / "repo_templates" / task["task_id"]
    workspace = run_dir / "workspaces" / task["task_id"]
    repo_template.mkdir(parents=True, exist_ok=True)
    atomic_write(repo_template / task["module"], task["before"])
    atomic_write(repo_template / "test_task.py", task["tests"])
    if workspace.exists():
        shutil.rmtree(workspace)
    shutil.copytree(repo_template, workspace)

    messages = build_prompt(task)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    ).to(device)
    model.eval()

    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    started_generate = time.time()
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args_dict["max_new_tokens"],
            do_sample=args_dict["temperature"] > 0,
            temperature=args_dict["temperature"] if args_dict["temperature"] > 0 else None,
            top_p=args_dict["top_p"],
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generate_sec = time.time() - started_generate
    output_ids_tensor = generated.sequences[0, input_ids.shape[1] :].detach().cpu()
    output_ids = output_ids_tensor.tolist()
    raw_response = tokenizer.decode(output_ids, skip_special_tokens=False)
    clean_response = tokenizer.decode(output_ids, skip_special_tokens=True)
    transition_scores = model.compute_transition_scores(
        generated.sequences,
        generated.scores,
        normalize_logits=True,
    )[0].detach().cpu().tolist()
    prompt_ids = input_ids[0].detach().cpu().tolist()

    action = parse_action(clean_response) or parse_action(raw_response)
    if action == "A":
        atomic_write(workspace / task["module"], task["patch_a"])
    elif action == "B":
        atomic_write(workspace / task["module"], task["patch_b"])

    eval_result = run_pytest(workspace, python_bin=python_bin, timeout=args_dict["pytest_timeout"])
    reward = 1.0 if eval_result["returncode"] == 0 else 0.0

    sid = task["task_id"]
    manager = TrajectoryManager()
    turn = TurnRecord(
        prompt_ids=prompt_ids,
        output_ids=output_ids,
        finish_reason="stop",
        output_log_probs=[float(x) for x in transition_scores],
    )
    manager.record_turn(
        sid,
        turn=turn,
        prompt_messages=messages,
        response_message={"role": "assistant", "content": clean_response},
        metadata={"task_id": task["task_id"], "action": action},
    )
    base_sample = Sample(
        index=worker_index,
        group_index=worker_index,
        rollout_id=worker_index,
        prompt=tokenizer.decode(prompt_ids, skip_special_tokens=False),
        label=task["correct"],
        metadata={
            "task_id": task["task_id"],
            "module": task["module"],
            "harness": "local_pytest",
            "gpu_id": gpu_id,
        },
    )
    samples = manager.get_trajectory(
        sid,
        base_sample=base_sample,
        reward=reward,
        extra_metadata={
            "task_id": task["task_id"],
            "action": action,
            "correct_action": task["correct"],
            "workspace": str(workspace),
            "eval_returncode": eval_result["returncode"],
            "generate_sec": round(generate_sec, 3),
        },
    )
    for sample in samples:
        sample.response = clean_response
        sample.train_metadata = {
            "loss_surface": "assistant_choice_tokens",
            "reward_source": "pytest_pass_fail",
        }

    if device.startswith("cuda"):
        peak_mem_gb = round(torch.cuda.max_memory_allocated(device) / 1024**3, 3)
    else:
        peak_mem_gb = 0.0

    task_record = {
        "task_id": task["task_id"],
        "gpu_id": gpu_id,
        "device": device,
        "module": task["module"],
        "correct_action": task["correct"],
        "model_action": action,
        "reward": reward,
        "raw_response": raw_response,
        "clean_response": clean_response,
        "prompt_tokens": len(prompt_ids),
        "response_tokens": len(output_ids),
        "loss_mask_tokens": [sum(s.loss_mask or []) for s in samples],
        "logprob_tokens": [len(s.rollout_log_probs or []) for s in samples],
        "generate_sec": round(generate_sec, 3),
        "peak_gpu_mem_gb": peak_mem_gb,
        "workspace": str(workspace),
        "eval": eval_result,
        "sample_dicts": [s.to_dict() for s in samples],
    }
    atomic_write(run_dir / "task_records" / f"{task['task_id']}.json", json.dumps(task_record, indent=2))
    return task_record


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/chenyizhou/models/Qwen3-0.6B")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--python-bin", required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--pytest-timeout", type=int, default=30)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    num_gpus = torch.cuda.device_count()
    args_dict = vars(args) | {"num_gpus": num_gpus}

    task_prompts = []
    for idx, task in enumerate(TASKS):
        task_prompts.append(
            {
                "index": idx,
                "task_id": task["task_id"],
                "prompt": build_prompt(task),
                "label": task["correct"],
                "metadata": {"module": task["module"], "recipe": "local_coding_agent_rl"},
            }
        )
    save_jsonl(run_dir / "prompt_data.jsonl", task_prompts)
    atomic_write(run_dir / "gpu_snapshot_before.txt", subprocess.getoutput("nvidia-smi"))

    started = time.time()
    records: list[dict[str, Any]] = []
    max_workers = min(args.num_workers, len(TASKS), max(1, num_gpus))
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn")) as executor:
        futures = {
            executor.submit(run_one, task, args_dict, idx): task["task_id"]
            for idx, task in enumerate(TASKS)
        }
        for future in as_completed(futures):
            records.append(future.result())
            last = records[-1]
            print(
                f"[done] {last['task_id']} gpu={last['gpu_id']} "
                f"action={last['model_action']} reward={last['reward']} "
                f"tokens={last['response_tokens']} gen_sec={last['generate_sec']}",
                flush=True,
            )

    records.sort(key=lambda r: r["task_id"])
    sample_groups: list[list[Sample]] = []
    for record in records:
        sample_groups.append([Sample.from_dict(d) for d in record["sample_dicts"]])

    rewards = [float(r["reward"]) for r in records]
    total_response_tokens = sum(int(r["response_tokens"]) for r in records)
    metrics = {
        "num_tasks": len(records),
        "num_samples": sum(len(g) for g in sample_groups),
        "pass_rate": sum(rewards) / len(rewards) if rewards else 0.0,
        "total_response_tokens": total_response_tokens,
        "avg_response_tokens": total_response_tokens / len(records) if records else 0.0,
        "wall_time_sec": round(time.time() - started, 3),
        "num_gpus_visible": num_gpus,
        "num_workers": max_workers,
    }
    rollout_output = RolloutFnTrainOutput(samples=sample_groups, metrics=metrics)
    torch.save(rollout_output, run_dir / "rollout_fn_train_output.pt")
    torch.save({"samples": sample_groups, "metrics": metrics, "records": records}, run_dir / "rollout_debug_dump.pt")
    save_jsonl(run_dir / "samples.jsonl", [s.to_dict() for group in sample_groups for s in group])
    atomic_write(run_dir / "records.json", json.dumps(records, indent=2))
    atomic_write(run_dir / "summary.json", json.dumps(metrics, indent=2))
    atomic_write(run_dir / "gpu_snapshot_after.txt", subprocess.getoutput("nvidia-smi"))

    print("SUMMARY", json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    main()
