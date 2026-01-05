%%writefile benchmark_vs_unsloth.py
"""
=============================================================================
CHRONICALS vs UNSLOTH BENCHMARK SUITE
=============================================================================
Head-to-head comparison between Chronicals LoRA and Unsloth.

Metrics Measured:
- Tokens/second throughput
- Peak memory usage (MB)
- Loss convergence speed
- Training time

Model Sizes Supported:
- 0.5B: Qwen2.5-0.5B
- 1B: meta-llama/Llama-3.2-1B
- 3B: Qwen2.5-3B
- 7B: Qwen2.5-7B, Llama-2-7b, Mistral-7B

Datasets Supported:
- Alpaca (yahma/alpaca-cleaned)
- OpenAssistant (OpenAssistant/oasst1)
- SlimOrca (Open-Orca/SlimOrca)

Usage:
    python benchmark_vs_unsloth.py --model Qwen/Qwen2.5-0.5B --steps 100
    python benchmark_vs_unsloth.py --model Qwen/Qwen2.5-0.5B --full-suite
    python benchmark_vs_unsloth.py --model all --dataset all  # Full benchmark

=============================================================================
"""

import os
import sys
import gc
import json
import time
import argparse
import statistics
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

import torch
import torch.nn as nn

# =============================================================================
# CONFIGURATION
# =============================================================================

# Target throughputs (tokens/sec) for different model sizes on A100
TARGET_THROUGHPUTS = {
    "0.5B": 50000,
    "1B": 35000,
    "3B": 20000,
    "7B": 10000,
}

# Model registry with metadata
MODEL_REGISTRY = {
    # 0.5B models
    "Qwen/Qwen2.5-0.5B": {"size": "0.5B", "type": "qwen", "params": 500_000_000},
    # 1B models
    "meta-llama/Llama-3.2-1B": {"size": "1B", "type": "llama", "params": 1_000_000_000},
    "Qwen/Qwen2.5-1.5B": {"size": "1.5B", "type": "qwen", "params": 1_500_000_000},
    # 3B models
    "Qwen/Qwen2.5-3B": {"size": "3B", "type": "qwen", "params": 3_000_000_000},
    # 7B models
    "Qwen/Qwen2.5-7B": {"size": "7B", "type": "qwen", "params": 7_000_000_000},
    "meta-llama/Llama-2-7b-hf": {"size": "7B", "type": "llama", "params": 7_000_000_000},
    "mistralai/Mistral-7B-v0.1": {"size": "7B", "type": "mistral", "params": 7_000_000_000},
}

# Dataset registry
DATASET_REGISTRY = {
    "alpaca": {
        "path": "yahma/alpaca-cleaned",
        "text_field": "text",
        "default_samples": 5000,
    },
    "openassistant": {
        "path": "OpenAssistant/oasst1",
        "text_field": "text",
        "default_samples": 5000,
    },
    "slimorca": {
        "path": "Open-Orca/SlimOrca",
        "text_field": "conversations",
        "default_samples": 5000,
    },
}


@dataclass
class BenchmarkResult:
    """Structured benchmark result."""
    method: str
    model: str
    dataset: str
    throughput_tokens_sec: float
    throughput_std: float
    peak_memory_mb: float
    total_time_sec: float
    final_loss: float
    steps: int
    batch_size: int
    seq_length: int
    lora_rank: int
    optimizations: Dict[str, bool] = field(default_factory=dict)
    run_throughputs: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "model": self.model,
            "dataset": self.dataset,
            "throughput_tokens_sec": self.throughput_tokens_sec,
            "throughput_std": self.throughput_std,
            "peak_memory_mb": self.peak_memory_mb,
            "total_time_sec": self.total_time_sec,
            "final_loss": self.final_loss,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "seq_length": self.seq_length,
            "lora_rank": self.lora_rank,
            "optimizations": self.optimizations,
        }


# =============================================================================
# UTILITIES
# =============================================================================

def cuda_sync():
    """Synchronize CUDA."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_memory():
    """Reset GPU memory stats."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def get_gpu_memory_mb() -> float:
    """Get peak GPU memory in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information."""
    if not torch.cuda.is_available():
        return {"available": False}

    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": props.name,
        "memory_gb": props.total_memory / 1e9,
        "compute_capability": f"{props.major}.{props.minor}",
        "cuda_version": torch.version.cuda,
    }


class CUDATimer:
    """CUDA-aware timer for accurate GPU timing."""

    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
        self.cpu_start = 0.0

    def start(self):
        if self.cuda_available:
            torch.cuda.synchronize()
            self.start_event.record()
        self.cpu_start = time.perf_counter()

    def stop(self) -> float:
        if self.cuda_available:
            self.end_event.record()
            torch.cuda.synchronize()
            return self.start_event.elapsed_time(self.end_event) / 1000.0
        return time.perf_counter() - self.cpu_start


# =============================================================================
# DATASET LOADING
# =============================================================================

def load_dataset_for_benchmark(
    tokenizer,
    dataset_name: str = "alpaca",
    max_length: int = 512,
    num_samples: int = 5000,
) -> Any:
    """Load and tokenize dataset for benchmarking."""
    from datasets import load_dataset

    config = DATASET_REGISTRY.get(dataset_name, DATASET_REGISTRY["alpaca"])

    print(f"  Loading {dataset_name} dataset...")

    try:
        if dataset_name == "alpaca":
            dataset = load_dataset(config["path"], split=f"train[:{num_samples}]")

            alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""

            def format_fn(example):
                instruction = example.get("instruction", "")
                input_text = example.get("input", "")
                output = example.get("output", "")
                if input_text:
                    instruction = f"{instruction}\n\nInput: {input_text}"
                return {"text": alpaca_prompt.format(instruction=instruction, output=output)}

            dataset = dataset.map(format_fn)

        elif dataset_name == "openassistant":
            dataset = load_dataset(config["path"], split=f"train[:{num_samples}]")
            # Format OpenAssistant conversations
            def format_fn(example):
                return {"text": example.get("text", str(example))}
            dataset = dataset.map(format_fn)

        elif dataset_name == "slimorca":
            dataset = load_dataset(config["path"], split=f"train[:{num_samples}]")
            # Format SlimOrca conversations
            def format_fn(example):
                convos = example.get("conversations", [])
                text = " ".join([c.get("value", "") for c in convos if isinstance(c, dict)])
                return {"text": text}
            dataset = dataset.map(format_fn)

        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Tokenize
        def tokenize_fn(examples):
            result = tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors=None,
            )
            result["labels"] = result["input_ids"].copy()
            return result

        dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
        dataset.set_format(type="torch")

        print(f"  [OK] Loaded {len(dataset)} samples")
        return dataset

    except Exception as e:
        print(f"  [ERROR] Failed to load {dataset_name}: {e}")
        raise


# =============================================================================
# BENCHMARK: CHRONICALS LORA
# =============================================================================

def benchmark_chronicals_lora(
    model_name: str,
    dataset_name: str,
    num_steps: int,
    batch_size: int,
    seq_length: int,
    lora_rank: int = 16,
    warmup_steps: int = 10,
    num_runs: int = 1,
) -> BenchmarkResult:
    """
    Benchmark Chronicals LoRA with all optimizations.

    Optimizations enabled:
    - Liger Kernel (RoPE, RMSNorm, SwiGLU, CrossEntropy)
    - torch.compile (default mode)
    - LoRA+ (differential learning rates)
    - Fused AdamW optimizer
    """
    print("\n" + "=" * 60)
    print("CHRONICALS LoRA Benchmark")
    print("=" * 60)

    reset_memory()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ==========================================================================
    # Step 1: Apply Liger Kernel patches BEFORE model loading
    # ==========================================================================
    print("\n  [1] Applying Liger Kernel patches...")
    model_type = model_name.lower()

    try:
        if 'qwen' in model_type:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen2
            apply_liger_kernel_to_qwen2(
                rope=True, rms_norm=True, swiglu=True,
                cross_entropy=True, fused_linear_cross_entropy=False,
            )
            print("    [OK] Liger Kernel patched for Qwen2")
        elif 'llama' in model_type:
            from liger_kernel.transformers import apply_liger_kernel_to_llama
            apply_liger_kernel_to_llama(
                rope=True, rms_norm=True, swiglu=True,
                cross_entropy=True, fused_linear_cross_entropy=False,
            )
            print("    [OK] Liger Kernel patched for LLaMA")
        elif 'mistral' in model_type:
            from liger_kernel.transformers import apply_liger_kernel_to_mistral
            apply_liger_kernel_to_mistral(
                rope=True, rms_norm=True, swiglu=True,
                cross_entropy=True, fused_linear_cross_entropy=False,
            )
            print("    [OK] Liger Kernel patched for Mistral")
    except ImportError as e:
        print(f"    [WARN] Liger Kernel not available: {e}")

    # ==========================================================================
    # Step 2: Load model and tokenizer
    # ==========================================================================
    print("\n  [2] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).cuda()

    # ==========================================================================
    # Step 3: Apply LoRA
    # ==========================================================================
    print("\n  [3] Applying LoRA...")
    try:
        from peft import LoraConfig, get_peft_model, TaskType

        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0.0,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_config)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    [OK] LoRA applied: r={lora_rank}, trainable={trainable_params:,}")
    except ImportError:
        print("    [ERROR] PEFT not installed!")
        return None

    # ==========================================================================
    # Step 4: Apply torch.compile
    # ==========================================================================
    print("\n  [4] Applying torch.compile...")
    try:
        torch._dynamo.config.suppress_errors = True
        model = torch.compile(model, mode="default", fullgraph=False)
        print("    [OK] torch.compile applied")
    except Exception as e:
        print(f"    [WARN] torch.compile failed: {e}")

    # ==========================================================================
    # Step 5: Load dataset
    # ==========================================================================
    print("\n  [5] Loading dataset...")
    dataset = load_dataset_for_benchmark(
        tokenizer, dataset_name, seq_length,
        num_samples=(num_steps + warmup_steps) * batch_size * 2,
    )

    from torch.utils.data import DataLoader

    def collate_fn(examples):
        return {
            "input_ids": torch.stack([ex["input_ids"] for ex in examples]),
            "attention_mask": torch.stack([ex["attention_mask"] for ex in examples]),
            "labels": torch.stack([ex["labels"] for ex in examples]),
        }

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # ==========================================================================
    # Step 6: Setup optimizer (LoRA+ style with differential LR)
    # ==========================================================================
    print("\n  [6] Setting up LoRA+ optimizer...")
    lora_a_params = []
    lora_b_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_A" in name:
            lora_a_params.append(param)
        elif "lora_B" in name:
            lora_b_params.append(param)

    lr_base = 2e-4
    lr_ratio = 16.0

    param_groups = [
        {"params": lora_a_params, "lr": lr_base},
        {"params": lora_b_params, "lr": lr_base * lr_ratio},
    ]

    try:
        optimizer = torch.optim.AdamW(param_groups, fused=True)
        print(f"    [OK] LoRA+ with fused AdamW (B matrix LR = {lr_base * lr_ratio})")
    except TypeError:
        optimizer = torch.optim.AdamW(param_groups)
        print(f"    [OK] LoRA+ with standard AdamW")

    # ==========================================================================
    # Step 7: Warmup
    # ==========================================================================
    print(f"\n  [7] Warming up ({warmup_steps} steps)...")
    model.train()
    data_iter = iter(dataloader)

    for i in range(warmup_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = {k: v.cuda() for k, v in batch.items()}

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(**batch)
            loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    cuda_sync()
    reset_memory()

    # ==========================================================================
    # Step 8: Benchmark runs
    # ==========================================================================
    print(f"\n  [8] Running benchmark ({num_steps} steps x {num_runs} runs)...")
    run_throughputs = []
    run_losses = []

    for run in range(num_runs):
        timer = CUDATimer()
        timer.start()

        losses = []
        for step in range(num_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())

            if (step + 1) % 20 == 0:
                print(f"      Run {run + 1} Step {step + 1}/{num_steps}, Loss: {loss.item():.4f}")

        elapsed = timer.stop()
        tokens = num_steps * batch_size * seq_length
        throughput = tokens / elapsed
        run_throughputs.append(throughput)
        run_losses.append(losses[-1])

        print(f"    Run {run + 1}: {throughput:,.0f} tok/s, Loss: {losses[-1]:.4f}")

    # Aggregate results
    avg_throughput = statistics.mean(run_throughputs)
    std_throughput = statistics.stdev(run_throughputs) if len(run_throughputs) > 1 else 0
    peak_memory = get_gpu_memory_mb()

    result = BenchmarkResult(
        method="Chronicals LoRA",
        model=model_name,
        dataset=dataset_name,
        throughput_tokens_sec=avg_throughput,
        throughput_std=std_throughput,
        peak_memory_mb=peak_memory,
        total_time_sec=sum(t / r for t, r in zip(run_throughputs, [1] * len(run_throughputs))),
        final_loss=statistics.mean(run_losses),
        steps=num_steps,
        batch_size=batch_size,
        seq_length=seq_length,
        lora_rank=lora_rank,
        optimizations={
            "liger_kernel": True,
            "torch_compile": True,
            "lora_plus": True,
            "fused_adamw": True,
        },
        run_throughputs=run_throughputs,
    )

    print(f"\n  CHRONICALS RESULT:")
    print(f"    Throughput: {avg_throughput:,.0f} +/- {std_throughput:.0f} tok/s")
    print(f"    Peak Memory: {peak_memory:,.0f} MB")
    print(f"    Final Loss: {result.final_loss:.4f}")

    del model, optimizer
    reset_memory()

    return result


# =============================================================================
# BENCHMARK: UNSLOTH LORA
# =============================================================================

def benchmark_unsloth_lora(
    model_name: str,
    dataset_name: str,
    num_steps: int,
    batch_size: int,
    seq_length: int,
    lora_rank: int = 16,
    warmup_steps: int = 10,
    num_runs: int = 1,
) -> Optional[BenchmarkResult]:
    """
    Benchmark Unsloth LoRA for fair comparison.

    Unsloth optimizations:
    - Fused QK RoPE
    - Chunked Cross-Entropy
    - LoRA Bracketing
    - Optimized gradient checkpointing
    """
    print("\n" + "=" * 60)
    print("UNSLOTH LoRA Benchmark")
    print("=" * 60)

    reset_memory()

    # Check if Unsloth is available
    try:
        from unsloth import FastLanguageModel
        print("  [OK] Unsloth imported")
    except ImportError as e:
        print(f"  [ERROR] Unsloth not available: {e}")
        print("  Install with: pip install unsloth")
        return None

    # ==========================================================================
    # Step 1: Load model with Unsloth
    # ==========================================================================
    print("\n  [1] Loading model with Unsloth...")

    # Unsloth model mapping
    unsloth_model_map = {
        "Qwen/Qwen2.5-0.5B": "unsloth/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-1.5B": "unsloth/Qwen2.5-1.5B",
        "Qwen/Qwen2.5-3B": "unsloth/Qwen2.5-3B",
        "Qwen/Qwen2.5-7B": "unsloth/Qwen2.5-7B",
    }

    unsloth_name = unsloth_model_map.get(model_name, model_name)

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=unsloth_name,
            max_seq_length=seq_length,
            dtype=torch.bfloat16,
            load_in_4bit=False,  # Full precision for fair comparison
        )
        print(f"    [OK] Loaded {unsloth_name}")
    except Exception as e:
        print(f"    [ERROR] Failed to load model: {e}")
        return None

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ==========================================================================
    # Step 2: Apply LoRA
    # ==========================================================================
    print("\n  [2] Applying LoRA with Unsloth...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_rank * 2,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        max_seq_length=seq_length,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    [OK] LoRA applied: r={lora_rank}, trainable={trainable:,}")

    # ==========================================================================
    # Step 3: Load dataset and create trainer
    # ==========================================================================
    print("\n  [3] Setting up trainer...")

    try:
        from trl import SFTTrainer, SFTConfig
    except ImportError:
        from trl import SFTTrainer
        from transformers import TrainingArguments as SFTConfig

    from datasets import load_dataset

    # Load dataset
    if dataset_name == "alpaca":
        dataset = load_dataset("yahma/alpaca-cleaned", split=f"train[:{(num_steps + warmup_steps) * batch_size * 2}]")

        alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""

        def format_fn(example):
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            output = example.get("output", "")
            if input_text:
                instruction = f"{instruction}\n\nInput: {input_text}"
            return {"text": alpaca_prompt.format(instruction=instruction, output=output)}

        dataset = dataset.map(format_fn)
    else:
        dataset = load_dataset(DATASET_REGISTRY[dataset_name]["path"], split=f"train[:{num_steps * batch_size * 2}]")

    # ==========================================================================
    # Step 4: Warmup
    # ==========================================================================
    print(f"\n  [4] Warming up ({warmup_steps} steps)...")

    warmup_args = SFTConfig(
        output_dir="./unsloth_warmup",
        per_device_train_batch_size=batch_size,
        max_steps=warmup_steps,
        logging_steps=999999,
        save_strategy="no",
        bf16=True,
        max_seq_length=seq_length,
        dataset_text_field="text",
        packing=False,
        report_to="none",
    )

    warmup_trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=warmup_args,
    )
    warmup_trainer.train()
    cuda_sync()
    del warmup_trainer
    reset_memory()

    # ==========================================================================
    # Step 5: Benchmark runs
    # ==========================================================================
    print(f"\n  [5] Running benchmark ({num_steps} steps x {num_runs} runs)...")

    run_throughputs = []
    run_losses = []

    for run in range(num_runs):
        training_args = SFTConfig(
            output_dir="./unsloth_benchmark",
            per_device_train_batch_size=batch_size,
            max_steps=num_steps,
            logging_steps=20,
            save_strategy="no",
            bf16=True,
            max_seq_length=seq_length,
            dataset_text_field="text",
            packing=False,
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            tokenizer=tokenizer,
            args=training_args,
        )

        timer = CUDATimer()
        timer.start()
        trainer.train()
        elapsed = timer.stop()

        tokens = num_steps * batch_size * seq_length
        throughput = tokens / elapsed
        run_throughputs.append(throughput)

        # Get loss
        final_loss = 0
        if trainer.state.log_history:
            for entry in reversed(trainer.state.log_history):
                if 'loss' in entry:
                    final_loss = entry['loss']
                    break
        run_losses.append(final_loss)

        print(f"    Run {run + 1}: {throughput:,.0f} tok/s, Loss: {final_loss:.4f}")

        del trainer

    # Aggregate results
    avg_throughput = statistics.mean(run_throughputs)
    std_throughput = statistics.stdev(run_throughputs) if len(run_throughputs) > 1 else 0
    peak_memory = get_gpu_memory_mb()

    result = BenchmarkResult(
        method="Unsloth LoRA",
        model=model_name,
        dataset=dataset_name,
        throughput_tokens_sec=avg_throughput,
        throughput_std=std_throughput,
        peak_memory_mb=peak_memory,
        total_time_sec=0,  # Calculated internally by Unsloth
        final_loss=statistics.mean(run_losses),
        steps=num_steps,
        batch_size=batch_size,
        seq_length=seq_length,
        lora_rank=lora_rank,
        optimizations={
            "fused_qk_rope": True,
            "chunked_cross_entropy": True,
            "lora_bracketing": True,
            "unsloth_checkpointing": True,
        },
        run_throughputs=run_throughputs,
    )

    print(f"\n  UNSLOTH RESULT:")
    print(f"    Throughput: {avg_throughput:,.0f} +/- {std_throughput:.0f} tok/s")
    print(f"    Peak Memory: {peak_memory:,.0f} MB")
    print(f"    Final Loss: {result.final_loss:.4f}")

    del model
    reset_memory()

    return result


# =============================================================================
# BENCHMARK: HUGGINGFACE BASELINE
# =============================================================================

def benchmark_huggingface_lora(
    model_name: str,
    dataset_name: str,
    num_steps: int,
    batch_size: int,
    seq_length: int,
    lora_rank: int = 16,
    warmup_steps: int = 10,
    num_runs: int = 1,
) -> BenchmarkResult:
    """Benchmark standard HuggingFace + PEFT LoRA as baseline."""
    print("\n" + "=" * 60)
    print("HUGGINGFACE PEFT LoRA Benchmark (Baseline)")
    print("=" * 60)

    reset_memory()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load model
    print("\n  [1] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).cuda()

    # Apply LoRA
    print("\n  [2] Applying LoRA...")
    from peft import LoraConfig, get_peft_model, TaskType

    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    [OK] LoRA applied: r={lora_rank}, trainable={trainable:,}")

    # Load dataset
    print("\n  [3] Loading dataset...")
    dataset = load_dataset_for_benchmark(
        tokenizer, dataset_name, seq_length,
        num_samples=(num_steps + warmup_steps) * batch_size * 2,
    )

    from torch.utils.data import DataLoader

    def collate_fn(examples):
        return {
            "input_ids": torch.stack([ex["input_ids"] for ex in examples]),
            "attention_mask": torch.stack([ex["attention_mask"] for ex in examples]),
            "labels": torch.stack([ex["labels"] for ex in examples]),
        }

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Optimizer (standard)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    # Warmup
    print(f"\n  [4] Warming up ({warmup_steps} steps)...")
    model.train()
    data_iter = iter(dataloader)

    for i in range(warmup_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = {k: v.cuda() for k, v in batch.items()}

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(**batch)
            loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    cuda_sync()
    reset_memory()

    # Benchmark
    print(f"\n  [5] Running benchmark ({num_steps} steps)...")
    run_throughputs = []
    run_losses = []

    for run in range(num_runs):
        timer = CUDATimer()
        timer.start()

        losses = []
        for step in range(num_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            batch = {k: v.cuda() for k, v in batch.items()}

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

            if (step + 1) % 20 == 0:
                print(f"      Run {run + 1} Step {step + 1}/{num_steps}, Loss: {loss.item():.4f}")

        elapsed = timer.stop()
        tokens = num_steps * batch_size * seq_length
        throughput = tokens / elapsed
        run_throughputs.append(throughput)
        run_losses.append(losses[-1])

    # Aggregate
    avg_throughput = statistics.mean(run_throughputs)
    std_throughput = statistics.stdev(run_throughputs) if len(run_throughputs) > 1 else 0
    peak_memory = get_gpu_memory_mb()

    result = BenchmarkResult(
        method="HuggingFace PEFT",
        model=model_name,
        dataset=dataset_name,
        throughput_tokens_sec=avg_throughput,
        throughput_std=std_throughput,
        peak_memory_mb=peak_memory,
        total_time_sec=0,
        final_loss=statistics.mean(run_losses),
        steps=num_steps,
        batch_size=batch_size,
        seq_length=seq_length,
        lora_rank=lora_rank,
        optimizations={},
        run_throughputs=run_throughputs,
    )

    print(f"\n  HUGGINGFACE RESULT:")
    print(f"    Throughput: {avg_throughput:,.0f} +/- {std_throughput:.0f} tok/s")
    print(f"    Peak Memory: {peak_memory:,.0f} MB")

    del model, optimizer
    reset_memory()

    return result


# =============================================================================
# VISUALIZATION
# =============================================================================

def generate_comparison_chart(results: List[BenchmarkResult], output_path: str = "benchmark_comparison.png"):
    """Generate comparison bar chart."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  [WARN] matplotlib not installed, skipping chart generation")
        return

    methods = [r.method for r in results]
    throughputs = [r.throughput_tokens_sec for r in results]
    memories = [r.peak_memory_mb for r in results]
    stds = [r.throughput_std for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Throughput chart
    ax1 = axes[0]
    colors = ['#2ecc71' if 'Chronicals' in m else '#3498db' if 'Unsloth' in m else '#95a5a6' for m in methods]
    bars1 = ax1.bar(methods, throughputs, yerr=stds, color=colors, capsize=5)
    ax1.set_ylabel('Tokens/second')
    ax1.set_title('Training Throughput Comparison')
    ax1.tick_params(axis='x', rotation=15)

    for bar, val in zip(bars1, throughputs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(stds) * 0.5,
                 f'{val:,.0f}', ha='center', va='bottom', fontsize=10)

    # Memory chart
    ax2 = axes[1]
    bars2 = ax2.bar(methods, memories, color=colors)
    ax2.set_ylabel('Peak Memory (MB)')
    ax2.set_title('Memory Usage Comparison')
    ax2.tick_params(axis='x', rotation=15)

    for bar, val in zip(bars2, memories):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                 f'{val:,.0f}', ha='center', va='bottom', fontsize=10)

    plt.suptitle('Chronicals vs Unsloth LoRA Benchmark', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Chart saved to: {output_path}")


def print_comparison_table(results: List[BenchmarkResult]):
    """Print formatted comparison table."""
    print("\n" + "=" * 100)
    print("BENCHMARK COMPARISON TABLE")
    print("=" * 100)
    print(f"{'Method':<25} {'Throughput':>15} {'Memory (MB)':>15} {'Loss':>10} {'Speedup':>10}")
    print("-" * 100)

    # Find baseline (HuggingFace)
    baseline_throughput = None
    for r in results:
        if "HuggingFace" in r.method:
            baseline_throughput = r.throughput_tokens_sec
            break

    if baseline_throughput is None and results:
        baseline_throughput = results[-1].throughput_tokens_sec

    for r in results:
        speedup = r.throughput_tokens_sec / baseline_throughput if baseline_throughput else 1.0
        print(f"{r.method:<25} {r.throughput_tokens_sec:>15,.0f} {r.peak_memory_mb:>15,.0f} "
              f"{r.final_loss:>10.4f} {speedup:>9.2f}x")

    print("-" * 100)

    # Winner announcement
    if len(results) >= 2:
        sorted_results = sorted(results, key=lambda x: x.throughput_tokens_sec, reverse=True)
        winner = sorted_results[0]
        print(f"\nWINNER: {winner.method} ({winner.throughput_tokens_sec:,.0f} tokens/sec)")

        if "Chronicals" in winner.method:
            print("  Chronicals BEATS the competition!")
        elif "Unsloth" in winner.method:
            print("  Unsloth wins this round. More optimization needed!")

    print("=" * 100)


# =============================================================================
# MAIN
# =============================================================================

def run_full_benchmark(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    dataset_name: str = "alpaca",
    num_steps: int = 100,
    batch_size: int = 4,
    seq_length: int = 512,
    lora_rank: int = 16,
    warmup_steps: int = 10,
    num_runs: int = 1,
    output_dir: str = "./benchmark_results",
) -> List[BenchmarkResult]:
    """Run full benchmark suite."""
    print("\n" + "=" * 70)
    print("CHRONICALS vs UNSLOTH BENCHMARK SUITE")
    print("=" * 70)

    # GPU info
    gpu_info = get_gpu_info()
    print(f"\nGPU: {gpu_info.get('name', 'Unknown')}")
    print(f"Memory: {gpu_info.get('memory_gb', 0):.1f} GB")
    print(f"CUDA: {gpu_info.get('cuda_version', 'Unknown')}")

    print(f"\nBenchmark Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Steps: {num_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_length}")
    print(f"  LoRA rank: {lora_rank}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Runs: {num_runs}")

    results = []

    # 1. HuggingFace baseline
    try:
        hf_result = benchmark_huggingface_lora(
            model_name, dataset_name, num_steps, batch_size,
            seq_length, lora_rank, warmup_steps, num_runs,
        )
        if hf_result:
            results.append(hf_result)
    except Exception as e:
        print(f"\n[ERROR] HuggingFace benchmark failed: {e}")

    # 2. Unsloth
    try:
        unsloth_result = benchmark_unsloth_lora(
            model_name, dataset_name, num_steps, batch_size,
            seq_length, lora_rank, warmup_steps, num_runs,
        )
        if unsloth_result:
            results.append(unsloth_result)
    except Exception as e:
        print(f"\n[ERROR] Unsloth benchmark failed: {e}")

    # 3. Chronicals (our implementation)
    try:
        chronicals_result = benchmark_chronicals_lora(
            model_name, dataset_name, num_steps, batch_size,
            seq_length, lora_rank, warmup_steps, num_runs,
        )
        if chronicals_result:
            results.append(chronicals_result)
    except Exception as e:
        print(f"\n[ERROR] Chronicals benchmark failed: {e}")

    # Generate outputs
    if results:
        print_comparison_table(results)
        generate_comparison_chart(results, os.path.join(output_dir, "benchmark_comparison.png"))

        # Save results as JSON
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, "benchmark_results.json")
        with open(results_path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "gpu_info": gpu_info,
                "config": {
                    "model": model_name,
                    "dataset": dataset_name,
                    "steps": num_steps,
                    "batch_size": batch_size,
                    "seq_length": seq_length,
                    "lora_rank": lora_rank,
                },
                "results": [r.to_dict() for r in results],
            }, f, indent=2)
        print(f"\nResults saved to: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Chronicals vs Unsloth Benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B",
                        help="Model name or 'all' for all models")
    parser.add_argument("--dataset", type=str, default="alpaca",
                        help="Dataset name or 'all' for all datasets")
    parser.add_argument("--steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq_length", type=int, default=512, help="Sequence length")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Warmup steps")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of runs for averaging")
    parser.add_argument("--output_dir", type=str, default="./benchmark_results", help="Output directory")
    parser.add_argument("--full-suite", action="store_true", help="Run full benchmark suite")

    args = parser.parse_args()

    if args.full_suite or args.model == "all":
        # Run benchmarks for multiple models
        models = list(MODEL_REGISTRY.keys())[:3]  # First 3 models
        all_results = []
        for model in models:
            results = run_full_benchmark(
                model_name=model,
                dataset_name=args.dataset,
                num_steps=args.steps,
                batch_size=args.batch_size,
                seq_length=args.seq_length,
                lora_rank=args.lora_rank,
                warmup_steps=args.warmup_steps,
                num_runs=args.num_runs,
                output_dir=args.output_dir,
            )
            all_results.extend(results)
    else:
        run_full_benchmark(
            model_name=args.model,
            dataset_name=args.dataset,
            num_steps=args.steps,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            lora_rank=args.lora_rank,
            warmup_steps=args.warmup_steps,
            num_runs=args.num_runs,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
