"""
UNSLOTH MAXIMUM PERFORMANCE BENCHMARK
=====================================
This benchmark gives Unsloth EVERY possible advantage:

1. IMPORT ORDER: Unsloth imported FIRST (before transformers/peft)
2. PACKING=TRUE: Up to 5x faster (per Unsloth docs)
3. GRADIENT CHECKPOINTING: "unsloth" mode (30% less VRAM)
4. FUSED QK ROPE: Automatically enabled by Unsloth
5. ADAMW_8BIT: Optimized optimizer
6. AUTO PADDING-FREE: Enabled by default in latest Unsloth
7. SAME LoRA CONFIG: rank=32, alpha=32 (matches Chronicals)

RUN THIS IN A FRESH RUNTIME - NO OTHER IMPORTS FIRST!

Usage:
    python unsloth_max_benchmark.py --model Qwen/Qwen2.5-0.5B --steps 50
"""

# =============================================================================
# CRITICAL: IMPORT UNSLOTH FIRST!
# =============================================================================
print("=" * 70)
print("UNSLOTH MAXIMUM PERFORMANCE BENCHMARK")
print("=" * 70)
print("\n[1/7] Importing Unsloth FIRST (critical for all optimizations)...")

try:
    from unsloth import FastLanguageModel
    print("  [OK] Unsloth imported FIRST - all optimizations active")
except ImportError:
    print("  [ERROR] Unsloth not installed!")
    print("  Install with: pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'")
    import sys
    sys.exit(1)

# Now import everything else
import torch
import time
import gc
import argparse
import json
from datetime import datetime

print("[2/7] Importing training utilities...")
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, Dataset
print("  [OK] TRL and datasets imported")


def reset_memory():
    """Reset GPU memory for accurate measurement."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def get_gpu_info():
    """Get GPU information."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return name, memory_gb
    return "CPU", 0


class CUDATimer:
    """Accurate CUDA timing using events."""
    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)

    def start(self):
        if self.cuda_available:
            torch.cuda.synchronize()
            self.start_event.record()
        self.cpu_start = time.perf_counter()

    def stop(self):
        if self.cuda_available:
            self.end_event.record()
            torch.cuda.synchronize()
            return self.start_event.elapsed_time(self.end_event) / 1000.0
        return time.perf_counter() - self.cpu_start


def benchmark_unsloth_maximum(
    model_name: str,
    num_steps: int = 50,
    batch_size: int = 2,
    seq_length: int = 512,
    warmup_steps: int = 10,
    use_packing: bool = True,
):
    """
    Benchmark Unsloth with MAXIMUM performance settings.

    ALL OPTIMIZATIONS ENABLED:
    - packing=True (5x faster per Unsloth docs)
    - use_gradient_checkpointing="unsloth" (30% less VRAM)
    - Fused QK RoPE (2.3x faster - automatic)
    - adamw_8bit optimizer
    - Auto padding-free (automatic in latest Unsloth)
    - Proper warmup for torch.compile
    """
    print("\n" + "=" * 70)
    print("UNSLOTH MAXIMUM PERFORMANCE BENCHMARK")
    print("=" * 70)

    gpu_name, gpu_memory = get_gpu_info()
    print(f"\n[3/7] GPU: {gpu_name} ({gpu_memory:.1f} GB)")

    reset_memory()

    # Map to Unsloth model name
    unsloth_model_map = {
        "Qwen/Qwen2.5-0.5B": "unsloth/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-1.5B": "unsloth/Qwen2.5-1.5B",
        "Qwen/Qwen2.5-7B": "unsloth/Qwen2.5-7B",
    }
    unsloth_model = unsloth_model_map.get(model_name, model_name)

    print(f"\n[4/7] Loading model: {unsloth_model}")
    print("  Settings:")
    print(f"    - dtype: bfloat16")
    print(f"    - max_seq_length: {seq_length}")
    print(f"    - load_in_4bit: False (full precision for fair comparison)")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=unsloth_model,
        max_seq_length=seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,  # Full precision for fair comparison with Chronicals
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n[5/7] Applying LoRA with ALL Unsloth optimizations...")
    print("  LoRA Settings (SAME AS CHRONICALS):")
    print("    - rank: 32")
    print("    - alpha: 32")
    print("    - target_modules: q,k,v,o,gate,up,down")
    print("    - dropout: 0 (optimized)")
    print("    - bias: none (optimized)")
    print("    - gradient_checkpointing: unsloth (30% less VRAM)")

    model = FastLanguageModel.get_peft_model(
        model,
        r=32,                             # SAME AS CHRONICALS
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=32,                    # SAME AS CHRONICALS
        lora_dropout=0,                   # Optimized
        bias="none",                      # Optimized
        use_gradient_checkpointing="unsloth",  # UNSLOTH'S BEST
        random_state=42,
        max_seq_length=seq_length,
        use_rslora=False,
        loftq_config=None,
    )

    # Count trainable params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Load Alpaca dataset (same as Chronicals benchmark)
    print("\n[6/7] Loading Alpaca dataset (same as Chronicals)...")
    total_samples_needed = (num_steps + warmup_steps + 10) * batch_size * 2

    alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

    try:
        raw_dataset = load_dataset("yahma/alpaca-cleaned", split=f"train[:{total_samples_needed}]")
        print(f"  [OK] Loaded yahma/alpaca-cleaned ({len(raw_dataset)} samples)")

        def format_alpaca(example):
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            output = example.get("output", "")
            if input_text:
                instruction = f"{instruction}\n\nInput: {input_text}"
            return {"text": alpaca_prompt.format(instruction, output)}

        dataset = raw_dataset.map(format_alpaca)
    except Exception as e:
        print(f"  [WARN] Alpaca failed ({e}), using synthetic data")
        dummy_texts = [
            "Below is an instruction. Respond appropriately.\n\n### Instruction:\nExplain machine learning.\n\n### Response:\nMachine learning is a subset of artificial intelligence. " * 5
            for _ in range(total_samples_needed)
        ]
        dataset = Dataset.from_dict({'text': dummy_texts})

    print("\n[7/7] Running benchmark...")
    print("  Benchmark Settings:")
    print(f"    - batch_size: {batch_size}")
    print(f"    - seq_length: {seq_length}")
    print(f"    - warmup_steps: {warmup_steps}")
    print(f"    - benchmark_steps: {num_steps}")
    print(f"    - packing: {use_packing} (UP TO 5X FASTER)")
    print(f"    - optimizer: adamw_8bit")
    print(f"    - gradient_checkpointing: unsloth")

    # WARMUP PHASE
    print(f"\n  Warming up ({warmup_steps} steps)...")
    warmup_args = SFTConfig(
        output_dir="./unsloth_max_warmup",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        max_steps=warmup_steps,
        logging_steps=999999,
        save_strategy="no",
        bf16=True,
        max_seq_length=seq_length,
        dataset_text_field="text",
        packing=use_packing,              # ENABLE PACKING
        optim="adamw_8bit",
        warmup_steps=0,
        report_to="none",
        seed=42,
    )

    warmup_trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=warmup_args,
    )

    warmup_trainer.train()
    torch.cuda.synchronize()
    del warmup_trainer
    reset_memory()

    # BENCHMARK PHASE
    print(f"  Running {num_steps} benchmark steps...")

    training_args = SFTConfig(
        output_dir="./unsloth_max_benchmark",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        max_steps=num_steps,
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        max_seq_length=seq_length,
        dataset_text_field="text",
        packing=use_packing,              # ENABLE PACKING - 5X FASTER!
        optim="adamw_8bit",
        warmup_steps=0,
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args,
    )

    # Time the benchmark
    timer = CUDATimer()
    timer.start()

    trainer.train()

    total_time = timer.stop()

    # Calculate metrics
    total_tokens = num_steps * batch_size * seq_length
    throughput = total_tokens / total_time
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024

    # Get final loss
    final_loss = 0
    if trainer.state.log_history:
        for entry in reversed(trainer.state.log_history):
            if 'loss' in entry:
                final_loss = entry['loss']
                break

    # Results
    packing_str = "packing=True" if use_packing else "packing=False"
    method_name = f"Unsloth MAX ({packing_str})"

    result = {
        'method': method_name,
        'throughput_tokens_sec': throughput,
        'peak_memory_mb': peak_memory,
        'total_time_sec': total_time,
        'final_loss': final_loss,
        'steps': num_steps,
        'warmup_steps': warmup_steps,
        'batch_size': batch_size,
        'seq_length': seq_length,
        'packing': use_packing,
        'gradient_checkpointing': 'unsloth',
        'optimizer': 'adamw_8bit',
        'trainable_params': trainable_params,
        'total_params': total_params,
        'gpu': gpu_name,
    }

    print("\n" + "=" * 70)
    print("UNSLOTH MAXIMUM PERFORMANCE RESULTS")
    print("=" * 70)
    print(f"  Method: {method_name}")
    print(f"  Throughput: {throughput:,.0f} tokens/sec")
    print(f"  Peak Memory: {peak_memory:,.0f} MB")
    print(f"  Total Time: {total_time:.2f}s")
    print(f"  Final Loss: {final_loss:.4f}")
    print(f"  Trainable Params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print("=" * 70)

    # Save results
    with open('unsloth_max_results.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: unsloth_max_results.json")

    del model, trainer
    reset_memory()

    return result


def main():
    parser = argparse.ArgumentParser(description="Unsloth Maximum Performance Benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B", help="Model name")
    parser.add_argument("--steps", type=int, default=50, help="Benchmark steps")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--seq_length", type=int, default=512, help="Sequence length")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Warmup steps")
    parser.add_argument("--no-packing", action="store_true", help="Disable packing")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("FAIR COMPARISON SETTINGS (MATCHING CHRONICALS)")
    print("=" * 70)
    print(f"  model: {args.model}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  seq_length: {args.seq_length}")
    print(f"  steps: {args.steps}")
    print(f"  warmup_steps: {args.warmup_steps}")
    print(f"  packing: {not args.no_packing}")
    print(f"  LoRA rank: 32")
    print(f"  LoRA alpha: 32")
    print("=" * 70)

    result = benchmark_unsloth_maximum(
        model_name=args.model,
        num_steps=args.steps,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        warmup_steps=args.warmup_steps,
        use_packing=not args.no_packing,
    )

    # Compare with Chronicals if available
    if result:
        print("\n" + "=" * 70)
        print("COMPARISON WITH CHRONICALS")
        print("=" * 70)
        chronicals_throughput = 11699  # From our benchmark
        speedup = chronicals_throughput / result['throughput_tokens_sec']
        print(f"  Chronicals LoRA+: {chronicals_throughput:,} tokens/sec")
        print(f"  Unsloth MAX: {result['throughput_tokens_sec']:,.0f} tokens/sec")
        if speedup > 1:
            print(f"  Chronicals is {speedup:.2f}x FASTER")
        else:
            print(f"  Unsloth is {1/speedup:.2f}x FASTER")
        print("=" * 70)


if __name__ == "__main__":
    main()
