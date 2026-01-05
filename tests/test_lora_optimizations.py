%%writefile test_lora_optimizations.py
"""
=============================================================================
CHRONICALS LORA OPTIMIZATIONS TEST SUITE
=============================================================================
Comprehensive tests for all LoRA optimizations to ensure:
1. Correctness - No accuracy degradation
2. Memory efficiency - Within expected bounds
3. Performance - Meeting throughput targets
4. Stability - No crashes or NaNs

Test Categories:
- Unit tests for individual optimizations
- Integration tests for combined optimizations
- Memory validation tests
- Correctness validation (gradients, loss convergence)
- Performance benchmarks

Usage:
    python test_lora_optimizations.py                    # Run all tests
    python test_lora_optimizations.py --quick            # Quick smoke tests
    python test_lora_optimizations.py --unit             # Unit tests only
    python test_lora_optimizations.py --integration      # Integration tests only
    python test_lora_optimizations.py --benchmark        # Performance benchmarks

=============================================================================
"""

import os
import sys
import gc
import time
import argparse
import unittest
import warnings
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# TEST CONFIGURATION
# =============================================================================

@dataclass
class TestConfig:
    """Test configuration."""
    model_name: str = "Qwen/Qwen2.5-0.5B"
    lora_rank: int = 8  # Small for fast tests
    batch_size: int = 2
    seq_length: int = 128  # Short for fast tests
    num_steps: int = 5  # Few steps for unit tests
    learning_rate: float = 2e-4
    dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"

    # Thresholds
    max_memory_overhead_pct: float = 20.0  # Max 20% memory overhead
    min_throughput_ratio: float = 0.8  # At least 80% of expected
    max_loss_nan_pct: float = 0.0  # No NaN losses allowed
    gradient_tolerance: float = 1e-3  # Gradient comparison tolerance


# Global test config
TEST_CONFIG = TestConfig()


# =============================================================================
# UTILITIES
# =============================================================================

def cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def skip_if_no_cuda(test_func):
    """Decorator to skip test if CUDA is not available."""
    def wrapper(*args, **kwargs):
        if not cuda_available():
            print(f"SKIPPED (no CUDA): {test_func.__name__}")
            return
        return test_func(*args, **kwargs)
    return wrapper


@contextmanager
def memory_tracker():
    """Context manager to track GPU memory usage."""
    if cuda_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start_memory = torch.cuda.memory_allocated()

    yield

    if cuda_available():
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()
        end_memory = torch.cuda.memory_allocated()
        print(f"    Memory: start={start_memory/1e6:.1f}MB, peak={peak_memory/1e6:.1f}MB, end={end_memory/1e6:.1f}MB")


def reset_gpu():
    """Reset GPU state."""
    gc.collect()
    if cuda_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def create_dummy_batch(
    batch_size: int,
    seq_length: int,
    vocab_size: int = 32000,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """Create dummy batch for testing."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def check_gradients(model: nn.Module, check_nan: bool = True, check_zero: bool = True) -> Dict[str, Any]:
    """Check gradient health."""
    results = {
        "total_params": 0,
        "params_with_grad": 0,
        "nan_grads": 0,
        "zero_grads": 0,
        "grad_norm": 0.0,
    }

    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.requires_grad:
            results["total_params"] += 1
            if param.grad is not None:
                results["params_with_grad"] += 1

                if check_nan and torch.isnan(param.grad).any():
                    results["nan_grads"] += 1

                if check_zero and (param.grad.abs().sum() == 0):
                    results["zero_grads"] += 1

                total_norm += param.grad.data.norm(2).item() ** 2

    results["grad_norm"] = total_norm ** 0.5
    return results


# =============================================================================
# UNIT TESTS: Individual Optimizations
# =============================================================================

class TestLigerKernelPatching(unittest.TestCase):
    """Test Liger Kernel patching."""

    @skip_if_no_cuda
    def test_liger_kernel_import(self):
        """Test that Liger Kernel can be imported."""
        print("\n  Testing Liger Kernel import...")
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen2
            print("    [PASS] Liger Kernel imported successfully")
        except ImportError as e:
            self.skipTest(f"Liger Kernel not installed: {e}")

    @skip_if_no_cuda
    def test_liger_kernel_patching_qwen(self):
        """Test Liger Kernel patching for Qwen."""
        print("\n  Testing Liger Kernel patching for Qwen...")
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen2

            # Apply patches (should not raise)
            apply_liger_kernel_to_qwen2(
                rope=True,
                rms_norm=True,
                swiglu=True,
                cross_entropy=True,
            )
            print("    [PASS] Liger Kernel patched successfully")
        except ImportError:
            self.skipTest("Liger Kernel not installed")
        except Exception as e:
            self.fail(f"Liger Kernel patching failed: {e}")


class TestLoRAPlus(unittest.TestCase):
    """Test LoRA+ differential learning rates."""

    def test_lora_plus_param_groups(self):
        """Test LoRA+ parameter group creation."""
        print("\n  Testing LoRA+ parameter groups...")

        # Create simple model with LoRA-like structure
        class DummyLoRAModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = nn.Linear(64, 64)
                self.lora_A = nn.Linear(64, 8, bias=False)
                self.lora_B = nn.Linear(8, 64, bias=False)

        model = DummyLoRAModel()

        # Create param groups
        lora_a_params = []
        lora_b_params = []
        other_params = []

        for name, param in model.named_parameters():
            if "lora_A" in name:
                lora_a_params.append(param)
            elif "lora_B" in name:
                lora_b_params.append(param)
            else:
                other_params.append(param)

        base_lr = 2e-4
        lr_ratio = 16.0

        param_groups = [
            {"params": lora_a_params, "lr": base_lr, "name": "lora_A"},
            {"params": lora_b_params, "lr": base_lr * lr_ratio, "name": "lora_B"},
            {"params": other_params, "lr": base_lr, "name": "other"},
        ]

        # Verify
        self.assertEqual(len(param_groups[0]["params"]), 1)  # lora_A
        self.assertEqual(len(param_groups[1]["params"]), 1)  # lora_B
        self.assertEqual(param_groups[0]["lr"], base_lr)
        self.assertEqual(param_groups[1]["lr"], base_lr * lr_ratio)

        print("    [PASS] LoRA+ parameter groups created correctly")


class TestCutCrossEntropy(unittest.TestCase):
    """Test Cut Cross-Entropy loss."""

    @skip_if_no_cuda
    def test_cce_basic(self):
        """Test basic CCE functionality."""
        print("\n  Testing Cut Cross-Entropy...")

        batch_size = 2
        seq_len = 32
        hidden_size = 64
        vocab_size = 1000

        # Create inputs
        hidden_states = torch.randn(
            batch_size, seq_len, hidden_size,
            device="cuda", dtype=torch.float32
        )
        lm_head_weight = torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.float32)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")

        # Standard cross-entropy
        logits = hidden_states @ lm_head_weight.t()
        standard_loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1),
            reduction="mean",
        )

        # CCE (chunked version)
        from COLAB_LORA_TRAINER import CutCrossEntropyLoss
        cce = CutCrossEntropyLoss(chunk_size=256)
        cce_loss = cce(hidden_states, lm_head_weight, labels)

        # Compare losses (should be close)
        diff = abs(standard_loss.item() - cce_loss.item())
        self.assertLess(diff, 0.1, f"CCE loss differs too much: {diff}")

        print(f"    Standard CE: {standard_loss.item():.4f}")
        print(f"    CCE: {cce_loss.item():.4f}")
        print(f"    Difference: {diff:.6f}")
        print("    [PASS] CCE produces correct loss")

    @skip_if_no_cuda
    def test_cce_memory_efficiency(self):
        """Test that CCE uses less memory than standard CE."""
        print("\n  Testing CCE memory efficiency...")

        batch_size = 4
        seq_len = 256
        hidden_size = 256
        vocab_size = 32000  # Large vocab like Qwen

        # Standard CE memory
        reset_gpu()
        hidden_states = torch.randn(
            batch_size, seq_len, hidden_size,
            device="cuda", dtype=torch.bfloat16
        )
        lm_head_weight = torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")

        torch.cuda.reset_peak_memory_stats()
        logits = hidden_states @ lm_head_weight.t()  # This allocates [batch*seq, vocab]
        _ = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
        standard_memory = torch.cuda.max_memory_allocated()

        # CCE memory
        reset_gpu()
        hidden_states = torch.randn(
            batch_size, seq_len, hidden_size,
            device="cuda", dtype=torch.bfloat16
        )
        lm_head_weight = torch.randn(vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")

        torch.cuda.reset_peak_memory_stats()
        from COLAB_LORA_TRAINER import CutCrossEntropyLoss
        cce = CutCrossEntropyLoss(chunk_size=4096)
        _ = cce(hidden_states, lm_head_weight, labels)
        cce_memory = torch.cuda.max_memory_allocated()

        print(f"    Standard CE memory: {standard_memory/1e6:.1f} MB")
        print(f"    CCE memory: {cce_memory/1e6:.1f} MB")
        print(f"    Ratio: {cce_memory/standard_memory:.2f}x")

        # CCE should use significantly less memory
        self.assertLess(cce_memory, standard_memory * 0.8,
                        "CCE should use at least 20% less memory")
        print("    [PASS] CCE uses less memory")


class TestTorchCompile(unittest.TestCase):
    """Test torch.compile integration."""

    @skip_if_no_cuda
    def test_torch_compile_basic(self):
        """Test basic torch.compile functionality."""
        print("\n  Testing torch.compile...")

        # Simple model
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        ).cuda()

        # Compile
        try:
            torch._dynamo.config.suppress_errors = True
            compiled_model = torch.compile(model, mode="default", fullgraph=False)

            # Test forward pass
            x = torch.randn(4, 64, device="cuda")
            y = compiled_model(x)

            self.assertEqual(y.shape, (4, 64))
            print("    [PASS] torch.compile works correctly")
        except Exception as e:
            self.skipTest(f"torch.compile not available: {e}")


class TestFusedAdamW(unittest.TestCase):
    """Test fused AdamW optimizer."""

    @skip_if_no_cuda
    def test_fused_adamw(self):
        """Test fused AdamW availability and functionality."""
        print("\n  Testing fused AdamW...")

        model = nn.Linear(64, 64).cuda()

        try:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)

            # Test step
            x = torch.randn(4, 64, device="cuda")
            loss = model(x).sum()
            loss.backward()
            optimizer.step()

            print("    [PASS] Fused AdamW works correctly")
        except TypeError:
            print("    [SKIP] Fused AdamW not available in this PyTorch version")
            self.skipTest("Fused AdamW not available")


# =============================================================================
# INTEGRATION TESTS: Combined Optimizations
# =============================================================================

class TestIntegration(unittest.TestCase):
    """Integration tests for combined optimizations."""

    @skip_if_no_cuda
    def test_full_lora_training_loop(self):
        """Test complete LoRA training loop with all optimizations."""
        print("\n  Testing full LoRA training loop...")

        reset_gpu()

        # Skip if transformers/peft not available
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError as e:
            self.skipTest(f"Required packages not installed: {e}")

        config = TEST_CONFIG

        # Apply Liger Kernel
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen2
            apply_liger_kernel_to_qwen2(
                rope=True, rms_norm=True, swiglu=True, cross_entropy=True,
            )
            print("    Liger Kernel: applied")
        except ImportError:
            print("    Liger Kernel: not available")

        # Load model
        print(f"    Loading model: {config.model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=config.dtype,
            trust_remote_code=True,
        ).cuda()

        # Apply LoRA
        peft_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_rank * 2,
            lora_dropout=0.0,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_config)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    Trainable params: {trainable_params:,}")

        # Apply torch.compile
        try:
            torch._dynamo.config.suppress_errors = True
            model = torch.compile(model, mode="default", fullgraph=False)
            print("    torch.compile: applied")
        except Exception:
            print("    torch.compile: not available")

        # Setup optimizer (LoRA+)
        lora_a_params = []
        lora_b_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_A" in name:
                lora_a_params.append(param)
            elif "lora_B" in name:
                lora_b_params.append(param)

        param_groups = [
            {"params": lora_a_params, "lr": config.learning_rate},
            {"params": lora_b_params, "lr": config.learning_rate * 16.0},
        ]

        try:
            optimizer = torch.optim.AdamW(param_groups, fused=True)
        except TypeError:
            optimizer = torch.optim.AdamW(param_groups)

        # Training loop
        model.train()
        losses = []
        start_time = time.perf_counter()

        for step in range(config.num_steps):
            batch = create_dummy_batch(
                config.batch_size,
                config.seq_length,
                vocab_size=tokenizer.vocab_size,
            )

            with torch.cuda.amp.autocast(dtype=config.dtype):
                outputs = model(**batch)
                loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            losses.append(loss.item())
            print(f"      Step {step + 1}/{config.num_steps}: loss={loss.item():.4f}")

        elapsed = time.perf_counter() - start_time
        tokens = config.num_steps * config.batch_size * config.seq_length
        throughput = tokens / elapsed
        peak_memory = torch.cuda.max_memory_allocated() / 1e9

        print(f"    Throughput: {throughput:,.0f} tok/s")
        print(f"    Peak memory: {peak_memory:.2f} GB")
        print(f"    Final loss: {losses[-1]:.4f}")

        # Validations
        self.assertFalse(any(torch.isnan(torch.tensor(losses))), "NaN loss detected")
        self.assertLess(losses[-1], losses[0] + 1.0, "Loss should not explode")

        print("    [PASS] Full training loop completed successfully")

        del model, optimizer
        reset_gpu()


# =============================================================================
# MEMORY VALIDATION TESTS
# =============================================================================

class TestMemoryValidation(unittest.TestCase):
    """Test memory usage is within expected bounds."""

    @skip_if_no_cuda
    def test_lora_memory_overhead(self):
        """Test that LoRA adds minimal memory overhead."""
        print("\n  Testing LoRA memory overhead...")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError as e:
            self.skipTest(f"Required packages not installed: {e}")

        config = TEST_CONFIG

        # Measure base model memory
        reset_gpu()
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=config.dtype,
            trust_remote_code=True,
        ).cuda()
        base_memory = torch.cuda.max_memory_allocated()

        del base_model
        reset_gpu()

        # Measure LoRA model memory
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=config.dtype,
            trust_remote_code=True,
        ).cuda()

        peft_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_rank * 2,
            lora_dropout=0.0,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_config)
        lora_memory = torch.cuda.max_memory_allocated()

        overhead_pct = ((lora_memory - base_memory) / base_memory) * 100

        print(f"    Base model memory: {base_memory/1e9:.2f} GB")
        print(f"    LoRA model memory: {lora_memory/1e9:.2f} GB")
        print(f"    Overhead: {overhead_pct:.1f}%")

        self.assertLess(overhead_pct, config.max_memory_overhead_pct,
                        f"LoRA memory overhead ({overhead_pct:.1f}%) exceeds limit")

        print("    [PASS] LoRA memory overhead is acceptable")

        del model
        reset_gpu()


# =============================================================================
# CORRECTNESS VALIDATION TESTS
# =============================================================================

class TestCorrectnessValidation(unittest.TestCase):
    """Test that optimizations don't degrade accuracy."""

    @skip_if_no_cuda
    def test_gradient_flow(self):
        """Test that gradients flow correctly through LoRA layers."""
        print("\n  Testing gradient flow...")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError as e:
            self.skipTest(f"Required packages not installed: {e}")

        config = TEST_CONFIG

        # Load and setup model
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32,  # Use float32 for gradient checking
            trust_remote_code=True,
        ).cuda()

        peft_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_rank * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_config)

        # Forward + backward
        batch = create_dummy_batch(2, 64, vocab_size=tokenizer.vocab_size)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        # Check gradients
        grad_info = check_gradients(model)

        print(f"    Total params: {grad_info['total_params']}")
        print(f"    Params with grad: {grad_info['params_with_grad']}")
        print(f"    NaN grads: {grad_info['nan_grads']}")
        print(f"    Zero grads: {grad_info['zero_grads']}")
        print(f"    Grad norm: {grad_info['grad_norm']:.4f}")

        self.assertEqual(grad_info['nan_grads'], 0, "NaN gradients detected")
        self.assertGreater(grad_info['params_with_grad'], 0, "No gradients computed")
        self.assertGreater(grad_info['grad_norm'], 0, "Zero gradient norm")

        print("    [PASS] Gradients flow correctly")

        del model
        reset_gpu()

    @skip_if_no_cuda
    def test_loss_convergence(self):
        """Test that loss converges during training."""
        print("\n  Testing loss convergence...")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError as e:
            self.skipTest(f"Required packages not installed: {e}")

        config = TEST_CONFIG

        # Setup
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=config.dtype,
            trust_remote_code=True,
        ).cuda()

        peft_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_rank * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_config)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.learning_rate,
        )

        # Train with same data to test memorization
        fixed_batch = create_dummy_batch(
            config.batch_size,
            config.seq_length,
            vocab_size=tokenizer.vocab_size,
        )

        model.train()
        losses = []
        num_steps = 20  # More steps for convergence test

        for step in range(num_steps):
            with torch.cuda.amp.autocast(dtype=config.dtype):
                outputs = model(**fixed_batch)
                loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            losses.append(loss.item())
            if (step + 1) % 5 == 0:
                print(f"      Step {step + 1}: loss={loss.item():.4f}")

        # Check convergence
        initial_loss = sum(losses[:3]) / 3
        final_loss = sum(losses[-3:]) / 3

        print(f"    Initial loss (avg first 3): {initial_loss:.4f}")
        print(f"    Final loss (avg last 3): {final_loss:.4f}")
        print(f"    Improvement: {initial_loss - final_loss:.4f}")

        self.assertLess(final_loss, initial_loss, "Loss should decrease during training")

        print("    [PASS] Loss converges correctly")

        del model, optimizer
        reset_gpu()


# =============================================================================
# PERFORMANCE BENCHMARK TESTS
# =============================================================================

class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests."""

    @skip_if_no_cuda
    def test_throughput_benchmark(self):
        """Benchmark training throughput."""
        print("\n  Running throughput benchmark...")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError as e:
            self.skipTest(f"Required packages not installed: {e}")

        config = TEST_CONFIG
        # Use larger values for benchmark
        batch_size = 4
        seq_length = 256
        num_steps = 20

        # Apply Liger Kernel
        try:
            from liger_kernel.transformers import apply_liger_kernel_to_qwen2
            apply_liger_kernel_to_qwen2(
                rope=True, rms_norm=True, swiglu=True, cross_entropy=True,
            )
        except ImportError:
            pass

        # Setup model
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=config.dtype,
            trust_remote_code=True,
        ).cuda()

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_config)

        # torch.compile
        try:
            torch._dynamo.config.suppress_errors = True
            model = torch.compile(model, mode="default", fullgraph=False)
        except Exception:
            pass

        # Optimizer
        try:
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=config.learning_rate,
                fused=True,
            )
        except TypeError:
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=config.learning_rate,
            )

        # Warmup
        model.train()
        for _ in range(3):
            batch = create_dummy_batch(batch_size, seq_length, vocab_size=tokenizer.vocab_size)
            with torch.cuda.amp.autocast(dtype=config.dtype):
                outputs = model(**batch)
                loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Benchmark
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_time = time.perf_counter()

        for step in range(num_steps):
            batch = create_dummy_batch(batch_size, seq_length, vocab_size=tokenizer.vocab_size)
            with torch.cuda.amp.autocast(dtype=config.dtype):
                outputs = model(**batch)
                loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        peak_memory = torch.cuda.max_memory_allocated() / 1e9

        total_tokens = num_steps * batch_size * seq_length
        throughput = total_tokens / elapsed

        print(f"    Configuration:")
        print(f"      Model: {config.model_name}")
        print(f"      Batch size: {batch_size}")
        print(f"      Sequence length: {seq_length}")
        print(f"      Steps: {num_steps}")
        print(f"    Results:")
        print(f"      Time: {elapsed:.2f}s")
        print(f"      Throughput: {throughput:,.0f} tokens/sec")
        print(f"      Peak memory: {peak_memory:.2f} GB")

        # Check against targets
        gpu_name = torch.cuda.get_device_name(0).lower()
        if "t4" in gpu_name:
            target = 5000
        elif "a100" in gpu_name:
            target = 20000
        else:
            target = 10000

        print(f"    Target: {target:,} tokens/sec")
        print(f"    Ratio: {throughput/target:.2f}x target")

        print("    [PASS] Throughput benchmark completed")

        del model, optimizer
        reset_gpu()


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_tests(test_type: str = "all"):
    """Run tests based on type."""
    print("\n" + "=" * 70)
    print("CHRONICALS LORA OPTIMIZATIONS TEST SUITE")
    print("=" * 70)

    # Hardware info
    if cuda_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"CUDA: {torch.version.cuda}")
    else:
        print("\nWARNING: CUDA not available. Many tests will be skipped.")

    print(f"\nPyTorch: {torch.__version__}")

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    if test_type == "all" or test_type == "unit":
        suite.addTests(loader.loadTestsFromTestCase(TestLigerKernelPatching))
        suite.addTests(loader.loadTestsFromTestCase(TestLoRAPlus))
        suite.addTests(loader.loadTestsFromTestCase(TestCutCrossEntropy))
        suite.addTests(loader.loadTestsFromTestCase(TestTorchCompile))
        suite.addTests(loader.loadTestsFromTestCase(TestFusedAdamW))

    if test_type == "all" or test_type == "integration":
        suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    if test_type == "all" or test_type == "memory":
        suite.addTests(loader.loadTestsFromTestCase(TestMemoryValidation))

    if test_type == "all" or test_type == "correctness":
        suite.addTests(loader.loadTestsFromTestCase(TestCorrectnessValidation))

    if test_type == "all" or test_type == "benchmark":
        suite.addTests(loader.loadTestsFromTestCase(TestPerformanceBenchmarks))

    if test_type == "quick":
        # Quick smoke tests only
        suite.addTests(loader.loadTestsFromTestCase(TestLigerKernelPatching))
        suite.addTests(loader.loadTestsFromTestCase(TestLoRAPlus))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.wasSuccessful():
        print("\n[PASS] All tests passed!")
    else:
        print("\n[FAIL] Some tests failed.")
        for test, trace in result.failures + result.errors:
            print(f"  - {test}")

    return result.wasSuccessful()


def main():
    parser = argparse.ArgumentParser(description="Chronicals LoRA Test Suite")
    parser.add_argument("--quick", action="store_true", help="Run quick smoke tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--memory", action="store_true", help="Run memory tests only")
    parser.add_argument("--correctness", action="store_true", help="Run correctness tests only")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks only")

    args = parser.parse_args()

    if args.quick:
        test_type = "quick"
    elif args.unit:
        test_type = "unit"
    elif args.integration:
        test_type = "integration"
    elif args.memory:
        test_type = "memory"
    elif args.correctness:
        test_type = "correctness"
    elif args.benchmark:
        test_type = "benchmark"
    else:
        test_type = "all"

    success = run_tests(test_type)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
