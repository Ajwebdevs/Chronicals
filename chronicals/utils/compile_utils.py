%%writefile compile_utils.py
"""
Chronicals torch.compile + LoRA Hotswapping Utilities
=======================================================
Production-grade torch.compile integration with LoRA adapter hotswapping
to avoid recompilation overhead.

KEY INNOVATIONS:
================
1. PEFT Hotswap Integration: Load adapters without recompilation
2. Graph Break Prevention: Strategies to avoid compilation overhead
3. Mode Selection: Optimal compile mode per use case
4. Compilation Caching: Reuse compiled artifacts across runs
5. Profiling Integration: Measure compilation vs runtime benefits

PERFORMANCE TARGETS:
===================
- 1.2-1.5x speedup from torch.compile optimizations
- Zero recompilation when hotswapping LoRA adapters
- <5% compilation overhead per training run

TORCH.COMPILE MODES:
===================
- "default": Best balance, good for most cases (~10% speedup)
- "reduce-overhead": Minimizes CUDA kernel launch overhead (~15% speedup)
- "max-autotune": Maximum optimization, longer compile time (~25% speedup)

LORA HOTSWAPPING:
================
The key insight is that compiled graphs can remain valid when:
1. Model architecture (shapes, ops) stays the same
2. Only adapter weights change (same rank/alpha)
3. PEFT hotswap API is used instead of adapter reload

Workflow:
    1. model.enable_peft_hotswap(target_rank=max_rank)
    2. model.load_adapter(adapter1_path)
    3. model = torch.compile(model, mode="default")
    4. model.load_adapter(adapter2_path, hotswap=True)  # No recompile!

REFERENCES:
===========
- PyTorch 2.0 torch.compile: https://pytorch.org/docs/stable/torch.compiler.html
- PEFT Hotswapping: https://huggingface.co/docs/peft
- Compilation Caching: https://pytorch.org/docs/stable/torch.compiler_cudagraphs.html

Author: Chronicals Framework
Version: 1.0.0
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Callable, Union, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
import functools
import warnings
import time
import os
import hashlib

# =============================================================================
# PEFT/LoRA Availability Detection
# =============================================================================

PEFT_AVAILABLE = False
PEFT_VERSION = None

try:
    import peft
    from peft import PeftModel, LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
    PEFT_VERSION = getattr(peft, '__version__', 'unknown')
except ImportError:
    PeftModel = None
    LoraConfig = None
    get_peft_model = None

# Check torch.compile availability
COMPILE_AVAILABLE = hasattr(torch, 'compile')
TORCH_VERSION = torch.__version__


def get_compile_info() -> Dict[str, Any]:
    """Get torch.compile and PEFT availability info."""
    info = {
        'torch_version': TORCH_VERSION,
        'compile_available': COMPILE_AVAILABLE,
        'peft_available': PEFT_AVAILABLE,
        'peft_version': PEFT_VERSION,
        'cuda_available': torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['device_name'] = torch.cuda.get_device_name(0)
        info['device_capability'] = torch.cuda.get_device_capability(0)

    return info


# =============================================================================
# Compilation Configuration
# =============================================================================

@dataclass
class CompileConfig:
    """
    Configuration for torch.compile with LoRA awareness.

    Attributes:
        mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune')
        fullgraph: Require full graph compilation (no breaks)
        dynamic: Allow dynamic shapes (reduces recompilation)
        backend: Compilation backend ('inductor', 'cudagraphs', etc.)
        disable: Disable compilation entirely
        cache_dir: Directory for compilation cache
        verbose: Print compilation information

    LoRA-specific:
        enable_hotswap: Enable PEFT hotswap mode
        max_lora_rank: Maximum LoRA rank for hotswap compatibility
        freeze_base_model: Freeze base model weights
    """
    # torch.compile settings
    mode: str = "default"
    fullgraph: bool = False
    dynamic: bool = True
    backend: str = "inductor"
    disable: bool = False
    cache_dir: Optional[str] = None
    verbose: bool = False

    # LoRA hotswap settings
    enable_hotswap: bool = True
    max_lora_rank: int = 64
    freeze_base_model: bool = True

    # Performance tuning
    use_triton: bool = True
    use_cudagraphs: bool = False  # Can conflict with dynamic shapes
    max_autotune_gemm: bool = True
    coordinate_descent_tuning: bool = False

    # Profiling
    profile_compilation: bool = False
    warmup_steps: int = 3

    def __post_init__(self):
        """Validate configuration."""
        valid_modes = ['default', 'reduce-overhead', 'max-autotune']
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid mode: {self.mode}. Must be one of {valid_modes}")

        valid_backends = ['inductor', 'cudagraphs', 'onnxrt', 'tvm', 'eager']
        if self.backend not in valid_backends:
            warnings.warn(f"Backend '{self.backend}' may not be available")

        # CUDAGraphs requires fullgraph and no dynamic shapes
        if self.use_cudagraphs:
            self.fullgraph = True
            self.dynamic = False

    def get_compile_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for torch.compile()."""
        kwargs = {
            'mode': self.mode,
            'fullgraph': self.fullgraph,
            'dynamic': self.dynamic,
            'backend': self.backend,
        }

        # Set inductor options if applicable
        if self.backend == 'inductor':
            # These are set via torch._inductor.config
            pass

        return kwargs

    def get_mode_description(self) -> str:
        """Get human-readable description of compilation mode."""
        descriptions = {
            'default': "Balanced optimization (~10% speedup, fast compile)",
            'reduce-overhead': "Minimize kernel launch overhead (~15% speedup)",
            'max-autotune': "Maximum optimization (~25% speedup, slow compile)",
        }
        return descriptions.get(self.mode, "Unknown mode")


# =============================================================================
# Graph Break Prevention
# =============================================================================

class GraphBreakPrevention:
    """
    Utilities to prevent graph breaks during torch.compile.

    Graph breaks force recompilation and hurt performance.
    Common causes:
    1. Python control flow with data-dependent conditions
    2. print/logging statements
    3. in-place operations on views
    4. Unsupported operations
    5. Dynamic shapes (if dynamic=False)
    """

    # Operations known to cause graph breaks
    BREAK_OPERATIONS = [
        'print', 'logging', 'pdb', 'breakpoint',
        'numpy', 'item', 'tolist',
    ]

    @staticmethod
    def check_module_for_breaks(module: nn.Module) -> List[str]:
        """
        Analyze module for potential graph break sources.

        Args:
            module: PyTorch module to analyze

        Returns:
            List of potential graph break warnings
        """
        warnings_list = []

        # Check for data-dependent control flow in forward
        if hasattr(module, 'forward'):
            import inspect
            try:
                source = inspect.getsource(module.forward)

                for op in GraphBreakPrevention.BREAK_OPERATIONS:
                    if op in source:
                        warnings_list.append(
                            f"Potential graph break: '{op}' found in forward()"
                        )

                # Check for if statements with tensor conditions
                if 'if ' in source and ('.item()' in source or '.any()' in source or '.all()' in source):
                    warnings_list.append(
                        "Data-dependent control flow may cause graph breaks"
                    )
            except (OSError, TypeError):
                pass  # Can't get source, skip

        return warnings_list

    @staticmethod
    @contextmanager
    def suppress_breaks():
        """
        Context manager to suppress graph break warnings during compilation.

        Usage:
            with GraphBreakPrevention.suppress_breaks():
                model = torch.compile(model)
        """
        import logging
        logger = logging.getLogger("torch._dynamo")
        original_level = logger.level
        logger.setLevel(logging.ERROR)
        try:
            yield
        finally:
            logger.setLevel(original_level)

    @staticmethod
    def mark_static_shapes(
        tensor: torch.Tensor,
        dims: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Mark tensor dimensions as static for better compilation.

        Args:
            tensor: Input tensor
            dims: Dimensions to mark as static (None = all)

        Returns:
            Same tensor (marking is a hint to compiler)
        """
        if hasattr(torch, '_dynamo') and hasattr(torch._dynamo, 'mark_static'):
            if dims is None:
                dims = list(range(tensor.dim()))
            for dim in dims:
                torch._dynamo.mark_static(tensor, dim)
        return tensor


# =============================================================================
# LoRA Hotswapping Manager
# =============================================================================

class LoRAHotswapManager:
    """
    Manager for LoRA adapter hotswapping without recompilation.

    The key insight is that torch.compile captures the computational graph,
    not the weight values. If the graph structure stays the same (same LoRA rank),
    we can swap weights without recompilation.

    Usage:
        manager = LoRAHotswapManager(model, max_rank=64)
        manager.prepare_for_hotswap()

        # Compile once
        compiled_model = torch.compile(manager.model)

        # Swap adapters without recompile
        manager.hotswap_adapter("path/to/adapter2")

    Note: Requires PEFT >= 0.7.0 for hotswap support.
    """

    def __init__(
        self,
        model: nn.Module,
        max_rank: int = 64,
        freeze_base: bool = True,
    ):
        """
        Initialize the hotswap manager.

        Args:
            model: Base model or PeftModel
            max_rank: Maximum LoRA rank to support (determines buffer sizes)
            freeze_base: Freeze base model parameters
        """
        self.model = model
        self.max_rank = max_rank
        self.freeze_base = freeze_base
        self._is_prepared = False
        self._current_adapter = None
        self._adapter_cache: Dict[str, Dict] = {}

    def prepare_for_hotswap(self) -> nn.Module:
        """
        Prepare model for adapter hotswapping.

        This allocates LoRA buffers at max_rank size and enables
        the PEFT hotswap mode if available.

        Returns:
            Prepared model ready for compilation
        """
        if not PEFT_AVAILABLE:
            raise ImportError(
                "PEFT not available. Install with: pip install peft"
            )

        # If not already a PeftModel, we need to add LoRA first
        if not isinstance(self.model, PeftModel):
            warnings.warn(
                "Model is not a PeftModel. Cannot prepare for hotswap. "
                "Use get_peft_model() first."
            )
            return self.model

        # Enable hotswap mode (PEFT >= 0.7.0)
        if hasattr(self.model, 'enable_peft_hotswap'):
            self.model.enable_peft_hotswap(target_rank=self.max_rank)
            self._is_prepared = True
            print(f"[LoRA Hotswap] Enabled with max_rank={self.max_rank}")
        else:
            warnings.warn(
                "PEFT version does not support hotswap. "
                "Upgrade with: pip install -U peft"
            )

        # Freeze base model if requested
        if self.freeze_base:
            self._freeze_base_model()

        return self.model

    def hotswap_adapter(
        self,
        adapter_path: str,
        adapter_name: Optional[str] = None,
    ) -> bool:
        """
        Hotswap to a different adapter without recompilation.

        Args:
            adapter_path: Path to adapter weights
            adapter_name: Optional adapter name

        Returns:
            True if hotswap successful, False otherwise
        """
        if not self._is_prepared:
            warnings.warn("Model not prepared for hotswap. Call prepare_for_hotswap() first.")
            return False

        if not PEFT_AVAILABLE:
            return False

        try:
            # Load new adapter with hotswap flag
            if hasattr(self.model, 'load_adapter'):
                self.model.load_adapter(
                    adapter_path,
                    adapter_name=adapter_name or "hotswap",
                    is_trainable=True,
                    # Key flag: tells PEFT to reuse existing compiled graph
                    hotswap=True if hasattr(self.model, 'load_adapter') else None,
                )
                self._current_adapter = adapter_path
                print(f"[LoRA Hotswap] Swapped to: {adapter_path}")
                return True
            else:
                # Fallback: direct weight loading
                return self._manual_weight_swap(adapter_path)

        except Exception as e:
            warnings.warn(f"Hotswap failed: {e}")
            return False

    def _freeze_base_model(self):
        """Freeze base model parameters."""
        for name, param in self.model.named_parameters():
            if 'lora_' not in name.lower():
                param.requires_grad = False

    def _manual_weight_swap(self, adapter_path: str) -> bool:
        """
        Manually swap LoRA weights without full reload.

        This is a fallback for older PEFT versions.
        """
        try:
            import safetensors.torch
            import os

            # Look for adapter weights
            adapter_file = None
            for fname in ['adapter_model.safetensors', 'adapter_model.bin']:
                fpath = os.path.join(adapter_path, fname)
                if os.path.exists(fpath):
                    adapter_file = fpath
                    break

            if adapter_file is None:
                warnings.warn(f"No adapter weights found in {adapter_path}")
                return False

            # Load weights
            if adapter_file.endswith('.safetensors'):
                state_dict = safetensors.torch.load_file(adapter_file)
            else:
                state_dict = torch.load(adapter_file, map_location='cpu')

            # Update model weights
            model_state = self.model.state_dict()
            for key, value in state_dict.items():
                if key in model_state:
                    if model_state[key].shape == value.shape:
                        model_state[key].copy_(value)
                    else:
                        warnings.warn(f"Shape mismatch for {key}, skipping")

            return True

        except Exception as e:
            warnings.warn(f"Manual weight swap failed: {e}")
            return False

    def get_current_adapter(self) -> Optional[str]:
        """Get path to currently loaded adapter."""
        return self._current_adapter

    def is_prepared(self) -> bool:
        """Check if model is prepared for hotswapping."""
        return self._is_prepared


# =============================================================================
# Compiled Model Wrapper
# =============================================================================

class CompiledLoRAModel:
    """
    Wrapper for compiled models with LoRA hotswap support.

    Provides a clean interface for:
    1. Compiling the model with optimal settings
    2. Hotswapping adapters without recompilation
    3. Profiling compilation and runtime

    Usage:
        compiled = CompiledLoRAModel(model, config)
        compiled.compile()

        # Training
        output = compiled(input_ids, attention_mask=mask)

        # Swap adapter
        compiled.hotswap_adapter("path/to/new/adapter")

        # Continue training without recompile
        output = compiled(input_ids, attention_mask=mask)
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[CompileConfig] = None,
    ):
        """
        Initialize compiled model wrapper.

        Args:
            model: Base model (can be PeftModel)
            config: Compilation configuration
        """
        self.base_model = model
        self.config = config or CompileConfig()
        self.compiled_model: Optional[nn.Module] = None
        self.hotswap_manager: Optional[LoRAHotswapManager] = None

        # Profiling stats
        self._compile_time: float = 0.0
        self._forward_times: List[float] = []
        self._warmup_complete: bool = False

    def compile(self) -> 'CompiledLoRAModel':
        """
        Compile the model with configured settings.

        Returns:
            self for chaining
        """
        if not COMPILE_AVAILABLE:
            warnings.warn("torch.compile not available. Using eager mode.")
            self.compiled_model = self.base_model
            return self

        if self.config.disable:
            print("[Compile] Disabled, using eager mode")
            self.compiled_model = self.base_model
            return self

        # Set up LoRA hotswap if applicable
        if self.config.enable_hotswap and PEFT_AVAILABLE:
            if isinstance(self.base_model, PeftModel):
                self.hotswap_manager = LoRAHotswapManager(
                    self.base_model,
                    max_rank=self.config.max_lora_rank,
                    freeze_base=self.config.freeze_base_model,
                )
                self.hotswap_manager.prepare_for_hotswap()

        # Configure inductor settings
        self._configure_inductor()

        # Compile
        print(f"[Compile] Mode: {self.config.mode}")
        print(f"[Compile] {self.config.get_mode_description()}")

        start_time = time.time()

        compile_kwargs = self.config.get_compile_kwargs()
        self.compiled_model = torch.compile(self.base_model, **compile_kwargs)

        self._compile_time = time.time() - start_time
        print(f"[Compile] Initial compile setup: {self._compile_time:.2f}s")

        return self

    def _configure_inductor(self):
        """Configure torch inductor settings."""
        if not hasattr(torch, '_inductor'):
            return

        try:
            config = torch._inductor.config

            # Enable Triton if available
            if self.config.use_triton:
                config.triton.cudagraphs = self.config.use_cudagraphs

            # Enable max-autotune for GEMM operations
            if self.config.max_autotune_gemm:
                config.max_autotune = True
                config.max_autotune_gemm = True

            # Coordinate descent tuning (slower but potentially better)
            if self.config.coordinate_descent_tuning:
                config.coordinate_descent_tuning = True

            # Cache settings
            if self.config.cache_dir:
                config.cache_dir = self.config.cache_dir

        except Exception as e:
            if self.config.verbose:
                print(f"[Compile] Could not configure inductor: {e}")

    def warmup(self, sample_input: Dict[str, torch.Tensor]) -> None:
        """
        Run warmup steps to trigger compilation.

        Args:
            sample_input: Sample model input dict
        """
        if self.compiled_model is None:
            self.compile()

        print(f"[Compile] Running {self.config.warmup_steps} warmup steps...")

        for i in range(self.config.warmup_steps):
            start = time.time()
            with torch.no_grad():
                _ = self.compiled_model(**sample_input)
            elapsed = time.time() - start
            print(f"  Warmup step {i+1}: {elapsed*1000:.1f}ms")

        self._warmup_complete = True
        print("[Compile] Warmup complete, model is compiled")

    def hotswap_adapter(self, adapter_path: str) -> bool:
        """
        Hotswap to a different LoRA adapter.

        Args:
            adapter_path: Path to new adapter

        Returns:
            True if successful
        """
        if self.hotswap_manager is None:
            warnings.warn("Hotswap manager not initialized. Model may not support hotswap.")
            return False

        return self.hotswap_manager.hotswap_adapter(adapter_path)

    def __call__(self, *args, **kwargs):
        """Forward pass through compiled model."""
        if self.compiled_model is None:
            self.compile()

        if self.config.profile_compilation:
            start = time.time()
            output = self.compiled_model(*args, **kwargs)
            self._forward_times.append(time.time() - start)
            return output

        return self.compiled_model(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """Forward pass (alias for __call__)."""
        return self.__call__(*args, **kwargs)

    def train(self, mode: bool = True):
        """Set training mode."""
        if self.compiled_model is not None:
            self.compiled_model.train(mode)
        return self

    def eval(self):
        """Set evaluation mode."""
        return self.train(False)

    def parameters(self):
        """Get model parameters."""
        return self.base_model.parameters()

    def named_parameters(self):
        """Get named model parameters."""
        return self.base_model.named_parameters()

    def state_dict(self):
        """Get model state dict."""
        return self.base_model.state_dict()

    def load_state_dict(self, state_dict):
        """Load model state dict."""
        return self.base_model.load_state_dict(state_dict)

    def get_profiling_stats(self) -> Dict[str, Any]:
        """Get compilation and runtime profiling stats."""
        stats = {
            'compile_time_s': self._compile_time,
            'warmup_complete': self._warmup_complete,
            'mode': self.config.mode,
        }

        if self._forward_times:
            import numpy as np
            times_ms = np.array(self._forward_times) * 1000
            stats['forward_times_ms'] = {
                'mean': float(np.mean(times_ms)),
                'std': float(np.std(times_ms)),
                'min': float(np.min(times_ms)),
                'max': float(np.max(times_ms)),
                'p50': float(np.percentile(times_ms, 50)),
                'p99': float(np.percentile(times_ms, 99)),
            }

        return stats


# =============================================================================
# Convenience Functions
# =============================================================================

def compile_model_with_lora(
    model: nn.Module,
    mode: str = "default",
    enable_hotswap: bool = True,
    max_lora_rank: int = 64,
    **compile_kwargs,
) -> CompiledLoRAModel:
    """
    Compile a model with LoRA hotswap support.

    Args:
        model: Model to compile (can be PeftModel)
        mode: Compilation mode
        enable_hotswap: Enable LoRA hotswapping
        max_lora_rank: Max rank for hotswap buffers
        **compile_kwargs: Additional compile arguments

    Returns:
        CompiledLoRAModel wrapper

    Example:
        model = AutoModelForCausalLM.from_pretrained(...)
        model = get_peft_model(model, lora_config)

        compiled = compile_model_with_lora(model, mode="reduce-overhead")

        # Training
        output = compiled(**batch)

        # Swap adapter
        compiled.hotswap_adapter("path/to/adapter2")
    """
    config = CompileConfig(
        mode=mode,
        enable_hotswap=enable_hotswap,
        max_lora_rank=max_lora_rank,
        **compile_kwargs,
    )

    wrapper = CompiledLoRAModel(model, config)
    wrapper.compile()

    return wrapper


def get_optimal_compile_mode(
    batch_size: int,
    seq_length: int,
    num_params: int,
) -> str:
    """
    Get recommended compile mode based on workload.

    Args:
        batch_size: Training batch size
        seq_length: Sequence length
        num_params: Number of model parameters

    Returns:
        Recommended compile mode
    """
    # Rough heuristics based on workload size
    tokens_per_batch = batch_size * seq_length

    # Small batches benefit more from reduce-overhead
    if tokens_per_batch < 4096:
        return "reduce-overhead"

    # Large models benefit from max-autotune
    if num_params > 3e9:  # > 3B params
        return "max-autotune"

    # Default for medium workloads
    return "default"


# =============================================================================
# Main Entry Point and Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Chronicals torch.compile + LoRA Hotswapping Utilities")
    print("=" * 70)

    # Print availability info
    info = get_compile_info()
    print("\nEnvironment Info:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    # Test configuration
    print("\n--- Testing CompileConfig ---")
    config = CompileConfig(
        mode="reduce-overhead",
        enable_hotswap=True,
        max_lora_rank=64,
    )
    print(f"Mode: {config.mode}")
    print(f"Description: {config.get_mode_description()}")
    print(f"Compile kwargs: {config.get_compile_kwargs()}")

    # Test with dummy model
    print("\n--- Testing Compilation ---")

    class DummyModel(nn.Module):
        def __init__(self, hidden_size=512):
            super().__init__()
            self.linear1 = nn.Linear(hidden_size, hidden_size)
            self.linear2 = nn.Linear(hidden_size, hidden_size)
            self.norm = nn.LayerNorm(hidden_size)

        def forward(self, x):
            x = self.linear1(x)
            x = torch.relu(x)
            x = self.linear2(x)
            x = self.norm(x)
            return x

    model = DummyModel()

    # Check for graph breaks
    print("\nChecking for potential graph breaks...")
    warnings_list = GraphBreakPrevention.check_module_for_breaks(model)
    if warnings_list:
        for w in warnings_list:
            print(f"  Warning: {w}")
    else:
        print("  No obvious graph breaks detected")

    # Compile
    if COMPILE_AVAILABLE:
        print("\nCompiling model...")
        compiled = CompiledLoRAModel(model, config)
        compiled.compile()

        # Warmup
        sample_input = {'x': torch.randn(2, 128, 512)}
        compiled.warmup(sample_input)

        # Benchmark
        print("\nBenchmarking...")
        import time
        times = []
        for _ in range(10):
            start = time.time()
            _ = compiled(**sample_input)
            times.append(time.time() - start)

        print(f"  Mean forward time: {sum(times)/len(times)*1000:.2f}ms")
        print(f"  Min forward time: {min(times)*1000:.2f}ms")

        # Get stats
        print("\nProfiling stats:")
        stats = compiled.get_profiling_stats()
        for k, v in stats.items():
            print(f"  {k}: {v}")
    else:
        print("\ntorch.compile not available, skipping compilation test")

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)
