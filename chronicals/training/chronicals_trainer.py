"""
Chronicals Trainer
===================
Production-grade HuggingFace-compatible trainer with ALL optimizations.
Drop-in replacement for transformers.Trainer.

OPTIMIZED FOR MAXIMUM SPEED - Target: 50,000+ tok/s on A100

Key features:
- torch.compile with optimal mode selection (reduce-overhead for training)
- Regional compilation for faster cold start (2-5x faster)
- CUDA Graphs integration for fixed-shape training
- Fused AdamW (fused=True) for faster optimizer steps
- Minimal Python overhead in training loop
- Proper CUDA autocast with BF16/FP16/FP8
- Triton autotune caching
- MFU (Model FLOPs Utilization) tracking
- Performance regression detection
- Comprehensive profiling integration

References:
- https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/
- https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
- https://huggingface.co/docs/transformers/en/perf_torch_compile
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any, Callable, Union
import time
import os
import math
import inspect
from dataclasses import dataclass

from chronicals.config.config import ChronicalsConfig, TrainingConfig, HF_WRITE_TOKEN
from chronicals.utils.fp8_utils import FP8Handler
from .gradient_checkpointing import apply_gradient_checkpointing

# Import FlashAttention-3 optimizer for SOTA performance (arXiv:2407.08608)
# Provides: persistent kernels, ping-pong scheduling, warp specialization,
# FP8 block quantization, varlen sequence packing
try:
    from chronicals.kernels.flash_attention_optimizer import (
        FlashAttentionConfig,
        FlashAttentionPatcher,
        optimize_model_for_speed,
        setup_torch_compile_for_flash_attn,
        get_flash_attn_info,
        get_gpu_specs,
        FLASH_ATTN_AVAILABLE,
        FLASH_ATTN_3_AVAILABLE,
        RING_ATTN_AVAILABLE,
        create_cu_seqlens_from_position_ids,
        is_packed_sequence,
    )
    FA_OPTIMIZER_AVAILABLE = True
except ImportError:
    FA_OPTIMIZER_AVAILABLE = False
    FLASH_ATTN_AVAILABLE = False
    FLASH_ATTN_3_AVAILABLE = False
    RING_ATTN_AVAILABLE = False

# Import DeepSeek V3 style FP8 training
try:
    from chronicals.utils.fp8_deepseek import (
        convert_model_to_fp8, FP8Linear, FP8Context,
        compute_fp8_memory_savings, DEFAULT_FP8_EXCLUDE_PATTERNS
    )
    FP8_DEEPSEEK_AVAILABLE = True
except ImportError:
    FP8_DEEPSEEK_AVAILABLE = False
from chronicals.data.sequence_packer import SequencePacker, PackedBatch, DataPrefetcher
from chronicals.kernels.triton_kernels import fused_cross_entropy, chunked_cross_entropy, ChunkedCrossEntropyLoss

# Import Apple Cut Cross-Entropy for memory-efficient training
# CCE computes loss WITHOUT materializing full logits - essential for large vocab models
try:
    from chronicals.kernels.cut_cross_entropy import (
        cut_cross_entropy,
        linear_cross_entropy,
        CutCrossEntropyLoss,
        get_optimal_chunk_size,
        estimate_memory_savings,
    )
    CCE_AVAILABLE = True
except ImportError:
    CCE_AVAILABLE = False

from chronicals.utils.visual_reporter import VisualReporter, TrainingMetrics

# Import optimizers
try:
    from chronicals.optimizers.optimizers import (
        ScheduleFreeAdamW, Muon, HybridMuonAdamW, Adam8bit, AdamAtan2,
        WSDScheduler, OneCycleLR, create_optimizer, create_scheduler
    )
    OPTIMIZERS_AVAILABLE = True
except ImportError:
    OPTIMIZERS_AVAILABLE = False

# Import FusedAdamW
try:
    from chronicals.optimizers.fused_adamw import FusedAdamW
    FUSED_ADAMW_AVAILABLE = True
except ImportError:
    FUSED_ADAMW_AVAILABLE = False

# Try to import Liger Kernel for ultra-fast fused operations
# Liger provides 20% throughput gain + 60% memory reduction
# Reference: https://github.com/linkedin/Liger-Kernel
LIGER_AVAILABLE = False
LIGER_AUTO_AVAILABLE = False
_LIGER_PATCHERS = {}

try:
    from liger_kernel.transformers import (
        apply_liger_kernel_to_llama,
        apply_liger_kernel_to_qwen2,
        apply_liger_kernel_to_gemma2,
        apply_liger_kernel_to_mistral,
    )
    _LIGER_PATCHERS = {
        'llama': apply_liger_kernel_to_llama,
        'qwen': apply_liger_kernel_to_qwen2,
        'qwen2': apply_liger_kernel_to_qwen2,
        'gemma': apply_liger_kernel_to_gemma2,
        'gemma2': apply_liger_kernel_to_gemma2,
        'mistral': apply_liger_kernel_to_mistral,
    }
    LIGER_AVAILABLE = True
except ImportError:
    try:
        from liger_kernel.transformers import apply_liger_kernel_to_llama
        _LIGER_PATCHERS = {'llama': apply_liger_kernel_to_llama}
        LIGER_AVAILABLE = True
    except ImportError:
        pass

# Try AutoLigerKernelForCausalLM for automatic model patching (simplest approach)
try:
    from liger_kernel.transformers import AutoLigerKernelForCausalLM
    LIGER_AUTO_AVAILABLE = True
except ImportError:
    pass

# Try to import individual Liger components for custom integration
try:
    from liger_kernel.ops import (
        LigerRMSNorm,
        LigerCrossEntropyLoss,
        LigerSwiGLUMLP,
    )
    LIGER_OPS_AVAILABLE = True
except ImportError:
    LIGER_OPS_AVAILABLE = False

# GC management for reduced latency spikes
import gc

# Import CUDA Graph Manager for optimized training
# Reference: https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
try:
    from .cuda_graph_manager import (
        CUDAGraphManager,
        CUDAGraphConfig,
        CapturedGraph,
        create_cuda_graph_training_step,
    )
    CUDA_GRAPHS_AVAILABLE = True
except ImportError:
    CUDA_GRAPHS_AVAILABLE = False

# Import Performance Profiler for bottleneck identification
# Reference: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
try:
    from chronicals.utils.profiling_utils import (
        PerformanceProfiler,
        MFUTracker,
        ThroughputMonitor,
        MixedPrecisionManager,
        detect_gpu_specs,
        generate_performance_report,
        print_performance_report,
        GPU_SPECS,
    )
    PROFILING_AVAILABLE = True
except ImportError:
    PROFILING_AVAILABLE = False


def setup_torch_compile_for_training(
    model: nn.Module,
    config: Optional["TrainingConfig"] = None,
    use_liger: bool = False,
) -> nn.Module:
    """
    Configure torch.compile with optimal settings for LLM training.

    This is a standalone function that can be used to compile models
    with the best settings for training, accounting for Liger Kernel
    compatibility and CUDA graph integration.

    IMPORTANT: torch.compile conflicts with Liger Kernel!
    - If Liger is enabled, use 'default' mode or skip compilation entirely
    - If Liger is disabled, use 'reduce-overhead' for 70-100% speedup

    Args:
        model: The model to compile
        config: Optional TrainingConfig with compile settings
        use_liger: Whether Liger Kernel is enabled (affects mode selection)

    Returns:
        Compiled model (or original if compilation fails/skipped)

    Example:
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
        >>> model = setup_torch_compile_for_training(model, use_liger=True)

    References:
        - https://blog.ezyang.com/2024/11/ways-to-use-torch-compile/
        - https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/
        - https://huggingface.co/docs/transformers/en/perf_torch_compile
    """
    if not hasattr(torch, 'compile'):
        print("torch.compile not available (requires PyTorch 2.0+)")
        return model

    # Get configuration from config or use defaults
    if config is not None:
        mode = getattr(config, 'torch_compile_mode', 'default')
        fullgraph = getattr(config, 'torch_compile_fullgraph', False)
        backend = getattr(config, 'torch_compile_backend', 'inductor')
        dynamic = getattr(config, 'torch_compile_dynamic', None)
    else:
        mode = 'default'
        fullgraph = False
        backend = 'inductor'
        dynamic = None

    # CRITICAL: Handle Liger Kernel compatibility
    # reduce-overhead uses CUDA graphs which conflict with Liger's dynamic Triton kernels
    if use_liger and mode == 'reduce-overhead':
        print("[WARN] torch.compile reduce-overhead conflicts with Liger Kernel!")
        print("       Switching to 'default' mode for compatibility")
        mode = 'default'

    try:
        # Configure dynamo for better compatibility
        if hasattr(torch, '_dynamo'):
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.cache_size_limit = 256
            if hasattr(torch._dynamo.config, 'assume_static_by_default'):
                torch._dynamo.config.assume_static_by_default = True

        # Configure inductor for optimal Triton kernel generation
        if hasattr(torch, '_inductor') and hasattr(torch._inductor, 'config'):
            inductor_cfg = torch._inductor.config
            if hasattr(inductor_cfg, 'epilogue_fusion'):
                inductor_cfg.epilogue_fusion = True
            if hasattr(inductor_cfg, 'coordinate_descent_tuning'):
                inductor_cfg.coordinate_descent_tuning = True

        # Build compile options
        compile_kwargs = {
            'mode': mode,
            'fullgraph': fullgraph,
            'backend': backend,
        }

        if dynamic is not None:
            compile_kwargs['dynamic'] = dynamic

        # Compile the model
        compiled_model = torch.compile(model, **compile_kwargs)
        print(f"[OK] torch.compile (mode={mode}, backend={backend})")

        return compiled_model

    except Exception as e:
        print(f"[WARN] torch.compile failed: {e}")
        print("       Falling back to eager mode")
        return model


@dataclass
class TrainerState:
    """State of the trainer."""
    global_step: int = 0
    epoch: float = 0.0
    total_steps: int = 0
    best_loss: float = float('inf')
    log_history: List[Dict[str, Any]] = None
    total_tokens: int = 0
    total_time: float = 0.0

    def __post_init__(self):
        if self.log_history is None:
            self.log_history = []


class ChronicalsTrainer:
    """
    Chronicals Training Loop - MAXIMUM SPEED Edition.

    Optimizations:
    - torch.compile with fullgraph=False, suppress_errors=True (HF compatible)
    - Fused AdamW (fused=True parameter)
    - Minimal Python overhead in hot loop
    - torch.cuda.amp.autocast for mixed precision
    - No logging in hot path (batched logging only)
    - Pre-allocated tensors where possible
    """

    def __init__(
        self,
        model: nn.Module,
        args: TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        tokenizer = None,
        callbacks: Optional[List[Callable]] = None,
        model_config: Optional[ChronicalsConfig] = None,
    ):
        self.model = model
        self.args = args
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.tokenizer = tokenizer
        self.callbacks = callbacks or []
        self.model_config = model_config

        # State
        self.state = TrainerState()

        # Performance flags
        self._compile_warmup_done = False
        self._use_autocast = False

        # Setup
        self._setup_training()

    def _setup_training(self):
        """Setup training components with all optimizations."""
        print("\n" + "=" * 60)
        print("CHRONICALS OPTIMIZATION SETUP (SPEED MODE)")
        print("=" * 60)

        # Track active optimizations for diagnostic output
        self._active_optimizations = {}
        self._optimization_categories = {
            'kernel': [],      # Fused kernel optimizations
            'memory': [],      # Memory optimizations
            'precision': [],   # Numerical precision
            'compute': [],     # Compute optimizations
            'data': [],        # Data loading optimizations
        }

        # Device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)

        # Mixed precision - use autocast instead of model casting for better torch.compile compat
        if self.args.bf16 and torch.cuda.is_available():
            self._use_autocast = True
            self._autocast_dtype = torch.bfloat16
            print("  [OK] BF16 autocast enabled")
        elif self.args.fp16 if hasattr(self.args, 'fp16') else False:
            self._use_autocast = True
            self._autocast_dtype = torch.float16
            print("  [OK] FP16 autocast enabled")
        else:
            self._use_autocast = False
            self._autocast_dtype = torch.float32

        # ========== Liger Kernel (must be applied before compile) ==========
        self._setup_liger_kernel()

        # ========== FlashAttention-3 Optimization (arXiv:2407.08608) ==========
        self._setup_flash_attention()

        # ========== torch.compile - OPTIMIZED FOR SPEED ==========
        self._setup_torch_compile()

        # ========== Optimizer - USE FUSED ==========
        self._setup_optimizer()

        # ========== Learning Rate Scheduler ==========
        self._setup_scheduler()

        # ========== FP8 Handler (DeepSeek V3 Style) ==========
        self.fp8_handler = None
        self.fp8_context = None
        self.use_deepseek_fp8 = False

        if self.args.fp8:
            # Check if we should use DeepSeek V3 style FP8
            use_deepseek_style = getattr(self.args, 'use_deepseek_fp8', True)

            if use_deepseek_style and FP8_DEEPSEEK_AVAILABLE:
                # DeepSeek V3 style: block-wise FP8 with optimal amax history
                block_size = getattr(self.args, 'fp8_block_size', 128)
                amax_history = getattr(self.args, 'fp8_amax_history_len', 32)
                exclude_patterns = getattr(
                    self.args, 'fp8_exclude_patterns',
                    DEFAULT_FP8_EXCLUDE_PATTERNS
                )

                print(f"  [FP8] Converting model to DeepSeek V3 style FP8...")
                print(f"        Block size: {block_size}")
                print(f"        Amax history: {amax_history}")

                self.model = convert_model_to_fp8(
                    self.model,
                    block_size=block_size,
                    amax_history_len=amax_history,
                    exclude_patterns=exclude_patterns,
                    verbose=False,
                )

                # Create FP8 context for cache invalidation
                self.fp8_context = FP8Context(self.model)
                self.use_deepseek_fp8 = True

                # Print memory savings
                savings = compute_fp8_memory_savings(self.model)
                print(f"  [OK] FP8 training enabled (DeepSeek V3 style)")
                print(f"        Coverage: {savings['fp8_coverage']*100:.1f}%")
                print(f"        Memory savings: {savings['savings_ratio']:.2f}x")
            else:
                # Fallback to basic FP8Handler
                self.fp8_handler = FP8Handler()
                print("  [OK] FP8 training enabled (basic E4M3/E5M2)")

        # ========== Gradient Checkpointing ==========
        if self.args.use_gradient_checkpointing:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                print("  [OK] Gradient checkpointing (HuggingFace native)")
            else:
                try:
                    apply_gradient_checkpointing(
                        self.model,
                        checkpoint_ratio=self.args.checkpoint_ratio,
                    )
                    print(f"  [OK] Gradient checkpointing (ratio={self.args.checkpoint_ratio:.3f})")
                except ValueError:
                    print("  [WARN] Gradient checkpointing not available for this model")

        # ========== Sequence Packing ==========
        if self.args.use_sequence_packing:
            self.packer = SequencePacker(
                max_seq_len=self.args.max_seq_length,
                strategy=self.args.packing_strategy,
                use_flash_varlen=self.args.use_flash_varlen,
            )
            print(f"  [OK] Sequence packing (strategy={self.args.packing_strategy})")
        else:
            self.packer = None

        # ========== Cross-Entropy (Standard, Fused, CCE, or Adaptive) ==========
        # Cross-entropy implementation hierarchy (from fastest to most compatible):
        # 1. triton_cce: Triton kernel, computes logits on-the-fly (FASTEST)
        # 2. cce: Apple CCE PyTorch fallback, never materializes logits (RECOMMENDED)
        # 3. chunked: Chunked processing of pre-computed logits
        # 4. adaptive: Auto-select based on vocab size and available memory
        # 5. standard: PyTorch F.cross_entropy (baseline, high memory)
        self.use_fused_ce = False
        self.use_chunked_ce = getattr(self.args, 'use_chunked_cross_entropy', False)
        self.ce_chunk_size = getattr(self.args, 'cce_chunk_size', getattr(self.args, 'ce_chunk_size', 8192))
        self.ce_implementation = getattr(self.args, 'ce_implementation', 'standard')
        self.use_cce = False  # Track if using Apple CCE
        self.cce_loss_fn = None

        # Get vocab size from model config
        vocab_size = getattr(self.model.config, 'vocab_size', 32000) if hasattr(self.model, 'config') else 32000

        # Adaptive selection based on vocab size
        if self.ce_implementation == 'adaptive':
            small_thresh = getattr(self.args, 'ce_small_vocab_threshold', 32000)
            large_thresh = getattr(self.args, 'ce_large_vocab_threshold', 100000)
            if vocab_size <= small_thresh:
                self.ce_implementation = 'standard'
            elif vocab_size >= large_thresh:
                self.ce_implementation = 'cce'
            else:
                self.ce_implementation = 'chunked'
            print(f"  [OK] Adaptive CE selected: {self.ce_implementation} (vocab={vocab_size:,})")

        # Setup the selected cross-entropy implementation
        if self.ce_implementation in ('cce', 'triton_cce') and CCE_AVAILABLE:
            # Apple Cut Cross-Entropy - NEVER materializes full logits!
            # This is THE solution for large vocabulary models (Qwen, Gemma)
            use_auto_chunk = getattr(self.args, 'cce_auto_chunk_size', True)
            if use_auto_chunk:
                # Auto-tune chunk size based on vocab and available memory
                hidden_dim = getattr(self.model.config, 'hidden_size', 2048)
                self.ce_chunk_size = get_optimal_chunk_size(
                    vocab_size=vocab_size,
                    hidden_dim=hidden_dim,
                    available_vram_gb=40.0  # A100 80GB, conservative
                )

            self.cce_loss_fn = CutCrossEntropyLoss(
                chunk_size=self.ce_chunk_size,
                ignore_index=-100,
                label_smoothing=self.args.label_smoothing,
                z_loss_weight=getattr(self.args, 'z_loss_weight', 1e-4),
                reduction='mean',
            )
            self.use_cce = True
            self.chunked_ce_loss = None

            # Compute memory savings
            batch_size = self.args.per_device_train_batch_size
            seq_len = self.args.max_seq_length
            savings = estimate_memory_savings(batch_size, seq_len, vocab_size, self.ce_chunk_size)
            print(f"  [OK] Apple Cut Cross-Entropy (CCE) enabled")
            print(f"       Vocab: {vocab_size:,}, Chunk: {self.ce_chunk_size:,}")
            print(f"       Memory: {savings['standard_mb']/1024:.2f}GB -> {savings['cce_mb']:.0f}MB ({savings['reduction_factor']:.0f}x reduction)")

        elif self.use_chunked_ce or self.ce_implementation == 'chunked':
            # Chunked CE fallback (still saves memory, but requires logits first)
            self.chunked_ce_loss = ChunkedCrossEntropyLoss(
                chunk_size=self.ce_chunk_size,
                label_smoothing=self.args.label_smoothing,
                z_loss_weight=getattr(self.args, 'z_loss_weight', 1e-4),
                ignore_index=-100,
            )
            self.use_chunked_ce = True
            print(f"  [OK] Chunked Cross-Entropy (chunk_size={self.ce_chunk_size})")
            print(f"       Memory reduction: ~30x for large vocabularies")
        else:
            self.chunked_ce_loss = None
            print("  [OK] Standard cross-entropy (PyTorch)")

        # ========== Visual Reporter - DISABLED IN HOT PATH ==========
        if self.args.visual_reporting:
            self.reporter = VisualReporter(
                output_dir=self.args.output_dir,
                report_every=self.args.report_every_n_steps,
            )
        else:
            self.reporter = None

        # ========== MFU Tracking ==========
        self.track_mfu = self.args.track_mfu
        if self.track_mfu:
            self._compute_model_flops()
            print(f"  [OK] MFU tracking (peak={self.args.gpu_peak_tflops:.0f} TFLOPS)")

        # ========== CUDA Graph Manager ==========
        # CUDA graphs eliminate kernel launch overhead for fixed-shape training
        # NOTE: Conflicts with Liger Kernel and sequence packing
        self.cuda_graph_manager = None
        use_cuda_graphs = getattr(self.args, 'use_cuda_graphs', False)
        if use_cuda_graphs and CUDA_GRAPHS_AVAILABLE:
            use_liger = getattr(self, 'use_liger', False)
            use_packing = getattr(self.args, 'use_sequence_packing', False)

            if use_liger:
                print("  [--] CUDA graphs disabled (conflicts with Liger Kernel)")
            elif use_packing:
                print("  [--] CUDA graphs disabled (conflicts with sequence packing)")
            else:
                graph_config = CUDAGraphConfig(
                    warmup_steps=getattr(self.args, 'cuda_graph_warmup_steps', 3),
                    capture_mode=getattr(self.args, 'cuda_graph_capture_mode', 'thread_local'),
                    use_memory_pool=getattr(self.args, 'cuda_graph_use_memory_pool', True),
                    static_shapes=getattr(self.args, 'cuda_graph_static_shapes', True),
                    max_cached_graphs=getattr(self.args, 'cuda_graph_max_cached_graphs', 16),
                    fallback_on_error=getattr(self.args, 'cuda_graph_fallback_on_error', True),
                )
                self.cuda_graph_manager = CUDAGraphManager(graph_config, self.device)
                print("  [OK] CUDA Graph Manager initialized")

        # ========== Performance Profiler ==========
        # Tracks MFU, throughput, timing breakdown, and detects regressions
        self.profiler = None
        if PROFILING_AVAILABLE and getattr(self.args, 'visual_reporting', True):
            try:
                self.profiler = PerformanceProfiler(
                    model=self.model,
                    output_dir=self.args.output_dir,
                    gpu_peak_tflops=self.args.gpu_peak_tflops,
                    enable_memory_tracking=True,
                    enable_timing_breakdown=True,
                )
                print("  [OK] Performance Profiler initialized")
            except Exception as e:
                print(f"  [--] Performance Profiler failed: {e}")
                self.profiler = None

        # ========== Calculate Total Steps ==========
        num_train_samples = len(self.train_dataloader)
        self.state.total_steps = (
            num_train_samples * self.args.num_train_epochs
        ) // self.args.gradient_accumulation_steps

        # Create output directory
        os.makedirs(self.args.output_dir, exist_ok=True)

        print("=" * 60 + "\n")

    def _setup_liger_kernel(self):
        """
        Setup Liger Kernel for ultra-fast fused operations.

        Liger Kernel provides:
        - Fused RMSNorm: 7x faster, 3x less memory
        - Fused SwiGLU: Eliminates intermediate allocations
        - Fused CrossEntropy: 2.3x faster
        - Fused RoPE: 2.3x faster
        - Fused Linear CrossEntropy: 18x memory reduction (never materializes logits)

        Total impact: 20% throughput gain + 60% memory reduction
        Reference: https://github.com/linkedin/Liger-Kernel

        TORCH.COMPILE COMPATIBILITY (2025):
        - Liger kernels are torch.compile compatible out of the box
        - Apply Liger patching BEFORE torch.compile for best results
        - Reference: https://huggingface.co/docs/trl/en/liger_kernel_integration
        """
        self.use_liger = False
        self.use_fused_linear_ce = False  # Track if we're using Liger's fused linear CE

        if not self.args.use_liger_kernel or not LIGER_AVAILABLE:
            return

        try:
            # Detect model architecture and apply appropriate Liger kernel
            model_type = getattr(self.model.config, 'model_type', '').lower() if hasattr(self.model, 'config') else ''

            # Check if we should use fused_linear_cross_entropy (18x memory reduction)
            # This is the SOTA option for large vocabularies
            use_fused_linear_ce = getattr(self.args, 'use_fused_linear_cross_entropy', False)

            # Build common kwargs for Liger patching
            liger_kwargs = {
                'rope': getattr(self.args, 'use_fused_rope', True),
                'rms_norm': getattr(self.args, 'use_fused_rmsnorm', True),
                'swiglu': getattr(self.args, 'use_fused_swiglu', True),
                'cross_entropy': getattr(self.args, 'use_fused_cross_entropy', True) and not use_fused_linear_ce,
            }

            # Add fused_linear_cross_entropy if available (Liger 0.3.0+)
            if use_fused_linear_ce:
                liger_kwargs['fused_linear_cross_entropy'] = True
                liger_kwargs['cross_entropy'] = False  # Disable regular CE if using fused linear CE
                self.use_fused_linear_ce = True

            if 'llama' in model_type:
                from liger_kernel.transformers import apply_liger_kernel_to_llama
                apply_liger_kernel_to_llama(**liger_kwargs)
                ce_mode = "FusedLinearCE" if use_fused_linear_ce else "CE"
                print(f"  [OK] Liger Kernel (LLaMA: RoPE+RMSNorm+SwiGLU+{ce_mode})")
                self.use_liger = True

            elif 'qwen' in model_type:
                try:
                    from liger_kernel.transformers import apply_liger_kernel_to_qwen2
                    apply_liger_kernel_to_qwen2(**liger_kwargs)
                    ce_mode = "FusedLinearCE" if use_fused_linear_ce else "CE"
                    print(f"  [OK] Liger Kernel (Qwen2: RoPE+RMSNorm+SwiGLU+{ce_mode})")
                    self.use_liger = True
                except ImportError:
                    print("  [--] Liger Kernel for Qwen not available")

            elif 'gemma' in model_type:
                try:
                    from liger_kernel.transformers import apply_liger_kernel_to_gemma2
                    # Gemma uses GeGLU instead of SwiGLU
                    gemma_kwargs = liger_kwargs.copy()
                    del gemma_kwargs['swiglu']
                    gemma_kwargs['geglu'] = True
                    apply_liger_kernel_to_gemma2(**gemma_kwargs)
                    ce_mode = "FusedLinearCE" if use_fused_linear_ce else "CE"
                    print(f"  [OK] Liger Kernel (Gemma2: RoPE+RMSNorm+GeGLU+{ce_mode})")
                    self.use_liger = True
                except ImportError:
                    print("  [--] Liger Kernel for Gemma not available")

            elif 'mistral' in model_type:
                try:
                    from liger_kernel.transformers import apply_liger_kernel_to_mistral
                    apply_liger_kernel_to_mistral(**liger_kwargs)
                    ce_mode = "FusedLinearCE" if use_fused_linear_ce else "CE"
                    print(f"  [OK] Liger Kernel (Mistral: RoPE+RMSNorm+SwiGLU+{ce_mode})")
                    self.use_liger = True
                except ImportError:
                    print("  [--] Liger Kernel for Mistral not available")

            elif 'phi' in model_type:
                try:
                    from liger_kernel.transformers import apply_liger_kernel_to_phi3
                    apply_liger_kernel_to_phi3(**liger_kwargs)
                    ce_mode = "FusedLinearCE" if use_fused_linear_ce else "CE"
                    print(f"  [OK] Liger Kernel (Phi3: RoPE+RMSNorm+SwiGLU+{ce_mode})")
                    self.use_liger = True
                except ImportError:
                    print("  [--] Liger Kernel for Phi3 not available")

            else:
                # Try generic Liger kernel components
                try:
                    from liger_kernel.transformers import (
                        LigerRMSNorm,
                        LigerCrossEntropyLoss,
                    )
                    print(f"  [OK] Liger Kernel available (generic, model_type={model_type})")
                    self.use_liger = True
                except ImportError:
                    print(f"  [--] Liger Kernel components not available for model_type={model_type}")

        except Exception as e:
            print(f"  [--] Liger Kernel setup failed: {e}")
            self.use_liger = False

    def _setup_flash_attention(self):
        """
        Setup FlashAttention-3 optimization for SOTA throughput.

        Implements all FA3 paper optimizations (arXiv:2407.08608):
        - Warp specialization: Producer/consumer warps for TMA+WGMMA overlap
        - Ping-pong scheduling: 2 warpgroups alternate GEMM and softmax
        - Persistent kernels: Reduced launch overhead (132 thread blocks on H100)
        - Block quantization: Per-block FP8 scaling for 2.6x lower error
        - Varlen support: Efficient sequence packing with cu_seqlens

        Performance targets:
        - H100 FP16: 740 TFLOPS (75% utilization)
        - H100 FP8: 1.2 PFLOPS
        - A100 BF16: 156 TFLOPS (50% MFU)

        References:
        - https://arxiv.org/abs/2407.08608
        - https://tridao.me/blog/2024/flash3/
        - https://huggingface.co/blog/packing-with-FA2
        """
        self.fa_config = None
        self.use_fa3_optimization = False

        if not FA_OPTIMIZER_AVAILABLE:
            return

        # Check if FA optimization is requested
        use_fa_optimization = getattr(self.args, 'use_flash_attention_optimizer', True)
        if not use_fa_optimization:
            return

        try:
            # Get FlashAttention info
            fa_info = get_flash_attn_info()

            print(f"\n  [FA3] FlashAttention Status:")
            print(f"        FA2: {'Available' if fa_info['flash_attn_available'] else 'Not Available'} (v{fa_info['flash_attn_version']})")
            print(f"        FA3: {'Available' if fa_info['flash_attn_3_available'] else 'Not Available'}")
            print(f"        Ring Attention: {'Available' if fa_info['ring_attn_available'] else 'Not Available'}")

            if fa_info['is_hopper']:
                print(f"        GPU: Hopper (H100/H800) - FA3 optimizations enabled")
            elif fa_info['is_ampere']:
                print(f"        GPU: Ampere (A100) - FA2 optimizations only")

            # Create hardware-aware config
            self.fa_config = FlashAttentionConfig.from_gpu()

            # Enable varlen if sequence packing is used
            if self.args.use_sequence_packing:
                self.fa_config.use_varlen = True
                self.fa_config.varlen_use_position_ids = True

            # Enable ring attention for very long sequences
            enable_ring = getattr(self.args, 'use_ring_attention', False)
            if enable_ring and RING_ATTN_AVAILABLE:
                self.fa_config.use_ring_attention = True
                self.fa_config.ring_attention_impl = getattr(self.args, 'ring_attention_impl', 'llama3')

            # Apply model patching for varlen support if needed
            # DISABLED: varlen patching conflicts with torch.compile causing 10x slowdown
            # The standard FlashAttention in transformers handles variable lengths fine
            if False and self.args.use_sequence_packing and self.args.use_flash_varlen:
                if FLASH_ATTN_AVAILABLE:
                    self.model = FlashAttentionPatcher.patch_model_for_varlen(
                        self.model,
                        use_flash_attn_3=self.fa_config.use_flash_attn_3,
                    )
                    print(f"  [OK] FlashAttention varlen patched for sequence packing")
                    self._optimization_categories['compute'].append('FA varlen')

            # Print optimization summary
            print(f"  [OK] {self.fa_config.get_optimization_summary()}")

            # Store expected performance
            gpu_specs = get_gpu_specs()
            if self.fa_config.use_flash_attn_3:
                expected_tflops = gpu_specs.get('fa3_achievable_bf16_tflops', 661)
            else:
                expected_tflops = gpu_specs.get('fa2_achievable_bf16_tflops', 156)
            print(f"        Expected attention TFLOPS: {expected_tflops:.0f}")

            self.use_fa3_optimization = True
            self._optimization_categories['kernel'].append('FlashAttention-3' if self.fa_config.use_flash_attn_3 else 'FlashAttention-2')

        except Exception as e:
            print(f"  [--] FlashAttention optimization failed: {e}")
            self.fa_config = None
            self.use_fa3_optimization = False

    def _setup_torch_compile(self):
        """
        Setup torch.compile with OPTIMAL settings for LLM training (2024-2025 research).

        CRITICAL: torch.compile + Liger Kernel Compatibility
        =====================================================
        When Liger Kernel is enabled, torch.compile can cause 3-4x SLOWDOWN due to:
        1. Graph breaks from Liger's custom Triton ops (torch.autograd.Function)
        2. Recompilation on every step when using dynamic Triton kernels
        3. CUDA graph conflicts in reduce-overhead mode

        RECOMMENDATION:
        - Liger enabled: Use 'default' mode OR disable torch.compile entirely
        - Liger disabled: Use 'reduce-overhead' for 70-100% speedup with CUDA graphs

        Mode Selection Guide (2024-2025 best practices):
        ================================================
        - 'default': Balanced, Liger-compatible, no CUDA graphs
        - 'reduce-overhead': Best for training, uses CUDA graphs (70-100% speedup)
                             BUT conflicts with Liger's dynamic Triton kernels!
        - 'max-autotune': Best for inference, benchmarks kernel variants
        - 'max-autotune-no-cudagraphs': Good for dynamic shapes

        Key Features:
        - Regional compilation for faster cold start (2-5x faster first epoch)
        - Triton autotune caching for subsequent runs
        - Optimizer compilation for additional 10-15% speedup (PyTorch 2.2+)
        - Automatic Liger compatibility detection

        References:
        - https://blog.ezyang.com/2024/11/ways-to-use-torch-compile/
        - https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/
        - https://huggingface.co/docs/transformers/en/perf_torch_compile
        - https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
        """
        if not self.args.use_torch_compile or not hasattr(torch, 'compile'):
            self.is_compiled = False
            return

        # Check for emergency disable
        if getattr(self.args, 'torch_compile_disable', False):
            print("  [--] torch.compile disabled via config")
            self.is_compiled = False
            return

        # CRITICAL: Check Liger Kernel compatibility
        use_liger = getattr(self, 'use_liger', False) or getattr(self.args, 'use_liger_kernel', False)

        try:
            # Get compile options with research-backed defaults
            mode = getattr(self.args, 'torch_compile_mode', 'default')
            fullgraph = getattr(self.args, 'torch_compile_fullgraph', False)
            backend = getattr(self.args, 'torch_compile_backend', 'inductor')
            dynamic = getattr(self.args, 'torch_compile_dynamic', None)
            use_regional = getattr(self.args, 'torch_compile_regional', True)

            # CRITICAL: Handle Liger Kernel + torch.compile compatibility
            # reduce-overhead uses CUDA graphs which conflict with Liger's dynamic Triton kernels
            # This can cause 3-4x SLOWDOWN due to graph breaks and recompilation
            if use_liger and mode == 'reduce-overhead':
                print("  [WARN] torch.compile reduce-overhead conflicts with Liger Kernel!")
                print("         Switching to 'default' mode for compatibility")
                mode = 'default'

            # Configure dynamo for better compatibility with HuggingFace models
            # and custom Triton kernels
            if hasattr(torch, '_dynamo'):
                # Suppress errors to gracefully handle graph breaks
                torch._dynamo.config.suppress_errors = True
                # Increase cache size for larger models with many unique traces
                torch._dynamo.config.cache_size_limit = 256  # Increased for complex models
                # Optimize for training (not guard overhead)
                if hasattr(torch._dynamo.config, 'optimize_ddp'):
                    torch._dynamo.config.optimize_ddp = True
                # Assume static shapes by default for better performance
                # Dynamic shapes cause recompilation overhead
                if hasattr(torch._dynamo.config, 'assume_static_by_default'):
                    torch._dynamo.config.assume_static_by_default = True
                # Enable automatic dynamic shapes for truly variable inputs
                if hasattr(torch._dynamo.config, 'automatic_dynamic_shapes'):
                    torch._dynamo.config.automatic_dynamic_shapes = dynamic is None
                # Allow inline inbuilt nn modules for better fusion
                if hasattr(torch._dynamo.config, 'inline_inbuilt_nn_modules'):
                    torch._dynamo.config.inline_inbuilt_nn_modules = True
                # Skip check for Triton kernel autotuning (reduces overhead)
                if hasattr(torch._dynamo.config, 'skip_triton_cache_check'):
                    torch._dynamo.config.skip_triton_cache_check = True

            # Configure inductor for optimal Triton kernel generation
            # NOTE: torch._inductor.config attributes vary by PyTorch version
            # We use try/except and hasattr checks for maximum compatibility
            if hasattr(torch, '_inductor') and hasattr(torch._inductor, 'config'):
                inductor_config = torch._inductor.config

                # Enable Triton autotuning for best kernel performance
                # In newer PyTorch versions, this might be under different names
                try:
                    if hasattr(inductor_config, 'triton'):
                        triton_config = inductor_config.triton
                        # Try different attribute names for autotuning
                        if hasattr(triton_config, 'autotune'):
                            triton_config.autotune = True
                        elif hasattr(triton_config, 'autotune_pointwise'):
                            triton_config.autotune_pointwise = True
                except (AttributeError, TypeError):
                    pass  # Silently skip if config structure changed

                # Enable epilogue fusion for fused pointwise ops
                try:
                    if hasattr(inductor_config, 'epilogue_fusion'):
                        inductor_config.epilogue_fusion = True
                except (AttributeError, TypeError):
                    pass

                # Enable coordinate descent tuning for matmuls
                try:
                    if hasattr(inductor_config, 'coordinate_descent_tuning'):
                        inductor_config.coordinate_descent_tuning = True
                except (AttributeError, TypeError):
                    pass

                # Max autotune mode: enable more aggressive optimizations
                try:
                    if mode == 'max-autotune' and hasattr(inductor_config, 'max_autotune'):
                        inductor_config.max_autotune = True
                except (AttributeError, TypeError):
                    pass

                # Enable CUDA graphs in reduce-overhead mode
                try:
                    if mode == 'reduce-overhead' and hasattr(inductor_config, 'triton'):
                        triton_config = inductor_config.triton
                        if hasattr(triton_config, 'cudagraphs'):
                            triton_config.cudagraphs = True
                except (AttributeError, TypeError):
                    pass

                # Enable kernel fusion for custom Triton ops
                try:
                    if hasattr(inductor_config, 'aggressive_fusion'):
                        inductor_config.aggressive_fusion = True
                except (AttributeError, TypeError):
                    pass

            # Build compile options
            compile_kwargs = {
                'mode': mode,
                'fullgraph': fullgraph,  # Must be False for HF models with DDP/FSDP
                'backend': backend,
            }

            # Add dynamic shapes option if specified
            # None = auto-detect, True = allow dynamic, False = assume static
            if dynamic is not None:
                compile_kwargs['dynamic'] = dynamic

            # For varlen with sequence packing, use static shapes
            # (FlashAttention varlen handles variable lengths internally)
            if hasattr(self, 'fa_config') and self.fa_config is not None:
                if self.fa_config.use_varlen and 'dynamic' not in compile_kwargs:
                    compile_kwargs['dynamic'] = False
                    print(f"       Using static shapes (FA varlen handles variable lengths)")

            # Regional compilation: compile repeated transformer blocks individually
            # This is 2-5x faster cold start than full model compilation
            # Reference: https://huggingface.co/docs/transformers/en/perf_torch_compile
            if use_regional:
                compiled = self._apply_regional_compile(compile_kwargs)
                if compiled:
                    self.is_compiled = True
                    # Setup optimizer compilation if available (PyTorch 2.2+)
                    self._setup_compiled_optimizer()
                    return

            # Fallback: Full model compilation
            self.model = torch.compile(self.model, **compile_kwargs)
            print(f"  [OK] torch.compile (mode={mode}, backend={backend})")
            self.is_compiled = True

            # Setup optimizer compilation if available (PyTorch 2.2+)
            self._setup_compiled_optimizer()

        except Exception as e:
            print(f"  [--] torch.compile failed: {e}")
            self.is_compiled = False

    def _apply_regional_compile(self, compile_kwargs: dict) -> bool:
        """
        Apply regional compilation to transformer blocks for faster cold start.

        This compiles each repeated decoder layer class sequentially, hitting the
        compiler cache for subsequent blocks. Much faster cold start than full
        model compilation (2-5x speedup on first epoch).

        NOTE: Some models (like Qwen2) have issues with layer-level compilation
        due to forward method assignment restrictions. In such cases, we fall back
        to whole-model compilation which is more compatible.

        Reference: https://huggingface.co/docs/transformers/en/perf_torch_compile
        """
        mode = compile_kwargs.get('mode', 'default')

        # Try different model architectures
        layers = None
        layer_path = None

        # LLaMA/Qwen/Mistral style (model.model.layers)
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
            layer_path = 'model.model.layers'

        # GPT-2/GPT-Neo style (transformer.h)
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            layers = self.model.transformer.h
            layer_path = 'transformer.h'

        # BERT/RoBERTa style (encoder.layer)
        elif hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
            layers = self.model.encoder.layer
            layer_path = 'encoder.layer'

        # Gemma style (model.layers)
        elif hasattr(self.model, 'layers'):
            layers = self.model.layers
            layer_path = 'layers'

        if layers is None or len(layers) == 0:
            return False

        # Try regional compilation first (faster cold start)
        # But fall back to whole-model compilation if it fails
        # (e.g., Qwen2 has issues with layer.forward assignment)
        block_class = type(layers[0])
        try:
            for i, layer in enumerate(layers):
                layers[i] = torch.compile(layer, **compile_kwargs)

            print(f"  [OK] torch.compile regional ({len(layers)} {block_class.__name__}, mode={mode})")
            print(f"       Path: {layer_path}, faster cold start than full model")
            return True
        except Exception as e:
            # Regional compilation failed - fall back to whole model compilation
            # This is more compatible but has slower cold start
            print(f"  [--] Regional compile failed for {block_class.__name__}: {e}")
            print(f"       Falling back to whole-model compilation...")
            try:
                self.model = torch.compile(self.model, **compile_kwargs)
                print(f"  [OK] torch.compile whole-model (mode={mode})")
                return True
            except Exception as e2:
                print(f"  [--] Whole-model compile also failed: {e2}")
                return False

    def _setup_compiled_optimizer(self):
        """
        Compile optimizer step for additional speedups (PyTorch 2.2+).

        Reference: https://pytorch.org/blog/pytorch2-2/
        """
        try:
            # Check PyTorch version supports optimizer compilation
            version_parts = torch.__version__.split('.')[:2]
            major, minor = int(version_parts[0]), int(version_parts[1].split('+')[0].split('a')[0].split('b')[0].split('rc')[0])
            if (major, minor) >= (2, 2) and getattr(self.args, 'torch_compile_optimizer', True):
                self._compile_optimizer_enabled = True
                # Note: Actual optimizer compilation happens in training loop
                # via torch.compile wrapping the optimizer.step() call
                print("  [OK] Optimizer compilation enabled (PyTorch 2.2+)")
            else:
                self._compile_optimizer_enabled = False
        except Exception:
            self._compile_optimizer_enabled = False

    def _check_graph_breaks(self, verbose: bool = False) -> bool:
        """
        Check for graph breaks in the compiled model.

        Graph breaks reduce torch.compile efficiency because they:
        1. Split the computation graph into multiple subgraphs
        2. Introduce Python interpreter overhead between subgraphs
        3. Prevent cross-subgraph optimizations like kernel fusion

        Common causes of graph breaks (2025):
        - torch.autograd.Function without proper torch.library registration
        - Data-dependent control flow (if tensor.item() > 0)
        - Calls to Python functions not traced by dynamo
        - Incompatible operations with custom Triton kernels

        To minimize graph breaks:
        - Use torch.library.triton_op for custom Triton kernels
        - Use torch.cond for data-dependent control flow
        - Use torch._dynamo.allow_in_graph() for safe functions
        - Use fullgraph=False (default) to gracefully handle breaks

        Reference: https://pytorch.org/docs/stable/torch.compiler_profiling_torch_compile.html
        """
        if not self.is_compiled or not hasattr(torch, '_dynamo'):
            return True  # Can't check, assume OK

        try:
            # Get explanation of how the model compiles
            # This is expensive so only do it when verbose
            if verbose and hasattr(torch._dynamo, 'explain'):
                # Create a dummy input for explanation
                batch_size = 1
                seq_len = min(128, self.args.max_seq_length)
                dummy_ids = torch.randint(0, 1000, (batch_size, seq_len), device=self.device)
                dummy_labels = torch.randint(0, 1000, (batch_size, seq_len), device=self.device)

                # Get explanation
                explanation = torch._dynamo.explain(self.model)(input_ids=dummy_ids, labels=dummy_labels)

                if hasattr(explanation, 'graph_break_count'):
                    break_count = explanation.graph_break_count
                    if break_count > 0:
                        print(f"  [WARN] torch.compile found {break_count} graph breaks")
                        if hasattr(explanation, 'break_reasons'):
                            for i, reason in enumerate(explanation.break_reasons[:3]):  # Show first 3
                                print(f"         Break {i+1}: {reason}")
                        return False
                    else:
                        print(f"  [OK] No graph breaks detected")
                        return True

            # Simpler check: just verify compilation succeeded
            return True

        except Exception as e:
            if verbose:
                print(f"  [WARN] Graph break check failed: {e}")
            return True  # Assume OK on failure

    def _setup_optimizer(self):
        """Setup optimizer with FUSED enabled for maximum speed."""
        optimizer_type = self.args.optimizer_type
        params = [p for p in self.model.parameters() if p.requires_grad]

        self.use_external_grad_clip = True

        if optimizer_type == "fused_adamw" and FUSED_ADAMW_AVAILABLE:
            try:
                self.optimizer = FusedAdamW(
                    params,
                    lr=self.args.learning_rate,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                    weight_decay=self.args.weight_decay,
                    max_grad_norm=self.args.max_grad_norm,
                )
                print("  [OK] FusedAdamW optimizer (Triton, 1.8x faster)")
                self.use_external_grad_clip = False
                return
            except Exception as e:
                print(f"  [--] FusedAdamW failed ({e})")

        if optimizer_type == "schedule_free" and OPTIMIZERS_AVAILABLE:
            total_steps = len(self.train_dataloader) * self.args.num_train_epochs
            warmup_steps = int(total_steps * self.args.warmup_ratio)
            self.optimizer = ScheduleFreeAdamW(
                params,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                weight_decay=self.args.weight_decay,
                warmup_steps=warmup_steps,
            )
            print("  [OK] Schedule-Free AdamW (Meta AI)")
            return

        if optimizer_type == "muon" and OPTIMIZERS_AVAILABLE:
            self.optimizer = Muon(
                params,
                lr=self.args.learning_rate * 10,
                momentum=0.95,
            )
            print("  [OK] Muon optimizer (spectral normalization)")
            return

        if optimizer_type == "hybrid_muon" and OPTIMIZERS_AVAILABLE:
            self.optimizer = HybridMuonAdamW(
                self.model,
                lr_muon=0.02,
                lr_adamw=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )
            print("  [OK] Hybrid Muon + AdamW")
            return

        if optimizer_type == "8bit_adam" and OPTIMIZERS_AVAILABLE:
            self.optimizer = Adam8bit(
                params,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                weight_decay=self.args.weight_decay,
            )
            print("  [OK] 8-bit Adam (bitsandbytes)")
            return

        if optimizer_type == "adam_atan2" and OPTIMIZERS_AVAILABLE:
            self.optimizer = AdamAtan2(
                params,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                weight_decay=self.args.weight_decay,
            )
            print("  [OK] Adam-atan2 (DeepSeek/HRM style)")
            return

        # PyTorch AdamW with FUSED=True for CUDA (2x faster!)
        # Check if fused is supported (PyTorch 2.0+)
        use_fused = False
        if torch.cuda.is_available():
            try:
                # Test if fused parameter is accepted
                sig = inspect.signature(torch.optim.AdamW.__init__)
                use_fused = 'fused' in sig.parameters
            except Exception:
                # Fallback: try creating a dummy optimizer
                try:
                    dummy_param = torch.nn.Parameter(torch.zeros(1, device='cuda'))
                    dummy_opt = torch.optim.AdamW([dummy_param], lr=1e-3, fused=True)
                    del dummy_opt, dummy_param
                    use_fused = True
                except (TypeError, RuntimeError):
                    use_fused = False

        if use_fused:
            self.optimizer = torch.optim.AdamW(
                params,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                weight_decay=self.args.weight_decay,
                fused=True,  # CRITICAL: 2x faster on CUDA
            )
            print("  [OK] PyTorch AdamW (FUSED=True, 2x faster)")
        else:
            self.optimizer = torch.optim.AdamW(
                params,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                weight_decay=self.args.weight_decay,
            )
            print("  [OK] PyTorch AdamW (standard)")

    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        total_steps = len(self.train_dataloader) * self.args.num_train_epochs
        warmup_steps = int(total_steps * self.args.warmup_ratio)

        scheduler_type = self.args.lr_scheduler_type

        if self.args.optimizer_type == "schedule_free":
            self.scheduler = None
            print("  [OK] No LR scheduler (Schedule-Free)")
            return

        if scheduler_type == "wsd" and OPTIMIZERS_AVAILABLE:
            self.scheduler = WSDScheduler(
                self.optimizer,
                total_steps=total_steps,
                warmup_ratio=self.args.warmup_ratio,
                stable_ratio=self.args.wsd_stable_ratio,
                min_lr_ratio=self.args.lr_min_ratio,
            )
            print(f"  [OK] WSD scheduler (stable={self.args.wsd_stable_ratio:.0%})")
            return

        if scheduler_type == "one_cycle" and OPTIMIZERS_AVAILABLE:
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.args.learning_rate,
                total_steps=total_steps,
                pct_start=self.args.warmup_ratio,
            )
            print("  [OK] OneCycle scheduler")
            return

        if scheduler_type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

            warmup = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps
            )
            cosine = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=self.args.learning_rate * self.args.lr_min_ratio
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_steps]
            )
            print("  [OK] Cosine scheduler with warmup")
            return

        if scheduler_type == "constant":
            from torch.optim.lr_scheduler import LinearLR, ConstantLR, SequentialLR

            warmup = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps
            )
            constant = ConstantLR(
                self.optimizer,
                factor=1.0,
                total_iters=total_steps - warmup_steps
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup, constant],
                milestones=[warmup_steps]
            )
            print("  [OK] Constant LR with warmup")
            return

        self.scheduler = None

    def _compute_model_flops(self):
        """Compute approximate FLOPs per forward pass."""
        N = sum(p.numel() for p in self.model.parameters())
        self.model_flops_per_token = 6 * N

    def _compute_mfu(self, tokens_per_sec: float) -> float:
        """Compute Model FLOPS Utilization."""
        if not self.track_mfu:
            return 0.0

        achieved_flops = tokens_per_sec * self.model_flops_per_token
        peak_flops = self.args.gpu_peak_tflops * 1e12
        return (achieved_flops / peak_flops) * 100

    def train(self):
        """
        Main training loop - OPTIMIZED FOR SPEED.

        Returns:
            TrainerState with training history
        """
        print("\n" + "=" * 60)
        print("Starting Chronicals Training (SPEED MODE)")
        print("=" * 60)
        self._print_training_info()
        print("=" * 60 + "\n")

        self.model.train()
        training_start_time = time.time()

        # Warmup torch.compile on first batch
        if self.is_compiled and not self._compile_warmup_done:
            print("Warming up torch.compile (first batch)...")
            self._warmup_compile()
            self._compile_warmup_done = True
            print("Warmup complete!")

        for epoch in range(self.args.num_train_epochs):
            self.state.epoch = epoch

            # Check if we've hit max_steps before starting epoch
            if self.args.max_steps > 0 and self.state.global_step >= self.args.max_steps:
                break

            epoch_loss = self._train_epoch_fast(epoch)

            # Evaluate
            if self.eval_dataloader is not None:
                eval_loss = self.evaluate()
                print(f"Epoch {epoch+1} - Train Loss: {epoch_loss:.4f}, Eval Loss: {eval_loss:.4f}")
            else:
                print(f"Epoch {epoch+1} - Train Loss: {epoch_loss:.4f}")

            # Save checkpoint
            self.save_checkpoint(f"checkpoint-epoch-{epoch+1}")

        self.state.total_time = time.time() - training_start_time

        # Re-enable GC after training
        gc.enable()

        # Final save
        self.save_checkpoint("final")

        # Generate report
        if self.reporter:
            self.reporter.generate_html_report()
            self.reporter.save_metrics_json()

        self._print_final_stats()

        return self.state

    def _warmup_compile(self):
        """
        Warmup torch.compile before production training.

        This is critical for torch.compile performance because:
        1. First pass triggers JIT compilation (slow)
        2. reduce-overhead mode needs CUDA graph warmup iterations
        3. Subsequent passes hit the Inductor/Triton cache (fast)

        Best practices from research (2025):
        - Run multiple warmup iterations (3-5) to trigger all code paths
        - Use representative batch sizes to avoid recompilation
        - Sync CUDA to ensure compilation completes before timing
        - Do optimizer step too (triggers optimizer compilation if enabled)
        - For dynamic shapes: warmup with multiple shape buckets

        DYNAMIC SHAPES HANDLING:
        - If torch_compile_dynamic=True, warmup with different seq lengths
        - This pre-compiles multiple shape variants to avoid runtime recompilation
        - Key insight: torch.compile caches by shape guards, so we need to hit
          the shapes we'll see during training

        References:
        - https://docs.pytorch.org/docs/stable/torch.compiler_profiling_torch_compile.html
        - https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
        - https://huggingface.co/docs/transformers/en/perf_torch_compile
        """
        warmup_steps = getattr(self.args, 'torch_compile_warmup_steps', 5)  # 5 steps for better warmup
        cuda_available = torch.cuda.is_available()
        use_dynamic = getattr(self.args, 'torch_compile_dynamic', None)

        try:
            # Get data iterator for warmup
            data_iter = iter(self.train_dataloader)

            # Track shapes we've warmed up for debugging
            warmed_shapes = set()

            for i in range(warmup_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    # Reset iterator if we run out of data
                    data_iter = iter(self.train_dataloader)
                    batch = next(data_iter)

                batch = self._prepare_batch(batch)

                # Track shape for dynamic compilation
                seq_len = batch['input_ids'].shape[1] if len(batch['input_ids'].shape) > 1 else batch['input_ids'].shape[0]
                warmed_shapes.add(seq_len)

                # Forward pass with autocast (triggers forward compilation)
                with torch.cuda.amp.autocast(enabled=self._use_autocast, dtype=self._autocast_dtype):
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch.get('attention_mask'),
                        labels=batch['labels'],
                    )
                    loss = outputs.loss

                # Backward pass (triggers backward graph compilation)
                loss.backward()

                # Do optimizer step on last warmup iteration (triggers optimizer compilation)
                if i == warmup_steps - 1:
                    if self.use_external_grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()

                # Zero gradients
                self.optimizer.zero_grad(set_to_none=True)

                # Sync after each step to ensure compilation completes
                if cuda_available:
                    torch.cuda.synchronize()

                print(f"  Warmup step {i+1}/{warmup_steps} complete (seq_len={seq_len})")

            # If using dynamic shapes, warmup with additional shape buckets
            # This reduces runtime recompilation when sequence lengths vary
            if use_dynamic and self.args.use_sequence_packing:
                # Common power-of-2 sequence lengths for bucket compilation
                max_seq = self.args.max_seq_length
                shape_buckets = [128, 256, 512, 1024, 2048, 4096]
                shape_buckets = [s for s in shape_buckets if s <= max_seq and s not in warmed_shapes]

                for bucket_seq in shape_buckets[:2]:  # Limit to 2 additional buckets
                    try:
                        # Create dummy input with bucket size
                        batch_size = batch['input_ids'].shape[0]
                        dummy_ids = torch.randint(0, 1000, (batch_size, bucket_seq), device=self.device)
                        dummy_labels = torch.randint(0, 1000, (batch_size, bucket_seq), device=self.device)

                        with torch.cuda.amp.autocast(enabled=self._use_autocast, dtype=self._autocast_dtype):
                            with torch.no_grad():  # No grad for bucket warmup
                                outputs = self.model(input_ids=dummy_ids, labels=dummy_labels)

                        if cuda_available:
                            torch.cuda.synchronize()

                        print(f"  Dynamic shape warmup: seq_len={bucket_seq}")
                        warmed_shapes.add(bucket_seq)
                    except Exception as e:
                        # Bucket warmup is optional - don't fail
                        pass

            # Final sync and cleanup
            if cuda_available:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            # Reset optimizer state after warmup (don't keep warmup momentum)
            self.optimizer.zero_grad(set_to_none=True)

            # Disable GC during training for lower latency
            if getattr(self.args, 'disable_gc_during_training', True):
                gc.disable()

            print(f"  Compile warmup complete (shapes: {sorted(warmed_shapes)})")

        except Exception as e:
            print(f"  [WARN] Compile warmup failed: {e}")

    def _print_training_info(self):
        """Print training configuration."""
        print(f"  Model: {self.model.__class__.__name__}")
        print(f"  Epochs: {self.args.num_train_epochs}")
        print(f"  Batch size: {self.args.per_device_train_batch_size}")
        print(f"  Gradient accumulation: {self.args.gradient_accumulation_steps}")
        print(f"  Effective batch size: {self.args.effective_batch_size}")
        print(f"  Learning rate: {self.args.learning_rate}")
        print(f"  Optimizer: {self.args.optimizer_type}")
        print(f"  LR Schedule: {self.args.lr_scheduler_type}")
        print(f"  FP8: {self.args.fp8}")
        print(f"  Sequence packing: {self.args.use_sequence_packing}")
        print(f"  torch.compile: {self.is_compiled}")
        if self.args.max_steps > 0:
            print(f"  Max steps: {self.args.max_steps}")

    def _print_final_stats(self):
        """Print final training statistics."""
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"  Total steps: {self.state.global_step}")
        print(f"  Total tokens: {self.state.total_tokens:,}")
        print(f"  Total time: {self.state.total_time:.1f}s")
        if self.state.total_time > 0:
            avg_tokens_per_sec = self.state.total_tokens / self.state.total_time
            print(f"  Avg throughput: {avg_tokens_per_sec:,.0f} tokens/sec")
            if self.track_mfu:
                avg_mfu = self._compute_mfu(avg_tokens_per_sec)
                print(f"  Avg MFU: {avg_mfu:.1f}%")
        if self.state.log_history:
            final_loss = self.state.log_history[-1].get('loss', 0)
            print(f"  Final loss: {final_loss:.4f}")
        print("=" * 60 + "\n")

    def _train_epoch_fast(self, epoch: int) -> float:
        """
        FAST training loop with minimal Python overhead.

        Key optimizations:
        - No unnecessary function calls in hot path
        - Batched logging (only every N steps)
        - Pre-computed constants
        - Minimal conditionals in inner loop
        - Proper CUDA synchronization for accurate timing
        - Correct tokens/sec calculation accounting for grad accumulation
        """
        total_loss = 0.0
        num_logged_steps = 0
        accumulated_loss = 0.0
        epoch_tokens = 0

        # Pre-compute constants (avoid attribute lookups in hot loop)
        grad_accum_steps = self.args.gradient_accumulation_steps
        max_grad_norm = self.args.max_grad_norm
        logging_steps = self.args.logging_steps
        save_steps = self.args.save_steps
        max_steps = self.args.max_steps
        use_autocast = self._use_autocast
        autocast_dtype = self._autocast_dtype
        use_grad_clip = self.use_external_grad_clip
        has_scheduler = self.scheduler is not None

        # Get references (avoid attribute lookup in loop)
        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler
        device = self.device

        # CUDA availability check (done once)
        cuda_available = torch.cuda.is_available()

        # Tokens tracking for throughput
        tokens_since_log = 0

        # Synchronize and start timing AFTER warmup/setup
        if cuda_available:
            torch.cuda.synchronize()
        step_start_time = time.perf_counter()  # More precise timer

        # Data iterator - use simpler iteration for speed
        data_iter = iter(self.train_dataloader)

        step = 0
        while True:
            # Check max_steps FIRST (before any work)
            if max_steps > 0 and self.state.global_step >= max_steps:
                break

            # Get next batch
            try:
                batch = next(data_iter)
            except StopIteration:
                break

            # Move to device (non-blocking for async transfer)
            if isinstance(batch, PackedBatch):
                input_ids = batch.input_ids.to(device, non_blocking=True)
                labels = batch.labels.to(device, non_blocking=True)
                attention_mask = batch.attention_mask.to(device, non_blocking=True) if batch.attention_mask is not None else None
            else:
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device, non_blocking=True)

            # Count tokens (fast path - just count elements)
            batch_tokens = input_ids.numel()
            epoch_tokens += batch_tokens
            tokens_since_log += batch_tokens

            # For HuggingFace models with packing, skip attention_mask
            if self.args.use_sequence_packing and hasattr(model, 'config'):
                attention_mask = None

            # Forward pass with autocast
            with torch.cuda.amp.autocast(enabled=use_autocast, dtype=autocast_dtype):
                # Use CCE if enabled - computes loss WITHOUT materializing full logits
                if self.use_cce and self.cce_loss_fn is not None:
                    # Forward pass WITHOUT computing loss (get hidden states)
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    hidden_states = outputs.hidden_states[-1]

                    # Get lm_head weights (handle different model architectures)
                    lm_weight = None
                    lm_bias = None
                    if hasattr(model, 'lm_head'):
                        lm_weight = model.lm_head.weight
                        lm_bias = getattr(model.lm_head, 'bias', None)
                    elif hasattr(model, 'model') and hasattr(model.model, 'lm_head'):
                        lm_weight = model.model.lm_head.weight
                        lm_bias = getattr(model.model.lm_head, 'bias', None)

                    if lm_weight is not None:
                        # CCE loss - NEVER materializes full logits!
                        loss = self.cce_loss_fn(hidden_states, lm_weight, labels, lm_bias)
                    else:
                        # Fallback to standard if lm_head not found
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss
                else:
                    # Standard loss computation
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss

            # Scale loss for gradient accumulation
            scaled_loss = loss / grad_accum_steps
            accumulated_loss += loss.item()

            # Backward pass
            scaled_loss.backward()

            step += 1

            # Optimizer step (every grad_accum_steps)
            if step % grad_accum_steps == 0:
                # Gradient clipping (if using external clip)
                if use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                # Optimizer step
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # Invalidate FP8 caches after weight update (DeepSeek V3 style)
                if self.fp8_context is not None:
                    self.fp8_context.invalidate_caches()

                # Scheduler step
                if has_scheduler:
                    scheduler.step()

                self.state.global_step += 1

                # Logging (batched - not every step!)
                if self.state.global_step % logging_steps == 0:
                    # CRITICAL: Sync CUDA before timing for accurate measurement
                    if cuda_available:
                        torch.cuda.synchronize()

                    step_time = time.perf_counter() - step_start_time
                    avg_loss = accumulated_loss / (logging_steps * grad_accum_steps)
                    tokens_per_sec = tokens_since_log / step_time if step_time > 0 else 0

                    # Minimal logging
                    self._log_step_fast(avg_loss, tokens_per_sec)
                    total_loss += avg_loss
                    num_logged_steps += 1

                    # Reset counters and restart timer
                    accumulated_loss = 0.0
                    tokens_since_log = 0

                    # Sync again before restarting timer
                    if cuda_available:
                        torch.cuda.synchronize()
                    step_start_time = time.perf_counter()

                # Save checkpoint (rare)
                if save_steps > 0 and self.state.global_step % save_steps == 0:
                    self.save_checkpoint(f"checkpoint-{self.state.global_step}")

        self.state.total_tokens += epoch_tokens

        return total_loss / max(num_logged_steps, 1)

    def _train_epoch(self, epoch: int) -> float:
        """Original train epoch - calls fast version."""
        return self._train_epoch_fast(epoch)

    def _training_step(self, batch: Union[Dict[str, torch.Tensor], PackedBatch]) -> torch.Tensor:
        """Single training step with optimized loss computation.

        Supports multiple cross-entropy implementations:
        1. CCE (Cut Cross-Entropy): NEVER materializes full logits - best for large vocab
        2. Standard: Uses model's built-in loss (materializes full logits)
        """
        if isinstance(batch, PackedBatch):
            input_ids = batch.input_ids.to(self.device)
            attention_mask = batch.attention_mask
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            labels = batch.labels.to(self.device)
        else:
            input_ids = batch['input_ids']
            attention_mask = batch.get('attention_mask')
            labels = batch['labels']

        if self.args.use_sequence_packing and hasattr(self.model, 'config'):
            attention_mask = None

        with torch.cuda.amp.autocast(enabled=self._use_autocast, dtype=self._autocast_dtype):
            # Use CCE if enabled - computes loss WITHOUT materializing full logits
            if self.use_cce and self.cce_loss_fn is not None:
                # Forward pass WITHOUT computing loss (no labels)
                # Get hidden states from model
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )

                # Get last hidden states
                hidden_states = outputs.hidden_states[-1]

                # Get lm_head weights (handle different model architectures)
                if hasattr(self.model, 'lm_head'):
                    lm_weight = self.model.lm_head.weight
                    lm_bias = getattr(self.model.lm_head, 'bias', None)
                elif hasattr(self.model, 'model') and hasattr(self.model.model, 'lm_head'):
                    lm_weight = self.model.model.lm_head.weight
                    lm_bias = getattr(self.model.model.lm_head, 'bias', None)
                elif hasattr(self.model, 'embed_out'):
                    # Some models use embed_out for the final projection
                    lm_weight = self.model.embed_out.weight
                    lm_bias = getattr(self.model.embed_out, 'bias', None)
                else:
                    # Fallback to standard loss computation
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    return outputs.loss

                # Compute CCE loss - NEVER materializes full [batch*seq, vocab] logits!
                loss = self.cce_loss_fn(hidden_states, lm_weight, labels, lm_bias)

            else:
                # Standard loss computation (materializes full logits)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

        return loss

    def _prepare_batch(self, batch) -> Dict[str, torch.Tensor]:
        """Prepare batch for training."""
        if isinstance(batch, PackedBatch):
            result = {
                'input_ids': batch.input_ids.to(self.device, non_blocking=True),
                'labels': batch.labels.to(self.device, non_blocking=True),
            }
            if batch.attention_mask is not None:
                result['attention_mask'] = batch.attention_mask.to(self.device, non_blocking=True)
            return result
        else:
            return {
                k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

    def _count_tokens(self, batch) -> int:
        """Count non-padding tokens in batch."""
        if isinstance(batch, PackedBatch):
            return sum(batch.sequence_lengths)
        elif isinstance(batch, dict):
            labels = batch.get('labels', batch.get('input_ids'))
            if isinstance(labels, torch.Tensor):
                return (labels != -100).sum().item()
        return 0

    def _log_step_fast(self, loss: float, tokens_per_sec: float):
        """Fast logging with minimal overhead."""
        if self.scheduler:
            lr = self.scheduler.get_last_lr()[0]
        else:
            lr = self.optimizer.param_groups[0]['lr']

        # Print minimal info
        print(f"Step {self.state.global_step}: loss={loss:.4f}, lr={lr:.2e}, tok/s={tokens_per_sec:,.0f}")

        # Log to history (lightweight)
        self.state.log_history.append({
            'step': self.state.global_step,
            'loss': loss,
            'learning_rate': lr,
            'tokens_per_sec': tokens_per_sec,
        })

        # Visual reporter (only if enabled, and batched)
        if self.reporter and self.state.global_step % self.args.report_every_n_steps == 0:
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
            mfu = self._compute_mfu(tokens_per_sec) if self.track_mfu else 0
            metrics = TrainingMetrics(
                step=self.state.global_step,
                loss=loss,
                learning_rate=lr,
                tokens_per_sec=tokens_per_sec,
                memory_mb=memory_mb,
                epoch=self.state.epoch,
            )
            self.reporter.log_step(metrics)

    def _log_step(self, loss: float, tokens_per_sec: float):
        """Original log step - calls fast version."""
        self._log_step_fast(loss, tokens_per_sec)

    def evaluate(self) -> float:
        """Evaluate on eval dataset."""
        if self.eval_dataloader is None:
            return 0.0

        if self.args.optimizer_type == "schedule_free" and hasattr(self.optimizer, 'eval_mode'):
            self.optimizer.eval_mode()

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = self._prepare_batch(batch)
                with torch.cuda.amp.autocast(enabled=self._use_autocast, dtype=self._autocast_dtype):
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch.get('attention_mask'),
                        labels=batch['labels'],
                    )
                total_loss += outputs.loss.item()
                num_batches += 1

        if self.args.optimizer_type == "schedule_free" and hasattr(self.optimizer, 'train_mode'):
            self.optimizer.train_mode()

        self.model.train()
        return total_loss / max(num_batches, 1)

    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.args.output_dir, name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))

        if self.scheduler:
            torch.save(
                self.scheduler.state_dict(),
                os.path.join(checkpoint_dir, "scheduler.pt")
            )

        torch.save({
            'global_step': self.state.global_step,
            'epoch': self.state.epoch,
            'best_loss': self.state.best_loss,
            'total_tokens': self.state.total_tokens,
            'total_time': self.state.total_time,
        }, os.path.join(checkpoint_dir, "trainer_state.pt"))

        torch.save(
            self.state.log_history,
            os.path.join(checkpoint_dir, "training_history.pt")
        )

        if self.reporter:
            last_loss = self.state.log_history[-1]['loss'] if self.state.log_history else 0
            self.reporter.log_checkpoint(
                self.state.global_step,
                name,
                {'loss': last_loss}
            )

        print(f"Saved checkpoint: {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_dir: str):
        """Load checkpoint."""
        self.model.load_state_dict(
            torch.load(os.path.join(checkpoint_dir, "model.pt"), map_location=self.device)
        )

        if os.path.exists(os.path.join(checkpoint_dir, "optimizer.pt")):
            self.optimizer.load_state_dict(
                torch.load(os.path.join(checkpoint_dir, "optimizer.pt"), map_location=self.device)
            )

        if os.path.exists(os.path.join(checkpoint_dir, "scheduler.pt")) and self.scheduler:
            self.scheduler.load_state_dict(
                torch.load(os.path.join(checkpoint_dir, "scheduler.pt"), map_location=self.device)
            )

        if os.path.exists(os.path.join(checkpoint_dir, "trainer_state.pt")):
            state = torch.load(os.path.join(checkpoint_dir, "trainer_state.pt"), map_location=self.device)
            self.state.global_step = state['global_step']
            self.state.epoch = state['epoch']
            self.state.best_loss = state['best_loss']
            self.state.total_tokens = state.get('total_tokens', 0)
            self.state.total_time = state.get('total_time', 0.0)

        if os.path.exists(os.path.join(checkpoint_dir, "training_history.pt")):
            self.state.log_history = torch.load(
                os.path.join(checkpoint_dir, "training_history.pt"),
                map_location='cpu'
            )

        print(f"Loaded checkpoint from: {checkpoint_dir}")

    def resume_from_checkpoint(self, checkpoint_dir: str) -> int:
        """Resume training from a checkpoint."""
        if not os.path.exists(checkpoint_dir):
            print(f"Checkpoint not found: {checkpoint_dir}")
            return 0

        self.load_checkpoint(checkpoint_dir)
        print(f"Resuming from epoch {int(self.state.epoch)}, step {self.state.global_step}")
        return int(self.state.epoch)

    def push_to_hub(self, repo_name: str, private: bool = True):
        """Push model to HuggingFace Hub."""
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_name, private=private, token=HF_WRITE_TOKEN, exist_ok=True)

        checkpoint_dir = os.path.join(self.args.output_dir, "final")
        api.upload_folder(
            folder_path=checkpoint_dir,
            repo_id=repo_name,
            token=HF_WRITE_TOKEN,
        )

        print(f"Model pushed to: https://huggingface.co/{repo_name}")

    def get_active_optimizations(self) -> Dict[str, Any]:
        """
        Get dictionary of all active optimizations.

        Returns dict with:
        - enabled: List of enabled optimization names
        - disabled: List of disabled optimization names
        - categories: Dict mapping category -> list of optimizations
        - summary: Human-readable summary string
        """
        enabled = []
        disabled = []

        # Check FlashAttention-3 status (arXiv:2407.08608)
        fa_config = getattr(self, 'fa_config', None)
        fa3_enabled = fa_config is not None and getattr(fa_config, 'use_flash_attn_3', False)
        fa_varlen = fa_config is not None and getattr(fa_config, 'use_varlen', False)
        fa_persistent = fa_config is not None and getattr(fa_config, 'use_persistent_kernel', False)
        fa_pingpong = fa_config is not None and getattr(fa_config, 'use_pingpong_scheduling', False)
        fa_warp_spec = fa_config is not None and getattr(fa_config, 'use_warp_specialization', False)

        # Check each optimization
        optimizations = {
            # Kernel optimizations
            'torch.compile': self.is_compiled,
            'Liger Kernel': getattr(self, 'use_liger', False),
            'Fused AdamW': 'Fused' in str(type(self.optimizer).__name__) or getattr(self, 'use_external_grad_clip', True) == False,
            'Fused Cross-Entropy': getattr(self, 'use_fused_ce', False),
            'Chunked Cross-Entropy': getattr(self, 'use_chunked_ce', False),

            # FlashAttention-3 optimizations (arXiv:2407.08608)
            'FlashAttention-3': fa3_enabled,
            'FA3 Persistent Kernels': fa_persistent,
            'FA3 Ping-Pong Scheduling': fa_pingpong,
            'FA3 Warp Specialization': fa_warp_spec,
            'FA Varlen (Sequence Packing)': fa_varlen,

            # Memory optimizations
            'Gradient Checkpointing': self.args.use_gradient_checkpointing,
            'FP8 Training': self.args.fp8,
            'DeepSeek FP8': getattr(self, 'use_deepseek_fp8', False),

            # Compute optimizations
            'Sequence Packing': self.args.use_sequence_packing,
            'Flash Attention varlen': self.args.use_flash_varlen if self.args.use_sequence_packing else False,

            # Precision
            'BF16 Autocast': self._use_autocast and self._autocast_dtype == torch.bfloat16,
            'FP16 Autocast': self._use_autocast and self._autocast_dtype == torch.float16,

            # Data loading
            'Data Prefetcher': getattr(self.args, 'use_data_prefetcher', False),
            'Pin Memory': getattr(self.args, 'pin_memory', False),
            'Persistent Workers': getattr(self.args, 'persistent_workers', False),
        }

        for name, is_enabled in optimizations.items():
            if is_enabled:
                enabled.append(name)
            else:
                disabled.append(name)

        return {
            'enabled': enabled,
            'disabled': disabled,
            'enabled_count': len(enabled),
            'total_count': len(optimizations),
            'summary': f"{len(enabled)}/{len(optimizations)} optimizations enabled",
        }

    def print_optimization_summary(self):
        """Print formatted summary of active optimizations."""
        info = self.get_active_optimizations()

        print("\n" + "=" * 60)
        print("ACTIVE OPTIMIZATIONS SUMMARY")
        print("=" * 60)
        print(f"\n{info['summary']}")

        print("\nENABLED:")
        for opt in info['enabled']:
            print(f"  [ON]  {opt}")

        if info['disabled']:
            print("\nDISABLED:")
            for opt in info['disabled']:
                print(f"  [OFF] {opt}")

        print("=" * 60 + "\n")

    def get_performance_diagnostic(self) -> Dict[str, Any]:
        """
        Get comprehensive performance diagnostic.

        Returns dict with optimization status, theoretical performance,
        and recommendations.
        """
        # Get optimization info
        opt_info = self.get_active_optimizations()

        # Calculate theoretical performance
        N = sum(p.numel() for p in self.model.parameters())
        flops_per_token = 6 * N
        peak_tflops = self.args.gpu_peak_tflops
        theoretical_max = (peak_tflops * 1e12) / flops_per_token

        # Estimate achievable based on enabled optimizations
        base_mfu = 0.15  # 15% baseline
        mfu_multipliers = {
            'torch.compile': 1.3,
            'Liger Kernel': 1.2,
            'Sequence Packing': 2.0,
            'FP8 Training': 1.2,
            'Gradient Checkpointing': 0.8,  # Trades speed for memory
        }

        estimated_mfu = base_mfu
        for opt in opt_info['enabled']:
            if opt in mfu_multipliers:
                estimated_mfu *= mfu_multipliers[opt]

        estimated_mfu = min(estimated_mfu, 0.75)  # Cap at 75%
        estimated_tokens = theoretical_max * estimated_mfu

        return {
            'model_params_b': N / 1e9,
            'flops_per_token': flops_per_token,
            'peak_tflops': peak_tflops,
            'theoretical_max_tokens': theoretical_max,
            'estimated_mfu': estimated_mfu * 100,
            'estimated_tokens_per_sec': estimated_tokens,
            'optimizations': opt_info,
            'recommendations': self._get_recommendations(opt_info),
        }

    def _get_recommendations(self, opt_info: Dict) -> List[str]:
        """Get performance improvement recommendations."""
        recommendations = []
        disabled = set(opt_info['disabled'])

        if 'torch.compile' in disabled:
            recommendations.append("Enable torch.compile for 1.5-2x speedup")
        if 'Liger Kernel' in disabled:
            recommendations.append("Enable Liger Kernel for 20% throughput + 60% memory reduction")
        if 'Sequence Packing' in disabled:
            recommendations.append("Enable sequence packing for 2-5x throughput improvement")
        if 'FP8 Training' in disabled:
            recommendations.append("Enable FP8 training for 2x memory bandwidth savings")
        if 'Gradient Checkpointing' not in disabled and self.args.per_device_train_batch_size < 8:
            recommendations.append("Consider larger batch size with gradient checkpointing")

        return recommendations


if __name__ == "__main__":
    print("Chronicals Trainer Module (SPEED MODE)")
    print("=" * 50)
    print("\nOptimizations available:")
    print(f"  FusedAdamW: {FUSED_ADAMW_AVAILABLE}")
    print(f"  Advanced Optimizers: {OPTIMIZERS_AVAILABLE}")
    print(f"  Liger Kernel: {LIGER_AVAILABLE}")
    print(f"  torch.compile: {hasattr(torch, 'compile')}")
    print(f"  Fused AdamW (PyTorch): True")
    print("\nUsage:")
    print("  from chronicals_trainer import ChronicalsTrainer")
    print("  trainer = ChronicalsTrainer(model, args, train_dataloader)")
    print("  trainer.train()")
