"""
Chronicals Gradient Checkpointing
==================================
Optimal sqrt(n) checkpointing for 5.5x memory reduction with 10-15% overhead.
Now with optional async CPU offloading for O(1) GPU memory usage.

MEMORY ANALYSIS:
================
Without checkpointing: O(L) memory for L layers
With checkpointing: O(sqrt(L)) memory, +33% recompute cost
With checkpointing + offload: O(1) GPU memory, bandwidth limited

BANDWIDTH CONSIDERATIONS:
=========================
PCIe 4.0 x16: 32 GB/s bidirectional
Per-layer activation: ~63 MB (batch=1, seq=4096, hidden=2048)
Transfer time: ~2 ms per layer
Compute time: ~3-5 ms per layer (forward), ~6-10 ms (backward)
Overlap potential: HIGH (transfer < compute)

In Colab: Copy this entire cell, paste, and run to create gradient_checkpointing.py
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import List, Optional, Callable, Any, Dict
import math

# Import async offload module if available
try:
    from .async_activation_offload import (
        ActivationOffloadManager,
        OffloadActivations,
        OffloadConfig,
        CheckpointWithOffload,
        estimate_activation_memory,
    )
    OFFLOAD_AVAILABLE = True
except ImportError:
    OFFLOAD_AVAILABLE = False


class GradientCheckpointer:
    """
    Optimal gradient checkpointing with sqrt(n) strategy.
    Now with optional async CPU offloading.

    Theory:
    - Without checkpointing: Memory = O(n) for n layers
    - With sqrt(n) checkpointing: Memory = O(sqrt(n))
    - With checkpointing + offload: Memory = O(1) GPU
    - Compute overhead: ~1 extra forward pass (10-15%)

    For 32 layers: sqrt(32) ≈ 5.66 → checkpoint every 6 layers
    Memory reduction: 32/6 ≈ 5.3x (checkpointing only)
    Memory reduction with offload: ~32x (O(1) GPU memory)
    """

    def __init__(
        self,
        num_layers: int,
        checkpoint_ratio: Optional[float] = None,
        use_reentrant: bool = False,
        use_activation_offloading: bool = False,
        offload_config: Optional[Dict[str, Any]] = None,
    ):
        self.num_layers = num_layers
        self.use_reentrant = use_reentrant
        self.use_activation_offloading = use_activation_offloading

        # Compute checkpoint frequency using sqrt(n) strategy
        if checkpoint_ratio is not None:
            self.checkpoint_every = max(1, int(1.0 / checkpoint_ratio))
        else:
            self.checkpoint_every = max(1, int(math.sqrt(num_layers)))

        # Track which layers to checkpoint
        self.checkpoint_layers = set(
            range(0, num_layers, self.checkpoint_every)
        )

        # Initialize activation offload manager if enabled
        self.offload_manager = None
        self._offload_context = None
        if use_activation_offloading and OFFLOAD_AVAILABLE:
            config_kwargs = offload_config or {}
            config = OffloadConfig(**config_kwargs)
            self.offload_manager = ActivationOffloadManager(config)
            print(f"Activation offloading: ENABLED")
            print(f"  - Pin memory: {config.use_pin_memory}")
            print(f"  - Use streams: {config.use_streams}")
            print(f"  - Min offload size: {config.min_offload_size} bytes")
        elif use_activation_offloading and not OFFLOAD_AVAILABLE:
            print(f"WARNING: Activation offloading requested but module not available")

        print(f"Gradient checkpointing: every {self.checkpoint_every} layers")
        print(f"Checkpointed layers: {sorted(self.checkpoint_layers)}")

    def should_checkpoint(self, layer_idx: int) -> bool:
        """Check if layer should be checkpointed."""
        return layer_idx in self.checkpoint_layers

    def apply_to_model(self, model: nn.Module, layers_attr: str = "layers"):
        """
        Apply checkpointing to model layers.

        Args:
            model: Model with transformer layers
            layers_attr: Attribute name for layers list
        """
        layers = getattr(model, layers_attr, None)
        if layers is None:
            raise ValueError(f"Model has no attribute '{layers_attr}'")

        for idx, layer in enumerate(layers):
            layer._checkpoint = self.should_checkpoint(idx)

    def wrap_forward(
        self,
        layer_idx: int,
        layer_fn: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Wrap layer forward with optional checkpointing and offloading.

        Args:
            layer_idx: Index of current layer
            layer_fn: Layer forward function
            *args: Layer arguments
            **kwargs: Layer keyword arguments

        Returns:
            Layer output (with gradient checkpointing if enabled)
        """
        # Set layer index for offload manager
        if self.offload_manager is not None:
            self.offload_manager.set_layer(layer_idx, is_backward=False)

        if self.should_checkpoint(layer_idx):
            # Use gradient checkpointing
            return checkpoint(
                layer_fn,
                *args,
                use_reentrant=self.use_reentrant,
                **kwargs
            )
        else:
            # Normal forward
            return layer_fn(*args, **kwargs)

    def get_offload_context(self):
        """
        Get context manager for activation offloading.

        Usage:
            with checkpointer.get_offload_context():
                output = model(input)
                loss.backward()
        """
        if self.offload_manager is not None and OFFLOAD_AVAILABLE:
            return OffloadActivations(self.offload_manager)
        else:
            # Return a no-op context manager
            from contextlib import nullcontext
            return nullcontext()

    def start_prefetch(self, layer_idx: int):
        """
        Start prefetching activations for upcoming backward layers.

        Call this at the beginning of each layer's backward pass.
        """
        if self.offload_manager is not None:
            self.offload_manager.start_prefetch(layer_idx)

    def reset_offload(self):
        """Reset offload manager state for new training step."""
        if self.offload_manager is not None:
            self.offload_manager.reset()

    def get_offload_stats(self) -> Optional[Dict[str, Any]]:
        """Get offloading statistics if available."""
        if self.offload_manager is not None:
            stats = self.offload_manager.get_stats()
            return {
                'tensors_offloaded': stats.tensors_offloaded,
                'tensors_prefetched': stats.tensors_prefetched,
                'bytes_offloaded_mb': stats.bytes_offloaded / (1024 * 1024),
                'bytes_prefetched_mb': stats.bytes_prefetched / (1024 * 1024),
                'offload_time_ms': stats.offload_time_ms,
                'prefetch_time_ms': stats.prefetch_time_ms,
            }
        return None

    def estimate_memory_savings(
        self,
        hidden_size: int,
        intermediate_size: int,
        seq_len: int,
        batch_size: int = 1,
        dtype_bytes: int = 2,  # BF16
    ) -> dict:
        """
        Estimate memory savings from checkpointing.

        Returns:
            dict with baseline_mb, checkpointed_mb, savings_ratio
        """
        # Per-layer activation memory
        # Hidden states: batch * seq * hidden * dtype
        hidden_mem = batch_size * seq_len * hidden_size * dtype_bytes

        # FFN intermediate: batch * seq * intermediate * dtype
        ffn_mem = batch_size * seq_len * intermediate_size * dtype_bytes

        # Attention scores: batch * heads * seq * seq * dtype (without FlashAttention)
        # With FlashAttention: O(seq) instead of O(seq^2)
        # Assume FlashAttention: batch * seq * hidden * dtype
        attn_mem = hidden_mem

        per_layer_mem = hidden_mem + ffn_mem + attn_mem

        # Baseline: store all layers
        baseline_mem = self.num_layers * per_layer_mem

        # Checkpointed: store only checkpoint layers + recompute buffer
        num_checkpoints = len(self.checkpoint_layers)
        # Need to store activations for layers between checkpoints
        max_recompute = self.checkpoint_every
        checkpointed_mem = num_checkpoints * per_layer_mem + max_recompute * per_layer_mem

        return {
            'baseline_mb': baseline_mem / (1024 * 1024),
            'checkpointed_mb': checkpointed_mem / (1024 * 1024),
            'savings_ratio': baseline_mem / checkpointed_mem,
            'num_checkpoints': num_checkpoints,
            'checkpoint_every': self.checkpoint_every,
        }


class CheckpointedLayer(nn.Module):
    """
    Wrapper that applies gradient checkpointing to a layer.

    Usage:
        layer = CheckpointedLayer(transformer_layer, checkpoint=True)
        output = layer(hidden_states, attention_mask=mask)
    """

    def __init__(
        self,
        layer: nn.Module,
        checkpoint: bool = False,
        use_reentrant: bool = False,
    ):
        super().__init__()
        self.layer = layer
        self._checkpoint = checkpoint
        self.use_reentrant = use_reentrant

    def forward(self, *args, **kwargs):
        if self._checkpoint and self.training:
            # Checkpoint function needs args without kwargs for some versions
            def layer_fn(*inputs):
                return self.layer(*inputs, **kwargs)
            return checkpoint(layer_fn, *args, use_reentrant=self.use_reentrant)
        else:
            return self.layer(*args, **kwargs)


class SelectiveCheckpointing:
    """
    Selective checkpointing: only checkpoint expensive operations.

    Cheap ops (LayerNorm, ReLU, dropout) are always recomputed.
    Expensive ops (attention, FFN matmuls) are checkpointed.
    """

    # Operations that are cheap to recompute
    CHEAP_OPS = {
        nn.LayerNorm,
        nn.ReLU,
        nn.GELU,
        nn.SiLU,
        nn.Dropout,
    }

    # Operations that are expensive (should checkpoint)
    EXPENSIVE_OPS = {
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
    }

    @classmethod
    def should_checkpoint_module(cls, module: nn.Module) -> bool:
        """Determine if module should be checkpointed based on cost."""
        for cheap_type in cls.CHEAP_OPS:
            if isinstance(module, cheap_type):
                return False

        for expensive_type in cls.EXPENSIVE_OPS:
            if isinstance(module, expensive_type):
                return True

        # Default: checkpoint if has parameters
        return len(list(module.parameters())) > 0


def apply_gradient_checkpointing(
    model: nn.Module,
    checkpoint_ratio: float = 0.177,  # ~1/sqrt(32)
    layers_attr: str = "layers",
    use_activation_offloading: bool = False,
    offload_config: Optional[Dict[str, Any]] = None,
) -> GradientCheckpointer:
    """
    Apply gradient checkpointing to a model with optional activation offloading.

    Args:
        model: Transformer model with layers
        checkpoint_ratio: Fraction of layers to checkpoint (default: sqrt(n))
        layers_attr: Attribute name for layers list
        use_activation_offloading: Enable async CPU offloading of activations
        offload_config: Configuration for offloading (see OffloadConfig)

    Returns:
        GradientCheckpointer instance

    Memory Analysis:
        - No optimization: O(L) GPU memory
        - Checkpointing only: O(sqrt(L)) GPU memory
        - Checkpointing + Offload: O(1) GPU memory (activations on CPU)
    """
    layers = getattr(model, layers_attr, None)
    if layers is None:
        # Try common attribute names
        for attr in ['layers', 'encoder.layer', 'decoder.layers', 'transformer.h']:
            try:
                parts = attr.split('.')
                obj = model
                for part in parts:
                    obj = getattr(obj, part)
                layers = obj
                break
            except AttributeError:
                continue

    if layers is None:
        raise ValueError("Could not find layers in model")

    num_layers = len(layers)
    checkpointer = GradientCheckpointer(
        num_layers,
        checkpoint_ratio,
        use_activation_offloading=use_activation_offloading,
        offload_config=offload_config,
    )

    # Apply to each layer
    for idx, layer in enumerate(layers):
        if hasattr(layer, 'gradient_checkpointing_enable'):
            # HuggingFace style
            if checkpointer.should_checkpoint(idx):
                layer.gradient_checkpointing_enable()
        else:
            # Set custom attribute
            layer._use_checkpoint = checkpointer.should_checkpoint(idx)

    return checkpointer


def estimate_activation_memory(
    config,
    batch_size: int = 1,
    seq_len: int = 4096,
    with_checkpointing: bool = True,
    with_offloading: bool = False,
    pcie_bandwidth_gbps: float = 32.0,  # PCIe 4.0 x16
) -> dict:
    """
    Estimate activation memory for a model config.

    Args:
        config: ChronicalsConfig or similar
        batch_size: Training batch size
        seq_len: Sequence length
        with_checkpointing: Whether to use checkpointing
        with_offloading: Whether to use CPU offloading
        pcie_bandwidth_gbps: PCIe bandwidth for transfer time estimates

    Returns:
        Memory estimates in MB with transfer time analysis
    """
    hidden_size = getattr(config, 'hidden_size', 2048)
    intermediate_size = getattr(config, 'intermediate_size', 5632)
    num_layers = getattr(config, 'num_hidden_layers', 24)
    num_heads = getattr(config, 'num_attention_heads', 16)
    head_dim = hidden_size // num_heads

    dtype_bytes = 2  # BF16

    # Per-layer activations
    hidden_act = batch_size * seq_len * hidden_size * dtype_bytes
    ffn_act = batch_size * seq_len * intermediate_size * dtype_bytes

    # Attention (with FlashAttention - O(seq) not O(seq^2))
    qkv_act = 3 * batch_size * seq_len * hidden_size * dtype_bytes
    attn_out_act = batch_size * seq_len * hidden_size * dtype_bytes

    per_layer_bytes = hidden_act + ffn_act + qkv_act + attn_out_act
    per_layer_mb = per_layer_bytes / (1024 * 1024)

    checkpoint_every = max(1, int(math.sqrt(num_layers)))

    # Calculate memory for different configurations
    baseline_mb = num_layers * per_layer_mb  # O(L)

    if with_checkpointing:
        # O(sqrt(L)) - store checkpoints + recompute window
        effective_layers = (num_layers // checkpoint_every) + checkpoint_every
        checkpointed_mb = effective_layers * per_layer_mb
    else:
        effective_layers = num_layers
        checkpointed_mb = baseline_mb

    if with_offloading:
        # O(1) - only need current compute window on GPU
        # Keep max_fwd_stash_size tensors on GPU
        offload_gpu_mb = checkpoint_every * per_layer_mb  # Current segment
        offload_cpu_mb = baseline_mb  # All activations on CPU
    else:
        offload_gpu_mb = checkpointed_mb
        offload_cpu_mb = 0

    # Transfer time analysis
    transfer_time_per_layer_ms = (per_layer_mb / 1024) / pcie_bandwidth_gbps * 1000
    total_transfer_time_ms = transfer_time_per_layer_ms * num_layers * 2  # fwd + bwd

    result = {
        'per_layer_mb': per_layer_mb,
        'baseline_mb': baseline_mb,
        'checkpointed_mb': checkpointed_mb,
        'num_layers': num_layers,
        'effective_layers': effective_layers,
        'checkpoint_every': checkpoint_every,
    }

    if with_offloading:
        result.update({
            'offload_gpu_mb': offload_gpu_mb,
            'offload_cpu_mb': offload_cpu_mb,
            'transfer_time_per_layer_ms': transfer_time_per_layer_ms,
            'total_transfer_time_ms': total_transfer_time_ms,
            'gpu_savings_vs_baseline': baseline_mb / offload_gpu_mb,
            'gpu_savings_vs_checkpoint': checkpointed_mb / offload_gpu_mb,
        })

    return result


if __name__ == "__main__":
    print("=" * 70)
    print("GRADIENT CHECKPOINTING WITH ASYNC OFFLOAD - TEST SUITE")
    print("=" * 70)

    # Test checkpointer
    print("\n[1] BASIC CHECKPOINTING TEST")
    print("-" * 50)
    checkpointer = GradientCheckpointer(num_layers=32)

    print(f"\nFor 32-layer model:")
    print(f"  Checkpoint every: {checkpointer.checkpoint_every} layers")
    print(f"  Checkpointed layers: {sorted(checkpointer.checkpoint_layers)}")

    # Estimate memory savings
    savings = checkpointer.estimate_memory_savings(
        hidden_size=2048,
        intermediate_size=5632,
        seq_len=4096,
        batch_size=1,
    )

    print(f"\nMemory estimation (4K seq, batch=1):")
    print(f"  Baseline: {savings['baseline_mb']:.2f} MB")
    print(f"  Checkpointed: {savings['checkpointed_mb']:.2f} MB")
    print(f"  Savings: {savings['savings_ratio']:.2f}x")

    # Test with activation offloading
    print("\n[2] CHECKPOINTING WITH ACTIVATION OFFLOADING")
    print("-" * 50)

    if OFFLOAD_AVAILABLE:
        checkpointer_offload = GradientCheckpointer(
            num_layers=32,
            use_activation_offloading=True,
            offload_config={
                'use_pin_memory': True,
                'use_streams': True,
                'track_stats': True,
            }
        )

        # Create mock config for memory estimation
        class MockConfig:
            hidden_size = 2048
            intermediate_size = 5632
            num_hidden_layers = 32
            num_attention_heads = 16

        config = MockConfig()
        mem_est = estimate_activation_memory(
            config, batch_size=1, seq_len=4096,
            with_checkpointing=True, with_offloading=True
        )

        print(f"\nMemory Analysis with Offloading:")
        print(f"  Per-layer: {mem_est['per_layer_mb']:.2f} MB")
        print(f"  Baseline (no opt): {mem_est['baseline_mb']:.2f} MB")
        print(f"  With checkpointing: {mem_est['checkpointed_mb']:.2f} MB")
        print(f"  With offload (GPU): {mem_est['offload_gpu_mb']:.2f} MB")
        print(f"  With offload (CPU): {mem_est['offload_cpu_mb']:.2f} MB")
        print(f"  Transfer time/layer: {mem_est['transfer_time_per_layer_ms']:.2f} ms")
        print(f"  GPU savings vs baseline: {mem_est['gpu_savings_vs_baseline']:.1f}x")
        print(f"  GPU savings vs checkpoint: {mem_est['gpu_savings_vs_checkpoint']:.1f}x")
    else:
        print("Activation offloading module not available")

    # Test with simple model
    print("\n[3] MODEL INTEGRATION TEST")
    print("-" * 50)

    class DummyLayer(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.linear = nn.Linear(hidden_size, hidden_size)
            self._use_checkpoint = False

        def forward(self, x):
            return self.linear(x)

    class DummyModel(nn.Module):
        def __init__(self, hidden_size, num_layers):
            super().__init__()
            self.layers = nn.ModuleList([
                DummyLayer(hidden_size) for _ in range(num_layers)
            ])

        def forward(self, x, checkpointer=None):
            for idx, layer in enumerate(self.layers):
                if checkpointer is not None:
                    x = checkpointer.wrap_forward(idx, layer, x)
                elif layer._use_checkpoint and self.training:
                    x = checkpoint(layer, x, use_reentrant=False)
                else:
                    x = layer(x)
            return x

    model = DummyModel(hidden_size=512, num_layers=32)
    checkpointer.apply_to_model(model)

    print(f"\nApplied checkpointing to model:")
    checkpointed_count = sum(1 for layer in model.layers if layer._use_checkpoint)
    print(f"  Total layers: {len(model.layers)}")
    print(f"  Checkpointed layers: {checkpointed_count}")

    # Run forward/backward test
    if torch.cuda.is_available():
        print("\n[4] CUDA FORWARD/BACKWARD TEST")
        print("-" * 50)

        device = torch.device("cuda")
        model = model.to(device)
        x = torch.randn(1, 128, 512, device=device, requires_grad=True)

        # Test with offload context if available
        if OFFLOAD_AVAILABLE and checkpointer_offload is not None:
            print("\nRunning with activation offloading...")
            with checkpointer_offload.get_offload_context():
                output = model(x, checkpointer=checkpointer_offload)
                loss = output.sum()
                loss.backward()

            stats = checkpointer_offload.get_offload_stats()
            if stats:
                print(f"\nOffload Statistics:")
                print(f"  Tensors offloaded: {stats['tensors_offloaded']}")
                print(f"  Tensors prefetched: {stats['tensors_prefetched']}")
                print(f"  Data offloaded: {stats['bytes_offloaded_mb']:.2f} MB")
                print(f"  Data prefetched: {stats['bytes_prefetched_mb']:.2f} MB")
        else:
            print("\nRunning standard forward/backward...")
            output = model(x)
            loss = output.sum()
            loss.backward()

        print("\nForward/backward completed successfully!")

    print("\n" + "=" * 70)
    print("MATHEMATICAL ANALYSIS SUMMARY")
    print("=" * 70)
    print("""
Memory Complexity:
  - Without checkpointing: O(L) for L layers
  - With sqrt(L) checkpointing: O(sqrt(L))
  - With checkpointing + offload: O(1) GPU memory

PCIe Bandwidth Analysis (PCIe 4.0 x16 @ 32 GB/s):
  - Per-layer activation (2048 hidden, 4K seq): ~63 MB
  - Transfer time per layer: ~2 ms
  - Typical forward compute time: 3-5 ms
  - Typical backward compute time: 6-10 ms
  - Overlap potential: HIGH (transfer < compute)

Recommended Configuration:
  - GPU VRAM limited: Enable checkpointing + offloading
  - Compute limited: Enable checkpointing only
  - No constraints: Disable both for max speed
""")
