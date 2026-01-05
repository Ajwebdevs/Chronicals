"""
CUDA Graph Manager for Chronicals Training
============================================
Manages CUDA graph capture and replay for maximum training throughput.

CUDA Graphs eliminate kernel launch overhead by recording GPU operations
once and replaying them, achieving near-zero CPU overhead for the training loop.

Key Features:
- Automatic warmup before capture
- Graph invalidation detection
- Multiple graphs for different batch sizes (shape bucketing)
- Seamless fallback when graphs can't be used
- Memory pool management for efficient allocation
- Integration with torch.compile reduce-overhead mode

Performance Benefits:
- 2-3x speedup for small batch training
- Near-zero Python overhead in hot path
- Reduced PCIe bandwidth from kernel launch elimination

Requirements:
- Static tensor shapes (or shape bucketing)
- No CPU synchronization in captured region
- No dynamic control flow

References:
- https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
- https://docs.pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html
- https://fireworks.ai/blog/speed-python-pick-two-how-cuda-graphs-enable-fast-python-code-for-deep-learning
"""

import torch
import torch.nn as nn
from torch.cuda import Stream
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
import warnings
import gc
import functools
from contextlib import contextmanager


@dataclass
class CUDAGraphConfig:
    """Configuration for CUDA Graph capture and replay."""

    # Warmup settings
    warmup_steps: int = 3  # Steps to run before capture

    # Capture settings
    capture_mode: str = "thread_local"  # "thread_local", "global", "relaxed"
    use_memory_pool: bool = True  # Use dedicated memory pool

    # Shape handling
    static_shapes: bool = True  # Require static shapes
    shape_buckets: List[int] = field(default_factory=lambda: [128, 256, 512, 1024, 2048, 4096])

    # Graph management
    max_cached_graphs: int = 16  # Maximum number of cached graph variants

    # Fallback behavior
    fallback_on_error: bool = True  # Fall back to eager on capture failure

    # Integration
    enable_with_torch_compile: bool = True  # Use with torch.compile reduce-overhead

    def __post_init__(self):
        # Validate capture mode
        valid_modes = ["thread_local", "global", "relaxed"]
        if self.capture_mode not in valid_modes:
            raise ValueError(f"capture_mode must be one of {valid_modes}")


@dataclass
class GraphStatistics:
    """Statistics for a captured CUDA graph."""
    shape_key: str
    capture_time_ms: float
    replay_count: int = 0
    total_replay_time_ms: float = 0.0
    memory_used_mb: float = 0.0

    @property
    def avg_replay_time_ms(self) -> float:
        if self.replay_count == 0:
            return 0.0
        return self.total_replay_time_ms / self.replay_count


class CUDAGraphMemoryPool:
    """
    Dedicated memory pool for CUDA graphs.

    Allocates a fixed memory pool upfront to avoid fragmentation
    and ensure deterministic memory layout across graph replays.
    """

    def __init__(self, pool_size_mb: float = 1024.0):
        self.pool_size_mb = pool_size_mb
        self.pool = None
        self._is_initialized = False

    def initialize(self, device: torch.device = None):
        """Initialize the memory pool."""
        if self._is_initialized:
            return

        if device is None:
            device = torch.device("cuda:0")

        if not torch.cuda.is_available():
            warnings.warn("CUDA not available, memory pool disabled")
            return

        # Get the default CUDA memory pool
        # Note: PyTorch 2.0+ manages pools automatically with graphs
        self.pool = torch.cuda.graph_pool_handle()
        self._is_initialized = True

    def get_pool_handle(self):
        """Get the memory pool handle for graph capture."""
        if not self._is_initialized:
            self.initialize()
        return self.pool

    def cleanup(self):
        """Clean up the memory pool."""
        self.pool = None
        self._is_initialized = False


class CapturedGraph:
    """
    A captured CUDA graph for a specific input shape.

    Stores the graph, static input/output buffers, and replay state.
    """

    def __init__(
        self,
        shape_key: str,
        graph: torch.cuda.CUDAGraph,
        static_inputs: Dict[str, torch.Tensor],
        static_outputs: Dict[str, torch.Tensor],
        memory_pool: Optional[CUDAGraphMemoryPool] = None,
    ):
        self.shape_key = shape_key
        self.graph = graph
        self.static_inputs = static_inputs
        self.static_outputs = static_outputs
        self.memory_pool = memory_pool

        # Statistics
        self.stats = GraphStatistics(
            shape_key=shape_key,
            capture_time_ms=0.0,
        )

        # State
        self._is_valid = True

    def replay(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Replay the captured graph with new inputs.

        Args:
            inputs: Dictionary of input tensors (must match static shapes)

        Returns:
            Dictionary of output tensors
        """
        if not self._is_valid:
            raise RuntimeError("Graph has been invalidated")

        # Copy inputs to static buffers
        for key, tensor in inputs.items():
            if key in self.static_inputs:
                self.static_inputs[key].copy_(tensor)

        # Replay the graph
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        self.graph.replay()
        end_event.record()

        torch.cuda.synchronize()

        # Update statistics
        self.stats.replay_count += 1
        self.stats.total_replay_time_ms += start_event.elapsed_time(end_event)

        # Return copies of static outputs
        return {key: tensor.clone() for key, tensor in self.static_outputs.items()}

    def invalidate(self):
        """Invalidate the graph (e.g., after weight update with incompatible changes)."""
        self._is_valid = False

    @property
    def is_valid(self) -> bool:
        return self._is_valid


class CUDAGraphManager:
    """
    Manages CUDA graph capture and replay for training.

    Features:
    - Automatic warmup before capture
    - Shape bucketing for variable input sizes
    - Graph caching and reuse
    - Seamless fallback when graphs can't be used
    - Integration with torch.compile

    Usage:
        manager = CUDAGraphManager(config)

        # Warmup phase
        for step in range(warmup_steps):
            manager.warmup_step(model, sample_batch)

        # Capture training step
        graph = manager.capture_training_step(model, sample_batch, optimizer)

        # Replay in training loop
        for batch in dataloader:
            outputs = manager.replay_or_run(model, batch, optimizer)
    """

    def __init__(
        self,
        config: Optional[CUDAGraphConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config or CUDAGraphConfig()
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Graph storage
        self._graphs: Dict[str, CapturedGraph] = {}
        self._warmup_count: Dict[str, int] = {}

        # Memory management
        self._memory_pool: Optional[CUDAGraphMemoryPool] = None
        if self.config.use_memory_pool:
            self._memory_pool = CUDAGraphMemoryPool()

        # State
        self._capture_enabled = True
        self._last_shape_key: Optional[str] = None

        # Stream for capture
        self._capture_stream: Optional[Stream] = None

    def _get_shape_key(self, batch: Dict[str, torch.Tensor]) -> str:
        """Generate a unique key for the input shapes."""
        shape_parts = []
        for key in sorted(batch.keys()):
            if isinstance(batch[key], torch.Tensor):
                shape_parts.append(f"{key}:{tuple(batch[key].shape)}")
        return "|".join(shape_parts)

    def _bucket_shape(self, seq_len: int) -> int:
        """Find the smallest bucket that fits the sequence length."""
        for bucket in self.config.shape_buckets:
            if seq_len <= bucket:
                return bucket
        return self.config.shape_buckets[-1]

    def _prepare_static_inputs(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Create static input buffers for graph capture."""
        static_inputs = {}
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor):
                static_inputs[key] = tensor.clone().to(self.device)
        return static_inputs

    def warmup_step(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[Callable] = None,
    ):
        """
        Run a warmup step before capture.

        Warmup is necessary to:
        1. Trigger any lazy initialization in PyTorch/CUDA
        2. Allow cuDNN autotuning to complete
        3. Populate kernel caches

        Args:
            model: The model to train
            batch: Input batch dictionary
            optimizer: Optional optimizer
            loss_fn: Optional custom loss function
        """
        shape_key = self._get_shape_key(batch)

        # Track warmup count per shape
        if shape_key not in self._warmup_count:
            self._warmup_count[shape_key] = 0

        model.train()

        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Forward pass
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(
                input_ids=batch.get("input_ids"),
                attention_mask=batch.get("attention_mask"),
                labels=batch.get("labels"),
            )
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

        # Backward pass
        loss.backward()

        # Optimizer step if provided
        if optimizer is not None:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        else:
            model.zero_grad(set_to_none=True)

        self._warmup_count[shape_key] += 1

    def is_warmed_up(self, batch: Dict[str, torch.Tensor]) -> bool:
        """Check if sufficient warmup has been done for this shape."""
        shape_key = self._get_shape_key(batch)
        return self._warmup_count.get(shape_key, 0) >= self.config.warmup_steps

    def capture_training_step(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        grad_scaler: Optional[torch.cuda.amp.GradScaler] = None,
        max_grad_norm: float = 1.0,
    ) -> Optional[CapturedGraph]:
        """
        Capture a full training step as a CUDA graph.

        Args:
            model: The model to train
            batch: Sample batch for capture (shapes must match future batches)
            optimizer: The optimizer
            grad_scaler: Optional gradient scaler for mixed precision
            max_grad_norm: Maximum gradient norm for clipping

        Returns:
            CapturedGraph object or None if capture failed
        """
        if not torch.cuda.is_available():
            warnings.warn("CUDA not available, graph capture disabled")
            return None

        shape_key = self._get_shape_key(batch)

        # Check if already captured
        if shape_key in self._graphs:
            return self._graphs[shape_key]

        # Check if warmed up
        if not self.is_warmed_up(batch):
            warnings.warn(f"Insufficient warmup for shape {shape_key}")
            return None

        # Prepare static input buffers
        static_inputs = self._prepare_static_inputs(batch)

        # Create CUDA graph
        graph = torch.cuda.CUDAGraph()

        # Initialize memory pool if needed
        pool_handle = None
        if self._memory_pool is not None:
            self._memory_pool.initialize(self.device)
            pool_handle = self._memory_pool.get_pool_handle()

        model.train()
        optimizer.zero_grad(set_to_none=True)

        try:
            # Capture the training step
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()

            # Side stream for capture
            capture_stream = torch.cuda.Stream()

            with torch.cuda.stream(capture_stream):
                # Wait for default stream
                capture_stream.wait_stream(torch.cuda.default_stream())

                # Begin capture
                with torch.cuda.graph(graph, pool=pool_handle):
                    # Forward pass with autocast
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        outputs = model(
                            input_ids=static_inputs.get("input_ids"),
                            attention_mask=static_inputs.get("attention_mask"),
                            labels=static_inputs.get("labels"),
                        )
                        loss = outputs.loss

                    # Backward pass
                    if grad_scaler is not None:
                        grad_scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    # Gradient clipping
                    if grad_scaler is not None:
                        grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                    # Optimizer step
                    if grad_scaler is not None:
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                    else:
                        optimizer.step()

                    optimizer.zero_grad(set_to_none=True)

            # Wait for capture to complete
            torch.cuda.default_stream().wait_stream(capture_stream)

            end_event.record()
            torch.cuda.synchronize()

            capture_time_ms = start_event.elapsed_time(end_event)

            # Create static output buffer
            static_outputs = {"loss": loss.detach().clone()}

            # Create captured graph object
            captured = CapturedGraph(
                shape_key=shape_key,
                graph=graph,
                static_inputs=static_inputs,
                static_outputs=static_outputs,
                memory_pool=self._memory_pool,
            )
            captured.stats.capture_time_ms = capture_time_ms
            captured.stats.memory_used_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

            # Cache the graph
            self._graphs[shape_key] = captured
            self._last_shape_key = shape_key

            # Manage cache size
            if len(self._graphs) > self.config.max_cached_graphs:
                self._evict_oldest_graph()

            return captured

        except Exception as e:
            if self.config.fallback_on_error:
                warnings.warn(f"CUDA graph capture failed: {e}. Falling back to eager mode.")
                return None
            else:
                raise

    def _evict_oldest_graph(self):
        """Evict the least recently used graph."""
        if not self._graphs:
            return

        # Find graph with lowest replay count
        oldest_key = min(
            self._graphs.keys(),
            key=lambda k: self._graphs[k].stats.replay_count
        )

        if oldest_key != self._last_shape_key:  # Don't evict the most recent
            del self._graphs[oldest_key]

    def replay(self, batch: Dict[str, torch.Tensor]) -> Optional[Dict[str, torch.Tensor]]:
        """
        Replay a captured graph if available.

        Args:
            batch: Input batch

        Returns:
            Output dictionary or None if no graph available
        """
        shape_key = self._get_shape_key(batch)

        if shape_key not in self._graphs:
            return None

        captured = self._graphs[shape_key]
        if not captured.is_valid:
            return None

        # Prepare inputs on device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                  for k, v in batch.items()}

        return captured.replay(inputs)

    def replay_or_run(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        grad_scaler: Optional[torch.cuda.amp.GradScaler] = None,
        max_grad_norm: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Replay graph if available, otherwise run eager forward/backward.

        This is the main entry point for the training loop.

        Args:
            model: The model
            batch: Input batch
            optimizer: Optimizer
            grad_scaler: Optional gradient scaler
            max_grad_norm: Gradient clipping norm

        Returns:
            Dictionary with "loss" key
        """
        # Try replay first
        outputs = self.replay(batch)
        if outputs is not None:
            return outputs

        # Fall back to eager execution
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        model.train()

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(
                input_ids=batch.get("input_ids"),
                attention_mask=batch.get("attention_mask"),
                labels=batch.get("labels"),
            )
            loss = outputs.loss

        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
        else:
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        if grad_scaler is not None:
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        return {"loss": loss.detach()}

    def invalidate_all_graphs(self):
        """Invalidate all cached graphs (call after model structure changes)."""
        for graph in self._graphs.values():
            graph.invalidate()
        self._graphs.clear()

    def get_statistics(self) -> Dict[str, GraphStatistics]:
        """Get statistics for all cached graphs."""
        return {key: graph.stats for key, graph in self._graphs.items()}

    def print_statistics(self):
        """Print formatted statistics."""
        stats = self.get_statistics()

        if not stats:
            print("No CUDA graphs captured")
            return

        print("\n" + "=" * 60)
        print("CUDA Graph Statistics")
        print("=" * 60)

        for shape_key, stat in stats.items():
            print(f"\n  Shape: {shape_key}")
            print(f"    Capture time: {stat.capture_time_ms:.2f} ms")
            print(f"    Replay count: {stat.replay_count}")
            print(f"    Avg replay time: {stat.avg_replay_time_ms:.3f} ms")
            print(f"    Memory used: {stat.memory_used_mb:.1f} MB")

        print("=" * 60)


def create_cuda_graph_training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    sample_batch: Dict[str, torch.Tensor],
    warmup_steps: int = 3,
    max_grad_norm: float = 1.0,
) -> Tuple[Callable, Optional[CapturedGraph]]:
    """
    Convenience function to create a CUDA graph-accelerated training step.

    Args:
        model: The model
        optimizer: The optimizer
        sample_batch: Sample batch for warmup/capture
        warmup_steps: Number of warmup steps
        max_grad_norm: Gradient clipping norm

    Returns:
        Tuple of (training_step_function, captured_graph)
    """
    manager = CUDAGraphManager(
        CUDAGraphConfig(warmup_steps=warmup_steps)
    )

    # Warmup
    for _ in range(warmup_steps):
        manager.warmup_step(model, sample_batch, optimizer)

    # Capture
    captured = manager.capture_training_step(
        model, sample_batch, optimizer, max_grad_norm=max_grad_norm
    )

    # Create training step function
    def training_step(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return manager.replay_or_run(
            model, batch, optimizer, max_grad_norm=max_grad_norm
        )

    return training_step, captured


# Decorator for easy CUDA graph capture
def cuda_graph_capture(warmup_steps: int = 3):
    """
    Decorator to capture a function as a CUDA graph.

    Usage:
        @cuda_graph_capture(warmup_steps=3)
        def training_step(model, batch, optimizer):
            ...
    """
    def decorator(fn):
        captured_graph = None
        warmup_count = 0
        static_inputs = None
        static_outputs = None
        graph = None

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            nonlocal captured_graph, warmup_count, static_inputs, static_outputs, graph

            if not torch.cuda.is_available():
                return fn(*args, **kwargs)

            # Warmup phase
            if warmup_count < warmup_steps:
                warmup_count += 1
                return fn(*args, **kwargs)

            # Capture phase
            if graph is None:
                graph = torch.cuda.CUDAGraph()

                # Run once to create static buffers
                static_outputs = fn(*args, **kwargs)

                # Capture
                with torch.cuda.graph(graph):
                    static_outputs = fn(*args, **kwargs)

            # Replay
            graph.replay()
            return static_outputs

        return wrapper
    return decorator


if __name__ == "__main__":
    print("CUDA Graph Manager for Chronicals Training")
    print("=" * 50)

    if torch.cuda.is_available():
        print(f"CUDA Available: {torch.cuda.get_device_name(0)}")

        # Test basic functionality
        config = CUDAGraphConfig()
        manager = CUDAGraphManager(config)

        print(f"\nConfig:")
        print(f"  Warmup steps: {config.warmup_steps}")
        print(f"  Capture mode: {config.capture_mode}")
        print(f"  Shape buckets: {config.shape_buckets}")
        print(f"  Max cached graphs: {config.max_cached_graphs}")

        # Create a simple model for testing
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(512, 512)

            def forward(self, input_ids, attention_mask=None, labels=None):
                x = self.linear(input_ids.float())
                loss = x.mean()
                return type('Outputs', (), {'loss': loss})()

        model = SimpleModel().cuda()
        optimizer = torch.optim.AdamW(model.parameters())

        # Create sample batch
        batch = {
            "input_ids": torch.randint(0, 100, (2, 512)).cuda(),
            "labels": torch.randint(0, 100, (2, 512)).cuda(),
        }

        # Warmup
        print("\nWarming up...")
        for i in range(3):
            manager.warmup_step(model, batch, optimizer)
            print(f"  Warmup step {i+1}/3")

        # Capture
        print("\nCapturing graph...")
        captured = manager.capture_training_step(model, batch, optimizer)

        if captured:
            print(f"  Capture time: {captured.stats.capture_time_ms:.2f} ms")

            # Test replay
            print("\nReplaying graph...")
            for i in range(5):
                outputs = manager.replay(batch)
                print(f"  Replay {i+1}: loss = {outputs['loss'].item():.4f}")

            manager.print_statistics()
        else:
            print("  Graph capture failed or disabled")

    else:
        print("CUDA not available")
