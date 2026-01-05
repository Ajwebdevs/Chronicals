%%writefile profiling_utils.py
"""
Profiling Utilities for Chronicals Training
=============================================
Built-in profiling for identifying bottlenecks and measuring performance.

Features:
- MFU (Model FLOPs Utilization) tracking
- Forward/backward/optimizer time breakdown
- Memory usage over time
- Kernel launch overhead measurement
- Performance regression detection
- Integration with PyTorch Profiler

References:
- https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/
"""

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
import time
import json
import os
from contextlib import contextmanager
from collections import deque
import warnings


# GPU specifications for common GPUs (peak TFLOPs at BF16)
GPU_SPECS = {
    # NVIDIA Hopper
    "H100": {"bf16_tflops": 989.0, "fp8_tflops": 1979.0, "memory_gb": 80, "bandwidth_tb_s": 3.35},
    "H200": {"bf16_tflops": 989.0, "fp8_tflops": 1979.0, "memory_gb": 141, "bandwidth_tb_s": 4.8},
    "H800": {"bf16_tflops": 989.0, "fp8_tflops": 1979.0, "memory_gb": 80, "bandwidth_tb_s": 3.35},

    # NVIDIA Ampere
    "A100": {"bf16_tflops": 312.0, "fp8_tflops": 624.0, "memory_gb": 80, "bandwidth_tb_s": 2.04},
    "A100-40GB": {"bf16_tflops": 312.0, "fp8_tflops": 624.0, "memory_gb": 40, "bandwidth_tb_s": 1.56},
    "A6000": {"bf16_tflops": 155.0, "fp8_tflops": 310.0, "memory_gb": 48, "bandwidth_tb_s": 0.768},

    # NVIDIA Ada Lovelace
    "L40": {"bf16_tflops": 181.0, "fp8_tflops": 362.0, "memory_gb": 48, "bandwidth_tb_s": 0.864},
    "L4": {"bf16_tflops": 121.0, "fp8_tflops": 242.0, "memory_gb": 24, "bandwidth_tb_s": 0.3},
    "RTX 4090": {"bf16_tflops": 330.0, "fp8_tflops": 660.0, "memory_gb": 24, "bandwidth_tb_s": 1.01},

    # NVIDIA Volta
    "V100": {"bf16_tflops": 125.0, "fp8_tflops": 0, "memory_gb": 32, "bandwidth_tb_s": 0.9},

    # Default fallback
    "default": {"bf16_tflops": 100.0, "fp8_tflops": 200.0, "memory_gb": 16, "bandwidth_tb_s": 0.5},
}


def detect_gpu_specs() -> Dict[str, Any]:
    """
    Detect GPU specifications from the current device.

    Returns:
        Dictionary with GPU specs (tflops, memory, bandwidth)
    """
    if not torch.cuda.is_available():
        return GPU_SPECS["default"]

    gpu_name = torch.cuda.get_device_name(0).lower()
    props = torch.cuda.get_device_properties(0)

    # Match GPU name to specs
    for key, specs in GPU_SPECS.items():
        if key.lower() in gpu_name:
            result = specs.copy()
            result["name"] = torch.cuda.get_device_name(0)
            result["detected"] = True
            return result

    # Fallback: estimate from compute capability
    cc = props.major * 10 + props.minor

    # Rough estimates based on architecture
    if cc >= 90:  # Hopper
        base = GPU_SPECS["H100"]
    elif cc >= 80:  # Ampere
        base = GPU_SPECS["A100"]
    elif cc >= 75:  # Turing/Ada
        base = GPU_SPECS["L4"]
    else:
        base = GPU_SPECS["V100"]

    result = base.copy()
    result["name"] = torch.cuda.get_device_name(0)
    result["detected"] = False
    result["memory_gb"] = props.total_memory / (1024**3)

    return result


@dataclass
class TimingBreakdown:
    """Breakdown of time spent in different phases."""
    forward_ms: float = 0.0
    backward_ms: float = 0.0
    optimizer_ms: float = 0.0
    data_loading_ms: float = 0.0
    other_ms: float = 0.0
    total_ms: float = 0.0

    @property
    def forward_pct(self) -> float:
        return (self.forward_ms / self.total_ms * 100) if self.total_ms > 0 else 0

    @property
    def backward_pct(self) -> float:
        return (self.backward_ms / self.total_ms * 100) if self.total_ms > 0 else 0

    @property
    def optimizer_pct(self) -> float:
        return (self.optimizer_ms / self.total_ms * 100) if self.total_ms > 0 else 0


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""
    step: int
    allocated_mb: float
    reserved_mb: float
    peak_allocated_mb: float
    timestamp: float = 0.0

    @classmethod
    def capture(cls, step: int) -> "MemorySnapshot":
        """Capture current memory state."""
        if not torch.cuda.is_available():
            return cls(step=step, allocated_mb=0, reserved_mb=0, peak_allocated_mb=0)

        return cls(
            step=step,
            allocated_mb=torch.cuda.memory_allocated() / 1024 / 1024,
            reserved_mb=torch.cuda.memory_reserved() / 1024 / 1024,
            peak_allocated_mb=torch.cuda.max_memory_allocated() / 1024 / 1024,
            timestamp=time.time(),
        )


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for a training run."""
    # Throughput
    tokens_per_second: float = 0.0
    samples_per_second: float = 0.0
    steps_per_second: float = 0.0

    # Efficiency
    mfu_percent: float = 0.0  # Model FLOPs Utilization
    gpu_utilization_percent: float = 0.0
    memory_efficiency_percent: float = 0.0

    # Timing
    timing: TimingBreakdown = field(default_factory=TimingBreakdown)

    # Memory
    peak_memory_mb: float = 0.0
    average_memory_mb: float = 0.0

    # Loss
    final_loss: float = 0.0
    loss_history: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tokens_per_second": self.tokens_per_second,
            "samples_per_second": self.samples_per_second,
            "steps_per_second": self.steps_per_second,
            "mfu_percent": self.mfu_percent,
            "gpu_utilization_percent": self.gpu_utilization_percent,
            "memory_efficiency_percent": self.memory_efficiency_percent,
            "peak_memory_mb": self.peak_memory_mb,
            "average_memory_mb": self.average_memory_mb,
            "final_loss": self.final_loss,
            "timing": {
                "forward_ms": self.timing.forward_ms,
                "backward_ms": self.timing.backward_ms,
                "optimizer_ms": self.timing.optimizer_ms,
                "total_ms": self.timing.total_ms,
            },
        }


class MFUTracker:
    """
    Model FLOPs Utilization (MFU) tracker.

    MFU measures what fraction of the GPU's peak theoretical FLOPs
    are actually being used for model computation.

    MFU = (Model FLOPs / Step Time) / Peak GPU TFLOPs

    For training:
    - Model FLOPs = 6 * N * T (N = params, T = tokens)
    - 2 for forward, 4 for backward (gradient for weights AND activations)
    """

    def __init__(
        self,
        model: nn.Module,
        gpu_peak_tflops: Optional[float] = None,
    ):
        """
        Initialize MFU tracker.

        Args:
            model: The model being trained
            gpu_peak_tflops: Peak GPU TFLOPs (auto-detected if None)
        """
        self.model = model
        self.num_params = sum(p.numel() for p in model.parameters())

        # Get GPU specs
        gpu_specs = detect_gpu_specs()
        self.gpu_peak_tflops = gpu_peak_tflops or gpu_specs["bf16_tflops"]
        self.gpu_name = gpu_specs.get("name", "Unknown")

        # FLOPs per token (6N for training)
        self.flops_per_token = 6 * self.num_params

        # History for averaging
        self._history: deque = deque(maxlen=100)

    def compute_mfu(self, tokens: int, elapsed_sec: float) -> float:
        """
        Compute MFU for a training step.

        Args:
            tokens: Number of tokens processed
            elapsed_sec: Time elapsed in seconds

        Returns:
            MFU as percentage (0-100)
        """
        if elapsed_sec <= 0:
            return 0.0

        # Achieved TFLOPs
        achieved_flops = self.flops_per_token * tokens
        achieved_tflops = achieved_flops / elapsed_sec / 1e12

        # MFU
        mfu = (achieved_tflops / self.gpu_peak_tflops) * 100

        self._history.append(mfu)
        return mfu

    def get_average_mfu(self) -> float:
        """Get average MFU over history."""
        if not self._history:
            return 0.0
        return sum(self._history) / len(self._history)

    def get_summary(self) -> Dict[str, Any]:
        """Get MFU summary."""
        return {
            "num_params": self.num_params,
            "params_b": self.num_params / 1e9,
            "flops_per_token": self.flops_per_token,
            "gpu_peak_tflops": self.gpu_peak_tflops,
            "gpu_name": self.gpu_name,
            "current_mfu": self._history[-1] if self._history else 0.0,
            "average_mfu": self.get_average_mfu(),
        }


class ThroughputMonitor:
    """
    Monitors training throughput over time.

    Tracks tokens/sec, samples/sec, and identifies performance regressions.
    """

    def __init__(
        self,
        window_size: int = 100,
        regression_threshold_pct: float = 10.0,
    ):
        """
        Initialize throughput monitor.

        Args:
            window_size: Number of steps for moving average
            regression_threshold_pct: Threshold for regression detection
        """
        self.window_size = window_size
        self.regression_threshold = regression_threshold_pct

        # History
        self._tokens_history: deque = deque(maxlen=window_size)
        self._time_history: deque = deque(maxlen=window_size)
        self._throughput_history: deque = deque(maxlen=window_size)

        # Baseline for regression detection
        self._baseline_throughput: Optional[float] = None
        self._baseline_set_step: int = 0

        # State
        self._total_tokens: int = 0
        self._total_time: float = 0.0
        self._start_time: float = time.time()

    def record_step(self, tokens: int, elapsed_sec: float):
        """Record a training step."""
        throughput = tokens / elapsed_sec if elapsed_sec > 0 else 0

        self._tokens_history.append(tokens)
        self._time_history.append(elapsed_sec)
        self._throughput_history.append(throughput)

        self._total_tokens += tokens
        self._total_time += elapsed_sec

    def set_baseline(self, step: int):
        """Set baseline throughput for regression detection."""
        if self._throughput_history:
            self._baseline_throughput = self.get_average_throughput()
            self._baseline_set_step = step

    def check_regression(self) -> Tuple[bool, float]:
        """
        Check for performance regression.

        Returns:
            (is_regressed, regression_percent)
        """
        if self._baseline_throughput is None or len(self._throughput_history) < 10:
            return False, 0.0

        current = self.get_average_throughput()
        regression_pct = ((self._baseline_throughput - current) / self._baseline_throughput) * 100

        is_regressed = regression_pct > self.regression_threshold
        return is_regressed, regression_pct

    def get_average_throughput(self) -> float:
        """Get average throughput over window."""
        if not self._throughput_history:
            return 0.0
        return sum(self._throughput_history) / len(self._throughput_history)

    def get_overall_throughput(self) -> float:
        """Get overall throughput since start."""
        if self._total_time <= 0:
            return 0.0
        return self._total_tokens / self._total_time

    def get_statistics(self) -> Dict[str, float]:
        """Get throughput statistics."""
        if not self._throughput_history:
            return {"mean": 0, "min": 0, "max": 0, "std": 0}

        history = list(self._throughput_history)
        mean = sum(history) / len(history)
        variance = sum((x - mean) ** 2 for x in history) / len(history)

        return {
            "mean": mean,
            "min": min(history),
            "max": max(history),
            "std": variance ** 0.5,
            "total_tokens": self._total_tokens,
            "total_time": self._total_time,
        }


class PerformanceProfiler:
    """
    Built-in profiling for identifying training bottlenecks.

    Tracks:
    - Forward/backward/optimizer time breakdown
    - Memory usage over time
    - MFU (Model FLOPs Utilization)
    - Kernel launch overhead
    - Performance regression detection
    """

    def __init__(
        self,
        model: nn.Module,
        output_dir: str = "./profiler_output",
        gpu_peak_tflops: Optional[float] = None,
        enable_memory_tracking: bool = True,
        enable_timing_breakdown: bool = True,
        regression_threshold_pct: float = 10.0,
    ):
        """
        Initialize profiler.

        Args:
            model: The model being trained
            output_dir: Directory for profiler output
            gpu_peak_tflops: Peak GPU TFLOPs (auto-detected if None)
            enable_memory_tracking: Track memory usage
            enable_timing_breakdown: Track time breakdown
            regression_threshold_pct: Threshold for regression alerts
        """
        self.model = model
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Sub-trackers
        self.mfu_tracker = MFUTracker(model, gpu_peak_tflops)
        self.throughput_monitor = ThroughputMonitor(
            regression_threshold_pct=regression_threshold_pct
        )

        # Tracking flags
        self.enable_memory_tracking = enable_memory_tracking
        self.enable_timing_breakdown = enable_timing_breakdown

        # History
        self._memory_snapshots: List[MemorySnapshot] = []
        self._timing_breakdowns: List[TimingBreakdown] = []

        # CUDA events for timing
        self._events: Dict[str, torch.cuda.Event] = {}
        self._timing_stack: List[Tuple[str, float]] = []

        # State
        self._step = 0
        self._epoch = 0
        self._is_profiling = False

    @contextmanager
    def profile_step(self, step: int):
        """
        Context manager to profile a training step.

        Usage:
            with profiler.profile_step(step):
                # training step code
        """
        self._step = step
        self._is_profiling = True

        # Reset events
        self._events.clear()
        self._timing_stack.clear()

        # Capture start memory
        if self.enable_memory_tracking:
            start_memory = MemorySnapshot.capture(step)

        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start_time = time.perf_counter()
            yield
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = time.perf_counter() - start_time

        finally:
            self._is_profiling = False

            # Capture end memory
            if self.enable_memory_tracking:
                end_memory = MemorySnapshot.capture(step)
                self._memory_snapshots.append(end_memory)

    @contextmanager
    def profile_phase(self, phase_name: str):
        """
        Profile a specific phase (forward, backward, optimizer).

        Usage:
            with profiler.profile_phase("forward"):
                outputs = model(inputs)
        """
        if not self._is_profiling:
            yield
            return

        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            try:
                yield
            finally:
                end_event.record()
                torch.cuda.synchronize()
                elapsed_ms = start_event.elapsed_time(end_event)
                self._events[phase_name] = elapsed_ms
        else:
            start = time.perf_counter()
            try:
                yield
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                self._events[phase_name] = elapsed_ms

    def record_step(
        self,
        tokens: int,
        loss: float,
        elapsed_sec: float,
    ):
        """Record metrics for a training step."""
        # Record throughput
        self.throughput_monitor.record_step(tokens, elapsed_sec)

        # Compute MFU
        mfu = self.mfu_tracker.compute_mfu(tokens, elapsed_sec)

        # Store timing breakdown
        if self._events:
            breakdown = TimingBreakdown(
                forward_ms=self._events.get("forward", 0),
                backward_ms=self._events.get("backward", 0),
                optimizer_ms=self._events.get("optimizer", 0),
                total_ms=elapsed_sec * 1000,
            )
            breakdown.other_ms = breakdown.total_ms - (
                breakdown.forward_ms + breakdown.backward_ms + breakdown.optimizer_ms
            )
            self._timing_breakdowns.append(breakdown)

        self._step += 1

    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        metrics = PerformanceMetrics()

        # Throughput
        stats = self.throughput_monitor.get_statistics()
        metrics.tokens_per_second = stats["mean"]
        metrics.steps_per_second = 1.0 / (stats.get("mean", 1) / 1000) if stats.get("mean", 0) > 0 else 0

        # MFU
        mfu_summary = self.mfu_tracker.get_summary()
        metrics.mfu_percent = mfu_summary["average_mfu"]

        # Memory
        if self._memory_snapshots:
            latest = self._memory_snapshots[-1]
            metrics.peak_memory_mb = max(s.peak_allocated_mb for s in self._memory_snapshots)
            metrics.average_memory_mb = sum(s.allocated_mb for s in self._memory_snapshots) / len(self._memory_snapshots)

        # Timing
        if self._timing_breakdowns:
            latest = self._timing_breakdowns[-1]
            metrics.timing = latest

        return metrics

    def check_regression(self) -> Tuple[bool, float]:
        """Check for performance regression."""
        return self.throughput_monitor.check_regression()

    def set_baseline(self):
        """Set baseline for regression detection."""
        self.throughput_monitor.set_baseline(self._step)

    def print_summary(self):
        """Print performance summary."""
        metrics = self.get_current_metrics()
        mfu_summary = self.mfu_tracker.get_summary()

        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)

        print(f"\n  GPU: {mfu_summary['gpu_name']}")
        print(f"  Model Parameters: {mfu_summary['params_b']:.2f}B")
        print(f"  Peak GPU TFLOPs: {mfu_summary['gpu_peak_tflops']:.1f}")

        print(f"\n  Throughput:")
        print(f"    Current: {metrics.tokens_per_second:,.0f} tokens/sec")
        print(f"    MFU: {metrics.mfu_percent:.1f}%")

        if self._timing_breakdowns:
            print(f"\n  Timing Breakdown:")
            print(f"    Forward:   {metrics.timing.forward_ms:.2f} ms ({metrics.timing.forward_pct:.1f}%)")
            print(f"    Backward:  {metrics.timing.backward_ms:.2f} ms ({metrics.timing.backward_pct:.1f}%)")
            print(f"    Optimizer: {metrics.timing.optimizer_ms:.2f} ms ({metrics.timing.optimizer_pct:.1f}%)")
            print(f"    Other:     {metrics.timing.other_ms:.2f} ms")

        print(f"\n  Memory:")
        print(f"    Peak: {metrics.peak_memory_mb:,.0f} MB")
        print(f"    Average: {metrics.average_memory_mb:,.0f} MB")

        is_regressed, pct = self.check_regression()
        if is_regressed:
            print(f"\n  [WARNING] Performance regression detected: {pct:.1f}% slower than baseline")

        print("=" * 60)

    def save_report(self, filename: str = "performance_report.json"):
        """Save performance report to JSON."""
        metrics = self.get_current_metrics()

        report = {
            "metrics": metrics.to_dict(),
            "mfu_summary": self.mfu_tracker.get_summary(),
            "throughput_stats": self.throughput_monitor.get_statistics(),
            "memory_snapshots": [
                {"step": s.step, "allocated_mb": s.allocated_mb, "peak_mb": s.peak_allocated_mb}
                for s in self._memory_snapshots[-100:]  # Last 100
            ],
            "timing_breakdowns": [
                {"forward_ms": t.forward_ms, "backward_ms": t.backward_ms, "optimizer_ms": t.optimizer_ms}
                for t in self._timing_breakdowns[-100:]  # Last 100
            ],
        }

        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Performance report saved to: {filepath}")


class MixedPrecisionManager:
    """
    Handles BF16/FP16/FP8 training with optimal settings.

    Features:
    - Automatic dtype selection based on GPU
    - GradScaler for FP16 if needed
    - FP8 support (DeepSeek V3 style)
    - Per-layer precision control
    """

    def __init__(
        self,
        model: nn.Module,
        preferred_dtype: str = "auto",  # "auto", "bf16", "fp16", "fp8"
        use_grad_scaler: bool = True,
        fp8_exclude_patterns: Optional[List[str]] = None,
    ):
        """
        Initialize mixed precision manager.

        Args:
            model: The model
            preferred_dtype: Preferred precision
            use_grad_scaler: Use GradScaler for FP16
            fp8_exclude_patterns: Patterns for layers to exclude from FP8
        """
        self.model = model
        self.fp8_exclude_patterns = fp8_exclude_patterns or ["embed", "lm_head", "norm"]

        # Detect GPU capabilities
        self.gpu_specs = detect_gpu_specs()
        self.supports_bf16 = self._check_bf16_support()
        self.supports_fp8 = self._check_fp8_support()

        # Select dtype
        self.dtype = self._select_dtype(preferred_dtype)

        # Setup GradScaler for FP16
        self.grad_scaler = None
        if self.dtype == torch.float16 and use_grad_scaler:
            self.grad_scaler = torch.cuda.amp.GradScaler()

        print(f"MixedPrecisionManager: Using {self.dtype}")

    def _check_bf16_support(self) -> bool:
        """Check if GPU supports BF16."""
        if not torch.cuda.is_available():
            return False
        cc = torch.cuda.get_device_capability()
        return cc[0] >= 8  # Ampere or newer

    def _check_fp8_support(self) -> bool:
        """Check if GPU supports FP8."""
        if not torch.cuda.is_available():
            return False
        cc = torch.cuda.get_device_capability()
        return cc[0] >= 9  # Hopper or newer

    def _select_dtype(self, preferred: str) -> torch.dtype:
        """Select optimal dtype based on GPU and preference."""
        if preferred == "auto":
            if self.supports_bf16:
                return torch.bfloat16
            else:
                return torch.float16
        elif preferred == "bf16":
            if not self.supports_bf16:
                warnings.warn("BF16 not supported, falling back to FP16")
                return torch.float16
            return torch.bfloat16
        elif preferred == "fp16":
            return torch.float16
        elif preferred == "fp8":
            if not self.supports_fp8:
                warnings.warn("FP8 not supported, falling back to BF16")
                return torch.bfloat16 if self.supports_bf16 else torch.float16
            return torch.float8_e4m3fn
        else:
            return torch.bfloat16

    @contextmanager
    def autocast(self):
        """Context manager for automatic mixed precision."""
        with torch.cuda.amp.autocast(dtype=self.dtype):
            yield

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for FP16 training."""
        if self.grad_scaler is not None:
            return self.grad_scaler.scale(loss)
        return loss

    def unscale_gradients(self, optimizer: torch.optim.Optimizer):
        """Unscale gradients before clipping."""
        if self.grad_scaler is not None:
            self.grad_scaler.unscale_(optimizer)

    def step(self, optimizer: torch.optim.Optimizer):
        """Optimizer step with gradient scaling."""
        if self.grad_scaler is not None:
            self.grad_scaler.step(optimizer)
            self.grad_scaler.update()
        else:
            optimizer.step()

    def get_summary(self) -> Dict[str, Any]:
        """Get precision configuration summary."""
        return {
            "dtype": str(self.dtype),
            "supports_bf16": self.supports_bf16,
            "supports_fp8": self.supports_fp8,
            "using_grad_scaler": self.grad_scaler is not None,
            "gpu_name": self.gpu_specs.get("name", "Unknown"),
        }


def generate_performance_report(
    model: nn.Module,
    metrics: PerformanceMetrics,
    output_path: str = "./performance_report.html",
) -> str:
    """
    Generate an HTML performance report.

    Args:
        model: The model
        metrics: Performance metrics
        output_path: Path for HTML output

    Returns:
        Path to generated report
    """
    num_params = sum(p.numel() for p in model.parameters())
    gpu_specs = detect_gpu_specs()

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chronicals Performance Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
            h2 {{ color: #666; margin-top: 30px; }}
            .metric {{ display: flex; justify-content: space-between; padding: 10px; border-bottom: 1px solid #eee; }}
            .metric-label {{ color: #666; }}
            .metric-value {{ font-weight: bold; color: #333; }}
            .good {{ color: #4CAF50; }}
            .warning {{ color: #FF9800; }}
            .bad {{ color: #F44336; }}
            .bar {{ height: 20px; background: #e0e0e0; border-radius: 10px; margin: 10px 0; }}
            .bar-fill {{ height: 100%; border-radius: 10px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Chronicals Performance Report</h1>

            <h2>Hardware</h2>
            <div class="metric">
                <span class="metric-label">GPU</span>
                <span class="metric-value">{gpu_specs.get('name', 'Unknown')}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Peak TFLOPs (BF16)</span>
                <span class="metric-value">{gpu_specs.get('bf16_tflops', 0):.1f}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Memory Bandwidth</span>
                <span class="metric-value">{gpu_specs.get('bandwidth_tb_s', 0):.2f} TB/s</span>
            </div>

            <h2>Model</h2>
            <div class="metric">
                <span class="metric-label">Parameters</span>
                <span class="metric-value">{num_params:,} ({num_params/1e9:.2f}B)</span>
            </div>

            <h2>Performance</h2>
            <div class="metric">
                <span class="metric-label">Throughput</span>
                <span class="metric-value">{metrics.tokens_per_second:,.0f} tokens/sec</span>
            </div>
            <div class="metric">
                <span class="metric-label">MFU</span>
                <span class="metric-value {'good' if metrics.mfu_percent > 40 else 'warning' if metrics.mfu_percent > 20 else 'bad'}">{metrics.mfu_percent:.1f}%</span>
            </div>

            <h3>MFU Progress</h3>
            <div class="bar">
                <div class="bar-fill" style="width: {min(metrics.mfu_percent, 100)}%; background: {'#4CAF50' if metrics.mfu_percent > 40 else '#FF9800' if metrics.mfu_percent > 20 else '#F44336'};"></div>
            </div>
            <p style="text-align: center; color: #666;">Target: 50%+ | Current: {metrics.mfu_percent:.1f}%</p>

            <h2>Timing Breakdown</h2>
            <div class="metric">
                <span class="metric-label">Forward Pass</span>
                <span class="metric-value">{metrics.timing.forward_ms:.2f} ms ({metrics.timing.forward_pct:.1f}%)</span>
            </div>
            <div class="metric">
                <span class="metric-label">Backward Pass</span>
                <span class="metric-value">{metrics.timing.backward_ms:.2f} ms ({metrics.timing.backward_pct:.1f}%)</span>
            </div>
            <div class="metric">
                <span class="metric-label">Optimizer</span>
                <span class="metric-value">{metrics.timing.optimizer_ms:.2f} ms ({metrics.timing.optimizer_pct:.1f}%)</span>
            </div>

            <h2>Memory</h2>
            <div class="metric">
                <span class="metric-label">Peak Memory</span>
                <span class="metric-value">{metrics.peak_memory_mb:,.0f} MB</span>
            </div>
            <div class="metric">
                <span class="metric-label">Average Memory</span>
                <span class="metric-value">{metrics.average_memory_mb:,.0f} MB</span>
            </div>
        </div>
    </body>
    </html>
    """

    with open(output_path, "w") as f:
        f.write(html_content)

    return output_path


def print_performance_report(metrics: PerformanceMetrics):
    """Print formatted performance report to console."""
    print("\n" + "=" * 60)
    print("PERFORMANCE REPORT")
    print("=" * 60)
    print(f"\n  Throughput: {metrics.tokens_per_second:,.0f} tokens/sec")
    print(f"  MFU: {metrics.mfu_percent:.1f}%")
    print(f"\n  Timing:")
    print(f"    Forward:   {metrics.timing.forward_ms:.2f} ms")
    print(f"    Backward:  {metrics.timing.backward_ms:.2f} ms")
    print(f"    Optimizer: {metrics.timing.optimizer_ms:.2f} ms")
    print(f"\n  Memory:")
    print(f"    Peak: {metrics.peak_memory_mb:,.0f} MB")
    print("=" * 60)


if __name__ == "__main__":
    print("Profiling Utilities for Chronicals Training")
    print("=" * 50)

    # Test GPU detection
    specs = detect_gpu_specs()
    print(f"\nDetected GPU: {specs.get('name', 'N/A')}")
    print(f"  BF16 TFLOPs: {specs.get('bf16_tflops', 0)}")
    print(f"  Memory: {specs.get('memory_gb', 0)} GB")
    print(f"  Bandwidth: {specs.get('bandwidth_tb_s', 0)} TB/s")

    if torch.cuda.is_available():
        # Create a simple model for testing
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1024, 1024)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel().cuda()

        # Test MFU tracker
        mfu_tracker = MFUTracker(model)
        print(f"\nMFU Tracker:")
        print(f"  Model params: {mfu_tracker.num_params:,}")
        print(f"  FLOPs per token: {mfu_tracker.flops_per_token:,}")

        # Simulate some training steps
        for i in range(10):
            mfu = mfu_tracker.compute_mfu(tokens=1024, elapsed_sec=0.01)
            print(f"  Step {i+1}: MFU = {mfu:.1f}%")

        print(f"  Average MFU: {mfu_tracker.get_average_mfu():.1f}%")

        # Test throughput monitor
        print("\nThroughput Monitor:")
        monitor = ThroughputMonitor()
        for i in range(20):
            monitor.record_step(tokens=1024, elapsed_sec=0.01)

        stats = monitor.get_statistics()
        print(f"  Mean: {stats['mean']:,.0f} tokens/sec")
        print(f"  Min: {stats['min']:,.0f} tokens/sec")
        print(f"  Max: {stats['max']:,.0f} tokens/sec")

    else:
        print("\nCUDA not available for testing")
