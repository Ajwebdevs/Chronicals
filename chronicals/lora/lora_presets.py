%%writefile lora_presets.py
"""
=============================================================================
CHRONICALS LORA PRESETS
=============================================================================
Pre-configured settings for different hardware and model sizes.
Automatically selects optimal configurations based on GPU detection.

Hardware Presets:
- T4: 16GB VRAM (Colab Free)
- A10G: 24GB VRAM (Colab Pro, AWS)
- L4: 24GB VRAM (GCP)
- A100-40GB: 40GB VRAM
- A100-80GB: 80GB VRAM
- RTX 4090: 24GB VRAM
- RTX 3090: 24GB VRAM
- H100: 80GB VRAM

Model Size Presets:
- 0.5B: Qwen2.5-0.5B, etc.
- 1B: Llama-3.2-1B, etc.
- 3B: Qwen2.5-3B, etc.
- 7B: Llama-2-7B, Mistral-7B, etc.
- 13B: Llama-2-13B, etc.
- 70B: Llama-2-70B (multi-GPU required)

Usage:
    from lora_presets import get_optimal_preset, detect_hardware

    # Auto-detect and get optimal settings
    preset = get_optimal_preset(model_name="Qwen/Qwen2.5-0.5B")

    # Or manually select
    preset = get_preset_for_hardware("a100-80gb", "7B")

=============================================================================
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

import torch


# =============================================================================
# HARDWARE DETECTION
# =============================================================================

class GPUType(Enum):
    """Supported GPU types."""
    T4 = "t4"
    A10G = "a10g"
    L4 = "l4"
    A100_40GB = "a100-40gb"
    A100_80GB = "a100-80gb"
    H100 = "h100"
    RTX_4090 = "rtx4090"
    RTX_3090 = "rtx3090"
    V100 = "v100"
    UNKNOWN = "unknown"


@dataclass
class GPUSpec:
    """GPU specifications."""
    name: str
    memory_gb: float
    compute_capability: str
    bf16_tflops: float
    supports_fp8: bool
    memory_bandwidth_gbps: float


# GPU specifications database
GPU_SPECS: Dict[GPUType, GPUSpec] = {
    GPUType.T4: GPUSpec(
        name="Tesla T4",
        memory_gb=16,
        compute_capability="7.5",
        bf16_tflops=65,
        supports_fp8=False,
        memory_bandwidth_gbps=320,
    ),
    GPUType.A10G: GPUSpec(
        name="NVIDIA A10G",
        memory_gb=24,
        compute_capability="8.6",
        bf16_tflops=125,
        supports_fp8=False,
        memory_bandwidth_gbps=600,
    ),
    GPUType.L4: GPUSpec(
        name="NVIDIA L4",
        memory_gb=24,
        compute_capability="8.9",
        bf16_tflops=121,
        supports_fp8=True,
        memory_bandwidth_gbps=300,
    ),
    GPUType.A100_40GB: GPUSpec(
        name="NVIDIA A100 40GB",
        memory_gb=40,
        compute_capability="8.0",
        bf16_tflops=312,
        supports_fp8=False,  # Simulated
        memory_bandwidth_gbps=1555,
    ),
    GPUType.A100_80GB: GPUSpec(
        name="NVIDIA A100 80GB",
        memory_gb=80,
        compute_capability="8.0",
        bf16_tflops=312,
        supports_fp8=False,  # Simulated
        memory_bandwidth_gbps=2039,
    ),
    GPUType.H100: GPUSpec(
        name="NVIDIA H100",
        memory_gb=80,
        compute_capability="9.0",
        bf16_tflops=989,
        supports_fp8=True,
        memory_bandwidth_gbps=3350,
    ),
    GPUType.RTX_4090: GPUSpec(
        name="GeForce RTX 4090",
        memory_gb=24,
        compute_capability="8.9",
        bf16_tflops=82.6,
        supports_fp8=True,
        memory_bandwidth_gbps=1008,
    ),
    GPUType.RTX_3090: GPUSpec(
        name="GeForce RTX 3090",
        memory_gb=24,
        compute_capability="8.6",
        bf16_tflops=35.6,
        supports_fp8=False,
        memory_bandwidth_gbps=936,
    ),
    GPUType.V100: GPUSpec(
        name="Tesla V100",
        memory_gb=32,
        compute_capability="7.0",
        bf16_tflops=125,
        supports_fp8=False,
        memory_bandwidth_gbps=900,
    ),
}


def detect_gpu() -> Tuple[GPUType, Optional[GPUSpec]]:
    """
    Detect the current GPU type.

    Returns:
        Tuple of (GPUType, GPUSpec or None)
    """
    if not torch.cuda.is_available():
        return GPUType.UNKNOWN, None

    gpu_name = torch.cuda.get_device_name(0).lower()
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    # Match GPU type
    if "t4" in gpu_name:
        return GPUType.T4, GPU_SPECS[GPUType.T4]
    elif "a10" in gpu_name:
        return GPUType.A10G, GPU_SPECS[GPUType.A10G]
    elif "l4" in gpu_name:
        return GPUType.L4, GPU_SPECS[GPUType.L4]
    elif "a100" in gpu_name:
        if memory_gb > 60:
            return GPUType.A100_80GB, GPU_SPECS[GPUType.A100_80GB]
        else:
            return GPUType.A100_40GB, GPU_SPECS[GPUType.A100_40GB]
    elif "h100" in gpu_name:
        return GPUType.H100, GPU_SPECS[GPUType.H100]
    elif "4090" in gpu_name:
        return GPUType.RTX_4090, GPU_SPECS[GPUType.RTX_4090]
    elif "3090" in gpu_name:
        return GPUType.RTX_3090, GPU_SPECS[GPUType.RTX_3090]
    elif "v100" in gpu_name:
        return GPUType.V100, GPU_SPECS[GPUType.V100]

    # Unknown GPU - create spec from detection
    props = torch.cuda.get_device_properties(0)
    unknown_spec = GPUSpec(
        name=torch.cuda.get_device_name(0),
        memory_gb=memory_gb,
        compute_capability=f"{props.major}.{props.minor}",
        bf16_tflops=50,  # Conservative estimate
        supports_fp8=props.major >= 9,
        memory_bandwidth_gbps=500,
    )
    return GPUType.UNKNOWN, unknown_spec


def detect_hardware() -> Dict[str, Any]:
    """
    Detect hardware and return comprehensive info.

    Returns:
        Dict with hardware information
    """
    gpu_type, gpu_spec = detect_gpu()

    info = {
        "cuda_available": torch.cuda.is_available(),
        "gpu_type": gpu_type.value,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if gpu_spec:
        info.update({
            "gpu_name": gpu_spec.name,
            "memory_gb": gpu_spec.memory_gb,
            "compute_capability": gpu_spec.compute_capability,
            "bf16_tflops": gpu_spec.bf16_tflops,
            "supports_fp8": gpu_spec.supports_fp8,
            "memory_bandwidth_gbps": gpu_spec.memory_bandwidth_gbps,
        })

    return info


# =============================================================================
# LORA PRESETS
# =============================================================================

@dataclass
class LoRAPreset:
    """
    Pre-configured LoRA training settings.

    Optimized for specific hardware + model size combinations.
    """
    # Identification
    name: str
    description: str

    # LoRA hyperparameters
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Training hyperparameters
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    seq_length: int = 512
    learning_rate: float = 2e-4
    num_epochs: int = 3
    warmup_ratio: float = 0.03

    # Optimizations
    use_liger_kernel: bool = True
    use_torch_compile: bool = True
    use_lora_plus: bool = True
    lora_plus_lr_ratio: float = 16.0
    use_fused_adamw: bool = True
    use_gradient_checkpointing: bool = False
    use_bf16: bool = True

    # Memory optimizations
    use_8bit_optimizer: bool = False
    use_4bit_quantization: bool = False

    # Expected performance
    expected_throughput_tokens_sec: int = 0
    expected_memory_gb: float = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert preset to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "seq_length": self.seq_length,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "warmup_ratio": self.warmup_ratio,
            "use_liger_kernel": self.use_liger_kernel,
            "use_torch_compile": self.use_torch_compile,
            "use_lora_plus": self.use_lora_plus,
            "use_fused_adamw": self.use_fused_adamw,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_bf16": self.use_bf16,
            "expected_throughput_tokens_sec": self.expected_throughput_tokens_sec,
            "expected_memory_gb": self.expected_memory_gb,
        }


# =============================================================================
# HARDWARE-SPECIFIC PRESETS
# =============================================================================

# T4 (16GB) - Colab Free
T4_PRESETS = {
    "0.5B": LoRAPreset(
        name="T4-0.5B",
        description="Qwen2.5-0.5B on T4 (Colab Free)",
        lora_rank=16,
        batch_size=4,
        gradient_accumulation_steps=4,
        seq_length=512,
        use_gradient_checkpointing=False,
        expected_throughput_tokens_sec=8000,
        expected_memory_gb=8,
    ),
    "1B": LoRAPreset(
        name="T4-1B",
        description="1B model on T4 (Colab Free)",
        lora_rank=16,
        batch_size=2,
        gradient_accumulation_steps=8,
        seq_length=512,
        use_gradient_checkpointing=True,
        expected_throughput_tokens_sec=5000,
        expected_memory_gb=12,
    ),
    "3B": LoRAPreset(
        name="T4-3B",
        description="3B model on T4 - requires 4-bit",
        lora_rank=8,
        batch_size=1,
        gradient_accumulation_steps=16,
        seq_length=256,
        use_gradient_checkpointing=True,
        use_4bit_quantization=True,
        expected_throughput_tokens_sec=2000,
        expected_memory_gb=14,
    ),
}

# A10G / L4 / RTX 4090 (24GB)
A10G_PRESETS = {
    "0.5B": LoRAPreset(
        name="A10G-0.5B",
        description="Qwen2.5-0.5B on A10G/L4/4090",
        lora_rank=32,
        batch_size=8,
        gradient_accumulation_steps=2,
        seq_length=512,
        expected_throughput_tokens_sec=15000,
        expected_memory_gb=10,
    ),
    "1B": LoRAPreset(
        name="A10G-1B",
        description="1B model on A10G/L4/4090",
        lora_rank=32,
        batch_size=4,
        gradient_accumulation_steps=4,
        seq_length=512,
        expected_throughput_tokens_sec=10000,
        expected_memory_gb=16,
    ),
    "3B": LoRAPreset(
        name="A10G-3B",
        description="3B model on A10G/L4/4090",
        lora_rank=16,
        batch_size=2,
        gradient_accumulation_steps=8,
        seq_length=512,
        use_gradient_checkpointing=True,
        expected_throughput_tokens_sec=6000,
        expected_memory_gb=20,
    ),
    "7B": LoRAPreset(
        name="A10G-7B",
        description="7B model on A10G/L4/4090 - requires 4-bit",
        lora_rank=16,
        batch_size=1,
        gradient_accumulation_steps=16,
        seq_length=512,
        use_gradient_checkpointing=True,
        use_4bit_quantization=True,
        expected_throughput_tokens_sec=3000,
        expected_memory_gb=22,
    ),
}

# A100-40GB
A100_40GB_PRESETS = {
    "0.5B": LoRAPreset(
        name="A100-40GB-0.5B",
        description="Qwen2.5-0.5B on A100-40GB",
        lora_rank=64,
        batch_size=16,
        gradient_accumulation_steps=1,
        seq_length=512,
        expected_throughput_tokens_sec=35000,
        expected_memory_gb=12,
    ),
    "1B": LoRAPreset(
        name="A100-40GB-1B",
        description="1B model on A100-40GB",
        lora_rank=64,
        batch_size=8,
        gradient_accumulation_steps=2,
        seq_length=512,
        expected_throughput_tokens_sec=25000,
        expected_memory_gb=20,
    ),
    "3B": LoRAPreset(
        name="A100-40GB-3B",
        description="3B model on A100-40GB",
        lora_rank=32,
        batch_size=4,
        gradient_accumulation_steps=4,
        seq_length=512,
        expected_throughput_tokens_sec=15000,
        expected_memory_gb=28,
    ),
    "7B": LoRAPreset(
        name="A100-40GB-7B",
        description="7B model on A100-40GB",
        lora_rank=32,
        batch_size=2,
        gradient_accumulation_steps=8,
        seq_length=512,
        use_gradient_checkpointing=True,
        expected_throughput_tokens_sec=8000,
        expected_memory_gb=36,
    ),
}

# A100-80GB
A100_80GB_PRESETS = {
    "0.5B": LoRAPreset(
        name="A100-80GB-0.5B",
        description="Qwen2.5-0.5B on A100-80GB - MAXIMUM SPEED",
        lora_rank=64,
        batch_size=32,
        gradient_accumulation_steps=1,
        seq_length=512,
        expected_throughput_tokens_sec=50000,
        expected_memory_gb=15,
    ),
    "1B": LoRAPreset(
        name="A100-80GB-1B",
        description="1B model on A100-80GB",
        lora_rank=64,
        batch_size=16,
        gradient_accumulation_steps=1,
        seq_length=512,
        expected_throughput_tokens_sec=35000,
        expected_memory_gb=25,
    ),
    "3B": LoRAPreset(
        name="A100-80GB-3B",
        description="3B model on A100-80GB",
        lora_rank=64,
        batch_size=8,
        gradient_accumulation_steps=2,
        seq_length=512,
        expected_throughput_tokens_sec=20000,
        expected_memory_gb=40,
    ),
    "7B": LoRAPreset(
        name="A100-80GB-7B",
        description="7B model on A100-80GB",
        lora_rank=64,
        batch_size=4,
        gradient_accumulation_steps=4,
        seq_length=512,
        expected_throughput_tokens_sec=10000,
        expected_memory_gb=55,
    ),
    "13B": LoRAPreset(
        name="A100-80GB-13B",
        description="13B model on A100-80GB",
        lora_rank=32,
        batch_size=2,
        gradient_accumulation_steps=8,
        seq_length=512,
        use_gradient_checkpointing=True,
        expected_throughput_tokens_sec=5000,
        expected_memory_gb=70,
    ),
}

# H100 (80GB)
H100_PRESETS = {
    "0.5B": LoRAPreset(
        name="H100-0.5B",
        description="Qwen2.5-0.5B on H100 - BLAZING FAST",
        lora_rank=64,
        batch_size=64,
        gradient_accumulation_steps=1,
        seq_length=512,
        expected_throughput_tokens_sec=100000,
        expected_memory_gb=20,
    ),
    "1B": LoRAPreset(
        name="H100-1B",
        description="1B model on H100",
        lora_rank=64,
        batch_size=32,
        gradient_accumulation_steps=1,
        seq_length=512,
        expected_throughput_tokens_sec=70000,
        expected_memory_gb=30,
    ),
    "3B": LoRAPreset(
        name="H100-3B",
        description="3B model on H100",
        lora_rank=64,
        batch_size=16,
        gradient_accumulation_steps=1,
        seq_length=512,
        expected_throughput_tokens_sec=40000,
        expected_memory_gb=45,
    ),
    "7B": LoRAPreset(
        name="H100-7B",
        description="7B model on H100",
        lora_rank=64,
        batch_size=8,
        gradient_accumulation_steps=2,
        seq_length=512,
        expected_throughput_tokens_sec=20000,
        expected_memory_gb=60,
    ),
    "13B": LoRAPreset(
        name="H100-13B",
        description="13B model on H100",
        lora_rank=64,
        batch_size=4,
        gradient_accumulation_steps=4,
        seq_length=512,
        expected_throughput_tokens_sec=10000,
        expected_memory_gb=75,
    ),
}


# Preset registry
PRESET_REGISTRY: Dict[GPUType, Dict[str, LoRAPreset]] = {
    GPUType.T4: T4_PRESETS,
    GPUType.A10G: A10G_PRESETS,
    GPUType.L4: A10G_PRESETS,  # Same as A10G
    GPUType.RTX_4090: A10G_PRESETS,  # Same memory class
    GPUType.RTX_3090: A10G_PRESETS,  # Same memory class
    GPUType.V100: A100_40GB_PRESETS,  # Similar to A100-40GB
    GPUType.A100_40GB: A100_40GB_PRESETS,
    GPUType.A100_80GB: A100_80GB_PRESETS,
    GPUType.H100: H100_PRESETS,
}


# =============================================================================
# PRESET SELECTION
# =============================================================================

def get_model_size(model_name: str) -> str:
    """
    Determine model size category from model name.

    Args:
        model_name: HuggingFace model name

    Returns:
        Size category: "0.5B", "1B", "3B", "7B", "13B", "70B"
    """
    name_lower = model_name.lower()

    # Check for explicit size in name
    if "0.5b" in name_lower or "500m" in name_lower:
        return "0.5B"
    elif "1b" in name_lower or "1.5b" in name_lower:
        return "1B"
    elif "3b" in name_lower:
        return "3B"
    elif "7b" in name_lower or "8b" in name_lower:
        return "7B"
    elif "13b" in name_lower:
        return "13B"
    elif "70b" in name_lower:
        return "70B"

    # Default to 1B for unknown models
    return "1B"


def get_preset_for_hardware(
    gpu_type: str,
    model_size: str,
) -> LoRAPreset:
    """
    Get preset for specific hardware and model size.

    Args:
        gpu_type: GPU type string (e.g., "t4", "a100-80gb")
        model_size: Model size (e.g., "0.5B", "7B")

    Returns:
        LoRAPreset configuration
    """
    # Convert string to enum
    try:
        gpu_enum = GPUType(gpu_type.lower())
    except ValueError:
        gpu_enum = GPUType.UNKNOWN

    # Get presets for this GPU
    presets = PRESET_REGISTRY.get(gpu_enum, T4_PRESETS)

    # Get preset for model size
    if model_size in presets:
        return presets[model_size]

    # Fallback to closest size
    sizes = ["0.5B", "1B", "3B", "7B", "13B", "70B"]
    try:
        idx = sizes.index(model_size)
        # Try smaller size first
        for i in range(idx, -1, -1):
            if sizes[i] in presets:
                return presets[sizes[i]]
        # Then try larger
        for i in range(idx, len(sizes)):
            if sizes[i] in presets:
                return presets[sizes[i]]
    except ValueError:
        pass

    # Final fallback
    return list(presets.values())[0]


def get_optimal_preset(
    model_name: str,
    gpu_type: Optional[str] = None,
) -> LoRAPreset:
    """
    Get optimal preset for model, auto-detecting hardware if not specified.

    Args:
        model_name: HuggingFace model name
        gpu_type: Optional GPU type override

    Returns:
        Optimized LoRAPreset
    """
    # Detect hardware if not specified
    if gpu_type is None:
        detected_gpu, _ = detect_gpu()
        gpu_type = detected_gpu.value

    # Get model size
    model_size = get_model_size(model_name)

    # Get preset
    preset = get_preset_for_hardware(gpu_type, model_size)

    print(f"\nSelected preset: {preset.name}")
    print(f"  Model size: {model_size}")
    print(f"  GPU type: {gpu_type}")
    print(f"  Batch size: {preset.batch_size}")
    print(f"  LoRA rank: {preset.lora_rank}")
    print(f"  Expected throughput: {preset.expected_throughput_tokens_sec:,} tok/s")

    return preset


# =============================================================================
# OPTIMIZATION RECOMMENDATIONS
# =============================================================================

def get_optimization_recommendations(
    preset: LoRAPreset,
    current_throughput: Optional[float] = None,
    current_memory_gb: Optional[float] = None,
) -> List[str]:
    """
    Get optimization recommendations based on current performance.

    Args:
        preset: Current preset configuration
        current_throughput: Measured throughput (tokens/sec)
        current_memory_gb: Current memory usage (GB)

    Returns:
        List of recommendation strings
    """
    recommendations = []

    # Check if Liger Kernel is enabled
    if not preset.use_liger_kernel:
        recommendations.append(
            "Enable Liger Kernel for 20%+ throughput boost. "
            "Set use_liger_kernel=True"
        )

    # Check if torch.compile is enabled
    if not preset.use_torch_compile:
        recommendations.append(
            "Enable torch.compile for kernel fusion. "
            "Set use_torch_compile=True"
        )

    # Check if LoRA+ is enabled
    if not preset.use_lora_plus:
        recommendations.append(
            "Enable LoRA+ for 1.5-2x faster convergence. "
            "Set use_lora_plus=True"
        )

    # Performance-based recommendations
    if current_throughput is not None:
        target = preset.expected_throughput_tokens_sec
        ratio = current_throughput / target if target > 0 else 0

        if ratio < 0.5:
            recommendations.append(
                f"Throughput is {ratio:.0%} of expected. Check:"
                "\n  - Is GPU being fully utilized? (nvidia-smi)"
                "\n  - Is data loading a bottleneck? (increase num_workers)"
                "\n  - Are optimizations applied correctly?"
            )
        elif ratio < 0.8:
            recommendations.append(
                f"Throughput at {ratio:.0%} of expected. Consider:"
                "\n  - Increasing batch size if memory allows"
                "\n  - Enabling sequence packing for variable-length data"
            )

    # Memory-based recommendations
    if current_memory_gb is not None:
        gpu_type, gpu_spec = detect_gpu()
        if gpu_spec:
            utilization = current_memory_gb / gpu_spec.memory_gb

            if utilization < 0.5:
                recommendations.append(
                    f"Only using {utilization:.0%} of GPU memory. Consider:"
                    "\n  - Increasing batch size"
                    "\n  - Increasing LoRA rank"
                    "\n  - Increasing sequence length"
                )
            elif utilization > 0.95:
                recommendations.append(
                    f"Memory usage at {utilization:.0%}. Consider:"
                    "\n  - Enabling gradient checkpointing"
                    "\n  - Reducing batch size"
                    "\n  - Using 8-bit optimizer"
                )

    if not recommendations:
        recommendations.append("Configuration looks optimal! No changes needed.")

    return recommendations


# =============================================================================
# PRESET COMPARISON
# =============================================================================

def compare_presets(presets: List[LoRAPreset]) -> str:
    """
    Generate comparison table for multiple presets.

    Args:
        presets: List of presets to compare

    Returns:
        Formatted comparison string
    """
    lines = [
        "\n" + "=" * 80,
        "PRESET COMPARISON",
        "=" * 80,
        f"{'Name':<20} {'Batch':>8} {'Rank':>6} {'Throughput':>12} {'Memory':>10}",
        "-" * 80,
    ]

    for preset in presets:
        lines.append(
            f"{preset.name:<20} {preset.batch_size:>8} {preset.lora_rank:>6} "
            f"{preset.expected_throughput_tokens_sec:>12,} {preset.expected_memory_gb:>9.1f}G"
        )

    lines.append("=" * 80)

    return "\n".join(lines)


def list_all_presets() -> None:
    """Print all available presets."""
    print("\n" + "=" * 80)
    print("CHRONICALS LORA PRESETS")
    print("=" * 80)

    for gpu_type, presets in PRESET_REGISTRY.items():
        print(f"\n{gpu_type.value.upper()}:")
        print("-" * 40)
        for size, preset in presets.items():
            print(f"  {size}: {preset.description}")
            print(f"      Batch={preset.batch_size}, Rank={preset.lora_rank}, "
                  f"~{preset.expected_throughput_tokens_sec:,} tok/s")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CHRONICALS LORA PRESETS")
    print("=" * 70)

    # Detect hardware
    print("\nHardware Detection:")
    hw_info = detect_hardware()
    for key, value in hw_info.items():
        print(f"  {key}: {value}")

    # List all presets
    list_all_presets()

    # Example: Get optimal preset for a model
    print("\n" + "=" * 70)
    print("EXAMPLE: Getting optimal preset")
    print("=" * 70)

    test_models = [
        "Qwen/Qwen2.5-0.5B",
        "meta-llama/Llama-3.2-1B",
        "Qwen/Qwen2.5-7B",
    ]

    for model in test_models:
        preset = get_optimal_preset(model)
        print(f"\n  {model}:")
        print(f"    Preset: {preset.name}")
        print(f"    Batch size: {preset.batch_size}")
        print(f"    LoRA rank: {preset.lora_rank}")
        print(f"    Expected: {preset.expected_throughput_tokens_sec:,} tok/s")
