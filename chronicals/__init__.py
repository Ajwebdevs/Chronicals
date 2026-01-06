"""
Chronicals: High-Performance LLM Fine-Tuning Framework
======================================================

Chronicals achieves 3.51x speedup over Unsloth through:
- Fused Triton kernels (RMSNorm, SwiGLU, Cross-Entropy, QK-RoPE)
- Cut Cross-Entropy with 37x memory reduction
- LoRA+ optimizer with differential learning rates
- Sequence packing for efficient batch utilization
- FlashAttention integration

Quick Start:
    from chronicals import ChronicalsTrainer, ChronicalsConfig

    config = ChronicalsConfig(use_flash_attention=True, use_fused_kernels=True)
    trainer = ChronicalsTrainer(model, tokenizer, config)
    trainer.train(dataset)

For more information, visit: https://github.com/Ajwebdevs/Chronicals
"""

__version__ = "0.1.4"
__author__ = "Arjun S. Nair"
__email__ = "5minutepodcastforyou@gmail.com"
__license__ = "Apache-2.0"
__url__ = "https://github.com/Ajwebdevs/Chronicals"

# Lazy imports for faster startup
def __getattr__(name):
    """Lazy import mechanism for heavy modules."""
    if name == "ChronicalsTrainer":
        from chronicals.training.chronicals_trainer import ChronicalsTrainer
        return ChronicalsTrainer
    elif name == "ChronicalsConfig":
        from chronicals.config.config import ChronicalsConfig
        return ChronicalsConfig
    elif name == "LoRAPlusOptimizer":
        from chronicals.optimizers.lora_plus_optimizer import LoRAPlusOptimizer
        return LoRAPlusOptimizer
    elif name == "SequencePacker":
        from chronicals.data.sequence_packer import SequencePacker
        return SequencePacker
    raise AttributeError(f"module 'chronicals' has no attribute '{name}'")


def get_version():
    """Return the current version string."""
    return __version__


def get_device_info():
    """Get information about available compute devices."""
    info = {"cuda_available": False, "device_count": 0, "triton_available": False}

    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["device_count"] = torch.cuda.device_count()
            info["device_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
    except ImportError:
        pass

    try:
        import triton
        info["triton_available"] = True
        info["triton_version"] = triton.__version__
    except ImportError:
        pass

    return info


def print_info():
    """Print Chronicals package information."""
    print(f"Chronicals v{__version__}")
    print(f"Author: {__author__}")
    print(f"License: {__license__}")
    print(f"URL: {__url__}")
    print()

    device_info = get_device_info()
    print("Device Information:")
    print(f"  CUDA Available: {device_info['cuda_available']}")
    if device_info["cuda_available"]:
        print(f"  Device Count: {device_info['device_count']}")
        print(f"  Device Name: {device_info.get('device_name', 'N/A')}")
        print(f"  CUDA Version: {device_info.get('cuda_version', 'N/A')}")
    print(f"  Triton Available: {device_info['triton_available']}")
    if device_info["triton_available"]:
        print(f"  Triton Version: {device_info.get('triton_version', 'N/A')}")


__all__ = [
    "__version__",
    "__author__",
    "ChronicalsTrainer",
    "ChronicalsConfig",
    "LoRAPlusOptimizer",
    "SequencePacker",
    "get_version",
    "get_device_info",
    "print_info",
]
