#!/usr/bin/env python
"""
Chronicals: High-Performance LLM Fine-Tuning Framework
======================================================

Chronicals achieves 3.51x speedup over Unsloth through:
- Fused Triton kernels (RMSNorm, SwiGLU, Cross-Entropy, QK-RoPE)
- Cut Cross-Entropy with 37x memory reduction
- LoRA+ optimizer with differential learning rates
- Sequence packing for efficient batch utilization
- FlashAttention integration

Installation:
    pip install chronicals
    pip install chronicals[all]  # With all optional dependencies

Usage:
    from chronicals.training import ChronicalsTrainer
    from chronicals.config import ChronicalsConfig

    config = ChronicalsConfig(use_flash_attention=True, use_fused_kernels=True)
    trainer = ChronicalsTrainer(model, tokenizer, config)
    trainer.train(dataset)

GitHub: https://github.com/Ajwebdevs/Chronicals
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# Read version from __init__.py
def get_version():
    init_path = os.path.join(os.path.dirname(__file__), "chronicals", "__init__.py")
    if os.path.exists(init_path):
        with open(init_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    return "0.1.0"

setup(
    name="chronicals",
    version=get_version(),
    author="Arjun S. Nair",
    author_email="5minutepodcastforyou@gmail.com",
    description="High-Performance LLM Fine-Tuning Framework - 3.51x faster than Unsloth",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/Ajwebdevs/Chronicals",
    project_urls={
        "Bug Tracker": "https://github.com/Ajwebdevs/Chronicals/issues",
        "Documentation": "https://github.com/Ajwebdevs/Chronicals#readme",
        "Source Code": "https://github.com/Ajwebdevs/Chronicals",
    },
    packages=find_packages(exclude=["tests*", "benchmarks*", "examples*", "paper*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "datasets>=2.14.0",
        "accelerate>=0.25.0",
        "peft>=0.7.0",
        "tqdm>=4.65.0",
        "numpy>=1.24.0",
        "safetensors>=0.4.0",
    ],
    extras_require={
        "triton": ["triton>=2.1.0"],
        "flash-attn": ["flash-attn>=2.4.0"],
        "8bit": ["bitsandbytes>=0.41.0"],
        "all": [
            "triton>=2.1.0",
            "flash-attn>=2.4.0",
            "bitsandbytes>=0.41.0",
            "wandb>=0.16.0",
            "tensorboard>=2.15.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "llm", "fine-tuning", "deep-learning", "transformers", "triton",
        "lora", "flash-attention", "pytorch", "gpu", "optimization",
    ],
)
