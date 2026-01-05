"""Test that all Chronicals modules can be imported correctly."""

import pytest


class TestCoreImports:
    """Test core package imports."""

    def test_import_chronicals(self):
        """Test main package import."""
        import chronicals
        assert hasattr(chronicals, "__version__")
        assert chronicals.__version__

    def test_import_version(self):
        """Test version function."""
        from chronicals import get_version
        version = get_version()
        assert version
        assert isinstance(version, str)
        assert "." in version

    def test_import_device_info(self):
        """Test device info function."""
        from chronicals import get_device_info
        info = get_device_info()
        assert isinstance(info, dict)
        assert "cuda_available" in info
        assert "triton_available" in info


class TestConfigImports:
    """Test configuration module imports."""

    def test_import_chronicals_config(self):
        """Test ChronicalsConfig import."""
        from chronicals.config import ChronicalsConfig
        assert ChronicalsConfig is not None

    def test_import_training_config(self):
        """Test TrainingConfig import."""
        from chronicals.config import TrainingConfig
        assert TrainingConfig is not None

    def test_config_instantiation(self):
        """Test config can be instantiated with defaults."""
        from chronicals.config import ChronicalsConfig
        config = ChronicalsConfig()
        assert config is not None
        assert hasattr(config, "hidden_size")
        assert hasattr(config, "num_attention_heads")


class TestOptimizerImports:
    """Test optimizer module imports."""

    def test_import_lora_plus_optimizer(self):
        """Test LoRAPlusOptimizer import."""
        from chronicals.optimizers import LoRAPlusOptimizer
        assert LoRAPlusOptimizer is not None

    def test_import_lora_plus_adamw(self):
        """Test LoRAPlusAdamW import."""
        from chronicals.optimizers import LoRAPlusAdamW
        assert LoRAPlusAdamW is not None

    def test_optimizer_alias(self):
        """Test that LoRAPlusOptimizer is alias for LoRAPlusAdamW."""
        from chronicals.optimizers import LoRAPlusOptimizer, LoRAPlusAdamW
        assert LoRAPlusOptimizer is LoRAPlusAdamW


class TestDataImports:
    """Test data module imports."""

    def test_import_sequence_packer(self):
        """Test SequencePacker import."""
        from chronicals.data import SequencePacker
        assert SequencePacker is not None

    def test_import_packed_batch(self):
        """Test PackedBatch import."""
        from chronicals.data import PackedBatch
        assert PackedBatch is not None


class TestTrainingImports:
    """Test training module imports."""

    def test_import_chronicals_trainer(self):
        """Test ChronicalsTrainer import."""
        from chronicals.training import ChronicalsTrainer
        assert ChronicalsTrainer is not None


class TestLazyImports:
    """Test lazy imports from main package."""

    def test_lazy_import_trainer(self):
        """Test lazy import of ChronicalsTrainer."""
        from chronicals import ChronicalsTrainer
        assert ChronicalsTrainer is not None

    def test_lazy_import_config(self):
        """Test lazy import of ChronicalsConfig."""
        from chronicals import ChronicalsConfig
        assert ChronicalsConfig is not None

    def test_lazy_import_optimizer(self):
        """Test lazy import of LoRAPlusOptimizer."""
        from chronicals import LoRAPlusOptimizer
        assert LoRAPlusOptimizer is not None

    def test_lazy_import_sequence_packer(self):
        """Test lazy import of SequencePacker."""
        from chronicals import SequencePacker
        assert SequencePacker is not None


class TestKernelImports:
    """Test kernel module imports (may fail without GPU/Triton)."""

    @pytest.mark.skipif(True, reason="Triton kernels require GPU")
    def test_import_triton_kernels(self):
        """Test Triton kernels import (requires GPU)."""
        from chronicals.kernels import triton_kernels
        assert triton_kernels is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
