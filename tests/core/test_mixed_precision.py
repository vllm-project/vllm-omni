"""Unit tests for mixed precision manager."""

from vllm_omni.core.mixed_precision import (
    MixedPrecisionConfig,
    MixedPrecisionManager,
    PrecisionType,
)


class TestMixedPrecisionManager:
    """Tests for MixedPrecisionManager."""

    def test_default_precision(self):
        """Test default precision selection."""
        config = MixedPrecisionConfig(precision=PrecisionType.AUTO, prefer_bf16=True)
        manager = MixedPrecisionManager(config)

        precision = manager.get_precision_for_layer("attention", PrecisionType.FP32)
        assert precision == PrecisionType.BF16

    def test_fixed_precision(self):
        """Test fixed precision returns configured value."""
        config = MixedPrecisionConfig(precision=PrecisionType.FP16)
        manager = MixedPrecisionManager(config)

        precision = manager.get_precision_for_layer("attention")
        assert precision == PrecisionType.FP16

    def test_layer_wise_precision(self):
        """Test layer-wise precision assignment."""
        config = MixedPrecisionConfig(precision=PrecisionType.AUTO, layer_wise_precision=True, prefer_bf16=True)
        manager = MixedPrecisionManager(config)

        attn_prec = manager.get_precision_for_layer("attention_layer")
        assert attn_prec == PrecisionType.BF16

        embed_prec = manager.get_precision_for_layer("embedding")
        assert embed_prec == PrecisionType.FP32

        norm_prec = manager.get_precision_for_layer("layer_norm")
        assert norm_prec == PrecisionType.FP32

    def test_should_cast_input(self):
        """Test input casting decision."""
        config = MixedPrecisionConfig(enable_casting=True)
        manager = MixedPrecisionManager(config)

        assert manager.should_cast_input(PrecisionType.FP32, PrecisionType.BF16) is True
        assert manager.should_cast_input(PrecisionType.BF16, PrecisionType.BF16) is False

    def test_should_not_cast_when_disabled(self):
        """Test no casting when disabled."""
        config = MixedPrecisionConfig(enable_casting=False)
        manager = MixedPrecisionManager(config)

        assert manager.should_cast_input(PrecisionType.FP32, PrecisionType.BF16) is False

    def test_precision_info(self):
        """Test precision info."""
        config = MixedPrecisionConfig(precision=PrecisionType.AUTO, layer_wise_precision=True, prefer_bf16=False)
        manager = MixedPrecisionManager(config)
        manager.set_precision(PrecisionType.FP16)

        info = manager.get_precision_info()
        assert info["configured_precision"] == "auto"
        assert info["current_precision"] == "fp16"
        assert info["layer_wise"] is True
        assert info["prefer_bf16"] is False


class TestMixedPrecisionConfig:
    """Tests for MixedPrecisionConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = MixedPrecisionConfig()

        assert config.precision == PrecisionType.AUTO
        assert config.enable_casting is True
        assert config.layer_wise_precision is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = MixedPrecisionConfig(precision=PrecisionType.FP16, layer_wise_precision=True, prefer_bf16=False)

        assert config.precision == PrecisionType.FP16
        assert config.layer_wise_precision is True
        assert config.prefer_bf16 is False
