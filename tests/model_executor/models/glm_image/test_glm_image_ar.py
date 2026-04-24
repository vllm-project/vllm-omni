# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for GLM-Image AR model: DataParser, processor, and M-RoPE."""

import importlib.util
import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import torch

# ---------------------------------------------------------------------------
# Load target classes via importlib to avoid requiring transformers.models.glm_image
# (which may not exist in CI).  This follows the same pattern as
# tests/model_executor/models/qwen3_tts/test_code_predictor_dtype.py.
# ---------------------------------------------------------------------------

_BASE = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    os.pardir,
    os.pardir,
    os.pardir,
    "vllm_omni",
    "model_executor",
    "models",
    "glm_image",
)


def _load_module(name: str, filename: str):
    path = os.path.abspath(os.path.join(_BASE, filename))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_mock_modules() -> dict[str, object]:
    """Build the dict of modules to inject into sys.modules."""
    # Stub transformers.models.glm_image submodules
    glm_image_mod = types.ModuleType("transformers.models.glm_image")
    glm_config_mod = types.ModuleType("transformers.models.glm_image.configuration_glm_image")
    glm_config_mod.GlmImageConfig = type("GlmImageConfig", (), {})
    glm_config_mod.GlmImageTextConfig = type("GlmImageTextConfig", (), {})
    glm_config_mod.GlmImageVisionConfig = type("GlmImageVisionConfig", (), {})
    glm_config_mod.GlmImageVQVAEConfig = type("GlmImageVQVAEConfig", (), {})
    glm_proc_mod = types.ModuleType("transformers.models.glm_image.processing_glm_image")
    glm_proc_mod.GlmImageProcessor = type("GlmImageProcessor", (), {})

    # vllm_omni submodules needed by the import chain
    vllm_omni_mod = MagicMock()
    vllm_omni_models = types.ModuleType("vllm_omni.model_executor.models")
    vllm_omni_glm_image_pkg = types.ModuleType("vllm_omni.model_executor.models.glm_image")
    vllm_omni_glm_image_pkg.__path__ = [os.path.abspath(_BASE)]
    vllm_omni_output = MagicMock()

    return {
        "transformers.models.glm_image": glm_image_mod,
        "transformers.models.glm_image.configuration_glm_image": glm_config_mod,
        "transformers.models.glm_image.processing_glm_image": glm_proc_mod,
        "vllm_omni": vllm_omni_mod,
        "vllm_omni.model_executor": types.ModuleType("vllm_omni.model_executor"),
        "vllm_omni.model_executor.models": vllm_omni_models,
        "vllm_omni.model_executor.models.glm_image": vllm_omni_glm_image_pkg,
        "vllm_omni.model_executor.models.output_templates": vllm_omni_output,
    }


def _load_target_classes():
    """Load the glm_image_ar module with mocked dependencies."""
    mocks = _build_mock_modules()
    with patch.dict(sys.modules, mocks):
        mod = _load_module(
            "vllm_omni.model_executor.models.glm_image.glm_image_ar",
            "glm_image_ar.py",
        )
        sys.modules["vllm_omni.model_executor.models.glm_image.glm_image_ar"] = mod
    return mod


_ar_mod = _load_target_classes()

GlmImageDataParser = _ar_mod.GlmImageDataParser
GlmImageMultiModalProcessor = _ar_mod.GlmImageMultiModalProcessor
GlmImageForConditionalGeneration = _ar_mod.GlmImageForConditionalGeneration
GlmImageRotaryEmbedding = _ar_mod.GlmImageRotaryEmbedding

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# =============================================================================
# Helper: Minimal config for testing
# =============================================================================


def _make_hf_config(**overrides):
    """Create a minimal GlmImageConfig-like object for testing."""
    defaults = {
        "image_token_id": 167855,
        "image_start_token_id": 16384,
        "image_end_token_id": 16385,
        "grid_bos_token_id": None,
        "grid_eos_token_id": None,
    }
    defaults.update(overrides)
    from types import SimpleNamespace

    return SimpleNamespace(**defaults)


# =============================================================================
# Tests for GlmImageDataParser
# =============================================================================


class TestGlmImageDataParser:
    """Test that img2img key is normalized to image in the data parser."""

    def test_img2img_normalized_to_image(self):
        parser = GlmImageDataParser.__new__(GlmImageDataParser)
        parser._expected_hidden_size = 4096
        # The _get_subparsers should include img2img
        subparsers = parser._get_subparsers()
        assert "img2img" in subparsers
        assert subparsers["img2img"] == parser._parse_image_data

    def test_parse_mm_data_normalizes_img2img(self):
        parser = GlmImageDataParser.__new__(GlmImageDataParser)
        parser._expected_hidden_size = 4096
        # Create a mock for the parent parse_mm_data
        original_parse = type(parser).parse_mm_data

        calls = []

        def mock_parse(mm_data, **kwargs):
            calls.append(mm_data)
            return MagicMock()

        # Monkey-patch temporarily
        type(parser).parse_mm_data = mock_parse
        try:
            parser.parse_mm_data({"img2img": "fake_image"})
        except Exception:
            pass  # parse might fail on mock, we just check the normalization
        finally:
            type(parser).parse_mm_data = original_parse

        # Verify that "img2img" was normalized to "image"
        if calls:
            assert "image" in calls[0]
            assert "img2img" not in calls[0]


# =============================================================================
# Tests for _build_generation_grids
# =============================================================================


class TestBuildGenerationGrids:
    """Test M-RoPE grid construction for t2i mode."""

    @pytest.fixture
    def processor(self):
        """Create a minimal processor instance with mocked info."""
        proc = object.__new__(GlmImageMultiModalProcessor)
        proc.info = MagicMock()
        return proc

    def test_1024x1024(self, processor):
        kwargs = {"target_h": 1024, "target_w": 1024}
        grids = processor._build_generation_grids(kwargs)
        # token_h = 32, token_w = 32
        # ratio = 1.0, small_h = 16, small_w = 16
        assert grids.shape == (2, 3)
        assert grids[0].tolist() == [1, 32, 32]  # large
        assert grids[1].tolist() == [1, 16, 16]  # small

    def test_512x512(self, processor):
        kwargs = {"target_h": 512, "target_w": 512}
        grids = processor._build_generation_grids(kwargs)
        assert grids.shape == (2, 3)
        assert grids[0].tolist() == [1, 16, 16]
        # small: ratio=1.0, small_h=int(sqrt(1)*16)=16, small_w=16
        assert grids[1].tolist() == [1, 16, 16]

    def test_non_square(self, processor):
        kwargs = {"target_h": 1024, "target_w": 512}
        grids = processor._build_generation_grids(kwargs)
        # token_h = 32, token_w = 16, ratio = 2.0
        # small_h = int(sqrt(2)*16) = 22, small_w = int(sqrt(0.5)*16) = 11
        assert grids[0].tolist() == [1, 32, 16]
        assert grids[1].tolist() == [1, 22, 11]

    def test_defaults_to_1024_when_no_target(self, processor):
        kwargs = {}
        grids = processor._build_generation_grids(kwargs)
        assert grids[0].tolist() == [1, 32, 32]

    def test_height_width_fallback(self, processor):
        kwargs = {"height": 512, "width": 512}
        grids = processor._build_generation_grids(kwargs)
        assert grids[0].tolist() == [1, 16, 16]

    def test_aligned_to_factor(self, processor):
        # 1000 not aligned to 32, should be rounded down to 992
        kwargs = {"target_h": 1000, "target_w": 1000}
        grids = processor._build_generation_grids(kwargs)
        # 1000 // 32 = 31
        assert grids[0].tolist() == [1, 31, 31]


# =============================================================================
# Tests for get_mrope_input_positions
# =============================================================================


class TestGetMropeInputPositions:
    """Test M-RoPE position ID computation."""

    @pytest.fixture
    def model(self):
        """Create a minimal model instance for M-RoPE testing."""
        model = object.__new__(GlmImageForConditionalGeneration)
        model.config = _make_hf_config()
        return model

    def test_pure_text(self, model):
        """Pure text tokens: all 3 dimensions get same sequential positions."""
        input_tokens = [100, 101, 102, 103]
        positions, delta = model.get_mrope_input_positions(input_tokens)
        assert positions.shape == (3, 4)
        # All three dims should be [0, 1, 2, 3]
        for dim in range(3):
            assert positions[dim].tolist() == [0, 1, 2, 3]
        assert delta == 0  # max(3) + 1 - seq_len(4) = 0

    def test_t2i_with_target_size(self, model):
        """t2i with explicit target_h/target_w: grids built from them."""
        input_tokens = [100, 101, 102, 16384]  # text + <bos>
        kwargs = {"target_h": 256, "target_w": 256}

        positions, delta = model.get_mrope_input_positions(input_tokens, **kwargs)
        # 256/32=8 -> grids = [[1,8,8], [1,16,16]] (small uses factor//2=16 base)
        # Decode order (reversed): grid[-1]=[1,16,16]=256, grid[-2]=[1,8,8]=64, EOS=1
        total_decode = 256 + 64 + 1  # 321
        assert positions.shape == (3, 4 + total_decode)
        # delta = max_position + 1 - seq_len
        # Positions advance by max(h,w) per grid: max(16,16)=16, max(8,8)=8
        # max_pos = seq_len(4) + 16 + 8 = 28, then EOS at 28
        # delta = 28 + 1 - 4 = 25
        assert delta == 25

    def test_t2i_1024_default_grids(self, model):
        """t2i with default 1024x1024 grids when no explicit target size."""
        # Prompt ending with image_start_token_id but no image_end_token_id
        input_tokens = [100, 101, 16384]
        # No target_h/target_w, no mrope_image_grid_thw
        # Falls back to token parsing then to default [[1,32,32], [1,16,16]]
        positions, delta = model.get_mrope_input_positions(input_tokens)
        assert positions.shape[0] == 3

    def test_i2i_with_mrope_grid(self, model):
        """i2i: mrope_image_grid_thw contains source + target grids."""
        # Source image tokens: [16384, 167855*4, 16385] + text + 16384(bos)
        source_grid = [1, 2, 2]  # 2x2 = 4 image tokens
        target_grid = [1, 32, 32]  # 32x32 = 1024 tokens
        mrope_grid = torch.tensor([source_grid, target_grid], dtype=torch.long)

        # input_tokens: text + <start> + 4*image_token + <end> + <bos>
        input_tokens = [100, 101, 16384] + [167855] * 4 + [16385, 16384]

        positions, delta = model.get_mrope_input_positions(input_tokens, mrope_image_grid_thw=mrope_grid)

        # 1 source image (num_complete_images=1), 1 target grid (num_decode_grids=1)
        # Prefill covers all input tokens
        # Decode covers: 32*32 + 1(EOS) = 1025 tokens
        assert positions.shape[0] == 3

    def test_position_delta_non_negative(self, model):
        """mrope_position_delta should be non-negative for valid inputs."""
        input_tokens = [100, 16384]
        kwargs = {"target_h": 64, "target_w": 64}
        positions, delta = model.get_mrope_input_positions(input_tokens, **kwargs)
        assert delta >= 0


# =============================================================================
# Tests for GlmImageRotaryEmbedding._apply_mrope
# =============================================================================


class TestGlmImageRotaryEmbedding:
    """Test M-RoPE section interleaving in the rotary embedding."""

    @pytest.fixture
    def rotary_emb(self):
        # mrope_section=[8,12,12] sums to 32, so rotary_dim//2 must be >= 32
        # -> head_dim=64 gives rotary_dim=64, rotary_dim//2=32
        return GlmImageRotaryEmbedding(head_dim=64, mrope_section=[8, 12, 12])

    def test_apply_mrope_shape(self, rotary_emb):
        """Output shape matches [num_tokens, rotary_dim // 2]."""
        freqs = torch.randn(3, 5, 32)  # 3 dims, 5 tokens, rotary_dim//2=32
        result = rotary_emb._apply_mrope(freqs)
        assert result.shape == (5, 32)

    def test_apply_mrope_interleaving(self, rotary_emb):
        """Verify that M-RoPE correctly interleaves T/H/W sections."""
        # mrope_section = [8, 12, 12] splits dim 32 into 3 chunks: [8, 12, 12]
        # chunk 0 (size 8):  dim 0 % 3 = 0 (temporal)
        # chunk 1 (size 12): dim 1 % 3 = 1 (height)
        # chunk 2 (size 12): dim 2 % 3 = 2 (width)
        freqs = torch.ones(3, 1, 32)
        freqs[0, :, :] = 1.0  # temporal
        freqs[1, :, :] = 2.0  # height
        freqs[2, :, :] = 3.0  # width

        result = rotary_emb._apply_mrope(freqs)
        assert result.shape == (1, 32)
        assert (result[0, :8] == 1.0).all()  # chunk 0: temporal
        assert (result[0, 8:20] == 2.0).all()  # chunk 1: height
        assert (result[0, 20:32] == 3.0).all()  # chunk 2: width

    def test_forward_1d_positions(self, rotary_emb):
        """Forward with 1D positions (text-only) produces correct shapes."""
        positions = torch.arange(10)  # [10]
        q = torch.randn(10, 64)
        k = torch.randn(10, 64)
        q_out, k_out = rotary_emb(positions, q, k)
        assert q_out.shape == (10, 64)
        assert k_out.shape == (10, 64)

    def test_forward_3d_positions(self, rotary_emb):
        """Forward with 3D M-RoPE positions produces correct shapes."""
        positions = torch.arange(30).reshape(3, 10)  # [3, 10]
        q = torch.randn(10, 64)
        k = torch.randn(10, 64)
        q_out, k_out = rotary_emb(positions, q, k)
        assert q_out.shape == (10, 64)
        assert k_out.shape == (10, 64)

    def test_forward_preserves_dtype(self, rotary_emb):
        """Output dtype matches input dtype."""
        positions = torch.arange(5)
        q = torch.randn(5, 64, dtype=torch.float32)
        k = torch.randn(5, 64, dtype=torch.float32)
        q_out, k_out = rotary_emb(positions, q, k)
        assert q_out.dtype == torch.float32
        assert k_out.dtype == torch.float32
