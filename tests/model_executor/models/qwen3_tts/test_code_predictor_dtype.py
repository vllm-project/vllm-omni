# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for code predictor dtype alignment (fix for #2385).

Verifies that the code predictor handles dtype mismatches between input
tensors and model parameters without raising RuntimeError. This can happen
when model weights are loaded in float16/bfloat16 but upstream modules
produce float32 hidden states.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from unittest.mock import MagicMock, patch

import torch

# Direct file import to avoid vllm_omni.__init__ patch dependencies.
_BASE = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    os.pardir,
    os.pardir,
    os.pardir,
    "vllm_omni",
    "model_executor",
    "models",
    "qwen3_tts",
)


def _load_module(name: str, filename: str):
    path = os.path.abspath(os.path.join(_BASE, filename))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_mock_modules() -> dict[str, object]:
    """Build the dict of modules to inject into sys.modules."""
    platforms_mock = MagicMock()
    platforms_mock.current_omni_platform.supports_torch_inductor.return_value = False

    logger_mock = MagicMock()
    logger_mock.init_logger = lambda name: MagicMock()

    vllm_config_mod = MagicMock()
    vllm_config_mod.set_current_vllm_config = lambda cfg: MagicMock(__enter__=MagicMock(), __exit__=MagicMock())

    weight_utils_mock = MagicMock()
    weight_utils_mock.default_weight_loader = lambda p, w: None

    pkg = types.ModuleType("vllm_omni.model_executor.models.qwen3_tts")
    pkg.__path__ = [os.path.abspath(_BASE)]

    return {
        "vllm_omni": MagicMock(),
        "vllm_omni.platforms": platforms_mock,
        "vllm.logger": logger_mock,
        "vllm.config": MagicMock(),
        "vllm.config.vllm": vllm_config_mod,
        "vllm.model_executor.model_loader.weight_utils": weight_utils_mock,
        "vllm_omni.model_executor": types.ModuleType("vllm_omni.model_executor"),
        "vllm_omni.model_executor.models": types.ModuleType("vllm_omni.model_executor.models"),
        "vllm_omni.model_executor.models.qwen3_tts": pkg,
    }


def _load_target_classes():
    """Load config and code predictor modules with mocked dependencies.

    Uses patch.dict to ensure sys.modules is always restored, even on failure.
    """
    mocks = _build_mock_modules()
    with patch.dict(sys.modules, mocks):
        config_mod = _load_module(
            "vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts",
            "configuration_qwen3_tts.py",
        )
        sys.modules["vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts"] = config_mod

        cp_mod = _load_module(
            "vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code_predictor_vllm",
            "qwen3_tts_code_predictor_vllm.py",
        )

    return config_mod, cp_mod


_config_mod, _cp_mod = _load_target_classes()

Qwen3TTSTalkerCodePredictorConfig = _config_mod.Qwen3TTSTalkerCodePredictorConfig
Qwen3TTSTalkerConfig = _config_mod.Qwen3TTSTalkerConfig
CodePredictorWrapper = _cp_mod.Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM
CodePredictorModel = _cp_mod.Qwen3TTSTalkerCodePredictorModelVLLM


def _make_tiny_config() -> tuple:
    """Create minimal configs for a tiny code predictor model."""
    cp_config = Qwen3TTSTalkerCodePredictorConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_code_groups=4,
        rms_norm_eps=1e-6,
    )
    talker_config = Qwen3TTSTalkerConfig(
        hidden_size=32,
        num_code_groups=4,
    )
    return cp_config, talker_config


def _make_vllm_config(max_num_seqs: int = 4) -> MagicMock:
    """Create a mock VllmConfig with scheduler_config."""
    vllm_config = MagicMock()
    vllm_config.scheduler_config.max_num_seqs = max_num_seqs
    return vllm_config


class TestCodePredictorDtypeAlignment:
    """Test that code predictor buffers match model parameter dtype."""

    def test_ensure_buffers_uses_given_dtype(self) -> None:
        """_ensure_buffers should create proj_buf with the given dtype."""
        cp_config, talker_config = _make_tiny_config()
        vllm_config = _make_vllm_config()

        predictor = CodePredictorWrapper(
            vllm_config=vllm_config,
            config=cp_config,
            talker_config=talker_config,
        )

        # Create buffer in float16
        predictor._ensure_buffers(torch.device("cpu"), torch.float16)
        assert predictor._proj_buf is not None
        assert predictor._proj_buf.dtype == torch.float16

        # Re-create buffer in float32 (different dtype triggers re-allocation)
        predictor._ensure_buffers(torch.device("cpu"), torch.float32)
        assert predictor._proj_buf.dtype == torch.float32

    def test_warmup_aligns_buffer_to_model_params(self) -> None:
        """_warmup_buckets should align proj_buf dtype to model parameters."""
        cp_config, talker_config = _make_tiny_config()
        vllm_config = _make_vllm_config(max_num_seqs=2)

        predictor = CodePredictorWrapper(
            vllm_config=vllm_config,
            config=cp_config,
            talker_config=talker_config,
        )

        # Cast model to float16 (simulating vLLM loading weights in half precision)
        predictor = predictor.to(torch.float16)

        # Pre-create proj_buf with WRONG dtype (float32) — simulating the bug
        predictor._ensure_buffers(torch.device("cpu"), torch.float32)
        assert predictor._proj_buf.dtype == torch.float32

        # Simulate _setup_compile having cached model dtype and compiled forward
        predictor._model_dtype = torch.float16
        predictor._compiled_model_fwd = predictor.model.forward

        # _warmup_buckets should fix the dtype mismatch
        predictor._warmup_buckets()

        assert predictor._proj_buf.dtype == torch.float16

    def test_setup_compile_caches_model_dtype(self) -> None:
        """_setup_compile should cache model parameter dtype."""
        cp_config, talker_config = _make_tiny_config()
        vllm_config = _make_vllm_config(max_num_seqs=2)

        predictor = CodePredictorWrapper(
            vllm_config=vllm_config,
            config=cp_config,
            talker_config=talker_config,
        )
        predictor = predictor.to(torch.float16)

        assert predictor._model_dtype is None
        predictor._setup_compile()
        assert predictor._model_dtype == torch.float16

    def test_forward_with_mismatched_input_dtype(self) -> None:
        """forward() should not crash when inputs are float32 but model is float16."""
        cp_config, talker_config = _make_tiny_config()
        vllm_config = _make_vllm_config(max_num_seqs=2)

        predictor = CodePredictorWrapper(
            vllm_config=vllm_config,
            config=cp_config,
            talker_config=talker_config,
        )

        # Model in float16
        predictor = predictor.to(torch.float16)

        bsz = 1
        num_groups = cp_config.num_code_groups
        hidden = talker_config.hidden_size

        # Inputs in float32 (simulating the dtype mismatch from #2385)
        layer0_code = torch.zeros(bsz, dtype=torch.long)
        layer0_embed = torch.randn(bsz, hidden, dtype=torch.float32)
        last_talker_hidden = torch.randn(bsz, hidden, dtype=torch.float32)

        # This should NOT raise RuntimeError about dtype mismatch
        result = predictor(
            layer0_code=layer0_code,
            layer0_embed=layer0_embed,
            last_talker_hidden=last_talker_hidden,
            do_sample=False,
        )

        assert result.shape == (bsz, num_groups)
        assert result.dtype == torch.long


class TestCodePredictorModelDtype:
    """Test the inner model forward with different dtypes."""

    def test_model_forward_float16(self) -> None:
        """Inner model forward should work in float16."""
        cp_config, _ = _make_tiny_config()
        model = CodePredictorModel(cp_config, talker_hidden_size=32).to(torch.float16)

        bsz, seq_len = 1, 4
        inputs = torch.randn(bsz, seq_len, 32, dtype=torch.float16)
        pos_ids = torch.arange(seq_len).unsqueeze(0).expand(bsz, -1)

        output = model(inputs, pos_ids)
        assert output.dtype == torch.float16
        assert output.shape == (bsz, seq_len, 32)

    def test_model_forward_float32(self) -> None:
        """Inner model forward should work in float32."""
        cp_config, _ = _make_tiny_config()
        model = CodePredictorModel(cp_config, talker_hidden_size=32).to(torch.float32)

        bsz, seq_len = 1, 4
        inputs = torch.randn(bsz, seq_len, 32, dtype=torch.float32)
        pos_ids = torch.arange(seq_len).unsqueeze(0).expand(bsz, -1)

        output = model(inputs, pos_ids)
        assert output.dtype == torch.float32
        assert output.shape == (bsz, seq_len, 32)
