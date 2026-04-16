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

import pytest
import torch
from pytest_mock import MockerFixture

# Direct file import to avoid vllm_omni.__init__ patch dependencies.
_MODELS = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    os.pardir,
    os.pardir,
    os.pardir,
    "vllm_omni",
    "model_executor",
    "models",
)
_BASE = os.path.join(_MODELS, "qwen3_tts")
_COMMON = os.path.join(_MODELS, "common")


def _load_module(name: str, filename: str):
    path = os.path.abspath(os.path.join(_BASE, filename))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # register before exec (needed for dataclasses etc.)
    spec.loader.exec_module(mod)
    return mod


def _build_mock_modules(mocker: MockerFixture) -> dict[str, object]:
    """Build the dict of modules to inject into sys.modules."""
    platforms_mock = mocker.MagicMock()
    platforms_mock.current_omni_platform.supports_torch_inductor.return_value = False

    logger_mock = mocker.MagicMock()
    logger_mock.init_logger = lambda name: mocker.MagicMock()

    vllm_config_mod = mocker.MagicMock()
    vllm_config_mod.set_current_vllm_config = lambda cfg: mocker.MagicMock(
        __enter__=mocker.MagicMock(),
        __exit__=mocker.MagicMock(),
    )

    weight_utils_mock = mocker.MagicMock()
    weight_utils_mock.default_weight_loader = lambda p, w: None

    tts_pkg = types.ModuleType("vllm_omni.model_executor.models.qwen3_tts")
    tts_pkg.__path__ = [os.path.abspath(_BASE)]

    common_pkg = types.ModuleType("vllm_omni.model_executor.models.common")
    common_pkg.__path__ = [os.path.abspath(_COMMON)]

    models_pkg = types.ModuleType("vllm_omni.model_executor.models")
    models_pkg.__path__ = [os.path.abspath(_MODELS)]

    vllm_parallel_mock = mocker.MagicMock()
    vllm_parallel_mock.VocabParallelEmbedding = torch.nn.Embedding

    return {
        "vllm_omni": mocker.MagicMock(),
        "vllm_omni.platforms": platforms_mock,
        "vllm.logger": logger_mock,
        "vllm.config": mocker.MagicMock(),
        "vllm.config.vllm": vllm_config_mod,
        "vllm.model_executor.model_loader.weight_utils": weight_utils_mock,
        "vllm.model_executor.layers.vocab_parallel_embedding": vllm_parallel_mock,
        "vllm_omni.model_executor": types.ModuleType("vllm_omni.model_executor"),
        "vllm_omni.model_executor.models": models_pkg,
        "vllm_omni.model_executor.models.common": common_pkg,
        "vllm_omni.model_executor.models.qwen3_tts": tts_pkg,
    }


def _load_target_classes(mocker: MockerFixture):
    """Load config and code predictor modules with mocked dependencies.

    Uses mocker.patch.dict to ensure sys.modules is always restored, even on failure.
    """
    mocks = _build_mock_modules(mocker)
    mocker.patch.dict(sys.modules, mocks)
    config_mod = _load_module(
        "vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts",
        "configuration_qwen3_tts.py",
    )
    sys.modules["vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts"] = config_mod

    # Load the shared common module (thin wrappers import from it)
    common_cp_path = os.path.abspath(os.path.join(_COMMON, "qwen3_code_predictor.py"))
    common_spec = importlib.util.spec_from_file_location(
        "vllm_omni.model_executor.models.common.qwen3_code_predictor", common_cp_path
    )
    common_cp_mod = importlib.util.module_from_spec(common_spec)
    sys.modules["vllm_omni.model_executor.models.common.qwen3_code_predictor"] = common_cp_mod
    common_spec.loader.exec_module(common_cp_mod)

    cp_mod = _load_module(
        "vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code_predictor_vllm",
        "qwen3_tts_code_predictor_vllm.py",
    )

    return config_mod, cp_mod


@pytest.fixture
def loaded_target_classes(mocker: MockerFixture):
    config_mod, cp_mod = _load_target_classes(mocker)
    return (
        config_mod.Qwen3TTSTalkerCodePredictorConfig,
        config_mod.Qwen3TTSTalkerConfig,
        cp_mod.Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM,
        cp_mod.Qwen3TTSTalkerCodePredictorModelVLLM,
        cp_mod.CodePredictorWrapperConfig,
    )


def _make_tiny_config(loaded_target_classes) -> tuple:
    """Create minimal configs for a tiny code predictor model."""
    (
        qwen3_tts_talker_code_predictor_config,
        qwen3_tts_talker_config,
        _,
        _,
        _,
    ) = loaded_target_classes
    cp_config = qwen3_tts_talker_code_predictor_config(
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
    talker_config = qwen3_tts_talker_config(
        hidden_size=32,
        num_code_groups=4,
    )
    return cp_config, talker_config


def _make_vllm_config(mocker: MockerFixture, max_num_seqs: int = 4):
    """Create a mock VllmConfig with scheduler_config."""
    vllm_config = mocker.MagicMock()
    vllm_config.scheduler_config.max_num_seqs = max_num_seqs
    return vllm_config


class TestCodePredictorDtypeAlignment:
    """Test that code predictor buffers match model parameter dtype."""

    def test_ensure_buffers_uses_given_dtype(self, mocker: MockerFixture, loaded_target_classes) -> None:
        """_ensure_buffers should create proj_buf with the given dtype."""
        _, _, code_predictor_wrapper, _, _ = loaded_target_classes
        cp_config, talker_config = _make_tiny_config(loaded_target_classes)
        vllm_config = _make_vllm_config(mocker)

        predictor = code_predictor_wrapper(
            vllm_config=vllm_config,
            config=cp_config,
            talker_config=talker_config,
        )

        # Create buffer in float16
        predictor._ensure_buffers(torch.device("cpu"), torch.float16, 4)
        assert predictor._proj_buf is not None
        assert predictor._proj_buf.dtype == torch.float16

        # Re-create buffer in float32 (different dtype triggers re-allocation)
        predictor._ensure_buffers(torch.device("cpu"), torch.float32, 4)
        assert predictor._proj_buf.dtype == torch.float32

    def test_warmup_aligns_buffer_to_model_params(self, mocker: MockerFixture, loaded_target_classes) -> None:
        """_warmup_buckets should align proj_buf dtype to model parameters."""
        _, _, code_predictor_wrapper, _, _ = loaded_target_classes
        cp_config, talker_config = _make_tiny_config(loaded_target_classes)
        vllm_config = _make_vllm_config(mocker, max_num_seqs=2)

        predictor = code_predictor_wrapper(
            vllm_config=vllm_config,
            config=cp_config,
            talker_config=talker_config,
        )

        # Cast model to float16 (simulating vLLM loading weights in half precision)
        predictor = predictor.to(torch.float16)

        # Pre-create proj_buf with WRONG dtype (float32) — simulating the bug
        predictor._ensure_buffers(torch.device("cpu"), torch.float32, 2)
        assert predictor._proj_buf.dtype == torch.float32

        # Simulate _setup_compile having cached model dtype and compiled forward
        predictor._model_dtype = torch.float16
        predictor._compiled_model_fwd = predictor.model.forward

        # _warmup_buckets should fix the dtype mismatch
        predictor._warmup_buckets()

        assert predictor._proj_buf.dtype == torch.float16

    def test_setup_compile_caches_model_dtype(self, mocker: MockerFixture, loaded_target_classes) -> None:
        """_setup_compile should cache model parameter dtype."""
        _, _, code_predictor_wrapper, _, _ = loaded_target_classes
        cp_config, talker_config = _make_tiny_config(loaded_target_classes)
        vllm_config = _make_vllm_config(mocker, max_num_seqs=2)

        predictor = code_predictor_wrapper(
            vllm_config=vllm_config,
            config=cp_config,
            talker_config=talker_config,
        )
        predictor = predictor.to(torch.float16)

        assert predictor._model_dtype is None
        predictor._setup_compile()
        assert predictor._model_dtype == torch.float16

    def test_forward_with_mismatched_input_dtype(self, mocker: MockerFixture, loaded_target_classes) -> None:
        """forward() should not crash when inputs are float32 but model is float16."""
        _, _, code_predictor_wrapper, _, _ = loaded_target_classes
        cp_config, talker_config = _make_tiny_config(loaded_target_classes)
        vllm_config = _make_vllm_config(mocker, max_num_seqs=2)

        predictor = code_predictor_wrapper(
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

    def test_model_forward_float16(self, loaded_target_classes) -> None:
        """Inner model forward should work in float16."""
        _, _, _, code_predictor_model, _ = loaded_target_classes
        cp_config, _ = _make_tiny_config(loaded_target_classes)
        model = code_predictor_model(cp_config, embedding_dim=32).to(torch.float16)

        bsz, seq_len = 1, 4
        inputs = torch.randn(bsz, seq_len, 32, dtype=torch.float16)
        pos_ids = torch.arange(seq_len).unsqueeze(0).expand(bsz, -1)

        output = model(inputs, pos_ids)
        assert output.dtype == torch.float16
        assert output.shape == (bsz, seq_len, 32)

    def test_model_forward_float32(self, loaded_target_classes) -> None:
        """Inner model forward should work in float32."""
        _, _, _, code_predictor_model, _ = loaded_target_classes
        cp_config, _ = _make_tiny_config(loaded_target_classes)
        model = code_predictor_model(cp_config, embedding_dim=32).to(torch.float32)

        bsz, seq_len = 1, 4
        inputs = torch.randn(bsz, seq_len, 32, dtype=torch.float32)
        pos_ids = torch.arange(seq_len).unsqueeze(0).expand(bsz, -1)

        output = model(inputs, pos_ids)
        assert output.dtype == torch.float32
        assert output.shape == (bsz, seq_len, 32)


class TestCodePredictorWrapperConfig:
    """Test wrapper configuration for different models."""

    def test_omni_config(self, loaded_target_classes) -> None:
        """Qwen3-Omni uses correct wrapper config."""
        _, _, _, _, code_predictor_wrapper_config = loaded_target_classes
        config = code_predictor_wrapper_config(
            use_cuda_graphs=False,
            use_parallel_embedding=True,
            use_projection=False,
            return_proj_buf=True,
            sampling_mode="stored",
        )
        assert config.use_cuda_graphs is False
        assert config.use_parallel_embedding is True
        assert config.return_proj_buf is True
        assert config.sampling_mode == "stored"

    def test_tts_config(self, loaded_target_classes) -> None:
        """Qwen3-TTS uses correct wrapper config."""
        _, _, _, _, code_predictor_wrapper_config = loaded_target_classes
        config = code_predictor_wrapper_config(
            use_cuda_graphs=True,
            use_parallel_embedding=False,
            use_projection=True,
            return_proj_buf=False,
            sampling_mode="per_call",
        )
        assert config.use_cuda_graphs is True
        assert config.use_parallel_embedding is False
        assert config.return_proj_buf is False
        assert config.sampling_mode == "per_call"
