"""Unit tests for init_omni_model_state factory dispatch."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_omni.worker_v2.model_states import (
    _OMNI_ARCHITECTURES,
    init_omni_model_state,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_vllm_config(architectures):
    model_config = SimpleNamespace(architectures=architectures)
    return SimpleNamespace(model_config=model_config)


@patch("vllm_omni.worker_v2.model_states.omni_model_state.OmniModelState.__init__", return_value=None)
def test_qwen3_omni_dispatches_to_omni_model_state(mock_init):
    cfg = _make_vllm_config(["Qwen3OmniMoeForConditionalGeneration"])
    model = MagicMock()
    device = torch.device("cpu")

    state = init_omni_model_state(cfg, model, None, device)

    from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

    assert isinstance(state, OmniModelState)


@patch("vllm_omni.worker_v2.model_states.omni_model_state.OmniModelState.__init__", return_value=None)
def test_mammoth_dispatches_to_omni_model_state(mock_init):
    cfg = _make_vllm_config(["MammothModa2ForConditionalGeneration"])
    model = MagicMock()
    device = torch.device("cpu")

    state = init_omni_model_state(cfg, model, None, device)

    from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

    assert isinstance(state, OmniModelState)


@patch("vllm_omni.worker_v2.model_states._upstream_init_model_state")
def test_unknown_arch_delegates_to_upstream(mock_upstream):
    mock_upstream.return_value = MagicMock()
    cfg = _make_vllm_config(["LlamaForCausalLM"])
    model = MagicMock()
    device = torch.device("cpu")

    state = init_omni_model_state(cfg, model, None, device)

    mock_upstream.assert_called_once_with(cfg, model, None, device)
    assert state is mock_upstream.return_value


def test_omni_architectures_set_contains_expected():
    expected = {
        "Qwen3OmniMoeForConditionalGeneration",
        "Qwen2_5OmniForConditionalGeneration",
        "MammothModa2ForConditionalGeneration",
        "MiMoAudioForConditionalGeneration",
        "MammothModa2ARForConditionalGeneration",
        "Qwen3TTSTalkerForConditionalGeneration",
        "Qwen3TTSCode2Wav",
    }
    assert _OMNI_ARCHITECTURES == expected


@patch("vllm_omni.worker_v2.model_states._upstream_init_model_state")
def test_none_architectures_delegates_to_upstream(mock_upstream):
    mock_upstream.return_value = MagicMock()
    cfg = _make_vllm_config(None)
    model = MagicMock()
    device = torch.device("cpu")

    init_omni_model_state(cfg, model, None, device)

    mock_upstream.assert_called_once()
