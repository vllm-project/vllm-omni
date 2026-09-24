# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
    MiniCPMO45OmniForConditionalGeneration,
)
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    MiniCPMO45OmniLLMForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_multimodal_runner_preserves_forwarded_token_identities(monkeypatch):
    stage_model = torch.nn.Module()
    stage_model.make_empty_intermediate_tensors = lambda: None
    monkeypatch.setattr(
        "vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni.init_vllm_registered_model",
        lambda **kwargs: stage_model,
    )
    monkeypatch.setattr(
        "vllm_omni.model_executor.models.minicpmo_4_5.duplex.compat.patch_minicpmo_remote_config",
        lambda config: None,
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(), multimodal_config=None, model_stage="llm")
    )
    wrapper = MiniCPMO45OmniForConditionalGeneration(vllm_config=config)
    hidden = torch.randn(2, 4)
    wrapper.thinker.forward = MagicMock(return_value=hidden)
    token_buffer = torch.tensor([99692, 3837, 0, 0])
    embed_buffer = torch.randn(4, 4)
    runner = SimpleNamespace(
        model=wrapper,
        input_ids=SimpleNamespace(gpu=token_buffer),
        inputs_embeds=SimpleNamespace(gpu=embed_buffer),
    )

    # This same upstream path prepares normal multimodal forwards and graph
    # capture inputs. Embeddings must not replace the real token identities.
    input_ids, inputs_embeds = GPUModelRunner._prepare_mm_inputs(runner, 2)
    out = wrapper.forward(input_ids=input_ids, positions=torch.tensor([10, 11]), inputs_embeds=inputs_embeds)

    torch.testing.assert_close(out.multimodal_outputs["latent_input_ids"], token_buffer[:2, None])
    torch.testing.assert_close(out.multimodal_outputs["latent_positions"], torch.tensor([[10], [11]]))
    torch.testing.assert_close(wrapper.thinker.forward.call_args.kwargs["inputs_embeds"], embed_buffer[:2])

    # The codec Talker has no text vocabulary for randomized raw-token
    # profiling. It must retain the existing embeddings-only input path.
    config.model_config.model_stage = "tts"
    runner.model = MiniCPMO45OmniForConditionalGeneration(vllm_config=config)
    talker_ids, talker_embeds = GPUModelRunner._prepare_mm_inputs(runner, 2)
    assert talker_ids is None
    torch.testing.assert_close(talker_embeds, embed_buffer[:2])


def test_thinker_forward_returns_bare_hidden_states():
    """Verify that MiniCPMO45OmniLLMForConditionalGeneration.forward returns
    a bare hidden_states Tensor (not a tuple), which stock vLLM GPUModelRunner
    expects.
    """
    model = MiniCPMO45OmniLLMForConditionalGeneration.__new__(MiniCPMO45OmniLLMForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.llm = MagicMock()
    mock_hidden_states = torch.randn(4, 4096)
    model.llm.model = MagicMock(return_value=mock_hidden_states)

    out = model.forward(
        input_ids=torch.tensor([1, 2, 3, 4]),
        positions=torch.tensor([0, 1, 2, 3]),
        inputs_embeds=torch.randn(4, 4096),
    )

    assert isinstance(out, torch.Tensor), f"Expected bare torch.Tensor, got {type(out)}"
    assert out.shape == (4, 4096)
    assert torch.equal(out, mock_hidden_states)


def test_thinker_forward_builds_embeddings_once_when_inputs_embeds_none():
    """Verify that when inputs_embeds is None, forward builds input embeddings
    exactly once and forwards them to self.llm.model with input_ids=None."""
    model = MiniCPMO45OmniLLMForConditionalGeneration.__new__(MiniCPMO45OmniLLMForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.llm = MagicMock()
    mock_hidden_states = torch.randn(4, 4096)
    model.llm.model = MagicMock(return_value=mock_hidden_states)

    mock_mm_embeds = [torch.randn(2, 4096)]
    mock_input_embeds = torch.randn(4, 4096)
    model.get_multimodal_embeddings = MagicMock(return_value=mock_mm_embeds)
    model.get_input_embeddings = MagicMock(return_value=mock_input_embeds)

    input_ids = torch.tensor([1, 2, 3, 4])
    positions = torch.tensor([0, 1, 2, 3])

    out = model.forward(
        input_ids=input_ids,
        positions=positions,
        inputs_embeds=None,
    )

    # Assert input embeddings are built exactly once
    model.get_multimodal_embeddings.assert_called_once()
    model.get_input_embeddings.assert_called_once_with(input_ids, mock_mm_embeds)

    # Verify forwarded to self.llm.model with input_ids=None and inputs_embeds
    model.llm.model.assert_called_once_with(
        None,
        positions,
        None,
        inputs_embeds=mock_input_embeds,
    )

    assert isinstance(out, torch.Tensor), f"Expected bare torch.Tensor, got {type(out)}"
    assert torch.equal(out, mock_hidden_states)


@pytest.mark.parametrize(
    ("thinker_shape", "expected_shape"),
    [
        ((1, 4096), (1, 4096)),
        ((5, 4096), (5, 4096)),
        ((1, 1, 4096), (1, 4096)),
        ((1, 5, 4096), (5, 4096)),
    ],
)
def test_wrapper_shape_normalization(thinker_shape, expected_shape):
    """Verify that MiniCPMO45OmniForConditionalGeneration normalizes 3D
    hidden states of shape (1, seq_len, H) to (seq_len, H), while leaving
    2D hidden states intact (especially (1, H) for single-token decode)."""
    wrapper = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    torch.nn.Module.__init__(wrapper)
    wrapper.model_stage = "llm"
    wrapper.thinker = torch.nn.Module()
    mock_hidden = torch.randn(*thinker_shape)
    wrapper.thinker.forward = MagicMock(return_value=mock_hidden)

    out = wrapper.forward(
        input_ids=torch.tensor([1]),
        positions=torch.tensor([0]),
    )

    assert out.text_hidden_states.shape == expected_shape, (
        f"Expected text_hidden_states shape {expected_shape}, got {out.text_hidden_states.shape}"
    )
    assert out.multimodal_outputs["latent"].shape == expected_shape, (
        f"Expected latent shape {expected_shape}, got {out.multimodal_outputs['latent'].shape}"
    )


if __name__ == "__main__":
    test_thinker_forward_returns_bare_hidden_states()
    test_thinker_forward_builds_embeddings_once_when_inputs_embeds_none()
    for shape, exp in [
        ((1, 4096), (1, 4096)),
        ((5, 4096), (5, 4096)),
        ((1, 1, 4096), (1, 4096)),
        ((1, 5, 4096), (5, 4096)),
    ]:
        test_wrapper_shape_normalization(shape, exp)
    print("ALL TESTS PASSED SUCCESSFULLY!")
