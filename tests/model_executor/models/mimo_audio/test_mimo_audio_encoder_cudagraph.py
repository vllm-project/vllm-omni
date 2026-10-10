# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from vllm.model_executor.models.interfaces import supports_encoder_cudagraph
from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_omni.model_executor.models.mimo_audio.mimo_audio import (
    MiMoAudioForConditionalGeneration,
)
from vllm_omni.model_executor.models.mimo_audio.mimo_audio_llm import MiMoAudioLLMForConditionalGeneration
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner


class _TinyInputLocalTransformer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        return_dict: bool,
        is_causal: bool,
    ) -> SimpleNamespace:
        assert return_dict
        assert not is_causal
        return SimpleNamespace(last_hidden_state=torch.tanh(self.proj(inputs_embeds)))


def _protocol_model(device: torch.device = torch.device("cpu")) -> MiMoAudioLLMForConditionalGeneration:
    model = MiMoAudioLLMForConditionalGeneration.__new__(MiMoAudioLLMForConditionalGeneration)
    nn.Module.__init__(model)
    model.group_size = 4
    model.input_local_config = SimpleNamespace(hidden_size=8)
    model.input_local_transformer = _TinyInputLocalTransformer(8).to(
        device=device,
        dtype=torch.bfloat16,
    )
    return model


def _generation_model() -> MiMoAudioLLMForConditionalGeneration:
    model = _protocol_model()
    model.audio_channels = 1
    model.hidden_states_downcast = nn.Identity()
    model.local_sampler = object()
    model.local_forward = Mock(return_value=torch.zeros((2, 1, model.group_size), dtype=torch.long))
    model.speech_embeddings = nn.ModuleList([nn.Embedding(2, model.input_local_config.hidden_size)])
    model.speech_empty_ids = [1]
    model._input_local_encoder_cudagraph_manager = None
    model.register_buffer(
        "_new_audio_emb_buffer",
        torch.zeros((2, 1, model.group_size, model.input_local_config.hidden_size)),
    )
    model.speech_group_downcast = nn.Identity()
    return model


def _stage_wrapper(
    child: MiMoAudioLLMForConditionalGeneration,
) -> MiMoAudioForConditionalGeneration:
    wrapper = MiMoAudioForConditionalGeneration.__new__(MiMoAudioForConditionalGeneration)
    nn.Module.__init__(wrapper)
    wrapper.fused_thinker_talker = child
    return wrapper


@pytest.mark.cpu
@pytest.mark.core_model
def test_stage_wrapper_exposes_protocol_and_runner_attaches_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = object()
    child = _protocol_model()
    wrapper = _stage_wrapper(child)

    assert supports_encoder_cudagraph(wrapper)
    assert wrapper.get_encoder_cudagraph_config().modalities == ["audio"]

    runner = object.__new__(OmniGPUModelRunner)
    runner.encoder_cudagraph_manager = manager
    runner.get_model = lambda: wrapper
    monkeypatch.setattr(GPUModelRunner, "_maybe_init_encoder_cudagraph_manager", lambda _self: None)

    OmniGPUModelRunner._maybe_init_encoder_cudagraph_manager(runner)

    assert child._input_local_encoder_cudagraph_manager is manager


@pytest.mark.cpu
@pytest.mark.core_model
def test_input_local_transformer_uses_attached_manager() -> None:
    model = _generation_model()
    manager = SimpleNamespace(
        is_captured=lambda: True,
        execute=Mock(
            side_effect=lambda kwargs: [
                torch.full(
                    (kwargs["inputs_embeds"].shape[0] * model.group_size, 8),
                    7.0,
                )
            ]
        ),
    )
    model.set_input_local_transformer_cudagraph_manager(manager)
    model.encoder_eager_forward = Mock(side_effect=AssertionError("eager fallback must not run"))

    _, audio_embeddings = model._generate_speech_tokens_and_audio_embeddings(torch.zeros((2, 1, 8)))

    manager.execute.assert_called_once()
    torch.testing.assert_close(audio_embeddings, torch.full((2, 1, 32), 7.0))


@pytest.mark.cpu
@pytest.mark.core_model
def test_input_local_transformer_stays_eager_without_manager() -> None:
    model = _generation_model()
    eager = Mock(
        side_effect=lambda kwargs, path="default": torch.full(
            (kwargs["inputs_embeds"].shape[0] * model.group_size, 8),
            3.0,
        )
    )
    model.encoder_eager_forward = eager

    _, audio_embeddings = model._generate_speech_tokens_and_audio_embeddings(torch.zeros((2, 1, 8)))

    eager.assert_called_once()
    torch.testing.assert_close(audio_embeddings, torch.full((2, 1, 32), 3.0))


def _manager_config() -> SimpleNamespace:
    multimodal_config = SimpleNamespace(
        get_limit_per_prompt=lambda modality: 0,
        mm_encoder_tp_mode="weights",
    )
    return SimpleNamespace(
        compilation_config=SimpleNamespace(
            encoder_cudagraph_token_budgets=[4, 16],
            encoder_cudagraph_max_vision_items_per_batch=1,
            encoder_cudagraph_max_frames_per_batch=None,
        ),
        model_config=SimpleNamespace(multimodal_config=multimodal_config),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )


@pytest.mark.cuda
@pytest.mark.core_model
def test_manager_matches_eager_across_tiers_and_falls_back_out_of_range() -> None:
    device = torch.device("cuda")
    child = _protocol_model(device)
    model = _stage_wrapper(child)
    manager = EncoderCudaGraphManager(
        _manager_config(),
        device=device,
        dtype=torch.bfloat16,
        model=model,
    )
    manager.capture(torch.cuda.graph_pool_handle())

    for rows, expected_hits, expected_misses in (
        (1, 1, 0),
        (3, 2, 0),
        (5, 2, 1),
    ):
        inputs = torch.linspace(
            -1,
            1,
            rows * child.group_size * child.input_local_config.hidden_size,
            device=device,
            dtype=torch.bfloat16,
        ).reshape(rows, child.group_size, child.input_local_config.hidden_size)
        mm_kwargs = {"inputs_embeds": inputs}
        eager = model.encoder_eager_forward(mm_kwargs)
        actual = manager.execute(mm_kwargs)[0]

        torch.testing.assert_close(actual, eager, atol=1e-3, rtol=1e-3)
        stats = manager.get_cumulative_stats()
        assert stats["graph_hits"] == expected_hits
        assert stats["graph_misses"] == expected_misses
