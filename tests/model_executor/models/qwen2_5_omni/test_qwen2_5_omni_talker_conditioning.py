# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression coverage for Qwen2.5-Omni thinker-to-talker conditioning."""

from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniConfig
from vllm.config import ModelConfig, VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.config.multimodal import MultiModalConfig
from vllm.v1.worker.gpu import cudagraph_utils
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.qwen2_5_omni import qwen2_5_omni
from vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni import (
    Qwen2_5OmniForConditionalGeneration,
)
from vllm_omni.model_executor.stage_input_processors.qwen2_5_omni import (
    thinker2talker_full_payload,
)
from vllm_omni.worker_v2 import staged_writes
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState
from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_AUDIO_TOKEN_ID = 50
_IMAGE_TOKEN_ID = 51
_VIDEO_TOKEN_ID = 52
_TEXT_EOS_TOKEN_ID = 60
_TEXT_PAD_TOKEN_ID = 61


@dataclass(frozen=True)
class _Status:
    name: str


@dataclass
class _Request:
    prompt_token_ids: list[int]
    output_token_ids: list[int]
    status: _Status
    request_id: str = "request-0"
    sampling_params: None = None
    additional_information: None = None

    @property
    def all_token_ids(self) -> list[int]:
        return [*self.prompt_token_ids, *self.output_token_ids]


@dataclass(frozen=True)
class _ThinkerConfig:
    audio_token_index: int = _AUDIO_TOKEN_ID
    image_token_index: int = _IMAGE_TOKEN_ID
    video_token_index: int = _VIDEO_TOKEN_ID


@dataclass(frozen=True)
class _TalkerConfig:
    hidden_size: int = 2
    tts_codec_mask_token_id: int = 70
    tts_codec_pad_token_id: int = 71
    tts_codec_start_token_id: int = 72


class _Talker(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.config = _TalkerConfig()


def _bare_talker_model() -> Qwen2_5OmniForConditionalGeneration:
    model = object.__new__(Qwen2_5OmniForConditionalGeneration)
    nn.Module.__init__(model)

    talker = _Talker()
    model.model_stage = "talker"
    model.model = talker
    model.talker = talker
    model.thinker_config = _ThinkerConfig()
    model.talker_config = talker.config
    model.thinker_embedding = nn.Embedding(64, 2)
    with torch.no_grad():
        token_values = torch.arange(64, dtype=torch.float32)
        model.thinker_embedding.weight.copy_(torch.stack([token_values, token_values + 0.5], dim=1))

    model.embed_text_bos_token = model.thinker_embedding(torch.tensor([59]))
    model.embed_text_spk_tokens = {"Chelsie": model.embed_text_bos_token}
    model.embed_text_eos_token = model.thinker_embedding(torch.tensor([_TEXT_EOS_TOKEN_ID]))
    model.embed_text_pad_token = model.thinker_embedding(torch.tensor([_TEXT_PAD_TOKEN_ID]))
    model.embed_codec_pad_token = torch.tensor([[100.0, 101.0]])
    model.embed_codec_bos_token = torch.tensor([[200.0, 201.0]])
    return model


def _assert_conditioning(status_name: str, output_token_ids: list[int], expected_output_ids: list[int]) -> None:
    model = _bare_talker_model()
    prompt_token_ids = [1, _AUDIO_TOKEN_ID, _IMAGE_TOKEN_ID, _VIDEO_TOKEN_ID]
    request = _Request(
        prompt_token_ids=prompt_token_ids,
        output_token_ids=output_token_ids,
        status=_Status(status_name),
    )
    # Generation captures the prompt plus one decode row for every sampled
    # token except the final token that closes the request.
    hidden_row_count = len(request.all_token_ids) - 1
    hidden_states = torch.arange(hidden_row_count * 2, dtype=torch.float32).reshape(-1, 2) + 10
    payload = thinker2talker_full_payload(
        transfer_manager=None,
        pooling_output={"hidden": hidden_states},
        request=request,
    )

    assert payload is not None
    assert payload["ids"]["output"] == expected_output_ids

    prompt_length = len(prompt_token_ids) + 2
    input_ids, input_embeds, updates = model.thinker_to_talker_process(
        input_ids=torch.zeros(prompt_length, dtype=torch.long),
        input_embeds=torch.zeros(prompt_length, 2),
        payload=payload,
    )

    prompt_hidden = payload["embed"]["prefill"]
    prompt_token_embeds = model.thinker_embedding(torch.tensor(prompt_token_ids))
    prompt_token_embeds = prompt_token_embeds.masked_fill(
        torch.tensor([False, True, True, True]).unsqueeze(-1),
        0,
    )
    expected_prompt_conditioning = prompt_hidden + prompt_token_embeds

    reply_hidden = payload["hidden_states"]["output"]
    reply_token_embeds = model.thinker_embedding(torch.tensor(expected_output_ids))
    expected_reply_conditioning = reply_hidden + reply_token_embeds
    expected_input_embeds = torch.cat(
        [
            expected_prompt_conditioning,
            model.embed_text_bos_token + model.embed_codec_pad_token,
            expected_reply_conditioning[:1] + model.embed_codec_bos_token,
        ]
    )
    expected_reply_queue = torch.cat(
        [
            expected_reply_conditioning[1:],
            model.embed_text_eos_token,
            model.embed_text_pad_token,
        ]
    )

    assert input_ids.tolist() == [
        *([model.talker_config.tts_codec_mask_token_id] * len(prompt_token_ids)),
        model.talker_config.tts_codec_pad_token_id,
        model.talker_config.tts_codec_start_token_id,
    ]
    torch.testing.assert_close(input_embeds, expected_input_embeds)
    torch.testing.assert_close(updates["embed"]["thinker_reply"], expected_reply_queue)


def test_natural_completion_replaces_stop_row_with_text_eos_and_pad() -> None:
    _assert_conditioning(
        status_name="FINISHED_STOPPED",
        output_token_ids=[2, 3, 63],
        expected_output_ids=[2, 3],
    )


def test_length_capped_completion_appends_text_eos_and_pad() -> None:
    _assert_conditioning(
        status_name="FINISHED_LENGTH_CAPPED",
        output_token_ids=[2, 3, 4],
        expected_output_ids=[2, 3],
    )


@pytest.mark.parametrize("stage", ["thinker", "talker", None, "code2wav"])
def test_ar_forward_supports_native_capture_and_replay(stage: str | None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep native persistent output allocation/copy and replay view handling real.

    GPU capture is replaced with CPU calls, and fake replay updates the native
    buffer without re-entering Python forward, as a captured graph would.
    """
    resolved_stage = stage or "thinker"
    hf_config = Qwen2_5OmniConfig()
    hf_config.thinker_config.text_config.vocab_size = 64
    hf_config.thinker_config.text_config.hidden_size = 2
    model_config = object.__new__(ModelConfig)
    model_config.hf_config = hf_config
    model_config.multimodal_config = MultiModalConfig()
    if stage is not None:
        model_config.model_stage = stage
    config = object.__new__(VllmConfig)
    config.model_config = model_config

    class HiddenModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self.make_empty_intermediate_tensors = lambda: None

        def set_suppress_start_id(self, start_id):
            self.suppress_start_id = start_id

        def forward(self, input_ids, **kwargs):
            self.calls += 1
            hidden = input_ids.float().reshape(-1, 1).expand(-1, 2).clone()
            # The thinker may also return its input embeddings.
            return (torch.zeros_like(hidden), hidden) if resolved_stage == "thinker" else hidden

        def compute_logits(self, hidden):
            return hidden + 1

    child = HiddenModel()
    # Exercise the real stage declaration; replace only weight construction.
    monkeypatch.setattr(qwen2_5_omni, "init_vllm_registered_model", lambda **kwargs: child)
    monkeypatch.setattr(Qwen2_5OmniForConditionalGeneration, "_init_special_tokens_embeddings", lambda self: None)
    model = Qwen2_5OmniForConditionalGeneration(vllm_config=config)
    assert model.model_stage == resolved_stage
    assert model.have_multimodal_outputs is True
    runner = object.__new__(OmniGPUModelRunner)
    runner._staged_writes = staged_writes.StagedWrites()
    runner.model = model
    runner._configure_cudagraph_output_contract()
    if stage == "code2wav":
        output = model(input_ids=torch.empty(0, dtype=torch.long), positions=torch.empty(3, 0, dtype=torch.long))
        assert isinstance(output, OmniOutput)
        assert output.text_hidden_states is None
        assert output.multimodal_outputs["model_outputs"].numel() == 0
        assert runner._model_has_structured_output is True
        assert runner._exclude_full_graph is True
        return
    assert runner._model_has_structured_output is False
    assert runner._exclude_full_graph is False
    assert runner._full_graph_aux_outputs is False
    small, large = [cudagraph_utils.BatchExecutionDescriptor(CUDAGraphMode.FULL, n, n) for n in (2, 4)]
    manager = object.__new__(cudagraph_utils.ModelCudaGraphManager)
    manager.__dict__.update(
        use_breakable_cg=False,
        cudagraph_mode=CUDAGraphMode.FULL,
        _capture_descs={CUDAGraphMode.FULL: [small, large]},
        max_num_reqs=4,
        is_first_pp_rank=True,
        is_last_pp_rank=True,
        vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=1)),
        hidden_states=None,
        aux_hidden_states=[],
        graphs={},
    )

    def cpu_capture(self, create_forward_fn, progress_bar_desc):
        for desc in (small, large):
            for warmup in (True, False):
                create_forward_fn(desc, warmup)(CUDAGraphMode.NONE)
            self.graphs[desc] = SimpleNamespace(replay=lambda: self.hidden_states.add_(10))

    monkeypatch.setattr(cudagraph_utils.CudaGraphManager, "capture", cpu_capture)
    monkeypatch.setattr(cudagraph_utils, "prepare_inputs_to_capture", lambda *args, **kwargs: (None, {}))
    monkeypatch.setattr(cudagraph_utils, "set_forward_context", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(cudagraph_utils, "get_offloader", lambda: SimpleNamespace(sync_prev_onload=lambda: None))

    def capture_model(_self, *, profile_only=False):
        manager.capture(
            model,
            SimpleNamespace(prepare_dummy_inputs=lambda nr, nt: {"positions": torch.arange(nt).expand(3, -1)}),
            SimpleNamespace(
                input_ids=torch.arange(4), positions=torch.arange(4), is_padding=torch.zeros(4, dtype=torch.bool)
            ),
            None,
            None,
            [],
            None,
        )
        return 0

    runner.cudagraph_manager = manager
    runner.use_aux_hidden_state_outputs = False
    monkeypatch.setattr(GPUModelRunner, "capture_model", capture_model)
    assert runner.capture_model() == 0
    assert manager._capture_descs == {CUDAGraphMode.FULL: [small, large]}
    # A smaller first capture must still allocate for the largest descriptor.
    assert manager.hidden_states.shape == (4, 2)
    torch.testing.assert_close(manager.hidden_states, torch.arange(4).float().reshape(-1, 1).expand(-1, 2))
    assert child.calls == 4
    replayed = manager.run_fullgraph(small)
    assert child.calls == 4
    torch.testing.assert_close(replayed, torch.tensor([[10.0, 10.0], [11.0, 11.0]]))
    assert replayed.data_ptr() == manager.hidden_states.data_ptr()
    state = object.__new__(OmniModelState)
    state.model = model
    hidden, multimodal = state.postprocess_model_output(replayed, None, None)
    assert hidden is replayed
    assert multimodal == {}
    torch.testing.assert_close(model.compute_logits(hidden), hidden + 1)
    manager.run_fullgraph(large)
    # Native output views are borrowed: the next replay overwrites them.
    torch.testing.assert_close(replayed, torch.tensor([[20.0, 20.0], [21.0, 21.0]]))
