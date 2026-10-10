# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.logits_processor.state import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import random_sample
from vllm.v1.worker.gpu_input_batch import InputBatch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_omni.config.model import OmniModelConfig
from vllm_omni.data_entry_keys import to_struct
from vllm_omni.model_executor.models.cosyvoice3 import cosyvoice3, cosyvoice3_talker
from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 import CosyVoice3Model, _normalize_request_conditioning
from vllm_omni.transformers_utils.configs.cosyvoice3 import CosyVoice3Config
from vllm_omni.worker import gpu_ar_model_runner
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _DummyCode2Wav:
    def __init__(
        self,
        vocab_size: int,
        num_samples: int = 32,
        outputs: list[tuple[torch.Tensor, dict[str, object] | None]] | None = None,
    ):
        self.input_embedding = SimpleNamespace(num_embeddings=vocab_size)
        self.num_samples = num_samples
        self.outputs = list(outputs or [])
        self.forward_calls: list[dict[str, object]] = []
        self.forward_streaming_calls: list[dict[str, object]] = []
        self.forward_streaming_batch_calls: list[list[dict[str, object]]] = []

    def forward(self, **kwargs):
        self.forward_calls.append(kwargs)
        token = kwargs["token"]
        num_samples = int(token.shape[-1])
        return torch.linspace(-1.0, 1.0, max(num_samples, 1), dtype=torch.float32).reshape(1, 1, -1)

    def forward_streaming(self, **kwargs):
        self.forward_streaming_calls.append(kwargs)
        if self.outputs:
            return self.outputs.pop(0)

        token = kwargs["token"]
        num_samples = int(token.shape[-1])
        audio = torch.linspace(-1.0, 1.0, max(num_samples, 1), dtype=torch.float32).reshape(1, 1, -1)
        new_state = None
        if not kwargs.get("finalize", False):
            new_state = {
                "mel": torch.ones((1, 80, max(num_samples, 1)), dtype=torch.float32),
                "speech_offset": audio.shape[-1],
            }
        return audio, new_state

    def forward_streaming_batch(self, items, *, n_timesteps: int = 10):
        self.forward_streaming_batch_calls.append(items)
        return [
            self.forward_streaming(
                token=item["token"],
                prompt_token=item["prompt_token"],
                prompt_feat=item["prompt_feat"],
                embedding=item["embedding"],
                cache_state=item.get("cache_state"),
                n_timesteps=n_timesteps,
                token_offset_tokens=int(item.get("token_offset_tokens", 0)),
                finalize=bool(item.get("finalize", False)),
            )
            for item in items
        ]


def _make_code2wav_model(
    *,
    with_stride_cfg: bool = False,
    num_samples: int = 32,
    outputs: list[tuple[torch.Tensor, dict[str, object] | None]] | None = None,
) -> CosyVoice3Model:
    model = object.__new__(CosyVoice3Model)
    nn.Module.__init__(model)
    model.model_stage = "cosyvoice3_code2wav"
    hift_cfg = {} if not with_stride_cfg else {"upsample_rates": [8, 5, 3], "istft_params": {"hop_len": 4}}
    model.config = SimpleNamespace(
        sample_rate=24000,
        hift=hift_cfg,
        token_frame_rate=25 if with_stride_cfg else 0,
        token_mel_ratio=2 if with_stride_cfg else 0,
    )
    model.code2wav = _DummyCode2Wav(vocab_size=4, num_samples=num_samples, outputs=outputs)
    # Short-circuit the lazy TensorRT estimator swap: these tests exercise the
    # forward audio logic, not the TRT path. On a GPU CI runner the swap would
    # otherwise run and dereference ``self.model_dir`` (only set in __init__,
    # which this fixture bypasses via object.__new__).
    model._code2wav_trt_done = True
    model.source_cache_len = 4
    model.speech_window = torch.hamming_window(8, periodic=False)
    model._stream_audio_cache_by_req = {}
    model._stream_audio_cache_lock = Lock()
    model._stream_vocoder_cache_by_req = {}
    return model


def _make_talker_model() -> CosyVoice3Model:
    model = object.__new__(CosyVoice3Model)
    nn.Module.__init__(model)
    model.model_stage = "cosyvoice3_talker"
    model.config = SimpleNamespace(
        llm={
            "speech_token_size": 6561,
            "eos_token_id": 6562,
            "sampling": {
                "top_p": 0.8,
                "top_k": 25,
                "win_size": 10,
                "tau_r": 0.1,
            },
        },
        vocab_size=151923,
    )
    return model


def _make_sampling_metadata(
    *,
    output_token_ids: list[list[int]],
    repetition_penalty: float = 2.0,
) -> SamplingMetadata:
    batch_size = len(output_token_ids)
    return SamplingMetadata(
        temperature=torch.ones(batch_size, dtype=torch.float32),
        all_greedy=False,
        all_random=True,
        top_p=torch.full((batch_size,), 0.8, dtype=torch.float32),
        top_k=torch.full((batch_size,), 25, dtype=torch.int32),
        generators={},
        max_num_logprobs=None,
        no_penalties=False,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(batch_size, dtype=torch.float32),
        presence_penalties=torch.zeros(batch_size, dtype=torch.float32),
        repetition_penalties=torch.full((batch_size,), repetition_penalty, dtype=torch.float32),
        output_token_ids=output_token_ids,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


@pytest.fixture
def output_payload_vllm_config(tmp_path, request):
    hf_config = CosyVoice3Config()
    hf_config.llm.update(llm_input_size=16, llm_output_size=16, speech_token_size=32)
    # Exercise the real config fields without model downloads or engine setup.
    model_config = object.__new__(OmniModelConfig)
    model_config.hf_config = hf_config
    model_config.model_stage = "cosyvoice3_talker"
    model_config.model = str(tmp_path)
    model_config.async_chunk = getattr(request, "param", True)
    model_config.enable_return_routed_experts = False
    model_config.engine_output_type = "latent"
    model_config.stage_connector_config = {"name": "SharedMemoryConnector", "extra": {"role": "sender"}}
    vllm_config = object.__new__(VllmConfig)
    vllm_config.model_config = model_config
    return vllm_config


@pytest.fixture
def output_payload_talker(monkeypatch, output_payload_vllm_config):
    class DummyEncoder(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

        def forward(self, inputs_embeds, positions):
            return inputs_embeds

    class DummyTalker(nn.Module):
        def __init__(self, *, llm: nn.Module, **kwargs):
            super().__init__()
            self.llm = llm

    monkeypatch.setattr(cosyvoice3_talker, "VLLMQwen2Encoder", DummyEncoder)
    monkeypatch.setattr(cosyvoice3_talker, "CosyVoice3LM", DummyTalker)
    monkeypatch.setattr(CosyVoice3Model, "_create_llm_vllm_config", lambda self, config: config)

    return CosyVoice3Model(vllm_config=output_payload_vllm_config)


@pytest.mark.parametrize(
    "output_payload_vllm_config", [True, False], indirect=True, ids=["async_chunks", "sync_chunks"]
)
@pytest.mark.parametrize("async_scheduling", [True, False], ids=["async_scheduling", "sync_scheduling"])
def test_talker_skips_unused_hidden_payload_without_async_materialization(
    output_payload_talker, output_payload_vllm_config, async_scheduling
):
    runner = object.__new__(GPUARModelRunner)
    runner.use_async_scheduling = async_scheduling
    runner.omni_prefix_cache = None
    runner.speculative_config = None
    runner.model_config = output_payload_vllm_config.model_config
    runner.model = output_payload_talker
    # load_model caches this policy before the output path runs.
    runner._pooler_payload_include_hidden_flag = bool(getattr(runner.model, "omni_pooler_payload_include_hidden", True))

    assert runner._model_omni_pooler_payload_include_hidden() is (not runner.model_config.async_chunk)
    assert runner._should_use_async_omni_output() is False


@pytest.mark.parametrize("is_prefill", [True, False], ids=["prefill", "decode"])
@pytest.mark.parametrize(
    "output_payload_vllm_config", [True, False], indirect=True, ids=["async_chunks", "sync_chunks"]
)
def test_talker_inline_output_preserves_conditioning_and_tokens(
    output_payload_talker, output_payload_vllm_config, monkeypatch, mocker, is_prefill
):
    """Token-only output retains prefill conditioning and sampled codec IDs."""

    runner = object.__new__(GPUARModelRunner)
    runner.model = output_payload_talker
    runner._pooler_payload_include_hidden_flag = bool(getattr(runner.model, "omni_pooler_payload_include_hidden", True))
    runner.vllm_config = output_payload_vllm_config
    runner.model_config = output_payload_vllm_config.model_config
    runner._async_chunk = runner.model_config.async_chunk
    runner.omni_prefix_cache = None
    runner.supports_mm_inputs = False
    runner.routed_experts_initialized = False
    runner.model_intermediate_buffer = {}
    runner.input_batch = object.__new__(InputBatch)
    runner.input_batch._req_ids = ["r1", "r2"]
    runner.input_batch.req_id_to_index = {"r1": 0, "r2": 1}
    monkeypatch.setattr(GPUARModelRunner, "_resolve_pooler_payload_req_ids", lambda self, req_ids: ("latent", req_ids))
    monkeypatch.setattr(GPUARModelRunner, "get_omni_connector_output", lambda self: None)

    conditioning = {
        "speech_token": torch.tensor([[11, 12], [21, 0]], dtype=torch.long),
        "speech_token_len": torch.tensor([2, 1], dtype=torch.long),
        "speech_feat": torch.arange(16, dtype=torch.float32).reshape(2, 4, 2),
        "embedding": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
    }
    expected = {key: value.clone() for key, value in conditioning.items()}
    hidden_states = output_payload_talker.forward(
        input_ids=torch.tensor([1, 2, 3]),
        positions=torch.arange(3),
        inputs_embeds=torch.ones(3, 16),
        **(conditioning if is_prefill else {}),
    )
    model_output = output_payload_talker.make_omni_output(hidden_states, **(conditioning if is_prefill else {}))

    cpu_materialization = mocker.spy(gpu_ar_model_runner, "_to_cpu_contiguous")
    scheduler_output = object.__new__(SchedulerOutput)
    scheduler_output.total_num_scheduled_tokens = 3
    scheduler_output.num_scheduled_tokens = {"r1": 2, "r2": 1}
    output = runner._build_omni_model_runner_output_from_snapshot(
        scheduler_output=scheduler_output,
        hidden_states=model_output.text_hidden_states,
        staged_hidden_states_cpu=None,
        multimodal_outputs=model_output.multimodal_outputs,
        req_ids_output_copy=["r1", "r2"],
        req_id_to_index_output_copy={"r1": 0, "r2": 1},
        valid_sampled_token_ids=[[101], [102]],
        logprobs_lists=None,
        prompt_logprobs_dict={},
        num_nans_in_logits=None,
        kv_connector_output=None,
        ec_connector_output=None,
        cudagraph_stats=None,
        kv_extracted_req_ids=None,
        num_scheduled_tokens_np=torch.tensor([2, 1], dtype=torch.int32).numpy(),
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.long),
    )
    assert output.req_ids == ["r1", "r2"]
    assert output.sampled_token_ids == [[101], [102]]
    hidden_storage = model_output.text_hidden_states.untyped_storage().data_ptr()
    hidden_copied = any(
        call.args[0].untyped_storage().data_ptr() == hidden_storage for call in cpu_materialization.call_args_list
    )
    assert hidden_copied is (not runner._async_chunk)
    if runner._async_chunk:
        assert output.multimodal_outputs is None
    if not is_prefill and runner._async_chunk:
        assert not output.inter_stage_outputs or all(not item for item in output.inter_stage_outputs)
        return

    assert len(output.inter_stage_outputs) == 2
    for idx, prompt_len in enumerate([2, 1]):
        item = output.inter_stage_outputs[idx]
        assert ("hidden" not in item) is runner._async_chunk
        if not is_prefill:
            continue
        # The downstream processor removes padding using speech_token_len.
        torch.testing.assert_close(item["embed.speech_token"], expected["speech_token"][idx : idx + 1])
        torch.testing.assert_close(item["embed.speech_feat"], expected["speech_feat"][idx : idx + 1])
        assert item["embed.speech_token_len"].item() == prompt_len
        torch.testing.assert_close(item["embed.embedding"], expected["embedding"][idx : idx + 1])


def test_talker_preserves_packed_payload_policy(output_payload_talker, output_payload_vllm_config, monkeypatch):
    output_payload_vllm_config.model_config.async_chunk = False
    monkeypatch.setattr(cosyvoice3, "cosyvoice3_packed_inference_enabled", lambda: True)
    model = CosyVoice3Model(vllm_config=output_payload_vllm_config)

    assert model.omni_pooler_payload_include_hidden is False
    assert not getattr(model, "use_async_omni_output", False)


def test_forward_prefers_token_offset_when_present():
    model = _make_code2wav_model()

    runtime_info = [
        {
            "embed": {
                "speech_token": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "speech_feat": torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32),
                "embedding": torch.tensor([[0.5, 0.6]], dtype=torch.float32),
            },
            "meta": {"left_context_size": 2},
        }
    ]

    out = model.forward(
        input_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        positions=torch.tensor([0, 1, 2], dtype=torch.long),
        model_intermediate_buffer=runtime_info,
        seq_token_counts=[3],
    )

    assert len(out.multimodal_outputs["audio"]) == 1
    assert out.multimodal_outputs["audio"][0].numel() > 0
    assert len(model.code2wav.forward_streaming_calls) == 1
    call = model.code2wav.forward_streaming_calls[0]
    assert call["token"].shape == (1, 3)
    assert call["token_offset_tokens"] == 2
    assert call["finalize"] is False


def test_forward_falls_back_to_left_context_size_for_backward_compat():
    model = _make_code2wav_model()

    runtime_info = [
        {
            "embed": {
                "speech_token": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "speech_feat": torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32),
                "embedding": torch.tensor([[0.5, 0.6]], dtype=torch.float32),
            },
            "meta": {"left_context_size": 2},
        }
    ]

    model.forward(
        input_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        positions=torch.tensor([0, 1, 2], dtype=torch.long),
        model_intermediate_buffer=runtime_info,
        seq_token_counts=[3],
    )

    assert model.code2wav.forward_streaming_calls[0]["token_offset_tokens"] == 2


def test_forward_ignores_single_request_padded_tail_tokens():
    model = _make_code2wav_model(with_stride_cfg=True)
    runtime_info = [
        {
            "embed": {
                "speech_token": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "speech_feat": torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32),
                "embedding": torch.tensor([[0.5, 0.6]], dtype=torch.float32),
            },
            "meta": {"left_context_size": 0},
        }
    ]

    out = model.forward(
        input_ids=torch.tensor([0, 1, 2, 3, 3], dtype=torch.long),
        positions=torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        model_intermediate_buffer=runtime_info,
        seq_token_counts=[3],
    )

    # The padded tail must not contribute to code2wav length.
    assert out.multimodal_outputs["audio"][0].numel() == 3
    assert model.code2wav.forward_streaming_calls[0]["token"].tolist() == [[0, 1, 2]]


def test_forward_uses_non_stream_decode_without_chunk_metadata():
    model = _make_code2wav_model()

    runtime_info = [
        {
            "embed": {
                "speech_token": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "speech_feat": torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32),
                "embedding": torch.tensor([[0.5, 0.6]], dtype=torch.float32),
            },
            "ids": {"prompt": [101, 102]},
            "generated_len": 3,
        }
    ]

    out = model.forward(
        input_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        positions=torch.tensor([0, 1, 2], dtype=torch.long),
        model_intermediate_buffer=runtime_info,
        seq_token_counts=[3],
    )

    assert out.multimodal_outputs["audio"][0].numel() == 3
    assert len(model.code2wav.forward_calls) == 1
    assert len(model.code2wav.forward_streaming_calls) == 0
    call = model.code2wav.forward_calls[0]
    assert call["token"].tolist() == [[0, 1, 2]]
    assert call["token_offset_tokens"] == 0


def test_forward_uses_non_stream_talker_prefill_offset():
    model = _make_code2wav_model()

    runtime_info = [
        {
            "embed": {
                "speech_token": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "speech_feat": torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32),
                "embedding": torch.tensor([[0.5, 0.6]], dtype=torch.float32),
            },
            "meta": {"talker_prefill_offset": 3},
        }
    ]

    model.forward(
        input_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        positions=torch.tensor([0, 1, 2], dtype=torch.long),
        model_intermediate_buffer=runtime_info,
        seq_token_counts=[3],
    )

    assert model.code2wav.forward_calls[0]["token_offset_tokens"] == 3


def test_forward_reuses_streaming_cache_state_between_chunks():
    model = _make_code2wav_model(
        outputs=[
            (
                torch.arange(4, dtype=torch.float32).reshape(1, 1, -1),
                {"mel": torch.ones((1, 80, 3), dtype=torch.float32), "speech_offset": 4},
            ),
            (
                torch.full((1, 1, 2), 9.0, dtype=torch.float32),
                {"mel": torch.ones((1, 80, 5), dtype=torch.float32), "speech_offset": 6},
            ),
        ]
    )
    runtime_info = [
        {
            "embed": {
                "speech_token": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "speech_feat": torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32),
                "embedding": torch.tensor([[0.5, 0.6]], dtype=torch.float32),
            },
            "meta": {
                "req_id": ["rid-stream"],
                "stream_finished": torch.tensor(False),
                "left_context_size": 0,
            },
        }
    ]

    out1 = model.forward(
        input_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        positions=torch.tensor([0, 1, 2], dtype=torch.long),
        model_intermediate_buffer=runtime_info,
        seq_token_counts=[3],
    )
    assert out1.multimodal_outputs["audio"][0].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert model.code2wav.forward_streaming_calls[0]["cache_state"] is None

    out2 = model.forward(
        input_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        positions=torch.tensor([0, 1, 2], dtype=torch.long),
        model_intermediate_buffer=runtime_info,
        seq_token_counts=[3],
    )
    assert out2.multimodal_outputs["audio"][0].tolist() == [9.0, 9.0]
    cache_state = model.code2wav.forward_streaming_calls[1]["cache_state"]
    assert cache_state is not None
    assert cache_state["speech_offset"] == 4
    assert "rid-stream" in model._stream_vocoder_cache_by_req


def test_forward_clears_streaming_cache_on_terminal_chunk():
    model = _make_code2wav_model(
        outputs=[
            (
                torch.arange(4, dtype=torch.float32).reshape(1, 1, -1),
                {"mel": torch.ones((1, 80, 3), dtype=torch.float32), "speech_offset": 4},
            ),
            (
                torch.full((1, 1, 1), 7.0, dtype=torch.float32),
                None,
            ),
        ]
    )
    runtime_info = [
        {
            "embed": {
                "speech_token": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "speech_feat": torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32),
                "embedding": torch.tensor([[0.5, 0.6]], dtype=torch.float32),
            },
            "meta": {
                "req_id": ["rid-stream"],
                "stream_finished": torch.tensor(False),
                "left_context_size": 0,
            },
        }
    ]

    model.forward(
        input_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        positions=torch.tensor([0, 1, 2], dtype=torch.long),
        model_intermediate_buffer=runtime_info,
        seq_token_counts=[3],
    )
    assert "rid-stream" in model._stream_vocoder_cache_by_req

    runtime_info[0]["meta"]["stream_finished"] = torch.tensor(True)
    out = model.forward(
        input_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        positions=torch.tensor([0, 1, 2], dtype=torch.long),
        model_intermediate_buffer=runtime_info,
        seq_token_counts=[3],
    )
    assert out.multimodal_outputs["audio"][0].tolist() == [7.0]
    assert "rid-stream" not in model._stream_vocoder_cache_by_req


def test_forward_batches_streaming_flow_items(monkeypatch):
    monkeypatch.setenv("COSYVOICE3_BATCH_FLOW", "1")
    model = _make_code2wav_model()
    runtime_info = [
        {
            "embed": {
                "speech_token": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "speech_feat": torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32),
                "embedding": torch.tensor([[0.5, 0.6]], dtype=torch.float32),
            },
            "meta": {
                "req_id": ["rid-a"],
                "stream_finished": torch.tensor(False),
                "left_context_size": 0,
            },
        },
        {
            "embed": {
                "speech_token": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "speech_feat": torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32),
                "embedding": torch.tensor([[0.7, 0.8]], dtype=torch.float32),
            },
            "meta": {
                "req_id": ["rid-b"],
                "stream_finished": torch.tensor(False),
                "left_context_size": 1,
            },
        },
    ]

    out = model.forward(
        input_ids=torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long),
        positions=torch.arange(6, dtype=torch.long),
        model_intermediate_buffer=runtime_info,
        seq_token_counts=[3, 3],
    )

    assert len(out.multimodal_outputs["audio"]) == 2
    assert len(model.code2wav.forward_streaming_batch_calls) == 1
    batch_items = model.code2wav.forward_streaming_batch_calls[0]
    assert [item["index"] for item in batch_items] == [0, 1]
    assert torch.equal(batch_items[0]["token"], torch.tensor([[0, 1, 2]]))
    assert batch_items[1]["token_offset_tokens"] == 1
    assert "rid-a" in model._stream_vocoder_cache_by_req
    assert "rid-b" in model._stream_vocoder_cache_by_req


def test_sample_uses_ras_rejection_for_recent_repetition():
    model = _make_talker_model()
    metadata = _make_sampling_metadata(output_token_ids=[[1] * 10])
    logits = torch.tensor([[-1e9, 10.0, 0.0]], dtype=torch.float32)

    out = model.sample(logits, metadata)

    assert out is not None
    assert out.sampled_token_ids.tolist() == [[2]]


def test_sample_tolerates_padded_rows_without_history():
    model = _make_talker_model()
    metadata = _make_sampling_metadata(output_token_ids=[[1] * 10])
    logits = torch.tensor(
        [
            [-1e9, 10.0, 0.0],
            [-1e9, 0.0, 10.0],
        ],
        dtype=torch.float32,
    )

    out = model.sample(logits, metadata)

    assert out is not None
    assert out.sampled_token_ids.shape == (2, 1)


def test_sample_excludes_non_finite_logits():
    model = _make_talker_model()
    metadata = _make_sampling_metadata(output_token_ids=[[]])
    metadata.temperature.fill_(0.5)
    logits = torch.tensor([[float("nan"), 1.0, float("inf"), float("-inf")]], dtype=torch.bfloat16)

    out = model.sample(logits, metadata)

    assert out is not None
    assert out.sampled_token_ids.tolist() == [[1]]


def test_sample_preserves_allowed_token_mask_with_invalid_logits():
    model = _make_talker_model()
    metadata = _make_sampling_metadata(output_token_ids=[[]])
    metadata.allowed_token_ids_mask = torch.tensor([[True, False, True]])
    logits = torch.tensor([[float("nan"), 1.0, float("inf")]], dtype=torch.float32)

    out = model.sample(logits, metadata)

    assert out is not None
    assert out.sampled_token_ids.tolist() == [[1]]


def test_sample_rejects_rows_without_finite_logits():
    model = _make_talker_model()
    metadata = _make_sampling_metadata(output_token_ids=[[]])
    metadata.allowed_token_ids_mask = torch.tensor([[True, False, True]])
    logits = torch.tensor([[0.0, float("nan"), 0.0]], dtype=torch.float32)

    with pytest.raises(ValueError, match="no finite logits"):
        model.sample(logits, metadata)


def test_sample_keeps_only_finite_token_after_ras_rejection():
    model = _make_talker_model()
    metadata = _make_sampling_metadata(output_token_ids=[[1] * 10])
    metadata.allowed_token_ids_mask = torch.tensor([[True, False, True]])
    logits = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)

    out = model.sample(logits, metadata)

    assert out is not None
    assert out.sampled_token_ids.tolist() == [[1]]


def _reference_param(param, row, default):
    if param is None or param.numel() == 0:
        return default
    value = param.reshape(-1)[min(row, param.numel() - 1)].item()
    return int(value) if isinstance(default, int) else float(value)


def _reference_ras_one(weighted_scores, history, *, top_p, top_k, win_size, tau_r, generator):
    """Serial RAS oracle from the pre-B1 implementation, including RNG draws."""
    probs, ids = weighted_scores.softmax(dim=0).sort(descending=True, stable=True)
    keep = probs.cumsum(dim=0) - probs < top_p
    if top_k > 0:
        keep &= torch.arange(probs.numel(), device=probs.device) < top_k
    generators = {} if generator is None else {0: generator}
    draw = random_sample((probs * keep).unsqueeze(0), generators).reshape(())
    token = int(ids[draw].item())
    if win_size > 0 and history:
        recent = torch.tensor(history[-win_size:], device=probs.device)
        if int((recent == token).sum().item()) >= win_size * tau_r:
            scores = weighted_scores.clone()
            original = scores[token].clone()
            scores[token] = float("-inf")
            scores[token] = torch.where(torch.isfinite(scores).any(), scores[token], original)
            token = int(random_sample(scores.softmax(dim=0).unsqueeze(0), generators).item())
    return token


def test_batched_ras_matches_serial_seeded_requests_and_rng_state():
    model = _make_talker_model()
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0], [0.0, 3.0, 1.0, -2.0], [1.0, 0.0, 2.0, -1.0]])
    metadata = _make_sampling_metadata(output_token_ids=[[0] * 10, [], [2] * 10])
    metadata.temperature = torch.tensor([0.5, 1.0, 1.5])
    metadata.top_k = torch.tensor([1, 2, 4])
    metadata.top_p = torch.tensor([0.7, 0.8, 0.9])
    metadata.generators = {i: torch.Generator().manual_seed(400 + i) for i in range(3)}
    reference_generators = {i: torch.Generator().manual_seed(400 + i) for i in range(3)}
    for _ in range(6):
        expected = []
        for i in range(3):
            scores = torch.log_softmax(logits[i] / metadata.temperature[i], dim=0)
            expected.append(
                _reference_ras_one(
                    scores,
                    metadata.output_token_ids[i],
                    top_p=float(metadata.top_p[i]),
                    top_k=int(metadata.top_k[i]),
                    win_size=10,
                    tau_r=0.1,
                    generator=reference_generators[i],
                )
            )
        actual = model._ras_sample_batch(logits, metadata, default_top_p=0.8, default_top_k=25, win_size=10, tau_r=0.1)
        assert actual.tolist() == expected
        for i, token in enumerate(expected):
            assert torch.equal(metadata.generators[i].get_state(), reference_generators[i].get_state())
            metadata.output_token_ids[i].append(token)


@pytest.mark.parametrize("parameters", ["mixed", "defaults", "all_random"])
@pytest.mark.parametrize("seeded_rows", [(0, 1, 2, 3), (0, 2)])
@pytest.mark.parametrize("use_host_params", [False, True])
def test_mixed_ras_preserves_seeded_trajectories_and_greedy_rng(parameters, seeded_rows, use_host_params, device="cpu"):
    model = _make_talker_model()
    logits = torch.tensor([[4.0, 1.0, 0.0, -1.0], [0.0, 2.0, 1.0, -2.0]] * 2, device=device)
    metadata = _make_sampling_metadata(output_token_ids=[[0] * 10, [1], [], [1] * 10])
    metadata.all_random = parameters == "all_random"
    # Include a greedy row between random rows to exercise both compactions.
    temperatures = [0.6, 0.7, 1.2, 1.1] if metadata.all_random else [0.0, 0.7, 1.2, 0.0]
    metadata.temperature = torch.tensor(temperatures)
    host_params = [SamplingParams(temperature=t) for t in temperatures]
    metadata.top_p = torch.tensor([0.8, 0.6, 0.9, 1.0])
    metadata.top_k = torch.tensor([1, 1, -1, 4], dtype=torch.int32)
    if parameters == "defaults":
        metadata.top_p = metadata.top_k = None
    for name in ("temperature", "top_p", "top_k", "frequency_penalties", "presence_penalties"):
        value = getattr(metadata, name)
        if value is not None:
            setattr(metadata, name, value.to(device))
    metadata.generators = {i: torch.Generator(device=device).manual_seed(800 + i) for i in seeded_rows}
    reference_generators = {i: torch.Generator(device=device).manual_seed(800 + i) for i in seeded_rows}
    for step in range(40):
        if step == 20:
            # A previously greedy request must start from its untouched RNG.
            temperatures = [1.0, 0.9, 1.1, 0.5] if metadata.all_random else [1.0, 0.0, 0.0, 0.5]
            metadata.temperature = torch.tensor(temperatures, device=device)
            for params, temperature in zip(host_params, temperatures):
                params.temperature = temperature
        expected = {}
        for i in range(len(logits)):
            temperature = _reference_param(metadata.temperature, i, 1.0)
            if temperature < model._sampling_eps:
                expected[i] = int(logits[i].argmax())
            elif i in seeded_rows:
                expected[i] = _reference_ras_one(
                    torch.log_softmax(logits[i] / temperature, dim=0),
                    metadata.output_token_ids[i],
                    top_p=_reference_param(metadata.top_p, i, 0.8),
                    top_k=_reference_param(metadata.top_k, i, 25),
                    win_size=10,
                    tau_r=0.1,
                    generator=reference_generators[i],
                )
        out = model.sample(logits.clone(), metadata, per_req_sampling_params=host_params if use_host_params else None)
        assert out.sampled_token_ids.dtype == torch.int32
        assert out.sampled_token_ids.shape == (4, 1)
        tokens = out.sampled_token_ids[:, 0].tolist()
        for i, token in enumerate(tokens):
            if i in expected:
                assert token == expected[i]
            assert 0 <= token < logits.shape[1]
            metadata.output_token_ids[i].append(token)
        for i in seeded_rows:
            assert torch.equal(metadata.generators[i].get_state(), reference_generators[i].get_state())


def test_mixed_ras_request_reordering_preserves_generator_ownership():
    model = _make_talker_model()
    logits = torch.tensor([[1.0, 4.0, 0.0], [3.0, 0.0, 1.0], [0.0, 2.0, 5.0]])
    temperatures = torch.tensor([0.7, 0.0, 1.2])
    histories = [[1] * 10, [], [2]]
    actual_generators = [torch.Generator().manual_seed(90 + i) for i in range(3)]
    reference_generators = [torch.Generator().manual_seed(90 + i) for i in range(3)]
    for order in ([0, 1, 2], [2, 0, 1], [1, 2, 0], [2, 1, 0]):
        metadata = _make_sampling_metadata(output_token_ids=[histories[i] for i in order])
        metadata.all_random = False
        metadata.temperature = temperatures[order]
        metadata.generators = {row: actual_generators[req] for row, req in enumerate(order)}
        expected = []
        for req in order:
            if float(temperatures[req]) < model._sampling_eps:
                expected.append(int(logits[req].argmax()))
            else:
                expected.append(
                    _reference_ras_one(
                        torch.log_softmax(logits[req] / float(temperatures[req]), dim=0),
                        histories[req],
                        top_p=float(metadata.top_p[0]),
                        top_k=25,
                        win_size=10,
                        tau_r=0.1,
                        generator=reference_generators[req],
                    )
                )
        out = model.sample(logits[order].clone(), metadata)
        assert out.sampled_token_ids[:, 0].tolist() == expected
        for req, token in zip(order, expected):
            histories[req].append(token)
            assert torch.equal(actual_generators[req].get_state(), reference_generators[req].get_state())


@pytest.mark.parametrize("temperature", [0.0, 1e-6, 1e-5])
def test_mixed_ras_greedy_rows_never_draw_random_numbers(temperature, monkeypatch):
    model = _make_talker_model()
    metadata = _make_sampling_metadata(output_token_ids=[[1] * 10, [0] * 10])
    metadata.all_random = False
    metadata.temperature = torch.tensor([temperature])
    assert float(metadata.temperature[0]) < model._sampling_eps
    monkeypatch.setattr(cosyvoice3, "random_sample", lambda *a, **k: pytest.fail("greedy must not consume RNG"))
    out = model.sample(torch.tensor([[0.0, 2.0], [3.0, 0.0]]), metadata)
    assert out.sampled_token_ids.tolist() == [[1], [0]]


def test_mixed_ras_restores_single_valid_token_and_handles_empty_history():
    model = _make_talker_model()
    metadata = _make_sampling_metadata(output_token_ids=[[0] * 10, [1] * 10, []])
    metadata.all_random = False
    metadata.temperature = torch.tensor([0.0, 1.0, 1.0])
    metadata.allowed_token_ids_mask = torch.tensor([[False, True], [True, False], [True, False]])
    logits = torch.tensor([[3.0, 1.0], [0.0, 2.0], [float("nan"), 1.0]])
    out = model.sample(logits, metadata)
    assert out.sampled_token_ids.tolist() == [[0], [1], [1]]


def test_mixed_ras_applies_processors_before_greedy_and_random_selection():
    model = _make_talker_model()
    metadata = _make_sampling_metadata(output_token_ids=[[], []])
    metadata.all_random = False
    metadata.temperature = torch.tensor([0.0, 1.0])

    class OnlyLastToken:
        def apply(self, logits):
            logits[:, :-1] = float("-inf")
            return logits

    metadata.logitsprocs.non_argmax_invariant.append(OnlyLastToken())
    out = model.sample(torch.tensor([[10.0, 1.0], [10.0, 1.0]]), metadata)
    assert out.sampled_token_ids.tolist() == [[1], [1]]


def test_mixed_ras_rejects_invalid_random_row_before_consuming_rng():
    model = _make_talker_model()
    metadata = _make_sampling_metadata(output_token_ids=[[], []])
    metadata.all_random = False
    metadata.temperature = torch.tensor([0.0, 1.0])
    generator = torch.Generator().manual_seed(123)
    metadata.generators = {1: generator}
    before = generator.get_state().clone()
    with pytest.raises(ValueError, match="no finite logits"):
        model.sample(torch.tensor([[1.0, 0.0], [float("nan"), float("-inf")]]), metadata)
    assert torch.equal(generator.get_state(), before)


def test_gpu_ar_model_runner_prefers_model_sampler_when_opted_in():
    metadata = _make_sampling_metadata(output_token_ids=[[1, 2, 3]])
    expected = SamplerOutput(
        sampled_token_ids=torch.tensor([[7]], dtype=torch.int32),
        logprobs_tensors=None,
    )
    calls: list[torch.Tensor] = []

    class _DummyInputBatch:
        def __init__(self):
            self.sampling_metadata = metadata
            self.updated = False

        def update_async_output_token_ids(self):
            # After PR 3681 fix, update_async_output_token_ids is called
            # BEFORE model sampler path to ensure async placeholder repair
            # runs for all sampling paths
            self.updated = True

    runner = object.__new__(GPUARModelRunner)
    runner.input_batch = _DummyInputBatch()

    def model_sample(logits, sampling_metadata):
        calls.append(logits.clone())
        return expected

    runner.model = SimpleNamespace(
        prefer_model_sampler=True,
        sample=model_sample,
    )
    runner.sampler = lambda **_: (_ for _ in ()).throw(AssertionError("fallback sampler should not be used"))

    out = runner._sample(torch.tensor([[0.1, 0.2]], dtype=torch.float32), spec_decode_metadata=None)

    assert out is expected
    assert runner.input_batch.updated is True
    assert len(calls) == 1


def test_gpu_ar_model_runner_supplies_req_output_history_to_model_sampler():
    metadata = _make_sampling_metadata(output_token_ids=[])
    seen_histories: list[list[list[int]]] = []

    class _DummyInputBatch:
        def __init__(self):
            self.sampling_metadata = metadata
            self.req_output_token_ids = [[1, 2, 3]]
            self.req_ids = ["rid-1"]
            self.sampled_token_ids_cpu = None
            self.async_copy_ready_event = None
            self.prev_req_id_to_index = None
            self.update_async_called = False

        def update_async_output_token_ids(self):
            # After PR 3681 fix, update_async_output_token_ids is called
            # BEFORE model sampler path to ensure async placeholder repair
            # runs for all sampling paths
            self.update_async_called = True

    runner = object.__new__(GPUARModelRunner)
    runner.input_batch = _DummyInputBatch()

    def model_sample(logits, sampling_metadata):
        seen_histories.append([list(x) for x in sampling_metadata.output_token_ids])
        return SamplerOutput(sampled_token_ids=torch.tensor([[7]], dtype=torch.int32), logprobs_tensors=None)

    runner.model = SimpleNamespace(
        prefer_model_sampler=True,
        sample=model_sample,
    )
    runner.sampler = lambda **_: (_ for _ in ()).throw(AssertionError("fallback sampler should not be used"))

    runner._sample(torch.tensor([[0.1, 0.2]], dtype=torch.float32), spec_decode_metadata=None)

    assert runner.input_batch.update_async_called is True
    assert seen_histories == [[[1, 2, 3]]]


def test_gpu_ar_model_runner_repairs_async_placeholders_for_model_sampler():
    metadata = _make_sampling_metadata(output_token_ids=[])
    seen_histories: list[list[list[int]]] = []

    class _ReadyEvent:
        def __init__(self):
            self.synced = False

        def synchronize(self):
            self.synced = True

    class _DummyInputBatch:
        def __init__(self):
            self.sampling_metadata = metadata
            self.req_output_token_ids = [[11, -1]]
            self.req_ids = ["rid-1"]
            self.sampled_token_ids_cpu = torch.tensor([[29]], dtype=torch.int32)
            self.async_copy_ready_event = _ReadyEvent()
            self.prev_req_id_to_index = {"rid-1": 0}
            self.update_async_called = False

        def update_async_output_token_ids(self):
            # After PR 3681 fix, update_async_output_token_ids is called
            # BEFORE model sampler path to ensure async placeholder repair
            # runs for all sampling paths (model sampler + fallback sampler)
            self.update_async_called = True

    runner = object.__new__(GPUARModelRunner)
    runner.input_batch = _DummyInputBatch()

    def model_sample(logits, sampling_metadata):
        seen_histories.append([list(x) for x in sampling_metadata.output_token_ids])
        return SamplerOutput(sampled_token_ids=torch.tensor([[7]], dtype=torch.int32), logprobs_tensors=None)

    runner.model = SimpleNamespace(
        prefer_model_sampler=True,
        sample=model_sample,
    )
    runner.sampler = lambda **_: (_ for _ in ()).throw(AssertionError("fallback sampler should not be used"))

    runner._sample(torch.tensor([[0.1, 0.2]], dtype=torch.float32), spec_decode_metadata=None)

    assert runner.input_batch.async_copy_ready_event.synced is True
    assert runner.input_batch.update_async_called is True
    assert seen_histories == [[[11, 29]]]


@pytest.mark.parametrize(
    "order",
    [("decode", "short", "decode", "long"), ("short", "decode", "long", "decode"), ("long", "short", "decode")],
)
def test_embed_input_ids_preserves_interleaved_request_boundaries(order):
    model = _make_talker_model()
    speech = nn.Embedding.from_pretrained(torch.arange(100 * 4, dtype=torch.float32).reshape(100, 4))
    text = nn.Embedding.from_pretrained(torch.arange(100 * 4, dtype=torch.float32).reshape(100, 4) + 1000)
    model.model = SimpleNamespace(
        speech_embedding=speech, sos=0, task_id=1, llm=SimpleNamespace(model=SimpleNamespace(embed_tokens=text))
    )
    requests = {
        "decode": (torch.tensor([7]), torch.tensor([False]), None, speech.weight[7:8]),
        "short": (
            torch.tensor([1, 1, 1, 1, 21]),
            torch.tensor([True] * 4 + [False]),
            speech.weight[2:4],
            torch.cat((speech.weight[0:1], text.weight[21:22], speech.weight[1:2], speech.weight[2:4])),
        ),
        "long": (
            torch.tensor([1, 1, 1, 1, 1, 30, 31]),
            torch.tensor([True] * 5 + [False] * 2),
            speech.weight[4:7],
            torch.cat((speech.weight[0:1], text.weight[30:32], speech.weight[1:2], speech.weight[4:7])),
        ),
    }
    rows = [requests[name] for name in order]
    boundaries = [0]
    for ids, *_ in rows:
        boundaries.append(boundaries[-1] + ids.numel())
    actual = model.embed_input_ids(
        torch.cat([r[0] for r in rows]),
        multimodal_embeddings=[r[2] for r in rows if r[2] is not None],
        is_multimodal=torch.cat([r[1] for r in rows]),
        query_start_loc=boundaries,
    )
    torch.testing.assert_close(actual, torch.cat([r[3] for r in rows]), rtol=0, atol=0)
    conditioning = [torch.full((1, 192), i + 1.0) for i in range(sum(name != "decode" for name in order))]
    aligned = model._align_prompt_conditioning(conditioning)
    assert len(aligned) == len(order)
    index = 0
    for name, value in zip(order, aligned):
        if name == "decode":
            assert value is None
        else:
            assert value is conditioning[index]
            index += 1
    with pytest.raises(ValueError, match="conditioning must match"):
        model._align_prompt_conditioning(conditioning[:-1])


def test_mrv2_conditioning_follows_encoded_prompts_to_batch_rows():
    model = _make_talker_model()
    speech = nn.Embedding.from_pretrained(torch.arange(100 * 4, dtype=torch.float32).reshape(100, 4))
    text = nn.Embedding.from_pretrained(torch.arange(100 * 4, dtype=torch.float32).reshape(100, 4) + 1000)
    model.model = SimpleNamespace(
        speech_embedding=speech, sos=0, task_id=1, llm=SimpleNamespace(model=SimpleNamespace(embed_tokens=text))
    )
    model._mrv2_encoded_conditioning = OrderedDict()
    tokens = [torch.tensor([2, 3]), torch.tensor([4, 5, 6])]
    feats = [torch.full((4, 2), 1.0), torch.full((6, 2), 2.0)]
    speakers = [torch.full((1, 3), 1.0), torch.full((1, 3), 2.0)]
    encoded = model.embed_multimodal(speech_token=tokens, speech_feat=feats, embedding=speakers)
    # Batch: decode, short prompt, decode, long prompt, short again (same
    # reference: the encoder cache hands back the same tensor).
    ids = torch.tensor([7, 1, 1, 1, 1, 21, 8, 1, 1, 1, 1, 1, 30, 1, 1, 1, 1, 22])
    is_mm = torch.tensor([False] + [True] * 4 + [False, False] + [True] * 5 + [False] + [True] * 4 + [False])
    model.embed_input_ids(
        ids,
        multimodal_embeddings=[encoded[0], encoded[1], encoded[0]],
        is_multimodal=is_mm,
        query_start_loc=[0, 1, 6, 7, 13, 18],
    )
    output = model.make_omni_output(torch.zeros(18, 4), model_intermediate_buffer=[None] * 5)
    embed = output.multimodal_outputs["embed"]
    assert [value is None for value in embed["speech_token"]] == [True, False, True, False, False]
    for row, item in ((1, 0), (3, 1), (4, 0)):
        torch.testing.assert_close(embed["speech_token"][row], tokens[item][None])
        torch.testing.assert_close(embed["speech_feat"][row], feats[item][None])
        torch.testing.assert_close(embed["embedding"][row], speakers[item])
    # A following decode step carries no conditioning.
    model.embed_input_ids(
        torch.tensor([9, 10]), multimodal_embeddings=[], is_multimodal=torch.zeros(2, dtype=torch.bool)
    )
    assert model.make_omni_output(torch.zeros(2, 4)).multimodal_outputs == {}
    with pytest.raises(ValueError, match="lost its prompt conditioning"):
        model.embed_input_ids(
            torch.tensor([1, 1, 1, 1, 21]),
            multimodal_embeddings=[speech.weight[2:4].clone()],
            is_multimodal=torch.tensor([True] * 4 + [False]),
            query_start_loc=[0, 5],
        )


def test_talker_declares_its_narrow_logits_head():
    model = _make_talker_model()
    assert model.logits_vocab_size == 6561 + 200
    model.model_stage = "cosyvoice3_code2wav"
    assert model.logits_vocab_size is None


def test_embed_input_ids_rejects_cross_request_placeholder_block():
    model = _make_talker_model()
    speech = nn.Embedding(100, 4)
    model.model = SimpleNamespace(
        speech_embedding=speech,
        sos=0,
        task_id=1,
        llm=SimpleNamespace(model=SimpleNamespace(embed_tokens=nn.Embedding(100, 4))),
    )
    with pytest.raises(ValueError, match="complete prefill block"):
        model.embed_input_ids(
            torch.tensor([7, 1, 1, 1, 1, 20]),
            multimodal_embeddings=[speech.weight[2:4]],
            is_multimodal=torch.tensor([False, True, True, True, True, False]),
        )


def test_full_response_singleton_uses_the_optimized_batch_path(monkeypatch):
    monkeypatch.setenv("COSYVOICE3_BATCH_FLOW", "1")
    monkeypatch.setattr(cosyvoice3, "cosyvoice3_packed_inference_enabled", lambda: True)
    model = _make_code2wav_model()
    model.forward(
        input_ids=torch.tensor([0, 1, 2]),
        positions=torch.arange(3),
        seq_token_counts=[3],
        model_intermediate_buffer=[
            {
                "embed": {
                    "speech_token": torch.tensor([[1, 2, 3]]),
                    "speech_feat": torch.ones(1, 6, 2),
                    "embedding": torch.ones(1, 2),
                },
                "meta": {"req_id": ["single"], "stream_finished": torch.tensor(True), "left_context_size": 0},
            }
        ],
    )
    assert len(model.code2wav.forward_streaming_batch_calls) == 1
    assert model.code2wav.forward_streaming_batch_calls[0][0]["finalize"]


def test_full_response_adapter_uses_live_prefill_then_empty_decode(monkeypatch):
    model = _make_talker_model()
    hidden = torch.ones(2, 4)
    runner = object.__new__(OmniGPUModelRunner)
    runner.model = model
    runner._build_model_kwargs_extra = lambda: {}
    # Replay the same graph tensor while the live conditioning changes.
    monkeypatch.setattr(GPUModelRunner, "_model_forward", lambda *_args, **_kwargs: hidden)
    model._conditioning_request_rows = [1]
    model._conditioning_request_count = 2
    prompt = torch.tensor([[7, 8]])
    output = runner._model_forward(speech_token=prompt)
    values = output.multimodal_outputs["embed"]["speech_token"]
    assert values[0] is None
    torch.testing.assert_close(values[1], prompt)
    decode = runner._model_forward()
    assert decode.text_hidden_states is hidden
    assert decode.multimodal_outputs == {}


def test_packed_flow_preserves_torch_estimator_when_speaker_trt_is_enabled(monkeypatch):
    model = _make_code2wav_model()
    model._code2wav_trt_done = False
    monkeypatch.setattr(cosyvoice3.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cosyvoice3, "cosyvoice3_packed_inference_enabled", lambda: True)
    monkeypatch.setattr(cosyvoice3, "_cosyvoice3_trt_enabled", lambda: True)

    def unexpected_estimator_resolution():
        raise AssertionError("Packed Flow must not resolve or replace its torch estimator")

    monkeypatch.setattr(model, "_resolve_flow_estimator_onnx", unexpected_estimator_resolution)
    model._maybe_enable_code2wav_trt()
    assert model._code2wav_trt_done


def test_standard_sampling_preserves_parameters_and_excludes_prompt_penalties(monkeypatch):
    model = _make_talker_model()
    model.config.cosyvoice3_sampling_mode = "standard"
    metadata = _make_sampling_metadata(output_token_ids=[[2, 3]], repetition_penalty=1.21)
    metadata.prompt_token_ids = torch.tensor([[2, 4]], dtype=torch.long)
    metadata.temperature = torch.tensor([0.7])
    metadata.top_k = torch.tensor([20])
    captured = {}
    expected = SamplerOutput(sampled_token_ids=torch.tensor([[5]], dtype=torch.int32), logprobs_tensors=None)

    def standard_sampler(*, logits, sampling_metadata):
        captured["metadata"] = sampling_metadata
        return expected

    model._talker_sampler = standard_sampler
    monkeypatch.setattr(model, "_ras_sample_batch", lambda *a, **k: pytest.fail("RAS must not run in standard mode"))
    assert model.sample(torch.zeros(1, 8), metadata) is expected
    actual = captured["metadata"]
    assert actual is not metadata
    assert actual.output_token_ids == [[2, 3]]
    torch.testing.assert_close(actual.prompt_token_ids, torch.tensor([[8, 8]]))
    torch.testing.assert_close(metadata.prompt_token_ids, torch.tensor([[2, 4]]))
    assert actual.temperature is metadata.temperature
    assert actual.top_k is metadata.top_k
    assert actual.top_p is metadata.top_p
    assert actual.repetition_penalties is metadata.repetition_penalties


@pytest.mark.parametrize("mode", ["ras", "standard"])
def test_sampling_mode_preserves_or_merges_control_logits(mode):
    model = _make_talker_model()
    model.config.cosyvoice3_sampling_mode = mode
    model.config.vocab_size = 6770
    model.model = nn.Module()
    model.model.llm_decoder = lambda hidden: hidden.clone()
    hidden = torch.linspace(-2, 2, 6761).reshape(1, -1)
    logits = model.compute_logits(hidden)
    torch.testing.assert_close(logits[:, :6561], hidden[:, :6561])
    assert torch.isneginf(logits[:, 6761:]).all()
    if mode == "standard":
        torch.testing.assert_close(logits[:, 6561:6761], hidden[:, 6561:])
    else:
        torch.testing.assert_close(logits[:, 6562], torch.logsumexp(hidden[:, 6561:], dim=-1))
        assert torch.isneginf(logits[:, 6561]).all()
        assert torch.isneginf(logits[:, 6563:6761]).all()


def test_cancelled_stream_releases_only_its_vocoder_state():
    model = SimpleNamespace(
        _stream_audio_cache_lock=Lock(),
        _stream_vocoder_cache_by_req={"cancelled": {"mel": torch.ones(2)}, "live": {"mel": torch.zeros(2)}},
    )
    CosyVoice3Model.on_requests_finished(model, {"cancelled", "unknown"})
    CosyVoice3Model.on_requests_finished(model, {"cancelled"})
    assert set(model._stream_vocoder_cache_by_req) == {"live"}


@pytest.mark.parametrize("mrv2,packed", [(False, False), (False, True), (True, False)])
def test_talker_output_contract(monkeypatch, mrv2, packed):
    monkeypatch.setattr(cosyvoice3, "cosyvoice3_packed_inference_enabled", lambda: packed)
    model = _make_talker_model()
    hidden = torch.zeros(2, 4)
    model.model = SimpleNamespace(llm=lambda embeddings, positions: embeddings)
    sentinel = object()
    model.make_omni_output = lambda *args, **kwargs: sentinel
    if mrv2:
        model._sampling_eps = 1e-6
        model.mrv2_custom_sampler(SimpleNamespace(penalties_state=None))
    output = model.forward(torch.ones(2, dtype=torch.long), torch.arange(2), inputs_embeds=hidden)
    assert output is hidden


def test_request_conditioning_normalizes_singletons_without_changing_schema():
    token = torch.tensor([[7, 8]], dtype=torch.int32)
    feat = torch.randn(1, 4, 80)
    speaker = torch.randn(1, 192)
    length = torch.tensor([[2]], dtype=torch.int32)
    raw = {"embed": {"speech_token": token, "speech_feat": feat, "embedding": speaker, "speech_token_len": [length]}}
    result = to_struct(_normalize_request_conditioning(raw))
    assert result.embed.speech_token is token
    assert result.embed.speech_feat is feat
    assert result.embed.embedding is speaker
    assert result.embed.speech_token_len is length
    assert raw["embed"]["speech_token_len"] == [length]


def test_request_conditioning_rejects_unsplit_voices():
    with pytest.raises(ValueError, match="unsplit batch"):
        _normalize_request_conditioning({"embed": {"embedding": [torch.ones(1, 192), torch.zeros(1, 192)]}})


def test_request_conditioning_reaches_code2wav_with_true_prompt_length():
    model = _make_code2wav_model()
    output = model.forward(
        torch.tensor([1, 2]),
        torch.arange(2),
        model_intermediate_buffer=[
            {
                "embed": {
                    "speech_token": torch.tensor([[3, 2, 0, 0]], dtype=torch.int32),
                    "speech_feat": torch.randn(1, 8, 80),
                    "embedding": torch.randn(1, 192),
                    "speech_token_len": [torch.tensor([[2]], dtype=torch.int32)],
                },
                "meta": {"req_id": ["voice"]},
            }
        ],
    )
    call = model.code2wav.forward_calls[0]
    assert call["prompt_token"].shape == (1, 2)
    assert call["prompt_feat"].shape == (1, 4, 80)
    assert output.multimodal_outputs


@pytest.mark.parametrize("stage", ["cosyvoice3_talker", "cosyvoice3_code2wav"])
@pytest.mark.parametrize("owned", [False, True])
def test_generation_output_ownership_is_exposed_only_for_owned_codec(stage, owned):
    model = object.__new__(CosyVoice3Model)
    nn.Module.__init__(model)
    model.model_stage = stage
    if stage == "cosyvoice3_code2wav":
        model.code2wav = SimpleNamespace(owns_generation_output_storage=owned)
    assert model.owns_generation_output_storage is (stage == "cosyvoice3_code2wav" and owned)
