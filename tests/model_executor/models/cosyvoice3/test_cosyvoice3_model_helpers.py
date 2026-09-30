# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import functools
from threading import Lock
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch
import torch.nn as nn
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.logits_processor.state import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

if TYPE_CHECKING:
    from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 import CosyVoice3Model


@functools.lru_cache(maxsize=1)
def _cosyvoice3_model_and_runner():
    """Defer heavy Omni/vLLM imports until a test runs (avoids duplicate CustomOp init)."""
    from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 import CosyVoice3Model
    from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

    return CosyVoice3Model, GPUARModelRunner


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
    CosyVoice3Model, _ = _cosyvoice3_model_and_runner()
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
    CosyVoice3Model, _ = _cosyvoice3_model_and_runner()
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
    return SamplingMetadata(
        temperature=torch.tensor([1.0], dtype=torch.float32),
        all_greedy=False,
        all_random=True,
        top_p=torch.tensor([0.8], dtype=torch.float32),
        top_k=torch.tensor([25], dtype=torch.int32),
        generators={},
        max_num_logprobs=None,
        no_penalties=False,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(1, dtype=torch.float32),
        presence_penalties=torch.zeros(1, dtype=torch.float32),
        repetition_penalties=torch.tensor([repetition_penalty], dtype=torch.float32),
        output_token_ids=output_token_ids,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


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
                model._ras_sample_one(
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

    _, GPUARModelRunner = _cosyvoice3_model_and_runner()
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

    _, GPUARModelRunner = _cosyvoice3_model_and_runner()
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

    _, GPUARModelRunner = _cosyvoice3_model_and_runner()
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
    from collections import OrderedDict

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
    import vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 as module

    monkeypatch.setenv("COSYVOICE3_BATCH_FLOW", "1")
    monkeypatch.setattr(module, "cosyvoice3_packed_inference_enabled", lambda: True)
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
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

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
    import vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 as module

    model = _make_code2wav_model()
    model._code2wav_trt_done = False
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module, "cosyvoice3_packed_inference_enabled", lambda: True)
    monkeypatch.setattr(module, "_cosyvoice3_trt_enabled", lambda: True)

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
    model_cls, _ = _cosyvoice3_model_and_runner()
    model = SimpleNamespace(
        _stream_audio_cache_lock=Lock(),
        _stream_vocoder_cache_by_req={"cancelled": {"mel": torch.ones(2)}, "live": {"mel": torch.zeros(2)}},
    )
    model_cls.on_requests_finished(model, {"cancelled", "unknown"})
    model_cls.on_requests_finished(model, {"cancelled"})
    assert set(model._stream_vocoder_cache_by_req) == {"live"}


@pytest.mark.parametrize("mrv2,packed", [(False, False), (False, True), (True, False)])
def test_talker_output_contract(monkeypatch, mrv2, packed):
    import vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 as mod

    monkeypatch.setattr(mod, "cosyvoice3_packed_inference_enabled", lambda: packed)
    model = _make_talker_model()
    hidden = torch.zeros(2, 4)
    model.model = SimpleNamespace(llm=lambda embeddings, positions: embeddings)
    sentinel = object()
    model.make_omni_output = lambda *args, **kwargs: sentinel
    if mrv2:
        model._sampling_eps = 1e-6
        model.mrv2_custom_sampler(SimpleNamespace(penalties_state=None))
    output = model.forward(torch.ones(2, dtype=torch.long), torch.arange(2), inputs_embeds=hidden)
    assert output is (hidden if mrv2 or packed else sentinel)
