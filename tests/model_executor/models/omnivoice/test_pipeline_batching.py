# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.omnivoice.pipeline_omnivoice import (
    OmniVoicePipeline,
    _PreparedOmniVoiceRequest,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.models.omnivoice.omnivoice_generator import (
    _attention_metadata_from_cu_seqs,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_sdpa_fallback_mask_preserves_packed_sequence_boundaries() -> None:
    cu_seqs = torch.tensor([0, 2, 3, 6, 6], dtype=torch.int32)

    metadata = _attention_metadata_from_cu_seqs(cu_seqs, 6, needs_sdpa_mask=True)

    assert metadata.attn_mask is not None
    expected = torch.tensor(
        [
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    torch.testing.assert_close(metadata.attn_mask[0, 0], expected)


def test_eager_attention_metadata_honors_exact_max_seqlen() -> None:
    cu_seqs = torch.tensor([0, 2, 3, 6, 6], dtype=torch.int32)

    metadata = _attention_metadata_from_cu_seqs(
        cu_seqs,
        6,
        needs_sdpa_mask=False,
        max_seqlen=3,
    )

    assert metadata.extra["max_seqlen_q"] == 3
    assert metadata.extra["max_seqlen_k"] == 3


def test_request_batch_prepare_error_does_not_skip_valid_request() -> None:
    """A malformed request must not prevent another request from completing."""
    prepared = _PreparedOmniVoiceRequest(
        input_ids=torch.zeros(5, 8, dtype=torch.long),
        audio_mask=torch.ones(5, dtype=torch.bool),
        cond_len=3,
        target_len=2,
        seed=42,
    )

    def prepare_request(prompt, extra):
        del extra
        if prompt == "bad":
            return DiffusionOutput(error="invalid prompt")
        return prepared

    def collate_requests(requests):
        assert requests == [prepared]
        return prepared.input_ids, prepared.audio_mask, [prepared.cond_len]

    generator_calls = []

    def generate_tokens(**kwargs):
        generator_calls.append(kwargs)
        assert kwargs["target_lens"] == [prepared.target_len]
        return torch.ones(1, 8, prepared.target_len, dtype=torch.long)

    pipeline = SimpleNamespace(
        _prepare_request_input=prepare_request,
        _collate_request_inputs=collate_requests,
        generator=generate_tokens,
        decoder=lambda tokens: tokens.float(),
        num_step=32,
        guidance_scale=7.0,
        t_shift=1.0,
        layer_penalty_factor=0.0,
        position_temperature=0.0,
        class_temperature=0.0,
    )
    sampling = OmniDiffusionSamplingParams(num_inference_steps=32, guidance_scale=7.0)
    batch = DiffusionRequestBatch(
        requests=[
            OmniDiffusionRequest(prompt="bad", sampling_params=sampling, request_id="bad"),
            OmniDiffusionRequest(
                prompt="good",
                sampling_params=OmniDiffusionSamplingParams(num_inference_steps=32, guidance_scale=7.0),
                request_id="good",
            ),
        ]
    )

    outputs = OmniVoicePipeline.forward(pipeline, batch)

    assert len(outputs) == 2
    assert outputs[0].error == "invalid prompt"
    assert outputs[1].error is None
    torch.testing.assert_close(outputs[1].output, torch.ones(1, 8, 2))
    assert generator_calls[0]["guidance_scale"] == 7.0


def test_request_batch_honors_explicit_sampling_overrides() -> None:
    """Request steps and guidance must override the OmniVoice defaults."""
    prepared = _PreparedOmniVoiceRequest(
        input_ids=torch.zeros(5, 8, dtype=torch.long),
        audio_mask=torch.ones(5, dtype=torch.bool),
        cond_len=3,
        target_len=2,
        seed=42,
    )
    captured_sampling = []

    def generate_tokens(**kwargs):
        captured_sampling.append((kwargs["num_step"], kwargs["guidance_scale"]))
        return torch.ones(1, 8, prepared.target_len, dtype=torch.long)

    pipeline = SimpleNamespace(
        _prepare_request_input=lambda prompt, extra: prepared,
        _collate_request_inputs=lambda requests: (prepared.input_ids, prepared.audio_mask, [prepared.cond_len]),
        generator=generate_tokens,
        decoder=lambda tokens: tokens.float(),
        num_step=32,
        guidance_scale=2.0,
        t_shift=1.0,
        layer_penalty_factor=0.0,
        position_temperature=0.0,
        class_temperature=0.0,
    )
    batch = DiffusionRequestBatch(
        requests=[
            OmniDiffusionRequest(
                prompt="hello",
                sampling_params=OmniDiffusionSamplingParams(num_inference_steps=12, guidance_scale=6.5),
                request_id="request",
            )
        ]
    )

    outputs = OmniVoicePipeline.forward(pipeline, batch)

    assert outputs[0].error is None
    assert captured_sampling == [(12, 6.5)]


def test_request_batch_prepare_error_preserves_surrounding_output_indices() -> None:
    """A middle prepare error must not shift valid outputs into the wrong slots."""
    prepared_before = _PreparedOmniVoiceRequest(
        input_ids=torch.zeros(3, 8, dtype=torch.long),
        audio_mask=torch.ones(3, dtype=torch.bool),
        cond_len=2,
        target_len=1,
        seed=1,
    )
    prepared_after = _PreparedOmniVoiceRequest(
        input_ids=torch.zeros(5, 8, dtype=torch.long),
        audio_mask=torch.ones(5, dtype=torch.bool),
        cond_len=3,
        target_len=2,
        seed=2,
    )

    def prepare_request(prompt, extra):
        del extra
        if prompt == "before":
            return prepared_before
        if prompt == "after":
            return prepared_after
        return DiffusionOutput(error="invalid middle request")

    def collate_requests(requests):
        assert requests == [prepared_before, prepared_after]
        return (
            torch.cat([prepared_before.input_ids, prepared_after.input_ids]),
            torch.cat([prepared_before.audio_mask, prepared_after.audio_mask]),
            [prepared_before.cond_len, prepared_after.cond_len],
        )

    def generate_tokens(**kwargs):
        assert kwargs["target_lens"] == [1, 2]
        before = torch.full((1, 8, 1), 11, dtype=torch.long)
        after = torch.full((1, 8, 2), 22, dtype=torch.long)
        return torch.cat([before, after], dim=-1)

    pipeline = SimpleNamespace(
        _prepare_request_input=prepare_request,
        _collate_request_inputs=collate_requests,
        generator=generate_tokens,
        decoder=lambda tokens: tokens.float(),
        num_step=32,
        guidance_scale=2.0,
        t_shift=1.0,
        layer_penalty_factor=0.0,
        position_temperature=0.0,
        class_temperature=0.0,
    )
    batch = DiffusionRequestBatch(
        requests=[
            OmniDiffusionRequest(
                prompt=prompt,
                sampling_params=OmniDiffusionSamplingParams(num_inference_steps=32),
                request_id=prompt,
            )
            for prompt in ("before", "bad", "after")
        ]
    )

    outputs = OmniVoicePipeline.forward(pipeline, batch)

    assert len(outputs) == 3
    torch.testing.assert_close(outputs[0].output, torch.full((1, 8, 1), 11.0))
    assert outputs[1].error == "invalid middle request"
    torch.testing.assert_close(outputs[2].output, torch.full((1, 8, 2), 22.0))
