# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUMULATIVE outputs re-consolidate the growing stream every step.

The growth buffer must return exactly what ``torch.cat`` returned and keep
already-emitted snapshots intact.
"""

import pytest
import torch
from vllm.outputs import PoolingRequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine import FinishReason

from vllm_omni.outputs.mm_outputs import MultimodalPayload
from vllm_omni.outputs.output_modality import OutputModality
from vllm_omni.outputs.output_processor import OmniRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _state(kind: RequestOutputKind) -> OmniRequestState:
    return OmniRequestState(
        request_id="audio",
        external_req_id="audio",
        parent_req=None,
        request_index=0,
        lora_request=None,
        prompt=None,
        prompt_token_ids=[0],
        prompt_embeds=None,
        logprobs_processor=None,
        detokenizer=None,
        max_tokens_param=None,
        arrival_time=0.0,
        queue=None,
        log_stats=False,
        stream_interval=1,
        output_kind=kind,
    )


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(torch.int32)


def _chunk(index: int, length: int = 24) -> torch.Tensor:
    generator = torch.Generator().manual_seed(index)
    return torch.randn(length, generator=generator)


def test_cumulative_stream_matches_torch_cat_and_keeps_old_snapshots():
    state = _state(RequestOutputKind.CUMULATIVE)
    chunks: list[torch.Tensor] = []
    emitted: list[torch.Tensor] = []
    for index in range(200):
        chunk = _chunk(index, length=17 + index % 5)
        chunks.append(chunk)
        state.add_multimodal_tensor({"model_outputs": chunk, "sr": torch.tensor(24000)}, "audio")
        output = state.make_request_output([], None, None, None)
        assert output is not None and not isinstance(output, PoolingRequestOutput)
        audio = output.outputs[0].multimodal_output["audio"]
        assert audio.is_contiguous() and audio.dtype == torch.float32
        assert torch.equal(_bits(audio), _bits(torch.cat(chunks)))
        emitted.append(audio)
    assert state._mm_growth

    # Snapshots handed out earlier still hold exactly their prefix.
    for index, audio in enumerate(emitted):
        assert torch.equal(_bits(audio), _bits(torch.cat(chunks[: index + 1])))

    final = state.make_request_output([], None, FinishReason.STOP, None)
    assert final is not None and final.finished
    assert torch.equal(_bits(final.outputs[0].multimodal_output["audio"]), _bits(torch.cat(chunks)))


@pytest.mark.parametrize("kind", [RequestOutputKind.DELTA, RequestOutputKind.FINAL_ONLY])
def test_non_cumulative_kinds_keep_plain_concatenation(kind):
    state = _state(kind)
    chunks = [_chunk(index) for index in range(5)]
    for chunk in chunks:
        state.add_multimodal_tensor({"model_outputs": chunk}, "audio")
        state.make_request_output([], None, None, None)
    final = state.make_request_output([], None, FinishReason.STOP, None)
    assert state._mm_growth == {}
    if kind == RequestOutputKind.FINAL_ONLY:
        assert torch.equal(_bits(final.outputs[0].multimodal_output["audio"]), _bits(torch.cat(chunks)))


def _consolidated(
    payload: MultimodalPayload, growth: dict | None, key: str = "audio", modality=OutputModality.AUDIO
) -> torch.Tensor:
    payload.consolidate_tensors(modality, growth=growth)
    return payload.tensors[key]


def test_growth_falls_back_for_mixed_dtype_and_multi_dim_chunks():
    growth: dict = {}
    mixed = MultimodalPayload(tensors={"audio": [torch.ones(3), torch.ones(2, dtype=torch.float64)]})
    assert torch.equal(_consolidated(mixed, growth), torch.cat([torch.ones(3), torch.ones(2, dtype=torch.float64)]))
    assert _consolidated(mixed, growth).dtype == torch.float64
    assert "audio" not in growth

    frames = [torch.arange(6.0).reshape(2, 3), torch.arange(6.0, 9.0).reshape(1, 3)]
    stacked = MultimodalPayload(tensors={"audio": list(frames)})
    # CONCAT_LAST on 2-D chunks keeps its existing shape-mismatch fallback.
    expected = MultimodalPayload(tensors={"audio": list(frames)})
    assert torch.equal(_consolidated(stacked, growth), _consolidated(expected, None))
    assert "audio" not in growth


def test_growth_restarts_when_the_head_is_not_the_last_view():
    growth: dict = {}
    first = MultimodalPayload(tensors={"audio": [torch.ones(4), torch.full((2,), 2.0)]})
    view = _consolidated(first, growth)
    # A foreign head (e.g. a replaced stream) must not append after `view`.
    other = MultimodalPayload(tensors={"audio": [torch.zeros(3), torch.full((1,), 5.0)]})
    assert torch.equal(_consolidated(other, growth), torch.tensor([0.0, 0.0, 0.0, 5.0]))
    assert torch.equal(view, torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0, 2.0]))


def test_growth_extends_dim0_streams_of_matrices():
    """A LATENT stream (e.g. the Thinker's [T, hidden] rows) grows in place along dim 0."""
    growth: dict = {}
    chunks: list[torch.Tensor] = []
    emitted: list[tuple[torch.Tensor, torch.Tensor]] = []
    head = None
    for index in range(300):
        chunk = torch.randn(1 + index % 3, 8, generator=torch.Generator().manual_seed(index)).to(torch.bfloat16)
        chunks.append(chunk)
        payload = MultimodalPayload(tensors={"latent": [chunk] if head is None else [head, chunk]})
        head = _consolidated(payload, growth, "latent", OutputModality.LATENT)
        assert torch.equal(head, torch.cat(chunks, dim=0))
        emitted.append((head, torch.cat(chunks, dim=0)))
    assert "latent" in growth
    # Snapshots handed out earlier are never written.
    for view, expected in emitted[::37]:
        assert torch.equal(view, expected)


def test_growth_falls_back_when_dim0_chunks_change_trailing_shape():
    growth: dict = {}
    chunks = [torch.ones(2, 4), torch.ones(1, 3)]
    payload = MultimodalPayload(tensors={"latent": list(chunks)})
    with pytest.raises(RuntimeError):
        _consolidated(payload, growth, "latent", OutputModality.LATENT)
    assert "latent" not in growth
