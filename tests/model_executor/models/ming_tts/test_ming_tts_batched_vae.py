# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MingAudioVAEModel vae_batched_decode knob: batched forward vs per-patch path."""

import types

import pytest
import torch

from tests.helpers.mark import hardware_marks, hardware_test
from vllm_omni.model_executor.models.ming_tts.ming_tts_audio_vae import (
    _STREAM_STATE_TTL_SECONDS,
    MingAudioVAEModel,
)

pytestmark = [pytest.mark.core_model, pytest.mark.tts]
cuda = pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="NVIDIA CUDA required")

# Chunk schedules cross the sliding window at different steps per request.
_SCHEDULES = [[4, 8, 16, 5], [1, 2, 4, 8, 3], [6], [3, 3, 3, 3, 3, 3]]
_STARTS = [0, 1, 0, 2]
LATENT_DIM = 64  # pinned by validate_ming_tts_config
PATCH_SIZE = 4

_BACKBONE = {
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 4096,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-6,
    "sliding_window": 8,
    "use_sliding_window": True,
    "max_window_layers": 0,
    "vocab_size": 1,
    "_attn_implementation": "sdpa",
}


def _vllm_config(extra: dict | None, device: str):
    hf_config = types.SimpleNamespace(
        audio_tokenizer_config={
            "sample_rate": 44100,
            "patch_size": PATCH_SIZE,
            "dec_kwargs": {"backbone": dict(_BACKBONE), "latent_dim": LATENT_DIM, "output_dim": 16},
        },
    )
    model_config = types.SimpleNamespace(
        hf_config=hf_config,
        stage_connector_config={"extra": dict(extra or {})},
    )
    return types.SimpleNamespace(
        model_config=model_config,
        scheduler_config=types.SimpleNamespace(max_num_seqs=8),
        device_config=types.SimpleNamespace(device=torch.device(device)),
    )


def _model(extra: dict | None, device: str, dtype: torch.dtype = torch.float32) -> MingAudioVAEModel:
    torch.manual_seed(0)
    model = MingAudioVAEModel(vllm_config=_vllm_config(extra, device))
    return model.to(device=device, dtype=dtype).eval()


def _latents(device: str, seed: int = 1) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(sum(sizes), PATCH_SIZE, LATENT_DIM, generator=generator).to(device) for sizes in _SCHEDULES]


def _payload(request_id: str, chunk: torch.Tensor | None, finished: bool) -> dict:
    return {
        "ming_request_id": request_id,
        "meta": {"stream_finished": finished},
        **({"latent": chunk} if chunk is not None else {}),
    }


def _forward(model: MingAudioVAEModel, info_list: list[dict]) -> list[torch.Tensor]:
    return model(model_intermediate_buffer=info_list).multimodal_outputs["model_outputs"]


def _run(model: MingAudioVAEModel, latents: list[torch.Tensor], *, graphs: bool = False) -> list[torch.Tensor]:
    """Drive every schedule through forward(), staggered by ``_STARTS``."""
    if graphs:
        model._batched_decoder().capture_graphs(batch_sizes=(1, 2, 4), lengths=(8, 16, 32, 64))
    cursor = [0] * len(_SCHEDULES)
    delivered = [0] * len(_SCHEDULES)
    outputs: list[list[torch.Tensor]] = [[] for _ in _SCHEDULES]
    step = 0
    while any(delivered[request] < len(_SCHEDULES[request]) for request in range(len(_SCHEDULES))):
        info_list, owners = [], []
        for request, sizes in enumerate(_SCHEDULES):
            if step >= _STARTS[request] and delivered[request] < len(sizes):
                size = sizes[delivered[request]]
                finished = delivered[request] == len(sizes) - 1
                info_list.append(
                    _payload(
                        f"req-{request}",
                        latents[request][cursor[request] : cursor[request] + size],
                        finished,
                    )
                )
                owners.append(request)
                cursor[request] += size
                delivered[request] += 1
        if info_list:
            for request, waveform in zip(owners, _forward(model, info_list), strict=True):
                outputs[request].append(waveform.detach().float().cpu())
        step += 1
    return [torch.cat(output) for output in outputs]


@pytest.mark.cpu
def test_knob_defaults_off_and_keeps_per_patch_path():
    model = _model(extra={}, device="cpu")
    assert model._batched_enabled is False
    assert model._batched is None
    assert model.audio.decoder.upsampling.frame_holdback is False

    latents = _latents("cpu")
    _forward(model, [_payload("req-0", latents[0][:4], False)])
    # The per-patch path keeps its streaming state in the model dicts.
    assert "req-0" in model._past_key_values
    assert "req-0" in model._stream_state


@pytest.mark.cpu
def test_knob_parses_strictly():
    with pytest.raises(ValueError, match="vae_batched_decode"):
        _model(extra={"vae_batched_decode": "yes"}, device="cpu")
    # An explicit null means "not set" and keeps the default.
    assert _model(extra={"vae_batched_decode": None}, device="cpu")._batched_enabled is False
    assert _model(extra={"vae_batched_decode": True}, device="cpu")._batched_enabled is True


@pytest.mark.cpu
def test_knob_on_builds_batched_decoder_lazily():
    model = _model(extra={"vae_batched_decode": True}, device="cpu")
    assert model._batched_enabled is True
    assert model.audio.decoder.upsampling.frame_holdback is True
    assert model._batched is None  # nothing built before the first decode
    decoder = model._batched_decoder()
    assert decoder is model._batched


@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu", marks=pytest.mark.cpu),
        pytest.param("cuda", marks=[cuda, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]),
    ],
)
@pytest.mark.parametrize("graphs", [False, True])
def test_batched_forward_matches_per_patch_streams(device, graphs):
    reference = _run(_model(extra={}, device=device), _latents(device))
    batched = _run(_model(extra={"vae_batched_decode": True}, device=device), _latents(device), graphs=graphs)

    for actual, want in zip(batched, reference, strict=True):
        assert actual.shape == want.shape
        torch.testing.assert_close(actual, want, rtol=1e-4, atol=1e-5)


@cuda
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_packed_bf16_batched_forward_within_stock_drift():
    latents = _latents("cuda")
    reference = _run(_model(extra={}, device="cuda"), latents)
    stock = _run(_model(extra={}, device="cuda", dtype=torch.bfloat16), [latent.bfloat16() for latent in latents])
    batched_model = _model(extra={"vae_batched_decode": True}, device="cuda", dtype=torch.bfloat16)
    assert batched_model._batched_decoder()._packed is not None
    batched = _run(batched_model, [latent.bfloat16() for latent in latents], graphs=True)

    for actual, want, baseline in zip(batched, reference, stock, strict=True):
        assert actual.shape == want.shape
        # The batched path differs from the fp32 reference no more than the
        # per-patch path's own bf16 drift does (ming3's tolerance rule).
        tolerance = 2 * (baseline - want).abs().max().item() + 1e-3
        assert (actual - want).abs().max().item() <= tolerance


@pytest.mark.cpu
def test_batched_forward_flushes_and_clears_on_finished_without_latent():
    model = _model(extra={"vae_batched_decode": True}, device="cpu")
    latents = _latents("cpu")
    _forward(model, [_payload("req-0", latents[0][:4], False)])
    assert "req-0" in model._batched_decoder()._streams

    # EOF marker: a finished payload with no latent flushes and releases the slot.
    assert _forward(model, [_payload("req-0", None, True)])[0].numel() == 0
    assert "req-0" not in model._batched_decoder()._streams

    # An unfinished payload with Ming context but no latent is a protocol error.
    with pytest.raises(RuntimeError, match="no latent chunk"):
        _forward(model, [_payload("req-1", None, False)])


@pytest.mark.cpu
def test_batched_forward_accepts_empty_final_chunk():
    model = _model(extra={"vae_batched_decode": True}, device="cpu")
    empty = torch.zeros((0, PATCH_SIZE, LATENT_DIM))
    assert _forward(model, [_payload("req-0", empty, True)])[0].numel() == 0
    assert model._batched_decoder()._streams == {}


@pytest.mark.cpu
def test_batched_forward_decodes_a_requests_chunks_in_order():
    latents = _latents("cpu")
    first, second = latents[0][:6], latents[0][6:12]

    # Reference: the two chunks arrive in two separate steps.
    stepped = _model(extra={"vae_batched_decode": True}, device="cpu")
    out_a = _forward(stepped, [_payload("req-0", first, False)])[0]
    out_b = _forward(stepped, [_payload("req-0", second, True)])[0]
    reference = torch.cat([out_a.float().cpu(), out_b.float().cpu()])

    # The same chunks as two rows of one step decode in order, not interleaved.
    model = _model(extra={"vae_batched_decode": True}, device="cpu")
    outputs = _forward(
        model,
        [
            _payload("req-0", first, False),
            _payload("req-0", second, True),
        ],
    )
    got = torch.cat([outputs[0].float().cpu(), outputs[1].float().cpu()])
    assert got.shape == reference.shape
    torch.testing.assert_close(got, reference, rtol=1e-4, atol=1e-5)


@pytest.mark.cpu
def test_batched_releases_slots_on_finish_and_ttl_eviction():
    model = _model(extra={"vae_batched_decode": True}, device="cpu")
    latents = _latents("cpu")
    decoder = model._batched_decoder()
    _run(model, latents)
    assert decoder._streams == {}
    assert sorted(decoder._free) == list(range(decoder._max_streams))

    # An abandoned stream (no final payload) is evicted by the TTL.
    _forward(model, [_payload("stale", latents[0][:1], False)])
    assert "stale" in decoder._streams
    model._state_access_times["stale"] -= _STREAM_STATE_TTL_SECONDS + 1
    _forward(model, [_payload("fresh", latents[1][:1], False)])
    assert "stale" not in decoder._streams
    assert "fresh" in decoder._streams


@pytest.mark.cpu
def test_on_requests_finished_is_a_batched_only_backstop():
    latents = _latents("cpu")

    batched = _model(extra={"vae_batched_decode": True}, device="cpu")
    _forward(batched, [_payload("req-0", latents[0][:1], False)])
    assert "req-0" in batched._batched_decoder()._streams
    batched.on_requests_finished({"req-0"})
    assert "req-0" not in batched._batched_decoder()._streams

    # Knob off: the hook must not touch the per-patch streaming state.
    per_patch = _model(extra={}, device="cpu")
    _forward(per_patch, [_payload("req-0", latents[0][:1], False)])
    per_patch.on_requests_finished({"req-0"})
    assert "req-0" in per_patch._past_key_values
