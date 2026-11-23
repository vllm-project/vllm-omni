# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from functools import partial

import numpy as np
import pytest
import torch

from tests.helpers.mark import hardware_test
from tests.model_executor.models.ming3_tts.test_ming3_batched_vae import _vae

pytestmark = [pytest.mark.core_model, pytest.mark.tts]
cuda = pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="NVIDIA CUDA required")

LENGTHS = [9, 2, 23]  # latents per stream; 23 latents cross the 8-token window many times
STARTS = [0, 3, 1]  # engine step of each stream's first latent


def _reference(vae, latents: torch.Tensor) -> torch.Tensor:
    """One latent per streaming decode call, as Stage 1 would receive them."""
    past, state, outputs = None, (None, None, None), []
    for index in range(latents.shape[0]):
        waveform, state, past = vae.decode(
            latents[index : index + 1].reshape(1, 1, -1),
            past_key_values=past,
            use_cache=True,
            stream_state=state,
            last_chunk=index == latents.shape[0] - 1,
        )
        outputs.append(waveform.reshape(-1).float())
    return torch.cat(outputs)


@cuda
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("graphs", [False, True])
def test_step_decode_matches_streaming_reference(graphs):
    from vllm_omni.model_executor.models.ming3_tts.step_vae_decoder import StepVAEDecoder

    vae = _vae(packed=True).cuda()
    torch.manual_seed(2)
    latents = [torch.randn(length, 8, device="cuda") for length in LENGTHS]
    with torch.inference_mode():
        reference = [_reference(vae, latent) for latent in latents]
        vae.to(torch.bfloat16)
        transformers_bf16 = [_reference(vae, latent.bfloat16()) for latent in latents]

    decoder = StepVAEDecoder(vae.decoder, max_slots=4, kv_block_size=16, ring_blocks=2)
    if graphs:
        # Steps of 1-3 rows replay padded 1/2/4-row graphs.
        decoder.capture_graphs()
    outputs: list[list[torch.Tensor]] = [[] for _ in LENGTHS]
    taken = [0] * len(LENGTHS)  # latents handed to the decoder
    positions = [0] * len(LENGTHS)
    for step in range(max(start + length + 1 for start, length in zip(STARTS, LENGTHS))):
        prime, rows = [], []
        for stream, (start, length) in enumerate(zip(STARTS, LENGTHS)):
            if step == start:
                prime.append(stream)
            elif start < step <= start + length:
                rows.append(stream)
        if prime:
            index = torch.tensor(prime, device="cuda")
            decoder.prime(index, torch.stack([latents[s][0] for s in prime]).bfloat16())
            for stream in prime:
                taken[stream] = 1
        if not rows:
            continue
        # The step after a stream's last latent flushes it; its new latent is ignored.
        last = [taken[s] == LENGTHS[s] for s in rows]
        new = torch.stack([latents[s][min(taken[s], LENGTHS[s] - 1)] for s in rows]).bfloat16()
        # An extra row for a stream that is not stepping must leave its state alone.
        spare = next((s for s in range(len(LENGTHS)) if s not in rows and taken[s]), None)
        slots = rows + ([spare] if spare is not None else [])
        audio, audio_start, audio_end = decoder.step(
            torch.tensor(slots, device="cuda"),
            torch.tensor([positions[s] for s in slots], device="cuda"),
            torch.tensor([positions[s] == 0 for s in slots], device="cuda"),
            torch.cat([new, torch.zeros(len(slots) - len(rows), 8, device="cuda", dtype=new.dtype)]),
            torch.tensor(last + [False] * (len(slots) - len(rows)), device="cuda"),
            torch.tensor([True] * len(rows) + [False] * (len(slots) - len(rows)), device="cuda"),
        )
        for row, stream in enumerate(rows):
            outputs[stream].append(audio[row, audio_start[row] : audio_end[row]])
            positions[stream] += decoder.scale
            taken[stream] += 1
        if spare is not None:
            assert audio_end[-1] == audio_start[-1]

    for chunks, want, baseline in zip(outputs, reference, transformers_bf16, strict=True):
        got = torch.cat(chunks)
        assert got.shape == want.shape
        tolerance = 2 * (baseline - want).abs().max().item() + 1e-3
        assert (got - want).abs().max().item() <= tolerance


@cuda
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("graphs", [False, True])
def test_invalid_step_rows_keep_kv_length_and_resume(graphs):
    from vllm_omni.model_executor.models.ming3_tts.step_vae_decoder import StepVAEDecoder

    vae = _vae(packed=True).cuda().bfloat16()
    decoder = StepVAEDecoder(vae.decoder, max_slots=4, kv_block_size=16, ring_blocks=2)
    metadata = []
    packed = decoder.packed

    def record_packed(frames, positions, cu_seqlens, scale, kv):
        # Keep device buffers from the actual eager call or graph replay.
        metadata.append((kv.seqused_k, kv.slot_mapping))
        return packed(frames, positions, cu_seqlens, scale, kv)

    decoder.packed = record_packed
    if graphs:
        decoder.capture_graphs((4,))
    slots = torch.arange(3, device="cuda")
    torch.manual_seed(17)
    latents = torch.randn(5, 3, 8, device="cuda", dtype=torch.bfloat16)
    decoder.prime(slots, latents[0])
    positions = torch.zeros(3, device="cuda", dtype=torch.long)
    for step in range(1, 5):
        # Pause at position zero and again after a valid step; each row resumes.
        valid = torch.tensor([True, step not in (1, 3), True], device="cuda")
        new = latents[step].clone()
        new[1] = torch.where(valid[1], new[1], float("nan"))
        old_held = decoder.held[1].clone()
        old_caches = [cache[2:4].clone() for cache in (*decoder.keys, *decoder.values)]
        audio, start, end = decoder.step(slots, positions, positions == 0, new, torch.zeros_like(valid), valid)
        expected = positions + valid.long() * decoder.scale
        used, mapping = metadata[-1]
        torch.testing.assert_close(used[:3].long(), expected)
        if graphs:
            assert used[3].item() == 0  # graph padding cannot expose unwritten KV
        assert torch.isfinite(audio[0, start[0] : end[0]]).all()
        assert torch.isfinite(audio[2, start[2] : end[2]]).all()
        if not valid[1]:
            assert start[1] == end[1]
            assert (mapping[decoder.scale : 2 * decoder.scale] == -1).all()
            torch.testing.assert_close(decoder.held[1], old_held)
            for cache, old in zip((*decoder.keys, *decoder.values), old_caches, strict=True):
                torch.testing.assert_close(cache[2:4], old)
        else:
            assert end[1] > start[1]
            assert torch.isfinite(audio[1, start[1] : end[1]]).all()
        positions = expected


@cuda
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("graphs", [False, True])
@torch.inference_mode()
def test_side_stream_matches_serial_decode_with_slot_reuse(graphs):
    from tests.model_executor.models.ming3_tts.test_ming3_mrv2_state import _batch
    from vllm_omni.model_executor.models.ming3_tts.model_state import Ming3ModelState
    from vllm_omni.model_executor.models.ming3_tts.step_vae_decoder import StepVAEDecoder
    from vllm_omni.worker_v2.omni_ar_model_runner import _async_copy_tensor

    vae = _vae(packed=True).cuda().bfloat16()
    serial, side = [StepVAEDecoder(vae.decoder, max_slots=4, kv_block_size=16, ring_blocks=2) for _ in range(2)]
    if graphs:
        for decoder in (serial, side):
            # Three-row steps exercise eager fallback between captured buckets.
            decoder.capture_graphs((1, 2))
    assert side.stream is not None
    pooled = {torch.cuda.Stream().cuda_stream for _ in range(128)}
    for graph, _ in side._graphs.values():
        assert graph._capture_stream_owner.cuda_stream not in pooled
    state = object.__new__(Ming3ModelState)
    state._step_vae = side
    state._vae_positions = torch.zeros(4, dtype=torch.long, device="cuda")
    state._sample_rate = torch.tensor(1600, dtype=torch.int32)
    state._audio_chunks = {}
    state._chunk_latents = (1,)
    state._samples_per_latent = side.scale * side.hop
    copy_stream = torch.cuda.Stream()
    positions = np.zeros(4, dtype=np.int64)
    pending = []
    # Prime, pause, flush, reuse a cancelled slot, and mix prime/decode rows.
    schedule = [
        ([0, 1, 2], [True, True, True], [True, True, True], [False, False, False]),
        ([2, 0, 1], [False, False, False], [True, False, True], [False, False, False]),
        ([1, 2], [False, False], [False, True], [True, False]),
        ([0, 2], [True, False], [True, True], [False, False]),
        ([2, 0, 3], [False, False, True], [False, True, True], [True, False, False]),
        ([3, 0], [False, False], [False, False], [True, True]),
    ]
    for slots, first, emitted, flush in schedule:
        count = len(slots)
        batch = _batch(
            slots,
            computed=[0 if prime else 1 for prime in first],
            scheduled=[1] * count,
            prompt_len=[1] * count,
            seq_lens=[1] * count,
        )
        latent = torch.randn(count, 8, device="cuda", dtype=torch.bfloat16)
        index = torch.tensor(slots, device="cuda")
        primed = [i for i, value in enumerate(first) if value]
        decoded = [i for i, value in enumerate(first) if not value]
        if primed:
            serial.prime(index[primed], latent[primed])
            positions[np.asarray(slots)[primed]] = 0
        expected = None
        if decoded:
            selected = np.asarray(slots)[decoded]
            cursor = torch.tensor(positions[selected], device="cuda")
            valid = np.asarray(emitted)[decoded] | np.asarray(flush)[decoded]
            expected = serial.step(
                index[decoded],
                cursor,
                cursor == 0,
                latent[decoded],
                torch.tensor(np.asarray(flush)[decoded], device="cuda"),
                torch.tensor(valid, device="cuda"),
            )
            positions[selected] += valid * serial.scale
        state._decode_step_audio(
            batch,
            np.arange(count),
            np.asarray(slots),
            latent,
            torch.tensor(emitted, device="cuda"),
            torch.tensor(flush, device="cuda"),
        )
        output = state.prepare_streaming_audio_output(batch, None, {})
        with torch.cuda.stream(copy_stream):
            copied = output.to_cpu(copy_stream, partial(_async_copy_tensor, copy_stream=copy_stream, pin_memory=True))
        if expected is not None:
            assert output.event is not None
            pending.append((copied, expected))
    copy_stream.synchronize()
    for actual, expected in pending:
        for got, want in zip((actual.audio, actual.start, actual.end), expected, strict=True):
            torch.testing.assert_close(got, want.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(state._vae_positions.cpu(), torch.from_numpy(positions))


@cuda
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_side_stream_exception_restores_caller_stream(monkeypatch):
    from tests.model_executor.models.ming3_tts.test_ming3_mrv2_state import _batch
    from vllm_omni.model_executor.models.ming3_tts.model_state import Ming3ModelState
    from vllm_omni.model_executor.models.ming3_tts.step_vae_decoder import StepVAEDecoder

    state = object.__new__(Ming3ModelState)
    state._step_vae = StepVAEDecoder(_vae(packed=True).cuda().bfloat16().decoder, max_slots=1)
    batch = _batch([0], computed=[0], scheduled=[1], prompt_len=[1], seq_lens=[1])
    main = torch.cuda.current_stream()

    def fail_prime(slots, latent):
        assert torch.cuda.current_stream() == state._step_vae.stream
        raise RuntimeError("injected prime failure")

    monkeypatch.setattr(state._step_vae, "prime", fail_prime)
    with pytest.raises(RuntimeError, match="injected prime failure"):
        state._decode_step_audio(
            batch,
            np.array([0]),
            np.array([0]),
            torch.zeros(1, 8, device="cuda"),
            torch.ones(1, dtype=torch.bool, device="cuda"),
            torch.zeros(1, dtype=torch.bool, device="cuda"),
        )
    assert torch.cuda.current_stream() == main
