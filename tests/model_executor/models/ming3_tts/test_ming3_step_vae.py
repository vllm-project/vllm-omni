# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

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
