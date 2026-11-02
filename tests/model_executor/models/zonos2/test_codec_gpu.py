# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Network-free GPU core_model contract using a tiny real DAC architecture."""

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.zonos2.zonos2_codec import DACStreamDecoder, LocalDAC, shear

pytestmark = [pytest.mark.core_model]


@hardware_test(res={"cuda": ["L4", "B200"]}, num_cards=1)
def test_tiny_dac_gpu_stream_boundaries_and_cleanup(tmp_path, monkeypatch):
    dac = pytest.importorskip("dac", reason="Install the ZONOS2 codec extras; the dedicated GPU CI job does this")

    kwargs = dict(
        encoder_dim=8,
        encoder_rates=[2, 4, 8, 8],
        latent_dim=16,
        decoder_dim=32,
        decoder_rates=[8, 8, 4, 2],
        n_codebooks=9,
        codebook_size=1024,
        codebook_dim=2,
        sample_rate=44100,
    )
    # A real decoder structure with deterministic small random weights; no
    # remote checkpoint, no placeholder zero-waveform implementation.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        codec = dac.DAC(**kwargs).float().eval()
    path = tmp_path / "tiny-dac.pth"
    torch.save({"metadata": {"kwargs": kwargs}, "state_dict": codec.state_dict()}, path)
    monkeypatch.setenv("VLLM_ZONOS2_DAC_PATH", str(path))
    decoder = LocalDAC(device="cuda:0")
    for n in (0, 1, 15, 16, 17):
        codes = torch.arange(9 * n, device="cuda:0").reshape(9, n) % 1024
        wav = decoder.decode(codes)
        assert wav.dtype == torch.float32 and wav.shape == (n * 512,)
        assert torch.isfinite(wav).all()
    # Delay and final EOS span are constructed independently from shear().
    aligned = torch.arange(33 * 9, device="cuda:0").reshape(33, 9) % 1024
    raw = torch.full((41, 9), 1025, device="cuda:0", dtype=torch.long)
    for j in range(9):
        raw[j : 33 + j, j] = aligned[:, j]
    assert torch.equal(shear(raw, up=True)[:33], aligned)
    stream = DACStreamDecoder(decoder.decode)
    pieces = [stream.push("r", raw[:24], final=False, target=16, sequence=0)]
    assert not stream.push("r", raw[:24], final=False, target=16, sequence=0).numel()
    pieces.append(stream.push("r", raw, final=False, target=33, sequence=1))
    pieces.append(stream.push("r", raw, final=True, target=33, sequence=2))
    wav = torch.cat(pieces)
    assert wav.numel() == 33 * 512 and torch.isfinite(wav).all() and wav.abs().max() > 0
    assert not stream.states
    stream.cleanup(["r"])
    stream.push("cancel", raw[:24], final=False, target=16, sequence=0)
    stream.cleanup(["cancel"])
    assert not stream.states and not stream.closed
