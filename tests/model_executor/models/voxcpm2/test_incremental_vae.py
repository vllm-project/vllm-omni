# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.voxcpm2.incremental_vae import incremental_vae_decode

pytestmark = [pytest.mark.core_model]


def _vae():
    torch.manual_seed(42)
    native = pytest.importorskip("voxcpm.modules.audiovae.audio_vae_v2")
    decoder = nn.Sequential(
        native.CausalConv1d(2, 2, 3, dilation=3, padding=3),
        nn.Tanh(),
        native.CausalTransposeConv1d(2, 2, 4, stride=2, padding=1),
        native.CausalConv1d(2, 2, 3, padding=1),
    )
    return SimpleNamespace(decoder=decoder, decode=lambda x, **_: decoder(x))


@pytest.mark.cpu
def test_incremental_matches_full_decode_with_short_patches_and_reordering():
    vae = _vae()
    caches: dict[str, torch.Tensor] = {}
    prompts = torch.randn(2, 2, 8)
    expected = vae.decode(prompts)
    actual: list[list[torch.Tensor]] = [[], []]
    with torch.no_grad():
        for step in range(4):
            order = [0, 1] if step % 2 == 0 else [1, 0]
            patch = prompts[order, :, step * 2 : step * 2 + 2]
            output = incremental_vae_decode(
                vae, patch, caches, torch.tensor(order), torch.ones(2, dtype=torch.bool), 3, None
            )
            for row, request in enumerate(order):
                actual[request].append(output[row])
    torch.testing.assert_close(torch.stack([torch.cat(parts, -1) for parts in actual]), expected)
    # The dilation history is longer than the patch, so it must roll forward.
    assert caches["0"].shape == (3, 2, 6)


@pytest.mark.cpu
def test_incremental_inactive_slots_and_resume_into_different_slot():
    vae = _vae()
    caches: dict[str, torch.Tensor] = {}
    audio = torch.randn(1, 2, 8)
    with torch.no_grad():
        first = incremental_vae_decode(vae, audio[..., :4], caches, torch.tensor([0]), torch.tensor([True]), 3, None)
        parked = {name: cache[0].clone() for name, cache in caches.items()}
        incremental_vae_decode(vae, audio[..., 4:], caches, torch.tensor([0]), torch.tensor([False]), 3, None)
        for name, cache in caches.items():
            torch.testing.assert_close(cache[0], parked[name])
            cache[1].copy_(parked[name])
            cache[0].zero_()
        second = incremental_vae_decode(vae, audio[..., 4:], caches, torch.tensor([1]), torch.tensor([True]), 3, None)
    torch.testing.assert_close(torch.cat([first, second], -1), vae.decode(audio))


@pytest.mark.cpu
def test_incremental_restores_native_decode_after_error():
    vae = _vae()
    original = [module.forward for module in vae.decoder]
    with pytest.raises(RuntimeError):
        incremental_vae_decode(vae, torch.zeros(1, 3, 2), {}, torch.tensor([0]), torch.tensor([True]), 2, None)
    assert [module.forward for module in vae.decoder] == original


@pytest.mark.cpu
def test_incremental_native_decoder_matches_full_history():
    native = pytest.importorskip("voxcpm.modules.audiovae.audio_vae_v2")
    torch.manual_seed(7)
    decoder = native.CausalDecoder(input_channel=2, channels=16, rates=[2, 3], sr_bin_boundaries=[24000, 40000]).eval()
    vae = SimpleNamespace(decoder=decoder, decode=lambda x, sr_cond: decoder(x, sr_cond))
    audio = torch.randn(2, 2, 32)
    sr = torch.tensor([48000], dtype=torch.int32)
    cache: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        expected = vae.decode(audio, sr)
        chunks = [
            incremental_vae_decode(vae, patch, cache, torch.tensor([0, 1]), torch.tensor([True, True]), 3, sr)
            for patch in audio.split(4, dim=-1)
        ]
    torch.testing.assert_close(torch.cat(chunks, -1), expected, atol=1e-5, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
def test_incremental_cuda_graph_replay_preserves_slot_caches():
    vae = _vae()
    vae.decoder.cuda()
    cache: dict[str, torch.Tensor] = {}
    patch = torch.zeros(2, 2, 2, device="cuda")
    slots = torch.tensor([0, 1], device="cuda")
    valid = torch.zeros(2, device="cuda", dtype=torch.bool)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.no_grad(), torch.cuda.stream(stream):
        for _ in range(3):
            incremental_vae_decode(vae, patch, cache, slots, valid, 3, None)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = incremental_vae_decode(vae, patch, cache, slots, valid, 3, None)
    torch.cuda.current_stream().wait_stream(stream)
    audio = torch.randn(2, 2, 4, device="cuda")
    actual: list[torch.Tensor] = []
    with torch.no_grad():
        valid.fill_(True)
        for step in range(2):
            order = [0, 1] if step == 0 else [1, 0]
            slots.copy_(torch.tensor(order, device="cuda"))
            patch.copy_(audio[order, :, step * 2 : step * 2 + 2])
            graph.replay()
            actual.append(output[order].clone())
        torch.testing.assert_close(torch.cat(actual, -1), vae.decode(audio))
