# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real packed ring kernel and graph replay; no mocked decoder or graph."""

from dataclasses import replace

import pytest
import torch

from tests.helpers.mark import hardware_marks
from tests.model_executor.models.test_zonos3_codec import small_config, spec
from vllm_omni.model_executor.models.zonos3.codec import (
    ZACCodecChunk,
    ZACConvNeXt,
    ZACDecoderNetwork,
    ZACMossLayer,
    ZACPackedCodec,
    ZACPackedRingState,
    ZACPackedSynthesis,
    _packed_convnext,
    zac_packed_ring_attention,
)
from vllm_omni.model_executor.models.zonos3.cuda_graph_wrapper import ZACCodecGraphWrapper
from vllm_omni.platforms import current_omni_platform

pytestmark = [
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": "L4"}),
    pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="Requires CUDA and Triton"),
]


@pytest.mark.parametrize("batch", [1, 4, 8, 16, 32])
@torch.inference_mode()
def test_packed_ring_kernel_capture_matches_cpu_reference(batch):
    torch.manual_seed(91)
    cpu = ZACMossLayer(spec(context=7)).eval()
    gpu = ZACMossLayer(spec(context=7)).cuda().eval()
    gpu.load_state_dict(cpu.state_dict())
    max_chunk, ring = 5, 11
    rows = batch * max_chunk + 3

    def state(device):
        return ZACPackedRingState(
            torch.zeros(batch * ring + rows, 2, 8, device=device),
            torch.zeros(batch * ring + rows, 2, 8, device=device),
            batch,
            ring,
            max_chunk,
        )

    oracle, captured = state("cpu"), state("cuda")
    x = torch.randn(rows, 16, device="cuda")
    slots = torch.arange(batch - 1, -1, -1, device="cuda")
    offsets = torch.zeros(batch + 1, dtype=torch.long, device="cuda")
    progress = torch.zeros(batch, dtype=torch.long, device="cuda")
    result = None

    def execute():
        nonlocal result
        result = zac_packed_ring_attention(gpu, x, slots, offsets, progress, captured)

    for _ in range(3):
        execute()
    current_omni_platform.synchronize()
    graph = current_omni_platform.capture_graph(execute)
    assert graph is not None
    for step in range(12):
        lengths = torch.tensor([(i + step) % 6 for i in range(batch)])
        offsets_cpu = torch.cat((torch.zeros(1, dtype=torch.long), lengths.cumsum(0)))
        offsets.copy_(offsets_cpu)
        host_x = torch.randn(rows, 16)
        x.copy_(host_x)
        expected = zac_packed_ring_attention(
            cpu,
            host_x,
            slots.cpu(),
            offsets_cpu,
            progress.cpu(),
            oracle,
        )
        graph.replay()
        torch.testing.assert_close(result.cpu(), expected, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(
            captured.keys[: batch * ring].cpu(), oracle.keys[: batch * ring], atol=2e-6, rtol=2e-5
        )
        torch.testing.assert_close(
            captured.values[: batch * ring].cpu(), oracle.values[: batch * ring], atol=2e-6, rtol=2e-5
        )
        progress.index_add_(0, slots, lengths.to("cuda"))


@pytest.mark.parametrize("batch", [1, 4, 8, 16, 32])
@torch.inference_mode()
def test_complete_packed_codec_graph_matches_same_eager_forward(batch):
    torch.manual_seed(98)
    network = ZACDecoderNetwork(small_config()).cuda().eval()
    rows = batch * 5 + 3
    captured, eager = (ZACPackedCodec(network, batch, 5, rows) for _ in range(2))
    codes = torch.zeros(rows, 10, dtype=torch.long, device="cuda")
    slots = torch.arange(batch, device="cuda")
    offsets = torch.zeros(batch + 1, dtype=torch.long, device="cuda")
    final = torch.zeros(batch, dtype=torch.bool, device="cuda")
    result = None

    def execute():
        nonlocal result
        result = captured(codes, slots, offsets, final)

    for _ in range(3):
        execute()
    current_omni_platform.synchronize()
    graph = current_omni_platform.capture_graph(execute)
    for step in range(14):
        lengths = torch.tensor([(i + step) % 6 if step < 13 else 0 for i in range(batch)], device="cuda")
        offsets[1:].copy_(lengths.cumsum(0))
        final.fill_(step == 13)
        codes.random_(0, 32)
        codes[:2, 4:] = -1
        expected = eager(codes, slots, offsets, final)
        graph.replay()
        torch.testing.assert_close(result[0], expected[0], atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(result[1], expected[1], atol=0, rtol=0)
        torch.testing.assert_close(result[2], expected[2], atol=0, rtol=0)


@torch.inference_mode()
def test_complete_packed_two_key_buckets_share_state():
    torch.manual_seed(99)
    network = ZACDecoderNetwork(small_config()).cuda().eval()
    wrapper = ZACCodecGraphWrapper(network, 4, 5, [(b, t) for b in (1, 4) for t in (1, 8, 20)])
    eager = ZACPackedCodec(network, 8, 5, 20)
    wrapper.capture()
    assert set(wrapper.buckets) == {(b, t) for b in (1, 4) for t in (1, 8, 20)}
    for lengths, finished in (([1, 3, 0, 4], False), ([5], False), ([2, 1, 5, 3], False), ([0, 0, 0, 0], True)):
        n = len(lengths)
        slots = torch.arange(n, device="cuda")
        valid = torch.tensor(lengths, device="cuda")
        codes = torch.randint(0, 32, (sum(lengths), 10), device="cuda")
        final = torch.full((n,), finished, device="cuda")
        actual = wrapper(codes, slots, valid, final)
        b = next(x for x in wrapper.batch_sizes if x >= n)
        t = next(t for bb, t in wrapper.capture_keys if bb == b and t >= max(1, sum(lengths)))
        bucket = wrapper.buckets[b, t]
        expected = eager(bucket.codes, bucket.slots, bucket.offsets, bucket.final)
        torch.testing.assert_close(actual[0], expected[0], atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(actual[1], expected[1][: n + 1], atol=0, rtol=0)
        torch.testing.assert_close(actual[2], expected[2][:n], atol=0, rtol=0)
        torch.testing.assert_close(wrapper.codec.state.frames[:4], eager.state.frames[:4], atol=0, rtol=0)
        assert not wrapper.codec.state.frames[4:].any()


@torch.inference_mode()
def test_segmented_conv_preserves_noncontiguous_packed_input():
    """RVQ pointwise transpose is column-major, unlike the small padded fixtures."""
    torch.manual_seed(102)
    block = ZACConvNeXt(256, 512).cuda().eval()
    offsets = torch.tensor([0, 1, 6, 8, 8], device="cuda")
    slots = torch.tensor([3, 1, 0, 2], device="cuda")
    x = torch.randn(256, 11, device="cuda").T
    assert not x.is_contiguous()
    history = torch.randn(4, 256, 6, device="cuda")
    reference_history = history.clone()
    expected_history = history.clone()
    for slot, start, end in zip(slots.tolist(), offsets[:-1].tolist(), offsets[1:].tolist(), strict=True):
        expected_history[slot] = torch.cat((history[slot], x[start:end].T), dim=-1)[:, -6:]
    actual = _packed_convnext(block, x, slots, offsets, history)
    reference = _packed_convnext(block, x.contiguous(), slots, offsets, reference_history)
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    torch.testing.assert_close(history, reference_history, atol=0, rtol=0)
    torch.testing.assert_close(history, expected_history, atol=0, rtol=0)


@torch.inference_mode()
def test_packed_model_state_graph_lifecycle_and_owned_pcm():
    """ModelState routing/ownership against identical direct packed graph math."""
    from tests.model_executor.models.test_zonos3_codec import graph_network, make_codec_state

    network = graph_network("cuda")
    manager = make_codec_state(network, 2, max_chunk=4, graph_batch_sizes=(1, 2), graph_frame_sizes=(1, 4))
    reference = ZACCodecGraphWrapper(network, 2, 4, manager.model.graph.capture_keys)
    reference.capture()
    codes = torch.randint(0, 4096, (8, 10))
    retained = []
    skip = {"a": 4, "b": 0, "c": 4}

    def check(chunks, slots, reset=()):
        slots = torch.tensor(slots, dtype=torch.long, device="cuda")
        if reset:
            reference.codec.reset(torch.tensor(reset, dtype=torch.long, device="cuda"))
        raw = torch.cat([c.codes for c in chunks])
        if not raw.numel():
            raw = torch.zeros(1, 10, dtype=torch.long)
        expected, offsets, _ = reference(
            raw.cuda(),
            slots,
            torch.tensor([len(c.codes) for c in chunks], device="cuda"),
            torch.tensor([c.finished for c in chunks], device="cuda"),
        )
        boundaries = offsets.cpu().tolist()
        expected = expected.clone()
        actual = manager.decode(chunks)
        for i, (c, out) in enumerate(zip(chunks, actual, strict=True)):
            count = boundaries[i + 1] - boundaries[i]
            crop = min(skip[c.request_id], count)
            skip[c.request_id] -= crop
            torch.testing.assert_close(
                out.audio, expected[boundaries[i] + crop : boundaries[i + 1]], atol=1e-6, rtol=1e-4
            )
            retained.append((out.audio, out.audio.clone()))

    check([ZACCodecChunk("a", 0, 0, codes[:2], skip_samples=4), ZACCodecChunk("b", 0, 0, codes[:1])], [0, 1], (0, 1))
    manager.release("a")
    check([ZACCodecChunk("b", 1, 1, codes[1:3]), ZACCodecChunk("c", 0, 0, codes[3:7], skip_samples=4)], [1, 0], (0,))
    check([ZACCodecChunk("c", 1, 4, codes[:0], True), ZACCodecChunk("b", 2, 3, codes[3:4])], [0, 1])
    check([ZACCodecChunk("b", 3, 4, codes[:0], True)], [1])
    manager.release("b")
    manager.release("c")
    assert not manager.states and not manager._finished and len(manager._free) == 2
    assert manager.model.graph.replay_count == 4 and manager.model.graph.eager_count == 0
    for audio, snapshot in retained:
        torch.testing.assert_close(audio, snapshot, atol=0, rtol=0)


@pytest.mark.parametrize("batch", [1, 4, 32])
@torch.inference_mode()
def test_release_shape_synthesis_matches_whole_waveform(batch):
    """Actual FFT/FIR/resampler shapes against stateless whole-audio arithmetic."""
    torch.manual_seed(100)
    config = replace(small_config(), n_ffts=(736, 2208), hop=184, fir_kernel=127)
    network = ZACDecoderNetwork(config).cuda().eval()
    capacity = batch * 240 + 3
    synthesis = ZACPackedSynthesis(network, batch, 240, capacity)
    slots = torch.arange(batch, device="cuda")
    lengths = [240 - i % 5 * 16 for i in range(batch)]
    spectra = [[torch.randn(2 * length, n + 2, device="cuda") * 0.2 for n in config.n_ffts] for length in lengths]
    parts = [[] for _ in lengths]
    for step in range(3):
        chunk_lengths = lengths if step < 2 else [0] * batch
        offsets = torch.tensor([0, *torch.tensor(chunk_lengths).cumsum(0).tolist()], device="cuda")
        packed = [
            torch.cat(
                [
                    *(
                        spectra[i][branch][step * length : (step + 1) * length]
                        for i, length in enumerate(chunk_lengths)
                    ),
                    torch.zeros(capacity - sum(chunk_lengths), n + 2, device="cuda"),
                ]
            )
            for branch, n in enumerate(config.n_ffts)
        ]
        audio, boundaries, _ = synthesis(packed, slots, offsets, torch.full((batch,), step == 2, device="cuda"))
        boundaries = boundaries.cpu().tolist()
        for i in range(batch):
            parts[i].append(audio[boundaries[i] : boundaries[i + 1]].clone())
    resampler = network.to_external.double()
    for i, sequence in enumerate(spectra):
        branches = []
        for branch, (y, n) in enumerate(zip(sequence, config.n_ffts, strict=True)):
            log_mag, phase = y.double().chunk(2, -1)
            magnitude = log_mag.clamp_max(network.head.max_log_magnitude).exp()
            spectrum = torch.complex(magnitude * phase.cos(), magnitude * phase.sin()).T
            branches.append(
                torch.istft(
                    spectrum,
                    n,
                    hop_length=config.hop,
                    window=synthesis.ola.windows[branch],
                    length=2 * lengths[i] * config.hop,
                )
            )
        mixed = torch.nn.functional.conv1d(
            torch.stack(branches)[None], network.head.mixer.weight.double(), padding=config.fir_kernel // 2
        )[0, 0].float()
        expected = resampler(mixed.double()).float()
        torch.testing.assert_close(torch.cat(parts[i]), expected, atol=2e-7, rtol=3e-4)
