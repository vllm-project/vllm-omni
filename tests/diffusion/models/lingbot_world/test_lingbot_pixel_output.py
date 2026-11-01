# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for final pixel assembly; device/full-model gates are separate."""

import gc
import importlib.util
import sys
import weakref
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture(scope="module")
def pixel_modules():
    source = Path(__file__).resolve().parents[4] / "vllm_omni"
    package_name = "sp8_pixels_cpu"
    original_modules = set(sys.modules)
    with pytest.MonkeyPatch.context() as patch:
        package = ModuleType(package_name)
        package.__path__ = [str(source / "diffusion/models/lingbot_world")]
        patch.setitem(sys.modules, package_name, package)

        def load(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            patch.setitem(sys.modules, name, module)
            spec.loader.exec_module(module)
            return module

        pixels = load(f"{package_name}.pixel_output", source / "diffusion/models/lingbot_world/pixel_output.py")
        decoder = load("sp8_streaming_cpu", source / "experimental/ar_diffusion/streaming_decode.py")
        try:
            yield pixels, decoder.WanStreamingDecoder
        finally:
            # Relative imports performed by the subject must not leak into other tests.
            for name in set(sys.modules) - original_modules:
                if name.startswith(package_name + "."):
                    sys.modules.pop(name, None)


@dataclass
class DecoderConfig:
    patch_size: tuple[int, ...] | None = None


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("split_dim,video_dim,pixel_dim", [("width", 4, 3), ("height", 3, 2)])
def test_allgather_preserves_source_order_and_trim(pixel_modules, monkeypatch, dtype, split_dim, video_dim, pixel_dim):
    pixels, _ = pixel_modules
    shape = (1, 3, 2, 3, 2) if split_dim == "width" else (1, 3, 2, 2, 3)
    shards = [torch.linspace(-2, 2, 36).reshape(shape).to(dtype) + rank / 32 for rank in range(8)]
    snapshots = [value.clone() for value in shards]
    reference = pixels.exact_planar_pixels(torch.cat(shards, dim=video_dim)).narrow(pixel_dim, 0, 14)

    def allgather(target, source, group):
        for out, shard in zip(target, shards, strict=True):
            out.copy_(pixels.exact_planar_pixels(shard))

    monkeypatch.setattr(torch.distributed, "all_gather", allgather)
    for rank in range(8):
        policy = pixels.PixelOutputPolicy("root_uint8_allgather", object(), rank, 8)
        out = pixels.gather_final_pixels(shards[rank], policy=policy, split_dim=split_dim, expected_extent=14)
        assert torch.equal(out, reference) if rank == 0 else out.numel() == 0
    assert all(torch.equal(value, old) for value, old in zip(shards, snapshots, strict=True))


def test_root_batch_receive_order_and_wait(pixel_modules, monkeypatch):
    pixels, _ = pixel_modules
    shards = [torch.full((1, 3, 2, 3, 2), rank / 8, dtype=torch.bfloat16) for rank in range(8)]
    ordered = []

    class Work:
        def wait(self):
            ordered.append("wait")

    def op(fn, tensor, peer, group):
        assert fn is torch.distributed.irecv
        ordered.append(peer)
        tensor.copy_(pixels.exact_planar_pixels(shards[peer - 10]))
        return object()

    policy = pixels.PixelOutputPolicy("root_uint8_p2p", object(), 0, 8)
    monkeypatch.setattr(torch.distributed, "get_global_rank", lambda group, rank: rank + 10)
    monkeypatch.setattr(torch.distributed, "P2POp", op)
    monkeypatch.setattr(torch.distributed, "batch_isend_irecv", lambda ops: [Work() for _ in ops])
    out = pixels.gather_final_pixels(shards[0], policy=policy, split_dim="width", expected_extent=16)
    assert ordered == list(range(11, 18)) + ["wait"] * 7
    assert torch.equal(out, pixels.exact_planar_pixels(torch.cat(shards, dim=4)))


def test_nonprimary_batch_sends_only_to_group_primary(pixel_modules, monkeypatch):
    pixels, _ = pixel_modules
    shard = torch.full((1, 3, 2, 3, 2), 0.25, dtype=torch.bfloat16)
    old = shard.clone()
    operations = []
    waits = []

    class Work:
        def wait(self):
            waits.append(True)

    def op(fn, tensor, peer, group):
        assert fn is torch.distributed.isend and peer == 10
        assert torch.equal(tensor, pixels.exact_planar_pixels(shard))
        operations.append((fn, peer, group))
        return operations[-1]

    policy = pixels.PixelOutputPolicy("root_uint8_p2p", object(), 5, 8)
    monkeypatch.setattr(torch.distributed, "get_global_rank", lambda group, rank: rank + 10)
    monkeypatch.setattr(torch.distributed, "P2POp", op)
    monkeypatch.setattr(torch.distributed, "batch_isend_irecv", lambda ops: [Work() for _ in ops])
    out = pixels.gather_final_pixels(shard, policy=policy, split_dim="width", expected_extent=16)
    assert out.numel() == 0 and len(operations) == len(waits) == 1
    assert torch.equal(shard, old)


def test_collective_failure_retains_owners_and_propagates_original_error(pixel_modules, monkeypatch):
    pixels, _ = pixel_modules
    failure = importlib.import_module(f"{pixels.__package__}.failure")
    quarantine: list[tuple[object, ...]] = []
    monkeypatch.setattr(failure, "_QUARANTINE", quarantine)
    source = torch.zeros(1, 3, 2, 2, 2)
    references = [weakref.ref(source)]
    error = RuntimeError("injected pixel collective failure")
    calls = []

    def failed_collective(received, send, group):
        calls.append(group)
        references.extend(weakref.ref(value) for value in (send, *received))
        raise error

    monkeypatch.setattr(torch.distributed, "all_gather", failed_collective)
    policy = pixels.PixelOutputPolicy("root_uint8_allgather", object(), 0, 8)
    with pytest.raises(RuntimeError) as observed:
        pixels.gather_final_pixels(source, policy=policy, split_dim="width", expected_extent=16)
    assert observed.value is error and calls == [policy.group]
    assert len(quarantine) == 1
    # Prove the owner quarantine, rather than the exception traceback, keeps
    # source/send/receive storage alive after the failed call has unwound.
    error.__traceback__ = None
    del observed, source
    gc.collect()
    assert all(reference() is not None for reference in references)
    quarantine.clear()
    gc.collect()
    assert all(reference() is None for reference in references)


def test_exhaustive_bf16_pixel_rounding(pixel_modules):
    pixels, _ = pixel_modules
    source = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).reshape(1, 1, 1, 256, 256)
    source = source.expand(1, 3, 1, 256, 256).clone()
    expected = source.clamp(-1, 1)
    expected = ((expected[0] / 2 + 0.5).clamp(0, 1).float() * 255).round().to(torch.uint8)
    assert torch.equal(expected, pixels.exact_planar_pixels(source))


def test_policy_context_resets_after_failure(pixel_modules):
    pixels, _ = pixel_modules
    policy = pixels.PixelOutputPolicy("root_float", object(), 0, 8)
    with pytest.raises(RuntimeError, match="controlled"):
        with pixels.pixel_output_context(policy):
            assert pixels.current_pixel_output_policy() is policy
            raise RuntimeError("controlled")
    assert pixels.current_pixel_output_policy() is None
    with pytest.raises(ValueError):
        pixels.validate_mode("root")


def test_nonprimary_keeps_temporal_counters_and_cache(pixel_modules):
    _, decoder_class = pixel_modules

    class VAE:
        config = DecoderConfig()
        _cached_conv_counts = {"decoder": 1}

        def post_quant_conv(self, value):
            return value

        def decoder(self, value, *, feat_cache, feat_idx, first_chunk):
            feat_cache[0] = value.clone()
            feat_idx[0] += 1
            return value.new_empty(0)

    decoder = decoder_class(VAE())
    state = decoder.new_decode_state("s")
    for index in range(2):
        output = decoder.decode_chunk(torch.ones(1, 4, 3, 2, 2), state, produce_output=False)
        assert output.numel() == 0
        assert (state.frames_decoded, state.chunks_decoded) == (3 * (index + 1), index + 1)
        assert state.feat_map[0] is not None


def test_planar_temporal_concat_bypasses_float_clamp(pixel_modules):
    _, decoder_class = pixel_modules

    class VAE:
        config = DecoderConfig()
        _cached_conv_counts = {"decoder": 0}

        def post_quant_conv(self, value):
            return value

        def decoder(self, value, *, feat_cache, feat_idx, first_chunk):
            return torch.full((3, 1 if first_chunk else 4, 2, 2), 200, dtype=torch.uint8)

    decoder = decoder_class(VAE())
    state = decoder.new_decode_state("s")
    output = decoder.decode_chunk(torch.ones(1, 4, 3, 2, 2), state, output_format="planar_uint8")
    assert output.shape == (3, 9, 2, 2)
    assert torch.all(output == 200)
