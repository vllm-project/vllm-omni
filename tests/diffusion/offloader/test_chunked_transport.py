# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU round trips for chunked weight storage, including quantization scales."""

import pytest
import torch
from torch import nn

from tests.diffusion.offloader.helpers import patch_offload_runtime
from vllm_omni.diffusion.offloader import distributed_layerwise_backend as dist_backend
from vllm_omni.diffusion.offloader.chunked_transport import build_part_manifest, pack_local_shard
from vllm_omni.diffusion.offloader.tensor_utils import flatten_physical_storage, physical_storage_numel

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("chunk_bytes", [32, 64])
def test_chunk_roundtrip_preserves_scalars_strides_and_padding(world_size, chunk_bytes):
    # Offset slice with holes: logical flattening loses the stride, while
    # uninitialized physical holes would make transmitted bytes nondeterministic.
    backing = torch.arange(100, dtype=torch.float32)
    sliced = backing[3:63:3].reshape(4, 5)
    weight = torch.arange(48, dtype=torch.float32).reshape(6, 8).t().to(torch.float8_e4m3fn)
    specs = [
        ("weight", weight, False),
        ("weight_scale", nn.Parameter(torch.tensor(0.25), requires_grad=False), False),
        ("input_scale", torch.tensor(0.5), True),
        ("strided_buffer", sliced, True),
        ("empty_buffer", torch.empty(0), True),
    ]
    manifests = [
        build_part_manifest(
            specs,
            block_id=0,
            part_id="test",
            weight_shard_size=world_size,
            weight_shard_rank=rank,
            chunk_size_bytes=chunk_bytes,
            alignment_bytes=4,
        )
        for rank in range(world_size)
    ]
    shards = [pack_local_shard(specs, manifest) for manifest in manifests]
    assert len({manifest.digest for manifest in manifests}) == 1

    names = {meta.name for dm in manifests[0].dtypes for meta in dm.tensors}
    assert names == {name for name, _, _ in specs}
    restored = {}
    for dm in manifests[0].dtypes:
        full = torch.empty(dm.padded_numel, dtype=dm.dtype)
        for chunk in dm.chunks:
            for rank in range(world_size):
                begin = chunk.full_offset + rank * chunk.local_numel
                full[begin : begin + chunk.local_numel].copy_(
                    shards[rank][dm.dtype][chunk.cpu_offset : chunk.cpu_offset + chunk.local_numel]
                )
            padding = full[chunk.full_offset + chunk.valid_numel : chunk.full_offset + chunk.padded_numel]
            assert torch.count_nonzero(padding.float()) == 0
        for meta in dm.tensors:
            restored[meta.name] = torch.as_strided(
                full[meta.offset : meta.offset + meta.numel],
                meta.shape,
                meta.stride,
            )
            if meta.name == "strided_buffer":
                physical = full[meta.offset : meta.offset + meta.numel]
                assert torch.count_nonzero(physical[1::3]) == 0
                assert torch.count_nonzero(physical[2::3]) == 0

    for name, expected, _ in specs:
        actual = restored[name]
        assert actual.shape == expected.shape
        assert actual.stride() == expected.stride()
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)


def test_chunk_manifest_uses_physical_storage_helpers():
    backing = torch.arange(12, dtype=torch.float32)
    sliced = backing[1:10:2]
    scalar = torch.tensor(3.0)
    specs = [
        ("sliced", sliced, True),
        ("scale", scalar, False),
    ]
    manifest = build_part_manifest(
        specs,
        block_id=0,
        part_id="helpers",
        weight_shard_size=2,
        weight_shard_rank=0,
        chunk_size_bytes=64,
        alignment_bytes=4,
    )
    by_name = {meta.name: meta for dm in manifest.dtypes for meta in dm.tensors}
    assert by_name["scale"].shape == ()
    assert by_name["scale"].stride == ()
    assert by_name["scale"].numel == 1
    assert by_name["sliced"].numel == physical_storage_numel(sliced)
    packed = pack_local_shard(specs, manifest)[torch.float32]
    assert packed[0].item() == pytest.approx(sliced.reshape(-1)[0].item())
    torch.testing.assert_close(
        flatten_physical_storage(sliced),
        sliced.new_tensor([1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0, 0.0, 9.0]),
    )


@pytest.mark.parametrize("kind", ["parameter", "buffer"])
@pytest.mark.parametrize("change", ["dtype", "shape", "stride"])
def test_chunked_hook_validates_transformed_metadata(monkeypatch, kind, change):
    patch_offload_runtime(monkeypatch, dist_backend.current_omni_platform)
    block = nn.Module()
    values = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    if kind == "parameter":
        block.register_parameter("weight", nn.Parameter(values.clone()))
    else:
        block.register_buffer("weight", values.clone())
    transforms = {
        "dtype": lambda tensor: tensor.to(torch.float64),
        "shape": lambda tensor: tensor.reshape(2, 8),
        "stride": lambda tensor: tensor.t(),
    }
    hook = dist_backend.DistributedLayerwiseOffloadHook(
        next_block=block,
        device=torch.device("cpu"),
        dp_group=object(),
        dp_size=2,
        rank=0,
        pin_memory=False,
        tensor_transforms={id(block.weight): transforms[change]},
    )
    if change == "stride":
        hook.initialize_hook(nn.Module())
        assert hook.metadata[torch.float32][0]["shape"] == values.shape
        assert hook.metadata[torch.float32][0]["stride"] == values.t().stride()
    else:
        with pytest.raises(ValueError, match="chunked weight transform changed tensor metadata for 'weight'"):
            hook.initialize_hook(nn.Module())
        # Rejected transforms must fail before the original weights are cleared.
        torch.testing.assert_close(block.weight, values)
