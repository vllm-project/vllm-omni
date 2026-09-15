# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Physical-layout packing across chunk and rank boundaries."""

import weakref
from unittest.mock import patch

import pytest
import torch

from vllm_omni.diffusion.offloader import chunked_transport as transport

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("layout", list(transport.WeightLayout))
@pytest.mark.parametrize("world", [1, 2, 4])
def test_pack_materializes_once_and_preserves_strides(layout, world):
    specs = [
        ("transposed", torch.arange(80, dtype=torch.float32).reshape(8, 10).t(), False),
        ("strided", torch.arange(105, dtype=torch.int8).reshape(7, 15)[1::2, 1::3], False),
        ("contiguous", torch.arange(9, dtype=torch.float32), False),
        ("scalar", torch.tensor(3, dtype=torch.int8), True),
    ]
    manifests, shards = [], []
    for rank in range(world):
        manifest = transport.build_part_manifest(
            specs,
            block_id=0,
            part_id="attention",
            weight_shard_size=world,
            weight_shard_rank=rank,
            chunk_size_bytes=32,
            alignment_bytes=1,
            layout=layout,
        )
        with patch.object(transport, "_flat_physical", wraps=transport._flat_physical) as materialize:
            shards.append(transport.pack_local_shard(specs, manifest))
        # A transposed tensor spans many chunks; packing must copy it only once.
        sources = [id(call.args[0]) for call in materialize.call_args_list]
        assert len(sources) == len(set(sources))
        assert sources.count(id(specs[0][1])) == 1
        manifests.append(manifest)

    original = {name: tensor for name, tensor, _ in specs}
    for dm in manifests[0].dtypes:
        restored = torch.empty(dm.padded_numel, dtype=dm.dtype)
        for chunk in dm.chunks:
            gathered = torch.cat([shard[dm.dtype].narrow(0, chunk.cpu_offset, chunk.local_numel) for shard in shards])
            restored.narrow(0, chunk.full_offset, chunk.padded_numel).copy_(gathered)
        for meta in dm.tensors:
            view = torch.as_strided(restored, meta.shape, meta.stride, storage_offset=meta.offset)
            assert view.stride() == original[meta.name].stride()
            torch.testing.assert_close(view, original[meta.name], rtol=0, atol=0)

    contiguous = original["contiguous"]
    assert transport._flat_physical(contiguous, contiguous.numel()).data_ptr() == contiguous.data_ptr()


@pytest.mark.parametrize("layout", list(transport.WeightLayout))
def test_pack_skips_unowned_tensors_and_releases_temporary(layout):
    specs = [(str(i), torch.arange(64, dtype=torch.float32).reshape(8, 8).t(), False) for i in range(4)]
    manifest = transport.build_part_manifest(
        specs,
        block_id=0,
        part_id="attention",
        weight_shard_size=4,
        weight_shard_rank=0,
        chunk_size_bytes=256,
        alignment_bytes=1,
        layout=layout,
    )
    previous: weakref.ReferenceType[torch.Tensor] | None = None
    original = transport._flat_physical

    def materialize(source, size):
        nonlocal previous
        assert previous is None or previous() is None, "Previous tensor's temporary is still retained"
        flat = original(source, size)
        previous = weakref.ref(flat)
        return flat

    with patch.object(transport, "_flat_physical", side_effect=materialize) as wrapped:
        transport.pack_local_shard(specs, manifest)
    # Chunk layout owns a slice of every tensor; whole-block rank 0 owns only tensor 0.
    assert wrapped.call_count == (4 if layout is transport.WeightLayout.CHUNK_MAJOR else 1)
    assert previous is not None and previous() is None
