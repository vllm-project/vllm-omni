# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""B4 decode adapter; original paired assembly is reused."""

import types
from contextlib import contextmanager
from functools import lru_cache

import torch

from .paired_vae import geometry, jobs


@lru_cache(None)
def batch_plan(rank):
    return tuple(
        tuple(round_jobs[rank][i : i + 4]) for round_jobs in jobs() for i in range(0, len(round_jobs[rank]), 4)
    )


@contextmanager
def mixed_decode(model, latent, rank):
    """Yield measured calls, without changing existing exact-op counters.

    Each original paired decode call still consumes its original tile. At the
    first call in a B4 group, compute the group together and retain its outputs
    until their original consumers run. Groups never cross a gather boundary. The only linear
    override executes attention to_out at its original B1 shape and stride.
    """
    spatial, _, _ = geometry(model, latent)
    plan = batch_plan(rank)
    ordered = tuple(job for batch in plan for job in batch)
    original_decode = model.decode
    originals = []
    stats = dict(tile_outputs=0, decoder_calls=0, batch_sizes=[], to_out_calls=0)
    pending = []
    batch_index = 0

    def tile_view(job):
        window, tile = job
        sy, sx = spatial[tile]
        return latent[:, :, window * 5 : window * 5 + 7, sy, sx]

    def decode(self, clip):
        nonlocal batch_index
        expected = tile_view(ordered[stats["tile_outputs"]])
        assert (clip.data_ptr(), clip.shape, clip.stride(), clip.dtype) == (
            expected.data_ptr(),
            expected.shape,
            expected.stride(),
            expected.dtype,
        )
        if not pending:
            batch = plan[batch_index]
            batch_index += 1
            x = clip if len(batch) == 1 else torch.cat([tile_view(job) for job in batch], dim=0)
            decoded = original_decode(x)
            assert decoded.shape == (len(batch), 3, 28, 256, 256)
            assert decoded.dtype == torch.float32
            pending.extend(decoded[i : i + 1] for i in range(len(batch)))
            stats["decoder_calls"] += 1
            stats["batch_sizes"].append(len(batch))
        stats["tile_outputs"] += 1
        return pending.pop(0)

    try:
        for block in model.decoder.transformer_blocks:
            module = block.attn.to_out
            original = module.forward
            originals.append((module, original))

            def forward(self, x, _original=original):
                assert x.ndim == 3 and x.shape[1:] == (1797, 2048)
                assert x.dtype == torch.float16 and x.stride() == (1797 * 2048, 2048, 1)
                stats["to_out_calls"] += x.shape[0]
                if x.shape[0] == 1:
                    return _original(x)
                return torch.cat([_original(x[i : i + 1]) for i in range(x.shape[0])], dim=0)

            module.forward = types.MethodType(forward, module)
        model.decode = types.MethodType(decode, model)
        yield stats
        assert not pending and batch_index == len(plan)
        assert stats["tile_outputs"] == len(ordered)
        assert stats["decoder_calls"] == 21
        assert stats["to_out_calls"] == 36 * len(ordered)
    finally:
        model.decode = original_decode
        for module, original in originals:
            module.forward = original
