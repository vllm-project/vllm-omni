# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-device byte accounting and the bounded VAE encoder ring (RFC #4480).

Two things are pinned here. First, that a session reports the bytes it actually
holds: both buckets walked, storages rather than logical element counts, and
aliases counted once. Second, that bounding the adapter's VAE encoder history
changes no numbers -- ``quant_conv`` is pointwise in time, so the frames the
reader keeps do not depend on the ones the ring drops.

All CPU, tiny tensors, no model.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vllm_omni.experimental.world_models.adapters.state_dreamzero_adapter import (
    DreamZeroStateAdapter,
)
from vllm_omni.experimental.world_models.session_state import (
    LatentBuffer,
    SessionStateManager,
    device_bytes,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _bytes(value: object) -> dict[str, int]:
    return device_bytes(value, set())


# -- device_bytes ------------------------------------------------------------


def test_counts_tensor_storage_under_its_device() -> None:
    tensor = torch.zeros(64, dtype=torch.float32)
    assert _bytes(tensor) == {"cpu": 256}


def test_views_on_one_storage_counted_once() -> None:
    tensor = torch.zeros(64, dtype=torch.float32)
    seen: set[tuple[str, int]] = set()
    first = device_bytes(tensor, seen)
    second = device_bytes(tensor[:8], seen)
    assert first == {"cpu": 256}
    assert second == {}


def test_slice_reports_the_storage_it_pins() -> None:
    # The whole point of counting storages: this slice keeps all 256 bytes
    # alive, and ``numel() * element_size()`` would claim 32.
    sliced = torch.zeros(64, dtype=torch.float32)[:8]
    assert _bytes(sliced) == {"cpu": 256}


def test_containers_recurse_and_scalars_contribute_nothing() -> None:
    payload = {
        "tensors": [torch.zeros(16, dtype=torch.float32), None],
        "count": 7,
        "flag": True,
        "name": "not-a-buffer",
    }
    assert _bytes(payload) == {"cpu": 64}


def test_numpy_arrays_land_on_the_host() -> None:
    assert _bytes(np.zeros(32, dtype=np.float32)) == {"cpu": 128}


# -- SessionState / manager --------------------------------------------------


def test_session_counts_both_buckets_and_dedupes_across_them() -> None:
    manager = SessionStateManager()
    session = manager.get_or_create_session("s")

    buffer: LatentBuffer[torch.Tensor] = LatentBuffer()
    buffer.allocate(maxlen=None)
    shared = torch.zeros(64, dtype=torch.float32)  # 256 bytes
    buffer.append(shared)
    session.put("latents", buffer)
    session.attrs["conditioning"] = torch.zeros(16, dtype=torch.float32)  # 64 bytes

    assert session.nbytes_by_device() == {"cpu": 320}

    # The same storage reachable from both buckets is one allocation, not two.
    session.attrs["alias"] = shared
    assert session.nbytes_by_device() == {"cpu": 320}


def test_stats_reports_per_device_totals() -> None:
    manager = SessionStateManager()
    manager.get_or_create_session("s").attrs["x"] = torch.zeros(64, dtype=torch.float32)

    stats = manager.stats()
    assert stats["nbytes:cpu"] == 256
    assert stats["total_nbytes"] == 256


def test_attribute_bytes_were_previously_invisible() -> None:
    """The defect behind tzhouam's review comment on PR #4487.

    Every byte here lives in ``attrs``, and the old ``nbytes`` walked only the
    typed objects -- so a session holding real memory reported zero.
    """
    manager = SessionStateManager()
    session = manager.get_or_create_session("s")
    session.attrs["conv_cache"] = [torch.zeros(1024, dtype=torch.float32)]

    assert session.nbytes_by_device() == {"cpu": 4096}


# -- the largest value DreamZero holds ---------------------------------------


def test_feat_map_bytes_are_counted() -> None:
    """``vae_enc_feat_map`` is the largest thing a DreamZero session holds --
    603 MiB at the shipped config -- and it stays an attribute, so this is
    exactly the memory the old accounting missed."""
    manager = SessionStateManager()
    adapter = DreamZeroStateAdapter("s", manager)
    adapter.vae_enc_feat_map = [torch.zeros(1024, dtype=torch.float32), None]

    assert manager.stats()["nbytes:cpu"] == 4096

    # Freeing it piecemeal is not something the manager can do: the cache has no
    # recompute source and the stream breaks without it. Ending the stream (or
    # the session) is the only release, and that reclaims attributes too.
    adapter.reset_vae_encoder_stream()
    assert manager.stats().get("nbytes:cpu", 0) == 0


# -- bounded VAE encoder ring ------------------------------------------------


def _quantize(encoder_out: torch.Tensor) -> torch.Tensor:
    """Stand-in for ``_vae_quantize_encoder_out``.

    The real one is ``WanCausalConv3d(z_dim*2, z_dim*2, 1)`` followed by a
    per-channel affine rescale. Every operation is pointwise in time, which is
    the property the ring relies on; a 1x1x1 convolution reproduces it.
    """
    weight = torch.arange(1, encoder_out.shape[1] + 1, dtype=encoder_out.dtype).view(1, -1, 1, 1, 1)
    return encoder_out * weight + 0.5


def test_ring_is_bounded() -> None:
    adapter = DreamZeroStateAdapter("s", SessionStateManager(), vae_encoder_window=2)
    adapter.vae_encoder_out = torch.randn(1, 4, 1, 2, 2)
    for _ in range(10):
        adapter.append_vae_encoder_chunk(torch.randn(1, 4, 1, 2, 2))

    assert adapter.vae_encoder_out is not None
    assert adapter.vae_encoder_out.shape[2] == 2


def test_ring_matches_unbounded_history_for_the_frames_that_are_read() -> None:
    window = 2
    adapter = DreamZeroStateAdapter("s", SessionStateManager(), vae_encoder_window=window)
    seed = torch.randn(1, 4, 1, 2, 2)
    chunks = [torch.randn(1, 4, 1, 2, 2) for _ in range(9)]

    adapter.vae_encoder_out = seed
    unbounded = seed
    for chunk in chunks:
        adapter.append_vae_encoder_chunk(chunk)
        unbounded = torch.cat([unbounded, chunk], dim=2)

    # What the reader actually returns: quantise, then keep the last `window`.
    from_ring = _quantize(adapter.vae_encoder_out)[:, :, -window:]
    from_history = _quantize(unbounded)[:, :, -window:]
    torch.testing.assert_close(from_ring, from_history, rtol=0, atol=0)


def test_ring_holds_a_single_frame_before_the_first_body_chunk() -> None:
    # After `_vae_stream_seed` there is exactly one latent frame, which is what
    # sends the reader down its padding branch.
    adapter = DreamZeroStateAdapter("s", SessionStateManager(), vae_encoder_window=2)
    seed = torch.randn(1, 4, 1, 2, 2)
    adapter.vae_encoder_out = seed

    assert adapter.vae_encoder_out is not None
    assert adapter.vae_encoder_out.shape[2] == 1
    torch.testing.assert_close(adapter.vae_encoder_out, seed, rtol=0, atol=0)


def test_resetting_the_stream_clears_the_ring() -> None:
    adapter = DreamZeroStateAdapter("s", SessionStateManager(), vae_encoder_window=2)
    adapter.vae_encoder_out = torch.randn(1, 4, 1, 2, 2)
    adapter.reset_vae_encoder_stream()

    assert adapter.vae_encoder_out is None


def test_window_must_be_positive() -> None:
    with pytest.raises(ValueError, match="vae_encoder_window"):
        DreamZeroStateAdapter("s", SessionStateManager(), vae_encoder_window=0)
