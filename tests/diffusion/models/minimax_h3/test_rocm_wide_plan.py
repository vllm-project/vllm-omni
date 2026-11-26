# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import ctypes
import hashlib
import struct

import pytest

from vllm_omni.diffusion.models.minimax_h3.rocm_wide_fmha import _CUTS, _AttentionArgs, _worklist

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("cut", sorted(_CUTS))
def test_worklist_covers_independent_segments(cut):
    rows = _worklist(cut)
    assert len(rows) == 1518
    for head in range(14):
        regions: dict[int, list[tuple[int, int]]] = {0: [], cut: []}
        for q0, sq, kv0, skv, first, lo, hi, h, batch, wide_waves, narrow_waves in rows:
            if h != head:
                continue
            assert (kv0, skv, lo, hi, batch) == (q0, sq, 0, (sq + 63) // 64, 0)
            assert (wide_waves, narrow_waves) in {(8, 0), (0, 8)}
            end = first + 48 * wide_waves + 32 * narrow_waves
            regions[q0].append((q0 + max(0, first), q0 + min(end, sq)))
        for start, end in ((0, cut), (cut, 37760)):
            ordered = sorted(regions[start])
            assert ordered[0][0] == start and ordered[-1][1] == end
            assert all(left < right for left, right in ordered)
            assert all(a[1] == b[0] for a, b in zip(ordered, ordered[1:]))


def test_identity_schedule_matches_hardware_replay():
    descriptors = b"".join(struct.pack("<11i", *row) for row in _worklist(37746))
    assert hashlib.sha256(descriptors).hexdigest() == "ac2fbdcef0f8dac0bf30bb5529d119ae3b74b685bfa71152210f437b106a413b"


def test_unsupported_cut():
    with pytest.raises(ValueError, match="Unsupported H3 segment"):
        _worklist(37752)


def test_module_abi():
    assert ctypes.sizeof(_AttentionArgs) == 224
    assert _AttentionArgs.work.offset == 128
    assert _AttentionArgs.packed_v.offset == 144
