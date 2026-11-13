# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Optional parity against the unmodified, checksum-pinned upstream module.

Set HY_NOMAS_REFERENCE_FILE to uniform_history_cache.py from Tencent-Hunyuan/
Hy-Nomas commit 911cea6458a89c2663513ff7099561ec190dd491. Only profiling is stubbed;
selection and history rebuilding execute the original upstream functions.
"""

import hashlib
import importlib.util
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType

import pytest
import torch

from vllm_omni.diffusion.models.hy_nomas.history_archive import HyNomasHistoryArchive
from vllm_omni.diffusion.models.hy_nomas.history_sampling import (
    ARCHIVED_HISTORY_MODES,
    history_frame_indices,
    select_history,
    selected_layers,
    selected_mask,
)

pytestmark = [pytest.mark.local_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def reference(monkeypatch):
    path = os.environ.get("HY_NOMAS_REFERENCE_FILE")
    if not path:
        pytest.skip("Set HY_NOMAS_REFERENCE_FILE to the pinned upstream source")
    assert path is not None
    assert (
        hashlib.sha256(Path(path).read_bytes()).hexdigest()
        == "d636526a536bdbe52c3785f8c73351933eb886dbd934efcaf7465101ad32a4be"
    )
    metrics = ModuleType("hymm.utils.inference_metrics")
    metrics.__dict__.update(profiled=lambda name: lambda function: function, stage=lambda name: nullcontext())
    monkeypatch.setitem(sys.modules, metrics.__name__, metrics)
    spec = importlib.util.spec_from_file_location("hy_nomas_pinned_sampling", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mode", ARCHIVED_HISTORY_MODES)
def test_sampling_grid_matches_pinned_reference(reference, mode):
    for total in (0, 1, 2, 17, 73, 793, 4096, 2**53 + 11, 2**63 + 17):
        for budget in (1, 2, 3, 9, 18, 36, 72):
            for gamma in (0.01, 0.5, 1.0, 2.0, 4.0, 100.0):
                assert history_frame_indices(total, budget, mode, gamma=gamma) == reference.history_frame_indices(
                    total, budget, mode, gamma=gamma
                )


@pytest.mark.parametrize("mode", ARCHIVED_HISTORY_MODES)
@pytest.mark.parametrize("budget", [18, 36, 72])
def test_packed_kv_mask_and_action_layout_match_pinned_reference(reference, mode, budget):
    archive = HyNomasHistoryArchive(max_bytes=1 << 24, max_frames=4096)
    records = []
    for segment in range(40):
        text = 4 + segment % 3
        # Three chunks per logical segment; only the first carries persistent text.
        keys, values, masks = [], [], []
        for chunk in range(3):
            text_count = text if chunk == 0 else 0
            length = 6 + text_count
            key = torch.arange(length * 8, dtype=torch.float32).reshape(length, 2, 4) + archive.frames * 100
            mask = torch.ones(1, length, dtype=torch.bool)
            if text_count:
                mask[:, -1] = False
            archive.stage(
                ((key, -key),),
                mask,
                frame_start=archive.frames,
                frame_count=3,
                tokens_per_frame=2,
                segment=segment,
                text_tokens=text_count,
                action_start=2 if text_count else -1,
            )
            archive.publish()
            keys.append(key[:6])
            values.append(-key[:6])
            masks.append(mask[:, :6])
            if chunk == 0:
                text_key, text_value, text_mask = key[6:], -key[6:], mask[:, 6:]
        records.append(
            reference.ArchivedSegment(
                segment,
                segment * 9,
                9,
                18,
                text,
                2,
                torch.cat([*keys, text_key])[None],
                torch.cat([*values, text_value])[None],
                torch.cat([*masks, text_mask], 1),
            )
        )
    positions = reference.history_frame_indices(archive.frames, budget, mode)
    selected = select_history(archive.records, positions)
    capacity = budget * 8
    cache = {"k": torch.zeros(1, capacity, 2, 4), "v": torch.zeros(1, capacity, 2, 4), "swa_capacity": capacity}
    mask = torch.zeros(1, capacity, dtype=torch.bool)
    reference._rebuild_history(cache, mask, records, positions, 2)
    key, value = next(selected_layers(selected, 1))
    count = cache["swa_valid_len"]
    torch.testing.assert_close(key, cache["k"][0, :count], rtol=0, atol=0)
    torch.testing.assert_close(value, cache["v"][0, :count], rtol=0, atol=0)
    torch.testing.assert_close(selected_mask(selected), mask[:, :count], rtol=0, atol=0)
    cursor, tails = 0, []
    for frame in selected:
        cursor += frame.num_tokens
        if frame.text is not None:
            tails.append((frame.video.segment, cursor - frame.text_tokens + frame.text.action_start, cursor))
    expected = [
        (segment, start, end)
        for segment, start, end, _ in reference.selected_action_tails(cache["swa_uniform_layout"], mask)
    ]
    assert tails == expected
