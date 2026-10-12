# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""``duplex_incremental_fbank`` is bitwise equal to the remote ``StreamingMelProcessorExact`` (skipped without it)."""

import importlib
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from transformers.dynamic_module_utils import init_hf_modules
from transformers.utils import HF_MODULES_CACHE

from vllm_omni.model_executor.models.minicpmo_4_5.duplex.stage0 import MiniCPMO45Stage0DuplexRuntime

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_SR = 16000


@pytest.fixture(scope="module")
def remote() -> ModuleType:  # openbmb/MiniCPM-o-4_5's remote code, from the transformers module cache
    candidates = Path(HF_MODULES_CACHE, "transformers_modules").glob("*/*/processing_minicpmo.py")
    paths = [path for path in candidates if "class StreamingMelProcessorExact" in path.read_text()]
    if not paths:
        pytest.skip("MiniCPM-o 4.5 remote processing code is not in the transformers module cache")
    init_hf_modules()
    path = max(paths, key=lambda path: path.stat().st_mtime)
    return importlib.import_module(".".join(path.relative_to(HF_MODULES_CACHE).with_suffix("").parts))


def _mel(remote: ModuleType, *, incremental: bool) -> Any:
    """Stage 0's streaming configuration, switched over by the runtime as a session's processor is."""
    mel = remote.StreamingMelProcessorExact(
        feature_extractor=remote.MiniCPMAAudioProcessor(),
        chunk_ms=1000,
        first_chunk_ms=1035,
        cnn_redundancy_ms=20,
        enable_sliding_window=True,
        slide_trigger_seconds=30.0,
        slide_stride_seconds=10.0,
    )
    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime._incremental_fbank = incremental
    runtime._maybe_enable_incremental_fbank(SimpleNamespace(_streaming_mel_processor=mel))
    assert (type(mel) is not remote.StreamingMelProcessorExact) is incremental
    return mel


@pytest.mark.parametrize("chunks", ["stage0", "random"])
def test_streams_are_bitwise_equal(remote, chunks: str) -> None:
    # Stage 0's chunk sizes, or random ones (unaligned buffer ends); a 66 s stream slides the window twice.
    rng = np.random.default_rng(7)
    audio = (rng.standard_normal(66 * _SR) * 0.05).astype(np.float32)
    audio[_SR * 3 : _SR * 7] *= 20  # loud stretch for the dynamic range
    audio[_SR * 12 : _SR * 14] = 0.0  # digital silence
    audio[_SR * 45 :] *= 1e-3  # quiet tail
    ref, inc = _mel(remote, incremental=False), _mel(remote, incremental=True)
    position, slides = 0, 0
    while position < len(audio):
        size = ref.get_chunk_size() if chunks == "stage0" else int(rng.integers(1, 2 * _SR))
        dropped = ref.left_samples_dropped
        expected, expected_info = ref.process(audio[position : position + size])
        got, got_info = inc.process(audio[position : position + size])
        assert torch.equal(got, expected) and got_info == expected_info, position
        assert torch.equal(inc._extract_full(), remote.StreamingMelProcessorExact._extract_full(inc)), position
        position += size
        slides += ref.left_samples_dropped != dropped
    assert slides >= 2
