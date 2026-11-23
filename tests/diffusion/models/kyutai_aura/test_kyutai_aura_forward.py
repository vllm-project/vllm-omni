# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Kyutai AURA ``forward`` without weights: the video ends with the speech, offline and when streaming.

The chunk generator and the speech loader are stand-ins; ``forward``, ``_request_inputs`` and the crop run
as written."""

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.kyutai_aura import pipeline_kyutai_aura as module
from vllm_omni.diffusion.models.kyutai_aura.config import KYUTAI_AURA_1_3B
from vllm_omni.diffusion.models.kyutai_aura.pipeline_kyutai_aura import KyutaiAuraPipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]
SAMPLE_RATE = 24000


def _run(monkeypatch, sizes: list[int], speech_samples: int, streaming: bool):
    requested: list[int] = []
    emitted: list = []

    def generate_chunks(image, audio, prompt, seed=42, num_chunks=None):
        for index, size in enumerate(sizes):
            requested.append(index)
            yield index, len(sizes), torch.zeros(3, size, 4, 4)

    pipeline = KyutaiAuraPipeline.__new__(KyutaiAuraPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.cfg = KYUTAI_AURA_1_3B
    pipeline.od_config = SimpleNamespace(streaming_output=streaming)
    pipeline.audio_features = cast(Any, SimpleNamespace(sample_rate=SAMPLE_RATE))
    pipeline.generate_chunks = generate_chunks
    monkeypatch.setattr(module, "load_speech", lambda audio, rate: np.zeros(speech_samples, dtype=np.float32))
    monkeypatch.setattr(module, "get_emit_chunk", lambda: emitted.append)
    request = SimpleNamespace(
        prompts=[{"prompt": "A person speaking.", "multi_modal_data": {"image": object(), "audio": "speech.wav"}}],
        sampling_params=SimpleNamespace(resolved_frame_rate=None, extra_args={}, seed=None),
        is_dummy_run=lambda: False,
    )
    return pipeline.forward(request), emitted, requested


def test_offline_video_ends_with_the_speech(monkeypatch):
    # The model-card sample: 9.08 s of speech, a plan of 10 chunks (157 frames).
    outputs, emitted, requested = _run(monkeypatch, [13] + [16] * 9, 217920, streaming=False)
    video, waveform, sample_rate, fps = outputs[0].output
    assert video.shape[2] == 146 and fps == KYUTAI_AURA_1_3B.fps
    assert len(waveform) == 217920 and sample_rate == SAMPLE_RATE
    assert emitted == [] and requested == list(range(10))


def test_streaming_stops_with_the_speech_and_reports_what_was_sent(monkeypatch):
    # 1.5 s of speech needs 24 frames: chunk 0 (13 frames) and 11 frames of chunk 1 of a 3-chunk plan.
    outputs, emitted, requested = _run(monkeypatch, [13, 16, 16], 36000, streaming=True)
    assert [(out.chunk_index, out.output.shape[2]) for out in emitted] == [(0, 13), (1, 11)]
    assert requested == [0, 1]  # the third chunk is never generated
    finished = outputs[0]
    assert finished.finished and (finished.chunk_index, finished.total_chunks) == (1, 2)
