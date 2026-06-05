# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""JoyAI-Echo offline e2e: real Hub weights, single-shot text-to-video+audio.

This is the L4 functionality test for ``JoyAIEchoPipeline`` (PR1). The
checkpoint lives at ``jdopensource/JoyAI-Echo`` on the Hugging Face Hub
(LTX-2 Community License). The pipeline is a self-contained port of the
JoyAI-Echo reference inference; multi-shot long-video support arrives in
PR2.

The test runs at the smallest documented resolution (480x832x121 frames,
8-step DMD few-step schedule) and verifies:

* The pipeline returns a video tensor of shape ``(T, H, W, 3) uint8``.
* A non-empty audio waveform with the expected sample rate is exposed via
  ``request_output.multimodal_output``.

Numerical-fidelity / PSNR comparison against the upstream reference
(``JoyAI-Echo/inference.py``) is intentionally out of scope here; that
arrives in PR2/PR3 alongside the multi-shot memory bank.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.helpers import skip_if_gated_repo_inaccessible
from tests.helpers.mark import hardware_test
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]

_MODEL_REPO = "jdopensource/JoyAI-Echo"

_HEIGHT = 480
_WIDTH = 832
_NUM_FRAMES = 121
_FPS = 25
_AUDIO_SAMPLE_RATE = 24000
_NUM_INFERENCE_STEPS = 8
_PROMPT = "A cute orange cat sitting on a sofa, soft natural light, gentle ambient music."


@hardware_test(res={"cuda": "B200"})
def test_joyai_echo_single_shot_t2v_audio() -> None:
    """End-to-end single-shot generation on real Hub weights.

    ``model_class_name`` is passed explicitly so the default-stage-cfg
    factory in ``async_omni_engine.py`` can pick the right
    ``final_output_type`` before the auto-resolution from
    ``model_index.json`` runs (mirrors the Stable-Audio test pattern).
    """
    skip_if_gated_repo_inaccessible(_MODEL_REPO)

    omni = Omni(
        model=_MODEL_REPO,
        model_class_name="JoyAIEchoPipeline",
        max_num_seqs=1,
    )
    try:
        sp = OmniDiffusionSamplingParams(
            seed=12345,
            height=_HEIGHT,
            width=_WIDTH,
            num_frames=_NUM_FRAMES,
            frame_rate=float(_FPS),
            num_inference_steps=_NUM_INFERENCE_STEPS,
        )

        outputs = omni.generate(prompts=[_PROMPT], sampling_params_list=sp)
        assert outputs, "Omni.generate returned no outputs"

        result = outputs[0]
        request_output = getattr(result, "request_output", result)
        assert isinstance(request_output, OmniRequestOutput)

        # Video tensor lives on ``request_output.images`` (DiffusionEngine
        # routes the ``"video"`` key from the post-process dict here for
        # downstream serialization).
        images = request_output.images or []
        assert images, "JoyAI-Echo produced no video frames"
        video = images[0]
        if isinstance(video, torch.Tensor):
            video_np = video.detach().cpu().numpy()
        else:
            video_np = np.asarray(video)
        assert video_np.ndim == 4, f"unexpected video shape {video_np.shape}; expected (T,H,W,3)"
        actual_T, actual_H, actual_W, actual_C = video_np.shape
        assert actual_C == 3, f"expected 3 colour channels, got {actual_C}"
        assert actual_H == _HEIGHT, f"height={actual_H} != requested {_HEIGHT}"
        assert actual_W == _WIDTH, f"width={actual_W} != requested {_WIDTH}"
        # The post-process drops trailing duplicated frames; tolerate codec /
        # post-process trim within (1-frame, requested].
        assert _NUM_FRAMES - 1 <= actual_T <= _NUM_FRAMES, (
            f"frames={actual_T} not within [{_NUM_FRAMES - 1}, {_NUM_FRAMES}]"
        )

        # Audio is exposed via ``multimodal_output``.
        mm = request_output.multimodal_output or {}
        audio = mm.get("audio")
        assert audio is not None, "JoyAI-Echo produced no audio waveform"
        if isinstance(audio, torch.Tensor):
            audio_np = audio.detach().cpu().numpy()
        else:
            audio_np = np.asarray(audio)
        assert audio_np.size > 0, "audio waveform is empty"
        sample_rate = int(mm.get("audio_sample_rate") or 0)
        assert sample_rate == _AUDIO_SAMPLE_RATE, f"audio_sample_rate={sample_rate} != expected {_AUDIO_SAMPLE_RATE}"
    finally:
        omni.close()
