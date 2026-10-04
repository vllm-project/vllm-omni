# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""LTX-2.5 text-to-audio offline smoke using the registered pipeline."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from tests.helpers import skip_if_gated_repo_inaccessible
from tests.helpers.mark import hardware_test

DEFAULT_MODEL = "Lightricks/LTX-2.5-Diffusers"
DEFAULT_REVISION = "a6de4b5354f078db24d9cf4778c14846788aea3d"
MODEL = os.environ.get("VLLM_TEST_LTX25_MODEL", DEFAULT_MODEL)
MODEL_REVISION = os.environ.get(
    "VLLM_TEST_LTX25_MODEL_REVISION",
    DEFAULT_REVISION if MODEL == DEFAULT_MODEL else "",
)
SAMPLE_RATE = 48_000
REQUESTED_DURATION_S = 2.0
EXPECTED_SAMPLE_COUNT = 96_480

pytestmark = [pytest.mark.diffusion, pytest.mark.slow]


@pytest.fixture(scope="module", autouse=True)
def require_ltx25_model_access() -> None:
    if not os.path.isdir(MODEL):
        skip_if_gated_repo_inaccessible(MODEL, filename="model_index.json")


@hardware_test(res={"cuda": ["H100", "B200"]}, num_cards=1)
def test_ltx25_text_to_audio_offline_generate() -> None:
    """A real LTX-2.5 checkpoint generates finite stereo 48 kHz audio offline."""
    from vllm_omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.outputs import OmniRequestOutput
    from vllm_omni.platforms import current_omni_platform

    omni_kwargs = {
        "model": MODEL,
        "model_class_name": "LTX2TextToAudioPipeline",
        "enforce_eager": True,
        "diffusion_attention_backend": "CUDNN_ATTN",
    }
    if MODEL_REVISION and not os.path.isdir(MODEL):
        omni_kwargs["revision"] = MODEL_REVISION
    omni = Omni(
        **omni_kwargs,
    )
    try:
        output = omni.generate(
            {"prompt": "A close-up recording of a concert grand piano playing a gentle melody."},
            OmniDiffusionSamplingParams(
                num_inference_steps=30,
                generator=torch.Generator(current_omni_platform.device_type).manual_seed(42),
                extra_args={
                    "audio_length": REQUESTED_DURATION_S,
                    "audio_cfg_scale": 7.0,
                    "audio_stg_scale": 1.0,
                    "audio_rescale_scale": 0.7,
                    "audio_stg_blocks": [28],
                },
            ),
            use_tqdm=False,
        )[0]
        assert isinstance(output, OmniRequestOutput)
        multimodal_output = output.multimodal_output or {}
        assert output.final_output_type == "audio"
        assert int(multimodal_output["audio_sample_rate"]) == SAMPLE_RATE

        audio = multimodal_output["audio"]
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().float().cpu().numpy()
        audio = np.asarray(audio)
        if audio.ndim == 3:
            audio = audio[0]
        assert audio.ndim == 2 and audio.shape[0] == 2, f"expected stereo [2, samples], got {audio.shape}"
        assert audio.shape[1] == EXPECTED_SAMPLE_COUNT
        assert np.isfinite(audio).all()
        assert np.any(audio != 0), "generated audio is empty"
    finally:
        omni.close()
