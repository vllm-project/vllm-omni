# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import numpy as np

from tests.helpers.media import get_asset_path
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput


def generate_irodori_tts(omni: Omni) -> np.ndarray:
    # 2 steps for extremely fast test execution in CI/pytest
    outputs = omni.generate(
        prompts={
            "prompt": "こんにちは、私はGeorgeです。前方路口左转。",
        },
        sampling_params_list=OmniDiffusionSamplingParams(
            num_inference_steps=2,
            seed=42,
            extra_args={
                "ref_wav": str(get_asset_path("qwen3_tts/clone_2.wav")),
                "duration_scale": 1.0,
                "cfg_scale_text": 1.5,
                "cfg_scale_speaker": 2.0,
            },
        ),
    )

    assert outputs is not None
    first_output = outputs[0]
    assert first_output.final_output_type == "audio"

    req_out = first_output.request_output
    assert isinstance(req_out, OmniRequestOutput)
    assert req_out.final_output_type == "audio"
    assert req_out.multimodal_output is not None

    audio = req_out.multimodal_output.get("audio")
    assert isinstance(audio, np.ndarray)
    return audio


def test_irodori_tts_e2e():
    print("Initializing Omni engine with IrodoriTTSPipeline...")
    omni = Omni(
        model="Aratako/Irodori-TTS-500M-v3",
        model_class_name="IrodoriTTSPipeline",
    )
    try:
        print("Starting e2e audio generation...")
        audio = generate_irodori_tts(omni)
        print(f"E2E generated audio shape: {audio.shape}")

        # Audio output should be 3D numpy array (B, 1, samples)
        assert audio.ndim == 3
        assert audio.shape[0] == 1
        assert audio.shape[1] == 1
        assert audio.shape[2] > 0
        print("End-to-end Irodori TTS integration test completed successfully!")
    finally:
        omni.close()
