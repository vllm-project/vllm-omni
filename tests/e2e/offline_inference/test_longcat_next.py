# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
E2E offline tests for LongCat-Next (2-stage thinker + combined multi-decoder
pipeline): text understanding, text-to-image generation, and voice-cloned
text-to-speech generation.

Prompt formats mirror the ones validated on real hardware while bisecting the
LongCat-Next generation-quality bug (see git history for
``longcat_next_debug_quality.py``, removed before this PR) -- LongCat-Next
does not use a chat template; prompts are built directly from its
``<longcat_*>`` control tokens.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest
from vllm.multimodal.media.audio import load_audio

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.inputs.data import OmniTextPrompt

MODEL = os.environ.get("LONGCAT_NEXT_MODEL_PATH", "meituan-longcat/LongCat-Next")
_DEPLOY = get_deploy_config_path("longcat_next_4gpu_80gb_multi_decoder.yaml")
_OMNI_RUNNER_PARAM = (MODEL, _DEPLOY)

# Matches the reference model's generation_config.json text block
# (do_sample=true, temperature=0.4, top_k=20, top_p=0.85,
# repetition_penalty=1.1). Without repetition_penalty, low-temperature
# decoding falls into a repeated-token loop.
_THINKER_SAMPLING_OVERRIDES = {
    "temperature": 0.4,
    "top_k": 20,
    "top_p": 0.85,
    "repetition_penalty": 1.1,
    "detokenize": True,
}

pytestmark = [
    pytest.mark.full_model,
    pytest.mark.omni,
    pytest.mark.parametrize("omni_runner", [_OMNI_RUNNER_PARAM], indirect=True),
]


def _thinker_output(outputs):
    for o in outputs:
        if getattr(o, "stage_id", None) == 0:
            return o
    return None


def _multi_decoder_output(outputs):
    for o in outputs:
        if getattr(o, "stage_id", None) == 1:
            return o
    return None


def _sampling_params_for(omni_runner: OmniRunner, *, stage0_overrides: dict, stage0_max_tokens: int):
    params_list = omni_runner.omni.default_sampling_params_list
    for key, value in _THINKER_SAMPLING_OVERRIDES.items():
        setattr(params_list[0], key, value)
    for key, value in stage0_overrides.items():
        setattr(params_list[0], key, value)
    params_list[0].max_tokens = stage0_max_tokens
    return params_list


@hardware_test(res={"cuda": "H100"}, num_cards=4)
def test_longcat_next_text(omni_runner: OmniRunner) -> None:
    """Plain-text prompt, no multimodal trigger token: exercises the thinker
    stage alone (MLA backbone decode path)."""
    prompt = OmniTextPrompt(
        prompt=(
            "<longcat_system>You are a helpful assistant. "
            "<longcat_user>What is the capital of France? Answer in one sentence. "
            "<longcat_assistant>"
        ),
        modalities=["text"],
    )
    params_list = _sampling_params_for(omni_runner, stage0_overrides={}, stage0_max_tokens=128)

    outputs = omni_runner.omni.generate(prompts=[prompt], sampling_params_list=params_list)
    thinker = _thinker_output(outputs)
    assert thinker is not None, "No stage-0 (thinker) output"

    text = thinker.request_output.outputs[0].text
    assert text is not None and len(text.strip()) > 0, "Thinker produced no text"
    assert str(thinker.request_output.outputs[0].finish_reason) != "length", (
        "Thinker hit max_tokens without finishing -- likely a repetition/EOS regression"
    )


@hardware_test(res={"cuda": "H100"}, num_cards=4)
def test_longcat_next_image_generation(omni_runner: OmniRunner) -> None:
    """Text-to-image trigger: thinker emits visual codes for the 37x37 grid,
    decoded by the multi-decoder stage's image path."""
    token_h, token_w = 37, 37
    prompt = OmniTextPrompt(
        prompt=(
            "<longcat_system>You are a helpful assistant. "
            "<longcat_user>Generate an image of a cat sitting on a laptop. "
            "<longcat_assistant>"
            f"<longcat_img_token_size>{token_h} {token_w}</longcat_img_token_size>"
            "<longcat_img_start>"
        ),
        # Pipeline schema only knows "text"/"audio" (stage 1's final_output_type is
        # statically "audio" for both image- and audio-gen; see pipeline.py) --
        # "image" is rejected by get_final_stage_id_for_e2e's modality validation.
        modalities=["audio"],
        additional_information={
            "token_w": token_w,
            "token_h": token_h,
            "cfg_scale": 3.0,
        },
    )
    # 1369 = 37*37 visual code positions + control tokens.
    params_list = _sampling_params_for(omni_runner, stage0_overrides={}, stage0_max_tokens=2048)

    outputs = omni_runner.omni.generate(prompts=[prompt], sampling_params_list=params_list)
    thinker = _thinker_output(outputs)
    assert thinker is not None, "No stage-0 (thinker) output"
    assert str(thinker.request_output.outputs[0].finish_reason) != "length", (
        "Thinker hit max_tokens before emitting <longcat_img_end> -- image generation overran the grid"
    )

    decoder = _multi_decoder_output(outputs)
    assert decoder is not None, "No stage-1 (multi-decoder) output"
    images = decoder.request_output.outputs[0].multimodal_output.get("image")
    assert images is not None, "Multi-decoder produced no decoded image"


@hardware_test(res={"cuda": "H100"}, num_cards=4)
def test_longcat_next_audio_generation(omni_runner: OmniRunner) -> None:
    """Voice-cloned TTS trigger: thinker takes a reference audio clip plus a
    text script and emits audio codes, decoded by the multi-decoder stage's
    audio path."""
    local_asset = os.path.join(MODEL, "assets", "vc_zh3.wav")
    if os.path.isdir(MODEL) and os.path.isfile(local_asset):
        ref_voice_path = local_asset
    else:
        from huggingface_hub import hf_hub_download

        ref_voice_path = hf_hub_download(repo_id=MODEL, filename="assets/vc_zh3.wav")
    audio_signal, sr = load_audio(ref_voice_path, sr=16000)
    placeholder = "<longcat_audio_start><longcat_audio_pad><longcat_audio_end>"
    prompt = OmniTextPrompt(
        prompt=(
            "<longcat_system>Replicate the voice in the audio clip to formulate an answer. "
            f"{placeholder} "
            "<longcat_user>Using this voice, say: the meeting tomorrow is on the third floor. "
            "<longcat_assistant><longcat_audiogen_start>"
        ),
        modalities=["audio"],
        multi_modal_data={"audio": (audio_signal, sr)},
    )
    params_list = _sampling_params_for(omni_runner, stage0_overrides={}, stage0_max_tokens=2048)

    outputs = omni_runner.omni.generate(prompts=[prompt], sampling_params_list=params_list)
    thinker = _thinker_output(outputs)
    assert thinker is not None, "No stage-0 (thinker) output"

    decoder = _multi_decoder_output(outputs)
    assert decoder is not None, "No stage-1 (multi-decoder) output"
    audio = decoder.request_output.outputs[0].multimodal_output.get("audio")
    assert audio is not None, "Multi-decoder produced no decoded audio"
