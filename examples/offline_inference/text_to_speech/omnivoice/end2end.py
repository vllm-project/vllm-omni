# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end OmniVoice TTS inference via vLLM-Omni.

Supports:
- Auto voice mode: text only → generated speech
- Voice cloning mode: text + reference audio → cloned voice speech

Usage:
    # Auto voice
    python end2end.py --model k2-fsa/OmniVoice --text "Hello world"

    # Voice cloning
    python end2end.py --model k2-fsa/OmniVoice --text "Hello" \
        --ref-audio ref.wav --ref-text "reference transcription"
"""

import argparse
import os

import numpy as np
import soundfile as sf

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def run_e2e():
    parser = argparse.ArgumentParser(description="OmniVoice E2E TTS inference")
    parser.add_argument(
        "--model",
        type=str,
        default="k2-fsa/OmniVoice",
        help="Model name or path (HuggingFace or local)",
    )
    parser.add_argument(
        "--stage-config",
        type=str,
        default="vllm_omni/model_executor/stage_configs/omnivoice.yaml",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello, this is a test of the OmniVoice text to speech system.",
    )
    parser.add_argument(
        "--ref-audio",
        type=str,
        default=None,
        help="Reference audio for voice cloning (WAV file)",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default=None,
        help="Transcription of reference audio",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help="Language code (e.g., 'en', 'zh')",
    )
    parser.add_argument(
        "--instruct",
        type=str,
        default=None,
        help="Voice design instruction (e.g., 'female, low pitch, british accent')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Output audio file path",
    )
    parser.add_argument(
        "--stage-init-timeout",
        type=int,
        default=600,
        help="Stage initialization timeout in seconds",
    )
    args = parser.parse_args()

    if not os.path.exists(args.stage_config):
        raise FileNotFoundError(f"Stage config not found: {args.stage_config}")

    print(f"Initializing OmniVoice with model={args.model}")

    omni = Omni(
        model=args.model,
        stage_configs_path=args.stage_config,
        log_stats=True,
    )

    print("Model initialized. Preparing inputs...")

    # Build prompt
    mm_processor_kwargs = {}
    multi_modal_data = {}

    if args.ref_audio:
        if not os.path.exists(args.ref_audio):
            raise FileNotFoundError(f"Reference audio not found: {args.ref_audio}")

        from vllm.multimodal.media.audio import load_audio

        audio_signal, sr = load_audio(args.ref_audio, sr=None)
        multi_modal_data["audio"] = (audio_signal.astype(np.float32), sr)
        mm_processor_kwargs["ref_text"] = args.ref_text or ""
        mm_processor_kwargs["sample_rate"] = sr

    if args.lang:
        mm_processor_kwargs["lang"] = args.lang
    if args.instruct:
        mm_processor_kwargs["instruct"] = args.instruct

    prompts = {"prompt": args.text}
    if multi_modal_data:
        prompts["multi_modal_data"] = multi_modal_data
    if mm_processor_kwargs:
        prompts["mm_processor_kwargs"] = mm_processor_kwargs

    sampling_params_list = [OmniDiffusionSamplingParams()]

    print(f"Generating speech for: {args.text}")

    outputs = list(omni.generate(prompts, sampling_params_list=sampling_params_list))

    print(f"Received {len(outputs)} outputs.")
    for i, output in enumerate(outputs):
        try:
            ro = output.request_output
            if ro is None:
                print("No request_output found.")
                continue

            mm = getattr(ro, "multimodal_output", None)
            if not mm and ro.outputs:
                mm = getattr(ro.outputs[0], "multimodal_output", None)

            if mm:
                print(f"Multimodal output keys: {mm.keys()}")
                if "audio" in mm:
                    audio_out = mm["audio"]
                    sr = mm.get("sr", 24000)
                    if isinstance(audio_out, np.ndarray):
                        audio_np = audio_out
                    else:
                        audio_np = audio_out.cpu().numpy().squeeze()
                    out_path = args.output if i == 0 else f"output_{i}.wav"
                    sf.write(out_path, audio_np, sr)
                    print(f"Saved audio to {out_path} ({sr}Hz, {len(audio_np) / sr:.2f}s)")
            else:
                print("No multimodal output found.")
        except Exception as e:
            print(f"Error inspecting output: {e}")

    omni.close()
    print("Done.")


if __name__ == "__main__":
    run_e2e()
