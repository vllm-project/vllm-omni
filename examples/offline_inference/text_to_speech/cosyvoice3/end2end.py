# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import os
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
from vllm import SamplingParams
from vllm.multimodal.media.audio import load_audio

from vllm_omni.engine.arg_utils import nullify_stage_engine_defaults
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.model_executor.models.cosyvoice3.tokenizer import get_qwen_tokenizer
from vllm_omni.model_executor.models.cosyvoice3.utils import extract_text_token
from vllm_omni.transformers_utils.configs.cosyvoice3 import CosyVoice3Config

# Upstream zero-shot reference clip
ZERO_SHOT_PROMPT_URL = "https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/asset/zero_shot_prompt.wav"


def _default_ref_audio() -> str:
    # Download the upstream zero_shot_prompt.wav into the current dir
    dest = Path("zero_shot_prompt.wav")
    if not dest.exists() or dest.stat().st_size == 0:
        print(f"Downloading default reference audio to {dest}")
        urllib.request.urlretrieve(ZERO_SHOT_PROMPT_URL, dest)

    return str(dest)


def run_e2e():
    parser = argparse.ArgumentParser()
    # ""FunAudioLLM/Fun-CosyVoice3-0.5B-2512
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to CosyVoice3 model directory (e.g., pretrained_models/Fun-CosyVoice3-0.5B/).",
    )
    parser.add_argument(
        "--deploy-config",
        type=str,
        default=None,
        help="Override the deploy config path. If unset, auto-loads "
        "vllm_omni/deploy/cosyvoice3.yaml based on the HF model_type.",
    )
    parser.add_argument("--text", type=str, default="Hello, this is a test of the CosyVoice system capability.")
    parser.add_argument(
        "--prompt-text",
        type=str,
        default="You are a helpful assistant.<|endofprompt|>希望你以后，能够做的比我还好呦!",
    )
    parser.add_argument(
        "--ref-audio",
        type=str,
        default=None,
        help="Path to reference audio for voice cloning. "
        "If unset, downloads the upstream CosyVoice3 zero-shot prompt audio clip",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to tokenizer directory (e.g., <model_path>/CosyVoice-BlankEN).",
    )
    nullify_stage_engine_defaults(parser)
    args = parser.parse_args()
    # Ensure tokenizer directory exists
    if not os.path.exists(args.tokenizer):
        raise FileNotFoundError(f"{args.tokenizer} does not exist!")

    if args.deploy_config is not None and not os.path.exists(args.deploy_config):
        raise FileNotFoundError(f"{args.deploy_config} does not exist!")

    print(f"Initializing cosyvoice E2E with model={args.model}")

    omni = Omni(
        model=args.model,
        deploy_config=args.deploy_config,
        tokenizer=args.tokenizer,
        log_stats=True,
    )

    sampling_cfg = {"top_p": 0.8, "top_k": 25, "eos_token_id": 6561 + 1}

    print("Model initialized. Preparing inputs...")
    ref_audio_path = args.ref_audio or _default_ref_audio()
    if not os.path.exists(ref_audio_path):
        raise FileNotFoundError(f"Audio file not found: {ref_audio_path}")
    # Load at native sample rate
    audio_signal, sr = load_audio(ref_audio_path, sr=None)

    # Validate sample rate before processing (similar to original CosyVoice)
    min_sr = 16000
    if sr < min_sr:
        raise ValueError(
            f"Audio sample rate {sr} Hz is too low. "
            f"Minimum required: {min_sr} Hz. "
            f"Please provide audio with sample rate >= {min_sr} Hz."
        )

    audio_data = (audio_signal.astype(np.float32), sr)

    prompts = {
        "prompt": args.text,
        "multi_modal_data": {
            "audio": audio_data,
        },
        "mm_processor_kwargs": {
            "prompt_text": args.prompt_text,
            "sample_rate": audio_data[1],
        },
    }

    print(f"Generating for prompt: {args.text}")

    config = CosyVoice3Config()
    tokenizer = get_qwen_tokenizer(
        token_path=args.tokenizer,
        skip_special_tokens=config.skip_special_tokens,
        version=config.version,
    )
    _, text_token_len = extract_text_token(args.text, tokenizer, config.allowed_special)
    base_len = int(text_token_len)
    min_len = int(base_len * config.min_token_text_ratio)
    max_len = int(base_len * config.max_token_text_ratio)

    # Build SamplingParams for each stage (GPT, S2Mel, Vocoder)
    gpt_sampling = SamplingParams(
        temperature=1.0,
        top_p=sampling_cfg["top_p"],
        top_k=sampling_cfg["top_k"],
        repetition_penalty=2.0,
        min_tokens=min_len,
        max_tokens=max_len,
        stop_token_ids=[sampling_cfg["eos_token_id"]],
        # allowed_token_ids=list(range(6561+3)),
        detokenize=False,
    )
    # Not used
    s2mel_sampling = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        repetition_penalty=2.0,
        max_tokens=256,
        detokenize=False,
    )

    sampling_params_list = [gpt_sampling, s2mel_sampling]

    # Start profiling (requires VLLM_TORCH_PROFILER_DIR env var)
    if os.environ.get("VLLM_TORCH_PROFILER_DIR"):
        print("Starting profiler...")
        omni.start_profile()

    # Generate (Omni orchestrator requires a per-stage SamplingParams list)
    outputs = list(omni.generate(prompts, sampling_params_list=sampling_params_list[:2]))

    # Stop profiling and get results
    if os.environ.get("VLLM_TORCH_PROFILER_DIR"):
        print("Stopping profiler...")
        profile_results = omni.stop_profile()
        print(f"Profile traces saved to: {profile_results}")

    print(outputs)
    # Verify outputs
    print(f"Received {len(outputs)} outputs.")
    for i, output in enumerate(outputs):
        try:
            ro = output.request_output
            if ro is None:
                print("No request_output found.")
                continue

            # Multimodal output may be attached to RequestOutput or CompletionOutput.
            mm = getattr(ro, "multimodal_output", None)
            if not mm and ro.outputs:
                mm = getattr(ro.outputs[0], "multimodal_output", None)

            if mm:
                print(f"Multimodal output keys: {mm.keys()}")
                if "audio" in mm:
                    audio_out = mm["audio"]
                    print(f"Generated Audio Shape: {audio_out.shape}")
                    out_path = f"output_{i}.wav"
                    sf.write(out_path, audio_out.cpu().numpy().squeeze(), 22050)
                    print(f"Saved audio to {out_path}")
            else:
                print("No multimodal output found.")
        except Exception as e:
            print(f"Error inspecting output: {e}")
    omni.close()


if __name__ == "__main__":
    run_e2e()
