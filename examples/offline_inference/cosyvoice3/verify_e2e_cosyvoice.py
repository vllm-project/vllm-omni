# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import os

import numpy as np
import soundfile as sf
from vllm import SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.multimodal.media.audio import load_audio

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.model_executor.models.cosyvoice3.config import CosyVoice3Config
from vllm_omni.model_executor.models.cosyvoice3.tokenizer import get_qwen_tokenizer
from vllm_omni.model_executor.models.cosyvoice3.utils import extract_text_token


def run_e2e():
    parser = argparse.ArgumentParser()
    # ""FunAudioLLM/Fun-CosyVoice3-0.5B-2512
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to CosyVoice3 model directory (e.g., pretrained_models/Fun-CosyVoice3-0.5B/).",
    )
    parser.add_argument("--stage-config", type=str, default="vllm_omni/model_executor/stage_configs/cosyvoice3.yaml")
    parser.add_argument("--prompt", type=str, default="Hello, this is a test of the CosyVoice system capability.")
    parser.add_argument(
        "--prompt-text",
        type=str,
        default="You are a helpful assistant.<|endofprompt|>Testing my voices. Why should I not?",
    )
    parser.add_argument("--audio-path", type=str, default="prompt.wav")
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to tokenizer directory (e.g., <model_path>/CosyVoice-BlankEN).",
    )
    args = parser.parse_args()
    # Ensure tokenizer directory exists
    if not os.path.exists(args.tokenizer):
        raise FileNotFoundError(f"{args.tokenizer} does not exist!")

    # Ensure stage config exists
    if not os.path.exists(args.stage_config):
        raise FileNotFoundError(f"{args.stage_config} does not exist!")

    print(f"Initializing cosyvoice E2E with model={args.model}")

    # Initialize Omni
    # This spins up the engine(s) based on the stage config
    # We pass trust_remote_code=True same as Qwen examples
    omni = Omni(
        model=args.model,
        stage_configs_path=args.stage_config,
        trust_remote_code=True,
        tokenizer=args.tokenizer,
        log_stats=True,
    )

    sampling_cfg = {"top_p": 0.8, "top_k": 25, "eos_token_id": 6561 + 1}

    print("Model initialized. Preparing inputs...")
    if args.audio_path:
        if not os.path.exists(args.audio_path):
            raise FileNotFoundError(f"Audio file not found: {args.audio_path}")
        # Load at native sample rate
        audio_signal, sr = load_audio(args.audio_path, sr=None)

        # Validate sample rate before processing (similar to original CosyVoice)
        min_sr = 16000
        if sr < min_sr:
            raise ValueError(
                f"Audio sample rate {sr} Hz is too low. "
                f"Minimum required: {min_sr} Hz. "
                f"Please provide audio with sample rate >= {min_sr} Hz."
            )

        audio_data = (audio_signal.astype(np.float32), sr)
    else:
        audio_data = AudioAsset("mary_had_lamb").audio_and_sample_rate

    prompts = {
        "prompt": args.prompt,
        "multi_modal_data": {
            "audio": audio_data,
        },
        "mm_processor_kwargs": {
            "prompt_text": args.prompt_text,
            "sample_rate": audio_data[1],
        },
    }

    print(f"Generating for prompt: {args.prompt}")

    config = CosyVoice3Config()
    tokenizer = get_qwen_tokenizer(
        token_path=args.tokenizer,
        skip_special_tokens=config.skip_special_tokens,
        version=config.version,
    )
    _, text_token_len = extract_text_token(args.prompt, tokenizer, config.allowed_special)
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
