"""
Offline Inference for Step-Audio-Editx.
"""

import argparse
import logging
import os
from pathlib import Path

import soundfile as sf
from vllm import SamplingParams

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.model_executor.models.step_audio_editx.step_audio_tokenizer import (
    estimate_step_audio_editx_prompt_len,
)

logger = logging.getLogger(__name__)


def _local_path_or_none(path: str) -> Path | None:
    candidate = Path(path).expanduser()
    if candidate.is_absolute() or path.startswith(("./", "../", "~")):
        return candidate
    return None


def _build_inputs(args):
    """Build sample inputs for StepAudioEditx.
    Returns:
        QueryResult with Omni inputs and the Base model path.
    """
    if args.ref_audio is not None:
        ref_audio_single = args.ref_audio
        if args.ref_text is not None:
            ref_text_single = args.ref_text
        else:
            raise ValueError("ref_text must be provided when ref_audio is specified.")
    else:
        # Default reference audio and text for voice cloning if not provided via CLI args.
        ref_audio_single = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav"
        ref_text_single = (
            "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
        )
    if args.text is not None:
        syn_text_single = args.text
    else:
        syn_text_single = ""
    additional_information = {
        "edit_type": args.edit_type,
        "ref_audio": [ref_audio_single],
        "ref_text": [ref_text_single],
        "text": [syn_text_single],
    }
    if args.edit_info is not None:
        additional_information.update({"edit_info": args.edit_info})
    input_length = estimate_step_audio_editx_prompt_len(additional_information, args.model)
    inputs = {
        "prompt_token_ids": [0] * input_length,
        "additional_information": additional_information,
    }
    inputs = inputs if isinstance(inputs, list) else [inputs]
    return inputs


def run_e2e():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to StepAudioEditx (e.g., stepfun-ai/Step-Audio-EditX).",
    )
    parser.add_argument(
        "--audio-tokenizer",
        type=str,
        required=True,
        help="Path to tokenizer directory (e.g., stepfun-ai/Step-Audio-Tokenizer).",
    )
    parser.add_argument(
        "--deploy-config",
        type=str,
        default="vllm-omni/vllm_omni/deploy/step_audio_editx.yaml",
        help="Override the deploy config path. If unset, auto-loads "
        "vllm_omni/deploy/step_audio_editx.yaml based on the HF model_type.",
    )
    parser.add_argument(
        "--edit-type",
        choices=("clone", "emotion", "paralinguistic", "style", "denoise", "vad", "speed"),
        default="clone",
        help="Task type: clone, emotion, paralinguistic, style, denoise, vad, speed",
    )
    parser.add_argument("--edit-info", type=str, default=None, help="Additional information for the edit. ")
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument(
        "--ref-text",
        type=str,
        default=None,
    )
    parser.add_argument("--ref-audio", type=str, default=None, help="Path to reference audio for voice cloning.")
    parser.add_argument("--output", type=str, default=None, help="Output audio path.")
    args = parser.parse_args()
    audio_tokenizer_path = _local_path_or_none(args.audio_tokenizer)
    if audio_tokenizer_path is not None and not audio_tokenizer_path.exists():
        raise FileNotFoundError(f"{args.audio_tokenizer} does not exist!")

    if args.deploy_config is not None and not os.path.exists(args.deploy_config):
        raise FileNotFoundError(f"{args.deploy_config} does not exist!")

    print(f"Initializing StepAudioEditx E2E with model={args.model}")
    print(f"Deploy config: {args.deploy_config}")
    os.environ["STEP_AUDIO_TOKENIZER_PATH"] = args.audio_tokenizer

    omni = Omni(
        model=args.model,
        deploy_config=args.deploy_config,
        log_stats=True,
        trust_remote_code=True,
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": "./perf",
        },
    )

    inputs = _build_inputs(args)

    # Start profiling (requires VLLM_TORCH_PROFILER_DIR env var)
    if os.environ.get("VLLM_TORCH_PROFILER_DIR"):
        print("Starting profiler...")
        omni.start_profile()

    prompt_token_ids = inputs[0].get("prompt_token_ids", [])
    print(f"Prompt length: {len(prompt_token_ids)}")
    prompt_len = len(prompt_token_ids)
    max_tokens = 8192 - prompt_len

    gpt_sampling = SamplingParams(
        temperature=0.7,
        max_tokens=max_tokens,
        skip_special_tokens=False,
    )
    s2mel_sampling = SamplingParams(temperature=0.7, max_tokens=max_tokens, skip_special_tokens=False)
    sampling_params_list = [gpt_sampling, s2mel_sampling]
    logger.info(
        f"Task is {'edit: ' + args.edit_type if args.edit_type != 'clone' else 'clone'} for prompt: {args.text}"
    )
    outputs = list(omni.generate(inputs, sampling_params_list=sampling_params_list))

    if os.environ.get("VLLM_TORCH_PROFILER_DIR"):
        print("Stopping profiler...")
        profile_results = omni.stop_profile()
        print(f"Profile traces saved to: {profile_results}")

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
                    out_path = args.output if args.output else f"output_{i}.wav"
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    sf.write(out_path, audio_out.cpu().numpy().squeeze(), 24000)
                    print(f"Saved audio to {out_path}")
            else:
                print("No multimodal output found.")
        except Exception as e:
            print(f"Error inspecting output: {e}")
    omni.close()


if __name__ == "__main__":
    run_e2e()
