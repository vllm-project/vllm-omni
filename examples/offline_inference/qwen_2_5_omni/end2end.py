import argparse
import os
import os as _os_env_toggle
import random

import numpy as np
import soundfile as sf
import torch
from utils import make_omni_prompt

from vllm.sampling_params import SamplingParams
from vllm_omni.entrypoints.omni_llm import OmniLLM

_os_env_toggle.environ["VLLM_USE_V1"] = "1"

SEED = 42
# Set all random seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Make PyTorch deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set environment variables for deterministic behavior
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="Path to merged model directory (will be created if downloading).",
    )
    parser.add_argument("--thinker-model", type=str, default=None)
    parser.add_argument("--talker-model", type=str, default=None)
    parser.add_argument("--code2wav-model", type=str, default=None)
    parser.add_argument(
        "--hf-hub-id",
        default="Qwen/Qwen2.5-Omni-7B",
        help="Hugging Face repo id to download if needed.",
    )
    parser.add_argument(
        "--hf-revision", default=None, help="Optional HF revision (branch/tag/commit)."
    )
    parser.add_argument(
        "--prompts", required=True, nargs="+", help="Input text prompts."
    )
    parser.add_argument(
        "--voice-type", default="default", help="Voice type, e.g., m02, f030, default."
    )
    parser.add_argument(
        "--code2wav-dir",
        default=None,
        help="Path to code2wav folder (contains spk_dict.pt).",
    )
    parser.add_argument(
        "--dit-ckpt", default=None, help="Path to DiT checkpoint file (e.g., dit.pt)."
    )
    parser.add_argument(
        "--bigvgan-ckpt", default=None, help="Path to BigVGAN checkpoint file."
    )
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"]
    )
    parser.add_argument("--max-model-len", type=int, default=32768)

    parser.add_argument("--thinker-only", action="store_true")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--do-wave", action="store_true")
    parser.add_argument(
        "--prompt_type",
        choices=[
            "text",
            "audio",
            "audio-long",
            "audio-long-chunks",
            "audio-long-expand-chunks",
            "image",
            "video",
            "video-frames",
            "audio-in-video",
            "audio-in-video-v2",
            "audio-multi-round",
            "badcase-vl",
            "badcase-text",
            "badcase-image-early-stop",
            "badcase-two-audios",
            "badcase-two-videos",
            "badcase-multi-round",
            "badcase-voice-type",
            "badcase-voice-type-v2",
            "badcase-audio-tower-1",
            "badcase-audio-only",
        ],
        default="text",
    )
    parser.add_argument("--use-torchvision", action="store_true")
    parser.add_argument("--tokenize", action="store_true")
    parser.add_argument(
        "--output-wav", default="output.wav", help="Output wav file path."
    )
    parser.add_argument(
        "--thinker-hidden-states-dir",
        default="thinker_hidden_states",
        help="Path to thinker hidden states directory.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model_name = args.model
    omni_llm = OmniLLM(model=model_name)
    thinker_sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic - no randomness
        top_p=1.0,  # Disable nucleus sampling
        top_k=-1,  # Disable top-k sampling
        max_tokens=2048,
        seed=SEED,  # Fixed seed for sampling
        detokenize=True,
        repetition_penalty=1.1,
    )
    talker_sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic - no randomness
        top_p=1.0,  # Disable nucleus sampling
        top_k=-1,  # Disable top-k sampling
        max_tokens=2048,
        seed=SEED,  # Fixed seed for sampling
        detokenize=True,
        repetition_penalty=1.1,
        stop_token_ids=[8294],
    )
    code2wav_sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic - no randomness
        top_p=1.0,  # Disable nucleus sampling
        top_k=-1,  # Disable top-k sampling
        max_tokens=2048,
        seed=SEED,  # Fixed seed for sampling
        detokenize=True,
        repetition_penalty=1.1,
    )

    sampling_params_list = [
        thinker_sampling_params,
        talker_sampling_params,
        code2wav_sampling_params,
    ]

    prompt = [make_omni_prompt(args, prompt) for prompt in args.prompts]
    omni_outputs = omni_llm.generate(prompt, sampling_params_list)

    os.makedirs(args.output_wav, exist_ok=True)
    for stage_outputs in omni_outputs:
        if stage_outputs.final_output_type == "text":
            for output in stage_outputs.request_output:
                request_id = output.request_id
                text_output = output.outputs[0].text
                print(f"Request ID: {request_id}, Text Output: {text_output}")
        elif stage_outputs.final_output_type == "audio":
            for output in stage_outputs.request_output:
                request_id = output.request_id
                audio_tensor = output.multimodal_output["audio"]
                output_wav = os.path.join(
                    args.output_wav, f"output_{output.request_id}.wav"
                )
                sf.write(
                    output_wav, audio_tensor.detach().cpu().numpy(), samplerate=24000
                )
                print(f"Request ID: {request_id}, Saved audio to {output_wav}")


if __name__ == "__main__":
    main()
