import argparse
import os
import os as _os_env_toggle
import random

import numpy as np
import soundfile as sf
import torch
from utils import make_omni_prompt
from vllm.sampling_params import SamplingParams

from vllm_omni.entrypoints.pipelined_omni_llm import PipelinedOmniLLM

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
        "--prompts", nargs="+", default=None, help="Input text prompts."
    )
    parser.add_argument(
        "--pt-prompts", type=str, default=None, help="Path to .pt file containing List[str] prompts (overrides --prompts when prompt_type=text)."
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
    parser.add_argument(
        "--init-sleep-seconds",
        type=int,
        default=20,
        help="Sleep seconds after starting each stage process to allow initialization (default: 20)",
    )

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
        "--output-wav", default="output.wav", help="[Deprecated] Output wav directory (use --output-dir)."
    )
    parser.add_argument(
        "--output-dir", default="outputs", help="Output directory to save text and wav files together."
    )
    parser.add_argument(
        "--thinker-hidden-states-dir",
        default="thinker_hidden_states",
        help="Path to thinker hidden states directory.",
    )
    parser.add_argument(
        "--batch-timeout",
        type=int,
        default=5,
        help="Timeout for batching in seconds (default: 5)",
    )
    parser.add_argument(
        "--init-timeout",
        type=int,
        default=300,
        help="Timeout for initializing stages in seconds (default: 300)",
    )
    parser.add_argument(
        "--shm-threshold-bytes",
        type=int,
        default=65536,
        help="Threshold for using shared memory in bytes (default: 65536)",
    )
    parser.add_argument(
        "--enable-stats",
        action="store_true",
        default=True,
        help="Enable writing detailed statistics (default: enabled)",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model_name = args.model
    # If pt-prompts provided and prompt_type is text, load and override args.prompts
    try:
        if getattr(args, "pt_prompts", None) and args.prompt_type == "text":
            loaded = torch.load(args.pt_prompts)
            if not isinstance(loaded, list) or (len(loaded) > 0 and not isinstance(loaded[0], str)):
                raise ValueError("--pt-prompts must point to a .pt containing List[str]")
            args.prompts = loaded
            print(f"[Info] Loaded {len(args.prompts)} prompts from {args.pt_prompts}")
    except Exception as e:
        print(f"[Error] Failed to load pt-prompts: {e}")
        raise

    if args.prompts is None:
        raise ValueError("No prompts provided. Use --prompts ... or --pt-prompts <file.pt> (with --prompt_type text)")
    omni_llm = PipelinedOmniLLM(
        model=model_name,
        log_stats=args.enable_stats,
        log_file="omni_llm_pipeline.log",
        init_sleep_seconds=args.init_sleep_seconds,
        batch_timeout=args.batch_timeout,
        init_timeout=args.init_timeout,
        shm_threshold_bytes=args.shm_threshold_bytes,
    )
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
        temperature=0.9,
        top_p=0.8,
        top_k=40,
        max_tokens=2048,
        seed=SEED,  # Fixed seed for sampling
        detokenize=True,
        repetition_penalty=1.05,
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

    # Determine output directory: prefer --output-dir; fallback to --output-wav
    output_dir = args.output_dir if getattr(args, "output_dir", None) else args.output_wav
    os.makedirs(output_dir, exist_ok=True)
    for stage_outputs in omni_outputs:
        if stage_outputs.final_output_type == "text":
            for output in stage_outputs.request_output:
                request_id = int(output.request_id)
                text_output = output.outputs[0].text
                # Save aligned text file per request
                prompt_text = args.prompts[request_id]
                out_txt = os.path.join(output_dir, f"{request_id:05d}.txt")
                lines = []
                lines.append("Prompt:\n")
                lines.append(str(prompt_text) + "\n")
                lines.append("vllm_text_output:\n")
                lines.append(str(text_output).strip() + "\n")
                lines.append("transformers_text_output:\n\n")
                try:
                    with open(out_txt, "w", encoding="utf-8") as f:
                        f.writelines(lines)
                except Exception as e:
                    print(f"[Warn] Failed writing text file {out_txt}: {e}")
                print(f"Request ID: {request_id}, Text saved to {out_txt}")
        elif stage_outputs.final_output_type == "audio":
            for output in stage_outputs.request_output:
                request_id = int(output.request_id)
                audio_tensor = output.multimodal_output["audio"]
                output_wav = os.path.join(output_dir, f"output_{output.request_id}.wav")
                sf.write(
                    output_wav, audio_tensor.detach().cpu().numpy(), samplerate=24000
                )
                # Save aligned text file with empty text (no text output in this stage)
                prompt_text = args.prompts[request_id]
                out_txt = os.path.join(output_dir, f"{request_id:05d}.txt")
                lines = []
                lines.append("Prompt:\n")
                lines.append(str(prompt_text) + "\n")
                lines.append("vllm_text_output:\n\n")
                lines.append("transformers_text_output:\n\n")
                try:
                    with open(out_txt, "w", encoding="utf-8") as f:
                        f.writelines(lines)
                except Exception as e:
                    print(f"[Warn] Failed writing text file {out_txt}: {e}")
                print(f"Request ID: {request_id}, Saved audio to {output_wav} and text to {out_txt}")


if __name__ == "__main__":
    main()
