# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end inference example for Fun-Audio-Chat-8B.

Supports two modes:
- S2T (Speech-to-Text): Audio understanding and transcription
- S2S (Speech-to-Speech): Full voice conversation pipeline

Usage:
    # S2T mode (default)
    python end2end.py --mode s2t --audio-path /path/to/audio.wav

    # S2S mode
    python end2end.py --mode s2s --audio-path /path/to/audio.wav --output-dir output
"""

import json
import os
from pathlib import Path
from typing import NamedTuple

import librosa
import numpy as np
import soundfile as sf
from vllm import SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.entrypoints.omni import Omni

SEED = 42

# Default system prompts for Fun-Audio-Chat
# S2T mode: generate text only
DEFAULT_S2T_SYSTEM = "You are asked to generate text tokens."
# S2S mode: generate both text and speech tokens
DEFAULT_S2S_SYSTEM = "You are asked to generate both text and speech tokens at the same time."

# Audio template with bos/eos markers
AUDIO_TEMPLATE = "<|audio_bos|><|AUDIO|><|audio_eos|>"


class QueryResult(NamedTuple):
    inputs: dict
    limit_mm_per_prompt: dict[str, int]


def get_audio_query(
    audio_path: str | None = None,
    question: str | None = None,
    sampling_rate: int = 16000,
    mode: str = "s2t",
) -> QueryResult:
    """Build audio query for Fun-Audio-Chat."""
    # Select system prompt based on mode
    system_prompt = DEFAULT_S2T_SYSTEM if mode == "s2t" else DEFAULT_S2S_SYSTEM

    # Build user content with audio template
    if question:
        user_content = f"{AUDIO_TEMPLATE}\n{question}"
    else:
        user_content = AUDIO_TEMPLATE

    # Fun-Audio-Chat uses <|audio_bos|><|AUDIO|><|audio_eos|> for audio placeholder
    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    if audio_path and os.path.exists(audio_path):
        audio_signal, _ = librosa.load(audio_path, sr=sampling_rate)
        # Pass only the waveform; avoid resampler complaining about target_sr
        audio_data = audio_signal.astype(np.float32)
        print(
            f"[Debug] Loaded audio from file: shape={audio_data.shape}, duration={len(audio_data) / sampling_rate:.2f}s"
        )
    else:
        # Try to use Fun-Audio-Chat's test audio first
        fun_audio_chat_test = Path(__file__).parent.parent.parent.parent / "Fun-Audio-Chat/examples/ck7vv9ag.wav"
        if fun_audio_chat_test.exists():
            audio_signal, orig_sr = librosa.load(str(fun_audio_chat_test), sr=None)
            print(f"[Debug] Using Fun-Audio-Chat test audio: {fun_audio_chat_test}")
            print(f"[Debug] Original: shape={audio_signal.shape}, sr={orig_sr}")
            if orig_sr != sampling_rate:
                audio_signal = librosa.resample(audio_signal, orig_sr=orig_sr, target_sr=sampling_rate)
                print(f"[Debug] Resampled to {sampling_rate}Hz: shape={audio_signal.shape}")
            audio_data = audio_signal.astype(np.float32)
        else:
            # Fall back to vLLM's built-in test audio
            audio_signal, orig_sr = AudioAsset("mary_had_lamb").audio_and_sample_rate
            print(f"[Debug] AudioAsset original: shape={audio_signal.shape}, sr={orig_sr}")
            # Resample to target sampling rate if needed
            if orig_sr != sampling_rate:
                audio_signal = librosa.resample(audio_signal, orig_sr=orig_sr, target_sr=sampling_rate)
                print(f"[Debug] Resampled to {sampling_rate}Hz: shape={audio_signal.shape}")
            audio_data = audio_signal.astype(np.float32)
        print(f"[Debug] Final audio: shape={audio_data.shape}, duration={len(audio_data) / sampling_rate:.2f}s")

    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {
                "audio": audio_data,
            },
        },
        limit_mm_per_prompt={"audio": 1},
    )


def get_text_query(question: str | None = None, mode: str = "s2t") -> QueryResult:
    """Build text-only query."""
    if question is None:
        question = "Hello! How can I help you today?"

    system_prompt = DEFAULT_S2T_SYSTEM if mode == "s2t" else DEFAULT_S2S_SYSTEM

    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    return QueryResult(
        inputs={"prompt": prompt},
        limit_mm_per_prompt={},
    )


def get_stage_configs_path(mode: str) -> str:
    """Get the appropriate stage config file path."""
    config_dir = Path(__file__).parent.parent.parent.parent / "vllm_omni/model_executor/stage_configs"

    if mode == "s2s":
        config_path = config_dir / "fun_audio_chat_s2s.yaml"
    else:
        config_path = config_dir / "fun_audio_chat_s2t.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Stage config not found: {config_path}")

    return str(config_path)


def main(args):
    model_name = args.model_path

    # Get stage configs
    if args.stage_configs:
        stage_configs_path = args.stage_configs
    else:
        stage_configs_path = get_stage_configs_path(args.mode)

    print(f"[Info] Mode: {args.mode.upper()}")
    print(f"[Info] Model: {model_name}")
    print(f"[Info] Stage configs: {stage_configs_path}")

    # Build query
    if args.audio_path:
        query_result = get_audio_query(
            audio_path=args.audio_path,
            question=args.question,
            sampling_rate=args.sampling_rate,
            mode=args.mode,
        )
    else:
        # Use default audio
        query_result = get_audio_query(
            question=args.question,
            sampling_rate=args.sampling_rate,
            mode=args.mode,
        )

    # Initialize Omni engine
    log_file = None
    if args.enable_stats:
        os.makedirs(args.log_dir, exist_ok=True)
        log_file = os.path.join(args.log_dir, f"fun_audio_chat_{args.mode}")

    omni_llm = Omni(
        model=model_name,
        stage_configs_path=stage_configs_path,
        log_file=log_file,
        log_stats=args.enable_stats,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # Sampling parameters for main model
    main_sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.9,
        top_k=50,
        max_tokens=args.max_tokens,
        repetition_penalty=1.05,
        seed=args.seed,
        detokenize=True,
    )

    if args.mode == "s2s":
        # S2S mode: 3 stages
        # CRQ decoder sampling params
        crq_sampling_params = SamplingParams(
            temperature=0.9,
            top_k=50,
            max_tokens=4096,
            seed=args.seed,
            detokenize=False,
            repetition_penalty=1.05,
        )

        # CosyVoice sampling params
        cosyvoice_sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=4096 * 16,
            seed=args.seed,
            detokenize=True,
        )

        sampling_params_list = [
            main_sampling_params,
            crq_sampling_params,
            cosyvoice_sampling_params,
        ]
    else:
        # S2T mode: 1 stage
        sampling_params_list = [main_sampling_params]

    # Build prompts
    prompts = [query_result.inputs for _ in range(args.num_prompts)]

    print(f"[Info] Running inference with {args.num_prompts} prompt(s)...")

    # Run inference
    omni_outputs = omni_llm.generate(prompts, sampling_params_list)

    # Save outputs
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    for stage_outputs in omni_outputs:
        if stage_outputs.final_output_type == "text":
            for output in stage_outputs.request_output:
                request_id = output.request_id
                text_output = output.outputs[0].text
                prompt_text = output.prompt

                # Save text file
                out_txt = os.path.join(output_dir, f"{request_id}.txt")
                with open(out_txt, "w", encoding="utf-8") as f:
                    f.write("Prompt:\n")
                    f.write(str(prompt_text) + "\n\n")
                    f.write("Response:\n")
                    f.write(str(text_output).strip() + "\n")

                # Save JSON metadata
                out_json = os.path.join(output_dir, f"{request_id}.json")
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "request_id": request_id,
                            "mode": args.mode,
                            "prompt": str(prompt_text),
                            "response": str(text_output).strip(),
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )

                print(f"[Info] Request {request_id}: Text saved to {out_txt}")

        elif stage_outputs.final_output_type == "audio":
            for output in stage_outputs.request_output:
                request_id = output.request_id
                audio_tensor = output.multimodal_output.get("audio")

                if audio_tensor is not None:
                    output_wav = os.path.join(output_dir, f"{request_id}.wav")

                    # Convert to numpy array
                    audio_numpy = audio_tensor.float().detach().cpu().numpy()

                    # Ensure audio is 1D
                    if audio_numpy.ndim > 1:
                        audio_numpy = audio_numpy.flatten()

                    # Save audio file (24kHz output from CosyVoice)
                    sf.write(output_wav, audio_numpy, samplerate=24000, format="WAV")
                    print(f"[Info] Request {request_id}: Audio saved to {output_wav}")
                else:
                    print(f"[Warn] Request {request_id}: No audio output")

    print(f"[Info] All outputs saved to {output_dir}/")


def parse_args():
    parser = FlexibleArgumentParser(description="End-to-end inference for Fun-Audio-Chat-8B")

    parser.add_argument(
        "--mode",
        type=str,
        default="s2t",
        choices=["s2t", "s2s"],
        help="Run mode: s2t (speech-to-text) or s2s (speech-to-speech). Default: s2t",
    )
    parser.add_argument(
        "--audio-path",
        "-a",
        type=str,
        default=None,
        help="Path to input audio file. If not provided, uses built-in test audio.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory. Default: output",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="FunAudioLLM/Fun-Audio-Chat-8B",
        help="Fun-Audio-Chat model path. Default: FunAudioLLM/Fun-Audio-Chat-8B",
    )
    parser.add_argument(
        "--stage-configs",
        type=str,
        default=None,
        help="Custom stage config file path. Default: auto-select based on mode",
    )
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        default=None,
        help="Question/prompt to use with audio. Default: auto-generated",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=16000,
        help="Input audio sampling rate. Default: 16000",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature. Default: 0.7",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate. Default: 2048",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed. Default: {SEED}",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1,
        help="Number of prompts to run. Default: 1",
    )
    parser.add_argument(
        "--enable-stats",
        action="store_true",
        default=False,
        help="Enable detailed statistics logging. Default: False",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Log directory. Default: logs",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum model length for KV cache. Default: 8192",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization. Default: 0.9",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
