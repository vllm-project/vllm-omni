# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline AURA Omni example.

The native AURA pipeline is:

    Qwen3-ASR -> AURA/Qwen3-VL -> Qwen3-TTS Talker -> Code2Wav

Stage 0 consumes audio and emits a transcript. The stage input processor then
combines that transcript with the original visual payload for AURA.
"""

from __future__ import annotations

import os
from typing import NamedTuple

import numpy as np
import soundfile as sf
import vllm
from vllm import SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.assets.video import VideoAsset, video_to_ndarrays
from vllm.multimodal.media.audio import load_audio
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.engine.arg_utils import nullify_stage_engine_defaults
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.model_executor.stage_input_processors.aura_omni import (
    DEFAULT_QWEN3_TTS_REF_TEXT,
    default_qwen3_tts_ref_audio_path,
)

SEED = 42
DEFAULT_MODEL = "aurateam/AURA"
DEFAULT_DEPLOY_CONFIG = "vllm_omni/deploy/aura_omni.yaml"


class QueryResult(NamedTuple):
    inputs: dict
    limit_mm_per_prompt: dict[str, int]


def _load_audio(audio_path: str | None, sampling_rate: int) -> tuple[np.ndarray, int]:
    if audio_path:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio_signal, sr = load_audio(audio_path, sr=sampling_rate)
        return audio_signal.astype(np.float32), int(sr)
    audio_signal, sr = AudioAsset("mary_had_lamb").audio_and_sample_rate
    return audio_signal.astype(np.float32), int(sr)


def _load_video(video_path: str | None, num_frames: int) -> list[np.ndarray]:
    if video_path:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        return video_to_ndarrays(video_path, num_frames=num_frames)
    return VideoAsset(name="baby_reading", num_frames=num_frames).np_ndarrays


def build_query(args) -> QueryResult:
    prompt = (
        "<|im_start|>user\n"
        "<|audio_start|><|audio_pad|><|audio_end|>"
        "Please transcribe this speech and use the video context to decide whether a reply is needed.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    audio_data = _load_audio(args.audio_path, args.sampling_rate)
    video_frames = _load_video(args.video_path, args.num_frames)
    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {
                "audio": audio_data,
                "video": video_frames,
            },
            "additional_information": {
                "aura_system_prompt": args.aura_system_prompt,
                "tts_task_type": args.tts_task_type,
                "tts_language": args.tts_language,
                "tts_speaker": args.tts_speaker,
                "tts_instruct": args.tts_instruct,
                "tts_ref_audio": args.tts_ref_audio,
                "tts_ref_text": args.tts_ref_text,
                "tts_x_vector_only_mode": args.tts_x_vector_only_mode,
                "tts_pass_token_ids": args.tts_pass_token_ids,
            },
        },
        limit_mm_per_prompt={"audio": 1, "video": 1},
    )


def build_sampling_params() -> list[SamplingParams]:
    asr_sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=256,
        detokenize=True,
        seed=SEED,
    )
    aura_sampling = SamplingParams(
        temperature=0.5,
        top_p=1.0,
        top_k=-1,
        max_tokens=256,
        detokenize=True,
        repetition_penalty=1.0,
        seed=SEED,
    )
    talker_sampling = SamplingParams(
        temperature=0.9,
        top_k=50,
        max_tokens=4096,
        detokenize=False,
        repetition_penalty=1.05,
        stop_token_ids=[2150],
        seed=SEED,
    )
    code2wav_sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=65536,
        detokenize=True,
        repetition_penalty=1.0,
        seed=SEED,
    )
    return [asr_sampling, aura_sampling, talker_sampling, code2wav_sampling]


def save_stage_output(stage_outputs, output_dir: str) -> None:
    output = stage_outputs.request_output
    request_id = output.request_id
    if stage_outputs.final_output_type == "text":
        text_output = output.outputs[0].text
        out_txt = os.path.join(output_dir, f"{request_id}.txt")
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(str(text_output).strip() + "\n")
        print(f"Request ID: {request_id}, text saved to {out_txt}")
        print(text_output)
        return

    if stage_outputs.final_output_type == "audio":
        mm = output.outputs[0].multimodal_output
        if not mm or "audio" not in mm:
            print(f"Request ID: {request_id}, no audio output")
            return
        audio = mm["audio"]
        if isinstance(audio, list):
            import torch

            audio = torch.cat([(item if hasattr(item, "detach") else torch.tensor(item)).flatten() for item in audio])
        sample_rate = int(mm.get("sr", 24000))
        audio_numpy = audio.float().detach().cpu().numpy().flatten()
        out_wav = os.path.join(output_dir, f"{request_id}.wav")
        sf.write(out_wav, audio_numpy, samplerate=sample_rate, format="WAV")
        print(f"Request ID: {request_id}, audio saved to {out_wav}")


def main(args) -> None:
    print("=" * 20, "\n", f"vllm version: {vllm.__version__}", "\n", "=" * 20)
    os.makedirs(args.output_dir, exist_ok=True)

    query = build_query(args)
    prompts = [query.inputs for _ in range(args.num_prompts)]
    if args.modalities:
        modalities = [item.strip() for item in args.modalities.split(",") if item.strip()]
        for prompt in prompts:
            prompt["modalities"] = modalities

    omni_kwargs = vars(args).copy()
    omni_kwargs["model"] = args.model
    omni = Omni(**omni_kwargs)
    sampling_params_list = build_sampling_params()[: omni.num_stages]

    try:
        for stage_outputs in omni.generate(prompts, sampling_params_list):
            save_stage_output(stage_outputs, args.output_dir)
    finally:
        omni.close()


def parse_args():
    parser = FlexibleArgumentParser(description="AURA Omni offline inference example")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Top-level model name/path.")
    parser.add_argument("--deploy-config", default=DEFAULT_DEPLOY_CONFIG, help="AURA Omni deploy config.")
    parser.add_argument("--audio-path", "-a", default=None, help="Local audio file. Uses a built-in asset by default.")
    parser.add_argument("--video-path", "-v", default=None, help="Local video file. Uses a built-in asset by default.")
    parser.add_argument("--num-frames", type=int, default=16, help="Number of video frames to sample.")
    parser.add_argument("--sampling-rate", type=int, default=16000, help="Audio sampling rate for ASR input.")
    parser.add_argument("--num-prompts", type=int, default=1, help="Number of repeated prompts.")
    parser.add_argument("--modalities", default="text,audio", help="Comma-separated final modalities.")
    parser.add_argument("--output-dir", default="output_aura_omni", help="Directory for generated files.")
    parser.add_argument(
        "--aura-system-prompt",
        default=(
            "You are receiving a live video stream where the final frame is the present moment. "
            "Respond only when a response is needed. Otherwise output '<|silent|>'. Respond in Chinese."
        ),
        help="System prompt for the AURA stage.",
    )
    parser.add_argument("--tts-task-type", default="Base", choices=["Base", "CustomVoice"], help="Qwen3-TTS task.")
    parser.add_argument("--tts-language", default="Chinese", help="Qwen3-TTS language.")
    parser.add_argument("--tts-speaker", default="Vivian", help="Qwen3-TTS speaker.")
    parser.add_argument("--tts-instruct", default="", help="Optional Qwen3-TTS style instruction.")
    parser.add_argument(
        "--tts-ref-audio",
        default=default_qwen3_tts_ref_audio_path(),
        help="Base-mode reference audio path/URL.",
    )
    parser.add_argument(
        "--tts-ref-text",
        default=DEFAULT_QWEN3_TTS_REF_TEXT,
        help="Base-mode reference audio transcript.",
    )
    parser.add_argument(
        "--tts-x-vector-only-mode",
        action="store_true",
        help="Use speaker embedding only for Base mode (disable ICL ref_text conditioning).",
    )
    parser.add_argument(
        "--tts-pass-token-ids",
        action="store_true",
        help="Pass AURA-generated assistant token ids directly to Qwen3-TTS. Defaults to sending text.",
    )
    parser.add_argument("--init-timeout", type=int, default=2000, help="Overall initialization timeout.")
    parser.add_argument("--stage-init-timeout", type=int, default=2000, help="Per-stage initialization timeout.")
    parser.add_argument("--log-stats", action="store_true", default=False, help="Enable stats logging.")
    nullify_stage_engine_defaults(parser)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
