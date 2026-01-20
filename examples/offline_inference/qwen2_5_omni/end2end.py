# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This example shows how to use vLLM-Omni for running offline inference
with the correct prompt format on Qwen2.5-Omni
"""

import os
import time
from typing import NamedTuple

import librosa
import numpy as np
import soundfile as sf
from PIL import Image
from vllm.assets.audio import AudioAsset
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset, video_to_ndarrays
from vllm.multimodal.image import convert_image_mode
from vllm.sampling_params import SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.entrypoints.omni import Omni

SEED = 42


class QueryResult(NamedTuple):
    inputs: dict
    limit_mm_per_prompt: dict[str, int]


# NOTE: The default `max_num_seqs` and `max_model_len` may result in OOM on
# lower-end GPUs.
# Unless specified, these settings have been tested to work on a single L4.

default_system = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
    "Group, capable of perceiving auditory and visual inputs, as well as "
    "generating text and speech."
)


def get_text_query(question: str = None) -> QueryResult:
    if question is None:
        question = "Explain the system architecture for a scalable audio generation pipeline. Answer in 15 words."
    prompt = (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return QueryResult(
        inputs={
            "prompt": prompt,
        },
        limit_mm_per_prompt={},
    )


def get_mixed_modalities_query(
    video_path: str | None = None,
    image_path: str | None = None,
    audio_path: str | None = None,
    num_frames: int = 16,
    sampling_rate: int = 16000,
) -> QueryResult:
    question = "What is recited in the audio? What is the content of this image? Why is this video funny?"
    prompt = (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        "<|im_start|>user\n<|audio_bos|><|AUDIO|><|audio_eos|>"
        "<|vision_bos|><|IMAGE|><|vision_eos|>"
        "<|vision_bos|><|VIDEO|><|vision_eos|>"
        f"{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    # Load video
    if video_path:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        video_frames = video_to_ndarrays(video_path, num_frames=num_frames)
    else:
        video_frames = VideoAsset(name="baby_reading", num_frames=num_frames).np_ndarrays

    # Load image
    if image_path:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        pil_image = Image.open(image_path)
        image_data = convert_image_mode(pil_image, "RGB")
    else:
        image_data = convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")

    # Load audio
    if audio_path:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio_signal, sr = librosa.load(audio_path, sr=sampling_rate)
        audio_data = (audio_signal.astype(np.float32), sr)
    else:
        audio_data = AudioAsset("mary_had_lamb").audio_and_sample_rate

    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {
                "audio": audio_data,
                "image": image_data,
                "video": video_frames,
            },
        },
        limit_mm_per_prompt={"audio": 1, "image": 1, "video": 1},
    )


def get_use_audio_in_video_query(
    video_path: str | None = None, num_frames: int = 16, sampling_rate: int = 16000
) -> QueryResult:
    question = "Describe the content of the video, then convert what the baby say into text."
    prompt = (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        "<|im_start|>user\n<|vision_bos|><|VIDEO|><|vision_eos|><|audio_bos|><|AUDIO|><|audio_eos|>"
        f"{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    if video_path:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        video_frames = video_to_ndarrays(video_path, num_frames=num_frames)
        # Extract audio from video file
        audio_signal, sr = librosa.load(video_path, sr=sampling_rate)
        audio = (audio_signal.astype(np.float32), sr)
    else:
        asset = VideoAsset(name="baby_reading", num_frames=num_frames)
        video_frames = asset.np_ndarrays
        audio = asset.get_audio(sampling_rate=sampling_rate)

    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {
                "video": video_frames,
                "audio": audio,
            },
            "mm_processor_kwargs": {
                "use_audio_in_video": True,
            },
        },
        limit_mm_per_prompt={"audio": 1, "video": 1},
    )


def get_multi_audios_query(audio_path: str | None = None, sampling_rate: int = 16000) -> QueryResult:
    question = "Are these two audio clips the same?"
    prompt = (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        "<|im_start|>user\n<|audio_bos|><|AUDIO|><|audio_eos|>"
        "<|audio_bos|><|AUDIO|><|audio_eos|>"
        f"{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    if audio_path:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio_signal, sr = librosa.load(audio_path, sr=sampling_rate)
        audio_data = (audio_signal.astype(np.float32), sr)
        # Use the provided audio as the first audio, default as second
        audio_list = [
            audio_data,
            AudioAsset("mary_had_lamb").audio_and_sample_rate,
        ]
    else:
        audio_list = [
            AudioAsset("winning_call").audio_and_sample_rate,
            AudioAsset("mary_had_lamb").audio_and_sample_rate,
        ]

    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {
                "audio": audio_list,
            },
        },
        limit_mm_per_prompt={
            "audio": 2,
        },
    )


def get_image_query(question: str = None, image_path: str | None = None) -> QueryResult:
    if question is None:
        question = "What is the content of this image?"
    prompt = (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        "<|im_start|>user\n<|vision_bos|><|IMAGE|><|vision_eos|>"
        f"{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    if image_path:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        pil_image = Image.open(image_path)
        image_data = convert_image_mode(pil_image, "RGB")
    else:
        image_data = convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")

    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {
                "image": image_data,
            },
        },
        limit_mm_per_prompt={"image": 1},
    )


def get_video_query(question: str = None, video_path: str | None = None, num_frames: int = 16) -> QueryResult:
    if question is None:
        question = "Why is this video funny?"
    prompt = (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        "<|im_start|>user\n<|vision_bos|><|VIDEO|><|vision_eos|>"
        f"{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    if video_path:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        video_frames = video_to_ndarrays(video_path, num_frames=num_frames)
    else:
        video_frames = VideoAsset(name="baby_reading", num_frames=num_frames).np_ndarrays

    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {
                "video": video_frames,
            },
        },
        limit_mm_per_prompt={"video": 1},
    )


def get_audio_query(question: str = None, audio_path: str | None = None, sampling_rate: int = 16000) -> QueryResult:
    if question is None:
        question = "What is the content of this audio?"
    prompt = (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        "<|im_start|>user\n<|audio_bos|><|AUDIO|><|audio_eos|>"
        f"{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    if audio_path:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio_signal, sr = librosa.load(audio_path, sr=sampling_rate)
        audio_data = (audio_signal.astype(np.float32), sr)
    else:
        audio_data = AudioAsset("mary_had_lamb").audio_and_sample_rate

    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {
                "audio": audio_data,
            },
        },
        limit_mm_per_prompt={"audio": 1},
    )


query_map = {
    "use_mixed_modalities": get_mixed_modalities_query,
    "use_audio_in_video": get_use_audio_in_video_query,
    "use_multi_audios": get_multi_audios_query,
    "use_image": get_image_query,
    "use_video": get_video_query,
    "use_audio": get_audio_query,
    "text": get_text_query,
}


def main(args):
    model_name = "Qwen/Qwen2.5-Omni-7B"

    # Get paths from args
    video_path = getattr(args, "video_path", None)
    image_path = getattr(args, "image_path", None)
    audio_path = getattr(args, "audio_path", None)
    num_frames = getattr(args, "num_frames", 16)
    sampling_rate = getattr(args, "sampling_rate", 16000)

    # Get the query function and call it with appropriate parameters
    query_func = query_map[args.query_type]
    if args.query_type == "mixed_modalities":
        query_result = query_func(
            video_path=video_path,
            image_path=image_path,
            audio_path=audio_path,
            num_frames=num_frames,
            sampling_rate=sampling_rate,
        )
    elif args.query_type == "use_audio_in_video":
        query_result = query_func(video_path=video_path, num_frames=num_frames, sampling_rate=sampling_rate)
    elif args.query_type == "multi_audios":
        query_result = query_func(audio_path=audio_path, sampling_rate=sampling_rate)
    elif args.query_type == "use_image":
        query_result = query_func(image_path=image_path)
    elif args.query_type == "use_video":
        query_result = query_func(video_path=video_path, num_frames=num_frames)
    elif args.query_type == "use_audio":
        query_result = query_func(audio_path=audio_path, sampling_rate=sampling_rate)
    else:
        query_result = query_func()
    omni_llm = Omni(
        model=model_name,
        log_stats=args.enable_stats,
        stage_init_timeout=args.stage_init_timeout,
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

    if args.txt_prompts is None:
        prompts = [query_result.inputs for _ in range(args.num_prompts)]
    else:
        assert args.query_type == "text", "txt-prompts is only supported for text query type"
        with open(args.txt_prompts, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines()]
            prompts = [get_text_query(ln).inputs for ln in lines if ln != ""]
            print(f"[Info] Loaded {len(prompts)} prompts from {args.txt_prompts}")

    if args.modalities is not None:
        output_modalities = args.modalities.split(",")
        for i, prompt in enumerate(prompts):
            prompt["modalities"] = output_modalities

    profiler_enabled = bool(os.getenv("VLLM_TORCH_PROFILER_DIR"))
    if profiler_enabled:
        omni_llm.start_profile(stages=[0])
    omni_generator = omni_llm.generate(prompts, sampling_params_list, py_generator=args.py_generator)

    # Determine output directory: prefer --output-dir; fallback to --output-wav
    output_dir = args.output_dir if getattr(args, "output_dir", None) else args.output_wav
    os.makedirs(output_dir, exist_ok=True)

    total_requests = len(prompts)
    processed_count = 0
    for stage_outputs in omni_generator:
        if stage_outputs.final_output_type == "text":
            for output in stage_outputs.request_output:
                request_id = output.request_id
                text_output = output.outputs[0].text
                # Save aligned text file per request
                prompt_text = output.prompt
                out_txt = os.path.join(output_dir, f"{request_id}.txt")
                lines = []
                lines.append("Prompt:\n")
                lines.append(str(prompt_text) + "\n")
                lines.append("vllm_text_output:\n")
                lines.append(str(text_output).strip() + "\n")
                try:
                    with open(out_txt, "w", encoding="utf-8") as f:
                        f.writelines(lines)
                except Exception as e:
                    print(f"[Warn] Failed writing text file {out_txt}: {e}")
                print(f"Request ID: {request_id}, Text saved to {out_txt}")
        elif stage_outputs.final_output_type == "audio":
            for output in stage_outputs.request_output:
                request_id = output.request_id
                audio_tensor = output.multimodal_output["audio"]
                output_wav = os.path.join(output_dir, f"output_{request_id}.wav")
                sf.write(output_wav, audio_tensor.detach().cpu().numpy(), samplerate=24000)
                print(f"Request ID: {request_id}, Saved audio to {output_wav}")

        processed_count += len(stage_outputs.request_output)
        if profiler_enabled and processed_count >= total_requests:
            print(f"[Info] Processed {processed_count}/{total_requests}. Stopping profiler inside active loop...")
            # Stop the profiler while workers are still alive
            omni_llm.stop_profile()

            print("[Info] Waiting 30s for workers to write massive trace files to disk...")
            time.sleep(30)
            print("[Info] Trace export wait finished.")

    omni_llm.close()


def parse_args():
    parser = FlexibleArgumentParser(description="Demo on using vLLM for offline inference with audio language models")
    parser.add_argument(
        "--query-type",
        "-q",
        type=str,
        default="use_mixed_modalities",
        choices=query_map.keys(),
        help="Query type.",
    )
    parser.add_argument(
        "--enable-stats",
        action="store_true",
        default=False,
        help="Enable writing detailed statistics (default: disabled)",
    )
    parser.add_argument(
        "--stage-init-timeout",
        type=int,
        default=300,
        help="Timeout for initializing a single stage in seconds (default: 300)",
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
        "--output-wav",
        default="output_audio",
        help="[Deprecated] Output wav directory (use --output-dir).",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1,
        help="Number of prompts to generate.",
    )
    parser.add_argument(
        "--txt-prompts",
        type=str,
        default=None,
        help="Path to a .txt file with one prompt per line (preferred).",
    )
    parser.add_argument(
        "--video-path",
        "-v",
        type=str,
        default=None,
        help="Path to local video file. If not provided, uses default video asset.",
    )
    parser.add_argument(
        "--image-path",
        "-i",
        type=str,
        default=None,
        help="Path to local image file. If not provided, uses default image asset.",
    )
    parser.add_argument(
        "--audio-path",
        "-a",
        type=str,
        default=None,
        help="Path to local audio file. If not provided, uses default audio asset.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Number of frames to extract from video (default: 16).",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=16000,
        help="Sampling rate for audio loading (default: 16000).",
    )
    parser.add_argument(
        "--worker-backend", type=str, default="multi_process", choices=["multi_process", "ray"], help="backend"
    )
    parser.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help="Address of the Ray cluster.",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default=None,
        help="Modalities to use for the prompts.",
    )
    parser.add_argument(
        "--py-generator",
        action="store_true",
        default=False,
        help="Use py_generator mode. The returned type of Omni.generate() is a Python Generator object.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
