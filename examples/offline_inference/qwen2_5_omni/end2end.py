# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Offline E2E example for vLLM-Omni (Qwen2.5-Omni).
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


def _req_index(rid: str) -> int:
    """Extract the integer prefix from request_id like '7_<uuid>' -> 7."""
    try:
        return int(str(rid).split("_", 1)[0])
    except Exception:
        return 10**18


def _print_run_header(model_name: str, args, num_prompts: int) -> None:
    print("=" * 72)
    print("[vLLM-Omni Offline Inference]")
    print(f"Model      : {model_name}")
    print(f"Query type : {args.query_type}")
    print(f"Backend    : {args.worker_backend}")
    if args.worker_backend == "ray":
        print(f"Ray addr   : {args.ray_address}")
    print(f"Prompts    : {num_prompts}")
    if getattr(args, "txt_prompts", None):
        print(f"Prompt file: {args.txt_prompts}")
    if getattr(args, "modalities", None):
        print(f"Modalities : {args.modalities}")
    print(f"Stats      : {'ON' if args.enable_stats else 'OFF'}")
    print(f"Output dir : {get_output_dir(args)}")
    print("=" * 72)


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
    "mixed_modalities": get_mixed_modalities_query,
    "use_audio_in_video": get_use_audio_in_video_query,
    "multi_audios": get_multi_audios_query,
    "use_image": get_image_query,
    "use_video": get_video_query,
    "use_audio": get_audio_query,
    "text": get_text_query,
}


def get_output_dir(args) -> str:
    out = getattr(args, "output_dir", None)
    if out:
        return out
    return getattr(args, "output_wav", "output_audio")


def _expand_sampling_params_list(
    sampling_params_list: list[SamplingParams],
    num_stages: int,
) -> list[SamplingParams]:
    """
    Make end2end robust to stage count changes without touching orchestrator code.

    Rules:
    - If user provides exactly num_stages -> use as-is.
    - If user provides 1 -> repeat it for all stages.
    - Otherwise -> raise.
    """
    if num_stages <= 0:
        return sampling_params_list
    if len(sampling_params_list) == num_stages:
        return sampling_params_list
    if len(sampling_params_list) == 1:
        return [sampling_params_list[0] for _ in range(num_stages)]
    raise ValueError(
        f"sampling_params_list length mismatch: got {len(sampling_params_list)}, expected {num_stages} "
        f"(or 1 to auto-repeat)."
    )


def main(args):
    model_name = "Qwen/Qwen2.5-Omni-3B"

    # --- STEP 0: Pre-flight checks --------------------------------------------------
    if args.txt_prompts:
        assert args.query_type == "text", "--txt-prompts only supported for text queries"
        if not os.path.isfile(args.txt_prompts):
            raise FileNotFoundError(f"Prompt file not found: {args.txt_prompts}")
        with open(args.txt_prompts, encoding="utf-8") as f:
            raw_lines = f.readlines()
        prompts = [ln.strip() for ln in raw_lines if ln.strip()]
        print(f"[Info] Loaded {len(prompts)} non-empty prompts from {args.txt_prompts}")
        if not prompts:
            raise ValueError("No valid prompts found in file — all lines empty/whitespace.")
    else:
        # Generate synthetic prompts
        prompts = [None] * args.num_prompts

    # --- STEP 1: Build query inputs ------------------------------------------------
    query_func = query_map[args.query_type]

    request_inputs = []
    for i, p in enumerate(prompts):
        try:
            if args.query_type == "text":
                qr = query_func(question=p)
            elif args.query_type == "mixed_modalities":
                qr = query_func(
                    video_path=args.video_path,
                    image_path=args.image_path,
                    audio_path=args.audio_path,
                    num_frames=args.num_frames,
                    sampling_rate=args.sampling_rate,
                )
            elif args.query_type == "use_audio_in_video":
                qr = query_func(
                    video_path=args.video_path, num_frames=args.num_frames, sampling_rate=args.sampling_rate
                )
            elif args.query_type == "multi_audios":
                qr = query_func(audio_path=args.audio_path, sampling_rate=args.sampling_rate)
            elif args.query_type == "use_image":
                qr = query_func(image_path=args.image_path)
            elif args.query_type == "use_video":
                qr = query_func(video_path=args.video_path, num_frames=args.num_frames)
            elif args.query_type == "use_audio":
                qr = query_func(audio_path=args.audio_path, sampling_rate=args.sampling_rate)
            else:
                qr = query_func()
        except Exception as e:
            raise RuntimeError(f"Failed to build prompt #{i}: {e}") from e

        # Inject modalities if specified
        if args.modalities:
            qr.inputs["modalities"] = args.modalities.split(",")

        request_inputs.append(qr.inputs)

    # --- STEP 2: Initialize Omni ----------------------------------------------------
    omni_llm = Omni(
        model=model_name,
        log_stats=args.enable_stats,
        log_file=("omni_llm_pipeline.log" if args.enable_stats else None),
        init_sleep_seconds=args.init_sleep_seconds,
        batch_timeout=args.batch_timeout,
        init_timeout=args.init_timeout,
        shm_threshold_bytes=args.shm_threshold_bytes,
        worker_backend=args.worker_backend,
        ray_address=args.ray_address,
    )

    # Auto-detect stage count safely
    num_stages = len(getattr(omni_llm, "stage_list", []))
    if num_stages == 0:
        raise RuntimeError("Failed to detect stage count from Omni instance")

    # --- STEP 3: Build sampling params list -----------------------------------------
    # Base params — match your original intent
    thinker = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=1024,  # ← safer for single GPU
        seed=SEED,
        detokenize=True,
        repetition_penalty=1.1,
    )
    talker = SamplingParams(
        temperature=0.9,
        top_p=0.8,
        top_k=40,
        max_tokens=1024,
        seed=SEED,
        detokenize=True,
        repetition_penalty=1.05,
        stop_token_ids=[8294],
    )
    code2wav = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=1024,
        seed=SEED,
        detokenize=True,
        repetition_penalty=1.1,
    )

    # Map stage_id → expected model_stage (from Omni config introspection)
    stage_stages = []
    try:
        for stage in omni_llm.stage_list:
            eng_args = getattr(stage, "engine_args", {})
            ms = getattr(eng_args, "model_stage", None) or eng_args.get("model_stage", "unknown")
            stage_stages.append(ms)
    except Exception:
        stage_stages = ["thinker"] * num_stages  # fallback

    # Auto-assign based on model_stage; fallback to thinker if unknown
    auto_sampling_params = []
    for stage_id, model_stage in enumerate(stage_stages):
        if model_stage == "talker":
            sp = talker
        elif model_stage == "code2wav":
            sp = code2wav
        else:  # thinker, or unknown
            sp = thinker
        auto_sampling_params.append(sp)
        print(
            f"[Info] Stage-{stage_id} (model_stage={model_stage}) → sampling_params: max_tokens={sp.max_tokens}, temp={sp.temperature}"
        )

    # --- STEP 4: Run generation -----------------------------------------------------
    _print_run_header(model_name, args, len(request_inputs))

    t0 = time.time()
    try:
        omni_outputs = omni_llm.generate(request_inputs, auto_sampling_params)
    except Exception as e:
        print(f"[CRITICAL] Generation failed: {e}")
        import traceback

        traceback.print_exc()
        raise
    t1 = time.time()

    print(f"[Done] Generated {len(omni_outputs)} stage outputs in {(t1 - t0):.2f}s")

    # --- STEP 5: Unified output saving ----------------------------------------------
    output_dir = get_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)

    saved = 0
    for stage_output in omni_outputs:
        final_type = getattr(stage_output, "final_output_type", None)
        if not final_type:
            print("[Skip] Stage output has no final_output_type — skipping")
            continue

        outputs = list(stage_output.request_output)
        if not outputs:
            print(f"[Skip] No request outputs in stage output (type={final_type})")
            continue

        # Sort by prompt index (0_, 1_, 2_, ...)
        outputs_sorted = sorted(outputs, key=lambda o: _req_index(o.request_id))

        for out in outputs_sorted:
            rid = out.request_id
            try:
                if final_type == "text":
                    text = getattr(out.outputs[0], "text", "").strip() if out.outputs else ""
                    path = os.path.join(output_dir, f"{rid}.txt")
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(f"Prompt:\n{out.prompt}\n\nvLLM Output:\n{text}\n")
                    print(f"[Save] Text → {path}")
                    saved += 1

                elif final_type == "audio":
                    audio = getattr(out, "multimodal_output", {}).get("audio")
                    if audio is None:
                        print(f"[Warn] No audio in multimodal_output for {rid}")
                        continue
                    path = os.path.join(output_dir, f"{rid}.wav")
                    sf.write(path, audio.detach().cpu().numpy(), samplerate=24000)
                    print(f"[Save] Audio → {path}")
                    saved += 1

                else:
                    print(f"[Warn] Skipping unsupported final_output_type: {final_type} for {rid}")

            except Exception as e:
                print(f"[Error] Failed to save {rid} ({final_type}): {e}")
                import traceback

                traceback.print_exc()

    print(f"[Summary] Saved {saved} artifacts into: {output_dir}")


def parse_args():
    parser = FlexibleArgumentParser(description="Demo on using vLLM-Omni for offline inference (Qwen2.5-Omni)")
    parser.add_argument(
        "--query-type",
        "-q",
        type=str,
        default="mixed_modalities",
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
        "--init-sleep-seconds",
        type=int,
        default=20,
        help="Sleep seconds after starting each stage process to allow initialization (default: 20)",
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
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for artifacts (preferred).",
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
        help="Number of prompts to generate (ignored if --txt-prompts is used).",
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
        "--worker-backend",
        type=str,
        default="multi_process",
        choices=["multi_process", "ray"],
        help="Worker backend.",
    )
    parser.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help="Address of the Ray cluster (only if --worker-backend=ray).",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default=None,
        help="Modalities to use for the prompts (comma-separated).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
