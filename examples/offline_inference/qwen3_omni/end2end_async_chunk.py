# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline inference with async_chunk enabled via AsyncOmni.

This script uses AsyncOmni (the async orchestrator) to run offline inference
with async_chunk semantics: downstream stages (Talker, Code2Wav) start
*before* upstream stages finish, consuming chunks as they arrive via
the in-worker OmniChunkTransferAdapter / connector.

Compared to the synchronous ``end2end.py`` (which uses ``Omni``), this
entry point achieves true stage-level concurrency -- stage-1/2 are
actively processing while stage-0 is still generating.

Usage
-----
    python end2end_async_chunk.py --query-type use_audio \
        --deploy-config <path-to-deploy-config-yaml>

See ``--help`` for all options.
"""

import asyncio
import logging
import os
import time
import uuid
from typing import NamedTuple

import numpy as np
import soundfile as sf
import torch

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from PIL import Image
from vllm import SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset, video_to_ndarrays
from vllm.multimodal.image import convert_image_mode
from vllm.multimodal.media.audio import load_audio
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.entrypoints.async_omni import AsyncOmni

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Query builders (reuse the patterns from end2end.py)
# ---------------------------------------------------------------------------

default_system = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
    "Group, capable of perceiving auditory and visual inputs, as well as "
    "generating text and speech."
)


class QueryResult(NamedTuple):
    inputs: dict
    limit_mm_per_prompt: dict[str, int]


def get_text_query(question: str = None) -> QueryResult:
    if question is None:
        question = "Explain the system architecture for a scalable audio generation pipeline. Answer in 15 words."
    prompt = (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return QueryResult(inputs={"prompt": prompt}, limit_mm_per_prompt={})


def get_audio_query(
    question: str = None,
    audio_path: str | None = None,
    sampling_rate: int = 16000,
) -> QueryResult:
    if question is None:
        question = "What is the content of this audio?"
    prompt = (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    if audio_path:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio_signal, sr = load_audio(audio_path, sr=sampling_rate)
        audio_data = (audio_signal.astype(np.float32), sr)
    else:
        audio_data = AudioAsset("mary_had_lamb").audio_and_sample_rate
    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {"audio": audio_data},
        },
        limit_mm_per_prompt={"audio": 1},
    )


def get_image_query(question: str = None, image_path: str | None = None) -> QueryResult:
    if question is None:
        question = "What is the content of this image?"
    prompt = (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
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
            "multi_modal_data": {"image": image_data},
        },
        limit_mm_per_prompt={"image": 1},
    )


def get_video_query(
    question: str = None,
    video_path: str | None = None,
    num_frames: int = 16,
) -> QueryResult:
    if question is None:
        question = "Why is this video funny?"
    prompt = (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
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
            "multi_modal_data": {"video": video_frames},
        },
        limit_mm_per_prompt={"video": 1},
    )


query_map = {
    "text": get_text_query,
    "use_audio": get_audio_query,
    "use_image": get_image_query,
    "use_video": get_video_query,
}

# ---------------------------------------------------------------------------
# Core async routine
# ---------------------------------------------------------------------------


def clone_prompt_for_request(template: dict) -> dict:
    """Shallow-clone prompt dict so concurrent requests own independent containers."""
    cloned = dict(template)
    for key in ("multi_modal_data", "mm_processor_kwargs", "additional_information"):
        value = template.get(key)
        if isinstance(value, dict):
            cloned[key] = dict(value)
        elif isinstance(value, list):
            cloned[key] = list(value)
    return cloned


def _default_deploy_config_path() -> str | None:
    """Best-effort default deploy config for running Qwen3-Omni with async_chunk.

    The default ``vllm_omni/deploy/qwen3_omni_moe.yaml`` ships with
    ``async_chunk: true`` at the top level, so loading it is enough to
    enable async-chunk semantics. To disable it, copy the YAML and set
    ``async_chunk: false`` (or pass ``--deploy-config`` to a YAML that
    overrides the flag).

    When this example is executed from within the repository, we resolve
    the default YAML path relative to this file. When installed elsewhere,
    the file may not exist and callers should pass ``--deploy-config``
    explicitly.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    candidate = os.path.join(
        repo_root,
        "vllm_omni",
        "deploy",
        "qwen3_omni_moe.yaml",
    )
    return candidate if os.path.exists(candidate) else None


async def run_single_request(
    async_omni: AsyncOmni,
    prompt: dict,
    request_id: str,
    sampling_params_list: list[SamplingParams] | None,
    output_dir: str,
    output_modalities: list[str] | None = None,
    stream_audio_to_disk: bool = False,
) -> dict:
    """Run one request through AsyncOmni and collect outputs.

    Returns a dict with timing information and saved file paths.
    """
    t_start = time.perf_counter()
    text_parts: list[str] = []
    audio_chunks: list[torch.Tensor] = []
    audio_sr: int | None = None
    first_audio_ts: float | None = None
    audio_list_consumed: int = 0
    audio_last_tensor: torch.Tensor | None = None
    stage_0_first_output_ts: float | None = None

    samplerate = 24000
    wav_file = os.path.join(output_dir, f"output_{request_id}.wav")
    sf_writer: sf.SoundFile | None = None
    audio_samples_written: int = 0

    try:
        async for omni_output in async_omni.generate(
            prompt=prompt,
            request_id=request_id,
            sampling_params_list=sampling_params_list,
            output_modalities=output_modalities,
        ):
            output = omni_output.request_output
            if omni_output.final_output_type == "text":
                if stage_0_first_output_ts is None:
                    stage_0_first_output_ts = time.perf_counter()
                text_output = output.outputs[0].text
                text_parts.append(text_output)
            elif omni_output.final_output_type == "audio":
                mm_out = output.outputs[0].multimodal_output
                if mm_out and "audio" in mm_out:
                    if first_audio_ts is None:
                        first_audio_ts = time.perf_counter()
                    if audio_sr is None and "sr" in mm_out:
                        sr_val = mm_out["sr"]
                        audio_sr = sr_val.item() if hasattr(sr_val, "item") else int(sr_val)
                        samplerate = audio_sr
                    audio_data = mm_out["audio"]
                    if isinstance(audio_data, list):
                        new_chunks = audio_data[audio_list_consumed:]
                        audio_list_consumed = len(audio_data)
                    elif isinstance(audio_data, torch.Tensor):
                        new_chunks = [audio_data]
                        audio_last_tensor = audio_data
                    else:
                        new_chunks = []

                    if stream_audio_to_disk and new_chunks:
                        if sf_writer is None:
                            sf_writer = sf.SoundFile(
                                wav_file,
                                mode="w",
                                samplerate=samplerate,
                                channels=1,
                                subtype="FLOAT",
                            )
                        for chunk in new_chunks:
                            chunk_np = chunk.float().detach().cpu().numpy().flatten()
                            sf_writer.write(chunk_np)
                            audio_samples_written += len(chunk_np)
                    else:
                        audio_chunks.extend(new_chunks)
    finally:
        if sf_writer is not None:
            sf_writer.close()

    t_end = time.perf_counter()
    result = {
        "request_id": request_id,
        "e2e_latency_s": t_end - t_start,
        "saved_files": [],
    }

    # Save text output
    if text_parts:
        text_file = os.path.join(output_dir, f"{request_id}.txt")
        with open(text_file, "w", encoding="utf-8") as f:
            f.write("".join(text_parts))
        result["saved_files"].append(text_file)
        print(
            f"[Request {request_id}] Text saved to {text_file} "
            f"(stage-0 first output at {stage_0_first_output_ts - t_start:.3f}s)"
        )

    # Save audio output
    if stream_audio_to_disk and audio_samples_written > 0:
        result["saved_files"].append(wav_file)
        result["audio_duration_s"] = audio_samples_written / samplerate
        result["num_audio_chunks"] = audio_list_consumed
        ttfa = (first_audio_ts - t_start) if first_audio_ts else None
        result["time_to_first_audio_s"] = ttfa
        ttfa_str = f"{ttfa:.3f}s" if ttfa is not None else "N/A"
        print(
            f"[Request {request_id}] Audio streamed to {wav_file} "
            f"(duration={result['audio_duration_s']:.2f}s, "
            f"TTFA={ttfa_str}, "
            f"e2e={result['e2e_latency_s']:.3f}s)"
        )
    elif audio_chunks or audio_last_tensor is not None:
        if audio_chunks:
            if len(audio_chunks) > 1:
                audio_tensor = torch.cat(audio_chunks, dim=-1)
            else:
                audio_tensor = audio_chunks[0]
        else:
            audio_tensor = audio_last_tensor
        audio_numpy = audio_tensor.float().detach().cpu().numpy()
        if audio_numpy.ndim > 1:
            audio_numpy = audio_numpy.flatten()
        sf.write(wav_file, audio_numpy, samplerate=samplerate, format="WAV")
        result["saved_files"].append(wav_file)
        result["audio_duration_s"] = len(audio_numpy) / samplerate
        result["num_audio_chunks"] = len(audio_chunks)
        ttfa = (first_audio_ts - t_start) if first_audio_ts else None
        result["time_to_first_audio_s"] = ttfa
        ttfa_str = f"{ttfa:.3f}s" if ttfa is not None else "N/A"
        print(
            f"[Request {request_id}] Audio saved to {wav_file} "
            f"({len(audio_chunks)} chunks, "
            f"duration={result['audio_duration_s']:.2f}s, "
            f"TTFA={ttfa_str}, "
            f"e2e={result['e2e_latency_s']:.3f}s)"
        )

    return result


async def run_all(args):
    """Main async entry: build prompts, create AsyncOmni, run requests."""
    # Build query
    query_func = query_map[args.query_type]
    if args.query_type == "use_video":
        query_result = query_func(
            video_path=getattr(args, "video_path", None),
            num_frames=getattr(args, "num_frames", 16),
        )
    elif args.query_type == "use_image":
        query_result = query_func(image_path=getattr(args, "image_path", None))
    elif args.query_type == "use_audio":
        query_result = query_func(
            audio_path=getattr(args, "audio_path", None),
            sampling_rate=getattr(args, "sampling_rate", 16000),
        )
    else:
        query_result = query_func()

    # Build prompt list
    if args.txt_prompts is not None:
        assert args.query_type == "text", "txt-prompts is only supported for text query type"
        with open(args.txt_prompts, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        prompts = [get_text_query(ln).inputs for ln in lines]
        print(f"[Info] Loaded {len(prompts)} prompts from {args.txt_prompts}")
    else:
        prompts = [clone_prompt_for_request(query_result.inputs) for _ in range(args.num_prompts)]

    # Inject output modalities if specified
    output_modalities = None
    if args.modalities is not None:
        output_modalities = args.modalities.split(",")
        for prompt in prompts:
            prompt["modalities"] = output_modalities

    # Create AsyncOmni
    print(f"[Info] Creating AsyncOmni with deploy_config={args.deploy_config}")
    async_omni = None
    try:
        # ``from_cli_args`` expands vars(args) into kwargs and auto-captures
        # ``_cli_explicit_keys`` from ``sys.argv[1:]`` so argparse defaults
        # do not silently override deploy YAML values. Mirrors the
        # ``EngineArgs.from_cli_args`` pattern used throughout vllm /
        # vllm-omni. ``deploy_config=None`` (the default) falls through to
        # the bundled ``vllm_omni/deploy/qwen3_omni_moe.yaml``.
        async_omni = AsyncOmni.from_cli_args(args)

        # Use default sampling params from stage config (they are pre-configured
        # in the YAML for each stage).
        #
        # NOTE: Since we do not set the sampling params directly, .generate in
        # will automatically set the output kind to delta, since this is what
        # makes sense for most multimodal use-cases.
        sampling_params_list = None

        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Run requests with concurrency control
        semaphore = asyncio.Semaphore(args.max_in_flight)
        request_timeout = getattr(args, "request_timeout_s", None)
        stream_audio = getattr(args, "stream_audio_to_disk", False)

        async def _run_one(idx: int, prompt: dict):
            async with semaphore:
                request_id = f"req_{idx}_{uuid.uuid4().hex[:8]}"
                coro = run_single_request(
                    async_omni=async_omni,
                    prompt=prompt,
                    request_id=request_id,
                    sampling_params_list=sampling_params_list,
                    output_dir=output_dir,
                    output_modalities=output_modalities,
                    stream_audio_to_disk=stream_audio,
                )
                if request_timeout and request_timeout > 0:
                    return await asyncio.wait_for(coro, timeout=request_timeout)
                return await coro

        wall_start = time.perf_counter()
        tasks = [_run_one(i, p) for i, p in enumerate(prompts)]
        all_results = await asyncio.gather(*tasks, return_exceptions=True)
        wall_end = time.perf_counter()

        # Print summary
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        success_count = 0
        total_audio_dur = 0.0
        for r in all_results:
            if isinstance(r, Exception):
                print(f"  [ERROR] {type(r).__name__}: {r}")
            else:
                success_count += 1
                total_audio_dur += r.get("audio_duration_s", 0.0)
                print(f"  [{r['request_id']}] e2e={r['e2e_latency_s']:.3f}s  files={r['saved_files']}")
        wall_time = wall_end - wall_start
        print(f"\nTotal: {success_count}/{len(prompts)} succeeded")
        print(f"Wall time: {wall_time:.3f}s")
        if total_audio_dur > 0:
            print(f"Total audio duration: {total_audio_dur:.2f}s")
            print(f"Real-time factor: {total_audio_dur / wall_time:.2f}x")
        print("=" * 60)
    finally:
        if async_omni is not None:
            async_omni.shutdown()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = FlexibleArgumentParser(
        description=(
            "Offline inference with async_chunk enabled via AsyncOmni. "
            "Downstream stages start before upstream stages finish, "
            "achieving true stage-level concurrency."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="Model name or path.",
    )
    parser.add_argument(
        "--query-type",
        "-q",
        type=str,
        default="use_audio",
        choices=query_map.keys(),
        help="Query type.",
    )
    parser.add_argument(
        "--deploy-config",
        type=str,
        default=_default_deploy_config_path(),
        help=(
            "Path to a deploy config YAML. "
            "If not set, uses the model's default config "
            "(make sure it has async_chunk: true)."
        ),
    )
    parser.add_argument(
        "--log-stats",
        action="store_true",
        default=False,
        help="Enable writing detailed statistics.",
    )
    parser.add_argument(
        "--stage-init-timeout",
        type=int,
        default=300,
        help="Timeout for initializing a single stage (seconds).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_audio_async_chunk",
        help="Directory to save output files.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1,
        help="Number of prompts to generate (duplicated from query).",
    )
    parser.add_argument(
        "--txt-prompts",
        type=str,
        default=None,
        help="Path to a .txt file with one prompt per line.",
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=1,
        help="Maximum concurrent requests (default: 1).",
    )
    parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=None,
        help=(
            "Per-request timeout in seconds. When set, a request that "
            "exceeds this duration is cancelled and reported as an error. "
            "Default: None (no timeout)."
        ),
    )
    parser.add_argument(
        "--batch-timeout-s",
        type=float,
        default=None,
        help=(
            "Global timeout for the entire batch in seconds. When set, "
            "the whole run_all() is cancelled if it exceeds this duration. "
            "Default: None (no global timeout)."
        ),
    )
    parser.add_argument(
        "--stream-audio-to-disk",
        action="store_true",
        default=False,
        help=(
            "Write audio chunks to WAV incrementally instead of "
            "accumulating in memory. Useful for very long audio or "
            "high --max-in-flight to reduce memory footprint."
        ),
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default=None,
        help="Comma-separated output modalities filter (e.g. 'text', 'audio', 'text,audio').",
    )
    parser.add_argument(
        "--audio-path",
        "-a",
        type=str,
        default=None,
        help="Path to local audio file.",
    )
    parser.add_argument(
        "--image-path",
        "-i",
        type=str,
        default=None,
        help="Path to local image file.",
    )
    parser.add_argument(
        "--video-path",
        "-v",
        type=str,
        default=None,
        help="Path to local video file.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Number of frames to extract from video.",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=16000,
        help="Sampling rate for audio loading.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    async def _main():
        batch_timeout = getattr(args, "batch_timeout_s", None)
        if batch_timeout and batch_timeout > 0:
            await asyncio.wait_for(run_all(args), timeout=batch_timeout)
        else:
            await run_all(args)

    try:
        asyncio.run(_main())
    except asyncio.TimeoutError:
        print(
            f"\n[TIMEOUT] Batch exceeded --batch-timeout-s="
            f"{args.batch_timeout_s}s. AsyncOmni shutdown was handled "
            f"by finally block."
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user. AsyncOmni shutdown was handled by finally block.")
