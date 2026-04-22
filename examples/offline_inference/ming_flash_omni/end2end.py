# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Partial example cases are referred from
# https://github.com/inclusionAI/Ming/blob/3954fcb880ff5e61ff128bcf7f1ec344d46a6fe3/cookbook.ipynb
import os
import time
from typing import NamedTuple

import numpy as np
import vllm
from PIL import Image
from transformers import AutoProcessor
from vllm import SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset, video_to_ndarrays
from vllm.multimodal.image import convert_image_mode
from vllm.multimodal.media.audio import load_audio
from vllm.utils.argparse_utils import FlexibleArgumentParser

import vllm_omni
from vllm_omni.entrypoints.omni import Omni

# Imports the processor also registers itself
from vllm_omni.transformers_utils.processors.ming import MingFlashOmniProcessor  # noqa: F401

SEED = 42
MODEL_NAME = "Jonathan1909/Ming-flash-omni-2.0"


class QueryResult(NamedTuple):
    inputs: dict
    limit_mm_per_prompt: dict[str, int]


def get_text_query(processor: MingFlashOmniProcessor, question: str | None = None) -> QueryResult:
    if question is None:
        question = "请详细介绍鹦鹉的生活习性。"
    conversation = [{"role": "HUMAN", "content": question}]
    prompt = processor.apply_chat_template(conversation, tokenize=False)
    return QueryResult(
        inputs={"prompt": prompt},
        limit_mm_per_prompt={},
    )


def get_image_query(
    processor: MingFlashOmniProcessor,
    question: str | None = None,
    image_path: str | None = None,
) -> QueryResult:
    if question is None:
        question = "Describe this image in detail."

    if image_path:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image_data = convert_image_mode(Image.open(image_path), "RGB")
    else:
        image_data = convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")

    conversation = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "image", "image": image_data},
                {"type": "text", "text": question},
            ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, tokenize=False)

    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {"image": image_data},
        },
        limit_mm_per_prompt={"image": 1},
    )


def get_audio_query(
    processor: MingFlashOmniProcessor,
    question: str | None = None,
    audio_path: str | None = None,
    sampling_rate: int = 16000,
) -> QueryResult:
    if question is None:
        question = "Please recognize the language of this speech and transcribe it. Format: oral."

    if audio_path:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio_signal, sr = load_audio(audio_path, sr=sampling_rate)
        audio_data = (audio_signal.astype(np.float32), sr)
    else:
        audio_data = AudioAsset("mary_had_lamb").audio_and_sample_rate

    # Use a string for "audio" so the processor counts it as 1 audio input
    conversation = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "audio", "audio": "input"},
                {"type": "text", "text": question},
            ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, tokenize=False)

    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {"audio": audio_data},
        },
        limit_mm_per_prompt={"audio": 1},
    )


def get_video_query(
    processor: MingFlashOmniProcessor,
    question: str | None = None,
    video_path: str | None = None,
    num_frames: int = 16,
) -> QueryResult:
    if question is None:
        question = "Describe what is happening in this video."

    if video_path:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        video_frames = video_to_ndarrays(video_path, num_frames=num_frames)
    else:
        video_frames = VideoAsset(name="baby_reading", num_frames=num_frames).np_ndarrays

    conversation = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "video"},
                {"type": "text", "text": question},
            ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, tokenize=False)

    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {"video": video_frames},
        },
        limit_mm_per_prompt={"video": 1},
    )


def get_mixed_modalities_query(
    processor: MingFlashOmniProcessor,
    image_path: str | None = None,
    audio_path: str | None = None,
    sampling_rate: int = 16000,
) -> QueryResult:
    """Mixed image + audio understanding."""
    question = "Describe the image, and recognize the language of this speech and transcribe it. Format: oral"

    if image_path:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image_data = convert_image_mode(Image.open(image_path), "RGB")
    else:
        image_data = convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")

    if audio_path:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        sig, sr = load_audio(audio_path, sr=sampling_rate)
        audio_data = (sig.astype(np.float32), sr)
    else:
        audio_data = AudioAsset("mary_had_lamb").audio_and_sample_rate

    conversation = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "image", "image": image_data},
                {"type": "audio", "audio": "input"},
                {"type": "text", "text": question},
            ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, tokenize=False)

    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {"image": image_data, "audio": audio_data},
        },
        limit_mm_per_prompt={"image": 1, "audio": 1},
    )


def get_reasoning_query(
    processor: MingFlashOmniProcessor,
    question: str | None = None,
    image_path: str | None = None,
) -> QueryResult:
    if question is None:
        # NOTE: To use the following default question, input with example figure provided by Ming
        # https://github.com/inclusionAI/Ming/blob/3954fcb880ff5e61ff128bcf7f1ec344d46a6fe3/figures/cases/3_0.png
        # E.g.,
        # python examples/offline_inference/ming_flash_omni/end2end.py -q reasoning --image-path ./3_0.png
        # Otherwise, the problem solving might be false.
        question = (
            "Based on the following rules:\n•\tYou control the smiley face character\n"
            "•\tYou can move up, down, left, and right, and only a single square at a time\n"
            "•\tWalls are dark grey and cannot be moved into\n•\tThe brown square is a box\n•"
            "\tThe box can be pushed by moving into it (i.e., if you are in the square "
            "adjacent to the box to the left, and move onto the square with the box, "
            "the box will move one square to the right).\n"
            "•\tThe box cannot be pushed into walls\n"
            "•\tThe blue door at the bottom is locked and cannot be passed through, "
            "unless the box is placed on the blue square\n"
            "•\tThe square beneath the blue door is the exit\n"
            "•\tMoving from one square to another\n\n"
            "Let's assume a coordinate system where the smiley face is "
            "on the top left at (1,1) and the square below it is (1,2). "
            "The smiley face performs the following moves: {down, right, right, right}, "
            "such that the smiley face is at square (4,2) and the box is in square (5,2). "
            "What are the next sequence of moves that must be done to move the box down to (5,3)? "
            "Give your answer as a comma separated list."
        )

    if image_path:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image_data = convert_image_mode(Image.open(image_path), "RGB")
        conversation = [
            {
                "role": "HUMAN",
                "content": [
                    {"type": "image", "image": image_data},
                    {"type": "text", "text": question},
                ],
            }
        ]
        prompt = processor.apply_chat_template(conversation, tokenize=False, use_cot_system_prompt=True)
        return QueryResult(
            inputs={
                "prompt": prompt,
                "multi_modal_data": {"image": image_data},
            },
            limit_mm_per_prompt={"image": 1},
        )

    conversation = [{"role": "HUMAN", "content": question}]
    prompt = processor.apply_chat_template(conversation, tokenize=False, use_cot_system_prompt=True)
    return QueryResult(
        inputs={"prompt": prompt},
        limit_mm_per_prompt={},
    )


query_map = {
    "text": get_text_query,
    "use_audio": get_audio_query,
    "use_image": get_image_query,
    "use_video": get_video_query,
    "use_mixed_modalities": get_mixed_modalities_query,
    "reasoning": get_reasoning_query,
}


def main(args):
    print(
        "=" * 20,
        "\n",
        f"vllm version: {vllm.__version__}\n",
        f"vllm-omni version: {vllm_omni.__version__}\n",
        "=" * 20,
        sep="",
    )

    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    assert isinstance(processor, MingFlashOmniProcessor), f"Wrong processor type being used: {type(processor)}"

    query_func = query_map[args.query_type]
    if args.query_type == "use_image":
        query_result = query_func(processor, image_path=args.image_path)
    elif args.query_type == "use_audio":
        query_result = query_func(processor, audio_path=args.audio_path, sampling_rate=args.sampling_rate)
    elif args.query_type == "use_video":
        query_result = query_func(processor, video_path=args.video_path, num_frames=args.num_frames)
    elif args.query_type == "use_mixed_modalities":
        query_result = query_func(
            processor,
            image_path=args.image_path,
            audio_path=args.audio_path,
            sampling_rate=args.sampling_rate,
        )
    elif args.query_type == "reasoning":
        query_result = query_func(processor, image_path=args.image_path)
    else:
        query_result = query_func(processor)

    # Initialize Omni (with thinker-only stage config)
    omni = Omni(
        model=MODEL_NAME,
        stage_configs_path=args.stage_configs_path,
        log_stats=args.log_stats,
        init_timeout=args.init_timeout,
        stage_init_timeout=args.stage_init_timeout,
    )

    # Thinker sampling params
    thinker_sampling_params = SamplingParams(
        temperature=0.4,
        top_p=0.9,
        max_tokens=args.max_tokens,
        repetition_penalty=1.05,
        seed=SEED,
        detokenize=True,
    )
    sampling_params_list = [thinker_sampling_params]

    prompts = [query_result.inputs for _ in range(args.num_prompts)]

    if args.modalities is not None:
        output_modalities = args.modalities.split(",")
        for prompt in prompts:
            prompt["modalities"] = output_modalities

    total_requests = len(prompts)
    processed_count = 0
    print(f"Query type: {args.query_type}")
    print(f"Number of prompts: {total_requests}")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    profiler_enabled = args.enable_profiler
    if profiler_enabled:
        omni.start_profile(stages=args.profiler_stages)

    for stage_outputs in omni.generate(prompts, sampling_params_list):
        output = stage_outputs.request_output
        if stage_outputs.final_output_type == "text":
            request_id = output.request_id
            text_output = output.outputs[0].text
            lines = []
            lines.append("Prompt:\n")
            lines.append(str(output.prompt) + "\n")
            lines.append("Text Output:\n")
            lines.append(str(text_output).strip() + "\n")
            print(*lines, sep="")

            # Save to file
            out_txt = os.path.join(output_dir, f"{request_id}.txt")
            try:
                with open(out_txt, "w", encoding="utf-8") as f:
                    f.writelines(lines)
                print(f"Request ID: {request_id}, text saved to {out_txt}")
            except Exception as e:
                print(f"Failed to write output file {out_txt}: {e}")

        elif stage_outputs.final_output_type == "audio":
            raise NotImplementedError("Add audio example after talker supported.")

        processed_count += 1
        if profiler_enabled and processed_count >= total_requests:
            print(f"[Info] Processed {processed_count}/{total_requests}. Stopping profiler inside active loop...")
            # Stop the profiler while workers are still alive
            omni.stop_profile(stages=args.profiler_stages)

            print("[Info] Waiting 30s for workers to write trace files to disk...")
            time.sleep(30)
            print("[Info] Trace export wait time finished.")

    omni.close()


def parse_args():
    parser = FlexibleArgumentParser(description="Ming-flash-omni 2.0 offline inference example")
    parser.add_argument(
        "--query-type",
        "-q",
        type=str,
        default="text",
        choices=query_map.keys(),
        help="Query type.",
    )
    parser.add_argument(
        "--stage-configs-path",
        type=str,
        default=None,
        help="Path to a stage configs YAML file.",
    )
    parser.add_argument(
        "--log-stats",
        action="store_true",
        default=False,
        help="Enable detailed statistics logging.",
    )
    parser.add_argument("--init-timeout", type=int, default=2000, help="Timeout for initializing in seconds.")
    parser.add_argument(
        "--stage-init-timeout",
        type=int,
        default=2000,
        help="Timeout for initializing a single stage in seconds.",
    )
    parser.add_argument(
        "--enable-profiler",
        action="store_true",
        default=False,
        help="Enables profiling when set.",
    )
    parser.add_argument(
        "--profiler-stages",
        type=int,
        nargs="*",
        default=[0],
        help="List of stage IDs to profile. If not set, profiles all stages.",
    )
    parser.add_argument(
        "--image-path",
        "-i",
        type=str,
        default=None,
        help="Path to local image file. Uses default asset if not provided.",
    )
    parser.add_argument(
        "--audio-path",
        "-a",
        type=str,
        default=None,
        help="Path to local audio file. Uses default asset if not provided.",
    )
    parser.add_argument(
        "--video-path",
        "-v",
        type=str,
        default=None,
        help="Path to local video file. Uses default asset if not provided.",
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
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Maximum tokens to generate.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1,
        help="Number of prompts to generate.",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default=None,
        help="Output modalities (comma-separated).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_ming",
        help="Output directory for results.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
