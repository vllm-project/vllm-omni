# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Offline end-to-end inference for MiniCPM-o 4.5 (thinker + talker/Token2Wav).

MiniCPM-o uses placeholders ``(<image>./</image>)``, ``(<audio>./</audio>)``,
and ``(<video>./</video>)``. Speech output requires a ``<|tts_bos|>`` suffix on
the assistant prefix (equivalent to online ``chat_template_kwargs.use_tts_template``).
"""

import os
import time
from typing import NamedTuple

import numpy as np
import soundfile as sf
import vllm
from PIL import Image
from vllm import SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset, video_to_ndarrays
from vllm.multimodal.image import convert_image_mode
from vllm.multimodal.media.audio import load_audio

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.utils.tracking_parser import TrackingArgumentParser

SEED = 42

default_system = (
    "You are MiniCPM-o, a helpful multimodal assistant that can "
    "understand images, audio and video, and respond in text and speech."
)


class QueryResult(NamedTuple):
    inputs: dict
    limit_mm_per_prompt: dict[str, int]


def _assistant_prefix(use_tts: bool) -> str:
    # Matches HF chat_template: assistant header, optional empty <think>, then TTS bos.
    prefix = "<|im_start|>assistant\n"
    if use_tts:
        prefix += "<|tts_bos|>"
    return prefix


def _build_prompt(user_body: str, use_tts: bool) -> str:
    return (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        f"<|im_start|>user\n{user_body}<|im_end|>\n"
        f"{_assistant_prefix(use_tts)}"
    )


def get_text_query(question: str | None = None, use_tts: bool = True) -> QueryResult:
    if question is None:
        question = "Say hello, then introduce vLLM-Omni in one sentence."
    return QueryResult(
        inputs={"prompt": _build_prompt(question, use_tts=use_tts)},
        limit_mm_per_prompt={},
    )


def get_image_query(
    question: str | None = None,
    image_path: str | None = None,
    use_tts: bool = True,
) -> QueryResult:
    if question is None:
        question = "What is the content of this image?"
    user_body = f"(<image>./</image>)\n{question}"

    if image_path:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image_data = convert_image_mode(Image.open(image_path), "RGB")
    else:
        image_data = convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")

    return QueryResult(
        inputs={
            "prompt": _build_prompt(user_body, use_tts=use_tts),
            "multi_modal_data": {"image": image_data},
        },
        limit_mm_per_prompt={"image": 1},
    )


def get_audio_query(
    question: str | None = None,
    audio_path: str | None = None,
    sampling_rate: int = 16000,
    use_tts: bool = True,
) -> QueryResult:
    if question is None:
        question = "What is the content of this audio?"
    user_body = f"(<audio>./</audio>)\n{question}"

    if audio_path:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio_signal, sr = load_audio(audio_path, sr=sampling_rate)
        audio_data = (audio_signal.astype(np.float32), sr)
    else:
        audio_data = AudioAsset("mary_had_lamb").audio_and_sample_rate

    return QueryResult(
        inputs={
            "prompt": _build_prompt(user_body, use_tts=use_tts),
            "multi_modal_data": {"audio": audio_data},
        },
        limit_mm_per_prompt={"audio": 1},
    )


def get_video_query(
    question: str | None = None,
    video_path: str | None = None,
    num_frames: int = 16,
    use_tts: bool = True,
) -> QueryResult:
    if question is None:
        question = "Why is this video funny?"
    user_body = f"(<video>./</video>)\n{question}"

    if video_path:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        video_frames = video_to_ndarrays(video_path, num_frames=num_frames)
    else:
        video_frames = VideoAsset(name="baby_reading", num_frames=num_frames).np_ndarrays

    return QueryResult(
        inputs={
            "prompt": _build_prompt(user_body, use_tts=use_tts),
            "multi_modal_data": {"video": video_frames},
        },
        limit_mm_per_prompt={"video": 1},
    )


def get_mixed_modalities_query(
    video_path: str | None = None,
    image_path: str | None = None,
    audio_path: str | None = None,
    num_frames: int = 16,
    sampling_rate: int = 16000,
    use_tts: bool = True,
) -> QueryResult:
    question = "What is recited in the audio? What is the content of this image? Why is this video funny?"
    user_body = f"(<audio>./</audio>)\n(<image>./</image>)\n(<video>./</video>)\n{question}"

    if video_path:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        video_frames = video_to_ndarrays(video_path, num_frames=num_frames)
    else:
        video_frames = VideoAsset(name="baby_reading", num_frames=num_frames).np_ndarrays

    if image_path:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image_data = convert_image_mode(Image.open(image_path), "RGB")
    else:
        image_data = convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")

    if audio_path:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio_signal, sr = load_audio(audio_path, sr=sampling_rate)
        audio_data = (audio_signal.astype(np.float32), sr)
    else:
        audio_data = AudioAsset("mary_had_lamb").audio_and_sample_rate

    return QueryResult(
        inputs={
            "prompt": _build_prompt(user_body, use_tts=use_tts),
            "multi_modal_data": {
                "audio": audio_data,
                "image": image_data,
                "video": video_frames,
            },
        },
        limit_mm_per_prompt={"audio": 1, "image": 1, "video": 1},
    )


def get_multi_audios_query(use_tts: bool = True) -> QueryResult:
    question = "Are these two audio clips the same?"
    user_body = f"(<audio>./</audio>)\n(<audio>./</audio>)\n{question}"
    return QueryResult(
        inputs={
            "prompt": _build_prompt(user_body, use_tts=use_tts),
            "multi_modal_data": {
                "audio": [
                    AudioAsset("winning_call").audio_and_sample_rate,
                    AudioAsset("mary_had_lamb").audio_and_sample_rate,
                ],
            },
        },
        limit_mm_per_prompt={"audio": 2},
    )


query_map = {
    "text": get_text_query,
    "use_audio": get_audio_query,
    "use_image": get_image_query,
    "use_video": get_video_query,
    "use_multi_audios": get_multi_audios_query,
    "use_mixed_modalities": get_mixed_modalities_query,
}


def _wants_tts(modalities: str | None) -> bool:
    """TTS bos is needed whenever audio is among the requested output modalities."""
    if modalities is None:
        return True
    parts = {p.strip() for p in modalities.split(",") if p.strip()}
    return "audio" in parts


def main(args):
    model_name = args.model
    print("=" * 20, "\n", f"vllm version: {vllm.__version__}", "\n", "=" * 20)

    video_path = getattr(args, "video_path", None)
    image_path = getattr(args, "image_path", None)
    audio_path = getattr(args, "audio_path", None)
    use_tts = _wants_tts(args.modalities)

    query_func = query_map[args.query_type]
    if args.query_type == "use_video":
        query_result = query_func(
            video_path=video_path,
            num_frames=getattr(args, "num_frames", 16),
            use_tts=use_tts,
        )
    elif args.query_type == "use_image":
        query_result = query_func(image_path=image_path, use_tts=use_tts)
    elif args.query_type == "use_audio":
        query_result = query_func(
            audio_path=audio_path,
            sampling_rate=getattr(args, "sampling_rate", 16000),
            use_tts=use_tts,
        )
    elif args.query_type == "use_mixed_modalities":
        query_result = query_func(
            video_path=video_path,
            image_path=image_path,
            audio_path=audio_path,
            num_frames=getattr(args, "num_frames", 16),
            sampling_rate=getattr(args, "sampling_rate", 16000),
            use_tts=use_tts,
        )
    elif args.query_type == "use_multi_audios":
        query_result = query_func(use_tts=use_tts)
    else:
        query_result = query_func(use_tts=use_tts)

    omni_kwargs = vars(args).copy()
    # Drop example-only CLI keys that are not Omni / engine args.
    for key in (
        "query_type",
        "output_wav",
        "output_dir",
        "num_prompts",
        "txt_prompts",
        "video_path",
        "image_path",
        "audio_path",
        "num_frames",
        "sampling_rate",
        "modalities",
        "py_generator",
        "enable_profiler",
        "profiler_stages",
        "log_dir",
    ):
        omni_kwargs.pop(key, None)
    omni_kwargs["model"] = model_name
    omni_kwargs["trust_remote_code"] = True
    omni = Omni(**omni_kwargs)

    # Stage 0 (thinker): multimodal understanding → text (+ TTS span when enabled).
    thinker_sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=2048,
        seed=SEED,
        detokenize=True,
        repetition_penalty=1.1,
    )
    # Stage 1 (talker + Token2Wav): max_tokens=1 satisfies the scheduler;
    # waveform is produced in-process by Token2Wav.
    talker_sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=1,
        seed=SEED,
        detokenize=False,
    )
    sampling_params_list = [thinker_sampling_params, talker_sampling_params][: omni.num_stages]

    if args.txt_prompts is None:
        prompts = [query_result.inputs for _ in range(args.num_prompts)]
    else:
        assert args.query_type == "text", "txt-prompts is only supported for text query type"
        with open(args.txt_prompts, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines()]
            prompts = [get_text_query(ln, use_tts=use_tts).inputs for ln in lines if ln != ""]
            print(f"[Info] Loaded {len(prompts)} prompts from {args.txt_prompts}")

    if args.modalities is not None:
        output_modalities = args.modalities.split(",")
        for prompt in prompts:
            prompt["modalities"] = output_modalities

    profiler_enabled = args.enable_profiler
    if profiler_enabled:
        omni.start_profile(stages=args.profiler_stages)

    omni_generator = omni.generate(prompts, sampling_params_list, py_generator=args.py_generator)
    output_dir = args.output_dir if getattr(args, "output_dir", None) else args.output_wav
    os.makedirs(output_dir, exist_ok=True)

    total_requests = len(prompts)
    processed_count = 0
    print(f"query type: {args.query_type}")
    print(f"use_tts (tts_bos): {use_tts}")

    for stage_outputs in omni_generator:
        output = stage_outputs.request_output
        if stage_outputs.final_output_type == "text":
            request_id = output.request_id
            text_output = output.outputs[0].text
            out_txt = os.path.join(output_dir, f"{request_id}.txt")
            lines = [
                "Prompt:\n",
                str(output.prompt) + "\n",
                "vllm_text_output:\n",
                str(text_output).strip() + "\n",
            ]
            try:
                with open(out_txt, "w", encoding="utf-8") as f:
                    f.writelines(lines)
            except Exception as e:
                print(f"[Warn] Failed writing text file {out_txt}: {e}")
            print(f"Request ID: {request_id}, Text saved to {out_txt}")
        elif stage_outputs.final_output_type == "audio":
            request_id = output.request_id
            audio_tensor = output.outputs[0].multimodal_output["audio"]
            output_wav = os.path.join(output_dir, f"output_{request_id}.wav")

            if isinstance(audio_tensor, list):
                import torch

                audio_tensor = torch.cat(
                    [(t if isinstance(t, torch.Tensor) else torch.tensor(t)).flatten() for t in audio_tensor]
                )
            audio_numpy = audio_tensor.float().detach().cpu().numpy()
            if audio_numpy.ndim > 1:
                audio_numpy = audio_numpy.flatten()

            # MiniCPM-o 4.5 Token2Wav emits 24 kHz mono.
            sf.write(output_wav, audio_numpy, samplerate=24000, format="WAV")
            print(f"Request ID: {request_id}, Saved audio to {output_wav}")

        processed_count += 1
        if profiler_enabled and processed_count >= total_requests:
            print(f"[Info] Processed {processed_count}/{total_requests}. Stopping profiler inside active loop...")
            omni.stop_profile(stages=args.profiler_stages)
            print("[Info] Waiting 30s for workers to write trace files to disk...")
            time.sleep(30)
            print("[Info] Trace export wait time finished.")
    omni.close()


def parse_args():
    parser = TrackingArgumentParser(
        description="Offline inference demo for MiniCPM-o 4.5 (text / image / audio / video → text + speech)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openbmb/MiniCPM-o-4_5",
        help="Model name or path.",
    )
    parser.add_argument(
        "--query-type",
        "-q",
        type=str,
        default="text",
        choices=query_map.keys(),
        help="Query type.",
    )
    parser.add_argument(
        "--log-stats",
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
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for text/audio results.",
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
        "--stage-configs-path",
        type=str,
        default=None,
        help="Path to a stage configs file (deprecated; prefer --deploy-config).",
    )
    parser.add_argument(
        "--deploy-config",
        type=str,
        default=None,
        help="Path to a deploy YAML (default: auto-load minicpmo_4_5.yaml).",
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
        "--log-dir",
        type=str,
        default="logs",
        help="Log directory (default: logs).",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default=None,
        help="Output modalities (comma-separated), e.g. text or text,audio.",
    )
    parser.add_argument(
        "--py-generator",
        action="store_true",
        default=False,
        help="Use py_generator mode. The returned type of Omni.generate() is a Python Generator object.",
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
        default=None,
        help="List of stage IDs to profile. If not set, profiles all stages.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Model dtype (auto, half, float16, bfloat16, float, float32).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
