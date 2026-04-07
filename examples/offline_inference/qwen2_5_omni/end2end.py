# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This example shows how to use vLLM-Omni for running offline inference
with the correct prompt format on Qwen2.5-Omni

Multimodal inputs default to **local files** under
``examples/offline_inference/qwen2_5_omni/local_assets/`` (auto-created
PNG/WAV, and MP4 via ffmpeg or OpenCV when available). Use
``--remote-vllm-assets`` to fall back to vLLM's bundled/downloaded assets.
"""

import os

# Triton (used by vLLM rotary / flash-attn kernels) calls `/sbin/ldconfig -p` to locate
# libcuda unless TRITON_LIBCUDA_PATH is set. On minimal hosts or broken linker cache
# (`/etc/ld.so.cache` missing), ldconfig exits non-zero and EngineCore crashes during
# multimodal profile_run. Set the directory containing libcuda.so.1 before importing vLLM.
def _ensure_triton_libcuda_path() -> None:
    if os.environ.get("TRITON_LIBCUDA_PATH"):
        return
    candidates = (
        "/usr/lib/x86_64-linux-gnu",
        "/lib/x86_64-linux-gnu",
        "/usr/lib/wsl/lib",
        "/usr/local/cuda/lib64",
        "/usr/lib64",
        "/opt/nvidia/lib",
    )
    for d in candidates:
        try:
            if os.path.isfile(os.path.join(d, "libcuda.so.1")):
                os.environ["TRITON_LIBCUDA_PATH"] = d
                return
        except OSError:
            continue


_ensure_triton_libcuda_path()

import shutil
import subprocess
import time
from pathlib import Path
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


def _vllm_omni_repo_root() -> Path:
    """vllm-omni repo root: .../examples/offline_inference/qwen2_5_omni/end2end.py -> parents[3]."""
    return Path(__file__).resolve().parents[3]


def default_local_mm_assets_dir() -> Path:
    """Directory for offline multimodal demo files (under vllm-omni tree)."""
    return _vllm_omni_repo_root() / "examples" / "offline_inference" / "qwen2_5_omni" / "local_assets"


def _write_sine_wav(path: Path, *, sr: int, seconds: float, freq_hz: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(sr * seconds)
    t = np.linspace(0.0, seconds, n, endpoint=False, dtype=np.float32)
    x = (0.08 * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)
    sf.write(str(path), x, int(sr))


def _ensure_demo_image_png(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    arr[:, :] = (64, 120, 200)
    arr[32:96, 32:96] = (200, 80, 80)
    Image.fromarray(arr, mode="RGB").save(str(path), format="PNG")


def _ensure_demo_video_mp4_cv2(path: Path, *, num_frames: int) -> bool:
    try:
        import cv2
    except ImportError:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = 128, 128
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 4.0, (w, h))
    if not out.isOpened():
        return False
    try:
        for i in range(max(8, num_frames)):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            frame[:, :] = ((i * 11) % 220, 100, 80)
            out.write(frame)
    finally:
        out.release()
    return path.exists()


def _ensure_demo_video_with_audio_ffmpeg(
    path: Path,
    audio_wav: Path,
    *,
    fps: int = 4,
    min_frames: int = 20,
) -> bool:
    """Mux synthetic color video + WAV using ffmpeg.

    ``min_frames`` must be at least the ``--num-frames`` used by ``video_to_ndarrays`` (often 16).
    We avoid ``-shortest`` so a short WAV does not truncate the video below ``min_frames``.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not audio_wav.is_file():
        return False
    duration_sec = max(3.0, (float(min_frames) + 2.0) / float(fps))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp_av.mp4")
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=blue:s=128x128:r={fps}:d={duration_sec}",
        "-i",
        str(audio_wav),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-t",
        str(duration_sec),
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=120)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return False
    if tmp.exists():
        tmp.replace(path)
        return path.exists()
    return False


def ensure_local_multimodal_demo_files(
    asset_dir: Path,
    *,
    sampling_rate: int = 16000,
    num_frames_hint: int = 16,
) -> dict[str, str]:
    """Create small offline-safe demo media under ``asset_dir`` and return absolute paths.

    Keys: ``image``, ``audio``, ``audio_b``, ``video`` (may be silent mp4 if only OpenCV worked),
    ``video_av`` (video+audio when ffmpeg succeeded, else same as ``video``).
    """
    asset_dir = Path(asset_dir).resolve()
    asset_dir.mkdir(parents=True, exist_ok=True)

    img = asset_dir / "demo_image.png"
    aud = asset_dir / "demo_audio.wav"
    aud_b = asset_dir / "demo_audio_b.wav"
    vid_av = asset_dir / "demo_video_with_audio.mp4"
    vid_silent = asset_dir / "demo_video_silent.mp4"

    _ensure_demo_image_png(img)
    if not aud.exists():
        _write_sine_wav(aud, sr=sampling_rate, seconds=2.5, freq_hz=440.0)
    if not aud_b.exists():
        _write_sine_wav(aud_b, sr=sampling_rate, seconds=2.5, freq_hz=523.25)

    min_vid_frames = max(20, num_frames_hint + 4)

    def _video_frame_count_estimate(p: Path) -> int | None:
        ffprobe = shutil.which("ffprobe")
        if ffprobe:
            cmd = [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_packets",
                "-show_entries",
                "stream=nb_read_packets",
                "-of",
                "csv=p=0",
                str(p),
            ]
            try:
                out = subprocess.check_output(cmd, text=True, timeout=60).strip()
                if out:
                    return int(out)
            except (subprocess.CalledProcessError, ValueError, subprocess.TimeoutExpired):
                pass
        try:
            import cv2

            cap = cv2.VideoCapture(str(p))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return n if n > 0 else None
        except Exception:
            return None

    if vid_av.exists():
        got = _video_frame_count_estimate(vid_av)
        if got is not None and got < num_frames_hint:
            vid_av.unlink(missing_ok=True)

    if not vid_av.exists():
        _ensure_demo_video_with_audio_ffmpeg(vid_av, aud, min_frames=min_vid_frames)

    if not vid_silent.exists() and not vid_av.exists():
        if not _ensure_demo_video_mp4_cv2(vid_silent, num_frames=max(16, num_frames_hint)):
            raise RuntimeError(
                "无法生成本地演示视频：请安装 ffmpeg（推荐，可生成带音频的 mp4）或 "
                "pip install opencv-python-headless，或手动放置 mp4 并传入 --video-path。"
            )

    primary_video = vid_av if vid_av.exists() else vid_silent
    return {
        "image": str(img.resolve()),
        "audio": str(aud.resolve()),
        "audio_b": str(aud_b.resolve()),
        "video": str(primary_video.resolve()),
        "video_av": str(vid_av.resolve()) if vid_av.exists() else str(primary_video.resolve()),
    }


def resolve_stage_config_path_for_end2end(args) -> str | None:
    """Pick a stage YAML that fits the machine (single-GPU vs default 2-GPU pipeline).

    The stock ``qwen2_5_omni.yaml`` expects two visible GPUs: Thinker on logical ``0``,
    Talker and Token2Wav on logical ``1`` (three processes; stage-1 and stage-2 share
    the second device with capped KV so code2wav can run).
    """
    if getattr(args, "stage_config", None):
        return args.stage_config
    if getattr(args, "use_default_two_gpu_stages", False):
        print(
            "[Info] 已指定 --use-default-two-gpu-stages：使用模型默认双卡布局 "
            "（随仓库的 qwen2_5_omni.yaml；与检测到 ≥2 张 GPU 时的默认行为相同）。"
        )
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        dev_count = torch.cuda.device_count()
        if dev_count == 1:
            cfg = (
                _vllm_omni_repo_root()
                / "vllm_omni"
                / "model_executor"
                / "stage_configs"
                / "qwen2_5_omni_single_gpu.yaml"
            )
            if cfg.is_file():
                print(
                    "[Info] 检测到 1 块 GPU：使用随仓库分发的单卡 stage 配置（三阶段均在 GPU0，"
                    "降低 gpu_memory_utilization / max_model_len）。"
                )
                print(
                    "[Info] 说明：Thinker/Talker/Token2Wav 为三个独立进程，显存会叠加占用；"
                    "若 Stage2（token2wav）仍 OOM，请使用多卡（例如 "
                    "CUDA_VISIBLE_DEVICES=0,1 python .../end2end.py）或自行调低单卡 YAML。"
                )
                print(f"       {cfg}")
                return str(cfg)
        elif dev_count >= 2:
            print(
                "[Info] 检测到多块 GPU：使用模型默认 stage（vllm_omni/.../qwen2_5_omni.yaml）。"
                "布局：Thinker 在「可见」cuda:0；Talker 与 Token2Wav 在 cuda:1（进程内编号）。"
            )
            print(
                "[Info] 指定物理 GPU 示例：CUDA_VISIBLE_DEVICES=4,5 python .../end2end.py "
                "（则逻辑 0→物理 4、逻辑 1→物理 5）。无需再传 --use-default-two-gpu-stages。"
            )
    except Exception as exc:
        print(f"[Warn] 无法自动选择单卡 stage 配置: {exc}")
    return None


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


def get_multi_audios_query(
    audio_path: str | None = None,
    audio_path_2: str | None = None,
    sampling_rate: int = 16000,
) -> QueryResult:
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
        first = (audio_signal.astype(np.float32), sr)
        if audio_path_2:
            if not os.path.exists(audio_path_2):
                raise FileNotFoundError(f"Audio file not found: {audio_path_2}")
            a2, sr2 = librosa.load(audio_path_2, sr=sampling_rate)
            audio_list = [first, (a2.astype(np.float32), sr2)]
        else:
            audio_list = [
                first,
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
    cuda_vis = getattr(args, "cuda_visible_devices", None)
    if cuda_vis is not None and str(cuda_vis).strip() != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_vis).strip()
        print(f"[Info] 使用 CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    if getattr(args, "enable_magi_compile", False):
        os.environ["VLLM_OMNI_MAGI_COMPILE"] = "1"
        print(
            "[Info] MagiCompiler enabled: VLLM_OMNI_MAGI_COMPILE=1 "
            "(Thinker/Talker LM + Token2Wav DiT/BigVGAN when constructed; requires `magi_compiler`)."
        )

    model_name = args.model

    # Get paths from args
    video_path = getattr(args, "video_path", None)
    image_path = getattr(args, "image_path", None)
    audio_path = getattr(args, "audio_path", None)
    audio_path_2 = args.audio_path_2
    num_frames = getattr(args, "num_frames", 16)
    sampling_rate = getattr(args, "sampling_rate", 16000)

    if not getattr(args, "remote_vllm_assets", False) and args.query_type != "text":
        mm_base = os.environ.get("QWEN25_OMNI_MM_ASSETS_DIR", args.mm_assets_dir)
        mm_dir = Path(mm_base).expanduser().resolve()
        try:
            demos = ensure_local_multimodal_demo_files(
                mm_dir,
                sampling_rate=sampling_rate,
                num_frames_hint=num_frames,
            )
        except RuntimeError as exc:
            print(f"[Error] {exc}")
            raise
        print(f"[Info] 离线多模态资源目录: {mm_dir}")
        print("[Info] 演示文件:", demos)
        if args.query_type in ("use_mixed_modalities", "use_video", "use_audio_in_video"):
            if not video_path:
                video_path = demos["video"]
        if args.query_type in ("use_mixed_modalities", "use_image"):
            if not image_path:
                image_path = demos["image"]
        if args.query_type in ("use_mixed_modalities", "use_audio"):
            if not audio_path:
                audio_path = demos["audio"]
        if args.query_type == "multi_audios":
            if not audio_path:
                audio_path = demos["audio"]
            if not audio_path_2:
                audio_path_2 = demos["audio_b"]

    # Get the query function and call it with appropriate parameters
    query_func = query_map[args.query_type]
    if args.query_type == "use_mixed_modalities":
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
        query_result = query_func(
            audio_path=audio_path,
            audio_path_2=audio_path_2,
            sampling_rate=sampling_rate,
        )
    elif args.query_type == "use_image":
        query_result = query_func(image_path=image_path)
    elif args.query_type == "use_video":
        query_result = query_func(video_path=video_path, num_frames=num_frames)
    elif args.query_type == "use_audio":
        query_result = query_func(audio_path=audio_path, sampling_rate=sampling_rate)
    else:
        query_result = query_func()
    omni_kwargs = dict(
        model=model_name,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
        batch_timeout=args.batch_timeout,
        init_timeout=args.init_timeout,
        shm_threshold_bytes=args.shm_threshold_bytes,
    )
    stage_cfg_path = resolve_stage_config_path_for_end2end(args)
    if stage_cfg_path is not None:
        omni_kwargs["stage_configs_path"] = stage_cfg_path
    omni = Omni(**omni_kwargs)
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
    if profiler_enabled and hasattr(omni, "start_profile"):
        omni.start_profile(stages=[0])
    elif profiler_enabled:
        print("[Warn] VLLM_TORCH_PROFILER_DIR is set, but current engine does not support profiler controls.")
    omni_generator = omni.generate(prompts, sampling_params_list, py_generator=args.py_generator)

    # Determine output directory: prefer --output-dir; fallback to --output-wav
    output_dir = args.output_dir if getattr(args, "output_dir", None) else args.output_wav
    os.makedirs(output_dir, exist_ok=True)

    total_requests = len(prompts)
    processed_count = 0
    for stage_outputs in omni_generator:
        output = stage_outputs.request_output
        if stage_outputs.final_output_type == "text":
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
            request_id = output.request_id
            audio_tensor = output.outputs[0].multimodal_output["audio"]
            output_wav = os.path.join(output_dir, f"output_{request_id}.wav")
            sf.write(output_wav, audio_tensor.detach().cpu().numpy(), samplerate=24000)
            print(f"Request ID: {request_id}, Saved audio to {output_wav}")

        processed_count += 1
        if profiler_enabled and hasattr(omni, "stop_profile") and processed_count >= total_requests:
            print(f"[Info] Processed {processed_count}/{total_requests}. Stopping profiler inside active loop...")
            # Stop the profiler while workers are still alive
            omni.stop_profile()

            print("[Info] Waiting 30s for workers to write massive trace files to disk...")
            time.sleep(30)
            print("[Info] Trace export wait finished.")

    omni.close()


def parse_args():
    parser = FlexibleArgumentParser(description="Demo on using vLLM for offline inference with audio language models")
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get(
            "QWEN25_OMNI_MODEL",
            "/data/zy/work_models/Qwen2.5-Omni-3B",
        ),
        help="Local model directory (offline default) or HF id. Override with env QWEN25_OMNI_MODEL.",
    )
    parser.add_argument(
        "--enable-magi-compile",
        action="store_true",
        default=False,
        help="Set VLLM_OMNI_MAGI_COMPILE=1 for Qwen2.5-Omni MagiCompiler hooks (install [magi] extra).",
    )
    parser.add_argument(
        "--mm-assets-dir",
        type=str,
        nargs="?",
        const=str(default_local_mm_assets_dir()),
        default=str(default_local_mm_assets_dir()),
        help=(
            "多模态离线演示文件目录。可写路径；仅写 --mm-assets-dir 不带参数时等同默认仓库内 local_assets/。"
            "环境变量 QWEN25_OMNI_MM_ASSETS_DIR 仍在 main 中优先生效。"
        ),
    )
    parser.add_argument(
        "--remote-vllm-assets",
        action="store_true",
        default=False,
        help="不自动生成本地演示文件，未指定 --*-path 时使用 vLLM ImageAsset/VideoAsset/AudioAsset（可能需联网或缓存）。",
    )
    parser.add_argument(
        "--stage-config",
        type=str,
        default=None,
        help="自定义 omni stage YAML 路径。不设且为单卡时，自动使用仓库内 qwen2_5_omni_single_gpu.yaml。",
    )
    parser.add_argument(
        "--use-default-two-gpu-stages",
        action="store_true",
        default=False,
        help=(
            "显式选用模型随附的 qwen2_5_omni.yaml（双卡布局）。"
            "当进程内已可见 ≥2 张 GPU 时，不传本参数也会自动使用该默认；本开关主要用于显式确认。"
        ),
    )
    parser.add_argument(
        "--cuda-visible-devices",
        "-g",
        type=str,
        default=None,
        help=(
            "指定可见的物理 GPU，写法同环境变量 CUDA_VISIBLE_DEVICES（如 0 只用第 0 张卡，2 只用第 2 张，"
            "1,3 表示进程内可见两张卡且编号重排为 0、1）。不设则沿用当前 shell 里的 CUDA_VISIBLE_DEVICES。"
        ),
    )
    parser.add_argument(
        "--query-type",
        "-q",
        type=str,
        default="use_mixed_modalities",
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
        "--output-dir",
        type=str,
        default=None,
        help="文本/音频输出目录（未设置则与 --output-wav 相同）。",
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
        help="本地视频路径。默认同本地模式：自动使用 mm 目录下生成的 demo MP4（优先 ffmpeg 带音频）。",
    )
    parser.add_argument(
        "--image-path",
        "-i",
        type=str,
        default=None,
        help="本地图片路径。默认：mm 目录下 demo_image.png（非 --remote-vllm-assets）。",
    )
    parser.add_argument(
        "--audio-path",
        "-a",
        type=str,
        default=None,
        help="本地音频路径。默认：mm 目录下 demo_audio.wav（非 --remote-vllm-assets）。",
    )
    parser.add_argument(
        "--audio-path-2",
        type=str,
        default=None,
        help="第二条本地音频（用于 multi_audios；离线模式下默认用 demo_audio_b.wav）。",
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
