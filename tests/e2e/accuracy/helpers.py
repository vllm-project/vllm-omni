# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
import os
import re
import shutil
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from base64 import b64decode, b64encode
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from fractions import Fraction
from hashlib import sha1
from io import BytesIO
from mimetypes import guess_type
from pathlib import Path
from typing import IO, Any
from urllib.parse import urlparse

import numpy as np
import pytest
import requests
import torch
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

FTFY_SITECUSTOMIZE_MOCK_DIR = Path(__file__).with_name("ftfy_mock")


class IdentityFtfy:
    @staticmethod
    def fix_text(text: str) -> str:
        return text


def apply_ftfy_mock(*, wan_i2v_module: Any | None = None) -> None:
    """Mock ftfy library for test cases in the main process"""
    module: Any = wan_i2v_module
    if module is None:
        from diffusers.pipelines.wan import pipeline_wan_i2v as wan_pipeline

        module = wan_pipeline

    if not hasattr(module, "ftfy"):
        module.ftfy = IdentityFtfy()
        print("ftfy (text encoding sanitizer) is not installed. Using mock ftfy implementation (identity function)")
    else:
        print("ftfy (text encoding sanitizer) is installed. Using actual ftfy implementation.")


def env_to_apply_ftfy_mock_in_subproc(env: dict[str, str] | None = None) -> dict[str, str]:
    """Mock ftfy library for test cases in subprocesses, by prepending the sitecustomize.py directory to PYTHONPATH"""
    env_dict = dict(env or {})
    pythonpath = env_dict.get("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
    path_entries = [str(FTFY_SITECUSTOMIZE_MOCK_DIR)]
    if pythonpath:
        path_entries.append(pythonpath)
    env_dict["PYTHONPATH"] = os.pathsep.join(path_entries)
    return env_dict


def reset_artifact_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def infer_model_label(model: str) -> str:
    label = Path(model.rstrip("/\\")).name or "model"
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in label)


def model_output_dir(parent_dir: Path, model: str) -> Path:
    safe_model_name = model.split("/")[-1].replace(".", "_")
    path = parent_dir / safe_model_name
    path.mkdir(parents=True, exist_ok=True)
    return path


_SSIM_RE = re.compile(r"All:(?P<score>[0-9.]+)")
_PSNR_RE = re.compile(r"average:(?P<score>[0-9.]+)")


def parse_video_metadata(payload: dict[str, Any]) -> dict[str, int | float]:
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        raise ValueError(f"ffprobe payload did not include video streams: {payload}")

    stream = streams[0]
    width = int(stream["width"])
    height = int(stream["height"])
    fps = float(Fraction(stream["avg_frame_rate"]))
    frame_count_value = stream.get("nb_read_frames") or stream.get("nb_frames")
    if frame_count_value is None:
        raise ValueError(f"ffprobe payload did not include frame count: {payload}")
    frame_count = int(frame_count_value)
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
    }


def parse_ssim_score(output: str) -> float:
    match = _SSIM_RE.search(output)
    if match is None:
        raise ValueError(f"Could not parse SSIM score from ffmpeg output:\n{output}")
    return float(match.group("score"))


def parse_psnr_score(output: str) -> float:
    match = _PSNR_RE.search(output)
    if match is None:
        raise ValueError(f"Could not parse PSNR score from ffmpeg output:\n{output}")
    return float(match.group("score"))


def probe_binary(binary: str, *, test_name: str = "video similarity e2e test") -> str:
    resolved = shutil.which(binary)
    if resolved is None:
        pytest.skip(f"{binary} is required for {test_name}.")
    assert resolved is not None
    return resolved


def probe_video(path: Path) -> dict[str, int | float]:
    probe_binary("ffprobe")
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return parse_video_metadata(json.loads(result.stdout))


def run_ffmpeg_similarity(filter_name: str, first: Path, second: Path) -> str:
    probe_binary("ffmpeg")
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(first),
            "-i",
            str(second),
            "-lavfi",
            f"[0:v][1:v]{filter_name}",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stderr


def online_timeout_seconds(configured: int | None, *, default: int = 1200) -> int:
    return int(configured) if configured is not None else default


def is_remote_image_source(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


def validate_image_source(source: str) -> None:
    if is_remote_image_source(source):
        response = requests.get(source, timeout=60)
        response.raise_for_status()
        return

    image_path = Path(source)
    if not image_path.exists():
        raise FileNotFoundError(f"Local image source does not exist: {image_path}")


def build_online_image_reference(source: str) -> str:
    if source.startswith("data:image"):
        return source

    if is_remote_image_source(source):
        response = requests.get(source, timeout=60)
        response.raise_for_status()
        mime_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip()
        if not mime_type.startswith("image/"):
            mime_type = guess_type(urlparse(source).path)[0] or "image/png"
        encoded = b64encode(response.content).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    image_path = Path(source)
    mime_type = guess_type(image_path.name)[0] or "image/png"
    encoded = b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def materialize_image_source(source: str, output_dir: Path) -> str:
    if not is_remote_image_source(source):
        return source

    from diffusers.utils import load_image

    output_dir.mkdir(parents=True, exist_ok=True)
    source_name = Path(urlparse(source).path).name or "input.png"
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_name)
    if "." not in safe_name:
        safe_name = f"{safe_name}.png"
    image_path = output_dir / safe_name
    if not image_path.exists():
        image = load_image(source).convert("RGB")
        image.save(image_path)
    return str(image_path)


def video_artifact_dir(result_root: Path, source: str) -> Path:
    if is_remote_image_source(source):
        source_name = Path(urlparse(source).path).stem or "remote"
    else:
        source_name = Path(source).stem or "local"

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_name)
    digest = sha1(source.encode("utf-8")).hexdigest()[:8]
    return result_root / f"{safe_name}-{digest}"


def send_video_request_with_timeout(
    online_client,
    request_config: dict[str, Any],
    *,
    timeout_seconds: int,
) -> bytes:
    form_data = request_config.get("form_data")
    if not isinstance(form_data, dict):
        raise ValueError("Video request_config must contain 'form_data'")

    if not form_data.get("prompt"):
        raise ValueError("Video request_config['form_data'] must contain 'prompt'")

    normalized_form_data = {key: str(value) for key, value in form_data.items() if value is not None}

    files: dict[str, tuple[str, BytesIO, str]] = {}
    image_reference = request_config.get("image_reference")
    if image_reference:
        if image_reference.startswith("data:image"):
            header, encoded = image_reference.split(",", 1)
            content_type = header.split(";")[0].removeprefix("data:")
            extension = content_type.split("/")[-1]
            file_data = b64decode(encoded)
            files["input_reference"] = (
                f"reference.{extension}",
                BytesIO(file_data),
                content_type,
            )
        else:
            normalized_form_data["image_reference"] = json.dumps({"image_url": image_reference})

    start_time = time.perf_counter()
    create_url = online_client._build_url("/v1/videos")
    response = requests.post(
        create_url,
        data=normalized_form_data,
        files=files,
        headers={"Accept": "application/json"},
        timeout=60,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise requests.HTTPError(f"{exc}; response body: {response.text}", response=response) from exc
    job_data = response.json()
    video_id = job_data["id"]
    online_client._wait_until_video_completed(video_id, timeout_seconds=timeout_seconds)
    video_content = online_client._download_video_content(video_id)
    print(f"online_video_e2e_latency_s={time.perf_counter() - start_time:.3f}")
    return video_content


def assert_video_metadata(
    metadata: dict[str, int | float],
    *,
    width: int,
    height: int,
    fps: int | float,
    frame_count: int,
) -> None:
    assert metadata["width"] == width
    assert metadata["height"] == height
    assert metadata["fps"] == float(fps)
    assert metadata["frame_count"] == frame_count


def assert_video_similarity_metrics(
    *,
    label: str,
    online_path: Path,
    offline_path: Path,
    ssim_threshold: float,
    psnr_threshold: float,
) -> tuple[float, float]:
    ssim_output = run_ffmpeg_similarity("ssim", online_path, offline_path)
    psnr_output = run_ffmpeg_similarity("psnr", online_path, offline_path)
    ssim_score = parse_ssim_score(ssim_output)
    psnr_score = parse_psnr_score(psnr_output)

    print(f"{label} similarity metrics:")
    print(
        "  SSIM:"
        f" value={ssim_score:.6f},"
        f" threshold>={ssim_threshold:.6f},"
        " range=[0, 1],"
        " higher_is_better=True,"
        " interpretation=structural_similarity"
    )
    print(
        "  PSNR:"
        f" value={psnr_score:.6f} dB,"
        f" threshold>={psnr_threshold:.6f} dB,"
        " range=[0, +inf),"
        " higher_is_better=True,"
        " interpretation=pixel_error_in_decibels"
    )
    print(f"online_video={online_path}")
    print(f"offline_video={offline_path}")

    assert ssim_score >= ssim_threshold, (
        f"SSIM below threshold: got {ssim_score:.6f}, expected >= {ssim_threshold:.6f}. "
        f"online={online_path} offline={offline_path}"
    )
    assert psnr_score >= psnr_threshold, (
        f"PSNR below threshold: got {psnr_score:.6f}, expected >= {psnr_threshold:.6f}. "
        f"online={online_path} offline={offline_path}"
    )
    return ssim_score, psnr_score


def assert_similarity(
    *,
    model_name: str,
    vllm_image: Image.Image,
    diffusers_image: Image.Image,
    ssim_threshold: float,
    psnr_threshold: float,
    width: int | None = None,
    height: int | None = None,
    compare_mode: str = "RGB",
) -> None:
    requested_size = (width, height) if width is not None and height is not None else None
    if requested_size is not None and diffusers_image.size != requested_size:
        pytest.skip(
            "Skipping as diffusers baseline output is corrupt and not comparable: "
            f"dimensions do not match requested size; requested={requested_size}, got={diffusers_image.size}."
        )

    assert vllm_image.size == diffusers_image.size, (
        f"Online and diffusers output sizes mismatch: online={vllm_image.size}, diffusers={diffusers_image.size}"
    )

    ssim_score, psnr_score = compute_image_ssim_psnr(
        prediction=vllm_image,
        reference=diffusers_image,
        compare_mode=compare_mode,
    )
    print(f"{model_name} similarity metrics:")
    print(f"  SSIM: value={ssim_score:.6f}, threshold>={ssim_threshold:.6f}, range=[-1, 1], higher_is_better=True")
    print(
        f"  PSNR: value={psnr_score:.6f} dB, threshold>={psnr_threshold:.6f} dB, range=[0, +inf), higher_is_better=True"
    )

    assert ssim_score >= ssim_threshold, (
        f"SSIM below threshold for {model_name}: got {ssim_score:.6f}, expected >= {ssim_threshold:.6f}."
    )
    assert psnr_score >= psnr_threshold, (
        f"PSNR below threshold for {model_name}: got {psnr_score:.6f}, expected >= {psnr_threshold:.6f}."
    )


def assert_image_sequence_similarity(
    *,
    model_name: str,
    vllm_images: list[Image.Image],
    diffusers_images: list[Image.Image],
    ssim_threshold: float,
    psnr_threshold: float,
    compare_mode: str = "RGB",
) -> None:
    assert len(vllm_images) == len(diffusers_images), (
        f"Output image count mismatch for {model_name}: online={len(vllm_images)}, diffusers={len(diffusers_images)}"
    )
    for index, (vllm_image, diffusers_image) in enumerate(zip(vllm_images, diffusers_images, strict=True), start=1):
        assert_similarity(
            model_name=f"{model_name}[layer={index}]",
            vllm_image=vllm_image,
            diffusers_image=diffusers_image,
            ssim_threshold=ssim_threshold,
            psnr_threshold=psnr_threshold,
            compare_mode=compare_mode,
        )


def assert_images_pixel_close(
    *,
    model_name: str,
    vllm_image: Image.Image,
    diffusers_image: Image.Image,
    mean_threshold: float,
    p99_threshold: float,
    compare_mode: str = "RGB",
) -> None:
    """Assert full-image pixel closeness between online serving and diffusers output.

    Match the threshold style used by diffusion parallelism tests: convert both
    images to float32 RGB tensors in the [0, 1] range, then compare mean and p99
    absolute channel differences.

    Accuracy improves as both values move lower:
    - ``mean_abs_diff`` tracks global average drift across the whole image.
    - ``p99_abs_diff`` tracks tail drift while ignoring the noisiest 1% of
      channel samples.

    The printed mismatch ratios are diagnostics only. ``pixel_ratio`` counts
    pixels where any channel exceeds a tolerance, while ``channel_ratio`` counts
    individual channel samples above that tolerance.
    """
    assert vllm_image.size == diffusers_image.size, (
        f"Online and diffusers output sizes mismatch: online={vllm_image.size}, diffusers={diffusers_image.size}"
    )

    vllm_array = np.asarray(vllm_image.convert(compare_mode), dtype=np.float32) / 255.0
    diffusers_array = np.asarray(diffusers_image.convert(compare_mode), dtype=np.float32) / 255.0
    channel_abs_diff = np.abs(vllm_array - diffusers_array)
    mean_abs_diff = float(channel_abs_diff.mean())
    p99_abs_diff = float(np.quantile(channel_abs_diff, 0.99))
    percentiles = {
        percentile: float(np.quantile(channel_abs_diff, percentile / 100.0)) for percentile in (50, 90, 95, 99, 99.9)
    }

    print(f"{model_name} pixel metrics:")
    print(
        f"  mean_abs_diff: value={mean_abs_diff:.6e}, threshold<={mean_threshold:.6e}, "
        "range=[0, 1], lower_is_better=True"
    )
    print(
        f"  p99_abs_diff: value={p99_abs_diff:.6e}, threshold<={p99_threshold:.6e}, range=[0, 1], lower_is_better=True"
    )
    print("  abs_diff_percentiles:")
    for percentile, value in percentiles.items():
        print(f"    p{percentile}: value={value:.6e}, range=[0, 1], lower_is_better=True")
    print("  mismatch_ratios_by_channel_threshold:")
    for tolerance in (0, 1, 2, 4, 8, 16, 32, 64, 128):
        normalized_tolerance = tolerance / 255.0
        pixel_mismatch = np.any(channel_abs_diff > normalized_tolerance, axis=-1)
        pixel_mismatch_ratio = float(np.count_nonzero(pixel_mismatch) / pixel_mismatch.size)
        channel_mismatch_ratio = float(
            np.count_nonzero(channel_abs_diff > normalized_tolerance) / channel_abs_diff.size
        )
        print(
            f"    threshold>{normalized_tolerance:.6e} ({tolerance}/255): pixel_ratio={pixel_mismatch_ratio:.8f}, "
            f"channel_ratio={channel_mismatch_ratio:.8f}"
        )

    assert mean_abs_diff <= mean_threshold and p99_abs_diff <= p99_threshold, (
        f"Image diff exceeded threshold for {model_name}: mean_abs_diff={mean_abs_diff:.6e}, "
        f"p99_abs_diff={p99_abs_diff:.6e} (thresholds: mean<={mean_threshold:.6e}, "
        f"p99<={p99_threshold:.6e})"
    )


def compute_image_ssim_psnr(
    *,
    prediction: Image.Image,
    reference: Image.Image,
    compare_mode: str = "RGB",
) -> tuple[float, float]:
    pred_tensor = _pil_to_batched_tensor(prediction, compare_mode=compare_mode)
    ref_tensor = _pil_to_batched_tensor(reference, compare_mode=compare_mode)

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0)

    ssim_value = float(ssim_metric(pred_tensor, ref_tensor).item())
    psnr_value = float(psnr_metric(pred_tensor, ref_tensor).item())
    return ssim_value, psnr_value


class CLIPScorer:
    def __init__(self, model_name_or_path: str = "openai/clip-vit-base-patch16"):
        from transformers import CLIPModel, CLIPProcessor

        self._model = CLIPModel.from_pretrained(model_name_or_path)
        self._processor = CLIPProcessor.from_pretrained(model_name_or_path)
        self._model.eval()

    def score(self, image: Image.Image, text: str) -> float:
        inputs = self._processor(text=[text], images=[image], return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self._model(**inputs)
        img_emb = outputs.image_embeds
        txt_emb = outputs.text_embeds
        img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(p=2, dim=-1, keepdim=True)
        similarity = (img_emb * txt_emb).sum(dim=-1)
        return float(similarity.item() * 100)

    def image_image_score(self, image1: Image.Image, image2: Image.Image) -> float:
        inputs = self._processor(text=[""], images=[image1, image2], return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self._model(**inputs)
        img_emb1, img_emb2 = outputs.image_embeds.chunk(2)
        img_emb1 = img_emb1 / img_emb1.norm(p=2, dim=-1, keepdim=True)
        img_emb2 = img_emb2 / img_emb2.norm(p=2, dim=-1, keepdim=True)
        similarity = (img_emb1 * img_emb2).sum(dim=-1)
        return float(similarity.item() * 100)

    def assert_score(self, *, model_name: str, image: Image.Image, text: str, threshold: float) -> None:
        value = self.score(image, text)
        print(f"{model_name} CLIP score:")
        print(f"  CLIPScore: value={value:.4f}, threshold>={threshold:.4f}, higher_is_better=True")
        assert value >= threshold, (
            f"CLIP score below threshold for {model_name}: got {value:.4f}, expected >= {threshold:.4f}."
        )


def _pil_to_clip_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor


def _pil_to_batched_tensor(image: Image.Image, *, compare_mode: str) -> torch.Tensor:
    array = np.asarray(image.convert(compare_mode), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor


class SemanticSimilarityScorer:
    """Semantic similarity scorer for text-text and image-text comparisons"""

    def __init__(
        self,
        text_model_name: str = "BAAI/bge-m3",
    ):
        """
        Initialize semantic similarity scorer

        Args:
            text_model_name: Model for text-text similarity (BGE-M3)
        """
        self._text_model = None
        self._text_model_name = text_model_name

    def _load_text_model(self):
        """Lazy load text similarity model (BGE-M3)"""
        if self._text_model is None:
            try:
                from FlagEmbedding import BGEM3FlagModel

                print(f"Loading text similarity model: {self._text_model_name}")
                self._text_model = BGEM3FlagModel(self._text_model_name, use_fp16=True)
            except ImportError:
                raise ImportError("FlagEmbedding not installed. Install with: pip install FlagEmbedding")
        return self._text_model

    def text_similarity(self, text1: str, text2: str) -> dict[str, float | int]:
        """
        Compute semantic similarity between two texts using BGE-M3
        """
        results: dict[str, float | int] = {}
        # 1. Text prefix matching (maximum matching character length)
        match_count = 0
        for char_vllm, char_ref in zip(text1, text2):
            if char_vllm == char_ref:
                match_count += 1
            else:
                break
        results["text_prefix_match_count"] = match_count

        # 2. Full CoT semantic similarity comparison
        model = self._load_text_model()
        emb1 = model.encode([text1], return_dense=True, return_sparse=False, return_colbert_vecs=False)["dense_vecs"][0]
        emb2 = model.encode([text2], return_dense=True, return_sparse=False, return_colbert_vecs=False)["dense_vecs"][0]
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        results["cot_semantic_sim"] = float(similarity)
        return results

    def release_models(self):
        """Release loaded models to free memory"""
        if self._text_model is not None:
            del self._text_model
            self._text_model = None


def download_images(urls: list[str]) -> list[Image.Image]:
    """Download and cache multiple images"""
    images = []
    for i, url in enumerate(urls):
        print(f"Downloading image {i} from {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        images.append(image)

    return images


_DREAMZERO_UPSTREAM_SERVER_SCRIPT = (
    Path(__file__).resolve().parent / "upstream_mock" / "upstream_socket_server_no_compile.py"
)
_DREAMZERO_DEFAULT_MODEL = "GEAR-Dreams/DreamZero-DROID"
_DREAMZERO_DEFAULT_READY_TIMEOUT_S = int(os.environ.get("OPENPI_SERVICE_READY_TIMEOUT_S", "900"))
_DREAMZERO_DEFAULT_PROMPT = (
    "Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan"
)
_DREAMZERO_DEFAULT_SESSION_ID = "openpi-e2e-parity-session"
_DREAMZERO_OPENPI_VLLM_PATH = "/v1/realtime/robot/openpi"


def require_dreamzero_parity_env() -> tuple[Path, Path]:
    repo_env = os.environ.get("DREAMZERO_REPO")
    if repo_env is None:
        raise ValueError("DREAMZERO_REPO environment variable is required.")
    repo = Path(repo_env).expanduser()
    if not repo.is_dir():
        raise FileNotFoundError(f"DREAMZERO_REPO does not exist: {repo}")
    checkpoint_dir = repo / "checkpoints" / "dreamzero"
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"DreamZero checkpoint not found: {checkpoint_dir}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return repo, checkpoint_dir


def load_dreamzero_upstream_modules():
    require_dreamzero_parity_env()
    import test_client_AR as dreamzero_upstream_client
    from eval_utils.policy_client import WebsocketClientPolicy

    return dreamzero_upstream_client, WebsocketClientPolicy


def dreamzero_cfg_parallel_size() -> int:
    return int(os.environ.get("OPENPI_E2E_CFG_PARALLEL_SIZE", "2"))


def _dreamzero_parity_vllm_env() -> dict[str, str]:
    from tests.helpers.clean import pick_least_used_device_indices
    from tests.helpers.runtime import get_open_port

    parallel_size = dreamzero_cfg_parallel_size()
    gpus = [str(index) for index in pick_least_used_device_indices(parallel_size)]
    if parallel_size > len(gpus):
        raise RuntimeError(
            f"cfg_parallel_size={parallel_size} requires at least {parallel_size} GPUs, but only got {gpus}."
        )
    return {
        "CUDA_VISIBLE_DEVICES": ",".join(gpus[:parallel_size]),
        "ATTENTION_BACKEND": "torch",
        "DIFFUSION_ATTENTION_BACKEND": "TORCH_SDPA",
        "VLLM_DISABLE_COMPILE_CACHE": "1",
        "MASTER_PORT": str(get_open_port()),
    }


def dreamzero_parity_vllm_params(model: str | None = None):
    from tests.helpers.runtime import OmniServerParams
    from tests.helpers.stage_config import get_deploy_config_path

    parallel_size = dreamzero_cfg_parallel_size()
    server_args = [
        "--served-model-name",
        "dreamzero-droid",
        "--enforce-eager",
    ]
    if parallel_size > 1:
        server_args.extend(["--cfg-parallel-size", str(parallel_size)])
    resolved_model = model or os.environ.get("VLLM_DREAMZERO_MODEL") or _DREAMZERO_DEFAULT_MODEL
    return OmniServerParams(
        model=resolved_model,
        stage_config_path=get_deploy_config_path("dreamzero_tp1_cfg2.yaml"),
        server_args=server_args,
        env_dict=_dreamzero_parity_vllm_env(),
    )


def _dreamzero_omni_server_from_params(params):
    from tests.helpers.runtime import OmniServer

    server_args = list(params.server_args or [])
    if params.stage_config_path is not None:
        server_args += ["--deploy-config", params.stage_config_path]
    return OmniServer(
        params.model,
        server_args,
        port=params.port,
        env_dict=params.env_dict,
        use_omni=params.use_omni,
    )


@contextmanager
def dreamzero_parity_vllm_client(
    params=None,
    *,
    ready_timeout_s: float = _DREAMZERO_DEFAULT_READY_TIMEOUT_S,
) -> Iterator["DreamZeroParityClient"]:
    """Start vLLM via ``OmniServer`` and return a ready OpenPI parity client."""
    with _dreamzero_omni_server_from_params(params or dreamzero_parity_vllm_params()) as server:
        client = _dreamzero_wait_for_client_ready(
            lambda: _dreamzero_create_openpi_vllm_client(server.host, server.port),
            timeout_s=ready_timeout_s,
            proc=server.proc,
        )
        yield DreamZeroParityClient(client)


def _dreamzero_create_openpi_vllm_client(host: str, port: int) -> Any:
    from openpi_client import msgpack_numpy

    _, websocket_client_policy_cls = load_dreamzero_upstream_modules()

    class _OpenPIWebsocketClientPolicy(websocket_client_policy_cls):  # type: ignore[misc, valid-type]
        def __init__(
            self,
            host: str = "127.0.0.1",
            port: int = 8000,
            path: str = _DREAMZERO_OPENPI_VLLM_PATH,
        ) -> None:
            self._uri = f"ws://{host}:{port}{path}"
            self._packer = msgpack_numpy.Packer()
            self._ws, self._server_metadata = self._wait_for_server()

    return _OpenPIWebsocketClientPolicy(host=host, port=port)


def _dreamzero_stop_process(proc: subprocess.Popen[str]) -> None:
    log_file = getattr(proc, "_codex_log_file", None)
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover - cleanup path
            proc.kill()
            proc.wait(timeout=10)
    if log_file is not None:
        log_file.close()


def _dreamzero_wait_for_client_ready(
    client_factory: Callable[[], Any],
    *,
    timeout_s: float,
    proc: subprocess.Popen[str] | None = None,
    log_path: Path | None = None,
) -> Any:
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            details = ""
            if log_path is not None and log_path.exists():
                details = log_path.read_text(errors="replace")[-8000:]
            raise RuntimeError(f"Service exited before becoming ready with code {proc.returncode}.\n{details}")
        try:
            return client_factory()
        except Exception as exc:  # pragma: no cover - retry path
            last_err = exc
            time.sleep(1)
    raise TimeoutError(f"Timed out waiting for websocket service: {last_err}")


def _dreamzero_normalize_reset_response(response: Any) -> str:
    from openpi_client import msgpack_numpy

    if isinstance(response, str):
        return response
    decoded = msgpack_numpy.unpackb(response)
    if isinstance(decoded, dict):
        return str(decoded.get("status"))
    return str(decoded)


def normalize_dreamzero_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(metadata)
    if isinstance(normalized.get("image_resolution"), tuple):
        normalized["image_resolution"] = list(normalized["image_resolution"])
    return normalized


def assert_dreamzero_logs_clean(log_path: Path) -> None:
    text = log_path.read_text(errors="replace")
    if "SignalException: Process" in text and "got signal: 15" in text:
        text = text.split("Traceback (most recent call last):", 1)[0]
    assert "Traceback" not in text, text
    assert "RuntimeError:" not in text, text


def assert_dreamzero_upstream_log_matches_vllm_baseline(log_path: Path) -> None:
    text = log_path.read_text(errors="replace")
    assert "DIT Compute Steps 8 steps" not in text, text
    assert "DIT Compute Steps 16 steps" in text, text


class DreamZeroParityClient:
    """Connected OpenPI websocket client for DreamZero parity inference."""

    def __init__(self, client: Any) -> None:
        self._client = client

    @property
    def raw_client(self) -> Any:
        return self._client

    def build_obs_sequence(self) -> tuple[dict[str, Any], dict[str, Any]]:
        dreamzero_upstream_client, _ = load_dreamzero_upstream_modules()
        camera_frames = dreamzero_upstream_client.load_camera_frames()
        chunks = dreamzero_upstream_client.build_frame_schedule(
            min(v.shape[0] for v in camera_frames.values()),
            1,
        )
        obs0 = dreamzero_upstream_client._make_obs_from_video(
            camera_frames,
            [0],
            _DREAMZERO_DEFAULT_PROMPT,
            _DREAMZERO_DEFAULT_SESSION_ID,
        )
        obs1 = dreamzero_upstream_client._make_obs_from_video(
            camera_frames,
            chunks[0],
            _DREAMZERO_DEFAULT_PROMPT,
            _DREAMZERO_DEFAULT_SESSION_ID,
        )
        return obs0, obs1

    def collect_parity_outputs(self) -> tuple[dict[str, Any], list[np.ndarray]]:
        metadata = self._client.get_server_metadata()
        obs0, obs1 = self.build_obs_sequence()
        outputs = [
            self._client.infer(dict(obs0)),
            self._client.infer(dict(obs1)),
        ]
        assert _dreamzero_normalize_reset_response(self._client.reset({})) == "reset successful"
        outputs.append(self._client.infer(dict(obs0)))
        self._client._ws.close()
        return metadata, outputs


class _DreamZeroParityServer(ABC):
    host = "127.0.0.1"

    def __init__(
        self,
        *,
        log_path: Path,
        port: int | None = None,
        ready_timeout_s: float = _DREAMZERO_DEFAULT_READY_TIMEOUT_S,
    ) -> None:
        from tests.helpers.runtime import get_open_port

        self.log_path = log_path
        self.port = get_open_port() if port is None else port
        self.ready_timeout_s = ready_timeout_s
        self.proc: subprocess.Popen[str] | None = None
        self._log_file: IO[str] | None = None

    @abstractmethod
    def _build_argv(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def _build_env(self) -> dict[str, str]:
        raise NotImplementedError

    @abstractmethod
    def _create_client(self) -> Any:
        raise NotImplementedError

    def _start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self.log_path.open("w")
        self.proc = subprocess.Popen(
            self._build_argv(),
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(Path.cwd()),
            env=self._build_env(),
        )
        self.proc._codex_log_file = self._log_file  # type: ignore[attr-defined]

    def __enter__(self) -> DreamZeroParityClient:
        self._start()
        client = _dreamzero_wait_for_client_ready(
            self._create_client,
            timeout_s=self.ready_timeout_s,
            proc=self.proc,
            log_path=self.log_path,
        )
        return DreamZeroParityClient(client)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.proc is not None:
            _dreamzero_stop_process(self.proc)
            self.proc = None


class DreamZeroUpstreamServer(_DreamZeroParityServer):
    """Official DreamZero upstream reference server launched via torchrun."""

    def _build_argv(self) -> list[str]:
        _, checkpoint_dir = require_dreamzero_parity_env()
        return [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(dreamzero_cfg_parallel_size()),
            str(_DREAMZERO_UPSTREAM_SERVER_SCRIPT),
            "--port",
            str(self.port),
            "--model_path",
            str(checkpoint_dir),
        ]

    def _build_env(self) -> dict[str, str]:
        from tests.helpers.clean import pick_least_used_device_indices

        dreamzero_repo, _ = require_dreamzero_parity_env()
        env = os.environ.copy()
        env.setdefault("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{Path.cwd()}:{dreamzero_repo}:{env['PYTHONPATH']}".rstrip(":")
        env["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(index) for index in pick_least_used_device_indices(max(dreamzero_cfg_parallel_size(), 1))
        )
        env.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
        env["ATTENTION_BACKEND"] = "torch"
        env.setdefault("ENABLE_TENSORRT", "false")
        env["ENABLE_DIT_CACHE"] = "false"
        env["NUM_DIT_STEPS"] = "16"
        env["DYNAMIC_CACHE_SCHEDULE"] = "false"
        return env

    def _create_client(self) -> Any:
        _, WebsocketClientPolicy = load_dreamzero_upstream_modules()
        return WebsocketClientPolicy(host=self.host, port=self.port)
