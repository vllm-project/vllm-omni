# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E test for DreamID-Omni offline inference with Cache-Dit.

Uses the official DreamID-Omni ``test_case/twoip`` example assets instead of
synthetic inputs so the coverage matches the upstream demo structure.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
import soundfile as sf
from PIL import Image

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "1"

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion, pytest.mark.cache, pytest.mark.slow]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = REPO_ROOT / "dreamid_omni"
MODEL_PATH_ENV = "DREAMID_OMNI_MODEL_PATH"
EXAMPLE_ROOT_ENV = "DREAMID_OMNI_EXAMPLE_ROOT"
NUM_INFERENCE_STEPS = 45

CACHE_CONFIG = {
    "Fn_compute_blocks": 1,
    "Bn_compute_blocks": 0,
    "max_warmup_steps": 4,
    "max_cached_steps": 20,
    "residual_diff_threshold": 0.24,
    "max_continuous_cached_steps": 3,
    "enable_taylorseer": False,
    "taylorseer_order": 1,
    "scm_steps_mask_policy": None,
    "scm_steps_policy": "dynamic",
}


class DreamIDGenerationResult(NamedTuple):
    video: np.ndarray
    audio: np.ndarray
    fps: int
    duration_s: float
    elapsed_s: float


class DreamIDExampleCase(NamedTuple):
    prompt: str
    image_paths: tuple[Path, ...]
    audio_paths: tuple[Path, ...]


def _require_model_path() -> str:
    model_path = os.environ.get(MODEL_PATH_ENV, str(DEFAULT_MODEL_PATH))
    if not Path(model_path).exists():
        pytest.skip(f"DreamID-Omni model bundle not found at {model_path}. Set {MODEL_PATH_ENV}.")

    if importlib.util.find_spec("dreamid_omni") is None:
        pytest.skip(
            "dreamid_omni dependency package is not importable. "
            "Run examples/offline_inference/x_to_video_audio/download_dreamid_omni.py first."
        )

    return model_path


def _dreamid_repo_root_candidates() -> list[Path]:
    candidates: list[Path] = []
    example_root = os.environ.get(EXAMPLE_ROOT_ENV)
    if example_root:
        candidates.append(Path(example_root).expanduser().resolve())

    spec = importlib.util.find_spec("dreamid_omni")
    if spec is not None and spec.origin:
        candidates.append(Path(spec.origin).resolve().parents[1] / "test_case" / "twoip")

    return candidates


@lru_cache(maxsize=1)
def _load_example_case() -> DreamIDExampleCase:
    for candidate_root in _dreamid_repo_root_candidates():
        caption_path = candidate_root / "captions" / "20.json"
        image_paths = (
            candidate_root / "imgs" / "20" / "0.png",
            candidate_root / "imgs" / "20" / "1.png",
        )
        audio_paths = (
            candidate_root / "audios" / "20" / "0.wav",
            candidate_root / "audios" / "20" / "1.wav",
        )
        if caption_path.exists() and all(path.exists() for path in image_paths + audio_paths):
            prompt = _load_prompt_from_json(caption_path)
            return DreamIDExampleCase(
                prompt=prompt,
                image_paths=image_paths,
                audio_paths=audio_paths,
            )

    searched = ", ".join(str(path) for path in _dreamid_repo_root_candidates()) or "<none>"
    pytest.skip(
        "DreamID-Omni official example assets not found. "
        f"Set {EXAMPLE_ROOT_ENV} to the local 'test_case/twoip' directory. "
        f"Searched: {searched}"
    )


def _load_prompt_from_json(path: Path) -> str:
    raw_text = path.read_text(encoding="utf-8").strip()
    parsed = json.loads(raw_text)
    if isinstance(parsed, str):
        prompt = parsed
    elif isinstance(parsed, dict) and isinstance(parsed.get("content"), str):
        prompt = parsed["content"]
    else:
        raise AssertionError(f"Unexpected DreamID caption format in {path}")

    prompt = re.sub(
        r"\[SPEAKER_TIMESTAMPS_START\].*?\[SPEAKER_TIMESTAMPS_END\]",
        "",
        prompt,
        flags=re.DOTALL,
    ).strip()
    prompt = re.sub(
        r"\[AUDIO_DESCRIPTION_START].*?\[AUDIO_DESCRIPTION_END]",
        "",
        prompt,
        flags=re.DOTALL,
    ).strip()
    prompt = re.sub(r"\[[A-Z_]+\]", "", prompt)
    prompt = re.sub(r"\n\s*\n", "\n", prompt).strip()
    return prompt


def _load_example_images(example_case: DreamIDExampleCase) -> list[Image.Image]:
    images: list[Image.Image] = []
    for path in example_case.image_paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


def _load_example_audios(example_case: DreamIDExampleCase) -> list[np.ndarray]:
    audios: list[np.ndarray] = []
    for path in example_case.audio_paths:
        audio_array, sample_rate = sf.read(str(path), dtype="float32")
        audio_np = np.asarray(audio_array, dtype=np.float32)
        if audio_np.ndim == 2:
            audio_np = np.mean(audio_np, axis=1)
        audio_np = audio_np[int(sample_rate * 1) : int(sample_rate * 3)]
        audios.append(audio_np.reshape(-1))
    return audios


def _sampling_params(num_inference_steps: int) -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        height=704,
        width=1280,
        num_inference_steps=num_inference_steps,
        seed=42,
        extra_args={
            "solver_name": "unipc",
            "shift": 5.0,
        },
    )


def _request_payload(example_case: DreamIDExampleCase) -> dict[str, object]:
    return {
        "prompt": example_case.prompt,
        "video_negative_prompt": "jitter, bad hands, blur, distortion",
        "audio_negative_prompt": "robotic, muffled, echo, distorted",
        "multi_modal_data": {
            "image": _load_example_images(example_case),
            "audio": _load_example_audios(example_case),
        },
    }


def _extract_generation(outputs: list[object], elapsed_s: float) -> DreamIDGenerationResult:
    """Extract video and audio from raw Omni outputs"""
    assert outputs, "DreamID-Omni returned no outputs"

    result = outputs[0]
    assert hasattr(result, "images") and result.images, "not return video frames"

    video = result.images[0]
    assert isinstance(video, np.ndarray), f"Expected video ndarray, got {type(video)}"
    assert video.ndim == 4, f"Expected 4D DreamID video tensor, got {video.shape}"

    multimodal_output = getattr(result, "multimodal_output", None) or {}
    assert "audio" in multimodal_output, "not return audio"
    audio = np.asarray(multimodal_output["audio"], dtype=np.float32).reshape(-1)
    assert audio.size > 0, "returned empty audio"

    audio_sample_rate = int(multimodal_output.get("audio_sample_rate", 16000))
    fps = multimodal_output.get("fps", 24)
    duration_s = float(audio.size) / float(audio_sample_rate)

    generation = DreamIDGenerationResult(
        video=video.astype(np.float32, copy=False),
        audio=audio,
        fps=fps,
        duration_s=duration_s,
        elapsed_s=elapsed_s,
    )
    return generation


def _run_generation(
    *,
    model_path: str,
    example_case: DreamIDExampleCase,
    use_cache_dit: bool,
) -> DreamIDGenerationResult:
    runner_kwargs = {
        "model_type": "dreamid-omni",
        "parallel_config": DiffusionParallelConfig(cfg_parallel_size=1),
    }
    if use_cache_dit:
        runner_kwargs["cache_backend"] = "cache_dit"
        runner_kwargs["cache_config"] = CACHE_CONFIG

    with OmniRunner(model_path, **runner_kwargs) as runner:
        warmup_payload = _request_payload(example_case)
        _ = runner.omni.generate(warmup_payload, _sampling_params(2))

        elapsed_samples: list[float] = []
        extracted_result: DreamIDGenerationResult | None = None

        request_payload = _request_payload(example_case)
        start = time.perf_counter()
        outputs = runner.omni.generate(request_payload, _sampling_params(NUM_INFERENCE_STEPS))
        elapsed_s = time.perf_counter() - start
        elapsed_samples.append(elapsed_s)
        if extracted_result is None:
            extracted_result = _extract_generation(outputs, elapsed_s)

        assert extracted_result is not None
        median_elapsed_s = float(np.median(np.asarray(elapsed_samples, dtype=np.float64)))
        return extracted_result._replace(elapsed_s=median_elapsed_s)


def _video_mae(a: np.ndarray, b: np.ndarray) -> float:
    assert a.shape == b.shape, f"Video shapes differ: {a.shape} vs {b.shape}"
    return float(np.mean(np.abs(a - b)))


def _audio_rms(audio: np.ndarray) -> float:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    return float(np.sqrt(np.mean(audio**2)))


def _audio_rms_ratio(a: DreamIDGenerationResult, b: DreamIDGenerationResult) -> float:
    rms_a = _audio_rms(a.audio)
    rms_b = _audio_rms(b.audio)
    return rms_b / max(rms_a, 1e-6)


@hardware_test(res={"cuda": "H100"})
def test_dreamid_omni_cache_dit_e2e() -> None:
    model_path = _require_model_path()
    example_case = _load_example_case()

    baseline = _run_generation(model_path=model_path, example_case=example_case, use_cache_dit=False)
    cached = _run_generation(model_path=model_path, example_case=example_case, use_cache_dit=True)

    video_mae = _video_mae(baseline.video, cached.video)
    audio_rms_ratio = _audio_rms_ratio(baseline, cached)
    speedup = baseline.elapsed_s / cached.elapsed_s if cached.elapsed_s > 0 else float("inf")

    assert video_mae <= 0.15, f"video drift too much: {video_mae:.6f}"
    assert 0.4 <= audio_rms_ratio <= 2.5, f"audio quality regressed too much: rms_ratio={audio_rms_ratio:.6f}"
    assert speedup >= 1.05, f"cache-dit improve not enough: {speedup:.3f}x"
