# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E accuracy guard against a pinned Lightricks LTX pipeline revision.

The comparison runs both runtimes through PyTorch SDPA and uses
``max_batch_size=4`` in the official reference to match Omni's fused guidance
batch. Video and audio guidance use the official non-HQ one-stage defaults;
only the generation shape and step count are reduced for CI runtime.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from tests.e2e.accuracy.helpers import reset_artifact_dir
from tests.helpers.mark import hardware_test

OFFICIAL_REPOSITORY = "https://github.com/Lightricks/LTX-2.git"
OFFICIAL_REVISION = "9377758131b1ffde4b7f766804590a6617bf2ab9"
# Version selected by this revision's uv.lock. Keep it out of Omni's runtime and dev dependencies.
OFFICIAL_OPENIMAGEIO_VERSION = "3.1.11.0"
PROMPT = (
    "A space shuttle launches vertically above a desert launch pad. Bright exhaust flames and a dense white "
    "plume billow beneath it while the camera remains fixed."
)
NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, "
    "background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting "
    "direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, "
    "incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, "
    "silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, incorrect "
    "dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural "
    "transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, cinematic oversaturation, "
    "stylized filters, or AI artifacts."
)

# Both runtimes use PyTorch SDPA with the current Torch dispatch defaults.
ATTENTION_BACKEND = "torch_sdpa"
VIDEO_SSIM_MEAN_THRESHOLD = 0.95
VIDEO_SSIM_MIN_THRESHOLD = 0.90
VIDEO_PSNR_MEAN_THRESHOLD = 30.0
AUDIO_RELATIVE_L2_THRESHOLD = 0.2
AUDIO_COSINE_THRESHOLD = 0.95


@dataclass(frozen=True)
class LTXAccuracyCase:
    name: str
    model_id: str
    model_revision: str
    model_env: str
    model_class_name: str
    checkpoint_repo: str
    checkpoint_filename: str
    checkpoint_revision: str
    checkpoint_env: str
    stg_block: int
    prompt: str = PROMPT
    image_repo: str | None = None
    image_filename: str | None = None
    image_revision: str | None = None


CASES = (
    LTXAccuracyCase(
        name="ltx2",
        model_id="Lightricks/LTX-2",
        model_revision="47da56e2ad66ce4125a9922b4a8826bf407f9d0a",
        model_env="VLLM_TEST_LTX2_MODEL",
        model_class_name="LTX2Pipeline",
        checkpoint_repo="Lightricks/LTX-2",
        checkpoint_filename="ltx-2-19b-dev.safetensors",
        checkpoint_revision="47da56e2ad66ce4125a9922b4a8826bf407f9d0a",
        checkpoint_env="VLLM_TEST_LTX2_OFFICIAL_CHECKPOINT",
        stg_block=29,
    ),
    LTXAccuracyCase(
        name="ltx2_3",
        model_id="diffusers/LTX-2.3-Diffusers",
        model_revision="8eee8edcf067e838b843f926ec4d4cc9b2be1aaf",
        model_env="VLLM_TEST_LTX23_MODEL",
        model_class_name="LTX2Pipeline",
        checkpoint_repo="Lightricks/LTX-2.3",
        checkpoint_filename="ltx-2.3-22b-dev.safetensors",
        checkpoint_revision="4229404625088d21c4f112eb640fb04a0900ee25",
        checkpoint_env="VLLM_TEST_LTX23_OFFICIAL_CHECKPOINT",
        stg_block=28,
    ),
    LTXAccuracyCase(
        name="ltx2_3_i2v",
        model_id="diffusers/LTX-2.3-Diffusers",
        model_revision="8eee8edcf067e838b843f926ec4d4cc9b2be1aaf",
        model_env="VLLM_TEST_LTX23_MODEL",
        model_class_name="LTX2Pipeline",
        checkpoint_repo="Lightricks/LTX-2.3",
        checkpoint_filename="ltx-2.3-22b-dev.safetensors",
        checkpoint_revision="4229404625088d21c4f112eb640fb04a0900ee25",
        checkpoint_env="VLLM_TEST_LTX23_OFFICIAL_CHECKPOINT",
        stg_block=28,
        image_repo="huggingface/documentation-images",
        image_filename="diffusers/svd/rocket.png",
        image_revision="645d8364f0c7a101180b364811b5a11a362e4010",
    ),
)


def _run(command: list[str], *, env: dict[str, str], timeout: int = 1800) -> None:
    start = time.perf_counter()
    subprocess.run(command, env=env, timeout=timeout, check=True)
    print(f"{' '.join(command[:3])} finished in {time.perf_counter() - start:.1f}s")


def _clone_official_source(root: Path, revision: str) -> None:
    root.mkdir(parents=True)
    repository = os.environ.get("VLLM_TEST_LTX_OFFICIAL_REPOSITORY", OFFICIAL_REPOSITORY)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "remote", "add", "origin", repository], check=True)
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(3):
        try:
            subprocess.run(
                ["git", "-C", str(root), "fetch", "--depth", "1", "origin", revision],
                check=True,
            )
            last_error = None
            break
        except subprocess.CalledProcessError as error:
            last_error = error
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
    if last_error is not None:
        raise last_error
    subprocess.run(["git", "-C", str(root), "checkout", "-q", "--detach", "FETCH_HEAD"], check=True)


def _git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return None


def _official_source(artifact_root: Path) -> tuple[Path, str]:
    revision = os.environ.get("VLLM_TEST_LTX_OFFICIAL_REVISION", OFFICIAL_REVISION)
    configured_root = os.environ.get("VLLM_TEST_LTX_OFFICIAL_ROOT")
    root = Path(configured_root) if configured_root else artifact_root / f"official-source-{revision[:12]}"
    actual_revision = _git_revision(root) if root.exists() else None
    if actual_revision != revision and configured_root:
        raise AssertionError(f"Official source revision mismatch: {actual_revision} != {revision}")
    if actual_revision != revision:
        if root.exists():
            shutil.rmtree(root)
        _clone_official_source(root, revision)
        actual_revision = _git_revision(root)
    assert actual_revision == revision, f"Official source revision mismatch: {actual_revision} != {revision}"
    return root, revision


def _official_runner_prefix() -> list[str]:
    """Run the reference with its missing binary dependency isolated from CI."""
    uv = shutil.which("uv")
    assert uv is not None, "uv is required to run the pinned official LTX reference"
    return [
        uv,
        "run",
        "--no-project",
        "--with",
        f"openimageio=={OFFICIAL_OPENIMAGEIO_VERSION}",
        "--python",
        sys.executable,
        "python",
    ]


def _resolve_model(case: LTXAccuracyCase) -> Path:
    configured_model = os.environ.get(case.model_env)
    if configured_model and Path(configured_model).exists():
        return Path(configured_model)
    model_id = configured_model or case.model_id
    revision = os.environ.get(f"{case.model_env}_REVISION")
    if revision is None and model_id == case.model_id:
        revision = case.model_revision
    return Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            allow_patterns=[
                "model_index.json",
                "audio_vae/*",
                "connectors/*",
                "processor/*",
                "scheduler/*",
                "text_encoder/config.json",
                "text_encoder/generation_config.json",
                "text_encoder/model*",
                "tokenizer/*",
                "transformer/*",
                "vae/*",
                "vocoder/*",
            ],
        )
    )


def _resolve_gemma_root(model: Path) -> Path:
    configured_root = os.environ.get("VLLM_TEST_LTX_GEMMA_ROOT")
    if configured_root:
        root = Path(configured_root)
        assert root.is_dir(), f"Gemma root not found: {root}"
        return root
    return model


def _resolve_checkpoint(case: LTXAccuracyCase, model: Path) -> Path:
    configured_checkpoint = os.environ.get(case.checkpoint_env)
    if configured_checkpoint:
        checkpoint = Path(configured_checkpoint)
        assert checkpoint.is_file(), f"Official checkpoint not found: {checkpoint}"
        return checkpoint
    model_checkpoint = model / case.checkpoint_filename
    if model_checkpoint.is_file():
        return model_checkpoint
    return Path(
        hf_hub_download(
            repo_id=case.checkpoint_repo,
            filename=case.checkpoint_filename,
            revision=case.checkpoint_revision,
        )
    )


def _resolve_image(case: LTXAccuracyCase) -> Path | None:
    if case.image_filename is None:
        return None
    if case.image_repo is None or case.image_revision is None:
        raise ValueError(f"Incomplete image source for LTX accuracy case {case.name!r}.")
    configured_image = os.environ.get("VLLM_TEST_LTX_I2V_IMAGE")
    if configured_image:
        image = Path(configured_image)
        assert image.is_file(), f"LTX I2V conditioning image not found: {image}"
        return image
    return Path(
        hf_hub_download(
            repo_id=case.image_repo,
            repo_type="dataset",
            filename=case.image_filename,
            revision=case.image_revision,
        )
    )


def _request(case: LTXAccuracyCase, image: Path | None) -> dict[str, object]:
    request: dict[str, object] = {
        "prompt": case.prompt,
        "negative_prompt": NEGATIVE_PROMPT,
        "width": 512,
        "height": 384,
        "num_frames": 25,
        "fps": 24,
        "num_inference_steps": 20,
        "seed": 42,
        "video_cfg_scale": 3.0,
        "audio_cfg_scale": 7.0,
        "video_stg_scale": 1.0,
        "audio_stg_scale": 1.0,
        "video_modality_scale": 3.0,
        "audio_modality_scale": 3.0,
        "video_rescale_scale": 0.7,
        "audio_rescale_scale": 0.7,
        "video_stg_blocks": [case.stg_block],
        "audio_stg_blocks": [case.stg_block],
    }
    if image is not None:
        request["image"] = str(image.resolve())
    return request


def _video_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    assert reference.shape == prediction.shape
    assert reference.ndim == 4 and reference.shape[-1] == 3
    ssim_scores: list[float] = []
    psnr_scores: list[float] = []
    for reference_frame, prediction_frame in zip(reference, prediction, strict=True):
        reference_tensor = torch.from_numpy(reference_frame).permute(2, 0, 1).unsqueeze(0)
        prediction_tensor = torch.from_numpy(prediction_frame).permute(2, 0, 1).unsqueeze(0)
        ssim_scores.append(float(StructuralSimilarityIndexMeasure(data_range=1.0)(prediction_tensor, reference_tensor)))
        psnr_scores.append(float(PeakSignalNoiseRatio(data_range=1.0)(prediction_tensor, reference_tensor)))
    difference = np.abs(reference.astype(np.float64) - prediction.astype(np.float64))
    return {
        "ssim_mean": float(np.mean(ssim_scores)),
        "ssim_min": float(np.min(ssim_scores)),
        "psnr_mean_db": float(np.mean(psnr_scores)),
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
    }


def _canonical_audio(audio: np.ndarray) -> np.ndarray:
    while audio.ndim > 2 and audio.shape[0] == 1:
        audio = audio[0]
    return audio.astype(np.float64)


def _audio_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float | bool]:
    reference = _canonical_audio(reference)
    prediction = _canonical_audio(prediction)
    assert reference.shape == prediction.shape
    difference = reference - prediction
    reference_norm = max(float(np.linalg.norm(reference)), 1e-12)
    prediction_norm = max(float(np.linalg.norm(prediction)), 1e-12)
    return {
        "bitwise_equal": bool(np.array_equal(reference, prediction)),
        "max_abs": float(np.abs(difference).max()),
        "mean_abs": float(np.abs(difference).mean()),
        "relative_l2": float(np.linalg.norm(difference) / reference_norm),
        "cosine_similarity": float(np.vdot(reference.ravel(), prediction.ravel()) / (reference_norm * prediction_norm)),
    }


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@pytest.mark.full_model
@pytest.mark.benchmark
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_ltx_one_stage_matches_official(case: LTXAccuracyCase, accuracy_artifact_root: Path) -> None:
    """Compare official and Omni raw AV outputs from the same E2E request."""
    output_root = reset_artifact_dir(accuracy_artifact_root / "ltx_official" / case.name)
    official_root, official_revision = _official_source(accuracy_artifact_root / "ltx_official")
    model = _resolve_model(case)
    gemma_root = _resolve_gemma_root(model)
    checkpoint = _resolve_checkpoint(case, model)
    image = _resolve_image(case)
    request_path = output_root / "request.json"
    request_path.write_text(json.dumps(_request(case, image), indent=2) + "\n")

    runner = Path(__file__).with_name("run_ltx_reference.py")
    runner_args = [
        str(runner),
        "--request",
        str(request_path),
    ]
    if os.environ.get("VLLM_TEST_LTX_ENABLE_LAYERWISE_OFFLOAD", "").lower() in {"1", "true", "yes", "on"}:
        runner_args.append("--enable-layerwise-offload")
    env = os.environ.copy()
    env["VLLM_TEST_LTX_OFFICIAL_REVISION"] = official_revision
    env["PYTHONUNBUFFERED"] = "1"
    repository_root = Path(__file__).resolve().parents[4]
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(repository_root) if not existing_pythonpath else f"{repository_root}{os.pathsep}{existing_pythonpath}"
    )

    official_output = output_root / "official"
    _run(
        _official_runner_prefix()
        + runner_args
        + [
            "--backend",
            "official",
            "--output-dir",
            str(official_output),
            "--official-root",
            str(official_root),
            "--checkpoint",
            str(checkpoint),
            "--gemma-root",
            str(gemma_root),
        ],
        env=env,
    )

    omni_output = output_root / "omni"
    _run(
        [sys.executable]
        + runner_args
        + [
            "--backend",
            "omni",
            "--output-dir",
            str(omni_output),
            "--model",
            str(model),
            "--model-class-name",
            case.model_class_name,
        ],
        env=env,
    )

    official_metadata = json.loads((official_output / "metadata.json").read_text())
    omni_metadata = json.loads((omni_output / "metadata.json").read_text())
    assert official_metadata["attention_backend"] == ATTENTION_BACKEND
    assert omni_metadata["attention_backend"] == ATTENTION_BACKEND
    assert official_metadata["audio_sample_rate"] == omni_metadata["audio_sample_rate"]
    video_metrics = _video_metrics(
        np.load(official_output / "video.npy"),
        np.load(omni_output / "video.npy"),
    )
    audio_metrics = _audio_metrics(
        np.load(official_output / "audio.npy"),
        np.load(omni_output / "audio.npy"),
    )
    result = {
        "case": case.name,
        "task": "i2v" if image is not None else "t2v",
        "attention_backend": ATTENTION_BACKEND,
        "official_revision": official_revision,
        "model_revision": case.model_revision,
        "checkpoint_revision": case.checkpoint_revision,
        "video": video_metrics,
        "audio": audio_metrics,
    }
    (output_root / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

    assert video_metrics["ssim_mean"] >= VIDEO_SSIM_MEAN_THRESHOLD
    assert video_metrics["ssim_min"] >= VIDEO_SSIM_MIN_THRESHOLD
    assert video_metrics["psnr_mean_db"] >= VIDEO_PSNR_MEAN_THRESHOLD
    assert audio_metrics["relative_l2"] <= AUDIO_RELATIVE_L2_THRESHOLD
    assert audio_metrics["cosine_similarity"] >= AUDIO_COSINE_THRESHOLD
