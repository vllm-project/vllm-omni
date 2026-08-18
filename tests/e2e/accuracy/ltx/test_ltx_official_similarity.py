# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E accuracy guard against a pinned Lightricks LTX pipeline revision.

The original reduced one-stage guards remain unchanged. Four complementary
default-shape cases cover both model versions, both two-stage weight families,
and T2V/I2V without expanding every combination into a duplicate golden.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

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


@dataclass(frozen=True)
class LTXArtifact:
    repo_id: str
    filename: str
    revision: str
    env: str
    repo_type: Literal["model", "dataset"] = "model"


@dataclass(frozen=True)
class LTXAccuracyThresholds:
    video_ssim_mean: float
    video_ssim_min: float
    video_psnr_mean_db: float
    # Zero disables gating for waveform-sensitive audio that is not expected
    # to align with the official output.
    audio_relative_l2: float
    audio_cosine_similarity: float


STRICT_THRESHOLDS = LTXAccuracyThresholds(
    video_ssim_mean=0.95,
    video_ssim_min=0.90,
    video_psnr_mean_db=30.0,
    audio_relative_l2=0.20,
    audio_cosine_similarity=0.95,
)


@dataclass(frozen=True)
class LTXAccuracyCase:
    name: str
    pipeline_kind: Literal["one_stage", "distilled", "two_stage"]
    model_id: str
    model_revision: str
    model_env: str
    model_class_name: str
    checkpoint: LTXArtifact
    width: int
    height: int
    num_frames: int
    num_inference_steps: int
    seed: int
    stg_block: int | None
    thresholds: LTXAccuracyThresholds
    prompt: str = PROMPT
    gemma_model_id: str | None = None
    gemma_model_revision: str | None = None
    gemma_model_env: str | None = None
    image: LTXArtifact | None = None
    spatial_upsampler: LTXArtifact | None = None
    distilled_lora: LTXArtifact | None = None
    # Official uses block streaming; Omni uses layerwise offload. Both preserve
    # bf16 arithmetic while keeping full-resolution two-stage cases on one H100.
    enable_layerwise_offload: bool = False


LTX2_REVISION = "47da56e2ad66ce4125a9922b4a8826bf407f9d0a"
LTX23_REVISION = "4229404625088d21c4f112eb640fb04a0900ee25"
LTX2_CHECKPOINT = LTXArtifact(
    repo_id="Lightricks/LTX-2",
    filename="ltx-2-19b-dev.safetensors",
    revision=LTX2_REVISION,
    env="VLLM_TEST_LTX2_OFFICIAL_CHECKPOINT",
)
LTX2_DISTILLED_CHECKPOINT = replace(
    LTX2_CHECKPOINT,
    filename="ltx-2-19b-distilled.safetensors",
    env="VLLM_TEST_LTX2_DISTILLED_OFFICIAL_CHECKPOINT",
)
LTX2_UPSAMPLER = replace(
    LTX2_CHECKPOINT,
    filename="ltx-2-spatial-upscaler-x2-1.0.safetensors",
    env="VLLM_TEST_LTX2_UPSAMPLER",
)
LTX2_DISTILLED_LORA = replace(
    LTX2_CHECKPOINT,
    filename="ltx-2-19b-distilled-lora-384.safetensors",
    env="VLLM_TEST_LTX2_DISTILLED_LORA",
)
LTX23_CHECKPOINT = LTXArtifact(
    repo_id="Lightricks/LTX-2.3",
    filename="ltx-2.3-22b-dev.safetensors",
    revision=LTX23_REVISION,
    env="VLLM_TEST_LTX23_OFFICIAL_CHECKPOINT",
)
LTX23_DISTILLED_CHECKPOINT = replace(
    LTX23_CHECKPOINT,
    # The pinned Diffusers checkpoint contains the original merged distilled
    # Transformer. Keep the reference Transformer on the same version; the
    # spatial upsampler is versioned independently and remains on 1.1 below.
    filename="ltx-2.3-22b-distilled.safetensors",
    env="VLLM_TEST_LTX23_DISTILLED_OFFICIAL_CHECKPOINT",
)
LTX23_UPSAMPLER = replace(
    LTX23_CHECKPOINT,
    filename="ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    env="VLLM_TEST_LTX23_UPSAMPLER",
)
LTX23_DISTILLED_LORA = replace(
    LTX23_CHECKPOINT,
    filename="ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
    env="VLLM_TEST_LTX23_DISTILLED_LORA",
)
I2V_IMAGE = LTXArtifact(
    repo_id="huggingface/documentation-images",
    filename="diffusers/svd/rocket.png",
    revision="645d8364f0c7a101180b364811b5a11a362e4010",
    env="VLLM_TEST_LTX_I2V_IMAGE",
    repo_type="dataset",
)


LEGACY_CASES = (
    LTXAccuracyCase(
        name="ltx2",
        pipeline_kind="one_stage",
        model_id="Lightricks/LTX-2",
        model_revision=LTX2_REVISION,
        model_env="VLLM_TEST_LTX2_MODEL",
        model_class_name="LTX2Pipeline",
        checkpoint=LTX2_CHECKPOINT,
        width=512,
        height=384,
        num_frames=25,
        num_inference_steps=20,
        seed=42,
        stg_block=29,
        thresholds=STRICT_THRESHOLDS,
    ),
    LTXAccuracyCase(
        name="ltx2_3",
        pipeline_kind="one_stage",
        model_id="diffusers/LTX-2.3-Diffusers",
        model_revision="8eee8edcf067e838b843f926ec4d4cc9b2be1aaf",
        model_env="VLLM_TEST_LTX23_MODEL",
        model_class_name="LTX2Pipeline",
        checkpoint=LTX23_CHECKPOINT,
        width=512,
        height=384,
        num_frames=25,
        num_inference_steps=20,
        seed=42,
        stg_block=28,
        thresholds=STRICT_THRESHOLDS,
    ),
    LTXAccuracyCase(
        name="ltx2_3_i2v",
        pipeline_kind="one_stage",
        model_id="diffusers/LTX-2.3-Diffusers",
        model_revision="8eee8edcf067e838b843f926ec4d4cc9b2be1aaf",
        model_env="VLLM_TEST_LTX23_MODEL",
        model_class_name="LTX2Pipeline",
        checkpoint=LTX23_CHECKPOINT,
        width=512,
        height=384,
        num_frames=25,
        num_inference_steps=20,
        seed=42,
        stg_block=28,
        thresholds=STRICT_THRESHOLDS,
        image=I2V_IMAGE,
    ),
)


TWO_STAGE_CASES = (
    LTXAccuracyCase(
        name="ltx2_distilled_t2v",
        pipeline_kind="distilled",
        model_id="rootonchair/LTX-2-19b-distilled",
        model_revision="388e2846f54aae51687498ffb6b27c7c2c9ce9e5",
        model_env="VLLM_TEST_LTX2_DISTILLED_MODEL",
        model_class_name="LTX2DistilledPipeline",
        checkpoint=LTX2_DISTILLED_CHECKPOINT,
        spatial_upsampler=LTX2_UPSAMPLER,
        width=1536,
        height=1024,
        num_frames=121,
        num_inference_steps=8,
        seed=10,
        stg_block=None,
        gemma_model_id="Lightricks/LTX-2",
        gemma_model_revision=LTX2_REVISION,
        gemma_model_env="VLLM_TEST_LTX2_MODEL",
        thresholds=STRICT_THRESHOLDS,
    ),
    LTXAccuracyCase(
        name="ltx23_distilled_i2v",
        pipeline_kind="distilled",
        model_id="diffusers/LTX-2.3-Distilled-Diffusers",
        model_revision="432e0d3c2d1769aaa4d295f9243f7062bf6b47ee",
        model_env="VLLM_TEST_LTX23_DISTILLED_MODEL",
        model_class_name="LTX2DistilledPipeline",
        checkpoint=LTX23_DISTILLED_CHECKPOINT,
        spatial_upsampler=LTX23_UPSAMPLER,
        width=1536,
        height=1024,
        num_frames=121,
        num_inference_steps=8,
        seed=10,
        stg_block=None,
        gemma_model_id="diffusers/LTX-2.3-Diffusers",
        gemma_model_revision="8eee8edcf067e838b843f926ec4d4cc9b2be1aaf",
        gemma_model_env="VLLM_TEST_LTX23_MODEL",
        thresholds=STRICT_THRESHOLDS,
        image=I2V_IMAGE,
    ),
    LTXAccuracyCase(
        name="ltx2_two_stage_layer_fused_t2v",
        pipeline_kind="two_stage",
        model_id="Lightricks/LTX-2",
        model_revision=LTX2_REVISION,
        model_env="VLLM_TEST_LTX2_MODEL",
        model_class_name="LTX2TwoStagePipeline",
        checkpoint=LTX2_CHECKPOINT,
        spatial_upsampler=LTX2_UPSAMPLER,
        distilled_lora=LTX2_DISTILLED_LORA,
        width=1536,
        height=1024,
        num_frames=121,
        num_inference_steps=40,
        seed=10,
        stg_block=29,
        enable_layerwise_offload=True,
        thresholds=STRICT_THRESHOLDS,
    ),
    LTXAccuracyCase(
        name="ltx23_two_stage_layer_fused_i2v",
        pipeline_kind="two_stage",
        model_id="diffusers/LTX-2.3-Diffusers",
        model_revision="8eee8edcf067e838b843f926ec4d4cc9b2be1aaf",
        model_env="VLLM_TEST_LTX23_MODEL",
        model_class_name="LTX2TwoStagePipeline",
        checkpoint=LTX23_CHECKPOINT,
        spatial_upsampler=LTX23_UPSAMPLER,
        distilled_lora=LTX23_DISTILLED_LORA,
        width=1536,
        height=1024,
        num_frames=121,
        num_inference_steps=30,
        seed=10,
        stg_block=28,
        enable_layerwise_offload=True,
        thresholds=STRICT_THRESHOLDS,
        image=I2V_IMAGE,
    ),
)

CASES = (*LEGACY_CASES, *TWO_STAGE_CASES)


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
                "latent_upsampler/*",
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


def _resolve_gemma_root(case: LTXAccuracyCase, model: Path) -> Path:
    configured_root = os.environ.get("VLLM_TEST_LTX_GEMMA_ROOT")
    if configured_root:
        root = Path(configured_root)
        assert root.is_dir(), f"Gemma root not found: {root}"
        return root
    if case.gemma_model_id is not None:
        configured_model = os.environ.get(case.gemma_model_env or "")
        if configured_model and Path(configured_model).is_dir():
            return Path(configured_model)
        return Path(
            snapshot_download(
                repo_id=case.gemma_model_id,
                revision=case.gemma_model_revision,
                allow_patterns=[
                    "processor/*",
                    "text_encoder/*",
                    "tokenizer/*",
                ],
            )
        )
    return model


def _resolve_artifact(artifact: LTXArtifact, model: Path | None = None) -> Path:
    configured_path = os.environ.get(artifact.env)
    if configured_path:
        path = Path(configured_path)
        assert path.is_file(), f"Configured LTX artifact not found: {path}"
        return path
    if model is not None:
        model_path = model / artifact.filename
        if model_path.is_file():
            return model_path
    return Path(
        hf_hub_download(
            repo_id=artifact.repo_id,
            repo_type=None if artifact.repo_type == "model" else artifact.repo_type,
            filename=artifact.filename,
            revision=artifact.revision,
        )
    )


def _resolve_image(case: LTXAccuracyCase) -> Path | None:
    if case.image is None:
        return None
    return _resolve_artifact(case.image)


def _request(case: LTXAccuracyCase, image: Path | None) -> dict[str, object]:
    request: dict[str, object] = {
        "pipeline_kind": case.pipeline_kind,
        "prompt": case.prompt,
        "negative_prompt": "" if case.pipeline_kind == "distilled" else NEGATIVE_PROMPT,
        "width": case.width,
        "height": case.height,
        "num_frames": case.num_frames,
        "fps": 24,
        "num_inference_steps": case.num_inference_steps,
        "seed": case.seed,
    }
    if case.pipeline_kind != "distilled":
        assert case.stg_block is not None
        request.update(
            video_cfg_scale=3.0,
            audio_cfg_scale=7.0,
            video_stg_scale=1.0,
            audio_stg_scale=1.0,
            video_modality_scale=3.0,
            audio_modality_scale=3.0,
            video_rescale_scale=0.7,
            audio_rescale_scale=0.7,
            video_stg_blocks=[case.stg_block],
            audio_stg_blocks=[case.stg_block],
        )
    if image is not None:
        request["image"] = str(image.resolve())
    return request


def _video_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    assert reference.shape == prediction.shape
    assert reference.ndim == 4 and reference.shape[-1] == 3
    device = torch.device("cpu")
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_scores: list[float] = []
    psnr_scores: list[float] = []
    max_abs = 0.0
    absolute_error_sum = 0.0
    with torch.inference_mode():
        for reference_frame, prediction_frame in zip(reference, prediction, strict=True):
            reference_tensor = torch.from_numpy(reference_frame).permute(2, 0, 1).unsqueeze(0).to(device)
            prediction_tensor = torch.from_numpy(prediction_frame).permute(2, 0, 1).unsqueeze(0).to(device)
            ssim_scores.append(float(ssim(prediction_tensor, reference_tensor)))
            psnr_scores.append(float(psnr(prediction_tensor, reference_tensor)))
            ssim.reset()
            psnr.reset()
            difference = np.abs(reference_frame.astype(np.float64) - prediction_frame.astype(np.float64))
            max_abs = max(max_abs, float(difference.max()))
            absolute_error_sum += float(difference.sum())
    return {
        "ssim_mean": float(np.mean(ssim_scores)),
        "ssim_min": float(np.min(ssim_scores)),
        "psnr_mean_db": float(np.mean(psnr_scores)),
        "max_abs": max_abs,
        "mean_abs": absolute_error_sum / reference.size,
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
@pytest.mark.slow
@pytest.mark.benchmark
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_ltx_matches_official(case: LTXAccuracyCase, accuracy_artifact_root: Path) -> None:
    """Compare official and Omni raw AV outputs from the same single-H100 request."""
    configured_artifact_root = os.environ.get("VLLM_TEST_LTX_ARTIFACT_ROOT")
    if configured_artifact_root:
        accuracy_artifact_root = Path(configured_artifact_root)
        accuracy_artifact_root.mkdir(parents=True, exist_ok=True)
    output_root = reset_artifact_dir(accuracy_artifact_root / "ltx_official" / case.name)
    official_root, official_revision = _official_source(accuracy_artifact_root / "ltx_official")
    model = _resolve_model(case)
    gemma_root = _resolve_gemma_root(case, model)
    checkpoint = _resolve_artifact(case.checkpoint, model)
    spatial_upsampler = _resolve_artifact(case.spatial_upsampler, model) if case.spatial_upsampler is not None else None
    distilled_lora = _resolve_artifact(case.distilled_lora, model) if case.distilled_lora is not None else None
    image = _resolve_image(case)
    request_path = output_root / "request.json"
    request_path.write_text(json.dumps(_request(case, image), indent=2) + "\n")

    runner = Path(__file__).with_name("run_ltx_reference.py")
    runner_args = [
        str(runner),
        "--request",
        str(request_path),
        "--pipeline-kind",
        case.pipeline_kind,
    ]
    enable_layerwise_offload = case.enable_layerwise_offload or os.environ.get(
        "VLLM_TEST_LTX_ENABLE_LAYERWISE_OFFLOAD", ""
    ).lower() in {"1", "true", "yes", "on"}
    if enable_layerwise_offload:
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
    official_args = _official_runner_prefix() + runner_args
    official_args.extend(
        [
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
        ]
    )
    if spatial_upsampler is not None:
        official_args.extend(["--spatial-upsampler", str(spatial_upsampler)])
    if distilled_lora is not None:
        official_args.extend(["--distilled-lora", str(distilled_lora)])
    _run(official_args, env=env)

    omni_output = output_root / "omni"
    omni_args = (
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
        ]
    )
    _run(omni_args, env=env)

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
        "pipeline_kind": case.pipeline_kind,
        "model_class_name": case.model_class_name,
        "attention_backend": ATTENTION_BACKEND,
        "official_revision": official_revision,
        "model_revision": case.model_revision,
        "checkpoint_revision": case.checkpoint.revision,
        "spatial_upsampler_revision": (case.spatial_upsampler.revision if case.spatial_upsampler is not None else None),
        "distilled_lora_revision": case.distilled_lora.revision if case.distilled_lora is not None else None,
        "enable_layerwise_offload": enable_layerwise_offload,
        "thresholds": asdict(case.thresholds),
        "request": _request(case, image),
        "video": video_metrics,
        "audio": audio_metrics,
    }
    (output_root / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

    assert video_metrics["ssim_mean"] >= case.thresholds.video_ssim_mean
    assert video_metrics["ssim_min"] >= case.thresholds.video_ssim_min
    assert video_metrics["psnr_mean_db"] >= case.thresholds.video_psnr_mean_db
    if case.thresholds.audio_relative_l2 > 0:
        assert audio_metrics["relative_l2"] <= case.thresholds.audio_relative_l2
    if case.thresholds.audio_cosine_similarity > 0:
        assert audio_metrics["cosine_similarity"] >= case.thresholds.audio_cosine_similarity
