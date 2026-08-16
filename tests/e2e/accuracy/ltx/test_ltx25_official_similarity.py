# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E accuracy guards against pinned official Lightricks LTX references.

The LTX-2/2.3 comparison runs both runtimes through PyTorch SDPA and uses
``max_batch_size=4`` in the official reference to match Omni's fused guidance
batch. Video and audio guidance use the official non-HQ one-stage defaults;
only the generation shape and step count are reduced for CI runtime.
LTX-2.5 runs the official split-artifact pipelines with connector weights from
the same Diffusers checkpoint under test. The distilled case covers the fixed
two-stage schedule; the Full/SFT case covers raw dev weights and an explicit
shared one-stage schedule.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from tests.e2e.accuracy.helpers import reset_artifact_dir
from tests.helpers import skip_if_gated_repo_inaccessible
from tests.helpers.mark import hardware_test

OFFICIAL_REPOSITORY = "https://github.com/Lightricks/LTX-2.git"
LTX25_OFFICIAL_REVISION = "7954dcb0d986bdc36ef272564a9789ade07fcc65"
LTX25_OMNI_MODEL_ID = "Lightricks/LTX-2.5-Diffusers"
LTX25_OMNI_MODEL_REVISION = "a6de4b5354f078db24d9cf4778c14846788aea3d"
LTX25_OFFICIAL_MODEL_ID = "Lightricks/LTX-2.5"
LTX25_OFFICIAL_MODEL_REVISION = "8a4ff96f581e72bedc1b44367581c49d544a05f1"
LTX25_OFFICIAL_COMMON_FILES = {
    "text_encoder": "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "video_vae": "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
    "audio_vae": "vae/ltx-2.5-audio-vae-bf16.safetensors",
}
LTX25_OFFICIAL_VARIANT_FILES = {
    "distilled": {
        "transformer": "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        "spatial_upsampler": "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    },
    "full": {
        "transformer": "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors",
    },
    "full_two_stage": {
        "transformer": "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors",
        "spatial_upsampler": "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        "distilled_lora": "loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
    },
}
# Version selected by the pinned official source's uv.lock. Keep it isolated
# from Omni's runtime and development dependencies.
OFFICIAL_OPENIMAGEIO_VERSION = "3.1.11.0"
LTX25_PROMPT = "A cinematic shot of a red fox walking through a snowy forest at dawn, the camera tracking alongside."
LTX25_I2V_IMAGE_REPO = "huggingface/documentation-images"
LTX25_I2V_IMAGE_FILENAME = "diffusers/svd/rocket.png"
LTX25_I2V_IMAGE_REVISION = "645d8364f0c7a101180b364811b5a11a362e4010"
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

# Release parity exercises the accelerated BF16 cuDNN path in both runtimes.
# Degenerate one-token attention falls back to PyTorch SDPA MATH.
ATTENTION_BACKEND = "torch_sdpa_cudnn_bf16"
VIDEO_SSIM_MEAN_THRESHOLD = 0.95
VIDEO_SSIM_MIN_THRESHOLD = 0.90
VIDEO_PSNR_MEAN_THRESHOLD = 30.0
AUDIO_RELATIVE_L2_THRESHOLD = 0.2
AUDIO_COSINE_THRESHOLD = 0.95

# Decoded-output gates for the pinned official/Omni BF16 cuDNN comparison.
# The metrics artifact preserves the exact observed values.
LTX25_VIDEO_SSIM_MEAN_THRESHOLD = 0.99
LTX25_I2V_VIDEO_SSIM_MEAN_THRESHOLD = 0.99
LTX25_VIDEO_SSIM_MIN_THRESHOLD = 0.99
LTX25_VIDEO_PSNR_MEAN_THRESHOLD = 40.0
LTX25_VIDEO_LPIPS_MEAN_THRESHOLD = 0.01
LTX25_AUDIO_RELATIVE_L2_THRESHOLD = 0.32
LTX25_AUDIO_COSINE_THRESHOLD = 0.95

# Full/SFT uses independently converted dev weights, but the release contract
# still requires decoded-video parity at the same 0.99 SSIM floor as the
# distilled paths when both runtimes use BF16 cuDNN attention.
LTX25_FULL_VIDEO_SSIM_MEAN_THRESHOLD = 0.99
LTX25_FULL_VIDEO_SSIM_MIN_THRESHOLD = 0.99
LTX25_FULL_VIDEO_PSNR_MEAN_THRESHOLD = 40.0
LTX25_FULL_AUDIO_RELATIVE_L2_THRESHOLD = 0.3
LTX25_FULL_VIDEO_LPIPS_MEAN_THRESHOLD = 0.01
LTX25_FULL_I2V_VIDEO_LPIPS_MEAN_THRESHOLD = 0.02
LTX25_FULL_AUDIO_COSINE_THRESHOLD = AUDIO_COSINE_THRESHOLD
LTX25_STAGE_2_SIGMAS = [0.909375, 0.725, 0.421875, 0.0]


def test_ltx_reference_runner_unwraps_flattened_pipeline_output() -> None:
    from vllm_omni.outputs import OmniRequestOutput

    from .run_ltx25_reference import _omni_worker_peak_memory_mb, _unwrap_omni_output

    frame = object()
    audio = object()
    output = OmniRequestOutput(
        stage_id=0,
        images=[frame],
        _multimodal_output={"audio": audio, "audio_sample_rate": 48_000},
    )

    frames, actual_audio, sample_rate = _unwrap_omni_output(output)

    assert frames is frame
    assert actual_audio is audio
    assert sample_rate == 48_000
    output.peak_memory_mb = 12_345.0
    assert _omni_worker_peak_memory_mb([output]) == 12_345.0


def _run(command: list[str], *, env: dict[str, str], timeout: int = 1800) -> None:
    start = time.perf_counter()
    subprocess.run(command, env=env, timeout=timeout, check=True)
    print(f"{' '.join(command[:3])} finished in {time.perf_counter() - start:.1f}s")


def _clone_pinned_source(root: Path, repository: str, revision: str) -> None:
    root.mkdir(parents=True)
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


def _ltx25_official_source(artifact_root: Path) -> tuple[Path, str]:
    repository = os.environ.get("VLLM_TEST_LTX25_OFFICIAL_REPOSITORY", OFFICIAL_REPOSITORY)
    revision = os.environ.get("VLLM_TEST_LTX25_OFFICIAL_REVISION", LTX25_OFFICIAL_REVISION)
    configured_root = os.environ.get("VLLM_TEST_LTX25_OFFICIAL_ROOT")
    root = Path(configured_root) if configured_root else artifact_root / f"official-source-{revision[:12]}"
    actual_revision = _git_revision(root) if root.exists() else None
    if actual_revision != revision and configured_root:
        raise AssertionError(f"Official LTX-2.5 source revision mismatch: {actual_revision} != {revision}")
    if actual_revision != revision:
        if root.exists():
            shutil.rmtree(root)
        _clone_pinned_source(root, repository, revision)
        actual_revision = _git_revision(root)
    assert actual_revision == revision, f"Official LTX-2.5 source revision mismatch: {actual_revision} != {revision}"
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


def _resolve_ltx25_omni_model(*, transformer_subfolder: str = "transformer") -> tuple[Path, str, str | None]:
    if transformer_subfolder not in {"transformer", "transformer_full"}:
        raise ValueError(f"Unsupported LTX-2.5 transformer subfolder: {transformer_subfolder!r}")
    configured_model = os.environ.get("VLLM_TEST_LTX25_MODEL")
    configured_revision = os.environ.get("VLLM_TEST_LTX25_MODEL_REVISION")
    if configured_model and Path(configured_model).exists():
        model = Path(configured_model).resolve()
        snapshot_revision = model.name if model.parent.name == "snapshots" else None
        assert (model / transformer_subfolder).is_dir(), f"LTX-2.5 component not found: {model / transformer_subfolder}"
        return model, configured_model, configured_revision or snapshot_revision
    model_id = configured_model or LTX25_OMNI_MODEL_ID
    revision = configured_revision
    if revision is None and model_id == LTX25_OMNI_MODEL_ID:
        revision = LTX25_OMNI_MODEL_REVISION
    model = Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            allow_patterns=[
                "model_index.json",
                "audio_vae/*",
                "connectors/config.json",
                "connectors/diffusion_pytorch_model.safetensors.index.json",
                "connectors/diffusion_pytorch_model-*.safetensors",
                "scheduler/*",
                "text_encoder/config.json",
                "text_encoder/generation_config.json",
                "text_encoder/model*",
                "tokenizer/*",
                f"{transformer_subfolder}/*",
                "vae/*",
                "vocoder/*",
            ],
        )
    ).resolve()
    snapshot_revision = model.name if model.parent.name == "snapshots" else None
    return model, model_id, snapshot_revision or revision


def _resolve_ltx25_official_artifacts(
    *, checkpoint_variant: str = "distilled"
) -> tuple[dict[str, Path], str, str | None]:
    try:
        variant_files = LTX25_OFFICIAL_VARIANT_FILES[checkpoint_variant]
    except KeyError as exc:
        raise ValueError(f"Unsupported official LTX-2.5 checkpoint variant: {checkpoint_variant!r}") from exc
    files = {**LTX25_OFFICIAL_COMMON_FILES, **variant_files}
    configured_model = os.environ.get("VLLM_TEST_LTX25_OFFICIAL_MODEL")
    configured_revision = os.environ.get("VLLM_TEST_LTX25_OFFICIAL_MODEL_REVISION")
    if configured_model and Path(configured_model).exists():
        root = Path(configured_model).resolve()
        model_source = configured_model
        snapshot_revision = root.name if root.parent.name == "snapshots" else None
        resolved_revision = configured_revision or snapshot_revision
    else:
        model_source = configured_model or LTX25_OFFICIAL_MODEL_ID
        revision = configured_revision
        if revision is None and model_source == LTX25_OFFICIAL_MODEL_ID:
            revision = LTX25_OFFICIAL_MODEL_REVISION
        root = Path(
            snapshot_download(
                repo_id=model_source,
                revision=revision,
                allow_patterns=list(files.values()),
            )
        ).resolve()
        snapshot_revision = root.name if root.parent.name == "snapshots" else None
        resolved_revision = snapshot_revision or revision

    artifacts = {name: root / relative_path for name, relative_path in files.items()}
    missing = [f"{name}: {path}" for name, path in artifacts.items() if not path.is_file()]
    assert not missing, f"Official LTX-2.5 artifacts are missing: {', '.join(missing)}"
    return artifacts, model_source, resolved_revision


def _resolve_gemma_root(model: Path) -> Path:
    configured_root = os.environ.get("VLLM_TEST_LTX_GEMMA_ROOT")
    if configured_root:
        root = Path(configured_root)
        assert root.is_dir(), f"Gemma root not found: {root}"
        return root
    return model


def _resolve_ltx25_image() -> Path:
    configured_image = os.environ.get("VLLM_TEST_LTX25_I2V_IMAGE")
    if configured_image:
        image = Path(configured_image)
        assert image.is_file(), f"LTX-2.5 I2V conditioning image not found: {image}"
        return image
    return Path(
        hf_hub_download(
            repo_id=LTX25_I2V_IMAGE_REPO,
            repo_type="dataset",
            filename=LTX25_I2V_IMAGE_FILENAME,
            revision=LTX25_I2V_IMAGE_REVISION,
        )
    )


def _require_ltx25_model_access() -> None:
    configured_omni_model = os.environ.get("VLLM_TEST_LTX25_MODEL")
    if not (configured_omni_model and Path(configured_omni_model).exists()):
        omni_model = configured_omni_model or LTX25_OMNI_MODEL_ID
        omni_revision = os.environ.get("VLLM_TEST_LTX25_MODEL_REVISION")
        if omni_revision is None and omni_model == LTX25_OMNI_MODEL_ID:
            omni_revision = LTX25_OMNI_MODEL_REVISION
        skip_if_gated_repo_inaccessible(
            omni_model,
            revision=omni_revision,
            filename="model_index.json",
        )

    configured_official_model = os.environ.get("VLLM_TEST_LTX25_OFFICIAL_MODEL")
    if not (configured_official_model and Path(configured_official_model).exists()):
        official_model = configured_official_model or LTX25_OFFICIAL_MODEL_ID
        official_revision = os.environ.get("VLLM_TEST_LTX25_OFFICIAL_MODEL_REVISION")
        if official_revision is None and official_model == LTX25_OFFICIAL_MODEL_ID:
            official_revision = LTX25_OFFICIAL_MODEL_REVISION
        skip_if_gated_repo_inaccessible(
            official_model,
            revision=official_revision,
            filename=LTX25_OFFICIAL_COMMON_FILES["text_encoder"],
        )


def _ltx25_request(image: Path | None = None) -> dict[str, object]:
    request: dict[str, object] = {
        "prompt": LTX25_PROMPT,
        # Keep the nightly parity test small enough to run the pinned official
        # pipeline and Omni sequentially on one H100. The public recipe covers
        # the release-resolution 1920x1088x121 configuration.
        "width": 1024,
        "height": 576,
        "num_frames": 25,
        "fps": 24,
        "num_inference_steps": 8,
        "seed": 42,
    }
    if image is not None:
        request.update(image=str(image.resolve()), image_crf=18)
    return request


def _ltx25_full_sigmas(num_inference_steps: int) -> list[float]:
    """Materialize the official LTX-2.5 one-stage schedule in FP32."""
    sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1)
    base_anchor = 1024
    max_anchor = 4096
    base_shift = 0.95
    max_shift = 2.05
    slope = (max_shift - base_shift) / (max_anchor - base_anchor)
    sigma_shift = max_anchor * slope + (base_shift - slope * base_anchor)
    exp_shift = math.exp(sigma_shift)
    sigmas = torch.where(sigmas != 0, exp_shift / (exp_shift + (1 / sigmas - 1)), 0)

    non_zero = sigmas != 0
    one_minus_sigmas = 1.0 - sigmas[non_zero]
    scale = one_minus_sigmas[-1] / (1.0 - 0.1)
    sigmas[non_zero] = 1.0 - one_minus_sigmas / scale
    return [float(sigma) for sigma in sigmas]


def _ltx25_full_request(image: Path | None = None) -> dict[str, object]:
    num_inference_steps = 30
    request: dict[str, object] = {
        "prompt": LTX25_PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        # Full guidance expands to a four-way batch. Keep the decoded output
        # small enough for the pinned official and Omni runs to share one H100.
        "width": 512,
        "height": 384,
        "num_frames": 25,
        "fps": 24,
        "num_inference_steps": num_inference_steps,
        "seed": 42,
        "video_cfg_scale": 3.0,
        "audio_cfg_scale": 7.0,
        "video_stg_scale": 1.0,
        "audio_stg_scale": 1.0,
        "video_modality_scale": 3.0,
        "audio_modality_scale": 3.0,
        "video_rescale_scale": 0.7,
        "audio_rescale_scale": 0.7,
        "video_stg_blocks": [28],
        "audio_stg_blocks": [28],
        "sigmas": _ltx25_full_sigmas(num_inference_steps),
    }
    if image is not None:
        request.update(image=str(image.resolve()), image_crf=18)
    return request


def test_ltx25_full_request_pins_official_schedule() -> None:
    request = _ltx25_full_request()
    sigmas = request["sigmas"]

    assert isinstance(sigmas, list)
    assert isinstance(request["num_inference_steps"], int)
    assert len(sigmas) == request["num_inference_steps"] + 1
    assert math.isclose(sigmas[0], 1.0, rel_tol=0.0, abs_tol=1e-6)
    assert sigmas[-1] == 0.0
    assert all(sigmas[index] > sigmas[index + 1] for index in range(len(sigmas) - 1))


def test_ltx25_i2v_requests_pin_official_crf() -> None:
    image = Path("conditioning.png")
    for request in (_ltx25_request(image), _ltx25_full_request(image)):
        assert request["image"] == str(image.resolve())
        assert request["image_crf"] == 18


def _video_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    from benchmarks.diffusion.quantization_quality import compute_lpips_video

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
        "lpips_alex_mean": compute_lpips_video(reference, prediction),
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


@pytest.mark.full_model
@pytest.mark.benchmark
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("task", ["t2v", "i2v"])
def test_ltx25_distilled_two_stage_matches_official(accuracy_artifact_root: Path, task: str) -> None:
    """Compare Omni with the official runtime using the same checkpoint connector weights."""
    _require_ltx25_model_access()
    artifact_parent = accuracy_artifact_root / "ltx_official"
    output_root = reset_artifact_dir(artifact_parent / f"ltx2_5_distilled_{task}")
    official_root, official_revision = _ltx25_official_source(artifact_parent)
    official_artifacts, official_model_source, official_model_revision = _resolve_ltx25_official_artifacts()
    omni_model, omni_model_source, omni_model_revision = _resolve_ltx25_omni_model()
    image = _resolve_ltx25_image() if task == "i2v" else None
    request_path = output_root / "request.json"
    request_path.write_text(json.dumps(_ltx25_request(image), indent=2) + "\n")

    runner = Path(__file__).with_name("run_ltx25_reference.py")
    runner_args = [
        str(runner),
        "--request",
        str(request_path),
        "--enable-layerwise-offload",
    ]
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
            "--transformer-path",
            str(official_artifacts["transformer"]),
            "--text-encoder-path",
            str(official_artifacts["text_encoder"]),
            "--video-vae-path",
            str(official_artifacts["video_vae"]),
            "--audio-vae-path",
            str(official_artifacts["audio_vae"]),
            "--spatial-upsampler-path",
            str(official_artifacts["spatial_upsampler"]),
            "--connector-model",
            str(omni_model),
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
            str(omni_model),
            "--model-class-name",
            "LTX2DistilledTwoStagePipeline",
        ],
        env=env,
    )

    official_metadata = json.loads((official_output / "metadata.json").read_text())
    omni_metadata = json.loads((omni_output / "metadata.json").read_text())
    assert official_metadata["official_revision"] == official_revision
    assert official_metadata["pipeline"] == "DistilledPipeline"
    assert official_metadata["attention_backend"] == ATTENTION_BACKEND
    assert official_metadata["connector_model"] == str(omni_model)
    assert omni_metadata["model_class_name"] == "LTX2DistilledTwoStagePipeline"
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
        "case": "ltx2_5_distilled",
        "task": task,
        "attention_backend": ATTENTION_BACKEND,
        "official_revision": official_revision,
        "official_model": official_model_source,
        "official_model_revision": official_model_revision,
        "official_connector_model": omni_model_source,
        "official_connector_revision": omni_model_revision,
        "omni_model": omni_model_source,
        "omni_model_revision": omni_model_revision,
        "resolved_omni_model_path": str(omni_model),
        "peak_memory_mb": {
            "official_allocated": official_metadata["peak_memory_allocated_mb"],
            "official_reserved": official_metadata["peak_memory_reserved_mb"],
            "omni_worker_reported": omni_metadata["worker_peak_memory_mb"],
            "omni_parent_allocated": omni_metadata["peak_memory_allocated_mb"],
            "omni_parent_reserved": omni_metadata["peak_memory_reserved_mb"],
        },
        "video": video_metrics,
        "audio": audio_metrics,
    }
    (output_root / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

    ssim_mean_threshold = LTX25_VIDEO_SSIM_MEAN_THRESHOLD if task == "t2v" else LTX25_I2V_VIDEO_SSIM_MEAN_THRESHOLD
    assert video_metrics["ssim_mean"] >= ssim_mean_threshold
    assert video_metrics["ssim_min"] >= LTX25_VIDEO_SSIM_MIN_THRESHOLD
    assert video_metrics["psnr_mean_db"] >= LTX25_VIDEO_PSNR_MEAN_THRESHOLD
    assert video_metrics["lpips_alex_mean"] <= LTX25_VIDEO_LPIPS_MEAN_THRESHOLD
    assert audio_metrics["cosine_similarity"] >= LTX25_AUDIO_COSINE_THRESHOLD
    assert audio_metrics["relative_l2"] <= LTX25_AUDIO_RELATIVE_L2_THRESHOLD


@pytest.mark.full_model
@pytest.mark.benchmark
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("task", ["t2v", "i2v"])
@pytest.mark.parametrize("pipeline_mode", ["full_one_stage", "full_two_stage"])
def test_ltx25_full_matches_official(accuracy_artifact_root: Path, task: str, pipeline_mode: str) -> None:
    """Compare raw official Full/SFT weights with Omni's converted Full pipeline."""
    _require_ltx25_model_access()
    artifact_parent = accuracy_artifact_root / "ltx_official"
    output_root = reset_artifact_dir(artifact_parent / f"ltx2_5_{pipeline_mode}_{task}")
    model_class_name = {
        "full_one_stage": "LTX2Pipeline",
        "full_two_stage": "LTX2TwoStagePipeline",
    }[pipeline_mode]
    official_root, official_revision = _ltx25_official_source(artifact_parent)
    official_artifacts, official_model_source, official_model_revision = _resolve_ltx25_official_artifacts(
        checkpoint_variant="full" if pipeline_mode == "full_one_stage" else "full_two_stage"
    )
    omni_model, omni_model_source, omni_model_revision = _resolve_ltx25_omni_model(
        transformer_subfolder="transformer_full"
    )
    image = _resolve_ltx25_image() if task == "i2v" else None
    request = _ltx25_full_request(image)
    if pipeline_mode == "full_two_stage":
        request["stage_1_sigmas"] = request.pop("sigmas")
        request["stage_2_sigmas"] = LTX25_STAGE_2_SIGMAS
    request_path = output_root / "request.json"
    request_path.write_text(json.dumps(request, indent=2) + "\n")

    runner = Path(__file__).with_name("run_ltx25_reference.py")
    runner_args = [
        str(runner),
        "--request",
        str(request_path),
        "--enable-layerwise-offload",
    ]
    env = os.environ.copy()
    env["VLLM_TEST_LTX_OFFICIAL_REVISION"] = official_revision
    env["PYTHONUNBUFFERED"] = "1"
    repository_root = Path(__file__).resolve().parents[4]
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(repository_root) if not existing_pythonpath else f"{repository_root}{os.pathsep}{existing_pythonpath}"
    )

    official_output = output_root / "official"
    official_sidecar_args: list[str] = []
    if pipeline_mode == "full_two_stage":
        official_sidecar_args = [
            "--spatial-upsampler-path",
            str(official_artifacts["spatial_upsampler"]),
            "--distilled-lora-path",
            str(official_artifacts["distilled_lora"]),
        ]

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
            "--official-pipeline",
            pipeline_mode,
            "--transformer-path",
            str(official_artifacts["transformer"]),
            "--text-encoder-path",
            str(official_artifacts["text_encoder"]),
            "--video-vae-path",
            str(official_artifacts["video_vae"]),
            "--audio-vae-path",
            str(official_artifacts["audio_vae"]),
            "--connector-model",
            str(omni_model),
        ]
        + official_sidecar_args,
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
            str(omni_model),
            "--model-class-name",
            model_class_name,
        ],
        env=env,
    )

    official_metadata = json.loads((official_output / "metadata.json").read_text())
    omni_metadata = json.loads((omni_output / "metadata.json").read_text())
    assert official_metadata["official_revision"] == official_revision
    assert (
        official_metadata["pipeline"]
        == {
            "full_one_stage": "TI2VidOneStagePipeline",
            "full_two_stage": "TI2VidTwoStagesPipeline",
        }[pipeline_mode]
    )
    assert official_metadata["attention_backend"] == ATTENTION_BACKEND
    assert official_metadata["connector_model"] == str(omni_model)
    assert omni_metadata["model_class_name"] == model_class_name
    assert omni_metadata["attention_backend"] == ATTENTION_BACKEND
    assert official_metadata["seed"] == omni_metadata["seed"]
    for schedule_name in ("sigmas", "stage_1_sigmas", "stage_2_sigmas"):
        assert official_metadata[schedule_name] == omni_metadata[schedule_name]
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
        "case": f"ltx2_5_{pipeline_mode}",
        "task": task,
        "attention_backend": ATTENTION_BACKEND,
        "official_revision": official_revision,
        "official_model": official_model_source,
        "official_model_revision": official_model_revision,
        "official_connector_model": omni_model_source,
        "official_connector_revision": omni_model_revision,
        "omni_model": omni_model_source,
        "omni_model_revision": omni_model_revision,
        "resolved_omni_model_path": str(omni_model),
        "peak_memory_mb": {
            "official_allocated": official_metadata["peak_memory_allocated_mb"],
            "official_reserved": official_metadata["peak_memory_reserved_mb"],
            "omni_worker_reported": omni_metadata["worker_peak_memory_mb"],
            "omni_parent_allocated": omni_metadata["peak_memory_allocated_mb"],
            "omni_parent_reserved": omni_metadata["peak_memory_reserved_mb"],
        },
        "video": video_metrics,
        "audio": audio_metrics,
    }
    (output_root / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

    assert video_metrics["ssim_mean"] >= LTX25_FULL_VIDEO_SSIM_MEAN_THRESHOLD
    assert video_metrics["ssim_min"] >= LTX25_FULL_VIDEO_SSIM_MIN_THRESHOLD
    assert video_metrics["psnr_mean_db"] >= LTX25_FULL_VIDEO_PSNR_MEAN_THRESHOLD
    lpips_threshold = (
        LTX25_FULL_VIDEO_LPIPS_MEAN_THRESHOLD if task == "t2v" else LTX25_FULL_I2V_VIDEO_LPIPS_MEAN_THRESHOLD
    )
    assert audio_metrics["relative_l2"] <= LTX25_FULL_AUDIO_RELATIVE_L2_THRESHOLD
    assert video_metrics["lpips_alex_mean"] <= lpips_threshold
    assert audio_metrics["cosine_similarity"] >= LTX25_FULL_AUDIO_COSINE_THRESHOLD
