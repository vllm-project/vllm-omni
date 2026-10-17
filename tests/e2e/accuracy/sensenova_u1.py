# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pinned inputs and reviewable artifacts for the SenseNova H100 accuracy gate.

References are captured explicitly and reviewed before committing. Normal test
runs only read the reference file; they never update it on a mismatch.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image

    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL_ID = "SenseNova/SenseNova-U1-8B-MoT"
MODEL_REVISION = "bfa9b436503cb8aed4f2bc60e3236710cc77468d"
REFERENCE_PATH = Path(__file__).with_name("sensenova_u1_reference.json")
PIXEL_TOLERANCE = 10
REPO_ROOT = Path(__file__).resolve().parents[3]


def require_h100() -> str:
    """Check the actual GPU; hardware_test markers only route CI jobs."""
    from vllm_omni.platforms import current_omni_platform

    name = current_omni_platform.get_device_name() if current_omni_platform.is_cuda() else "non-CUDA platform"
    assert "H100" in name, f"SenseNova pixel accuracy requires an H100, got {name}"
    return name


def accuracy_model_path() -> str:
    """Resolve the exact weights before the pipeline can resolve a floating ID."""
    from vllm_omni.transformers_utils.repo_utils import hf_api

    require_h100()
    return hf_api().snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        allow_patterns=[
            "added_tokens.json",
            "chat_template.jinja",
            "config.json",
            "merges.txt",
            "model-*.safetensors",
            "model.safetensors.index.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "vocab.json",
        ],
    )


def _git_output(*args: str) -> bytes | None:
    result = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, check=False)
    return result.stdout if result.returncode == 0 else None


def _source_metadata() -> dict[str, Any]:
    commit = _git_output("rev-parse", "HEAD")
    patch = _git_output("diff", "HEAD", "--binary")
    source_paths = [
        "tests/e2e/accuracy/sensenova_u1.py",
        "tests/e2e/offline_inference/test_sensenova_u1_text2img_expansion.py",
        "tests/e2e/offline_inference/test_sensenova_u1_img2img_expansion.py",
    ]
    return {
        "git_commit": commit.decode().strip() if commit is not None else None,
        "declared_commit": os.environ.get("SENSENOVA_U1_SOURCE_COMMIT"),
        "git_diff_sha256": hashlib.sha256(patch).hexdigest() if patch is not None else None,
        # These also identify untracked test helpers in an uncommitted capture.
        "test_source_sha256": {
            path: hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest() for path in source_paths
        },
    }


def _environment_metadata() -> dict[str, Any]:
    import torch

    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "gpu": require_h100(),
        "driver": sorted(set(driver.stdout.splitlines())),
        "python": platform.python_version(),
        "cuda": torch.version.cuda,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "vllm", "transformers", "diffusers", "cache-dit")
        },
        "ci_image": os.environ.get("SENSENOVA_U1_CI_IMAGE"),
        "build_url": os.environ.get("BUILDKITE_BUILD_URL"),
    }


def record_accuracy(
    image: Image.Image,
    *,
    case: str,
    prompt: str,
    sampling_params: OmniDiffusionSamplingParams,
    pixel_positions: list[tuple[int, int]],
    input_image: Image.Image | None = None,
) -> dict[str, Any]:
    """Write the output and its inputs/provenance before checking the golden."""
    assert case in {"t2i", "i2i"}, f"Unknown SenseNova accuracy case: {case}"
    assert len(pixel_positions) == len(set(pixel_positions)) == 10, "Expected ten distinct reference positions"
    artifact_dir = Path(os.environ.get("SENSENOVA_U1_ARTIFACT_DIR", "sensenova-accuracy"))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    png_path = artifact_dir / f"{case}.png"
    image = image.convert("RGB")
    image.save(png_path)
    input_record = None
    if input_image is not None:
        rgb = input_image.convert("RGB")
        input_record = {"size": list(rgb.size), "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()}
    # Normalize tuples in extra_args so in-memory and JSON contracts compare.
    sampling = json.loads(
        json.dumps(
            {
                name: getattr(sampling_params, name)
                for name in ("height", "width", "seed", "num_inference_steps", "extra_args")
            }
        )
    )
    record = {
        "case": case,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "prompt": prompt,
            "sampling_params": sampling,
            "output_size": list(image.size),
            "input_image": input_record,
        },
        "reference_pixels": [
            {"position": list(position), "rgb": list(image.getpixel(position))} for position in pixel_positions
        ],
        "environment": _environment_metadata(),
        "source": _source_metadata(),
        "output_png_sha256": hashlib.sha256(png_path.read_bytes()).hexdigest(),
    }
    (artifact_dir / f"{case}.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def assert_accuracy(record: dict[str, Any], reference_path: Path = REFERENCE_PATH) -> None:
    """Compare like-for-like inputs and the original ten points at ±10 RGB."""
    assert reference_path.is_file(), (
        f"Missing reviewed H100 reference: {reference_path}. Capture and review the pinned H100 goldens; "
        "normal accuracy runs do not create or refresh references. The current PNG and metadata were saved."
    )
    reference = json.loads(reference_path.read_text())
    assert reference["schema_version"] == 1, "Unsupported SenseNova reference schema"
    golden = reference["cases"][record["case"]]
    assert "H100" in golden["environment"]["gpu"], "SenseNova reference must be captured on an H100"
    context = (
        f"Compare {reference_path} with the current PNG/JSON in "
        f"{os.environ.get('SENSENOVA_U1_ARTIFACT_DIR', 'sensenova-accuracy')}; "
        "check the recorded software, driver, CI image and source versions before refreshing goldens."
    )
    for key, expected in golden["contract"].items():
        assert record["contract"].get(key) == expected, f"SenseNova accuracy input mismatch: {key}. {context}"
    assert record["contract"].keys() == golden["contract"].keys(), (
        f"SenseNova accuracy contract fields differ. {context}"
    )
    actual_pixels = record["reference_pixels"]
    expected_pixels = golden["reference_pixels"]
    assert len(expected_pixels) == len(actual_pixels) == 10, "Expected ten reference pixels"
    for actual, expected in zip(actual_pixels, expected_pixels):
        assert actual["position"] == expected["position"], f"SenseNova reference positions differ. {context}"
        rgb = expected["rgb"]
        assert len(rgb) == 3 and all(type(channel) is int and 0 <= channel <= 255 for channel in rgb), (
            f"Invalid RGB reference at {expected['position']}"
        )
        assert all(abs(a - e) <= PIXEL_TOLERANCE for a, e in zip(actual["rgb"], rgb)), (
            f"Pixel mismatch at {expected['position']}: expected {rgb}, got {actual['rgb']} "
            f"(tolerance {PIXEL_TOLERANCE}). {context}"
        )
