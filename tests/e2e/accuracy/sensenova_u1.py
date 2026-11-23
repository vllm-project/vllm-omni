# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pinned inputs and reviewable artifacts for SenseNova accuracy gates.

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
H100_REFERENCE_PATH = Path(__file__).with_name("sensenova_u1_reference.json")
MI300_REFERENCE_PATH = Path(__file__).with_name("sensenova_u1_reference_mi300.json")
REFERENCE_PATHS = {
    "H100": H100_REFERENCE_PATH,
    "MI300": MI300_REFERENCE_PATH,
}
# Keep the original public constant for CPU contract tests and downstream tools.
REFERENCE_PATH = H100_REFERENCE_PATH
PIXEL_TOLERANCE = 10
REPO_ROOT = Path(__file__).resolve().parents[3]


def require_reference_hardware() -> tuple[str, str]:
    """Return the qualified hardware family and actual device name."""
    from vllm_omni.platforms import current_omni_platform

    if current_omni_platform.is_cuda():
        name = current_omni_platform.get_device_name()
        assert "H100" in name, f"SenseNova CUDA pixel accuracy requires an H100, got {name}"
        return "H100", name
    if current_omni_platform.is_rocm():
        name = current_omni_platform.get_device_name()
        assert "MI300" in name, f"SenseNova ROCm pixel accuracy requires an MI300, got {name}"
        return "MI300", name
    raise AssertionError("SenseNova pixel accuracy requires an H100 or MI300 accelerator")


def reference_path_for_current_hardware() -> Path:
    hardware, _ = require_reference_hardware()
    return REFERENCE_PATHS[hardware]


def accuracy_model_path() -> str:
    """Resolve the exact weights before the pipeline can resolve a floating ID."""
    from vllm_omni.transformers_utils.repo_utils import hf_api

    require_reference_hardware()
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

    hardware, device_name = require_reference_hardware()
    driver_command = (
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        if hardware == "H100"
        else ["rocm-smi", "--showdriverversion"]
    )
    driver = subprocess.run(
        driver_command,
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "reference_hardware": hardware,
        "gpu": device_name,
        "driver": sorted({line.strip() for line in driver.stdout.splitlines() if line.strip()}),
        "python": platform.python_version(),
        "cuda": torch.version.cuda,
        "rocm": torch.version.hip,
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


def assert_accuracy(record: dict[str, Any], reference_path: Path | None = None) -> None:
    """Compare like-for-like inputs and the original ten points at ±10 RGB."""
    if reference_path is None:
        reference_path = reference_path_for_current_hardware()
    actual_hardware = record.get("environment", {}).get("reference_hardware")
    if actual_hardware is None:
        gpu_name = str(record.get("environment", {}).get("gpu", ""))
        actual_hardware = next((name for name in REFERENCE_PATHS if name in gpu_name), "selected hardware")
    assert reference_path.is_file(), (
        f"Missing reviewed {actual_hardware} reference: {reference_path}. "
        f"Capture and review the pinned {actual_hardware} goldens; "
        "normal accuracy runs do not create or refresh references. The current PNG and metadata were saved."
    )
    reference = json.loads(reference_path.read_text())
    assert reference["schema_version"] == 1, "Unsupported SenseNova reference schema"
    golden = reference["cases"][record["case"]]
    golden_gpu = golden["environment"]["gpu"]
    golden_hardware = golden["environment"].get("reference_hardware")
    if golden_hardware is None:
        golden_hardware = next((name for name in REFERENCE_PATHS if name in golden_gpu), None)
    assert golden_hardware == actual_hardware, (
        f"SenseNova reference hardware mismatch: expected {actual_hardware}, got {golden_hardware} ({golden_gpu})"
    )
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
