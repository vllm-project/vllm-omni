# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
import requests
import torch
from safetensors.torch import load_file
from torch import nn

from tests.e2e.accuracy.helpers import assert_video_metadata, assert_video_similarity_metrics, probe_video
from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler

GOLDEN_BASE_URL = os.getenv("SANA_VIDEO_GOLDEN_BASE_URL")
SCHEMA_VERSION = 2
PROMPT = "A cat walking on the grass, facing the camera. motion score: 30."
NEGATIVE_PROMPT = (
    "A chaotic sequence with misshapen, deformed limbs in heavy motion blur, sudden disappearance, jump cuts, "
    "jerky movements, rapid shot changes, frames out of sync, inconsistent character shapes, temporal artifacts, "
    "jitter, and ghosting effects, creating a disorienting visual experience."
)
VARIANTS = {
    "480p": ("Efficient-Large-Model/SANA-Video_2B_480p_diffusers", 480, 832),
    "720p": ("Efficient-Large-Model/SANA-Video_2B_720p_diffusers", 704, 1280),
}
REVISIONS = {
    "480p": "fed3bce411c58a0f688a31afe8f52e61acc2b15f",
    "720p": "8bda5e623d0f48cd6da3b387b10ca35d15cf1c4e",
}
SSIM_THRESHOLDS = {
    ("480p", "t2v"): 0.91,
    ("480p", "i2v"): 0.93,
    ("720p", "t2v"): 0.93,
    ("720p", "i2v"): 0.93,
}

pytestmark = [
    pytest.mark.full_model,
    pytest.mark.diffusion,
    pytest.mark.skipif(
        not GOLDEN_BASE_URL,
        reason="Set SANA_VIDEO_GOLDEN_BASE_URL after publishing the frozen v2 S3 assets.",
    ),
]


class _TransformerCheckpoint(nn.Module):
    def __init__(self, transformer: nn.Module, model: str, revision: str) -> None:
        super().__init__()
        from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader

        self.transformer = transformer
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model,
                subfolder="transformer",
                revision=revision,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _assert_hex_digest(value: Any, *, length: int = 64) -> None:
    assert isinstance(value, str) and len(value) == length
    assert all(character in "0123456789abcdef" for character in value)


def _validate_manifest(payload: Any, variant: str, task: str) -> dict[str, Any]:
    assert isinstance(payload, dict)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["variant"] == variant
    assert payload["task"] == task
    # Local checkpoint generation is deliberately marked non-publishable. Do
    # not permit such output to become the canonical reference by accident.
    assert payload["publishable"] is True
    assert payload["model_provenance"] == {
        "kind": "huggingface_hub",
        "model_id": VARIANTS[variant][0],
        "revision": REVISIONS[variant],
    }
    _assert_hex_digest(payload["metadata_sha256"])

    expected_files = {
        "metadata.json",
        "pipeline.mp4",
        "pipeline_reference.safetensors",
        "transformer_case.safetensors",
    }
    if task == "i2v":
        expected_files.add("input.png")
    files = payload["files"]
    assert isinstance(files, dict)
    assert set(files) == expected_files
    for filename, expected in files.items():
        # Prevent a compromised manifest from writing outside output_dir.
        assert filename == Path(filename).name
        assert isinstance(expected, dict) and set(expected) == {"sha256", "size"}
        _assert_hex_digest(expected["sha256"])
        assert isinstance(expected["size"], int) and expected["size"] > 0
    return payload


def _validate_metadata(payload: Any, manifest: dict[str, Any], variant: str, task: str) -> None:
    assert isinstance(payload, dict)
    _, height, width = VARIANTS[variant]
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["variant"] == variant
    assert payload["task"] == task
    assert payload["publishable"] is True
    assert payload["model_provenance"] == manifest["model_provenance"]
    assert payload["prompt"] == PROMPT
    assert payload["negative_prompt"] == NEGATIVE_PROMPT
    assert payload["seed"] == 42
    assert payload["generator_device"] == "cuda"
    assert payload["dtype"] == "bfloat16"
    assert payload["height"] == height
    assert payload["width"] == width
    assert payload["num_frames"] == 81
    assert payload["fps"] == 16
    assert payload["num_inference_steps"] == 50
    assert payload["guidance_scale"] == 6.0

    implementation = payload["reference_implementation"]
    assert implementation["library"] == "diffusers"
    assert implementation["pipeline_class"] == ("SanaImageToVideoPipeline" if task == "i2v" else "SanaVideoPipeline")
    assert implementation["version"]
    assert implementation["torch_version"]

    generator = payload["generator_provenance"]
    assert generator["repository"] == "vllm-project/vllm-omni"
    _assert_hex_digest(generator["revision"], length=40)
    _assert_hex_digest(generator["generator_sha256"])
    assert generator["dirty"] is False

    scheduler = payload["scheduler"]
    assert scheduler["class"] == "DPMSolverMultistepScheduler"
    _assert_hex_digest(scheduler["config_sha256"])
    assert scheduler["config_sha256"] == _json_sha256(scheduler["config"])

    input_image = payload["input_image"]
    if task == "i2v":
        assert input_image["filename"] == "input.png"
        _assert_hex_digest(input_image["sha256"])
        assert input_image["size"] == manifest["files"]["input.png"]["size"]
        assert input_image["sha256"] == manifest["files"]["input.png"]["sha256"]
        assert input_image["format"] == "png-rgb"
    else:
        assert input_image is None


def _download_case(variant: str, task: str, output_dir: Path, filenames: set[str]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"{GOLDEN_BASE_URL.rstrip('/')}/{variant}/{task}"
    response = requests.get(f"{base_url}/manifest.json", timeout=60)
    response.raise_for_status()
    payload = _validate_manifest(response.json(), variant, task)
    filenames = filenames | {"metadata.json"}
    if task == "i2v":
        filenames.add("input.png")
    assert filenames <= payload["files"].keys()
    for filename in filenames:
        expected = payload["files"][filename]
        path = output_dir / filename
        response = requests.get(f"{base_url}/{filename}", timeout=300)
        response.raise_for_status()
        path.write_bytes(response.content)
        assert path.stat().st_size == expected["size"]
        assert _sha256(path) == expected["sha256"]
    assert _sha256(output_dir / "metadata.json") == payload["metadata_sha256"]
    metadata = json.loads((output_dir / "metadata.json").read_text())
    _validate_metadata(metadata, payload, variant, task)
    return payload


@pytest.mark.benchmark
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize(
    "_hardware",
    [pytest.param(None, marks=hardware_marks(res={"cuda": "H100"}))],
)
def test_sana_video_transformer_matches_frozen_golden(
    variant: str,
    _hardware,
    accuracy_artifact_root: Path,
) -> None:
    del _hardware
    from vllm.config import LoadConfig
    from vllm.transformers_utils.config import get_hf_file_to_dict

    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
    from vllm_omni.diffusion.models.sana_video import SanaVideoTransformer3DModel

    model, _, _ = VARIANTS[variant]
    revision = REVISIONS[variant]
    output_dir = accuracy_artifact_root / "sana_video" / variant / "t2v"
    _download_case(variant, "t2v", output_dir, {"transformer_case.safetensors"})
    case = load_file(output_dir / "transformer_case.safetensors")
    transformer_config = get_hf_file_to_dict("transformer/config.json", model, revision=revision)
    assert transformer_config is not None
    transformer = SanaVideoTransformer3DModel.from_config(transformer_config).to(
        device="cuda",
        dtype=torch.bfloat16,
    )
    checkpoint = _TransformerCheckpoint(transformer, model, revision)
    od_config = OmniDiffusionConfig(
        model=model,
        model_class_name="SanaVideoPipeline",
        dtype=torch.bfloat16,
        revision=revision,
    )
    loader = DiffusersPipelineLoader(LoadConfig(), od_config=od_config)
    transformer.load_weights(
        (name.removeprefix("transformer."), tensor)
        for name, tensor in loader.get_all_weights(checkpoint)
        if name.startswith("transformer.")
    )
    transformer.eval()
    with torch.inference_mode():
        actual = transformer(
            case["hidden_states"].to("cuda"),
            case["encoder_hidden_states"].to("cuda"),
            case["timestep"].to("cuda"),
            encoder_attention_mask=case["encoder_attention_mask"].to("cuda"),
        ).sample.float()
    expected = case["output"].to("cuda").float()
    error = actual - expected
    max_abs = error.abs().max()
    relative_l2 = torch.linalg.vector_norm(error) / torch.linalg.vector_norm(expected)
    cosine = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
    # CUDNN_ATTN BF16 reduction order is not bitwise deterministic on
    # Blackwell. Bound rare outliers separately while keeping the aggregate
    # relative-L2 and cosine checks strict.
    max_abs_threshold = 0.2 if variant == "480p" else 0.1
    relative_l2_threshold = 0.017 if variant == "480p" else 0.015
    assert max_abs.item() <= max_abs_threshold
    assert relative_l2.item() <= relative_l2_threshold
    assert cosine.item() >= 0.999


SERVER_CASES = [
    pytest.param(
        OmniServerParams(
            model=model,
            server_args=[
                "--model-class-name",
                pipeline_class,
            ],
        ),
        id=f"{variant}-{task}",
        marks=hardware_marks(res={"cuda": "H100"}),
    )
    for variant, (model, _, _) in VARIANTS.items()
    for task, pipeline_class in (
        ("t2v", "SanaVideoPipeline"),
        ("i2v", "SanaImageToVideoPipeline"),
    )
]


@pytest.mark.benchmark
@pytest.mark.parametrize("omni_server", SERVER_CASES, indirect=True)
def test_sana_video_pipeline_matches_frozen_golden(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
    accuracy_artifact_root: Path,
) -> None:
    variant = "720p" if "720p" in omni_server.model else "480p"
    task = "i2v" if "SanaImageToVideoPipeline" in omni_server.serve_args else "t2v"
    _, height, width = VARIANTS[variant]
    output_dir = accuracy_artifact_root / "sana_video" / variant / task
    _download_case(variant, task, output_dir, {"pipeline.mp4"})

    request_config = {
        "model": omni_server.model,
        "form_data": {
            "prompt": PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
            "height": height,
            "width": width,
            "num_frames": 81,
            "fps": 16,
            "num_inference_steps": 50,
            "guidance_scale": 6.0,
            "seed": 42,
        },
    }
    if task == "i2v":
        encoded = base64.b64encode((output_dir / "input.png").read_bytes()).decode()
        request_config["image_reference"] = f"data:image/png;base64,{encoded}"
    result = openai_client.send_video_diffusion_request(request_config)[0]
    actual_path = output_dir / "actual.mp4"
    actual_path.write_bytes(result.videos[0])
    golden_path = output_dir / "pipeline.mp4"

    metadata = probe_video(actual_path)
    assert_video_metadata(metadata, width=width, height=height, fps=16, frame_count=81)
    assert_video_similarity_metrics(
        label=f"sana_video_{variant}_{task}",
        online_path=actual_path,
        offline_path=golden_path,
        # The native prompt cross-attention uses vLLM-Omni Attention rather
        # than Diffusers SDPA. Frozen stage checks prove exact inputs and
        # scheduler state, while the BF16 backend reduction difference grows
        # across CFG denoising. Calibrate only the affected 480p T2V case;
        # its frozen result is 0.919861 SSIM / 28.7442 dB PSNR.
        ssim_threshold=SSIM_THRESHOLDS[(variant, task)],
        psnr_threshold=28.0,
    )
