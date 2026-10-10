# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

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

GOLDEN_BASE_URL = os.getenv("PAN2_GOLDEN_BASE_URL")
# A local copy of the tiny checkpoint is accepted when the golden was generated from the same bytes.
LOCAL_MODEL = os.getenv("VLLM_OMNI_PAN2_TINY_MODEL")
MODEL = LOCAL_MODEL or "wuqing157/tiny-pan2-modular-pipe"
# Pin the immutable commit once the tiny checkpoint is published on the Hub.
REVISION: str | None = "3ff3130b4527f61aaab7bde01465c9884dc84cdd"
SCHEMA_VERSION = 1
PROMPT = "A red fox trots through fresh snow in a pine forest at dawn."
NEGATIVE_PROMPT = "blurry, low quality, distorted"
HEIGHT = 128
WIDTH = 224
NUM_FRAMES = 313
FPS = 24
SSIM_THRESHOLDS = {"t2v": 0.93, "i2v": 0.93}

pytestmark = [
    pytest.mark.slow,
    pytest.mark.diffusion,
    pytest.mark.skipif(
        not GOLDEN_BASE_URL,
        reason="Set PAN2_GOLDEN_BASE_URL after publishing the frozen v1 S3 assets.",
    ),
]


class _TransformerCheckpoint(nn.Module):
    def __init__(self, transformer: nn.Module, model: str, revision: str | None) -> None:
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


def _directory_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(f"{path.relative_to(root).as_posix()}\0{path.stat().st_size}\0{_sha256(path)}\n".encode())
    return digest.hexdigest()


def _assert_hex_digest(value: Any, *, length: int = 64) -> None:
    assert isinstance(value, str) and len(value) == length
    assert all(character in "0123456789abcdef" for character in value)


def _assert_model_provenance(payload: dict[str, Any]) -> None:
    provenance = payload["model_provenance"]
    if LOCAL_MODEL is None:
        # Local checkpoint generation is deliberately marked non-publishable. Do
        # not permit such output to become the canonical reference by accident.
        assert payload["publishable"] is True
        assert provenance == {"kind": "huggingface_hub", "model_id": MODEL, "revision": REVISION}
    else:
        assert payload["publishable"] is False
        assert provenance["kind"] == "local_checkpoint"
        assert provenance["checkpoint_fingerprint"]["sha256"] == _directory_sha256(Path(LOCAL_MODEL).resolve())


def _validate_manifest(payload: Any, task: str) -> dict[str, Any]:
    assert isinstance(payload, dict)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["task"] == task
    _assert_model_provenance(payload)
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


def _validate_metadata(payload: Any, manifest: dict[str, Any], task: str) -> None:
    assert isinstance(payload, dict)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["task"] == task
    assert payload["publishable"] == manifest["publishable"]
    assert payload["model_provenance"] == manifest["model_provenance"]
    assert payload["prompt"] == PROMPT
    assert payload["negative_prompt"] == NEGATIVE_PROMPT
    assert payload["seed"] == 42
    assert payload["generator_device"] == "cuda"
    assert payload["dtype"] == "bfloat16"
    assert payload["height"] == HEIGHT
    assert payload["width"] == WIDTH
    assert payload["num_frames"] == NUM_FRAMES
    assert payload["fps"] == FPS
    assert payload["num_inference_steps"] == 50
    assert payload["guidance_scale"] == 3.0

    implementation = payload["reference_implementation"]
    assert implementation["library"] == "diffusers"
    assert implementation["pipeline_class"] == "PAN2ModularPipeline"
    assert implementation["version"]
    assert implementation["torch_version"]

    generator = payload["generator_provenance"]
    assert generator["repository"] == "vllm-project/vllm-omni"
    _assert_hex_digest(generator["revision"], length=40)
    _assert_hex_digest(generator["generator_sha256"])
    assert generator["dirty"] is False

    scheduler = payload["scheduler"]
    assert scheduler["class"] == "FlowMatchEulerDiscreteScheduler"
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


def _download_case(task: str, output_dir: Path, filenames: set[str]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    assert GOLDEN_BASE_URL is not None
    base_url = f"{GOLDEN_BASE_URL.rstrip('/')}/{task}"
    response = requests.get(f"{base_url}/manifest.json", timeout=60)
    response.raise_for_status()
    payload = _validate_manifest(response.json(), task)
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
    _validate_metadata(metadata, payload, task)
    return payload


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "_hardware",
    [pytest.param(None, marks=hardware_marks(res={"cuda": "L4"}))],
)
def test_pan2_transformer_matches_frozen_golden(
    _hardware,
    accuracy_artifact_root: Path,
    pan2_transformer_runtime,
) -> None:
    del _hardware
    from vllm.config import LoadConfig
    from vllm.transformers_utils.config import get_hf_file_to_dict

    from vllm_omni.diffusion.data import TransformerConfig
    from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
    from vllm_omni.diffusion.models.pan2 import PAN2Transformer3DModel
    from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs

    output_dir = accuracy_artifact_root / "pan2" / "t2v"
    _download_case("t2v", output_dir, {"transformer_case.safetensors"})
    case = load_file(output_dir / "transformer_case.safetensors")
    transformer_config = get_hf_file_to_dict("transformer/config.json", MODEL, revision=REVISION)
    assert transformer_config is not None
    od_config = pan2_transformer_runtime
    transformer_kwargs = get_transformer_config_kwargs(
        TransformerConfig.from_dict(transformer_config), PAN2Transformer3DModel
    )
    transformer = PAN2Transformer3DModel(od_config=od_config, **transformer_kwargs).to(
        device="cuda", dtype=torch.bfloat16
    )
    checkpoint = _TransformerCheckpoint(transformer, MODEL, REVISION)
    loader = DiffusersPipelineLoader(LoadConfig(), od_config=od_config)
    transformer.load_weights(
        (name.removeprefix("transformer."), tensor)
        for name, tensor in loader.get_all_weights(checkpoint)
        if name.startswith("transformer.")
    )
    transformer.eval()
    with torch.inference_mode():
        actual = transformer(
            hidden_states=case["hidden_states"].to("cuda"),
            timestep=case["timestep"].to("cuda"),
            encoder_hidden_states=case["encoder_hidden_states"].to("cuda"),
        ).float()
    expected = case["output"].to("cuda").float()
    error = actual - expected
    max_abs = error.abs().max()
    relative_l2 = torch.linalg.vector_norm(error) / torch.linalg.vector_norm(expected)
    cosine = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
    assert max_abs.item() <= 0.1
    assert relative_l2.item() <= 0.015
    assert cosine.item() >= 0.999


SERVER_CASES = [
    pytest.param(
        OmniServerParams(model=MODEL, server_args=["--no-guardrails"]),
        task,
        id=task,
        marks=hardware_marks(res={"cuda": "L4"}),
    )
    for task in ("t2v", "i2v")
]


@pytest.mark.benchmark
@pytest.mark.parametrize(("omni_server", "task"), SERVER_CASES, indirect=["omni_server"])
def test_pan2_pipeline_matches_frozen_golden(
    omni_server: OmniServer,
    task: str,
    openai_client: OpenAIClientHandler,
    accuracy_artifact_root: Path,
) -> None:
    output_dir = accuracy_artifact_root / "pan2" / task
    _download_case(task, output_dir, {"pipeline.mp4"})

    request_config = {
        "model": omni_server.model,
        "form_data": {
            "prompt": PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
            "height": HEIGHT,
            "width": WIDTH,
            "num_frames": NUM_FRAMES,
            "fps": FPS,
            "num_inference_steps": 50,
            "guidance_scale": 3.0,
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
    assert_video_metadata(metadata, width=WIDTH, height=HEIGHT, fps=FPS, frame_count=NUM_FRAMES)
    assert_video_similarity_metrics(
        label=f"pan2_{task}",
        online_path=actual_path,
        offline_path=golden_path,
        ssim_threshold=SSIM_THRESHOLDS[task],
        psnr_threshold=28.0,
    )
