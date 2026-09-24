# SPDX-License-Identifier: Apache-2.0
"""Exercise the offline runner with a CPU engine double and real MP4 metadata."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from test_cookbook import action_record, manifest

from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.config import (
    COSMOS3_NANO_SIM_BIMANUAL_ARTIFACT_FIELDS,
)
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.cookbook import resolve_asset

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def runner():
    path = ROOT / "examples/offline_inference/cosmos3_nano_sim_bimanual/cosmos3_nano_sim_bimanual.py"
    spec = importlib.util.spec_from_file_location("bimanual_cookbook_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_module(monkeypatch, name: str, **attributes: object) -> None:
    module = ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, module)


def test_asset_downloads_use_hf_cache_and_http_cache(tmp_path: Path, monkeypatch) -> None:
    download = Mock(return_value=str(tmp_path / "cached.mp4"))
    install_module(monkeypatch, "huggingface_hub", hf_hub_download=download)
    assert (
        resolve_asset(
            "https://huggingface.co/nvidia/private-preview/resolve/main/samples/a%20b.mp4",
            base_dir=tmp_path,
            cache_dir=tmp_path,
        )
        == tmp_path / "cached.mp4"
    )
    download.assert_called_once_with(repo_id="nvidia/private-preview", revision="main", filename="samples/a b.mp4")
    request = Mock(return_value=io.BytesIO(b"asset"))
    monkeypatch.setattr("urllib.request.urlopen", request)
    url = "https://example.test/conditioning.png"
    cached = resolve_asset(url, base_dir=tmp_path, cache_dir=tmp_path)
    assert cached.read_bytes() == b"asset"
    assert resolve_asset(url, base_dir=tmp_path, cache_dir=tmp_path) == cached
    assert request.call_count == 1


@pytest.mark.parametrize("kind", ["legacy", "cookbook"])
@pytest.mark.parametrize("use_overrides", [False, True])
def test_runner_batch_status_and_camera_poses(
    tmp_path: Path, monkeypatch, runner, kind: str, use_overrides: bool
) -> None:
    imageio = pytest.importorskip("imageio.v2")
    pytest.importorskip("imageio_ffmpeg")
    artifact = manifest(kind)
    payload = {
        key: getattr(artifact, key)
        for key in COSMOS3_NANO_SIM_BIMANUAL_ARTIFACT_FIELDS
        if key not in ("conditioning", "fixed_step_sampler_config")
    }
    payload["conditioning"] = artifact.conditioning.model_dump(mode="json", exclude_none=True)
    payload["fixed_step_sampler_config"] = {
        "sample_type": artifact.sample_type,
        "t_list": list(artifact.t_list),
        "num_train_timesteps": artifact.num_train_timesteps,
    }
    model = tmp_path / "model"
    (model / "transformer").mkdir(parents=True)
    (model / "transformer/config.json").write_text(json.dumps({"cosmos3_nano_sim_bimanual": payload}))
    action = {**action_record(tmp_path, 5), "name": "../action", "height": 32, "width": 32}
    invalid = {**action, "name": "invalid", "num_frames": 6}
    camera = {
        "name": "camera",
        "camera_trajectory": "w-5",
        "model_mode": "text2video",
        "camera_num_frames": 5,
        "num_frames": 901,
        "height": 32,
        "width": 32,
        "seed": 13,
        "prompt": "A room.",
        "duration_template": " D={duration:.1f}",
        "resolution_template": None,
    }
    if kind == "legacy":
        camera.update(
            camera_pose_convention="backward_framewise",
            camera_action_normalization="scale",
            camera_translation_scale=10.0,
        )
    decoder_failure = {**camera, "name": "decoder_failure", "seed": 99}
    jsonl = tmp_path / "inputs.jsonl"
    jsonl.write_text("\n".join(json.dumps(record) for record in (action, invalid, camera, decoder_failure)))
    output_dir = tmp_path / "outputs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--model",
            str(model),
            "--jsonl",
            str(jsonl),
            "--input-format",
            "cookbook",
            "--all-samples",
            "--output-dir",
            str(output_dir),
        ],
    )
    if use_overrides:
        sys.argv.extend(["--deploy-config", str(ROOT / "vllm_omni/deploy/cosmos3_nano_sim_bimanual_i4.yaml")])
    calls = []
    engines = []

    class FakeOmni:
        def __init__(self, **kwargs):
            engines.append(self)
            self.closed = False

        def generate(self, prompt, params):
            calls.append((prompt, params))
            if params.seed == 99:
                raise RuntimeError("simulated decoder failure")
            return np.full((1, params.num_frames, params.height, params.width, 3), 127, dtype=np.uint8)

        def close(self):
            self.closed = True

    def export_to_video(frames, path, fps):
        imageio.mimwrite(path, [(frame * 255).astype(np.uint8) for frame in frames], fps=fps)

    install_module(monkeypatch, "vllm_omni.entrypoints.omni", Omni=FakeOmni)
    install_module(monkeypatch, "vllm_omni.inputs.data", OmniDiffusionSamplingParams=SimpleNamespace)
    install_module(monkeypatch, "vllm_omni.platforms", current_omni_platform=SimpleNamespace(device_type="cpu"))
    install_module(monkeypatch, "vllm_omni.outputs", OmniRequestOutput=type("OmniRequestOutput", (), {}))
    install_module(monkeypatch, "diffusers.utils", export_to_video=export_to_video)
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 1
    statuses = json.loads((output_dir / "sample_outputs.json").read_text())
    assert [item["status"] for item in statuses] == ["success", "failed", "success", "failed"]
    assert "divisible" in statuses[1]["error"]
    assert len(engines) == 1 and engines[0].closed
    assert len(calls) == 3
    for status, (_, params) in zip((statuses[0], statuses[2]), calls):
        assert status["actual_num_frames"] == 5
        assert status["actual_fps"] == 30
        assert (status["actual_height"], status["actual_width"]) == (32, 32)
        assert Path(status["output"]).parent == output_dir
        assert params.extra_args["reset"] and params.extra_args["close_session"]
        assert params.guidance_scale == 1 and params.num_inference_steps is None
        assert status["window_frames"] == (226 if use_overrides else artifact.window_frames)
        assert status["num_steps_by_frame"] == ([4, 2] if use_overrides else [4])
        assert params.seed == params.generator.initial_seed()
        assert status["action_contract_sha256"] == artifact.action_contract_sha256
    assert [params.seed for _, params in calls] == [7, 13, 99]
    assert calls[0][1].extra_args["action_space"] == "model"
    assert calls[1][1].extra_args["action_space"] == "raw"
    assert statuses[2]["requested_num_frames"] == 901
    assert statuses[2]["effective_num_frames"] == 5
    assert np.asarray(json.loads(Path(statuses[2]["camera_trajectory_path"]).read_text())).shape == (5, 4, 4)
    for status, (prompt, _) in zip(statuses[2:], calls[1:]):
        assert status["effective_camera_recipe"]["pose_convention"] == (
            "backward_framewise" if kind == "legacy" else "backward_chunk_anchored_16f"
        )
        assert status["effective_camera_recipe"]["camera_num_frames"] == 5
        assert status["effective_prompt"] == prompt["prompt"] == "A room. D=0.0"
        assert status["prompt_templates"] == {"duration_template": " D={duration:.1f}", "resolution_template": None}
    assert "simulated decoder failure" in statuses[3]["error"]


def test_runner_bfloat16_video_conversion(runner) -> None:
    frames = runner._video_frames(torch.full((1, 3, 5, 32, 32), -1, dtype=torch.bfloat16))
    assert len(frames) == 5 and frames[0].shape == (32, 32, 3)
    assert np.count_nonzero(frames) == 0
