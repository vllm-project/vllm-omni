# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare a real tiny Motubrain policy through the native CLI and the Omni adapter."""

import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import load_file

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.motubrain2 import pipeline_motubrain2
from vllm_omni.diffusion.output_formatter import format_diffusion_outputs, normalize_diffusion_postprocess_output
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.entrypoints.openpi.connection import _pack, _unpack
from vllm_omni.entrypoints.openpi.serving import ServingRealtimeRobotOpenPI
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.cpu, pytest.mark.local_model, pytest.mark.diffusion]


def test_native_cli_matches_pipeline(tmp_path, monkeypatch):
    pytest.importorskip("motubrain2.inference", reason="Install the optional Motubrain runtime to run parity")
    assets = tmp_path / "tiny"
    subprocess.run([sys.executable, "-m", "motubrain2.tools.make_tiny_data", str(assets)], check=True)
    reference = tmp_path / "native.safetensors"
    seed = 17
    steps = 2
    subprocess.run(
        [
            sys.executable,
            "-m",
            "motubrain2.inference",
            "--model",
            str(assets / "model"),
            "--input",
            str(assets / "observation.json"),
            "--profile",
            str(assets / "profile.json"),
            "--wan",
            str(assets / "wan"),
            "--text-encoder",
            str(assets / "text"),
            "--device",
            "cpu",
            "--steps",
            str(steps),
            "--seed",
            str(seed),
            "--output",
            str(reference),
        ],
        check=True,
    )

    monkeypatch.setattr(pipeline_motubrain2, "get_local_device", lambda: torch.device("cpu"))
    config_path = assets / "model" / "config.json"
    architecture = json.loads(config_path.read_text())
    architecture["embodiments"] = {"test_robot": json.loads((assets / "profile.json").read_text())}
    config_path.write_text(json.dumps(architecture))
    pipeline = pipeline_motubrain2.Motubrain2Pipeline(
        od_config=OmniDiffusionConfig(
            model=str(assets),
            dtype=torch.float32,
            model_config={
                "bundle": str(assets / "model"),
                "embodiment": "test_robot",
                "wan": str(assets / "wan"),
                "text_encoder": str(assets / "text"),
                "steps": 3,
            },
        )
    )
    record = json.loads((assets / "observation.json").read_text())
    obs = {"prompt": record["instruction"], "return_native_outputs": True}
    for camera, (filename,) in record["cameras"].items():
        with Image.open(assets / filename) as frame:
            obs[camera] = np.array(frame.convert("RGB"))
    params = OmniDiffusionSamplingParams(
        generator=torch.Generator().manual_seed(seed),
        num_inference_steps=steps,
        extra_args={"session_id": "parity", "robot_obs": obs},
    )
    req = OmniDiffusionRequest(prompt=obs["prompt"], sampling_params=params, request_id="tiny-policy")
    result = pipeline(DiffusionRequestBatch([req]))
    assert result.error is None
    native = load_file(reference)
    assert native["decoded_valid"].all()
    np.testing.assert_allclose(result.output["actions"], native["decoded_action"].numpy(), atol=1e-6, rtol=0)
    (formatted,) = format_diffusion_outputs(
        request=req,
        od_config=pipeline.od_config,
        diffusion_output=result,
        output_data=result.output,
        postprocess_output=normalize_diffusion_postprocess_output(result.output),
    )
    capability = {"action_horizon": pipeline.horizon, "action_dim": pipeline.action_dim}
    serving = ServingRealtimeRobotOpenPI(
        engine_client=SimpleNamespace(model_config={"policy_server_config": capability})
    )
    response = _unpack(_pack(serving._extract_actions(formatted, return_native_outputs=True)))
    assert response.keys() == native.keys() == {"video", "action", "decoded_action", "decoded_valid"}
    for name, tensor in native.items():
        expected = tensor.numpy()
        assert response[name].dtype == expected.dtype
        np.testing.assert_allclose(response[name], expected, atol=1e-6, rtol=0)
    np.testing.assert_array_equal(serving._extract_actions(formatted), response["decoded_action"])
