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
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from safetensors.torch import load_file

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.motubrain2 import pipeline_motubrain2
from vllm_omni.diffusion.output_formatter import format_diffusion_outputs, normalize_diffusion_postprocess_output
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.entrypoints.motubrain2 import router
from vllm_omni.entrypoints.motubrain2_codec import pack_message, unpack_message

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

    class TinyEngine:
        def get_diffusion_od_config(self):
            metadata = {"action_horizon": pipeline.horizon, "action_dim": pipeline.action_dim}
            return SimpleNamespace(model_config={"motubrain2_metadata": metadata})

        async def generate(self, *, prompt, sampling_params_list, request_id):
            (params,) = sampling_params_list
            request = OmniDiffusionRequest(prompt=prompt, sampling_params=params, request_id=request_id)
            result = pipeline(DiffusionRequestBatch([request]))
            assert result.error is None
            (formatted,) = format_diffusion_outputs(
                request=request,
                od_config=pipeline.od_config,
                diffusion_output=result,
                output_data=result.output,
                postprocess_output=normalize_diffusion_postprocess_output(result.output),
            )
            yield formatted

    app = FastAPI()
    app.state.engine_client = TinyEngine()
    endpoint = "/v1/realtime/robot/motubrain2"
    app.include_router(router)
    obs.update(seed=seed, sampling_params={"num_inference_steps": steps})
    with TestClient(app) as client, client.websocket_connect(endpoint) as socket:
        metadata = unpack_message(socket.receive_bytes())
        assert metadata["action_horizon"] == pipeline.horizon
        socket.send_bytes(pack_message(obs))
        response = unpack_message(socket.receive_bytes())
        socket.send_bytes(pack_message({**obs, "return_native_outputs": False}))
        actions = unpack_message(socket.receive_bytes())

    native = load_file(reference)
    assert native["decoded_valid"].all()
    assert response.keys() == native.keys() == {"video", "action", "decoded_action", "decoded_valid"}
    for name, tensor in native.items():
        expected = tensor.numpy()
        assert response[name].dtype == expected.dtype
        np.testing.assert_array_equal(response[name], expected)
    assert actions.dtype == np.float32
    np.testing.assert_array_equal(actions, response["decoded_action"])
