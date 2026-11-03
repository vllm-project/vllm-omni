# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Serve a native VoxCPM2 adapter through the speech API."""

import json
from pathlib import Path

import pytest
import torch
import yaml
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

from tests.helpers.fixtures.runtime import omni_fixture_lock
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams, get_model_prefix, iter_omni_server
from tests.helpers.stage_config import get_deploy_config_path

pytestmark = [pytest.mark.slow, pytest.mark.tts]
MODEL = "openbmb/VoxCPM2"


@pytest.fixture(scope="module")
def omni_server(request, run_level, tmp_path_factory):
    directory = tmp_path_factory.mktemp("voxcpm2-lora")
    model = get_model_prefix() + MODEL
    weights = Path(model) / "model.safetensors"
    if not weights.is_file():
        weights = Path(hf_hub_download(MODEL, "model.safetensors"))
    projections = ["enc_to_lm_proj", "lm_to_dit_proj", "res_to_dit_proj", "fusion_concat_proj"]
    generator = torch.Generator().manual_seed(42)
    tensors = {}
    with safe_open(weights, framework="pt", device="cpu") as checkpoint:
        for key in checkpoint.keys():
            if not key.endswith(".weight"):
                continue
            name = key.removesuffix(".weight")
            attention = name.startswith(("base_lm.", "residual_lm.", "feat_decoder.estimator.")) and name.rsplit(
                ".", 1
            )[-1] in {"q_proj", "k_proj", "v_proj", "o_proj"}
            if not attention and name not in projections:
                continue
            out_features, in_features = checkpoint.get_slice(key).get_shape()
            # Small nonzero deltas exercise every group without retraining a voice.
            tensors[f"{name}.lora_A"] = torch.randn(2, in_features, generator=generator) * 0.001
            tensors[f"{name}.lora_B"] = torch.randn(out_features, 2, generator=generator) * 0.001
    assert tensors
    save_file(tensors, str(directory / "lora_weights.safetensors"))
    (directory / "lora_config.json").write_text(
        json.dumps({"lora_config": {"r": 2, "alpha": 2, "enable_lm": True, "enable_dit": True, "enable_proj": True}})
    )

    with open(get_deploy_config_path("voxcpm2.yaml")) as stream:
        config = yaml.safe_load(stream)
    stage = config["stages"][0]
    stage.update(
        max_num_seqs=4,
        kv_cache_memory_bytes=1024**3,
        gpu_memory_utilization=0.65,
        max_model_len=2048,
        max_num_batched_tokens=2048,
    )
    stage["default_sampling_params"]["max_tokens"] = 256
    runtime = stage["engine_extras"]["hf_overrides"]["voxcpm2_runtime_config"]
    runtime.update(startup_lora_path=str(directory), unified_decode_graph_max_batch_size=4)
    deploy_path = directory / "deploy.yaml"
    deploy_path.write_text(yaml.safe_dump(config))
    request.param = OmniServerParams(
        model=MODEL,
        stage_config_path=str(deploy_path),
        server_args=["--trust-remote-code", "--disable-log-stats"],
    )
    yield from iter_omni_server(request, run_level, omni_fixture_lock)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_startup_lora_concurrent_speech(omni_server, online_client):
    online_client.send_audio_speech_request(
        {
            "model": omni_server.model,
            "input": "The weather is nice today, perfect for a walk in the park.",
            "voice": "default",
            "response_format": "wav",
            "stream": False,
            "timeout": 300,
            "min_audio_bytes": 40_000,
        },
        request_num=4,
    )


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_startup_lora_streaming_speech(omni_server, online_client):
    online_client.send_audio_speech_request(
        {
            "model": omni_server.model,
            "input": "The weather is nice today, perfect for a walk in the park.",
            "voice": "default",
            "response_format": "wav",
            "stream": True,
            "stream_format": "audio",
            "timeout": 300,
            "min_audio_bytes": 40_000,
        }
    )
