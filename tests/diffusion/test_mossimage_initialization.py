# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Initialize MOSS-Image with real scheduler metadata and mocked weights."""

import json
from pathlib import Path

import pytest
import torch

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.mossimage import pipeline_mossimage as pipeline_module

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("source", ["hub", "local"])
@pytest.mark.parametrize("revision", [None, "pinned-revision"])
def test_initialization_resolves_scheduler_and_preserves_revision(tmp_path, monkeypatch, mocker, source, revision):
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", False)

    snapshot = tmp_path / "hf-cache" / "snapshots" / (revision or "main")
    scheduler_dir = snapshot / "scheduler"
    scheduler_dir.mkdir(parents=True)
    config_path = scheduler_dir / "scheduler_config.json"
    config_path.write_text(
        json.dumps(
            {
                "_class_name": "MOSSImageScheduler",
                "num_train_timesteps": 64,
                "shift": 7.0,
                "training_shift": 1.25,
            }
        ),
        encoding="utf-8",
    )
    repo_id = "OpenMOSS-Team/MOSS-Image-Base"
    model = repo_id if source == "hub" else str(snapshot)
    assert not Path(repo_id).exists()

    download = mocker.patch("diffusers.configuration_utils.hf_hub_download", return_value=str(config_path))
    prefetch = mocker.patch.object(pipeline_module, "prefetch_subfolders")
    mocker.patch.object(pipeline_module, "get_local_device", return_value=torch.device("cpu"))
    encoder = mocker.patch.object(pipeline_module.AutoModel, "from_pretrained", return_value=torch.nn.Identity())
    tokenizer = mocker.patch.object(
        pipeline_module.AutoTokenizer, "from_pretrained", return_value=mocker.sentinel.tokenizer
    )
    vae = mocker.patch.object(pipeline_module.AutoencoderKLFlux2, "from_pretrained", return_value=torch.nn.Identity())
    mocker.patch.object(pipeline_module, "MOSSImageTransformer", return_value=torch.nn.Identity())

    config = OmniDiffusionConfig(model=model, revision=revision)
    pipeline = pipeline_module.MOSSImagePipeline(od_config=config)

    assert type(pipeline.scheduler) is pipeline_module.MOSSImageScheduler
    assert pipeline.scheduler.config.num_train_timesteps == 64
    assert pipeline.scheduler.config.shift == 7.0
    assert pipeline.scheduler.config.training_shift == 1.25
    pipeline.scheduler.set_timesteps(4, device="cpu")
    expected = pipeline_module.MOSSImageScheduler(num_train_timesteps=64, shift=7.0, training_shift=1.25)
    expected.set_timesteps(4, device="cpu")
    torch.testing.assert_close(pipeline.scheduler.sigmas, expected.sigmas)

    local_files_only = source == "local"
    if source == "hub":
        download.assert_called_once()
        assert download.call_args.args == (repo_id,)
        assert download.call_args.kwargs["filename"] == "scheduler_config.json"
        assert download.call_args.kwargs["subfolder"] == "scheduler"
        assert download.call_args.kwargs["revision"] == revision
        assert download.call_args.kwargs["local_files_only"] is False
    else:
        download.assert_not_called()
    assert not Path(repo_id).exists()

    prefetch.assert_called_once_with(
        model,
        ["scheduler", "text_encoder", "tokenizer", "vae"],
        local_files_only=local_files_only,
        revision=revision,
    )
    for loader, subfolder in ((encoder, "text_encoder"), (tokenizer, "tokenizer"), (vae, "vae")):
        loader.assert_called_once()
        assert loader.call_args.args == (model,)
        assert loader.call_args.kwargs["subfolder"] == subfolder
        assert loader.call_args.kwargs["revision"] == revision
        assert loader.call_args.kwargs["local_files_only"] is local_files_only
    assert pipeline.weights_sources[0].model_or_path == model
    assert pipeline.weights_sources[0].revision == revision
