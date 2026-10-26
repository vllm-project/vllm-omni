# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from importlib import import_module

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3_decoder import (
    resolve_minimax_h3_decoder_model_path,
)
from vllm_omni.model_executor.models.minimax_h3.pipeline import MINIMAX_H3_DECODE_PIPELINE

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("model_subdir", "task_type", "partition"),
    [
        (None, None, "FL2VA"),
        (None, "t2va", "FL2VA"),
        (None, "ref2va", "Ref2VA"),
        (None, "combined", "FL2VA"),
        ("FL2VA", "auto", "FL2VA"),
        ("Ref2VA", None, "Ref2VA"),
        ("FL2VA", "ref2va", "Ref2VA"),
    ],
)
def test_decoder_stage_and_constructor_resolve_the_same_local_partition(tmp_path, model_subdir, task_type, partition):
    for subdir in ("FL2VA", "Ref2VA"):
        (tmp_path / subdir / "video_vae").mkdir(parents=True)
        (tmp_path / subdir / "audio_vae").mkdir()
    model = str(tmp_path / model_subdir) if model_subdir else str(tmp_path)
    resolver_path = MINIMAX_H3_DECODE_PIPELINE.stages[2].model_path_resolver
    module_name, function_name = resolver_path.rsplit(".", 1)
    resolver = getattr(import_module(module_name), function_name)
    assert resolver is resolve_minimax_h3_decoder_model_path

    stage_path = resolver(model, "decoder-revision", task_type)

    assert stage_path == str(tmp_path / partition)
    assert resolve_minimax_h3_decoder_model_path(stage_path, "decoder-revision", task_type) == stage_path


def test_decoder_resolver_rejects_an_unknown_task(tmp_path):
    with pytest.raises(ValueError, match="task_type"):
        resolve_minimax_h3_decoder_model_path(str(tmp_path), None, "unknown")


@pytest.mark.parametrize(
    ("quantized", "offload_mode", "expected_device"),
    [(True, "legacy", "cpu"), (True, "resident", "meta"), (True, "compact", "meta"), (False, "legacy", "meta")],
)
def test_decoder_constructor_routes_vae_quantization_and_legacy_residency(
    monkeypatch, quantized, offload_mode, expected_device
):
    from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3_decoder as pipeline_module
    from vllm_omni.quantization.component_config import ComponentQuantizationConfig
    from vllm_omni.quantization.int8_config import DiffusionInt8Config

    quant_config = DiffusionInt8Config() if quantized else None
    od_config = OmniDiffusionConfig(
        model="unused",
        revision=None,
        task_type=None,
        parallel_config=DiffusionParallelConfig(),
        quantization_config=ComponentQuantizationConfig({"video_vae": quant_config}),
        diffusion_offload_config={"mode": "layer", "components": ["dit"]} if offload_mode == "compact" else None,
        enable_cpu_offload=False,
        enable_layerwise_offload=offload_mode == "legacy",
        enable_distributed_layerwise_offload=False,
        trust_remote_code=True,
        enable_diffusion_pipeline_profiler=False,
    )
    video_calls: list[dict] = []
    audio_calls: list[dict] = []

    class FakeVideoVAE(torch.nn.Module):
        def __init__(self, _component_path, **kwargs):
            super().__init__()
            video_calls.append(kwargs)

    class FakeAudioVAE(torch.nn.Module):
        def __init__(self, _component_path, **kwargs):
            super().__init__()
            audio_calls.append(kwargs)

    monkeypatch.setattr(pipeline_module, "get_local_device", lambda: torch.device("meta"))
    monkeypatch.setattr(pipeline_module, "resolve_minimax_h3_decoder_model_path", lambda *_args: "unused/FL2VA")
    monkeypatch.setattr(pipeline_module, "MiniMaxH3VideoVAE", FakeVideoVAE)
    monkeypatch.setattr(pipeline_module, "MiniMaxH3AudioVAE", FakeAudioVAE)
    monkeypatch.setattr(
        pipeline_module.MiniMaxH3DecoderPipeline, "setup_diffusion_pipeline_profiler", lambda self, **_kwargs: None
    )

    pipeline = pipeline_module.MiniMaxH3DecoderPipeline(od_config=od_config)

    assert pipeline.vae is pipeline.video_vae
    assert video_calls[0]["quant_config"] is quant_config
    assert video_calls[0]["load_device"] == torch.device(expected_device)
    assert video_calls[0]["decode_only"] is True
    assert "load_device" not in audio_calls[0]
