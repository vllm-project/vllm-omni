# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for Qwen2.5-Omni stage routing and image encoder graph gating."""

from contextlib import nullcontext
from inspect import getattr_static
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.model_executor.models.interfaces import supports_encoder_cudagraph
from vllm.v1.attention.backends.registry import AttentionBackendEnum

import vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni as outer_module
import vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_thinker as thinker_module
import vllm_omni.model_executor.models.qwen2_5_omni.vision_encoder_cudagraph as graph_module
from vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_thinker import (
    Qwen2_5OmniThinkerForConditionalGeneration,
)
from vllm_omni.worker.encoder_cudagraph import SingleReplayEncoderCudaGraphManager
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _first_pipeline_rank(monkeypatch):
    monkeypatch.setattr(graph_module, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True))


def _visual():
    return SimpleNamespace(
        attn_backend=AttentionBackendEnum.FLASH_ATTN,
        out_hidden_size=16,
        spatial_merge_size=2,
    )


def _config(stage="thinker", *, embeds=False):
    mm = SimpleNamespace(
        enable_mm_embeds=embeds,
        get_limit_per_prompt=lambda modality: 4,
        mm_encoder_tp_mode="weights",
        mm_encoder_attn_dtype=None,
        skip_mm_profiling=False,
    )
    text = SimpleNamespace(vocab_size=8, hidden_size=4, rms_norm_eps=1e-6)
    hf = SimpleNamespace(
        thinker_config=SimpleNamespace(text_config=text, audio_config=None, vision_config=None),
        talker_config=SimpleNamespace(),
        token2wav_config=None,
    )
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf, multimodal_config=mm, model_stage=stage, max_model_len=4096),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=32768),
        compilation_config=SimpleNamespace(
            cudagraph_mm_encoder=True,
            encoder_cudagraph_token_budgets=[64, 128],
            encoder_cudagraph_max_vision_items_per_batch=2,
            encoder_cudagraph_max_frames_per_batch=None,
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        quant_config=None,
    )
    return cfg


def _thinker(cfg=None):
    model = object.__new__(Qwen2_5OmniThinkerForConditionalGeneration)
    nn.Module.__init__(model)
    model.visual = _visual()
    model.vllm_config = cfg or _config()
    model.multimodal_config = model.vllm_config.model_config.multimodal_config
    model.make_empty_intermediate_tensors = lambda **kwargs: None
    model._enable_image_encoder_cudagraph()
    return model


def _combined(monkeypatch, stage, *, embeds=False):
    cfg = _config(stage, embeds=embeds)
    thinker = _thinker(cfg)
    submodule = thinker if stage == "thinker" else nn.Identity()
    monkeypatch.setattr(outer_module, "init_vllm_registered_model", lambda **kwargs: submodule)
    cls = outer_module.Qwen2_5OmniForConditionalGeneration
    monkeypatch.setattr(cls, "_init_special_tokens_embeddings", lambda self: None, raising=False)
    monkeypatch.setattr(cls, "_mark_language_model", lambda self, **kwargs: nullcontext())
    return cls(vllm_config=cfg), cfg


def _runner_factory(model, cfg):
    runner = object.__new__(OmniGPUModelRunner)
    runner.compilation_config = cfg.compilation_config
    runner.supports_mm_inputs = True
    runner.model = model
    runner.vllm_config = cfg
    runner.device = torch.device("cpu")
    runner.dtype = torch.float32
    return runner._create_encoder_cudagraph_manager()


@pytest.mark.parametrize("stage,enabled", [("thinker", True), ("talker", False), ("code2wav", False)])
def test_wrapper_exposes_the_image_graph_on_the_thinker_stage_only(monkeypatch, stage, enabled):
    model, cfg = _combined(monkeypatch, stage)
    assert supports_encoder_cudagraph(model) is enabled
    manager = _runner_factory(model, cfg)
    assert (manager is not None) is enabled
    if enabled:
        assert getattr_static(model, "encoder_cudagraph_forward").__self__ is model.thinker
        # A group that does not fit one graph goes back to one eager call.
        assert type(manager) is SingleReplayEncoderCudaGraphManager
        assert manager.model is model
        assert manager.config.out_hidden_size == 16
        assert manager.supports_modality("image")
        assert not manager.supports_modality("video")
        assert not manager.supports_modality("audio")
        # Qwen2.5-VL's window-attention buffers come with the reused contract.
        assert {"window_index", "reverse_indices", "cu_window_seqlens"} <= set(manager.config.buffer_keys)


def test_thinker_construction_enables_the_image_graph(monkeypatch):
    cfg = _config()
    # The thinker is built from the thinker sub-config itself.
    cfg.model_config.hf_config = SimpleNamespace(
        audio_config=SimpleNamespace(),
        vision_config=SimpleNamespace(),
        text_config=cfg.model_config.hf_config.thinker_config.text_config,
    )
    language_model = nn.Module()
    language_model.make_empty_intermediate_tensors = None
    monkeypatch.setattr(thinker_module, "Qwen2_5OmniAudioEncoder", lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(thinker_module, "Qwen2_5_VisionTransformer", lambda **kwargs: _visual())
    monkeypatch.setattr(thinker_module, "init_vllm_registered_model", lambda **kwargs: language_model)
    cls = Qwen2_5OmniThinkerForConditionalGeneration
    monkeypatch.setattr(cls, "_mark_tower_model", lambda self, *args: nullcontext())
    monkeypatch.setattr(cls, "_mark_language_model", lambda self, *args: nullcontext())
    assert supports_encoder_cudagraph(cls(vllm_config=cfg))


@pytest.mark.parametrize(
    "change",
    ["no_image_limit", "no_visual", "sdpa_backend", "flashinfer_backend", "fp8_attention", "later_pipeline_rank", None],
)
def test_image_graph_gate(monkeypatch, change):
    model = _thinker()
    del model.supports_encoder_cudagraph
    mm = model.multimodal_config
    if change == "no_image_limit":
        mm.get_limit_per_prompt = lambda modality: 0 if modality == "image" else 4
    elif change == "no_visual":
        model.visual = None
    elif change == "sdpa_backend":
        model.visual.attn_backend = AttentionBackendEnum.TORCH_SDPA
    elif change == "flashinfer_backend":
        model.visual.attn_backend = AttentionBackendEnum.FLASHINFER
    elif change == "fp8_attention":
        mm.mm_encoder_attn_dtype = "fp8"
    elif change == "later_pipeline_rank":
        monkeypatch.setattr(graph_module, "get_pp_group", lambda: SimpleNamespace(is_first_rank=False))
    warnings = []
    monkeypatch.setattr(graph_module, "logger", SimpleNamespace(warning=lambda msg, *args: warnings.append(msg % args)))
    model._enable_image_encoder_cudagraph()
    assert supports_encoder_cudagraph(model) is (change is None)
    # An explicit opt-in that cannot apply says why; later pipeline ranks never run the encoder.
    assert len(warnings) == (0 if change in (None, "later_pipeline_rank") else 1)

    warnings.clear()
    model.vllm_config.compilation_config.cudagraph_mm_encoder = False
    model._enable_image_encoder_cudagraph()
    assert not warnings


def test_precomputed_embeddings_keep_existing_encoder_entry(monkeypatch):
    model, cfg = _combined(monkeypatch, "thinker", embeds=True)
    assert not supports_encoder_cudagraph(model.thinker)
    assert _runner_factory(model, cfg) is None


def test_video_items_are_not_graph_inputs():
    model = _thinker()
    assert model.get_input_modality({"image_grid_thw": torch.tensor([[1, 2, 2]])}) == "image"
    with pytest.raises(ValueError, match="only take images"):
        model.get_input_modality({"video_grid_thw": torch.tensor([[2, 2, 2]])})
    assert model.get_max_frames_per_video() == 1


def test_budget_range_follows_scheduler_and_model_limits():
    model = _thinker()
    cfg = _config()
    cfg.scheduler_config.max_num_batched_tokens = 128
    assert model.get_encoder_cudagraph_budget_range(cfg) == (64, 128)
    cfg.scheduler_config.max_num_batched_tokens = 32
    assert model.get_encoder_cudagraph_budget_range(cfg) == (32, 32)
    # The shipped deploy's 32768-token batch stops at the replay-profitable cap.
    cfg.scheduler_config.max_num_batched_tokens = 32768
    assert model.get_encoder_cudagraph_budget_range(cfg) == (64, 256)
