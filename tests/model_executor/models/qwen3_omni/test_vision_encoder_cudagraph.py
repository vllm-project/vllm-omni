# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for Qwen3-Omni stage routing and encoder graph buffers."""

from contextlib import nullcontext
from inspect import getattr_static
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeConfig
from vllm.model_executor.models.interfaces import supports_encoder_cudagraph
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.encoder_cudagraph import BudgetGraphMetadata, EncoderCudaGraphManager
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

import vllm_omni.model_executor.models.qwen3_omni.qwen3_omni as outer_module
import vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker as thinker_module
import vllm_omni.model_executor.models.qwen3_omni.vision_encoder_cudagraph as graph_module
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3Omni_VisionTransformer,
    Qwen3OmniMoeThinkerForConditionalGeneration,
)
from vllm_omni.worker.encoder_cudagraph import SingleReplayEncoderCudaGraphManager
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _first_pipeline_rank(monkeypatch):
    monkeypatch.setattr(graph_module, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True))


class _PatchEmbed(nn.Module):
    patch_size = 2
    temporal_patch_size = 1

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)
        self.proj.in_channels = 1
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(4))

    def forward(self, pixels):
        return self.proj(pixels)


class _Block(nn.Module):
    def forward(self, x, *, cu_seqlens, rotary_pos_emb_cos, rotary_pos_emb_sin, max_seqlen, sequence_lengths):
        return x + rotary_pos_emb_cos[:, None, :] + rotary_pos_emb_sin[:, None, :]


class _Merger(nn.Module):
    def forward(self, x):
        return x.reshape(-1, 4, 4).sum(dim=1)


def _vision():
    model = object.__new__(Qwen3Omni_VisionTransformer)
    nn.Module.__init__(model)
    model.hidden_size = 4
    model.spatial_merge_size = 2
    model.tp_size = 1
    model.attn_backend = AttentionBackendEnum.FLASH_ATTN
    model.apply_vit_abs_pos_embed = True
    model.deepstack_visual_indexes = [0, 1]
    model.patch_embed = _PatchEmbed()
    model.blocks = nn.ModuleList([_Block(), _Block()])
    model.merger = _Merger()
    model.merger_list = nn.ModuleList([_Merger(), _Merger()])

    def positions(grid):
        return torch.cat([torch.arange(int(t * h * w)).float().unsqueeze(1).expand(-1, 4) + int(w) for t, h, w in grid])

    model.fast_pos_embed_interpolate = positions
    model.rot_pos_emb = lambda grid: (positions(grid) / 10, positions(grid) / 20)
    return model


def _config(stage="thinker", *, embeds=False):
    mm = SimpleNamespace(
        enable_mm_embeds=embeds,
        get_limit_per_prompt=lambda modality: 4,
        mm_encoder_tp_mode="weights",
        mm_encoder_attn_dtype=None,
    )
    hf = Qwen3OmniMoeConfig(tts_bos_token_id=1, tts_eos_token_id=2, tts_pad_token_id=3)
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf, multimodal_config=mm, model_stage=stage, max_model_len=32),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=32),
        compilation_config=SimpleNamespace(
            cudagraph_mm_encoder=True,
            encoder_cudagraph_token_budgets=[2, 4],
            encoder_cudagraph_max_vision_items_per_batch=2,
            encoder_cudagraph_max_frames_per_batch=None,
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        device_config=SimpleNamespace(device="cpu"),
        speculative_config=None,
    )
    cfg.with_hf_config = lambda *args, **kwargs: cfg
    return cfg


def _thinker(cfg=None):
    model = object.__new__(Qwen3OmniMoeThinkerForConditionalGeneration)
    nn.Module.__init__(model)
    model.visual = _vision()
    model.vllm_config = cfg or _config()
    model.multimodal_config = model.vllm_config.model_config.multimodal_config
    model.visual_dim = 4
    model.multiscale_dim = 8
    model.make_empty_intermediate_tensors = lambda **kwargs: None
    # The thinker wrapper reads the first pipeline layer to size captures.
    model.language_model = SimpleNamespace(model=SimpleNamespace(start_layer=0))
    model._enable_image_encoder_cudagraph()
    return model


def _combined(monkeypatch, stage, *, embeds=False):
    cfg = _config(stage, embeds=embeds)
    thinker = _thinker(cfg)
    monkeypatch.setattr(outer_module, "init_vllm_registered_model", lambda **kwargs: thinker)
    monkeypatch.setattr(
        outer_module, "current_omni_platform", SimpleNamespace(supports_talker_mtp_graph_capture=lambda: False)
    )
    cls = outer_module.Qwen3OmniMoeForConditionalGeneration
    monkeypatch.setattr(cls, "_init_special_tokens_embeddings", lambda self: None)
    monkeypatch.setattr(cls, "_get_talker_suppressed_tokens", lambda self: [])
    return cls(vllm_config=cfg), cfg


def _runner_factory(model, cfg, supports_mm_inputs=True):
    runner = object.__new__(OmniGPUModelRunner)
    runner.compilation_config = cfg.compilation_config
    runner.supports_mm_inputs = supports_mm_inputs
    runner.model = model
    runner.vllm_config = cfg
    runner.device = torch.device("cpu")
    runner.dtype = torch.float32
    return runner._create_encoder_cudagraph_manager()


@pytest.mark.parametrize("stage,enabled", [("thinker", True), ("talker", False), ("code2wav", False), (None, True)])
def test_real_wrapper_construction_controls_runner_protocol(monkeypatch, stage, enabled):
    model, cfg = _combined(monkeypatch, stage)
    assert supports_encoder_cudagraph(model) is enabled
    manager = _runner_factory(model, cfg)
    assert (manager is not None) is enabled
    if enabled:
        assert getattr_static(model, "encoder_cudagraph_forward").__self__ is model.thinker
        assert type(manager) is SingleReplayEncoderCudaGraphManager
        assert manager.model is model
        assert manager.config.out_hidden_size == 12
        assert manager.supports_modality("image")
        assert not manager.supports_modality("video")
        assert not manager.supports_modality("audio")


def test_runner_keeps_the_upstream_manager_without_the_opt_in(monkeypatch):
    model, cfg = _combined(monkeypatch, "thinker")
    model.encoder_cudagraph_single_replay = False
    assert type(_runner_factory(model, cfg)) is EncoderCudaGraphManager


def test_runner_builds_no_manager_without_multimodal_inputs(monkeypatch):
    model, cfg = _combined(monkeypatch, "thinker")
    assert _runner_factory(model, cfg, supports_mm_inputs=False) is None


def test_thinker_construction_enables_the_image_graph(monkeypatch):
    cfg = _config()
    cfg.quant_config = None
    thinker_config = cfg.model_config.hf_config.thinker_config
    thinker_config.architectures = ["Qwen3OmniMoeThinkerForConditionalGeneration"]
    thinker_config.vision_config.deepstack_visual_indexes = [0, 1]
    thinker_config.vision_config.out_hidden_size = 4
    thinker_config.text_config.rms_norm_eps = 1e-6
    thinker_config.text_config.hidden_size = 4
    language_model = nn.Module()
    language_model.make_empty_intermediate_tensors = None
    monkeypatch.setattr(thinker_module, "Qwen3OmniMoeAudioEncoder", lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(thinker_module, "Qwen3Omni_VisionTransformer", lambda **kwargs: _vision())
    monkeypatch.setattr(thinker_module, "Qwen3MoeLLMForCausalLM", lambda **kwargs: language_model)
    cls = Qwen3OmniMoeThinkerForConditionalGeneration
    monkeypatch.setattr(cls, "_mark_tower_model", lambda self, *args: nullcontext())
    monkeypatch.setattr(cls, "_mark_language_model", lambda self, *args: nullcontext())
    assert supports_encoder_cudagraph(cls(vllm_config=cfg))


@pytest.mark.parametrize("change", ["no_image_limit", "sdpa_backend", "fp8_attention", "later_pipeline_rank", None])
def test_image_graph_gate(monkeypatch, change):
    model = _thinker()
    del model.supports_encoder_cudagraph
    mm = model.multimodal_config
    if change == "no_image_limit":
        mm.get_limit_per_prompt = lambda modality: 0 if modality == "image" else 4
    elif change == "sdpa_backend":
        model.visual.attn_backend = AttentionBackendEnum.TORCH_SDPA
    elif change == "fp8_attention":
        mm.mm_encoder_attn_dtype = "fp8"
    elif change == "later_pipeline_rank":
        monkeypatch.setattr(graph_module, "get_pp_group", lambda: SimpleNamespace(is_first_rank=False))
    model._enable_image_encoder_cudagraph()
    assert supports_encoder_cudagraph(model) is (change is None)


def test_precomputed_embeddings_keep_existing_encoder_entry(monkeypatch):
    model, cfg = _combined(monkeypatch, "thinker", embeds=True)
    assert not supports_encoder_cudagraph(model.thinker)
    assert _runner_factory(model, cfg) is None


def _images(grids, offset=0):
    patches = sum(t * h * w for t, h, w in grids)
    return {
        "image_grid_thw": torch.tensor(grids),
        "pixel_values": torch.arange(patches * 4).float().reshape(patches, 4) + offset,
    }


def _cpu_manager(model):
    manager = SingleReplayEncoderCudaGraphManager(model.vllm_config, torch.device("cpu"), torch.float32, model)
    # Replace only CUDA launch/capture; keep production packing, selectors,
    # metadata copy, output slicing and ownership behavior.
    for budget in manager.token_budgets:
        values = model.prepare_encoder_cudagraph_capture_inputs(
            budget, manager.max_batch_size, manager.max_frames_per_batch, torch.device("cpu"), torch.float32
        ).values
        output = torch.empty_like(model.encoder_cudagraph_forward(values))

        def replay(values=values, output=output):
            output.copy_(model.encoder_cudagraph_forward(values))

        manager.budget_graphs.setdefault("default", {})[budget] = BudgetGraphMetadata(
            token_budget=budget,
            max_batch_size=manager.max_batch_size,
            max_frames_per_batch=manager.max_frames_per_batch,
            graph=SimpleNamespace(replay=replay),
            input_buffers=values,
            output_buffer=output,
        )
    return manager


def test_manager_preserves_item_order_deepstack_and_previous_outputs():
    model = _thinker()
    manager = _cpu_manager(model)
    inputs = _images([[1, 2, 4], [1, 2, 2]])  # packing reorders 2/1-token images
    expected = model.encoder_eager_forward(inputs).split([2, 1])
    outputs = manager.execute(inputs)
    assert manager.graph_hits == 2
    assert [output.shape for output in outputs] == [(2, 12), (1, 12)]
    for output, reference in zip(outputs, expected):
        torch.testing.assert_close(output, reference)
        # The two captured intermediate levels must remain in their own slots.
        assert not torch.equal(output[:, 4:8], output[:, 8:12])
    saved = [output.clone() for output in outputs]
    changed = _images([[1, 4, 2], [1, 2, 4]], offset=1000)
    actual = manager.execute(changed)
    for output, reference in zip(actual, model.encoder_eager_forward(changed).split([2, 2])):
        torch.testing.assert_close(output, reference)
    for output, reference in zip(outputs, saved):
        torch.testing.assert_close(output, reference)


@pytest.mark.parametrize(
    "grids",
    [
        [[1, 2, 10]],  # one image above the largest budget
        [[1, 2, 4], [1, 2, 6]],  # 2 + 3 tokens exceed the largest budget
        [[1, 2, 2]] * 3,  # more images than one replay holds
    ],
)
def test_group_that_does_not_fit_one_replay_goes_back_to_the_runner(grids):
    model = _thinker()
    manager = _cpu_manager(model)
    assert manager.execute(_images(grids)) is None
    assert manager.graph_hits == 0 and manager.graph_misses == len(grids)


def test_empty_shard_is_valid():
    model = _thinker()
    manager = _cpu_manager(model)
    inputs = _images([[1, 2, 10]])
    empty = model.select_encoder_cudagraph_items(inputs, [])
    assert empty["pixel_values"].shape == (0, 4)
    assert manager.execute(empty) == []


def test_budget_range_follows_scheduler_and_model_limits():
    model = _thinker()
    cfg = _config()
    assert model.get_encoder_cudagraph_budget_range(cfg) == (32, 32)
    cfg.scheduler_config.max_num_batched_tokens = 128
    cfg.model_config.max_model_len = 4096
    assert model.get_encoder_cudagraph_budget_range(cfg) == (64, 128)
    # The shipped deploy's 32768-token batch stops at the replay-profitable cap.
    cfg.scheduler_config.max_num_batched_tokens = 32768
    cfg.model_config.max_model_len = 65536
    assert model.get_encoder_cudagraph_budget_range(cfg) == (64, 256)


@pytest.mark.parametrize("grid", [[2, 2, 2], [1, 3, 2], [1, 2, 0]])
def test_multi_frame_or_unmergeable_image_grid_is_rejected(grid):
    with pytest.raises(ValueError, match="Invalid Qwen3-Omni image grid"):
        _thinker().get_encoder_cudagraph_item_specs(_images([grid]))


def test_capture_has_full_budget_and_replay_keeps_scalar_bound():
    model = _thinker()
    values = model.prepare_encoder_cudagraph_capture_inputs(3, 2, 0, torch.device("cpu"), torch.float32).values
    assert values["pixel_values"].shape == (12, 4)  # non-divisible budget/max-items
    replay = model.prepare_encoder_cudagraph_replay_buffers(_images([[1, 2, 2]]), 2, 0)
    assert "max_seqlen" not in replay.values
    assert values["max_seqlen"].item() == 12
    model.get_encoder_cudagraph_config().padding_logics["cu_seqlens"](values["cu_seqlens"], replay.values["cu_seqlens"])
    assert values["cu_seqlens"].tolist() == [0, 4, 4]


def test_graph_entry_preserves_vision_module_hooks():
    model = _thinker()
    values = model.prepare_encoder_cudagraph_capture_inputs(2, 2, 0, torch.device("cpu"), torch.float32).values
    calls = []
    hook = model.visual.register_forward_pre_hook(lambda module, args: calls.append(args[0]))
    try:
        model.encoder_cudagraph_forward(values)
    finally:
        hook.remove()
    assert len(calls) == 1 and calls[0] is values["pixel_values"]


def _run_encoder_step(monkeypatch, model, manager, batches):
    import vllm.v1.worker.gpu_model_runner as runner_module

    monkeypatch.setattr(runner_module, "group_and_batch_mm_kwargs", lambda *args, **kwargs: iter(batches))
    keys = [f"item{index}" for index in range(sum(count for _, count, _ in batches))]
    cached: dict[str, torch.Tensor] = {}
    runner = SimpleNamespace(
        _batch_mm_inputs_from_scheduler=lambda _: (keys, [(m, {}) for m, _, _ in batches], []),
        observability_config=None,
        model=model,
        lora_config=None,
        device=torch.device("cpu"),
        is_multimodal_pruning_enabled=False,
        requires_sequential_video_encoding=False,
        timed_encoder_operation=lambda *args: nullcontext(),
        encoder_cudagraph_manager=manager,
        _cache_encoder_output=lambda key, value, *args: cached.__setitem__(key, value),
    )
    scheduler = SimpleNamespace(ec_manager_metadata=None, free_encoder_mm_hashes=[])
    GPUModelRunner._execute_mm_encoder(runner, scheduler)
    return [cached[key] for key in keys]


def test_actual_runner_dispatch_keeps_video_and_audio_eager(monkeypatch):
    model = _thinker()
    manager = _cpu_manager(model)
    eager_modalities = []

    def embed_multimodal(**kwargs):
        eager_modalities.append(kwargs["kind"])
        return [torch.zeros(1, 12)]

    model.embed_multimodal = embed_multimodal
    image = _images([[1, 2, 2]])
    expected = model.encoder_eager_forward(image)
    batches = [("image", 1, image), ("video", 1, {"kind": "video"}), ("audio", 1, {"kind": "audio"})]
    cached = _run_encoder_step(monkeypatch, model, manager, batches)
    assert eager_modalities == ["video", "audio"]
    assert len(cached) == 3
    torch.testing.assert_close(cached[0], expected)
    assert manager.graph_hits == 1


def test_runner_encodes_a_group_beyond_one_replay_in_one_eager_call(monkeypatch):
    model = _thinker()
    manager = _cpu_manager(model)
    calls = []

    def embed_multimodal(**kwargs):
        calls.append(kwargs["image_grid_thw"].tolist())
        return list(model.encoder_eager_forward(kwargs).split([2, 3]))

    model.embed_multimodal = embed_multimodal
    images = _images([[1, 2, 4], [1, 2, 6]])
    cached = _run_encoder_step(monkeypatch, model, manager, [("image", 2, images)])
    assert calls == [[[1, 2, 4], [1, 2, 6]]]
    for output, reference in zip(cached, model.encoder_eager_forward(images).split([2, 3])):
        torch.testing.assert_close(output, reference)
    assert manager.graph_hits == 0 and manager.graph_misses == 2
