# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Captured Qwen3-Omni image encoder graphs replay the eager tower on CUDA."""

import socket
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from tests.helpers.mark import hardware_marks

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": "L4"}),
]

DTYPE = torch.bfloat16
# Four input patches per output token; budgets are in output tokens.
BUDGETS = [8, 64]


def _tol(reference):
    # bf16 relative tolerance, with the absolute floor scaled to the output.
    return {"rtol": 1.6e-2, "atol": 1.6e-2 * reference.abs().max().item()}


def _vision_config():
    return SimpleNamespace(
        hidden_size=128,
        num_heads=2,
        image_size=128,
        patch_size=8,
        spatial_merge_size=2,
        temporal_patch_size=2,
        in_channels=3,
        apply_vit_abs_pos_embed=True,
        deepstack_visual_indexes=[0, 1],
        intermediate_size=256,
        hidden_act="gelu_pytorch_tanh",
        depth=2,
        out_hidden_size=64,
    )


def _engine_config():
    mm = SimpleNamespace(
        enable_mm_embeds=False,
        get_limit_per_prompt=lambda modality: 4,
        mm_encoder_tp_mode="weights",
        mm_encoder_attn_dtype=None,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(multimodal_config=mm, max_model_len=256),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=256),
        compilation_config=SimpleNamespace(
            cudagraph_mm_encoder=True,
            encoder_cudagraph_token_budgets=BUDGETS,
            encoder_cudagraph_max_vision_items_per_batch=2,
            encoder_cudagraph_max_frames_per_batch=None,
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )


@pytest.fixture(scope="module")
def _parallel_state():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import parallel_state

    if parallel_state.model_parallel_is_initialized():
        yield
        return
    previous_device = torch.accelerator.current_device_index()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    torch.accelerator.set_device_index(0)
    parallel_state.init_distributed_environment(
        world_size=1, rank=0, local_rank=0, distributed_init_method=f"tcp://127.0.0.1:{port}", backend="nccl"
    )
    with set_current_vllm_config(VllmConfig()):
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)
    try:
        yield
    finally:
        parallel_state.destroy_model_parallel()
        parallel_state.destroy_distributed_environment()
        torch.accelerator.set_device_index(previous_device)


@pytest.fixture(scope="module", params=[AttentionBackendEnum.FLASH_ATTN, AttentionBackendEnum.TRITON_ATTN])
def encoder(request, _parallel_state):
    import vllm.model_executor.models.vision as vision_module
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.config.multimodal import MultiModalConfig
    from vllm.utils.torch_utils import set_default_torch_dtype

    from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
        Qwen3Omni_VisionTransformer,
        Qwen3OmniMoeThinkerForConditionalGeneration,
    )

    override = MultiModalConfig(mm_encoder_attn_backend=request.param)
    with (
        pytest.MonkeyPatch.context() as patch,
        set_current_vllm_config(VllmConfig()),
        set_default_torch_dtype(DTYPE),
        torch.device("cuda"),
    ):
        patch.setattr(vision_module, "get_multimodal_config", lambda: override)
        visual = Qwen3Omni_VisionTransformer(vision_config=_vision_config())
    assert visual.attn_backend == request.param
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, param in visual.named_parameters():
            if "norm" in name:
                param.fill_(1.0 if name.endswith("weight") else 0.0)
            elif name.startswith("pos_embed"):
                # Comparable to the patch embedding, so a misplaced row shows.
                param.copy_(torch.randn(param.shape, generator=generator))
            elif param.dim() > 1:
                fan_in = param[0].numel()
                param.copy_(torch.randn(param.shape, generator=generator) / fan_in**0.5)
            else:
                param.copy_(torch.randn(param.shape, generator=generator) * 0.02)
    model = object.__new__(Qwen3OmniMoeThinkerForConditionalGeneration)
    nn.Module.__init__(model)
    model.visual = visual
    model.vllm_config = _engine_config()
    model.multimodal_config = model.vllm_config.model_config.multimodal_config
    model.visual_dim = 64
    model.multiscale_dim = 128
    model._enable_image_encoder_cudagraph()
    return model


def _images(grids, seed):
    generator = torch.Generator().manual_seed(seed)
    patches = sum(t * h * w for t, h, w in grids)
    pixels = torch.randn(patches, 3 * 2 * 8 * 8, generator=generator)
    # The processor keeps image_grid_thw on the CPU; pixels reach the GPU.
    return {"image_grid_thw": torch.tensor(grids), "pixel_values": pixels.to("cuda", DTYPE)}


def _forward(model, pixels, metadata):
    from vllm.config import VllmConfig
    from vllm.forward_context import set_forward_context

    # The tower on caller-supplied metadata, inside a forward context like the thinker.
    with torch.inference_mode(), set_forward_context(None, VllmConfig()):
        return model.visual.forward_with_encoder_metadata(pixels, metadata)


def _eager(model, inputs):
    from vllm.config import VllmConfig

    # The thinker's own image path, which the runner calls without graphs.
    engine_config, model.vllm_config = model.vllm_config, VllmConfig()
    try:
        with torch.inference_mode():
            return list(model.embed_multimodal(**inputs))
    finally:
        model.vllm_config = engine_config


def _assert_detectable(model, inputs, wrong_metadata):
    # A replay built from wrong metadata must fall outside the tolerance.
    right = torch.cat(_eager(model, inputs))
    wrong = _forward(model, inputs["pixel_values"], wrong_metadata)
    assert not torch.allclose(wrong, right, **_tol(right))


def test_tolerance_detects_wrong_encoder_metadata(encoder):
    visual = encoder.visual
    two = _images([[1, 8, 8], [1, 4, 8]], seed=2)
    grids = two["image_grid_thw"].tolist()
    metadata = visual.prepare_encoder_metadata(grids)
    shifted = dict(metadata, pos_embeds=metadata["pos_embeds"].roll(1, dims=0))
    _assert_detectable(encoder, two, shifted)
    transposed = visual.prepare_encoder_metadata([[1, 8, 8], [1, 8, 4]])
    rotated = dict(metadata, rotary_pos_emb_cos=transposed["rotary_pos_emb_cos"])
    rotated["rotary_pos_emb_sin"] = transposed["rotary_pos_emb_sin"]
    _assert_detectable(encoder, two, rotated)
    total = int(metadata["cu_seqlens"][-1])
    merged = dict(metadata, cu_seqlens=metadata["cu_seqlens"][[0, -1]], max_seqlen=torch.tensor(total))
    merged.pop("sequence_lengths", None)
    _assert_detectable(encoder, two, merged)


def test_replay_metadata_does_not_read_the_device(encoder):
    # Replay buffers are built from the host grid every step; a device read
    # here would stall the CPU behind the queued GPU work.
    torch.accelerator.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        encoder.visual.prepare_encoder_metadata([[1, 8, 8], [1, 4, 8]])
    finally:
        torch.cuda.set_sync_debug_mode("default")


def test_captured_graphs_match_eager_and_keep_previous_outputs(encoder):
    from vllm.compilation.monitor import set_cudagraph_capturing_enabled
    from vllm.distributed.parallel_state import graph_capture
    from vllm.model_executor.models.interfaces import supports_encoder_cudagraph

    from vllm_omni.worker.encoder_cudagraph import SingleReplayEncoderCudaGraphManager

    assert supports_encoder_cudagraph(encoder)
    manager = SingleReplayEncoderCudaGraphManager(encoder.vllm_config, torch.device("cuda"), DTYPE, encoder)
    try:
        # The runner captures inside graph_capture(), which supplies the
        # non-default capture stream.
        with torch.inference_mode(), graph_capture(device=torch.device("cuda")):
            set_cudagraph_capturing_enabled(True)
            try:
                manager.capture(graph_pool=torch.cuda.graph_pool_handle())
            finally:
                set_cudagraph_capturing_enabled(False)
        assert manager.get_cumulative_stats()["num_budgets"] == len(BUDGETS)
        # A small image; two images that exactly fill the smallest budget; one
        # 256-patch sequence, longer than one attention tile, filling a budget
        # alone; two images the manager packs into one replay in reverse order.
        cases = [[[1, 4, 4]], [[1, 2, 8], [1, 4, 4]], [[1, 16, 16]], [[1, 8, 8], [1, 4, 8]]]
        retained = []
        for seed, grids in enumerate(cases):
            inputs = _images(grids, seed)
            expected = _eager(encoder, inputs)
            with torch.inference_mode():
                actual = manager.execute(inputs)
            assert [tuple(x.shape) for x in actual] == [tuple(x.shape) for x in expected]
            for output, reference in zip(actual, expected):
                torch.testing.assert_close(output, reference, **_tol(reference))
                retained.append((output, output.clone()))
        stats = manager.get_cumulative_stats()
        # Hits and misses count images, not replays.
        assert stats["graph_hits"] == 6 and stats["graph_misses"] == 0, stats
        for output, saved in retained:
            torch.testing.assert_close(output, saved, rtol=0, atol=0)

        # Larger than every budget, or more than one replay holds: the group
        # goes back to the runner's single eager call.
        with torch.inference_mode():
            assert manager.execute(_images([[1, 16, 32]], seed=9)) is None
            assert manager.execute(_images([[1, 16, 16], [1, 4, 4]], seed=10)) is None
        assert manager.get_cumulative_stats()["graph_misses"] == 3
    finally:
        manager.clear()
