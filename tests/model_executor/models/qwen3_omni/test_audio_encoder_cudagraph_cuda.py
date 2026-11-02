# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Captured Qwen3-Omni audio encoder graphs replay the eager tower on CUDA."""

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
MEL = 32
# 100-frame chunks of 13 tokens; n_window_infer=400 gives 52-token attention windows,
# which also cap the audio budgets.
BUDGETS = [13, 52]


def _tol(reference):
    return {"rtol": 1.6e-2, "atol": 1.6e-2 * reference.abs().max().item()}


def _audio_config():
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig

    return Qwen3OmniMoeAudioEncoderConfig(
        num_mel_bins=MEL,
        encoder_layers=2,
        encoder_attention_heads=2,
        encoder_ffn_dim=128,
        d_model=64,
        output_dim=48,
        downsample_hidden_size=16,
        n_window=50,
        n_window_infer=400,
        conv_chunksize=500,
        max_source_positions=1500,
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
        Qwen3OmniMoeAudioEncoder,
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
        tower = Qwen3OmniMoeAudioEncoder(_audio_config())
    assert tower.attn_backend == request.param
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, param in tower.named_parameters():
            if "norm" in name or "ln_post" in name:
                param.fill_(1.0 if name.endswith("weight") else 0.0)
            elif param.dim() > 1:
                fan_in = param[0].numel()
                param.copy_(torch.randn(param.shape, generator=generator) / fan_in**0.5)
            else:
                param.copy_(torch.randn(param.shape, generator=generator) * 0.02)
    model = object.__new__(Qwen3OmniMoeThinkerForConditionalGeneration)
    nn.Module.__init__(model)
    model.audio_tower = tower
    model.vllm_config = _engine_config()
    model.multimodal_config = model.vllm_config.model_config.multimodal_config
    model._enable_audio_encoder_cudagraph(model.vllm_config)
    return model


def _audio(lengths, seed):
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(MEL, sum(lengths), generator=generator)
    # The processor keeps audio_feature_lengths on the CPU; features reach the GPU.
    return {
        "input_audio_features": features.to("cuda", DTYPE),
        "audio_feature_lengths": torch.tensor(lengths),
        "feature_attention_mask": [torch.ones(n, device="cuda") for n in lengths],
    }


def _eager(model, inputs):
    from vllm.config import VllmConfig

    # The thinker's own audio path, which the runner calls without graphs.
    engine_config, model.vllm_config = model.vllm_config, VllmConfig()
    try:
        with torch.inference_mode():
            return list(model.embed_multimodal(**inputs))
    finally:
        model.vllm_config = engine_config


def _forward(model, features, metadata):
    with torch.inference_mode():
        return model.audio_tower.forward_with_encoder_metadata(features, metadata)


def test_tolerance_detects_wrong_encoder_metadata(encoder):
    tower = encoder.audio_tower
    inputs = _audio([250, 150], seed=1)
    right = torch.cat(_eager(encoder, inputs))
    metadata = tower.prepare_encoder_metadata([250, 150])
    shifted = dict(metadata, valid_index=metadata["valid_index"].roll(1))
    total = int(metadata["cu_seqlens"][-1])
    merged = dict(metadata, cu_seqlens=metadata["cu_seqlens"][[0, -1]], max_seqlen=torch.tensor(total))
    for wrong in (shifted, merged):
        actual = _forward(encoder, inputs["input_audio_features"], wrong)
        assert not torch.allclose(actual, right, **_tol(right))


def test_replay_metadata_does_not_read_the_device(encoder):
    # Replay buffers are built from host lengths every step; a device read
    # here would stall the CPU behind the queued GPU work.
    inputs = _audio([250, 150], seed=0)
    torch.accelerator.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        encoder.audio_tower.prepare_encoder_metadata([250, 150])
        encoder.get_encoder_cudagraph_item_specs(inputs)
        encoder.prepare_encoder_cudagraph_replay_buffers(inputs, 2, 0, "audio")
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
        with torch.inference_mode(), graph_capture(device=torch.device("cuda")):
            set_cudagraph_capturing_enabled(True)
            try:
                manager.capture(graph_pool=torch.cuda.graph_pool_handle())
            finally:
                set_cudagraph_capturing_enabled(False)
        # Explicit budgets set the audio budgets (whole 13-token chunks).
        assert manager.path_token_budgets == {"audio": [13, 52]}
        # One full chunk filling the smallest budget; a tail chunk padded to a
        # larger budget; two clips in one replay; four chunks filling a budget.
        cases = [[100], [250], [100, 150], [400]]
        exact = {0, 3}
        retained = []
        for seed, lengths in enumerate(cases):
            inputs = _audio(lengths, seed)
            expected = _eager(encoder, inputs)
            with torch.inference_mode():
                actual = manager.execute(inputs)
            assert [tuple(x.shape) for x in actual] == [tuple(x.shape) for x in expected]
            for output, reference in zip(actual, expected):
                # Whole chunks that exactly fill a budget give the eager result bit for bit.
                tol = {"rtol": 0, "atol": 0} if seed in exact else _tol(reference)
                torch.testing.assert_close(output, reference, **tol)
                retained.append((output, output.clone()))
        stats = manager.get_cumulative_stats()
        assert stats["graph_hits"] == 5 and stats["graph_misses"] == 0, stats
        for output, saved in retained:
            torch.testing.assert_close(output, saved, rtol=0, atol=0)

        # More chunks than the largest budget, more clips than one replay
        # holds, or only sub-chunk clips: the group goes back to the runner.
        with torch.inference_mode():
            assert manager.execute(_audio([700], seed=9)) is None
            assert manager.execute(_audio([100, 100, 100], seed=10)) is None
            assert manager.execute(_audio([50, 80], seed=11)) is None
        assert manager.get_cumulative_stats()["graph_misses"] == 6
    finally:
        manager.clear()
