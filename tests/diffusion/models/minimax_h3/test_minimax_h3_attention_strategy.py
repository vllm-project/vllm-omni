# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Full-forward capture, compilation and replay with CPU or real CUDA providers."""

from types import SimpleNamespace

import pytest
import torch

from tests.helpers.attention_strategy import (
    configure_strategy_test,
    move_strategy_inputs,
)
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.forward_context import ForwardContext, override_forward_context

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]


STRATEGY_EXECUTION_MODES = [
    pytest.param("cpu", "eager", "none", marks=pytest.mark.cpu),
    pytest.param("cpu", "inductor", "none", marks=pytest.mark.cpu),
    pytest.param("cuda", "inductor", "none", marks=pytest.mark.cuda),
    pytest.param("cuda", "inductor", "component", marks=pytest.mark.cuda),
    pytest.param("cuda", "inductor", "layerwise", marks=pytest.mark.cuda),
]


def install_strategy_layerwise_offload(blocks, request):
    from vllm_omni.diffusion.offloader.layerwise_backend import _install_layerwise_hook_group, remove_block_hook

    blocks = list(blocks)
    hooks = _install_layerwise_hook_group(blocks, torch.device("cuda:0"), torch.Stream(device="cuda"), True)
    hooks[0].prefetch_layer(non_blocking=False)

    def restore_blocks():
        for hook in hooks:
            hook.restore_next_block()
        for block in blocks:
            remove_block_hook(block)

    request.addfinalizer(restore_blocks)


def check_strategy_capture_and_reuse(
    model, inputs, device, compiler_backend, offload_mode, steps, tolerances, offload_component=None
):
    references = {}
    for step in steps:
        with override_forward_context(ForwardContext(denoise_step_idx=step, total_denoise_steps=len(steps))):
            references[step] = model(**inputs)

    graphs = []

    def backend(graph, example_inputs):
        compiled = torch._inductor.compile(graph, example_inputs) if compiler_backend == "inductor" else graph.forward
        # Inductor may request that Dynamo restart tracing with specialized
        # dimensions. Count completed compilations, not abandoned attempts.
        graphs.append(graph)
        return compiled

    torch._dynamo.reset()
    model._attention_strategy_runner.compile(
        backend=backend, dynamic=True, granularity="full" if offload_mode == "none" else "regional"
    )
    runner = model._attention_strategy_runner

    runner.set_warmup(True)
    try:
        with override_forward_context(ForwardContext(denoise_step_idx=0, total_denoise_steps=2)):
            torch.testing.assert_close(model(**inputs), references[0], **tolerances)
    finally:
        runner.set_warmup(False)
    assert not runner.warmup_status()["layouts_pending"]
    for step in steps:
        with override_forward_context(ForwardContext(denoise_step_idx=step, total_denoise_steps=len(steps))):
            torch.testing.assert_close(model(**inputs), references[step], **tolerances)
    compiled_count = len(graphs)
    if offload_mode == "none":
        assert compiled_count == len(steps)
    else:
        assert compiled_count > 0
    if offload_component is not None:
        offload_component()
    for step in steps:
        with override_forward_context(ForwardContext(denoise_step_idx=step, total_denoise_steps=len(steps))):
            torch.testing.assert_close(model(**inputs), references[step], **tolerances)
    assert len(graphs) == compiled_count
    if device == "cuda":
        sparse_calls = [
            sum("vllm_omni.block_sparse_request" in str(node.target) for node in graph.graph.nodes) for graph in graphs
        ]
        if offload_mode == "none":
            assert sparse_calls == [0, 1, 2]
        else:
            assert sum(sparse_calls) > 0  # Sparse attention remains compiled between offload boundaries.
    return graphs, compiled_count


@pytest.mark.parametrize(("device", "compiler_backend", "offload_mode"), STRATEGY_EXECUTION_MODES)
@torch.inference_mode()
def test_minimax_forward_capture_and_reuse(monkeypatch, request, device, compiler_backend, offload_mode):
    from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as h3

    common, steps = configure_strategy_test(monkeypatch, device, "minimax_h3.dit")
    monkeypatch.setattr(h3, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr("vllm.distributed.get_tensor_model_parallel_world_size", lambda: 1)
    with set_current_diffusion_config(SimpleNamespace(**common)):
        cfg = SimpleNamespace(
            **common,
            tf_model_config={
                "num_layers": 2,
                "token_refiner_num_layers": 1,
                "hidden_size": 32,
                "num_attention_heads": 2,
                "attention_head_dim": 16,
                "ffn_hidden_size": 64,
                "latents_dim": 2,
                "audio_latents_dim": 2,
                "patch_size": (1, 1, 1),
                "text_dim": 32,
                "timestep_input_dim": 8,
                "time_embed_hidden_size": 16,
                "time_embed_dim": 16,
                "adaln_out_features": 18 * 32,
                "final_adaln_out_features": 2 * 32,
                "rope_inv_freq_len": 2,
            },
        )
        if device == "cuda":
            cfg.tf_model_config.update(
                hidden_size=256,
                attention_head_dim=128,
                ffn_hidden_size=512,
                adaln_out_features=18 * 256,
                final_adaln_out_features=2 * 256,
                rope_inv_freq_len=16,
            )
        # Exact AdaLN caching deliberately introduces eager boundaries. Keep
        # the single-graph gate cache-free; regional offload cases cover caching.
        cfg.cache_config = {"minimax_h3_adaln_cache": offload_mode != "none"}
        model = h3.MiniMaxH3DiTModel(cfg)
        inputs = dict(
            x=torch.randn(1, 8, 2),
            audio_x=torch.randn(1, 8, 2),
            img_position_ids=torch.zeros(1, 8, 3),
            unique_timesteps=torch.ones(1),
            inverse_indices=torch.zeros(8, dtype=torch.long),
            update_mask=torch.ones(3),
            token_tags=torch.tensor([0, 0, 0, 1, 1, 1, 2, 2]),
            prompt_embeds=torch.randn(3, 32),
            img_pos_info={"position_ids": torch.tensor([3, 4, 5])},
            audio_pos_info={"position_ids": torch.tensor([6, 7])},
            text_pos_info={"position_ids": torch.tensor([0, 1, 2])},
            img_pos_for_infer_output_info={"position_ids": torch.tensor([3, 4, 5])},
            packed_seq_params={"cu_seqlens_q": torch.tensor([0, 8], dtype=torch.int32), "max_seqlen_q": 8},
            refiner_packed_seq_params={"cu_seqlens_q": torch.tensor([0, 3], dtype=torch.int32), "max_seqlen_q": 3},
        )
    if device == "cuda":
        inputs.update(
            update_mask=torch.ones(128),
            x=torch.randn(1, 256, 2),
            audio_x=torch.randn(1, 256, 2),
            img_position_ids=torch.zeros(1, 256, 3),
            inverse_indices=torch.zeros(256, dtype=torch.long),
            token_tags=torch.tensor([0] * 64 + [1] * 128 + [2] * 64),
            prompt_embeds=torch.randn(64, 32),
            img_pos_info={"position_ids": torch.arange(64, 192)},
            audio_pos_info={"position_ids": torch.arange(192, 256)},
            text_pos_info={"position_ids": torch.arange(64)},
            img_pos_for_infer_output_info={"position_ids": torch.arange(64, 192)},
            packed_seq_params={"cu_seqlens_q": torch.tensor([0, 256], dtype=torch.int32), "max_seqlen_q": 256},
            refiner_packed_seq_params={"cu_seqlens_q": torch.tensor([0, 64], dtype=torch.int32), "max_seqlen_q": 64},
        )

    inputs = move_strategy_inputs(inputs, device, common["dtype"])
    model.to(device=device)
    freq_dim = model.arch.rope_inv_freq_len
    model.rope.inv_freq.copy_(model._rope_theta ** (-torch.arange(freq_dim, device=device).float() / freq_dim))
    model.post_load_weights()
    model.eval()
    for p in model.parameters():
        p.fill_(1) if p.ndim == 1 else p.normal_(std=0.02)
    offload_component = None
    if offload_mode == "component":
        from vllm_omni.diffusion.offloader.sequential_backend import (
            SequentialOffloadHook,
            apply_sequential_offload,
            remove_sequential_offload,
        )

        apply_sequential_offload([model], [], torch.device("cuda:0"), offload_initial_dits=True)
        request.addfinalizer(lambda: remove_sequential_offload([model]))

        def offload_component():
            SequentialOffloadHook([], torch.device("cuda:0"))._to_cpu(model)
    elif offload_mode == "layerwise":
        install_strategy_layerwise_offload(model.blocks, request)
    # MiniMax uses BF16 internally even though final projections return FP32.
    tolerances = {"atol": 0.005, "rtol": 0.005}
    graphs, _ = check_strategy_capture_and_reuse(
        model, inputs, device, compiler_backend, offload_mode, steps, tolerances, offload_component
    )
    if device == "cuda":
        # A second packed geometry exercises symbolic shape bindings across
        # offload resumptions. Host packing metadata may add specializations;
        # once warmed, replay must not compile again.
        resized = move_strategy_inputs(
            {
                **inputs,
                "x": torch.randn(1, 320, 2),
                "audio_x": torch.randn(1, 320, 2),
                "img_position_ids": torch.zeros(1, 320, 3),
                "inverse_indices": torch.zeros(320, dtype=torch.long),
                "token_tags": torch.tensor([0] * 64 + [1] * 192 + [2] * 64),
                "update_mask": torch.ones(192),
                "img_pos_info": {"position_ids": torch.arange(64, 256)},
                "audio_pos_info": {"position_ids": torch.arange(256, 320)},
                "img_pos_for_infer_output_info": {"position_ids": torch.arange(64, 256)},
                "packed_seq_params": {"cu_seqlens_q": torch.tensor([0, 320], dtype=torch.int32), "max_seqlen_q": 320},
            },
            device,
            common["dtype"],
        )
        resized_references = {}
        for step in steps:
            resized_references[step] = model._attention_strategy_runner._eager_forwards[step](**resized)
            with override_forward_context(ForwardContext(denoise_step_idx=step, total_denoise_steps=len(steps))):
                torch.testing.assert_close(model(**resized), resized_references[step], **tolerances)
        warmed_count = len(graphs)
        for step in steps:
            with override_forward_context(ForwardContext(denoise_step_idx=step, total_denoise_steps=len(steps))):
                torch.testing.assert_close(model(**resized), resized_references[step], **tolerances)
        assert len(graphs) == warmed_count
    torch._dynamo.reset()
