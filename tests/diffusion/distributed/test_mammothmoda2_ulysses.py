# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Two-GPU FP32/math control for the MammothModa2 model SP boundary.

Uses real NCCL, shared attention and model hooks, with released 21/7/120 head
geometry and reduced depth. A separate tiny native pipeline replay checks the
runtime/CFG/scheduler/noise contract. No checkpoint, serving or speedup is tested.
"""

import json
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.nn.attention import SDPBackend, sdpa_kernel
from vllm.utils.network_utils import get_open_port

from tests.diffusion.models.mammoth_moda2.test_pipeline_sp import _config as _runtime_config
from tests.diffusion.models.mammoth_moda2.test_pipeline_sp import _request as _runtime_request
from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelConfig
from vllm_omni.diffusion.forward_context import get_forward_context, set_forward_context
from vllm_omni.diffusion.hooks.sequence_parallel import apply_sequence_parallel
from vllm_omni.diffusion.models.mammoth_moda2.mammothmoda2_dit_model import Transformer2DModel
from vllm_omni.diffusion.models.mammoth_moda2.pipeline_mammothmoda2_dit import MammothModa2DiTPipeline
from vllm_omni.diffusion.registry import _apply_sequence_parallel_if_enabled
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.parallel]


def _config(degree):
    return OmniDiffusionConfig(
        dtype=torch.float32,
        parallel_config=DiffusionParallelConfig(ulysses_degree=degree, ulysses_mode="advanced_uaa"),
        diffusion_attention_config={"default": {"backend": "TORCH_SDPA"}},
    )


def _model(config, device):
    torch.manual_seed(42)
    with set_current_diffusion_config(config), set_forward_context(omni_diffusion_config=config):
        return (
            Transformer2DModel(
                hidden_size=2520,
                num_attention_heads=21,
                num_kv_heads=7,
                axes_dim_rope=(40, 40, 40),
                axes_lens=(64, 64, 64),
                num_layers=2,
                num_refiner_layers=1,
                in_channels=16,
                text_feat_dim=32,
            )
            .to(device=device, dtype=torch.float32)
            .eval()
        )


def _runtime_replay(rank, device):
    baseline_config, sp_config = _runtime_config(1), _runtime_config(2)
    with set_current_diffusion_config(baseline_config), set_forward_context(omni_diffusion_config=baseline_config):
        baseline = MammothModa2DiTPipeline(od_config=baseline_config).to(device).eval()
    with set_current_diffusion_config(sp_config), set_forward_context(omni_diffusion_config=sp_config):
        pipeline = MammothModa2DiTPipeline(od_config=sp_config).to(device).eval()
        _apply_sequence_parallel_if_enabled(pipeline, sp_config)
    pipeline.load_state_dict(baseline.state_dict(), strict=True)
    records = []
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        for seed in (42, None):
            requests = {length: _runtime_request(length, 4.0, seed=seed) for length in (3, 2)}
            # The request owner materializes omitted seeds once, before the
            # request is sent to workers. Reproduce that same-request contract,
            # including A/B/A replay of the originally resolved default seed.
            resolved_seeds = torch.tensor(
                [request.sampling_params.seed for request in requests.values()], device=device, dtype=torch.int64
            )
            dist.broadcast(resolved_seeds, src=0)
            for request, resolved_seed in zip(requests.values(), resolved_seeds.tolist()):
                request.sampling_params.seed = resolved_seed
            first = None
            for text_len in (3, 2, 3):
                request = requests[text_len]
                torch.manual_seed(500 + text_len)
                with set_forward_context(omni_diffusion_config=baseline_config):
                    expected = baseline(request).output
                # Request-level seeds must dominate different rank-local RNG
                # states, including when the original caller omitted a seed.
                torch.manual_seed(500 + text_len + rank)
                with set_forward_context(omni_diffusion_config=sp_config):
                    actual = pipeline(request).output
                    ctx = get_forward_context()
                    assert (ctx.sp_original_seq_len, ctx.sp_padding_size, ctx._sp_shard_depth) == (None, 0, 0)
                assert actual.shape == expected.shape == (1, 3, 32, 48)
                assert torch.isfinite(actual).all()
                torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
                if text_len == 3:
                    if first is None:
                        first = actual
                    else:
                        torch.testing.assert_close(actual, first, atol=0, rtol=0)
                records.append(
                    {
                        "text_len": text_len,
                        "seed": seed,
                        "resolved_seed": request.sampling_params.seed,
                        "max_abs_error": (actual - expected).abs().max().item(),
                    }
                )
    return records


def _worker(rank, port, output_dir):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda", rank)
    current_omni_platform.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    try:
        init_distributed_environment(world_size=2, rank=rank, local_rank=rank, backend="nccl")
        initialize_model_parallel(sequence_parallel_size=2, ulysses_degree=2, backend="nccl")
        baseline_config, sp_config = _config(1), _config(2)
        baseline = _model(baseline_config, device)
        model = _model(sp_config, device)
        model.load_state_dict(baseline.state_dict(), strict=True)
        apply_sequence_parallel(model, SequenceParallelConfig(ulysses_degree=2), model._sp_plan)
        rotary = model.rope_embedder.get_freqs_real((40, 40, 40), (64, 64, 64), 10000)
        records = []

        # The runner creates a new context per request, whereas CFG reuses it.
        for hooks_applied in (False, True):
            with set_forward_context(omni_diffusion_config=sp_config), torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
                ctx = get_forward_context()
                ctx.sp_plan_hooks_applied = hooks_applied
                first_conditional = None
                for text_len in (3, 0, 3):
                    generator = torch.Generator().manual_seed(100 + text_len)
                    latent = torch.randn(1, 16, 16, 16, generator=generator).transpose(2, 3).to(device)
                    text = torch.randn(1, 32, text_len, generator=generator).transpose(1, 2).to(device)
                    mask = torch.ones(1, text_len, dtype=torch.bool, device=device)
                    args = (latent, torch.tensor([0.5], device=device), text, rotary, mask)
                    with set_forward_context(omni_diffusion_config=baseline_config):
                        expected = baseline(*args)
                        repeated = baseline(*args)
                    torch.testing.assert_close(repeated, expected, atol=0, rtol=0)

                    global_seq = text_len + 64
                    local_seq = (global_seq + 1) // 2
                    seen_main, seen_refiners, seen_attention = [], [], []

                    def main_hook(module, inputs):
                        assert inputs[0].shape == (1, local_seq, 2520)
                        assert inputs[1].shape == (1, local_seq * 2)
                        assert inputs[4].shape == (1, local_seq)
                        assert ctx._sp_shard_depth == 1
                        assert ctx.sp_padding_size == global_seq % 2
                        assert module.attn.omni_attn._get_active_parallel_strategy().name == "ulysses"
                        seen_main.append(True)

                    def refiner_hook(module, inputs):
                        assert inputs[0].shape[1] in (text_len, 64)
                        assert module.attn.omni_attn._get_active_parallel_strategy().name == "none"
                        seen_refiners.append(True)

                    def attention_hook(module, inputs, output):
                        assert inputs[0].shape == output.shape == (1, local_seq, 21, 120)
                        assert inputs[1].shape == inputs[2].shape == (1, local_seq, 7, 120)
                        seen_attention.append(True)

                    handles = [layer.register_forward_pre_hook(main_hook) for layer in model.layers]
                    handles += [layer.attn.omni_attn.register_forward_hook(attention_hook) for layer in model.layers]
                    handles += [
                        layer.register_forward_pre_hook(refiner_hook)
                        for layer in (*model.context_refiner, *model.noise_refiner)
                    ]
                    try:
                        actual = model(*args)
                    finally:
                        for handle in handles:
                            handle.remove()
                    assert len(seen_main) == len(seen_attention) == 2
                    assert len(seen_refiners) == 2
                    assert ctx._sp_shard_depth == 0
                    assert ctx.sp_padding_size == 0 and ctx.sp_original_seq_len is None
                    assert actual.shape == expected.shape and torch.isfinite(actual).all()
                    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
                    if text_len == 3:
                        if first_conditional is None:
                            first_conditional = actual.clone()
                        else:
                            torch.testing.assert_close(actual, first_conditional, atol=0, rtol=0)
                    records.append(
                        {
                            "text_len": text_len,
                            "hooks_applied_flag": hooks_applied,
                            "local_seq_len": local_seq,
                            "original_seq_len": global_seq,
                            "sp_padding_size": global_seq % 2,
                            "max_abs_error": (actual - expected).abs().max().item(),
                        }
                    )
        result = {
            "rank": rank,
            "gpu_uuid": str(torch.cuda.get_device_properties(rank).uuid),
            "backend": dist.get_backend(),
            "dtype": "float32",
            "scope": "reduced-depth real-head-geometry model; not full-checkpoint E2E",
            "cases": records,
            "runtime_cases": _runtime_replay(rank, device),
        }
        Path(output_dir, f"rank-{rank}.json").write_text(json.dumps(result, indent=2))
    finally:
        destroy_distributed_env()


@hardware_test(res={"cuda": "L4"}, num_cards=2)
def test_mammothmoda2_two_rank_ulysses_fp32(tmp_path):
    if not torch.cuda.is_available() or torch.accelerator.device_count() < 2:
        pytest.skip("requires two distinct CUDA devices")
    torch.multiprocessing.spawn(_worker, args=(get_open_port(), str(tmp_path)), nprocs=2, join=True)
    results = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
    assert {result["rank"] for result in results} == {0, 1}
    assert len({result["gpu_uuid"] for result in results}) == 2
    assert all(result["backend"] == "nccl" and len(result["cases"]) == 6 for result in results)
    assert all(len(result["runtime_cases"]) == 6 for result in results)
