# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Two-GPU native tiny-pipeline DLO lifecycle, not checkpoint qualification."""

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.nn.attention import SDPBackend, sdpa_kernel
from vllm.utils.network_utils import get_open_port

from tests.diffusion.models.mammoth_moda2.test_dit_offload import _dlo_config
from tests.diffusion.models.mammoth_moda2.test_pipeline_sp import _config, _request
from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.forward_context import get_forward_context, set_forward_context
from vllm_omni.diffusion.models.mammoth_moda2.pipeline_mammothmoda2_dit import MammothModa2DiTPipeline
from vllm_omni.diffusion.offloader.base import OffloadConfig
from vllm_omni.diffusion.offloader.distributed_layerwise_backend import DistributedLayerwiseOffloadBackend
from vllm_omni.diffusion.registry import _apply_sequence_parallel_if_enabled
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.parallel]


def _new_pipeline(config, dtype):
    torch.manual_seed(42)
    with set_current_diffusion_config(config), set_forward_context(omni_diffusion_config=config):
        pipeline = MammothModa2DiTPipeline(od_config=config).to(dtype=dtype).eval()
        _apply_sequence_parallel_if_enabled(pipeline, config)
    return pipeline


def _run(pipeline, config, request):
    with set_forward_context(omni_diffusion_config=config):
        result = pipeline(request).output
        context = get_forward_context()
        assert context._sp_shard_depth == 0
        assert context._sp_equal_pad_stack == []
        return result


def _worker(rank, port, output_dir, allgather, dtype, real_geometry=False, recover_after_failure=False):
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
    backend = None
    try:
        init_distributed_environment(world_size=2, rank=rank, local_rank=rank, backend="nccl")
        initialize_model_parallel(sequence_parallel_size=2, ulysses_degree=2, backend="nccl")
        baseline_config = replace(_config(2), dtype=dtype)
        config = replace(_dlo_config(2, allgather=allgather), dtype=dtype)
        if recover_after_failure:
            # An intermediate failure needs a tail that has not yet reloaded
            # the first block; a two-block ring does not expose this state.
            for selected in (baseline_config, config):
                selected.tf_model_config.params["gen_dit_config"]["num_layers"] = 3
        if real_geometry:
            for selected in (baseline_config, config):
                selected.tf_model_config.params["gen_dit_config"].update(
                    hidden_size=2520,
                    num_attention_heads=21,
                    num_kv_heads=7,
                    axes_dim_rope=(40, 40, 40),
                    num_layers=3,
                    multiple_of=256,
                )
                selected.tf_model_config.params["gen_axes_dim_rope"] = (40, 40, 40)
        baseline = _new_pipeline(baseline_config, dtype).to(device)
        candidate = _new_pipeline(config, dtype)
        candidate.load_state_dict(baseline.state_dict(), strict=True)
        master = {name: tensor.detach().cpu().clone() for name, tensor in candidate.state_dict().items()}
        backend = DistributedLayerwiseOffloadBackend(OffloadConfig.from_od_config(config), device)
        records = []
        recovery_records = []
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            for cycle in range(2):
                backend.enable(candidate)
                assert backend.enabled
                assert len(backend._all_hook_groups) == 1
                assert len(backend._all_hook_groups[0]) == (3 if real_geometry or recover_after_failure else 2)
                hooks = backend._all_hook_groups[0]
                assert all(hook.dp_size == (2 if allgather else 1) for hook in hooks)
                assert all(hook.gpu_buffers is hooks[0].gpu_buffers for hook in hooks)
                assert not any(hasattr(block, "_hook_registry") for block in candidate.gen_transformer.context_refiner)
                first = None
                for text_len, guidance in ((3, 4.0), (2, 1.0), (3, 4.0)):
                    request = _request(text_len, guidance, seed=42)
                    expected = _run(baseline, baseline_config, request)
                    actual = _run(candidate, config, request)
                    # Same SP, weights and kernels: transport must not change arithmetic.
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    if text_len == 3:
                        if first is None:
                            first = actual.clone()
                        else:
                            torch.testing.assert_close(actual, first, rtol=0, atol=0)
                    records.append({"cycle": cycle, "text_len": text_len, "guidance": guidance, "max_abs_error": 0.0})
                if recover_after_failure and cycle == 1:
                    # Both ranks fail at the same point, leaving a valid NCCL
                    # group. Asymmetric collective failures are a different
                    # process-recovery contract and are not retried here.
                    def fail_intermediate(module, args):
                        raise RuntimeError("injected intermediate Mammoth DLO failure")

                    for guidance in (1.0, 4.0):
                        request = _request(3, guidance, seed=42)
                        expected = _run(baseline, baseline_config, request)
                        handle = candidate.gen_transformer.layers[1].register_forward_pre_hook(fail_intermediate)
                        try:
                            with pytest.raises(RuntimeError, match="injected intermediate Mammoth DLO"):
                                _run(candidate, config, request)
                        finally:
                            handle.remove()
                        first_hook = next(hook for hook in hooks if hook._is_group_first)
                        assert not first_hook.is_materialized
                        for retry in range(2):
                            assert backend.enabled and backend._all_hook_groups[0] is hooks
                            actual = _run(candidate, config, _request(3, guidance, seed=42))
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                            recovery_records.append({"guidance": guidance, "retry": retry, "max_abs_error": 0.0})
                elif not allgather and cycle == 1:
                    # Both ranks fail before the same block; this tests local
                    # storage cleanup, not asymmetric collective-failure recovery.
                    def fail(module, args):
                        raise RuntimeError("injected Mammoth DLO block failure")

                    handle = candidate.gen_transformer.layers[1].register_forward_pre_hook(fail)
                    try:
                        with pytest.raises(RuntimeError, match="injected Mammoth DLO"):
                            _run(candidate, config, _request(3, 4.0))
                    finally:
                        handle.remove()
                backend.disable()
                assert not backend.enabled and not backend._all_hook_groups
                for name, tensor in candidate.state_dict().items():
                    torch.testing.assert_close(tensor.cpu(), master[name], rtol=0, atol=0)
        Path(output_dir, f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "gpu_uuid": str(torch.cuda.get_device_properties(rank).uuid),
                    "backend": dist.get_backend(),
                    "allgather": allgather,
                    "dtype": str(dtype),
                    "real_head_geometry": real_geometry,
                    "scope": "tiny native pipeline; no checkpoint/performance claim",
                    "restored_state_tensors": len(master),
                    "cases": records,
                    "recovery_cases": recovery_records,
                },
                indent=2,
            )
        )
    finally:
        # Normal disable above validates restoration. Process teardown must not
        # enter restoration collectives after an asymmetric rank failure.
        if backend is not None and backend.enabled and not allgather:
            backend.disable()
        destroy_distributed_env()


@hardware_test(res={"cuda": "L4"}, num_cards=2)
@pytest.mark.parametrize("allgather", [False, True], ids=["rank_local", "allgather"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("recover_after_failure", [False, True], ids=["lifecycle", "recovery"])
def test_mammothmoda2_two_rank_dlo(tmp_path, allgather, dtype, recover_after_failure):
    if not torch.cuda.is_available() or torch.accelerator.device_count() < 2 or torch.version.hip is not None:
        pytest.skip("requires two distinct NVIDIA CUDA devices")
    torch.multiprocessing.spawn(
        _worker,
        args=(get_open_port(), str(tmp_path), allgather, dtype, False, recover_after_failure),
        nprocs=2,
        join=True,
    )
    results = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
    assert len({result["gpu_uuid"] for result in results}) == 2
    assert all(result["backend"] == "nccl" and len(result["cases"]) == 6 for result in results)
    assert all(len(result["recovery_cases"]) == (4 if recover_after_failure else 0) for result in results)


@hardware_test(res={"cuda": "L4"}, num_cards=2)
def test_mammothmoda2_dlo_real_geometry(tmp_path):
    """Released head geometry and large weight blocks, with an odd ring.

    Three main layers and a small image fixture keep this below full-checkpoint
    qualification. AllGather is enabled with the original norm expressions.
    """
    if not torch.cuda.is_available() or torch.accelerator.device_count() < 2 or torch.version.hip is not None:
        pytest.skip("requires two distinct NVIDIA CUDA devices")
    torch.multiprocessing.spawn(
        _worker, args=(get_open_port(), str(tmp_path), True, torch.bfloat16, True), nprocs=2, join=True
    )
    results = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
    assert len({result["gpu_uuid"] for result in results}) == 2
    assert all(result["real_head_geometry"] and result["allgather"] for result in results)
