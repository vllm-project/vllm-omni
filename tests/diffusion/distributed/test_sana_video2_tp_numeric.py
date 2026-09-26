# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SANA-Video 2.0 dense/TP numerical checks with real process groups."""

import os
import socket
import time

import pytest
import torch

from tests.diffusion.distributed.test_sana_video2_sp_numeric import _CONFIG, _inputs


def _model(device):
    from vllm_omni.diffusion.models.sana_video2.components import RMSNorm
    from vllm_omni.diffusion.models.sana_video2.transformer_sana_video2 import (
        SanaVideo2TransformerConfig,
        SanaVideo2TransformerModel,
    )

    torch.manual_seed(8006)
    model = SanaVideo2TransformerModel(SanaVideo2TransformerConfig(**_CONFIG)).to(device).eval()
    for module in model.modules():
        if isinstance(module, RMSNorm):
            torch.nn.init.uniform_(module.weight, 0.5, 1.5)
    for projection in (model.attn_res.attn_proj, model.attn_res.mlp_proj, model.attn_res.final_proj):
        torch.nn.init.normal_(projection.weight, std=0.1)
    return model


def _worker(rank, tp, sp, port, directory):
    from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
    from vllm_omni.diffusion.distributed.parallel_state import (
        destroy_distributed_env,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm_omni.diffusion.forward_context import set_forward_context

    torch.set_num_threads(1)
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(tp * sp),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
    )
    init_distributed_environment(local_rank=rank, backend="gloo")
    try:
        initialize_model_parallel(
            tensor_parallel_size=tp, sequence_parallel_size=sp, ulysses_degree=sp, ring_degree=1, backend="gloo"
        )
        with torch.device("meta"):
            model = _model(torch.device("meta"))
        model.materialize(device=torch.device("cpu"), dtype=torch.float32)
        weights = torch.load(f"{directory}/weights.pt", map_location="cpu", weights_only=True)
        report = model.load_weights(weights.items())
        assert len(report.loaded_keys) == len(weights)
        assert model.blocks[0].attn.qkv.weight.shape == (180, 120)
        assert model.blocks[3].attn.heads == 5
        tp_rank = rank % tp
        qkv = weights["blocks.0.attn.qkv.weight"]
        expected_qkv = torch.cat(
            [qkv[part * 120 + tp_rank * 60 : part * 120 + (tp_rank + 1) * 60] for part in range(3)]
        )
        torch.testing.assert_close(model.blocks[0].attn.qkv.weight, expected_qkv, rtol=0, atol=0)
        q_norm = weights["blocks.0.attn.q_norm.weight"]
        torch.testing.assert_close(
            model.blocks[0].attn.q_norm.weight, q_norm[tp_rank * 60 : (tp_rank + 1) * 60], rtol=0, atol=0
        )
        kv = weights["blocks.0.cross_attn.kv_linear.weight"]
        expected_kv = torch.cat([kv[part * 120 + tp_rank * 60 : part * 120 + (tp_rank + 1) * 60] for part in range(2)])
        torch.testing.assert_close(model.blocks[0].cross_attn.kv_linear.weight, expected_kv, rtol=0, atol=0)
        torch.testing.assert_close(
            model.blocks[0].attn.beta_proj.weight,
            weights["blocks.0.attn.beta_proj.weight"][tp_rank * 10 : (tp_rank + 1) * 10],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            model.blocks[0].attn.output_gate.bias,
            weights["blocks.0.attn.output_gate.bias"][tp_rank * 60 : (tp_rank + 1) * 60],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            model.blocks[0].cross_attn.q_linear.weight,
            weights["blocks.0.cross_attn.q_linear.weight"][tp_rank * 60 : (tp_rank + 1) * 60],
            rtol=0,
            atol=0,
        )
        down = weights["blocks.0.mlp.down_proj.weight"]
        torch.testing.assert_close(
            model.blocks[0].mlp.down_proj.weight, down[:, tp_rank * 60 : (tp_rank + 1) * 60], rtol=0, atol=0
        )
        torch.testing.assert_close(
            model.blocks[0].mlp.down_proj.bias, weights["blocks.0.mlp.down_proj.bias"], rtol=0, atol=0
        )
        with torch.device("meta"):
            bf16_model = _model(torch.device("meta"))
        bf16_model.materialize(device=torch.device("cpu"), dtype=torch.bfloat16)
        bf16_model.load_weights(weights.items())
        torch.testing.assert_close(
            bf16_model.blocks[0].attn.qkv.weight, expected_qkv.to(torch.bfloat16), rtol=0, atol=0
        )
        torch.testing.assert_close(
            bf16_model.blocks[0].attn.q_norm.weight,
            q_norm[tp_rank * 60 : (tp_rank + 1) * 60].to(torch.bfloat16),
            rtol=0,
            atol=0,
        )
        malformed = dict(weights)
        malformed["blocks.0.attn.q_norm.weight"] = q_norm[:60]
        with pytest.raises(ValueError, match="shape mismatch"):
            model.load_weights(malformed.items())
        with pytest.raises(ValueError, match="duplicate key"):
            model.load_weights([*weights.items(), next(iter(weights.items()))])
        with pytest.raises(ValueError, match="missing keys"):
            model.load_weights(list(weights.items())[1:])
        with pytest.raises(ValueError, match="unexpected key"):
            model.load_weights([*weights.items(), ("unexpected.weight", torch.ones(1))])
        if sp > 1:
            from vllm_omni.diffusion.distributed.parallel_state import get_sp_group

            model.set_sequence_parallel(get_sp_group())
        config = OmniDiffusionConfig(
            model=directory,
            dtype=torch.float32,
            parallel_config=DiffusionParallelConfig(
                tensor_parallel_size=tp,
                sequence_parallel_size=sp,
                ulysses_degree=sp,
                ring_degree=1,
                ulysses_mode="advanced_uaa" if sp > 1 else "strict",
            ),
        )
        outputs = {}
        with torch.no_grad(), set_forward_context(omni_diffusion_config=config):
            for task in ("t2v", "ti2v"):
                tensors = _inputs(3, 1, 3, task, 8765, torch.device("cpu"))
                outputs[task] = model(*tensors).cpu()
        torch.save(outputs, f"{directory}/rank_{rank}.pt")
    finally:
        destroy_distributed_env()


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.parallel
@pytest.mark.cpu
@pytest.mark.parametrize("tp,sp", [(2, 1), (2, 2)])
def test_tp_full_depth_shards_and_reports_dense_drift(tp, sp, tmp_path):
    import torch.multiprocessing as mp

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    model = _model(torch.device("cpu"))
    expected = {}
    with torch.no_grad():
        for task in ("t2v", "ti2v"):
            expected[task] = model(*_inputs(3, 1, 3, task, 8765, torch.device("cpu"))).cpu()
    torch.save(model.state_dict(), tmp_path / "weights.pt")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    context = mp.spawn(_worker, args=(tp, sp, port, str(tmp_path)), nprocs=tp * sp, join=False)
    try:
        deadline = time.monotonic() + 180
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                raise TimeoutError(f"TP{tp}+SP{sp} workers exceeded 180 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    first = None
    for rank in range(tp * sp):
        observed = torch.load(tmp_path / f"rank_{rank}.pt", map_location="cpu", weights_only=True)
        if first is None:
            first = observed
        for task, target in expected.items():
            delta = (observed[task].double() - target.double()).flatten()
            absolute = delta.abs().max().item()
            relative = (delta.norm() / target.double().flatten().norm().clamp_min(1e-12)).item()
            if rank == 0:
                print(f"TP{tp} SP{sp} {task} max_abs={absolute:.3e} rel_l2={relative:.3e}")
            torch.testing.assert_close(observed[task], first[task], rtol=0, atol=0)
