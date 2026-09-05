# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import json
import os
import tempfile
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from safetensors.torch import load_file, save_file

import vllm_omni.diffusion.models.magi2.layers as layers_module
import vllm_omni.diffusion.models.magi2.mh_moe as mh_moe_module
from vllm_omni.diffusion.model_loader.host_weight_plan import (
    build_checkpoint_mmap_plan,
)
from vllm_omni.diffusion.models.magi2.configuration_magi2 import (
    Magi2MHCConfig,
    Magi2MoEConfig,
    Magi2PreviewConfig,
)
from vllm_omni.diffusion.models.magi2.modeling_magi2 import (
    Magi2PreviewTransformer,
)
from vllm_omni.diffusion.models.magi2.parallel import Magi2ParallelGroup
from vllm_omni.diffusion.models.magi2.pipeline_magi2 import Magi2Pipeline
from vllm_omni.diffusion.offloader.distributed_layerwise_backend import (
    DistributedLayerwiseOffloadBackend,
    DistributedLayerwiseOffloadHook,
)


@dataclass(frozen=True)
class _WeightSourceStub:
    model_or_path: str
    subfolder: str
    revision: str | None
    prefix: str


@dataclass(frozen=True)
class _PipelineModulesStub:
    dits: list[nn.Module]
    dit_names: list[str]


_WORLD_SIZE = 4
pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


def _tiny_config() -> Magi2PreviewConfig:
    return Magi2PreviewConfig(
        # The released checkpoint has 40 layers. Keep this tiny fixture at two
        # so it exercises the shared backend's streamable-block contract.
        num_layers=2,
        hidden_size=16,
        head_dim=8,
        num_query_groups=2,
        video_in_channels=4,
        audio_in_channels=4,
        text_in_channels=4,
        intermediate_factor=2,
        multimodal_layers=(0, 1),
        params_dtype=torch.float32,
        mhc=Magi2MHCConfig(num_streams=2),
        moe=Magi2MoEConfig(
            num_heads=2,
            num_experts=4,
            top_k=2,
            expert_intermediate_size=8,
            shared_expert_intermediate_size=8,
            modality_shared_expert_intermediate_size=8,
            layers=(0, 1),
        ),
    )


@contextmanager
def _patched_weight_groups(ep_group: Magi2ParallelGroup):
    singleton = Magi2ParallelGroup(None, world_size=1, rank=0)
    with ExitStack() as stack:
        stack.enter_context(patch.object(layers_module, "get_magi2_tp_group", return_value=singleton))
        stack.enter_context(patch.object(mh_moe_module, "get_magi2_ep_group", return_value=ep_group))
        yield


def _initialize_checkpoint(model: Magi2PreviewTransformer) -> None:
    generator = torch.Generator(device="cpu").manual_seed(211)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(
                torch.randn(
                    parameter.shape,
                    dtype=parameter.dtype,
                    generator=generator,
                )
                * 0.025
            )
        # The normal loader intentionally mirrors EMA into the runtime router
        # bias. Keep both released checkpoint entries equal so this test can
        # compare every reconstructed tensor, not only inference-visible ones.
        moe = model.block.layers[0].mlp.moe_mlp
        moe.router.expert_bias.copy_(moe.router.expert_bias_ema)


def _write_checkpoint(checkpoint_root: str) -> None:
    singleton = Magi2ParallelGroup(None, world_size=1, rank=0)
    with _patched_weight_groups(singleton):
        source = Magi2PreviewTransformer(_tiny_config())
        _initialize_checkpoint(source)

    weights = {name: tensor.detach().cpu().contiguous() for name, tensor in source.state_dict().items()}
    preview_dir = os.path.join(checkpoint_root, "preview")
    os.makedirs(preview_dir)
    weight_file = os.path.join(preview_dir, "model.safetensors")
    save_file(weights, weight_file)
    with open(os.path.join(preview_dir, "model.safetensors.index.json"), "w") as index:
        json.dump(
            {"weight_map": {name: os.path.basename(weight_file) for name in weights}},
            index,
        )


def _new_groups(
    rank_sets: tuple[tuple[int, ...], ...],
) -> tuple[tuple[tuple[int, ...], dist.ProcessGroup], ...]:
    return tuple((ranks, dist.new_group(ranks=list(ranks), backend="gloo")) for ranks in rank_sets)


def _current_group(
    rank: int,
    rank_groups: tuple[tuple[tuple[int, ...], dist.ProcessGroup], ...],
) -> tuple[tuple[int, ...], dist.ProcessGroup]:
    return next((ranks, group) for ranks, group in rank_groups if rank in ranks)


def _load_mmap_transform_and_reconstruct(
    checkpoint_root: str,
    dp_group: dist.ProcessGroup,
    dp_size: int,
    dp_rank: int,
    ep_group: Magi2ParallelGroup,
) -> None:
    checkpoint_file = os.path.join(checkpoint_root, "preview", "model.safetensors")
    checkpoint = load_file(checkpoint_file, device="cpu")

    with _patched_weight_groups(ep_group):
        oracle = Magi2PreviewTransformer(_tiny_config())
        assert oracle.load_weights(checkpoint.items()) == set(oracle.state_dict())

        with torch.device("meta"):
            target = Magi2PreviewTransformer(_tiny_config())
        target.requires_grad_(False)
        pipeline = Magi2Pipeline.__new__(Magi2Pipeline)
        nn.Module.__init__(pipeline)
        pipeline.checkpoint_root = checkpoint_root
        pipeline.transformer = target

        plan_result = build_checkpoint_mmap_plan(
            pipeline,
            dit_modules=(("transformer", target),),
            sources=(
                _WeightSourceStub(
                    model_or_path=checkpoint_root,
                    subfolder="preview",
                    revision=None,
                    prefix="transformer.",
                ),
            ),
            model_path=checkpoint_root,
            tensor_parallel_size=1,
            use_hsdp=False,
            online_quantization=False,
        )
        assert plan_result.fallback_reason is None
        assert plan_result.plan is not None

        backend = object.__new__(DistributedLayerwiseOffloadBackend)
        backend.device = torch.device("cpu")
        backend._using_rank_local_mmap = False
        backend._mmap_transforms_by_tensor_id = {}
        backend._load_weights_via_mmap(
            pipeline,
            _PipelineModulesStub(dits=[target], dit_names=["transformer"]),
            plan_result.plan,
        )

    target_block = target.block.layers[0]
    oracle_block = oracle.block.layers[0]
    target_parameters = dict(target_block.named_parameters())
    oracle_parameters = dict(oracle_block.named_parameters())

    # The mmap loader binds the released full-head tensor. The DLO shard step
    # must apply the model-declared EP transform before it computes the DP
    # shard, so DP peers gather identical SP-local layouts in DP2SP2.
    moe_name = "mlp.moe_mlp.gate"
    mmap_moe = target_parameters[moe_name]
    mmap_transform = backend._mmap_transforms_by_tensor_id[id(mmap_moe)]
    assert callable(mmap_transform)
    assert tuple(mmap_moe.shape) == tuple(checkpoint[f"block.layers.0.{moe_name}"].shape)
    assert tuple(mmap_transform(mmap_moe).shape) == tuple(oracle_parameters[moe_name].shape)

    # Keep this gloo regression accelerator-isolated. Pinned allocation is the
    # final, layout-preserving operation in _shard_and_pin and has separate
    # backend coverage; enabling it here would initialize CUDA in every worker.
    cpu_shards, metadata = DistributedLayerwiseOffloadHook._shard_and_pin(
        target_parameters,
        dict(target_block.named_buffers()),
        dp_size=dp_size,
        rank=dp_rank,
        pin_memory=False,
        tensor_transforms=backend._mmap_transforms_by_tensor_id,
    )
    assert all(not shard.requires_grad for shard in cpu_shards.values())
    assert all(parameter.numel() == 0 for parameter in target_block.parameters())

    for dtype, local_shard in cpu_shards.items():
        gathered = torch.empty(
            local_shard.numel() * dp_size,
            dtype=dtype,
            device="cpu",
        )
        dist.all_gather_into_tensor(gathered, local_shard, group=dp_group)

        for tensor_metadata in metadata[dtype]:
            name = tensor_metadata["name"]
            offset = tensor_metadata["offset"]
            numel = tensor_metadata["numel"]
            reconstructed = gathered[offset : offset + numel].view(tensor_metadata["shape"])
            torch.testing.assert_close(reconstructed, oracle_parameters[name])


def _distributed_worker(rank: int, rendezvous: str, checkpoint_root: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=_WORLD_SIZE,
    )
    try:
        dp2_groups = _new_groups(((0, 2), (1, 3)))
        sp2_groups = _new_groups(((0, 1), (2, 3)))

        _load_mmap_transform_and_reconstruct(
            checkpoint_root,
            dist.group.WORLD,
            dp_size=4,
            dp_rank=rank,
            ep_group=Magi2ParallelGroup(None, world_size=1, rank=0),
        )
        dist.barrier()

        dp_ranks, dp_group = _current_group(rank, dp2_groups)
        sp_ranks, sp_group = _current_group(rank, sp2_groups)
        _load_mmap_transform_and_reconstruct(
            checkpoint_root,
            dp_group,
            dp_size=2,
            dp_rank=dp_ranks.index(rank),
            ep_group=Magi2ParallelGroup(
                sp_group,
                world_size=2,
                rank=sp_ranks.index(rank),
            ),
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="requires torch.distributed gloo",
)
def test_dlo_mmap_transform_dp_shard_and_allgather_for_dp4_and_dp2sp2() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        _write_checkpoint(temp_dir)
        rendezvous = f"file://{os.path.join(temp_dir, 'gloo-rendezvous')}"
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
            mp.spawn(
                _distributed_worker,
                args=(rendezvous, temp_dir),
                nprocs=_WORLD_SIZE,
                join=True,
            )
