# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import os
import tempfile
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy

import vllm_omni.diffusion.offloader.distributed_layerwise_backend as dlo_module
from tests.diffusion.models.magi2.test_native_distributed_parity import (
    _current_group,
    _new_groups,
    _patched_groups,
)
from tests.diffusion.models.magi2.test_native_packing import _tiny_config, _tiny_model, _tiny_sampler
from tests.diffusion.offloader.helpers import patch_offload_runtime
from vllm_omni.diffusion.distributed.hsdp import shard_model
from vllm_omni.diffusion.models.magi2.modeling_magi2 import Magi2PreviewTransformer
from vllm_omni.diffusion.models.magi2.parallel import Magi2ParallelGroup
from vllm_omni.diffusion.models.magi2.pipeline_magi2 import Magi2Pipeline
from vllm_omni.diffusion.models.magi2.preview_data_proxy import Magi2PackedLayout
from vllm_omni.diffusion.models.magi2.sampler_magi2 import CFGConfig
from vllm_omni.diffusion.offloader.base import OffloadConfig
from vllm_omni.diffusion.offloader.config import OffloadStrategy

_WORLD_SIZE = 4
pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


@contextmanager
def _cpu_offload_runtime():
    with pytest.MonkeyPatch.context() as monkeypatch:
        patch_offload_runtime(monkeypatch, dlo_module.current_omni_platform, synchronize=True)
        yield


def _enable_dlo(
    model: Magi2PreviewTransformer,
    *,
    dp_group: dist.ProcessGroup | None,
    dp_size: int,
    dp_rank: int,
) -> None:
    pipeline = Magi2Pipeline.__new__(Magi2Pipeline)
    nn.Module.__init__(pipeline)
    pipeline.transformer = model
    backend = dlo_module.DistributedLayerwiseOffloadBackend(
        OffloadConfig(
            strategy=OffloadStrategy.DISTRIBUTED_LAYER_WISE,
            pin_cpu_memory=False,
            dp_size=dp_size,
            dlo_use_allgather=dp_size > 1,
        ),
        torch.device("cpu"),
    )
    backend.dp_group = dp_group
    backend.rank = dp_rank
    backend.enable(pipeline)
    assert backend.enabled


def _enable_hsdp(model: Magi2PreviewTransformer) -> None:
    mesh = init_device_mesh("cpu", mesh_shape=(1, _WORLD_SIZE), mesh_dim_names=("replicate", "shard"))
    ignored = {
        parameter for name in model._hsdp_ignored_modules for parameter in model.get_submodule(name).parameters()
    }
    shard_model(
        model,
        mesh=mesh,
        mp_policy=MixedPrecisionPolicy(cast_forward_inputs=False),
        hsdp_shard_conditions=model._hsdp_shard_conditions,
        ignored_params=ignored,
    )
    model.requires_grad_(False)


def _sampler_tensors(seed: int) -> dict[str, torch.Tensor]:
    # 14 tokens per CFG branch, so the packed sequence splits evenly across
    # two Ulysses ranks; gloo cannot gather uneven shards.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return {
        "latent": torch.randn(1, 4, 2, 1, 3, generator=generator),
        "audio_latent": torch.randn(1, 5, 4, generator=generator),
        "txt_feat": torch.randn(1, 3, 4, generator=generator),
        "null_txt_feat": torch.randn(1, 3, 4, generator=generator),
    }


def _steps(sampler, layout: Magi2PackedLayout):
    return [
        sampler.prepare_model_input(**_sampler_tensors(seed), t=t, cfg_config=CFGConfig(), layout=layout)
        for seed, t in enumerate((torch.tensor([900.0]), torch.tensor([450.0])))
    ]


def _run(model: Magi2PreviewTransformer, *, compile_regions: bool) -> list[tuple[torch.Tensor, torch.Tensor]]:
    sampler = _tiny_sampler(model)
    steps = _steps(sampler, Magi2PackedLayout())
    if not compile_regions:
        return [sampler.forward(step) for step in steps]
    torch._dynamo.reset()
    model.compile_regions(fullgraph=True, backend="eager", dynamic=True)
    outputs = [sampler.forward(steps[0])]
    with torch._dynamo.config.patch(error_on_recompile=True):
        outputs.append(sampler.forward(steps[1]))
    return outputs


def _assert_same(actual, expected, *, exact: bool, label: str) -> None:
    for (video, audio), (video_ref, audio_ref) in zip(actual, expected, strict=True):
        if exact:
            assert torch.equal(video, video_ref), label
            assert torch.equal(audio, audio_ref), label
        else:
            torch.testing.assert_close(video, video_ref, atol=2e-5, rtol=2e-4, msg=label)
            torch.testing.assert_close(audio, audio_ref, atol=2e-5, rtol=2e-4, msg=label)


def _distributed_worker(rank: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=_WORLD_SIZE)
    try:
        singleton = Magi2ParallelGroup(None, world_size=1, rank=0)
        with _patched_groups(singleton, singleton):
            oracle = _tiny_model()
            checkpoint = [(name, value.detach().clone()) for name, value in oracle.state_dict().items()]
            expected = _run(oracle, compile_regions=False)

        sp2_groups = _new_groups(((0, 1), (2, 3)))
        dp2_groups = _new_groups(((0, 2), (1, 3)))
        sp_group = _current_group(rank, sp2_groups)
        dp_ranks, dp_process_group = next((ranks, group) for ranks, group in dp2_groups if rank in ranks)

        def dlo_rank_local(model: Magi2PreviewTransformer) -> None:
            _enable_dlo(model, dp_group=None, dp_size=1, dp_rank=0)

        def dlo_allgather(model: Magi2PreviewTransformer) -> None:
            _enable_dlo(model, dp_group=dp_process_group, dp_size=len(dp_ranks), dp_rank=dp_ranks.index(rank))

        variants = (
            ("dlo_rank_local_sp2", sp_group, dlo_rank_local),
            ("dlo_allgather_dp2sp2", sp_group, dlo_allgather),
            ("hsdp4", singleton, _enable_hsdp),
        )
        for name, ulysses_group, enable in variants:
            with ExitStack() as stack:
                stack.enter_context(_patched_groups(singleton, ulysses_group))
                stack.enter_context(_cpu_offload_runtime())
                outputs = {}
                for compile_regions in (False, True):
                    model = Magi2PreviewTransformer(_tiny_config())
                    assert model.load_weights(checkpoint) == set(model.state_dict())
                    enable(model)
                    outputs[compile_regions] = _run(model, compile_regions=compile_regions)
            _assert_same(outputs[False], expected, exact=False, label=f"{name} eager vs single-rank oracle")
            _assert_same(outputs[True], outputs[False], exact=True, label=f"{name} compiled vs eager")
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="requires torch.distributed gloo",
)
def test_compiled_regions_match_eager_under_dlo_and_hsdp() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        rendezvous = f"file://{os.path.join(temp_dir, 'gloo-rendezvous')}"
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
            mp.spawn(_distributed_worker, args=(rendezvous,), nprocs=_WORLD_SIZE, join=True)
