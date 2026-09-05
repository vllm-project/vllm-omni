# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.offloader.base import OffloadConfig, OffloadStrategy
from vllm_omni.diffusion.offloader.distributed_layerwise_backend import (
    DistributedLayerwiseOffloadBackend,
)
from vllm_omni.diffusion.offloader.module_collector import PipelineModules
from vllm_omni.diffusion.offloader.offload_plan import OffloadPlan

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


def _od_config(**overrides):
    values = {
        "enable_cpu_offload": False,
        "enable_layerwise_offload": False,
        "enable_distributed_layerwise_offload": True,
        "dlo_use_allgather": False,
        "dlo_resident_layers": 0,
        "dlo_offload_components": {},
        "pin_cpu_memory": False,
        "parallel_config": SimpleNamespace(
            use_hsdp=False,
            data_parallel_size=1,
            sequence_parallel_size=1,
        ),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _backend(policy: dict[str, bool]) -> DistributedLayerwiseOffloadBackend:
    backend = object.__new__(DistributedLayerwiseOffloadBackend)
    backend.config = OffloadConfig(
        strategy=OffloadStrategy.DISTRIBUTED_LAYER_WISE,
        pin_cpu_memory=False,
        dlo_use_allgather=False,
        dlo_offload_components=policy,
    )
    backend.device = torch.device("cpu")
    return backend


def _modules() -> PipelineModules:
    return PipelineModules(
        dits=[],
        dit_names=[],
        encoders=[nn.Module()],
        encoder_names=["text_encoder"],
        vaes=[nn.Module(), nn.Module()],
        vae_names=["video_vae", "audio_vae"],
    )


def test_component_policy_resolves_exact_then_default():
    backend = _backend({"text_encoder": False, "default": True})

    assert not backend._component_offload_enabled("text_encoder")
    assert backend._component_offload_enabled("video_vae")


@pytest.mark.parametrize(
    ("value", "error", "match"),
    [
        ([], TypeError, "must be a mapping"),
        (["text_encoder"], TypeError, "must be a mapping"),
        (False, TypeError, "must be a mapping"),
        ({1: False}, TypeError, "keys must be non-empty strings"),
        ({"text_encoder": 0}, TypeError, "must be a boolean"),
    ],
)
def test_component_policy_rejects_invalid_shapes(value, error, match):
    with pytest.raises(error, match=match):
        OffloadConfig.from_od_config(_od_config(dlo_offload_components=value))


def test_component_policy_accepts_none_as_default():
    config = OffloadConfig.from_od_config(_od_config(dlo_offload_components=None))

    assert config.dlo_offload_components == {}


def test_component_policy_requires_dlo():
    with pytest.raises(ValueError, match="requires distributed layerwise offload"):
        OffloadConfig.from_od_config(
            _od_config(
                enable_distributed_layerwise_offload=False,
                dlo_offload_components={"text_encoder": False},
            )
        )


def test_component_policy_rejects_unknown_or_dit_names():
    backend = _backend({"transformer": False})

    with pytest.raises(ValueError, match="unknown auxiliary components.*transformer"):
        backend._validate_component_offload_policy(_modules())


@pytest.mark.parametrize(
    ("policy", "encoder_stage", "video_stage", "audio_stage", "try_encoder"),
    [
        ({}, True, True, True, True),
        ({"text_encoder": False}, False, True, True, False),
        ({"default": False, "video_vae": True}, False, True, False, False),
    ],
)
def test_prepare_auxiliary_components_applies_policy(
    mocker,
    policy,
    encoder_stage,
    video_stage,
    audio_stage,
    try_encoder,
):
    backend = _backend(policy)
    backend._try_layerwise_offload_encoder = mocker.Mock()
    backend._register_on_demand_hook = mocker.Mock()
    modules = _modules()
    plan = OffloadPlan(
        encoder_block_attrs={"text_encoder": ("layers",)},
        on_demand_component_paths=frozenset({"text_encoder", "video_vae", "audio_vae"}),
    )

    backend._prepare_auxiliary_components(modules, plan)

    assert backend._try_layerwise_offload_encoder.called is try_encoder
    assert modules.encoders[0]._omni_dlo_offload_enabled is encoder_stage
    assert modules.vaes[0]._omni_dlo_offload_enabled is video_stage
    assert modules.vaes[1]._omni_dlo_offload_enabled is audio_stage
    assert backend._register_on_demand_hook.call_args_list == [
        mocker.call(modules.encoders[0], "text_encoder", stage_on_demand=encoder_stage),
        mocker.call(modules.vaes[0], "video_vae", stage_on_demand=video_stage),
        mocker.call(modules.vaes[1], "audio_vae", stage_on_demand=audio_stage),
    ]
