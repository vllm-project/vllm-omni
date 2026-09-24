# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _OffloadAbort(BaseException):
    pass


def test_no_allgather_example_selects_distributed_backend():
    from examples.offline_inference.minimax_h3.dlo_lifecycle import engine_kwargs
    from vllm_omni.diffusion.offloader.base import OffloadConfig
    from vllm_omni.diffusion.offloader.config import OffloadStrategy

    args = SimpleNamespace(
        model="/fake/MiniMax-H3/FL2VA",
        mode="dlo-no-allgather",
        dp_size=2,
        tp_size=2,
        batch_wait_ms=500.0,
        init_timeout=1800.0,
    )
    kwargs = engine_kwargs(args)
    od_config = SimpleNamespace(
        diffusion_offload_config=None,
        enable_cpu_offload=False,
        enable_layerwise_offload=False,
        enable_distributed_layerwise_offload=kwargs["enable_distributed_layerwise_offload"],
        dlo_use_allgather=kwargs["dlo_use_allgather"],
        dlo_resident_layers=kwargs["dlo_resident_layers"],
        dlo_host_registration_limit_gib=0.0,
        host_weight_runtime_mode="disabled",
        pin_cpu_memory=True,
        parallel_config=SimpleNamespace(
            data_parallel_size=kwargs["data_parallel_size"],
            sequence_parallel_size=1,
            use_hsdp=False,
        ),
    )

    config = OffloadConfig.from_od_config(od_config)

    assert config.strategy is OffloadStrategy.DISTRIBUTED_LAYER_WISE
    assert config.dlo_resident_layers == 0
    assert not config.uses_allgather("dit")


def test_encoder_non_block_children_use_one_shared_snapshot_stager(monkeypatch, mocker):
    from vllm_omni.diffusion.models.minimax_h3 import encoder as encoder_module

    class VisionStack(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = torch.nn.Linear(2, 2)
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)])

    class TextStack(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(4, 2)
            self.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)])

    encoder = object.__new__(encoder_module.MiniMaxH3Qwen3VLEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.device_target = torch.device("cpu")
    encoder.vision = VisionStack()
    encoder.text_model = TextStack()
    hook = mocker.Mock()
    hook.pin_memory = False
    encoder._omni_layerwise_hooks = [hook]
    encoder._omni_layerwise_enabled = True
    cache = mocker.Mock()
    encoder.set_omni_component_cache(cache)
    stager = mocker.Mock()
    stager_cls = mocker.Mock(return_value=stager)
    monkeypatch.setattr(encoder_module, "PinnedModuleStager", stager_cls)

    encoder.load_to_device()
    encoder.offload_to_cpu()

    stager_cls.assert_called_once_with(
        [encoder.vision.patch_embed, encoder.text_model.embed_tokens],
        torch.device("cpu"),
        pin_memory=False,
        cache_retention=cache,
    )
    stager.load.assert_called_once_with()
    stager.offload.assert_called_once_with()
    hook.offload_layer.assert_called_once_with()


def test_manual_component_failure_forces_retained_cache_release(mocker):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = mocker.Mock()
    pipeline.od_config.diffusion_offload_config = None
    pipeline.od_config.enable_layerwise_offload = False
    pipeline.od_config.enable_distributed_layerwise_offload = True
    pipeline._model_cpu_offload_modules = []
    pipeline._dlo_component_cache = mocker.Mock()
    component = mocker.Mock()
    component.offload_to_cpu.side_effect = _OffloadAbort("offload failed")

    with pytest.raises(RuntimeError, match="component failed"):
        with pipeline._component_on_device(component):
            raise RuntimeError("component failed")

    component.load_to_device.assert_called_once_with()
    component.offload_to_cpu.assert_called_once_with()
    pipeline._dlo_component_cache.release_if_needed.assert_called_once_with(force=True)


def test_manual_component_offload_failure_forces_retained_cache_release(mocker):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = mocker.Mock()
    pipeline.od_config.diffusion_offload_config = None
    pipeline.od_config.enable_layerwise_offload = False
    pipeline.od_config.enable_distributed_layerwise_offload = True
    pipeline._model_cpu_offload_modules = []
    pipeline._dlo_component_cache = mocker.Mock()
    component = mocker.Mock()
    component.offload_to_cpu.side_effect = [None, _OffloadAbort("offload failed")]

    with pipeline._component_on_device(component):
        pass
    with pytest.raises(_OffloadAbort, match="offload failed"):
        with pipeline._component_on_device(component):
            pass

    assert component.load_to_device.call_count == 2
    assert component.offload_to_cpu.call_count == 2
    pipeline._dlo_component_cache.release_if_needed.assert_called_once_with(force=True)
