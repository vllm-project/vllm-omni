# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _OffloadAbort(BaseException):
    pass


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
