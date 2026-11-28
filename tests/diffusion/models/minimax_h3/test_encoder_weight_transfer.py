# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3 import encoder as encoder_module
from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3
from vllm_omni.diffusion.models.minimax_h3.encoder import MiniMaxH3Qwen3VLEncoder
from vllm_omni.diffusion.models.minimax_h3.encoder_config import EncoderWeightTransfer, encoder_weight_transfer
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("config", [None, {}, {"other_setting": 1}])
def test_default_preserves_original_weight_transfer(config):
    assert encoder_weight_transfer(config) is EncoderWeightTransfer.OFF


@pytest.mark.parametrize("mode", list(EncoderWeightTransfer))
def test_modes_are_resolved_without_mutating_deployment(mode):
    config = {"encoder_weight_transfer": mode.value, "other_setting": 1}
    assert encoder_weight_transfer(config) is mode
    assert config == {"encoder_weight_transfer": mode.value, "other_setting": 1}


@pytest.mark.parametrize("value", ["packed_prefetch", "prefetch", "on", "PACKED", "", "packed "])
def test_unknown_modes_are_rejected(value):
    with pytest.raises(ValueError, match="off or packed"):
        encoder_weight_transfer({"encoder_weight_transfer": value})


@pytest.mark.parametrize("value", [None, True, 1, ["packed"], {"mode": "packed"}])
def test_non_string_modes_are_rejected(value):
    with pytest.raises(TypeError, match="must be a string"):
        encoder_weight_transfer({"encoder_weight_transfer": value})


@pytest.mark.parametrize("unsupported", ["cpu", "external_encoder", "no_layer_offload", "unselected_encoder"])
def test_pipeline_rejects_unsupported_transfer_before_loading_weights(mocker, unsupported):
    mocker.patch.object(
        pipeline_minimax_h3,
        "get_local_device",
        return_value=mocker.Mock(type="cpu" if unsupported == "cpu" else "npu"),
    )
    mocker.patch.object(pipeline_minimax_h3, "offload_streams_blocks", return_value=unsupported != "no_layer_offload")
    mocker.patch.object(
        pipeline_minimax_h3, "should_offload_component", return_value=unsupported != "unselected_encoder"
    )
    config = mocker.Mock(
        parallel_config=mocker.Mock(cfg_parallel_size=1),
        diffusion_attention_schedule=None,
        additional_config={"encoder_weight_transfer": EncoderWeightTransfer.PACKED.value},
        model_loaded={"text_encoder": unsupported != "external_encoder"},
        diffusion_offload_config={"mode": "layer", "components": ["dit"]},
    )
    with pytest.raises(ValueError, match="local MiniMax-H3 NPU text encoder selected for layer offload"):
        MiniMaxH3Pipeline(od_config=config)


def _bare_layerwise_encoder(mocker):
    encoder = MiniMaxH3Qwen3VLEncoder.__new__(MiniMaxH3Qwen3VLEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.text_model = torch.nn.Identity()
    encoder.device_target = mocker.Mock(type="npu")
    encoder._omni_non_block_stager = None
    encoder._omni_layerwise_enabled = True
    encoder._omni_layerwise_hooks = ()
    encoder._omni_component_cache = None
    mocker.patch.object(encoder, "_omni_non_block_modules", return_value=[mocker.sentinel.module])
    return encoder


@pytest.mark.parametrize("supplied_cache", [False, True])
def test_packed_staging_reuses_supplied_cache_or_creates_bounded_policy(mocker, supplied_cache):
    existing = mocker.sentinel.component_cache if supplied_cache else None
    factory = mocker.patch.object(encoder_module, "BoundedAllocatorCache", return_value=mocker.sentinel.bounded_cache)
    stager = mocker.patch.object(encoder_module, "PinnedModuleStager")
    encoder = _bare_layerwise_encoder(mocker)
    encoder._omni_component_cache = existing
    encoder.prepare_packed_offload()
    stager.assert_called_once_with(
        [mocker.sentinel.module],
        encoder.device_target,
        pin_memory=True,
        cache_retention=existing if supplied_cache else factory.return_value,
        pack_by_dtype=True,
    )
    assert stager.call_args.kwargs["cache_retention"] is (existing if supplied_cache else factory.return_value)
    assert stager.call_args.kwargs["pack_by_dtype"] is True
    # Cache reuse must not turn into live GPU weight residency.
    assert "retain_device_storage" not in stager.call_args.kwargs
    if supplied_cache:
        factory.assert_not_called()
    else:
        factory.assert_called_once_with(encoder.device_target)


def test_default_unpacked_layerwise_staging_does_not_enable_new_cache_policy(mocker):
    factory = mocker.patch.object(encoder_module, "BoundedAllocatorCache")
    stager = mocker.patch.object(encoder_module, "PinnedModuleStager")
    encoder = _bare_layerwise_encoder(mocker)
    encoder.load_to_device()
    assert stager.call_args.kwargs["cache_retention"] is None
    assert "pack_by_dtype" not in stager.call_args.kwargs
    factory.assert_not_called()
    stager.return_value.load.assert_called_once_with()
