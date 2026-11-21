# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import ast
from pathlib import Path

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3.encoder_config import EncoderWeightTransfer, encoder_weight_transfer

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
    source = Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MiniMaxH3Pipeline")
    init = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    # Run the production constructor's early validation without importing model dependencies.
    cls = ast.ClassDef(
        name=owner.name,
        bases=[ast.parse("torch.nn.Module", mode="eval").body],
        keywords=[],
        body=[init],
        decorator_list=[],
    )
    scope = {
        "torch": torch,
        "EncoderWeightTransfer": EncoderWeightTransfer,
        "encoder_weight_transfer": encoder_weight_transfer,
        "get_local_device": lambda: mocker.Mock(type="cpu" if unsupported == "cpu" else "npu"),
        "offload_streams_blocks": lambda config: unsupported != "no_layer_offload",
        "should_offload_component": lambda config, component: unsupported != "unselected_encoder",
        "TEXT_ENCODER_COMPONENT": "text_encoder",
    }
    module = ast.Module(body=ast.parse("from __future__ import annotations").body + [cls], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), scope)
    config = mocker.Mock(
        parallel_config=mocker.Mock(cfg_parallel_size=1),
        diffusion_attention_schedule=None,
        additional_config={"encoder_weight_transfer": EncoderWeightTransfer.PACKED.value},
        model_loaded={"text_encoder": unsupported != "external_encoder"},
        diffusion_offload_config={"mode": "layer", "components": ["dit"]},
    )
    with pytest.raises(ValueError, match="local MiniMax-H3 NPU text encoder selected for layer offload"):
        scope[owner.name](od_config=config)


def _encoder_staging_methods():
    source = Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/minimax_h3/encoder.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MiniMaxH3Qwen3VLEncoder")
    methods: list[ast.stmt] = [
        n
        for n in owner.body
        if isinstance(n, ast.FunctionDef) and n.name in {"prepare_packed_offload", "load_to_device"}
    ]
    module = ast.Module(body=ast.parse("from __future__ import annotations").body + methods, type_ignores=[])
    return source, ast.fix_missing_locations(module)


@pytest.mark.parametrize("supplied_cache", [False, True])
def test_packed_staging_reuses_supplied_cache_or_creates_bounded_policy(mocker, supplied_cache):
    source, module = _encoder_staging_methods()
    existing = mocker.sentinel.component_cache if supplied_cache else None
    factory = mocker.Mock(return_value=mocker.sentinel.bounded_cache)
    stager = mocker.Mock()
    scope = {"PinnedModuleStager": stager, "BoundedAllocatorCache": factory}
    exec(compile(module, str(source), "exec"), scope)
    encoder = mocker.Mock(
        is_loaded=True,
        device_target=mocker.Mock(type="npu"),
        _omni_non_block_stager=None,
        _omni_layerwise_enabled=True,
        _omni_layerwise_hooks=(),
        _omni_component_cache=existing,
        _omni_non_block_modules=mocker.Mock(return_value=[mocker.sentinel.module]),
    )
    scope["prepare_packed_offload"](encoder)
    assert stager.call_args.kwargs["cache_retention"] is (existing if supplied_cache else factory.return_value)
    assert stager.call_args.kwargs["pack_by_dtype"] is True
    # Cache reuse must not turn into live GPU weight residency.
    assert "retain_device_storage" not in stager.call_args.kwargs
    if supplied_cache:
        factory.assert_not_called()
    else:
        factory.assert_called_once_with(encoder.device_target)


def test_default_unpacked_layerwise_staging_does_not_enable_new_cache_policy(mocker):
    source, module = _encoder_staging_methods()
    factory = mocker.Mock()
    stager = mocker.Mock()
    scope = {"PinnedModuleStager": stager, "BoundedAllocatorCache": factory}
    exec(compile(module, str(source), "exec"), scope)
    encoder = mocker.Mock(
        is_loaded=True,
        device_target=mocker.Mock(type="npu"),
        _omni_non_block_stager=None,
        _omni_layerwise_enabled=True,
        _omni_layerwise_hooks=(),
        _omni_component_cache=None,
        _omni_non_block_modules=mocker.Mock(return_value=[mocker.sentinel.module]),
    )
    scope["load_to_device"](encoder)
    assert stager.call_args.kwargs["cache_retention"] is None
    assert "pack_by_dtype" not in stager.call_args.kwargs
    factory.assert_not_called()
    stager.return_value.load.assert_called_once_with()
