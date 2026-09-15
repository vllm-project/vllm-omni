# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-side DLO contracts. CPU tests do not claim transfer validation."""

from dataclasses import replace

import pytest
import torch

from tests.diffusion.models.mammoth_moda2.test_pipeline_sp import _config, _pipeline, _request
from tests.diffusion.offloader.helpers import patch_offload_runtime
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.model_loader.host_weight_plan import _planned_source_prefixes
from vllm_omni.diffusion.models.mammoth_moda2.pipeline_mammothmoda2_dit import (
    _build_mammoth_config,
    _validate_sequence_parallel_runtime,
)
from vllm_omni.diffusion.offloader.base import OffloadConfig
from vllm_omni.diffusion.offloader.component_utils import iter_streamable_dits
from vllm_omni.diffusion.offloader.distributed_layerwise_backend import DistributedLayerwiseOffloadBackend
from vllm_omni.diffusion.offloader.offload_plan import get_offload_plan
from vllm_omni.diffusion.offloader.plan_resolver import resolve_offload_plan
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_native_diffusion_extras_default_is_an_independent_dict():
    first, second = _config(), _config()
    assert first.extras == second.extras == {}
    first.extras["mammoth_experimental_dlo"] = True
    assert second.extras == {}


def _dlo_config(degree=1, *, allgather=False, **kwargs):
    return replace(
        _config(degree),
        enable_distributed_layerwise_offload=True,
        dlo_use_allgather=allgather,
        extras={"mammoth_experimental_dlo": True},
        **kwargs,
    )


@pytest.mark.parametrize("value", ["true", 1, None])
def test_dlo_flag_requires_boolean(monkeypatch, value):
    config = replace(_dlo_config(), extras={"mammoth_experimental_dlo": value})
    with pytest.raises(ValueError, match="mammoth_experimental_dlo must be a bool"):
        _pipeline(config, monkeypatch)


def test_dlo_plan_discovers_only_main_transformer_blocks(monkeypatch):
    config = _dlo_config()
    pipeline = _pipeline(config, monkeypatch)
    plan = get_offload_plan(pipeline)
    assert plan is not None
    assert plan.block_attrs == {"gen_transformer": ("layers",)}
    assert not plan.offload_submodules
    assert not plan.encoder_block_attrs
    assert not plan.resident_dit_paths
    resolved = resolve_offload_plan(pipeline, OffloadConfig.from_od_config(config))
    entries = list(iter_streamable_dits(resolved, pipeline.device))
    assert len(entries) == 1
    component, stack = entries[0]
    model, blocks = component.module, stack.streaming
    assert component.path == "gen_transformer"
    assert model is pipeline.gen_transformer
    assert stack.attrs == ("layers",)
    assert blocks == tuple(model.layers)
    assert not {id(block) for block in blocks}.intersection(id(block) for block in model.context_refiner)


def test_mixed_root_checkpoint_rejects_dedicated_mmap_source_selection(monkeypatch):
    config = _dlo_config()
    pipeline = _pipeline(config, monkeypatch)
    # Test the real source-selection boundary without initializing unrelated
    # CUDA-only checkpoint adapters in a CPU test. The planner catches this
    # incompatibility and selects the ordinary loader.
    with pytest.raises(RuntimeError, match="dedicated component weight source"):
        _planned_source_prefixes(pipeline.weights_sources, [("gen_transformer", pipeline.gen_transformer)])


@pytest.mark.parametrize("transfer", ["rank-local", "allgather"])
def test_compact_config_does_not_mislabel_ordinary_layer_backend(transfer):
    config = replace(
        _config(2),
        diffusion_offload_config={
            "mode": "layer",
            "components": ["dit"],
            "layer_options": {"dit": {"weight_transfer": transfer}},
        },
        extras={"mammoth_experimental_dlo": True},
    )
    if transfer == "allgather":
        _validate_sequence_parallel_runtime(config, _build_mammoth_config(config))
    else:
        with pytest.raises(ValueError, match="distributed layerwise offload backend"):
            _validate_sequence_parallel_runtime(config, _build_mammoth_config(config))


@pytest.mark.parametrize("allgather", [False, True])
def test_sp_dlo_requires_explicit_experimental_opt_in(allgather):
    config = _dlo_config(2, allgather=allgather)
    # CPU validation checks admission only. Actual SP group construction and
    # communication are exercised by the separate two-GPU test.
    _validate_sequence_parallel_runtime(config, _build_mammoth_config(config))
    with pytest.raises(ValueError, match="offload"):
        _validate_sequence_parallel_runtime(replace(config, extras={}), _build_mammoth_config(config))


@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("enforce_eager", False, "eager"),
        ("cache_backend", "cache_dit", "cache"),
        ("max_num_seqs", 2, "max_num_seqs"),
        ("dlo_resident_layers", 1, "resident"),
    ],
)
def test_experimental_dlo_rejects_unqualified_modes(monkeypatch, degree, field, value, message):
    with pytest.raises(ValueError, match=message):
        _pipeline(_dlo_config(degree, **{field: value}), monkeypatch)


def test_dlo_rejects_dev_before_construction(monkeypatch):
    config = _dlo_config()
    config.tf_model_config.params["llm_config"]["model_type"] = "mammothmoda2_qwen3_vl"
    with pytest.raises(ValueError, match="Preview"):
        _pipeline(config, monkeypatch)


@pytest.mark.parametrize("guidance", [1.0, 4.0])
def test_dlo_forward_preserves_runner_no_grad_for_storage_rebinding(monkeypatch, guidance):
    config = _dlo_config()
    pipeline = _pipeline(config, monkeypatch)
    observed = []
    handle = pipeline.gen_transformer.layers[0].register_forward_pre_hook(
        lambda module, args: observed.append((torch.is_grad_enabled(), torch.is_inference_mode_enabled()))
    )
    try:
        with torch.no_grad(), set_forward_context(omni_diffusion_config=config):
            for _ in range(2):
                result = pipeline(_request(3, guidance))
                assert torch.isfinite(result.output).all()
    finally:
        handle.remove()
    assert len(observed) == (4 if guidance == 1.0 else 8)
    assert all(state == (False, False) for state in observed)


@pytest.mark.parametrize("guidance", [1.0, 4.0])
def test_dlo_request_recovers_after_intermediate_layer_failure(monkeypatch, guidance):
    config = _dlo_config()
    config.tf_model_config.params["gen_dit_config"]["num_layers"] = 3
    pipeline = _pipeline(config, monkeypatch)
    patch_offload_runtime(monkeypatch, current_omni_platform, synchronize=True)
    backend = DistributedLayerwiseOffloadBackend(
        replace(OffloadConfig.from_od_config(config), pin_cpu_memory=False), torch.device("cpu")
    )

    def fail_layer(module, args):
        raise RuntimeError("injected intermediate layer failure")

    with torch.no_grad(), set_forward_context(omni_diffusion_config=config):
        expected = pipeline(_request(3, guidance)).output.clone()
        backend.enable(pipeline)
        try:
            torch.testing.assert_close(pipeline(_request(3, guidance)).output, expected, rtol=0, atol=0)
            handle = pipeline.gen_transformer.layers[1].register_forward_pre_hook(fail_layer)
            try:
                with pytest.raises(RuntimeError, match="injected intermediate layer failure"):
                    pipeline(_request(3, guidance))
            finally:
                handle.remove()
            for _ in range(2):
                torch.testing.assert_close(pipeline(_request(3, guidance)).output, expected, rtol=0, atol=0)
        finally:
            backend.disable()
