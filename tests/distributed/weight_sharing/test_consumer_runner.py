# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Consumer load ownership: production runner/loader methods with CPU IPC tokens."""

import runpy
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.distributed.weight_sharing.helpers import FakeCodec, load_class_methods, make_context, make_weights

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class LoadError(RuntimeError):
    pass


@pytest.fixture
def consumer_load(weight_sharing, contracts, monkeypatch):
    offload_config = runpy.run_path(
        str(Path(__file__).resolve().parents[3] / "vllm_omni/diffusion/offloader/config.py")
    )
    codec = FakeCodec(contracts)
    provider = weight_sharing.TorchIpcBackend(codec=codec)
    backend = weight_sharing.TorchIpcBackend(codec=codec)
    context = make_context(contracts)
    bindings, spec = make_weights(contracts, two_storages=True)
    exported = provider.prepare_export(bindings, spec, context)
    payload = provider.claim_export(exported)
    group = SimpleNamespace(role="consumer", session_id="session", payload=payload)
    factory = Mock(return_value=backend)
    monkeypatch.setitem(sys.modules, "vllm_omni.distributed.weight_sharing", weight_sharing)
    monkeypatch.setattr(weight_sharing, "TorchIpcBackend", factory)
    platform = Mock(
        current_device=Mock(return_value=3),
        get_torch_device=Mock(side_effect=lambda index: torch.device("cuda", index)),
    )
    monkeypatch.setitem(sys.modules, "vllm_omni.platforms", SimpleNamespace(current_omni_platform=platform))
    # Model binding is outside this test; imported views still have real backend lifetimes.
    adapter = SimpleNamespace(
        shared_param_full_names=Mock(return_value=[]), is_shared_checkpoint_name=Mock(return_value=False)
    )
    model = SimpleNamespace(
        named_parameters=lambda: [],
        named_buffers=lambda: [],
        named_modules=lambda: [],
        eval=lambda: model,
        get_weight_sharing_adapter=Mock(return_value=adapter),
    )

    def attach(received_backend, received_model, *, payload, group, device_index):
        assert received_model is model
        assert device_index == 3
        assert received_backend is runner._weight_share_backend is backend
        assert group is loader.od_config.weight_sharing
        assert runner._weight_share_session_id == "session"
        imported = backend.open_import(payload, spec, context)
        model.views = list(imported.values())  # Model-owned aliases make close_import return HELD.
        return imported

    adapter.open_backbone_import = Mock(side_effect=attach)
    loader_cls = load_class_methods(
        "vllm_omni/diffusion/model_loader/diffusers_loader.py",
        "DiffusersPipelineLoader",
        ["load_model", "_get_shared_weight_group", "_load_model_with_shared_backbone"],
        {
            "torch": torch,
            "set_default_torch_dtype": lambda _: nullcontext(),
            **{name: offload_config[name] for name in ("resolve_offload", "OffloadStrategy", "DIT_COMPONENT")},
        },
    )
    loader = loader_cls()
    loader.od_config = SimpleNamespace(weight_sharing=group, dtype="float32")
    loader.quant_config = None
    loader.parallel_config = SimpleNamespace(use_hsdp=False)
    loader._init_from_load_format = Mock(return_value=model)
    loader.load_weights = Mock()
    loader._process_weights_after_loading = Mock()
    loader._attach_offload_startup_state = Mock()
    namespace = {
        "resolve_prompt_embed_cache_config": lambda **_: (False, 0),
        "resolve_offload": offload_config["resolve_offload"],
        "offload_enabled": offload_config["offload_enabled"],
        "current_omni_platform": platform,
        "LoadConfig": Mock(),
        "DiffusersPipelineLoader": Mock(return_value=loader),
        "time": time,
        "nullcontext": nullcontext,
        "DeviceMemoryProfiler": nullcontext,
    }
    runner_cls = load_class_methods(
        "vllm_omni/diffusion/worker/diffusion_model_runner.py",
        "DiffusionModelRunner",
        ["load_model"],
        namespace,
    )
    runner = runner_cls()
    runner.od_config = loader.od_config
    runner.vllm_config = SimpleNamespace()
    runner.device = "cuda:3"
    return SimpleNamespace(
        runner=runner, loader=loader, model=model, adapter=adapter, backend=backend, codec=codec, factory=factory
    )


@pytest.mark.parametrize("failure", ["preflight", "partial_import", "private_weights", "postprocess"])
def test_load_failure_retains_backend_and_blocks_retry(consumer_load, contracts, failure):
    r = consumer_load
    if failure == "preflight":
        r.adapter.open_backbone_import.side_effect = LoadError("preflight failed")
    elif failure == "partial_import":
        r.codec.failures["import"] = (2, RuntimeError("import failed"))
    elif failure == "private_weights":
        r.loader.load_weights.side_effect = LoadError("private weights failed")
    else:
        r.loader._process_weights_after_loading.side_effect = LoadError("postprocess failed")

    with pytest.raises((LoadError, contracts.WeightSharingError)):
        r.runner.load_model()
    assert not hasattr(r.runner, "pipeline")
    assert r.runner._weight_share_backend is r.backend
    assert r.runner._weight_share_session_id == "session"
    if failure == "preflight":
        with pytest.raises(contracts.WeightSharingError, match="unknown session"):
            r.backend.session_status("session")
    else:
        snapshot = r.backend.session_status("session")
        assert snapshot.state == contracts.SessionState.POISONED
        if failure == "partial_import":
            assert snapshot.imported_storage_ids == ("storage-0",)
            assert snapshot.active_storage_id == "storage-1"
        else:
            assert all(not lease.expired() for lease in r.codec.leases)

    calls = list(r.codec.calls)
    for load_format in ("default", "dummy"):
        with pytest.raises(RuntimeError, match="already attempted"):
            r.runner.load_model(load_format=load_format)
    assert r.codec.calls == calls
    r.factory.assert_called_once_with()
    r.adapter.open_backbone_import.assert_called_once()


@pytest.mark.parametrize("load_format", ["default", None])
@pytest.mark.parametrize("device", ["cuda:3", "cuda"])
def test_pipeline_return_keeps_same_backend_on_runner(consumer_load, load_format, device):
    r = consumer_load
    r.runner.device = device
    r.model.requires_early_lora_init = True
    # Stop after pipeline assignment, before unrelated compilation/cache setup.
    with pytest.raises(LoadError, match="after load"):
        r.runner.load_model(load_format=load_format, initialize_lora=Mock(side_effect=LoadError("after load")))
    assert r.loader._init_from_load_format.call_args.args[0] == "default"
    assert r.runner.pipeline is r.model
    assert r.model._weight_share_backend is r.runner._weight_share_backend is r.backend
    assert r.model._weight_share_import.session_id == r.runner._weight_share_session_id
    r.model.get_weight_sharing_adapter.assert_called_once_with()
    r.adapter.shared_param_full_names.assert_called_once_with(r.model)
    assert r.loader.load_weights.call_args.kwargs["weight_name_filter"] is r.adapter.is_shared_checkpoint_name
    r.factory.assert_called_once_with()


@pytest.mark.parametrize("load_format", ["diffusers", "custom_pipeline", "dummy", "unknown"])
def test_shared_backbone_rejects_load_format_before_construction(consumer_load, load_format):
    r = consumer_load
    r.loader._init_from_load_format.side_effect = AssertionError("must reject before model construction")
    with pytest.raises(ValueError, match="shared-backbone loading requires load_format='default'"):
        r.runner.load_model(load_format=load_format)
    r.factory.assert_not_called()
    r.loader._init_from_load_format.assert_not_called()
    r.adapter.open_backbone_import.assert_not_called()
    r.loader.load_weights.assert_not_called()
    assert not r.codec.imports


def test_dummy_without_sharing_skips_loading(consumer_load):
    r = consumer_load
    r.runner.od_config.weight_sharing = None
    r.loader.load_model = Mock()
    r.runner.load_model(load_format="dummy")
    r.loader.load_model.assert_not_called()
    r.factory.assert_not_called()
    assert not hasattr(r.runner, "pipeline")


def test_loader_requires_caller_owned_backend(consumer_load):
    r = consumer_load
    with pytest.raises(ValueError, match="runner-owned"):
        r.loader.load_model("cuda:0")
    r.loader._init_from_load_format.assert_not_called()
    r.factory.assert_not_called()


@pytest.mark.parametrize("offload", [False, True])
def test_disabled_sharing_does_not_create_backend(consumer_load, offload):
    r = consumer_load
    r.runner.od_config.weight_sharing = None
    r.runner.od_config.enable_layerwise_offload = offload
    r.loader.load_model = Mock(side_effect=LoadError("ordinary load"))
    with pytest.raises(LoadError, match="ordinary load"):
        r.runner.load_model()
    assert r.loader.load_model.call_args.kwargs.get("weight_sharing_backend") is None
    assert not hasattr(r.runner, "_weight_share_backend")
    r.factory.assert_not_called()


@pytest.mark.parametrize("declared", [False, None])
def test_unsupported_postload_is_rejected_before_import(consumer_load, monkeypatch, declared):
    r = consumer_load
    method = SimpleNamespace()
    if declared is not None:
        method.supports_pre_processed_weights = declared
    r.model.named_parameters = lambda: [("shared.weight", object())]
    r.model.named_modules = lambda: [("shared", SimpleNamespace(quant_method=method))]
    r.adapter.shared_param_full_names.return_value = ["shared.weight"]
    monkeypatch.setitem(r.loader._load_model_with_shared_backbone.__globals__, "QuantizeMethodBase", SimpleNamespace)
    with pytest.raises(ValueError, match="does not support preprocessed shared weights"):
        r.runner.load_model()
    r.adapter.open_backbone_import.assert_not_called()
    assert not r.codec.imports


def test_missing_adapter_is_rejected_before_import(consumer_load):
    r = consumer_load
    del r.model.get_weight_sharing_adapter
    with pytest.raises(ValueError, match="get_weight_sharing_adapter"):
        r.runner.load_model()
    r.adapter.open_backbone_import.assert_not_called()
    r.loader.load_weights.assert_not_called()
    assert not r.codec.imports


@pytest.mark.parametrize(
    "offload_settings",
    [
        {"enable_cpu_offload": True},
        {"enable_layerwise_offload": True},
        {"enable_distributed_layerwise_offload": True},
        {"diffusion_offload_config": {"mode": "module", "components": ["dit"]}},
        {"diffusion_offload_config": {"mode": "layer", "components": ["dit", "text_encoder"]}},
        {
            "diffusion_offload_config": {
                "mode": "layer",
                "components": ["dit"],
                "layer_options": {"dit": {"weight_transfer": "allgather"}},
            }
        },
    ],
)
def test_shared_backbone_rejects_offload_before_construction(consumer_load, offload_settings):
    r = consumer_load
    vars(r.loader.od_config).update(offload_settings)
    r.loader._init_from_load_format.side_effect = AssertionError("must reject before model construction")
    with pytest.raises(ValueError, match="shared-backbone loading does not support DiT offload"):
        r.runner.load_model()
    r.loader._init_from_load_format.assert_not_called()
    r.adapter.open_backbone_import.assert_not_called()
    assert not r.codec.imports


@pytest.mark.parametrize("mode", ["module", "layer"])
def test_shared_backbone_allows_private_encoder_offload(consumer_load, mode):
    r = consumer_load
    r.loader.od_config.diffusion_offload_config = {"mode": mode, "components": ["text_encoder"]}
    r.model.requires_early_lora_init = True
    with pytest.raises(LoadError, match="after load"):
        r.runner.load_model(initialize_lora=Mock(side_effect=LoadError("after load")))
    assert r.runner.pipeline is r.model
    r.adapter.open_backbone_import.assert_called_once()


@pytest.mark.parametrize("device", ["cuda:0", "xpu:3", "xpu"])
def test_device_mismatch_is_rejected_before_construction(consumer_load, device):
    r = consumer_load
    r.runner.device = device
    with pytest.raises(ValueError, match="does not match current device cuda:3"):
        r.runner.load_model()
    r.loader._init_from_load_format.assert_not_called()
    r.adapter.open_backbone_import.assert_not_called()
    assert not r.codec.imports
