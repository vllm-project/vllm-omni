# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise the provider runner hook with the real backend and CPU codec."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.distributed.weight_sharing.helpers import FakeCodec, load_class_methods, make_context, make_weights

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def provider_runner(weight_sharing, contracts, monkeypatch):
    runner_cls = load_class_methods(
        "vllm_omni/worker_v2/omni_model_runner.py",
        "OmniGPUModelRunner",
        ["_publish_shared_backbone_if_configured"],
        {"logger": Mock()},
    )

    codec = FakeCodec(contracts)
    backend = weight_sharing.TorchIpcBackend(codec=codec)
    factory = Mock(return_value=backend)
    monkeypatch.setitem(sys.modules, "vllm_omni.distributed.weight_sharing", weight_sharing)
    monkeypatch.setattr(weight_sharing, "TorchIpcBackend", factory)
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.platforms",
        SimpleNamespace(current_omni_platform=SimpleNamespace(current_device=lambda: 3)),
    )
    context = make_context(contracts)

    def publish(backend, model, *, group, device_index):
        assert model is runner.model
        assert device_index == 3  # Current device, not runner.device.index or TP rank.
        bindings, spec = make_weights(contracts, two_storages=True)
        return backend.prepare_export(bindings, spec, context)

    runner = runner_cls()
    runner.vllm_config = SimpleNamespace(
        weight_sharing=SimpleNamespace(role="provider", session_id=context.identity.session_id, group_id="group")
    )
    runner.device = SimpleNamespace(index=0)
    runner.adapter = SimpleNamespace(publish_backbone=Mock(side_effect=publish))
    runner.model = SimpleNamespace(get_weight_sharing_adapter=Mock(return_value=runner.adapter))
    runner.publish = runner._publish_shared_backbone_if_configured
    return runner, backend, codec, factory


@pytest.mark.parametrize("failure", ["before_prepare", "second_storage"])
def test_failed_publish_retains_owner_and_rejects_retry(provider_runner, contracts, failure):
    runner, backend, codec, factory = provider_runner
    if failure == "before_prepare":
        runner.adapter.publish_backbone.side_effect = RuntimeError("adapter failed")
        error = RuntimeError
    else:
        codec.failures["export"] = (2, RuntimeError("reducer failed"))
        error = contracts.WeightSharingError

    with pytest.raises(error):
        runner.publish()

    assert runner._weight_share_backend is backend
    assert runner._weight_share_session_id == "session"
    assert getattr(runner, "_weight_share_session", None) is None
    if failure == "second_storage":
        status = backend.session_status(runner._weight_share_session_id)
        assert status.state == contracts.SessionState.POISONED
        assert status.exported_storage_ids == ("storage-0",)
        assert status.active_storage_id == "storage-1"
    else:
        with pytest.raises(contracts.WeightSharingError, match="unknown session"):
            backend.session_status(runner._weight_share_session_id)

    calls = list(codec.calls)
    with pytest.raises(RuntimeError, match="previous weight-sharing publish attempt"):
        runner.publish()
    factory.assert_called_once_with()
    runner.adapter.publish_backbone.assert_called_once()
    assert codec.calls == calls
    assert not codec.cancellations


def test_successful_publish_retains_owner_before_export_and_is_idempotent(provider_runner, contracts):
    runner, backend, codec, factory = provider_runner

    def check_owner():
        assert runner._weight_share_backend is backend
        assert runner._weight_share_session_id == "session"

    codec.before_export = check_owner
    runner.publish()
    session = runner._weight_share_session
    assert backend.session_status(session).state == contracts.SessionState.EXPORTED
    runner.publish()
    factory.assert_called_once_with()
    runner.adapter.publish_backbone.assert_called_once()
    assert runner._weight_share_session is session
    assert len(codec.exports) == 2
    assert backend.release_export(session).status == "CLOSED"


def test_missing_adapter_is_rejected_before_export(provider_runner):
    runner, backend, codec, factory = provider_runner
    del runner.model.get_weight_sharing_adapter
    with pytest.raises(ValueError, match="get_weight_sharing_adapter"):
        runner.publish()
    factory.assert_not_called()
    assert not codec.exports
