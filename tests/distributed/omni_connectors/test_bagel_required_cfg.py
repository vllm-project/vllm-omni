# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Required Bagel CFG receive contract (CPU, single-rank boundary).

A missing required branch must fail before any rank enters KV distribution.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.distributed.omni_connectors.kv_transfer_manager import (
    KVCacheReceiveError,
    OmniKVCacheConfig,
    OmniKVTransferManager,
    ReceiveRole,
    _TransferTopoConfig,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.stage_input_processors.bagel import collect_cfg_kv_caches

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.cache]


def _receive(monkeypatch, required_roles, available_roles, *, receive_error=False):
    manager = OmniKVTransferManager(OmniKVCacheConfig(need_recv_cache=True))
    manager._topo_config = _TransferTopoConfig(ReceiveRole.LOCAL, False, 1, 0, None, 1, 0, None, None)
    params = OmniDiffusionSamplingParams(
        seed=0,
        past_key_values=None,
        cfg_kv_request_ids={role: f"R__{role}" for role in required_roles},
    )
    request = OmniDiffusionRequest(prompt="test", request_id="R", sampling_params=params)
    primary = SimpleNamespace(key_cache={}, value_cache={})

    def receive_primary(req, target_device):
        req.sampling_params.past_key_values = primary
        return True

    def receive_companion(request_id, target_device):
        if receive_error:
            raise OSError("injected connector failure")
        if request_id.removeprefix("R__") not in available_roles:
            return None, 0  # timeout outcome; no real sleeping
        # An explicitly received empty branch is valid, unlike absent payload.
        return {"layer_blocks": {"key_cache": {}, "value_cache": {}}}, 1

    monkeypatch.setattr(manager, "receive_kv_cache", receive_primary)
    monkeypatch.setattr(manager, "receive_kv_cache_for_request", receive_companion)
    monkeypatch.setattr(manager, "_resolve_request_id", lambda req: req.request_id)
    # Single-rank test: collective mechanics are deliberately excluded.
    monkeypatch.setattr(manager, "distribute_kv_cache", Mock(return_value=None))
    monkeypatch.setattr(manager, "_apply_request_kv_payload", Mock())
    result = manager.receive_multi_kv_cache_distributed(request, collect_cfg_kv_caches)
    return result, params


@pytest.mark.parametrize(
    ("required", "available", "missing"),
    [
        (("cfg_text",), (), "cfg_text"),
        (("cfg_text", "cfg_img"), ("cfg_text",), "cfg_img"),
        (("cfg_text", "cfg_img"), ("cfg_img",), "cfg_text"),
    ],
    ids=["negative-text-timeout", "image-branch-timeout", "text-branch-timeout"],
)
def test_primary_success_does_not_hide_required_cfg_timeout(monkeypatch, required, available, missing):
    with pytest.raises(KVCacheReceiveError, match=missing):
        _receive(monkeypatch, required, available)


def test_companion_connector_exception_ends_request(monkeypatch):
    with pytest.raises(KVCacheReceiveError, match="cfg_text"):
        _receive(monkeypatch, ("cfg_text",), (), receive_error=True)


@pytest.mark.parametrize("roles", [(), ("cfg_text",), ("cfg_text", "cfg_img")])
def test_unrequested_or_received_empty_cfg_is_valid(monkeypatch, roles):
    received, params = _receive(monkeypatch, roles, roles)
    assert received is True
    assert params.past_key_values is not None
    for role in roles:
        assert getattr(params, f"{role}_past_key_values") is not None


@pytest.mark.parametrize("samples", [1, 2])
@pytest.mark.parametrize("missing", [False, True])
def test_upstream_sample_receive_preserved(samples, missing):
    from vllm_omni.distributed.omni_connectors.utils.kv_utils import PER_SAMPLE_KV_COUNT_KEY

    m = OmniKVTransferManager(OmniKVCacheConfig(need_recv_cache=True))
    m._topo_config = _TransferTopoConfig(ReceiveRole.LOCAL, False, 1, 0, None, 1, 0, None, None)
    p = OmniDiffusionSamplingParams(
        seed=0, extra_args={PER_SAMPLE_KV_COUNT_KEY: samples}, cfg_kv_request_ids={"cfg_text": "R__cfg_text"}
    )
    r = OmniDiffusionRequest(prompt="test", request_id="R", sampling_params=p)

    def fetch(rid, device=None, **kwargs):
        if missing and rid == "R__cfg_text":
            return None, 0
        return {"layer_blocks": {"key_cache": [], "value_cache": []}, "metadata": {"owner": rid}}, 1

    m.receive_kv_cache_for_request = Mock(side_effect=fetch)
    if missing:
        with pytest.raises(KVCacheReceiveError, match="cfg_text"):
            m.receive_multi_kv_cache_distributed(r, collect_cfg_kv_caches)
        assert p.past_key_values is None
    else:
        assert m.receive_multi_kv_cache_distributed(r, collect_cfg_kv_caches)
        if samples == 2:
            assert len(p.past_key_values) == 2
            assert [x["owner"] for x in p.kv_metadata] == ["R-sample0", "R-sample1"]
    expected = ["R"] if samples == 1 else ["R-sample0", "R-sample1"]
    assert [c.args[0] for c in m.receive_kv_cache_for_request.call_args_list] == expected + ["R__cfg_text"]


def test_required_cfg_error_survives_single_target_rank():
    """Source TP can be active while the receiving stage has only one rank."""
    manager = OmniKVTransferManager(OmniKVCacheConfig(need_recv_cache=True))
    manager._topo_config = _TransferTopoConfig(
        ReceiveRole.LOCAL, True, 1, 0, None, 1, 0, None, None, Mock(world_size=1), 1
    )
    req = OmniDiffusionRequest(prompt="test", request_id="R", sampling_params=OmniDiffusionSamplingParams(seed=0))
    manager.receive_multi_kv_cache = Mock(side_effect=OSError("injected connector failure"))
    with pytest.raises(KVCacheReceiveError, match="injected connector failure"):
        manager.receive_multi_kv_cache_distributed(req)
