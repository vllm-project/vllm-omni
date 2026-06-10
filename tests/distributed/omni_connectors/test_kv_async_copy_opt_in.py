"""Async KV H2D copy opt-in contract.

The async pinned-pool -> GPU copy is only safe for callers that wait via
``wait_kv_copy()`` before consuming the KV (the diffusion runner).  These tests
lock the invariants:

* ``from_od_config`` honors ``OmniDiffusionConfig.enable_kv_async_copy`` and
  defaults to the synchronous path when the flag is absent.
* ``from_vllm_config`` (the AR receive path, which never waits) always stays on
  the synchronous path.
* Direct construction defaults to synchronous.
"""

from types import SimpleNamespace

import pytest

from vllm_omni.distributed.omni_connectors.kv_transfer_manager import (
    OmniKVCacheConfig,
    OmniKVTransferManager,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_direct_construction_defaults_to_sync():
    mgr = OmniKVTransferManager(OmniKVCacheConfig())
    assert mgr._async_kv_copy is False


def test_from_od_config_enables_async_when_flag_set():
    od_config = SimpleNamespace(omni_kv_config=None, enable_kv_async_copy=True)
    mgr = OmniKVTransferManager.from_od_config(od_config)
    assert mgr._async_kv_copy is True


def test_from_od_config_stays_sync_when_flag_unset():
    od_config = SimpleNamespace(omni_kv_config=None, enable_kv_async_copy=False)
    mgr = OmniKVTransferManager.from_od_config(od_config)
    assert mgr._async_kv_copy is False


def test_from_od_config_defaults_sync_when_flag_missing():
    # Older config objects may not carry the field at all.
    od_config = SimpleNamespace(omni_kv_config=None)
    mgr = OmniKVTransferManager.from_od_config(od_config)
    assert mgr._async_kv_copy is False


def test_from_od_config_preserves_async_with_connector_config():
    od_config = SimpleNamespace(
        omni_kv_config={"connector_config": {"type": "mock"}, "need_recv_cache": True},
        enable_kv_async_copy=True,
    )
    mgr = OmniKVTransferManager.from_od_config(od_config)
    assert mgr._async_kv_copy is True


def test_from_vllm_config_ar_path_is_always_sync():
    # AR receivers never call wait_kv_copy(); the async path must never engage
    # for them regardless of any config.
    model_config = SimpleNamespace(omni_kv_config=None)
    vllm_config = SimpleNamespace(kv_transfer_config=None)
    mgr = OmniKVTransferManager.from_vllm_config(vllm_config, model_config)
    assert mgr._async_kv_copy is False
