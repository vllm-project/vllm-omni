# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for MooncakeStoreConnector config handling.

These tests do NOT require a running mooncake master/store: the
module-level MooncakeDistributedStore / ReplicateConfig symbols are
replaced with fakes whose setup() succeeds, so only the connector's own
config parsing (stage_id, host env expansion) is exercised.
"""

import pytest

from vllm_omni.distributed.omni_connectors.connectors import mooncake_store_connector as msc
from vllm_omni.distributed.omni_connectors.utils.env import EnvVarExpansionError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeReplicateConfig:
    with_soft_pin = False


class _FakeStore:
    """Records the setup() args so tests can assert what the connector passed."""

    def __init__(self):
        self.setup_args = None

    def setup(self, host, metadata, segment, localbuf, proto, rdma, master):
        self.setup_args = {
            "host": host,
            "metadata": metadata,
            "segment": segment,
            "localbuf": localbuf,
            "proto": proto,
            "rdma": rdma,
            "master": master,
        }
        return 0


@pytest.fixture()
def fakes(monkeypatch):
    """Install fake mooncake symbols regardless of whether mooncake is installed."""
    created = []

    def make_store():
        store = _FakeStore()
        created.append(store)
        return store

    monkeypatch.setattr(msc, "MooncakeDistributedStore", make_store)
    monkeypatch.setattr(msc, "ReplicateConfig", _FakeReplicateConfig)
    return created


def make_connector(fakes, **config):
    connector = msc.MooncakeStoreConnector(config)
    assert len(fakes) == 1
    return connector, fakes[0]


class TestStageId:
    def test_stage_id_from_config(self, fakes):
        connector, _ = make_connector(fakes, stage_id=2)
        assert connector.stage_id == 2

    def test_stage_id_defaults_to_minus_one(self, fakes):
        # ChunkTransferAdapter reads connector.stage_id whenever async_chunk
        # is on; the default must exist (not raise AttributeError) even when
        # the deploy config omits it.
        connector, _ = make_connector(fakes)
        assert connector.stage_id == -1


class TestHostExpansion:
    def test_host_env_var_expanded(self, fakes, monkeypatch):
        monkeypatch.setenv("MC_STORE_HOST", "10.2.1.42")
        connector, store = make_connector(fakes, host="$MC_STORE_HOST")
        # The expanded value must reach the transfer engine: the listen
        # address handed to setup() is what other stage pods dial.
        assert connector.host == "10.2.1.42"
        assert store.setup_args["host"] == "10.2.1.42"

    def test_host_env_var_embedded_in_string(self, fakes, monkeypatch):
        monkeypatch.setenv("MC_STORE_HOST_SUFFIX", "42")
        connector, store = make_connector(fakes, host="10.2.1.$MC_STORE_HOST_SUFFIX")
        assert connector.host == "10.2.1.42"
        assert store.setup_args["host"] == "10.2.1.42"

    def test_host_braced_env_var_expanded(self, fakes, monkeypatch):
        monkeypatch.setenv("MC_STORE_HOST", "10.2.1.43")
        connector, _ = make_connector(fakes, host="${MC_STORE_HOST}")
        assert connector.host == "10.2.1.43"

    def test_literal_host_untouched(self, fakes):
        connector, store = make_connector(fakes, host="10.2.1.44")
        assert connector.host == "10.2.1.44"
        assert store.setup_args["host"] == "10.2.1.44"

    def test_host_default(self, fakes):
        connector, store = make_connector(fakes)
        assert connector.host == "127.0.0.1"
        assert store.setup_args["host"] == "127.0.0.1"

    def test_unset_env_var_raises_with_var_name(self, fakes, monkeypatch):
        monkeypatch.delenv("MC_STORE_HOST", raising=False)
        # A typo'd/missing env var must fail at connector init with the
        # variable named, not leak "$MC_STORE_HOST" into the transfer
        # engine as a literal hostname.
        with pytest.raises(EnvVarExpansionError, match="MC_STORE_HOST"):
            make_connector(fakes, host="$MC_STORE_HOST")
