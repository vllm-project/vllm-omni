# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Unit tests for OmniEngineArgs._map_offload_config() LMCache support.

Tests the config bridge that maps omni_kv_config YAML surface to vLLM's
KV transfer infrastructure (LMCacheConnectorV1).
"""

import os

import pytest

from vllm_omni.engine.arg_utils import (
    OmniEngineArgs,
    _build_lmcache_connector_config,
    _map_offload_config,
    _set_lmcache_env,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestBuildLmcacheConnectorConfig:
    """Test _build_lmcache_connector_config() helper."""

    def test_basic_config(self):
        lmcache_config = {"config_file": "/tmp/lmcache.yaml"}
        entry = _build_lmcache_connector_config(lmcache_config)

        assert entry["kv_connector"] == "LMCacheConnectorV1"
        assert entry["kv_role"] == "kv_both"
        assert entry["kv_connector_extra_config"]["lmcache.config_file"] == "/tmp/lmcache.yaml"

    def test_multiple_fields(self):
        lmcache_config = {"config_file": "/tmp/lmcache.yaml", "chunk_size": 256}
        entry = _build_lmcache_connector_config(lmcache_config)

        assert entry["kv_connector_extra_config"]["lmcache.config_file"] == "/tmp/lmcache.yaml"
        assert entry["kv_connector_extra_config"]["lmcache.chunk_size"] == 256

    def test_already_prefixed_keys(self):
        lmcache_config = {"lmcache.chunk_size": 256}
        entry = _build_lmcache_connector_config(lmcache_config)

        assert entry["kv_connector_extra_config"]["lmcache.chunk_size"] == 256

    def test_original_config_not_mutated(self):
        lmcache_config = {"config_file": "/tmp/lmcache.yaml"}
        entry = _build_lmcache_connector_config(lmcache_config)
        entry["kv_connector_extra_config"]["extra_key"] = "value"
        assert "extra_key" not in lmcache_config


class TestMapOffloadConfig:
    """Test the full _map_offload_config()."""

    def _make_args_with_config(self, omni_kv_config):
        args = object.__new__(OmniEngineArgs)
        args.omni_kv_config = omni_kv_config
        args.kv_offloading_size = None
        args.kv_transfer_config = None
        args.disable_hybrid_kv_cache_manager = False
        return args

    def test_lmcache_only_sets_kv_transfer_config(self):
        args = self._make_args_with_config(
            {
                "kv_store_config": {
                    "lmcache_config": {
                        "config_file": "/tmp/lmcache.yaml",
                    }
                }
            }
        )

        try:
            _map_offload_config(args)
        except ImportError:
            pytest.skip("vLLM not installed, cannot import KVTransferConfig")

        assert args.kv_transfer_config is not None
        assert args.kv_transfer_config.kv_connector == "LMCacheConnectorV1"
        assert args.kv_transfer_config.kv_connector_extra_config["lmcache.config_file"] == "/tmp/lmcache.yaml"
        assert args.kv_transfer_config.kv_role == "kv_both"

    def test_no_config_is_noop(self):
        args = self._make_args_with_config(None)
        _map_offload_config(args)
        assert args.kv_offloading_size is None
        assert args.kv_transfer_config is None

    def test_empty_kv_store_config_is_noop(self):
        args = self._make_args_with_config({"kv_store_config": {}})
        _map_offload_config(args)
        assert args.kv_offloading_size is None
        assert args.kv_transfer_config is None

    def test_lmcache_config_with_multiple_fields(self):
        args = self._make_args_with_config(
            {
                "kv_store_config": {
                    "lmcache_config": {
                        "config_file": "/tmp/lmcache.yaml",
                        "chunk_size": 256,
                        "max_local_cache_size": "10GiB",
                    }
                }
            }
        )

        try:
            _map_offload_config(args)
        except ImportError:
            pytest.skip("vLLM not installed, cannot import KVTransferConfig")

        cfg = args.kv_transfer_config
        assert cfg.kv_connector == "LMCacheConnectorV1"
        assert cfg.kv_connector_extra_config["lmcache.config_file"] == "/tmp/lmcache.yaml"
        assert cfg.kv_connector_extra_config["lmcache.chunk_size"] == 256
        assert cfg.kv_connector_extra_config["lmcache.max_local_cache_size"] == "10GiB"

    def test_custom_kv_role(self):
        args = self._make_args_with_config(
            {
                "kv_store_config": {
                    "kv_role": "kv_producer",
                    "lmcache_config": {"config_file": "/tmp/lmcache.yaml"},
                }
            }
        )

        try:
            _map_offload_config(args)
        except ImportError:
            pytest.skip("vLLM not installed, cannot import KVTransferConfig")

        assert args.kv_transfer_config.kv_role == "kv_producer"

    def test_lmcache_yaml_config_format(self):
        saved = os.environ.get("LMCACHE_CONFIG_FILE")
        try:
            args = self._make_args_with_config(
                {
                    "kv_store_config": {
                        "lmcache_config": {
                            "config_file": "/tmp/mylmcache.yaml",
                            "chunk_size": 256,
                        }
                    }
                }
            )
            try:
                _map_offload_config(args)
            except ImportError:
                pytest.skip("vLLM not installed, cannot import KVTransferConfig")
            _set_lmcache_env(args)

            extra = args.kv_transfer_config.kv_connector_extra_config
            assert extra.get("lmcache.config_file") == "/tmp/mylmcache.yaml"
            assert extra.get("lmcache.chunk_size") == 256
            assert os.environ.get("LMCACHE_CONFIG_FILE") == "/tmp/mylmcache.yaml"
        finally:
            if saved is not None:
                os.environ["LMCACHE_CONFIG_FILE"] = saved
            elif "LMCACHE_CONFIG_FILE" in os.environ:
                os.environ.pop("LMCACHE_CONFIG_FILE")
