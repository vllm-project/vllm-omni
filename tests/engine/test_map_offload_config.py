# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Unit tests for OmniEngineArgs._map_offload_config() LMCache support.

Tests the config bridge that maps omni_kv_config YAML surface to vLLM's
KV transfer infrastructure (LMCacheConnectorV1).
"""

import logging
import os

import pytest

from vllm_omni.distributed.kv_transfer.mooncake.config import MooncakeConfig
from vllm_omni.engine.arg_utils import (
    OmniEngineArgs,
    _build_lmcache_connector_config,
    _map_offload_config,
    _set_lmcache_env,
    _warn_if_hidden_state_pool_undersized,
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


class TestHiddenStatePoolWarning:
    """The warning has to fire on LMCache's own defaults, which are undersized."""

    def test_warns_when_neither_size_is_given(self, caplog):
        with caplog.at_level(logging.WARNING):
            _warn_if_hidden_state_pool_undersized({})

        assert "smaller than max_local_cpu_size" in caplog.text

    def test_warns_when_only_the_kv_size_is_raised(self, caplog):
        with caplog.at_level(logging.WARNING):
            _warn_if_hidden_state_pool_undersized({"lmcache.max_local_cpu_size": 20.0})

        assert "smaller than max_local_cpu_size" in caplog.text

    def test_quiet_when_the_hidden_state_pool_is_large_enough(self, caplog):
        with caplog.at_level(logging.WARNING):
            _warn_if_hidden_state_pool_undersized(
                {
                    "lmcache.max_hidden_state_cpu_size": 8.0,
                    "lmcache.max_local_cpu_size": 5.0,
                }
            )

        assert caplog.text == ""


class TestMooncakeConfig:
    def test_required_identity_and_defaults(self):
        config = MooncakeConfig.from_dict(
            {"config_file": "/tmp/mooncake.json", "namespace": "test", "model_revision": "pinned-revision"}
        )
        assert config.max_hs_bytes == 256 * 1024 * 1024
        assert config.max_pending_transfers == 16
        assert config.load_timeout_seconds == 30.0
        assert MooncakeConfig.from_dict(config.to_dict()) == config

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("namespace", ""),
            ("model_revision", None),
            ("config_file", " "),
            ("max_hs_bytes", 0),
            ("max_hs_bytes", True),
            ("max_pending_transfers", -1),
            ("max_pending_transfers", 1.5),
            ("load_timeout_seconds", float("inf")),
            ("load_timeout_seconds", float("nan")),
            ("load_timeout_seconds", 0),
            ("load_timeout_seconds", True),
            ("load_timeout_seconds", "30"),
            ("max_hs_byte", 1024),
        ],
    )
    def test_invalid_configuration_is_rejected(self, key, value):
        config = {"config_file": "/tmp/mooncake.json", "namespace": "test", "model_revision": "pinned-revision"}
        config[key] = value
        with pytest.raises(ValueError, match="mooncake_config"):
            MooncakeConfig.from_dict(config)

    @pytest.mark.parametrize("missing", ["config_file", "namespace", "model_revision"])
    def test_identity_is_required(self, missing):
        config = {"config_file": "/tmp/mooncake.json", "namespace": "test", "model_revision": "pinned-revision"}
        del config[missing]
        with pytest.raises(ValueError, match=missing):
            MooncakeConfig.from_dict(config)


class TestMapMooncakeConfig:
    @staticmethod
    def _args():
        args = object.__new__(OmniEngineArgs)
        args.omni_kv_config = {
            "kv_store_config": {
                "mooncake_config": {
                    "config_file": "/tmp/mooncake.json",
                    "namespace": "qwen-test",
                    "model_revision": "pinned-revision",
                }
            }
        }
        args.kv_transfer_config = None
        args.enable_prefix_caching = None
        return args

    def test_maps_extension_and_explicit_recovery_policy(self):
        args = self._args()
        _map_offload_config(args)
        config = args.kv_transfer_config
        assert config.kv_connector == "OmniMooncakeStoreConnector"
        assert config.kv_connector_module_path == "vllm_omni.distributed.kv_transfer.mooncake.connector"
        assert config.kv_load_failure_policy == "recompute"
        assert config.kv_role == "kv_both"
        assert args.enable_prefix_caching is False
        extra = config.kv_connector_extra_config["omni_mooncake"]
        assert extra["namespace"] == "qwen-test"
        assert extra["max_hs_bytes"] == 256 * 1024 * 1024
        assert "max_hs_bytes" not in args.omni_kv_config["kv_store_config"]["mooncake_config"]

    @pytest.mark.parametrize("lmcache", [{}, {"config_file": "/tmp/lmcache.yaml"}])
    def test_rejects_two_backends(self, lmcache):
        args = self._args()
        args.omni_kv_config["kv_store_config"]["lmcache_config"] = lmcache
        with pytest.raises(ValueError, match="either"):
            _map_offload_config(args)

    def test_rejects_local_prefix_cache(self):
        args = self._args()
        args.enable_prefix_caching = True
        with pytest.raises(ValueError, match="enable_prefix_caching=false"):
            _map_offload_config(args)

    def test_rejects_explicit_transfer_config(self):
        args = self._args()
        args.kv_transfer_config = {"kv_connector": "OtherConnector"}
        with pytest.raises(ValueError, match="explicit kv_transfer_config"):
            _map_offload_config(args)

    @pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
    def test_requires_read_and_write(self, role):
        args = self._args()
        args.omni_kv_config["kv_store_config"]["kv_role"] = role
        with pytest.raises(ValueError, match="kv_role=kv_both"):
            _map_offload_config(args)

    def test_empty_mooncake_config_is_not_silently_ignored(self):
        args = self._args()
        args.omni_kv_config["kv_store_config"]["mooncake_config"] = {}
        with pytest.raises(ValueError, match="Missing mooncake_config"):
            _map_offload_config(args)
