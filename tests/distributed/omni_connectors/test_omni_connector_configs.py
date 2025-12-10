# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest

# Use the new import path for initialization utilities
from vllm_omni.distributed.omni_connectors.utils.initialization import load_omni_transfer_config


def get_config_files():
    """Helper to find config files."""
    # Go up two levels from 'tests/distributed/omni_connectors' (approx) to 'vllm-omni' root
    # Adjust based on file location: vllm-omni/tests/distributed/omni_connectors/test_omni_connector_configs.py
    # This file is 4 levels deep from root if we count from tests?
    # vllm-omni/tests/distributed/omni_connectors -> parent -> distributed -> parent -> tests -> parent -> vllm-omni
    # Let's use resolve to be safe.

    # Path(__file__) = .../vllm-omni/tests/distributed/omni_connectors/test_omni_connector_configs.py
    # .parent = omni_connectors
    # .parent = distributed
    # .parent = tests
    # .parent = vllm-omni

    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    config_dir = base_dir / "vllm_omni" / "model_executor" / "stage_configs"

    if not config_dir.exists():
        return []

    return list(config_dir.glob("qwen*.yaml"))


# Collect files at module level for parametrization
config_files = get_config_files()


@pytest.mark.skipif(len(config_files) == 0, reason="No config files found or directory missing")
@pytest.mark.parametrize("yaml_file", config_files, ids=lambda p: p.name)
def test_load_qwen_yaml_configs(yaml_file):
    """
    Scan and test loading of all qwen*.yaml config files.
    This ensures that existing stage configs are compatible with the OmniConnector system.
    """
    print(f"Testing config load: {yaml_file.name}")
    try:
        # Attempt to load the config
        # default_shm_threshold doesn't matter much for loading correctness, using default
        config = load_omni_transfer_config(yaml_file)

        assert config is not None, "Config should not be None"

        # Basic validation
        # Note: Some configs might not have 'runtime' or 'connectors' section if they rely on auto-shm
        # but the load function should succeed regardless.

        # If the config defines stages, we expect connectors to be populated (either explicit or auto SHM)
        # We can't strictly assert len(config.connectors) > 0 because a single stage pipeline might have 0 edges.

        print(f"  -> Successfully loaded. Connectors: {len(config.connectors)}")

    except Exception as e:
        pytest.fail(f"Failed to load config {yaml_file.name}: {e}")
