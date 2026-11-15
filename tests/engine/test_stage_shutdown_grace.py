# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from unittest.mock import Mock

import pytest

from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClientBase

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class RecordingTransport:
    def shutdown(self, timeout=None):
        self.received_timeout = timeout


class Client(StageEngineCoreClientBase, RecordingTransport):
    def __init__(self, configured):
        self.vllm_config = Mock(shutdown_timeout=0)
        self._stage_shutdown_timeout = configured


@pytest.mark.parametrize(
    ("configured", "explicit", "expected"),
    [(30, None, 30), (30, 0, 0), (30, 5, 5), (0, None, 0), (None, None, None)],
)
def test_configured_grace_reaches_transport(configured, explicit, expected):
    client = Client(configured)
    client.shutdown(timeout=explicit)
    assert client.received_timeout == expected


def test_runtime_grace_validation():
    from vllm_omni.config.omni_config import OmniStageRuntimeConfig

    assert OmniStageRuntimeConfig(shutdown_timeout=30).shutdown_timeout == 30
    with pytest.raises(ValueError):
        OmniStageRuntimeConfig(shutdown_timeout=-1)
