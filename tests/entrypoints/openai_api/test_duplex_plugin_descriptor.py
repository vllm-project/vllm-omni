# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import pytest

from vllm_omni.engine.duplex.contracts import (
    DUPLEX_CONTRACT_VERSION,
    DuplexPluginDescriptor,
)
from vllm_omni.engine.duplex.control_plane import (
    DuplexControlPlane,
)
from vllm_omni.engine.duplex.messages import DuplexFence
from vllm_omni.engine.duplex.session import (
    DuplexReplayAppend,
    DuplexSessionRuntimeManager,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_plugin_descriptor_rejects_unknown_contract_version() -> None:
    with pytest.raises(ValueError, match="unsupported duplex contract version"):
        DuplexPluginDescriptor(
            contract_version="duplex.capabilities.v0",
            adapter_id="minicpmo45",
            runtime_extension_id="minicpmo45",
            stage_count=3,
        )


def test_plugin_descriptor_requires_complete_binding() -> None:
    with pytest.raises(ValueError, match="requires adapter_id"):
        DuplexPluginDescriptor(
            contract_version=DUPLEX_CONTRACT_VERSION,
            adapter_id="",
            runtime_extension_id="minicpmo45",
            stage_count=3,
        )


def test_coerce_capabilities_preserves_plugin_descriptor() -> None:
    capabilities = DuplexControlPlane.coerce_capabilities(
        {
            "contract_version": DUPLEX_CONTRACT_VERSION,
            "adapter_id": "minicpmo45",
            "runtime_extension_id": "minicpmo45",
            "stage_count": 3,
            "input_modes": ["append_audio_chunk"],
        }
    )

    descriptor = capabilities.plugin_descriptor()
    assert descriptor is not None
    assert descriptor.adapter_id == "minicpmo45"
    assert descriptor.runtime_extension_id == "minicpmo45"
    assert descriptor.stage_count == 3


def test_coerce_capabilities_rejects_partial_plugin_descriptor() -> None:
    with pytest.raises(ValueError, match="must declare adapter_id"):
        DuplexControlPlane.coerce_capabilities(
            {
                "contract_version": DUPLEX_CONTRACT_VERSION,
                "adapter_id": "minicpmo45",
                "input_modes": ["append_audio_chunk"],
            }
        )


def test_context_ledger_is_a_read_only_snapshot_over_session_state() -> None:
    manager = DuplexSessionRuntimeManager()
    fence = DuplexFence("ledger-session")
    session = manager.open_session(fence)
    session.update_scheduler_context(tokens=128, limit=512)
    session.replay_appends.append(
        DuplexReplayAppend(
            operation_id="op-1",
            operation_fingerprint=b"fingerprint",
            prompt={"prompt_token_ids": [1, 2]},
            token_count=2,
            byte_count=32,
        )
    )
    session.replay_token_count = 2
    session.replay_byte_count = 32

    snapshot = session.context_ledger()
    assert snapshot.session_id == "ledger-session"
    assert snapshot.context_tokens == 128
    assert snapshot.context_utilization == 0.25
    assert snapshot.replay_append_count == 1
    assert manager.context_ledgers() == (snapshot,)
