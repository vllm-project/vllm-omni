# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from pytest_mock import MockerFixture

from vllm_omni.distributed.omni_connectors.adapter import (
    compute_talker_prompt_ids_length,
    try_recv_via_connector,
    try_send_via_connector,
)
from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector
from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec, OmniTransferConfig
from vllm_omni.distributed.omni_connectors.utils.initialization import get_connectors_config_for_stage

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def mock_objects(mocker: MockerFixture):
    return {"connector": mocker.MagicMock(), "metrics": mocker.MagicMock(), "queue_fn": mocker.MagicMock()}


def test_send_success(mock_objects):
    """Test try_send_via_connector success path."""
    # Setup
    mock_connector = mock_objects["connector"]
    mock_metrics = mock_objects["metrics"]
    mock_queue_fn = mock_objects["queue_fn"]

    stage_id = 0
    next_stage_id = 1
    req_id = "req_123"
    inputs = {"input_ids": [1, 2, 3]}
    sampling_params = {"temperature": 0.7}
    prompt = "test prompt"

    # Mock connector.put return
    # Returns: (success, size, metadata)
    mock_metadata = {"handle": "xyz"}
    mock_connector.put.return_value = (True, 100, mock_metadata)

    # Execute
    result = try_send_via_connector(
        connector=mock_connector,
        stage_id=stage_id,
        next_stage_id=next_stage_id,
        req_id=req_id,
        next_inputs=inputs,
        sampling_params=sampling_params,
        original_prompt=prompt,
        next_stage_queue_submit_fn=mock_queue_fn,
        metrics=mock_metrics,
    )

    # Verify
    assert result is True

    # 1. Verify connector.put called correctly
    mock_connector.put.assert_called_once()
    args, _ = mock_connector.put.call_args
    assert args[0] == "0"  # from_stage
    assert args[1] == "1"  # to_stage
    assert args[2] == req_id
    # Verify payload structure in put
    payload = args[3]
    assert payload["engine_inputs"] == inputs
    assert payload["sampling_params"] == sampling_params

    # 2. Verify queue notification submitted
    mock_queue_fn.assert_called_once()
    notify_payload = mock_queue_fn.call_args[0][0]
    assert notify_payload["request_id"] == req_id
    assert notify_payload["from_connector"] is True
    assert notify_payload["connector_metadata"] == mock_metadata

    # 3. Verify metrics recorded
    mock_metrics.on_forward.assert_called_once()


def test_send_fail(mock_objects):
    """Test try_send_via_connector when connector fails."""
    mock_connector = mock_objects["connector"]
    mock_metrics = mock_objects["metrics"]
    mock_queue_fn = mock_objects["queue_fn"]

    mock_connector.put.return_value = (False, 0, None)

    result = try_send_via_connector(
        connector=mock_connector,
        stage_id=0,
        next_stage_id=1,
        req_id="req_fail",
        next_inputs={},
        sampling_params={},
        original_prompt="",
        next_stage_queue_submit_fn=mock_queue_fn,
        metrics=mock_metrics,
    )

    assert result is False
    mock_queue_fn.assert_not_called()


def test_recv_success(mock_objects):
    """Test try_recv_via_connector success path."""
    mock_connector = mock_objects["connector"]

    # Setup task received from queue
    task = {
        "request_id": "req_recv",
        "from_connector": True,
        "from_stage": "0",
        "connector_metadata": {"handle": "xyz"},
    }

    # Setup connectors dict
    connectors = {("0", "1"): mock_connector}

    # Mock connector.get return
    expected_data = {"engine_inputs": {"ids": [1]}}
    # get returns: (data_obj, size)
    mock_connector.get.return_value = (expected_data, 50)
    # serialize_obj needed for metrics calculation if size not returned directly
    mock_connector.serialize_obj.return_value = b"bytes"

    # Execute
    # We are stage 1 receiving from stage 0
    inputs, rx_metrics = try_recv_via_connector(task, connectors, stage_id=1)

    # Verify
    assert inputs == expected_data["engine_inputs"]
    assert rx_metrics is not None
    mock_connector.get.assert_called_once_with("0", "1", "req_recv", metadata={"handle": "xyz"})


def test_recv_no_connector():
    """Test recv fails when no connector exists for edge."""
    task = {"request_id": "req_missing", "from_connector": True, "from_stage": "0"}
    connectors = {}  # Empty connectors

    inputs, _ = try_recv_via_connector(task, connectors, stage_id=1)
    assert inputs is None


def test_shm_connector_flow(mocker: MockerFixture):
    """
    Verify the full flow: Send -> Adapter -> Connector -> Adapter -> Recv.
    Using real SharedMemoryConnector (inline mode for simplicity).
    """
    # 1. Setup Connector
    config = {"shm_threshold_bytes": 1024}  # Large threshold to use inline
    connector = SharedMemoryConnector(config)
    connectors_map = {("0", "1"): connector}

    # 2. Setup Data
    stage_id = 0
    next_stage_id = 1
    req_id = "flow_req"
    inputs = {"tokens": [10, 20, 30]}
    sampling_params = {"n": 1}

    # Queue capture mechanism
    queue_capture = []

    def mock_submit(payload):
        queue_capture.append(payload)

    mock_metrics = mocker.MagicMock()

    # 3. Send
    success = try_send_via_connector(
        connector=connector,
        stage_id=stage_id,
        next_stage_id=next_stage_id,
        req_id=req_id,
        next_inputs=inputs,
        sampling_params=sampling_params,
        original_prompt="prompt",
        next_stage_queue_submit_fn=mock_submit,
        metrics=mock_metrics,
    )
    assert success is True
    assert len(queue_capture) == 1

    # 4. Recv
    # The 'task' is what would be popped from the queue
    received_task = queue_capture[0]

    # Verify queue payload contains what we expect
    assert received_task["from_connector"] is True
    assert received_task["from_stage"] == "0"

    # Decode
    decoded_inputs, _ = try_recv_via_connector(received_task, connectors_map, stage_id=1)

    # 5. Verify Data Integrity
    assert decoded_inputs == inputs


def test_get_connectors_for_stage():
    """Test filtering logic for stage config."""
    # Config has edges: 0->1, 1->2
    config = OmniTransferConfig(connectors={("0", "1"): ConnectorSpec(name="C1"), ("1", "2"): ConnectorSpec(name="C2")})

    # Get config for Stage 1
    # Stage 1 receives from 0 (input) and sends to 2 (output)
    # get_connectors_config_for_stage ONLY returns INPUT connectors for the worker to initialize

    stage_config = get_connectors_config_for_stage(config, stage_id=1)

    # Should contain "from_stage_0"
    assert "from_stage_0" in stage_config
    assert stage_config["from_stage_0"]["spec"]["name"] == "C1"

    # Should NOT contain "from_stage_1" or related to output
    assert "from_stage_1" not in stage_config

    # Verify Stage 2
    stage_2_config = get_connectors_config_for_stage(config, stage_id=2)
    assert "from_stage_1" in stage_2_config
    assert stage_2_config["from_stage_1"]["spec"]["name"] == "C2"


def test_recv_with_missing_metadata(mocker: MockerFixture):
    """Test recv when queue payload is malformed (missing metadata)."""
    # Connector expects metadata but task doesn't have it
    task = {
        "request_id": "req_bad",
        "from_connector": True,
        "from_stage": "0",
        # Missing "connector_metadata"
    }
    mock_conn = mocker.MagicMock()
    # If get is called with None metadata, connector usually handles it or adapter handles exception
    mock_conn.get.side_effect = Exception("Get failed")

    connectors = {("0", "1"): mock_conn}

    inputs, _ = try_recv_via_connector(task, connectors, stage_id=1)
    assert inputs is None


# Qwen3-Omni chat-template token IDs used by compute_talker_prompt_ids_length.
_IM_START = 151644
_SYSTEM = 8948
_USER = 872
_ASSISTANT = 77091
_AUDIO_PAD = 151675
_IMAGE_PAD = 151655
_VIDEO_PAD = 151656
_FILLER = 1  # any non-special id


def _block(role: int, *body: int) -> list[int]:
    """Build a Qwen3 turn block: <|im_start|> <role> <body...>."""
    return [_IM_START, role, *body]


class TestComputeTalkerPromptIdsLength:
    """``compute_talker_prompt_ids_length`` must mirror the segments that
    ``_thinker_to_talker_prefill`` actually appends to the talker input."""

    def test_empty(self) -> None:
        assert compute_talker_prompt_ids_length([]) == 0

    def test_system_block_skipped(self) -> None:
        # System-only — system blocks are skipped by the talker prefill.
        prompt = _block(_SYSTEM, _FILLER, _FILLER)
        assert compute_talker_prompt_ids_length(prompt) == 0

    def test_audio_user_block_counted(self) -> None:
        # User block containing an audio token contributes its full length.
        prompt = _block(_USER, _AUDIO_PAD, _FILLER, _FILLER)
        assert compute_talker_prompt_ids_length(prompt) == len(prompt)

    def test_image_user_block_counted(self) -> None:
        prompt = _block(_USER, _IMAGE_PAD, _FILLER)
        assert compute_talker_prompt_ids_length(prompt) == len(prompt)

    def test_video_user_block_counted(self) -> None:
        prompt = _block(_USER, _VIDEO_PAD, _FILLER)
        assert compute_talker_prompt_ids_length(prompt) == len(prompt)

    def test_text_only_user_block_skipped(self) -> None:
        # Text-only user blocks (e.g. STT transcripts, tool responses) are
        # skipped by the prefill — counting them oversizes the talker input
        # and hangs stage-1 waiting for tokens that never arrive.
        prompt = _block(_USER, _FILLER, _FILLER, _FILLER)
        assert compute_talker_prompt_ids_length(prompt) == 0

    def test_last_assistant_block_adds_nine(self) -> None:
        # The last assistant block contributes a fixed 9-token codec preamble.
        prompt = _block(_ASSISTANT, _FILLER, _FILLER)
        assert compute_talker_prompt_ids_length(prompt) == 9

    def test_non_last_assistant_block_skipped(self) -> None:
        # Only the LAST assistant block contributes 9 — earlier ones are skipped.
        audio_user = _block(_USER, _AUDIO_PAD, _FILLER)
        prompt = (
            _block(_ASSISTANT, _FILLER, _FILLER)
            + audio_user
            + _block(_ASSISTANT, _FILLER)
        )
        # Non-last assistant: 0; audio user: full length; last assistant: 9.
        assert compute_talker_prompt_ids_length(prompt) == len(audio_user) + 9

    def test_full_conversation(self) -> None:
        # System + audio user + tool-response (text-only user) + last assistant.
        audio_user = _block(_USER, _AUDIO_PAD, _FILLER, _FILLER)
        tool_response_user = _block(_USER, _FILLER, _FILLER, _FILLER, _FILLER)
        prompt = (
            _block(_SYSTEM, _FILLER)
            + audio_user
            + tool_response_user
            + _block(_ASSISTANT, _FILLER)
        )
        # Only the audio user (full length) + last assistant (+9) count.
        assert compute_talker_prompt_ids_length(prompt) == len(audio_user) + 9
