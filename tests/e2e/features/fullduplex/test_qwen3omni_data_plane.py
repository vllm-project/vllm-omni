# tests/e2e/features/fullduplex/test_qwen3omni_data_plane.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility data-plane contract tests for Qwen3-Omni."""

import pytest

from vllm_omni.experimental.fullduplex.qwen3omni.data_plane import (
    Qwen3OmniDataPlaneSession,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _encode(samples, sample_rate, fmt, speed=None):
    return "audio-b64"


def test_project_fails_closed_because_chat_fallback_owns_output():
    plane = Qwen3OmniDataPlaneSession(_encode)

    with pytest.raises(RuntimeError, match="chat fallback"):
        list(plane.project(object()))


def test_terminal_lifecycle():
    plane = Qwen3OmniDataPlaneSession(_encode)
    plane.begin_request("req-1")
    plane.mark_terminal("req-1")
    assert plane.is_terminal("req-1") is True
    plane.close_session("s1", active_request_id="req-1")
    assert plane.is_terminal("req-1") is True


def test_is_terminal_none_is_terminal():
    plane = Qwen3OmniDataPlaneSession(_encode)
    assert plane.is_terminal(None) is True


def test_close_stream_marks_terminal():
    plane = Qwen3OmniDataPlaneSession(_encode)
    plane.close_stream("req-1")
    assert plane.is_terminal("req-1") is True
