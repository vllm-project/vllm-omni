# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for stage context logging in vllm_omni.logger."""

import logging

import pytest

from vllm_omni.logger import (
    StageContextFilter,
    clear_stage_context,
    get_stage_context,
    set_stage_context,
)

pytestmark = [pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _clean_stage_context() -> None:
    """Ensure stage context is cleared before and after each test."""
    clear_stage_context()
    yield  # type: ignore[misc]
    clear_stage_context()


class TestStageContextAPI:
    """Tests for the set/get/clear stage context API."""

    def test_default_context_is_none(self) -> None:
        """get_stage_context returns None when no context has been set."""
        assert get_stage_context() is None

    def test_set_and_get_stage_context(self) -> None:
        """set_stage_context stores a (stage_id, model_stage) tuple."""
        set_stage_context(0, "thinker")
        result = get_stage_context()
        assert result == (0, "thinker")

    def test_clear_stage_context(self) -> None:
        """clear_stage_context resets context to None."""
        set_stage_context(1, "talker")
        clear_stage_context()
        assert get_stage_context() is None

    def test_overwrite_stage_context(self) -> None:
        """Calling set_stage_context again overwrites the previous value."""
        set_stage_context(0, "thinker")
        set_stage_context(2, "code2wav")
        assert get_stage_context() == (2, "code2wav")


class TestStageContextFilter:
    """Tests for the StageContextFilter logging filter."""

    def _make_record(self, msg: str = "test message") -> logging.LogRecord:
        return logging.LogRecord(
            name="vllm_omni.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg=msg,
            args=(),
            exc_info=None,
        )

    def test_filter_always_returns_true(self) -> None:
        """Filter never suppresses log records."""
        f = StageContextFilter()
        record = self._make_record()
        assert f.filter(record) is True

    def test_filter_no_context_no_prefix(self) -> None:
        """Without stage context, the message is unchanged."""
        f = StageContextFilter()
        record = self._make_record("original msg")
        f.filter(record)
        assert record.msg == "original msg"
        assert not hasattr(record, "stage_id")

    def test_filter_prepends_tag(self) -> None:
        """With stage context set, the filter prepends [STAGE:<name>]."""
        set_stage_context(1, "talker")
        f = StageContextFilter()
        record = self._make_record("hello")
        f.filter(record)
        assert record.msg == "[STAGE:talker] hello"
        assert record.stage_id == 1  # type: ignore[attr-defined]
        assert record.stage_tag == "talker"  # type: ignore[attr-defined]

    def test_different_stages_produce_different_tags(self) -> None:
        """Different stage contexts produce different prefixes."""
        f = StageContextFilter()

        set_stage_context(0, "thinker")
        r1 = self._make_record("msg1")
        f.filter(r1)
        assert r1.msg == "[STAGE:thinker] msg1"

        set_stage_context(2, "code2wav")
        r2 = self._make_record("msg2")
        f.filter(r2)
        assert r2.msg == "[STAGE:code2wav] msg2"


class TestStageLoggingIntegration:
    """Integration tests using actual vllm_omni loggers."""

    @staticmethod
    def _capture_from_vllm_omni(logger_name: str, msg: str) -> list[logging.LogRecord]:
        captured: list[logging.LogRecord] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured.append(record)

        vllm_omni_logger = logging.getLogger("vllm_omni")
        handler = _Capture()
        vllm_omni_logger.addHandler(handler)
        try:
            logging.getLogger(logger_name).info(msg)
        finally:
            vllm_omni_logger.removeHandler(handler)
        return captured

    def test_integration_tag_applied(self) -> None:
        """Log via a vllm_omni child logger with stage context has tag."""
        set_stage_context(0, "thinker")
        records = self._capture_from_vllm_omni("vllm_omni.test_integration", "integration test")
        assert len(records) == 1
        assert "[STAGE:thinker]" in records[0].msg

    def test_no_tag_without_context(self) -> None:
        """Log via a vllm_omni child logger without context has no prefix."""
        records = self._capture_from_vllm_omni("vllm_omni.test_integration", "no context here")
        assert len(records) == 1
        assert "[STAGE:" not in records[0].msg
