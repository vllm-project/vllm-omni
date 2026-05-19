"""Fixtures for the process-wide speaker cache singleton."""

from __future__ import annotations

import pytest


@pytest.fixture
def fresh_speaker_cache():
    """Reset the process-wide speaker cache singleton before and after the test."""
    import vllm_omni.utils.speaker_cache as _sc

    def _reset():
        with _sc._SINGLETON_LOCK:
            if _sc._SINGLETON is not None:
                _sc._SINGLETON.clear()
            _sc._SINGLETON = None

    _reset()
    yield
    _reset()
