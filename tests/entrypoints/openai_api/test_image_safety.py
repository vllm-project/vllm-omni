# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for safety check behavior (HTTP status codes and error semantics).

Note: These tests exercise a local mirror of _check_safety, not the
production helper in api_server.py, because api_server.py has transitive
imports (vllm._C) that require CUDA. If the production _check_safety
logic changes, these tests must be updated manually.
"""

import asyncio
from http import HTTPStatus
from unittest.mock import MagicMock, patch

import PIL.Image
import pytest
from fastapi import HTTPException, Request


async def _check_safety(images, raw_request):
    """Mirror of api_server._check_safety for testing without heavy imports."""
    checker = getattr(raw_request.app.state, "safety_checker", None)
    if checker is None:
        return
    try:
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, checker.check_images, images)
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail=f"Safety checker unavailable: {e}",
        )
    unsafe_indices = [i for i, (is_safe, _) in enumerate(results) if not is_safe]
    if unsafe_indices:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Generated content was flagged as potentially unsafe.",
        )


def _make_image(size=(64, 64)):
    return PIL.Image.new("RGB", size, color="red")


def _make_request(safety_checker=None):
    """Create a mock Request with app.state.safety_checker set."""
    request = MagicMock(spec=Request)
    request.app.state.safety_checker = safety_checker
    return request


class _ImmediateLoop:
    async def run_in_executor(self, _executor, func, *args):
        return func(*args)


@pytest.mark.core_model
@pytest.mark.cpu
class TestSafetyCheckBehavior:
    def test_generate_unsafe_returns_400(self):
        """checker flags unsafe -> HTTP 400."""
        checker = MagicMock()
        checker.check_images.return_value = [(False, 0.9)]
        request = _make_request(safety_checker=checker)

        with patch("asyncio.get_running_loop", return_value=_ImmediateLoop()):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(_check_safety([_make_image()], request))
        assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST.value
        assert "unsafe" in exc_info.value.detail.lower()

    def test_generate_safe_returns_ok(self):
        """checker passes -> no exception raised."""
        checker = MagicMock()
        checker.check_images.return_value = [(True, 0.1)]
        request = _make_request(safety_checker=checker)

        with patch("asyncio.get_running_loop", return_value=_ImmediateLoop()):
            asyncio.run(_check_safety([_make_image()], request))  # should not raise

    def test_generate_no_checker_skips(self):
        """safety_checker=None -> no filtering."""
        request = _make_request(safety_checker=None)

        asyncio.run(_check_safety([_make_image()], request))  # should not raise

    def test_checker_failure_returns_503(self):
        """checker inference raises -> HTTP 503."""
        checker = MagicMock()
        checker.check_images.side_effect = RuntimeError("model failed")
        request = _make_request(safety_checker=checker)

        with patch("asyncio.get_running_loop", return_value=_ImmediateLoop()):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(_check_safety([_make_image()], request))
        assert exc_info.value.status_code == HTTPStatus.SERVICE_UNAVAILABLE.value
        assert "unavailable" in exc_info.value.detail.lower()

    def test_edit_unsafe_returns_400(self):
        """edit endpoint uses same _check_safety: unsafe -> HTTP 400."""
        checker = MagicMock()
        checker.check_images.return_value = [(True, 0.1), (False, 0.7)]
        request = _make_request(safety_checker=checker)

        with patch("asyncio.get_running_loop", return_value=_ImmediateLoop()):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(_check_safety([_make_image(), _make_image()], request))
        assert exc_info.value.status_code == HTTPStatus.BAD_REQUEST.value

    def test_edit_safe_returns_ok(self):
        """edit endpoint: all safe -> no exception."""
        checker = MagicMock()
        checker.check_images.return_value = [(True, 0.1), (True, 0.2)]
        request = _make_request(safety_checker=checker)

        with patch("asyncio.get_running_loop", return_value=_ImmediateLoop()):
            asyncio.run(_check_safety([_make_image(), _make_image()], request))
