# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Helpers for resolving ``concurrent.futures.Future`` objects shared across threads."""

import concurrent.futures
from typing import Any, TypeVar

from vllm.logger import init_logger

logger = init_logger(__name__)

_ResultT = TypeVar("_ResultT")


def try_set_result(fut: concurrent.futures.Future[_ResultT], result: _ResultT) -> bool:
    # fut may be cancelled concurrently (e.g. asyncio.wait_for timeout) between
    # the caller's fut.done() check and this call; drop the late result instead
    # of crashing the pump thread. Return whether the future was actually
    # resolved so callers that need to mirror the terminal state elsewhere
    # (e.g. record the id as dropped) can react to a lost delivery.
    try:
        fut.set_result(result)
    except concurrent.futures.InvalidStateError:
        logger.debug("Dropping late result for already-resolved/cancelled future")
        return False
    return True


def try_set_exception(fut: concurrent.futures.Future[Any], exc: BaseException) -> bool:
    try:
        fut.set_exception(exc)
    except concurrent.futures.InvalidStateError:
        logger.debug("Dropping late exception for already-resolved/cancelled future")
        return False
    return True
