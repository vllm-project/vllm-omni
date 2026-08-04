# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniCPM-o invalid-param tests share the session-scoped duplex server."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest


@pytest.fixture(scope="module")
def omni_server(minicpmo_duplex_server: Any) -> Generator[Any, Any, None]:
    yield minicpmo_duplex_server
