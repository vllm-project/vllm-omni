# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import os

import pytest

from tests.helpers.runtime import assign_ephemeral_master_port

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_assign_ephemeral_master_port_replaces_a_stale_port(monkeypatch):
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "29501")

    port = assign_ephemeral_master_port()

    assert os.environ["MASTER_ADDR"] == "127.0.0.1"
    assert os.environ["MASTER_PORT"] == str(port)
    assert port != 29501
