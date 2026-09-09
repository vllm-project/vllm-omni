# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Fault injection must never target unrelated services on a shared GPU host."""

from types import SimpleNamespace

import pytest

from tests.dfx.reliability import helpers

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_process_fault_requires_a_known_test_server_tree(monkeypatch):
    killed = []
    monkeypatch.setattr(helpers, "_log_server_process_tree", lambda server: None)
    monkeypatch.setattr(helpers, "_list_server_process_tree", lambda server: [])
    monkeypatch.setattr(helpers, "_pids_in_server_tree_substring_match", lambda *a, **kw: [])
    monkeypatch.setattr(helpers, "inject_process_kill", lambda **kw: [987654])
    monkeypatch.setattr(helpers, "_safe_proc_info", lambda pid: ("unrelated", "unrelated service"))
    monkeypatch.setattr(helpers.os, "kill", lambda pid, sig: killed.append(pid))
    inject = helpers.make_process_kill_fault_injector(grep_patterns="worker")
    with pytest.raises(ValueError, match="server process tree"):
        inject(SimpleNamespace())
    assert killed == []


def test_process_fault_filters_global_discovery_to_owned_pids(monkeypatch):
    killed = []
    monkeypatch.setattr(helpers, "_log_server_process_tree", lambda server: None)
    monkeypatch.setattr(helpers, "_list_server_process_tree", lambda server: [100, 101])
    monkeypatch.setattr(helpers, "_pids_in_server_tree_substring_match", lambda *a, **kw: [])
    monkeypatch.setattr(helpers, "inject_process_kill", lambda **kw: [987654, 101])
    monkeypatch.setattr(helpers, "_safe_proc_info", lambda pid: ("worker", "worker"))
    monkeypatch.setattr(helpers.os, "kill", lambda pid, sig: killed.append(pid))
    helpers.make_process_kill_fault_injector(grep_patterns="worker")(SimpleNamespace())
    assert killed == [101]
