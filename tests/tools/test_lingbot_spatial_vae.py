# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for the spatial decoder validation tool's failure boundary."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools/lingbot_world_spatial_vae" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rank_failure_reaps_all_children(monkeypatch):
    import signal

    import torch.multiprocessing as mp

    module = load("worker")
    children = [Mock() for _ in range(4)]
    for child in children:
        child.is_alive.side_effect = [True, False]
    context = SimpleNamespace(processes=children, join=Mock(side_effect=RuntimeError("rank failed")))
    monkeypatch.setattr(mp, "spawn", Mock(return_value=context))
    monkeypatch.setattr(signal, "signal", Mock())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    conn = Mock()
    with pytest.raises(RuntimeError, match="rank failed"):
        module.worker(conn, "0,1,2,3", "unused-model", 29500)
    for child in children:
        child.terminate.assert_called_once()
        child.join.assert_called_once_with(timeout=10)
    assert "rank failed" in conn.send.call_args.args[0]["error"]
