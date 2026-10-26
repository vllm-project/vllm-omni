# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Serving layout and collective-group contracts; real NPU gates are separate."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def load_module():
    path = Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/minimax_h3/vision_parallel.py"
    spec = importlib.util.spec_from_file_location("service_parallel_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_transport_uses_explicit_encoder_group(monkeypatch):
    m = load_module()
    group = object()
    monkeypatch.setattr(m.dist, "get_rank", lambda group=None: 1 if group is not None else 9)
    monkeypatch.setattr(m.dist, "get_world_size", lambda group=None: 2 if group is not None else 16)
    calls = []

    def exchange(out, value, *, group=None):
        calls.append(group)
        out.copy_(value)

    monkeypatch.setattr(m.dist, "all_to_all_single", exchange)
    plan = m.build_vision_plan((8, 12), 2)
    transport = m.VisionAllToAll(plan, group=group)
    assert (transport.rank, transport.size) == (1, 2)
    transport.exchange(torch.ones(2, 2))
    assert calls == [group]


def test_full_output_gather_restores_global_merger_order(monkeypatch):
    m = load_module()
    group = object()
    plan = m.build_vision_plan((12, 4, 20), 2)
    full = torch.arange(4 * 9 * 3).reshape(4, 9, 3).float()
    local_ids = plan.rank_merger_indices[0]
    local = full[:, list(local_ids)]
    seen = []

    def gather(received, value, *, group=None):
        seen.append(group)
        for rank, target in enumerate(received):
            ids = plan.rank_merger_indices[rank]
            target.zero_()
            target[:, : len(ids)] = full[:, list(ids)]

    monkeypatch.setattr(m.dist, "all_gather", gather)
    final, deep = m.gather_vision_outputs((local[0], list(local[1:])), plan, group=group)
    assert seen == [group]
    assert torch.equal(torch.stack([final, *deep]), full)


@pytest.mark.parametrize("value", ["2", "16", "yes", "-1"])
def test_service_rejects_unsupported_sp_size(mocker, value):
    m = load_module()
    with pytest.raises(ValueError):
        m.resolve_vision_sp_group(value, mocker.Mock(world_size=8, device_group=object()))


def test_service_is_opt_in_and_requires_exact_tp8(mocker):
    m = load_module()
    assert m.resolve_vision_sp_group("0", None) is None
    group = mocker.Mock(world_size=8, device_group=object())
    assert m.resolve_vision_sp_group("8", group) is group.device_group
    with pytest.raises(ValueError):
        m.resolve_vision_sp_group("8", mocker.Mock(world_size=4))


def test_empty_rank_shard_keeps_explicit_hidden_dimension(mocker):
    # Exercise the real attention body without importing the serving dependency tree.
    import ast

    path = Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/minimax_h3/vision_sp.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "attention_sp8")
    namespace = {"torch": torch, "segment_attention": lambda q, k, v: v}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    attention = mocker.Mock(num_heads=16, head_dim=2, qkv=lambda x: x.new_empty((0, 96)), proj=lambda x: x)
    transport = mocker.Mock(
        plan=mocker.Mock(segment_lengths=(4,)),
        to_heads=lambda x: x.new_zeros((4, 2, 2)),
        to_tokens=lambda x: x.new_empty((0, 16, 2)),
    )
    hidden = torch.empty(0, 32)
    result = namespace["attention_sp8"](attention, hidden, (None, None), transport, lambda q, k, c, s: (q, k))
    assert result.shape == (0, 32)
