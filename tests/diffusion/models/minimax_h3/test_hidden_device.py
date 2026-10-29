# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import ast
import os
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]
ROOT = Path(__file__).resolve().parents[4]


class MiniMaxH3Qwen3VLEncoder:
    """Typed name required by the extracted pipeline method."""


def method(file, owner, name):
    """Exercise actual lightweight methods without loading model/runtime dependencies."""
    source = ROOT / "vllm_omni/diffusion/models/minimax_h3" / file
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=ast.parse("from __future__ import annotations").body + [fn], type_ignores=[])
    scope = {
        "torch": torch,
        "os": os,
        "contextmanager": contextmanager,
        "MiniMaxH3Qwen3VLEncoder": MiniMaxH3Qwen3VLEncoder,
    }
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), scope)
    return scope[name]


@pytest.mark.parametrize("device_output", [False, True])
def test_return_placement_and_sdp_restore(mocker, device_output):
    encode = method("encoder.py", "MiniMaxH3Qwen3VLEncoder", "encode_ids")
    raw = mocker.Mock(cpu=mocker.Mock(return_value="host result"))
    obj = mocker.Mock(
        is_loaded=True,
        device_target=torch.device("cpu"),
        parameters=lambda: iter([torch.zeros(1)]),
        _encode=mocker.Mock(return_value=raw),
    )
    initial = torch.backends.cuda.cudnn_sdp_enabled()
    result = encode(obj, torch.tensor([1]), **({"return_on_device": True} if device_output else {}))
    assert result is raw if device_output else result == "host result"
    assert raw.cpu.call_count == (0 if device_output else 1)
    assert torch.backends.cuda.cudnn_sdp_enabled() == initial


@pytest.mark.parametrize("device_output", [False, True])
def test_exception_restores_sdp(mocker, device_output):
    encode = method("encoder.py", "MiniMaxH3Qwen3VLEncoder", "encode_ids")
    obj = mocker.Mock(
        is_loaded=True,
        device_target=torch.device("cpu"),
        parameters=lambda: iter([torch.zeros(1)]),
        _encode=mocker.Mock(side_effect=RuntimeError("failure")),
    )
    initial = torch.backends.cuda.cudnn_sdp_enabled()
    with pytest.raises(RuntimeError, match="failure"):
        encode(obj, torch.tensor([1]), return_on_device=device_output)
    assert torch.backends.cuda.cudnn_sdp_enabled() == initial


def test_forward_keeps_default_call_and_forwards_opt_in(mocker):
    forward = method("encoder.py", "MiniMaxH3Qwen3VLEncoder", "forward")
    obj = mocker.Mock(encode_ids=mocker.Mock(return_value="result"))
    ids = torch.tensor([1])
    assert forward(obj, ids) == "result"
    assert "return_on_device" not in obj.encode_ids.call_args.kwargs
    forward(obj, ids, return_on_device=True)
    assert obj.encode_ids.call_args.kwargs["return_on_device"] is True


@pytest.mark.parametrize("branch", ["resident", "manual", "sequential"])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_pipeline_routing_lifetime_and_input_ownership(mocker, monkeypatch, branch, enabled, fail):
    call = method("pipeline_minimax_h3.py", "MiniMaxH3Pipeline", "_encode_text_hidden")
    monkeypatch.setenv("VLLM_OMNI_H3_HIDDEN_DEVICE", "1" if enabled else "0")
    events: list[object] = []
    hidden = torch.arange(12).reshape(3, 4).to(torch.bfloat16)

    def encode(ids, **kwargs):
        events.append(("encode", kwargs.copy()))
        if fail:
            raise RuntimeError("failure")
        return hidden

    encoder = mocker.Mock(side_effect=encode)
    encoder.encode_ids.side_effect = encode

    @contextmanager
    def scope(component):
        events.append("load")
        try:
            yield
        finally:
            events.append("offload")

    obj = mocker.Mock(
        text_encoder=encoder,
        _model_cpu_offload_modules=["module"] if branch == "sequential" else [],
        _uses_manual_component_offload=lambda c: branch == "manual",
        _component_on_device=scope,
    )
    kwargs = {"pixel_values": torch.zeros(1)}
    if fail:
        with pytest.raises(RuntimeError, match="failure"):
            call(obj, torch.tensor([1]), kwargs)
    else:
        assert call(obj, torch.tensor([1]), kwargs) is hidden
        assert torch.equal(hidden, torch.arange(12).reshape(3, 4).to(torch.bfloat16))
    passed = next(e[1] for e in events if isinstance(e, tuple))
    assert passed.get("return_on_device", False) == enabled
    assert set(kwargs) == {"pixel_values"}
    if branch == "manual":
        assert events[0] == "load" and events[-1] == "offload"
    if branch == "sequential":
        encoder.assert_called_once()
        encoder.encode_ids.assert_not_called()
    if branch == "resident":
        encoder.load_to_device.assert_called_once()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("branch", ["manual", "sequential"])
@pytest.mark.parametrize("fail", [False, True])
def test_real_module_hooks_and_component_cleanup(mocker, monkeypatch, enabled, branch, fail):
    monkeypatch.setenv("VLLM_OMNI_H3_HIDDEN_DEVICE", "1" if enabled else "0")
    events = []

    class Encoder(torch.nn.Module):
        forward = method("encoder.py", "MiniMaxH3Qwen3VLEncoder", "forward")
        encode_ids = method("encoder.py", "MiniMaxH3Qwen3VLEncoder", "encode_ids")

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(4))
            self.is_loaded = True
            self.device_target = torch.device("cpu")

        def load_to_device(self):
            events.append("load")

        def offload_to_cpu(self):
            events.append("offload")
            self.weight = torch.nn.Parameter(torch.empty(0))

        def _encode(self, ids, **kwargs):
            if fail:
                raise RuntimeError("failure")
            return self.weight + ids.float()

    enc = Encoder()
    obj = mocker.Mock(
        text_encoder=enc,
        _model_cpu_offload_modules=[enc] if branch == "sequential" else [],
        _uses_manual_component_offload=lambda c: True,
    )
    obj._component_on_device = MethodType(
        method("pipeline_minimax_h3.py", "MiniMaxH3Pipeline", "_component_on_device"), obj
    )
    if branch == "sequential":
        enc.register_forward_pre_hook(lambda m, args: m.load_to_device())
        enc.register_forward_hook(lambda m, args, out: m.offload_to_cpu(), always_call=True)
    call = method("pipeline_minimax_h3.py", "MiniMaxH3Pipeline", "_encode_text_hidden")
    if fail:
        with pytest.raises(RuntimeError, match="failure"):
            call(obj, torch.arange(4), {})
    else:
        hidden = call(obj, torch.arange(4), {})
        # The real scope/hooks have released the parameter; activation remains usable.
        assert enc.weight.numel() == 0
        assert torch.equal(hidden, torch.arange(4) + 1)
        assert not torch._C._is_alias_of(hidden, enc.weight)
    assert events == ["load", "offload"]
