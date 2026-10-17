# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch.nn as nn

import vllm_omni.diffusion.compile as compile_module
from vllm_omni.diffusion.compile import regionally_compile
from vllm_omni.diffusion.hooks import HookRegistry, ModelHook

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _WrappedBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.compile_called = False
        self.forward_compiled = False

    def compile(self, *args, **kwargs):
        self.compile_called = True
        return self

    def forward(self, x):
        return x


class _ModelWithWrappedRepeatedBlocks(nn.Module):
    _repeated_blocks = ["OriginalBlock"]
    _layerwise_offload_blocks_attrs = ["transformer_blocks"]

    def __init__(self) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_WrappedBlock(), _WrappedBlock()])
        self.other_blocks = nn.ModuleList([_WrappedBlock()])


def test_regionally_compile_matches_wrapped_blocks_by_declared_container_attr(monkeypatch):
    model = _ModelWithWrappedRepeatedBlocks()
    compile_calls = []

    def _compile(fn, *args, **kwargs):
        compile_calls.append((fn, args, kwargs))

        def _compiled(*fn_args, **fn_kwargs):
            return f"compiled:{fn(*fn_args, **fn_kwargs)}"

        return _compiled

    monkeypatch.setattr(compile_module.torch, "compile", _compile)

    regionally_compile(model, dynamic=True)

    assert len(compile_calls) == 2
    assert all(not block.compile_called for block in model.transformer_blocks)
    assert not model.other_blocks[0].compile_called
    assert model.transformer_blocks[0].forward("ok") == "compiled:ok"


def test_regionally_compile_does_not_partially_mutate_on_setup_failure(monkeypatch):
    model = _ModelWithWrappedRepeatedBlocks()
    original_forwards = [block.forward.__func__ for block in model.transformer_blocks]
    compile_calls = 0

    def _compile(fn, *args, **kwargs):
        nonlocal compile_calls
        compile_calls += 1
        if compile_calls == 2:
            raise RuntimeError("compile setup failed")
        return lambda *fn_args, **fn_kwargs: fn(*fn_args, **fn_kwargs)

    monkeypatch.setattr(compile_module.torch, "compile", _compile)

    with pytest.raises(RuntimeError, match="compile setup failed"):
        regionally_compile(model, dynamic=True)

    assert [block.forward.__func__ for block in model.transformer_blocks] == original_forwards


def test_regionally_compile_keeps_hook_dispatch_outside_compiled_graph(monkeypatch):
    model = _ModelWithWrappedRepeatedBlocks()
    block = model.transformer_blocks[0]
    registry = HookRegistry.get_or_create(block)
    hook = ModelHook()
    registry.register_hook("test", hook)

    wrapped_forward = block.forward
    original_forward = block._omni_original_forward
    compile_calls = []

    def _compile(fn, *args, **kwargs):
        compile_calls.append(fn)

        def _compiled(*fn_args, **fn_kwargs):
            return f"compiled:{fn(*fn_args, **fn_kwargs)}"

        return _compiled

    monkeypatch.setattr(compile_module.torch, "compile", _compile)

    regionally_compile(model)

    assert compile_calls[0] is original_forward
    assert compile_calls[0] is not wrapped_forward
    assert block.forward is wrapped_forward
    assert block._omni_original_forward is not original_forward
    assert hook.fn_ref.original_forward is block._omni_original_forward
    assert block("ok") == "compiled:ok"


def test_compiled_block_preserves_forward_signature_for_inspection(monkeypatch):
    """cache-dit matches blocks via inspect.signature(block.forward).

    Whatever regionally_compile installs as the block forward must stay
    signature-transparent: parameter names and the return annotation drive
    cache-dit's ForwardPattern match (build 2954, Multi-GPU Layered job
    failed when a bare *args/**kwargs wrapper hid them; torch.compile's own
    wrapper preserves the signature).
    """
    import inspect

    import torch

    class _SignatureBlock(nn.Module):
        def forward(self, hidden_states, encoder_hidden_states=None) -> "torch.Tensor":
            return hidden_states

    class _Model(nn.Module):
        _repeated_blocks = ["_SignatureBlock"]

        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([_SignatureBlock()])

    model = _Model()
    monkeypatch.setattr(compile_module.torch, "compile", lambda fn, *a, **k: fn)
    regionally_compile(model)

    sig = inspect.signature(model.blocks[0].forward)
    assert set(sig.parameters.keys()) == {"hidden_states", "encoder_hidden_states"}
    assert "torch.Tensor" in str(sig.return_annotation)
    assert model.blocks[0].forward("x") == "x"
