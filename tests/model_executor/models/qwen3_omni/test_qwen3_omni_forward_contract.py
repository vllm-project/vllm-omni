# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the Qwen3-Omni thinker forward return contract.

Background
----------
Since omni architectures unconditionally override upstream's registry entries,
``Qwen3OmniMoeForConditionalGeneration`` (and the standalone thinker) are also
constructed by plain, non-staged vLLM runs — e.g. ``vllm serve
Qwen/Qwen3-Omni-30B-A3B-Instruct`` in the nightly vLLM-text perf job. Stock
vLLM's ``GPUModelRunner`` indexes the forward output directly
(``hidden_states[logit_indices]``), so a thinker forward that always returns a
``(hidden_states, captured_hidden_states)`` tuple kills the engine during
``profile_run`` with ``TypeError: only integer tensors of a single element can
be converted to an index`` (Buildkite build 2952).

Contract under test:

- No capture requested → the forward returns a **bare tensor** (stock-runner
  compatible).
- Capture requested (``capture_layer_indices`` + ``return_hidden_states``,
  staged runs with a talker downstream) → the forward returns the
  ``(hidden_states, captured_layer_dict)`` tuple omni's runner consumes via
  ``make_omni_output``.
- The combined model requests capture only in staged runs
  (``model_config.model_stage`` injected), never in plain vLLM runs.
- ``make_omni_output`` tolerates both return shapes.

Pure shape/contract tests: CPU-only, no weights, no GPU.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import (
    Qwen3OmniMoeForConditionalGeneration,
)
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_HIDDEN = 8
_TOKENS = 4


def _hidden_states() -> torch.Tensor:
    return torch.zeros(_TOKENS, _HIDDEN)


class _InnerModelStub:
    """Stands in for ``Qwen3MoeLLMModel``: always returns the internal 2-tuple."""

    def __init__(self, output=None):
        self.calls: list[dict] = []
        self._output = output

    def __call__(self, input_ids, positions, intermediate_tensors, **kwargs):
        self.calls.append(kwargs)
        if self._output is not None:
            return self._output
        if kwargs.get("return_hidden_states"):
            return _hidden_states(), {"hidden_states": {"layers": {0: _hidden_states()}}}
        return _hidden_states(), None


def _make_thinker(inner_output=None) -> tuple[Qwen3OmniMoeThinkerForConditionalGeneration, _InnerModelStub]:
    thinker = object.__new__(Qwen3OmniMoeThinkerForConditionalGeneration)
    nn.Module.__init__(thinker)
    inner = _InnerModelStub(output=inner_output)
    thinker.use_deepstack = False
    thinker.language_model = SimpleNamespace(model=inner)
    return thinker, inner


def _forward_args():
    return {
        "input_ids": torch.zeros(_TOKENS, dtype=torch.long),
        "positions": torch.arange(_TOKENS),
    }


def test_thinker_forward_returns_bare_tensor_without_capture():
    thinker, _ = _make_thinker()
    out = thinker.forward(**_forward_args())
    assert isinstance(out, torch.Tensor)
    assert out.shape == (_TOKENS, _HIDDEN)


def test_thinker_forward_returns_tuple_when_capturing():
    thinker, _ = _make_thinker()
    out = thinker.forward(
        **_forward_args(),
        capture_layer_indices=[0, 2],
        return_hidden_states=True,
    )
    assert isinstance(out, tuple)
    hidden, captured = out
    assert isinstance(hidden, torch.Tensor)
    assert "hidden_states" in captured


def test_thinker_forward_passes_through_intermediate_tensors():
    it = IntermediateTensors({"hidden_states": _hidden_states(), "residual": _hidden_states()})
    thinker, _ = _make_thinker(inner_output=it)
    out = thinker.forward(**_forward_args())
    assert out is it


class _ThinkerStub(nn.Module):
    """Records capture kwargs; mirrors the fixed thinker return contract."""

    def __init__(self):
        super().__init__()
        self.received: dict = {}

    def forward(self, **kwargs):
        self.received = kwargs
        if kwargs.get("return_hidden_states"):
            return _hidden_states(), {"hidden_states": {"layers": {}}}
        return _hidden_states()


def _make_combined(*, is_staged_run: bool, accept_hidden_layer=6) -> Qwen3OmniMoeForConditionalGeneration:
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    nn.Module.__init__(model)
    model.model_stage = "thinker"
    model.is_staged_run = is_staged_run
    model.talker_config = SimpleNamespace(accept_hidden_layer=accept_hidden_layer)
    model.thinker = _ThinkerStub()
    return model


def test_combined_forward_plain_run_skips_capture_and_returns_tensor():
    model = _make_combined(is_staged_run=False)
    out = model.forward(**_forward_args())
    assert "capture_layer_indices" not in model.thinker.received
    assert isinstance(out, torch.Tensor)


def test_combined_forward_staged_run_captures_and_returns_tuple():
    model = _make_combined(is_staged_run=True)
    out = model.forward(**_forward_args())
    assert model.thinker.received["capture_layer_indices"] == [0, 6]
    assert model.thinker.received["return_hidden_states"] is True
    assert isinstance(out, tuple)


def test_combined_forward_staged_run_without_accept_layer_returns_tensor():
    model = _make_combined(is_staged_run=True, accept_hidden_layer=None)
    out = model.forward(**_forward_args())
    assert "capture_layer_indices" not in model.thinker.received
    assert isinstance(out, torch.Tensor)


def test_make_omni_output_accepts_bare_tensor():
    model = _make_combined(is_staged_run=False)
    out = model.make_omni_output(_hidden_states())
    assert isinstance(out, OmniOutput)
    assert out.text_hidden_states.shape == (_TOKENS, _HIDDEN)
    assert out.multimodal_outputs == {}


def test_make_omni_output_accepts_capture_tuple():
    model = _make_combined(is_staged_run=True)
    captured = {"hidden_states": {"layers": {}}}
    out = model.make_omni_output((_hidden_states(), captured))
    assert isinstance(out, OmniOutput)
    assert out.multimodal_outputs == captured
