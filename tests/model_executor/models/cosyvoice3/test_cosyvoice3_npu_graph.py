# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU regressions for the opt-in CosyVoice3 Stage-1 NPU graph adapter."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.cfm import ConditionalCFM
from vllm_omni.model_executor.models.cosyvoice3.runtime import (
    cosyvoice3_batch_flow_profile,
    cosyvoice3_stage1_npu_graph_enabled,
    cosyvoice3_stage1_npu_graph_max,
)
from vllm_omni.platforms.npu.models.cosyvoice3_graph import (
    CosyVoice3HiFTGraphs,
    cosyvoice3_fia_blocked_mask,
    enable_cosyvoice3_graphs,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _SyntheticEstimator(nn.Module):
    def forward(self, x, mask, mu, t, spks, cond):
        return x + mask + mu + cond + spks.unsqueeze(-1) + t[:, None, None]


class _FakeExactGraphRunner:
    """Exercise production solve_euler dispatch without requiring an NPU."""

    def __init__(self, max_graphs: int = 8):
        self.max_graphs = max_graphs
        self.graphs: dict[tuple[object, ...], tuple[torch.Tensor, ...]] = {}
        self.inputs: list[tuple[torch.Tensor, ...]] = []
        self.fallbacks = 0

    def run(self, operation, inputs, constants, compute):
        key = (
            operation,
            constants,
            tuple((tuple(value.shape), value.dtype) for value in inputs),
        )
        self.inputs.append(tuple(value.detach().clone() for value in inputs))
        static_inputs = self.graphs.get(key)
        if static_inputs is None:
            if len(self.graphs) >= self.max_graphs:
                self.fallbacks += 1
                return tuple(value.detach().clone() for value in compute(*inputs))
            static_inputs = tuple(value.detach().clone() for value in inputs)
            self.graphs[key] = static_inputs
        else:
            for static, current in zip(static_inputs, inputs, strict=True):
                static.copy_(current)
        return tuple(value.detach().clone() for value in compute(*static_inputs))


def _make_cfm() -> ConditionalCFM:
    params = SimpleNamespace(
        solver="euler",
        sigma_min=1e-4,
        t_scheduler="linear",
        training_cfg_rate=0.0,
        inference_cfg_rate=0.5,
    )
    return ConditionalCFM(
        in_channels=80,
        cfm_params=params,
        n_spks=1,
        spk_emb_dim=80,
        estimator=_SyntheticEstimator(),
    )


def _solve_inputs(offset: float = 0.0):
    x = torch.full((1, 80, 3), offset)
    t_span = torch.tensor([0.0, 0.5, 1.0])
    mu = torch.full_like(x, 0.25 + offset)
    mask = torch.tensor([[[1.0, 1.0 - offset, 1.0]]])
    spks = torch.full((1, 80), 0.5 + offset)
    cond = torch.full_like(x, 0.75 + offset)
    return x, t_span, mu, mask, spks, cond


def test_solve_euler_fake_graph_refreshes_inputs_and_owns_outputs(monkeypatch):
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda *_args, **_kwargs: False)
    cfm = _make_cfm()
    runner = _FakeExactGraphRunner()
    cfm._npu_euler_graphs = runner

    first = cfm.solve_euler(*_solve_inputs(0.0))
    first_snapshot = first.clone()
    second = cfm.solve_euler(*_solve_inputs(0.5))

    assert not torch.equal(first, second)
    assert torch.equal(first, first_snapshot)
    assert first.data_ptr() != second.data_ptr()
    assert torch.equal(runner.inputs[-1][0], _solve_inputs(0.5)[0])


def test_solve_euler_graph_signature_tracks_cfg_and_autocast(monkeypatch):
    state = {"autocast": False}
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda *_args, **_kwargs: state["autocast"])
    monkeypatch.setattr(torch, "get_autocast_dtype", lambda *_args, **_kwargs: torch.bfloat16)
    cfm = _make_cfm()
    runner = _FakeExactGraphRunner()
    cfm._npu_euler_graphs = runner

    cfm.solve_euler(*_solve_inputs())
    cfm.inference_cfg_rate = 1.0
    cfm.solve_euler(*_solve_inputs())
    state["autocast"] = True
    cfm.solve_euler(*_solve_inputs())

    assert len(runner.graphs) == 3


def test_solve_euler_graph_limit_uses_eager_fallback(monkeypatch):
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda *_args, **_kwargs: False)
    cfm = _make_cfm()
    runner = _FakeExactGraphRunner(max_graphs=1)
    cfm._npu_euler_graphs = runner

    cfm.solve_euler(*_solve_inputs())
    cfm.inference_cfg_rate = 1.0
    output = cfm.solve_euler(*_solve_inputs(0.25))

    assert runner.fallbacks == 1
    assert output.shape == (1, 80, 3)


def test_fia_keep_mask_conversion_and_dtype_validation():
    keep_mask = torch.tensor([[True, False, True]])
    blocked = cosyvoice3_fia_blocked_mask(keep_mask, query_len=2, key_len=3)

    assert blocked.shape == (1, 1, 2, 3)
    assert torch.equal(blocked[0, 0, 0], torch.tensor([False, True, False]))
    assert torch.equal(blocked[0, 0, 0], blocked[0, 0, 1])
    with pytest.raises(TypeError, match="keep_mask must be bool"):
        cosyvoice3_fia_blocked_mask(keep_mask.int(), query_len=2, key_len=3)


class _CaptureOnceRunner:
    def __init__(self):
        self.captures = 0

    @property
    def stats(self):
        return {"captures": self.captures, "failed": 0, "hits": 0}

    def run(self, _operation, inputs, _constants, compute):
        output = compute(*inputs)
        self.captures += 1
        return output


class _FakeHiFT(nn.Module):
    def __init__(self):
        super().__init__()
        self._npu_istft_envelopes = {("unrelated",): torch.tensor([1.0])}
        self._npu_istft_envelope_accesses = None

    def _decode_eager(self, mel, source, *, finalize):
        key = ("final" if finalize else "stream",)
        self._npu_istft_envelopes[key] = torch.tensor([2.0])
        if self._npu_istft_envelope_accesses is not None:
            self._npu_istft_envelope_accesses.add(key)
        return mel + source


def test_hift_capture_retains_only_touched_envelopes():
    hift = _FakeHiFT()
    graphs = CosyVoice3HiFTGraphs(hift, max_graphs=1)
    graphs.runner = _CaptureOnceRunner()

    graphs.run(torch.ones(1), torch.ones(1), finalize=True)

    assert set(graphs._captured_envelopes) == {("final",)}
    assert graphs._captured_envelopes[("final",)] is hift._npu_istft_envelopes[("final",)]


def test_stage1_graph_environment_defaults_and_validation(monkeypatch):
    monkeypatch.delenv("COSYVOICE3_STAGE1_NPU_GRAPH", raising=False)
    monkeypatch.delenv("COSYVOICE3_STAGE1_NPU_GRAPH_MAX", raising=False)
    assert not cosyvoice3_stage1_npu_graph_enabled()
    assert cosyvoice3_stage1_npu_graph_max() == 8

    monkeypatch.setenv("COSYVOICE3_STAGE1_NPU_GRAPH", "1")
    monkeypatch.setenv("COSYVOICE3_STAGE1_NPU_GRAPH_MAX", "bad")
    assert cosyvoice3_stage1_npu_graph_enabled()
    with pytest.raises(ValueError, match="must be a positive integer, got 'bad'"):
        cosyvoice3_stage1_npu_graph_max()


def test_graph_install_rejects_fp32_before_process_runtime_changes(monkeypatch):
    decoder = SimpleNamespace(training=False, estimator=nn.Linear(2, 2).float())
    hift = SimpleNamespace(training=False)

    def unexpected_runtime_change():
        pytest.fail("process-wide NPU settings must not change for an unsupported dtype")

    monkeypatch.setattr(
        "vllm_omni.platforms.npu.models.cosyvoice3_graph.prepare_cosyvoice3_graph_runtime",
        unexpected_runtime_change,
    )
    with pytest.raises(ValueError, match="requires a float16 or bfloat16 DiT, got torch.float32"):
        enable_cosyvoice3_graphs(decoder, hift)


def test_profiler_scope_is_skipped_during_npu_capture(monkeypatch):
    monkeypatch.setenv("COSYVOICE3_BATCH_FLOW_DEBUG", "1")
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(is_current_stream_capturing=lambda: True),
        raising=False,
    )
    monkeypatch.setattr(
        torch.profiler,
        "record_function",
        lambda _name: pytest.fail("record_function must not run during graph capture"),
    )

    with cosyvoice3_batch_flow_profile("capture_safe"):
        pass
