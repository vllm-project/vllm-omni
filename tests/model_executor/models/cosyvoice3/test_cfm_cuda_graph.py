# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the flow estimator CUDA-graph path in CausalConditionalCFM."""

import pytest
import torch
from omegaconf import DictConfig

from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.cfm import (
    CausalConditionalCFM,
)

# The level mark is module-wide; the hardware mark is per test because the gate
# tests need no device while capture and replay do.
pytestmark = pytest.mark.core_model

CFM_PARAMS = DictConfig(
    {
        "sigma_min": 1e-06,
        "solver": "euler",
        "t_scheduler": "cosine",
        "training_cfg_rate": 0.2,
        "inference_cfg_rate": 0.7,
        "reg_loss_type": "l1",
    }
)


class _StubEstimator(torch.nn.Module):
    """Deterministic estimator with the DiT call signature."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(80, 80)

    def forward(self, x, mask, mu, t, spks, cond):
        h = (x + mu + cond).transpose(1, 2)
        h = self.proj(h) + t.view(-1, 1, 1) + spks.unsqueeze(1)
        return h.transpose(1, 2) * mask


def _make_cfm(device: str, estimator: torch.nn.Module | None = None, **graph_cfg) -> CausalConditionalCFM:
    cfm = CausalConditionalCFM(
        in_channels=240,
        cfm_params=CFM_PARAMS,
        n_spks=1,
        spk_emb_dim=80,
        estimator=estimator or _StubEstimator(),
        flow_graph_config={"enabled": True, **graph_cfg},
    )
    cfm.estimator.to(device).eval()
    return cfm


def _make_inputs(device: str, mel_len: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 80, mel_len, generator=g).to(device)
    mu = torch.randn(1, 80, mel_len, generator=g).to(device)
    mask = torch.ones(1, 1, mel_len, device=device)
    spks = torch.randn(1, 80, generator=g).to(device)
    cond = torch.randn(1, 80, mel_len, generator=g).to(device)
    t_span = torch.linspace(0, 1, 11, device=device)
    t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
    return x, t_span, mu, mask, spks, cond


@pytest.mark.cpu
def test_graphs_stay_off_unless_the_config_enables_them():
    default = CausalConditionalCFM(
        in_channels=240, cfm_params=CFM_PARAMS, n_spks=1, spk_emb_dim=80, estimator=_StubEstimator()
    )
    x, *_ = _make_inputs("cpu", 40)
    assert not default._estimator_graph_enabled
    assert not default._estimator_graph_usable(x)
    assert not _make_cfm("cpu", enabled=False)._estimator_graph_usable(x)


@pytest.mark.cpu
def test_config_overrides_are_read():
    assert _make_cfm("cpu", max_graphs=7)._estimator_graph_max == 7


@pytest.mark.cpu
def test_cpu_input_skips_graphs_and_matches_eager():
    cfm = _make_cfm("cpu")
    ref = _make_cfm("cpu", enabled=False)
    ref.estimator.load_state_dict(cfm.estimator.state_dict())
    x, t_span, mu, mask, spks, cond = _make_inputs("cpu", 40)
    assert not cfm._estimator_graph_usable(x)
    out = cfm.solve_euler(x, t_span, mu, mask, spks, cond)
    expected = ref.solve_euler(x, t_span, mu, mask, spks, cond)
    assert torch.equal(out, expected)
    assert not cfm._estimator_graphs


@pytest.mark.cpu
def test_training_estimator_skips_graphs():
    cfm = _make_cfm("cpu")
    cfm.estimator.train()
    x, *_ = _make_inputs("cpu", 40)
    assert not cfm._estimator_graph_usable(x)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
class TestCudaGraphEquivalence:
    def _gpu_pair(self, **kw):
        """A graphed CFM and an eager one sharing identical weights."""
        from vllm_omni.diffusion.models.cosyvoice3_audio.cosyvoice3_dit import DiT

        def _dit():
            torch.manual_seed(0)
            return DiT(dim=64, depth=2, heads=2, dim_head=32, dropout=0.0, mel_dim=80, mu_dim=80, spk_dim=80)

        graphed = _make_cfm("cuda", estimator=_dit(), **kw)
        eager = _make_cfm("cuda", estimator=_dit(), enabled=False)
        eager.estimator.load_state_dict(graphed.estimator.state_dict())
        return graphed, eager

    def test_captures_on_first_call_and_matches_eager(self):
        """No recurrence heuristic: one call captures, then replays it 9 times."""
        graphed, eager = self._gpu_pair()
        inputs = _make_inputs("cuda", 254)
        ref = eager.solve_euler(*inputs)
        out = graphed.solve_euler(*inputs)
        assert len(graphed._estimator_graphs) == 1
        stats = graphed.flow_graph_stats()
        assert stats["captures"] == 1
        assert stats["hits"] == 9  # the remaining Euler steps replayed it
        torch.testing.assert_close(out, ref, rtol=0, atol=1e-5)

    def test_replay_reads_fresh_inputs(self):
        graphed, eager = self._gpu_pair()
        first = _make_inputs("cuda", 128, seed=0)
        second = _make_inputs("cuda", 128, seed=1)
        out1 = graphed.solve_euler(*first)
        out2 = graphed.solve_euler(*second)
        assert len(graphed._estimator_graphs) == 1  # same shape, replayed
        assert not torch.allclose(out1, out2)
        torch.testing.assert_close(out2, eager.solve_euler(*second), rtol=0, atol=1e-5)

    def test_multiple_shapes_cached(self):
        graphed, eager = self._gpu_pair()
        for mel_len in (96, 160, 254):
            inputs = _make_inputs("cuda", mel_len)
            out = graphed.solve_euler(*inputs)
            torch.testing.assert_close(out, eager.solve_euler(*inputs), rtol=0, atol=1e-5)
        assert len(graphed._estimator_graphs) == 3

    def test_full_cache_retires_whole_generation(self):
        """A full cache must retire as one generation, not evict one entry.

        Freeing a single graph invalidates the pool its peers replay from.
        """
        graphed, eager = self._gpu_pair(max_graphs=2)
        for mel_len in (96, 160):
            graphed.solve_euler(*_make_inputs("cuda", mel_len))
        assert len(graphed._estimator_graphs) == 2

        overflow = _make_inputs("cuda", 254)
        out = graphed.solve_euler(*overflow)
        assert graphed.flow_graph_stats()["flushes"] == 1
        # That step fell back to eager; the next one re-captures into the
        # emptied cache, so exactly one entry survives the call.
        assert len(graphed._estimator_graphs) == 1
        torch.testing.assert_close(out, eager.solve_euler(*overflow), rtol=0, atol=1e-5)

    def test_low_free_memory_stays_eager(self, monkeypatch):
        from vllm_omni.platforms import current_omni_platform

        graphed, eager = self._gpu_pair()

        def no_free_memory(*_args, **_kwargs):
            return 0, 1 << 40

        monkeypatch.setattr(current_omni_platform, "get_device_memory", no_free_memory)
        inputs = _make_inputs("cuda", 254)
        out = graphed.solve_euler(*inputs)
        assert not graphed._estimator_graphs  # floor kept it eager instead of OOMing
        torch.testing.assert_close(out, eager.solve_euler(*inputs), rtol=0, atol=1e-5)

    def test_capture_failure_disables_for_the_process(self, monkeypatch):
        graphed, eager = self._gpu_pair()

        def boom(*_args, **_kwargs):
            raise RuntimeError("capture exploded")

        monkeypatch.setattr(graphed, "_capture_estimator_graph", boom)
        inputs = _make_inputs("cuda", 64)
        out = graphed.solve_euler(*inputs)
        torch.testing.assert_close(out, eager.solve_euler(*inputs), rtol=0, atol=1e-5)
        assert not graphed._estimator_graph_enabled
