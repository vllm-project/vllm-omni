# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bit-exactness tests for the fused rep-penalty/temperature prelude.

The fused Triton kernel is the default stage-1 codec sampling prologue on NPU
(disable it with ``talker_codec_fused_prelude: false`` in the stage connector
extra). These tests compare it against the eager chain it replaces
(``_repetition_penalty_scaling`` + ``logits / temperature``)
through the public ``prepare_codec_logits`` entry point with the fused switch
forced on and off, and additionally capture and replay the fused path under an
NPU graph (``torch.npu.NPUGraph``) -- a Triton launch under graph capture is a
different support path from an eager launch.

Everything is skipped outside NPU+Triton: the eager chain runs anywhere, but
the parity claim these tests guard is only made where the fused path ships.
"""

from typing import Any

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.models.minicpmo_4_5 import talker_codec_sample as tcs
from vllm_omni.platforms import current_omni_platform

pytestmark = [
    pytest.mark.core_model,
    *hardware_marks(res={"npu": "A3"}, num_cards=1),
    pytest.mark.skipif(not current_omni_platform.is_npu(), reason="requires Ascend NPU"),
]


def _npu_and_triton_available() -> bool:
    return (hasattr(torch, "npu") and torch.npu.is_available()) and tcs._HAS_TRITON


def _device() -> torch.device:
    """Build the NPU device lazily.

    ``torch.device("npu")`` raises at import time when ``torch_npu`` has not
    registered the backend, which would turn collection on a CPU/GPU host into
    a collection error instead of a skip. Module scope must stay free of it.
    """
    return torch.device("npu")


def _make_state(codes: list[int]) -> tcs.TalkerCodecDeviceState:
    return tcs.make_device_state(
        torch.tensor(codes, dtype=torch.int32, device=_device()),
        step=0,
        max_tokens=100,
        finished=False,
    )


def _run_both(raw_logits, state, repetition_penalty, temperature):
    """Run prepare_codec_logits with the fused path on and off, return both.

    The fused path is guarded by the module-level latch `_FUSED_PRELUDE_OK`,
    so tests flip that attribute directly and restore it in a try/finally.
    """
    device = _device()
    common = dict(
        state=state,
        min_tokens=torch.tensor([0], dtype=torch.int32, device=device),
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        eos_token_id=0,
        top_k=0,
        top_p=1.0,
        min_p=0.0,
    )
    original = tcs._FUSED_PRELUDE_OK
    try:
        tcs._FUSED_PRELUDE_OK = True
        fused = tcs.prepare_codec_logits(raw_logits.clone(), **common)
        tcs._FUSED_PRELUDE_OK = False
        eager = tcs.prepare_codec_logits(raw_logits.clone(), **common)
    finally:
        tcs._FUSED_PRELUDE_OK = original
    return fused, eager


CASES: dict[str, dict[str, Any]] = {
    # history_len < HISTORY_WINDOW: the window tail is zero-filled and the
    # kernel must gate counts on hlen, not on the padding.
    "short_history": dict(codes=[11, 22, 33, 44, 55], rep=1.1, temp=1.0, neg=False, neg_inf=False),
    # repeated ids inside the window: counts must accumulate per occurrence.
    "repeated_ids": dict(
        codes=[7, 7, 7, 101, 202, 303, 404, 505, 606, 707], rep=1.2, temp=0.9, neg=False, neg_inf=False
    ),
    # negative logits take the multiplication branch of the penalty.
    "negative_logits": dict(codes=[1, 2, 3, 4], rep=1.3, temp=1.0, neg=True, neg_inf=False),
    # -inf censored entries must stay -inf through penalty and temperature.
    "neg_inf_entries": dict(codes=[1, 2, 3, 4], rep=1.1, temp=1.0, neg=False, neg_inf=True),
    # a penalty other than 1.0 with a temperature that is not 1 either.
    "penalty_and_temperature": dict(codes=[5, 6, 7, 8, 9, 10], rep=1.5, temp=0.8, neg=False, neg_inf=False),
}


@pytest.mark.parametrize("case_name", sorted(CASES))
@pytest.mark.skipif(not _npu_and_triton_available(), reason="requires NPU + Triton")
def test_fused_prelude_bit_exact_eager(case_name):
    device = _device()
    cfg = CASES[case_name]
    state = _make_state(cfg["codes"])
    raw = torch.randn(1, tcs.VOCAB_SIZE, device=device)
    if cfg["neg"]:
        raw = -raw.abs() - 0.5
    if cfg["neg_inf"]:
        raw[:, [100, 200, 3000]] = float("-inf")
    raw[:, 0] = 0.0  # keep the codec EOS out of the censoring paths

    fused, eager = _run_both(
        raw,
        state,
        torch.tensor([cfg["rep"]], dtype=torch.float32, device=device),
        torch.tensor([cfg["temp"]], dtype=torch.float32, device=device),
    )
    assert torch.equal(fused, eager)


@pytest.mark.skipif(not _npu_and_triton_available(), reason="requires NPU + Triton")
def test_fused_prelude_bit_exact_under_graph_capture_and_replay():
    """Capture the fused kernel in an NPUGraph, then replay on new inputs.

    A Triton launch under graph capture is a different support path from an
    eager launch, so the kernel is exercised in both modes. The capture scope
    is the fused kernel itself (the unit that production wraps), not the whole
    prepare_codec_logits chain: the shared censoring step allocates its
    stop_index tensor per call (H2D from a Python tuple), which is not a
    capture-safe pattern and is independent of the fused prelude.

    The eager warmup launch (which also compiles the kernel) is asserted on the
    captured inputs. The capture-phase output itself is not asserted -- graph
    capture records ops without executing them -- so the assertion that matters
    is the replay output matching the eager chain bit-for-bit on the replayed
    inputs.
    """
    device = _device()
    state = _make_state([3, 1, 4, 1, 5, 9, 2, 6])
    rep = torch.tensor([1.2], dtype=torch.float32, device=device)
    temp = torch.tensor([0.9], dtype=torch.float32, device=device)
    raw_a = torch.randn(1, tcs.VOCAB_SIZE, device=device)
    raw_b = torch.randn(1, tcs.VOCAB_SIZE, device=device)
    raw_b[:, 0] = 0.0

    def _eager(x):
        logits = tcs._repetition_penalty_scaling(x.float(), state, rep)
        return logits / temp.reshape(1, 1)

    vocab = tcs.VOCAB_SIZE
    static_in = raw_a.clone().reshape(1, vocab)
    # Warmup (also compiles the kernel) and sanity-check the fused path is live.
    warm = tcs._rep_pen_temp_fused(static_in, state, rep, temp)
    assert warm is not None
    assert torch.equal(warm, _eager(raw_a))

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        static_out = tcs._rep_pen_temp_fused(static_in, state, rep, temp)
    # Graph capture records ops without executing them, so static_out is only
    # meaningful after the first replay.

    # Replay with a different input; output must match eager on raw_b.
    static_in.copy_(raw_b)
    graph.replay()
    torch.npu.synchronize()
    assert torch.equal(static_out, _eager(raw_b))


class _BoomKernel:
    """Stand-in for the jitted kernel whose compile/launch always fails."""

    def __getitem__(self, grid):
        def _launch(*args, **kwargs):
            raise RuntimeError("simulated Triton compile failure")

        return _launch


@pytest.mark.skipif(not _npu_and_triton_available(), reason="requires NPU + Triton")
def test_fused_prelude_config_switch_forces_eager():
    """``fused_prelude=False`` must produce exactly the eager chain output."""
    device = _device()
    state = _make_state([11, 22, 33, 44])
    raw = torch.randn(1, tcs.VOCAB_SIZE, device=device)
    raw[:, 0] = 0.0
    common = dict(
        state=state,
        min_tokens=torch.tensor([0], dtype=torch.int32, device=device),
        temperature=torch.tensor([0.9], dtype=torch.float32, device=device),
        repetition_penalty=torch.tensor([1.2], dtype=torch.float32, device=device),
        eos_token_id=0,
        top_k=0,
        top_p=1.0,
        min_p=0.0,
    )
    original = tcs._FUSED_PRELUDE_OK
    try:
        tcs._FUSED_PRELUDE_OK = False
        eager = tcs.prepare_codec_logits(raw.clone(), **common)
        tcs._FUSED_PRELUDE_OK = True
        switched = tcs.prepare_codec_logits(raw.clone(), fused_prelude=False, **common)
        fused = tcs.prepare_codec_logits(raw.clone(), **common)
    finally:
        tcs._FUSED_PRELUDE_OK = original
    assert torch.equal(switched, eager)
    assert torch.equal(fused, eager)


@pytest.mark.skipif(not _npu_and_triton_available(), reason="requires NPU + Triton")
def test_fused_prelude_latches_off_after_kernel_failure(monkeypatch):
    """A failing kernel returns None, latches off and leaves the eager path intact."""
    device = _device()
    state = _make_state([5, 6, 7])
    raw = torch.randn(1, tcs.VOCAB_SIZE, device=device)
    raw[:, 0] = 0.0
    rep = torch.tensor([1.1], dtype=torch.float32, device=device)
    temp = torch.tensor([1.0], dtype=torch.float32, device=device)

    monkeypatch.setattr(tcs, "_FUSED_PRELUDE_OK", True)
    monkeypatch.setattr(tcs, "_rep_pen_temp_kernel", _BoomKernel())
    assert tcs._rep_pen_temp_fused(raw.clone(), state, rep, temp) is None
    assert tcs._FUSED_PRELUDE_OK is False

    out = tcs.prepare_codec_logits(
        raw.clone(),
        state=state,
        min_tokens=torch.tensor([0], dtype=torch.int32, device=device),
        temperature=temp,
        repetition_penalty=rep,
        eos_token_id=0,
        top_k=0,
        top_p=1.0,
        min_p=0.0,
    )
    assert out.shape == raw.shape
    assert torch.isfinite(out).all()


@pytest.mark.skipif(not _npu_and_triton_available(), reason="requires NPU + Triton")
def test_warmup_fused_prelude_is_idempotent(monkeypatch):
    """Warmup compiles once; a second call is a no-op."""
    monkeypatch.setattr(tcs, "_FUSED_PRELUDE_WARMED", False)
    tcs.warmup_fused_prelude(_device())
    assert tcs._FUSED_PRELUDE_WARMED is True
    tcs.warmup_fused_prelude(_device())


def test_warmup_disabled_never_touches_the_device():
    """An eager-only configuration must not compile or launch the kernel."""

    class _Bomb:
        @property
        def type(self):
            raise AssertionError("disabled warmup touched the device")

    tcs.warmup_fused_prelude(_Bomb(), enabled=False)


def test_bool_flag_parses_strings_strictly():
    """Quoted YAML/CLI values must not silently fall back to truthiness."""
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import _bool_flag

    assert _bool_flag(True, "k") is True
    assert _bool_flag(False, "k") is False
    assert _bool_flag(1, "k") is True
    assert _bool_flag(0, "k") is False
    assert _bool_flag("true", "k") is True
    assert _bool_flag(" FALSE ", "k") is False
    assert _bool_flag("1", "k") is True
    assert _bool_flag("0", "k") is False
    for bogus in ("yes", "", "fa1se", None, 2):
        with pytest.raises(ValueError):
            _bool_flag(bogus, "k")


def test_extra_config_reaches_the_switch_from_the_connector_dict():
    """The dict shape ``arg_utils`` builds must resolve to the switch value."""
    from types import SimpleNamespace

    from vllm_omni.model_executor.models.minicpmo_4_5 import minicpmo_4_5_omni_tts as tts_mod

    holder = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                stage_connector_config={
                    "name": "SharedMemoryConnector",
                    "extra": {"talker_codec_fused_prelude": "false"},
                }
            )
        )
    )
    extra = tts_mod.MiniCPMO45OmniTTSForConditionalGeneration._extra_config(holder)
    assert extra["talker_codec_fused_prelude"] == "false"
    assert tts_mod._bool_flag(extra["talker_codec_fused_prelude"], "talker_codec_fused_prelude") is False


def _eager_top_k_top_p(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """The eager chain ``_topk_topp_fused`` replaces (talker_codec_sample.py)."""
    out = logits.clone()
    if top_k > 0:
        keep = min(tcs.VOCAB_SIZE, int(top_k))
        threshold = torch.topk(out, keep, dim=-1).values[..., -1, None]
        out.masked_fill_(out < threshold, float("-inf"))
    if 0.0 < float(top_p) < 1.0:
        sorted_logits, sorted_indices = torch.sort(out, descending=False, dim=-1)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative_probs <= (1.0 - float(top_p))
        remove[..., -1:] = False
        remove = remove.scatter(-1, sorted_indices, remove)
        out.masked_fill_(remove, float("-inf"))
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.skipif(not _npu_and_triton_available(), reason="requires NPU + Triton")
def test_fused_tail_matches_eager_top_k_top_p(seed):
    """The fused tail must keep exactly the eager chain's candidates and logits."""
    torch.manual_seed(seed)
    logits = torch.randn(1, tcs.VOCAB_SIZE, device=_device())
    fused = tcs._topk_topp_fused(logits.clone(), 25, 0.85)
    assert fused is not None
    eager = _eager_top_k_top_p(logits, 25, 0.85)
    kept_fused = torch.isfinite(fused)
    kept_eager = torch.isfinite(eager)
    assert torch.equal(kept_fused, kept_eager)
    assert torch.equal(fused[kept_fused], eager[kept_eager])


@pytest.mark.skipif(not _npu_and_triton_available(), reason="requires NPU + Triton")
def test_fused_tail_falls_back_for_batched_logits():
    """The kernel masks one row; batched logits must fall back to the eager chain."""
    logits = torch.randn(2, tcs.VOCAB_SIZE, device=_device())
    assert tcs._topk_topp_fused(logits, 25, 0.85) is None


@pytest.mark.skipif(not _npu_and_triton_available(), reason="requires NPU + Triton")
def test_fused_tail_falls_back_above_the_candidate_cap():
    """A candidate set the kernel cannot hold must fall back, not get truncated."""
    logits = torch.randn(1, tcs.VOCAB_SIZE, device=_device())
    assert tcs._topk_topp_fused(logits, tcs._TAIL_MAX_CANDIDATES + 1, 0.85) is None


@pytest.mark.skipif(not _npu_and_triton_available(), reason="requires NPU + Triton")
def test_fused_tail_defers_a_single_candidate_to_eager():
    """top_k == 1 must never reach the kernel: KP == 1 aborts the Ascend compiler."""
    logits = torch.randn(1, tcs.VOCAB_SIZE, device=_device())
    assert tcs._topk_topp_fused(logits, 1, 0.85) is None


@pytest.mark.parametrize("top_k,top_p", [(25, 1.0), (tcs._TAIL_MAX_CANDIDATES, 0.85)])
@pytest.mark.skipif(not _npu_and_triton_available(), reason="requires NPU + Triton")
def test_fused_tail_matches_eager_at_the_parameter_edges(top_k, top_p):
    """top_p == 1.0 and top_k == 1 / the cap must keep exactly the eager set."""
    torch.manual_seed(7)
    logits = torch.randn(1, tcs.VOCAB_SIZE, device=_device())
    fused = tcs._topk_topp_fused(logits.clone(), top_k, top_p)
    assert fused is not None
    eager = _eager_top_k_top_p(logits, top_k, top_p)
    kept_fused = torch.isfinite(fused)
    kept_eager = torch.isfinite(eager)
    assert torch.equal(kept_fused, kept_eager)
    assert torch.equal(fused[kept_fused], eager[kept_eager])


def _tied_logits(name: str) -> torch.Tensor:
    """One B=1 row whose finite logits contain exact ties."""
    vocab = tcs.VOCAB_SIZE
    logits = torch.full((1, vocab), -20.0, device=_device())
    if name == "all_tied":
        logits[0, :25] = 0.0
    elif name == "partial_tie":
        logits[0, :10] = 1.0
        logits[0, 10:25] = torch.linspace(0.5, 0.1, 15, device=_device())
    else:  # pragma: no cover - guards the caller below
        raise AssertionError(name)
    return logits


@pytest.mark.parametrize("top_p", [0.5, 0.85, 1.0])
@pytest.mark.parametrize("name", ["all_tied", "partial_tie"])
@pytest.mark.skipif(not _npu_and_triton_available(), reason="requires NPU + Triton")
def test_fused_tail_keeps_a_valid_distribution_on_tied_logits(name, top_p):
    """Tied finite logits must keep the eager support, not collapse to all ``-inf``.

    ``smaller`` used to sum strictly-smaller candidates only, so 25 equal logits
    with ``top_p=0.85`` left every candidate at ``smaller == 0`` and the kernel
    returned a row without support, whose softmax is NaN.
    """
    logits = _tied_logits(name)
    fused = tcs._topk_topp_fused(logits.clone(), 25, top_p)
    assert fused is not None
    kept = torch.isfinite(fused)
    assert bool(kept.any()), f"{name}/{top_p}: fused mask has no support"
    probs = torch.softmax(fused, dim=-1)
    assert not bool(torch.isnan(probs).any()), f"{name}/{top_p}: softmax is NaN"
    assert torch.allclose(probs.sum(-1), torch.ones(1, device=probs.device), atol=1e-5)
    eager = _eager_top_k_top_p(logits, 25, top_p)
    kept_eager = torch.isfinite(eager)
    # Membership inside a tie group is order-dependent -- ``torch.topk``'s here,
    # ``torch.sort(stable=False)``'s in the reference -- so compare the kept
    # count and the sorted kept values, which the kernel guarantees by
    # construction, not the identity of the kept members.
    assert int(kept.sum()) == int(kept_eager.sum()), f"{name}/{top_p}: kept count differs"
    assert torch.equal(
        torch.sort(fused[kept]).values, torch.sort(eager[kept_eager]).values
    ), f"{name}/{top_p}: kept logits differ"
