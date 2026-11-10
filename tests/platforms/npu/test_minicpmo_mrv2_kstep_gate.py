# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Gate tests for the MRv2 K-step frame loop (npu_omni_v2_model_runner).

The gate must separate benign declines (prefill steps, warmup/sizing batches,
one-token decode rows) -- where a single forward is the correct path -- from
corrupting ones (a K-step-armed decode step with inconsistent spans), where
falling back to a single forward would sample one frame per span and desync
both the codec stream and the scheduler's accounting. V1 raises in the
latter case.
"""

import types

import pytest

pytest.importorskip("vllm_ascend")

from tests.helpers.mark import hardware_marks
from vllm_omni.platforms.npu.worker import npu_omni_v2_model_runner as runner_mod

pytestmark = [pytest.mark.core_model, pytest.mark.omni, *hardware_marks(res={"npu": "A3"}, num_cards=1)]

KSTEP_SPEC = {"method": "ngram", "num_speculative_tokens": 7}  # K = 8


def _make_runner(spans, infos, has_prefill=False):
    ns = types.SimpleNamespace()
    ns.vllm_config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(model_stage="tts"),
        speculative_config=KSTEP_SPEC,
    )
    ns.model_state = types.SimpleNamespace(
        _request_token_spans=lambda input_batch: spans,
    )
    ns.model = types.SimpleNamespace()  # no .talker: codec_model falls back
    # A real MRv2 step's batch carries no vocab_size field (the sampling
    # vocab lives on model_config); the fake mirrors that.
    input_batch = types.SimpleNamespace(has_prefill=has_prefill)
    model_inputs = {"model_intermediate_buffer": infos}
    return ns, input_batch, model_inputs


def _run(ns, input_batch, model_inputs, run_model=None):
    func = runner_mod.NPUOmniARModelRunner._run_kstep_frames
    return func(ns, model_inputs, input_batch, run_model)


def test_prefill_step_declines_without_raising():
    # Prefill spans booked at prompt length: single forward is correct.
    spans = [(0, 14), (14, 25)]
    infos = [{"req_id": str(i)} for i in range(2)]
    ns, input_batch, model_inputs = _make_runner(spans, infos, has_prefill=True)
    assert _run(ns, input_batch, model_inputs) is None


def test_decode_width_outside_k_window_declines():
    # Uniform 8x1024 batch (warmup/sizing booked at prompt length): benign.
    spans = [(i * 1024, (i + 1) * 1024) for i in range(8)]
    infos = [{"req_id": str(i)} for i in range(8)]
    ns, input_batch, model_inputs = _make_runner(spans, infos)
    assert _run(ns, input_batch, model_inputs) is None


def test_one_token_decode_rows_decline():
    # Plain single-token decode rows never enter the frame loop.
    spans = [(0, 1), (1, 2), (2, 3)]
    infos = [{"req_id": str(i)} for i in range(3)]
    ns, input_batch, model_inputs = _make_runner(spans, infos)
    assert _run(ns, input_batch, model_inputs) is None


def test_non_uniform_decode_spans_raise():
    # K-step-armed decode with mixed span widths: a single forward would
    # sample one frame from the last row of each span and corrupt the codec
    # stream and the scheduler accounting; this must fail loudly.
    spans = [(0, 8), (8, 15)]
    infos = [{"req_id": str(i)} for i in range(2)]
    ns, input_batch, model_inputs = _make_runner(spans, infos)
    with pytest.raises(RuntimeError, match="multi-token decode step"):
        _run(ns, input_batch, model_inputs)


def test_span_info_count_mismatch_raises():
    spans = [(0, 8), (8, 16)]
    infos = [{"req_id": "0"}]
    ns, input_batch, model_inputs = _make_runner(spans, infos)
    with pytest.raises(RuntimeError, match="multi-token decode step"):
        _run(ns, input_batch, model_inputs)


def test_gate_passes_to_frame_loop(monkeypatch):
    import vllm_omni.platforms.npu.worker.talker_multiframe as multiframe

    seen = {}

    def fake_run(**kwargs):
        seen["frames"] = kwargs["frames"]
        return "frame-loop-output"

    monkeypatch.setattr(multiframe, "run", fake_run)

    spans = [(i * 8, (i + 1) * 8) for i in range(2)]
    infos = [{"req_id": str(i)} for i in range(2)]
    ns, input_batch, model_inputs = _make_runner(spans, infos)
    out = _run(ns, input_batch, model_inputs)
    assert out == "frame-loop-output"
    assert seen["frames"] == 8
    # The per-step arm is on only after a successful frame loop.
    assert ns._kstep_ran_last is True


def test_declined_step_disarms_kstep_state():
    # A width-1 decode step falls back to the plain forward and must not
    # leave per-step K-step state (drafts/postprocess arm) behind.
    spans = [(0, 1), (1, 2), (2, 3)]
    infos = [{"req_id": str(i)} for i in range(3)]
    ns, input_batch, model_inputs = _make_runner(spans, infos)
    ns._kstep_ran_last = True
    assert _run(ns, input_batch, model_inputs) is None
    assert ns._kstep_ran_last is False


def test_prefill_decline_disarms_kstep_state():
    spans = [(0, 14), (14, 25)]
    infos = [{"req_id": str(i)} for i in range(2)]
    ns, input_batch, model_inputs = _make_runner(spans, infos, has_prefill=True)
    ns._kstep_ran_last = True
    assert _run(ns, input_batch, model_inputs) is None
    assert ns._kstep_ran_last is False


def test_propose_draft_tokens_falls_back_after_decline(monkeypatch):
    # After a declined step the runner must defer to the base proposal flow
    # instead of writing speculative CONTINUE rows.
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        runner_mod.OmniARModelRunner,
        "_propose_draft_tokens",
        lambda self, ib, draft, ns_, nr, mm: seen.setdefault("called", (ib, draft)) and None,
    )

    class _FakeRunner(runner_mod.NPUOmniARModelRunner):
        pass

    runner = object.__new__(_FakeRunner)
    runner._kstep_ran_last = False
    draft = types.SimpleNamespace(num_spec_tokens=2)
    out = runner_mod.NPUOmniARModelRunner._propose_draft_tokens(runner, "ib", draft, 1, 0, None)
    assert seen["called"] == ("ib", draft)
    assert out is None  # the fake base returns None; we propagate it


def test_propose_draft_tokens_writes_continue_after_run():
    # After a successful frame loop the K-step arm reserves K-1 KV rows by
    # writing CONTINUE_TOKEN_ID (0) drafts, as V1 does.
    import numpy as np

    class _FakeRunner(runner_mod.NPUOmniARModelRunner):
        pass

    runner = object.__new__(_FakeRunner)
    runner._kstep_ran_last = True
    runner.req_states = types.SimpleNamespace(draft_tokens=np.full((3, 8), 5, dtype=np.int64))
    ib = types.SimpleNamespace(idx_mapping=np.array([0, 1, 2]))
    written = {}

    class _Handler:
        def set_draft_tokens(self, input_batch, arr):
            written["arr"] = arr

    runner.draft_tokens_handler = _Handler()
    draft = types.SimpleNamespace(num_spec_tokens=2)
    runner_mod.NPUOmniARModelRunner._propose_draft_tokens(runner, ib, draft, 1, 0, None)
    assert (written["arr"] == 0).all()
    assert written["arr"].shape == (3, 2)
