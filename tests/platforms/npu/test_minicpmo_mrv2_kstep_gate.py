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

from vllm_omni.platforms.npu.worker import npu_omni_v2_model_runner as runner_mod

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
    input_batch = types.SimpleNamespace(has_prefill=has_prefill, vocab_size=6562)
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

    calls = {}

    def fake_run(**kwargs):
        calls.update(kwargs)
        return "frame-loop-output"

    monkeypatch.setattr(multiframe, "run", fake_run)

    spans = [(i * 8, (i + 1) * 8) for i in range(2)]
    infos = [{"req_id": str(i)} for i in range(2)]
    ns, input_batch, model_inputs = _make_runner(spans, infos)
    out = _run(ns, input_batch, model_inputs)
    assert out == "frame-loop-output"
    assert calls["frames"] == 8
    # The stop-row vocab sync ran against the real input batch.
    assert input_batch.vocab_size == 2
