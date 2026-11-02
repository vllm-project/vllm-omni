# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA entry points into the Talker's vertical K-frame loop.

The loop itself is ``talker_multiframe.run`` (main's own NPU-side module, pure
Python + torch). CUDA only adds two hooks:

* ``gpu_model_runner.OmniGPUModelRunner._model_forward`` calls the loop when a
  step schedules K positions per request;
* ``gpu_ar_model_runner.GPUARModelRunner.propose_draft_token_ids`` books those
  K positions with the scheduler by emitting ``continue`` drafts.

These tests pin the properties the hooks rely on, without a GPU.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.platforms.npu.worker import talker_multiframe

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _mf_model(stage: str = "tts"):
    """Wrapper surface the multi-frame contract check demands.

    ``supports_multi_frame_decode`` is not a bare flag: when it reads True the
    contract check requires the four per-frame hooks to exist, otherwise the
    loop could run against silent defaults.
    """
    return SimpleNamespace(
        supports_multi_frame_decode=(stage == "tts"),
        model_stage=stage,
        batch_stop_logits=None,
        take_batch_stop_logits=lambda: None,
        set_batch_stop_logits=lambda logits: None,
        merge_frame_outputs=lambda outputs, stop_logits: outputs[-1],
    )


class _DraftedRunner:
    """Minimal runner surface ``drafts_this_step`` reads."""

    def __init__(self, num_spec_tokens: int, stage: str = "tts"):
        self.num_spec_tokens = num_spec_tokens
        self.model = _mf_model(stage)

    def _kstep_constant_draft_tensor(self, rows: int, drafts: int) -> torch.Tensor:
        """Mirror of the runner's cached CONTINUE tensor, kept shape-faithful."""
        return torch.full(
            (rows, drafts),
            talker_multiframe.CONTINUE_TOKEN_ID,
            dtype=torch.int32,
            device="cpu",
        )


def test_drafts_this_step_is_k_frames():
    """K = num_speculative_tokens + 1, and 0 when the stage is not the Talker."""
    assert talker_multiframe.drafts_this_step(_DraftedRunner(7)) == 8
    assert talker_multiframe.drafts_this_step(_DraftedRunner(3)) == 4
    assert talker_multiframe.drafts_this_step(_DraftedRunner(0)) == 0
    assert talker_multiframe.drafts_this_step(_DraftedRunner(7, stage="thinker")) == 0


def test_constant_drafts_fill_every_row_on_a_drafted_step():
    """A full drafted step books K-1 positions per request, all `continue`."""
    continue_id = talker_multiframe.STOP_TOKEN_ID - 1  # 0 is `continue`
    drafted = talker_multiframe.constant_drafts(
        valid_sampled_token_ids=[[0], [0], [0], [0]],
        frames=8,
        num_reqs=4,
    )
    assert len(drafted) == 4
    for row in drafted:
        assert len(row) == 7, "K-1 draft positions per request"
        assert set(row) == {continue_id}


def test_constant_drafts_fold_when_a_row_sampled_nothing():
    """An empty row is a stop firing mid-span: the batch folds, nobody drafts."""
    drafted = talker_multiframe.constant_drafts(
        valid_sampled_token_ids=[[0], [], [0], [0]],
        frames=8,
        num_reqs=4,
    )
    assert not any(drafted), "mixed drafted/undrafted rows must not share a span"


def test_constant_drafts_fold_on_a_stop_truncated_row():
    """A live short row (stop fired mid-span) folds the batch.

    The split needs the scheduler's per-request draft counts: without them
    every row is assumed to sit on a drafted step, so a short row is read as
    scheduler padding and does keep drafting.
    """
    drafted = talker_multiframe.constant_drafts(
        valid_sampled_token_ids=[[0, 0, 0, 0], [0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        frames=8,
        num_reqs=4,
        scheduled_draft_counts=[8, 8, 8, 8],
        fold_short_rows=True,
    )
    assert not any(drafted), "a stop-truncated row must fold the batch"


def test_applies_returns_frames_for_a_uniform_decode_step():
    """`applies` is the gate the CUDA hook relies on: K uniform spans -> K."""
    model = _mf_model()
    kwargs = {
        "request_token_spans": [(0, 8), (8, 16)],
        "model_intermediate_buffer": [
            {"_omni_is_prefill": False, "audio_state": {"step": 5}},
            {"_omni_is_prefill": False, "audio_state": {"step": 3}},
        ],
    }
    assert talker_multiframe.applies(model, kwargs) == 8


def test_applies_refuses_non_uniform_spans():
    model = _mf_model()
    kwargs = {
        "request_token_spans": [(0, 8), (8, 9)],
        "model_intermediate_buffer": [
            {"_omni_is_prefill": False, "audio_state": {"step": 5}},
            {"_omni_is_prefill": False, "audio_state": {"step": 3}},
        ],
    }
    assert talker_multiframe.applies(model, kwargs) == 0


def test_applies_skips_a_stage_without_multi_frame_support():
    model = _mf_model(stage="thinker")
    assert talker_multiframe.applies(model, {}) == 0


@pytest.mark.parametrize("short", [False, True])
def test_gpu_propose_uses_scheduled_width_for_tensor_rows(short):
    import torch

    from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

    runner = _DraftedRunner(3)
    runner.input_batch = SimpleNamespace(num_reqs=2, req_ids=["a", "b"])
    scheduler = SimpleNamespace(scheduled_spec_decode_tokens={"a": [0] * 3, "b": [0] * 3})
    samples = torch.tensor([[0, 0, 0, 0], [0, 0, -1, -1] if short else [0, 0, 0, 0]])
    drafts = GPUARModelRunner.propose_draft_token_ids(runner, scheduler, samples, None, None, None)
    # Under async scheduling the draft must be a device tensor
    # (torch.int32): `_prepare_input_ids` asserts isinstance(..., torch.Tensor)
    # and the async event copy only arms for tensors.
    assert isinstance(drafts, torch.Tensor)
    assert drafts.dtype == torch.int32
    if short:
        assert drafts.shape == (2, 0)
    else:
        assert drafts.shape == (2, 3)
        assert drafts.tolist() == [[0] * 3, [0] * 3]


@pytest.mark.parametrize("enabled", [False, True])
def test_gpu_sample_tokens_neutralizes_only_kstep_control_head(enabled):
    from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

    runner = _DraftedRunner(3)
    if not enabled:
        runner.model = object()

    class FakeMinTokensLogitsProcessor:
        def __init__(self):
            self.min_toks = {"r1": 8}

    processor = FakeMinTokensLogitsProcessor()
    runner.input_batch = SimpleNamespace(
        sampling_metadata=SimpleNamespace(logitsprocs=SimpleNamespace(non_argmax_invariant=[processor]))
    )
    runner.execute_model_state = (None,) * 13

    class SampleReachedError(Exception):
        pass

    def sample(*args):
        raise SampleReachedError

    runner._sample = sample
    with pytest.raises(SampleReachedError):
        GPUARModelRunner.sample_tokens(runner, None)
    assert processor.min_toks == ({} if enabled else {"r1": 8})


def test_kstep_frame_rows_picks_the_row_the_loop_samples():
    """The per-frame projection must gather ``end - 1`` of each span.

    The sampling loop reads ``hidden[end - 1 : end]``, so projecting ``start``
    instead would silently score the wrong row once a span is wider than one.
    """
    import torch

    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
        kstep_frame_rows,
    )

    # One-row spans (what the K-frame loop passes per frame).
    assert kstep_frame_rows([(0, 1), (1, 2), (2, 3)], torch.device("cpu")).tolist() == [0, 1, 2]
    # Wider spans (a step that scheduled several positions per request).
    assert kstep_frame_rows([(0, 8), (8, 16)], torch.device("cpu")).tolist() == [7, 15]


@pytest.mark.parametrize("fail", [False, True])
def test_cuda_runner_restores_narrow_context(monkeypatch, fail):
    import torch
    import vllm.forward_context

    from vllm_omni.worker import talker_cuda_graph
    from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

    owner = SimpleNamespace(
        model=_mf_model(),
        input_batch=SimpleNamespace(num_reqs=1),
    )
    extra = {
        "request_token_spans": [(0, 2)],
        "model_intermediate_buffer": [{"audio_state": {"step": 1}}],
    }
    owner._build_model_kwargs_extra = lambda: extra
    restored = []
    narrow = SimpleNamespace(restore=lambda: restored.append(True))
    context = object()
    monkeypatch.setattr(vllm.forward_context, "get_forward_context", lambda: context)

    def begin(**kwargs):
        assert kwargs["graph_owner"] is owner
        assert kwargs["forward_context"] is context
        return narrow

    monkeypatch.setattr(talker_cuda_graph, "begin_cuda_narrow_step", begin)
    output = SimpleNamespace(text_hidden_states=torch.zeros(2, 2))

    def run(**kwargs):
        assert kwargs["narrow"] is narrow
        if fail:
            raise RuntimeError("forward failed")
        return output

    monkeypatch.setattr(talker_multiframe, "run", run)
    monkeypatch.setattr(talker_multiframe, "ensure_stop_token_vocab", lambda *args: None)
    if fail:
        with pytest.raises(RuntimeError, match="forward failed"):
            OmniGPUModelRunner._model_forward(owner)
    else:
        assert OmniGPUModelRunner._model_forward(owner) is output
        assert owner._omni_last_model_output is output
    assert restored == [True]
