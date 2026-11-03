# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

# Token2Wav imports the optional `step-audio2` extras (flashcosyvoice / s3tokenizer
# / hyperpyyaml) at module load. Skip instead of erroring at collection time on CI
# images that do not install them; a genuine import error still surfaces.
try:
    from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import (
        BatchedToken2WavState,
    )
    from vllm_omni.model_executor.models.step_audio2.step_audio2_token2wav import (
        _BATCHED_PROMPT_CACHE_ID,
        StepAudio2Token2WavForConditionalGeneration,
    )
except ModuleNotFoundError as exc:
    pytest.skip(
        f"Step-Audio2 Token2Wav extras not installed ({exc.name})",
        allow_module_level=True,
    )

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _DummyModelConfig:
    def __init__(self, **hf_overrides) -> None:
        self.hf_config = SimpleNamespace(
            token2wav_path="/tmp/fake-token2wav",
            token2wav_float16=False,
            **hf_overrides,
        )
        self.model = "/tmp/fake-model"

    def get_hidden_size(self) -> int:
        return 8


def _build_model(**hf_overrides) -> StepAudio2Token2WavForConditionalGeneration:
    vllm_config = SimpleNamespace(
        model_config=_DummyModelConfig(**hf_overrides),
        device_config=SimpleNamespace(device="cpu"),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
    )
    return StepAudio2Token2WavForConditionalGeneration(vllm_config=vllm_config)


def _armed_model(backend: Mock) -> StepAudio2Token2WavForConditionalGeneration:
    """A model with the knob on and a mocked backend already built."""
    model = _build_model(token2wav_batched_backend=True)
    model._batched_backend = backend
    model._batched_precapture_pending = False
    return model


def _fake_state(tag: int = 0) -> BatchedToken2WavState:
    return BatchedToken2WavState(
        flow_cache={"estimator_cnn_cache": torch.zeros(1, 2, 3 + tag)},
        hift_cache={"source": torch.zeros(1)},
    )


def _fake_audio(samples: int = 100) -> torch.Tensor:
    return torch.zeros(samples, dtype=torch.float32)


def _mock_backend() -> Mock:
    backend = Mock()
    backend.prepare_prompt.return_value = object()
    backend.setup_batch.side_effect = lambda features, n: [_fake_state() for _ in range(n)]
    return backend


def _chunk_info(request_id: str, tokens: int, last: bool) -> dict:
    return {
        "meta": {
            "left_context_size": 1 if last else 0,
            "last_chunk": last,
            "request_id": request_id,
            "code_flat_numel": tokens,
        }
    }


def _forward(model, rows, *, runner_counts=True, request_ids=None):
    """rows: list of (request_id, token_count, last); None token_count = no payload.

    ``runner_counts=False`` omits the ``seq_token_counts`` kwarg so the meta
    fallback paths are exercised.
    """
    ids, infos, counts = [], [], []
    for request_id, tokens, last in rows:
        if tokens is None:
            infos.append({"other": "bookkeeping"})
            counts.append(0)
            continue
        ids.extend([7] * tokens)
        infos.append(_chunk_info(request_id, tokens, last))
        counts.append(tokens)
    input_ids = torch.tensor(ids, dtype=torch.long)
    kwargs = {}
    if runner_counts:
        kwargs["seq_token_counts"] = counts
    if request_ids is not None:
        kwargs["request_ids"] = request_ids
    return model.forward(
        input_ids=input_ids,
        positions=torch.zeros(max(len(ids), 1), dtype=torch.long),
        runtime_additional_information=infos,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Knob resolution
# ---------------------------------------------------------------------------


def test_batched_knob_defaults_off_and_capture_is_noop():
    model = _build_model()

    assert model._batched_enabled is False
    assert model._batched_backend is None
    assert getattr(model, "requires_exact_input_shape", False) is False
    assert getattr(model, "requires_request_ids", False) is False
    # Must not raise or build anything for the default (knob-off) model.
    model.capture_auxiliary_graphs()
    assert model._batched_backend is None


def test_batched_knob_invalid_value_fails_at_startup():
    with pytest.raises(ValueError, match="token2wav_batched_backend"):
        _build_model(token2wav_batched_backend="maybe")


def test_batched_knob_on_sets_runner_flags():
    model = _build_model(token2wav_batched_backend=True)

    assert model._batched_enabled is True
    assert model._batched_tf32 is True
    assert model.requires_exact_input_shape is True
    assert model.requires_request_ids is True
    assert model._batched_states_cap == 16  # max(16, 2 * max_num_seqs=8)


# ---------------------------------------------------------------------------
# Flat-token splitting
# ---------------------------------------------------------------------------


def test_split_flat_tokens_splits_by_counts():
    flat = torch.arange(10, dtype=torch.long)
    segments = StepAudio2Token2WavForConditionalGeneration._split_flat_tokens(flat, [3, 7], "counts")

    assert [int(s.numel()) for s in segments] == [3, 7]
    assert segments[1].tolist() == list(range(3, 10))


def test_split_flat_tokens_rejects_count_mismatch():
    flat = torch.arange(10, dtype=torch.long)
    with pytest.raises(RuntimeError, match="sum 9 != 10"):
        StepAudio2Token2WavForConditionalGeneration._split_flat_tokens(flat, [3, 6], "counts")


def test_split_flat_tokens_rejects_negative_and_non_sequence():
    flat = torch.arange(4, dtype=torch.long)
    with pytest.raises(RuntimeError, match="negative"):
        StepAudio2Token2WavForConditionalGeneration._split_flat_tokens(flat, [-1, 5], "counts")
    with pytest.raises(RuntimeError, match="not a sequence"):
        StepAudio2Token2WavForConditionalGeneration._split_flat_tokens(flat, 4, "counts")
    with pytest.raises(RuntimeError, match="invalid"):
        StepAudio2Token2WavForConditionalGeneration._split_flat_tokens(flat, "abc", "counts")


# ---------------------------------------------------------------------------
# Batched async-chunk forward (mocked backend)
# ---------------------------------------------------------------------------


def test_batched_forward_streams_state_and_drops_on_last_chunk(monkeypatch):
    monkeypatch.setattr("os.path.exists", lambda _: True)
    backend = _mock_backend()
    setup_state = _fake_state(0)
    state_b, state_c, state_final = _fake_state(0), _fake_state(0), _fake_state(0)
    backend.setup_batch.side_effect = lambda features, n: [setup_state]
    decoded_states = iter([state_b, state_c, state_final])
    backend.decode_batch.side_effect = lambda tokens, features, states, last_chunk: (
        [_fake_audio(200)],
        [next(decoded_states)],
    )
    model = _armed_model(backend)

    _forward(model, [("r1", 28, False)])
    # The first decode consumed the setup state and stored the decoded one.
    assert backend.decode_batch.call_args.args[2] == [setup_state]
    assert model._batched_states["r1"] is state_b
    call = backend.decode_batch.call_args
    assert call.args[0].shape == (1, 28)
    assert call.kwargs["last_chunk"] is False

    _forward(model, [("r1", 28, False)])
    # The second step decodes from the stored state, not a fresh setup.
    assert backend.decode_batch.call_args.args[2] == [state_b]
    assert model._batched_states["r1"] is state_c

    out = _forward(model, [("r1", 10, True)])
    assert "r1" not in model._batched_states
    assert backend.decode_batch.call_args.kwargs["last_chunk"] is True
    audio = out.multimodal_outputs["model_outputs"]
    assert len(audio) == 1 and int(audio[0].numel()) == 200


def test_batched_forward_uniform_rows_take_stacked_decode_batch(monkeypatch):
    monkeypatch.setattr("os.path.exists", lambda _: True)
    backend = _mock_backend()
    backend.decode_batch.return_value = ([_fake_audio(), _fake_audio()], [_fake_state(), _fake_state()])
    model = _armed_model(backend)

    _forward(model, [("a", 28, False), ("b", 28, False)])

    backend.decode_ragged_batch.assert_not_called()
    assert backend.decode_batch.call_args.args[0].shape == (2, 28)
    assert set(model._batched_states) == {"a", "b"}


def test_batched_forward_mixed_rows_take_ragged(monkeypatch):
    monkeypatch.setattr("os.path.exists", lambda _: True)
    backend = _mock_backend()
    backend.decode_ragged_batch.return_value = ([_fake_audio(), _fake_audio()], [_fake_state(), _fake_state()])
    model = _armed_model(backend)

    # Mixed widths (steady 28 next to a short final chunk) must go ragged.
    _forward(model, [("a", 28, False), ("b", 10, True)])

    backend.decode_batch.assert_not_called()
    call = backend.decode_ragged_batch.call_args
    assert [int(t.numel()) for t in call.args[0]] == [28, 10]
    assert call.kwargs["last_chunks"] == [False, True]
    assert set(model._batched_states) == {"a"}  # b finished and dropped


def test_batched_forward_empty_eof_marker_drops_state_without_decode(monkeypatch):
    monkeypatch.setattr("os.path.exists", lambda _: True)
    backend = _mock_backend()
    model = _armed_model(backend)
    model._batched_states["r1"] = _fake_state()

    out = _forward(model, [("r1", 0, True)])

    backend.decode_batch.assert_not_called()
    backend.decode_ragged_batch.assert_not_called()
    assert "r1" not in model._batched_states
    audio = out.multimodal_outputs["model_outputs"]
    assert len(audio) == 1 and int(audio[0].numel()) == 1


def test_batched_forward_nonpayload_entries_get_placeholders(monkeypatch):
    monkeypatch.setattr("os.path.exists", lambda _: True)
    backend = _mock_backend()
    backend.decode_batch.return_value = ([_fake_audio()], [_fake_state()])
    model = _armed_model(backend)

    out = _forward(model, [("a", None, None), ("b", 28, False), ("c", None, None)])

    audio = out.multimodal_outputs["model_outputs"]
    assert len(audio) == 3
    assert int(audio[0].numel()) == 1 and int(audio[2].numel()) == 1
    assert int(audio[1].numel()) == 100


def test_batched_forward_splits_by_meta_counts_without_runner_counts(monkeypatch):
    monkeypatch.setattr("os.path.exists", lambda _: True)
    backend = _mock_backend()
    backend.decode_ragged_batch.return_value = ([_fake_audio(), _fake_audio()], [_fake_state(), _fake_state()])
    model = _armed_model(backend)

    # No seq_token_counts kwarg: the per-payload code_flat_numel meta splits the flat ids.
    _forward(model, [("a", 28, False), ("b", 10, True)], runner_counts=False)

    call = backend.decode_ragged_batch.call_args
    assert [int(t.numel()) for t in call.args[0]] == [28, 10]


def test_batched_forward_requires_counts_for_multiple_legacy_payloads(monkeypatch):
    monkeypatch.setattr("os.path.exists", lambda _: True)
    backend = _mock_backend()
    model = _armed_model(backend)

    # Legacy payloads (marker only, no code_flat_numel) cannot be split
    # across requests without runner counts.
    with pytest.raises(RuntimeError, match="cannot split input_ids"):
        model.forward(
            input_ids=torch.zeros(56, dtype=torch.long),
            positions=torch.zeros(56, dtype=torch.long),
            runtime_additional_information=[
                {"meta": {"left_context_size": 0}},
                {"meta": {"left_context_size": 0}},
            ],
        )


def test_batched_forward_single_payload_owns_all_tokens_without_counts(monkeypatch):
    monkeypatch.setattr("os.path.exists", lambda _: True)
    backend = _mock_backend()
    backend.decode_batch.return_value = ([_fake_audio()], [_fake_state()])
    model = _armed_model(backend)

    _forward(model, [("b", 28, False), ("a", None, None)], runner_counts=False)

    assert backend.decode_batch.call_args.args[0].shape == (1, 28)


def test_batched_forward_decode_failure_drops_involved_states(monkeypatch):
    monkeypatch.setattr("os.path.exists", lambda _: True)
    backend = _mock_backend()
    backend.decode_batch.side_effect = RuntimeError("backend boom")
    model = _armed_model(backend)
    model._batched_states["a"] = _fake_state()

    with pytest.raises(RuntimeError, match="backend boom"):
        _forward(model, [("a", 28, False)])

    # A retry must start from a fresh setup instead of a half-consumed cache.
    assert "a" not in model._batched_states


def test_batched_state_lru_cap_evicts_oldest(monkeypatch):
    monkeypatch.setattr("os.path.exists", lambda _: True)
    backend = _mock_backend()
    backend.decode_batch.return_value = ([_fake_audio()], [_fake_state()])
    model = _armed_model(backend)
    model._batched_states_cap = 2

    for request_id in ("r1", "r2", "r3"):
        _forward(model, [(request_id, 28, False)])

    assert set(model._batched_states) == {"r2", "r3"}


def test_batched_forward_uses_default_prompt_cache_id(monkeypatch):
    monkeypatch.setattr("os.path.exists", lambda _: True)
    backend = _mock_backend()
    backend.decode_batch.return_value = ([_fake_audio()], [_fake_state()])
    model = _armed_model(backend)

    _forward(model, [("a", 28, False)])

    assert backend.prepare_prompt.call_args.args[0] == _BATCHED_PROMPT_CACHE_ID


# ---------------------------------------------------------------------------
# Graph warmup hook
# ---------------------------------------------------------------------------


def test_capture_auxiliary_graphs_sweeps_once_then_noop():
    model = _build_model(token2wav_batched_backend=True)
    graph = SimpleNamespace(_failed=False)
    backend = SimpleNamespace(
        _encoder_graphs=graph,
        _chunk_encoder_graph=graph,
        hift_graph_wrapper=graph,
        _cfm_graph_wrapper=graph,
        _whole_euler_graph_wrapper=graph,
        precapture_hift=lambda: 2,
        prepare_prompt=lambda *args: object(),
        precapture_whole_euler=lambda features: 3,
    )
    model._batched_backend = backend
    monkeypatch_exists = pytest.MonkeyPatch()
    monkeypatch_exists.setattr("os.path.exists", lambda _: True)
    try:
        model.capture_auxiliary_graphs()
        # Second call is a no-op (the pending flag cleared by the first).
        model.capture_auxiliary_graphs()
    finally:
        monkeypatch_exists.undo()


def test_capture_auxiliary_graphs_poisoned_wrapper_fails_startup():
    model = _build_model(token2wav_batched_backend=True)
    graph = SimpleNamespace(_failed=False)
    backend = SimpleNamespace(
        _encoder_graphs=graph,
        _chunk_encoder_graph=graph,
        hift_graph_wrapper=graph,
        _cfm_graph_wrapper=graph,
        _whole_euler_graph_wrapper=graph,
        precapture_hift=lambda: 0,
        prepare_prompt=lambda *args: object(),
        precapture_whole_euler=lambda features: (_ for _ in ()).throw(RuntimeError("capture failed")),
    )
    backend._whole_euler_graph_wrapper._failed = True
    model._batched_backend = backend
    monkeypatch_exists = pytest.MonkeyPatch()
    monkeypatch_exists.setattr("os.path.exists", lambda _: True)
    try:
        with pytest.raises(RuntimeError, match="capture failed"):
            model.capture_auxiliary_graphs()
    finally:
        monkeypatch_exists.undo()


def test_capture_auxiliary_graphs_knob_off_never_builds():
    model = _build_model()
    model.capture_auxiliary_graphs()
    assert model._batched_backend is None
