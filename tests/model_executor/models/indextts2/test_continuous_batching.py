from __future__ import annotations

from collections import deque
from types import MethodType, SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.indextts2 import indextts2_s2mel_decoder
from vllm_omni.model_executor.models.indextts2.indextts2_s2mel_decoder import (
    IndexTTS2S2MelDecoder,
)
from vllm_omni.model_executor.models.indextts2.s2mel.modules.flow_matching import BASECFM


class _RecordingEstimator(torch.nn.Module):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)
        self.timesteps: list[torch.Tensor] = []
        self.input_shapes: list[tuple[int, ...]] = []
        self.workspace_ptrs: list[tuple[int, int]] = []
        self.conditioning_ptrs: list[tuple[int, ...]] = []

    def forward(
        self,
        x: torch.Tensor,
        prompt_x: torch.Tensor,
        x_lens: torch.Tensor,
        t: torch.Tensor,
        style: torch.Tensor,
        cond: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        pre_mask = kwargs.get("pre_mask")
        unpad_data = kwargs.get("unpad_data")
        self.timesteps.append(t.detach().clone())
        self.input_shapes.append(tuple(x.shape))
        self.workspace_ptrs.append((x.data_ptr(), t.data_ptr()))
        self.conditioning_ptrs.append(
            (
                prompt_x.data_ptr(),
                x_lens.data_ptr(),
                style.data_ptr(),
                cond.data_ptr(),
                pre_mask[0].data_ptr(),
                0 if pre_mask[1] is None else pre_mask[1].data_ptr(),
                0 if unpad_data is None else unpad_data[0].data_ptr(),
                0 if unpad_data is None else unpad_data[1].data_ptr(),
            )
        )
        return torch.ones_like(x) * t[:, None, None]


def _make_cfm() -> tuple[BASECFM, _RecordingEstimator]:
    cfm = object.__new__(BASECFM)
    torch.nn.Module.__init__(cfm)
    estimator = _RecordingEstimator()
    cfm.estimator = estimator
    cfm.in_channels = 1
    cfm.zero_prompt_speech_token = False
    cfm.estimator_autocast_dtype = None
    return cfm, estimator


def _init_state(
    cfm: BASECFM,
    *,
    request_id: str,
    cfg_rate: float = 0.0,
    length: int = 4,
):
    return cfm.init_euler_state(
        request_id=request_id,
        mu=torch.zeros(1, length, 2),
        x_lens=torch.tensor([length]),
        prompt=torch.zeros(1, 1, 0),
        style=torch.zeros(1, 3),
        n_timesteps=2,
        inference_cfg_rate=cfg_rate,
        initial_noise=torch.zeros(1, 1, length),
        prompt_lens=torch.tensor([0]),
    )


def test_resumable_euler_batches_different_request_timesteps() -> None:
    cfm, estimator = _make_cfm()
    state_a = _init_state(cfm, request_id="a")
    state_b = _init_state(cfm, request_id="b")

    cfm.run_euler_step([state_b])
    assert state_b.step_index == 1

    cfm.run_euler_step([state_a, state_b])

    assert estimator.timesteps[-1].tolist() == [0.0, 0.5]
    assert state_a.step_index == 1
    assert state_b.step_index == 2
    assert state_a.finished is False
    assert state_b.finished is True
    torch.testing.assert_close(state_b.x, torch.full_like(state_b.x, 0.25))


def test_resumable_euler_uses_per_request_dt_for_mixed_steps() -> None:
    cfm, _ = _make_cfm()
    state_a = _init_state(cfm, request_id="a")
    state_b = _init_state(cfm, request_id="b")
    state_a.t_span = torch.tensor([0.0, 0.25, 1.0])
    state_b.t_span = torch.tensor([0.0, 0.5, 1.0])

    cfm.run_euler_step([state_b])
    cfm.run_euler_step([state_a, state_b])

    assert state_a.t.item() == 0.25
    assert state_b.t.item() == 1.0
    assert state_a.step_index == 1
    assert state_b.step_index == 2
    torch.testing.assert_close(state_a.x, torch.zeros_like(state_a.x))
    torch.testing.assert_close(state_b.x, torch.full_like(state_b.x, 0.25))


def test_resumable_euler_preserves_cfg_pair_order_for_mixed_steps() -> None:
    cfm, estimator = _make_cfm()
    state_a = _init_state(cfm, request_id="a", cfg_rate=0.7)
    state_b = _init_state(cfm, request_id="b", cfg_rate=0.7)
    cfm.run_euler_step([state_b])

    cfm.run_euler_step([state_a, state_b])

    assert estimator.timesteps[-1].tolist() == [0.0, 0.5, 0.0, 0.5]


def test_resumable_euler_reuses_single_request_cfg_conditioning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfm, estimator = _make_cfm()
    state = _init_state(cfm, request_id="a", cfg_rate=0.7)

    assert state.cfg_prompt_x is not None
    assert state.cfg_style is not None
    assert state.cfg_mu is not None
    assert state.cfg_pre_mask is not None

    def fail_zeros_like(*args, **kwargs):
        del args, kwargs
        raise AssertionError("single-request Euler step rebuilt CFG conditioning")

    monkeypatch.setattr(torch, "zeros_like", fail_zeros_like)
    cfm.run_euler_step([state])

    assert estimator.input_shapes[-1] == (2, 1, 4)


def test_resumable_euler_batches_variable_lengths_without_padding_state() -> None:
    cfm, estimator = _make_cfm()
    state_a = _init_state(cfm, request_id="a", length=4)
    state_b = _init_state(cfm, request_id="b", length=6)
    cfm.run_euler_step([state_b])

    cfm.run_euler_step([state_a, state_b])

    assert estimator.timesteps[-1].tolist() == [0.0, 0.5]
    assert estimator.input_shapes[-1] == (2, 1, 6)
    assert state_a.x.shape == (1, 1, 4)
    assert state_b.x.shape == (1, 1, 6)
    assert state_a.step_index == 1
    assert state_b.step_index == 2
    torch.testing.assert_close(state_b.x, torch.full_like(state_b.x, 0.25))


def test_resumable_euler_reuses_group_conditioning_and_recurrent_workspaces() -> None:
    cfm, estimator = _make_cfm()
    state_a = _init_state(cfm, request_id="a", cfg_rate=0.7, length=4)
    state_b = _init_state(cfm, request_id="b", cfg_rate=0.7, length=6)

    cfm.run_euler_step([state_a, state_b])
    cfm.run_euler_step([state_a, state_b])

    assert estimator.workspace_ptrs[0] == estimator.workspace_ptrs[1]
    assert estimator.conditioning_ptrs[0] == estimator.conditioning_ptrs[1]


def test_resumable_euler_keeps_state_in_group_workspace_between_steps() -> None:
    cfm, _ = _make_cfm()
    state_a = _init_state(cfm, request_id="a", cfg_rate=0.7, length=4)
    state_b = _init_state(cfm, request_id="b", cfg_rate=0.7, length=6)
    original_ptrs = (state_a.x.data_ptr(), state_b.x.data_ptr())

    cfm.run_euler_step([state_a, state_b])
    entry = list(cfm._euler_group_cache.values())[-1]
    resident_ptrs = (state_a.x.data_ptr(), state_b.x.data_ptr())

    assert resident_ptrs != original_ptrs
    assert state_a.x.data_ptr() == entry.estimator_x[0:1, :, :4].data_ptr()
    assert state_b.x.data_ptr() == entry.estimator_x[1:2, :, :6].data_ptr()

    cfm.run_euler_step([state_a, state_b])

    assert (state_a.x.data_ptr(), state_b.x.data_ptr()) == resident_ptrs


def test_resumable_euler_discards_cached_groups_for_finished_requests() -> None:
    cfm, _ = _make_cfm()
    state_a = _init_state(cfm, request_id="a", length=4)
    state_b = _init_state(cfm, request_id="b", length=6)
    state_c = _init_state(cfm, request_id="c", length=5)
    cfm.run_euler_step([state_a, state_b])
    cfm.run_euler_step([state_c])

    assert len(cfm._euler_group_cache) == 2

    cfm.discard_euler_group_cache({"a"})

    assert len(cfm._euler_group_cache) == 1
    remaining_request_ids = {request_id for key in cfm._euler_group_cache for request_id, _state_id, _length in key[0]}
    assert remaining_request_ids == {"c"}


def test_stacked_noncausal_mask_keeps_stride_zero_query_dimension() -> None:
    cfm, _ = _make_cfm()
    states = [
        _init_state(cfm, request_id="a", length=6),
        _init_state(cfm, request_id="b", length=6),
    ]

    first, second = cfm._stack_state_masks(states)

    assert second is not None
    assert second.stride(-2) == 0
    assert second.data_ptr() == first.data_ptr()


class _FakeEulerState:
    def __init__(self, request_id: str, length: int, *, step_index: int = 0) -> None:
        self.request_id = request_id
        self.x = torch.zeros(1, 80, length)
        self.step_index = step_index

    @property
    def finished(self) -> bool:
        return self.step_index >= 2


class _FakeDecoderState:
    def __init__(self, cfm_state: _FakeEulerState) -> None:
        self.cfm_state = cfm_state
        self.target_length = int(cfm_state.x.shape[-1])
        self.ref_length = 0
        self.output_emitted = False
        self.cfm_admit_after = 0.0


class _RecordingCFM:
    def __init__(self) -> None:
        self.calls: list[list[tuple[str, int]]] = []
        self.discard_calls: list[set[str]] = []

    def run_euler_step(self, states: list[_FakeEulerState]) -> None:
        self.calls.append([(state.request_id, state.step_index) for state in states])
        for state in states:
            state.step_index += 1

    def discard_euler_group_cache(self, request_ids: set[str]) -> None:
        self.discard_calls.append(request_ids)


def _make_decoder_for_state_tests(cfm: _RecordingCFM | None = None) -> IndexTTS2S2MelDecoder:
    decoder = object.__new__(IndexTTS2S2MelDecoder)
    torch.nn.Module.__init__(decoder)
    decoder.s2mel_cfm_batch_size = 4
    decoder.s2mel_continuous_max_padding_ratio = 1.0
    decoder.s2mel_continuous_singleton_wait_ms = 0.0
    decoder._continuous_cfm_states = {}
    decoder._last_finished_request_ids = set()
    decoder._deferred_cleanup_ids = set()
    decoder.s2mel_async_vocoder = False
    decoder.s2mel_async_vocoder_max_pending_batches = 2
    decoder._async_vocoder_stream = None
    decoder._pending_vocoder_batches = deque()
    decoder._ready_vocoder_outputs = {}
    if cfm is not None:
        decoder.s2mel = SimpleNamespace(models={"cfm": cfm})
    return decoder


class _QueryEvent:
    def __init__(self, ready: bool) -> None:
        self.ready = ready

    def query(self) -> bool:
        return self.ready

    def synchronize(self) -> None:
        self.ready = True


def test_async_vocoder_collects_only_ready_live_requests() -> None:
    decoder = _make_decoder_for_state_tests()
    live = _FakeDecoderState(_FakeEulerState("live", 8))
    cancelled = _FakeDecoderState(_FakeEulerState("cancelled", 8))
    pending = _FakeDecoderState(_FakeEulerState("pending", 8))
    decoder._continuous_cfm_states = {"live": live, "pending": pending}
    decoder._pending_vocoder_batches.extend(
        [
            indextts2_s2mel_decoder._PendingVocoderBatch(
                ("live", "cancelled"),
                (live, cancelled),
                (torch.ones(4), torch.full((4,), 3.0)),
                _QueryEvent(True),
            ),
            indextts2_s2mel_decoder._PendingVocoderBatch(
                ("pending",),
                (pending,),
                (torch.full((4,), 2.0),),
                _QueryEvent(False),
            ),
        ]
    )

    outputs = decoder._collect_ready_vocoder_outputs({"live", "cancelled"})

    torch.testing.assert_close(outputs["live"], torch.ones(4))
    assert "cancelled" not in outputs
    assert decoder._ready_vocoder_outputs["live"][0] is live
    assert live.output_emitted is False

    decoder._commit_ready_vocoder_outputs(set(outputs))

    assert live.output_emitted is True
    assert pending.output_emitted is False
    assert "live" not in decoder._ready_vocoder_outputs
    assert len(decoder._pending_vocoder_batches) == 1


def test_ready_async_vocoder_output_survives_batch_init_failure() -> None:
    cfm = _RecordingCFM()
    decoder = _make_decoder_for_state_tests(cfm)
    ready = _FakeDecoderState(_FakeEulerState("ready", 8))
    ready.vocoder_queued = True
    decoder._continuous_cfm_states = {"ready": ready}
    decoder._ready_vocoder_outputs = {
        "ready": (ready, torch.ones(4)),
    }

    def initialize(
        self: IndexTTS2S2MelDecoder,
        *,
        request_id: str,
        info: dict[str, object],
        device: torch.device,
        model_dtype: torch.dtype,
    ) -> _FakeDecoderState:
        del self, info, device, model_dtype
        raise RuntimeError(f"cannot initialize {request_id}")

    decoder._initialize_continuous_request = MethodType(initialize, decoder)

    with pytest.raises(RuntimeError, match="cannot initialize bad"):
        decoder._forward_continuous(
            request_ids=["ready", "bad"],
            request_infos=[{}, {}],
            device=torch.device("cpu"),
            model_dtype=torch.float32,
        )

    assert decoder._ready_vocoder_outputs["ready"][0] is ready
    assert ready.output_emitted is False
    assert decoder.take_finished_request_ids() == set()

    output = decoder._forward_continuous(
        request_ids=["ready"],
        request_infos=[{}],
        device=torch.device("cpu"),
        model_dtype=torch.float32,
    )

    assert output.multimodal_outputs is not None
    torch.testing.assert_close(
        output.multimodal_outputs["audio"][0],
        torch.ones(4),
    )
    assert ready.output_emitted is True
    assert decoder._ready_vocoder_outputs == {}
    assert decoder.take_finished_request_ids() == {"ready"}


def test_continuous_decoder_initializes_each_request_once() -> None:
    decoder = _make_decoder_for_state_tests()
    initialized: list[tuple[str, dict[str, object]]] = []

    def initialize(
        self: IndexTTS2S2MelDecoder,
        *,
        request_id: str,
        info: dict[str, object],
        device: torch.device,
        model_dtype: torch.dtype,
    ) -> SimpleNamespace:
        del self, device, model_dtype
        initialized.append((request_id, info))
        return _FakeDecoderState(_FakeEulerState(request_id, 8))

    decoder._initialize_continuous_request = MethodType(initialize, decoder)
    payload = {"mel_codes": torch.tensor([1, 2, 3])}

    first = decoder._get_or_initialize_continuous_request(
        request_id="request-a",
        info=payload,
        device=torch.device("cpu"),
        model_dtype=torch.float32,
    )
    second = decoder._get_or_initialize_continuous_request(
        request_id="request-a",
        info=payload,
        device=torch.device("cpu"),
        model_dtype=torch.float32,
    )

    assert first is second
    assert initialized == [("request-a", payload)]


def test_continuous_decoder_rolls_back_new_batch_states_after_init_failure() -> None:
    cfm = _RecordingCFM()
    decoder = _make_decoder_for_state_tests(cfm)
    existing = _FakeDecoderState(_FakeEulerState("existing", 8))
    decoder._continuous_cfm_states = {"existing": existing}
    initialized: list[str] = []

    def initialize(
        self: IndexTTS2S2MelDecoder,
        *,
        request_id: str,
        info: dict[str, object],
        device: torch.device,
        model_dtype: torch.dtype,
    ) -> _FakeDecoderState:
        del self, info, device, model_dtype
        initialized.append(request_id)
        if request_id == "new-bad":
            raise RuntimeError("bad request")
        return _FakeDecoderState(_FakeEulerState(request_id, 8))

    decoder._initialize_continuous_request = MethodType(initialize, decoder)

    with pytest.raises(RuntimeError, match="bad request"):
        decoder._forward_continuous(
            request_ids=["existing", "new-good", "new-bad"],
            request_infos=[{}, {}, {}],
            device=torch.device("cpu"),
            model_dtype=torch.float32,
        )

    assert initialized == ["new-good", "new-bad"]
    assert decoder._continuous_cfm_states == {"existing": existing}
    assert cfm.discard_calls == [{"new-good"}]


def test_continuous_decoder_discards_new_state_after_advance_failure() -> None:
    cfm = _RecordingCFM()
    decoder = _make_decoder_for_state_tests(cfm)
    existing = _FakeDecoderState(_FakeEulerState("existing", 8))
    decoder._continuous_cfm_states = {"existing": existing}

    def initialize(
        self: IndexTTS2S2MelDecoder,
        *,
        request_id: str,
        info: dict[str, object],
        device: torch.device,
        model_dtype: torch.dtype,
    ) -> _FakeDecoderState:
        del self, info, device, model_dtype
        return _FakeDecoderState(_FakeEulerState(request_id, 8))

    def fail_after_advance(
        self: IndexTTS2S2MelDecoder,
        request_ids: list[str],
    ) -> list[_FakeDecoderState]:
        for request_id in request_ids:
            self._continuous_cfm_states[request_id].cfm_state.step_index += 1
        raise RuntimeError("advance failed")

    decoder._initialize_continuous_request = MethodType(initialize, decoder)
    decoder._advance_continuous_cfm = MethodType(fail_after_advance, decoder)

    with pytest.raises(RuntimeError, match="advance failed"):
        decoder._forward_continuous(
            request_ids=["existing", "new"],
            request_infos=[{}, {}],
            device=torch.device("cpu"),
            model_dtype=torch.float32,
        )

    assert decoder._continuous_cfm_states == {"existing": existing}
    assert existing.cfm_state.step_index == 1
    assert cfm.discard_calls == [{"new"}]


def test_continuous_decoder_groups_exact_shapes_and_keeps_mixed_steps() -> None:
    cfm = _RecordingCFM()
    decoder = _make_decoder_for_state_tests(cfm)
    decoder._continuous_cfm_states = {
        "a": _FakeDecoderState(_FakeEulerState("a", 8, step_index=0)),
        "b": _FakeDecoderState(_FakeEulerState("b", 8, step_index=1)),
        "c": _FakeDecoderState(_FakeEulerState("c", 12, step_index=0)),
    }

    finished = decoder._advance_continuous_cfm(["a", "b", "c"])

    assert cfm.calls == [[("a", 0), ("b", 1)], [("c", 0)]]
    assert [state.cfm_state.request_id for state in finished] == ["b"]


def test_continuous_decoder_groups_compatible_length_ratios() -> None:
    cfm = _RecordingCFM()
    decoder = _make_decoder_for_state_tests(cfm)
    decoder.s2mel_continuous_max_padding_ratio = 1.2
    decoder._continuous_cfm_states = {
        "a": _FakeDecoderState(_FakeEulerState("a", 8)),
        "b": _FakeDecoderState(_FakeEulerState("b", 9)),
        "c": _FakeDecoderState(_FakeEulerState("c", 12)),
    }

    decoder._advance_continuous_cfm(["a", "b", "c"])

    assert cfm.calls == [[("a", 0), ("b", 0)], [("c", 0)]]


def test_continuous_decoder_waits_once_for_new_singleton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfm = _RecordingCFM()
    decoder = _make_decoder_for_state_tests(cfm)
    decoder.s2mel_continuous_singleton_wait_ms = 1.0
    state = _FakeDecoderState(_FakeEulerState("a", 8))
    decoder._continuous_cfm_states = {"a": state}
    now = [10.0]
    monkeypatch.setattr(
        indextts2_s2mel_decoder.time,
        "monotonic",
        lambda: now[0],
    )

    decoder._advance_continuous_cfm(["a"])
    assert cfm.calls == []
    assert state.cfm_admit_after == pytest.approx(10.001)

    now[0] = 10.002
    decoder._advance_continuous_cfm(["a"])
    decoder._advance_continuous_cfm(["a"])

    assert cfm.calls == [[("a", 0)], [("a", 1)]]


def test_continuous_decoder_batches_compatible_request_before_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfm = _RecordingCFM()
    decoder = _make_decoder_for_state_tests(cfm)
    decoder.s2mel_continuous_singleton_wait_ms = 1.0
    state_a = _FakeDecoderState(_FakeEulerState("a", 8))
    decoder._continuous_cfm_states = {"a": state_a}
    now = [10.0]
    monkeypatch.setattr(
        indextts2_s2mel_decoder.time,
        "monotonic",
        lambda: now[0],
    )

    decoder._advance_continuous_cfm(["a"])
    state_b = _FakeDecoderState(_FakeEulerState("b", 8))
    decoder._continuous_cfm_states["b"] = state_b
    now[0] = 10.0005
    decoder._advance_continuous_cfm(["a", "b"])

    assert cfm.calls == [[("a", 0), ("b", 0)]]


def test_continuous_decoder_routes_variable_group_through_padding_aware_attention() -> None:
    cfm = _RecordingCFM()
    cfm.estimator = object()
    decoder = _make_decoder_for_state_tests(cfm)
    decoder.s2mel_continuous_max_padding_ratio = 1.2
    decoder._continuous_cfm_states = {
        "a": _FakeDecoderState(_FakeEulerState("a", 8)),
        "b": _FakeDecoderState(_FakeEulerState("b", 9)),
    }
    full_mask_values: list[bool] = []

    decoder._set_dit_full_mask_fast_path = MethodType(
        lambda self, estimator, *, enabled: full_mask_values.append(enabled),
        decoder,
    )

    decoder._advance_continuous_cfm(["a", "b"])

    assert full_mask_values == [False]


def test_continuous_decoder_reports_completion_once_and_cleans_aborts() -> None:
    cfm = _RecordingCFM()
    decoder = _make_decoder_for_state_tests(cfm)
    decoder._continuous_cfm_states = {
        "done": _FakeDecoderState(_FakeEulerState("done", 8)),
        "aborted": _FakeDecoderState(_FakeEulerState("aborted", 8)),
    }
    decoder._last_finished_request_ids = {"done"}

    assert decoder.take_finished_request_ids() == {"done"}
    assert decoder.take_finished_request_ids() == set()

    decoder.on_requests_finished(["aborted", "unknown"])

    assert set(decoder._continuous_cfm_states) == {"done", "aborted"}
    assert decoder._deferred_cleanup_ids == {"aborted", "unknown"}
    assert cfm.discard_calls == []

    decoder._flush_deferred_cleanup()

    assert set(decoder._continuous_cfm_states) == {"done"}
    assert cfm.discard_calls == [{"aborted", "unknown"}]

    decoder.on_requests_finished(["done"])
    decoder.flush_finished_requests()

    assert decoder._continuous_cfm_states == {}
    assert cfm.discard_calls[-1] == {"done"}


def test_completed_request_is_not_reinitialized_before_scheduler_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfm = _RecordingCFM()
    cfm.finalize_calls = 0

    def finalize(state: _FakeEulerState) -> torch.Tensor:
        del state
        cfm.finalize_calls += 1
        return torch.zeros(1, 80, 8)

    cfm.finalize_euler_state = finalize
    decoder = _make_decoder_for_state_tests(cfm)
    decoder.s2mel_vocoder_bf16 = False
    initialize_calls: list[str] = []
    vocode_calls: list[int] = []

    def initialize(
        self: IndexTTS2S2MelDecoder,
        *,
        request_id: str,
        info: dict[str, object],
        device: torch.device,
        model_dtype: torch.dtype,
    ) -> _FakeDecoderState:
        del self, info, device, model_dtype
        initialize_calls.append(request_id)
        return _FakeDecoderState(_FakeEulerState(request_id, 8, step_index=1))

    decoder._initialize_continuous_request = MethodType(initialize, decoder)
    decoder._get_resolved_vocoder_source = MethodType(
        lambda self: "fake-vocoder",
        decoder,
    )
    decoder._get_vocoder_runner = MethodType(
        lambda self, bigvgan, device, dtype: bigvgan,
        decoder,
    )

    def vocode_mels(
        self: IndexTTS2S2MelDecoder,
        *,
        mels: list[torch.Tensor],
        vocode: object,
        voc_dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        del self, vocode, voc_dtype
        vocode_calls.append(len(mels))
        return [torch.ones(4) for _ in mels]

    decoder._vocode_mels = MethodType(vocode_mels, decoder)
    monkeypatch.setattr(
        indextts2_s2mel_decoder,
        "_load_bigvgan",
        lambda *args, **kwargs: object(),
    )

    first = decoder._forward_continuous(
        request_ids=["request-a"],
        request_infos=[{"mel_codes": torch.tensor([1])}],
        device=torch.device("cpu"),
        model_dtype=torch.float32,
    )
    second = decoder._forward_continuous(
        request_ids=["request-a"],
        request_infos=[{"mel_codes": torch.tensor([1])}],
        device=torch.device("cpu"),
        model_dtype=torch.float32,
    )

    assert initialize_calls == ["request-a"]
    assert cfm.finalize_calls == 1
    assert vocode_calls == [1]
    torch.testing.assert_close(
        first.multimodal_outputs["audio"][0],
        torch.ones(4),
    )
    assert second.multimodal_outputs is None
    assert "request-a" in decoder._continuous_cfm_states

    decoder.on_requests_finished(["request-a"])

    assert "request-a" in decoder._continuous_cfm_states

    third = decoder._forward_continuous(
        request_ids=["request-a"],
        request_infos=[{"mel_codes": torch.tensor([1])}],
        device=torch.device("cpu"),
        model_dtype=torch.float32,
    )

    assert third.multimodal_outputs is None
    assert initialize_calls == ["request-a"]
    assert cfm.finalize_calls == 1
    assert vocode_calls == [1]
    assert "request-a" not in decoder._continuous_cfm_states


def test_continuous_decoder_emits_payload_only_for_completed_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfm = _RecordingCFM()
    cfm.finalize_euler_state = lambda state: torch.zeros(1, 80, 8)
    decoder = _make_decoder_for_state_tests(cfm)
    decoder.s2mel_vocoder_bf16 = False
    decoder._continuous_cfm_states = {
        "done": _FakeDecoderState(_FakeEulerState("done", 8, step_index=1)),
        "pending": _FakeDecoderState(_FakeEulerState("pending", 8)),
    }
    decoder._get_resolved_vocoder_source = MethodType(
        lambda self: "fake-vocoder",
        decoder,
    )
    decoder._get_vocoder_runner = MethodType(
        lambda self, bigvgan, device, dtype: bigvgan,
        decoder,
    )
    decoder._vocode_mels = MethodType(
        lambda self, *, mels, vocode, voc_dtype: [torch.ones(4) for _ in mels],
        decoder,
    )
    monkeypatch.setattr(
        indextts2_s2mel_decoder,
        "_load_bigvgan",
        lambda *args, **kwargs: object(),
    )

    output = decoder._forward_continuous(
        request_ids=["done", "pending"],
        request_infos=[
            {"mel_codes": torch.tensor([1])},
            {"mel_codes": torch.tensor([2])},
        ],
        device=torch.device("cpu"),
        model_dtype=torch.float32,
    )

    assert output.multimodal_outputs is not None
    assert output.multimodal_outputs["audio"][1] is None
    assert output.multimodal_outputs["sr"][1] is None
    torch.testing.assert_close(output.multimodal_outputs["audio"][0], torch.ones(4))
    assert output.multimodal_outputs["sr"][0].item() == 22050


def test_continuous_decoder_allows_vllm_dummy_forward_without_request_ids() -> None:
    decoder = _make_decoder_for_state_tests()
    decoder.stepwise_generation = True
    decoder.use_gpt_latent = False
    decoder._s2mel_model_dtype = MethodType(
        lambda self: torch.float32,
        decoder,
    )

    output = decoder(
        input_ids=torch.zeros(1, dtype=torch.long),
        model_intermediate_buffer=None,
    )

    assert output.multimodal_outputs is not None
    torch.testing.assert_close(
        output.multimodal_outputs["audio"],
        torch.zeros(1),
    )


def test_continuous_decoder_still_rejects_real_payload_without_request_ids() -> None:
    decoder = _make_decoder_for_state_tests()
    decoder.stepwise_generation = True
    decoder.use_gpt_latent = False
    decoder._s2mel_model_dtype = MethodType(
        lambda self: torch.float32,
        decoder,
    )

    with pytest.raises(
        ValueError,
        match="continuous batching requires request_ids",
    ):
        decoder(
            input_ids=torch.zeros(1, dtype=torch.long),
            model_intermediate_buffer=[{"mel_codes": torch.tensor([1, 2, 3])}],
        )
