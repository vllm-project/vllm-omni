# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU dispatch/state tests. Fake NPU routing here is not hardware validation."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.model_executor.models.minicpmo_4_5.test_code2wav_batching import _config, _forward, _info, _model
from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import (
    BatchedToken2WavState,
)
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav import MiniCPMO45Code2Wav, _WorkItem

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _voice_prompt(path):
    voice = int(path.rsplit("-", 1)[-1].split(".")[0])
    return (
        torch.tensor([[5 + voice, 6 + voice]]),
        torch.tensor([2], dtype=torch.int32),
        torch.tensor([[(-1.0 if voice % 2 else 1.0) * (voice + 1)]]),
        torch.full((1, 4, 1), float(voice + 1)),
        torch.tensor([4], dtype=torch.int32),
    )


def _candidate(enabled=True):
    model, core = _model(setup_cache_size=16)
    core._prepare_prompt = _voice_prompt
    core.flow.decoder.rand_noise = torch.zeros(1, 1, 1024)
    model._npu_cross_reference_batching = enabled
    backend = model.backend
    backend._ragged_fused_body = True
    calls = []

    def replay(**kwargs):
        # Refuse the graph to exercise the uniform solve's existing fallback.
        # Real NPUGraph numerical parity is covered by the NPU test file.
        if kwargs.get("prompt_lengths") is not None:
            caches = kwargs["att_cache"]
            if isinstance(caches, list):
                assert len({int(cache.shape[4]) for cache in caches}) == 1
            calls.append(int(kwargs["x"].shape[0]))
        return None

    backend._whole_euler_graph_wrapper = SimpleNamespace(
        enabled=True,
        device=SimpleNamespace(type="npu"),
        dtype=torch.float32,
        att_slots=0,
        ragged_body=object(),
        replay=replay,
    )
    return model, core, calls


def _request(row, chunk, *, count=20, last=False, epoch=0):
    info = _info(f"voice-{row}", chunk, list(range(10 + row, 10 + row + count)), last_chunk=last, cache_epoch=epoch)
    info["meta"].update(prompt_cache_id=f"voice-{row}", prompt_wav=f"/fake/voice-{row}.wav")
    return info


@pytest.mark.parametrize(
    "value,expected", [(None, False), (False, False), (True, True), ("false", False), ("true", True)]
)
def test_opt_in_config(value, expected):
    config = _config()
    if value is not None:
        config.model_config.stage_connector_config["extra"]["cfm_npu_cross_reference_batching"] = value
    model = MiniCPMO45Code2Wav(vllm_config=config)
    assert model._npu_cross_reference_batching is expected


def test_invalid_config_fails_at_construction():
    config = _config()
    config.model_config.stage_connector_config["extra"]["cfm_npu_cross_reference_batching"] = "sometimes"
    with pytest.raises(ValueError, match="cfm_npu_cross_reference_batching"):
        MiniCPMO45Code2Wav(vllm_config=config)


@pytest.mark.parametrize("enabled,minimum", [(False, 1), (False, 2), (True, 1), (True, 2)])
def test_minimum_batch_conflict_warns_at_construction(enabled, minimum, monkeypatch):
    warning = Mock()
    monkeypatch.setattr("vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav.logger.warning", warning)
    config = _config(minimum=minimum)
    config.model_config.stage_connector_config["extra"]["cfm_npu_cross_reference_batching"] = enabled
    model = MiniCPMO45Code2Wav(vllm_config=config)
    assert model._min_batch_size == minimum
    assert model._npu_cross_reference_batching is enabled
    if enabled and minimum != 1:
        warning.assert_called_once()
        message, value = warning.call_args.args
        assert "requires code2wav_min_batch_size=1" in message
        assert "will remain inactive" in message
        assert value == minimum
        model.backend = SimpleNamespace()
        assert not model._npu_cross_reference_active()
    else:
        warning.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
def test_single_request_keeps_original_dispatch_without_npu_metadata(enabled, monkeypatch):
    model, _, _ = _candidate(enabled)
    independent, _, _ = _candidate(False)
    # A singleton cannot share work, even with the feature enabled. Guard
    # both capability lookup and the two prepared-shape validation sites.
    capability = Mock(side_effect=AssertionError("singleton must not inspect NPU batching capability"))
    grouping = Mock(side_effect=AssertionError("singleton must not regroup prepared states"))
    monkeypatch.setattr(model, "_npu_cross_reference_active", capability)
    monkeypatch.setattr(model, "_npu_same_shape_groups", grouping)
    original_key = model._bucket_key

    def singleton_key(item, *, cross_turn, row_offsets, cross_reference, npu_same_shape):
        assert not npu_same_shape, "singleton must retain reference-specific bucket keys"
        return original_key(
            item,
            cross_turn=cross_turn,
            row_offsets=row_offsets,
            cross_reference=cross_reference,
            npu_same_shape=npu_same_shape,
        )

    monkeypatch.setattr(model, "_bucket_key", singleton_key)
    for chunk in range(10):
        info = _request(0, chunk, last=chunk == 9)
        actual = _forward(model, [info])
        expected = _forward(independent, [info])
        torch.testing.assert_close(
            actual.multimodal_outputs["model_outputs"][0], expected.multimodal_outputs["model_outputs"][0]
        )
        if chunk != 9:
            state = model._states["voice-0"].token2wav
            reference = independent._states["voice-0"].token2wav
            for name, cache in reference.flow_cache.items():
                torch.testing.assert_close(state.flow_cache[name], cache)
            for name, cache in reference.hift_cache.items():
                torch.testing.assert_close(state.hift_cache[name], cache)
    capability.assert_not_called()
    grouping.assert_not_called()
    assert not model._states


def test_singleton_buckets_skip_prepared_shape_partition(monkeypatch):
    model, _, calls = _candidate()
    independent = [_candidate(False)[0] for _ in range(2)]
    grouping = Mock(side_effect=AssertionError("a one-row bucket needs no additional shape partition"))
    monkeypatch.setattr(model, "_npu_same_shape_groups", grouping)
    infos = [_request(0, 0, count=20), _request(1, 0, count=12)]
    actual = _forward(model, infos)
    assert calls == [1, 1]
    grouping.assert_not_called()
    for row, single in enumerate(independent):
        expected = _forward(single, [infos[row]])
        torch.testing.assert_close(
            actual.multimodal_outputs["model_outputs"][row], expected.multimodal_outputs["model_outputs"][0]
        )


def test_singletons_after_reference_setup_skip_revalidation(monkeypatch):
    model, core, calls = _candidate()

    def different_prompt_lengths(path):
        values = list(_voice_prompt(path))
        if "voice-1.wav" in path:
            values[0] = torch.tensor([[5, 6, 7]])
            values[3] = torch.ones(1, 5, 1)
        return tuple(values)

    core._prepare_prompt = different_prompt_lengths
    grouping = Mock(wraps=model._npu_same_shape_groups)
    monkeypatch.setattr(model, "_npu_same_shape_groups", grouping)
    _forward(model, [_request(0, 0), _request(1, 0)])
    assert calls == [1, 1]
    # Fresh rows need one authoritative partition after setup, but the
    # singleton groups it produces must not perform the same validation again.
    grouping.assert_called_once()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("batch", [1, 2, 3, 4])
def test_equal_shapes_match_independent_voices_through_streaming_trim(batch, enabled):
    model, core, calls = _candidate(enabled)
    independent = [_candidate(False)[0] for _ in range(batch)]
    for chunk in range(10):
        infos = [_request(row, chunk, last=chunk == 9) for row in range(batch)]
        calls.clear()
        output = _forward(model, infos)
        assert calls == ([batch] if enabled else [1] * batch)
        assert core.flow.decoder.estimator.cfg_batches[-2:] == [2 * (batch if enabled else 1)] * 2
        if enabled:
            # _project_speakers L2-normalizes each one-dimensional embedding
            # before the fake identity affine layer, so only its sign remains.
            speakers = [-1.0 if row % 2 else 1.0 for row in range(batch)]
            assert core.flow.decoder.estimator.speaker_order[-1] == speakers + [0.0] * batch
        for row, single in enumerate(independent):
            expected = _forward(single, [infos[row]])
            torch.testing.assert_close(
                output.multimodal_outputs["model_outputs"][row], expected.multimodal_outputs["model_outputs"][0]
            )
            if chunk != 9:
                state = model._states[f"voice-{row}"].token2wav
                reference = single._states[f"voice-{row}"].token2wav
                for name, cache in reference.flow_cache.items():
                    torch.testing.assert_close(state.flow_cache[name], cache)
                for name, cache in reference.hift_cache.items():
                    torch.testing.assert_close(state.hift_cache[name], cache)
        if chunk != 9:
            caches = [state.token2wav.flow_cache["estimator_att_cache"] for state in model._states.values()]
            assert len({cache.data_ptr() for cache in caches}) == batch
    assert not model._states


@pytest.mark.parametrize(
    "ineligible",
    ["cuda", "cpu", "disabled_graph", "no_ragged", "unfused", "fp16", "bf16_cache", "slots", "trt", "minimum"],
)
def test_ineligible_policies_do_not_enable_npu_mode(ineligible):
    model, _, _ = _candidate()
    wrapper = model.backend._whole_euler_graph_wrapper
    if ineligible in {"cuda", "cpu"}:
        wrapper.device = SimpleNamespace(type=ineligible)
    elif ineligible == "disabled_graph":
        wrapper.enabled = False
    elif ineligible == "no_ragged":
        wrapper.ragged_body = None
    elif ineligible == "unfused":
        model.backend._ragged_fused_body = False
    elif ineligible == "fp16":
        wrapper.dtype = torch.float16
    elif ineligible == "bf16_cache":
        model.backend._estimator_att_cache_dtype = torch.bfloat16
    elif ineligible == "slots":
        wrapper.att_slots = 4
    elif ineligible == "trt":
        model.backend._trt_stepper = object()
    else:
        model._min_batch_size = 2
    assert not model._npu_cross_reference_active()


@pytest.mark.parametrize("enabled", [False, True])
def test_cuda_merging_is_unchanged_by_npu_switch(enabled):
    model, _, calls = _candidate(enabled)
    model.backend._whole_euler_graph_wrapper.device = SimpleNamespace(type="cuda")
    model.backend._ragged_fused_body = False
    _forward(model, [_request(0, 0), _request(1, 0)])
    assert calls == [2]


def test_step_graph_is_preserved_when_npu_opt_in_is_ineligible():
    model, core, _ = _candidate()
    model.backend._whole_euler_graph_wrapper = None
    calls = []

    def replay(inputs, time, cnn, att, cnn_out, att_out, mask=None):
        calls.append(int(inputs.shape[0]))
        result = core.flow.decoder.estimator.blocks_forward_chunk(inputs, time, mask, cnn, att, cnn_out, att_out)
        return result, cnn_out, att_out

    model.backend._cfm_graph_wrapper = SimpleNamespace(enabled=True, replay=replay)
    _forward(model, [_request(0, 0), _request(1, 0)])
    calls.clear()
    _forward(model, [_request(0, 1), _request(1, 1)])
    assert calls == [2] * (2 * model.backend.n_timesteps)


def test_different_token_lengths_are_split_without_reordering():
    model, _, calls = _candidate()
    singles = [_candidate(False)[0] for _ in range(3)]
    infos = [_request(row, 0, count=count) for row, count in enumerate([20, 12, 20])]
    actual = _forward(model, infos)
    assert calls == [2, 1]
    for row, single in enumerate(singles):
        expected = _forward(single, [infos[row]])
        torch.testing.assert_close(
            actual.multimodal_outputs["model_outputs"][row], expected.multimodal_outputs["model_outputs"][0]
        )


@pytest.mark.parametrize("onset_merge", [False, True])
def test_fresh_references_are_regrouped_after_actual_prompt_setup(onset_merge):
    model, core, calls = _candidate()
    model._onset_merge = onset_merge
    singles = [_candidate(False)[0] for _ in range(3)]

    def varied_prompt(path):
        values = list(_voice_prompt(path))
        if "voice-1.wav" in path:
            values[0] = torch.tensor([[5, 6, 7]])
            values[3] = torch.ones(1, 5, 1)
        return tuple(values)

    core._prepare_prompt = varied_prompt
    for single in singles:
        single.backend._token2wav._prepare_prompt = varied_prompt
    infos = [_request(row, 0) for row in range(3)]
    actual = _forward(model, infos)
    assert calls == [2, 1]
    for row, single in enumerate(singles):
        expected = _forward(single, [infos[row]])
        torch.testing.assert_close(
            actual.multimodal_outputs["model_outputs"][row], expected.multimodal_outputs["model_outputs"][0]
        )


def test_mixed_history_and_final_flags_remain_separate():
    model, _, calls = _candidate()
    _forward(model, [_request(0, 0), _request(1, 0)])
    _forward(model, [_request(0, 1)])
    calls.clear()
    _forward(model, [_request(0, 2), _request(1, 1)])
    assert calls == [1, 1]
    other, _, other_calls = _candidate()
    _forward(other, [_request(0, 0), _request(1, 0)])
    other_calls.clear()
    _forward(other, [_request(0, 1, last=True), _request(1, 1)])
    assert other_calls == [1, 1]
    assert "voice-0" not in other._states and "voice-1" in other._states
    other.on_requests_finished(["voice-1"])
    assert not other._states


def test_permuted_request_order_matches_independent_streams():
    model, _, _ = _candidate()
    independent = {row: _candidate(False)[0] for row in range(4)}
    for chunk in range(3):
        order = [2, 0, 3, 1] if chunk % 2 == 0 else [1, 3, 0, 2]
        infos = [_request(row, chunk, last=chunk == 2) for row in order]
        result = _forward(model, infos)
        for index, row in enumerate(order):
            expected = _forward(independent[row], [infos[index]])
            torch.testing.assert_close(
                result.multimodal_outputs["model_outputs"][index],
                expected.multimodal_outputs["model_outputs"][0],
            )
    assert not model._states


def test_completed_and_cancelled_rows_do_not_pollute_new_voice():
    model, _, _ = _candidate()
    independent = {row: _candidate(False)[0] for row in range(3)}
    for chunk in range(6):
        infos = [_request(1, chunk, last=chunk == 5)]
        if chunk < 3:
            infos.append(_request(0, chunk, last=chunk == 2))
        if chunk >= 3:
            infos.append(_request(2, chunk - 3, last=chunk == 5))
        actual = _forward(model, infos)
        for index, info in enumerate(infos):
            row = 1 if index == 0 else (0 if chunk < 3 else 2)
            expected = _forward(independent[row], [info])
            torch.testing.assert_close(
                actual.multimodal_outputs["model_outputs"][index],
                expected.multimodal_outputs["model_outputs"][0],
            )
    assert not model._states
    model, _, _ = _candidate()
    _forward(model, [_request(0, 0), _request(1, 0)])
    model.on_requests_finished(["voice-0"])
    assert "voice-0" not in model._states and "voice-1" in model._states
    model.on_requests_finished(["voice-1"])
    assert not model._states


@pytest.mark.parametrize("cross_turn", [False, True])
def test_epoch_policy_is_preserved(cross_turn):
    model, _, calls = _candidate()
    model._cross_turn_buckets = cross_turn
    _forward(model, [_request(0, 0, epoch=0), _request(1, 0, epoch=1)])
    assert calls == ([2] if cross_turn else [1, 1])


def _prepared_rows():
    _, core, _ = _candidate()
    from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import BatchedToken2Wav

    # A standalone CPU adapter avoids the fake graph while creating fixtures.
    backend = BatchedToken2Wav(core)
    features = [backend.prepare_prompt(f"voice-{i}", f"/fake/voice-{i}.wav") for i in range(2)]
    states = [backend.setup_batch(feature, 1)[0] for feature in features]
    items = [
        _WorkItem(
            i,
            f"r{i}",
            f"r{i}",
            0,
            0,
            f"voice-{i}",
            f"/fake/voice-{i}.wav",
            False,
            torch.tensor([10, 11]),
            None,
            None,
            0,
            0,
            torch.empty(0, dtype=torch.uint8),
            False,
            False,
            False,
        )
        for i in range(2)
    ]
    return items, states, features


@pytest.mark.parametrize(
    "change",
    [
        "cache_dtype",
        "cache_device",
        "cache_shape",
        "hift_shape",
        "prompt_length",
        "prompt_dtype",
        "speaker_shape",
        "token_dtype",
        "token_device",
        "last",
        "epoch",
    ],
)
def test_authoritative_shape_group_rejects_incompatible_rows(change):
    items, states, features = _prepared_rows()
    assert MiniCPMO45Code2Wav._npu_same_shape_groups(items, states, features, cross_turn=False) == [[0, 1]]
    if change == "cache_dtype":
        cache = dict(states[1].flow_cache)
        cache["estimator_att_cache"] = cache["estimator_att_cache"].to(torch.bfloat16)
        states[1] = BatchedToken2WavState(cache, states[1].hift_cache)
    elif change in {"cache_device", "cache_shape"}:
        cache = dict(states[1].flow_cache)
        value = cache["estimator_att_cache"]
        cache["estimator_att_cache"] = value.to("meta") if change == "cache_device" else value[..., :-1, :]
        states[1] = BatchedToken2WavState(cache, states[1].hift_cache)
    elif change == "hift_shape":
        cache = dict(states[1].hift_cache)
        # setup_batch starts with a zero-length HiFT mel cache, so slicing
        # off a frame would leave the shape unchanged. Add one instead.
        value = cache["mel"]
        cache["mel"] = value.new_zeros((*value.shape[:-1], value.shape[-1] + 1))
        assert cache["mel"].shape != states[0].hift_cache["mel"].shape
        states[1] = BatchedToken2WavState(states[1].flow_cache, cache)
    elif change == "prompt_length":
        features[1] = replace(features[1], mels=torch.zeros(1, 5, 1))
    elif change == "speaker_shape":
        features[1] = replace(features[1], speaker_embedding=torch.zeros(1, 2))
    elif change == "prompt_dtype":
        features[1] = replace(features[1], mels=features[1].mels.double())
    elif change == "token_dtype":
        items[1] = replace(items[1], tokens=items[1].tokens.to(torch.int32))
    elif change == "token_device":
        items[1] = replace(items[1], tokens=items[1].tokens.to("meta"))
    elif change == "last":
        items[1] = replace(items[1], last_chunk=True)
    else:
        items[1] = replace(items[1], cache_epoch=1)
    assert MiniCPMO45Code2Wav._npu_same_shape_groups(items, states, features, cross_turn=False) == [[0], [1]]


@pytest.mark.parametrize("mismatch", ["tokens", "last_chunk"])
def test_direct_uniform_dispatch_rejects_mismatch_before_backend_call(mismatch):
    model, _, _ = _candidate()
    items, states, features = _prepared_rows()
    if mismatch == "tokens":
        items[1] = replace(items[1], tokens=torch.tensor([1, 2, 3]))
    else:
        items[1] = replace(items[1], last_chunk=not items[0].last_chunk)
    model.backend.decode_batch = Mock(side_effect=AssertionError("must not run"))
    with pytest.raises(RuntimeError, match="npu_cross_reference_shape_mismatch"):
        model._decode_rows(items, states, features, ragged=False)
    model.backend.decode_batch.assert_not_called()


def test_existing_npu_mixed_history_guard_is_preserved(monkeypatch):
    from tests.model_executor.models.minicpmo_4_5.test_cuda_graph_wrapper import (
        test_whole_euler_mixed_histories_do_not_enable_npu_arena,
    )

    test_whole_euler_mixed_histories_do_not_enable_npu_arena(monkeypatch)
