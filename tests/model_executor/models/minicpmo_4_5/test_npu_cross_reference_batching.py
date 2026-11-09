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
    mixed_history = enabled
    model, core = _model(setup_cache_size=16)
    core._prepare_prompt = _voice_prompt
    core.flow.decoder.rand_noise = torch.zeros(1, 1, 1024)
    model._npu_cross_reference_batching = enabled
    backend = model.backend
    backend._ragged_fused_body = True
    if mixed_history:
        # This fixture exercises dispatch/state ownership with a synthetic
        # estimator, not real DiT blocks. Its output and cache markers depend
        # only on the first valid frame, so padding does not change them.
        # Trained-weight NPU tests validate the actual ragged fused kernel.
        def fake_ragged_body(estimator, inputs, time, mask, cnn, att, cnn_out, att_out, lengths):
            assert len(lengths) == inputs.shape[0]
            assert all(0 < length <= inputs.shape[-1] for length in lengths)
            return estimator.blocks_forward_chunk(inputs, time, mask, cnn, att, cnn_out, att_out)

        backend._ragged_body = fake_ragged_body
    calls = []

    def replay(**kwargs):
        # Refuse the graph to exercise the uniform solve's existing fallback.
        # Real NPUGraph numerical parity is covered by the NPU test file.
        if kwargs.get("prompt_lengths") is not None:
            caches = kwargs["att_cache"]
            if isinstance(caches, list) and not mixed_history:
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


def test_different_lengths_merge_after_reference_setup():
    model, _, calls = _candidate()
    independent = [_candidate(False)[0] for _ in range(2)]
    infos = [_request(0, 0, count=20), _request(1, 0, count=12)]
    actual = _forward(model, infos)
    assert calls == [2]
    for row, single in enumerate(independent):
        expected = _forward(single, [infos[row]])
        torch.testing.assert_close(
            actual.multimodal_outputs["model_outputs"][row], expected.multimodal_outputs["model_outputs"][0]
        )


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


def test_different_token_lengths_share_cfm_without_reordering():
    model, _, calls = _candidate()
    singles = [_candidate(False)[0] for _ in range(3)]
    infos = [_request(row, 0, count=count) for row, count in enumerate([20, 12, 20])]
    actual = _forward(model, infos)
    assert calls == [3]
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
    assert calls == [3]
    for row, single in enumerate(singles):
        expected = _forward(single, [infos[row]])
        torch.testing.assert_close(
            actual.multimodal_outputs["model_outputs"][row], expected.multimodal_outputs["model_outputs"][0]
        )


def test_mixed_history_and_final_flags_share_cfm():
    model, _, calls = _candidate()
    _forward(model, [_request(0, 0), _request(1, 0)])
    _forward(model, [_request(0, 1)])
    calls.clear()
    _forward(model, [_request(0, 2), _request(1, 1)])
    assert calls == [2]
    other, _, other_calls = _candidate()
    _forward(other, [_request(0, 0), _request(1, 0)])
    other_calls.clear()
    _forward(other, [_request(0, 1, last=True), _request(1, 1)])
    assert other_calls == [2]
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
def test_direct_dispatch_routes_variable_shapes_to_ragged_backend(mismatch):
    model, _, _ = _candidate()
    items, states, features = _prepared_rows()
    if mismatch == "tokens":
        items[1] = replace(items[1], tokens=torch.tensor([1, 2, 3]))
    else:
        items[1] = replace(items[1], last_chunk=not items[0].last_chunk)
    model.backend.decode_batch = Mock(side_effect=AssertionError("must not run"))
    expected: tuple[list[torch.Tensor], list[BatchedToken2WavState]] = ([], [])
    model.backend.decode_ragged_batch = Mock(return_value=expected)
    assert model._decode_rows(items, states, features, ragged=False) is expected
    model.backend.decode_ragged_batch.assert_called_once()
    model.backend.decode_batch.assert_not_called()


def test_existing_npu_mixed_history_guard_is_preserved(monkeypatch):
    from tests.model_executor.models.minicpmo_4_5.test_cuda_graph_wrapper import (
        test_whole_euler_mixed_histories_do_not_enable_npu_arena,
    )

    test_whole_euler_mixed_histories_do_not_enable_npu_arena(monkeypatch)


@pytest.mark.parametrize("batch", [2, 3, 4])
def test_ragged_npu_inputs_match_independent_streams(batch):
    model, core, calls = _candidate()
    independent_pairs = [_candidate(False) for _ in range(batch)]
    independent = [value[0] for value in independent_pairs]

    def variable_prompt(path):
        values = list(_voice_prompt(path))
        row = int(path.rsplit("-", 1)[-1].split(".")[0])
        values[0] = torch.arange(5, 7 + row).reshape(1, -1)
        values[1] = torch.tensor([2 + row], dtype=torch.int32)
        values[3] = torch.full((1, 4 + row, 1), float(row + 1))
        values[4] = torch.tensor([4 + row], dtype=torch.int32)
        return tuple(values)

    core._prepare_prompt = variable_prompt
    for _, single_core, _ in independent_pairs:
        single_core._prepare_prompt = variable_prompt
    for chunk in range(10):
        infos = [_request(row, chunk, count=12 + 2 * row, last=chunk == 9) for row in range(batch)]
        calls.clear()
        actual = _forward(model, infos)
        assert calls == [batch], "Encoder subgroups must converge into one CFM solve"
        for row, single in enumerate(independent):
            expected = _forward(single, [infos[row]])
            torch.testing.assert_close(
                actual.multimodal_outputs["model_outputs"][row],
                expected.multimodal_outputs["model_outputs"][0],
            )
            if chunk != 9:
                state = model._states[f"voice-{row}"].token2wav
                reference = single._states[f"voice-{row}"].token2wav
                for name, cache in reference.flow_cache.items():
                    torch.testing.assert_close(state.flow_cache[name], cache)
                for name, cache in reference.hift_cache.items():
                    torch.testing.assert_close(state.hift_cache[name], cache)
    assert not model._states


def test_ragged_npu_groups_allow_lengths_but_keep_hift_boundary():
    items, states, features = _prepared_rows()
    states[1].flow_cache["conformer_att_cache"] = torch.cat([states[1].flow_cache["conformer_att_cache"]] * 2, dim=3)
    states[1].flow_cache["estimator_att_cache"] = torch.cat([states[1].flow_cache["estimator_att_cache"]] * 2, dim=4)
    features[1] = replace(
        features[1],
        speech_tokens=torch.cat([features[1].speech_tokens] * 2, dim=1),
        mels=torch.cat([features[1].mels] * 2, dim=1),
    )
    items[1] = replace(items[1], tokens=items[1].tokens[:3], last_chunk=True)
    group = MiniCPMO45Code2Wav._npu_same_shape_groups
    assert group(items, states, features, cross_turn=False, mixed_history=True) == [[0, 1]]
    assert group(items, states, features, cross_turn=False) == [[0], [1]]
    states[1].hift_cache["mel"] = torch.nn.functional.pad(states[1].hift_cache["mel"], (0, 1))
    assert group(items, states, features, cross_turn=False, mixed_history=True) == [[0], [1]]


def test_ragged_npu_staggered_final_and_fresh_request():
    model, _, calls = _candidate()
    singles = {row: _candidate(False)[0] for row in range(3)}
    for chunk in range(10):
        # Row 0 finishes while row 1 continues; row 2 then joins as a fresh row.
        rows = [0, 1] if chunk <= 4 else [1, 2]
        infos = [
            _request(
                row, chunk if row != 2 else chunk - 5, count=12 + row * 2, last=(row == 0 and chunk == 4) or chunk == 9
            )
            for row in rows
        ]
        calls.clear()
        actual = _forward(model, infos)
        if chunk == 4:
            assert calls == [2], "Mixed final flags must share CFM"
        if chunk == 5:
            assert calls == [1, 1], "Fresh and continuing rows retain separate buckets"
        for index, row in enumerate(rows):
            expected = _forward(singles[row], [infos[index]])
            torch.testing.assert_close(
                actual.multimodal_outputs["model_outputs"][index], expected.multimodal_outputs["model_outputs"][0]
            )
            key = f"voice-{row}"
            assert (key in model._states) == (key in singles[row]._states)
            if key in model._states:
                for kind in ("flow_cache", "hift_cache"):
                    got = getattr(model._states[key].token2wav, kind)
                    wanted = getattr(singles[row]._states[key].token2wav, kind)
                    for name in wanted:
                        torch.testing.assert_close(got[name], wanted[name])
        if chunk >= 4:
            assert "voice-0" not in model._states
    assert not model._states


@pytest.mark.parametrize(
    "policy,expected",
    [
        ("eligible", True),
        ("disabled", False),
        ("fp16", False),
        ("slots", False),
        ("bf16_cache", False),
        ("unfused", False),
        ("no_ragged", False),
        ("disabled_graph", False),
        ("minimum", False),
        ("cpu", False),
        ("no_wrapper", False),
    ],
)
def test_backend_initialization_gates_actual_wrapper_flag(policy, expected, monkeypatch, tmp_path):
    import vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav as stage_module
    import vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_token2wav as core_module

    model, core, _ = _candidate()
    backend = model.backend
    wrapper = backend._whole_euler_graph_wrapper
    # A stale True must be overwritten when configuration/admission rejects it.
    wrapper.npu_mixed_history = True
    if policy == "disabled":
        model._npu_cross_reference_batching = False
    elif policy == "fp16":
        wrapper.dtype = torch.float16
    elif policy == "slots":
        wrapper.att_slots = 4
    elif policy == "bf16_cache":
        backend._estimator_att_cache_dtype = torch.bfloat16
    elif policy == "unfused":
        backend._ragged_fused_body = False
    elif policy == "no_ragged":
        wrapper.ragged_body = None
    elif policy == "disabled_graph":
        wrapper.enabled = False
    elif policy == "minimum":
        model._min_batch_size = 2
    elif policy == "cpu":
        wrapper.device = SimpleNamespace(type="cpu")
    elif policy == "no_wrapper":
        backend._whole_euler_graph_wrapper = None

    (tmp_path / "assets" / "token2wav").mkdir(parents=True)
    prompt = tmp_path / "prompt.wav"
    prompt.touch()
    extra = model.vllm_config.model_config.stage_connector_config["extra"]
    extra.update(
        prompt_wav=str(prompt), token2wav_device_spk=False, token2wav_trt=False, token2wav_spk_cpu_reference=False
    )
    model._prompt_wav_override = str(prompt)
    monkeypatch.delenv("MINICPMO_TOKEN2WAV_TRT", raising=False)
    monkeypatch.setattr(stage_module, "_resolve_model_dir", lambda *args: str(tmp_path))
    monkeypatch.setattr(core_module, "MiniCPMO45Token2wav", lambda *args, **kwargs: core)
    monkeypatch.setattr(stage_module, "BatchedToken2Wav", lambda *args, **kwargs: backend)
    warning = Mock()
    monkeypatch.setattr(stage_module.logger, "warning", warning)
    model.backend = None
    model._build_backend()

    assert model.backend is backend
    if policy != "no_wrapper":
        assert wrapper.npu_mixed_history is expected
    else:
        assert wrapper.npu_mixed_history is True
    inactive_warnings = [
        call for call in warning.call_args_list if "NPU cross-reference CFM batching is inactive" in str(call)
    ]
    assert bool(inactive_warnings) is (policy not in ("eligible", "disabled"))

    reasons = {
        "fp16": "wrapper dtype must be FP32",
        "slots": "attention slot pool must be disabled",
        "bf16_cache": "attention cache dtype must be FP32",
        "unfused": "fused ragged body is required",
        "no_ragged": "ragged Whole-Euler body is unavailable",
        "disabled_graph": "Whole-Euler graph is disabled",
        "minimum": "code2wav_min_batch_size must be 1",
        "cpu": "wrapper device must be NPU",
        "no_wrapper": "Whole-Euler wrapper is unavailable",
    }
    if policy in reasons:
        assert len(inactive_warnings) == 1
        assert inactive_warnings[0].args[1] == reasons[policy]
