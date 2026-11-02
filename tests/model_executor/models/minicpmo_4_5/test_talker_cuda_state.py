# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.talker_codec_sample import (
    HISTORY_WINDOW,
    VOCAB_SIZE,
    TalkerCodecDeviceState,
    TalkerCudaKStepState,
    codec_sample_result,
    prepare_codec_logits,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def workspace(rows=2, frames=3):
    state = TalkerCudaKStepState.allocate(rows, frames, torch.device("cpu"))
    state.temperatures.zero_()
    state.max_tokens.fill_(10)
    return state


def logits(tokens):
    result = torch.full((len(tokens), VOCAB_SIZE), -100.0)
    result[torch.arange(len(tokens)), tokens] = 10
    return result


def test_eos_suppresses_remaining_frames_without_affecting_other_request():
    state = workspace()
    pointers = [state.history.data_ptr(), state.steps.data_ptr(), state.finished.data_ptr()]
    state.sample(logits([VOCAB_SIZE - 1, 7]), 0, VOCAB_SIZE - 1)
    state.sample(logits([4, 8]), 1, VOCAB_SIZE - 1)
    assert state.steps.tolist() == [1, 2]
    assert state.emits[:2].tolist() == [[False, True], [False, True]]
    assert state.stops[:2].tolist() == [[True, False], [True, False]]
    assert state.history[0].tolist() == [-1] * HISTORY_WINDOW
    assert state.history[1, -2:].tolist() == [7, 8]
    assert pointers == [state.history.data_ptr(), state.steps.data_ptr(), state.finished.data_ptr()]


def test_stop_and_max_length_semantics():
    state = workspace()
    state.stop_mask[0, 3] = True
    state.max_tokens[1] = 1
    state.sample(logits([3, 4]), 0, VOCAB_SIZE - 1)
    assert state.finished.tolist() == [True, True]
    assert state.emits[0].tolist() == [False, False]
    assert state.last_tokens.tolist() == [0, 0]


@pytest.mark.parametrize("token,maximum", [(4, 10), (4, 1), (3, 10), (VOCAB_SIZE - 1, 10)])
def test_stop_and_length_match_native_state(token, maximum):
    state = workspace(rows=1)
    state.stop_mask[0, 3] = True
    state.max_tokens.fill_(maximum)
    native = TalkerCodecDeviceState(
        history=torch.zeros((1, HISTORY_WINDOW), dtype=torch.int32),
        history_len=torch.zeros(1, dtype=torch.int32),
        step=torch.zeros(1, dtype=torch.int32),
        max_tokens=torch.tensor([maximum], dtype=torch.int32),
        finished=torch.zeros(1, dtype=torch.bool),
    )
    for frame in range(3):
        result = codec_sample_result(native, torch.tensor([token]), eos_token_id=VOCAB_SIZE - 1, stop_token_ids=(3,))
        native = result.state
        state.sample(logits([token]), frame, VOCAB_SIZE - 1)
        assert state.steps.tolist() == native.step.tolist()
        assert state.emits[frame].tolist() == result.emit.tolist()
        assert state.stops[frame].tolist() == native.finished.tolist()
        length = int(native.history_len.item())
        if length:
            assert state.history[0, -length:].tolist() == native.history[0, :length].tolist()


def test_min_tokens_masks_eos_and_request_stop_ids():
    state = workspace()
    state.min_tokens.fill_(2)
    state.stop_mask[:, 3] = True
    scores = logits([VOCAB_SIZE - 1, 3])
    scores[:, 4] = 5
    state.sample(scores, 0, VOCAB_SIZE - 1)
    assert state.tokens[0].tolist() == [4, 4]
    assert not state.finished.any()


def test_penalty_counts_duplicate_history_tokens():
    state = workspace(rows=1)
    state.penalties.fill_(2)
    state.history[0, -2:] = 5
    scores = logits([5])
    scores[0, 6] = 4
    state.sample(scores, 0, VOCAB_SIZE - 1)
    assert state.tokens[0].item() == 6


def test_stochastic_topk_and_top_p_use_external_noise():
    state = workspace(rows=1)
    state.temperatures.fill_(1)
    state.top_k.fill_(2)
    state.uniforms.fill_(0.99)
    scores = logits([5])
    scores[0, 6] = 10
    state.sample(scores, 0, VOCAB_SIZE - 1)
    assert state.tokens[0].item() in (5, 6)
    state.top_p.fill_(0.1)
    state.sample(scores, 1, VOCAB_SIZE - 1)
    assert state.tokens[1].item() in (5, 6)


def test_snapshot_restores_all_fields_without_rebinding_storage():
    state = workspace(rows=1)
    state.uniforms.fill_(0.2)
    initial = state.snapshot()
    key = state.storage_key()
    state.sample(logits([5]), 0, VOCAB_SIZE - 1)
    for tensor in state._tensor_fields().values():
        tensor.zero_()
    state.restore(initial)
    assert state.storage_key() == key
    for name, tensor in state._tensor_fields().items():
        assert torch.equal(tensor, initial[name])
        assert tensor.data_ptr() != initial[name].data_ptr()
    state.sample(logits([5]), 0, VOCAB_SIZE - 1)
    assert state.steps.tolist() == [1]
    assert state.history[0, -1].item() == 5


def test_external_noise_changes_sample_without_changing_storage_key():
    state = workspace(rows=1)
    state.temperatures.fill_(1)
    state.top_k.fill_(2)
    scores = logits([5])
    scores[0, 6] = 10
    initial = state.snapshot()
    key = state.storage_key()
    state.uniforms.fill_(0.1)
    state.sample(scores, 0, VOCAB_SIZE - 1)
    first = state.tokens[0].clone()
    state.restore(initial)
    state.uniforms.fill_(0.9)
    state.sample(scores, 0, VOCAB_SIZE - 1)
    assert not torch.equal(state.tokens[0], first)
    assert state.storage_key() == key


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("heterogeneous", [False, True])
@pytest.mark.parametrize("fused", [False, True])
def test_native_sampling_matches_independent_generators_and_history(device, heterogeneous, fused):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    state = TalkerCudaKStepState.allocate(4, 3, torch.device(device))
    state.temperatures.copy_(torch.tensor([0.8, 1.2, 0.6, 0.0 if heterogeneous else 0.9], device=device))
    state.penalties.copy_(torch.tensor([1.1, 1.2, 1.0, 0.9], device=device))
    state.max_tokens.fill_(100)
    state.min_tokens.fill_(5)
    state.steps.copy_(torch.tensor([0, 6, 2, 5], device=device))
    state.history[0, -3:] = torch.tensor([5, 5, 7], device=device)
    state.history[1, -2:] = torch.tensor([8, 9], device=device)
    state.history[3, :] = 10
    actual = tuple(torch.Generator(device=device).manual_seed(seed) for seed in (123, 456, 789, 101))
    expected = tuple(torch.Generator(device=device).manual_seed(seed) for seed in (123, 456, 789, 101))
    config = (
        (0.8, 25, 0.9, (3,)),
        (1.2, 10 if heterogeneous else 25, 0.7 if heterogeneous else 0.9, (4,) if heterogeneous else (3,)),
        (0.6, 25, 0.9, (3,)),
        (0.0 if heterogeneous else 0.9, 25, 0.9, (3,)),
    )
    for row, (_, _, _, stop_ids) in enumerate(config):
        state.stop_mask[row, list(stop_ids)] = True

    if fused:
        if device == "cuda":
            pytest.importorskip("triton")
        assert state.warmup_native_prelude(VOCAB_SIZE - 1) == (device == "cuda")

    source = torch.Generator().manual_seed(789)
    for frame in range(3):
        scores = torch.randn(4, VOCAB_SIZE, generator=source).to(device)
        tokens = []
        for row, generator in enumerate(expected):
            history = state.history[row][state.history[row] >= 0].reshape(1, -1)
            padded = torch.zeros((1, HISTORY_WINDOW), device=device, dtype=torch.long)
            padded[:, : history.numel()] = history
            native = TalkerCodecDeviceState(
                history=padded,
                history_len=torch.tensor([history.numel()], device=device),
                step=state.steps[row : row + 1].clone(),
                max_tokens=state.max_tokens[row : row + 1],
                finished=state.finished[row : row + 1],
            )
            temperature, top_k, top_p, stop_ids = config[row]
            filtered = prepare_codec_logits(
                scores[row : row + 1],
                native,
                state.min_tokens[row : row + 1],
                state.temperatures[row : row + 1] if temperature else state.temperatures.new_ones(1),
                state.penalties[row : row + 1],
                eos_token_id=VOCAB_SIZE - 1,
                top_k=top_k if temperature else 0,
                top_p=top_p if temperature else 1.0,
                min_tokens_stop_ids=stop_ids,
                fused_prelude=False,
            )
            tokens.append(
                torch.multinomial(filtered.softmax(-1), 1, generator=generator).reshape(1)
                if temperature
                else filtered.argmax(dim=-1)
            )
        state.sample_native(scores, frame, VOCAB_SIZE - 1, config, actual)
        assert torch.equal(state.tokens[frame], torch.cat(tokens))
        for observed, reference in zip(actual, expected):
            assert torch.equal(observed.get_state(), reference.get_state())
    continuation = torch.ones((1, VOCAB_SIZE), device=device)
    for observed, reference in zip(actual, expected):
        assert torch.equal(
            torch.multinomial(continuation, 1, generator=observed),
            torch.multinomial(continuation, 1, generator=reference),
        )


def test_native_prelude_host_metadata_does_not_change_snapshot_or_storage():
    state = workspace(rows=1)
    snapshot = state.snapshot()
    key = state.storage_key()
    assert not state.warmup_native_prelude(VOCAB_SIZE - 1)
    state._native_prelude_eos = VOCAB_SIZE - 1
    assert state.storage_key() == key
    state.steps.add_(3)
    state.restore(snapshot)
    assert state.steps.tolist() == [0]
    assert set(state.snapshot()) == set(snapshot)


def test_native_prelude_capture_warmup_is_rejected(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    state = TalkerCudaKStepState.allocate(1, 1, torch.device("cuda"))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="outside capture"):
        state.warmup_native_prelude(VOCAB_SIZE - 1)


def test_native_prelude_launch_failure_is_not_retried_eager(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    state = TalkerCudaKStepState.allocate(1, 1, torch.device("cuda"))
    state._native_prelude_eos = VOCAB_SIZE - 1

    def fail(*args):
        raise RuntimeError("launch failed")

    monkeypatch.setattr(state, "_launch_native_prelude", fail)
    scores = torch.zeros((1, VOCAB_SIZE), device="cuda")
    with pytest.raises(RuntimeError, match="launch failed"):
        state.sample_native(scores, 0, VOCAB_SIZE - 1, ((0.0, 0, 1.0, ()),), (None,))


def test_model_binding_commit_and_batch_compaction():
    from types import SimpleNamespace

    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
        MiniCPMO45OmniTTSForConditionalGeneration,
    )

    model = SimpleNamespace(
        _request_audio_states={},
        _request_codec_history={},
        _request_generators={},
        _codec_min_tokens=0,
        _codec_max_tokens=10,
        _codec_temperature=0,
        _codec_repetition_penalty=1,
        _codec_top_k=0,
        _codec_top_p=1,
        _codec_seed=7,
        _codec_eos_id=VOCAB_SIZE - 1,
        _codec_fused_prelude=True,
        _merge_request_codec_params=lambda *args, **kwargs: None,
    )
    hidden = torch.zeros(2, 4)
    infos = [{"request_id": "a"}, {"request_id": "b"}]
    kwargs = {"model_intermediate_buffer": infos, "request_token_spans": [(0, 1), (1, 2)]}
    state = MiniCPMO45OmniTTSForConditionalGeneration.prepare_cuda_kstep(model, hidden, 2, **kwargs)
    address = state.history.data_ptr()
    storage_key = state.storage_key()

    def captured_frame():
        return state.sample(logits([6, 7]), 0, VOCAB_SIZE - 1)

    state.sample(logits([3, 4]), 0, VOCAB_SIZE - 1)
    state.sample(logits([VOCAB_SIZE - 1, 5]), 1, VOCAB_SIZE - 1)
    output = MiniCPMO45OmniTTSForConditionalGeneration.commit_cuda_kstep(model, hidden)
    deltas_out = output.multimodal_outputs["codes"]["audio"]
    valids_out = output.multimodal_outputs["meta"]["codec_frame_valid"]
    assert [d[v.detach().cpu().bool()].flatten().tolist() for d, v in zip(deltas_out, valids_out)] == [[3], [4, 5]]
    # The full-width delta itself keeps post-stop frames for the consumer crop.
    assert [delta.flatten().tolist() for delta in deltas_out] == [[3, VOCAB_SIZE - 1], [4, 5]]
    assert model._request_audio_states["a"]["step"] == 2
    assert model._request_audio_states["a"]["finished"]
    assert model.batch_stop_logits.argmax(-1).tolist() == [0, 1, 0, 0]
    kwargs["model_intermediate_buffer"] = [{"request_id": "b"}, {"request_id": "c"}]
    rebound = MiniCPMO45OmniTTSForConditionalGeneration.prepare_cuda_kstep(model, hidden, 2, **kwargs)
    assert rebound is state
    assert rebound.storage_key() == storage_key
    assert rebound.history.data_ptr() == address
    assert [flag.item() for flag in output.multimodal_outputs["meta"]["finished"]] == [True, False]
    assert rebound.history[0, -2:].tolist() == [4, 5]
    assert rebound.history[1].tolist() == [-1] * HISTORY_WINDOW
    assert rebound.steps.tolist() == [2, 0]
    assert not rebound.finished.any()
    assert model._request_generators == {}
    assert torch.equal(rebound.uniforms, torch.full_like(rebound.uniforms, 0.5))
    model._request_audio_states["b"]["finished"] = True
    model._request_audio_states["c"]["max_tokens"] = 1
    refreshed = MiniCPMO45OmniTTSForConditionalGeneration.prepare_cuda_kstep(model, hidden, 2, **kwargs)
    assert refreshed is state
    assert refreshed.storage_key() == storage_key
    assert refreshed.max_tokens.tolist() == [10, 1]
    captured_frame()
    assert refreshed.steps.tolist() == [2, 1]
    assert refreshed.finished.tolist() == [True, True]
    assert not refreshed.emits[0, 0]
    assert refreshed.history[0, -2:].tolist() == [4, 5]
    assert [flag.item() for flag in output.multimodal_outputs["meta"]["finished"]] == [True, False]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("eos_frame", [0, 1, 2])
def test_native_model_graph_eos_restores_request_rng(eos_frame):
    from types import MethodType, SimpleNamespace

    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
        MiniCPMO45OmniTTSForConditionalGeneration as Model,
    )
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
        _flush_cuda_kstep_host_pending,
    )
    from vllm_omni.worker.talker_cuda_graph import CUDANarrowStep

    generators = {name: torch.Generator(device="cuda").manual_seed(seed) for name, seed in (("a", 123), ("b", 456))}
    references = {name: torch.Generator(device="cuda").manual_seed(seed) for name, seed in (("a", 123), ("b", 456))}
    model = SimpleNamespace(
        _request_audio_states={},
        _request_codec_history={},
        _codec_temperature=0.8,
        _codec_top_p=1.0,
        _codec_top_k=0,
        _codec_repetition_penalty=1.0,
        _codec_min_tokens=0,
        _codec_max_tokens=100,
        _codec_eos_id=VOCAB_SIZE - 1,
        _codec_seed=7,
        _codec_fused_prelude=True,
        audio_bos_token_id=VOCAB_SIZE - 2,
        _control_logits_dtype=torch.float32,
        head_code=[lambda hidden: hidden],
        emb_code=[lambda tokens: torch.zeros((tokens.numel(), VOCAB_SIZE), device="cuda")],
        _request_generator=lambda request_id, device: generators[request_id],
        _merge_request_codec_params=lambda *args, **kwargs: None,
    )
    for name in ("prepare_cuda_kstep", "cuda_kstep_frame", "commit_cuda_kstep"):
        setattr(model, name, MethodType(getattr(Model, name), model))
    hidden = torch.zeros((2, VOCAB_SIZE), device="cuda")
    narrow = CUDANarrowStep(
        2,
        3,
        hidden,
        torch.zeros(2, device="cuda", dtype=torch.long),
        torch.zeros((2, 3), device="cuda", dtype=torch.long),
        torch.zeros((2, 3), device="cuda", dtype=torch.long),
        torch.ones((2, 3), device="cuda", dtype=torch.int),
        {},
    )
    scores = torch.full((3, 2, VOCAB_SIZE), -torch.inf, device="cuda")
    scores[:, :, 7] = 0
    scores[eos_frame, 0, 7] = -torch.inf
    scores[eos_frame, 0, VOCAB_SIZE - 1] = 0
    for iteration in range(4):
        order = ("a", "b") if iteration < 2 else ("b", "a")
        model._request_audio_states.clear()
        model._request_codec_history.clear()
        state = model.prepare_cuda_kstep(
            hidden,
            3,
            model_intermediate_buffer=[{"request_id": name} for name in order],
            request_token_spans=[(0, 1), (1, 2)],
        )
        assert state is not None

        graph_generators = model._cuda_kstep_generators

        def snapshot():
            return state.snapshot(), tuple(g.get_state() for g in graph_generators)

        def restore(saved):
            state.restore(saved[0])
            for generator, saved_state in zip(graph_generators, saved[1]):
                generator.set_state(saved_state)

        def forward():
            for frame in range(3):
                model.cuda_kstep_frame(scores[frame], frame)
            return state.tokens

        narrow.run_fused(
            forward,
            state_key=(state.storage_key(), tuple(id(g) for g in graph_generators)),
            snapshot=snapshot,
            restore=restore,
            generators=graph_generators,
        )
        result = model.commit_cuda_kstep(hidden)
        # Full-width deltas plus the emit mask: the consumer crops
        # (``meta.codec_frame_valid`` contract), so assert on the cropped rows.
        deltas_out = result.multimodal_outputs["codes"]["audio"]
        valids_out = result.multimodal_outputs["meta"]["codec_frame_valid"]
        assert [d[v.detach().cpu().bool()].flatten().tolist() for d, v in zip(deltas_out, valids_out)] == [
            [7] * eos_frame,
            [7] * 3,
        ]
        for valid in valids_out:
            # Sticky prefix semantics: once invalid, every later frame stays
            # invalid; a pre-finished row is fully invalid.
            flags = valid.cpu().bool().tolist()
            assert all(not f for f in flags[flags.index(False) :]) if False in flags else True
        # Commit only defers the host bookkeeping (RNG rewind included); the
        # next host-state reader would land it, so drive that seam explicitly
        # before asserting generator states.
        _flush_cuda_kstep_host_pending(model)
        for row, name in enumerate(order):
            draws = eos_frame + 1 if row == 0 else 3
            for frame in range(draws):
                torch.multinomial(scores[frame, row].softmax(-1), 1, generator=references[name])
            assert torch.equal(generators[name].get_state(), references[name].get_state())
            continuation = torch.ones(VOCAB_SIZE, device="cuda")
            assert torch.equal(
                torch.multinomial(continuation, 1, generator=generators[name]),
                torch.multinomial(continuation, 1, generator=references[name]),
            )
        assert narrow.captures == 1
        assert narrow.replays == iteration + 1


@pytest.mark.parametrize("params", [{"codec_temperature": 0.7}, {"codec_ignore_eos": True}])
def test_model_binding_declines_unsupported_sampling_without_rng(params):
    from types import SimpleNamespace

    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
        MiniCPMO45OmniTTSForConditionalGeneration,
    )

    generator = torch.Generator().manual_seed(42)
    before = generator.get_state().clone()
    model = SimpleNamespace(
        _request_audio_states={"a": params},
        _request_generators={"a": generator},
        _codec_temperature=0,
        _merge_request_codec_params=lambda *args, **kwargs: None,
    )
    result = MiniCPMO45OmniTTSForConditionalGeneration.prepare_cuda_kstep(
        model,
        torch.zeros(1, 4),
        2,
        model_intermediate_buffer=[{"request_id": "a"}],
        request_token_spans=[(0, 1)],
    )
    assert result is None
    assert not hasattr(model, "_cuda_kstep_workspaces")
    assert torch.equal(generator.get_state(), before)
