# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""V1 device codec state: EOS, rolling history, output ownership and sampling."""

import pytest
import torch
from torch.overrides import TorchFunctionMode

from tests.helpers.mark import hardware_marks, hardware_test
from tests.model_executor.models.minicpmo_4_5.test_talker_batching import (
    _CodecSamplingMetadata,
    _make_talker,
    _reference_repetition_penalty,
)
from vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni import _extract_codec_delta
from vllm_omni.utils.mm_outputs import build_mm_cpu, to_payload_element

pytestmark = [pytest.mark.core_model]


@pytest.fixture(
    params=[
        pytest.param("cuda", marks=hardware_marks(res={"cuda": "L4"}, num_cards=1)),
        pytest.param("npu", marks=hardware_marks(res={"npu": "A3"}, num_cards=1)),
    ]
)
def device(request):
    backend = getattr(torch, request.param, None)
    if backend is None or not backend.is_available():
        pytest.skip(f"requires {request.param}")
    return request.param


class NoHostRead(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func in (torch.Tensor.item, torch.Tensor.tolist, torch.Tensor.__bool__, torch.Tensor.__int__):
            assert args[0].device.type not in ("cuda", "npu"), "device codec state must not be read on the host"
        if func in (torch.Tensor.to, torch.Tensor.cpu) and args[0].device.type in ("cuda", "npu"):
            if func is torch.Tensor.cpu or "cpu" in str(args[1:]) + str(kwargs):
                raise AssertionError("device codec state must not be copied to the host")
        return func(*args, **(kwargs or {}))


def test_codec_slots_grow_for_preempted_requests_without_losing_state(device):
    from vllm_omni.model_executor.models.minicpmo_4_5.codec_state import CodecState

    codec = CodecState(torch.device(device), 8, capacity=1)
    first: dict[str, int] = {}
    second: dict[str, int] = {}
    with NoHostRead():
        codec.update([("preempted", first, 2)], torch.tensor([2], device=device), eos=7, cadence=25, boundary=2)
        # max_num_seqs bounds running requests, not retained preempted requests.
        codec.update([("new", second, 3)], torch.tensor([3], device=device), eos=7, cadence=25, boundary=2)
        history, valid, done, _ = codec.update(
            [("preempted", first, 4)], torch.tensor([4], device=device), eos=7, cadence=25, boundary=2
        )
    assert first["_gpu_slot"] != second["_gpu_slot"]
    assert history[0, -2:].tolist() == [2, 4]
    assert valid.item() and not done.item()
    assert codec.state[first["_gpu_slot"], 0].item() == 2
    assert codec.state[second["_gpu_slot"], 0].item() == 1


def test_v1_gpu_codec_state_matches_cpu_over_reordered_mixed_steps(device):
    gpu, cpu = _make_talker(), _make_talker()
    gpu.emb_code = torch.nn.ModuleList([torch.nn.Embedding(8, 2, device=device)])
    cpu.emb_code = torch.nn.ModuleList([torch.nn.Embedding(8, 2)])
    gpu._request_audio_states = {
        "a": {"finished": False, "step": 23, "max_tokens": 26, "min_tokens": 26, "recent_codes": [1] * 16},
        "b": {"finished": False, "step": 0, "turn_end_drain": True, "recent_codes": [2, 3]},
        "c": {"finished": False, "step": 24, "max_tokens": 101, "turn_end_drain": True},
    }
    import copy

    cpu._request_audio_states = copy.deepcopy(gpu._request_audio_states)
    snapshots = []
    for req_ids, values in [(list("abc"), [4, 7, 5]), (list("cab"), [6, 2, 3]), (list("bca"), [1, 1, 1])]:
        infos = [{"request_id": rid, "audio_state": gpu._request_audio_states[rid]} for rid in req_ids]
        ids = torch.tensor(values, device=device)
        with NoHostRead():
            _, _, updates = gpu.preprocess_decode_batch(input_ids=ids, req_infos=infos)
            gpu_infos = [dict(info, **update) for info, update in zip(infos, updates)]
            out = gpu.make_omni_output(
                torch.ones(3, 2, device=device),
                model_intermediate_buffer=gpu_infos,
                request_token_spans=[(0, 1), (1, 2), (2, 3)],
            )
            logits = torch.arange(-4, 4, device=device).float().expand(3, -1)
            actual, _ = gpu._apply_codec_repetition_penalty(
                logits, _CodecSamplingMetadata(torch.tensor([1.05, 1.1, 1.2], device=device))
            )
        cpu_infos = []
        for rid, value in zip(req_ids, values):
            _, _, update = cpu.preprocess(
                torch.tensor([value]), None, request_id=rid, audio_state=cpu._request_audio_states[rid]
            )
            cpu_infos.append(dict(request_id=rid, **update))
        expected_out = cpu.make_omni_output(
            torch.ones(3, 2), model_intermediate_buffer=cpu_infos, request_token_spans=[(0, 1), (1, 2), (2, 3)]
        )
        expected, _ = cpu._apply_codec_repetition_penalty(
            logits.cpu(), _CodecSamplingMetadata(torch.tensor([1.05, 1.1, 1.2]))
        )
        torch.testing.assert_close(actual.cpu(), expected)
        mm = build_mm_cpu(out.multimodal_outputs)
        for row, rid in enumerate(req_ids):
            payload = to_payload_element(mm, row, row, row + 1, seq_len=3)
            reference = to_payload_element(expected_out.multimodal_outputs, row, row, row + 1, seq_len=3)
            assert _extract_codec_delta(payload, rid) == _extract_codec_delta(reference, rid)
            assert payload["meta"]["finished"].item() == reference["meta"]["finished"].item()
            state = gpu._request_audio_states[rid]
            assert gpu._device_codec_state.state[state["_gpu_slot"], 0].item() == cpu._request_audio_states[rid]["step"]
        torch.testing.assert_close(gpu._force_eos_rows.cpu(), torch.tensor(cpu._force_eos_rows))
        torch.testing.assert_close(gpu._mask_eos_rows.cpu(), torch.tensor(cpu._mask_eos_rows))
        snapshots.append((out, mm))
        ids.fill_(0)
        torch.testing.assert_close(out.multimodal_outputs["codes"]["audio"][0].cpu(), mm["codes"]["audio"][0])
    # Previous payloads remain owned when subsequent input/state buffers change.
    for output, snapshot in snapshots:
        for codec_delta, reference in zip(output.multimodal_outputs["codes"]["audio"], snapshot["codes"]["audio"]):
            torch.testing.assert_close(codec_delta.cpu(), reference)
    gpu.on_requests_finished(set("abc"))
    gpu._flush_deferred_cleanup()
    assert not gpu._request_audio_states and not gpu._decode_codec_id_map()


def test_v1_gpu_eos_masks_apply_without_host_reads(device):
    from types import SimpleNamespace

    talker = _make_talker().to(device)
    talker._force_eos_rows = torch.tensor([False, True], device=device)
    talker._mask_eos_rows = torch.tensor([True, False], device=device)
    with NoHostRead():
        logits = talker.compute_logits(torch.ones(2, 2, device=device))
        sampled = SimpleNamespace(sampled_token_ids=torch.tensor([[3], [4]], device=device))
        talker._force_eos_on_sampled_ids(sampled, talker._pending_force_eos_rows)
    assert logits[0, 7].item() == float("-inf")
    assert logits[1, 7].item() == 0
    assert sampled.sampled_token_ids.cpu().tolist() == [[3], [7]]


def test_v1_gpu_history_survives_duplex_prefill_and_mixed_batch(device, mocker):
    talker = _make_talker()
    talker.emb_code = torch.nn.ModuleList([torch.nn.Embedding(8, 2, device=device)])
    talker._tts_config.attention_type = "other"
    condition = torch.ones(2, 2, device=device)
    mocker.patch.object(talker, "_build_condition_embeddings", return_value=condition)
    common = dict(
        request_id="a",
        native_duplex=True,
        _omni_is_prefill=True,
        _omni_prompt_len=2,
        tts_token_ids=torch.tensor([1]),
        tts_hidden_states=torch.ones(1, 2),
    )
    _, _, first = talker.preprocess(
        torch.zeros(2, dtype=torch.long, device=device), None, meta={"streaming_condition_seq": 0}, **common
    )
    talker.make_omni_output(
        condition, model_intermediate_buffer=[dict(request_id="a", **first)], request_token_spans=[(0, 2)]
    )
    for token in [1, 2, 3]:
        _, _, updates = talker.preprocess_decode_batch(
            input_ids=torch.tensor([token], device=device),
            req_infos=[dict(request_id="a", audio_state=talker._request_audio_states["a"])],
        )
        talker.make_omni_output(
            condition[:1], model_intermediate_buffer=[dict(request_id="a", **updates[0])], request_token_spans=[(0, 1)]
        )
    old_slot = talker._request_audio_states["a"]["_gpu_slot"]
    with NoHostRead():
        _, _, update = talker.preprocess(
            torch.zeros(2, dtype=torch.long, device=device), None, meta={"streaming_condition_seq": 1}, **common
        )
        output = talker.make_omni_output(
            condition, model_intermediate_buffer=[dict(request_id="a", **update)], request_token_spans=[(0, 2)]
        )
    assert talker._request_audio_states["a"]["_gpu_slot"] == old_slot
    assert talker._device_codec_state.state[old_slot, 0].item() == 0
    assert output.multimodal_outputs["codes"]["audio"][0].numel() == 0
    assert talker._penalty_histories[0].cpu().tolist() == [8] * 13 + [1, 2, 3]
    # A GPU decode row and a CPU prefill row score their own histories.
    _, _, updates = talker.preprocess_decode_batch(
        input_ids=torch.tensor([4], device=device),
        req_infos=[dict(request_id="a", audio_state=talker._request_audio_states["a"])],
    )
    with NoHostRead():
        output = talker.make_omni_output(
            torch.ones(3, 2, device=device),
            model_intermediate_buffer=[dict(request_id="a", **updates[0]), dict(request_id="b", audio_state={})],
            request_token_spans=[(0, 1), (1, 3)],
        )
        actual, _ = talker._apply_codec_repetition_penalty(
            torch.ones(2, 8, device=device), _CodecSamplingMetadata(torch.tensor([1.05, 1.1], device=device))
        )
    expected = _reference_repetition_penalty(torch.ones(1, 8), torch.tensor([1, 2, 3, 4]), penalty=1.05, window_size=16)
    torch.testing.assert_close(actual[0:1].cpu(), expected)
    torch.testing.assert_close(actual[1].cpu(), torch.ones(8))
    assert _extract_codec_delta(
        to_payload_element(build_mm_cpu(output.multimodal_outputs), 0, 0, 1, seq_len=3), "a"
    ) == [4]
    assert (
        _extract_codec_delta(to_payload_element(build_mm_cpu(output.multimodal_outputs), 1, 1, 3, seq_len=3), "b") == []
    )


@pytest.mark.parametrize("batch", [1, 4, 33])
def test_fused_gpu_window_penalty_matches_reference(device, batch):
    from vllm_omni.model_executor.models.minicpmo_4_5.codec_state import apply_window_penalty

    generator = torch.Generator().manual_seed(42)
    logits = torch.randn(batch, 6562, generator=generator)
    histories = torch.randint(0, 6563, (batch, 16), generator=generator)
    histories[0].fill_(17)
    histories[-1].fill_(17)
    penalties = torch.linspace(1.0, 1.5, batch)
    actual = apply_window_penalty(logits.to(device), histories.to(device), penalties.to(device))
    expected = torch.cat(
        [
            _reference_repetition_penalty(
                logits[row : row + 1],
                histories[row][histories[row] < 6562],
                penalty=float(penalties[row]),
                window_size=16,
            )
            for row in range(batch)
        ]
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-6, atol=1e-6)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_v1_gpu_output_uses_async_snapshot_and_reuses_finished_slot():
    from vllm_omni.worker.gpu_ar_model_runner import _snapshot_tensor_payload_to_cpu_async

    talker = _make_talker()
    talker.emb_code = torch.nn.ModuleList([torch.nn.Embedding(8, 2, device="cuda")])
    talker._request_audio_states["a"] = {"finished": False, "step": 0}
    ids = torch.tensor([2], device="cuda")
    _, _, updates = talker.preprocess_decode_batch(
        input_ids=ids, req_infos=[dict(request_id="a", audio_state=talker._request_audio_states["a"])]
    )
    output = talker.make_omni_output(
        torch.ones(1, 2, device="cuda"),
        model_intermediate_buffer=[dict(request_id="a", **updates[0])],
        request_token_spans=[(0, 1)],
    )
    slot = talker._request_audio_states["a"]["_gpu_slot"]
    snapshot = _snapshot_tensor_payload_to_cpu_async(
        output.multimodal_outputs, copy_stream=torch.cuda.Stream(), pin_memory=True
    )
    # Scalar fallback after GPU batching must still emit the device codec ID.
    with NoHostRead():
        _, _, update = talker.preprocess(
            torch.tensor([3], device="cuda"), None, request_id="a", audio_state=talker._request_audio_states["a"]
        )
        second = talker.make_omni_output(
            torch.ones(1, 2, device="cuda"),
            model_intermediate_buffer=[dict(request_id="a", **update)],
            request_token_spans=[(0, 1)],
        )
    ids.zero_()
    talker.on_requests_finished({"a"})
    talker._flush_deferred_cleanup()
    assert not talker._device_codec_state.slots
    talker._request_audio_states["b"] = {"finished": False, "step": 0}
    _, _, updates = talker.preprocess_decode_batch(
        input_ids=torch.tensor([4], device="cuda"),
        req_infos=[dict(request_id="b", audio_state=talker._request_audio_states["b"])],
    )
    talker.make_omni_output(
        torch.ones(1, 2, device="cuda"),
        model_intermediate_buffer=[dict(request_id="b", **updates[0])],
        request_token_spans=[(0, 1)],
    )
    assert talker._request_audio_states["b"]["_gpu_slot"] == slot
    snapshot.wait()
    assert _extract_codec_delta(to_payload_element(snapshot.payload, 0, 0, 1, seq_len=1), "a") == [2]
    assert _extract_codec_delta(
        to_payload_element(build_mm_cpu(second.multimodal_outputs), 0, 0, 1, seq_len=1), "a"
    ) == [3]


def test_gpu_decode_masks_finished_request_when_new_request_joins(device):
    talker = _make_talker()
    talker.emb_code = torch.nn.ModuleList([torch.nn.Embedding(8, 2, device=device)])
    talker._request_audio_states["a"] = {"finished": False, "step": 0}
    _, _, updates = talker.preprocess_decode_batch(
        input_ids=torch.tensor([7], device=device),
        req_infos=[dict(request_id="a", audio_state=talker._request_audio_states["a"])],
    )
    talker.make_omni_output(
        torch.ones(1, 2, device=device),
        model_intermediate_buffer=[dict(request_id="a", **updates[0])],
        request_token_spans=[(0, 1)],
    )
    talker._request_audio_states["b"] = {"finished": False, "step": 0}
    with NoHostRead():
        _, embeds, updates = talker.preprocess_decode_batch(
            input_ids=torch.tensor([2, 3], device=device),
            req_infos=[dict(request_id=rid, audio_state=talker._request_audio_states[rid]) for rid in ["a", "b"]],
        )
        output = talker.make_omni_output(
            torch.ones(2, 2, device=device),
            model_intermediate_buffer=[dict(request_id=rid, **update) for rid, update in zip(["a", "b"], updates)],
            request_token_spans=[(0, 1), (1, 2)],
        )
    torch.testing.assert_close(embeds[0].cpu(), torch.zeros(2))
    torch.testing.assert_close(embeds[1], talker.emb_code[0](torch.tensor(3, device=device)))
    mm = build_mm_cpu(output.multimodal_outputs)
    assert _extract_codec_delta(to_payload_element(mm, 0, 0, 1, seq_len=2), "a") == []
    assert _extract_codec_delta(to_payload_element(mm, 1, 1, 2, seq_len=2), "b") == [3]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("aux_penalty", ["neutral", "frequency", "presence", "mixed"])
def test_sampler_fast_path_preserves_seeded_tokens_and_auxiliary_penalties(aux_penalty, mocker):
    from dataclasses import replace
    from types import SimpleNamespace

    from vllm.sampling_params import SamplingParams
    from vllm.v1.sample.logits_processor import LogitsProcessors
    from vllm.v1.sample.metadata import SamplingMetadata
    from vllm.v1.sample.sampler import Sampler

    from vllm_omni.model_executor.models.minicpmo_4_5.codec_state import apply_window_penalty
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )
    from vllm_omni.worker.sampling_utils import call_model_sampler

    batch, vocab = 3, 64
    params = [
        SamplingParams(
            temperature=0.8,
            top_k=25,
            top_p=0.85,
            repetition_penalty=1.05,
            frequency_penalty=0.2 if aux_penalty == "frequency" or (aux_penalty == "mixed" and row == 1) else 0,
            presence_penalty=-0.2 if aux_penalty == "presence" else 0,
            seed=42 + row,
        )
        for row in range(batch)
    ]

    def metadata():
        return SamplingMetadata(
            temperature=torch.tensor([p.temperature for p in params], device="cuda"),
            all_greedy=False,
            all_random=True,
            top_p=torch.tensor([p.top_p for p in params], device="cuda"),
            top_k=torch.tensor([p.top_k for p in params], device="cuda", dtype=torch.int32),
            generators={row: torch.Generator(device="cuda").manual_seed(p.seed) for row, p in enumerate(params)},
            max_num_logprobs=None,
            no_penalties=False,
            prompt_token_ids=torch.zeros(batch, 8, device="cuda", dtype=torch.long),
            frequency_penalties=torch.tensor([p.frequency_penalty for p in params], device="cuda"),
            presence_penalties=torch.tensor([p.presence_penalty for p in params], device="cuda"),
            repetition_penalties=torch.tensor([p.repetition_penalty for p in params], device="cuda"),
            output_token_ids=[[row] * 32 for row in range(batch)],
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=LogitsProcessors(),
        )

    actual_meta, reference_meta = metadata(), metadata()
    talker = _make_talker()
    talker._num_audio_tokens, talker._codec_eos_id = vocab, vocab - 1
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.model_stage = "tts"
    model.model_sampler_wants_sampling_params = True
    model.model = talker
    input_batch = SimpleNamespace(req_ids=["c", "a", "b"])
    requests = {rid: SimpleNamespace(sampling_params=params[row]) for row, rid in enumerate(input_batch.req_ids)}
    reference_sampler = Sampler()
    calls = mocker.spy(talker._codec_sampler, "forward")
    raw = torch.arange(batch * vocab, device="cuda").float().reshape(batch, vocab).sin()
    for step in range(20):
        histories = torch.tensor([ids[-16:] for ids in reference_meta.output_token_ids], device="cuda")
        talker._penalty_histories = histories
        forced = torch.tensor([step == 19, False, False], device="cuda")
        talker._pending_force_eos_rows = forced
        expected_logits = apply_window_penalty(raw.clone(), histories, reference_meta.repetition_penalties)
        prepared = replace(
            reference_meta,
            prompt_token_ids=torch.full_like(reference_meta.prompt_token_ids, vocab),
            repetition_penalties=torch.ones_like(reference_meta.repetition_penalties),
        )
        expected = reference_sampler(expected_logits, prepared)
        expected.sampled_token_ids.masked_fill_(forced[:, None], vocab - 1)
        with NoHostRead():
            actual = call_model_sampler(
                model, model.sample, raw.clone(), actual_meta, input_batch=input_batch, requests=requests
            )
        torch.testing.assert_close(actual.sampled_token_ids, expected.sampled_token_ids, rtol=0, atol=0)
        passed_metadata = calls.call_args.args[1]
        assert passed_metadata.no_penalties is (aux_penalty == "neutral")
        assert actual_meta.no_penalties is False
        assert actual_meta.prompt_token_ids is not None
        torch.testing.assert_close(actual_meta.prompt_token_ids, torch.zeros_like(actual_meta.prompt_token_ids))
        tokens = actual.sampled_token_ids.cpu().reshape(-1).tolist()
        for row, token in enumerate(tokens):
            actual_meta.output_token_ids[row].append(token)
            reference_meta.output_token_ids[row].append(token)
