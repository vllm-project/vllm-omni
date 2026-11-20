# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU contract tests for speculative payload token alignment."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

from vllm_omni.data_entry_keys import deserialize_payload, flatten_payload, serialize_payload
from vllm_omni.model_executor.models.qwen3_omni.speculative_payload import (
    STAGING_KEY,
    apply_speculative_payload,
    finalize_speculative_payload,
    prepare_speculative_payload,
)
from vllm_omni.model_executor.stage_input_processors.qwen3_omni import (
    thinker2talker_async_chunk,
    thinker2talker_full_payload,
)
from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner
from vllm_omni.worker_v2.omni_data_plane import OmniRunnerDataPlane, _NativeRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _prepare(
    *,
    consumed=False,
    samples=((11, 12, -1),),
    inputs=(10, 11, 12),
    positions=(3, 4, 5),
    prompt_lens=(3,),
    invalid=(),
    req_ids=("r1",),
    speculative_ids=frozenset({"r1"}),
    offsets=(0, 3),
    embed=None,
    dtype=torch.float32,
):
    if embed is None:

        def embed(ids):
            return ids.to(dtype).view(-1, 1)

    return prepare_speculative_payload(
        multimodal_outputs={
            "hidden_states": {
                "layers": {
                    0: torch.tensor(inputs, dtype=dtype).view(-1, 1) * 10,
                    24: torch.ones(len(inputs), 1),
                }
            }
        },
        req_ids=req_ids,
        speculative_request_ids=speculative_ids,
        invalid_req_indices=invalid,
        sampled_token_ids=torch.tensor(samples),
        input_token_ids=torch.tensor(inputs),
        input_positions=torch.tensor(positions),
        query_start_loc=offsets,
        prompt_lens=prompt_lens,
        vocab_size=100,
        consumed_input_span=consumed,
        embed_input_ids=embed,
    )


@pytest.mark.parametrize("first_committed", [10, 11])
@pytest.mark.parametrize("first_position", [3, 4])
def test_prefill_seed_uses_input_position_even_for_repeated_tokens(first_committed, first_position):
    mm = finalize_speculative_payload(_prepare(samples=((first_committed, 12, -1),), positions=(first_position, 5, 6)))
    payload = apply_speculative_payload({"hidden": torch.ones(3, 1), "hidden_states.layer_24": torch.ones(3, 1)}, mm, 0)
    expected = [10, first_committed, 12] if first_position == 3 else [first_committed, 12]
    assert payload["ids"]["output"] == expected
    assert payload["hidden_states.layer_0"].flatten().tolist() == expected
    assert payload["meta"] == {"decode_span": True, "committed_token_count": len(expected)}
    assert "hidden" not in payload and "hidden_states.layer_24" not in payload


@pytest.mark.parametrize("samples", [((11, 12, -1),), ((-1, -1, -1),)])
def test_inline_batched_lookup_filters_sentinels_and_keeps_dtype(samples):
    dtype = torch.float16
    seen = []

    def embed(ids):
        seen.append(ids)
        assert bool(((ids >= 0) & (ids < 100)).all())
        return ids.to(dtype).view(-1, 1)

    result = finalize_speculative_payload(_prepare(samples=samples, embed=embed, dtype=dtype))[STAGING_KEY][0]
    expected = [10, 11, 12] if samples[0][0] != -1 else []
    assert result["ids"]["output"] == expected
    assert result["hidden_states.layer_0"].dtype == dtype
    assert len(seen) == 1


def test_other_models_keep_generic_pooler_payload():
    [payload] = OmniARModelRunner._build_pooler_output_from_cpu(
        hidden_cpu=torch.ones(3, 2),
        mm_cpu={"hidden_states.layer_0": torch.full((3, 2), 8.0), "hidden_states.layer_24": torch.full((3, 2), 7.0)},
        query_start_loc_np=np.array([0, 3]),
        num_scheduled_tokens=np.array([3]),
        num_reqs=1,
        req_ids=["r1"],
        engine_output_type="latent",
    )
    assert torch.equal(payload["hidden_states.layer_0"], torch.full((3, 2), 8.0))
    assert torch.equal(payload["hidden_states.layer_24"], torch.full((3, 2), 7.0))
    assert "hidden" in payload and "meta" not in payload and "ids" not in payload


def test_async_processor_emits_committed_decode_span() -> None:
    transfer_manager = SimpleNamespace(
        put_req_chunk={"r1": 1},
        request_payload={},
        config=SimpleNamespace(hf_config=SimpleNamespace(talker_config=SimpleNamespace(accept_hidden_layer=24))),
    )
    request = SimpleNamespace(
        external_req_id="r1",
        output_token_ids=[10, 11, 12],
        all_token_ids=[1, 2, 3, 10, 11, 12],
        prompt_token_ids=[1, 2, 3],
        additional_information=None,
    )
    output = thinker2talker_async_chunk(
        transfer_manager,
        {
            "hidden_states": {"layers": {0: torch.tensor([[101.0], [102.0]])}},
            "meta": {"decode_span": torch.tensor(True)},
            "ids": {"output": torch.tensor([11, 12])},
        },
        request,
        new_token_ids=[11, 12],
    )

    assert output is not None
    assert output.embed is not None
    assert torch.equal(output.embed.decode, torch.tensor([[101.0], [102.0]]))
    assert output.embed.decode_token_start == 1
    assert output.embed.decode_token_end == 3
    assert output.ids is not None and output.ids.output == [11, 12]
    assert output.hidden_states is None


def test_async_processor_keeps_singleton_tts_embeddings_from_wire_payload() -> None:
    transfer_manager = SimpleNamespace(
        put_req_chunk={"r1": 0},
        request_payload={},
        config=SimpleNamespace(
            hf_config=SimpleNamespace(
                im_start_token_id=151644,
                user_token_id=872,
                talker_config=SimpleNamespace(accept_hidden_layer=24),
                thinker_config=SimpleNamespace(audio_token_id=151675, image_token_id=151655, video_token_id=151656),
            )
        ),
    )
    request = SimpleNamespace(
        external_req_id="r1",
        output_token_ids=[],
        all_token_ids=[1, 2],
        prompt_token_ids=[1, 2],
        additional_information=None,
        resumable=False,
    )
    special = torch.tensor([[3.0]])
    output = thinker2talker_async_chunk(
        transfer_manager,
        {
            "hidden_states": {"layers": {0: torch.ones(2, 1), 24: torch.ones(2, 1)}},
            "embed": {"tts_bos": [special], "tts_eos": [special], "tts_pad": [special]},
        },
        request,
        is_finished=True,
    )

    assert output is not None and output.embed is not None
    assert torch.equal(output.embed.tts_bos, special)
    assert torch.equal(output.embed.tts_eos, special)
    assert torch.equal(output.embed.tts_pad, special)


@pytest.mark.parametrize("is_finished, expected", [(False, [11, 12]), (True, [11])])
def test_async_processor_trims_scheduler_truncated_span(is_finished, expected) -> None:
    transfer_manager = SimpleNamespace(
        put_req_chunk={"r1": 1},
        request_payload={},
        config=SimpleNamespace(hf_config=SimpleNamespace(talker_config=SimpleNamespace(accept_hidden_layer=24))),
    )
    request = SimpleNamespace(
        external_req_id="r1",
        output_token_ids=[10, 11, 12],
        additional_information=None,
        resumable=False,
    )
    output = thinker2talker_async_chunk(
        transfer_manager,
        {
            "hidden_states": {"layers": {0: torch.tensor([[11.0], [12.0], [13.0]])}},
            "meta": {"decode_span": True, "committed_token_count": 3},
            "ids": {"output": [11, 12, 13]},
        },
        request,
        is_finished=is_finished,
        new_token_ids=[11, 12],
    )

    assert output is not None
    assert output.ids.output == expected
    assert output.embed.decode.flatten().tolist() == expected
    assert output.embed.decode_token_start == 1
    assert output.embed.decode_token_end == 1 + len(expected)
    assert output.meta.committed_token_count == len(expected)


def test_async_processor_rejects_committed_count_mismatch() -> None:
    transfer_manager = SimpleNamespace(
        put_req_chunk={"r1": 1},
        request_payload={},
        config=SimpleNamespace(hf_config=SimpleNamespace(talker_config=SimpleNamespace(accept_hidden_layer=24))),
    )
    request = SimpleNamespace(
        external_req_id="r1",
        output_token_ids=[10, 11],
        all_token_ids=[1, 10, 11],
        prompt_token_ids=[1],
        additional_information=None,
    )

    with pytest.raises(RuntimeError, match="Committed token count mismatch"):
        thinker2talker_async_chunk(
            transfer_manager,
            {
                "hidden_states": {"layers": {0: torch.ones(2, 1)}},
                "meta": {"decode_span": True, "committed_token_count": 3},
                "ids": {"output": torch.tensor([10, 11])},
            },
            request,
        )


def test_async_processor_flushes_held_prefill_with_first_decode_span() -> None:
    transfer_manager = SimpleNamespace(
        put_req_chunk={"r1": 0},
        request_payload={
            "r1": {
                "embed": {
                    "prefill": torch.tensor([[1.0], [2.0]]),
                    "tts_bos": torch.tensor([[5.0]]),
                    "tts_eos": torch.tensor([[6.0]]),
                    "tts_pad": torch.tensor([[7.0]]),
                },
                "hidden_states": {"output": torch.tensor([[3.0], [4.0]])},
            }
        },
        config=SimpleNamespace(hf_config=SimpleNamespace(talker_config=SimpleNamespace(accept_hidden_layer=24))),
    )
    request = SimpleNamespace(
        external_req_id="r1",
        output_token_ids=[10, 11, 12],
        all_token_ids=[1, 2, 10, 11, 12],
        prompt_token_ids=[1, 2],
        additional_information=None,
    )

    output = thinker2talker_async_chunk(
        transfer_manager,
        {
            "hidden_states": {"layers": {0: torch.tensor([[10.0], [11.0], [12.0]])}},
            "meta": {"decode_span": True, "committed_token_count": 3},
            "ids": {"output": [10, 11, 12]},
        },
        request,
        new_token_ids=[11, 12],
    )

    assert output is not None
    assert output.embed is not None and output.embed.prefill is not None
    assert torch.equal(output.embed.prefill, torch.tensor([[1.0], [2.0], [10.0]]))
    assert torch.equal(output.embed.decode, torch.tensor([[11.0], [12.0]]))
    assert torch.equal(output.embed.tts_bos, torch.tensor([[5.0]]))
    assert torch.equal(output.embed.tts_eos, torch.tensor([[6.0]]))
    assert torch.equal(output.embed.tts_pad, torch.tensor([[7.0]]))
    assert output.hidden_states is not None
    assert torch.equal(output.hidden_states.output, torch.tensor([[3.0], [4.0]]))
    assert output.ids is not None
    assert output.ids.all == [1, 2, 10]
    assert output.ids.prompt == [1, 2]
    assert output.meta is not None and output.meta.decode_span is True
    assert transfer_manager.request_payload == {}


def test_async_decode_span_metadata_survives_payload_wire_round_trip() -> None:
    decode = torch.tensor([[1.0], [2.0]])
    payload = {
        "meta": {"decode_span": True, "committed_token_count": 2},
        "ids": {"output": [11, 12]},
        "embed": {"decode": decode},
    }

    wire = serialize_payload(payload)
    assert wire is not None
    restored = deserialize_payload(wire)
    assert restored["meta"]["decode_span"] is True
    assert restored["meta"]["committed_token_count"] == 2
    assert restored["ids"]["output"] == [11, 12]
    assert torch.equal(restored["embed"]["decode"], decode)


@pytest.mark.parametrize(
    ("samples", "expected"),
    [([11, -1, -1], [10]), ([11, 12, -1], [10, 11]), ([11, 12, 13], [10, 11, 12]), ([-1, -1, -1], [])],
)
def test_async_speculative_reuses_only_consumed_forward_rows(samples, expected):
    mm = _prepare(consumed=True, samples=(samples,), embed=lambda ids: pytest.fail("extra embedding lookup"))
    result = finalize_speculative_payload(mm)[STAGING_KEY][0]
    assert result["ids"]["output"] == expected
    assert result["hidden_states.layer_0"].flatten().tolist() == [x * 10 for x in expected]
    assert result["meta"]["consumed_input_span"] is True


def test_consumed_batch_mask_keeps_ragged_request_boundaries_and_skips_unowned_rows():
    # Prompt, zero/partial/full accepted drafts, discarded decode, ordinary
    # decode, and an empty sample share one batch with different query widths.
    mm = _prepare(
        consumed=True,
        samples=(
            (3, -1, -1, -1),
            (21, -1, -1, -1),
            (31, 32, 33, -1),
            (41, 42, 100, -1),
            (51, -1, -1, -1),
            (61, 62, 63, 64),
            (-1, 100, 101, -1),
        ),
        inputs=(1, 2, 3, 4, 5, 20, 21, 22, 23, 30, 31, 40, 41, 42, 50, 60, 61, 62, 63, 70, 71),
        req_ids=("prompt", "zero", "discard", "partial", "ordinary", "full", "empty"),
        speculative_ids={"zero", "discard", "partial", "full", "empty"},
        invalid=(2,),
        offsets=(0, 5, 9, 11, 14, 15, 19, 21),
        embed=lambda ids: pytest.fail("extra embedding lookup"),
    )
    stage = mm[STAGING_KEY]
    assert stage["indices"] == (1, 3, 5, 6)
    assert stage["offsets"] == (0, 4, 7, 11, 13)
    assert stage["ids"].tolist() == [20, -1, -1, -1, 40, 41, -1, 60, 61, 62, 63, -1, -1]
    result = finalize_speculative_payload(mm)[STAGING_KEY]
    for idx, expected in ((1, [20]), (3, [40, 41]), (5, [60, 61, 62, 63]), (6, [])):
        assert result[idx]["ids"]["output"] == expected
        assert result[idx]["hidden_states.layer_0"].shape == (len(expected), 1)
        assert result[idx]["hidden_states.layer_0"].flatten().tolist() == [x * 10 for x in expected]
        assert result[idx]["meta"] == {
            "decode_span": True,
            "consumed_input_span": True,
            "committed_token_count": len(expected),
        }
    assert all(result[idx] is None for idx in (0, 2, 4))


@pytest.mark.parametrize("consumed", [False, True])
def test_mixed_partial_prefill_and_decode_preserves_prompt_rows(consumed):
    calls = []

    def embed(ids):
        calls.append(ids)
        return ids.float().view(-1, 1)

    mm = _prepare(
        consumed=consumed,
        samples=((3, -1, -1), (21, 23, -1)),
        inputs=(1, 2, 20, 21, 22),
        positions=(0, 1, 4, 5, 6),
        prompt_lens=(8, 3),
        invalid=(0,),
        req_ids=("r1", "r2"),
        speculative_ids={"r2"},
        offsets=(0, 2, 5),
        embed=embed,
    )
    mm = finalize_speculative_payload(mm)
    prompt = {"hidden_states.layer_0": torch.tensor([[10.0], [20.0]]), "hidden_states.layer_24": torch.ones(2, 1)}
    assert apply_speculative_payload(dict(prompt), mm, 0).keys() == prompt.keys()
    assert torch.equal(
        apply_speculative_payload(dict(prompt), mm, 0)["hidden_states.layer_0"], prompt["hidden_states.layer_0"]
    )
    decode = apply_speculative_payload({}, mm, 1)
    assert decode["ids"]["output"] == ([20, 21] if consumed else [21, 23])
    assert len(calls) == (0 if consumed else 1)


def test_inline_mixed_batch_keeps_prefill_and_discard_rows_and_batches_zero_draft_decode():
    calls = []

    def embed(ids):
        calls.append(ids)
        return ids.float().view(-1, 1)

    mm = _prepare(
        samples=((9, -1, -1), (10, -1, -1), (11, -1, -1), (21, 23, -1), (30, -1, -1), (41, -1, -1)),
        inputs=(1, 2, 3, 4, 5, 10, 20, 21, 22, 30, 40),
        positions=(0, 1, 2, 3, 4, 3, 8, 9, 10, 10, 3),
        prompt_lens=(8, 5, 3, 3, 3, 3),
        req_ids=("partial", "final_prefill", "first_decode", "spec", "later_decode", "discard"),
        speculative_ids={"spec", "discard"},
        invalid=(0, 5),
        offsets=(0, 2, 5, 6, 9, 10, 11),
        embed=embed,
    )
    assert mm[STAGING_KEY]["indices"] == (1, 2, 3, 4)
    result = finalize_speculative_payload(mm)
    original = {"hidden_states.layer_0": torch.ones(3, 1), "hidden_states.layer_24": torch.ones(3, 1)}
    for idx in (0, 1, 5):
        preserved = apply_speculative_payload(dict(original), result, idx)
        assert preserved.keys() == original.keys()
        assert preserved["hidden_states.layer_0"] is original["hidden_states.layer_0"]
    for idx, expected in ((2, [10, 11]), (3, [21, 23]), (4, [30])):
        payload = apply_speculative_payload(dict(original), result, idx)
        assert payload["ids"]["output"] == expected
        assert payload["hidden_states.layer_0"].flatten().tolist() == expected
        assert "hidden_states.layer_24" not in payload
    assert len(calls) == 1


@pytest.mark.parametrize("consumed", [False, True])
@pytest.mark.parametrize("draft_steps", [(False, True, True), (True, False, True), (False, False, False)])
def test_first_decode_without_drafts_then_speculation_delivers_each_input_once(consumed, draft_steps):
    """Use one connector across prefill, ordinary decode and verifier steps."""
    manager = SimpleNamespace(
        put_req_chunk={"r1": 0},
        request_payload={},
        config=SimpleNamespace(
            hf_config=SimpleNamespace(
                im_start_token_id=151644,
                user_token_id=872,
                talker_config=SimpleNamespace(accept_hidden_layer=24),
                thinker_config=SimpleNamespace(audio_token_id=151675, image_token_id=151655, video_token_id=151656),
            )
        ),
    )
    spoken = []
    history = [10]
    steps: list[tuple[tuple[int, ...], list[int], int, bool, list[int]]] = [((1, 2, 3), [10], 0, False, list(history))]
    for drafts in draft_steps:
        position = 3 + len(history) - 1
        inputs = tuple(range(history[-1], history[-1] + (3 if drafts else 1)))
        delta = list(range(history[-1] + 1, history[-1] + (4 if drafts else 2)))
        history.extend(delta)
        steps.append((inputs, delta, position, drafts, list(history)))
    for inputs, delta, position, drafts, history in steps:
        samples = tuple(delta + [-1] * (3 - len(delta)))
        mm = _prepare(
            consumed=consumed,
            inputs=inputs,
            samples=(samples,),
            positions=tuple(range(position, position + len(inputs))),
            offsets=(0, len(inputs)),
            speculative_ids={"r1"} if drafts else set(),
            embed=lambda ids: ids.float().view(-1, 1) * 10,
        )
        assert (STAGING_KEY in mm) == (drafts or not consumed)
        mm = finalize_speculative_payload(mm)
        if STAGING_KEY in mm:
            raw = {
                "hidden_states.layer_0": mm["hidden_states"]["layers"][0],
                "hidden_states.layer_24": mm["hidden_states"]["layers"][24],
            }
            payload = apply_speculative_payload(raw, mm, 0)
            # Exercise the actual flat-key transport conversion too.
            payload = deserialize_payload(serialize_payload(payload))
            if position == 0:
                assert mm[STAGING_KEY][0] is None
                assert "meta" not in payload
        else:
            payload = mm
        request = SimpleNamespace(
            external_req_id="r1",
            prompt_token_ids=[1, 2, 3],
            output_token_ids=history,
            all_token_ids=[1, 2, 3] + history,
            additional_information=None,
            resumable=False,
            num_computed_tokens=position + len(inputs),
        )
        chunk = thinker2talker_async_chunk(manager, payload, request, new_token_ids=delta)
        if chunk is None:
            assert not drafts and position == 0
            continue
        manager.put_req_chunk["r1"] += 1
        if chunk.embed.prefill is not None:
            spoken.extend(chunk.embed.prefill[3:].flatten().tolist())
        if chunk.embed.decode is not None:
            spoken.extend(chunk.embed.decode.flatten().tolist())
        expected = history[:-1] if consumed else history
        assert spoken == [token * 10 for token in expected]
    assert manager.put_req_chunk["r1"] == 3


def test_inline_zero_draft_handoff_accumulates_full_payload_without_missing_or_duplicate_rows():
    runner = object.__new__(OmniRunnerDataPlane)
    runner._pending_full_payload_send = {}
    runner._full_payload_replace_keys_cached = frozenset()
    request = SimpleNamespace(prompt_token_ids=[1, 2, 3], output_token_ids=[], additional_information=None)
    for inputs, samples, position, drafts in (
        ((1, 2, 3), (10, -1, -1), 0, False),
        ((10,), (11, -1, -1), 3, False),
        ((11, 12, 13), (12, 13, 14), 4, True),
        ((14,), (15, -1, -1), 7, False),
        ((15, 16, 17), (16, 99, -1), 8, True),
    ):
        mm = finalize_speculative_payload(
            _prepare(
                inputs=inputs,
                samples=(samples,),
                positions=tuple(range(position, position + len(inputs))),
                offsets=(0, len(inputs)),
                speculative_ids={"r1"} if drafts else set(),
            )
        )
        raw = {
            "hidden_states.layer_0": torch.tensor(inputs).float().view(-1, 1),
            "hidden_states.layer_24": torch.ones(len(inputs), 1),
        }
        payload = flatten_payload(apply_speculative_payload(raw, mm, 0))
        request.output_token_ids.extend(x for x in samples if x >= 0)
        request.all_token_ids = request.prompt_token_ids + request.output_token_ids
        runner.accumulate_full_payload_output("r1", payload, request)
    combined, _ = runner._materialize_full_payload_entry(runner._pending_full_payload_send["r1"])
    assert combined["hidden_states.layer_0"].flatten().tolist() == request.all_token_ids
    manager = SimpleNamespace(
        config=SimpleNamespace(hf_config=SimpleNamespace(talker_config=SimpleNamespace(accept_hidden_layer=24)))
    )
    result = thinker2talker_full_payload(manager, combined, request)
    assert result["embed"]["prefill"].flatten().tolist() == request.all_token_ids[:-1]


@pytest.mark.parametrize(
    ("samples", "stop_ids", "max_tokens", "max_model_len"),
    [
        pytest.param((11, 99, 13), [], 20, 100, id="eos-middle"),
        pytest.param((99, 12, 13), [], 20, 100, id="eos-first"),
        pytest.param((11, 12, 99), [], 20, 100, id="eos-last"),
        pytest.param((11, 12, 13), [12], 20, 100, id="stop-token-middle"),
        pytest.param((11, 12, 13), [], 3, 100, id="max-tokens-middle"),
        pytest.param((11, 12, 13), [], 20, 6, id="max-model-len-middle"),
    ],
)
def test_full_payload_flush_honors_scheduler_stop_boundary(samples, stop_ids, max_tokens, max_model_len):
    """Verifier acceptance can extend past the scheduler's terminal token."""
    params = SamplingParams(max_tokens=max_tokens, stop_token_ids=stop_ids)
    params.update_from_generation_config({}, eos_token_id=99)
    scheduled_request = Request("r1", [1, 2, 3], params, pooling_params=None)
    scheduled_request.append_output_token_ids(10)
    scheduled_request.num_computed_tokens = 6
    scheduler = object.__new__(Scheduler)
    scheduler.max_model_len = max_model_len
    _, stopped = scheduler._update_request_with_output(scheduled_request, list(samples))
    assert stopped
    final_ids = list(scheduled_request.output_token_ids)

    runner = object.__new__(OmniRunnerDataPlane)
    runner._pending_full_payload_send = {}
    runner._full_payload_replace_keys_cached = frozenset()
    runner._async_chunk = False
    # The V2 transport owns accepted token history instead of V1's cached
    # placeholder list. The scheduler's terminal span is authoritative here.
    request = _NativeRequestState(
        request_id="r1",
        external_req_id="r1",
        prompt_token_ids=[1, 2, 3],
        sampling_params=params,
        num_computed_tokens=6,
        output_token_ids=final_ids,
    )
    runner._native_requests = {"r1": request}
    for inputs, step_samples, position in (((1, 2, 3), (10, -1, -1), 0), ((10, 11, 12), samples, 3)):
        mm = finalize_speculative_payload(
            _prepare(
                inputs=inputs,
                samples=(step_samples,),
                positions=tuple(range(position, position + len(inputs))),
                offsets=(0, len(inputs)),
                speculative_ids={"r1"} if position else set(),
            )
        )
        raw = {
            "hidden_states.layer_0": torch.tensor(inputs).float().view(-1, 1),
            "hidden_states.layer_24": torch.ones(len(inputs), 1),
        }
        runner.accumulate_full_payload_output("r1", flatten_payload(apply_speculative_payload(raw, mm, 0)), request)

    manager = SimpleNamespace(
        config=SimpleNamespace(hf_config=SimpleNamespace(talker_config=SimpleNamespace(accept_hidden_layer=24)))
    )
    entries, finished = runner._build_chunk_entries(
        req_ids=[], inter_stage_outputs=None, sampled_token_ids=None, terminal_req_ids={"r1"}
    )
    assert finished == ["r1"]
    [(terminal_request, payload)] = entries
    result = thinker2talker_full_payload(manager, payload, terminal_request)
    assert result["ids"]["all"] == [1, 2, 3, *final_ids]
    assert result["embed"]["prefill"].flatten().tolist() == [1, 2, 3, *final_ids][:-1]
    assert request.output_token_ids is final_ids
    assert request.output_token_ids == list(scheduled_request.output_token_ids)
    assert not runner._pending_full_payload_send


@pytest.mark.parametrize(
    ("finished", "history", "delta", "captured", "expected"),
    [
        (False, [10, 11, 12, 99], [12, 99], [11, 12], [11, 12]),
        (True, [10, 11, 12, 99], [12, 99], [11, 12, 99], [11, 12]),
        (True, [10, 11, 12], [12], [11, 12, 13], [11]),
    ],
)
def test_consumed_input_processor_tracks_scheduler_clipping(finished, history, delta, captured, expected):
    manager = SimpleNamespace(
        put_req_chunk={"r1": 1},
        request_payload={},
        config=SimpleNamespace(hf_config=SimpleNamespace(talker_config=SimpleNamespace(accept_hidden_layer=24))),
    )
    request = SimpleNamespace(
        external_req_id="r1",
        output_token_ids=history,
        prompt_token_ids=[1],
        all_token_ids=[1] + history,
        additional_information=None,
        resumable=False,
    )
    payload = {
        "hidden_states": {"layers": {0: torch.tensor(captured).float().view(-1, 1)}},
        "meta": {"decode_span": True, "consumed_input_span": True, "committed_token_count": len(captured)},
        "ids": {"output": captured},
    }
    output = thinker2talker_async_chunk(manager, payload, request, is_finished=finished, new_token_ids=delta)
    assert output.ids.output == expected
    assert output.embed.decode.flatten().tolist() == expected
    assert output.embed.decode_token_start == 1
    assert output.embed.decode_token_end == 1 + len(expected)


def test_consumed_input_first_span_completes_held_prefill_without_reembedding():
    manager = SimpleNamespace(
        put_req_chunk={"r1": 0},
        request_payload={
            "r1": {"embed": {"prefill": torch.tensor([[1.0], [2.0]])}, "hidden_states": {"output": torch.ones(2, 1)}}
        },
        config=SimpleNamespace(hf_config=SimpleNamespace(talker_config=SimpleNamespace(accept_hidden_layer=24))),
    )
    request = SimpleNamespace(
        external_req_id="r1",
        output_token_ids=[10, 11, 12],
        prompt_token_ids=[1, 2],
        additional_information=None,
        resumable=False,
    )
    payload = {
        "hidden_states": {"layers": {0: torch.tensor([[100.0], [110.0]])}},
        "meta": {"decode_span": True, "consumed_input_span": True, "committed_token_count": 2},
        "ids": {"output": [10, 11]},
    }
    restored = deserialize_payload(serialize_payload(payload))
    output = thinker2talker_async_chunk(manager, restored, request, new_token_ids=[11, 12])
    assert output.embed.prefill.flatten().tolist() == [1.0, 2.0, 100.0]
    assert output.embed.decode.flatten().tolist() == [110.0]
    assert output.ids.all == [1, 2, 10]


def test_consumed_input_span_rejects_wrong_token_identity():
    manager = SimpleNamespace(
        put_req_chunk={"r1": 1},
        request_payload={},
        config=SimpleNamespace(hf_config=SimpleNamespace(talker_config=SimpleNamespace(accept_hidden_layer=24))),
    )
    request = SimpleNamespace(
        external_req_id="r1",
        output_token_ids=[10, 11, 12],
        prompt_token_ids=[1],
        additional_information=None,
        resumable=False,
    )
    payload = {
        "hidden_states": {"layers": {0: torch.tensor([[99.0]])}},
        "meta": {"decode_span": True, "consumed_input_span": True, "committed_token_count": 1},
        "ids": {"output": [99]},
    }
    with pytest.raises(RuntimeError, match="does not match scheduler"):
        thinker2talker_async_chunk(manager, payload, request, new_token_ids=[12])
