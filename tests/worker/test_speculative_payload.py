# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU contract tests for speculative payload token alignment."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.data_entry_keys import deserialize_payload, serialize_payload
from vllm_omni.model_executor.stage_input_processors.qwen3_omni import thinker2talker_async_chunk
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_payload_adapter_uses_committed_tokens_and_drops_dflash_mask_rows() -> None:
    """DFlash query/mask rows must not become Talker condition rows."""

    class _Model:
        def embed_input_ids(self, input_ids):
            return input_ids.to(dtype=torch.float32).view(-1, 1)

    runner = object.__new__(GPUARModelRunner)
    runner.model = _Model()
    runner.device = torch.device("cpu")
    payload = runner._build_omni_pooler_payload(
        rid="r1",
        idx=0,
        start=0,
        end=3,
        hidden_states_cpu=torch.ones(3, 2),
        req_hidden_states_cpu={"r1": torch.full((3, 2), 9.0)},
        combined_hidden_states=None,
        combined_multimodal_outputs=None,
        mm_cpu={
            "hidden_states.layer_0": torch.full((3, 2), 8.0),
            "hidden_states.layer_24": torch.full((3, 2), 7.0),
        },
        audio_sparse_output=False,
        sparse_mm_index={},
        hidden_seq_len=3,
        scheduled_seq_len=3,
        committed_token_ids=[101, 102],
    )

    assert torch.equal(payload["hidden_states.layer_0"], torch.tensor([[101.0], [102.0]]))
    assert "hidden_states.layer_24" not in payload
    assert "hidden" not in payload
    assert payload["meta"] == {"decode_span": True, "committed_token_count": 2}
    assert payload["ids"]["output"] == [101, 102]


@pytest.mark.parametrize(
    ("use_dflash", "sampled", "expected"),
    [
        (True, [10, 11], [10, 11]),
        (True, [], []),
        (False, [10, 11], None),
    ],
)
def test_speculative_token_selection_tracks_committed_rows(use_dflash, sampled, expected) -> None:
    runner = object.__new__(GPUARModelRunner)
    runner.speculative_config = SimpleNamespace(use_dflash=lambda: use_dflash)
    scheduler_output = SimpleNamespace(scheduled_spec_decode_tokens={"r1": [1, 2, 3]})

    assert runner._get_speculative_committed_token_ids("r1", 0, scheduler_output, [sampled]) == expected


@pytest.mark.parametrize("first_committed", [10, 11])
@pytest.mark.parametrize("prior_rows", [0, 3, 4])
def test_first_speculative_delta_seeds_prefill_sampled_token(first_committed, prior_rows) -> None:
    runner = object.__new__(GPUARModelRunner)
    runner.speculative_config = SimpleNamespace(use_dflash=lambda: True)
    runner.model_config = SimpleNamespace(async_chunk=False)
    runner._pending_full_payload_send = {"r1": ({}, {}, {"hidden_states.layer_0": prior_rows}, object())}
    runner.requests = {"r1": SimpleNamespace(prompt_token_ids=[1, 2, 3], output_token_ids=[10, 11])}
    scheduler_output = SimpleNamespace(scheduled_spec_decode_tokens={"r1": [4, 5, 6]})

    committed = [first_committed, 12]
    expected = [10, *committed] if prior_rows <= 3 else committed
    assert runner._get_speculative_committed_token_ids("r1", 0, scheduler_output, [committed]) == expected


def test_async_speculative_delta_seeds_only_the_first_prefill_sample() -> None:
    runner = object.__new__(GPUARModelRunner)
    runner.speculative_config = SimpleNamespace(use_dflash=lambda: True)
    runner.model_config = SimpleNamespace(async_chunk=True)
    runner.requests = {
        "r1": SimpleNamespace(prompt_token_ids=[1, 2, 3], output_token_ids=[10, 11, 12]),
    }
    scheduler_output = SimpleNamespace(scheduled_spec_decode_tokens={"r1": [4, 5]})

    assert runner._get_speculative_committed_token_ids("r1", 0, scheduler_output, [[11, 12]]) == [10, 11, 12]

    # A later verifier step may reject every draft and commit only one
    # replacement token. Its span length is one, but the request history is
    # already longer than the initial prefill sample, so it must not be seeded.
    runner.requests["r1"].output_token_ids = [10, 11, 12, 13]
    assert runner._get_speculative_committed_token_ids("r1", 0, scheduler_output, [[13]]) == [13]

    runner.requests["r1"].output_token_ids = [10, 10]
    assert runner._get_speculative_committed_token_ids("r1", 0, scheduler_output, [[10]]) == [10, 10]


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
    payload = {
        "meta": {"decode_span": True, "committed_token_count": 2},
        "ids": {"output": [11, 12]},
        "embed": {"decode": torch.tensor([[1.0], [2.0]])},
    }

    wire = serialize_payload(payload)
    assert wire is not None
    restored = deserialize_payload(wire)
    assert restored["meta"]["decode_span"] is True
    assert restored["meta"]["committed_token_count"] == 2
    assert restored["ids"]["output"] == [11, 12]
    assert torch.equal(restored["embed"]["decode"], payload["embed"]["decode"])


def _async_speculative_runner():
    runner = object.__new__(GPUARModelRunner)
    runner.requests = {rid: SimpleNamespace(output_token_ids=[-1]) for rid in ("r1", "r2")}
    runner.input_batch = SimpleNamespace(vocab_size=100)

    def embed_input_ids(input_ids):
        assert bool(((input_ids >= 0) & (input_ids < 100)).all())
        return input_ids.float().unsqueeze(-1)

    runner.model = SimpleNamespace(embed_input_ids=embed_input_ids)
    return runner


@pytest.mark.parametrize(
    ("samples", "expected"),
    [([11, -1, -1], [10]), ([11, 12, -1], [10, 11]), ([11, 12, 13], [10, 11, 12]), ([-1, -1, -1], [])],
)
def test_async_speculative_reuses_only_consumed_forward_rows(samples, expected):
    runner = _async_speculative_runner()
    runner.model.embed_input_ids = lambda **kwargs: pytest.fail("extra embedding lookup")
    scheduler = SimpleNamespace(scheduled_spec_decode_tokens={"r1": [11, 12]}, num_scheduled_tokens={"r1": 3})
    payloads = runner._prepare_async_speculative_payloads(
        scheduler,
        torch.tensor([samples]),
        ["r1"],
        [],
        input_token_ids=torch.tensor([10, 11, 12]),
        input_embeddings=torch.tensor([[100.0], [110.0], [120.0]]),
        query_start_loc_cpu=[0, 3],
    )
    resolved = runner._resolve_async_speculative_payloads(payloads, 100)["r1"]
    assert resolved[0] == expected
    assert resolved[1].flatten().tolist() == [float(x * 10) for x in expected]


def test_async_speculative_captured_rows_follow_batch_offsets_and_discard_mask():
    runner = _async_speculative_runner()
    scheduler = SimpleNamespace(scheduled_spec_decode_tokens={"r2": [21, 22]}, num_scheduled_tokens={"r1": 2, "r2": 3})
    args = dict(
        input_token_ids=torch.tensor([1, 2, 20, 21, 22]),
        input_embeddings=torch.arange(5.0).view(-1, 1),
        query_start_loc_cpu=[0, 2, 5],
    )
    payloads = runner._prepare_async_speculative_payloads(
        scheduler, torch.tensor([[3, -1, -1], [21, 23, -1]]), ["r1", "r2"], [], **args
    )
    resolved = runner._resolve_async_speculative_payloads(payloads, 100)
    assert set(resolved) == {"r2"}
    assert resolved["r2"][0] == [20, 21]
    assert resolved["r2"][1].flatten().tolist() == [2.0, 3.0]
    assert (
        runner._prepare_async_speculative_payloads(
            scheduler, torch.tensor([[3, -1, -1], [21, 23, -1]]), ["r1", "r2"], [1], **args
        )
        == {}
    )


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


def test_async_speculative_cpu_payload_does_not_run_model_embedding():
    runner = _async_speculative_runner()
    runner.device = torch.device("cpu")

    def unexpected_lookup(*args, **kwargs):
        pytest.fail("background builder must not execute model embedding")

    runner.model.embed_input_ids = unexpected_lookup
    prepared = torch.tensor([[11.0], [12.0]])
    output = runner._build_omni_pooler_payload(
        rid="r1",
        idx=0,
        start=0,
        end=3,
        hidden_states_cpu=None,
        req_hidden_states_cpu=None,
        combined_hidden_states=None,
        combined_multimodal_outputs=None,
        mm_cpu={"hidden_states.layer_24": torch.ones(3, 1)},
        audio_sparse_output=False,
        sparse_mm_index={},
        hidden_seq_len=3,
        scheduled_seq_len=3,
        committed_token_ids=[11, 12],
        committed_token_embeddings=prepared,
    )
    assert output["ids"]["output"] == [11, 12]
    assert torch.equal(output["hidden_states.layer_0"], prepared)
    assert "hidden_states.layer_24" not in output


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
