# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.duplex_sampling import DuplexSamplingHelper
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.mrv2_sampling import (
    MiniCPMO45DuplexSampler,
    _v1_shaped_runner,
)

pytestmark = [pytest.mark.core_model]


@pytest.mark.cpu
def test_mrv2_rows_read_sampling_params_from_intermediate_buffers():
    helper = DuplexSamplingHelper()
    infos = {
        "a": {"duplex": {"data_plane": True}, "sampling_params": SamplingParams(temperature=0.2, top_k=7, top_p=0.5)},
        "b": {"duplex": {"data_plane": True}, "sampling_params": SamplingParams(temperature=0.8, top_k=15, top_p=0.9)},
    }
    runner = _v1_shaped_runner(SimpleNamespace(req_ids=["b", "a"]), infos)
    for request_id in infos:
        helper.refresh_active_request(runner, request_id)
    rows = helper.rows(runner)
    assert [(row.request_id, row.temperature, row.top_k, row.top_p) for row in rows] == [
        ("b", 0.8, 15, 0.9),
        ("a", 0.2, 7, 0.5),
    ]
    assert all(row.max_tokens == 16 for row in rows)


@pytest.mark.cpu
def test_mrv2_thinker_payload_uses_output_channel_contract():
    sampler = MiniCPMO45DuplexSampler(object(), SimpleNamespace())
    tokens = torch.tensor([[7]])
    standard = (SimpleNamespace(sampled_token_ids=tokens), torch.ones(1), torch.zeros(1))
    output = sampler.sample_step(None, None, None, None, lambda *_args: standard)
    payload = {"latent": torch.ones(1, 2)}
    assert output.sampler_output is standard[0]
    assert output.include_hidden_states is False
    assert output.finalize_multimodal(payload, [1]) is payload


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("partial_prefill", [True, False])
def test_mrv2_sampler_preserves_history_seed_counts_and_prefill_eligibility(mocker, partial_prefill):
    device = "cuda"
    params = SamplingParams(temperature=0.2, top_k=7, top_p=0.5, seed=13)
    model = SimpleNamespace(
        _mrv2_sampling_infos={
            "a": {"duplex": {"data_plane": True, "session_id": "session"}, "sampling_params": params}
        },
        prepare_duplex_sampling=mocker.Mock(),
        sample=mocker.Mock(
            return_value=SimpleNamespace(sampled_token_ids=torch.tensor([[9]], device=device), logprobs_tensors=None)
        ),
    )
    # Request a occupies slot 2; neither slot number nor padded tokens are
    # sampling row indices. The uncomputed tail must never enter its history.
    tokens = torch.tensor([[0] * 8, [0] * 8, [1, 2, 3, 4, 6, 8, 77, 88]], device=device)
    states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=tokens),
        prompt_len=SimpleNamespace(np=np.array([0, 0, 4])),
        prefill_len=SimpleNamespace(gpu=torch.tensor([0, 0, 7 if partial_prefill else 4], device=device)),
    )
    batch = SimpleNamespace(
        req_ids=["a"],
        num_reqs=1,
        num_draft_tokens=0,
        idx_mapping_np=np.array([2]),
        idx_mapping=torch.tensor([2], device=device),
        seq_lens=torch.tensor([6], device=device),
        cu_num_logits=torch.tensor([0, 1], device=device),
        is_prefilling_np=np.array([partial_prefill]),
        num_computed_prefill_tokens_np=np.array([5]),
        num_computed_tokens_np=np.array([5]),
        num_scheduled_tokens=np.array([1]),
        prefill_len_np=np.array([7 if partial_prefill else 4]),
    )
    base_output = SimpleNamespace(num_sampled=torch.tensor([0], device=device))
    base = mocker.Mock(return_value=base_output)
    base.req_states = states
    sampler = MiniCPMO45DuplexSampler(base, model)
    output = sampler(torch.zeros(1, 40, device=device), batch)
    if partial_prefill:
        assert output is base_output
        model.sample.assert_not_called()
        assert model.prepare_duplex_sampling.call_args.args[2] == ()
        assert sampler.generators == {}
    else:
        assert output.num_sampled.tolist() == [1]
        assert output.num_rejected.tolist() == [0]
        assert output.sampled_token_ids.tolist() == [[9]]
        md = model.sample.call_args.args[1]
        assert md.output_token_ids == [[6, 8]]
        assert md.generators[0].initial_seed() == 13
        assert md.top_k.tolist() == [7]
        generator = md.generators[0]
        sampler(torch.zeros(1, 40, device=device), batch)
        assert model.sample.call_args.args[1].generators[0] is generator
        sampler.forget_requests(["a"])
        assert sampler.generators == {}


@pytest.mark.cpu
def test_thinker_history_copies_only_new_ids_and_resets_at_condition_change():
    reads = []
    values = torch.arange(24).reshape(3, 8)

    class Ledger:
        def __getitem__(self, index):
            reads.append((index[0], index[1].start, index[1].stop))
            return values[index]

    states = SimpleNamespace(
        prompt_len=SimpleNamespace(np=np.array([0, 0, 2])),
        all_token_ids=SimpleNamespace(gpu=Ledger()),
    )
    model = SimpleNamespace(_mrv2_sampling_infos={})
    sampler = MiniCPMO45DuplexSampler(SimpleNamespace(req_states=states), model)
    row = SimpleNamespace(row_idx=0, request_id="a", seq=1)
    batch = SimpleNamespace(
        num_reqs=1,
        idx_mapping_np=np.array([2]),
        num_computed_tokens_np=np.array([3]),
        num_scheduled_tokens=np.array([1]),
    )
    infos = {"a": {"sampling_params": SamplingParams()}}
    first = sampler._metadata(batch, [row], infos, "cpu")
    first.output_token_ids[0].append(999)  # A consumer cannot mutate the cache.
    batch.num_computed_tokens_np[0] = 4
    second = sampler._metadata(batch, [row], infos, "cpu")
    assert second.output_token_ids == [[18, 19, 20]]
    assert reads == [(2, 2, 4), (2, 4, 5)]
    row.seq = 2
    sampler._metadata(batch, [row], infos, "cpu")
    assert reads[-1] == (2, 2, 5)
    batch.num_computed_tokens_np[0] = 2
    assert sampler._metadata(batch, [row], infos, "cpu").output_token_ids == [[18]]
    sampler.forget_requests(["a"])
    assert sampler._histories == {}


@pytest.mark.cpu
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_thinker_histories_and_rng_follow_requests_across_reorder_and_slot_reuse(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    request_ids = [f"request-{slot}" for slot in range(4)]
    tokens = torch.arange(32, device=device).reshape(4, 8)
    states = SimpleNamespace(
        prompt_len=SimpleNamespace(np=np.full(4, 2)),
        all_token_ids=SimpleNamespace(gpu=tokens),
    )
    infos = {req_id: {"sampling_params": SamplingParams(seed=slot)} for slot, req_id in enumerate(request_ids)}
    sampler = MiniCPMO45DuplexSampler(SimpleNamespace(req_states=states), SimpleNamespace(_mrv2_sampling_infos=infos))
    reference_rng = {
        req_id: torch.Generator(device=device).manual_seed(slot) for slot, req_id in enumerate(request_ids)
    }
    for step in range(4):
        if step == 2:
            sampler.forget_requests([request_ids[0]])
            assert request_ids[0] not in sampler.generators
            assert request_ids[0] not in sampler._histories
            request_ids[0] = "replacement"
            tokens[0].add_(100)
            infos["replacement"] = {"sampling_params": SamplingParams(seed=31)}
            reference_rng["replacement"] = torch.Generator(device=device).manual_seed(31)
        slots = np.roll(np.arange(4)[::-1], step)
        # Each request has a different accepted length; a new condition can
        # replace tokens without changing either the slot or the prompt length.
        lengths = [3 + (int(slot) + step) % 4 for slot in slots]
        if step == 3:
            tokens.add_(1000)
        rows = [
            SimpleNamespace(row_idx=row, request_id=request_ids[slot], seq=int(step == 3))
            for row, slot in enumerate(slots)
        ]
        batch = SimpleNamespace(
            num_reqs=4,
            idx_mapping_np=slots,
            num_computed_tokens_np=np.array(lengths) - 1,
            num_scheduled_tokens=np.ones(4, dtype=int),
        )
        metadata = sampler._metadata(batch, rows, infos, device)
        for row, slot in enumerate(slots):
            assert metadata.output_token_ids[row] == tokens[slot, 2 : lengths[row]].tolist()
            torch.testing.assert_close(
                torch.rand(4, device=device, generator=metadata.generators[row]),
                torch.rand(4, device=device, generator=reference_rng[request_ids[slot]]),
            )
    sampler.forget_requests(request_ids)
    assert sampler._histories == sampler.generators == infos == {}


@pytest.mark.cpu
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_thinker_decode_reuses_deferred_samples_without_device_ledger_reads(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")

    class NoLedgerReads:
        def __getitem__(self, index):
            raise AssertionError("steady-state sampling must not read back the device token ledger")

    states = SimpleNamespace(
        prompt_len=SimpleNamespace(np=np.array([2, 2])),
        all_token_ids=SimpleNamespace(gpu=NoLedgerReads()),
    )
    infos = {req_id: {"sampling_params": SamplingParams(seed=seed)} for seed, req_id in enumerate(("a", "b"))}
    sampler = MiniCPMO45DuplexSampler(SimpleNamespace(req_states=states), SimpleNamespace(_mrv2_sampling_infos=infos))
    expected: dict[str, list[int]] = {"a": [], "b": []}
    for step in range(8):
        slots = [0, 1] if step % 2 == 0 else [1, 0]
        rows = [SimpleNamespace(row_idx=i, request_id=("a", "b")[slot], seq=0) for i, slot in enumerate(slots)]
        batch = SimpleNamespace(
            num_reqs=2,
            idx_mapping_np=np.array(slots),
            num_computed_tokens_np=np.full(2, 1 + step),
            num_scheduled_tokens=np.ones(2, dtype=int),
        )
        metadata = sampler._metadata(batch, rows, infos, device)
        assert metadata.output_token_ids == [expected[row.request_id] for row in rows]
        sampled = [step * 10 + slot for slot in slots]
        sampler._defer_history(rows, torch.tensor(sampled, device=device))
        for row, token in zip(rows, sampled):
            expected[row.request_id].append(token)
    # Finishing a request before the pending copy is consumed cannot restore
    # its history or append that sample to another request reusing its slot.
    sampler.forget_requests(["a"])
    sampler._commit_history()
    assert "a" not in sampler._histories
    assert sampler._histories["b"][1] == expected["b"]


@pytest.mark.cpu
def test_thinker_reuses_policy_snapshot_without_another_device_copy(mocker):
    pending = SimpleNamespace(host=torch.tensor([[7, 3, 0], [9, 4, 1]]), event=mocker.Mock(), row_idxs=[0, 1])
    model = SimpleNamespace(_minicpmo45_duplex_pending_samples=pending)
    sampler = MiniCPMO45DuplexSampler(SimpleNamespace(), model)
    sampler._histories = {"a": ((0, 2, 0), []), "b": ((1, 2, 0), [])}
    rows = [SimpleNamespace(request_id=req_id) for req_id in ("a", "b")]
    # No device tensor is needed: the policy has already captured the result.
    sampler._defer_history(rows, None)
    sampler._commit_history()
    assert sampler._histories["a"][1] == [7]
    assert sampler._histories["b"][1] == [9]
    pending.event.synchronize.assert_called_once()


@pytest.mark.cpu
@pytest.mark.parametrize("all_policy_rows", [True, False])
def test_thinker_policy_reads_logits_in_place_when_every_row_is_duplex(mocker, all_policy_rows):
    params = SamplingParams(temperature=0.0)
    infos = {"a": {"duplex": {"data_plane": True, "session_id": "s"}, "sampling_params": params}}
    infos["b"] = dict(infos["a"]) if all_policy_rows else {"sampling_params": params}
    model = SimpleNamespace(_mrv2_sampling_infos=infos, prepare_duplex_sampling=mocker.Mock(), sample=mocker.Mock())
    model.sample.return_value = None  # the stock sampler takes over
    states = SimpleNamespace(prompt_len=SimpleNamespace(np=np.array([4, 4])))
    batch = SimpleNamespace(
        req_ids=["a", "b"],
        num_reqs=2,
        num_draft_tokens=0,
        idx_mapping_np=np.array([0, 1]),
        is_prefilling_np=np.array([False, False]),
        num_computed_tokens_np=np.array([3, 3]),
        num_scheduled_tokens=np.array([1, 1]),
    )
    base = mocker.Mock(return_value="stock")
    base.req_states = states
    sampler = MiniCPMO45DuplexSampler(base, model)
    logits = torch.randn(2, 8)
    assert sampler(logits, batch) == "stock"
    policy_logits = model.prepare_duplex_sampling.call_args.args[0]
    if all_policy_rows:
        assert policy_logits is logits
    else:
        assert policy_logits is not logits
        torch.testing.assert_close(policy_logits, logits[:1])
    assert base.call_args.args[0] is logits
