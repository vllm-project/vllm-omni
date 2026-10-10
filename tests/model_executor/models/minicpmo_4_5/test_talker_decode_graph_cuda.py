# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA: the Talker's fused decode output and graph-replayed seeded codec draw are bit-identical to eager."""

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]

_VOCAB = 6562
_EOS = _VOCAB - 1


def _talker():
    from vllm.config import VllmConfig

    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
        MiniCPMO45OmniTTSForConditionalGeneration,
    )

    talker = object.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    torch.nn.Module.__init__(talker)
    talker.vllm_config = VllmConfig()
    talker._codec_eos_id = _EOS
    talker._request_condition_states = {}
    talker._request_audio_states = {}
    return talker


def _setup(graph: bool, capacity: int, requests: list[tuple[str, int, list[int], dict]]):
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.states import RequestState

    device = torch.device("cuda")
    reqs = RequestState(capacity, 512, 64, 0, _VOCAB, device)
    talker = _talker()
    base = Sampler(talker.vllm_config, capacity, _VOCAB, device, reqs)
    # The live installer (#8576 window state in sampler.logits_processors).
    sampler, _ = talker.mrv2_custom_sampler(base)
    assert talker._mrv2_seeded_codec_sampler.base_sampler is base
    assert base.penalties_state is talker._mrv2_penalties
    if not graph:
        # Exercise the supported eager fallback against the default graph path.
        talker._mrv2_seeded_codec_sampler.decode_graphs = None
    # Worker readiness captures before any request arrives.
    talker.capture_auxiliary_graphs()
    slots = [_admit(talker, sampler, reqs, *request) for request in requests]
    return talker, sampler, reqs, slots


def _admit(talker, sampler, reqs, req_id, prompt_len, token_ids, params, condition_seq=0):
    """Runner admission of one (re)scheduled Talker condition, with its staged writes."""
    from vllm import SamplingParams

    reqs.remove_request(req_id)
    reqs.add_request(req_id, prompt_len, token_ids, len(token_ids), 256)
    # A live request carries its condition sequence (RNG rewind reads it).
    talker._request_condition_states[req_id] = {"condition_seq": condition_seq}
    slot = reqs.req_id_to_index[req_id]
    sampler.add_request(slot, SamplingParams(**params))
    # Each staged write rotates vLLM's UvaBackedTensor buffers.
    reqs.apply_staged_writes()
    sampler.apply_staged_writes()
    return slot


def _batch(reqs, slots, step):
    device = torch.device("cuda")
    rows = len(slots)
    prefill = np.array([int(reqs.prefill_len.np[s]) for s in slots])
    idx = torch.tensor(slots, dtype=torch.int32, device=device)
    seq_lens = torch.tensor(prefill + step + 1, dtype=torch.int32, device=device)
    return SimpleNamespace(
        num_reqs=rows,
        idx_mapping_np=np.array(slots),
        idx_mapping=idx,
        expanded_idx_mapping=idx,
        expanded_local_pos=torch.zeros(rows, dtype=torch.int32, device=device),
        positions=seq_lens.long() - 1,
        logits_indices=torch.arange(rows, device=device),
        input_ids=torch.zeros(rows, dtype=torch.int32, device=device),
        cu_num_logits_np=np.arange(rows + 1),
        cu_num_logits=torch.arange(rows + 1, dtype=torch.int32, device=device),
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens.cpu(),
        num_computed_prefill_tokens_np=prefill,
        num_scheduled_tokens=np.ones(rows, dtype=np.int32),
        prefill_len_np=prefill,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("rows", [1, 2, 3, 5])
@pytest.mark.parametrize("min_tokens", [0, 50])
@torch.inference_mode()
def test_seeded_codec_graph_matches_eager_draws_and_generators(rows, min_tokens):
    capacity = 8
    rng = random.Random(rows)
    # Deploy stage-1 defaults; native duplex overrides min_tokens to 0. With
    # min_tokens > 0 vLLM's LogitBiasState masks codec EOS (a stage stop id)
    # until the request reaches min_tokens; the graph replays that stage.
    params = dict(
        temperature=0.8,
        top_k=25,
        top_p=0.85,
        repetition_penalty=1.05,
        min_tokens=min_tokens,
        stop_token_ids=[_EOS],
        max_tokens=256,
        detokenize=False,
    )
    requests = []
    for i in range(rows):
        prompt = [rng.randrange(_VOCAB - 1) for _ in range(8 + i)]
        # Repeated codes make the 16-frame window penalty bite.
        history = [rng.choice([3, 5, 7]) for _ in range(rng.randrange(0, 20))]
        requests.append((f"r{i}", len(prompt), prompt + history, dict(params, seed=40 + i)))
    eager = _setup(False, capacity, requests)
    graph = _setup(True, capacity, requests)
    graphs = graph[0]._mrv2_seeded_codec_sampler.decode_graphs
    assert graphs is not None and sorted(graphs.graphs) == list(range(1, capacity + 1))
    assert eager[0]._mrv2_seeded_codec_sampler.decode_graphs is None
    replays = []
    original = graphs.try_sample

    def spy(*args, **kwargs):
        result = original(*args, **kwargs)
        replays.append(result is not None)
        return result

    graphs.try_sample = spy
    gen = torch.Generator(device="cuda").manual_seed(rows)
    onset = 0
    # The updated condition (prompt 30, 3 history codes) crosses min_tokens=50
    # 47 steps after the update.
    for step in range(84):
        if step == 30:
            # A condition update re-admits r0 with new parameters and prompt:
            # more staged writes, new UVA buffers, possibly another slot.
            update = ("r0", 30, list(range(30)) + [5, 5, 5], dict(params, seed=40, temperature=0.7, top_k=20))
            for setup in (eager, graph):
                talker, sampler, reqs, slots = setup
                slots[0] = _admit(talker, sampler, reqs, *update, condition_seq=1)
            onset = step
        logits = torch.randn((rows, _VOCAB), generator=gen, device="cuda") * 3
        # Codec EOS is the likeliest code: masking and top-k/top-p decide the onset.
        logits[:, _EOS] += 6.0
        forced = torch.rand(rows, generator=gen, device="cuda") < 0.05
        # Turn-start EOS masking for the first frames of each condition.
        mask = torch.full((rows,), step - onset < 6, dtype=torch.bool, device="cuda") & ~forced
        outputs = []
        for talker, sampler, reqs, slots in (eager, graph):
            talker._mrv2_output_infos = [{"native_duplex": True} for _ in range(rows)]
            talker._mrv2_forced_eos = forced.clone()
            talker._mrv2_mask_eos = mask.clone()
            out = sampler(logits.clone(), _batch(reqs, slots, step - onset))
            outputs.append(out)
            # Feed the sample back into the device history the penalty reads.
            tokens = out.sampled_token_ids.view(-1).tolist()
            for slot, token in zip(slots, tokens):
                total = int(reqs.total_len.gpu[slot])
                reqs.all_token_ids.stage_write(slot, total, [token])
                reqs.total_len.stage_write_elem(slot, total + 1)
            reqs.apply_staged_writes()
        torch.testing.assert_close(outputs[1].sampled_token_ids, outputs[0].sampled_token_ids, rtol=0, atol=0)
        torch.testing.assert_close(outputs[1].num_sampled, outputs[0].num_sampled, rtol=0, atol=0)
        for i in range(rows):
            got = graph[0]._mrv2_seeded_codec_sampler._generators[f"r{i}"]
            want = eager[0]._mrv2_seeded_codec_sampler._generators[f"r{i}"]
            torch.testing.assert_close(got.get_state(), want.get_state(), rtol=0, atol=0)
    # Every step replays, min_tokens included: the graph runs LogitBiasState.
    assert replays and all(replays)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "unsupported",
    [
        dict(min_p=0.1),  # min-p is not replayed
        dict(frequency_penalty=0.3),  # the window state's stock base would act
        dict(presence_penalty=0.3),
        dict(logprobs=1),
        dict(temperature=0.0),  # greedy rows keep the eager argmax
        dict(top_k=-1, top_p=1.0),  # the captured draw assumes active top-k/top-p
    ],
)
@torch.inference_mode()
def test_seeded_codec_graph_declines_unsupported_rows(unsupported):
    params = dict(temperature=0.8, top_k=25, top_p=0.85, repetition_penalty=1.05, max_tokens=256)
    requests = [
        ("a", 4, [1, 2, 3, 4], dict(params, seed=1)),
        ("b", 4, [1, 2, 3, 4], dict(params, seed=2, **unsupported)),
    ]
    talker, sampler, reqs, slots = _setup(True, 4, requests)
    graphs = talker._mrv2_seeded_codec_sampler.decode_graphs
    assert graphs.slot_ok[slots[0]] and not graphs.slot_ok[slots[1]]
    talker._mrv2_output_infos = [{"native_duplex": True}, {"native_duplex": True}]
    logits = torch.randn((2, _VOCAB), device="cuda")
    batch = _batch(reqs, slots, 0)
    false = torch.zeros(2, dtype=torch.bool, device="cuda")
    assert graphs.try_sample(logits, batch, false, false) is None
    one = _batch(reqs, slots[:1], 0)
    talker._mrv2_output_infos = [{"native_duplex": True}]
    assert graphs.try_sample(logits[:1].contiguous(), one, false[:1], false[:1]) is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_seeded_codec_graph_declines_processor_it_does_not_replay(mocker):
    from vllm.v1.worker.gpu.sample.logits_processor import LogitsProcessor

    params = dict(temperature=0.8, top_k=25, top_p=0.85, repetition_penalty=1.05, max_tokens=256, seed=1)
    talker, sampler, reqs, slots = _setup(True, 2, [("a", 4, [1, 2, 3, 4], params)])
    graphs = talker._mrv2_seeded_codec_sampler.decode_graphs
    talker._mrv2_output_infos = [{"native_duplex": True}]
    logits = torch.randn((1, _VOCAB), device="cuda")
    false = torch.zeros(1, dtype=torch.bool, device="cuda")
    assert graphs.try_sample(logits, _batch(reqs, slots, 0), false, false) is not None
    base = talker._mrv2_seeded_codec_sampler.base_sampler
    base.logits_processors.append(mocker.Mock(spec=LogitsProcessor))
    assert graphs.try_sample(logits, _batch(reqs, slots, 0), false, false) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("native", [False, True])
@torch.inference_mode()
def test_fused_decode_output_matches_eager_path(native):
    from vllm_omni.model_executor.models.minicpmo_4_5 import minicpmo_4_5_omni_tts as tts

    rng = random.Random(int(native))
    for _ in range(50):
        slots = 16
        num_reqs = rng.randint(1, slots)
        num_tokens = num_reqs + rng.choice([0, 1, 3])
        slot_ids = rng.sample(range(slots), num_reqs)
        prompt = torch.tensor([rng.choice([2, 10, 400, 4090, 4100]) for _ in range(slots)], dtype=torch.int32)
        steps = [rng.choice([-1, 0, 1, 24, 25, 29, 30, 50, 2047, 2048, 3000]) for _ in range(num_reqs)]
        seq_lens = torch.tensor([int(prompt[s]) + st for s, st in zip(slot_ids, steps)], dtype=torch.int32)
        ids = torch.tensor([rng.choice([0, 17, _EOS - 1, _EOS]) for _ in range(num_tokens)], dtype=torch.int32)
        empty = torch.tensor([rng.random() < 0.2 for _ in range(slots)], dtype=torch.bool)
        controls = torch.tensor(
            [[rng.choice([-1, 0, 10, 50]), rng.choice([-1, 0, 30]), rng.choice([-1, 0, 1])] for _ in range(slots)],
            dtype=torch.long,
        )
        outputs = []
        for device in ("cpu", "cuda"):
            talker = object.__new__(tts.MiniCPMO45OmniTTSForConditionalGeneration)
            torch.nn.Module.__init__(talker)
            talker._codec_eos_id = _EOS
            talker._tts_config = SimpleNamespace(max_position_embeddings=4096)
            talker._mrv2_decode_rows_logged = True
            talker._mrv2_codec_controls = controls.to(device)
            meta = {key: [torch.tensor(0)] for key in tts._DUPLEX_OUTPUT_META_KEYS}
            talker._mrv2_metadata_by_slot = {slot: meta for slot in range(slots)}
            talker._mrv2_empty_speech = empty.to(device)
            batch = SimpleNamespace(
                num_reqs=num_reqs,
                has_prefill=False,
                is_prefilling_np=np.zeros(num_reqs, dtype=bool),
                idx_mapping_np=np.array(slot_ids),
                idx_mapping=torch.tensor(slot_ids, dtype=torch.int32, device=device),
                input_ids=ids.to(device),
                seq_lens=seq_lens.to(device),
                query_start_loc_np=np.arange(num_reqs + 1),
                logits_indices=torch.arange(num_reqs, device=device),
            )
            out = talker.make_omni_output_mrv2(
                torch.zeros(num_tokens, 4, device=device),
                input_batch=batch,
                req_states=SimpleNamespace(prompt_len=SimpleNamespace(gpu=prompt.to(device))),
                model_intermediate_buffer=[{"native_duplex": native} for _ in range(num_reqs)],
            )
            mm = out.multimodal_outputs
            mask = talker._mrv2_mask_eos
            outputs.append(
                [
                    mm["codes"]["audio"].cpu(),
                    mm["meta"]["codec_frame_valid"].cpu(),
                    talker._mrv2_forced_eos.cpu(),
                    None if mask is None else mask.cpu(),
                ]
            )
        for got, want in zip(outputs[1], outputs[0]):
            if want is None:
                assert got is None
            else:
                assert got.dtype == want.dtype and torch.equal(got, want)
