# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MRv2 request lifecycle and real encoder dispatch with weight-free Hunyuan."""

from copy import copy
from dataclasses import replace

import numpy as np
import pytest
import torch
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.states import RequestState

from tests.model_executor.models import test_hunyuan_request_rng as hunyuan_rng
from vllm_omni.model_executor.mm_request_rng import MM_RNG_ITEM_KEY
from vllm_omni.worker_v2 import omni_model_runner, staged_writes

pytestmark = [pytest.mark.core_model]
runner = hunyuan_rng.runner
make_request = hunyuan_rng.make_request


@pytest.fixture
def v2(runner, mocker):
    model = runner.model
    model._mm_rng_provider = None
    model._timestep_token_id = -1
    model._timestep_encode = lambda value: torch.zeros(3, device=value.device, dtype=value.dtype)
    result = object.__new__(omni_model_runner.OmniGPUModelRunner)
    result.model, result.device = model, runner.device
    result.vllm_config = None
    result.encoder_cache = EncoderCache()
    result._staged_writes = staged_writes.StagedWrites()
    result.jit_warmup_registry = JitWarmupRegistry(None)
    result.supports_mm_inputs = True
    result._omni_data_plane = result.sampler = result.pooling_runner = result.pp_handler = None
    result.adaptive_verification = result.prompt_logprobs_worker = None
    result.is_last_pp_rank = False
    result.block_tables = mocker.Mock()
    result.lora_state = mocker.Mock()
    result.req_states = mocker.Mock(spec=RequestState)
    slots = result.req_states.req_id_to_index = {}
    result.req_states.add_request.side_effect = lambda req_id, **_: slots.setdefault(req_id, len(slots))
    result.req_states.remove_request.side_effect = lambda req_id: slots.pop(req_id, None)
    state = object.__new__(DefaultModelState)
    state.model, state.encoder_cache = model, result.encoder_cache
    state.rope_state = state.mm_pruner = state.prompt_embeds_state = None
    state.supports_mm_inputs = True
    state.encoder_runner = EncoderRunner(model, 64, 3, result.encoder_cache, model.vae.dtype, runner.device)
    mocker.patch.object(omni_model_runner, "init_omni_model_state", return_value=state)

    def load(parent):
        from vllm.v1.worker.gpu import model_runner as upstream

        parent.model_state = upstream.init_model_state(None, model, result.encoder_cache, runner.device)

    mocker.patch.object(GPUModelRunner, "load_model", load)
    result.load_model()
    result.empty_batch = mocker.Mock(
        spec=InputBatch,
        req_ids=[],
        num_tokens=0,
        num_tokens_after_padding=0,
        input_ids=torch.empty(0, dtype=torch.long, device=runner.device),
        num_scheduled_tokens=np.array([], dtype=np.int32),
        query_start_loc_np=np.array([0]),
        prefill_len_np=np.array([], dtype=np.int32),
        num_computed_tokens_np=np.array([], dtype=np.int32),
    )
    return result


def admit(v2, *requests):
    new = [NewRequestData.from_request(req, ([],), req.all_token_ids, uses_mrope=True) for req in requests]
    v2.add_requests(hunyuan_rng.schedule(new=new))


def encode(v2, scheduled):
    v2._get_mm_embeddings(scheduled, v2.empty_batch)
    return [
        v2.encoder_cache.encoder_outputs[v2.encoder_cache.mm_features[req_id][i].identifier]
        for req_id, indices in scheduled.items()
        for i in indices
    ]


@pytest.mark.parametrize(
    "runner",
    [
        hunyuan_rng.runner_case("cpu", torch.float32),
        hunyuan_rng.runner_case("cuda", torch.float32),
    ],
    indirect=True,
)
def test_mrv2_replays_released_prefix_and_mixed_requests(v2, make_request, mocker):
    refs = hunyuan_rng.images()
    a, b = make_request("A", [*refs, refs[0]], 42), make_request("B", refs[:1], 0)
    covered = a.mm_features[0].mm_position
    a.num_computed_tokens = covered.offset + covered.length
    admit(v2, a, b)
    provider = v2.model._mm_rng_provider
    assert provider is not None and not provider.streams
    # vLLM 0.33 releases already consumed encoder inputs, including CPU fields.
    update = hunyuan_rng.schedule()
    update.encoder_input_updates = {"A": {0: None}}
    v2._apply_encoder_input_updates(update)
    expected_a, state_a = hunyuan_rng.reference(v2, [*refs, refs[0]], 42)
    expected_b, _ = hunyuan_rng.reference(v2, refs[:1], 0)
    counts = np.array([len(a.prompt_token_ids) - a.num_computed_tokens, len(b.prompt_token_ids)])
    batch = mocker.Mock(
        spec=InputBatch,
        req_ids=["A", "B"],
        num_tokens=int(counts.sum()),
        num_tokens_after_padding=int(counts.sum()),
        input_ids=torch.tensor(a.prompt_token_ids[a.num_computed_tokens :] + b.prompt_token_ids, device=v2.device),
        num_scheduled_tokens=counts,
        query_start_loc_np=np.array([0, counts[0], counts.sum()]),
        prefill_len_np=np.array([len(a.prompt_token_ids), len(b.prompt_token_ids)]),
        num_computed_tokens_np=np.array([a.num_computed_tokens, 0]),
    )
    # Encode in B/A order, then gather in A/B order through the real nonempty
    # token merge. Include separator, ViT and ordinary text embeddings.
    actual = v2._get_mm_embeddings({"B": [0], "A": [1, 2, 3]}, batch)
    zero = expected_a[0].new_zeros((1, 3))
    expected = []
    for images, latents, start in (
        ([*refs, refs[0]], expected_a, a.num_computed_tokens),
        (refs[:1], expected_b, 0),
    ):
        parts = [zero]
        for image, latent in zip(images, latents, strict=True):
            parts.extend((latent, zero, image[:, 0, 0][None].to(zero)))
        expected.append(torch.cat([*parts, zero])[start:])
    assert torch.equal(actual, torch.cat(expected))
    handle = provider.handles["A"]
    assert torch.equal(provider.streams[handle].generator.get_state(), state_a)
    assert all(MM_RNG_ITEM_KEY not in f.data for req in (a, b) for f in req.mm_features)
    # Eviction and reordered recomputation must restore the original draw positions.
    for i in (1, 3):
        v2.encoder_cache.free_encoder_cache(a.mm_features[i].identifier)
    hunyuan_rng.assert_latents(encode(v2, {"A": [3, 1]}), [expected_a[3], expected_a[1]])


@pytest.mark.cpu
@pytest.mark.parametrize("consumed", [1, 3])
@pytest.mark.parametrize("prefix_hit", [False, True])
def test_mrv2_readmits_after_input_release_and_preemption(v2, make_request, consumed, prefix_hit):
    refs = hunyuan_rng.images()
    req = make_request("A", refs, 42)
    first = NewRequestData.from_request(req, ([],), req.all_token_ids, uses_mrope=True)
    v2.add_requests(hunyuan_rng.schedule(new=[first]))
    encode(v2, {"A": list(range(consumed))})
    provider = v2.model._mm_rng_provider
    handle, plan = provider.handles["A"], v2._mm_rng_state.plans["A"]

    # Pins vLLM 0.33 behavior: _get_encoder_input_updates reads only the four
    # attributes set below. Revisit this stub on upstream scheduler changes.
    scheduler = object.__new__(Scheduler)
    scheduler.use_v2_model_runner, scheduler.is_encoder_decoder = True, False
    scheduler.requests, scheduler._pending_encoder_input_ids = {"A": req}, {}
    pos = req.mm_features[consumed - 1].mm_position
    req.num_computed_tokens = pos.offset + pos.length
    output = hunyuan_rng.schedule()
    output.encoder_input_updates = scheduler._get_encoder_input_updates({"A": 1}, [first], {})
    assert output.encoder_input_updates == {"A": {i: None for i in range(consumed)}}
    v2._apply_encoder_input_updates(output)
    assert all(v2.encoder_cache.mm_features["A"][i].data is None for i in range(consumed))
    # Even the uniprocess executor must preserve EngineCore's original pixels.
    assert all(f.data is not None and "vae_pixel_values" in f.data for f in req.mm_features)
    output.preempted_req_ids = {"A"}
    v2.finish_requests(output)

    # Re-admit independent Request/feature objects using schedule()'s actual
    # serialization entry point, with either retained prefix KV or recompute.
    resumed = copy(req)
    resumed.mm_features = [replace(f, data=copy(f.data)) for f in req.mm_features]
    resumed.num_computed_tokens = req.num_computed_tokens if prefix_hit else 0
    new = NewRequestData.from_request(resumed, ([],), resumed.all_token_ids, uses_mrope=True)
    for feature in new.mm_features:
        assert feature.data is not None
        assert "vae_generator_seed" in feature.data and "vae_token_grid_hw" in feature.data
    if prefix_hit:
        assert all("vae_pixel_values" not in f.data for f in new.mm_features[:consumed])
    v2.add_requests(hunyuan_rng.schedule(new=[new]))
    assert provider.handles["A"] == handle and v2._mm_rng_state.plans["A"] == plan
    indices = list(range(consumed if prefix_hit else 0, len(refs)))
    v2.encoder_cache.reset_encoder_cache()
    expected, expected_state = hunyuan_rng.reference(v2, refs, 42)
    hunyuan_rng.assert_latents(encode(v2, {"A": indices}), [expected[i] for i in indices])
    assert torch.equal(provider.streams[handle].generator.get_state(), expected_state)


@pytest.mark.cpu
def test_mrv2_cache_hits_preemption_finish_and_shutdown(v2, make_request, mocker):
    req = make_request("A", hunyuan_rng.images(), 42)
    admit(v2, req)
    expected, _ = hunyuan_rng.reference(v2, hunyuan_rng.images(), 42)
    for feature, latent in zip(req.mm_features, expected, strict=True):
        v2.encoder_cache.encoder_outputs[feature.identifier] = latent
    encode(v2, {"A": [0, 1, 2]})
    provider = v2.model._mm_rng_provider
    assert provider is not None and not provider.streams
    # Cancellation must also release a plan which never needed a generator.
    v2.finish_requests(hunyuan_rng.schedule(finished=["A"]))
    assert not v2._mm_rng_state.plans
    admit(v2, req)
    v2.encoder_cache.reset_encoder_cache()
    hunyuan_rng.assert_latents(encode(v2, {"A": [1]}), [expected[1]])

    handle = provider.handles["A"]
    output = hunyuan_rng.schedule()
    output.preempted_req_ids = {"A"}
    v2.finish_requests(output)
    assert provider.handles["A"] == handle and "A" not in v2.encoder_cache.mm_features
    admit(v2, req)
    hunyuan_rng.assert_latents(encode(v2, {"A": [2]}), [expected[2]])
    v2.finish_requests(hunyuan_rng.schedule(finished=["A"]))
    assert not provider.handles and not provider.streams and not v2._mm_rng_state.plans
    admit(v2, make_request("A", hunyuan_rng.images(), 0))
    v2.encoder_cache.reset_encoder_cache()
    encode(v2, {"A": [0]})
    assert provider.handles["A"] != handle
    mocker.patch.object(GPUModelRunner, "shutdown")
    v2.shutdown()
    v2.shutdown()
    assert not provider.handles and not provider.streams and not v2._mm_rng_state.plans


@pytest.mark.cpu
def test_mrv2_reuses_another_requests_prefix(v2, make_request):
    refs = hunyuan_rng.images()
    a, b = make_request("A", refs[:1], 42), make_request("B", refs, 42)
    admit(v2, a, b)
    encode(v2, {"A": [0]})
    cached = v2.encoder_cache.encoder_outputs[a.mm_features[0].identifier]
    expected, _ = hunyuan_rng.reference(v2, refs, 42)
    outputs = encode(v2, {"B": [0, 1, 2]})
    assert outputs[0] is cached
    hunyuan_rng.assert_latents(outputs, expected)


@pytest.mark.cpu
@pytest.mark.parametrize("seed", [None, 42])
def test_mrv2_live_request_plan_is_immutable(v2, make_request, seed):
    refs = hunyuan_rng.images()
    admit(v2, make_request("A", refs, seed))
    with pytest.raises(ValueError, match="RNG.*changed"):
        admit(v2, make_request("A", refs, 0))
    if seed is not None:
        with pytest.raises(ValueError, match="RNG.*changed"):
            admit(v2, make_request("A", refs[::-1], seed))
    else:
        assert len(encode(v2, {"A": [0, 1, 2]})) == 3
        assert not v2.model._mm_rng_provider.streams


@pytest.mark.cpu
def test_mrv2_failed_encode_restores_inputs_for_retry(v2, make_request, mocker):
    req = make_request("A", hunyuan_rng.images(), 0)
    admit(v2, req)
    original = v2.model.patch_embed
    mocker.patch.object(v2.model, "patch_embed", side_effect=RuntimeError("encoder failure"))
    with pytest.raises(RuntimeError, match="encoder failure"):
        encode(v2, {"A": [1]})
    assert all(a is b for a, b in zip(req.mm_features, v2.encoder_cache.mm_features["A"], strict=True))
    v2.model.patch_embed = original
    expected, _ = hunyuan_rng.reference(v2, hunyuan_rng.images(), 0)
    hunyuan_rng.assert_latents(encode(v2, {"A": [1]}), [expected[1]])
