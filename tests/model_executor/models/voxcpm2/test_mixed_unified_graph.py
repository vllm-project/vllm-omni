# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Packed-to-request layout and admission contracts for mixed VoxCPM2 graphs."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.voxcpm2.model_state import VoxCPM2RequestState
from vllm_omni.model_executor.models.voxcpm2.scheduler import VoxCPM2OmniARAsyncScheduler
from vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker import VoxCPM2TalkerForConditionalGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _model():
    model = VoxCPM2TalkerForConditionalGeneration.__new__(VoxCPM2TalkerForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model._enable_unified_decode_graph = True
    model._mrv2_model_state = None
    model._active_states = {}
    model.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(enable_chunked_prefill=False, max_num_batched_tokens=16)
    )
    return model


def _ready_metadata_model():
    model = _model()
    model._enable_cuda_graph = True
    model._tts = SimpleNamespace(feat_decoder=SimpleNamespace(estimator=SimpleNamespace(_compiled=True)))
    model._cuda_graph_warmup_steps = model._cuda_graph_warmup_threshold = 3
    model._max_cached_graphs = 8
    model._mixed_graph_cache_limit = 0
    model._unified_decode_graph_max_batch_size = 4
    model._mixed_graphs = {}
    model._unified_graphs = {}
    model._select_unified_graph_bucket_size = lambda count: 4
    return model


@pytest.mark.parametrize("chunked,uses_v2,expected", [(False, True, True), (True, True, True), (False, False, False)])
def test_mixed_graph_requires_v2_for_both_prefill_modes(chunked, uses_v2, expected):
    model = _model()
    model._mrv2_model_state = object() if uses_v2 else None
    model.vllm_config.scheduler_config.enable_chunked_prefill = chunked
    assert model._mixed_unified_graph_enabled() is expected


def test_mixed_token_capacity_bounds_and_buckets():
    model = _model()
    assert model._mixed_token_capacity(5, 4) == 8
    assert model._mixed_token_capacity(1, 4) == 4
    assert model._mixed_token_capacity(16, 4) == 16
    assert model._mixed_token_capacity(17, 4) is None


def test_mixed_graph_packs_prefill_masks_and_decode_conditions(monkeypatch):
    model = _model()
    prefill = VoxCPM2RequestState(request_id="prefill")
    prefill.prefill_masks = (
        torch.tensor([[1, 0, 1]]),
        torch.tensor([[0, 1, 0]]),
        torch.arange(12.0).reshape(1, 3, 2, 2),
        torch.full((1, 3, 4), 5.0),
    )
    decode = VoxCPM2RequestState(request_id="decode")
    decode.prev_feat_embed = torch.full((1, 4), 7.0)
    decode.curr_prefix_feat_cond = torch.full((2, 2), 9.0)
    model._active_states = {s.request_id: s for s in (prefill, decode)}
    model._pending_requests = [("prefill", True, None, 3), ("decode", False, None, 1)]
    graph = SimpleNamespace(
        batch_size=4,
        text_mask=torch.empty(8, 1),
        prev_feat_embed=torch.empty(8, 4),
        prefix_feat_cond=torch.empty(4, 2, 2),
        last_token_indices=torch.empty(4, dtype=torch.long),
    )
    model._prepare_mixed_graph_inputs(graph, 2)
    assert graph.text_mask[:, 0].tolist() == [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert graph.prev_feat_embed[:, 0].tolist() == [0.0, 5.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0]
    assert graph.last_token_indices.tolist() == [2, 3, 3, 3]
    torch.testing.assert_close(graph.prefix_feat_cond[0], prefill.prefill_masks[2][0, -1])
    assert graph.prefix_feat_cond[1].tolist() == [[9.0, 9.0]] * 2
    assert not graph.prefix_feat_cond[2:].any()


@pytest.mark.parametrize("uses_v2,chunked,defer", [(True, False, False), (True, True, False), (False, False, True)])
def test_mixed_admission_removes_only_graph_specific_restriction(monkeypatch, uses_v2, chunked, defer):
    from vllm.v1.request import RequestStatus

    scheduler = object.__new__(VoxCPM2OmniARAsyncScheduler)
    scheduler.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(use_v2_model_runner=uses_v2),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=chunked),
    )
    scheduler.waiting = [object()]
    scheduler.kv_holding_waiting = []
    scheduler.running = [SimpleNamespace(status=RequestStatus.RUNNING, is_finished=lambda: False, num_prompt_tokens=5)]
    monkeypatch.setattr(scheduler, "_unified_decode_graph_enabled", lambda: True)
    monkeypatch.setattr(scheduler, "_get_confirmed_num_computed_tokens", lambda _: 5)
    assert scheduler._should_defer_waiting_admission() is defer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "per_patch_audio,incremental_vae,patch_count",
    [(False, False, None), (True, False, None), (True, True, None), (True, True, 0), (True, True, 1), (True, True, 2)],
)
def test_mixed_graph_capture_replays_changed_packed_layout(monkeypatch, per_patch_audio, incremental_vae, patch_count):
    """Exercise actual CUDA capture; attention is stubbed, so this is not FA2 E2E."""
    from contextlib import nullcontext

    from vllm_omni.model_executor.models.voxcpm2 import voxcpm2_talker as module

    model = _model()
    model.config = SimpleNamespace(hidden_size=4)
    model._device = torch.device("cuda")
    model._side_dtype = torch.float32
    model._feat_dim = model._patch_size = 2
    model._inference_timesteps = 1
    model._cfg_value = model._cfg_cutoff_ratio = 1.0
    model._unified_graph_stats = SimpleNamespace(captures=0)

    class Backbone:
        def precompute_fused_qkv(self):
            pass

        def __call__(self, **kwargs):
            return kwargs["inputs_embeds"] * 2

    model.model = Backbone()
    model.residual_model = Backbone()
    projection = torch.nn.Linear(4, 2, device="cuda")
    tts = SimpleNamespace(
        lm_to_dit_proj=projection,
        res_to_dit_proj=projection,
        fsq_layer=lambda x: x + 10,
        fusion_concat_proj=lambda x: x[:, :4] + x[:, 4:],
        feat_decoder=SimpleNamespace(estimator=torch.nn.Identity()),
        feat_encoder=lambda x: x.mean(dim=2),
        enc_to_lm_proj=lambda x: torch.cat([x, x], dim=-1),
    )
    model._tts = tts
    model._dit_proj_fn = lambda lm, res: torch.cat([projection(lm), projection(res)], dim=-1)
    monkeypatch.setattr(model, "_setup_cfm_buffers", lambda: None)
    monkeypatch.setattr(model, "_setup_execution_optimizations", lambda: None)
    monkeypatch.setattr(model, "_voxcpm2_compile_unified_capture_estimator", lambda x: x)
    monkeypatch.setattr(model, "_voxcpm2_compile_unified_capture_feat_encoder", lambda x: x)
    monkeypatch.setattr(module, "_CFMBufferManager", lambda **kwargs: None)
    monkeypatch.setattr(module, "get_forward_context", lambda: SimpleNamespace(attn_metadata=None))
    monkeypatch.setattr(module, "override_forward_context", lambda _: nullcontext())
    cfm_calls = []

    def solve(decoder, mu, patch, cond, *args, **kwargs):
        cfm_calls.append(mu.shape[0])
        return mu[:, :2, None] + cond

    monkeypatch.setattr(module, "_optimized_solve_euler_with_noise", solve)
    if per_patch_audio:
        from vllm_omni.model_executor.models.voxcpm2.model_state import VoxCPM2ModelState

        owner = object.__new__(VoxCPM2ModelState)
        owner.model = model
        owner.per_patch_audio = True
        owner.incremental_vae = incremental_vae
        owner.vae_feature_cache = {}
        owner.slots = [object(), object()]
        owner.next_embed = torch.zeros(3, 4, device="cuda")
        owner.prefix_feat = torch.zeros(3, 2, 2, device="cuda")
        owner.stop_logits = torch.zeros(3, 2, device="cuda")
        owner.active_slots = torch.tensor([True, True, False], device="cuda")
        owner.latent_context = torch.zeros(3, 4, 2, device="cuda")
        owner.latent_context_lengths = torch.zeros(3, dtype=torch.long, device="cuda")
        model._mrv2_model_state = owner
        tts.audio_vae = SimpleNamespace(
            decoder=torch.nn.Identity(),
            decode_chunk_size=3,
            decode=lambda x, **kwargs: x.sum(1).repeat_interleave(3, dim=1),
        )
        model._stop_fn = lambda x: x[:, :2]
        monkeypatch.setattr(model, "_get_vae_decode_sr_cond", lambda _: None)

    graph = model._capture_unified_decode_graph(4, 8, patch_count)
    if patch_count == 0:
        assert not cfm_calls
    else:
        assert cfm_calls and all(count == (4 if patch_count is None else patch_count) for count in cfm_calls)
    expected_valid = (
        [False, True, False, False]
        if patch_count == 1
        else ([False] * 4 if patch_count == 0 else [True, True, False, False])
    )
    if graph.patch_indices is not None:
        graph.patch_indices.copy_(torch.tensor([1, 0][:patch_count], device="cuda"))
    if per_patch_audio:
        assert (graph.pcm is not None) == incremental_vae
        assert not owner.latent_context_lengths.any()
        graph.audio_slots.copy_(torch.tensor([0, 1, 2, 2], device="cuda"))
        graph.audio_rows_valid.copy_(torch.tensor([True, True, False, False], device="cuda"))
    for order, length in (("prefill-first", 3), ("decode-first", 5)):
        prefill = VoxCPM2RequestState(request_id="prefill")
        prefill.prefill_masks = (
            torch.ones(1, length, device="cuda"),
            torch.zeros(1, length, device="cuda"),
            torch.ones(1, length, 2, 2, device="cuda"),
            torch.zeros(1, length, 4, device="cuda"),
        )
        decode = VoxCPM2RequestState(
            request_id="decode",
            prev_feat_embed=torch.ones(1, 4, device="cuda"),
            curr_prefix_feat_cond=torch.ones(2, 2, device="cuda"),
        )
        entries = [(prefill, True, None, length), (decode, False, None, 1)]
        if order == "decode-first":
            entries.reverse()
        model._mrv2_model_state = None  # Use request tensors for this layout-only test.
        model._active_states = {s.request_id: s for s, *_ in entries}
        model._pending_requests = [(s.request_id, *entry) for s, *entry in entries]
        model._prepare_mixed_graph_inputs(graph, 2)
        graph.input_embeds.copy_(torch.arange(32.0, device="cuda").reshape(8, 4))
        graph.graph.replay()
        scaffold_h = graph.input_embeds * 2
        expected = (scaffold_h + 10) * (1 - graph.text_mask) + scaffold_h * graph.text_mask
        expected = expected.index_select(0, graph.last_token_indices)
        torch.testing.assert_close(graph.lm_hidden, expected)
        assert graph.next_feat_embed.shape == (4, 4)
        if per_patch_audio:
            if not incremental_vae:
                graph.pcm, graph.pcm_valid = owner.decode_patch(
                    graph.audio_slots, graph.audio_rows_valid, graph.cfm_output.transpose(1, 2)
                )
            assert graph.pcm.shape == (4, 6)
            assert graph.pcm_valid.tolist() == expected_valid
            torch.testing.assert_close(graph.stop_logits, graph.lm_hidden[:, :2])

    if patch_count == 1:
        graph.patch_indices.fill_(0)
        graph.graph.replay()
        assert graph.pcm_valid.tolist() == [True, False, False, False]

    if per_patch_audio:
        model._mrv2_model_state = owner
        decode_graph = model._capture_unified_decode_graph(4, patch_batch_size=patch_count)
        if decode_graph.patch_indices is not None:
            decode_graph.patch_indices.copy_(torch.tensor([1, 0][:patch_count], device="cuda"))
        decode_graph.audio_slots.copy_(torch.tensor([0, 1, 2, 2], device="cuda"))
        decode_graph.audio_rows_valid.copy_(torch.tensor([True, True, False, False], device="cuda"))
        decode_graph.input_embeds.fill_(1)
        decode_graph.prev_feat_embed.fill_(1)
        decode_graph.prefix_feat_cond.fill_(1)
        decode_graph.graph.replay()
        if not incremental_vae:
            assert decode_graph.pcm is None
            decode_graph.pcm, decode_graph.pcm_valid = owner.decode_patch(
                decode_graph.audio_slots, decode_graph.audio_rows_valid, decode_graph.cfm_output.transpose(1, 2)
            )
        assert decode_graph.pcm.shape == (4, 6)
        assert decode_graph.pcm_valid.tolist() == expected_valid
        torch.testing.assert_close(decode_graph.stop_logits, torch.full((4, 2), 12.0, device="cuda"))


def test_mixed_metadata_request_uses_token_and_request_buckets():
    model = _ready_metadata_model()
    model._mrv2_model_state = object()
    request = model.get_runner_assisted_full_attention_metadata_request(
        req_ids=["prefill", "decode"],
        num_reqs=2,
        num_scheduled_tokens=[5, 1],
        num_computed_tokens=[0, 20],
        max_num_scheduled_tokens=5,
    )
    assert request.num_reqs_padded == 4
    assert request.num_tokens_padded == 8
    assert request.for_cudagraph_capture
    model._mixed_graphs[(4, 8)] = object()
    replay = model.get_runner_assisted_full_attention_metadata_request(
        req_ids=["a", "b"],
        num_reqs=2,
        num_scheduled_tokens=[1, 7],
        num_computed_tokens=[20, 0],
        max_num_scheduled_tokens=7,
    )
    assert not replay.for_cudagraph_capture
    model._mixed_graph_cache_limit = 1
    args = dict(
        req_ids=["a", "b"],
        num_reqs=2,
        num_scheduled_tokens=[8, 1],
        num_computed_tokens=[0, 20],
        max_num_scheduled_tokens=8,
    )
    assert model.get_runner_assisted_full_attention_metadata_request(**args) is None
    model._mixed_graph_cache_limit = 0
    assert model.get_runner_assisted_full_attention_metadata_request(**args).for_cudagraph_capture
    model._cuda_graph_warmup_steps = 0
    assert (
        model.get_runner_assisted_full_attention_metadata_request(
            req_ids=["a", "b"],
            num_reqs=2,
            num_scheduled_tokens=[5, 1],
            num_computed_tokens=[0, 20],
            max_num_scheduled_tokens=5,
        )
        is None
    )


def test_all_decode_keeps_uniform_metadata_when_mixed_graph_enabled():
    model = _ready_metadata_model()
    # Metadata selection needs slot readiness, not the GPU buffers themselves.
    model._mrv2_model_state = SimpleNamespace(tensor_for=lambda *_: torch.ones(1))
    model._active_states = {
        req_id: VoxCPM2RequestState(request_id=req_id, prefill_completed=True) for req_id in ("a", "b")
    }
    request = model.get_runner_assisted_full_attention_metadata_request(
        req_ids=["a", "b"],
        num_reqs=2,
        num_scheduled_tokens=[1, 1],
        num_computed_tokens=[20, 30],
        max_num_scheduled_tokens=1,
    )
    assert request.num_reqs_padded == 4
    assert request.num_tokens_padded is None
    assert request.for_cudagraph_capture
    model._unified_graphs[4] = object()
    replay = model.get_runner_assisted_full_attention_metadata_request(
        req_ids=["a", "b"],
        num_reqs=2,
        num_scheduled_tokens=[1, 1],
        num_computed_tokens=[20, 30],
        max_num_scheduled_tokens=1,
    )
    assert not replay.for_cudagraph_capture


def test_per_patch_single_decode_requests_runner_metadata():
    model = _ready_metadata_model()
    model._mrv2_model_state = SimpleNamespace(per_patch_audio=True, tensor_for=lambda *_: torch.ones(1))
    model._select_unified_graph_bucket_size = lambda count: count
    model._active_states = {"a": VoxCPM2RequestState(request_id="a", prefill_completed=True)}
    request = model.get_runner_assisted_full_attention_metadata_request(
        req_ids=["a"],
        num_reqs=1,
        num_scheduled_tokens=[1],
        num_computed_tokens=[20],
        max_num_scheduled_tokens=1,
    )
    assert request.num_reqs_padded == 1
    assert request.num_tokens_padded is None
    assert request.for_cudagraph_capture


def test_chunked_prefill_graph_uses_current_chunk_masks_and_conditioning():
    model = _model()
    model.vllm_config.scheduler_config.enable_chunked_prefill = True
    model._mrv2_model_state = SimpleNamespace(per_patch_audio=True)
    assert model._mixed_unified_graph_enabled()
    state = VoxCPM2RequestState(request_id="chunk", prefill_chunk_start=2)
    state.prefill_masks = (
        torch.tensor([[1.0, 1.0, 0.0, 1.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 1.0, 0.0, 1.0]]),
        torch.arange(20.0).reshape(1, 5, 2, 2),
        torch.arange(20.0).reshape(1, 5, 4),
    )
    graph = SimpleNamespace(
        batch_size=1,
        text_mask=torch.empty(4, 1),
        prev_feat_embed=torch.empty(4, 4),
        prefix_feat_cond=torch.empty(1, 2, 2),
        last_token_indices=torch.empty(1, dtype=torch.long),
    )
    model._active_states = {state.request_id: state}
    model._pending_requests = [(state.request_id, True, None, 2)]
    model._prepare_mixed_graph_inputs(graph, 1)
    assert graph.text_mask[:, 0].tolist() == [0.0, 1.0, 0.0, 0.0]
    torch.testing.assert_close(graph.prev_feat_embed[0], state.prefill_masks[3][0, 2])
    assert not graph.prev_feat_embed[1:].any()
    torch.testing.assert_close(graph.prefix_feat_cond[0], state.prefill_masks[2][0, 3])
    assert graph.last_token_indices.tolist() == [1]


@pytest.mark.parametrize("missing", ["request_bucket", "token_capacity"])
def test_mixed_invalid_capacity_falls_through_without_decode_graph(monkeypatch, missing):
    model = _ready_metadata_model()
    model._perf = SimpleNamespace(start=lambda *_: None)
    model._mrv2_model_state = SimpleNamespace(per_patch_audio=True)
    model._runner_assisted_unified_decode_graph_active = True
    model._active_states = {"p": VoxCPM2RequestState(request_id="p")}
    model._pending_requests = [("p", True, None, 2)]
    if missing == "request_bucket":
        model._select_unified_graph_bucket_size = lambda _: None
        model._mixed_token_capacity = lambda *_: pytest.fail("capacity must not receive a missing bucket")
    else:
        model._mixed_token_capacity = lambda *_: None
    model._forward_unified_decode = lambda *_: pytest.fail("invalid mixed capacity selected a decode graph")

    class FallbackReachedError(Exception):
        pass

    def normal_dispatch(**kwargs):
        assert not kwargs["is_all_decode"]
        raise FallbackReachedError

    monkeypatch.setattr(model, "_unified_decode_graph_skip_reason", normal_dispatch)
    with pytest.raises(FallbackReachedError):
        model.forward(torch.zeros(2, dtype=torch.long), torch.arange(2), inputs_embeds=torch.zeros(2, 4))


@pytest.mark.parametrize("kind,patch_count", [("mid", 1), ("last", 2), ("zero", 0), ("decode", 2)])
def test_exact_metadata_reuses_actual_patch_variant(kind, patch_count):
    model = _ready_metadata_model()
    model._runtime_config = SimpleNamespace(enable_exact_patch_batch=True)
    model._mrv2_model_state = SimpleNamespace(per_patch_audio=True, tensor_for=lambda *_: torch.ones(1))
    first = VoxCPM2RequestState(request_id="a", prefill_completed=kind == "decode")
    second = VoxCPM2RequestState(request_id="b", prefill_completed=kind != "zero")
    # Deliberately stale: selection must use current scheduler progress.
    first.is_last_prefill_chunk = kind != "last"
    model._active_states = {"a": first, "b": second}
    scheduled = [1, 1] if kind == "decode" else [2, 2] if kind == "zero" else [2, 1]
    computed = [20, 30] if kind == "decode" else [4, 20] if kind == "last" else [0, 0] if kind == "zero" else [0, 20]
    prompts = [10, 10] if kind == "decode" else [6, 6] if kind == "zero" else [6, 10]
    args = dict(
        req_ids=["a", "b"],
        num_reqs=2,
        num_scheduled_tokens=scheduled,
        num_computed_tokens=computed,
        max_num_scheduled_tokens=max(scheduled),
        num_prompt_tokens=prompts,
    )
    request = model.get_runner_assisted_full_attention_metadata_request(**args)
    assert request.for_cudagraph_capture
    if kind == "decode":
        model._unified_graphs[(4, patch_count)] = object()
    else:
        model._mixed_graphs[(4, 4, patch_count)] = object()
        model._mixed_graph_cache_limit = 1
    assert not model.get_runner_assisted_full_attention_metadata_request(**args).for_cudagraph_capture
    if kind == "mid":
        args["num_computed_tokens"] = [4, 20]  # Same request/token bucket, new patch count.
        assert model.get_runner_assisted_full_attention_metadata_request(**args) is None
        model._mixed_graph_cache_limit = 0
        assert model.get_runner_assisted_full_attention_metadata_request(**args).for_cudagraph_capture
