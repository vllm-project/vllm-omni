# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Transfer page visibility against the dense temporal-frame contract."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tests.diffusion.models.cosmos3_nano_sim_bimanual.test_transfer_resolution_contract import params, pipeline
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.geometry import (
    Cosmos3NanoSimBimanualGeometry,
    Cosmos3NanoSimBimanualResolutionPolicy,
)
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.pipeline_cosmos3_nano_sim_transfer import (
    _TransferConditioning,
)
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.state_cosmos3_nano_sim_bimanual import (
    Cosmos3NanoSimBimanualSessionState,
)
from vllm_omni.experimental.ar_diffusion.kv_cache import ARDiffusionKVCache, ARDiffusionKVConfig, paged_write_attn
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def paged_pipeline(chunk_size=1):
    p = pipeline(chunk_size)
    p.resolution_policy = Cosmos3NanoSimBimanualResolutionPolicy()
    p.transformer = SimpleNamespace(num_hidden_layers=2, num_kv_heads_local=1, head_dim=4)
    return p


def make_cache(spec):
    return ARDiffusionKVCache(
        ARDiffusionKVConfig(
            enable=True,
            chunk_size=spec.tokens_per_frame,
            window_chunks=spec.window_frames,
            sink_chunks=spec.sink_frames,
            gpu_memory_fraction=1.0,
        ),
        num_layers=spec.num_layers,
        num_kv_heads=spec.num_kv_heads,
        head_size=spec.head_size,
        dtype=torch.float32,
        block_size=spec.tokens_per_frame,
        max_model_len=spec.max_model_len,
        available_bytes=1 << 20,
        kv_branches=spec.kv_branches,
        session_capacity=spec.session_capacity,
        frames_per_block=spec.frames_per_block,
        cross_attention_lengths=spec.cross_attention_lengths,
        max_scratch_frames_per_branch=spec.max_scratch_frames_per_branch,
        max_scratch_tokens_per_branch=spec.max_scratch_tokens_per_branch,
        eviction_group_frames=spec.eviction_group_frames,
        device=torch.device("cpu"),
    )


@pytest.mark.parametrize("window,sink", [(1, 0), (2, 1), (4, 0), (4, 2), (30, 3), (51, 1)])
def test_transfer_paged_rollout_visibility_and_attention(window, sink, monkeypatch):
    """Exercise real allocation, scratch writes, attention, eviction and reuse.

    Frame labels are independent of physical page IDs and absolute positions;
    a uniform query makes attention's expected output the visible-label mean.
    """
    monkeypatch.delenv("VLLM_OMNI_AR_DIFFUSION_KV_GATHER", raising=False)
    p = paged_pipeline()
    geometry = Cosmos3NanoSimBimanualGeometry(height=32, width=32)
    spec = p._kv_spec_for_geometry(geometry, window_frames=window, sink_frames=sink, text_capacity=3)
    cache = make_cache(spec)
    free_before = cache.manager.block_pool.get_num_free_blocks()
    adapter = cache.begin_request("paired")
    paged = ARDiffusionKVState(cache, "paired", {"main": adapter}, num_layers=2)
    state = Cosmos3NanoSimBimanualSessionState(session_id="paired")
    total = 3 * window + 1
    request = p._validate_conditioning_request(
        params(1 + 4 * (total - 1), kv_cache_inference_size=window, attention_sink_size=sink), None
    )
    conditioning = _TransferConditioning(request, torch.zeros(1, 48, total, 2, 2))
    calls = []
    scratch_slots = []
    table_shapes = set()

    def run(frame, phase, commit):
        past = [j for j in range(frame) if j < sink or j >= frame - (window - sink - 1)]
        labels = [float(v) for j in past for v in (2 * j, 2 * j + 1)]
        if phase != "control":
            labels.append(float(2 * frame))
        current = float(2 * frame if phase == "control" else 2 * frame + 1)
        contexts = paged.get_kv_caches("main", seq_len=1, commit_current=commit, extra_visible_tokens=1)
        ctx = contexts[0].forward_ctx
        # The retained K/V must be unchanged across all four denoising calls.
        for layer in range(2):
            actual = cache.key_cache(layer)[ctx.history_block_ids, 0, 0, 0].tolist()
            assert actual == labels, (phase, frame, actual, labels)
        before = adapter.completed_chunks
        ctx.prepare(device=torch.device("cpu"), action_len=3, query_len=1)
        for layer_context in contexts:
            inputs = layer_context.to_layer_inputs()
            table_shapes.add(tuple(inputs.block_table.shape))
            q = torch.zeros(1, 1, 4)
            kv = torch.full_like(q, current)
            text = torch.full((3, 1, 4), -1.0)
            actual = paged_write_attn(inputs, q, kv, kv, text, text, 0.5)
            expected = (sum(labels) + current - 3) / (len(labels) + 4)
            torch.testing.assert_close(actual, torch.full_like(actual, expected), rtol=1e-5, atol=1e-5)
        if not commit:
            scratch_slots.append(ctx.current_video_slot_mapping.clone())
        paged.commit_paged_context("main")
        assert adapter.completed_chunks == before + int(commit)
        assert len(cache.window_block_ids(adapter)) <= 2 * window - 1
        calls.append((phase, frame))

    def control(*args, frame_start, **kwargs):
        run(frame_start, "control", True)

    def denoise(*args, chunk_start, **kwargs):
        for _ in range(4):
            run(chunk_start, "denoise", False)
        return torch.zeros(1, 48, 1, 2, 2)

    def clean(*args, frame_idx, **kwargs):
        run(frame_idx, "clean", True)

    p._transformer_forward = control
    p._denoise_chunk = denoise
    p._commit_clean_frame = clean
    for frame in range(total):
        p._run_chunk(
            state,
            geometry=geometry,
            chunk_start=frame,
            chunk_end=frame + 1,
            target_frame=total,
            terminal_request=True,
            request_start_frame=0,
            seed=42,
            text_kv=[],
            real_text_kv_len=3,
            fps=30,
            conditioning=conditioning,
            tick_durations={},
            measure_tick_latency=False,
        )
    assert ("clean", total - 1) not in calls
    assert adapter.completed_chunks == 2 * total - 1
    assert adapter.compacted_tokens > 0
    assert len(table_shapes) == 1
    assert all(torch.equal(slots, scratch_slots[0]) for slots in scratch_slots)
    assert not state.dense_kv_by_branch
    paged.close()
    assert cache.manager.block_pool.get_num_free_blocks() == free_before


@pytest.mark.parametrize("window,sink,text", [(30, 3, 4096), (1, 0, 512), (4, 2, 8192)])
def test_transfer_request_spec_uses_control_geometry_and_overrides(window, sink, text):
    p = paged_pipeline()
    sp = params(
        kv_cache_inference_size=window,
        attention_sink_size=sink,
        max_prompt_tokens=text,
        depth={"control": {"height": 832, "width": 480}},
        resolution="480",
    )
    request = SimpleNamespace(prompt={"prompt": "full cookbook caption"}, sampling_params=sp)
    result = p.ar_diffusion_request_spec(request)
    spec = result.kv_spec
    assert result.geometry_key == ((832, 480), window, sink, text)
    assert spec.tokens_per_frame == Cosmos3NanoSimBimanualGeometry(height=832, width=480).vision_tokens_per_frame
    assert (spec.window_frames, spec.sink_frames, spec.eviction_group_frames) == (2 * (window - sink) - 1, 2 * sink, 2)
    assert spec.cross_attention_lengths == {"text": text}
    assert spec.max_scratch_tokens_per_branch == text
    p.validate_ar_diffusion_effective_spec(spec)
    batch = SimpleNamespace(prompts=[request.prompt], sampling_params=sp)
    assert p.ar_diffusion_request_spec(batch) == result


def test_transfer_paged_rejects_unpaired_spec():
    p = paged_pipeline()
    spec = p.ar_diffusion_kv_cache_spec()
    with pytest.raises(ValueError, match="history"):
        p.validate_ar_diffusion_effective_spec(replace(spec, eviction_group_frames=1))
    with pytest.raises(ValueError, match="history"):
        p.validate_ar_diffusion_effective_spec(replace(spec, window_frames=2))


@pytest.mark.parametrize("chunk_size", [2, 3, 4, 8])
def test_chunkwise_transfer_full_history_and_partial_spans(chunk_size):
    p = paged_pipeline(chunk_size)
    geometry = Cosmos3NanoSimBimanualGeometry(height=32, width=32)
    spec = p._kv_spec_for_geometry(geometry, text_capacity=3)
    assert spec.eviction_group_frames == 1 and spec.sink_frames == 0
    assert spec.frames_per_block == spec.max_scratch_frames_per_branch == chunk_size
    p.validate_ar_diffusion_effective_spec(spec)
    cache = make_cache(spec)
    state = ARDiffusionKVState(cache, "chunk", {"main": cache.begin_request("chunk")}, num_layers=2)
    # Initial singleton, full chunks, then a shorter terminal span. The low
    # level paging API supports partial spans independent of clip admission.
    history = []
    cursor = 0

    def forward(labels, commit):
        contexts = state.get_kv_caches(
            "main", seq_len=len(labels), commit_current=commit, extra_visible_tokens=len(labels)
        )
        ctx = contexts[0].forward_ctx
        for layer in range(2):
            assert cache.key_cache(layer)[ctx.history_block_ids, 0, 0, 0].tolist() == history
        ctx.prepare(device=torch.device("cpu"), action_len=3, query_len=len(labels))
        kv = torch.tensor(labels, dtype=torch.float32)[:, None, None].expand(-1, 1, 4)
        text = torch.full((3, 1, 4), -1.0)
        expected = (sum(history) + sum(labels) - 3) / (len(history) + len(labels) + 3)
        before = state.adapter("main").completed_chunks
        for layer in contexts:
            actual = paged_write_attn(layer.to_layer_inputs(), torch.zeros_like(kv), kv, kv, text, text, 0.5)
            torch.testing.assert_close(actual, torch.full_like(actual, expected), rtol=1e-5, atol=1e-5)
        state.commit_paged_context("main")
        assert state.adapter("main").completed_chunks == before + (len(labels) if commit else 0)
        if commit:
            history.extend(labels)

    for span in (1, chunk_size, chunk_size, chunk_size - 1):
        control = [float(2 * t) for t in range(cursor, cursor + span)]
        generated = [label + 1 for label in control]
        forward(control, True)
        for _ in range(4):
            forward(generated, False)
        for label in generated:
            forward([label], True)
        cursor += span
    assert state.adapter("main").compacted_tokens == 0
    assert len(cache.window_block_ids(state.adapter("main"))) == 2 * cursor
    state.close()


@pytest.mark.parametrize("chunk_size", [2, 3, 4, 8])
def test_chunkwise_request_spec_preserves_full_history_contract(chunk_size):
    p = paged_pipeline(chunk_size)
    sp = params(
        1 + 4 * chunk_size * 2,
        depth={"control": {"height": 480, "width": 832}},
        resolution="480",
        max_prompt_tokens=4096,
    )
    request = SimpleNamespace(prompt="caption", sampling_params=sp)
    spec = p.ar_diffusion_request_spec(request).kv_spec
    assert (spec.window_frames, spec.sink_frames, spec.eviction_group_frames) == (96, 0, 1)
    assert spec.max_scratch_frames_per_branch == chunk_size
    sp.extra_args["kv_cache_inference_size"] = 2
    with pytest.raises(ValueError, match="full history"):
        p.ar_diffusion_request_spec(request)


def test_transfer_bound_pool_checks_request_limits():
    p = paged_pipeline()
    geometry = Cosmos3NanoSimBimanualGeometry(height=32, width=32)
    spec = p._kv_spec_for_geometry(geometry, window_frames=30, sink_frames=3, text_capacity=4096)
    cache = make_cache(spec)
    state = SimpleNamespace(kv_cache=cache)
    p._validate_bound_kv_geometry(state)
    p._validate_bound_kv_geometry(state, geometry, expected_spec=spec)
    wrong = p._kv_spec_for_geometry(geometry, window_frames=31, sink_frames=3, text_capacity=4096)
    with pytest.raises(RuntimeError, match="window_frames"):
        p._validate_bound_kv_geometry(state, geometry, expected_spec=wrong)


def test_paged_text_padding_uses_request_capacity():
    p = paged_pipeline()
    geometry = Cosmos3NanoSimBimanualGeometry(height=32, width=32)
    spec = p._kv_spec_for_geometry(geometry, window_frames=1, sink_frames=0, text_capacity=640)
    cache = make_cache(spec)
    paged = ARDiffusionKVState(cache, "text", {"main": cache.begin_request("text")}, num_layers=2)
    p._ar_diffusion_kv_state = paged
    raw = [(torch.randn(1, 600, 1, 4), torch.randn(1, 600, 1, 4)) for _ in range(2)]
    p.transformer.encode_und_kv = lambda *args: (raw, 600)

    def pad(values, *, max_len):
        assert max_len == 640  # checkpoint default remains 512
        return [
            (
                torch.nn.functional.pad(k, (0, 0, 0, 0, 0, max_len - 600)),
                torch.nn.functional.pad(v, (0, 0, 0, 0, 0, max_len - 600)),
            )
            for k, v in values
        ]

    p.transformer.pad_text_kv = pad
    result = p._ensure_text_kv(
        Cosmos3NanoSimBimanualSessionState(session_id="text"), torch.ones(1, 600), torch.ones(1, 600), max_length=640
    )
    for (k, v), (expected_k, expected_v) in zip(result, raw):
        torch.testing.assert_close(k[:, :600], expected_k)
        torch.testing.assert_close(v[:, :600], expected_v)
    paged.close()


def test_transfer_uses_existing_bimanual_paged_forward_and_transformer():
    from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.pipeline_cosmos3_nano_sim_bimanual import (
        Cosmos3NanoSimBimanualPipeline,
    )
    from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.pipeline_cosmos3_nano_sim_transfer import (
        Cosmos3NanoSimTransferPipeline,
    )
    from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.transformer_cosmos3_nano_sim_bimanual import (
        Cosmos3NanoSimBimanualTransformer,
    )
    from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.transformer_cosmos3_nano_sim_transfer import (
        Cosmos3NanoSimTransferTransformer,
    )

    assert Cosmos3NanoSimTransferPipeline._transformer_forward is Cosmos3NanoSimBimanualPipeline._transformer_forward
    assert Cosmos3NanoSimTransferTransformer._gen_layer_cls is Cosmos3NanoSimBimanualTransformer._gen_layer_cls
