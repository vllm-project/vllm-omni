# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU coverage for fused CausVid Cell dispatch.

The fused path must run one batched tensor forward per cell tick while every
member keeps its own conditioning, seed, sampler, KV owner and output row.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.causvid_wan import transformer as transformer_module
from vllm_omni.diffusion.models.causvid_wan.transformer import StageWanTransformer
from vllm_omni.diffusion.models.causvid_wan.wan_cell import WanCell, WanCellBatch
from vllm_omni.experimental.ar_diffusion.flow_euler import FlowEuler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SHAPE = (1, 1, 1, 1, 1)


class _StageTransformer:
    """Stand-in DiT that applies row-local conditioning and records the batch."""

    start_layer = 0
    end_layer = 1
    is_layer_group_first = True
    is_layer_group_last = True
    dim = 1

    def __init__(self):
        self.texts = {}
        self.shared = None
        self.blocks = []
        self.unembeds = []

    def prepare_cell(self, encoder_hidden_states, *, members=None, **_kwargs):
        if members is None:
            self.shared = encoder_hidden_states
            self.texts = {}
            return
        self.texts = {name: encoder_hidden_states[name] for name in members}
        self.shared = None

    def seq_len_for_latent(self, shape):
        return shape[-1]

    def embed(self, latent):
        return latent[:, :1].flatten(2).transpose(1, 2)

    def block(self, index, tokens, *, chunk, step, latent, timestep, kv_ctx=None, member=""):
        return self.block_batch(
            index,
            tokens,
            members=(member,),
            chunks=(chunk,),
            steps=(step,),
            latents=(latent,),
            timesteps=(timestep,),
            kv_ctxs=None if kv_ctx is None else (kv_ctx,),
        )

    def block_batch(self, index, tokens, *, members, chunks, steps, latents, timesteps, kv_ctxs):
        if tokens.shape[0] != len(members):
            raise AssertionError(f"fused block saw {tokens.shape[0]} rows for {len(members)} members")
        self.blocks.append(
            {
                "batch": tokens.shape[0],
                "members": tuple(members),
                "chunks": tuple(chunks),
                "steps": tuple(steps),
                "timesteps": tuple(float(timestep) for timestep in timesteps),
                "kv_ctxs": None if kv_ctxs is None else tuple(kv_ctxs),
                "latents": tuple(latents),
            }
        )
        return tokens + self._conditioning(members)

    def unembed(self, tokens, *, chunk, step, latent, timestep, member=""):
        return self.unembed_batch(tokens, members=(member,), steps=(step,), timesteps=(timestep,), latent=latent)

    def unembed_batch(self, tokens, *, members, steps, timesteps, latent):
        self.unembeds.append({"batch": tokens.shape[0], "members": tuple(members)})
        return latent + self._conditioning(members).reshape(len(members), 1, 1, 1, 1)

    def _conditioning(self, members) -> torch.Tensor:
        return torch.cat([self.texts[member] for member in members], dim=0)


class _MidStageTransformer(_StageTransformer):
    end_layer = 2
    is_layer_group_last = False


def _cell(transformer, condition, seed, shape=SHAPE):
    return WanCell(
        transformer,
        sampler=FlowEuler(1, shift=1.0),
        prompt_embeds=torch.full((1, 1, 1), condition),
        latent_shape=shape,
        seed=seed,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_fused_tick_is_one_batched_tensor_forward():
    transformer = _StageTransformer()
    cells = {"a": _cell(transformer, 2.0, 4), "b": _cell(transformer, 9.0, 8)}
    batch = WanCellBatch(cells)
    result = batch.forward((("a", (0, 0, 0)), ("b", (0, 0, 0))), [1.0, 2.0], hidden=None)

    assert len(transformer.blocks) == 1, "members must share one block forward"
    assert len(transformer.unembeds) == 1
    call = transformer.blocks[0]
    assert call["batch"] == 2
    assert call["members"] == ("a", "b")
    assert call["kv_ctxs"] == (1.0, 2.0), "each row keeps its own KV view"
    assert call["timesteps"] == (1000.0, 1000.0)
    assert list(batch._rows(result)) == [("a", 0, 0, 0), ("b", 0, 0, 0)]
    assert result["latent"].shape[0] == 2


def test_fused_rows_match_per_member_singleton_oracle():
    """The fused batch must equal the same members run one at a time."""

    def singleton(condition, seed):
        transformer = _StageTransformer()
        cell = _cell(transformer, condition, seed)
        return cell.forward((("only", (0, 0, 0)),), [None], hidden=None), cell

    transformer = _StageTransformer()
    cells = {"a": _cell(transformer, 2.0, 4), "b": _cell(transformer, 9.0, 8)}
    fused = WanCellBatch(cells).forward((("a", (0, 0, 0)), ("b", (0, 0, 0))), [None, None], hidden=None)

    for row, (condition, seed) in enumerate(((2.0, 4), (9.0, 8))):
        oracle, oracle_cell = singleton(condition, seed)
        torch.testing.assert_close(fused["latent"][row : row + 1], oracle["latent"])
        member = "ab"[row]
        torch.testing.assert_close(cells[member].finished[member][0], oracle_cell.finished["only"][0])
        torch.testing.assert_close(cells[member]._live[member][0], oracle_cell._live["only"][0])
    assert not torch.equal(fused["latent"][0:1], fused["latent"][1:2])


def test_fused_mid_stage_matches_per_member_singleton_oracle():
    """A non-last stage returns batched hidden states that equal the singletons."""

    def singleton(condition, seed):
        transformer = _MidStageTransformer()
        cell = _cell(transformer, condition, seed)
        return cell.forward((("only", (0, 0, 0)),), [None], hidden=None)

    transformer = _MidStageTransformer()
    cells = {"a": _cell(transformer, 2.0, 4), "b": _cell(transformer, 9.0, 8)}
    fused = WanCellBatch(cells).forward((("a", (0, 0, 0)), ("b", (0, 0, 0))), [None, None], hidden=None)

    assert set(fused) == {"latent", "hidden_states", WanCellBatch._META}
    for row, (condition, seed) in enumerate(((2.0, 4), (9.0, 8))):
        oracle = singleton(condition, seed)
        torch.testing.assert_close(fused["hidden_states"][row : row + 1], oracle["hidden_states"])
        torch.testing.assert_close(fused["latent"][row : row + 1], oracle["latent"])


def test_member_history_and_seeds_stay_isolated():
    transformer = _MidStageTransformer()
    cells = {"a": _cell(transformer, 2.0, 4), "b": _cell(transformer, 9.0, 8)}
    batch = WanCellBatch(cells)
    payload = batch.forward((("a", (0, 0, 0)), ("b", (0, 0, 0))), [None, None], hidden=None)

    assert set(cells["a"]._live) == {"a"} and set(cells["b"]._live) == {"b"}
    torch.testing.assert_close(cells["a"]._live["a"][0], cells["a"]._init_noise(0))
    torch.testing.assert_close(cells["b"]._live["b"][0], cells["b"]._init_noise(0))
    assert not torch.equal(cells["a"]._live["a"][0], cells["b"]._live["b"][0])
    assert cells["a"].finished == {} and cells["b"].finished == {}

    # A later tick for one member reuses only that member's own latent history.
    before = {name: dict(cell._live[name]) for name, cell in cells.items()}
    previous = batch._rows(payload)[("a", 0, 0, 0)]
    batch.forward((("a", (0, 0, 1)),), [None], hidden=previous)
    assert len(transformer.blocks) == 2
    assert transformer.blocks[1]["batch"] == 1 and transformer.blocks[1]["members"] == ("a",)
    assert cells["b"]._live["b"] == before["b"] and cells["b"].finished == {}


def test_failed_tick_leaves_member_state_untouched(monkeypatch):
    transformer = _StageTransformer()
    cells = {"a": _cell(transformer, 1.0, 1), "b": _cell(transformer, 2.0, 2)}
    batch = WanCellBatch(cells)
    stashed = {"latent": torch.zeros(SHAPE)}
    cells["b"]._stash[("b", 0, 0)] = stashed

    def fail(*_args, **_kwargs):
        raise RuntimeError("fused forward failure")

    monkeypatch.setattr(transformer, "block_batch", fail)
    with pytest.raises(RuntimeError, match="fused forward failure"):
        batch.forward((("a", (0, 0, 0)), ("b", (0, 0, 0))), [None, None], hidden=None)

    assert cells["a"].finished == {} and cells["b"].finished == {}
    assert cells["a"]._live == {} and cells["b"]._live == {}
    assert cells["b"]._stash[("b", 0, 0)] is stashed, "a failed tick must not consume peer activation"


def test_batch_rejects_incompatible_activation_shapes():
    transformer = _StageTransformer()
    a = _cell(transformer, 1.0, 1, (1, 1, 1, 1, 2))
    b = _cell(transformer, 2.0, 2, (1, 1, 1, 1, 3))
    with pytest.raises(ValueError, match="incompatible WanCell activation"):
        WanCellBatch({"a": a, "b": b})


def test_batch_rejects_members_on_different_transformers():
    a = _cell(_StageTransformer(), 1.0, 1)
    b = _cell(_StageTransformer(), 2.0, 2)
    with pytest.raises(ValueError, match="share one transformer"):
        WanCellBatch({"a": a, "b": b})


def test_row_metadata_stash_routes_each_member():
    transformer = _StageTransformer()
    cells = {"a": _cell(transformer, 1.0, 1), "b": _cell(transformer, 2.0, 2)}
    batch = WanCellBatch(cells)
    tasks = (("a", (2, 1, 0)), ("b", (4, 1, 0)))
    payload = batch.forward(tasks, [None, None], hidden=None)
    batch.stash_activation(batch.pack_activation(payload), meta=("", 2, 1))
    assert cells["a"].take_stashed_activation("a", 2, 1) is not None
    assert cells["b"].take_stashed_activation("b", 4, 1) is not None


def test_single_cell_batch_path_is_valid():
    transformer = _StageTransformer()
    cell = _cell(transformer, 3.0, 5)
    result = WanCellBatch({"only": cell}).forward((("only", (0, 0, 0)),), [None], hidden=None)
    assert result["latent"].shape == SHAPE
    assert len(transformer.blocks) == 1 and transformer.blocks[0]["batch"] == 1
    assert set(cell.finished) == {"only"}


def _bare_transformer(*, layers=1, head_dim=24, patch=(1, 1, 1), dim=4):
    """Real ``StageWanTransformer`` with stubbed DiT submodules (no weights)."""
    tf = StageWanTransformer.__new__(StageWanTransformer)
    torch.nn.Module.__init__(tf)
    tf.start_layer = 0
    tf.end_layer = layers
    tf.is_layer_group_first = True
    tf.is_layer_group_last = True
    tf.patch_size = patch
    tf.dim = dim
    tf._text = None
    tf._texts = None
    tf._cell_kv_ctx = None
    tf._cell_kv_ctxs = None
    tf._embeddings = {}
    tf._rotary = {}
    rope = SimpleNamespace(
        patch_size=patch,
        max_seq_len=16,
        attention_head_dim=head_dim,
        freqs_cos=torch.arange(16, dtype=torch.float32).unsqueeze(1).expand(16, head_dim).contiguous(),
        freqs_sin=torch.arange(16, dtype=torch.float32).unsqueeze(1).expand(16, head_dim).contiguous(),
    )
    calls: dict[str, list] = {"condition": []}

    def condition_embedder(timestep, text, image, timestep_seq_len=None):
        del image, timestep_seq_len
        calls["condition"].append((tuple(timestep.shape), tuple(text.shape), timestep.tolist()))
        batch = timestep.shape[0]
        temb = timestep.reshape(batch, 1).repeat(1, dim)
        enc = text + timestep.reshape(batch, 1, 1)
        return temb, temb.unsqueeze(1), enc, None

    def block(hidden, enc, timestep_proj, rotary, mask, meta, preserve):
        del timestep_proj, rotary, mask, meta, preserve
        return hidden + enc

    tf._wan = SimpleNamespace(
        rope=rope,
        condition_embedder=condition_embedder,
        timestep_proj_prepare=lambda proj, seq_len: proj,
        blocks=[block] * layers,
        norm_out=lambda hidden, scale, shift: hidden * scale + shift,
        proj_out=lambda hidden: hidden,
        output_scale_shift_prepare=lambda temb: (temb, temb),
    )
    tf.calls = calls
    return tf


def test_transformer_block_batch_conditions_every_member_in_one_pass():
    tf = _bare_transformer()
    latent = torch.zeros(SHAPE)
    hidden = torch.zeros(2, 1, 4)
    texts = {"a": torch.ones(1, 1, 4), "b": torch.full((1, 1, 4), 5.0)}
    tf.prepare_cell(texts, members=("a", "b"))

    fused = tf.block_batch(
        0,
        hidden,
        members=("a", "b"),
        chunks=(0, 0),
        steps=(0, 0),
        latents=(latent, latent),
        timesteps=(torch.tensor(1.0), torch.tensor(2.0)),
        kv_ctxs=(None, None),
    )

    assert len(tf.calls["condition"]) == 1, "one conditioning pass for the whole batch"
    timestep_shape, text_shape, timesteps = tf.calls["condition"][0]
    assert timestep_shape == (2,) and text_shape == (2, 1, 4)
    assert timesteps == [1.0, 2.0]

    singletons: list[torch.Tensor] = []
    for member, timestep in (("a", 1.0), ("b", 2.0)):
        tf.prepare_cell(texts, members=(member,))
        singletons.append(
            tf.block(
                0,
                hidden[len(singletons) : len(singletons) + 1],
                chunk=0,
                step=0,
                latent=latent,
                timestep=torch.tensor(timestep),
                member=member,
            )
        )
    torch.testing.assert_close(fused, torch.cat(singletons, dim=0))


def test_transformer_unembed_batch_matches_singletons():
    tf = _bare_transformer()
    latent = torch.zeros(2, 1, 1, 1, 1)
    hidden = torch.arange(8, dtype=torch.float32).reshape(2, 1, 4)
    texts = {"a": torch.ones(1, 1, 4), "b": torch.full((1, 1, 4), 5.0)}
    tf.prepare_cell(texts, members=("a", "b"))

    fused = tf.unembed_batch(
        hidden, members=("a", "b"), steps=(0, 0), timesteps=(torch.tensor(1.0), torch.tensor(2.0)), latent=latent
    )

    singletons = []
    for index, (member, timestep) in enumerate((("a", 1.0), ("b", 2.0))):
        tf.prepare_cell(texts, members=(member,))
        singletons.append(
            tf.unembed(
                hidden[index : index + 1],
                chunk=0,
                step=0,
                latent=latent[index : index + 1],
                timestep=torch.tensor(timestep),
                member=member,
            )
        )
    torch.testing.assert_close(fused, torch.cat(singletons, dim=0))
    assert fused.shape == (2, 4, 1, 1, 1)


class _Attn:
    num_heads = 1
    num_kv_heads = 1
    head_dim = 2

    def __init__(self):
        self.dense = []
        self.rotary = []

    def to_qkv(self, hidden):
        return torch.cat([hidden, hidden, hidden], dim=-1), None

    def norm_q(self, value):
        return value

    def norm_k(self, value):
        return value

    def rotary_embedding(self, value, cos, sin):
        self.rotary.append((value.shape, cos.shape, sin.shape))
        return value

    def attn(self, query, key, value, meta):
        self.dense.append(tuple(query.shape))
        return query

    def to_out(self, hidden):
        return hidden

    def dropout(self, hidden):
        return hidden


def test_self_attention_keeps_one_kv_view_per_row(monkeypatch):
    written = []

    def fake_paged(inputs, query, key, value, k_act, v_act, scale):
        written.append((inputs, query.clone(), key.clone(), value.clone(), scale))
        return query

    monkeypatch.setattr(transformer_module, "paged_write_attn", fake_paged)
    attn = _Attn()
    hidden = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
    contexts = (SimpleNamespace(name="a"), SimpleNamespace(name="b"))

    out = transformer_module._wan_self_attn_with_kv(attn, hidden, None, contexts)

    assert attn.dense == [], "request-scoped KV must not fall back to dense attention"
    assert [call[0].name for call in written] == ["a", "b"]
    torch.testing.assert_close(written[0][1].flatten(), hidden[0].flatten())
    torch.testing.assert_close(written[1][2].flatten(), hidden[1].flatten())
    torch.testing.assert_close(out, hidden)
    assert written[0][4] == pytest.approx(1.0 / 2**0.5)


def test_self_attention_batches_rows_when_no_row_owns_kv():
    attn = _Attn()
    hidden = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])

    out = transformer_module._wan_self_attn_with_kv(attn, hidden, None, (None, None))

    assert attn.dense == [(2, 1, 1, 2)], "dense attention runs once for the whole batch"
    torch.testing.assert_close(out, hidden)


def test_self_attention_applies_shared_rope_once_per_batch():
    attn = _Attn()
    hidden = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
    rotary = (torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1))

    transformer_module._wan_self_attn_with_kv(attn, hidden, rotary, (None, None))

    assert attn.rotary == [((2, 1, 1, 2), (1, 1, 1, 1), (1, 1, 1, 1)), ((2, 1, 1, 2), (1, 1, 1, 1), (1, 1, 1, 1))]


def _pipeline(monkeypatch, *, max_batch=2):
    from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.noisy_pp_executor import (
        ARDiffusionCellContext,
        CellRunSpec,
        CellTopology,
    )
    from vllm_omni.diffusion.models.causvid_wan import pipeline_causvid_wan as module
    from vllm_omni.experimental.ar_diffusion.kv_cache.noisy import ARDiffusionNoisyKVSpec, NoisyKVCache, NoisyKVState

    class Model(torch.nn.Module, _StageTransformer):
        num_layers = 2
        local_num_layers = 2
        end_layer = 2
        dim = 1

        def __init__(self):
            torch.nn.Module.__init__(self)
            _StageTransformer.__init__(self)
            self.register_parameter("device_anchor", torch.nn.Parameter(torch.zeros(1)))

        def seq_len_for_latent(self, shape):
            return shape[2] * shape[3] * shape[4]

        def block_batch(self, index, tokens, *, members, chunks, steps, latents, timesteps, kv_ctxs):
            assert kv_ctxs is not None
            # Exercise real request-local KV write views without an attention
            # kernel or model weights. Every owned layer writes its own slot.
            for kv_ctx in kv_ctxs:
                kv_ctx.key_pool.zero_()
                kv_ctx.value_pool.zero_()
            return _StageTransformer.block_batch(
                self,
                index,
                tokens,
                members=members,
                chunks=chunks,
                steps=steps,
                latents=latents,
                timesteps=timesteps,
                kv_ctxs=kv_ctxs,
            )

        def unembed_batch(self, tokens, *, members, steps, timesteps, latent):
            self.unembeds.append({"batch": tokens.shape[0], "members": tuple(members)})
            return tokens.transpose(1, 2).reshape_as(latent)

    pipeline = module.CausVidWanPipeline.__new__(module.CausVidWanPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.transformer = Model()
    pipeline.step_replicas = pipeline.layer_groups = 1
    pipeline.max_history_chunks = 1
    pipeline._default_shift = 1.0
    pipeline._default_denoise = 1
    pipeline.vae = None
    pipeline._streaming_decoder = None
    pipeline.od_config = SimpleNamespace(video_output_transport=SimpleNamespace(enable_device_postprocess=False))
    pipeline._encode_prompt = lambda text, device, dtype: torch.full((1, 1, 1), float(text), dtype=dtype, device=device)
    pipeline._decode_latents = lambda latents: latents
    cache = NoisyKVCache(
        ARDiffusionNoisyKVSpec(2, 1, 1, 2, 2, 1),
        dtype=torch.float32,
        device=torch.device("cpu"),
        max_batch_size=max_batch,
        layer_groups=1,
    )
    ctx = ARDiffusionCellContext(CellRunSpec(CellTopology(1, 1), max_batch_size=max_batch), NoisyKVState(cache))
    pipeline._chunk_ctx = ctx
    monkeypatch.setattr(module, "resolve_pp_rank_and_group", lambda: (0, None))
    return pipeline, ctx, module


def _request(name, *, prompt="2", seed=7, shape=(1, 1, 1, 1, 2), **extra):
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    return OmniDiffusionRequest(
        request_id=name,
        prompt=prompt,
        sampling_params=OmniDiffusionSamplingParams(
            seed=seed,
            extra_args={"latent_shape": shape, "num_chunks": 2, "num_denoise_steps": 1, **extra},
        ),
    )


def test_pipeline_real_context_batch_matches_independent_requests(monkeypatch):
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

    pipeline, ctx, module = _pipeline(monkeypatch)
    requests = [_request("a", prompt="2", seed=5), _request("b", prompt="7", seed=11)]
    run = module.run_cell_pipeline
    calls = []
    task_sizes = []
    batch_sizes = []

    def counted(*, ctx, adapter):
        calls.append(len(ctx.pending))
        forward = adapter.forward

        def capture(tasks, contexts, *, hidden):
            task_sizes.append(len(tasks))
            return forward(tasks, contexts, hidden=hidden)

        adapter.forward = capture
        return run(ctx=ctx, adapter=adapter)

    monkeypatch.setattr(module, "run_cell_pipeline", counted)
    actual = pipeline.forward(DiffusionRequestBatch(requests))
    batch_sizes = [call["batch"] for call in pipeline.transformer.blocks]
    assert calls == [2]
    assert max(task_sizes) == 2
    assert batch_sizes and max(batch_sizes) == 2, "members of one tick share one batched forward"
    assert not ctx.pending and not ctx.inflight and not ctx.kv.cache.pool.keys
    expected = [pipeline.forward(request)[0] for request in requests]
    for batched, single in zip(actual, expected):
        torch.testing.assert_close(batched.output, single.output)
    assert not torch.equal(actual[0].output, actual[1].output)


def test_pipeline_serializes_equal_token_but_different_shapes_and_steps(monkeypatch):
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

    pipeline, ctx, module = _pipeline(monkeypatch)
    calls = []
    run = module.run_cell_pipeline

    def counted(*, ctx, adapter):
        calls.append(len(ctx.pending))
        return run(ctx=ctx, adapter=adapter)

    monkeypatch.setattr(module, "run_cell_pipeline", counted)
    requests = [
        _request("a"),
        _request("b", shape=(1, 1, 1, 2, 1)),
        _request("c", shape=(1, 1, 1, 2, 1), num_denoise_steps=2),
    ]
    outputs = pipeline.forward(DiffusionRequestBatch(requests))
    assert calls == [1, 1, 1]
    assert len(outputs) == 3
    assert outputs[0].output.shape[-2:] == (1, 2)
    assert outputs[1].output.shape[-2:] == (2, 1)
    assert not ctx.pending and not ctx.inflight and not ctx.kv.cache.pool.keys


@pytest.mark.parametrize("failure", ["setup", "forward", "emit"])
def test_pipeline_failure_cancels_all_jobs_and_drops_only_call_ids(monkeypatch, failure):
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

    pipeline, ctx, module = _pipeline(monkeypatch)
    jobs = []

    class Job:
        cancelled = False

        def __init__(self, **_kwargs):
            jobs.append(self)

        def on_finished(self, *_args):
            pass

        def join(self):
            raise RuntimeError("emit failure")

        def cancel(self):
            self.cancelled = True

    monkeypatch.setattr(module, "start_async_stream_decode", Job)
    # Unrelated history must survive cancellation of this invocation.
    sentinel = ("unrelated", 0, 0, 0)
    ctx.kv.cache.pool.alloc(sentinel)
    if failure == "setup":
        encode = pipeline._encode_prompt

        def fail_encode(prompt, device, dtype):
            if prompt == "bad":
                raise RuntimeError("setup failure")
            return encode(prompt, device, dtype)

        pipeline._encode_prompt = fail_encode
    elif failure == "forward":

        def fail_block(*_args, **_kwargs):
            raise RuntimeError("forward failure")

        # A tick with two members enters through the fused batch entry point.
        pipeline.transformer.block = fail_block
        pipeline.transformer.block_batch = fail_block
    requests = [
        _request("a", stream_decode=True),
        _request("b", prompt="bad" if failure == "setup" else "3", stream_decode=True),
    ]
    with pytest.raises(RuntimeError, match=f"{failure} failure"):
        pipeline.forward(DiffusionRequestBatch(requests))
    assert jobs and all(job.cancelled for job in jobs)
    assert len(jobs) == (1 if failure == "setup" else 2)
    assert not ctx.pending and not ctx.inflight
    assert set(ctx.kv.cache.pool.keys) == {sentinel}
    ctx.kv.cache.pool.release(sentinel)
    assert all(len(free) == ctx.kv.cache.pool.capacity for free in ctx.kv.cache.pool.free)
