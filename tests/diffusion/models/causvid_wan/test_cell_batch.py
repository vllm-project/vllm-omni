# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU integration-shaped coverage for request-row CausVid Cell dispatch."""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.models.causvid_wan.wan_cell import WanCell, WanCellBatch
from vllm_omni.experimental.ar_diffusion.flow_euler import FlowEuler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Transformer:
    start_layer = 0
    end_layer = 1
    is_layer_group_first = True
    is_layer_group_last = True
    dim = 4

    def __init__(self):
        self.text = None
        self.seen = []

    def prepare_cell(self, prompt):
        self.text = prompt

    def seq_len_for_latent(self, shape):
        return shape[-1]

    def embed(self, latent):
        return latent[:, :1].flatten(2).transpose(1, 2)

    def block(self, index, tokens, **_kwargs):
        assert self.text is not None
        self.seen.append(float(self.text[0, 0, 0]))
        return tokens + self.text[:, :1, :1]

    def unembed(self, tokens, *, latent, **_kwargs):
        return torch.zeros_like(latent)


def _cell(tf, name, condition, seed, shape=(1, 1, 1, 1, 1)):
    return WanCell(
        tf,
        sampler=FlowEuler(1, shift=1.0),
        prompt_embeds=torch.full((1, 1, 1), condition),
        latent_shape=shape,
        seed=seed,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_dispatches_per_request_conditioning_seed_and_outputs():
    tf = _Transformer()
    cells = {"a": _cell(tf, "a", 2.0, 4), "b": _cell(tf, "b", 9.0, 8)}
    batch = WanCellBatch(cells)
    tasks = (("a", (0, 0, 0)), ("b", (0, 0, 0)))
    result = batch.forward(tasks, [None, None], hidden=None)
    assert tf.seen == [2.0, 9.0]
    torch.testing.assert_close(cells["a"]._init_noise(0), _cell(tf, "a", 2.0, 4)._init_noise(0))
    assert not torch.equal(cells["a"]._init_noise(0), cells["b"]._init_noise(0))
    assert list(batch._rows(result)) == [("a", 0, 0, 0), ("b", 0, 0, 0)]
    assert set(cells["a"].finished) == {"a"} and set(cells["b"].finished) == {"b"}


def test_batch_rejects_incompatible_activation_shapes():
    tf = _Transformer()
    a = _cell(tf, "a", 1.0, 1, (1, 1, 1, 1, 2))
    b = _cell(tf, "b", 2.0, 2, (1, 1, 1, 1, 3))
    with pytest.raises(ValueError, match="incompatible WanCell activation"):
        WanCellBatch({"a": a, "b": b})


def test_row_metadata_stash_routes_each_member():
    tf = _Transformer()
    cells = {"a": _cell(tf, "a", 1.0, 1), "b": _cell(tf, "b", 2.0, 2)}
    batch = WanCellBatch(cells)
    tasks = (("a", (2, 1, 0)), ("b", (4, 1, 0)))
    payload = batch.forward(tasks, [None, None], hidden=None)
    batch.stash_activation(batch.pack_activation(payload), meta=("", 2, 1))
    assert cells["a"].take_stashed_activation("a", 2, 1) is not None
    assert cells["b"].take_stashed_activation("b", 4, 1) is not None


def test_single_cell_batch_path_is_valid():
    tf = _Transformer()
    cell = _cell(tf, "only", 3.0, 5)
    result = WanCellBatch({"only": cell}).forward((("only", (0, 0, 0)),), [None], hidden=None)
    assert result["latent"].shape == (1, 1, 1, 1, 1)
    assert set(cell.finished) == {"only"}


def _pipeline(monkeypatch, *, max_batch=2):
    from types import SimpleNamespace

    from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.noisy_pp_executor import (
        ARDiffusionCellContext,
        CellRunSpec,
        CellTopology,
    )
    from vllm_omni.diffusion.models.causvid_wan import pipeline_causvid_wan as module
    from vllm_omni.experimental.ar_diffusion.kv_cache.noisy import ARDiffusionNoisyKVSpec, NoisyKVCache, NoisyKVState

    class Model(torch.nn.Module, _Transformer):
        num_layers = 2
        local_num_layers = 2
        end_layer = 2
        dim = 1

        def __init__(self):
            torch.nn.Module.__init__(self)
            _Transformer.__init__(self)
            self.register_parameter("device_anchor", torch.nn.Parameter(torch.zeros(1)))

        def seq_len_for_latent(self, shape):
            return shape[2] * shape[3] * shape[4]

        def block(self, index, tokens, *, kv_ctx=None, **kwargs):
            assert kv_ctx is not None
            # Exercise real request-local KV write views without an attention
            # kernel or model weights. Every owned layer writes its own slot.
            kv_ctx.key_pool.zero_()
            kv_ctx.value_pool.zero_()
            return _Transformer.block(self, index, tokens, **kwargs)

        def unembed(self, tokens, *, latent, **_kwargs):
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
    assert calls == [2]
    assert max(task_sizes) == 2
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

        pipeline.transformer.block = fail_block
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
