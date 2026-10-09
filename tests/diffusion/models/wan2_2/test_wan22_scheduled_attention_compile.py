# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A schedule-configured Wan service compiles each block as one graph, through the real model.

A tiny ``WanTransformer3DModel`` runs its real ``forward``: every block gets the post-patch grid as
``vsa_dit_seq_shape``, so self-attention metadata always carries a ``VideoTokenLayout`` and
``extra["vsa_dit_seq_shape"]``. The model is regionally compiled with Inductor and
``dynamic=True``, as the model runner does. Attention runs the production SDPA math on CPU.
"""

import socket

import pytest
import torch
from torch._dynamo.utils import counters as dynamo_counters

import vllm_omni.diffusion.attention.layer as layer_mod
import vllm_omni.diffusion.models.wan2_2.wan2_2_transformer as wan_mod
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend, SDPAImpl
from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention
from vllm_omni.diffusion.attention.schedule import AttentionScheduleRange
from vllm_omni.diffusion.compile import regionally_compile
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import AttentionConfig, AttentionScheduleConfig, AttentionSpec, OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import (
    DenoiseProgressMixin,
    begin_scheduled_denoise,
    bind_attention_schedule,
    set_forward_context,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_BLOCKS = 3
_STEPS = 3
# Latents [batch, channels, frames, height, width]; patch (1, 2, 2) gives grid (3, 8, 8), 192 tokens.
_SHAPE = (1, 4, 3, 16, 16)
_OTHER_SHAPE = (1, 4, 2, 12, 20)
_SEEN: list[tuple[str, object]] = []


@pytest.fixture
def wan_env(monkeypatch):
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.layers.utils import default_unquantized_gemm

    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.dispatch_unquantized_gemm",
        lambda *_args, **_kwargs: default_unquantized_gemm,
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    init_distributed_environment(
        world_size=1, rank=0, local_rank=0, distributed_init_method=f"tcp://127.0.0.1:{port}", backend="gloo"
    )
    initialize_model_parallel()
    monkeypatch.setattr(wan_mod, "get_pipeline_parallel_world_size", lambda: 1)
    monkeypatch.setattr(wan_mod, "is_pipeline_first_stage", lambda: True)
    monkeypatch.setattr(wan_mod, "is_pipeline_last_stage", lambda: True)
    monkeypatch.setattr(layer_mod, "get_attn_backend_for_role", _resolve)
    monkeypatch.setattr(layer_mod, "build_parallel_attention_strategy", lambda **kwargs: NoParallelAttention())
    monkeypatch.setattr(SDPAImpl, "forward", _recording_sdpa)
    # Multi-threaded CPU GEMMs and reductions are not bit-reproducible from run to run, even
    # without a schedule; one thread makes bitwise comparisons meaningful.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    _SEEN.clear()
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()
    _SEEN.clear()
    torch.set_num_threads(threads)
    cleanup_dist_env_and_memory()


def _resolve(*, role, head_size, attention_config=None, role_category=None, allow_trtllm_default=True):
    spec = None
    if attention_config is not None:
        spec, _source = attention_config.resolve_with_source(role=role, role_category=role_category)
    return SDPABackend, spec


def _recording_sdpa(self, query, key, value, attn_metadata=None):
    """The production SDPA math; records the metadata of every call that runs outside a graph."""
    if not torch.compiler.is_compiling():
        _SEEN.append((type(self).__name__, attn_metadata))
    return self.forward_cuda(query, key, value, attn_metadata)


class _CountingInductor:
    """Compiles each captured graph with Inductor and counts its executions."""

    def __init__(self):
        self.graphs: list[set[str]] = []
        self.executions: list[int] = []

    def __call__(self, gm, example_inputs):
        from torch._inductor.compile_fx import compile_fx

        index = len(self.graphs)
        self.graphs.append(
            {getattr(n.target, "__name__", str(n.target)) for n in gm.graph.nodes if n.op == "call_function"}
        )
        self.executions.append(0)
        compiled = compile_fx(gm, example_inputs)

        def run(*args):
            self.executions[index] += 1
            return compiled(*args)

        return run


class _Pipeline(DenoiseProgressMixin):
    pass


def _service(schedule_config):
    # Dynamo keeps automatic-dynamic state per code object, so start each service from scratch.
    torch._dynamo.reset()
    config = OmniDiffusionConfig(diffusion_attention_schedule=schedule_config)
    with set_current_diffusion_config(config):
        torch.manual_seed(0)
        model = wan_mod.WanTransformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads=2,
            attention_head_dim=64,
            in_channels=4,
            out_channels=4,
            text_dim=32,
            freq_dim=32,
            ffn_dim=256,
            num_layers=_BLOCKS,
            rope_max_seq_len=64,
        )
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        # vLLM linear layers allocate with torch.empty and expect a weight loader.
        for name, param in sorted(model.named_parameters()):
            scale = 0.02 if param.ndim > 1 else 0.1
            offset = 1.0 if "norm" in name and param.ndim == 1 else 0.0
            param.copy_(torch.randn(param.shape, generator=generator) * scale + offset)
    model = model.to(torch.bfloat16).eval()
    counter = _CountingInductor()
    regionally_compile(model, backend=counter, dynamic=True)
    return model, config, counter


def _run(model, config, counter, schedule, shape=_SHAPE):
    """One request of ``_STEPS`` steps; returns each step's output and graph executions per block."""
    generator = torch.Generator().manual_seed(1)
    latents = torch.randn(shape, generator=generator).to(torch.bfloat16)
    text = torch.randn(1, 16, 32, generator=generator).to(torch.bfloat16)
    outputs, executions = [], []
    pipeline = _Pipeline()
    with torch.no_grad(), set_forward_context(omni_diffusion_config=config), bind_attention_schedule(schedule):
        total = begin_scheduled_denoise(_STEPS)
        for step in range(_STEPS):
            pipeline.record_denoise_step(step, 1000.0 * (_STEPS - step) / _STEPS, total_steps=total)
            before = sum(counter.executions)
            out = model(latents, torch.tensor([999 - 100 * step]), text, return_dict=False)[0]
            executions.append((sum(counter.executions) - before) / _BLOCKS)
            outputs.append(out)
            latents = latents - 0.1 * out
        pipeline.record_denoise_step(None)
    return outputs, executions


def _schedule():
    return AttentionScheduleConfig(profiles={"approx": AttentionConfig(default=AttentionSpec(backend="SDPA"))})


def test_real_wan_forward_is_one_graph_per_block_and_matches_a_service_without_a_schedule(wan_env):
    plain = _service(None)
    plain_outputs, plain_executions = _run(*plain, None)
    dynamo_counters.clear()
    scheduled = _service(_schedule())
    dense_outputs, dense_executions = _run(*scheduled, ())

    assert dynamo_counters["graph_break"] == {}
    assert plain_executions == dense_executions == [1.0] * _STEPS
    counter = scheduled[2]
    assert len(counter.graphs) == len(plain[2].graphs)
    assert all("scheduled_attention" in ops and "scaled_dot_product_attention" not in ops for ops in counter.graphs)
    assert all("scaled_dot_product_attention" in ops for ops in plain[2].graphs)
    # Same kernels in the same partition, so the dense request is bit-identical, not just close.
    for step, (expected, actual) in enumerate(zip(plain_outputs, dense_outputs)):
        assert torch.equal(expected, actual), step


def test_op_hands_wan_metadata_to_attention_and_schedules_do_not_recompile(wan_env):
    model, config, counter = _service(_schedule())
    _run(model, config, counter, ())
    graphs = len(counter.graphs)

    _SEEN.clear()
    with torch._dynamo.config.patch(error_on_recompile=True):
        _, executions = _run(model, config, counter, (AttentionScheduleRange(start=1, end=2, profile="approx"),))
    assert executions == [1.0] * _STEPS and len(counter.graphs) == graphs

    # Eager reference: the metadata the real block builds, without compile.
    self_attn = [md for _name, md in _SEEN if md is not None]
    assert len(self_attn) == _STEPS * _BLOCKS
    for md in self_attn:
        assert md.attn_mask is None
        assert md.video_layout == layer_mod.VideoTokenLayout(prefix_len=0, latent_grid=(3, 8, 8), used_len=192)
        assert md.extra == {"vsa_dit_seq_shape": (3, 8, 8)}
    # Cross-attention keeps no metadata.
    assert sum(md is None for _name, md in _SEEN) == _STEPS * _BLOCKS


def test_new_resolution_compiles_no_more_than_a_service_without_a_schedule(wan_env):
    plain = _service(None)
    _run(*plain, None)
    _run(*plain, None, shape=_OTHER_SHAPE)
    scheduled = _service(_schedule())
    _run(*scheduled, ())
    _SEEN.clear()
    _, executions = _run(*scheduled, (), shape=_OTHER_SHAPE)

    assert executions == [1.0] * _STEPS
    assert len(scheduled[2].graphs) == len(plain[2].graphs)
    grids = {md.video_layout.latent_grid for _name, md in _SEEN if md is not None}
    assert grids == {(2, 6, 10)}
    assert {md.extra["vsa_dit_seq_shape"] for _name, md in _SEEN if md is not None} == {(2, 6, 10)}
