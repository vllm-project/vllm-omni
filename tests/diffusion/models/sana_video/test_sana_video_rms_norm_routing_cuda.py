# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CUDA regressions for SANA-Video's distributed RMSNorm fast path.

These tests deliberately exercise ``SanaDistributedRMSNorm`` rather than the
ordinary caption RMSNorm.  The former normalizes the video Q/K projections and
has a sum/count contract so that TP ranks can contribute disjoint channel
shards.  TP=1 may use the exact fused implementation; TP>1 must retain the
collective and global element count.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator

import pytest
import torch
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

import vllm_omni.diffusion.layers.sana_rms_norm as sana_rms
import vllm_omni.diffusion.models.sana_video.transformer_sana_video as sana_transformer
from vllm_omni.platforms import current_omni_platform

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.diffusion,
    pytest.mark.skipif(not current_platform.is_cuda(), reason="NVIDIA CUDA required"),
]


@pytest.fixture(autouse=True)
def _reset_fusion_state() -> Iterator[None]:
    caches = (
        sana_rms._VERIFIED_SIGNATURES,
        sana_rms._DISABLED_SIGNATURES,
        sana_rms._VERIFIED_SUM_SIGNATURES,
        sana_rms._DISABLED_SUM_SIGNATURES,
    )
    for cache in caches:
        cache.clear()
    yield
    for cache in caches:
        cache.clear()


@pytest.fixture
def tp1_cuda_group(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Initialize the real TP=1 group used by vLLM parallel linear layers."""
    from vllm.utils.network_utils import get_open_port

    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import (
        AttentionConfig,
        AttentionSpec,
        DiffusionParallelConfig,
        OmniDiffusionConfig,
    )
    from vllm_omni.diffusion.distributed.parallel_state import (
        destroy_distributed_env,
        init_distributed_environment,
        initialize_model_parallel,
    )

    device = torch.device(f"{current_omni_platform.device_type}:0")
    current_omni_platform.set_device(device)
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", str(get_open_port()))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="env://",
    )
    initialize_model_parallel(tensor_parallel_size=1)

    old_deterministic = torch.backends.cudnn.deterministic
    old_benchmark = torch.backends.cudnn.benchmark
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    parallel_config = DiffusionParallelConfig(
        pipeline_parallel_size=1,
        data_parallel_size=1,
        tensor_parallel_size=1,
        sequence_parallel_size=1,
        ulysses_degree=1,
        ring_degree=1,
        cfg_parallel_size=1,
    )
    omni_config = OmniDiffusionConfig(
        model="test",
        dtype=torch.bfloat16,
        parallel_config=parallel_config,
        diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA")),
    )
    try:
        with set_current_diffusion_config(omni_config):
            yield
    finally:
        torch.backends.cudnn.deterministic = old_deterministic
        torch.backends.cudnn.benchmark = old_benchmark
        destroy_distributed_env()


def _sum_count_reference(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """The exact pre-fast-path ``SanaDistributedRMSNorm.forward`` expression."""
    hidden_states_float = hidden_states.float()
    sum_sq = hidden_states_float.pow(2).sum(dim=-1, keepdim=True)
    normalized = hidden_states_float * torch.rsqrt(sum_sq / hidden_states.shape[-1] + eps)
    if weight.dtype in (torch.float16, torch.bfloat16):
        normalized = normalized.to(weight.dtype)
    return normalized * weight


def _raw_bf16_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    assert left.dtype is right.dtype is torch.bfloat16
    return torch.equal(left.view(torch.int16), right.view(torch.int16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize("shape", [(1, 1024, 2240), (2, 8192, 224)])
def test_exact_sum_rms_norm_matches_video_widths(shape: tuple[int, ...]) -> None:
    """Cover the production width and the reduced-width integration fixture."""
    torch.manual_seed(101)
    hidden_states = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(shape[-1], device="cuda", dtype=torch.bfloat16)
    expected = _sum_count_reference(hidden_states, weight, 1e-5)

    with torch.no_grad():
        result = sana_rms.exact_sana_rms_norm_sum(hidden_states, weight, 1e-5)

    signature = sana_rms._signature(hidden_states)
    assert _raw_bf16_equal(result, expected)
    assert signature in sana_rms._VERIFIED_SUM_SIGNATURES
    assert signature not in sana_rms._DISABLED_SUM_SIGNATURES


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize("failure", ["mismatch", "exception"])
def test_sum_rms_norm_failure_is_fail_closed(
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hidden_states = torch.randn((1, 1024, 2240), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2240, device="cuda", dtype=torch.bfloat16)
    expected = _sum_count_reference(hidden_states, weight, 1e-5)
    calls = 0

    def broken(*_args) -> torch.Tensor:
        nonlocal calls
        calls += 1
        if failure == "exception":
            raise RuntimeError("synthetic sum-kernel failure")
        return torch.zeros_like(hidden_states)

    monkeypatch.setattr(sana_rms, "_launch_exact_sana_rms_norm_sum", broken)
    with torch.no_grad():
        first = sana_rms.exact_sana_rms_norm_sum(hidden_states, weight, 1e-5)
        second = sana_rms.exact_sana_rms_norm_sum(hidden_states, weight, 1e-5)

    signature = sana_rms._signature(hidden_states)
    assert calls == 1
    assert _raw_bf16_equal(first, expected)
    assert _raw_bf16_equal(second, expected)
    assert signature in sana_rms._DISABLED_SUM_SIGNATURES
    assert signature not in sana_rms._VERIFIED_SUM_SIGNATURES


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_small_sum_rms_norm_stays_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    hidden_states = torch.randn((2, 300, 2240), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2240, device="cuda", dtype=torch.bfloat16)
    expected = _sum_count_reference(hidden_states, weight, 1e-5)
    monkeypatch.setattr(
        sana_rms,
        "_launch_exact_sana_rms_norm_sum",
        lambda *_args: pytest.fail("small tensors must not launch the sum/count fast path"),
    )

    with torch.no_grad():
        result = sana_rms.exact_sana_rms_norm_sum(hidden_states, weight, 1e-5)

    assert _raw_bf16_equal(result, expected)
    assert not sana_rms._VERIFIED_SUM_SIGNATURES
    assert not sana_rms._DISABLED_SUM_SIGNATURES


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize("case", ["fp16", "mixed_weight", "noncontiguous", "grad_enabled"])
def test_unsupported_sum_inputs_stay_eager(case: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if case == "fp16":
        hidden_states = torch.randn((1, 1024, 2240), device="cuda", dtype=torch.float16)
        weight = torch.randn(2240, device="cuda", dtype=torch.float16)
    elif case == "mixed_weight":
        hidden_states = torch.randn((1, 1024, 2240), device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(2240, device="cuda", dtype=torch.float32)
    elif case == "noncontiguous":
        hidden_states = torch.randn((1, 2240, 1024), device="cuda", dtype=torch.bfloat16).transpose(1, 2)
        weight = torch.randn(2240, device="cuda", dtype=torch.bfloat16)
    else:
        hidden_states = torch.randn((1, 1024, 2240), device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(2240, device="cuda", dtype=torch.bfloat16)

    expected = _sum_count_reference(hidden_states, weight, 1e-5)
    monkeypatch.setattr(
        sana_rms,
        "_launch_exact_sana_rms_norm_sum",
        lambda *_args: pytest.fail("unsupported inputs must not launch the sum/count fast path"),
    )
    if case == "grad_enabled":
        result = sana_rms.exact_sana_rms_norm_sum(hidden_states, weight, 1e-5)
    else:
        with torch.no_grad():
            result = sana_rms.exact_sana_rms_norm_sum(hidden_states, weight, 1e-5)

    if result.dtype is torch.bfloat16:
        assert _raw_bf16_equal(result, expected)
    else:
        assert torch.equal(result, expected)
    assert not sana_rms._VERIFIED_SUM_SIGNATURES
    assert not sana_rms._DISABLED_SUM_SIGNATURES


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_unverified_sum_signature_uses_eager_during_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    hidden_states = torch.randn((1, 1024, 2240), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2240, device="cuda", dtype=torch.bfloat16)
    expected = _sum_count_reference(hidden_states, weight, 1e-5)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        sana_rms,
        "_launch_exact_sana_rms_norm_sum",
        lambda *_args: pytest.fail("an unverified signature must not launch during capture"),
    )

    with torch.no_grad():
        result = sana_rms.exact_sana_rms_norm_sum(hidden_states, weight, 1e-5)

    assert _raw_bf16_equal(result, expected)
    assert not sana_rms._VERIFIED_SUM_SIGNATURES
    assert not sana_rms._DISABLED_SUM_SIGNATURES


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize("verified", [False, True])
def test_sum_oom_preserves_signature_and_allows_retry(
    verified: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hidden_states = torch.randn((1, 1024, 2240), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2240, device="cuda", dtype=torch.bfloat16)
    expected = _sum_count_reference(hidden_states, weight, 1e-5)
    signature = sana_rms._signature(hidden_states)
    if verified:
        with torch.no_grad():
            sana_rms.exact_sana_rms_norm_sum(hidden_states, weight, 1e-5)
        assert signature in sana_rms._VERIFIED_SUM_SIGNATURES

    original_launch = sana_rms._launch_exact_sana_rms_norm_sum
    oom = torch.OutOfMemoryError("synthetic temporary memory pressure")
    calls = 0

    def fail_once(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise oom
        return original_launch(*args)

    monkeypatch.setattr(sana_rms, "_launch_exact_sana_rms_norm_sum", fail_once)
    with monkeypatch.context() as patch:
        patch.setattr(
            sana_rms,
            "_eager_sana_rms_norm_sum",
            lambda *_args: pytest.fail("OOM must propagate without an eager fallback"),
        )
        with torch.no_grad(), pytest.raises(torch.OutOfMemoryError) as exc_info:
            sana_rms.exact_sana_rms_norm_sum(hidden_states, weight, 1e-5)

    assert exc_info.value is oom
    assert (signature in sana_rms._VERIFIED_SUM_SIGNATURES) is verified
    assert signature not in sana_rms._DISABLED_SUM_SIGNATURES

    with torch.no_grad():
        result = sana_rms.exact_sana_rms_norm_sum(hidden_states, weight, 1e-5)

    assert calls == 2
    assert _raw_bf16_equal(result, expected)
    assert signature in sana_rms._VERIFIED_SUM_SIGNATURES
    assert signature not in sana_rms._DISABLED_SUM_SIGNATURES


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_mean_and_sum_verification_caches_are_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A verified mean reduction must not bless the distinct sum/count path."""
    hidden_states = torch.randn((1, 1024, 2240), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2240, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        sana_rms.exact_sana_rms_norm(hidden_states, weight, 1e-5)

    signature = sana_rms._signature(hidden_states)
    assert signature in sana_rms._VERIFIED_SIGNATURES
    assert signature not in sana_rms._VERIFIED_SUM_SIGNATURES

    original_launch = sana_rms._launch_exact_sana_rms_norm_sum
    launches = 0

    def count_launch(*args):
        nonlocal launches
        launches += 1
        return original_launch(*args)

    monkeypatch.setattr(sana_rms, "_launch_exact_sana_rms_norm_sum", count_launch)
    with torch.no_grad():
        result = sana_rms.exact_sana_rms_norm_sum(hidden_states, weight, 1e-5)

    assert launches == 1
    assert _raw_bf16_equal(result, _sum_count_reference(hidden_states, weight, 1e-5))
    assert signature in sana_rms._VERIFIED_SUM_SIGNATURES
    assert signature in sana_rms._VERIFIED_SIGNATURES


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_verified_sum_signature_supports_cuda_graph_capture_and_replay() -> None:
    torch.manual_seed(103)
    static_input = torch.randn((1, 1024, 2240), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2240, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        sana_rms.exact_sana_rms_norm_sum(static_input, weight, 1e-5)
    assert sana_rms._signature(static_input) in sana_rms._VERIFIED_SUM_SIGNATURES
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(graph):
        result = sana_rms.exact_sana_rms_norm_sum(static_input, weight, 1e-5)

    replacement = torch.randn_like(static_input)
    static_input.copy_(replacement)
    graph.replay()
    torch.accelerator.synchronize()

    expected = _sum_count_reference(replacement, weight, 1e-5)
    assert _raw_bf16_equal(result, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sum_helper_compile_guard_preserves_original_expression(monkeypatch: pytest.MonkeyPatch) -> None:
    class SumRMSNorm(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(2240, device="cuda", dtype=torch.bfloat16))

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return sana_rms.exact_sana_rms_norm_sum(hidden_states, self.weight, 1e-5)

    hidden_states = torch.randn((1, 1024, 2240), device="cuda", dtype=torch.bfloat16)
    module = SumRMSNorm().eval()
    expected = _sum_count_reference(hidden_states, module.weight, 1e-5)
    monkeypatch.setattr(
        sana_rms,
        "_launch_exact_sana_rms_norm_sum",
        lambda *_args: pytest.fail("compiled regions must retain the eager sum/count expression"),
    )
    compiled = torch.compile(module, backend="eager", fullgraph=True)

    with torch.no_grad():
        result = compiled(hidden_states)

    assert _raw_bf16_equal(result, expected)


_ROUTING_CONFIG = {
    "in_channels": 4,
    "out_channels": 4,
    "num_attention_heads": 2,
    "attention_head_dim": 112,
    "num_layers": 1,
    "num_cross_attention_heads": 2,
    "cross_attention_head_dim": 112,
    "cross_attention_dim": 224,
    "caption_channels": 8,
    "mlp_ratio": 2.0,
    "patch_size": (1, 2, 2),
    "sample_size": 128,
    "rope_max_seq_len": 128,
}


def _pre_fast_path_distributed_forward(
    self: sana_transformer.SanaDistributedRMSNorm,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Frozen copy of the TP-aware implementation before video fusion."""
    tp_size = sana_transformer.get_tensor_model_parallel_world_size()
    hidden_states_float = hidden_states.float()
    sum_sq = hidden_states_float.pow(2).sum(dim=-1, keepdim=True)
    count = hidden_states.shape[-1]
    if tp_size > 1:
        sum_sq = sana_transformer.tensor_model_parallel_all_reduce(sum_sq)
        count *= tp_size
    normalized = hidden_states_float * torch.rsqrt(sum_sq / count + self.eps)
    if self.weight.dtype in (torch.float16, torch.bfloat16):
        normalized = normalized.to(self.weight.dtype)
    return normalized * self.weight


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_transformer_video_norms_really_launch_sum_fast_path(
    tp1_cuda_group: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the real one-block model and prove all three video norms fuse.

    B=2, F=2, H=W=128 produces 16,384 video-token rows after patching.
    At width 224 each video Q/K tensor contains 3,670,016 elements, above the
    production threshold without modifying it.  Caption and cross-attention K
    remain small and therefore must not appear in the launch log.
    """
    from vllm_omni.diffusion.models.sana_video import SanaVideoTransformer3DModel

    torch.manual_seed(107)
    model = SanaVideoTransformer3DModel(**_ROUTING_CONFIG).eval()
    for _, parameter in sorted(model.named_parameters()):
        torch.nn.init.normal_(parameter, mean=0.0, std=0.02)
        parameter.requires_grad_(False)
    model = model.to(device="cuda", dtype=torch.bfloat16)

    hidden_states = torch.randn((2, 4, 2, 128, 128), device="cuda", dtype=torch.bfloat16)
    encoder_hidden_states = torch.randn((2, 16, 8), device="cuda", dtype=torch.bfloat16)
    encoder_attention_mask = torch.ones((2, 16), device="cuda", dtype=torch.int64)
    timestep = torch.tensor([500.0, 700.0], device="cuda")

    launch_shapes: list[tuple[int, ...]] = []
    mean_launch_shapes: list[tuple[int, ...]] = []
    original_launch = sana_rms._launch_exact_sana_rms_norm_sum
    original_mean_launch = sana_rms._launch_exact_sana_rms_norm

    def count_real_launch(*args):
        launch_shapes.append(tuple(args[0].shape))
        return original_launch(*args)

    def count_mean_launch(*args):
        mean_launch_shapes.append(tuple(args[0].shape))
        return original_mean_launch(*args)

    monkeypatch.setattr(sana_rms, "_launch_exact_sana_rms_norm_sum", count_real_launch)
    monkeypatch.setattr(sana_rms, "_launch_exact_sana_rms_norm", count_mean_launch)

    block = model.transformer_blocks[0]
    norms = {
        "self_q": block.attn1.norm_q,
        "self_k": block.attn1.norm_k,
        "cross_q": block.attn2.norm_q,
        "cross_k": block.attn2.norm_k,
    }
    calls: Counter[str] = Counter()
    observed_shapes: dict[str, tuple[int, ...]] = {}

    def make_hook(name: str):
        def check_output(module, args, output):
            norm_input = args[0]
            expected = _sum_count_reference(norm_input, module.weight, module.eps)
            assert _raw_bf16_equal(output, expected), name
            calls[name] += 1
            observed_shapes[name] = tuple(norm_input.shape)

        return check_output

    handles = [module.register_forward_hook(make_hook(name)) for name, module in norms.items()]
    try:
        with torch.no_grad():
            first = model(
                hidden_states,
                encoder_hidden_states,
                timestep,
                encoder_attention_mask=encoder_attention_mask,
            ).sample
            second = model(
                hidden_states,
                encoder_hidden_states,
                timestep,
                encoder_attention_mask=encoder_attention_mask,
            ).sample
    finally:
        for handle in handles:
            handle.remove()

    video_shape = (2, 2 * 64 * 64, 224)
    assert observed_shapes == {
        "self_q": video_shape,
        "self_k": video_shape,
        "cross_q": video_shape,
        "cross_k": (2, 16, 224),
    }
    assert calls == Counter({"self_q": 2, "self_k": 2, "cross_q": 2, "cross_k": 2})
    assert launch_shapes == [video_shape] * 6
    assert mean_launch_shapes == []
    assert _raw_bf16_equal(first, second)

    monkeypatch.setattr(
        sana_transformer.SanaDistributedRMSNorm,
        "forward",
        _pre_fast_path_distributed_forward,
    )
    with torch.no_grad():
        reference = model(
            hidden_states,
            encoder_hidden_states,
            timestep,
            encoder_attention_mask=encoder_attention_mask,
        ).sample

    assert _raw_bf16_equal(first, reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_tp2_keeps_collective_and_global_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """An otherwise eligible local shard must not bypass TP's global RMS."""
    local_width = 112
    hidden_states = torch.randn((1, 20_000, local_width), device="cuda", dtype=torch.bfloat16)
    norm = sana_transformer.SanaDistributedRMSNorm(local_width, eps=1e-5).to(device="cuda", dtype=torch.bfloat16)
    norm.weight.data.normal_()

    local_sum = hidden_states.float().pow(2).sum(dim=-1, keepdim=True)
    remote_sum = torch.linspace(0.25, 3.0, hidden_states.shape[1], device="cuda").view(1, -1, 1)
    global_sum = local_sum + remote_sum
    collective_inputs: list[torch.Tensor] = []

    def fake_all_reduce(value: torch.Tensor) -> torch.Tensor:
        collective_inputs.append(value)
        return global_sum

    def unexpected_fast_path(*_args):
        pytest.fail("TP>1 must keep the collective sum/count implementation")

    monkeypatch.setattr(sana_transformer, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(sana_transformer, "tensor_model_parallel_all_reduce", fake_all_reduce)
    monkeypatch.setattr(sana_transformer, "exact_sana_rms_norm_sum", unexpected_fast_path)
    monkeypatch.setattr(sana_rms, "exact_sana_rms_norm_sum", unexpected_fast_path)

    normalized = hidden_states.float() * torch.rsqrt(global_sum / (2 * local_width) + norm.eps)
    expected = normalized.to(torch.bfloat16) * norm.weight
    with torch.no_grad():
        result = norm(hidden_states)

    assert len(collective_inputs) == 1
    assert torch.equal(collective_inputs[0], local_sum)
    assert _raw_bf16_equal(result, expected)
