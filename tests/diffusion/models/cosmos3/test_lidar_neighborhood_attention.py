# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Independent local-window oracle and sparse metadata/compilation contracts."""

from __future__ import annotations

import math
import weakref

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm_omni.diffusion.models.cosmos3.lidar_encoder import neighborhood_attention as attention
from vllm_omni.diffusion.models.cosmos3.lidar_encoder.transformer_vae import CircularNeighborhoodSelfAttentionBlock

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def axis_neighbors(q, length, kernel, dilation):
    # Construct the residue sequence and move a sliding window in that list.
    # This deliberately does not call the production start/predicate helpers.
    group = list(range(q % dilation, length, dilation))
    center = group.index(q)
    window = group[max(0, center - kernel // 2) :][:kernel]
    return group[-kernel:] if len(window) < kernel else window


def neighbor_indices(height, width, kernel, dilation):
    return [
        [
            h * width + w
            for h in axis_neighbors(qh, height, kernel[0], dilation[0])
            for w in axis_neighbors(qw, width, kernel[1], dilation[1])
        ]
        for qh in range(height)
        for qw in range(width)
    ]


def reference_attention(query, key, value, *, kernel_size, dilation, scale):
    batch, height, width, heads, dim = query.shape
    q, k, v = (t.double().reshape(batch, height * width, heads, dim) for t in (query, key, value))
    indices = torch.tensor(neighbor_indices(height, width, kernel_size, dilation), device=query.device)
    scores = torch.einsum("bshd,bskhd->bshk", q, k[:, indices]) * scale
    return torch.einsum("bshk,bskhd->bshd", scores.softmax(-1), v[:, indices]).reshape(query.shape)


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))],
)
@pytest.mark.parametrize("circular", [False, True])
@pytest.mark.parametrize(
    "hw,kernel,dilation",
    [
        ((1, 1), (1, 1), (1, 1)),
        ((5, 13), (3, 5), (1, 1)),
        ((7, 17), (3, 5), (2, 3)),
        ((9, 11), (1, 3), (4, 2)),
        ((8, 13), (4, 2), (2, 3)),
    ],
)
def test_operator_matches_fp64_neighborhoods(hw, kernel, dilation, circular, device):
    generator = torch.Generator(device=device).manual_seed(721)
    tensors = [torch.randn(2, *hw, 2, 8, device=device, generator=generator) for _ in range(3)]
    block = CircularNeighborhoodSelfAttentionBlock(16, 2, kernel, dilation, circular=circular)
    q, k, v = block.before_attn(*tensors)
    precision = torch.get_float32_matmul_precision()
    rng = torch.random.get_rng_state()
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        result = attention.neighborhood_attention_2d(q, k, v, kernel_size=kernel, dilation=dilation, scale=1.0)
    assert result.is_contiguous()
    expected = reference_attention(q, k, v, kernel_size=kernel, dilation=dilation, scale=1.0)
    result, expected = block.after_attn(result), block.after_attn(expected)
    assert result.shape == tensors[0].shape and result.dtype == torch.float32
    assert torch.get_float32_matmul_precision() == precision
    assert torch.equal(torch.random.get_rng_state(), rng)
    torch.testing.assert_close(result.double(), expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "kernel,dilation", [((3, 3), (2, 1)), ((1, 7), (1, 1)), ((0, 1), (1, 1)), ((1, 1), (0, 1)), (True, 1)]
)
def test_invalid_geometry_is_rejected(kernel, dilation):
    q = torch.zeros(1, 5, 6, 1, 4)
    with pytest.raises(ValueError, match="LiDAR attention"):
        attention.neighborhood_attention_2d(q, q, q, kernel_size=kernel, dilation=dilation, scale=1.0)


@pytest.mark.parametrize("hw", [(7, 17), (65, 67)])
def test_block_map_and_partial_predicate_cover_exact_neighborhoods(hw):
    kernel, dilation = (3, 5), (2, 3)
    mask = attention._get_block_mask(torch.device("cpu"), *hw, kernel, dilation)
    sequence = math.prod(hw)
    blocks = math.ceil(sequence / 64)
    assert mask.seq_lengths == (sequence, sequence) and mask.BLOCK_SIZE == (64, 64)
    assert mask.kv_indices.shape == (1, 1, blocks, blocks)
    assert mask.kv_indices.dtype == torch.int32 and mask.kv_indices.is_contiguous()
    assert mask.q_num_blocks is mask.q_indices is mask.full_q_indices is None
    assert mask.full_kv_indices is None
    expected = [set() for _ in range(blocks)]
    neighbors = neighbor_indices(*hw, kernel, dilation)
    for q, keys in enumerate(neighbors):
        expected[q // 64].update(p // 64 for p in keys)
    for q_block in range(blocks):
        count = mask.kv_num_blocks[0, 0, q_block]
        assert mask.kv_indices[0, 0, q_block, :count].tolist() == sorted(expected[q_block])
    # All keys for boundary queries, including padded lanes in partial blocks.
    keys = torch.arange(blocks * 64)
    for q in [0, hw[1] - 1, sequence // 2, sequence - 1, sequence]:
        visible = mask.mask_mod(0, 0, torch.tensor(q), keys).nonzero().flatten().tolist()
        assert visible == (neighbors[q] if q < sequence else [])


def test_production_mask_has_bounded_workspace_and_no_token_square():
    sequence = 64 * 908
    largest = 0

    class Allocations(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            nonlocal largest
            result = func(*args, **(kwargs or {}))
            for tensor in torch.utils._pytree.tree_leaves(result):
                if isinstance(tensor, torch.Tensor):
                    largest = max(largest, tensor.numel())
                    assert tensor.numel() < sequence * sequence
            return result

    attention._get_block_mask.cache_clear()
    with Allocations():
        mask = attention._get_block_mask(torch.device("cpu"), 64, 908, (3, 9), (1, 2))
    assert largest <= max(math.ceil(sequence / 64) ** 2, 4096 * 3 * 9)
    assert mask.kv_num_blocks.max() < math.ceil(sequence / 64) // 4


def test_geometry_lru_reuse_eviction_and_no_request_tensor_retention():
    attention._get_block_mask.cache_clear()
    q = torch.randn(1, 3, 7, 1, 4)
    tensor_ref = weakref.ref(q)
    attention.neighborhood_attention_2d(q, q, q, kernel_size=(3, 3), dilation=(1, 1), scale=1.0)
    first = attention._get_block_mask(torch.device("cpu"), 3, 7, (3, 3), (1, 1))
    larger_batch = q.repeat(3, 1, 1, 2, 1)
    attention.neighborhood_attention_2d(larger_batch, larger_batch, larger_batch, kernel_size=3, dilation=1, scale=1.0)
    assert attention._get_block_mask.cache_info().misses == 1
    del q
    assert tensor_ref() is None
    for width in range(8, 41):
        attention._get_block_mask(torch.device("cpu"), 3, width, (3, 3), (1, 1))
    assert attention._get_block_mask.cache_info().currsize == 32
    assert attention._get_block_mask(torch.device("cpu"), 3, 7, (3, 3), (1, 1)) is not first


def test_compiled_specializations_have_isolated_code_and_bounded_cache(monkeypatch):
    attention._get_compiled_runner.cache_clear()
    codes = []

    def compile_runner(function, **kwargs):
        assert kwargs == {"fullgraph": True, "dynamic": False}
        assert function.__closure__ is None
        codes.append(function.__code__)
        return function

    monkeypatch.setattr(torch, "compile", compile_runner)
    for batch in range(1, 35):
        key = (torch.device("cuda", 0), (batch, 7, 17, 2, 8), (3, 5), (2, 3), 1.0)
        runner = attention._get_compiled_runner(*key)
        assert attention._get_compiled_runner(*key) is runner
    assert len({id(code) for code in codes}) == 34
    assert attention._get_compiled_runner.cache_info().currsize == 32
    attention._get_compiled_runner.cache_clear()


def test_fullgraph_specializations_execute_beyond_dynamo_recompile_limit(monkeypatch):
    attention._get_compiled_runner.cache_clear()
    compile_operator = torch.compile
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    # Exercise Dynamo's real graph/cache behavior on CPU, bypassing only CUDA
    # kernel generation. Each distinct shape must get its own full graph.
    monkeypatch.setattr(torch, "compile", lambda function, **kw: compile_operator(function, backend=backend, **kw))
    mask = attention._get_block_mask(torch.device("cpu"), 3, 5, (3, 3), (1, 1))
    with torch.inference_mode(), torch._dynamo.config.patch(recompile_limit=2):
        for batch in range(1, 6):
            q = torch.ones(batch, 1, 15, 8)
            runner = attention._get_compiled_runner(q.device, (batch, 3, 5, 1, 8), (3, 3), (1, 1), 1.0)
            torch.testing.assert_close(runner(q, q, q, mask, 1.0), q)
    assert len(graphs) == 5
    attention._get_compiled_runner.cache_clear()


def test_fp32_layout_scale_and_local_kernel_precision(monkeypatch):
    def flex(query, key, value, *, block_mask, scale, kernel_options):
        assert all(t.shape == (2, 3, 35, 8) and t.is_contiguous() for t in (query, key, value))
        assert all(t.dtype == torch.float32 for t in (query, key, value))
        assert not torch.is_autocast_enabled("cpu") and scale == 1.0
        assert kernel_options["FLOAT32_PRECISION"] == "'ieee'"
        assert (kernel_options["BLOCK_M"], kernel_options["BLOCK_N"]) == (64, 64)
        assert (kernel_options["num_warps"], kernel_options["num_stages"]) == (4, 1)
        assert kernel_options["USE_TMA"] is False
        return value

    monkeypatch.setattr(attention, "flex_attention", flex)
    q = torch.randn(2, 5, 7, 3, 8)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = attention.neighborhood_attention_2d(q, q, q, kernel_size=3, dilation=1, scale=1.0)
    torch.testing.assert_close(actual, q, rtol=0, atol=0)


def test_real_encoder_local_attention_matches_independent_reference(monkeypatch):
    from vllm_omni.diffusion.models.cosmos3.lidar_encoder import transformer_vae
    from vllm_omni.diffusion.models.cosmos3.lidar_encoder.encoding import generate_polar_coords

    with torch.random.fork_rng():
        torch.manual_seed(721)
        model = (
            transformer_vae.Encoder(
                resolution=[32, 48],
                patch_size=[2, 2],
                in_channels=3,
                z_dim=2,
                base_channels=8,
                depths=[2, 2, 1, 1],
                num_heads=[1, 1, 2, 2],
                dilation=[2, 2, 1, 1],
                window_size=[3, 3],
                temporal_downsample=[False, False, False],
            )
            .float()
            .eval()
        )
        for name, parameter in model.named_parameters():
            if name.endswith("out_proj.weight"):
                torch.nn.init.normal_(parameter, std=0.1)
        pixels = torch.randn(1, 3, 4, 32, 48)
    coords = generate_polar_coords(32, 48)

    def stream():
        outputs, cache = [], None
        for first in range(0, 4, 3):
            output, cache = model.forward_stream(pixels[:, :, first : first + 3], coords, cache)
            outputs.append(output)
        return torch.cat(outputs, 2)

    with torch.inference_mode():
        actual = stream()
        monkeypatch.setattr(
            transformer_vae, "neighborhood_attention_2d", lambda *args, **kw: reference_attention(*args, **kw).float()
        )
        expected = stream()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_cpu_production_grid_rejected_before_eager_allocation():
    q = torch.empty(1, 64, 904, 1, 4)
    with pytest.raises(RuntimeError, match="use CUDA"):
        attention.neighborhood_attention_2d(q, q, q, kernel_size=3, dilation=1, scale=1.0)


def test_validation_measurements_exclude_capture_and_release_components(monkeypatch):
    from tools import validate_cosmos3_lidar_decoder as validation

    for name in ("synchronize", "reset_peak_memory_stats", "empty_cache"):
        monkeypatch.setattr(torch.accelerator, name, lambda: None)
    monkeypatch.setattr(torch.accelerator, "max_memory_allocated", lambda: 128)
    monkeypatch.setattr(torch.accelerator, "memory_allocated", lambda: 64)
    monkeypatch.setattr(torch.cuda, "get_rng_state", torch.random.get_rng_state)
    placements, calls = [], []

    class Component(torch.nn.Module):
        def to(self, device):
            placements.append(device)
            return self

        def cpu(self):
            placements.append("cpu")
            return self

        def forward(self):
            calls.append(bool(self._forward_hooks))
            return torch.ones(1, 3, 2, 2, 2), {}

    component = Component()
    actual, raw, timings = validation.evaluate(
        [component], lambda: component()[0], raw_module=component, preserve_rng=True
    )
    assert calls == [False] * 14 + [True]  # First call + 3 warm-ups + 10 timed runs, then capture.
    assert placements == ["cuda", "cpu"] and not component._forward_hooks
    assert timings["warm_peak_increment_bytes"] == 64
    assert len(timings["warm_seconds"]) == 10
    torch.testing.assert_close(actual, raw, rtol=0, atol=0)
    calls.clear()
    with pytest.raises(AssertionError, match="mismatch"):
        validation.evaluate([component], lambda: (_ for _ in ()).throw(AssertionError("mismatch")))
    assert placements[-1] == "cpu"


def test_validation_keeps_strict_numeric_and_binary_thresholds():
    from tools.validate_cosmos3_lidar_decoder import compare

    compare(torch.zeros(1), torch.full((1,), 9e-5))
    with pytest.raises(AssertionError):
        compare(torch.zeros(1), torch.full((1,), 2e-4))
    with pytest.raises(AssertionError):
        compare(torch.zeros(1), torch.full((1,), 1e-8), exact=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_compile_failure_propagates_without_fallback(monkeypatch):
    def fail(*args):
        raise RuntimeError("injected compilation failure")

    monkeypatch.setattr(attention, "_get_compiled_runner", fail)
    q = torch.ones(1, 3, 7, 1, 8, device="cuda")
    with pytest.raises(RuntimeError, match="compiled sparse attention is required") as error:
        attention.neighborhood_attention_2d(q, q, q, kernel_size=3, dilation=1, scale=1.0)
    assert "injected compilation failure" in str(error.value.__cause__)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.accelerator.device_count() < 2, reason="Two CUDA devices required"
)
def test_masks_follow_device_after_offload_reload():
    masks = []
    for index in [0, 1, 0]:
        q = torch.ones(1, 3, 7, 1, 8).to(torch.device("cuda", index))
        attention.neighborhood_attention_2d(q, q, q, kernel_size=3, dilation=1, scale=1.0)
        mask = attention._get_block_mask(q.device, 3, 7, (3, 3), (1, 1))
        assert mask.kv_indices.device == q.device
        masks.append(mask)
    assert masks[0] is masks[2] and masks[0] is not masks[1]
