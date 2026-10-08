# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import subprocess
import sys

import pytest
import torch

from vllm_omni.diffusion.models.ltx2.ops import qk_rms_norm as qk_rms_norm_ops
from vllm_omni.diffusion.models.ltx2.ops import residual_adaln as residual_adaln_ops
from vllm_omni.diffusion.models.ltx2.ops import resolve_ltx2_vae_operators
from vllm_omni.diffusion.models.ltx2.ops import swiglu as swiglu_ops
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cuda, pytest.mark.diffusion]

_DIM_SPLIT = (16, 24, 24)


def _selected_operators():
    if not current_omni_platform.is_available():
        pytest.skip("No accelerator is available")
    operators = resolve_ltx2_vae_operators(current_omni_platform.get_torch_device())
    if operators is None:
        pytest.skip("No LTX DiffVAE operators are available for this device")
    return operators


def _selected_fna():
    operators = _selected_operators()
    if operators.fna is None:
        pytest.skip("LTX DiffVAE FNA is unavailable for this device or installation")
    return operators.fna


def _load_fna_ops():
    return pytest.importorskip("vllm_omni.diffusion.models.ltx2.ops.fna", exc_type=ImportError)


def test_diffvae_model_and_tests_do_not_import_tilelang_backend() -> None:
    # Use a fresh interpreter: reloading the model in this process would leave
    # already-imported distributed subclasses pointing at the old base class.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "sys.modules['vllm_omni.diffusion.models.ltx2.ops.fna'] = None; "
            "sys.modules['vllm.tilelang_utils'] = None; "
            "from vllm_omni.diffusion.models.ltx2.vae.decoder import LTX2VideoDiffusionDecoderModel; "
            "import tests.diffusion.models.ltx2.test_ltx2_diffvae_attention; "
            "import tests.diffusion.models.ltx2.test_ltx2_diffvae_ops",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_diffvae_fna_tile_map_covers_exact_neighborhood() -> None:
    fna_ops = _load_fna_ops()
    problem = fna_ops.Problem(13, 14, 17, (11, 11, 11))
    indices, counts = fna_ops.build_kv_tile_map(problem)
    q_grid_t, q_grid_h, _ = problem.q_grid
    kv_grid_t, kv_grid_h, _ = problem.kv_grid

    for q_block in range(indices.shape[0]):
        selected = set(indices[q_block, : counts[q_block]].tolist())
        q_outer_t = q_block % q_grid_t
        q_outer_h = (q_block // q_grid_t) % q_grid_h
        q_outer_w = q_block // (q_grid_t * q_grid_h)
        for q_inner in range(fna_ops.Q_TOKENS):
            query = (
                q_outer_t * 4 + q_inner % 4,
                q_outer_h * 4 + (q_inner // 4) % 4,
                q_outer_w * 4 + q_inner // 16,
            )
            if any(index >= length for index, length in zip(query, problem.shape, strict=True)):
                continue
            starts = tuple(
                fna_ops._window_start(index, window, length)
                for index, window, length in zip(query, problem.window, problem.shape, strict=True)
            )
            for key_t in range(starts[0], starts[0] + problem.window[0]):
                for key_h in range(starts[1], starts[1] + problem.window[1]):
                    for key_w in range(starts[2], starts[2] + problem.window[2]):
                        key_block = ((key_w // 8) * kv_grid_h + key_h // 4) * kv_grid_t + key_t // 4
                        assert key_block in selected


def test_diffvae_fna_metadata_uses_static_kernel_shapes() -> None:
    fna_ops = _load_fna_ops()
    for shape in ((11, 11, 11), (13, 14, 17), (66, 96, 128), (79, 192, 192)):
        problem = fna_ops.Problem(*shape, (11, 11, 11))
        _, t_patterns, h_patterns, w_patterns = fna_ops.build_kv_tile_metadata(problem)
        assert {table.numel() for table in (t_patterns, h_patterns, w_patterns)} == {fna_ops.PATTERN_TABLE_SIZE}
        assert {limit for _, _, limit in fna_ops.build_kv_tile_buckets(problem)} <= set(fna_ops.KV_BUCKET_LIMITS)
        for ids, metadata, limit in fna_ops.build_kv_tile_buckets(problem):
            assert metadata.shape == (ids.numel(), limit)


def test_diffvae_fna_rejects_oversized_packed_tile_grid() -> None:
    fna_ops = _load_fna_ops()
    with pytest.raises(ValueError, match="15-bit metadata"):
        fna_ops.Problem(128, 256, 256, (11, 11, 11))


def test_diffvae_fna_propagates_launch_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fna_ops = _load_fna_ops()
    fna = _selected_fna()
    problem = fna_ops.Problem(11, 11, 11, (11, 11, 11))
    query = torch.randn(1, problem.q_length, fna_ops.HEADS, 64, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(1, problem.kv_length, fna_ops.HEADS, 64, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    calls = 0

    def fail_launch(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("injected launch failure")

    monkeypatch.setattr(fna_ops, "_fna3d_tilelang", fail_launch)
    with torch.inference_mode():
        for _ in range(2):
            with pytest.raises(RuntimeError, match="injected launch failure"):
                fna(
                    query,
                    key,
                    value,
                    shape=problem.shape,
                    window=problem.window,
                )
    assert calls == 2


@pytest.fixture
def isolated_runtime_state():
    targets = (
        (qk_rms_norm_ops, "_FAILED_ROPE_KEYS"),
        (qk_rms_norm_ops, "_VERIFIED_ROPE_KEYS"),
        (swiglu_ops, "_FAILED_KEYS"),
        (swiglu_ops, "_VERIFIED_KEYS"),
        (residual_adaln_ops, "_FAILED_ADALN_KEYS"),
        (residual_adaln_ops, "_VERIFIED_ADALN_KEYS"),
        (residual_adaln_ops, "_FAILED_ADD_DEVICES"),
        (residual_adaln_ops, "_VERIFIED_ADD_DEVICES"),
    )
    saved = [set(getattr(module, name)) for module, name in targets]
    for module, name in targets:
        getattr(module, name).clear()
    try:
        yield
    finally:
        for (module, name), state in zip(targets, saved, strict=True):
            target = getattr(module, name)
            target.clear()
            target.update(state)


@pytest.mark.parametrize(
    ("dim", "rows"),
    [
        (256, 257),
        (256, 16385),
        (256, 131075),
        (512, 65539),
        (1024, 32771),
        (2048, 16387),
    ],
)
def test_diffvae_swiglu_tiled_is_bit_exact(dim: int, rows: int, isolated_runtime_state) -> None:
    from diffusers.models.autoencoders.ltx2_diffusion_decoder import LTX2VideoVaeSwiGLU

    operators = _selected_operators()
    generator = torch.Generator(device="cuda").manual_seed(20260829)
    hidden_states = torch.randn((1, 1, 1, rows, dim), device="cuda", dtype=torch.bfloat16, generator=generator)
    gate_weight = torch.randn((4 * dim, dim), device="cuda", dtype=torch.bfloat16, generator=generator)
    up_weight = torch.randn_like(gate_weight)
    down_weight = torch.randn((dim, 4 * dim), device="cuda", dtype=torch.bfloat16, generator=generator)
    with torch.device("meta"):
        reference = LTX2VideoVaeSwiGLU(dim, 4 * dim)
    reference.w_gate.weight = torch.nn.Parameter(gate_weight, requires_grad=False)
    reference.w_up.weight = torch.nn.Parameter(up_weight, requires_grad=False)
    reference.w_down.weight = torch.nn.Parameter(down_weight, requires_grad=False)

    with torch.inference_mode():
        expected = reference(hidden_states)
        actual = operators.swiglu(hidden_states, gate_weight, up_weight, down_weight)

    assert actual is not None
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("dim", "expected_rows"),
    [(256, 131072), (512, 65536), (1024, 32768), (2048, 16384)],
)
def test_diffvae_swiglu_tile_rows_keep_a_constant_hidden_budget(dim: int, expected_rows: int) -> None:
    assert swiglu_ops._tile_rows(1_000_000, 4 * dim) == expected_rows


def test_diffvae_swiglu_tiled_permanently_falls_back_after_failure(
    monkeypatch: pytest.MonkeyPatch,
    isolated_runtime_state,
) -> None:
    operators = _selected_operators()
    hidden_states = torch.randn(1, 1, 1, 17, 256, device="cuda", dtype=torch.bfloat16)
    gate_weight = torch.randn(1024, 256, device="cuda", dtype=torch.bfloat16)
    up_weight = torch.randn_like(gate_weight)
    down_weight = torch.randn(256, 1024, device="cuda", dtype=torch.bfloat16)
    calls = 0

    def fail_launch(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("injected launch failure")

    monkeypatch.setattr(swiglu_ops, "_launch", fail_launch)
    with torch.inference_mode():
        assert operators.swiglu(hidden_states, gate_weight, up_weight, down_weight) is None
        assert operators.swiglu(hidden_states, gate_weight, up_weight, down_weight) is None

    assert calls == 1


@pytest.mark.parametrize("residual_count", [1, 2])
def test_diffvae_residual_rms_norm_modulate_is_bit_exact(
    residual_count: int,
    isolated_runtime_state,
) -> None:
    operators = _selected_operators()
    generator = torch.Generator(device="cuda").manual_seed(20260829)
    shape = (2, 3, 7, 7, 256)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    residual_a = torch.randn_like(x)
    residual_b = torch.randn_like(x) if residual_count == 2 else None
    norm_weight = torch.randn((256,), device="cuda", dtype=torch.bfloat16, generator=generator)
    scale = torch.randn((2, 1, 1, 1, 256), device="cuda", dtype=torch.bfloat16, generator=generator)
    shift = torch.randn_like(scale)
    hidden_states = x + residual_a
    if residual_b is not None:
        hidden_states = hidden_states + residual_b
    expected = torch.nn.functional.rms_norm(hidden_states, (256,), norm_weight, eps=1e-6) * (1 + scale) + shift

    with torch.inference_mode():
        actual = operators.residual_norm(
            x,
            residual_a,
            residual_b,
            norm_weight,
            scale,
            shift,
            1e-6,
        )

    assert actual is not None
    assert torch.equal(actual, expected)


def test_diffvae_residual_add3_is_bit_exact(isolated_runtime_state) -> None:
    operators = _selected_operators()
    generator = torch.Generator(device="cuda").manual_seed(20260829)
    tensors = [
        torch.randn((2, 3, 7, 7, 256), device="cuda", dtype=torch.bfloat16, generator=generator) for _ in range(4)
    ]
    expected = tensors[0] + tensors[1]
    expected = expected + tensors[2]
    expected = expected + tensors[3]

    with torch.inference_mode():
        actual = operators.residual_add(*tensors)

    assert actual is not None
    assert torch.equal(actual, expected)


def test_diffvae_residual_adaln_permanently_falls_back_after_failure(
    monkeypatch: pytest.MonkeyPatch,
    isolated_runtime_state,
) -> None:
    operators = _selected_operators()
    shape = (1, 3, 7, 7, 256)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(256, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(1, 1, 1, 1, 256, device="cuda", dtype=torch.bfloat16)
    shift = torch.randn_like(scale)
    calls = 0

    def fail_launch(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("injected launch failure")

    monkeypatch.setattr(residual_adaln_ops, "_launch_adaln", fail_launch)
    with torch.inference_mode():
        for _ in range(2):
            assert operators.residual_norm(x, residual, None, weight, scale, shift, 1e-6) is None

    assert calls == 1


def _tables(
    frames: int,
    height: int,
    width: int,
    dim_split: tuple[int, int, int] = _DIM_SPLIT,
    base: float = 10000.0,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    tables = []
    for length, dim in zip((frames, height, width), dim_split, strict=True):
        exponents = torch.arange(0, dim, 2, dtype=torch.float64, device="cuda") / dim
        inv_freqs = (1.0 / base**exponents).to(torch.float32)
        positions = torch.arange(length, dtype=torch.float32, device="cuda")
        angles = positions[:, None] * inv_freqs[None, :]
        tables.append((angles.cos(), angles.sin()))
    return tuple(tables)


def _apply_rope_reference(
    hidden_states: torch.Tensor,
    tables: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    dim_split: tuple[int, int, int] = _DIM_SPLIT,
) -> torch.Tensor:
    outputs = []
    offset = 0
    for axis, (dim, (cos, sin)) in enumerate(zip(dim_split, tables, strict=True), 1):
        chunk = hidden_states[..., offset : offset + dim]
        pairs = chunk.reshape(*chunk.shape[:-1], dim // 2, 2)
        even = pairs[..., 0].float()
        odd = pairs[..., 1].float()
        shape = [1, 1, 1, 1, 1, dim // 2]
        shape[axis] = cos.shape[0]
        cos = cos.reshape(shape)
        sin = sin.reshape(shape)
        rotated = torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1)
        outputs.append(rotated.reshape(chunk.shape).to(hidden_states.dtype))
        offset += dim
    return torch.cat(outputs, dim=-1)


@pytest.mark.parametrize(
    ("frames", "height", "width", "heads"),
    [(3, 7, 7, 1), (31, 136, 192, 4)],
)
def test_diffvae_qk_rms_norm_scale_rope_3d_is_bit_exact(
    frames: int,
    height: int,
    width: int,
    heads: int,
    isolated_runtime_state,
) -> None:
    operators = _selected_operators()
    generator = torch.Generator(device="cuda").manual_seed(20260829)
    shape = (1, frames, height, width, heads, 64)
    query = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    key = torch.randn_like(query)
    query_weight = torch.randn((64,), device="cuda", dtype=torch.bfloat16, generator=generator)
    key_weight = torch.randn((64,), device="cuda", dtype=torch.bfloat16, generator=generator)
    tables = _tables(frames, height, width)
    expected = (
        _apply_rope_reference(torch.nn.functional.rms_norm(query, (64,), query_weight, 1e-6) * 0.125, tables),
        _apply_rope_reference(torch.nn.functional.rms_norm(key, (64,), key_weight, 1e-6), tables),
    )

    with torch.inference_mode():
        actual = operators.qk_norm_rope(
            query,
            key,
            query_weight,
            key_weight,
            1e-6,
            0.125,
            _DIM_SPLIT,
            10000.0,
        )

    assert actual is not None
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_diffvae_qk_rms_norm_scale_rope_3d_permanently_falls_back_after_failure(
    monkeypatch: pytest.MonkeyPatch,
    isolated_runtime_state,
) -> None:
    operators = _selected_operators()
    query = torch.randn(1, 3, 7, 7, 1, 64, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    weight = torch.ones(64, device="cuda", dtype=torch.bfloat16)
    calls = 0

    def fail_launch(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("injected launch failure")

    monkeypatch.setattr(qk_rms_norm_ops, "_launch_combined", fail_launch)
    with torch.inference_mode():
        for _ in range(2):
            assert operators.qk_norm_rope(query, key, weight, weight, 1e-6, 0.125, _DIM_SPLIT, 10000.0) is None

    assert calls == 1
