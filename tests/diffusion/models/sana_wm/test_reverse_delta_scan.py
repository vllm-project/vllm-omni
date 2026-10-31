# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Exact SANA-WM reverse-scan parity without full-video flip/shift copies."""

import math
from typing import TypedDict

import pytest
import torch

import vllm_omni.diffusion.models.sana_wm.sana_wm_transformer as sana_wm

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
]

_CUDA_ONLY = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class _ScanKwargs(TypedDict):
    query_rot: torch.Tensor
    key_rot: torch.Tensor
    value: torch.Tensor
    beta: torch.Tensor
    decay: torch.Tensor
    spatial_tokens: int
    query: torch.Tensor | None
    key: torch.Tensor | None
    skip_z: bool


def _reverse_frames(
    tensor: torch.Tensor,
    *,
    frames: int,
    spatial_tokens: int,
    shift_value: float | None = None,
) -> torch.Tensor:
    """Materialize the baseline's reverse/optional one-frame shift."""
    lead = tensor.shape[:-1]
    reversed_ = torch.flip(tensor.reshape(*lead, frames, spatial_tokens), dims=[-2])
    if shift_value is not None:
        padding = torch.full(
            (*lead, 1, spatial_tokens),
            shift_value,
            device=tensor.device,
            dtype=tensor.dtype,
        )
        reversed_ = torch.cat([padding, reversed_.narrow(-2, 0, frames - 1)], dim=-2)
    return reversed_.reshape(*lead, frames * spatial_tokens)


def _flip_and_shift(tensor: torch.Tensor, *, dim: int, shift_value: float) -> torch.Tensor:
    """Materialize the baseline's reverse/one-frame shift for gates."""
    flipped = torch.flip(tensor, dims=[dim])
    shifted = flipped.narrow(dim, 0, tensor.shape[dim] - 1)
    pad_shape = list(tensor.shape)
    pad_shape[dim] = 1
    padding = torch.full(
        pad_shape,
        shift_value,
        device=tensor.device,
        dtype=tensor.dtype,
    )
    return torch.cat([padding, shifted], dim=dim)


def _forward_scan_oracle(
    query_rot: torch.Tensor,
    key_rot: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    *,
    spatial_tokens: int,
    query: torch.Tensor | None = None,
    key: torch.Tensor | None = None,
    skip_z: bool,
    flip_output: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Copy of the pre-H001 scan, kept only as an independent test oracle."""
    batch_size, num_heads, head_dim, token_count = query_rot.shape
    frames = beta.shape[2]

    def to_frames(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(
            batch_size,
            num_heads,
            head_dim,
            frames,
            spatial_tokens,
        ).permute(0, 1, 3, 2, 4)

    query_rot_f = to_frames(query_rot)
    key_rot_f = to_frames(key_rot)
    value_f = to_frames(value)
    state_kv = torch.zeros(
        batch_size,
        num_heads,
        head_dim,
        head_dim,
        device=query_rot.device,
        dtype=query_rot.dtype,
    )
    numerators: list[torch.Tensor] = []

    if skip_z:
        query_f = key_f = None
        state_z = None
        denominators: list[torch.Tensor] | None = None
    else:
        assert query is not None and key is not None
        query_f = to_frames(query)
        key_f = to_frames(key)
        state_z = torch.zeros(
            batch_size,
            num_heads,
            head_dim,
            1,
            device=query_rot.device,
            dtype=query_rot.dtype,
        )
        denominators = []

    for frame_idx in range(frames):
        query_rot_t = query_rot_f[:, :, frame_idx]
        key_rot_t = key_rot_f[:, :, frame_idx]
        value_t = value_f[:, :, frame_idx]
        beta_t = beta[:, :, frame_idx].unsqueeze(2)
        decay_t = decay[:, :, frame_idx].view(batch_size, num_heads, 1, 1)

        state_kv = state_kv * decay_t
        value_pred = torch.matmul(state_kv, key_rot_t)
        delta_value = (value_t - value_pred) * beta_t
        state_kv = state_kv + torch.matmul(
            delta_value,
            key_rot_t.transpose(-1, -2),
        )
        numerators.append(torch.matmul(state_kv, query_rot_t))

        if skip_z:
            continue
        assert query_f is not None and key_f is not None and state_z is not None
        assert denominators is not None
        query_t = query_f[:, :, frame_idx]
        key_t = key_f[:, :, frame_idx]
        state_z = state_z * decay_t
        z_pred = torch.matmul(state_z.transpose(-1, -2), key_t)
        delta_z = (1.0 - z_pred) * beta_t
        state_z = state_z + torch.matmul(key_t, delta_z.transpose(-1, -2))
        denominators.append(torch.matmul(state_z.transpose(-1, -2), query_t))

    def restore(tensors: list[torch.Tensor], dim: int) -> torch.Tensor:
        ordered = tensors[::-1] if flip_output else tensors
        return torch.stack(ordered, dim=2).permute(0, 1, 3, 2, 4).reshape(batch_size, num_heads, dim, token_count)

    numerator = restore(numerators, head_dim)
    denominator = None if denominators is None else restore(denominators, 1)
    return numerator, denominator


def _materialized_reverse_oracle(
    query_rot: torch.Tensor,
    key_rot: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    *,
    spatial_tokens: int,
    query: torch.Tensor | None,
    key: torch.Tensor | None,
    skip_z: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run exactly the baseline's materialized exclusive backward direction."""
    frames = beta.shape[2]
    backward_kwargs: dict[str, torch.Tensor] = {}
    if not skip_z:
        assert query is not None and key is not None
        backward_kwargs["query"] = _reverse_frames(
            query,
            frames=frames,
            spatial_tokens=spatial_tokens,
        )
        backward_kwargs["key"] = _reverse_frames(
            key,
            frames=frames,
            spatial_tokens=spatial_tokens,
            shift_value=0.0,
        )
    return _forward_scan_oracle(
        _reverse_frames(
            query_rot,
            frames=frames,
            spatial_tokens=spatial_tokens,
        ),
        _reverse_frames(
            key_rot,
            frames=frames,
            spatial_tokens=spatial_tokens,
            shift_value=0.0,
        ),
        _reverse_frames(
            value,
            frames=frames,
            spatial_tokens=spatial_tokens,
            shift_value=0.0,
        ),
        _flip_and_shift(beta, dim=2, shift_value=0.0),
        _flip_and_shift(decay, dim=2, shift_value=1.0),
        spatial_tokens=spatial_tokens,
        skip_z=skip_z,
        flip_output=True,
        **backward_kwargs,
    )


def _materialized_bidirectional_oracle(
    query_rot: torch.Tensor,
    key_rot: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    *,
    spatial_tokens: int,
    query: torch.Tensor | None,
    key: torch.Tensor | None,
    skip_z: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run both directions using only the pre-H001 materialized algorithm."""
    num_fwd, den_fwd = _forward_scan_oracle(
        query_rot,
        key_rot,
        value,
        beta,
        decay,
        spatial_tokens=spatial_tokens,
        query=query,
        key=key,
        skip_z=skip_z,
    )
    num_bwd, den_bwd = _materialized_reverse_oracle(
        query_rot,
        key_rot,
        value,
        beta,
        decay,
        spatial_tokens=spatial_tokens,
        query=query,
        key=key,
        skip_z=skip_z,
    )
    if skip_z:
        return num_fwd + num_bwd, None
    assert den_fwd is not None and den_bwd is not None
    return num_fwd + num_bwd, den_fwd + den_bwd


def _alternating(
    shape: tuple[int, ...],
    *,
    scale: float,
    device: str | torch.device,
) -> torch.Tensor:
    values = torch.arange(math.prod(shape), device=device)
    values = values.remainder(2).mul_(2).sub_(1).to(torch.float32).mul_(scale)
    return values.reshape(shape)


def _make_inputs(
    frames: int,
    *,
    spatial_tokens: int,
    heads: int,
    head_dim: int,
    layout: str,
    mode: str,
    batch_size: int = 1,
    device: str | torch.device = "cuda",
) -> dict[str, torch.Tensor]:
    """Build either contiguous inputs or the model's projected tensor strides."""
    tokens = frames * spatial_tokens

    def projected(scale: float) -> torch.Tensor:
        shape = (batch_size, tokens, heads, head_dim)
        if mode == "random":
            raw = torch.randn(shape, device=device, dtype=torch.float32) * scale
        elif mode == "zeros":
            raw = torch.zeros(shape, device=device, dtype=torch.float32)
        else:
            raw = _alternating(shape, scale=scale, device=device)
        result = raw.permute(0, 2, 3, 1)
        return result.contiguous() if layout == "contiguous" else result

    values = [projected(scale) for scale in (0.03, -0.02, 0.04, -0.01, 0.05)]
    gate_shape = (batch_size, frames, spatial_tokens, heads)
    decay_shape = (batch_size, frames, heads)
    if mode == "random":
        beta_raw = torch.rand(gate_shape, device=device) * 0.02
        decay_raw = torch.rand(decay_shape, device=device) * 0.5 + 0.4
    elif mode == "zeros":
        beta_raw = torch.zeros(gate_shape, device=device)
        decay_raw = torch.zeros(decay_shape, device=device)
    else:
        beta_raw = _alternating(gate_shape, scale=0.5, device=device).add_(0.5)
        decay_raw = _alternating(decay_shape, scale=0.625, device=device).add_(0.625)

    beta = beta_raw.permute(0, 3, 1, 2)
    decay = decay_raw.transpose(1, 2)
    if layout == "contiguous":
        beta = beta.contiguous()
        decay = decay.contiguous()
    query, key, value, query_rot, key_rot = values
    return {
        "query": query,
        "key": key,
        "value": value,
        "query_rot": query_rot,
        "key_rot": key_rot,
        "beta": beta,
        "decay": decay,
    }


def _call_kwargs(
    inputs: dict[str, torch.Tensor],
    *,
    spatial_tokens: int,
    skip_z: bool,
) -> _ScanKwargs:
    return {
        "query_rot": inputs["query_rot"],
        "key_rot": inputs["key_rot"],
        "value": inputs["value"],
        "beta": inputs["beta"],
        "decay": inputs["decay"],
        "spatial_tokens": spatial_tokens,
        "query": None if skip_z else inputs["query"],
        "key": None if skip_z else inputs["key"],
        "skip_z": skip_z,
    }


def _assert_exact_nested(
    actual: tuple[torch.Tensor, torch.Tensor | None],
    expected: tuple[torch.Tensor, torch.Tensor | None],
) -> None:
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        if expected_tensor is None:
            assert actual_tensor is None
        else:
            assert actual_tensor is not None
            torch.testing.assert_close(
                actual_tensor,
                expected_tensor,
                rtol=0,
                atol=0,
                equal_nan=True,
            )


@pytest.mark.cpu
@pytest.mark.parametrize("layout", ["contiguous", "projected"])
@pytest.mark.parametrize("frames", [1, 3])
@pytest.mark.parametrize("skip_z", [False, True], ids=["main", "camera"])
@torch.inference_mode()
def test_cpu_batched_matches_materialized_scan(
    layout: str,
    frames: int,
    skip_z: bool,
) -> None:
    """Exercise B>1 and both the synthetic-only and recurrent reverse cases."""
    torch.manual_seed(1300 + frames + int(skip_z))
    spatial_tokens = 6
    inputs = _make_inputs(
        frames,
        spatial_tokens=spatial_tokens,
        heads=2,
        head_dim=4,
        layout=layout,
        mode="random",
        batch_size=2,
        device="cpu",
    )
    kwargs = _call_kwargs(inputs, spatial_tokens=spatial_tokens, skip_z=skip_z)

    _assert_exact_nested(
        sana_wm._bidirectional_delta_scan(**kwargs),
        _materialized_bidirectional_oracle(**kwargs),
    )


@pytest.mark.cpu
@pytest.mark.parametrize("skip_z", [False, True], ids=["main", "camera"])
def test_cpu_batched_gradients_match_materialized_scan(skip_z: bool) -> None:
    """Keep the pure-PyTorch scan differentiable despite its new traversal."""
    torch.manual_seed(7700 + int(skip_z))
    base_inputs = _make_inputs(
        3,
        spatial_tokens=6,
        heads=2,
        head_dim=4,
        layout="projected",
        mode="random",
        batch_size=2,
        device="cpu",
    )
    candidate_inputs = {name: tensor.detach().clone().requires_grad_(True) for name, tensor in base_inputs.items()}
    oracle_inputs = {name: tensor.detach().clone().requires_grad_(True) for name, tensor in base_inputs.items()}

    candidate = sana_wm._bidirectional_delta_scan(**_call_kwargs(candidate_inputs, spatial_tokens=6, skip_z=skip_z))
    oracle = _materialized_bidirectional_oracle(**_call_kwargs(oracle_inputs, spatial_tokens=6, skip_z=skip_z))
    candidate_loss = candidate[0].square().sum()
    oracle_loss = oracle[0].square().sum()
    if not skip_z:
        assert candidate[1] is not None and oracle[1] is not None
        candidate_loss = candidate_loss + candidate[1].square().sum()
        oracle_loss = oracle_loss + oracle[1].square().sum()
    candidate_loss.backward()
    oracle_loss.backward()

    used_names = ("value", "query_rot", "key_rot") if skip_z else tuple(base_inputs)
    for name in used_names:
        torch.testing.assert_close(
            candidate_inputs[name].grad,
            oracle_inputs[name].grad,
            rtol=1e-6,
            atol=1e-8,
        )


@pytest.mark.parametrize("frames", [1, 2, 4, 21])
@pytest.mark.parametrize("skip_z", [False, True], ids=["main", "camera"])
@pytest.mark.cuda
@_CUDA_ONLY
@torch.inference_mode()
def test_matches_materialized_scan_at_real_shape_and_strides(
    frames: int,
    skip_z: bool,
) -> None:
    """Cover edge, source-micro, and 161-frame latent T values at real shape."""
    torch.manual_seed(4200 + frames + int(skip_z))
    spatial_tokens = 880  # 22 * 40: production size and deliberately non-square.
    inputs = _make_inputs(
        frames,
        spatial_tokens=spatial_tokens,
        heads=20,
        head_dim=112,
        layout="projected",
        mode="random",
    )
    tokens = frames * spatial_tokens
    assert inputs["query"].stride() == (20 * 112 * tokens, 112, 1, 20 * 112)
    assert not inputs["query"].is_contiguous()
    kwargs = _call_kwargs(inputs, spatial_tokens=spatial_tokens, skip_z=skip_z)

    expected = _materialized_bidirectional_oracle(**kwargs)
    actual = sana_wm._bidirectional_delta_scan(**kwargs)

    _assert_exact_nested(actual, expected)


@pytest.mark.parametrize("layout", ["contiguous", "projected"])
@pytest.mark.parametrize("mode", ["zeros", "extreme"])
@pytest.mark.parametrize("skip_z", [False, True], ids=["main", "camera"])
@pytest.mark.cuda
@_CUDA_ONLY
@torch.inference_mode()
def test_matches_zero_and_extreme_inputs(
    layout: str,
    mode: str,
    skip_z: bool,
) -> None:
    spatial_tokens = 15  # Intentionally not an H*H square.
    inputs = _make_inputs(
        4,
        spatial_tokens=spatial_tokens,
        heads=2,
        head_dim=8,
        layout=layout,
        mode=mode,
    )
    kwargs = _call_kwargs(inputs, spatial_tokens=spatial_tokens, skip_z=skip_z)
    _assert_exact_nested(
        sana_wm._bidirectional_delta_scan(**kwargs),
        _materialized_bidirectional_oracle(**kwargs),
    )


@pytest.mark.parametrize("skip_z", [False, True], ids=["main", "camera"])
@pytest.mark.cuda
@_CUDA_ONLY
@torch.inference_mode()
def test_t1_retains_nonfinite_query_matmuls(skip_z: bool) -> None:
    """The zero-state final query must still execute: 0*NaN/Inf is NaN."""
    spatial_tokens = 15
    inputs = _make_inputs(
        1,
        spatial_tokens=spatial_tokens,
        heads=2,
        head_dim=8,
        layout="projected",
        mode="random",
    )
    inputs["query_rot"][..., 0] = float("nan")
    if not skip_z:
        inputs["query"][..., 1] = float("inf")
    kwargs = _call_kwargs(inputs, spatial_tokens=spatial_tokens, skip_z=skip_z)

    actual = sana_wm._delta_scan(**kwargs, reverse_exclusive=True)
    expected = _materialized_reverse_oracle(**kwargs)

    _assert_exact_nested(actual, expected)
    assert torch.isnan(actual[0][..., 0]).all()
    if not skip_z:
        assert actual[1] is not None
        assert torch.isnan(actual[1][..., 1]).all()


@pytest.mark.parametrize("skip_z", [False, True], ids=["main", "camera"])
@pytest.mark.cuda
@_CUDA_ONLY
@torch.inference_mode()
def test_t1_does_not_consume_a_nonexistent_future_source(skip_z: bool) -> None:
    spatial_tokens = 15
    inputs = _make_inputs(
        1,
        spatial_tokens=spatial_tokens,
        heads=2,
        head_dim=8,
        layout="projected",
        mode="random",
    )
    for name in ("key", "value", "key_rot", "beta", "decay"):
        inputs[name].fill_(float("nan"))
    kwargs = _call_kwargs(inputs, spatial_tokens=spatial_tokens, skip_z=skip_z)

    actual = sana_wm._delta_scan(**kwargs, reverse_exclusive=True)
    expected = _materialized_reverse_oracle(**kwargs)

    _assert_exact_nested(actual, expected)
    assert torch.count_nonzero(actual[0]) == 0
    if not skip_z:
        assert actual[1] is not None
        assert torch.count_nonzero(actual[1]) == 0


@pytest.mark.cuda
@_CUDA_ONLY
@torch.inference_mode()
def test_bidirectional_route_hits_direct_reverse_scan_with_original_inputs(mocker) -> None:
    spatial_tokens = 15
    inputs = _make_inputs(
        2,
        spatial_tokens=spatial_tokens,
        heads=2,
        head_dim=8,
        layout="projected",
        mode="random",
    )
    kwargs = _call_kwargs(inputs, spatial_tokens=spatial_tokens, skip_z=False)

    scan = mocker.patch.object(
        sana_wm,
        "_delta_scan",
        wraps=sana_wm._delta_scan,
    )
    sana_wm._bidirectional_delta_scan(**kwargs)

    assert scan.call_count == 2
    forward_call, backward_call = scan.call_args_list
    assert forward_call.kwargs.get("reverse_exclusive", False) is False
    assert backward_call.kwargs["reverse_exclusive"] is True
    assert backward_call.args[0] is inputs["query_rot"]
    assert backward_call.args[1] is inputs["key_rot"]
    assert backward_call.args[2] is inputs["value"]
    assert backward_call.args[3] is inputs["beta"]
    assert backward_call.args[4] is inputs["decay"]
    assert not hasattr(sana_wm, "_reverse_frames")
    assert not hasattr(sana_wm, "_flip_and_shift")
