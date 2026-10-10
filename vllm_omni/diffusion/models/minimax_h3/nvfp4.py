# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""NVFP4 (W4A4) DiT linears for the H3 lane - experimental arm, opt-in via VLLM_OMNI_DIT_NVFP4=1.

A sibling of ``vllm_omni.diffusion.models.minimax_h3.b12x_mxfp8`` (``Mxfp8Linear`` /
``prepare_shared``) with the same contract, so the model's single construction site can select it
with an env-gated import alias. It uses the *same* b12x op the DiT already runs
(``b12x::blockscaled_bf16``, via ``blockscaled.mm`` with a bf16 source), which quantises the
activation internally; the differences are only that the weight is packed NVFP4 and that the call hands
the op a per-call activation global scale.

Measured at M=19904, eager, isolated: the four linears together run 13.929 ms in MXFP8
and 6.540 ms in NVFP4 (2.13x, 944-1284 TFLOP/s), i.e. about -370 ms/step over the 50 blocks. Mean
relative error against a bf16 reference rises from 3.8e-2 to 1.35e-1. Receipts:
``profile/sparse-attn-01/fp4/FINDINGS-nvfp4.md`` and ``fp4/fp4_vs_mxfp8.py``.

Global-scale convention (derived empirically, ``fp4/fp4_gemm_test.py``): b12x "reciprocal" kind,
``G = 448*6/amax``. The per-block E4M3 scale is ``G*max_abs/6`` and the dequant multiplier is
``1/G``. With ``global_scale_kind='multiplier'`` and the same G the output came out exactly G^2 too
large, which is how the convention was identified.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

_CAPACITY_STEP = 4096
_DEFAULT_CAPACITY = 40960
FP4_MAX = 448.0 * 6.0


def enabled() -> bool:
    return os.environ.get("VLLM_OMNI_DIT_NVFP4", "0") == "1"


def warmup_capacity() -> int:
    return int(os.environ.get("VLLM_OMNI_DIT_NVFP4_CAPACITY", _DEFAULT_CAPACITY))


def _capacity(n: int) -> int:
    return max(_CAPACITY_STEP, ((int(n) + _CAPACITY_STEP - 1) // _CAPACITY_STEP) * _CAPACITY_STEP)


def _device() -> torch.device:
    return torch.device("cuda", torch.accelerator.current_device_index())


def _tp_size() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_world_size

        return int(get_tensor_model_parallel_world_size())
    except Exception:
        return 1


def reduce_needed(module) -> bool:
    """Whether a vLLM parallel linear all-reduces its output (same rule as the MXFP8 path)."""
    return bool(getattr(module, "reduce_results", False)) and _tp_size() > 1


def act_scale(flat: torch.Tensor) -> torch.Tensor:
    """Per-call activation global scale G = 448*6/amax, shape (1,) fp32 on the input's device.

    Written as two reductions rather than ``flat.abs().amax()`` so no full-size fp32/bf16
    temporary is materialised in eager mode; inside the DiT the call site sits in a
    torch.compile region, where inductor fuses this into a single pass anyway. The read itself is
    unavoidable for a *dynamic* scale (b12x's MXFP8 path gets its per-32-block scales inside the
    quantizer kernel, but NVFP4 needs a per-tensor scale from outside) - about 0.2-0.4 ms per
    linear call on a 19904-row activation, i.e. of order 40-80 ms/step over 200 calls. Folding it
    into the producing RMSNorm/gate chain is the follow-up lever.
    """
    amax = torch.maximum(flat.amax(), flat.amin().neg()).to(torch.float32).clamp_min(1e-30)
    return (FP4_MAX / amax).reshape(1)


def quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """bf16 [N,K] -> (packed values [N,K/2] uint8, swizzled E4M3 scales, G) via b12x's NVFP4 quantizer."""
    from b12x.preparation import PreparationSession, PreparedCall
    from b12x.quantization import nvfp4 as nvq

    w = weight.detach().to(torch.bfloat16).contiguous()
    n, k = map(int, w.shape)
    if n % 128 or k % 128:
        raise ValueError(f"h3-nvfp4: weight shape {n}x{k} must be a multiple of 128 for the TMA quantizer")
    g = (FP4_MAX / w.float().abs().amax().clamp_min(1e-30)).to(torch.float32).reshape(1)
    plan = nvq.plan(n, k)
    outs = nvq.allocate_outputs(plan, device=w.device)

    def prepare_call(state, x=w, gs=g, o=outs):
        return PreparedCall(run=lambda: state.run(x, gs, o))

    with PreparationSession(device=w.device, autotune=False, compile_workers=1) as session:
        session.prepare((plan.request(name=f"h3-nvfp4-w-{n}x{k}", prepare_call=prepare_call),))
    nvq.run(plan=plan, x=w, global_scale=g, outputs=outs)
    del w
    return outs.packed_a_storage.reshape(n, k // 2), outs.scale_storage, g


def _values_scales(packed) -> tuple[torch.Tensor, torch.Tensor]:
    """(values, scale_mma) of a packed weight.

    MXFP8LinearWeight wraps a base Weight in ``.weight``; NVFP4LinearWeight carries the fields
    directly, so unwrap only when the wrapper is present.
    """
    inner = getattr(packed, "weight", packed)
    return inner.values, inner.scale_mma


class Nvfp4Linear:
    """Drop-in for a locally-sharded linear ``weight [out, in]`` on b12x NVFP4 (W4A4)."""

    def __init__(
        self, weight: torch.Tensor, *, name: str = "h3-linear", reduce: bool = False, bias: torch.Tensor | None = None
    ):
        from b12x.gemm import blockscaled

        self.name = name
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])
        self.reduce = bool(reduce)
        self.bias = bias
        values, scales, self.global_scale = quantize_weight(weight)
        self.packed = blockscaled.pack_weight(
            values, scales, recipe="nvfp4", global_scale=self.global_scale, global_scale_kind="reciprocal"
        )
        self._plan = None
        self._capacity = 0
        _log(f"packed {self.name}: {self.out_features}x{self.in_features}")

    @classmethod
    def from_quantized(cls, values: torch.Tensor, scale_u8: torch.Tensor, **kwargs):
        raise NotImplementedError(
            "h3-nvfp4: the pre-quantised (H3_MX_PREQUANT) checkpoint path is MXFP8-only; "
            "run this arm with runtime quantisation"
        )

    @property
    def _weight_parts(self):
        return _values_scales(self.packed)

    def __call__(self, x: torch.Tensor):
        from b12x.gemm import blockscaled

        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        if not flat.is_contiguous():
            flat = flat.contiguous()
        m = int(flat.shape[0])
        if self._plan is None:
            raise RuntimeError(f"{self.name}: plan not prepared; call prepare_shared() at load")
        if m > self._capacity:
            raise ValueError(f"{self.name}: {m} rows exceeds the prepared capacity {self._capacity}")
        scale = act_scale(flat)
        if _TRACE_CALLS:
            _log(f"first call {self.name}: m={m} k={self.in_features} n={self.out_features}")
        out = blockscaled.mm(flat, self.packed, plan=self._plan, activation_global_scale=scale)
        out = out.reshape(*shape[:-1], self.out_features)
        if self.reduce and _tp_size() > 1:
            from vllm.distributed import tensor_model_parallel_all_reduce

            try:
                from . import ar_wire as h3_ar_wire

                out = h3_ar_wire.tp_all_reduce(out, tensor_model_parallel_all_reduce)
            except Exception:
                out = tensor_model_parallel_all_reduce(out)
        if self.bias is not None:
            out = out + self.bias
        return out, None


_LOGGED: set[str] = set()
# Per-call logging is OFF by default: __call__ runs inside a torch.compile region (the AR-wire
# lesson - a Python float()/f-string on a traced tensor raises InternalTorchDynamoError and latches
# a silent fallback). Load-time logging stays on; it happens outside any compiled region.
_TRACE_CALLS = os.environ.get("H3_NVFP4_TRACE_CALLS", "0") == "1"


def _log(line: str) -> None:
    """One durable line per distinct event, so a run record can prove the arm engaged."""
    if line not in _LOGGED:
        _LOGGED.add(line)
        logger.info("h3_nvfp4 %s", line)


def prepare_shared(modules, capacity: int | None = None) -> dict[tuple[int, int], int]:
    """One plan per (in, out) shape, shared by every module of that shape (as the MXFP8 path does)."""
    from b12x.gemm import blockscaled
    from b12x.preparation import PreparationSession, PreparedCall

    cap = _capacity(warmup_capacity() if capacity is None else int(capacity))
    device = _device()

    groups: dict[tuple[int, int], list[Nvfp4Linear]] = {}
    for mod in modules:
        groups.setdefault((mod.in_features, mod.out_features), []).append(mod)
    if not groups:
        return {}

    requests = []
    for (in_f, out_f), mods in groups.items():
        # A finite, realistic placeholder: only its shape/dtype matter to the plan, but the value
        # feeds the priming call, so torch.empty (garbage, possibly NaN) is not acceptable here
        # the way it is for the MXFP8 path, which takes no activation scale.
        placeholder = torch.ones((cap, in_f), dtype=torch.bfloat16, device=device)
        packed = mods[0].packed
        gw = mods[0].global_scale
        ga = act_scale(placeholder)
        query = blockscaled.query_from_call(
            placeholder, packed, activation_mode="quantized", activation_global_scale=ga
        )
        plan = blockscaled.plan(query)

        def call(state, placeholder=placeholder, packed=packed, gw=gw, ga=ga):
            values, scales = _values_scales(packed)
            return PreparedCall(run=lambda: state.run(placeholder, values, scales, gw, activation_scale=ga))

        requests.append(plan.request(name=f"h3-nvfp4-{out_f}x{in_f}", prepare_call=call))
        _log(
            f"plan {out_f}x{in_f}: weight_global_scale={float(gw):.4f} "
            f"activation_global_scale={float(ga):.4f} capacity={cap}"
        )
        for mod in mods:
            mod._plan = plan
            mod._capacity = cap

    with PreparationSession(device=device, autotune=False, compile_workers=1) as session:
        session.prepare(tuple(requests))
    _log(f"W4A4 arm active: {len(modules)} modules, {len(groups)} plans, capacity={cap}, shapes={sorted(groups)}")
    return {key: cap for key in groups}
