# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MXFP8 linears for the H3 DiT on b12x (SM120/SM121), opt-in via VLLM_OMNI_DIT_MXFP8=1.

At M=18,748 the FFN up-projection runs 29.22 ms in bf16 and 8.54 ms through b12x's MXFP8
path with the activation quant fused (3.42x). vLLM's parallel linear stays the owner of
weight loading and tensor-parallel sharding; only the math changes, on the local shard.

Contracts preserved from the modules being replaced:

* a RowParallel linear's output is all-reduced across the tensor-parallel group, so
  ``reduce=True`` reproduces it;
* callers unpack a tuple, so ``__call__`` returns ``(output, None)``.

A b12x plan carries shapes and capacity only, never weights, so ONE plan is shared by every
module with the same (in, out) shape -- four plans for the whole DiT rather than four per
block. They are prepared eagerly at load (``prepare_shared``) so a served model does not pay
compilation on its first request. ``expected_m`` is deliberately not pinned, so a request
with a different token count reuses the program.
"""

from __future__ import annotations

import os

import torch

_CAPACITY_STEP = 4096
_DEFAULT_CAPACITY = 40960


def enabled() -> bool:
    return os.environ.get("VLLM_OMNI_DIT_MXFP8", "0") == "1"


def warmup_capacity() -> int:
    return int(os.environ.get("VLLM_OMNI_DIT_MXFP8_CAPACITY", _DEFAULT_CAPACITY))


def prequant_manifest() -> set[str] | None:
    """Keys of a pre-quantised DiT checkpoint, or None for the runtime-quantisation path.

    Armed by ``H3_MX_PREQUANT=1`` plus ``H3_MX_PREQUANT_MANIFEST=<transformer>/quantization.json``.
    With it, the wide DiT linears are never materialised as bf16: the loader keeps the stored
    e4m3 values and uint8 scales and packs them straight into the same b12x path. That is what
    makes TP1 x USP4 reachable -- 62 GiB of bf16 transformer plus the encoder half does not fit
    a 95 GiB card, while the stored e4m3 plus scales does.

    Only valid at tensor-parallel size 1: with TP > 1 the stored shards are full-width and the
    loader's TP slicing has to run, so the runtime path stays in charge.
    """
    if os.environ.get("H3_MX_PREQUANT", "0") != "1":
        return None
    path = os.environ.get("H3_MX_PREQUANT_MANIFEST", "")
    if not path or not os.path.exists(path):
        return None
    import json

    with open(path) as fh:
        keys = json.load(fh).get("quantized") or []
    return set(keys) if keys else None


def quantize_rows(source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """ModelOpt-style MXFP8 rows: 32-element blocks, E8M0 scale, e4m3 values."""
    rows, width = map(int, source.shape)
    blocked = source.to(torch.float32).reshape(rows, width // 32, 32)
    max_abs = blocked.abs().amax(dim=-1)
    safe = torch.where(max_abs > 0.0, max_abs / 448.0, torch.ones_like(max_abs))
    scale_u8 = (torch.ceil(torch.log2(safe)).clamp(-127, 127) + 127).to(torch.uint8)
    scale = scale_u8.view(torch.float8_e8m0fnu).to(torch.float32)
    values = (blocked / scale[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).reshape(rows, width).contiguous()
    return values, scale_u8.contiguous()


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
    """Whether a vLLM parallel linear all-reduces its output."""
    return bool(getattr(module, "reduce_results", False)) and _tp_size() > 1


class Mxfp8Linear:
    """Drop-in for a locally-sharded linear ``weight [out, in]`` on b12x MXFP8."""

    def __init__(
        self, weight: torch.Tensor, *, name: str = "h3-linear", reduce: bool = False, bias: torch.Tensor | None = None
    ):
        from b12x.gemm import mxfp8_linear

        self.name = name
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])
        self.reduce = bool(reduce)
        self.bias = bias
        self.packed = mxfp8_linear.pack_weight(*quantize_rows(weight.detach().to(torch.bfloat16)))
        self._plan = None
        self._capacity = 0

    @classmethod
    def from_quantized(
        cls,
        values: torch.Tensor,
        scale_u8: torch.Tensor,
        *,
        name: str = "h3-linear",
        reduce: bool = False,
        bias: torch.Tensor | None = None,
    ) -> Mxfp8Linear:
        """Build from a stored (e4m3 values, uint8 E8M0 scales) pair.

        ``quantize_rows`` emits exactly this pair, so packing a pre-quantised shard yields the
        same ``packed`` state as packing the bf16 shard it came from: the kernels, the plans
        and the numerics are unchanged, only the storage is.
        """
        from b12x.gemm import mxfp8_linear

        self = cls.__new__(cls)
        self.name = name
        self.in_features = int(values.shape[1])
        self.out_features = int(values.shape[0])
        self.reduce = bool(reduce)
        self.bias = bias
        device = _device()
        self.packed = mxfp8_linear.pack_weight(values.to(device), scale_u8.to(device))
        self._plan = None
        self._capacity = 0
        return self

    def __call__(self, x: torch.Tensor):
        from b12x.gemm import mxfp8_linear

        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        if not flat.is_contiguous():
            flat = flat.contiguous()
        m = int(flat.shape[0])
        if self._plan is None:
            raise RuntimeError(f"{self.name}: plan not prepared; call prepare_shared() at load")
        if m > self._capacity:
            raise ValueError(f"{self.name}: {m} rows exceeds the prepared capacity {self._capacity}")
        out = mxfp8_linear.mm(flat, self.packed, plan=self._plan)
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


def prepare_shared(modules, capacity: int | None = None) -> dict[tuple[int, int], int]:
    """Prepare one plan per (in, out) shape and share it across every module of that shape.

    A plan describes shapes and capacity, not weights, so all four kinds of wide linear in
    every block collapse to four prepared programs. Returns {(in, out): capacity}.
    """
    from b12x.gemm import blockscaled
    from b12x.preparation import PreparationSession, PreparedCall

    cap = _capacity(warmup_capacity() if capacity is None else int(capacity))
    device = _device()

    groups: dict[tuple[int, int], list[Mxfp8Linear]] = {}
    for mod in modules:
        groups.setdefault((mod.in_features, mod.out_features), []).append(mod)
    if not groups:
        return {}

    requests = []
    for (in_f, out_f), mods in groups.items():
        placeholder = torch.empty((cap, in_f), dtype=torch.bfloat16, device=device)
        packed = mods[0].packed
        query = blockscaled.query_from_call(placeholder, packed, activation_mode="quantized")
        plan = blockscaled.plan(query)

        def call(state, placeholder=placeholder, packed=packed):
            return PreparedCall(
                run=lambda: state.run(
                    placeholder, packed.weight.values, packed.weight.scale_mma, None, activation_scale=None
                )
            )

        requests.append(plan.request(name=f"h3-mxfp8-{out_f}x{in_f}", prepare_call=call))
        for mod in mods:
            mod._plan = plan
            mod._capacity = cap

    with PreparationSession(device=device, autotune=False, compile_workers=1) as session:
        session.prepare(tuple(requests))
    return {key: cap for key in groups}
