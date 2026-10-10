# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sol-Attn diffusion attention backend (SM120/SM121).

Runs packed attention through the released `sol-attn` Triton backend (NVlabs/Sana
`sol-engine`, Apache-2.0, vendored under site-packages `sol_attn/`). Sol-Attn is
**training-free**: each 64-token query block derives its own threshold
(mean + tau * std of the block-centroid score distribution, `thresh_type="diag"`) and
attends exactly the KV blocks above it, plus a +/-1 block local band and a contiguous
sink range. Omitted blocks are approximated by one virtual key per block (value = the
V sum, multiplicity = block length) merged inside the same online-softmax pass.

Two consequences the caller must respect, both from the reference integration
(`Sol-H3/h3_runtime/sparse_attention.py`):

* the sink keeps sink **keys** exact for every query, but sink **queries** are still
  threshold-routed, so they are recomputed densely -- "SOL keeps the released
  sparse-query policy and recomputes sink queries densely";
* a configuration that asks for sparse and silently gets dense is a wrong measurement
  wearing the right label, so the call counters are checked and reported.

Sink choice for the packed H3 layout `[text | condition video | audio | target video]`:
`video_layout.video_spans[role="target"].start` is the first generated-video row, so
`[0, target.start)` is exactly the prefix the reference calls `sink_mode="prefix"`.
No permutation is needed because this backend passes an explicit `sink_start`.

SM120 runs the **Triton** backend: NVIDIA's own SM120 integration removes the CuTe entry
(`_CUTE_BACKENDS.pop((12, 0))`) and validates `sol_backend == "triton"`.
"""

from __future__ import annotations

import os
import time
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)

logger = init_logger(__name__)

try:  # sol_attn is vendored for SM120/SM121 hosts only.
    from sol_attn import sol_attn as _sol_attn

    _SOL_IMPORT_ERROR: str | None = None
except Exception as _exc:  # pragma: no cover - import-time environment probe
    _sol_attn = None
    _SOL_IMPORT_ERROR = str(_exc)

BLOCK = 64


class SolAttnBackend(AttentionBackend):
    """Sol-Attn: threshold-routed block-sparse attention with a centroid residual."""

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        # the released kernel asserts head_dim == 128 (bf16 only)
        return [128]

    @classmethod
    def supports_packed_mask_free(cls) -> bool:
        return True

    @staticmethod
    def get_name() -> str:
        return "SOL_ATTN"

    @staticmethod
    def get_impl_cls() -> type[SolAttnImpl]:
        return SolAttnImpl


def _dense(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """SDPA on [rows, H, D]. Same convention as the reference `_dense` helper."""
    out = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0),
        dropout_p=0.0,
        is_causal=False,
    )
    return out.squeeze(0).transpose(0, 1)


def _sink_rows(attn_metadata: AttentionMetadata | None) -> int:
    """Rows in [0, n) that are prefix (text / conditioning / audio) rather than target video.

    Read from the layout the H3 model already publishes for its VSA routing; absent
    metadata means "no sink", which keeps the backend usable before the probe lands.
    """
    if attn_metadata is None:
        return 0
    layout = getattr(attn_metadata, "video_layout", None)
    if layout is None:
        return 0
    try:
        target = next((span for span in reversed(layout.video_spans) if span.role == "target"), None)
    except Exception:  # pragma: no cover - layout shape drift
        return 0
    if target is None:
        return 0
    return max(0, int(target.start))


class SolAttnImpl(AttentionImpl):
    """Threshold-routed sparse attention over the packed sequence."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float | None = None,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        backend_kwargs: dict | None = None,
        **extra_impl_args,
    ) -> None:
        if causal:
            raise ValueError("SOL_ATTN diffusion attention is bidirectional; causal is unsupported")
        if head_size != 128:
            raise ValueError(f"SOL_ATTN requires head_size 128, got {head_size}")
        if num_kv_heads is not None and num_kv_heads != num_heads:
            raise ValueError("SOL_ATTN does not shard KV heads; GQA is unsupported here")
        self.num_heads = num_heads
        self.head_size = head_size
        self.softmax_scale = softmax_scale
        # Reference defaults for this model family (Sol-H3 engine.py, ref2va branch).
        self.tau = float(os.environ.get("SOL_ATTN_TAU", "1.0"))
        self.thresh_type = os.environ.get("SOL_ATTN_THRESH_TYPE", "diag")
        self.gate = os.environ.get("SOL_ATTN_CORRECTNESS_GATE", "0") == "1"
        self.strict = os.environ.get("SOL_ATTN_STRICT", "1") == "1"

        self.sparse_calls = 0
        self.dense_calls = 0
        self.gated_shapes: set = set()
        self._logged_layout = False
        self._timed = 0
        self.last_density: dict | None = None
        self.last_sink_rows: int | None = None
        if backend_kwargs:
            logger.warning("SolAttnImpl ignoring backend_kwargs: %s", list(backend_kwargs.keys()))

    # -- gates -----------------------------------------------------------------
    def _run_gate(self, qb, kb, vb) -> None:
        """Route everything (tau=-1000) and check the kernel against SDPA, once per shape.

        On the real QKV, not random tensors: a random probe answers a question about the
        kernel, not about this model's tensors at this shape. Limits are the reference's.
        """
        key = (int(qb.shape[1]), int(qb.shape[2]), int(qb.shape[3]))
        if key in self.gated_shapes:
            return
        got = _sol_attn(
            qb,
            kb,
            vb,
            scale=self.softmax_scale,
            tau=-1000.0,
            thresh_type=self.thresh_type,
            kv_splits=1,
            sink_tokens=0,
            sink_start=0,
        )
        want = _dense(qb[0], kb[0], vb[0]).unsqueeze(0)
        # fp32 oracle: the published ABSOLUTE limits were calibrated on the reference
        # stack's activations. Ours peak at ~54, where a 0.25 absolute difference is
        # ~1 bf16 ulp. Comparing two bf16 kernels cannot separate kernel error from
        # common-mode rounding, so the gate measures each against an fp32 reference and
        # requires the kernel to be no worse than the dense path it replaces.
        q32, k32, v32 = qb.float(), kb.float(), vb.float()
        ref32 = torch.nn.functional.scaled_dot_product_attention(
            q32.transpose(1, 2), k32.transpose(1, 2), v32.transpose(1, 2), dropout_p=0.0, is_causal=False
        ).transpose(1, 2)
        e_sol = float((got.float() - ref32).abs().max())
        e_dense = float((want.float() - ref32).abs().max())
        scale = float(want.float().abs().max())
        stats = {
            "kernel_vs_fp32_max": e_sol,
            "dense_vs_fp32_max": e_dense,
            "max_abs": float((got.float() - want.float()).abs().max()),
            "max_rel": float((got.float() - want.float()).abs().max()) / max(scale, 1e-12),
            "activation_max_abs": scale,
            "rel_l2": float(
                torch.linalg.vector_norm(got.float() - want.float())
                / torch.linalg.vector_norm(want.float()).clamp_min(1e-12)
            ),
        }
        limits = {
            "rel_l2": float(os.environ.get("SOL_ATTN_GATE_REL_L2", "0.005")),
            "kernel_error_ratio": float(os.environ.get("SOL_ATTN_GATE_ERROR_RATIO", "2.0")),
        }
        ratio = e_sol / max(e_dense, 1e-12)
        passed = stats["rel_l2"] <= limits["rel_l2"] and ratio <= limits["kernel_error_ratio"]
        logger.info(
            "SOL_ATTN correctness gate %s shape=%s kernel_vs_fp32 %.6g dense_vs_fp32 %.6g "
            "ratio %.3f (<= %.1f) rel_l2 %.6f (<= %.5f) activation_max %.4g tau=-1000",
            "PASS" if passed else "FAIL",
            tuple(qb.shape),
            e_sol,
            e_dense,
            ratio,
            limits["kernel_error_ratio"],
            stats["rel_l2"],
            limits["rel_l2"],
            scale,
        )
        if not passed:
            raise RuntimeError(f"Sol-Attn correctness gate failed on real QKV: {stats} > {limits}")
        self.gated_shapes.add(key)

    # -- attention -------------------------------------------------------------
    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        if _sol_attn is None:
            raise ImportError(
                "SOL_ATTN requires the vendored sol-attn package on an SM120/SM121 device. "
                f"Import failed with: {_SOL_IMPORT_ERROR}"
            )
        if attn_metadata is not None and getattr(attn_metadata, "attn_mask", None) is not None:
            raise ValueError("SOL_ATTN does not support attn_mask; select a mask-capable backend")

        q3 = query.flatten(0, 1)
        k3 = key.flatten(0, 1)
        v3 = value.flatten(0, 1)
        packed = getattr(attn_metadata, "packed_padding", None) if attn_metadata is not None else None
        n = int(packed.q_length) if packed is not None else int(q3.shape[0])
        if packed is not None and int(packed.kv_length) != n:
            raise ValueError("SOL_ATTN requires packed Q and KV lengths to match")

        sink = _sink_rows(attn_metadata)
        if sink >= n:
            sink = 0
        if not self._logged_layout:
            self._logged_layout = True
            layout = getattr(attn_metadata, "video_layout", None)
            spans = [
                (getattr(s, "role", None), int(getattr(s, "start", -1)), int(getattr(s, "length", -1)))
                for s in getattr(layout, "video_spans", []) or []
            ]
            logger.info(
                "SOL_ATTN layout: q=%s kv=%s n=%d sink=%d spans=%s", tuple(q3.shape), tuple(k3.shape), n, sink, spans
            )
        qb = q3[:n].unsqueeze(0).contiguous()
        kb = k3[:n].unsqueeze(0).contiguous()
        vb = v3[:n].unsqueeze(0).contiguous()
        _t0 = time.time()
        if self.gate:
            self._run_gate(qb, kb, vb)

        out = _sol_attn(
            qb,
            kb,
            vb,
            scale=self.softmax_scale,
            tau=self.tau,
            thresh_type=self.thresh_type,
            kv_splits=1,
            sink_tokens=sink,
            sink_start=0,
        )
        if sink:
            # Reference contract: the released `sol` policy keeps sink KEYS exact for
            # every query; sink QUERIES are threshold-routed and recomputed densely.
            out[:, :sink] = _dense(qb[0, :sink], kb[0], vb[0]).unsqueeze(0)
        self.sparse_calls += 1
        self.last_sink_rows = sink
        if self._timed < 3:
            self._timed += 1
            torch.accelerator.synchronize()
            logger.info(
                "SOL_ATTN call %d: n=%d sink=%d took %.1f ms", self.sparse_calls, n, sink, (time.time() - _t0) * 1000.0
            )

        if out.shape[1] != q3.shape[0]:
            full = q3.new_zeros((q3.shape[0],) + tuple(out.shape[2:]))
            full[: out.shape[1]] = out[0]
            return full.reshape_as(query)
        return out[0].reshape_as(query)

    def stats(self) -> dict[str, Any]:
        return {
            "backend": "SOL_ATTN",
            "tau": self.tau,
            "thresh_type": self.thresh_type,
            "sparse_calls": self.sparse_calls,
            "dense_calls": self.dense_calls,
            "sink_rows": self.last_sink_rows,
            "gate": sorted(self.gated_shapes),
            "route_density": self.last_density,
        }

    def forward_xpu(self, query, key, value, attn_metadata=None):  # pragma: no cover
        raise NotImplementedError("SOL_ATTN is CUDA-only")
