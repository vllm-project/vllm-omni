# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tensor-only Stage-1 NPUGraph regions for CosyVoice3.

Capture the complete Flow Euler solve and deterministic HiFT decode separately.
Reference preprocessing, fresh noise, CPU F0, oscillator state and request-cache
updates remain outside capture. Exact shapes preserve convolution boundaries
and attention-mask semantics without padding the sequence.
"""

from __future__ import annotations

import torch
from vllm.logger import init_logger

from vllm_omni.model_executor.models.cosyvoice3.runtime import cosyvoice3_stage1_npu_graph_max
from vllm_omni.platforms.npu.graph_tools import NPUExactGraphRunner

logger = init_logger(__name__)
_DISABLE_HINT = "set enforce_eager: true on Stage 1"


def prepare_cosyvoice3_graph_runtime() -> None:
    """Select graph-capturable process-wide NPU runtime behavior."""
    import os

    if os.environ.get("ASCEND_LAUNCH_BLOCKING") == "1":
        raise RuntimeError(
            "CosyVoice3 Stage-1 NPUGraph is incompatible with ASCEND_LAUNCH_BLOCKING=1; "
            "unset it or set enforce_eager: true on Stage 1."
        )
    if not NPUExactGraphRunner.is_supported():
        raise RuntimeError("CosyVoice3 Stage-1 requires torch.npu NPUGraph APIs; " + _DISABLE_HINT)
    torch.npu.config.allow_internal_format = False
    torch.npu.set_compile_mode(jit_compile=False)


def cosyvoice3_fia_blocked_mask(
    keep_mask: torch.Tensor,
    query_len: int,
    key_len: int,
) -> torch.Tensor:
    """Materialize FIA's blocked-mask convention once per Flow solve."""
    if keep_mask.dtype is not torch.bool:
        raise TypeError(f"CosyVoice3 FIA keep_mask must be bool, got {keep_mask.dtype}")
    if keep_mask.ndim != 2 or keep_mask.shape[1] != key_len:
        raise ValueError(f"CosyVoice3 FIA keep_mask must have shape (batch, {key_len}), got {tuple(keep_mask.shape)}")
    return (~keep_mask[:, None, None, :]).expand(keep_mask.shape[0], 1, query_len, key_len).contiguous()


def cosyvoice3_graph_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    blocked_mask: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    """Static noncausal attention using the graph-capturable inference API.

    Ascend 950's SDPA route leaves an unjoined auxiliary stream at capture_end.
    FIA consumes the same key-padding mask, with True meaning masked instead
    of keep. DiT masks padded query outputs after the output projection.
    """
    import torch_npu

    heads = query.shape[2]
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("CosyVoice3 Stage-1 NPU graph attention requires float16 or bfloat16")
    if blocked_mask is not None and blocked_mask.dtype is not torch.bool:
        raise TypeError(f"CosyVoice3 FIA blocked_mask must be bool, got {blocked_mask.dtype}")
    out = torch_npu.npu_fused_infer_attention_score(
        query.transpose(1, 2).contiguous(),
        key.transpose(1, 2).contiguous(),
        value.transpose(1, 2).contiguous(),
        atten_mask=blocked_mask,
        num_heads=heads,
        input_layout="BNSD",
        scale=scale,
    )[0]
    return out.transpose(1, 2)


class CosyVoice3HiFTGraphs:
    """Exact-shape decode graphs returning request-owned waveform storage.

    The shared runner clones graph outputs before another request replays.
    """

    def __init__(self, hift: torch.nn.Module, max_graphs: int) -> None:
        self.hift = hift
        # ISTFT envelopes originate outside capture. The bounded eager cache
        # can evict them later, but recorded tasks must keep their storage alive.
        self._captured_envelopes: dict[tuple[object, ...], torch.Tensor] = {}
        self.runner = NPUExactGraphRunner(
            max_graphs=max_graphs,
            component_name="CosyVoice3 Stage-1 HiFT decode",
            disable_config_hint=_DISABLE_HINT,
        )

    def run(self, mel: torch.Tensor, source: torch.Tensor, finalize: bool) -> torch.Tensor:
        autocast = torch.is_autocast_enabled("npu")
        captures_before = self.runner.stats["captures"]
        touched_keys: set[tuple[object, ...]] = set()
        self.hift._npu_istft_envelope_accesses = touched_keys
        try:
            output = self.runner.run(
                "hift_decode",
                (mel, source),
                (finalize, autocast, torch.get_autocast_dtype("npu") if autocast else None),
                lambda x, s: (self.hift._decode_eager(x, s, finalize=finalize),),
            )[0]
        finally:
            self.hift._npu_istft_envelope_accesses = None
        if self.runner.stats["captures"] > captures_before:
            for key in touched_keys:
                envelope = self.hift._npu_istft_envelopes.get(key)
                if envelope is not None:
                    self._captured_envelopes[key] = envelope
        return output


def enable_cosyvoice3_graphs(decoder: torch.nn.Module, hift: torch.nn.Module) -> None:
    """Install graph regions after weights, evaluation mode and DFT setup."""
    if decoder.training or hift.training:
        raise ValueError("CosyVoice3 Stage-1 NPUGraph requires eval() before capture")
    estimator_dtype = next(decoder.estimator.parameters()).dtype
    if estimator_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            "CosyVoice3 Stage-1 NPU graph adapter requires a float16 or bfloat16 DiT, "
            f"got {estimator_dtype}; {_DISABLE_HINT}."
        )
    prepare_cosyvoice3_graph_runtime()
    max_graphs = cosyvoice3_stage1_npu_graph_max()
    for block in decoder.estimator.transformer_blocks:
        block.attn._use_npu_graph_attention = True
    decoder.estimator._use_npu_graph_attention = True
    decoder._npu_euler_graphs = NPUExactGraphRunner(
        max_graphs=max_graphs,
        component_name="CosyVoice3 Stage-1 Flow Euler",
        disable_config_hint=_DISABLE_HINT,
    )
    # Complex STFT/ISTFT offload to CPU on this backend. Use the existing
    # real-valued DFT implementation so the deterministic decoder is capturable.
    # Each shape's eager warmup primes the ISTFT envelope before capture.
    hift.enable_npu_graph_stft()
    hift._decode_graphs = CosyVoice3HiFTGraphs(hift, max_graphs=max_graphs)
    logger.info(
        "CosyVoice3 Stage-1 NPUGraph enabled: full Flow Euler + HiFT decode; "
        "attention=torch_npu.npu_fused_infer_attention_score (%d DiT blocks); "
        "lazy exact-shape capture, at most %d graphs per region",
        len(decoder.estimator.transformer_blocks),
        max_graphs,
    )
