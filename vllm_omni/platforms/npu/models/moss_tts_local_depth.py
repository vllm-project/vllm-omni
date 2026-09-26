# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Inject MOSS-TTS local depth whole-loop NPUGraph acceleration on Ascend.

On NPU, captures the entire n_vq-iteration autoregressive loop of
``generate_frame`` as ONE NPUGraph (per exact batch signature, lazy capture)
via :class:`NPUExactGraphRunner`, replacing eager re-prefill + multinomial
with KV-cache incremental decode + Gumbel-max sampling.

The graph lifecycle (buffers, capture, replay, pooling, failure handling) is
owned entirely by ``NPUExactGraphRunner``; this adapter only supplies the
tensor-only compute body and the dispatch gating.
"""

from __future__ import annotations

from collections.abc import Callable
from weakref import WeakKeyDictionary

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

from vllm_omni.platforms.npu.graph_tools import NPUExactGraphRunner

logger = init_logger(__name__)

_PATCHED = False
_original_setup_compile: Callable | None = None
_original_generate_frame: Callable | None = None
_depth_graph_runners: WeakKeyDictionary[object, NPUExactGraphRunner] = WeakKeyDictionary()

_MAX_GRAPHS = 64


def _apply_topk_topp_mask(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """Top-k/top-p masking as pure tensor ops (graph-capturable).

    Mirrors ``_sample_token``'s masking minus the final ``multinomial`` --
    the whole-loop graph samples via Gumbel-max instead (a captured
    NPUGraph cannot run ``multinomial`` with a per-request generator on
    NPU).
    """
    if top_k and 0 < top_k < logits.shape[-1]:
        top_vals, _ = torch.topk(logits, top_k, dim=-1)
        logits = torch.where(
            logits < top_vals[..., -1:],
            torch.full_like(logits, float("-inf")),
            logits,
        )
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(dim=-1)
        drop = cum > top_p
        drop[..., 1:] = drop[..., :-1].clone()
        drop[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, sorted_idx, sorted_logits)
    return logits


def _make_gumbel_noise(
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    n_vq: int,
    vocab: int,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gumbel(0,1) noise ``(B, n_vq, V)`` and ``(B, 2)`` seeded by ``generator``.

    Generated eagerly per frame (outside the captured graph) so the graph
    has no RNG and per-request reproducibility is preserved (same request
    seed -> same noise -> same Gumbel-max -> same codes).
    """
    u = torch.empty(batch_size, n_vq, vocab, device=device, dtype=torch.float32).uniform_(
        1e-6, 1 - 1e-6, generator=generator
    )
    gc = -torch.log(-torch.log(u))
    ub = torch.empty(batch_size, 2, device=device, dtype=torch.float32).uniform_(1e-6, 1 - 1e-6, generator=generator)
    gb = -torch.log(-torch.log(ub))
    return gc.to(dtype), gb.to(dtype)


def _fwd_incremental_graph(
    depth_model: nn.Module,
    x_c: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    c: int,
) -> torch.Tensor:
    """One-position KV-cache forward for position ``c`` (graph-captured).

    Reuses the existing ``forward(kv_cache=...)`` path but passes an
    ``attn_mask`` so the full ``cache_k`` (fixed shape) is used instead
    of a dynamic ``[:c+1]`` slice -- required for NPUGraph capture.
    """
    n_vq = cache_k.shape[2]
    mask = torch.ones(1, n_vq, dtype=torch.bool, device=x_c.device)
    mask[:, c + 1 :] = False
    return depth_model.ln_f(depth_model.h[0](x_c, (cache_k, cache_v), c, mask))


def _whole_loop_compute(
    depth_model: nn.Module,
    audio_lm_heads: nn.ModuleList,
    audio_embeddings: nn.ModuleList,
    local_text_lm_head: nn.Module,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    text_temperature: float,
    text_top_k: int,
    text_top_p: float,
    n_vq: int,
    backbone_last_hidden: torch.Tensor,
    gumb_codes: torch.Tensor | None = None,
    gumb_bin: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The full n_vq-iteration autoregressive loop as a tensor-only function.

    ``backbone_last_hidden`` ``(B, H)``: position 0 input.
    ``gumb_codes`` ``(B, n_vq, V)`` / ``gumb_bin`` ``(B, 2)``: Gumbel noise
    (``None`` when ``do_sample=False``).  ``do_sample``/``temperature``/``top_k``
    /``top_p``/``text_*`` are captured Python constants (the graph is only
    used when the call's params match the captured ones -- see dispatch in
    ``_patched_generate_frame``).
    """
    B = backbone_last_hidden.shape[0]
    H = depth_model.hidden_size
    dtype = backbone_last_hidden.dtype
    device = backbone_last_hidden.device
    attn = depth_model.h[0].attn

    embeds_buf = torch.zeros(B, n_vq, H, device=device, dtype=dtype)
    cache_k = torch.zeros(B, attn.n_head, n_vq, attn.head_dim, device=device, dtype=dtype)
    cache_v = torch.zeros(B, attn.n_head, n_vq, attn.head_dim, device=device, dtype=dtype)
    codes_buf = torch.zeros(B, n_vq, device=device, dtype=torch.long)
    cont_buf = torch.zeros(B, device=device, dtype=torch.long)

    embeds_buf[:, 0, :].copy_(backbone_last_hidden)

    h0 = _fwd_incremental_graph(depth_model, embeds_buf[:, 0:1], cache_k, cache_v, 0)
    lh = h0[:, 0]
    bl = local_text_lm_head(lh).float()
    if do_sample:
        bl = _apply_topk_topp_mask(bl / text_temperature, text_top_k, text_top_p)
        bin_tok = (bl + gumb_bin).argmax(-1)
    else:
        bin_tok = bl.argmax(-1)
    cont_buf.copy_(bin_tok)

    cl = audio_lm_heads[0](lh).float()
    if do_sample:
        cl = _apply_topk_topp_mask(cl / temperature, top_k, top_p)
        tok = (cl + gumb_codes[:, 0]).argmax(-1)
    else:
        tok = cl.argmax(-1)
    codes_buf[:, 0].copy_(tok)
    embeds_buf[:, 1].copy_(audio_embeddings[0](tok))

    for c in range(1, n_vq):
        hc = _fwd_incremental_graph(depth_model, embeds_buf[:, c : c + 1], cache_k, cache_v, c)
        lh = hc[:, 0]
        cl = audio_lm_heads[c](lh).float()
        if do_sample:
            cl = _apply_topk_topp_mask(cl / temperature, top_k, top_p)
            tok = (cl + gumb_codes[:, c]).argmax(-1)
        else:
            tok = cl.argmax(-1)
        codes_buf[:, c].copy_(tok)
        if c + 1 < n_vq:
            embeds_buf[:, c + 1].copy_(audio_embeddings[c](tok))

    return (cont_buf, codes_buf)


def _patched_setup_compile(self) -> None:
    assert _original_setup_compile is not None
    _original_setup_compile(self)
    if self in _depth_graph_runners:
        return
    from vllm_omni.platforms import current_omni_platform

    if not current_omni_platform.is_npu():
        return
    if not NPUExactGraphRunner.is_supported():
        logger.warning_once("MOSS-TTS local depth NPUGraph not supported on this torch_npu; eager fallback")
        return
    runner = NPUExactGraphRunner(
        max_graphs=_MAX_GRAPHS,
        component_name="MOSS-TTS local depth",
        disable_config_hint="set enforce_eager=True",
    )
    _depth_graph_runners[self] = runner
    logger.info(
        "MOSS-TTS local depth NPU whole-loop graph enabled (max_graphs=%d); lazy capture on first generate_frame",
        _MAX_GRAPHS,
    )


@torch.no_grad()
def _patched_generate_frame(
    self,
    backbone_last_hidden: torch.Tensor,
    audio_lm_heads: nn.ModuleList,
    audio_embeddings: nn.ModuleList,
    local_text_lm_head: nn.Module,
    *,
    n_vq: int,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    text_temperature: float = 1.0,
    text_top_k: int = 50,
    text_top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    history_per_codebook: list[list[int]] | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    runner = _depth_graph_runners.get(self)
    if (
        runner is not None
        and repetition_penalty == 1.0
        and history_per_codebook is None
        and backbone_last_hidden.device.type == "npu"
        and not torch.npu.is_current_stream_capturing()
    ):
        dtype = self.ln_f.weight.dtype
        for block in self.h:
            block.attn.prepare_rope_cache(n_vq, backbone_last_hidden.device, dtype)

        constants = (
            n_vq,
            do_sample,
            temperature,
            top_k,
            top_p,
            text_temperature,
            text_top_k,
            text_top_p,
            id(audio_lm_heads),
            id(audio_embeddings),
            id(local_text_lm_head),
        )

        if do_sample:
            vocab = audio_lm_heads[0].out_features
            gumb_codes, gumb_bin = _make_gumbel_noise(
                backbone_last_hidden.device,
                dtype,
                backbone_last_hidden.shape[0],
                n_vq,
                vocab,
                generator,
            )
            inputs = (backbone_last_hidden.to(dtype), gumb_codes, gumb_bin)

            def compute(bh: torch.Tensor, gc: torch.Tensor, gb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                return _whole_loop_compute(
                    self,
                    audio_lm_heads,
                    audio_embeddings,
                    local_text_lm_head,
                    do_sample,
                    temperature,
                    top_k,
                    top_p,
                    text_temperature,
                    text_top_k,
                    text_top_p,
                    n_vq,
                    bh,
                    gc,
                    gb,
                )

        else:
            inputs = (backbone_last_hidden.to(dtype),)

            def compute(bh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                return _whole_loop_compute(
                    self,
                    audio_lm_heads,
                    audio_embeddings,
                    local_text_lm_head,
                    do_sample,
                    temperature,
                    top_k,
                    top_p,
                    text_temperature,
                    text_top_k,
                    text_top_p,
                    n_vq,
                    bh,
                )

        cont_buf, codes_buf = runner.run("depth_whole_loop", inputs, constants, compute)
        should_continue = cont_buf.eq(0)
        return should_continue, codes_buf

    return _original_generate_frame(
        self,
        backbone_last_hidden,
        audio_lm_heads,
        audio_embeddings,
        local_text_lm_head,
        n_vq=n_vq,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        text_temperature=text_temperature,
        text_top_k=text_top_k,
        text_top_p=text_top_p,
        repetition_penalty=repetition_penalty,
        history_per_codebook=history_per_codebook,
        generator=generator,
    )


def apply_moss_tts_local_depth_patch() -> None:
    """Patch the shared depth transformer with Ascend NPUGraph acceleration."""
    global _PATCHED, _original_setup_compile, _original_generate_frame
    if _PATCHED:
        return

    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_local_depth import (
        MossTTSLocalDepthTransformer,
    )

    _original_setup_compile = MossTTSLocalDepthTransformer.setup_compile
    _original_generate_frame = MossTTSLocalDepthTransformer.generate_frame
    MossTTSLocalDepthTransformer.setup_compile = _patched_setup_compile  # type: ignore[method-assign]
    MossTTSLocalDepthTransformer.generate_frame = _patched_generate_frame  # type: ignore[method-assign]
    _PATCHED = True
    logger.debug("Applied NPU patch for MOSS-TTS local depth whole-loop graph")
