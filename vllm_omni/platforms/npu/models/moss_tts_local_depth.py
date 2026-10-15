# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Inject MOSS-TTS local depth whole-loop NPUGraph acceleration on Ascend.

Captures the entire ``n_vq``-iteration autoregressive loop of
``generate_frame`` as one NPUGraph (per exact batch signature, lazy capture)
via :class:`NPUExactGraphRunner`, replacing eager re-prefill + multinomial
with KV-cache incremental decode + Gumbel-max sampling. The graph only
removes host-side dispatch, so it wins at small batch and falls back to eager
above the crossover gated in ``_patched_generate_frame``. Graph lifecycle is
owned by ``NPUExactGraphRunner``; this adapter supplies only the tensor-only
compute body and the dispatch gating.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
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

# Largest batch that uses the captured whole-loop graph; larger batches stay
# eager. The graph only removes host-side dispatch, so it wins while the
# n_vq-step loop is launch-bound and loses once device time dominates (the
# captured body also pays for fixed-shape full-length attention). Sampling
# crosses over earlier than greedy and is the path real requests take, so gate
# at the last batch that still wins in both modes. Measured A/B table in the PR
# description.
#
# Override with MOSS_TTS_LOCAL_DEPTH_GRAPH_MAX_BATCH (0 disables the graph).
_MAX_GRAPH_BATCH = int(os.environ.get("MOSS_TTS_LOCAL_DEPTH_GRAPH_MAX_BATCH", "16"))


def _apply_topk_topp_mask(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """Top-k/top-p masking as pure tensor ops (graph-capturable).

    Mirrors ``_sample_token``'s masking minus the final ``multinomial`` -- the
    captured graph samples via Gumbel-max instead (a captured NPUGraph cannot
    run ``multinomial`` with a per-request generator on NPU).

    The top-k step scatters the ``topk`` values back into a ``-inf`` tensor so
    the candidate set is exactly the ``k`` indices ``topk`` returned; a
    threshold mask (``logits < kth``) would instead keep every token tied with
    the kth value and widen the set.
    """
    if top_k and 0 < top_k < logits.shape[-1]:
        top_vals, top_indices = torch.topk(logits, top_k, dim=-1)
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, top_indices, top_vals)
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

    Drawn eagerly per frame (outside the captured graph) so the graph has no
    RNG and per-request reproducibility is preserved.
    """
    u = torch.empty(batch_size, n_vq, vocab, device=device, dtype=torch.float32).uniform_(
        1e-6, 1 - 1e-6, generator=generator
    )
    gc = -torch.log(-torch.log(u))
    ub = torch.empty(batch_size, 2, device=device, dtype=torch.float32).uniform_(1e-6, 1 - 1e-6, generator=generator)
    gb = -torch.log(-torch.log(ub))
    return gc.to(dtype), gb.to(dtype)


def _causal_mask_table(n_vq: int, device: torch.device) -> torch.Tensor:
    """``(n_vq, n_vq)`` bool table whose row ``c`` allows positions ``<= c``.

    Built once so the per-step mask is a host-side row view with no device op.
    """
    ar = torch.arange(n_vq, device=device)
    return ar.unsqueeze(0) <= ar.unsqueeze(1)


def _fwd_incremental_graph(
    depth_model: nn.Module,
    x_c: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    c: int,
    mask: torch.Tensor,
) -> torch.Tensor:
    """One-position KV-cache forward for position ``c`` (graph-captured).

    Passes row ``c`` of :func:`_causal_mask_table` as ``attn_mask`` so the full
    fixed-shape ``cache_k`` is attended with a causal mask instead of a dynamic
    ``[:c+1]`` slice -- required for NPUGraph capture.
    """
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
    (``None`` when ``do_sample=False``). ``do_sample``/``temperature``/``top_k``
    /``top_p``/``text_*`` are captured Python constants, and the graph is only
    used when the call's params match the captured ones (see dispatch in
    ``_patched_generate_frame``).
    """
    B = backbone_last_hidden.shape[0]
    H = depth_model.hidden_size
    dtype = backbone_last_hidden.dtype
    device = backbone_last_hidden.device
    attn = depth_model.h[0].attn

    masks = _causal_mask_table(n_vq, device)

    # (n_vq, B, H) so the per-step write embeds_buf[c + 1] = <B, H> is a
    # contiguous row and the read embeds_buf[c].unsqueeze(1) is a contiguous
    # (B, 1, H) view. Every position is written before it is read.
    embeds_buf = backbone_last_hidden.new_empty(n_vq, B, H)
    # The cache is attended at full length (positions > c masked out), so it
    # must stay zero-initialised: a masked softmax cannot undo an uninitialised
    # NaN already baked into the product.
    cache_k = torch.zeros(B, attn.n_head, n_vq, attn.head_dim, device=device, dtype=dtype)
    cache_v = torch.zeros(B, attn.n_head, n_vq, attn.head_dim, device=device, dtype=dtype)

    embeds_buf[0].copy_(backbone_last_hidden)

    h0 = _fwd_incremental_graph(depth_model, embeds_buf[0].unsqueeze(1), cache_k, cache_v, 0, masks[0:1])
    lh = h0[:, 0]
    bl = local_text_lm_head(lh).float()
    # Mirror _sample_token: temperature<=0 (even with do_sample=True) is greedy,
    # a positive temperature is clamped to >= 1e-6. The heads are guarded
    # independently so a mixed mode is exact.
    if do_sample and text_temperature > 0:
        assert gumb_bin is not None
        bl = _apply_topk_topp_mask(bl / max(text_temperature, 1e-6), text_top_k, text_top_p)
        bin_tok = (bl + gumb_bin).argmax(-1)
    else:
        bin_tok = bl.argmax(-1)

    cl = audio_lm_heads[0](lh).float()
    if do_sample and temperature > 0:
        assert gumb_codes is not None
        cl = _apply_topk_topp_mask(cl / max(temperature, 1e-6), top_k, top_p)
        tok = (cl + gumb_codes[:, 0]).argmax(-1)
    else:
        tok = cl.argmax(-1)
    # Pack the codebook tokens in one stack at the end; a per-step strided
    # codes_buf[:, c].copy_(tok) measured slower.
    toks = [tok]
    if n_vq > 1:
        embeds_buf[1].copy_(audio_embeddings[0](tok))

    for c in range(1, n_vq):
        hc = _fwd_incremental_graph(depth_model, embeds_buf[c].unsqueeze(1), cache_k, cache_v, c, masks[c : c + 1])
        lh = hc[:, 0]
        cl = audio_lm_heads[c](lh).float()
        if do_sample and temperature > 0:
            assert gumb_codes is not None
            cl = _apply_topk_topp_mask(cl / max(temperature, 1e-6), top_k, top_p)
            tok = (cl + gumb_codes[:, c]).argmax(-1)
        else:
            tok = cl.argmax(-1)
        toks.append(tok)
        if c + 1 < n_vq:
            embeds_buf[c + 1].copy_(audio_embeddings[c](tok))

    return (bin_tok, torch.stack(toks, dim=1))


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
        "MOSS-TTS local depth NPU whole-loop graph enabled (max_graphs=%d, max_batch=%d); "
        "lazy capture on first generate_frame",
        _MAX_GRAPHS,
        _MAX_GRAPH_BATCH,
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
    generators: Sequence[torch.Generator | None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Per-row generators draw independent Gumbel noise per request, which the
    # single-generator graph path cannot reproduce; stay eager for them.
    per_row_generators = generators is not None and any(gen is not None for gen in generators)
    runner = _depth_graph_runners.get(self)
    if (
        runner is not None
        and repetition_penalty == 1.0
        and history_per_codebook is None
        and not per_row_generators
        and backbone_last_hidden.device.type == "npu"
        and backbone_last_hidden.shape[0] <= _MAX_GRAPH_BATCH
        and not torch.npu.is_current_stream_capturing()
    ):
        active_runner: NPUExactGraphRunner = runner
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

        inputs: tuple[torch.Tensor, ...]
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
        else:
            gumb_codes = gumb_bin = None
            inputs = (backbone_last_hidden.to(dtype),)

        def compute(
            bh: torch.Tensor,
            gc: torch.Tensor | None = None,
            gb: torch.Tensor | None = None,
            *,
            _model: nn.Module = self,
            _heads: nn.ModuleList = audio_lm_heads,
            _embeds: nn.ModuleList = audio_embeddings,
            _text_head: nn.Module = local_text_lm_head,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return _whole_loop_compute(
                _model,
                _heads,
                _embeds,
                _text_head,
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

        cont_buf, codes_buf = active_runner.run("depth_whole_loop", inputs, constants, compute)
        should_continue = cont_buf.eq(0)
        return should_continue, codes_buf

    # Set in `apply_moss_tts_local_depth_patch`, so non-None whenever the patch
    # is active.
    assert _original_generate_frame is not None
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
        generators=generators,
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
