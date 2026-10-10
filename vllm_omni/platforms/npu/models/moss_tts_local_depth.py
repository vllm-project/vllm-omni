# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Inject MOSS-TTS local depth whole-loop NPUGraph acceleration on Ascend.

On NPU, captures the entire n_vq-iteration autoregressive loop of
``generate_frame`` as ONE NPUGraph (per exact batch signature, lazy capture)
via :class:`NPUExactGraphRunner`, replacing eager re-prefill + multinomial
with KV-cache incremental decode + Gumbel-max sampling.

The graph's only saving is host-side dispatch: at batch 1 the 12-step loop is
launch-bound and the graph replays in 3.7ms against 5.3ms eager, while at
batch 64 the device time itself exceeds the eager host time, so the dispatch
saving is fully hidden and the graph's fixed-shape full-length attention
costs slightly more device work than eager's dynamic ``[:c+1]`` slice. See
``_MAX_GRAPH_BATCH`` for the measured crossover. Larger batches stay on the
eager path.

The graph lifecycle (buffers, capture, replay, pooling, failure handling) is
owned entirely by ``NPUExactGraphRunner``; this adapter only supplies the
tensor-only compute body and the dispatch gating.
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

# Largest batch that uses the captured whole-loop graph; bigger batches stay eager.
#
# The graph's only saving is host-side dispatch. While the n_vq-iteration loop
# is launch-bound that is a large win; once the device time of the 12 steps
# exceeds the eager host time the saving is completely hidden, and the
# captured body then also pays for its fixed-shape full-length attention,
# which needs a causal mask instead of eager's dynamic [:c+1] slice.
#
# Measured on Ascend 910B2C: one process, one model instance, the only
# difference between arms being whether a runner is registered, so graph
# dispatch vs fall-through to the original eager generate_frame. No RNG
# (do_sample=False for the greedy rows), 8 warm groups of median-of-30 with
# per-call synchronize, reported as the mean. Median ms/call:
#
#     B  |   1      8     16     20     24     32     48     64
#  greedy  3.71   4.31   4.46   4.96   5.14   5.39   6.23   6.45
#   eager  5.27   4.98   4.97   5.06   5.17   5.38   6.06   6.33
#  spd-up 1.42x  1.16x  1.11x  1.02x  1.01x  1.00x  0.97x  0.98x
#
#     B  |   1      8     16     20     24     32     48     64
#  sample  4.88   6.04   6.54   7.19   7.68   8.15   9.66   9.78
#   eager  8.07   7.65   8.36   7.78   7.61   8.07   8.77   9.02
#  spd-up 1.65x  1.27x  1.28x  1.08x  0.99x  0.99x  0.91x  0.92x
#
# The greedy crossover is at B~32 and the sampling one is earlier, at B~20-24
# -- sampling is the path real requests take, since the Gumbel noise is drawn
# eagerly outside the graph. B<=16 is a solid win in both (1.11x-1.65x), so
# gate there; beyond that the two paths are at parity and the graph would only
# add fixed-shape capture and its extra buffers.
#
# Override with MOSS_TTS_LOCAL_DEPTH_GRAPH_MAX_BATCH (0 disables the graph).
_MAX_GRAPH_BATCH = int(os.environ.get("MOSS_TTS_LOCAL_DEPTH_GRAPH_MAX_BATCH", "16"))


def _apply_topk_topp_mask(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """Top-k/top-p masking as pure tensor ops (graph-capturable).

    Mirrors ``_sample_token``'s masking minus the final ``multinomial`` --
    the whole-loop graph samples via Gumbel-max instead (a captured
    NPUGraph cannot run ``multinomial`` with a per-request generator on
    NPU).

    The top-k step keeps the *exact* candidate set returned by ``topk``
    (scattering the selected values back into a -inf-filled tensor), so
    tied logits do not widen the candidate set beyond ``k`` -- matching
    ``_sample_token``, which operates on the compact top-k indices.
    Threshold masking (``logits < kth``) would instead retain every token
    tied with the kth value (e.g. ``[3, 2, 2, 2]`` with ``top_k=2`` keeps
    four candidates, not two).
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


def _causal_mask_table(n_vq: int, device: torch.device) -> torch.Tensor:
    """``(n_vq, n_vq)`` bool table whose row ``c`` allows positions ``<= c``.

    Building the whole table once replaces the per-step
    ``torch.ones(...)`` + ``mask[:, c + 1:] = False`` pair, which cost two
    extra device ops on each of the ``n_vq`` steps. Inside a captured graph
    the table is built a single time and its address is then fixed, so
    ``row = table[c : c + 1]`` is a host-side view with no device cost.
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

    Reuses the existing ``forward(kv_cache=...)`` path but passes an
    ``attn_mask`` (row ``c`` of :func:`_causal_mask_table`, shape ``(1, n_vq)``)
    so the full ``cache_k`` of fixed shape is attended with a causal mask
    instead of a dynamic ``[:c+1]`` slice -- required for NPUGraph capture.
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
    (``None`` when ``do_sample=False``).  ``do_sample``/``temperature``/``top_k``
    /``top_p``/``text_*`` are captured Python constants (the graph is only
    used when the call's params match the captured ones -- see dispatch in
    ``_patched_generate_frame``).
    """
    # `gumb_*` are read only under `do_sample and <head temperature> > 0`,
    # which is exactly the condition under which the dispatch passes them
    # (see `_patched_generate_frame`). Each assert documents that invariant and
    # turns a violated contract into a clear error instead of a `None`
    # subscript; they cost nothing in the captured path.
    B = backbone_last_hidden.shape[0]
    H = depth_model.hidden_size
    dtype = backbone_last_hidden.dtype
    device = backbone_last_hidden.device
    attn = depth_model.h[0].attn

    masks = _causal_mask_table(n_vq, device)

    # (n_vq, B, H) rather than (B, n_vq, H) so the per-step write
    # embeds_buf[c + 1] = <B, H> is one contiguous row instead of a
    # stride-n_vq*H column, and the read embeds_buf[c].unsqueeze(1) is a
    # contiguous (B, 1, H) view. Every position is written before it is read,
    # so uninitialised memory never reaches the model.
    embeds_buf = backbone_last_hidden.new_empty(n_vq, B, H)
    # The K/V cache is read at full length (positions > c are masked out), so
    # it must stay zero-initialised: uninitialised memory can hold NaN and a
    # masked softmax cannot undo a NaN that is already in the product.
    cache_k = torch.zeros(B, attn.n_head, n_vq, attn.head_dim, device=device, dtype=dtype)
    cache_v = torch.zeros(B, attn.n_head, n_vq, attn.head_dim, device=device, dtype=dtype)

    embeds_buf[0].copy_(backbone_last_hidden)

    h0 = _fwd_incremental_graph(depth_model, embeds_buf[0].unsqueeze(1), cache_k, cache_v, 0, masks[0:1])
    lh = h0[:, 0]
    bl = local_text_lm_head(lh).float()
    # Mirror _sample_token per head: temperature<=0 (even with do_sample=True)
    # is greedy, and a positive temperature is clamped to >= 1e-6 to avoid
    # division by zero. The text and audio heads are guarded independently so
    # a mixed mode (e.g. text_temperature<=0 greedy + audio sampled) is exact.
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
    # Collect the codebook tokens and pack once at the end. A per-step
    # codes_buf[:, c].copy_(tok) is a strided device write on every step and
    # measured more expensive than this single pack.
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
    # single-generator graph path cannot reproduce, so stay eager for them.
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

    # Set in `apply_moss_tts_local_depth_patch`, which installs this function as
    # the class method, so it is non-None whenever the patch is active.
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
