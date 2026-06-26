# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Generic, model-agnostic publication of per-forward video geometry.

Structure-aware sparse attention kernels need to map the flat token sequence back
to ``(frame, patch)`` coordinates. The primitives required for that — the
post-VAE pre-patch grid ``(T, H, W)`` and the patch size ``(p_t, p_h, p_w)`` — are
fully determined by the transformer's forward input (the ``(B, C, T, H, W)``
latent) plus the model's patch configuration. So instead of asking every DiT to
publish geometry from inside its ``forward`` (an ``O(models)`` cost that couples
core model code to the sparse-attention feature), the framework registers a single
``forward_pre_hook`` on the transformer module(s) at load time and publishes the
raw primitives onto :class:`~vllm_omni.diffusion.forward_context.ForwardContext`.

Everything downstream is unchanged: the attention bridge surfaces the primitives
onto :class:`PerForwardState`, and the SPARSE_ATTN dispatcher's default resolver
(or a plugin ``geometry_fn``) derives ``total_latent_frames`` / ``patches_per_frame``.

The hook is best-effort: it no-ops for non-patchified models (no ``patch_size``,
or a non-5D input such as image/audio latents), so geometry is simply left unset
and a kernel falls back to its own path — it never breaks a run.
"""

from __future__ import annotations

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.forward_context import set_forward_context_video_geometry

logger = init_logger(__name__)

_GEOMETRY_HOOK_FLAG = "_vllm_omni_video_geometry_hook"


def _resolve_patch_size(module: torch.nn.Module) -> tuple[int, int, int] | None:
    """Normalize a transformer's patch size to ``(p_t, p_h, p_w)``.

    Handles the two conventions in-tree:
      * a single 3-tuple (e.g. Wan ``config.patch_size == (p_t, p_h, p_w)``);
      * a scalar spatial patch plus a separate temporal patch (e.g. HunyuanVideo
        ``patch_size`` + ``patch_size_t``).
    A scalar ``patch_size`` with no ``patch_size_t`` defaults to ``p_t = 1``:
    video DiTs overwhelmingly do not patch time, so an unpatched temporal axis is
    the safer prior than assuming an isotropic patch (which would silently halve
    the published frame count for a model with ``patch_size=2``).
    Reads from ``module.config`` first, then falls back to module attributes, so
    it works whether or not the transformer carries a diffusers-style ``config``.
    Returns ``None`` when no usable patch size is found.
    """
    cfg = getattr(module, "config", None)

    def _get(name: str):
        val = getattr(cfg, name, None) if cfg is not None else None
        return val if val is not None else getattr(module, name, None)

    ps = _get("patch_size")
    pt = _get("patch_size_t")

    if isinstance(ps, (tuple, list)) and len(ps) == 3:
        return int(ps[0]), int(ps[1]), int(ps[2])
    if isinstance(ps, int):
        return int(pt) if pt is not None else 1, int(ps), int(ps)
    return None


def _video_geometry_pre_hook(module: torch.nn.Module, args, kwargs) -> None:
    """forward_pre_hook: publish raw geometry from the transformer's latent input."""
    try:
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and args:
            hidden_states = args[0]
        # Only patchified video latents (B, C, T, H, W) carry a 3D grid.
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.dim() < 5:
            return
        patch_size = _resolve_patch_size(module)
        if patch_size is None:
            return
        t, h, w = (int(s) for s in hidden_states.shape[-3:])
        set_forward_context_video_geometry(latent_shape=(t, h, w), patch_size=patch_size)
    except Exception:  # best-effort: geometry stays unset, never break a forward
        logger.debug("video geometry pre-hook failed; leaving geometry unset", exc_info=True)


def register_video_geometry_hooks(pipeline) -> None:
    """Register the geometry pre-hook on a pipeline's transformer module(s).

    Idempotent (guarded by a flag attribute) and safe to call after
    ``torch.compile`` has replaced the transformer attributes — the pre-hook fires
    before the compiled forward and only reads shapes, so it cannot cause a graph
    break. A no-op for pipelines without a transformer.
    """
    if pipeline is None:
        return
    for attr in ("transformer", "transformer_2"):
        module = getattr(pipeline, attr, None)
        if not isinstance(module, torch.nn.Module):
            continue
        if getattr(module, _GEOMETRY_HOOK_FLAG, False):
            continue
        module.register_forward_pre_hook(_video_geometry_pre_hook, with_kwargs=True)
        setattr(module, _GEOMETRY_HOOK_FLAG, True)
        logger.debug("Registered video geometry pre-hook on pipeline.%s", attr)
