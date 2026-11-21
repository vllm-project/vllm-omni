# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""LTX-2 placement boundary: permute before SP hooks, undo after final gather."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from vllm_omni.diffusion.attention.sparsp.placement import (
    SparSPVideoLayout,
    interval_periodic_layout,
    supports_interval_periodic_layout,
)
from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available

if TYPE_CHECKING:
    from vllm_omni.diffusion.models.ltx2.ltx2_transformer import LTX2VideoTransformer3DModel


class LTX2SparSPLayout:
    def __init__(self, model: LTX2VideoTransformer3DModel) -> None:
        self.signature = inspect.signature(model.forward)
        self.layout: SparSPVideoLayout | None = None
        self.active = False
        self.previous: SparSPVideoLayout | None = None
        # Native module hooks surround the ModelHook-wrapped forward, so the
        # permutation precedes existing root SP splitting and RoPE-output hooks.
        model.register_forward_pre_hook(self.before, with_kwargs=True)
        model.register_forward_hook(self.after, with_kwargs=True, always_call=True)

    def before(self, model: LTX2VideoTransformer3DModel, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        if not is_forward_context_available():
            return args, kwargs
        ctx = get_forward_context()
        specs = [block.attn1.attn.effective_attention()[2] for block in model.transformer_blocks]
        placements = [spec.sparsp.placement if spec and spec.sparsp else "none" for spec in specs]
        if not any(placement in ("interval_periodic", "auto") for placement in placements):
            return args, kwargs
        if self.active or len(set(placements)) != 1:
            raise ValueError("LTX-2 SparSP placement requires serialized calls and the same policy on all video layers")
        config = ctx.omni_diffusion_config
        if config is None or config.diffusion_attention_schedule is not None:
            raise ValueError("LTX-2 SparSP placement does not support attention scheduling")
        bound = self.signature.bind_partial(*args, **kwargs)
        hidden = bound.arguments["hidden_states"]
        grid = (bound.arguments.get("num_frames"), bound.arguments.get("height"), bound.arguments.get("width"))
        if any(type(value) is not int for value in grid):
            if placements[0] == "auto":
                return args, kwargs
            raise ValueError("LTX-2 SparSP placement requires explicit num_frames, height and width")
        degree = config.parallel_config.ring_degree
        if hidden.shape[1] != grid[0] * grid[1] * grid[2]:
            raise ValueError("LTX-2 SparSP placement requires a complete unpadded video grid")
        if placements[0] == "auto" and not supports_interval_periodic_layout(grid, degree):
            return args, kwargs
        if (
            self.layout is None
            or self.layout.grid != grid
            or self.layout.degree != degree
            or self.layout.permutation.device != hidden.device
        ):
            self.layout = interval_periodic_layout(grid, degree, hidden.device)
        order = self.layout.permutation
        coordinates = bound.arguments.get("video_coords")
        if coordinates is None:
            coordinates = model.rope.prepare_video_coords(
                hidden.shape[0], *grid, hidden.device, fps=bound.arguments.get("fps", 24.0)
            )
        if coordinates.ndim != 4 or coordinates.shape[2] != hidden.shape[1]:
            raise ValueError("LTX-2 SparSP expects video_coords in [B,3,S,2] layout")
        bound.arguments["hidden_states"] = hidden.index_select(1, order)
        bound.arguments["video_coords"] = coordinates.index_select(2, order)
        for name in ("timestep", "keyframes_mask"):
            tensor = bound.arguments.get(name)
            if tensor is not None and tensor.ndim > 1 and tensor.shape[1] == hidden.shape[1]:
                bound.arguments[name] = tensor.index_select(1, order)
        self.previous = ctx.sparsp_video_layout
        ctx.sparsp_video_layout = self.layout
        self.active = True
        return bound.args, bound.kwargs

    def after(self, model: LTX2VideoTransformer3DModel, args: tuple, kwargs: dict, output: object) -> object:
        del model, args, kwargs
        if not self.active:
            return output
        try:
            if output is None:
                return output
            if self.layout is None:
                raise RuntimeError("active LTX-2 SparSP placement lost its layout")
            inverse = self.layout.inverse
            if isinstance(output, tuple):
                return (output[0].index_select(1, inverse), *output[1:])
            from vllm_omni.diffusion.models.ltx2.ltx2_transformer import AudioVisualModelOutput

            if not isinstance(output, AudioVisualModelOutput):
                raise TypeError("LTX-2 SparSP expected AudioVisualModelOutput or a video/audio tuple")
            output.sample = output.sample.index_select(1, inverse)
            return output
        finally:
            get_forward_context().sparsp_video_layout = self.previous
            self.previous = None
            self.active = False
