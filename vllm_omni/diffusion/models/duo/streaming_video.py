# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SolarX's immediate causal 5-code decode, using native spatial tile groups."""

import torch

from .exact_video import DuoExactVideoDecoder
from .solarx import TileStreamingDecoder


class DuoStreamingVideoDecoder(DuoExactVideoDecoder):
    def __init__(self, model, *, runtime, block=0):
        super().__init__(model)
        self.runtime = runtime
        self.runtime.reset()
        self.received = self.emitted = block

    def _decode(self, parts, crop):
        model = self.model
        device = next(model.parameters()).device
        z = parts[0][None].to(device).float()
        z = z * model.latents_std[None, :, None, None, None] + model.latents_mean[None, :, None, None, None]
        with torch.inference_mode(), torch.autocast("cuda", enabled=False):
            pixels = torch.cat([self.runtime.push(z[:, :, i : i + 1]) for i in range(5)], dim=2)
            pixels = model.processor.revert_tensor(pixels.float())
            rgb = (pixels[0].permute(1, 2, 3, 0).clamp(0, 1) * 255).round().to(torch.uint8).contiguous()
        result = dict(block=self.emitted, frames=self._copy_rgb_to_cpu(rgb))
        self.emitted += 1
        return result


def make_streaming_runtime(model, group, *, use_graph=True):
    model.decoder.to(dtype=torch.bfloat16)
    model.post_quant_conv.to(dtype=torch.bfloat16)
    return TileStreamingDecoder(model, group=group, use_graph=use_graph)
