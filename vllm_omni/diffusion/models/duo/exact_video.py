# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Five-code / seventeen-frame video decode with native boundary padding.

Each push returns its own block. The optional next-block lookahead retains
previous continuous-decoder behavior for comparisons only.
"""

import importlib

import torch


class DuoExactVideoDecoder:
    def __init__(self, model, *, cache_temporal_overlap=False, next_block_lookahead=False):
        if model.causal_decoder:
            raise ValueError("Exact duo_5_17 expects the released noncausal decoder")
        self.model = model
        self.cache_temporal_overlap = cache_temporal_overlap
        self.next_block_lookahead = next_block_lookahead
        self.overlap = None
        self.history = []
        self.received = self.emitted = 0
        self.closed = False
        self.shape = None
        self._rgb_host = self._rgb_ready = None

    def _decode(self, parts, crop):
        model = self.model
        device = next(model.parameters()).device
        z = torch.cat(parts, dim=1)[None].to(device).float()
        z = z * model.latents_std[None, :, None, None, None] + model.latents_mean[None, :, None, None, None]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            if self.cache_temporal_overlap and self.next_block_lookahead:
                # decode_temporal(12) executes [0:7] and [5:12]. The former
                # is the exact seven-code window decoded on the previous push.
                raw = model._adaptive_decode(z[:, :, -7:])
                dtype = importlib.import_module(type(model).__module__)._resolve_temporal_cat_dtype()
                if dtype is not None:
                    raw = raw.to(dtype)
                chunk = model.tokens_chunk_size * model.vae_ratio_t
                padding = model.frame_pre_padding
                pixels = raw[:, :, padding:chunk]
                if self.overlap is not None:
                    pixels = model.blend(self.overlap, pixels, model.frame_overlap, dim=-3)
                self.overlap = raw[:, :, chunk + padding :].contiguous()
                if not torch.isfinite(pixels).all():
                    raise ValueError("Invalid exact Realization video decode")
                pixels = model.processor.revert_tensor(pixels.float())
            else:
                pixels = model.decode_temporal(z)
                expected = {5: 17, 7: 22, 12: 39}[z.shape[2]]
                if pixels.shape[2] != expected or not torch.isfinite(pixels).all():
                    raise ValueError("Invalid exact Realization video decode")
                pixels = model.processor.revert_tensor(pixels.float())[:, :, crop : crop + 17]
        rgb = (pixels[0].permute(1, 2, 3, 0).clamp(0, 1) * 255).round().to(torch.uint8).contiguous()
        rgb = self._copy_rgb_to_cpu(rgb)
        if len(rgb) != 17:
            raise ValueError("Exact Realization decoder must publish seventeen frames")
        result = dict(block=self.emitted, frames=rgb)
        self.emitted += 1
        return result

    def _copy_rgb_to_cpu(self, rgb: torch.Tensor) -> torch.Tensor:
        # Keep pageable allocation/staging out of CUDA D2H. The returned clone
        # is packet-owned; a later push may immediately overwrite the staging area.
        if rgb.device.type == "cpu":
            return rgb
        if self._rgb_host is None:
            self._rgb_host = torch.empty(rgb.shape, dtype=rgb.dtype, device="cpu", pin_memory=True)
            self._rgb_ready = torch.Event(device=rgb.device)
        self._rgb_host.copy_(rgb, non_blocking=True)
        self._rgb_ready.record(torch.accelerator.current_stream(rgb.device))
        self._rgb_ready.synchronize()
        return self._rgb_host.clone()

    def push(self, current):
        if self.closed:
            raise RuntimeError("Exact Realization decoder is already closed")
        if current.ndim != 4 or current.shape[:2] != (24, 5) or not torch.isfinite(current).all():
            raise ValueError("duo_5_17 requires five normalized video codes")
        if self.shape is not None and current.shape != self.shape:
            raise ValueError("Video geometry changed within a session")
        self.shape = current.shape
        z = current.detach().cpu().clone()
        if not self.next_block_lookahead:
            packet = self._decode([z], 0)
            self.received += 1
            return [packet]
        packet = None
        if self.received:
            packet = self._decode([*self.history, z[:, :2]], 0 if self.received == 1 else 17)
        self.history = (self.history + [z])[-2:]
        self.received += 1
        return [] if packet is None else [packet]

    def finish(self):
        if self.closed:
            return []
        packets = []
        if self.next_block_lookahead and self.received:
            packets.append(self._decode([*self.history, self.history[-1][:, -2:]], 0 if self.received == 1 else 17))
        if self.emitted != self.received:
            raise RuntimeError("Exact Realization decoder lost a block")
        self.closed = True
        return packets
