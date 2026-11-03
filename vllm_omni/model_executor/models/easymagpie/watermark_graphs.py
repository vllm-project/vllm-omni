# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bounded CUDA graphs for the native, immutable Perth inference processor."""

import torch
import torch.nn.functional as F


class PerthCudaGraphs:
    """Serial codec use only; returned audio owns storage outside the graph pool."""

    def __init__(self, perth_net, sample_rate):
        self.perth_net = perth_net
        self.sample_rate = sample_rate
        self.graphs = {}
        self.stream = None
        self.pool = None

    def __call__(self, originals, process_batch):
        ap = self.perth_net.ap
        if (
            not originals.is_cuda
            or originals.dtype != torch.float32
            or torch.is_autocast_enabled("cuda")
            or type(ap).__module__ != "perth.perth_net.perth_net_implicit.audio_processor"
        ):
            return None
        from perth.perth_net.perth_net_implicit.audio_processor import AudioProcessor

        if type(ap) is not AudioProcessor:
            return None
        inverse = ap.inv_spectrogram
        spec = ap.spectrogram
        if (
            spec.window.dtype != torch.float32
            or inverse.window.dtype != torch.float32
            or spec.window.device != originals.device
            or inverse.window.device != originals.device
            or spec.n_fft != inverse.n_fft
            or spec.hop_length != inverse.hop_length
            or spec.center is not True
            or spec.pad != 0
            or spec.onesided is not True
            or spec.normalized is not False
        ):
            return None
        batch, samples = originals.shape
        bucket = 1 << (batch - 1).bit_length()
        key = (originals.device, bucket, samples)
        if key not in self.graphs:
            if len(self.graphs) >= 64 or (self.stream is not None and self.stream.device != originals.device):
                return None
            rate = int(self.perth_net.hp.sample_rate)
            signal_samples = (samples * rate + self.sample_rate - 1) // self.sample_rate
            if signal_samples < int(self.perth_net.hp.n_fft) // 2 + 1:
                return None
            frames = 1 + (signal_samples + 2 * (spec.n_fft // 2) - spec.n_fft) // spec.hop_length
            prepared = _PreparedInverse(inverse, frames)
            if not prepared.supported:
                return None
            if self.stream is None:
                self.stream = torch.cuda.Stream(device=originals.device)
                self.pool = torch.cuda.graph_pool_handle()
            static = torch.zeros(bucket, samples, device=originals.device, dtype=torch.float32)
            current = torch.cuda.current_stream(originals.device)
            self.stream.wait_stream(current)
            with torch.cuda.stream(self.stream):
                for _ in range(3):
                    process_batch(static, inverse=prepared)
            self.stream.synchronize()
            if prepared.used_fallback:
                return None
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool, stream=self.stream):
                output = process_batch(static, inverse=prepared)
            current.wait_stream(self.stream)
            self.graphs[key] = graph, static, output, prepared
        graph, static, output, _ = self.graphs[key]
        static[:batch].copy_(originals)
        graph.replay()
        return output[:batch].clone()


class _PreparedInverse:
    """Cache and validate the native ISTFT window envelope before capture."""

    def __init__(self, inverse, frames):
        self.inverse = inverse
        self.frames = frames
        self.n_fft = inverse.n_fft
        self.hop = inverse.hop_length
        self.supported = False
        self.used_fallback = False
        if not (
            inverse.center is True
            and inverse.normalized is False
            and inverse.onesided is True
            and inverse.pad == 0
            and self.n_fft > 1
            and 0 < self.hop <= inverse.win_length <= self.n_fft
            and frames > 1
            and inverse.window.shape == (inverse.win_length,)
            and inverse.window.dtype in (torch.float32, torch.float64)
            and not inverse.window.requires_grad
        ):
            return
        self.window = inverse.window.detach().clone()
        left = (self.n_fft - inverse.win_length) // 2
        self.window = F.pad(self.window, (left, self.n_fft - inverse.win_length - left))
        self.samples = self.n_fft + self.hop * (frames - 1)
        self.trim = self.n_fft // 2
        envelope = torch.ops.aten.unfold_backward.default(
            self.window.pow(2).expand(1, frames, self.n_fft), [1, self.samples], 1, self.n_fft, self.hop
        )
        self.envelope = envelope[:, self.trim : -self.trim]
        self.supported = self.envelope.abs().min().item() >= 1e-11

    def __call__(self, spectrum, length=None):
        if (
            not self.supported
            or length is not None
            or not spectrum.is_complex()
            or spectrum.numel() == 0
            or spectrum.shape[-2:] != (self.n_fft // 2 + 1, self.frames)
            or spectrum.device != self.window.device
            or spectrum.real.dtype != self.window.dtype
        ):
            self.used_fallback = True
            return self.inverse(spectrum, length=length)
        shape = spectrum.shape
        frames = torch.fft.irfft(
            spectrum.reshape(-1, shape[-2], shape[-1]).transpose(1, 2), n=self.n_fft, dim=-1, norm="backward"
        )
        frames = frames * self.window.view(1, 1, self.n_fft)
        waveform = torch.ops.aten.unfold_backward.default(
            frames, [frames.shape[0], self.samples], 1, self.n_fft, self.hop
        )
        waveform = waveform[:, self.trim : -self.trim] / self.envelope
        return waveform.reshape(shape[:-2] + waveform.shape[-1:])
