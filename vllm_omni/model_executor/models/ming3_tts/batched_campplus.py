# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CAM++ speaker embeddings of many reference clips in one GPU pass.

The API process ran CAM++ (Kaldi fbank and the ``campplus.onnx`` session) on
the CPU for every reference clip, about 40 ms on each request's path. Stage 0
instead embeds the reference clips of a step's new requests together, with
the same weights moved into the torch CAM++ module:

- Clips are zero-padded to one length. Every CAM++ operation that mixes time
  steps (the convolutions with a time kernel, the CAM context and segment
  means and the statistics pooling) reads the padded positions as zeros or
  leaves them out, so a clip's embedding is the one it gets alone.
- Kaldi fbank frames are independent, so the padded batch is framed once and
  each clip keeps its own frames; the per-clip mean is taken over those.
- Fbank and model replay from CUDA graphs captured per (rows, frames) bucket.
  Resampling to 16 kHz runs before, once per source rate.

The model runs in fp32 with TF32 convolutions and matmuls (cosine >= 0.99999
to the CPU session; the embedding enters the LLM through a bf16 projection).
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.compliance.kaldi as kaldi

from vllm_omni.model_executor.models.indextts2.utils.campplus.dtdnn import CAMPPlus
from vllm_omni.model_executor.models.indextts2.utils.campplus.layers import CAMDenseTDNNBlock, CAMDenseTDNNLayer
from vllm_omni.utils.device_copy import index_to_device, to_device_nonblocking

SAMPLE_RATE = 16000
FRAME_LENGTH, FRAME_SHIFT, FFT_SIZE, MEL_BINS = 400, 160, 512, 80
SEGMENT_FRAMES = 100
# Graph buckets: frames per clip, and clips per pass up to MAX_FRAME_ROWS frames.
FRAME_BUCKETS = (256, 384, 512, 768, 1024, 1536, 2048, 3072)
ROW_BUCKETS = (1, 2, 4, 8)
MAX_FRAME_ROWS = 8192


def num_frames(samples: int) -> int:
    """Kaldi fbank frames of a 16 kHz clip (``snip_edges``)."""
    return 1 + (samples - FRAME_LENGTH) // FRAME_SHIFT


def _bucket_samples(frames: int) -> int:
    return FRAME_LENGTH + FRAME_SHIFT * (frames - 1)


def _frame_bucket(frames: int) -> int | None:
    return next((bucket for bucket in FRAME_BUCKETS if bucket >= frames), None)


def _max_rows(bucket: int) -> int:
    return max((rows for rows in ROW_BUCKETS if rows * bucket <= MAX_FRAME_ROWS), default=1)


def _time_mask(lengths: torch.Tensor, size: int) -> torch.Tensor:
    return (torch.arange(size, device=lengths.device) < lengths[:, None]).float()


class BatchedFbank:
    """``kaldi.fbank(num_mel_bins=80, dither=0)`` of each row of zero-padded 16 kHz audio."""

    def __init__(self, device: torch.device):
        self.window = kaldi._feature_window_function("povey", FRAME_LENGTH, 0.42, device, torch.float32)
        mel, _ = kaldi.get_mel_banks(MEL_BINS, FFT_SIZE, float(SAMPLE_RATE), 20.0, 0.0, 100.0, -500.0, 1.0)
        self.mel = F.pad(mel, (0, 1)).to(device=device, dtype=torch.float32).T.contiguous()

    def __call__(self, waves: torch.Tensor) -> torch.Tensor:
        frames = waves.unfold(-1, FRAME_LENGTH, FRAME_SHIFT)
        frames = frames - frames.mean(-1, keepdim=True)
        frames = frames - 0.97 * torch.cat([frames[..., :1], frames[..., :-1]], dim=-1)
        frames = F.pad(frames * self.window, (0, FFT_SIZE - FRAME_LENGTH))
        power = torch.fft.rfft(frames).abs().pow(2.0)
        return torch.matmul(power, self.mel).clamp_min(torch.finfo(torch.float32).eps).log()


def _cam_dense_layer(layer: CAMDenseTDNNLayer, x: torch.Tensor, mask: torch.Tensor, scale: torch.Tensor, count):
    x = layer.nonlinear2(layer.linear1(layer.nonlinear1(x))) * mask
    cam = layer.cam_layer
    size = x.shape[-1]
    # CAMLayer's context is the clip mean plus the mean of the frame's 100-frame
    # segment (avg_pool1d with ceil_mode divides by the frames present), so the
    # gate is constant over a segment: compute it per segment, then spread it.
    sums = F.pad(x, (0, -size % SEGMENT_FRAMES)).unflatten(-1, (-1, SEGMENT_FRAMES)).sum(-1)
    context = sums.sum(-1, keepdim=True) / count + sums * scale
    gate = cam.sigmoid(cam.linear2(cam.relu(cam.linear1(context))))
    return cam.linear_local(x) * gate.repeat_interleave(SEGMENT_FRAMES, dim=-1)[..., :size]


def masked_campplus(model: CAMPPlus, feats: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """``model`` on ``feats`` [B, T, 80] whose row ``b`` holds ``lengths[b]`` frames; [B, embedding]."""
    head = model.head
    mask = _time_mask(lengths, feats.shape[1])[:, None, None, :]
    out = F.relu(head.bn1(head.conv1(feats.transpose(1, 2).unsqueeze(1) * mask))) * mask
    for block in (*head.layer1, *head.layer2):
        inner = F.relu(block.bn1(block.conv1(out))) * mask
        out = F.relu(block.bn2(block.conv2(inner)) + block.shortcut(out)) * mask
    out = F.relu(head.bn2(head.conv2(out))) * mask

    xvector = model.xvector
    x = xvector.tdnn(out.flatten(1, 2))
    # The TDNN's stride 2 (kernel 5, padding 2) keeps ceil(T / 2) frames.
    lengths = (lengths + 1) // 2
    mask = _time_mask(lengths, x.shape[-1])[:, None, :]
    count = lengths.float()[:, None, None]
    # 1 / frames present in each segment (a padded segment's value is unused).
    present = F.pad(mask, (0, -x.shape[-1] % SEGMENT_FRAMES)).unflatten(-1, (-1, SEGMENT_FRAMES)).sum(-1)
    scale = 1.0 / present.clamp_min(1)
    for name, module in xvector.named_children():
        if name in ("tdnn", "stats", "dense"):
            continue
        if isinstance(module, CAMDenseTDNNBlock):
            for layer in module:
                x = torch.cat([x, _cam_dense_layer(layer, x, mask, scale, count)], dim=1)
        else:
            x = module(x)
    x = x * mask
    mean = x.sum(-1, keepdim=True) / count
    std = (((x - mean) * mask).pow(2).sum(-1, keepdim=True) / (count - 1)).sqrt()
    return xvector.dense(torch.cat([mean, std], dim=1).squeeze(-1))


@contextmanager
def _tf32_math():
    matmul = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=False, allow_tf32=True):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul


class BatchedCampplus:
    def __init__(self, model: CAMPPlus, device: torch.device):
        self.device = torch.device(device)
        self.model = model.to(device=self.device, dtype=torch.float32).eval()
        self.fbank = BatchedFbank(self.device)
        self._resamplers: dict[int, torchaudio.transforms.Resample] = {}
        self._graphs: dict[tuple[int, int], tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    @classmethod
    def from_onnx(cls, onnx_path: str | os.PathLike[str], device: torch.device) -> BatchedCampplus:
        from vllm_omni.model_executor.models.cosyvoice3.speaker_embedding_torch import _build_campplus_fp32

        return cls(_build_campplus_fp32(str(onnx_path)), device)

    def _forward(self, waves: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        feats = self.fbank(waves)
        mask = _time_mask(lengths, feats.shape[1])[..., None]
        mean = (feats * mask).sum(1, keepdim=True) / lengths.float()[:, None, None]
        return masked_campplus(self.model, (feats - mean) * mask, lengths)

    def capture_graphs(self) -> None:
        pool = torch.cuda.graph_pool_handle()
        with torch.inference_mode(), _tf32_math():
            # Largest first, so the smaller graphs reuse the pool's blocks.
            for frames in sorted(FRAME_BUCKETS, reverse=True):
                for rows in sorted(ROW_BUCKETS, reverse=True):
                    if rows * frames > MAX_FRAME_ROWS:
                        continue
                    waves = torch.zeros((rows, _bucket_samples(frames)), device=self.device)
                    lengths = torch.full((rows,), frames, dtype=torch.long, device=self.device)
                    stream = torch.cuda.Stream(self.device)
                    stream.wait_stream(torch.cuda.current_stream(self.device))
                    with torch.cuda.stream(stream):
                        self._forward(waves, lengths)
                    torch.cuda.current_stream(self.device).wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=pool):
                        output = self._forward(waves, lengths)
                    self._graphs[(rows, frames)] = (graph, waves, lengths, output)

    def _resample(self, clips: list[tuple[torch.Tensor, int]]) -> list[torch.Tensor]:
        """Each clip at 16 kHz on the device, resampled together per source rate."""
        out: list[torch.Tensor | None] = [None] * len(clips)
        by_rate: dict[int, list[int]] = {}
        for index, (_, rate) in enumerate(clips):
            by_rate.setdefault(int(rate), []).append(index)
        for rate, indices in by_rate.items():
            waves = [to_device_nonblocking(clips[i][0].reshape(-1), self.device).float() for i in indices]
            if rate == SAMPLE_RATE:
                for i, wave in zip(indices, waves, strict=True):
                    out[i] = wave
                continue
            resampler = self._resamplers.get(rate)
            if resampler is None:
                resampler = torchaudio.transforms.Resample(orig_freq=rate, new_freq=SAMPLE_RATE).to(self.device)
                self._resamplers[rate] = resampler
            # Zero padding after a clip is what the resampler pads it with alone.
            padded = torch.nn.utils.rnn.pad_sequence(waves, batch_first=True)
            resampled = resampler(padded)
            for row, (i, wave) in enumerate(zip(indices, waves, strict=True)):
                out[i] = resampled[row, : -(-SAMPLE_RATE * wave.numel() // rate)]
        return out

    @torch.inference_mode()
    def embed(self, clips: list[tuple[torch.Tensor, int]]) -> list[torch.Tensor]:
        """Float32 embeddings on the device of ``(waveform, sample rate)`` clips."""
        if not clips:
            return []
        waves = self._resample(clips)
        frames = [num_frames(wave.numel()) for wave in waves]
        if min(frames) < 2:
            raise ValueError("CAM++ needs at least 2 fbank frames (25 ms) of reference audio")
        out: list[torch.Tensor | None] = [None] * len(clips)
        order = sorted(range(len(clips)), key=frames.__getitem__)
        start = 0
        while start < len(order):
            # The longest clip of a group sets its frame bucket and how many rows fit.
            stop = start + 1
            while stop < len(order):
                bucket = _frame_bucket(frames[order[stop]])
                if bucket is None or stop - start + 1 > _max_rows(bucket):
                    break
                stop += 1
            group = order[start:stop]
            embeddings = self._embed_group([waves[i] for i in group], [frames[i] for i in group])
            for index, embedding in zip(group, embeddings, strict=True):
                out[index] = embedding
            start = stop
        return out

    def _embed_group(self, waves: list[torch.Tensor], frames: list[int]) -> torch.Tensor:
        bucket = _frame_bucket(max(frames)) or max(frames)
        rows = next((r for r in ROW_BUCKETS if r >= len(waves) and r * bucket <= MAX_FRAME_ROWS), len(waves))
        entry = self._graphs.get((rows, bucket))
        if entry is None:
            # Eager on the same bucketed shapes, so cuDNN reuses its plans.
            static = torch.zeros((rows, _bucket_samples(bucket)), device=self.device)
            lengths = torch.full((rows,), bucket, dtype=torch.long, device=self.device)
        else:
            graph, static, lengths, output = entry
            static.zero_()
        for row, (wave, count) in enumerate(zip(waves, frames, strict=True)):
            used = _bucket_samples(count)
            static[row, :used].copy_(wave[:used])
        lengths.copy_(index_to_device(frames + [bucket] * (rows - len(waves)), self.device))
        if entry is None:
            with _tf32_math():
                output = self._forward(static, lengths)
        else:
            graph.replay()
        return output[: len(waves)].clone()


__all__ = ["BatchedCampplus", "BatchedFbank", "masked_campplus", "num_frames"]
