# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Same-worker acoustic batches, using the upstream bounded synthesis job."""

import math
import time
from collections import defaultdict
from itertools import zip_longest

import torch

from .batched_nar import BatchedCachedNAR
from .yue2 import (
    CODEC_OFFSET,
    CODEC_SIZE,
    ODE_STEPS,
    SAMPLE_RATE,
    VAE_CORE_FRAMES,
    VAE_HALO_FRAMES,
    CachedNAR,
    _SynthesisJob,
    _SynthesisQueue,
    chunk_ranges,
    song_chunks,
    synthesize_iter,
)


def decode_tiles_iter(vae, latents, batch_size, core_frames=VAE_CORE_FRAMES, cancelled=None):
    """Batch equal-shaped halo windows, then scatter their exact output cores.

    No padding, cross-song concatenation, or wait to fill a batch. A batch may
    contain tiles from the same song or different songs already in this job.
    """
    if core_frames < 1 or batch_size < 1 or VAE_HALO_FRAMES < vae.required_halo(core_frames):
        raise ValueError("Invalid VAE tile size, batch size, or insufficient halo")
    cancelled = cancelled if cancelled is not None else set()
    waves, buckets = [], defaultdict(list)
    for index, latent in enumerate(latents):
        z = vae._latent(latent.T.unsqueeze(0), check_finite=False)
        frames = z.shape[-1]
        total = vae.natural_output_length(frames)
        wave = z.new_zeros((vae.config.audio_channels, total), dtype=torch.float32)
        waves.append(wave)
        for start in range(0, frames, core_frames):
            end = min(frames, start + core_frames)
            left, right = max(0, start - VAE_HALO_FRAMES), min(frames, end + VAE_HALO_FRAMES)
            ratio = vae.config.downsampling_ratio
            a, b = start * ratio, min(end * ratio, total)
            buckets[right - left].append((index, z[..., left:right], a, b, (start - left) * ratio))
    for tasks in buckets.values():
        while tasks:
            batch, tasks = tasks[:batch_size], tasks[batch_size:]
            batch = [task for task in batch if task[0] not in cancelled]
            if not batch:
                continue
            tiles = batch[0][1] if len(batch) == 1 else torch.cat([task[1] for task in batch])
            decoded = vae.decode(tiles, check_finite=False)
            for row, (index, _, a, b, crop_start) in enumerate(batch):
                crop = decoded[row, :, crop_start : crop_start + b - a]
                if crop.shape[-1] != b - a:
                    raise RuntimeError("VAE tile did not cover its output core")
                waves[index][:, a:b].copy_(crop)
            yield
    return waves


class BatchedSynthesisQueue(_SynthesisQueue):
    """FIFO ready requests; B=1 remains the original NAR arithmetic.

    Cancellation drops a waiting member immediately. An active packed NAR
    keeps a cancelled row until the solve ends; its VAE tiles/output are
    skipped. If all members cancel, the original job releases in-flight work.
    """

    def __init__(self, model, nar_batch_size, vae_batch_size, tile_frames, batch_wait_ms=0, batch_max_tokens=0):
        if (
            isinstance(batch_wait_ms, bool)
            or not isinstance(batch_wait_ms, (int, float))
            or not math.isfinite(batch_wait_ms)
            or batch_wait_ms < 0
        ):
            raise ValueError("YuE2 NAR batch wait must be finite nonnegative milliseconds")
        if isinstance(batch_max_tokens, bool) or not isinstance(batch_max_tokens, int) or batch_max_tokens < 0:
            raise ValueError("YuE2 NAR batch token budget must be a nonnegative integer")
        super().__init__()
        self.batch_wait_s = batch_wait_ms / 1000
        self.batch_max_tokens = batch_max_tokens
        self.model = model
        self.nar_batch_size = nar_batch_size
        self.vae_batch_size = vae_batch_size
        self.tile_frames = tile_frames
        self.members = []
        self.retrying = []
        self.cancelled = set()
        self.sizes = []
        self.ready = {}

    def submit(self, state):
        codec = [token - CODEC_OFFSET for token in state.history]
        if not codec or min(codec) < 0 or max(codec) >= CODEC_SIZE:
            self.ready[state.request_id] = self._payload(torch.zeros((2, 0)), True, state.truncated)
            return
        if self.batch_max_tokens:
            try:
                ranges = chunk_ranges(len(codec), len(state.prefix_ids))
            except ValueError:
                self.ready[state.request_id] = self._payload(torch.zeros((2, 0)), True, state.truncated)
                return
            # Both conditioning and NAR contain the chunk's codec frames.
            state.acoustic_batch_tokens = len(state.prefix_ids) + 2 * max(b - a for a, b in ranges) + 3
        # The runner pumps once after all rows in this AR step are submitted.
        state.acoustic_ready_at = time.monotonic()
        self.waiting.append(state)

    @staticmethod
    def _payload(audio, failed, truncated):
        return {
            # Native output transport may outlive this job. Do not export
            # views of its shared pinned batch buffer to another request.
            "audio": audio.clone(memory_format=torch.contiguous_format),
            "sr": torch.tensor(SAMPLE_RATE, dtype=torch.int32),
            "meta.error": torch.tensor(int(failed)),
            "meta.truncated": torch.tensor(int(truncated)),
        }

    def _work(self, keep):
        model = self.model
        inputs = [(s.prefix_ids, [t - CODEC_OFFSET for t in s.history], s.constants.seed) for s in self.members]
        if len(inputs) == 1:
            latent = yield from synthesize_iter(model, *inputs[0], keep, self.graph_pool)
            latents = [latent]
        else:
            chunks = [song_chunks(*values) for values in inputs]
            pieces = [[] for _ in inputs]
            pool = self.graph_pool.handle(next(model.vae2llm.parameters()).device)
            for row in zip_longest(*chunks):
                owners = [index for index, chunk in enumerate(row) if chunk is not None]
                members = [row[index] for index in owners]
                if len(members) == 1:
                    engine = CachedNAR(model, members[0], pool)
                    keep.append(engine)
                    yield  # Bound condition prefill submission as in the original job.
                else:
                    engine = BatchedCachedNAR(model, members, pool)
                    keep.append(engine)
                    yield from engine.prefill_iter()
                    yield
                value = yield from engine.solve_iter(ODE_STEPS)
                values = [value] if len(members) == 1 else value.split(engine.lengths)
                for index, value in zip(owners, values):
                    pieces[index].append(value)
                engine.last_use = torch.cuda.Event()
                engine.last_use.record()
            latents = [parts[0] if len(parts) == 1 else torch.cat(parts) for parts in pieces]
        keep.extend(latents)
        # Retire completed NAR buffers before allocating VAE workspace.
        yield torch.cuda.Event()
        vae = model._vae_model(latents[0].device)
        waves = yield from decode_tiles_iter(vae, latents, self.vae_batch_size, self.tile_frames, self.cancelled)
        self.sizes = [wave.shape[-1] for wave in waves]
        return waves[0] if len(waves) == 1 else torch.cat(waves, dim=-1)

    def pump(self):
        if self.active is not None:
            if not self.active.done():
                return
            audio, failed = self.active.finish()
            # Retry recoverable packed failures once as B1; unknown CUDA errors stay terminal.
            if (
                failed
                and len(self.members) > 1
                and (self.active.error is None or isinstance(self.active.error, (torch.OutOfMemoryError, ValueError)))
            ):
                self.retrying.extend(state for index, state in enumerate(self.members) if index not in self.cancelled)
            else:
                waves = [torch.zeros((2, 0)) for _ in self.members] if failed else audio.split(self.sizes, dim=-1)
                for index, (state, wave) in enumerate(zip(self.members, waves)):
                    if index not in self.cancelled:
                        self.ready[state.request_id] = self._payload(wave, failed, state.truncated)
            self.active = None
            self.members = []
        if self.retrying:
            self.members = [self.retrying.pop(0)]
        elif self.waiting:
            count, tokens = 0, 0
            for state in self.waiting[: self.nar_batch_size]:
                cost = state.acoustic_batch_tokens if self.batch_max_tokens else 0
                if count and self.batch_max_tokens and tokens + cost > self.batch_max_tokens:
                    break
                count += 1
                tokens += cost
            # Preserve FIFO and let an oversized head run alone; this is a
            # packing budget, not an OOM guarantee or a per-request length cap.
            full = count == self.nar_batch_size or (
                self.batch_max_tokens and (tokens >= self.batch_max_tokens or count < len(self.waiting))
            )
            if (
                not full
                and self.batch_wait_s
                and time.monotonic() < self.waiting[0].acoustic_ready_at + self.batch_wait_s
            ):
                return
            self.members = self.waiting[:count]
            del self.waiting[:count]
        else:
            return
        self.cancelled = set()
        self.sizes = []
        self.active = _SynthesisJob(self.members[0].request_id, self._work)
        self.active.start(self.stream())

    def poll(self):
        self.pump()
        ready, self.ready = self.ready, {}
        return ready

    def cancel_request(self, request_id):
        self.waiting[:] = [state for state in self.waiting if state.request_id != request_id]
        self.retrying[:] = [state for state in self.retrying if state.request_id != request_id]
        self.ready.pop(request_id, None)
        for index, state in enumerate(self.members):
            if state.request_id == request_id:
                self.cancelled.add(index)
        if self.active is not None and len(self.cancelled) == len(self.members):
            self.active.cancel()
            self.active = None
            self.members = []
