# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in, bounded ACLGraphs for the API-process MOSS Local reference encoder."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator

import torch
from vllm.logger import init_logger

from vllm_omni.model_executor.models.moss_tts.reference_encoder_graphs import MossReferenceEncoderGraphs
from vllm_omni.platforms.npu.graph_tools import NPUExactGraphRunner

logger = init_logger(__name__)


class MossNPUReferenceEncoderGraphs(MossReferenceEncoderGraphs):
    """Retain true clip lengths while padding to a small, fixed graph set.

    The upstream tokenizer's SDPA attention is retained; the CUDA windowed
    FlashAttention replacement is never installed. Calls are serialized by the
    local reference batcher. Shared-host graph workers are deliberately excluded.
    """

    def __init__(self, tokenizer: torch.nn.Module, *, n_vq: int) -> None:
        super().__init__(tokenizer, n_vq=n_vq, batch_sizes=(1, 2), bucket_seconds=(4, 8), max_batch_seconds=8)
        # Bound padded shapes by the frame-rounded eight-second bucket.
        self._max_batch_samples = self._buckets[-1]
        self._shapes = [
            (batch, samples)
            for batch in self._batch_sizes
            for samples in self._buckets
            if batch * samples <= self._max_batch_samples
        ]
        self._buffers: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self._runner = NPUExactGraphRunner(
            max_graphs=len(self._shapes),
            component_name="MOSS Local reference encoder",
            disable_config_hint="set VLLM_OMNI_MOSS_REF_GRAPHS=0",
        )
        self._stream = None
        self._failed = False

    @property
    def captured(self) -> list[tuple[int, int]]:
        return sorted(self._buffers)

    @property
    def stats(self) -> dict[str, int]:
        return self._runner.stats

    def _check_healthy(self) -> None:
        if self._failed:
            raise RuntimeError(
                "MOSS reference NPUGraph failed; restart the API process with VLLM_OMNI_MOSS_REF_GRAPHS=0"
            )

    @contextlib.contextmanager
    def _context(self) -> Iterator[None]:
        npu = torch.get_device_module("npu")
        with npu.device(self._device):
            if self._stream is None:
                self._stream = npu.Stream(device=self._device)
            current = npu.current_stream(self._device)
            self._stream.wait_stream(current)
            with npu.stream(self._stream), torch.no_grad():
                try:
                    yield
                finally:
                    current.wait_stream(self._stream)

    def _core(self, audio: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = getattr(self._tokenizer, "compute_dtype", None)
        # Match eager's outer autocast, including the quantizer, around the shared core.
        amp = torch.autocast(device_type="npu", dtype=dtype) if dtype in (torch.float16, torch.bfloat16) else None
        with amp if amp is not None else contextlib.nullcontext():
            return super()._core(audio, lengths)

    def _run(
        self, inputs: tuple[torch.Tensor, torch.Tensor], shape: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._check_healthy()
        try:
            return self._runner.run("moss_reference_encode", inputs, shape, self._core)
        except RuntimeError:
            # Capture failures can invalidate allocator/RNG state. Replay errors
            # are also terminal: the caller must not retry eagerly in this process.
            self._failed = True
            raise

    def capture(self) -> None:
        self._check_healthy()
        if self._buffers:
            return
        if os.environ.get("ASCEND_LAUNCH_BLOCKING") == "1":
            raise ValueError("MOSS reference ACLGraph requires ASCEND_LAUNCH_BLOCKING to be unset or 0")
        if not self._runner.is_supported():
            return
        npu = torch.get_device_module("npu")
        with self._context():
            # Select graph-capturable ACLNN kernels in this API process only.
            # The codec worker configures its own process separately.
            npu.config.allow_internal_format = False
            npu.set_compile_mode(jit_compile=False)
            for batch, samples in reversed(self._shapes):
                audio = torch.zeros(batch, int(self._tokenizer.number_channels), samples, device=self._device)
                lengths = torch.full((batch,), samples, dtype=torch.long, device=self._device)
                self._run((audio, lengths), (batch, samples))
                self._buffers[(batch, samples)] = (audio, lengths)
        logger.info("MOSS Local reference ACLGraph capture complete: shapes=%s stats=%s", self.captured, self.stats)

    def encode(self, wavs: list[torch.Tensor]) -> list[torch.Tensor | None]:
        self._check_healthy()
        # Invalid clips remain the eager processor's responsibility.
        indices = [
            i for i, wav in enumerate(wavs) if wav.ndim == 2 and wav.shape[0] == self._channels and wav.shape[-1] > 0
        ]
        out: list[torch.Tensor | None] = [None] * len(wavs)
        with self._context():
            for index, codes in zip(indices, super().encode([wavs[i] for i in indices]), strict=True):
                out[index] = codes
        return out

    def _replay_bucket(self, batch: int, samples: int, wavs: list[torch.Tensor]) -> list[torch.Tensor]:
        audio, lengths = self._buffers[(batch, samples)]
        audio.zero_()
        true_lengths = [wav.shape[-1] for wav in wavs] + [samples] * (batch - len(wavs))
        lengths.copy_(torch.tensor(true_lengths, dtype=torch.long, device=audio.device))
        for row, wav in enumerate(wavs):
            audio[row, :, : wav.shape[-1]].copy_(wav)
        codes, code_lengths = self._run((audio, lengths), (batch, samples))
        codes = codes[:, : len(wavs)].cpu()
        code_lengths = code_lengths[: len(wavs)].cpu().tolist()
        return [codes[:, row, : int(code_lengths[row])].transpose(0, 1).long().contiguous() for row in range(len(wavs))]
