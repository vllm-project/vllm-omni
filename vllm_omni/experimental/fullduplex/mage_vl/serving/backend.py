# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Transformers backend for the Mage-VL production WebSocket transport."""

from __future__ import annotations

import asyncio
import binascii
import contextlib
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pybase64 as base64

from vllm_omni.experimental.fullduplex.core.session import DuplexSession
from vllm_omni.experimental.fullduplex.mage_vl import (
    MageVLCodecWindow,
    MageVLDuplexAdapter,
    MageVLGateDecision,
)

_STREAMING_PROMPT = "Please describe the video content in detail based on the provided information."


class MageVLTransformersBackend:
    """One shared checkpoint with serialized GPU access and isolated adapters."""

    def __init__(
        self,
        checkpoint: str = "microsoft/Mage-VL",
        *,
        device: str = "cuda",
        attn_impl: str = "sdpa",
        video_backend: str = "frames",
        num_frames: int = 8,
        target_fps: float = 1.0,
        max_new_tokens: int = 128,
        gate_threshold: float = 0.5,
        window_size: int = 4,
        max_windows: int = 64,
    ) -> None:
        if video_backend not in {"frames", "codec"}:
            raise ValueError("video_backend must be frames or codec")
        if video_backend == "codec" and shutil.which("cv-preinfer") is None:
            raise RuntimeError(
                "video_backend='codec' requires cv-preinfer from codec-video-prep>=0.2.5; "
                "install it and ensure cv-preinfer is on PATH, or use video_backend='frames'"
            )
        if num_frames <= 0 or target_fps <= 0 or max_new_tokens <= 0:
            raise ValueError("num_frames, target_fps, and max_new_tokens must be positive")
        self.checkpoint = checkpoint
        self.device = device
        self.attn_impl = attn_impl
        self.video_backend = video_backend
        self.num_frames = num_frames
        self.target_fps = target_fps
        self.max_new_tokens = max_new_tokens
        self.gate_threshold = gate_threshold
        self.window_size = window_size
        self.max_windows = max_windows
        self.model: Any = None
        self.processor: Any = None
        self._inference_lock = asyncio.Lock()

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.checkpoint, trust_remote_code=True)
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.checkpoint,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                attn_implementation=self.attn_impl,
            )
            .to(self.device)
            .eval()
        )
        if not hasattr(self.model, "streammind_gate_forward_segments"):
            raise AttributeError("Mage-VL checkpoint does not expose streammind_gate_forward_segments()")

    def adapter_factory(self) -> MageVLDuplexAdapter:
        if self.model is None or self.processor is None:
            raise RuntimeError("MageVLTransformersBackend.load() must be called before serving")
        return MageVLDuplexAdapter(
            gate=self.gate,
            generate=self.generate,
            window_size=self.window_size,
            max_windows=self.max_windows,
        )

    async def gate(
        self,
        _session: DuplexSession,
        windows: Sequence[MageVLCodecWindow],
    ) -> MageVLGateDecision:
        score = await self._run_inference(self._gate_sync, windows)
        latest = windows[-1]
        return MageVLGateDecision(
            should_respond=score >= self.gate_threshold,
            event_id=latest.segment_id,
            score=score,
            reason="streammind_gate",
        )

    async def generate(
        self,
        _session: DuplexSession,
        windows: Sequence[MageVLCodecWindow],
        query: str | None,
        _decision: MageVLGateDecision | None,
    ) -> str:
        return await self._run_inference(self._generate_sync, windows[-1], query or _STREAMING_PROMPT)

    async def _run_inference(self, function, *args):
        """Serialize GPU calls and drain worker threads before cancellation."""
        async with self._inference_lock:
            worker = asyncio.create_task(asyncio.to_thread(function, *args))
            try:
                return await asyncio.shield(worker)
            except asyncio.CancelledError:
                # Cancelling an asyncio wrapper cannot stop a CUDA worker thread.
                # Retain the lock until it really exits so a replacement request
                # cannot race the same shared model.
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await worker
                raise

    def _gate_sync(self, windows: Sequence[MageVLCodecWindow]) -> float:
        import torch

        prepared: list[tuple[dict[str, Any], Path | None]] = []
        try:
            prepared = [self._prepare(window, _STREAMING_PROMPT) for window in windows]
            visual = [self._gate_inputs(item[0]) for item in prepared]
            with torch.inference_mode():
                logits = self.model.streammind_gate_forward_segments(visual)[0]
            lengths = [int(segment["image_grid_thw"][:, 0].sum()) for segment in visual]
            boundary = sum(lengths) - 1
            return float(torch.softmax(logits[boundary].float(), dim=-1)[1].item())
        finally:
            for _, temporary_path in prepared:
                _remove_temporary(temporary_path)

    def _generate_sync(self, window: MageVLCodecWindow, query: str) -> str:
        import torch

        inputs, temporary_path = self._prepare(window, query)
        try:
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            new_tokens = output[0, inputs["input_ids"].shape[1] :]
            return self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        finally:
            _remove_temporary(temporary_path)

    def _prepare(self, window: MageVLCodecWindow, query: str) -> tuple[dict[str, Any], Path | None]:
        video_path, temporary_path = _materialize_video(window.data)
        messages = [
            {
                "role": "user",
                "content": [{"type": "video"}, {"type": "text", "text": query}],
            }
        ]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        kwargs: dict[str, Any] = {
            "text": [prompt],
            "videos": [str(video_path)],
            "video_backend": self.video_backend,
            "return_tensors": "pt",
            "padding": False,
        }
        if self.video_backend == "codec":
            kwargs.update(codec_config={"patch": 16, "max_pixels": 150000}, max_pixels=150000)
        else:
            kwargs.update(num_frames=self.num_frames, target_fps=self.target_fps)
        try:
            inputs = self.processor(**kwargs)
            inputs = {
                key: (
                    value.to(device=self.device, dtype=self.model.dtype)
                    if key == "pixel_values"
                    else value.to(self.device)
                )
                for key, value in inputs.items()
            }
            return inputs, temporary_path
        except Exception:
            _remove_temporary(temporary_path)
            raise

    @staticmethod
    def _gate_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
        return {key: inputs[key] for key in ("pixel_values", "image_grid_thw", "patch_positions") if key in inputs}


def _materialize_video(data: Any) -> tuple[Path, Path | None]:
    if not isinstance(data, dict):
        raise ValueError("video window data must be an object containing video_base64")
    encoded = data.get("video_base64")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("video window requires non-empty video_base64")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("video_base64 is not valid base64") from error
    if not payload:
        raise ValueError("decoded video payload is empty")
    with tempfile.NamedTemporaryFile(prefix="mage-vl-", suffix=".mp4", delete=False) as handle:
        handle.write(payload)
        path = Path(handle.name)
    return path, path


def _remove_temporary(path: Path | None) -> None:
    if path is not None:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
