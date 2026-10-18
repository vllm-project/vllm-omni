# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native Matrix-Game 4.0 rendering, one resident session per worker.

The engine schedules ordinary requests. A request can render a complete clip or
advance a named session by one chunk; session KV and codec state are model-owned.
There is no dependency on the reference repository at inference time.
"""

from __future__ import annotations

import gc
import html
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import ftfy
import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from transformers import AutoTokenizer

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.media import (
    DiffusionMediaOutput,
    VideoMediaOutput,
    VideoTensorEncoding,
    VideoTensorLayout,
    VideoTensorSpec,
    VideoValueRange,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.platforms import current_omni_platform
from vllm_omni.quantization.fp8_config import DiffusionFp8Config

if TYPE_CHECKING:
    from .stream import MatrixStream

DEFAULT_PROMPT = (
    "a third-person shooter game in a medieval castle with stone walls, towers and courtyards, "
    "stylized cartoon characters firing glowing projectiles, trees and grass, snowy mountains "
    "in the distance, realistic AAA game rendering, physically based materials, natural daylight, "
    "detailed vegetation and terrain"
)
SESSION_KEY = "matrix_session_id"
START_KEY = "matrix_frame_start"
END_KEY = "matrix_end_session"
_REQUEST_KEYS = {SESSION_KEY, START_KEY, END_KEY}
_FRAME_SIZES = {(480, 864), (736, 1280)}
_VIDEO_SPEC = VideoTensorSpec(
    VideoTensorLayout.BCTHW, VideoTensorEncoding.NORMALIZED_FLOAT, VideoValueRange.NEGATIVE_ONE_TO_ONE
)


@dataclass(frozen=True)
class MatrixInput:
    frames: torch.Tensor | None
    session_id: str | None
    frame_start: int
    end_session: bool
    steps: int


def prepare_request(request: OmniDiffusionRequest, *, height: int = 480, width: int = 864) -> OmniDiffusionRequest:
    params = request.sampling_params
    if params.height not in (None, height) or params.width not in (None, width):
        raise ValueError(f"Matrix-Game worker requires height={height}, width={width}")
    if params.num_outputs_per_prompt != 1 or params.guidance_scale not in (None, 1.0):
        raise ValueError("Matrix-Game 4.0 supports one output and guidance_scale=1")
    if params.generator is not None:
        raise ValueError("Use seed for Matrix-Game's separate CPU and device generators")
    steps = 1 if params.num_inference_steps is None else params.num_inference_steps
    if type(steps) is not int or steps not in (1, 2):
        raise ValueError("Matrix-Game 4.0 supports one or two denoising steps")
    if params.seed is None:
        params.seed = 0
    if params.output_type not in (None, "np"):
        raise ValueError("Matrix-Game 4.0 supports decoded RGB output (output_type='np')")
    unknown = params.extra_args.keys() - _REQUEST_KEYS
    if unknown:
        raise ValueError(f"Unknown Matrix-Game request options: {sorted(unknown)}")
    session_id = params.extra_args.get(SESSION_KEY)
    frame_start = params.extra_args.get(START_KEY, 0)
    end_session = params.extra_args.get(END_KEY, session_id is None)
    if session_id is not None and (not isinstance(session_id, str) or not 1 <= len(session_id) <= 128):
        raise ValueError("matrix_session_id must be a nonempty string of at most 128 characters")
    if type(frame_start) is not int or frame_start < 0 or type(end_session) is not bool:
        raise ValueError("matrix_frame_start must be a nonnegative integer; matrix_end_session must be boolean")
    if session_id is None and (frame_start or not end_session):
        raise ValueError("Clip requests start at zero and close automatically")
    if not isinstance(request.prompt, dict) or request.prompt.get("negative_prompt"):
        raise ValueError("Provide a prompt dictionary with multi_modal_data.video; negative prompts are unsupported")
    text = request.prompt.get("prompt", "")
    if not isinstance(text, str) or len(text) > 8192 or (text and not text.strip()):
        raise ValueError("prompt must be empty (retain caption) or a nonempty string of at most 8192 characters")
    media = request.prompt.get("multi_modal_data", {})
    if not isinstance(media, dict) or media.keys() - {"video"}:
        raise ValueError("Matrix-Game accepts only multi_modal_data.video")
    frames = media.get("video")
    if frames is None:
        if session_id is None or not end_session:
            raise ValueError("Missing RGB frames; omit video only to close a named session")
    else:
        if not isinstance(frames, torch.Tensor) or frames.device.type != "cpu" or frames.dtype != torch.uint8:
            raise ValueError(f"video must be a CPU uint8 RGB tensor with shape [T,{height},{width},3]")
        if frames.ndim != 4 or tuple(frames.shape[1:]) != (height, width, 3):
            raise ValueError(f"video must have shape [T,{height},{width},3]")
        count = frames.shape[0]
        if session_id is None:
            if count < 1 or (count - 1) % 4:
                raise ValueError("A complete clip must contain 1+4k frames")
        elif count != (1 if frame_start == 0 else 4):
            raise ValueError("A session starts with one frame, then accepts four frames per request")
    request.prepared_layout = MatrixInput(frames, session_id, frame_start, end_session, steps)
    params.height, params.width = height, width
    params.fps = params.fps or 60
    return request


def get_matrix_game4_pre_process_func(od_config: OmniDiffusionConfig) -> Callable:
    height, width = _frame_size(od_config)
    return partial(prepare_request, height=height, width=width)


def _frame_size(config: OmniDiffusionConfig) -> tuple[int, int]:
    height, width = config.model_config.get("height", 480), config.model_config.get("width", 864)
    if type(height) is not int or type(width) is not int or (height, width) not in _FRAME_SIZES:
        raise ValueError("Matrix-Game frame size must be 864x480 or experimental 1280x736")
    return height, width


def validate_config(config: OmniDiffusionConfig) -> None:
    if config.num_gpus != 1 or config.parallel_config.world_size != 1 or config.max_num_seqs != 1:
        raise ValueError("Matrix-Game 4.0 requires one GPU and max_num_seqs=1")
    if config.dtype != torch.bfloat16 or not config.enforce_eager:
        raise ValueError("Use dtype=bfloat16 and enforce_eager=True; generic graph wrapping is not qualified")
    if any(
        (
            config.enable_cpu_offload,
            config.enable_layerwise_offload,
            config.enable_distributed_layerwise_offload,
            config.enable_sleep_mode,
            config.diffusion_offload_config is not None,
        )
    ):
        raise ValueError("Matrix-Game currently requires resident weights; offload and sleep are unsupported")
    if config.step_execution or config.streaming_output:
        raise ValueError("Submit consecutive chunk requests; diffusion step execution is unsupported")
    if config.lora_path:
        raise ValueError("Matrix-Game does not support LoRA")
    if config.quantization_config is not None and not isinstance(config.quantization_config, DiffusionFp8Config):
        raise ValueError("Matrix-Game supports only online quantization_config='fp8' or None")
    if config.cache_backend not in (None, "none") or config.enable_prefix_caching:
        raise ValueError("Matrix-Game owns its temporal KV cache; generic cache acceleration is unsupported")
    for role in ("self", "cross"):
        spec, _ = config.diffusion_attention_config.resolve_with_source(role)
        if spec is None or spec.backend.upper() not in {"TORCH_SDPA", "CUDNN_ATTN", "FLASH_ATTN"}:
            raise ValueError(
                "Matrix-Game requires explicit TORCH_SDPA, CUDNN_ATTN or FLASH_ATTN attention for self and cross; "
                "use TORCH_SDPA for the exact-reference path"
            )
    options = config.model_config
    if options.keys() - {"prompt", "prompt_mode", "max_latents", "scene", "cuda_graph", "height", "width"}:
        raise ValueError(
            "Matrix-Game model_config accepts only prompt, prompt_mode, max_latents, scene, "
            "cuda_graph, height and width; "
            "use quantization_config='fp8' for online FP8"
        )
    _frame_size(config)
    if type(options.get("cuda_graph", False)) is not bool:
        raise ValueError("cuda_graph must be boolean")
    if options.get("scene", "castle") not in ("castle", "ancient-egypt", "racing"):
        raise ValueError("scene must be castle, ancient-egypt or racing")
    if options.get("prompt_mode", "fixed") not in ("fixed", "dynamic"):
        raise ValueError("prompt_mode must be fixed or dynamic")
    if options.get("prompt_mode") == "dynamic" and options.get("scene", "castle") != "racing":
        raise ValueError("Dynamic captions require the Racing scene and frame-aligned game captions")
    limit = options.get("max_latents", 8192)
    if type(limit) is not int or not 1 <= limit <= 8192:
        raise ValueError("max_latents must be an integer in [1,8192]")
    if not isinstance(options.get("prompt", DEFAULT_PROMPT), str) or not options.get("prompt", DEFAULT_PROMPT).strip():
        raise ValueError("prompt must be a nonempty string")


class MatrixGame4Pipeline(nn.Module):
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    _encoder_modules: ClassVar[list[str]] = []
    supports_request_batch = True

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        validate_config(od_config)
        if not current_omni_platform.is_cuda():
            raise ValueError("Matrix-Game currently requires an NVIDIA CUDA GPU")
        from .codec import MatrixCodec
        from .transformer import load_transformer

        self.device = current_omni_platform.get_torch_device()
        self.height, self.width = _frame_size(od_config)
        self.max_latents = od_config.model_config.get("max_latents", 8192)
        root = Path(od_config.model)
        self.scene = od_config.model_config.get("scene", "castle")
        scene_root = root / "models" / self.scene
        required = [
            scene_root / "model.safetensors",
            scene_root / "sink.png",
            root / "base/models_t5_umt5-xxl-enc-bf16.pth",
            root / "base/taew2_2.pth",
            scene_root / "inference.json",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if not (root / "base/google/umt5-xxl").is_dir():
            missing.append(str(root / "base/google/umt5-xxl"))
        if missing:
            raise FileNotFoundError(f"Incomplete local Matrix-Game resources: {missing}")
        profile = json.loads(required[4].read_text())
        contract = {
            "frame_size": [480, 864],
            "kv_window_latents": 21,
            "num_frame_per_block": 1,
            "scheduler_shift": 5,
            "cfg_scale": 1,
        }
        if any(profile.get(key) != value for key, value in contract.items()):
            raise ValueError("Scene inference.json does not match the released Matrix-Game contract")
        self.prompt_text = od_config.model_config.get("prompt", profile["prompt"])
        self.prompt_mode = od_config.model_config.get("prompt_mode", "fixed")
        self.sink_mode = "first_frame" if self.scene == "racing" else "kv_only"
        self.sink_rope_mode = "absolute" if self.scene == "racing" else "fixed"
        # vLLM online quantization materializes and processes each loaded linear
        # on its construction device. Keep unquantized construction on the CPU.
        with torch.device(self.device if od_config.quantization_config is not None else "cpu"):
            self.runtime = load_transformer(required[0], device=self.device, quant_config=od_config.quantization_config)
        self.runtime.use_graph = od_config.model_config.get("cuda_graph", False)
        self.transformer = self.runtime.model
        with torch.device("cpu"):
            from vllm_omni.diffusion.models.wan2_2.text_encoder import load_wan_umt5_encoder

            from .graph import MatrixGraph

            self.tokenizer = AutoTokenizer.from_pretrained(root / "base/google/umt5-xxl", local_files_only=True)
            self.text_encoder = load_wan_umt5_encoder(str(required[2]), torch.bfloat16)
            self.text_encoder = self.text_encoder.to(self.device).eval().requires_grad_(False)
            self.text_graph = (
                MatrixGraph(self._encode_tokens, num_tokens=512)
                if self.prompt_mode == "dynamic" and self.runtime.use_graph
                else None
            )
            self.register_buffer("prompt_embeds", self.encode_prompt(self.prompt_text), persistent=False)
            if self.prompt_mode == "fixed":
                self.text_encoder = None
                gc.collect()
                torch.accelerator.empty_cache()
            self.vae = MatrixCodec(str(required[3])).to(self.device, torch.bfloat16).eval()
            self.vae.use_graph = od_config.model_config.get("cuda_graph", False)
        self.register_buffer("sink_latent", self._encode_sink(required[1]), persistent=False)
        if od_config.quantization_config is not None:
            from .pointwise import compiled_ops

            self.runtime.fast_ops = compiled_ops()
        self._stream: MatrixStream | None = None
        self._session_id: str | None = None
        self._steps: int | None = None
        self._active_prompt = self.prompt_text
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """Use the runner's shared prompt cache when enabled; cached outputs stay read-only."""
        text = " ".join(html.unescape(html.unescape(ftfy.fix_text(prompt))).split())
        tokens = self.tokenizer(
            [text], padding="max_length", truncation=True, max_length=512, return_tensors="pt", add_special_tokens=True
        )
        ids, mask = tokens.input_ids.to(self.device), tokens.attention_mask.to(self.device)
        return self.text_graph(ids, mask) if self.text_graph is not None else self._encode_tokens(ids, mask)

    def _encode_tokens(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        assert self.text_encoder is not None
        embeddings = self.text_encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        return embeddings.masked_fill(~mask.bool().unsqueeze(-1), 0)

    @torch.no_grad()
    def _encode_sink(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            if self.sink_mode == "first_frame":
                image = ImageOps.fit(image, (self.width, self.height), method=Image.Resampling.LANCZOS)
            else:
                width, height = image.size
                scale = max(self.width / width, self.height / height)
                size = (round(width * scale), round(height * scale))
                image = image.resize(size, Image.Resampling.BILINEAR)
                left, top = round((size[0] - self.width) / 2), round((size[1] - self.height) / 2)
                image = image.crop((left, top, left + self.width, top + self.height))
            pixels = torch.from_numpy(np.array(image))
        pixels = (pixels.permute(2, 0, 1).float() * (2.0 / 255.0) - 1.0)[None, :, None]
        self.vae.clear_cache()
        try:
            latent = self.vae.encode(pixels.to(self.device, torch.bfloat16)).permute(0, 2, 1, 3, 4)
        finally:
            self.vae.clear_cache()
        shape = (1, 1, 48, self.height // 16, self.width // 16)
        if tuple(latent.shape) != shape or not torch.isfinite(latent).all():
            raise ValueError(f"Sink image did not encode to a finite {shape} latent")
        return latent

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The three native checkpoint formats are loaded strictly during construction.
        # No ComponentSource is exposed, so generic loading must supply no tensors.
        if next(iter(weights), None) is not None:
            raise ValueError("Matrix-Game loads its native checkpoints during construction")
        return set(dict(self.named_parameters()))

    def close_session(self) -> None:
        if self._stream is not None:
            self._stream.close()
        self._stream = None
        self._session_id = None
        self._steps = None
        self._active_prompt = self.prompt_text

    @torch.no_grad()
    def forward(self, batch: DiffusionRequestBatch) -> list[DiffusionOutput]:
        if batch.num_reqs != 1:
            raise ValueError("Matrix-Game only supports one request per worker")
        request = batch.requests[0]
        if not isinstance(request.prepared_layout, MatrixInput):
            prepare_request(request, height=self.height, width=self.width)
        prepared = request.prepared_layout
        assert isinstance(prepared, MatrixInput)
        text = request.prompt.get("prompt", "")
        if self.prompt_mode == "fixed" and text and text != self.prompt_text:
            raise ValueError("The scene caption is fixed at startup; set model_config.prompt to override it")
        if self._stream is not None and prepared.session_id != self._session_id:
            raise ValueError("Another Matrix-Game session owns this worker; close it before starting a new one")
        if prepared.frames is None:
            self.close_session()
            return [DiffusionOutput(custom_output={"session_closed": True})]
        if tuple(prepared.frames.shape[1:]) != (self.height, self.width, 3):
            raise ValueError("Prepared video resolution does not match this worker")
        count = prepared.frames.shape[0]
        needed = (count + 3) // 4
        if (self._stream.index if self._stream is not None else 0) + needed > self.max_latents:
            raise ValueError("Input exceeds this session's configured latent budget")
        if self._stream is not None and (prepared.frame_start != self._stream.frame or prepared.steps != self._steps):
            raise ValueError("Continuation must match the next frame offset and original denoising step count")
        if self._stream is None:
            if prepared.frame_start:
                raise ValueError("A new Matrix-Game session must begin at frame zero")
            from .stream import MatrixStream

            steps = (1000,) if prepared.steps == 1 else (1000, 500)
            self._stream = MatrixStream(
                self.runtime,
                self.vae,
                self.prompt_embeds,
                seed=request.sampling_params.seed,
                max_latents=self.max_latents,
                steps=steps,
                first_steps=steps,
                sink_latent=self.sink_latent,
                sink_mode=self.sink_mode,
                sink_rope_mode=self.sink_rope_mode,
                frame_size=(self.height, self.width),
            )
            self._session_id, self._steps = prepared.session_id, prepared.steps
        stream = self._stream
        outputs: list[torch.Tensor] = []
        succeeded = False
        try:
            if text and text != self._active_prompt:
                embeddings = self.prompt_embeds if text == self.prompt_text else self.encode_prompt(text)
                stream.set_prompt(embeddings)
                self._active_prompt = text
            offset = 0
            while offset < count:
                size = 1 if stream.index == 0 else 4
                pixels = prepared.frames[offset : offset + size]
                outputs.append(stream.push(pixels, frame_start=stream.frame).pixels)
                offset += size
            # Emit normalized device pixels; the runner owns dtype conversion and D2H.
            frames = outputs[0] if len(outputs) == 1 else torch.cat(outputs)
            media = DiffusionMediaOutput(VideoMediaOutput(frames.permute(1, 0, 2, 3).unsqueeze(0), _VIDEO_SPEC))
            result = DiffusionOutput(
                media=media,
                custom_output={
                    "frame_start": prepared.frame_start,
                    "frame_count": count,
                    "session_closed": prepared.end_session,
                },
            )
            succeeded = True
            return [result]
        finally:
            if prepared.end_session or not succeeded:
                self.close_session()
