# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage-1 acoustic decoder for MiniMax Music 3.

Consumes the AR stage's conditioning spans and returns 32 kHz stereo audio.
Each span is cut into 200-frame windows that overlap by half; every window is
solved by the flow-matching DiT with the previous window's tail latent pinned
as a prompt, decoded by the vocoder, and cropped so that concatenating the
survivors reconstructs the song exactly once.

The window arithmetic lives in :mod:`.chunking` and is driven entirely by the
span's global frame range, so the streaming path (one window per engine step)
and the non-streaming path (the whole song in one payload) produce identical
audio for the same seed.

The stage runs in float32. TF32 keeps that affordable, and bfloat16 measurably
degrades a 30-step solve.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .chunking import ChunkWindow, chunk_windows, crop_sample_bounds, overlap_mel_length
from .constants import (
    AR_CHUNK_HOP_FRAMES,
    AR_HIDDEN_SIZE,
    DEFAULT_DIT_CFG_SCALE,
    DEFAULT_DIT_STEPS,
    OUTPUT_CHANNELS,
    OUTPUT_SAMPLE_RATE,
)
from .dav import MiniMaxMusic3Vocoder, StreamingResampler, remove_weight_norm
from .dit import MiniMaxMusic3FlowMatchingDiT, remap_transformer_state
from .weights import (
    CONDITION_ENCODER_DIR,
    TRANSFORMER_DIR,
    VOCODER_DIR,
    load_component_state,
    resolve_repo_root,
)

logger = init_logger(__name__)

__all__ = ["MiniMaxMusic3AcousticForConditionalGeneration"]

# Namespaced so a request seed reused by another component cannot collide with
# the noise draw, and so the draw is reproducible across processes. Python's
# hash() is randomized per interpreter and must not be used here.
_SEED_PERSON = b"minimax-ttm"
_SEED_BYTES = 8


def _derive_seed(seed: int, *parts: object) -> int:
    """Derive a stable 63-bit sub-seed from a request seed and some labels."""
    digest = hashlib.blake2b(digest_size=_SEED_BYTES, person=_SEED_PERSON)
    digest.update(int(seed).to_bytes(8, "little", signed=False))
    for part in parts:
        raw = str(part).encode("utf-8")
        digest.update(len(raw).to_bytes(4, "little"))
        digest.update(raw)
    return int.from_bytes(digest.digest(), "little") & ((1 << 63) - 1)


def _meta_int(value: Any, default: int = 0) -> int:
    """Read an int that transport may have wrapped in a tensor or a list."""
    if isinstance(value, torch.Tensor):
        return int(value.reshape(-1)[0].item()) if value.numel() else default
    if isinstance(value, list | tuple):
        return _meta_int(value[0], default) if value else default
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _meta_bool(value: Any) -> bool:
    """Read a bool that transport may have wrapped in a tensor or a list."""
    if isinstance(value, torch.Tensor):
        return bool(value.reshape(-1)[0].item()) if value.numel() else False
    if isinstance(value, list | tuple):
        return _meta_bool(value[0]) if value else False
    return bool(value)


def _meta_str(value: Any) -> str | None:
    """Read a string that transport may have wrapped in a list."""
    if isinstance(value, list | tuple):
        return _meta_str(value[0]) if value else None
    return value if isinstance(value, str) else None


@dataclass
class _RequestState:
    """Cross-window streaming state, keyed by request id and never by row."""

    last_latent: Tensor | None = None
    last_condition: Tensor | None = None
    next_window_index: int = 0
    decoded_windows: int = 0
    resampler: StreamingResampler = field(default_factory=StreamingResampler)

    def release(self) -> None:
        self.last_latent = None
        self.last_condition = None
        self.resampler.reset()


class MiniMaxMusic3AcousticForConditionalGeneration(nn.Module):
    """Flow-matching DiT plus DAC vocoder, driven by the generation runner."""

    input_modalities = "audio"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        del prefix
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model

        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True

        self._configure_backends()

        # Skeleton only. vLLM's memory profiler walks the module tree at
        # startup, before any weight loader has run.
        self.dit = MiniMaxMusic3FlowMatchingDiT()
        self.vocoder = MiniMaxMusic3Vocoder()

        extra = self._connector_extra_config()
        self.dit_steps = max(1, _meta_int(extra.get("dit_steps"), DEFAULT_DIT_STEPS))
        cfg_scale = extra.get("dit_cfg_scale")
        self.dit_cfg_scale = (
            float(cfg_scale)
            if isinstance(cfg_scale, int | float) and not isinstance(cfg_scale, bool)
            else DEFAULT_DIT_CFG_SCALE
        )

        self._states: dict[str, _RequestState] = {}
        self._loaded = False

    @staticmethod
    def _configure_backends() -> None:
        """Pin the numeric backends this checkpoint was validated against.

        These are process-global switches, which is safe here only because a
        stage owns its own engine-core process. Never call this from a module
        shared with the AR stage: cuDNN's fused attention is not
        bit-reproducible for grouped-query shapes, so flipping it underneath
        the backbone makes the same seed produce a different song depending on
        which stage loaded first.

        cuDNN itself is off because its 1-D convolution paths are slower than
        the native ones for the vocoder's dilated stacks, and its SDPA backend
        is not needed because the DiT runs the flash path. TF32 keeps
        float32's range with a 10-bit mantissa, which is what makes a float32
        solve affordable.
        """
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = False
            torch.backends.cuda.enable_cudnn_sdp(False)

    def _connector_extra_config(self) -> dict[str, Any]:
        """Return the stage connector's ``extra`` block, or an empty mapping."""
        connector = getattr(self.vllm_config.model_config, "stage_connector_config", None)
        if isinstance(connector, dict):
            extra = connector.get("extra", connector)
            return extra if isinstance(extra, dict) else {}
        extra = getattr(connector, "extra", None)
        return extra if isinstance(extra, dict) else {}

    # ------------------------------------------------------------- vLLM hooks

    def embed_input_ids(self, input_ids: Tensor, **_: Any) -> Tensor:
        """Stable dummy embedding: this stage never reads token embeddings."""
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: Tensor | OmniOutput, sampling_metadata: Any = None) -> None:
        del hidden_states, sampling_metadata
        return None

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]:
        """Load the three acoustic components from their own folders.

        The iterator vLLM hands over covers the stage's own checkpoint folder,
        which holds nothing this stage needs. It is drained anyway: the loader
        streams from a file handle the caller closes only once the iterator is
        exhausted, so abandoning it wedges startup.
        """
        try:
            for _ in weights:
                pass
        except RuntimeError as exc:
            # The iterator is lazy, so draining it is what actually asks the
            # default loader for root-level weight files. This repository keeps
            # every component in its own subfolder and has none at the root, so
            # that lookup raises. Nothing here needs those weights; any other
            # RuntimeError is a real problem and is re-raised.
            if "Cannot find any model weights" not in str(exc):
                raise
            logger.debug("MiniMax Music 3 acoustic stage has no root-level weights, as expected")
        return self._load_components()

    def _ensure_loaded(self) -> None:
        """Load on first use when the weight loader skipped ``load_weights``.

        ``load_format=dummy`` (used by warmup and memory profiling) never calls
        ``load_weights``, which would leave the modules holding their random
        skeleton and make the dummy run produce noise or crash.
        """
        if not self._loaded:
            self._load_components()

    def _load_components(self) -> set[str]:
        """Read ``condition_encoder/``, ``transformer/`` and ``vocoder/``.

        Every component loads with ``strict=True``. A silently partial load
        would produce audio that sounds plausible but is wrong, which is far
        worse than a startup failure.
        """
        root = resolve_repo_root(
            self.model_path,
            revision=self.vllm_config.model_config.revision,
            download_dir=self.vllm_config.load_config.download_dir,
        )
        encoder = self.dit.condition_encoder
        encoder.load_state_dict(load_component_state(root, CONDITION_ENCODER_DIR), strict=True)
        transformer_state = remap_transformer_state(load_component_state(root, TRANSFORMER_DIR))
        self.dit.transformer.load_state_dict(transformer_state, strict=True)
        self.vocoder.load_state_dict(load_component_state(root, VOCODER_DIR), strict=True)

        device = self.vllm_config.device_config.device
        self.dit.to(device=device, dtype=torch.float32).eval()
        self.vocoder.to(device=device, dtype=torch.float32).eval()
        folded = remove_weight_norm(self.vocoder)
        self._loaded = True

        loaded = {f"dit.{name}" for name, _ in self.dit.named_parameters()}
        loaded |= {f"vocoder.{name}" for name, _ in self.vocoder.named_parameters()}
        logger.info(
            "MiniMax Music 3 acoustic loaded from %s device=%s dit_steps=%d "
            "dit_cfg_scale=%.3f attention=%s folded_weight_norms=%d parameters=%d",
            root,
            device,
            self.dit_steps,
            self.dit_cfg_scale,
            self.dit.transformer.attention_backend_name,
            folded,
            sum(p.numel() for p in self.parameters()),
        )
        return loaded

    def on_requests_finished(self, finished_req_ids: Iterable[str]) -> None:
        """Drop the streaming state of finished or aborted requests."""
        for req_id in finished_req_ids:
            state = self._states.pop(str(req_id), None)
            if state is not None:
                logger.info(
                    "MiniMax Music 3 acoustic done request=%s windows=%d",
                    req_id,
                    state.decoded_windows,
                )
                state.release()

    def make_omni_output(self, model_outputs: Tensor | OmniOutput | tuple, **kwargs: Any) -> OmniOutput:
        del kwargs
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        # The runner may unpack the NamedTuple in transit; rebuild it.
        if isinstance(model_outputs, tuple) and len(model_outputs) == len(OmniOutput._fields):
            return OmniOutput(*model_outputs)
        raise TypeError(f"MiniMaxMusic3AcousticForConditionalGeneration expected OmniOutput, got {type(model_outputs)}")

    # --------------------------------------------------------------- decoding

    @staticmethod
    def _silence() -> Tensor:
        """Placeholder for a step that produced no audio.

        Deliberately rank 2. A rank-1 empty tensor mixed with rank-2 chunks
        makes the multimodal collector's ``torch.cat(..., dim=-1)`` fall back
        to a flattening path that would planar-ize the stereo.
        """
        return torch.zeros((OUTPUT_CHANNELS, 0), dtype=torch.float32)

    def _decode_window(self, state: _RequestState, hidden: Tensor, window: ChunkWindow, seed: int) -> Tensor:
        """Solve and vocode one window, cropped to its unique contribution.

        Everything stays on the device: the crop is a slice and the result is
        handed straight to the resampler, so a step costs one device-to-host
        copy however many windows it decoded.
        """
        device = self.vllm_config.device_config.device
        hidden = hidden.unsqueeze(0).to(device=device, dtype=torch.float32)
        generator = torch.Generator(device=device).manual_seed(_derive_seed(seed, "dit", window.index))
        align = self.dit.aligned_condition(hidden)
        latent = self.dit(
            align,
            generator=generator,
            initial_latent=state.last_latent,
            initial_condition=state.last_condition,
            num_steps=self.dit_steps,
            cfg_scale=self.dit_cfg_scale,
        )
        waveform = self.vocoder(latent)

        # Carry the middle of the overlap forward. The solver re-imposes it on
        # the next window, so the two windows agree where they meet.
        overlap = overlap_mel_length()
        start = max(0, latent.shape[-1] - 2 * overlap)
        end = max(start, latent.shape[-1] - overlap)
        state.last_latent = latent[..., start:end].clone()
        state.last_condition = align[..., start:end].clone()

        wave = waveform[0].clamp(-1.0, 1.0).float()
        left, right = crop_sample_bounds(window)
        stop = wave.shape[-1] - right if right else wave.shape[-1]
        return wave[:, left:stop]

    def _decode_span(self, req_id: str, span: Tensor | None, meta: dict[str, Any]) -> Tensor:
        """Decode every window a span covers and return this step's audio.

        A span is normally one window, but the terminal emission carries the
        held-back complete window plus the truncated final one, and the
        non-streaming path carries the whole song. Window boundaries and the
        crop are re-derived from the span's global frame range so all three
        cases go through one code path.

        Raises:
            ValueError: If the span is malformed or lands on a window other
                than the one this request is waiting for.
        """
        span_start = _meta_int(meta.get("num_processed_tokens"), 0)
        is_terminal = _meta_bool(meta.get("last_chunk")) or _meta_bool(meta.get("stream_finished"))
        seed = _meta_int(meta.get("audio_seed"), 0)
        state = self._states.setdefault(req_id, _RequestState())

        # A metadata-only terminal marker carries no conditioning; it decodes
        # zero windows and exists only to flush the resampler.
        windows = chunk_windows(int(span.shape[0])) if span is not None else []
        pieces: list[Tensor] = []
        with torch.inference_mode():
            for local in windows:
                start = span_start + local.start
                index = start // AR_CHUNK_HOP_FRAMES
                if index != state.next_window_index:
                    # Each window is prompted by its predecessor's tail, so a
                    # gap or a replay corrupts the join. Fail the request here
                    # rather than return audio with a seam in it.
                    raise ValueError(
                        f"MiniMax Music 3 acoustic request {req_id} expected window "
                        f"{state.next_window_index} but received window {index} "
                        f"(span starts at AR frame {span_start}); the overlap prompt "
                        f"for this window is stale and the audio would not join"
                    )
                window = ChunkWindow(
                    index=index,
                    start=start,
                    end=span_start + local.end,
                    is_first=start == 0,
                    is_last=local.is_last and is_terminal,
                )
                pieces.append(self._decode_window(state, span[local.start : local.end], window, seed))
                state.next_window_index = index + 1
                state.decoded_windows += 1

            emitted = state.resampler.push(pieces, flush=is_terminal)
            return emitted.cpu()

    def _span_from_info(self, info: dict[str, Any], req_id: str) -> Tensor | None:
        """Return the conditioning span as ``[T, 32768]``, or ``None``.

        ``None`` means this step carried no conditioning, which is normal: a
        request can be scheduled before its span arrives, and the terminal
        marker is metadata only. A span that is present but the wrong shape is
        not normal and stops the request.

        Raises:
            ValueError: If a span is present but is not ``[T, 32768]``.
        """
        latent = info.get("latent")
        if not isinstance(latent, Tensor) or latent.numel() == 0:
            return None
        if latent.ndim == 3 and latent.shape[0] == 1:
            latent = latent[0]
        if latent.ndim != 2 or latent.shape[1] != AR_HIDDEN_SIZE:
            raise ValueError(
                f"MiniMax Music 3 acoustic request {req_id} carries a conditioning "
                f"span of shape {tuple(latent.shape)}; expected [T,{AR_HIDDEN_SIZE}]"
            )
        return latent

    def forward(
        self,
        input_ids: Tensor | None = None,
        positions: Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        """Decode this step's spans into one stereo waveform per request.

        Raises:
            ValueError: If a payload is malformed. Failing the step is
                deliberate: a request whose conditioning cannot be decoded has
                no audio to return, and reporting an error is better than
                leaving the caller waiting on a stream that will never end.
        """
        del input_ids, positions, intermediate_tensors, inputs_embeds
        self._ensure_loaded()

        infos = runtime_additional_information
        if infos is None:
            infos = kwargs.get("model_intermediate_buffer")
        infos = infos or []

        # One entry per scheduled request, including steps where a request's
        # span has not arrived yet.
        seq_token_counts = kwargs.get("seq_token_counts")
        num_reqs = len(seq_token_counts) if seq_token_counts is not None else len(infos)

        audios: list[Tensor] = []
        sr_tensor = torch.tensor(OUTPUT_SAMPLE_RATE, dtype=torch.int32)
        for index in range(num_reqs):
            info = infos[index] if index < len(infos) else {}
            info = info if isinstance(info, dict) else {}
            meta = info.get("meta")
            meta = meta if isinstance(meta, dict) else {}
            req_id = _meta_str(meta.get("req_id")) or _meta_str(info.get("req_id"))
            has_span = isinstance(info.get("latent"), Tensor) and info["latent"].numel() > 0
            is_terminal = _meta_bool(meta.get("last_chunk")) or _meta_bool(meta.get("stream_finished"))
            if not has_span and not is_terminal:
                audios.append(self._silence())
                continue
            if req_id is None:
                raise ValueError(
                    "MiniMax Music 3 acoustic payload carries no meta.req_id, so its "
                    "window cannot be joined to the rest of its request"
                )
            audios.append(self._decode_span(req_id, self._span_from_info(info, req_id), meta))

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": audios, "sr": [sr_tensor] * num_reqs},
        )
