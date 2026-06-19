"""Fish Speech S2 Pro -- DAC Decoder (Stage 1).

Loads the DAC codec from ``codec.pth`` and decodes codebook indices
[num_codebooks, T] → audio waveform at 44.1 kHz.

Analogous to ``Qwen3TTSCode2Wav`` in qwen3_tts.

Requires the ``fish-speech`` package for the DAC model architecture.
Install with: ``pip install fish-speech``
"""

from __future__ import annotations

import inspect
import os
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from torch.nn.utils.parametrize import remove_parametrizations
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

from vllm_omni.model_executor.models.fish_speech.dac_utils import (
    DAC_HOP_LENGTH,
    DAC_NUM_CODEBOOKS,
    DAC_SAMPLE_RATE,
    build_dac_codec,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


def _connector_extra_config(vllm_config: VllmConfig) -> dict[str, Any]:
    model_config = getattr(vllm_config, "model_config", None)
    connector_cfg = getattr(model_config, "stage_connector_config", None)
    if isinstance(connector_cfg, dict):
        return connector_cfg.get("extra", connector_cfg)
    extra = getattr(connector_cfg, "extra", None)
    return extra if isinstance(extra, dict) else {}


def _get_int_config(extra_cfg: dict[str, Any], default: int, *names: str) -> int:
    value = None
    for name in names:
        if name in extra_cfg:
            value = extra_cfg[name]
            break
    if value is None:
        return default
    try:
        return max(0, int(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid Fish Speech DAC integer config {names[0]}={value!r}") from exc


def _get_dac_dtype(extra_cfg: dict[str, Any]) -> torch.dtype:
    value = extra_cfg.get("fish_speech_dac_dtype", "float32")
    value = str(value).strip().lower()
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp32", "float32", ""}:
        return torch.float32
    raise ValueError(f"Invalid Fish Speech DAC dtype: {value!r}")


class FishSpeechDACDecoder(nn.Module):
    """Stage-1 DAC decoder for Fish Speech S2 Pro (GenerationModelRunner).

    Consumes frame-aligned codec tokens from input_ids and decodes waveform
    via the DAC codec decoder.
    """

    input_modalities = "audio"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model

        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True

        self._codec: nn.Module | None = None
        self._num_codebooks: int = DAC_NUM_CODEBOOKS
        self._output_sample_rate: int = DAC_SAMPLE_RATE
        self._hop_length: int = DAC_HOP_LENGTH
        self._logged_codec_stats = False
        self._codec_decode_takes_lengths: bool | None = None
        extra_cfg = _connector_extra_config(vllm_config)
        self._dac_dtype = _get_dac_dtype(extra_cfg)
        self._decode_batch_max_padded_frames = _get_int_config(
            extra_cfg,
            0,
            "fish_speech_dac_max_padded_frames",
            "dac_max_padded_frames",
        )
        self._decode_batch_max_batch = _get_int_config(
            extra_cfg,
            0,
            "fish_speech_dac_max_batch",
            "dac_max_batch",
        )

    def _decode_codes(
        self,
        codes_bqf: torch.Tensor,
        feature_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._codec is not None
        if self._codec_decode_takes_lengths:
            return self._codec.decode(codes_bqf, feature_lengths)

        if hasattr(self._codec, "from_indices"):
            wav_batch = self._codec.from_indices(codes_bqf)
        else:
            wav_batch = self._codec.decode(codes_bqf)
        max_frames = max(int(codes_bqf.shape[-1]), 1)
        scale = wav_batch.shape[-1] / max_frames
        audio_lengths = torch.clamp(
            torch.round(feature_lengths.to(torch.float32) * scale).to(torch.long),
            min=0,
            max=wav_batch.shape[-1],
        )
        return wav_batch, audio_lengths

    def _bake_weight_norm(self, codec: nn.Module) -> None:
        baked = 0
        for module in codec.modules():
            parametrizations = getattr(module, "parametrizations", None)
            if not parametrizations:
                continue
            for name in list(parametrizations.keys()):
                remove_parametrizations(module, name, leave_parametrized=True)
                baked += 1
        if baked > 0:
            logger.info("Baked %d DAC parametrized weights for inference", baked)

    def _cache_attention_masks(self, codec: nn.Module) -> None:
        for module in codec.modules():
            if not hasattr(module, "make_mask") or not hasattr(module, "make_window_limited_mask"):
                continue

            base_make_mask = module.make_mask
            base_make_window_mask = module.make_window_limited_mask
            mask_cache: dict[int, torch.Tensor] = {}
            window_mask_cache: dict[int, torch.Tensor] = {}

            def make_mask_cached(max_length: int, x_lens: torch.Tensor | None = None, *, _orig=base_make_mask):
                if x_lens is not None:
                    return _orig(max_length, x_lens)
                key = int(max_length)
                cached = mask_cache.get(key)
                if cached is None:
                    cached = _orig(max_length, x_lens)
                    mask_cache[key] = cached
                return cached

            def make_window_mask_cached(
                max_length: int,
                x_lens: torch.Tensor | None = None,
                *,
                _orig=base_make_window_mask,
            ):
                if x_lens is not None:
                    return _orig(max_length, x_lens)
                key = int(max_length)
                cached = window_mask_cache.get(key)
                if cached is None:
                    cached = _orig(max_length, x_lens)
                    window_mask_cache[key] = cached
                return cached

            module.make_mask = make_mask_cached
            module.make_window_limited_mask = make_window_mask_cached

    def _ensure_codec_loaded(self) -> None:
        if self._codec is not None:
            return

        codec_path = os.path.join(self.model_path, "codec.pth")
        if not os.path.exists(codec_path):
            # Try HuggingFace cache.
            try:
                from transformers.utils.hub import cached_file

                cached = cached_file(self.model_path, "codec.pth")
                if cached is not None:
                    codec_path = cached
            except Exception:
                pass

        if not os.path.exists(codec_path):
            raise FileNotFoundError(
                f"codec.pth not found at {codec_path}. Make sure the Fish Speech S2 Pro model includes codec.pth."
            )

        codec = build_dac_codec()

        # Load weights.
        state_dict = torch.load(codec_path, map_location="cpu", weights_only=True)
        # Some checkpoints wrap under "generator" key.
        if "generator" in state_dict:
            state_dict = state_dict["generator"]
        codec.load_state_dict(state_dict, strict=False)
        self._bake_weight_norm(codec)
        self._cache_attention_masks(codec)

        # Decode path only uses quantizer.decode() + decoder; prune
        # encode-only components before moving to device to avoid
        # unnecessary GPU allocation.
        codec.encoder = None
        codec.quantizer.pre_module = None
        codec.quantizer.downsample = None

        device = self.vllm_config.device_config.device
        codec = codec.to(device=device, dtype=self._dac_dtype)
        codec.eval()
        self._codec = codec
        self._codec_decode_takes_lengths = len(inspect.signature(codec.decode).parameters) >= 2

        logger.info(
            "Fish Speech DAC codec loaded from %s (device=%s, dtype=%s, sample_rate=%d)",
            codec_path,
            device,
            self._dac_dtype,
            self._output_sample_rate,
        )

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> None:
        return None

    def _split_request_ids(
        self,
        ids: torch.Tensor,
        seq_token_counts: list[int] | None = None,
    ) -> list[torch.Tensor]:
        if seq_token_counts is not None and len(seq_token_counts) > 1:
            boundaries = [0]
            for count in seq_token_counts:
                boundaries.append(boundaries[-1] + count)
            n = ids.numel()
            return [ids[boundaries[i] : min(boundaries[i + 1], n)] for i in range(len(seq_token_counts))]
        if is_forward_context_available():
            slices = get_forward_context().ubatch_slices
            if slices is not None and len(slices) > 1 and not any(hasattr(s, "token_slice") for s in slices):
                boundaries = [0]
                for s in slices:
                    boundaries.append(boundaries[-1] + s)
                return [ids[boundaries[i] : boundaries[i + 1]] for i in range(len(boundaries) - 1)]
        return [ids]

    def _codes_from_runtime_info(
        self,
        info: dict[str, Any] | None,
        device: torch.device,
    ) -> torch.Tensor | None:
        if not isinstance(info, dict):
            return None
        codes = info.get("codes", {}).get("audio")
        if not isinstance(codes, torch.Tensor) or codes.numel() == 0:
            return None

        q = self._num_codebooks
        codes = codes.to(device=device, dtype=torch.long, non_blocking=True).contiguous()
        if codes.ndim != 2 or codes.shape[0] != q:
            logger.warning(
                "DAC tensor codes must have shape [num_codebooks, frames], got %s for num_codebooks=%d.",
                tuple(codes.shape),
                q,
            )
            return None
        return codes

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        """Decode codec codes into audio waveform.

        input_ids layout per request: flat codes [num_codebooks * num_frames].
        Codes are codebook-major: [cb0_f0, cb0_f1, ..., cb0_fN, cb1_f0, ...].
        """
        q = self._num_codebooks
        sr_val = self._output_sample_rate
        sr_tensor = torch.tensor(sr_val, dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32)

        self._ensure_codec_loaded()
        assert self._codec is not None

        if input_ids is None or input_ids.numel() == 0:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [empty], "sr": [sr_tensor]},
            )

        ids = input_ids.reshape(-1).to(dtype=torch.long)
        request_ids_list = self._split_request_ids(ids, kwargs.get("seq_token_counts"))

        num_req = len(request_ids_list)
        parsed_ctx_frames = [0] * num_req
        parsed_total_frames = [0] * num_req
        valid_codes_qf: list[torch.Tensor] = []
        valid_indices: list[int] = []
        left_context_size = [0] * len(request_ids_list)
        if runtime_additional_information is not None:
            for i, info in enumerate(runtime_additional_information):
                if i >= len(left_context_size):
                    break
                meta = info.get("meta", {}) if isinstance(info, dict) else {}
                if "left_context_size" in meta:
                    left_context_size[i] = meta["left_context_size"]

        for i, req_ids in enumerate(request_ids_list):
            ctx_frames = left_context_size[i]
            runtime_codes = (
                self._codes_from_runtime_info(runtime_additional_information[i], ids.device)
                if runtime_additional_information is not None and i < len(runtime_additional_information)
                else None
            )
            if runtime_codes is not None:
                frames = runtime_codes.shape[1]
                parsed_ctx_frames[i] = ctx_frames
                parsed_total_frames[i] = frames
                valid_codes_qf.append(runtime_codes)
                valid_indices.append(i)
                continue

            if req_ids.numel() < 1:
                continue
            n = req_ids.numel()
            if n % q != 0:
                logger.warning(
                    "DAC decoder input_ids length %d not divisible by num_codebooks %d; returning empty audio.",
                    n,
                    q,
                )
                continue
            frames = n // q
            codes_qf = req_ids.reshape(q, frames)
            parsed_ctx_frames[i] = ctx_frames
            parsed_total_frames[i] = frames
            valid_codes_qf.append(codes_qf)
            valid_indices.append(i)
        if not valid_codes_qf:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "model_outputs": [empty] * num_req,
                    "sr": [sr_tensor] * num_req,
                },
            )

        if not self._logged_codec_stats:
            self._logged_codec_stats = True
            try:
                c = valid_codes_qf[0]
                logger.info(
                    "DAC decoder: frames=%d q=%d uniq=%d range=[%d,%d] batch=%d",
                    c.shape[1],
                    q,
                    int(torch.unique(c).numel()),
                    int(c.min().item()),
                    int(c.max().item()),
                    len(valid_codes_qf),
                )
            except Exception:
                pass

        wav_tensors: list[torch.Tensor | None] = [None] * len(valid_codes_qf)

        def _iter_bounded_groups(group: list[tuple[int, torch.Tensor]]) -> Iterable[list[tuple[int, torch.Tensor]]]:
            max_padded_frames = int(getattr(self, "_decode_batch_max_padded_frames", 0))
            max_batch = int(getattr(self, "_decode_batch_max_batch", 0))
            if max_padded_frames <= 0 and max_batch <= 0:
                yield group
                return

            current: list[tuple[int, torch.Tensor]] = []
            current_max_frames = 0
            for item in sorted(group, key=lambda pair: int(pair[1].shape[1])):
                frames = int(item[1].shape[1])
                next_max_frames = max(current_max_frames, frames)
                next_batch_size = len(current) + 1
                exceeds_batch = max_batch > 0 and next_batch_size > max_batch
                exceeds_work = max_padded_frames > 0 and next_batch_size * next_max_frames > max_padded_frames
                if current and (exceeds_batch or exceeds_work):
                    yield current
                    current = [item]
                    current_max_frames = frames
                else:
                    current.append(item)
                    current_max_frames = next_max_frames
            if current:
                yield current

        def _decode_group(group: list[tuple[int, torch.Tensor]]) -> None:
            actual_frames = [int(codes_qf.shape[1]) for _, codes_qf in group]
            target_frames = max(actual_frames)
            batch_size = len(group)
            first_codes = group[0][1]
            feature_lengths = torch.tensor(actual_frames, device=first_codes.device, dtype=torch.long)
            codes_bqf = torch.zeros(
                (batch_size, q, target_frames),
                device=first_codes.device,
                dtype=torch.long,
            )
            for row, (_, codes_qf) in enumerate(group):
                frame_count = int(codes_qf.shape[1])
                codes_bqf[row, :, :frame_count] = codes_qf

            with torch.amp.autocast("cuda", enabled=False):
                wav_batch, audio_lengths = self._decode_codes(codes_bqf, feature_lengths)
            audio_lengths_list = (
                audio_lengths.detach().to(device="cpu", dtype=torch.long).reshape(-1).tolist()
                if audio_lengths.numel() > 0
                else []
            )
            for row, (j, _) in enumerate(group):
                audio_len = int(audio_lengths_list[row]) if len(audio_lengths_list) > row else int(wav_batch.shape[-1])
                wav_tensors[j] = wav_batch[row, 0, :audio_len]

        for bounded_group in _iter_bounded_groups(list(enumerate(valid_codes_qf))):
            _decode_group(bounded_group)

        audios: list[torch.Tensor] = [empty] * num_req
        srs = [sr_tensor] * num_req

        for j, idx in enumerate(valid_indices):
            ctx_frames = parsed_ctx_frames[idx]
            total_frames = parsed_total_frames[idx]
            wav = wav_tensors[j]
            assert wav is not None
            # Trim context frames (left overlap for streaming).
            if ctx_frames > 0:
                # Decode length may deviate from (frames * hop_length) due to model
                # internals (padding/rounding). Use proportional trimming to keep
                # overlap removal aligned with the actual decoded length.
                denom = max(int(total_frames), 1)
                cut = int(ctx_frames / denom * wav.shape[0])
                cut = max(0, min(cut, int(wav.shape[0])))
                if cut < wav.shape[0]:
                    wav = wav[cut:]
                else:
                    logger.warning(
                        "Context trim %d >= decoded length %d; returning empty audio.",
                        cut,
                        wav.shape[0],
                    )
                    continue
            if wav.shape[0] > 0:
                audios[idx] = wav.to(dtype=torch.float32).reshape(-1)

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": audios, "sr": srs},
        )

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        if not (isinstance(model_outputs, tuple) and len(model_outputs) == 2):
            raise TypeError(f"FishSpeechDACDecoder expected (audio_tensor, sr), got {type(model_outputs)}")
        audio_tensor, sr = model_outputs
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": audio_tensor, "sr": sr},
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # DAC codec weights are loaded lazily from codec.pth, not from the main checkpoint.
        return set()
