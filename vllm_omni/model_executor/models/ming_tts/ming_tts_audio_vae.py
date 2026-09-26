# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adopted from https://github.com/inclusionAI/Ming-omni-tts/blob/main/audio_tokenizer/modeling_audio_vae.py
from __future__ import annotations

import os
import time
import warnings
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.model_executor.models.common.ming.audio_vae import AudioVAE
from vllm_omni.model_executor.models.output_templates import OmniOutput

from .config_ming_tts import KEY_CHUNK_ID, KEY_REQUEST_ID, MingTTSConfig
from .patch_emission import MING_STOP_REASON_KEY

logger = init_logger(__name__)

MING_FINAL_DECODE_STEP_KEY = "ming_final_decode_step"
_STREAM_STATE_TTL_SECONDS = 15 * 60
_MAX_STREAM_STATES = 1024


class MingAudioVAEModel(nn.Module):
    input_modalities = "audio"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        del prefix
        self.vllm_config = vllm_config
        self.ming_config = MingTTSConfig.from_hf_config(vllm_config.model_config.hf_config)
        if self.ming_config.audio_tokenizer_config is None:
            raise ValueError("MingAudioVAEModel requires audio_tokenizer_config")
        self.ming_config.validate()

        self.audio = AudioVAE(self.ming_config.audio_tokenizer_config)
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True

        self._past_key_values: dict[str, Any] = {}
        self._stream_state: dict[str, tuple[Any, Any, Any]] = {}
        self._patch_totals: dict[str, int] = {}
        self._sample_totals: dict[str, int] = {}
        self._state_access_times: OrderedDict[str, float] = OrderedDict()

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        hidden_size = int(self.ming_config.llm_hidden_size)
        if input_ids is None or input_ids.numel() == 0:
            return torch.empty((0, hidden_size), device=self.vllm_config.device_config.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], hidden_size), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> None:
        del hidden_states, sampling_metadata
        return None

    def chunked_decode_streaming(
        self,
        latent_chunk: torch.Tensor,
        *,
        request_id: str,
        finished: bool,
    ) -> tuple[torch.Tensor, Any, Any, bool, bool]:
        had_past_key_values = request_id in self._past_key_values
        had_stream_state = _has_stream_state(self._stream_state.get(request_id))
        stream_state = self._stream_state.get(request_id, (None, None, None))
        past_key_values = self._past_key_values.get(request_id)
        waveform_parts: list[torch.Tensor] = []

        patch_count = int(latent_chunk.shape[0])
        for patch_idx in range(patch_count):
            # [Batch, Time, Dimension] = [1, patch_size, latent_dim]
            latent_patch = latent_chunk[patch_idx : patch_idx + 1]
            is_last_patch = finished and patch_idx == patch_count - 1
            waveform, stream_state, past_key_values = self.audio.decode(
                latent_patch,
                past_key_values=past_key_values,
                use_cache=True,
                stream_state=stream_state,
                last_chunk=is_last_patch,
            )
            waveform_parts.append(waveform.reshape(-1).to(torch.float32))

        waveform_flat = (
            torch.cat(waveform_parts, dim=0)
            if waveform_parts
            else torch.zeros((0,), dtype=torch.float32, device=latent_chunk.device)
        )
        return waveform_flat, stream_state, past_key_values, had_past_key_values, had_stream_state

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        model_intermediate_buffer: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        del input_ids, positions, intermediate_tensors, inputs_embeds

        info_list = model_intermediate_buffer if isinstance(model_intermediate_buffer, list) else None
        if info_list is None:
            runtime_infos = kwargs.get("runtime_additional_information")
            info_list = runtime_infos if isinstance(runtime_infos, list) else None
        info_list = info_list or [{}]
        num_reqs = max(len(info_list), 1)
        sr_tensor = torch.tensor(int(self.ming_config.sample_rate), dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32, device=self.vllm_config.device_config.device)

        outputs: list[torch.Tensor] = []
        srs: list[torch.Tensor] = []

        for idx in range(num_reqs):
            info = info_list[idx] if idx < len(info_list) and isinstance(info_list[idx], dict) else {}
            meta = info.get("meta") or {}
            kv_meta = info.get("kv_metadata") or {}
            has_ming_context = _has_ming_context(info)
            if has_ming_context and KEY_REQUEST_ID not in info and KEY_REQUEST_ID not in kv_meta:
                raise RuntimeError(
                    f"Ming Stage-2 received a payload without {KEY_REQUEST_ID}. keys={sorted(info.keys())}"
                )
            request_id = _resolve_request_id(info, idx)
            self._touch_request_state(request_id)
            chunk_id = _coerce_optional_int(_first_present(info.get(KEY_CHUNK_ID), kv_meta.get(KEY_CHUNK_ID)))
            finished = _coerce_finished(
                _first_present(
                    meta.get("stream_finished"),
                    meta.get("finished"),
                    info.get("stream_finished"),
                    kv_meta.get("stream_finished"),
                    torch.tensor(True),
                )
            )
            latent = _first_present(
                info.get("latent"),
                info.get("ming_latent_patches"),
                kv_meta.get("ming_latent_patches"),
            )
            stripped = bool(
                _first_present(
                    meta.get("_ming_payload_stripped"),
                    info.get("_ming_payload_stripped"),
                    kv_meta.get("_ming_payload_stripped"),
                    False,
                )
            )
            if stripped:
                raise RuntimeError(
                    "Ming Stage-2 payload was stripped before model entry. "
                    f"request_id={request_id} chunk_id={chunk_id} keys={sorted(info.keys())}"
                )
            if latent is None:
                if has_ming_context and not finished:
                    raise RuntimeError(
                        "Ming Stage-2 received no latent chunk for an unfinished request. "
                        f"request_id={request_id} chunk_id={chunk_id} keys={sorted(info.keys())}"
                    )
                if finished:
                    self._clear_request_state(request_id)
                outputs.append(empty)
                srs.append(sr_tensor)
                continue

            latent_tensor = _coerce_latent_chunk(
                latent,
                device=self.vllm_config.device_config.device,
                dtype=next(self.audio.parameters()).dtype,
                latent_dim=self.ming_config.latent_dim,
                patch_size=self.ming_config.patch_size,
            )
            if latent_tensor is None or latent_tensor.numel() == 0:
                if not finished:
                    raise RuntimeError(
                        "Ming Stage-2 received an empty latent chunk before final flush. "
                        f"request_id={request_id} chunk_id={chunk_id} latent_shape={_shape_of(latent_tensor)}"
                    )
                if finished:
                    self._clear_request_state(request_id)
                outputs.append(empty)
                srs.append(sr_tensor)
                continue

            patch_count = int(latent_tensor.shape[0])
            waveform_flat, stream_state, past_key_values, had_past_key_values, had_stream_state = (
                self.chunked_decode_streaming(
                    latent_tensor,
                    request_id=request_id,
                    finished=finished,
                )
            )
            total_patch_count = self._patch_totals.get(request_id, 0) + patch_count
            total_waveform_numel = self._sample_totals.get(request_id, 0) + int(waveform_flat.numel())
            self._patch_totals[request_id] = total_patch_count
            self._sample_totals[request_id] = total_waveform_numel
            if (had_past_key_values or had_stream_state) and waveform_flat.numel() == 0 and not finished:
                raise RuntimeError(
                    "Ming Stage-2 produced an empty waveform after cached streaming state already existed. "
                    f"request_id={request_id} chunk_id={chunk_id} latent_shape={tuple(latent_tensor.shape)} "
                    f"had_past_key_values={had_past_key_values} had_stream_state={had_stream_state}"
                )

            if finished:
                if os.environ.get("MING_TTS_STAGE1_FINAL_LOG") == "1":
                    logger.info(
                        "MING_STAGE1_FINAL %s",
                        {
                            "request_id": request_id,
                            "chunk_id": chunk_id,
                            "stop_reason": _first_present(
                                info.get(MING_STOP_REASON_KEY),
                                kv_meta.get(MING_STOP_REASON_KEY),
                            ),
                            "final_decode_step": _coerce_optional_int(
                                _first_present(
                                    info.get(MING_FINAL_DECODE_STEP_KEY),
                                    kv_meta.get(MING_FINAL_DECODE_STEP_KEY),
                                )
                            ),
                            "final_chunk_patch_count": patch_count,
                            "total_patch_count": total_patch_count,
                            "final_chunk_waveform_numel": int(waveform_flat.numel()),
                            "total_waveform_numel": total_waveform_numel,
                        },
                    )
                self._clear_request_state(request_id)
            else:
                self._past_key_values[request_id] = past_key_values
                self._stream_state[request_id] = stream_state

            outputs.append(waveform_flat)
            srs.append(sr_tensor)

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "model_outputs": outputs,
                "sr": srs,
            },
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = list(weights)
        if not weights:
            raise RuntimeError("MingAudioVAEModel received no checkpoint weights.")

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded: set[str] = set()
        skipped: list[str] = []

        for name, loaded_weight in weights:
            if name not in params_dict:
                skipped.append(name)
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded.add(name)

        missing = {name for name in params_dict if name.startswith("audio.")} - loaded
        if missing:
            raise RuntimeError(f"MingAudioVAEModel: {len(missing)} params not loaded. First few: {sorted(missing)[:5]}")
        if skipped:
            warnings.warn(
                f"MingAudioVAEModel: skipped {len(skipped)} checkpoint keys during load. First few: {skipped[:8]}",
                stacklevel=2,
            )
        return loaded

    def _clear_request_state(self, request_id: str) -> None:
        self._past_key_values.pop(request_id, None)
        self._stream_state.pop(request_id, None)
        self._patch_totals.pop(request_id, None)
        self._sample_totals.pop(request_id, None)
        self._state_access_times.pop(request_id, None)

    def _touch_request_state(self, request_id: str) -> None:
        now = time.monotonic()
        self._evict_expired_request_states(now)
        self._state_access_times[request_id] = now
        self._state_access_times.move_to_end(request_id)
        while len(self._state_access_times) > _MAX_STREAM_STATES:
            evicted_request_id, _ = self._state_access_times.popitem(last=False)
            self._clear_request_state(evicted_request_id)

    def _evict_expired_request_states(self, now: float) -> None:
        cutoff = now - _STREAM_STATE_TTL_SECONDS
        while self._state_access_times:
            request_id, last_access = next(iter(self._state_access_times.items()))
            if last_access >= cutoff:
                break
            self._clear_request_state(request_id)


def _coerce_finished(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return False
        return bool(value.reshape(-1)[0].item())
    return bool(value)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _coerce_optional_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    if isinstance(value, list):
        value = value[0] if value else default
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return default
        return int(value.reshape(-1)[0].item())
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_request_id(info: dict[str, Any], idx: int) -> str:
    kv_meta = info.get("kv_metadata") or {}
    request_id = info.get(KEY_REQUEST_ID) or kv_meta.get(KEY_REQUEST_ID)
    if request_id is None:
        return str(idx)
    if not isinstance(request_id, str) or not request_id:
        raise RuntimeError(f"Ming Stage-2 received invalid request id: {request_id!r}")
    return request_id


def _has_ming_context(info: dict[str, Any]) -> bool:
    kv_meta = info.get("kv_metadata") or {}
    keys = (
        KEY_REQUEST_ID,
        KEY_CHUNK_ID,
        "ming_latent_patches",
        "_ming_payload_stripped",
    )
    return any(key in info or key in kv_meta for key in keys) or "latent" in info


def _shape_of(value: Any) -> tuple[int, ...] | None:
    if isinstance(value, torch.Tensor):
        return tuple(value.shape)
    return None


def _has_stream_state(value: Any) -> bool:
    if not isinstance(value, tuple):
        return False
    return any(item is not None for item in value)


def _coerce_latent_chunk(
    value: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
    latent_dim: int,
    patch_size: int,
) -> torch.Tensor | None:
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)

    latent = value.detach()
    if latent.ndim == 2:
        latent = latent.unsqueeze(0)
    if latent.ndim != 3:
        raise ValueError(f"Expected latent chunk rank-3 [Batch, Time, Dimension], got {tuple(latent.shape)}")
    if latent.shape[-2] != patch_size:
        raise ValueError(f"Latent patch size mismatch: got {latent.shape[-2]}, expected {patch_size}")
    if latent.shape[-1] != latent_dim:
        raise ValueError(f"Latent dim mismatch: got {latent.shape[-1]}, expected {latent_dim}")
    return latent.to(device=device, dtype=dtype)
