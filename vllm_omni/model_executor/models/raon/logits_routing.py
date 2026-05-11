# SPDX-License-Identifier: Apache-2.0
"""Logits routing and audio-token sampling for RaonModel."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from vllm_omni.model_executor.models.raon.raon_utils import (
    normalize_runtime_request_id,
    unwrap_singleton_list,
)
from vllm_omni.transformers_utils.configs.raon import ENV

if TYPE_CHECKING:
    from vllm_omni.model_executor.models.raon.raon import AudioDecodeState, RaonModel


def _info_int(req_info: dict[str, Any] | None, key: str, default: int = 0) -> int:
    if not isinstance(req_info, dict):
        return default
    value = unwrap_singleton_list(req_info.get(key, default))
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def audio_end_suppress_until(req_state: AudioDecodeState | None, req_info: dict[str, Any] | None) -> int:
    if req_state is None:
        return 0
    seam_until = (
        req_state.continuation_silence_frames + int(ENV.tts_long_eos_suppress_grace_steps)
        if req_state.continuation_silence_frames > 0
        else 0
    )
    return max(seam_until, _info_int(req_info, "audio_min_steps", 0))


class LogitsRouter:
    def __init__(self, model: RaonModel) -> None:
        self.model = model

    def sample_audio_token(
        self,
        *,
        logits: torch.Tensor,
        row_idx: int,
        req_info: dict[str, Any] | None,
        req_runtime_id: str | None,
        req_state: AudioDecodeState | None,
        audio_hidden_states: torch.Tensor | None,
    ) -> None:
        """Sample audio codec token and generate remaining RVQ codes."""
        audio_logits_row = (
            self.model.audio_lm_head(audio_hidden_states[row_idx : row_idx + 1]).squeeze(0)
            if audio_hidden_states is not None
            else None
        )
        if audio_logits_row is None:
            self.model._force_token_in_logits(logits, row_idx, int(self.model.audio_output_token_id))
            return

        in_silence_window = (
            req_state is not None
            and req_state.continuation_silence_frames > 0
            and req_state.audio_step_index < req_state.continuation_silence_frames
        )

        # Suppress AUDIO_END during continuation silence, near rolling-ICL
        # seams, and optionally until a request-scoped minimum audio length.
        _eos_suppress_until = audio_end_suppress_until(req_state, req_info)
        if req_state is not None and req_state.audio_step_index < _eos_suppress_until:
            audio_logits_row = audio_logits_row.clone()
            audio_logits_row[self.model.codebook_size] = float("-inf")

        if ENV.tts_temperature != 1.0 or ENV.tts_top_k > 0 or ENV.tts_top_p < 1.0:
            audio_logits_row = self.model._apply_audio_sampling_params(audio_logits_row)
        first_code = int(torch.multinomial(torch.softmax(audio_logits_row.float(), dim=-1), 1).item())

        if in_silence_window:
            silence = self.model._get_silence_codes()
            step_idx = req_state.audio_step_index
            silence_frame = silence[step_idx] if step_idx < silence.shape[0] else silence[-1]
            row_state = req_state
            row_state.pending_audio_codes = silence_frame.unsqueeze(0).to(device=logits.device)
            row_state.is_generating_audio = True
            # audio_step_index is incremented in audio_preprocess; do NOT double-increment here.
            self.model._force_token_in_logits(logits, row_idx, int(self.model.audio_output_token_id))
            return

        # Build layer-0 code history from per-request state.
        recent_codes = req_state.code_history
        first_code = self.model._ras.maybe_resample(
            recent_codes, first_code, audio_logits_row, self.model.codebook_size
        )

        if first_code == self.model.codebook_size:
            logits[row_idx, :] = float("-inf")
            if isinstance(self.model.audio_end_token_id, int) and 0 <= self.model.audio_end_token_id < int(
                logits.shape[-1]
            ):
                logits[row_idx, self.model.audio_end_token_id] = 0.0
            return

        req_state.code_history.append(first_code)
        row_state = req_state
        audio_hidden_row = audio_hidden_states[row_idx : row_idx + 1] if audio_hidden_states is not None else None

        speaker_batch: torch.Tensor | None = None
        if isinstance(req_info, dict) and self.model.proj_speaker_code is not None:
            speaker_embed = unwrap_singleton_list(req_info.get("speaker_embeds"))
            if isinstance(speaker_embed, torch.Tensor):
                if speaker_embed.ndim == 3:
                    speaker_embed = speaker_embed[:, 0, :]
                if speaker_embed.ndim == 2:
                    speaker_embed = speaker_embed[0]
                if speaker_embed.ndim == 1:
                    speaker_batch = speaker_embed.to(
                        device=logits.device, dtype=self.model.proj_code.weight.dtype
                    ).view(1, 1, -1)

        if audio_hidden_row is not None:
            full_codes = self.model._predict_rvq_codes(
                first_code=first_code,
                audio_hidden_row=audio_hidden_row,
                device=logits.device,
                speaker_embeds=speaker_batch,
            )
            row_state.pending_audio_codes = full_codes

        self.model._force_token_in_logits(logits, row_idx, int(self.model.audio_output_token_id))

    def generate_audio_for_row(
        self,
        *,
        logits: torch.Tensor,
        row_idx: int,
        req_info: dict[str, Any] | None,
        req_runtime_id: str | None,
        req_state: AudioDecodeState | None,
        force_audio_first_token: bool,
        is_sampled_row: bool,
        row_output_ids: list[int],
        audio_hidden_states: torch.Tensor | None,
    ) -> None:
        forced_audio_bootstrap = False
        if force_audio_first_token and is_sampled_row:
            forced_audio_bootstrap = (
                not bool(req_state.forced_audio_bootstrap_done) if req_state is not None else not row_output_ids
            )

        if forced_audio_bootstrap:
            self.model._forced_bootstrap_audio(logits, row_idx, req_state)
            return
        if not is_sampled_row:
            return

        self.sample_audio_token(
            logits=logits,
            row_idx=row_idx,
            req_info=req_info,
            req_runtime_id=req_runtime_id,
            req_state=req_state,
            audio_hidden_states=audio_hidden_states,
        )

    def apply_row_mode_adjustments(
        self,
        *,
        logits: torch.Tensor,
        row_runtime_info: list[Any],
        output_token_ids: Any,
        audio_hidden_states: torch.Tensor | None,
    ) -> None:
        row_count = len(row_runtime_info)
        for row_idx, req_info_raw in enumerate(row_runtime_info):
            req_info = req_info_raw if isinstance(req_info_raw, dict) else None
            mode = self.model._normalize_output_mode(req_info)

            force_audio_first_token = False
            req_runtime_id: str | None = None
            req_state: AudioDecodeState | None = None
            if req_info is not None:
                force_audio_first_token = bool(unwrap_singleton_list(req_info.get("force_audio_first_token", False)))
                csf_raw = unwrap_singleton_list(
                    req_info.get("continuation_silence_frames", 0),
                )
                csf = int(csf_raw) if csf_raw else 0
                req_runtime_id = normalize_runtime_request_id(
                    req_info.get("global_request_id", req_info.get("_omni_req_id"))
                )
                req_state = self.model._get_audio_decode_state(req_info)
                if req_state.continuation_silence_frames == 0 and csf > 0:
                    req_state.continuation_silence_frames = csf
            if (
                req_runtime_id is not None
                and audio_hidden_states is not None
                and row_idx < audio_hidden_states.shape[0]
            ):
                self.model._step_talker_hidden_rows[req_runtime_id] = audio_hidden_states[row_idx : row_idx + 1].clone()

            row_output_ids, is_sampled_row = self.model._resolve_row_sampling_state(
                output_token_ids=output_token_ids,
                row_idx=row_idx,
                row_count=row_count,
            )
            if mode == "audio_only":
                self.generate_audio_for_row(
                    logits=logits,
                    row_idx=row_idx,
                    req_info=req_info,
                    req_runtime_id=req_runtime_id,
                    req_state=req_state,
                    force_audio_first_token=force_audio_first_token,
                    is_sampled_row=is_sampled_row,
                    row_output_ids=row_output_ids,
                    audio_hidden_states=audio_hidden_states,
                )
                if req_runtime_id is not None and req_state is not None:
                    self.model._step_decode_states[req_runtime_id] = req_state
            else:
                self.model._mask_audio_logits_for_text_mode(logits, row_idx)
