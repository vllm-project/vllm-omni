# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Raw ZONOS2 frame handoff; de-shear and OLA belong to the DAC worker."""

from __future__ import annotations

from typing import Any

import torch

from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct, unflatten_payload
from vllm_omni.model_executor.models.zonos2.zonos2_codec import eos_boundary, shear


def _frames(audio: torch.Tensor) -> torch.Tensor:
    if audio.ndim == 1 and audio.numel() % 9 == 0:
        audio = audio.reshape(-1, 9)
    if audio.ndim != 2 or audio.shape[1] != 9 or audio.dtype not in (torch.int32, torch.int64):
        raise ValueError("ZONOS2 talker audio must have shape [T,9]")
    return audio.detach().to(device="cpu", dtype=torch.long)


def _revert_delay_pattern(audio_codes_qt: torch.Tensor) -> torch.Tensor:
    if audio_codes_qt.ndim != 2 or audio_codes_qt.shape[0] != 9:
        raise ValueError("ZONOS2 de-shear expects [9,T]")
    return shear(audio_codes_qt.T, up=True)[: max(0, audio_codes_qt.shape[1] - 8)].T.contiguous()


def _eos_frame(meta: dict) -> int | None:
    eos = meta.get("eos_frame")
    if isinstance(eos, torch.Tensor):
        eos = int(eos.reshape(-1)[-1]) if eos.numel() else -1
    return int(eos) if eos is not None else None


def _target(frame_count: int, eos: int | None, final: bool) -> int:
    target = frame_count if final else max(0, frame_count - 8)
    return min(target, max(0, int(eos))) if eos is not None and int(eos) >= 0 else target


def talker2dac_token_only(
    source_outputs: list[Any], prompt: Any = None, _requires_multimodal_data: bool = False
) -> list[Any]:
    """Admit sync DAC requests; the worker connector owns their codec data."""
    from vllm_omni.inputs.data import OmniTokensPrompt

    return [OmniTokensPrompt(prompt_token_ids=[0]) for source in source_outputs if source.finished]


def talker2dac_full_payload(transfer_manager: Any, pooling_output: dict[str, Any], request: Any) -> OmniPayloadStruct:
    """Send accumulated raw frames once, without de-shearing or edge trims."""
    del transfer_manager, request
    mm = unflatten_payload(pooling_output)
    audio = mm.get("codes", {}).get("audio")
    if not isinstance(audio, torch.Tensor):
        raise ValueError("ZONOS2 sync transport requires accumulated audio codes")
    frames = _frames(audio)
    eos = _eos_frame(mm.get("meta", {}))
    if eos is None:
        eos = eos_boundary(frames)
    return OmniPayloadStruct(
        codes=CodesStruct(audio=frames.T.contiguous()),
        meta=MetaStruct(
            finished=torch.tensor(True),
            last_chunk=True,
            num_processed_tokens=_target(len(frames), eos, True),
            chunk_seq=0,
            replace_runtime_additional_information=True,
        ),
    )


def talker2dac_async_chunk(
    transfer_manager: Any,
    multimodal_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
    new_token_ids: tuple[int, ...] | None = None,
) -> OmniPayloadStruct | None:
    key = request.external_req_id
    final = bool(is_finished or request.is_finished())
    buffer = transfer_manager.code_prompt_token_ids[key]
    mm = multimodal_output or {}
    audio = mm.get("codes", {}).get("audio")
    # Native MRv2 snapshots current-step tokens on the request; the legacy
    # transfer adapter supplies the explicit argument. Neither channel includes
    # old tokens on terminal flush, so the last frame is never appended twice.
    if new_token_ids is None:
        new_token_ids = tuple(getattr(request, "sampled_token_ids", ()))
    if new_token_ids and isinstance(audio, torch.Tensor) and audio.numel():
        frames = _frames(audio)
        if len(frames) != len(new_token_ids):
            raise ValueError("ZONOS2 step frame/token count mismatch")
        buffer.extend(frames.unbind(0))
    eos = _eos_frame(mm.get("meta", {}))
    frames = None
    if eos is None:
        # Legacy producers omit EOS metadata; their raw frames remain the oracle.
        frames = torch.stack(buffer) if buffer else torch.empty((0, 9), dtype=torch.long)
        eos = eos_boundary(frames)
    target = _target(len(buffer), eos, final)
    if not final and (target < 16 or target % 16 != 0):
        return None
    if frames is None:
        frames = torch.stack(buffer) if buffer else torch.empty((0, 9), dtype=torch.long)
    # The receiver replaces runtime information and DACStreamDecoder validates
    # cumulative prefixes. Send an owned full snapshot on emission/flush only;
    # changing to deltas requires a matching decoder/transport contract change.
    payload = OmniPayloadStruct(
        codes=CodesStruct(audio=frames.T.contiguous()),
        meta=MetaStruct(
            finished=torch.tensor(final),
            last_chunk=final,
            num_processed_tokens=target,
            chunk_seq=int(transfer_manager.put_req_chunk[key]),
            replace_runtime_additional_information=True,
        ),
    )
    if final:
        # The transfer manager also clears this map on cancellation. There
        # is no additional producer cache outside its existing lifecycle.
        transfer_manager.code_prompt_token_ids.pop(key, None)
    return payload
