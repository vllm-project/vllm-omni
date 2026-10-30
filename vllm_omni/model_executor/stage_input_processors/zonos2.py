# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Raw ZONOS2 frame handoff; de-shear and OLA belong to the DAC worker."""

from __future__ import annotations

from typing import Any

import torch

from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct
from vllm_omni.model_executor.models.zonos2.zonos2_codec import eos_boundary, shear
from vllm_omni.model_executor.models.zonos2.zonos2_keys import TARGET_FRAMES


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


def talker2dac(source_outputs: list[Any], prompt: Any = None, _requires_multimodal_data: bool = False) -> list[Any]:
    from vllm_omni.inputs.data import OmniTokensPrompt

    inputs = []
    for source in source_outputs:
        if not source.finished:
            continue
        mm = source.outputs[0].multimodal_output or {}
        audio = mm.get("codes", {}).get("audio")
        frames = _frames(audio) if isinstance(audio, torch.Tensor) else torch.empty((0, 9), dtype=torch.long)
        eos = _eos_frame(mm.get("meta", {}))
        if eos is None:
            eos = eos_boundary(frames)
        inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0],
                additional_information={
                    "codes": {"audio": frames.T.contiguous()},
                    TARGET_FRAMES: _target(len(frames), eos, True),
                    "meta": {"finished": True, "chunk_seq": 0},
                },
            )
        )
    return inputs


def talker2dac_async_chunk(
    transfer_manager: Any,
    multimodal_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
    new_token_ids: tuple[int, ...] = (),
) -> OmniPayloadStruct | None:
    key = request.external_req_id
    final = bool(is_finished or request.is_finished())
    buffer = transfer_manager.code_prompt_token_ids[key]
    mm = multimodal_output or {}
    audio = mm.get("codes", {}).get("audio")
    # save_async snapshots new_token_ids; a terminal flush with no sampled
    # token must not append the last frame a second time.
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
