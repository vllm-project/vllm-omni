# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processor for IndexTTS2: Talker (GPT AR) → S2Mel decoder."""

from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

STOP_MEL_TOKEN = 8193


def _cpu_view(tensor: torch.Tensor) -> torch.Tensor:
    """Return a contiguous CPU tensor suitable for connector serialization.

    Always returns an owning copy so downstream consumers cannot mutate the
    source tensors that may still be referenced by the producer stage.
    """
    out = tensor.detach()
    if out.device.type != "cpu":
        return out.cpu().contiguous()
    return out.clone().contiguous()


def _strip_stop_token(
    codes: torch.Tensor,
    latent: torch.Tensor,
    stop_mel_token: int = STOP_MEL_TOKEN,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Strip at the first stop token, matching official IndexTTS2 v2.

    Returns: (codes [B, T'], latent [B, T', D], code_lens [B]).
    """
    if codes.ndim == 1:
        codes = codes.unsqueeze(0)
    if latent.ndim == 2:
        latent = latent.unsqueeze(0)

    if codes.device.type != "cpu":
        codes = codes.detach().cpu()
    if latent.device.type != "cpu":
        latent = latent.detach().cpu()

    device = codes.device
    code_lens = []
    codes_out = []
    latent_out = []

    for i in range(codes.shape[0]):
        code = codes[i]
        lat = latent[i]

        stop_mask = (code == stop_mel_token).nonzero(as_tuple=False)
        if stop_mask.numel() > 0:
            valid_len = int(stop_mask[0].item())
        else:
            valid_len = int(code.shape[0])
        code_lens.append(valid_len)
        codes_out.append(code[:valid_len])
        latent_out.append(lat[:valid_len])

    max_len = max(code_lens) if code_lens else 0
    if max_len == 0:
        return (
            torch.zeros(codes.shape[0], 0, dtype=torch.long, device=device),
            torch.zeros(codes.shape[0], 0, latent.shape[-1], device=device, dtype=latent.dtype),
            torch.zeros(codes.shape[0], dtype=torch.long, device=device),
        )

    padded_codes = torch.zeros((len(codes_out), max_len), dtype=torch.long, device=device)
    padded_latent = torch.zeros(
        len(latent_out),
        max_len,
        latent_out[0].shape[-1],
        device=device,
        dtype=latent_out[0].dtype,
    )
    for i, (c, lat) in enumerate(zip(codes_out, latent_out)):
        padded_codes[i, : c.shape[0]] = c
        padded_latent[i, : lat.shape[0]] = lat

    return padded_codes, padded_latent, torch.tensor(code_lens, dtype=torch.long, device=device)


def _normalize_mel_sequence(mel_codes: torch.Tensor) -> torch.Tensor:
    mel_codes = mel_codes.to(torch.long)
    if mel_codes.ndim <= 1:
        return mel_codes.reshape(-1).contiguous() if mel_codes.ndim == 0 else mel_codes.contiguous()
    if mel_codes.ndim == 2 and (mel_codes.shape[0] == 1 or mel_codes.shape[1] == 1):
        return mel_codes.reshape(-1).contiguous()
    return mel_codes.reshape(-1).contiguous()


def _normalize_latent_sequence(latent: torch.Tensor) -> torch.Tensor:
    if latent.ndim <= 1:
        return latent.reshape(1, -1).contiguous()
    if latent.ndim == 2:
        return latent.contiguous()
    if latent.ndim == 3 and latent.shape[0] == 1:
        return latent[0].contiguous()
    return latent.reshape(-1, latent.shape[-1]).contiguous()


def _build_s2mel_additional_information(
    mel_codes: torch.Tensor,
    latent: torch.Tensor,
    meta: dict[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    """Build the Stage-1 S2Mel tensor contract shared by legacy and connector paths."""
    mel_codes_clean, latent_clean, code_lens = _strip_stop_token(mel_codes, latent)

    additional_information = {
        "latent": _cpu_view(latent_clean),
        "mel_codes": _cpu_view(mel_codes_clean),
        "code_lens": _cpu_view(code_lens),
    }

    for key in ("S_ref", "ref_mel", "style"):
        val = meta.get(key)
        if isinstance(val, torch.Tensor):
            additional_information[key] = _cpu_view(val)
        else:
            logger.warning("[%s] %s MISSING — Stage 1 will use fallback", context, key)

    return additional_information


def _get_payload_value(pooling_output: dict[str, Any], dotted_key: str, nested_parent: str, nested_key: str) -> Any:
    value = pooling_output.get(dotted_key)
    if value is not None:
        return value
    nested = pooling_output.get(nested_parent)
    if isinstance(nested, dict):
        return nested.get(nested_key)
    return None


def _request_id(request: Any) -> str:
    return str(getattr(request, "external_req_id", None) or getattr(request, "request_id", "?"))


def talker2s2mel_token_only(
    source_outputs: list[Any],
    prompt: Any = None,
    _requires_multimodal_data: bool = False,
) -> list[Any]:
    """Sync-side placeholder for Stage 1; tensors arrive via full-payload connector."""
    from vllm_omni.inputs.data import OmniTokensPrompt

    del prompt
    s2mel_inputs: list[OmniTokensPrompt] = []
    for talker_output in source_outputs:
        if not talker_output.finished:
            continue
        s2mel_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0],
                additional_information=None,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return s2mel_inputs


def talker2s2mel_full_payload(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: Any,
    **_: Any,
) -> dict[str, Any] | None:
    """Build the complete S2Mel input from accumulated per-step talker deltas."""
    del transfer_manager
    rid = _request_id(request)
    if not isinstance(pooling_output, dict):
        logger.warning("talker2s2mel_full_payload: pooling_output not a dict for req=%s", rid)
        return None

    mel_codes = _get_payload_value(pooling_output, "codes.mel", "codes", "mel")
    if not isinstance(mel_codes, torch.Tensor) or mel_codes.numel() == 0:
        logger.warning("talker2s2mel_full_payload: missing codes.mel for req=%s", rid)
        return None

    latent = _get_payload_value(pooling_output, "hidden_states.latent", "hidden_states", "latent")
    if not isinstance(latent, torch.Tensor) or latent.numel() == 0:
        logger.warning("talker2s2mel_full_payload: missing hidden_states.latent for req=%s", rid)
        return None

    mel_seq = _normalize_mel_sequence(mel_codes)
    latent_seq = _normalize_latent_sequence(latent)
    if mel_seq.numel() == 0 or latent_seq.numel() == 0:
        logger.warning("talker2s2mel_full_payload: empty normalized mel/latent for req=%s", rid)
        return None

    common_len = min(int(mel_seq.shape[0]), int(latent_seq.shape[0]))
    if common_len <= 0:
        logger.warning("talker2s2mel_full_payload: no common mel/latent length for req=%s", rid)
        return None

    mel_seq = mel_seq[:common_len]
    latent_seq = latent_seq[:common_len]

    meta = {
        "S_ref": _get_payload_value(pooling_output, "meta.S_ref", "meta", "S_ref"),
        "ref_mel": _get_payload_value(pooling_output, "meta.ref_mel", "meta", "ref_mel"),
        "style": _get_payload_value(pooling_output, "meta.style", "meta", "style"),
    }
    return _build_s2mel_additional_information(mel_seq, latent_seq, meta, context="full_payload")
