# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processor for IndexTTS2: Talker (GPT AR) → S2Mel decoder."""

from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

STOP_MEL_TOKEN = 8193


def _shape(tensor: Any) -> tuple[int, ...] | None:
    return tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else None


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

    device = codes.device
    code_lens = []
    codes_out = []
    latent_out = []

    for i in range(codes.shape[0]):
        code = codes[i]
        lat = latent[i]

        # Find stop token
        stop_mask = (code == stop_mel_token).nonzero(as_tuple=False)
        if stop_mask.numel() > 0:
            valid_len = int(stop_mask[0].item())
        else:
            valid_len = int(code.shape[0])
        code = code[:valid_len]
        lat = lat[:valid_len]

        code_lens.append(int(code.shape[0]))
        codes_out.append(code)
        latent_out.append(lat)

    # Pad to max length
    max_len = max(code_lens) if code_lens else 0
    if max_len == 0:
        return (
            torch.zeros(codes.shape[0], 0, dtype=torch.long, device=device),
            torch.zeros(codes.shape[0], 0, latent.shape[-1], device=device, dtype=latent.dtype),
            torch.zeros(codes.shape[0], dtype=torch.long, device=device),
        )

    padded_codes = torch.full((len(codes_out), max_len), stop_mel_token, dtype=torch.long, device=device)
    lat_dtype = latent_out[0].dtype
    padded_latent = torch.zeros(
        len(latent_out),
        max_len,
        latent_out[0].shape[-1],
        device=device,
        dtype=lat_dtype,
    )
    for i, (c, lat) in enumerate(zip(codes_out, latent_out)):
        padded_codes[i, : c.shape[0]] = c
        padded_latent[i, : lat.shape[0]] = lat

    return padded_codes, padded_latent, torch.tensor(code_lens, dtype=torch.long, device=device)


def _normalize_mel_sequence(mel_codes: torch.Tensor) -> torch.Tensor:
    """Normalize accumulated full-payload mel rows to one 1-D sequence."""
    mel_codes = mel_codes.to(torch.long)
    if mel_codes.ndim == 0:
        return mel_codes.reshape(1)
    if mel_codes.ndim == 1:
        return mel_codes.contiguous()
    if mel_codes.ndim == 2:
        if mel_codes.shape[1] == 1:
            return mel_codes[:, 0].contiguous()
        if mel_codes.shape[0] == 1:
            return mel_codes[0].contiguous()
    return mel_codes.reshape(-1).contiguous()


def _normalize_latent_sequence(latent: torch.Tensor) -> torch.Tensor:
    """Normalize accumulated full-payload latent rows to [T, D]."""
    if latent.ndim == 1:
        return latent.reshape(1, -1).contiguous()
    if latent.ndim == 2:
        return latent.contiguous()
    if latent.ndim == 3 and latent.shape[0] == 1:
        return latent[0].contiguous()
    if latent.ndim >= 3:
        return latent.reshape(-1, latent.shape[-1]).contiguous()
    return latent.contiguous()


def _cpu_float_clone(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().cpu().clone()


def _cpu_clone(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.cpu().clone()


def _build_s2mel_additional_information(
    mel_codes: torch.Tensor,
    latent: torch.Tensor,
    meta: dict[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    """Build the Stage-1 S2Mel tensor contract shared by legacy and connector paths."""
    mel_codes_clean, latent_clean, code_lens = _strip_stop_token(mel_codes, latent)

    logger.debug(
        "[%s] after stop trim — mel_codes=%s→%s, latent=%s→%s, max_code_len=%d",
        context,
        _shape(mel_codes),
        _shape(mel_codes_clean),
        _shape(latent),
        _shape(latent_clean),
        int(code_lens.max().item()) if code_lens.numel() else 0,
    )

    additional_information = {
        "latent": _cpu_float_clone(latent_clean),
        "mel_codes": _cpu_clone(mel_codes_clean),
        "code_lens": _cpu_clone(code_lens),
    }

    s_ref = meta.get("S_ref")
    ref_mel = meta.get("ref_mel")
    style = meta.get("style")
    if isinstance(s_ref, torch.Tensor):
        additional_information["S_ref"] = _cpu_float_clone(s_ref)
    else:
        logger.warning("[%s] S_ref MISSING — Stage 1 will skip ref conditioning", context)
    if isinstance(ref_mel, torch.Tensor):
        additional_information["ref_mel"] = _cpu_float_clone(ref_mel)
    else:
        logger.warning("[%s] ref_mel MISSING — Stage 1 will skip ref conditioning", context)
    if isinstance(style, torch.Tensor):
        additional_information["style"] = _cpu_float_clone(style)
    else:
        logger.warning("[%s] style MISSING — Stage 1 will use zeros", context)

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


def talker2s2mel(
    source_outputs: list[Any],
    prompt: Any = None,
    _requires_multimodal_data: bool = False,
) -> list[Any]:
    """Legacy orchestrator path: collect all Stage 0 output, format Stage 1 input."""
    # Clamp intra-op threads: this runs in the orchestrator/APIServer process
    # where torch defaults to all cores; the tiny CPU tensor ops below pay
    # ~85ms of OMP fork/join overhead per request otherwise (measured 4090 host).
    _old_nt = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _talker2s2mel_impl(source_outputs)
    finally:
        torch.set_num_threads(_old_nt)


def _talker2s2mel_impl(source_outputs: list[Any]) -> list[Any]:
    from vllm_omni.inputs.data import OmniTokensPrompt

    s2mel_inputs: list[OmniTokensPrompt] = []

    for i, talker_output in enumerate(source_outputs):
        if not talker_output.finished:
            continue

        output = talker_output.outputs[0]
        mm = output.multimodal_output
        if not isinstance(mm, dict):
            logger.warning("Talker output %d has no multimodal_output dict", i)
            continue

        codes_dict = mm.get("codes", {})
        mel_codes = codes_dict.get("mel")
        if not isinstance(mel_codes, torch.Tensor) or mel_codes.numel() == 0:
            logger.warning("Talker output %d has empty mel_codes", i)
            continue
        mel_codes = mel_codes.to(torch.long)

        hs_dict = mm.get("hidden_states", {})
        latent = hs_dict.get("latent")
        if not isinstance(latent, torch.Tensor) or latent.numel() == 0:
            logger.warning("Talker output %d has empty latent", i)
            continue

        meta = mm.get("meta", {})
        if not isinstance(meta, dict):
            meta = {}

        logger.debug(
            "[talker2s2mel] shapes — mel_codes=%s, latent=%s, S_ref=%s, ref_mel=%s, style=%s",
            _shape(mel_codes),
            _shape(latent),
            _shape(meta.get("S_ref")),
            _shape(meta.get("ref_mel")),
            _shape(meta.get("style")),
        )

        additional_information = _build_s2mel_additional_information(mel_codes, latent, meta, context="talker2s2mel")
        s2mel_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0],  # dummy token for vLLM scheduler
                multi_modal_data=None,
                mm_processor_kwargs=None,
                additional_information=additional_information,
            )
        )

    return s2mel_inputs


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
        logger.warning(
            "indextts2.talker2s2mel_full_payload: pooling_output not a dict (type=%s) for req=%s; "
            "consumer wait gate may hang.",
            type(pooling_output).__name__,
            rid,
        )
        return None

    mel_codes = _get_payload_value(pooling_output, "codes.mel", "codes", "mel")
    if not isinstance(mel_codes, torch.Tensor) or mel_codes.numel() == 0:
        logger.warning(
            "indextts2.talker2s2mel_full_payload: missing/empty codes.mel (keys=%s) for req=%s; "
            "consumer wait gate may hang.",
            list(pooling_output.keys()),
            rid,
        )
        return None

    latent = _get_payload_value(pooling_output, "hidden_states.latent", "hidden_states", "latent")
    if not isinstance(latent, torch.Tensor) or latent.numel() == 0:
        logger.warning(
            "indextts2.talker2s2mel_full_payload: missing/empty hidden_states.latent (keys=%s) for req=%s; "
            "consumer wait gate may hang.",
            list(pooling_output.keys()),
            rid,
        )
        return None

    mel_seq = _normalize_mel_sequence(mel_codes)
    latent_seq = _normalize_latent_sequence(latent)
    if mel_seq.numel() == 0 or latent_seq.numel() == 0 or latent_seq.shape[0] == 0:
        logger.warning("indextts2.talker2s2mel_full_payload: empty normalized mel/latent for req=%s", rid)
        return None

    common_len = min(int(mel_seq.shape[0]), int(latent_seq.shape[0]))
    if common_len <= 0:
        logger.warning("indextts2.talker2s2mel_full_payload: no common mel/latent length for req=%s", rid)
        return None
    if int(mel_seq.shape[0]) != int(latent_seq.shape[0]):
        logger.warning(
            "indextts2.talker2s2mel_full_payload: mel/latent length mismatch for req=%s; cropping %d/%d to %d",
            rid,
            int(mel_seq.shape[0]),
            int(latent_seq.shape[0]),
            common_len,
        )
    mel_seq = mel_seq[:common_len]
    latent_seq = latent_seq[:common_len]

    meta = {
        "S_ref": _get_payload_value(pooling_output, "meta.S_ref", "meta", "S_ref"),
        "ref_mel": _get_payload_value(pooling_output, "meta.ref_mel", "meta", "ref_mel"),
        "style": _get_payload_value(pooling_output, "meta.style", "meta", "style"),
    }
    additional_information = _build_s2mel_additional_information(
        mel_seq,
        latent_seq,
        meta,
        context="talker2s2mel_full_payload",
    )
    additional_information["meta"] = {"finished": torch.tensor(True, dtype=torch.bool)}
    return additional_information
