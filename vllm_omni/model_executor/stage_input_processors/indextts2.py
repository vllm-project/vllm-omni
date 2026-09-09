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
    latent: torch.Tensor | None,
    stop_mel_token: int = STOP_MEL_TOKEN,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Strip at the first stop token, matching official IndexTTS2 v2.

    Returns codes, optional aligned latent, and code lengths.
    """
    if codes.ndim == 1:
        codes = codes.unsqueeze(0)
    if latent is not None and latent.ndim == 2:
        latent = latent.unsqueeze(0)

    if codes.device.type != "cpu":
        codes = codes.detach().cpu()
    if latent is not None and latent.device.type != "cpu":
        latent = latent.detach().cpu()

    device = codes.device
    code_lens = []
    codes_out = []
    latent_out = []

    for i in range(codes.shape[0]):
        code = codes[i]
        stop_mask = (code == stop_mel_token).nonzero(as_tuple=False)
        if stop_mask.numel() > 0:
            valid_len = int(stop_mask[0].item())
        else:
            valid_len = int(code.shape[0])
        code_lens.append(valid_len)
        codes_out.append(code[:valid_len])
        if latent is not None:
            latent_out.append(latent[i, :valid_len])

    max_len = max(code_lens) if code_lens else 0
    if max_len == 0:
        empty_codes = torch.zeros(codes.shape[0], 0, dtype=torch.long, device=device)
        empty_latent = (
            torch.zeros(
                codes.shape[0],
                0,
                latent.shape[-1],
                device=device,
                dtype=latent.dtype,
            )
            if latent is not None
            else None
        )
        return (
            empty_codes,
            empty_latent,
            torch.zeros(
                codes.shape[0],
                dtype=torch.long,
                device=device,
            ),
        )

    padded_codes = torch.zeros((len(codes_out), max_len), dtype=torch.long, device=device)
    padded_latent = (
        torch.zeros(
            len(latent_out),
            max_len,
            latent_out[0].shape[-1],
            device=device,
            dtype=latent_out[0].dtype,
        )
        if latent_out
        else None
    )
    for i, c in enumerate(codes_out):
        padded_codes[i, : c.shape[0]] = c
        if padded_latent is not None:
            lat = latent_out[i]
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
    latent: torch.Tensor | None,
    meta: dict[str, Any],
    *,
    use_gpt_latent: bool,
    context: str,
) -> dict[str, Any]:
    """Build the Stage-1 S2Mel tensor contract shared by legacy and connector paths."""
    mel_codes_clean, latent_clean, code_lens = _strip_stop_token(mel_codes, latent)

    additional_information = {
        "mel_codes": _cpu_view(mel_codes_clean),
        "code_lens": _cpu_view(code_lens),
        "use_gpt_latent": use_gpt_latent,
    }
    if use_gpt_latent:
        if latent_clean is None:
            raise ValueError(f"{context} requires GPT latent but none was provided")
        additional_information["latent"] = _cpu_view(latent_clean)

    duration_factor = meta.get("duration_factor", 1.0)
    additional_information["duration_factor"] = 1.0 if duration_factor is None else float(duration_factor)

    for key in ("S_ref", "ref_mel", "style"):
        val = meta.get(key)
        if not isinstance(val, torch.Tensor) or val.numel() == 0:
            raise ValueError(f"{context} requires non-empty IndexTTS conditioning field {key}")
        additional_information[key] = _cpu_view(val)

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


def _request_seed(request: Any) -> int | None:
    sampling_params = getattr(request, "sampling_params", None)
    seed = getattr(sampling_params, "seed", None)
    return int(seed) if seed is not None else None


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
        raise ValueError(f"IndexTTS Stage 0 payload must be a dict for req={rid}")

    use_gpt_latent_value = _get_payload_value(
        pooling_output,
        "meta.use_gpt_latent",
        "meta",
        "use_gpt_latent",
    )
    if use_gpt_latent_value is None:
        raise ValueError(f"IndexTTS payload is missing meta.use_gpt_latent for req={rid}")
    use_gpt_latent = bool(use_gpt_latent_value)
    payload_latent = _get_payload_value(
        pooling_output,
        "hidden_states.latent",
        "hidden_states",
        "latent",
    )
    payload_mel_codes = _get_payload_value(
        pooling_output,
        "codes.mel",
        "codes",
        "mel",
    )
    if use_gpt_latent:
        mel_codes = payload_mel_codes
        latent = payload_latent
    else:
        output_token_ids = list(getattr(request, "output_token_ids", None) or [])
        if not output_token_ids:
            raise ValueError(f"IndexTTS code-only payload has no completed output_token_ids for req={rid}")
        mel_codes = torch.tensor(output_token_ids, dtype=torch.long)
        latent = None

    if not isinstance(mel_codes, torch.Tensor) or mel_codes.numel() == 0:
        raise ValueError(f"IndexTTS payload is missing mel codes for req={rid} (use_gpt_latent={use_gpt_latent})")

    mel_seq = _normalize_mel_sequence(mel_codes)
    if mel_seq.numel() == 0:
        raise ValueError(f"IndexTTS payload has an empty mel sequence for req={rid}")

    latent_seq: torch.Tensor | None = None
    if use_gpt_latent:
        if not isinstance(latent, torch.Tensor) or latent.numel() == 0:
            raise ValueError(f"IndexTTS latent payload is missing hidden_states.latent for req={rid}")
        latent_seq = _normalize_latent_sequence(latent)
        if latent_seq.numel() == 0:
            raise ValueError(f"IndexTTS payload has an empty latent sequence for req={rid}")
        if mel_seq.shape[0] != latent_seq.shape[0]:
            raise ValueError(
                f"IndexTTS mel/latent length mismatch for req={rid}: "
                f"mel={mel_seq.shape[0]} latent={latent_seq.shape[0]}"
            )

    meta = {
        "S_ref": _get_payload_value(pooling_output, "meta.S_ref", "meta", "S_ref"),
        "ref_mel": _get_payload_value(pooling_output, "meta.ref_mel", "meta", "ref_mel"),
        "style": _get_payload_value(pooling_output, "meta.style", "meta", "style"),
        "duration_factor": _get_payload_value(
            pooling_output,
            "meta.duration_factor",
            "meta",
            "duration_factor",
        ),
    }
    additional_information = _build_s2mel_additional_information(
        mel_seq,
        latent_seq,
        meta,
        use_gpt_latent=use_gpt_latent,
        context="full_payload",
    )
    seed = _request_seed(request)
    if seed is not None:
        additional_information["seed"] = seed
    return additional_information
