# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processor for IndexTTS2: Talker (GPT AR) → S2Mel decoder."""

from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


def _shape(tensor: Any) -> tuple[int, ...] | None:
    return tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else None


def _strip_stop_token(
    codes: torch.Tensor,
    latent: torch.Tensor,
    stop_mel_token: int = 8193,
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


def talker2s2mel(
    source_outputs: list[Any],
    prompt: Any = None,
    _requires_multimodal_data: bool = False,
) -> list[Any]:
    """Non-async: collect all Stage 0 output, format for Stage 1 S2Mel decoder."""
    # Clamp intra-op threads: this runs in the orchestrator/APIServer process
    # where torch defaults to all cores; the tiny CPU tensor ops below pay
    # ~85ms of OMP fork/join overhead per request otherwise (measured 4090 host).
    _old_nt = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _talker2s2mel_impl(source_outputs)
    finally:
        torch.set_num_threads(_old_nt)


def _talker2s2mel_impl(source_outputs):
    from vllm_omni.inputs.data import OmniTokensPrompt

    talker_outputs = source_outputs
    s2mel_inputs: list[OmniTokensPrompt] = []

    for i, talker_output in enumerate(talker_outputs):
        if not talker_output.finished:
            continue

        output = talker_output.outputs[0]
        mm = output.multimodal_output
        if not isinstance(mm, dict):
            logger.warning("Talker output %d has no multimodal_output dict", i)
            continue

        # Extract mel_codes
        codes_dict = mm.get("codes", {})
        mel_codes = codes_dict.get("mel")
        if not isinstance(mel_codes, torch.Tensor) or mel_codes.numel() == 0:
            logger.warning("Talker output %d has empty mel_codes", i)
            continue
        mel_codes = mel_codes.to(torch.long)

        # Extract latent hidden states
        hs_dict = mm.get("hidden_states", {})
        latent = hs_dict.get("latent")
        if not isinstance(latent, torch.Tensor) or latent.numel() == 0:
            logger.warning("Talker output %d has empty latent", i)
            continue

        # Extract metadata from Stage 0
        meta = mm.get("meta", {})
        s_ref = meta.get("S_ref")
        ref_mel = meta.get("ref_mel")
        style = meta.get("style")

        logger.debug(
            "[talker2s2mel] shapes — mel_codes=%s, latent=%s, S_ref=%s, ref_mel=%s, style=%s",
            _shape(mel_codes),
            _shape(latent),
            _shape(s_ref),
            _shape(ref_mel),
            _shape(style),
        )

        # Official infer_v2.py only strips at the first stop token in this path.
        mel_codes_clean, latent_clean, code_lens = _strip_stop_token(mel_codes, latent)

        logger.debug(
            "[talker2s2mel] after stop trim — mel_codes=%s→%s, latent=%s→%s, max_code_len=%d",
            _shape(mel_codes),
            _shape(mel_codes_clean),
            _shape(latent),
            _shape(latent_clean),
            int(code_lens.max().item()) if code_lens.numel() else 0,
        )

        # Build additional_information for Stage 1.
        # .clone() ensures tensors are independent copies — without it,
        # .float().cpu() is a no-op when the source is already float32+CPU,
        # and the resulting shared reference can cause batch-dim inflation
        # in model_intermediate_buffer under concurrent requests.
        additional_information = {
            "latent": latent_clean.float().cpu().clone(),
            "mel_codes": mel_codes_clean.cpu().clone(),
            "code_lens": code_lens.cpu().clone(),
        }
        if isinstance(s_ref, torch.Tensor):
            additional_information["S_ref"] = s_ref.float().cpu().clone()
        else:
            logger.warning("[talker2s2mel] S_ref MISSING — Stage 1 will skip ref conditioning")
        if isinstance(ref_mel, torch.Tensor):
            additional_information["ref_mel"] = ref_mel.float().cpu().clone()
        else:
            logger.warning("[talker2s2mel] ref_mel MISSING — Stage 1 will skip ref conditioning")
        if isinstance(style, torch.Tensor):
            additional_information["style"] = style.float().cpu().clone()
        else:
            logger.warning("[talker2s2mel] style MISSING — Stage 1 will use zeros")

        s2mel_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0],  # dummy token for vLLM scheduler
                multi_modal_data=None,
                mm_processor_kwargs=None,
                additional_information=additional_information,
            )
        )

    return s2mel_inputs
