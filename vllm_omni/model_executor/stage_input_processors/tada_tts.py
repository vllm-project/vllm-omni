"""Stage input processor for TADA TTS: AR Stage → Vocoder."""

from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


def ar2vocoder(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> list[Any]:
    """Batch mode: pass accumulated acoustic_features [T, 512] to TadaVocoder."""
    from vllm_omni.inputs.data import OmniTokensPrompt
    from vllm_omni.model_executor.stage_input_processors.qwen3_omni import _validate_stage_inputs

    ar_outputs = _validate_stage_inputs(stage_list, engine_input_source)
    vocoder_inputs: list[OmniTokensPrompt] = []

    for ar_output in ar_outputs:
        if not ar_output.finished:
            # Batch mode processes only fully-finished AR outputs.
            continue

        output = ar_output.outputs[0]
        mm = output.multimodal_output or {}

        af = mm.get("acoustic_features")
        tm = mm.get("text_token_mask")

        if not isinstance(af, torch.Tensor) or af.numel() == 0:
            logger.warning("ar2vocoder: no acoustic_features in Stage 0 output; skipping request")
            continue

        af = af.cpu().contiguous()
        T = af.shape[0]

        if isinstance(tm, torch.Tensor) and tm.numel() == T:
            tm = tm.cpu().contiguous()
        else:
            tm = torch.ones(T, dtype=torch.long)

        dummy_ids = [0] * T  # one per frame for vLLM-Omni sequence-length bookkeeping

        vocoder_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=dummy_ids,
                multi_modal_data=None,
                mm_processor_kwargs=None,
                additional_information={
                    "acoustic_features": af,
                    "text_token_mask": tm,
                },
            )
        )

    return vocoder_inputs


def ar2vocoder_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> dict[str, Any] | None:
    """Async-chunk mode: accumulate per-step acoustic features, emit when finished."""
    request_id = request.external_req_id
    finished = bool(is_finished or request.is_finished())

    if not hasattr(transfer_manager, "_tada_feat_buf"):
        transfer_manager._tada_feat_buf = {}
    feat_buf: dict[str, list[torch.Tensor]] = transfer_manager._tada_feat_buf

    if isinstance(pooling_output, dict):
        af = pooling_output.get("acoustic_features")
        if isinstance(af, torch.Tensor) and af.numel() > 0:
            feat_buf.setdefault(request_id, []).append(af.detach().cpu())

    if not finished:
        return None

    frames = feat_buf.pop(request_id, [])
    if frames:
        acoustic_features = torch.cat(frames, dim=0)  # [T, 512]
    else:
        acoustic_features = torch.zeros((0, 512), dtype=torch.float32)

    T = acoustic_features.shape[0]
    text_token_mask = torch.ones(T, dtype=torch.long)

    return {
        "acoustic_features": acoustic_features,
        "text_token_mask": text_token_mask,
        "code_predictor_codes": [0] * max(T, 1),  # dummy ids for sequence-length bookkeeping
        "finished": True,
    }
