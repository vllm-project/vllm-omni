"""Stage input processor for TADA TTS: AR Stage → Vocoder."""

from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


def ar2vocoder(
    source_outputs: list[Any],
    _prompt: Any = None,
    _requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[Any]:
    """Batch mode: pass accumulated acoustic_features [T, 512] + durations to TadaVocoder."""
    from vllm_omni.inputs.data import OmniTokensPrompt

    vocoder_inputs: list[OmniTokensPrompt] = []

    for ar_output in source_outputs:
        if not getattr(ar_output, "finished", True):
            # Batch mode processes only fully-finished AR outputs.
            continue

        output = ar_output.outputs[0]
        mm = getattr(output, "multimodal_output", None) or {}

        af = mm.get("acoustic_features")
        tm = mm.get("text_token_mask")
        tb = mm.get("time_before")

        if not isinstance(af, torch.Tensor) or af.numel() == 0:
            logger.warning("ar2vocoder: no acoustic_features in Stage 0 output; skipping request")
            continue

        af = af.cpu().contiguous()
        T = af.shape[0]

        if isinstance(tm, torch.Tensor) and tm.numel() == T:
            tm = tm.cpu().contiguous()
        else:
            tm = torch.ones(T, dtype=torch.long)

        # Per-token durations (frames) needed by the vocoder for frame expansion.
        if isinstance(tb, torch.Tensor) and tb.numel() == T:
            tb = tb.cpu().contiguous().to(torch.long)
        else:
            tb = torch.ones(T, dtype=torch.long)

        dummy_ids = [0] * T  # one per frame for vLLM-Omni sequence-length bookkeeping

        vocoder_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=dummy_ids,
                multi_modal_data=None,
                mm_processor_kwargs=None,
                additional_information={
                    "acoustic_features": af,
                    "text_token_mask": tm,
                    "time_before": tb,
                },
            )
        )

    return vocoder_inputs


def ar2vocoder_async_chunk(
    transfer_manager: Any,
    multimodal_output: dict[str, Any] | None = None,
    request: Any = None,
    is_finished: bool = False,
) -> Any:
    """Async-chunk mode: accumulate per-step acoustic features, emit once at finish.

    The chunk-transfer adapter calls this each step with the AR stage's per-step
    ``multimodal_output`` and expects an ``OmniPayloadStruct`` (or ``None`` to defer).
    TADA carries *continuous* acoustic features (no discrete-code slot like qwen3). The
    receive side (``_poll_single_request``) only preserves a **2-D** ``codes.audio`` tensor
    in the downstream request info (1-D is consumed into ``prompt_token_ids`` and dropped;
    other fields like ``latent`` are not routed to the vocoder). So we pack BOTH signals
    into one 2-D tensor: ``codes.audio[:, :512]`` = acoustic features, ``codes.audio[:, 512]``
    = per-token duration (``time_before``). ``TadaVocoder.forward`` unpacks them. We
    accumulate every step and emit only at finish, so the codec decodes the whole sequence
    in one pass (no chunk seams). Trimmed/transition frames never arrive (``make_omni_output``
    skips them).
    """
    from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct

    request_id = request.external_req_id
    finished = bool(is_finished or request.is_finished())

    if not hasattr(transfer_manager, "_tada_feat_buf"):
        transfer_manager._tada_feat_buf = {}
        transfer_manager._tada_time_buf = {}
    feat_buf: dict[str, list[torch.Tensor]] = transfer_manager._tada_feat_buf
    time_buf: dict[str, list[torch.Tensor]] = transfer_manager._tada_time_buf

    if isinstance(multimodal_output, dict):
        af = multimodal_output.get("acoustic_features")
        if isinstance(af, torch.Tensor) and af.numel() > 0:
            feat_buf.setdefault(request_id, []).append(af.detach().cpu().reshape(-1, 512))
            tb = multimodal_output.get("time_before")
            tb = (tb.detach().cpu().reshape(-1).to(torch.float32)
                  if isinstance(tb, torch.Tensor) and tb.numel() > 0
                  else torch.ones(af.reshape(-1, 512).shape[0], dtype=torch.float32))
            time_buf.setdefault(request_id, []).append(tb)

    if not finished:
        return None

    frames = feat_buf.pop(request_id, [])
    times = time_buf.pop(request_id, [])
    acoustic_features = torch.cat(frames, dim=0) if frames else torch.zeros((0, 512), dtype=torch.float32)
    T = acoustic_features.shape[0]
    time_before = torch.cat(times, dim=0) if times else torch.ones(T, dtype=torch.float32)

    # Pack acoustic [T,512] + duration [T,1] -> 2-D codes.audio [T,513] (the only field
    # the receive side keeps in the vocoder's request info; see _poll_single_request).
    packed = torch.cat([acoustic_features, time_before.reshape(-1, 1)], dim=-1).contiguous()
    return OmniPayloadStruct(
        codes=CodesStruct(audio=packed),
        meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)),
    )
