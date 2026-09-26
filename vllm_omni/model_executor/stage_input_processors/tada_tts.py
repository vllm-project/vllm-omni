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
    """Batch mode: pass accumulated acoustic features and durations to the vocoder."""
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


# Acoustic feature dimension produced by the AR stage.
ACOUSTIC_DIM = 512

# Default streaming chunk geometry (in 50 Hz frames) when the connector config omits them.
_DEFAULT_CHUNK_FRAMES = 50
_DEFAULT_LEFT_CTX_FRAMES = 128
_DEFAULT_RIGHT_CTX_FRAMES = 32


def _expand_token(acoustic: torch.Tensor, tb: int, *, lead: bool) -> list[torch.Tensor]:
    """Expand one token into frames: ``tb - 1`` leading silence frames followed by the
    acoustic frame. The first token (``lead=False``) drops its leading silence."""
    parts: list[torch.Tensor] = []
    nz = max(0, int(tb) - 1)
    if lead and nz > 0:
        parts.append(torch.zeros(nz, ACOUSTIC_DIM, dtype=torch.float32))
    parts.append(acoustic.reshape(1, ACOUSTIC_DIM).to(torch.float32))
    return parts


def _stream_chunk_cfg(transfer_manager: Any) -> tuple[int, int, int]:
    """Read the chunk/left/right frame counts from the connector's ``extra`` config."""
    connector = getattr(transfer_manager, "connector", None)
    raw = getattr(connector, "config", {}) or {}
    cfg = raw.get("extra", raw) if isinstance(raw, dict) else {}
    chunk = int(cfg.get("tada_chunk_frames", _DEFAULT_CHUNK_FRAMES))
    left = int(cfg.get("tada_left_context_frames", _DEFAULT_LEFT_CTX_FRAMES))
    right = int(cfg.get("tada_right_context_frames", _DEFAULT_RIGHT_CTX_FRAMES))
    return chunk, left, right


def ar2vocoder_async_chunk(
    transfer_manager: Any,
    multimodal_output: dict[str, Any] | None = None,
    request: Any = None,
    is_finished: bool = False,
) -> Any:
    """Stream audio as overlapping decode windows.

    Called once per AR step with that step's ``acoustic_features`` and ``time_before``.
    Each token's duration is expanded into a running frame buffer. Once a chunk's worth of
    new frames plus the right-lookahead margin is available, one window
    ``buffer[emit - left : emit + chunk + right]`` is emitted as a 2-D ``codes.audio`` tensor.
    ``meta.left_context_size`` / ``meta.right_holdback_size`` tell the vocoder how many frames
    to trim from each end so only the new ``[emit, emit + chunk)`` region reaches the output.
    The codec needs both left context and right lookahead to make a windowed decode match a
    full decode. At finish the remaining tail is flushed without right lookahead.
    """
    from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct

    request_id = request.external_req_id
    finished = bool(is_finished or request.is_finished())
    chunk_frames, left_ctx, right_ctx = _stream_chunk_cfg(transfer_manager)

    streams = getattr(transfer_manager, "_tada_stream", None)
    if streams is None:
        streams = transfer_manager._tada_stream = {}
    state = streams.setdefault(
        request_id,
        {"buf": torch.zeros(0, ACOUSTIC_DIM, dtype=torch.float32), "emit": 0, "started": False, "last_tb": 1},
    )

    # Append the new step's frames to the running buffer (shape [num_frames, ACOUSTIC_DIM]).
    if isinstance(multimodal_output, dict):
        af = multimodal_output.get("acoustic_features")
        if isinstance(af, torch.Tensor) and af.numel() > 0:
            af = af.detach().cpu().reshape(-1, ACOUSTIC_DIM)
            tbv = multimodal_output.get("time_before")
            tbv = (
                tbv.detach().cpu().reshape(-1).to(torch.long)
                if isinstance(tbv, torch.Tensor) and tbv.numel() >= af.shape[0]
                else torch.ones(af.shape[0], dtype=torch.long)
            )
            new_parts: list[torch.Tensor] = []
            for i in range(af.shape[0]):
                tb_i = int(tbv[i].item())
                # First token drops its leading silence; later tokens keep the inter-token gap.
                new_parts.extend(_expand_token(af[i], tb_i, lead=state["started"]))
                state["started"] = True
                state["last_tb"] = tb_i
            if new_parts:
                state["buf"] = torch.cat([state["buf"], *new_parts], dim=0)

    def _emit(window: torch.Tensor, left_drop: int, right_drop: int, fin: bool) -> Any:
        return OmniPayloadStruct(
            codes=CodesStruct(audio=window.contiguous()),
            meta=MetaStruct(
                left_context_size=int(left_drop),
                right_holdback_size=int(right_drop),
                finished=torch.tensor(bool(fin), dtype=torch.bool),
            ),
        )

    buf = state["buf"]
    emit = state["emit"]
    total = buf.shape[0]

    if not finished:
        # Wait until a full chunk plus the right-lookahead margin is buffered.
        if total - emit < chunk_frames + right_ctx:
            return None
        start = max(0, emit - left_ctx)
        window = buf[start : emit + chunk_frames + right_ctx]
        payload = _emit(window, emit - start, right_ctx, fin=False)
        state["emit"] = emit + chunk_frames
        return payload

    # Finish: append the trailing silence, then flush the remaining frames as the final window.
    if state["last_tb"] > 0:
        buf = state["buf"] = torch.cat(
            [buf, torch.zeros(int(state["last_tb"]), ACOUSTIC_DIM, dtype=torch.float32)], dim=0
        )
        total = buf.shape[0]
    if emit >= total:
        return _emit(torch.zeros(0, ACOUSTIC_DIM, dtype=torch.float32), 0, 0, fin=True)
    start = max(0, emit - left_ctx)
    payload = _emit(buf[start:total], emit - start, 0, fin=True)
    state["emit"] = total
    streams.pop(request_id, None)
    return payload
