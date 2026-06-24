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


# Default chunk geometry (frames @ 50 Hz) if the connector config omits them.
_DEFAULT_CHUNK_FRAMES = 50
_DEFAULT_LEFT_CTX_FRAMES = 128
_DEFAULT_RIGHT_CTX_FRAMES = 32


def _expand_token(acoustic: torch.Tensor, tb: int, *, lead: bool) -> list[torch.Tensor]:
    """Expand one token to frames: ``(tb-1)`` leading zero (silence) frames + the frame
    (mirrors upstream ``_decode_wav``). ``lead=False`` (first token) skips the leading
    zeros to drop the initial silence (upstream trims ``time_before[0]`` frames)."""
    parts: list[torch.Tensor] = []
    nz = max(0, int(tb) - 1)
    if lead and nz > 0:
        parts.append(torch.zeros(nz, 512, dtype=torch.float32))
    parts.append(acoustic.reshape(1, 512).to(torch.float32))
    return parts


def _stream_chunk_cfg(transfer_manager: Any) -> tuple[int, int, int]:
    """Read chunk geometry from the connector's ``extra`` config (qwen3 pattern)."""
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
    """Async-chunk mode: stream audio in overlapping codec-decode windows (qwen3 template).

    Called per AR step with the stage's per-step ``multimodal_output`` (acoustic_features +
    time_before, already trimmed of prompt/transition by make_omni_output). We expand each new
    token's duration into a running frame buffer and, once ``CHUNK + RIGHT_CTX`` new frames are
    available, emit ONE overlapping window ``buf[emit-LEFT_CTX : emit+CHUNK+RIGHT_CTX]`` as a 2-D
    ``codes.audio`` tensor [W, 512]. ``meta.left_context_size`` / ``meta.right_holdback_size`` tell
    the vocoder how many frames to drop from each end (left context + right lookahead); it keeps the
    ``[emit, emit+CHUNK)`` region. The 2-D ``codes.audio`` makes the non-AR receiver REPLACE the
    prior chunk and run exactly one vocoder forward per chunk; the framework CONCAT_LAST-accumulates
    each chunk's audio into the cumulative output. At finish the tail is flushed (no right lookahead).

    TADA's codec is non-causal, so RIGHT lookahead is required (unlike qwen3/glm) — verified offline
    to match a full decode to rel ~0.003.
    """
    from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct

    request_id = request.external_req_id
    finished = bool(is_finished or request.is_finished())
    chunk_frames, left_ctx, right_ctx = _stream_chunk_cfg(transfer_manager)

    st = getattr(transfer_manager, "_tada_stream", None)
    if st is None:
        st = transfer_manager._tada_stream = {}
    s = st.setdefault(
        request_id, {"buf": torch.zeros(0, 512, dtype=torch.float32), "emit": 0, "started": False, "last_tb": 1}
    )

    # Append newly generated tokens to the expanded frame buffer (row-indexed [N, 512]).
    if isinstance(multimodal_output, dict):
        af = multimodal_output.get("acoustic_features")
        if isinstance(af, torch.Tensor) and af.numel() > 0:
            af = af.detach().cpu().reshape(-1, 512)
            tbv = multimodal_output.get("time_before")
            tbv = (
                tbv.detach().cpu().reshape(-1).to(torch.long)
                if isinstance(tbv, torch.Tensor) and tbv.numel() >= af.shape[0]
                else torch.ones(af.shape[0], dtype=torch.long)
            )
            new_parts: list[torch.Tensor] = []
            for i in range(af.shape[0]):
                tb_i = int(tbv[i].item())
                first = not s["started"]
                new_parts.extend(_expand_token(af[i], tb_i, lead=not first))
                if first:
                    s["started"] = True  # drop the initial leading silence at the source
                s["last_tb"] = tb_i
            if new_parts:
                s["buf"] = torch.cat([s["buf"], *new_parts], dim=0)

    def _emit(window: torch.Tensor, left_drop: int, right_drop: int, fin: bool) -> Any:
        return OmniPayloadStruct(
            codes=CodesStruct(audio=window.contiguous()),  # [W, 512] pre-expanded frames (2-D)
            meta=MetaStruct(
                left_context_size=int(left_drop),
                right_holdback_size=int(right_drop),
                finished=torch.tensor(bool(fin), dtype=torch.bool),
            ),
        )

    buf = s["buf"]
    emit = s["emit"]
    E = buf.shape[0]

    if not finished:
        # Emit one chunk once CHUNK new frames + the right lookahead are buffered.
        if E - emit >= chunk_frames + right_ctx:
            start = max(0, emit - left_ctx)
            window = buf[start : emit + chunk_frames + right_ctx]
            payload = _emit(window, emit - start, right_ctx, fin=False)
            s["emit"] = emit + chunk_frames
            return payload
        return None

    # Finish: append the trailing silence (upstream appends time_before[-1] zero frames),
    # then flush everything remaining as one final window (no right lookahead).
    if s["last_tb"] > 0:
        buf = s["buf"] = torch.cat([buf, torch.zeros(int(s["last_tb"]), 512, dtype=torch.float32)], dim=0)
        E = buf.shape[0]
    if emit >= E:
        return _emit(torch.zeros(0, 512, dtype=torch.float32), 0, 0, fin=True)
    start = max(0, emit - left_ctx)
    payload = _emit(buf[start:E], emit - start, 0, fin=True)
    s["emit"] = E
    st.pop(request_id, None)
    return payload
