"""Stage input processors for VibeVoice TTS.

Bridges Stage 0 (audio generation) output — continuous acoustic latents —
to Stage 1 (audio tokenizer decoder) input.

Unlike Voxtral which streams discrete integer codes, VibeVoice streams
continuous float tensors (shape [latent_dim] per frame).
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniTokensPrompt

logger = init_logger(__name__)


def _extract_last_frame(pooling_output: dict[str, Any]) -> torch.Tensor | None:
    """Extract the latest acoustic latent frame from stage output."""
    audio = pooling_output.get("audio")
    if not isinstance(audio, torch.Tensor) or audio.numel() == 0:
        return None
    return audio.flatten()


def generator2tokenizer(
    stage_list,
    engine_input_source,
    prompt: OmniTokensPrompt | TextPrompt = None,
    requires_multimodal_data: bool = False,
):
    """Non-streaming: collect all latent frames and pass to Stage 1."""
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")
    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")
    if stage_list[source_stage_id].engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    generator_outputs = stage_list[source_stage_id].engine_outputs
    tokenizer_inputs = []
    for generator_output in generator_outputs:
        output = generator_output.outputs[0]
        # Each frame in multimodal_output["audio"] is (1, 1, latent_dim)
        frames = output.multimodal_output["audio"]
        flat_latents = torch.cat(frames, dim=1).squeeze(0).flatten()
        # Pack as: [0 (ctx_frames), num_frames (context_length), flat_values...]
        num_frames = len(frames)
        header = torch.tensor([0.0, float(num_frames)], dtype=torch.float32)
        token_ids = torch.cat([header, flat_latents.float().cpu()])
        tokenizer_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=token_ids.tolist(),
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return tokenizer_inputs


def generator2tokenizer_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: Any,
    is_finished: bool = False,
) -> dict[str, Any] | None:
    """Streaming: accumulate latent frames and emit chunks with context.

    Each acoustic latent frame is a 1-D float tensor of shape (latent_dim,).
    Frames are accumulated in transfer_manager.code_prompt_token_ids[request_id]
    as lists of floats.
    """
    request_id = request.external_req_id
    finished = bool(is_finished or request.is_finished())

    if isinstance(pooling_output, dict):
        frame = _extract_last_frame(pooling_output)
        if frame is not None:
            latent_values = frame.cpu().tolist()
            transfer_manager.code_prompt_token_ids[request_id].append(latent_values)
    elif not finished:
        return None

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", 25))
    chunk_size_at_begin = int(cfg.get("codec_chunk_frames_at_begin", 5))
    left_context_size = int(cfg.get("codec_left_context_frames", 25))
    if chunk_size <= 0 or left_context_size < 0:
        raise ValueError(
            f"Invalid codec chunk config: codec_chunk_frames={chunk_size}, "
            f"codec_left_context_frames={left_context_size}"
        )
    length = len(transfer_manager.code_prompt_token_ids[request_id])

    if length <= 0:
        if finished:
            return {
                "code_predictor_codes": [],
                "finished": torch.tensor(True, dtype=torch.bool),
            }
        return None

    if length <= chunk_size:
        chunk_size = chunk_size_at_begin

    chunk_length = length % chunk_size
    if chunk_length != 0 and not finished:
        return None

    context_length = chunk_length if chunk_length != 0 else chunk_size
    end_index = min(length, left_context_size + context_length)
    ctx_frames = max(0, int(end_index - context_length))
    window_frames = transfer_manager.code_prompt_token_ids[request_id][-end_index:]

    # Pack: [ctx_frames, context_length, flat_latent_values...]
    flat_values: list[float] = []
    for frame_vals in window_frames:
        flat_values.extend(frame_vals)

    code_predictor_codes = [float(ctx_frames), float(context_length)] + flat_values

    return {
        "code_predictor_codes": code_predictor_codes,
        "finished": torch.tensor(finished, dtype=torch.bool),
    }
