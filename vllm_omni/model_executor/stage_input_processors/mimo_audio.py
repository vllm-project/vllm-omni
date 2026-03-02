from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.mimo_audio.config_mimo_audio import TALKER_CODEC_PAD_TOKEN_ID

logger = init_logger(__name__)


def prepend_and_flatten_colmajor(x: torch.Tensor, pad_vec: torch.Tensor) -> torch.Tensor:
    """
    Prepend a padding vector to the input tensor and flatten in column-major order.

    This function expands the padding vector to match the batch dimensions of the input
    tensor, prepends it to the row dimension, and then flattens the result in column-major
    order (transposing before flattening).

    Args:
        x: Input tensor with shape (..., R, C) where R is the row dimension and C is
            the column dimension. Example: (B, 1, 8, 4) where B is batch size.
        pad_vec: Padding vector with shape (C,) to be prepended to x. The vector will
            be expanded to match the batch dimensions of x.

    Returns:
        A flattened 1D tensor in column-major order with shape (-1,). The result
        contains the padded row followed by all rows of x, flattened column by column.
    """
    pad_row = pad_vec.view(1, -1)

    # Expand pad_row to the front of x, keeping other batch dimensions consistent
    # Example: x shape = (B,1,R,C) → pad shape = (B,1,1,C)
    pad_expand = pad_row.view(*([1] * (x.dim() - 2)), 1, x.size(-1)).expand(*x.shape[:-2], 1, x.size(-1))

    # Prepend to the row dimension
    y = torch.cat([pad_expand, x], dim=-2)  # (..., R+1, C)

    # Flatten in column-major order:
    # First transpose (..., R+1, C) → (..., C, R+1)
    # Then flatten
    y_col_major = y.permute(*range(y.dim() - 2), -1, -2).reshape(-1)

    return y_col_major


def _make_finished_sentinel() -> dict[str, Any]:
    """Return a minimal payload with finished=True so Stage-1 can end the request."""
    return {"code_predictor_codes": [], "finished": torch.tensor(True, dtype=torch.bool)}


def llm2code2wav_async_chunk(
    transfer_manager: Any, pooling_output: dict[str, Any], request: Any
) -> dict[str, Any] | None:
    """
    Async chunk version: convert stage-0 pooling_output to code2wav payload (pooling / connector accumulation).

    Accumulates codes in connector per request_id,
    returns payload only when chunk_size is full or request is finished; returns None when waiting.
    """
    if "code_predictor_codes" not in pooling_output:
        if request.is_finished():
            return _make_finished_sentinel()
        return None

    code_predictor_codes = pooling_output["code_predictor_codes"]

    if code_predictor_codes is None:
        if request.is_finished():
            return _make_finished_sentinel()
        return None
    if isinstance(code_predictor_codes, torch.Tensor):
        if code_predictor_codes.numel() == 0:
            if request.is_finished():
                return _make_finished_sentinel()
            return None
    elif hasattr(code_predictor_codes, "__len__"):
        if len(code_predictor_codes) == 0:
            return None

    if isinstance(code_predictor_codes, torch.Tensor):
        if not code_predictor_codes.any():
            if request.is_finished():
                return _make_finished_sentinel()
            return None
        code_tensor = code_predictor_codes.to(torch.long)
    else:
        code_tensor = torch.tensor(code_predictor_codes, dtype=torch.long)
        if not code_tensor.any():
            if request.is_finished():
                return _make_finished_sentinel()
            return None

    if code_tensor.ndim == 3:
        code_tensor = code_tensor.unsqueeze(0)
    if code_tensor.ndim != 4 or code_tensor.shape[-2:] != (8, 4):
        return None

    pad_vec = torch.tensor([TALKER_CODEC_PAD_TOKEN_ID] * 4, device=code_tensor.device, dtype=code_tensor.dtype)
    code_final = prepend_and_flatten_colmajor(code_tensor, pad_vec)
    code_list = code_final.tolist()
    if sum(code_list) == 0:
        if request.is_finished():
            return _make_finished_sentinel()
        return None

    request_id = getattr(request, "external_req_id", None)
    if request_id is None:
        return None

    chunk_size = left_context_size = 10
    transfer_manager.code_prompt_token_ids[request_id].append(code_list)
    length = len(transfer_manager.code_prompt_token_ids[request_id])
    chunk_length = length % chunk_size
    if chunk_length != 0 and not request.is_finished():
        return None

    context_length = chunk_length if chunk_length != 0 else chunk_size
    end_index = min(length, left_context_size + context_length)

    info = {
        "code_predictor_codes": (
            torch.tensor(transfer_manager.code_prompt_token_ids[request_id][-end_index:]).reshape(-1).tolist()
        ),
        "finished": torch.tensor(request.is_finished(), dtype=torch.bool),
    }
    return info


def llm2code2wav(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """
    Process talker outputs to create code2wav inputs.

    Workflow:
    1. Extract talker's codec code outputs (8-layer RVQ codes)
    2. Flatten codes for code2wav input
    3. Package for code2wav stage

    Args:
        stage_list: List of stage objects
        engine_input_source: Source stage IDs (typically [1] for talker)
        prompt: Original prompt data
        requires_multimodal_data: Whether multimodal data is required

    Returns:
        List of OmniTokensPrompt for code2wav stage
    """
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")

    if stage_list[source_stage_id].engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    talker_outputs = stage_list[source_stage_id].engine_outputs
    code2wav_inputs = []

    # Process each talker output
    for i, talker_output in enumerate(talker_outputs):
        output = talker_output.outputs[0]

        # Extract codec codes from talker output
        # Expected shape: [8, seq_len] (8-layer RVQ codes)
        if "code_predictor_codes" in output.multimodal_output:
            codec_codes = output.multimodal_output["code_predictor_codes"].to(torch.long)  # [seq_batch_size, 1, 8, 4]
            is_all_zero = (codec_codes == 0).all(dim=(1, 2, 3))
            non_zero_indices = (~is_all_zero).nonzero(as_tuple=True)[0]
            if len(non_zero_indices) == 0:
                # All codec codes are zero - skip this request with a warning
                request_id = getattr(talker_output, "request_id", f"unknown_{i}")
                logger.warning(
                    "Skipping request %s: all codec codes are zero (empty output from Stage-0). "
                    "This may indicate the model failed to generate valid audio codes.",
                    request_id,
                )
            else:
                if len(non_zero_indices) < codec_codes.shape[0]:
                    codec_codes = codec_codes[non_zero_indices]
        elif "latent" in output.multimodal_output and "code_predictor_codes" not in output.multimodal_output:
            codec_codes = torch.zeros(1, 1, 8, 4, dtype=torch.long)
        else:
            raise ValueError(f"Invalid multimodal_output: {output.multimodal_output}")

        pad_vec = torch.tensor([TALKER_CODEC_PAD_TOKEN_ID] * 4)

        code_final = prepend_and_flatten_colmajor(codec_codes, pad_vec)
        code_final = code_final.tolist()

        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=code_final,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return code2wav_inputs
