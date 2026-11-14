# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Step-local Thinker-to-Talker spans, independent of runner bookkeeping."""

from collections.abc import Callable, Sequence

import torch

STAGING_KEY = "_thinker_speculative_payload"


def prepare_speculative_payload(
    *,
    multimodal_outputs: dict,
    req_ids: Sequence[str],
    speculative_request_ids: set[str],
    invalid_req_indices: Sequence[int],
    sampled_token_ids: torch.Tensor,
    input_token_ids: torch.Tensor,
    input_positions: torch.Tensor,
    query_start_loc: Sequence[int],
    prompt_lens: Sequence[int],
    vocab_size: int,
    consumed_input_span: bool,
    embed_input_ids: Callable[[torch.Tensor], torch.Tensor],
) -> dict:
    """Prepare GPU tensors before drafting can overwrite target inputs.

    Deferred chunks reuse consumed verifier rows (previous tail + accepted
    drafts). Inline output instead embeds the committed samples in one batch,
    including zero-draft decode steps and the first prefill sample exactly
    once, by its input position. Prompt rows retain their ordinary payload.
    No request history is consulted: async bookkeeping may contain only -1s.
    """
    invalid = set(invalid_req_indices)
    spans = [
        (idx, int(query_start_loc[idx]), int(query_start_loc[idx + 1]))
        for idx, rid in enumerate(req_ids)
        if idx not in invalid and (not consumed_input_span or rid in speculative_request_ids)
    ]
    if not spans:
        return multimodal_outputs

    indices, offsets, token_parts, embedding_parts, mask_parts = [], [0], [], [], []
    decode_parts = []
    layers = multimodal_outputs.get("hidden_states", {}).get("layers", {})
    if consumed_input_span:
        # One reduction and padded mask for the batch. Slice views below use
        # CPU-known query widths; no count readback or per-request kernels.
        valid = (sampled_token_ids >= 0) & (sampled_token_ids < vocab_size)
        counts = valid.sum(dim=1)
        max_width = max(end - start for _, start, end in spans)
        row_mask = torch.arange(max_width, device=input_token_ids.device)[None, :] < counts[:, None]

    for idx, start, end in spans:
        if consumed_input_span:
            ids = input_token_ids[start:end]
            mask_parts.append(row_mask[idx, : end - start])
            embedding_parts.append(layers[0][start:end])
        else:
            samples = sampled_token_ids[idx]
            valid = (samples >= 0) & (samples < vocab_size)
            # Linear target positions are corrected on GPU under async
            # scheduling. MRoPE coordinates and optimistic CPU counts cannot
            # identify the first generated input position reliably.
            is_decode = input_positions[start] >= prompt_lens[idx]
            decode_parts.append(is_decode)
            valid = valid & is_decode
            seed = (input_positions[start] == prompt_lens[idx]) & valid.any()
            first = torch.where(seed, input_token_ids[start], -1).reshape(1)
            ids = torch.cat((first, torch.where(valid, samples, -1)))
        indices.append(idx)
        token_parts.append(ids)
        offsets.append(offsets[-1] + ids.numel())

    ids = torch.cat(token_parts)
    if consumed_input_span:
        ids = torch.where(torch.cat(mask_parts), ids, -1)
        embeddings = torch.cat(embedding_parts)
    else:
        # Fixed-size padded lookup avoids a GPU->CPU count synchronization.
        # All TP ranks submit exactly one embedding collective per step.
        # Prefill padding is intentionally included: is_decode depends on
        # GPU-corrected positions; filtering it here would require a host sync.
        safe_ids = torch.where((ids >= 0) & (ids < vocab_size), ids, 0)
        embeddings = embed_input_ids(safe_ids).reshape(ids.numel(), -1)
    return {
        **multimodal_outputs,
        STAGING_KEY: {
            "indices": tuple(indices),
            "offsets": tuple(offsets),
            "ids": ids,
            "embeddings": embeddings,
            "vocab_size": vocab_size,
            "consumed_input_span": consumed_input_span,
            "decode_mask": torch.stack(decode_parts) if decode_parts else None,
        },
    }


def finalize_speculative_payload(multimodal_outputs: dict | None) -> dict | None:
    """Filter one completed CPU snapshot; never read live runner/model state."""
    if not multimodal_outputs or STAGING_KEY not in multimodal_outputs:
        return multimodal_outputs
    stage = multimodal_outputs[STAGING_KEY]
    # Inline fallback reaches this hook with GPU tensors; deferred output has
    # already copied both tensors using the ordinary Omni snapshot/event.
    ids, embeddings = stage["ids"].cpu(), stage["embeddings"].cpu()
    decode_mask = stage["decode_mask"]
    decode_rows = decode_mask.cpu().tolist() if decode_mask is not None else [True] * len(stage["indices"])
    payloads: list[dict | None] = [None] * (max(stage["indices"]) + 1)
    for idx, start, end, is_decode in zip(stage["indices"], stage["offsets"], stage["offsets"][1:], decode_rows):
        if not is_decode:
            continue
        tokens = ids[start:end]
        valid = (tokens >= 0) & (tokens < stage["vocab_size"])
        tokens = tokens[valid].tolist()
        meta = {"decode_span": True, "committed_token_count": len(tokens)}
        if stage["consumed_input_span"]:
            meta["consumed_input_span"] = True
        payloads[idx] = {
            "hidden_states.layer_0": embeddings[start:end][valid].contiguous(),
            "meta": meta,
            "ids": {"output": tokens},
        }
    return {**multimodal_outputs, STAGING_KEY: payloads}


def apply_speculative_payload(payload: dict, multimodal_outputs: dict | None, idx: int) -> dict:
    """Replace verifier query rows with the model's aligned speech span."""
    payload.pop(STAGING_KEY, None)
    spans = (multimodal_outputs or {}).get(STAGING_KEY, ())
    prepared = spans[idx] if idx < len(spans) else None
    if prepared is None:
        return payload
    payload = {key: value for key, value in payload.items() if key != "hidden" and not key.startswith("hidden_states.")}
    payload.update(prepared)
    return payload
