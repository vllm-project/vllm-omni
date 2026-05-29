from __future__ import annotations

import os
from typing import Any

import torch

BAGEL_VLM_THINK_SYSTEM_PROMPT = (
    "You should first think about the reasoning process in the mind and then "
    "provide the user with the answer.\n"
    "The reasoning process is enclosed within <think> </think> tags, i.e. "
    "<think> reasoning process here </think> answer here"
)

TRUE_VALUES = {"1", "true", "yes", "on"}


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in TRUE_VALUES


def bagel_vqa_reference_prefill_enabled() -> bool:
    return env_flag("BAGEL_VQA_REFERENCE_PREFILL")


def bagel_vqa_reference_layout_enabled() -> bool:
    return (
        bagel_vqa_reference_prefill_enabled()
        or env_flag("BAGEL_VQA_REFERENCE_LAYOUT")
    )


def bagel_token_id(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id < 0:
        raise ValueError(f"BAGEL tokenizer is missing required token {token!r}")
    return int(token_id)


def bagel_encode_text_segment(tokenizer: Any, text: str) -> list[int]:
    im_start = bagel_token_id(tokenizer, "<|im_start|>")
    im_end = bagel_token_id(tokenizer, "<|im_end|>")
    return [im_start] + list(tokenizer.encode(text)) + [im_end]


def strip_bagel_think_prompt(text: str) -> str:
    if text.startswith(BAGEL_VLM_THINK_SYSTEM_PROMPT):
        return text[len(BAGEL_VLM_THINK_SYSTEM_PROMPT) :].lstrip()
    return text


def messages_to_dicts(messages: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        if hasattr(msg, "model_dump"):
            out.append(msg.model_dump())
        elif isinstance(msg, dict):
            out.append(msg)
        else:
            out.append(
                {
                    "role": getattr(msg, "role", "user"),
                    "content": getattr(msg, "content", ""),
                }
            )
    return out


def build_bagel_reference_text_and_image_count(
    messages: list[Any],
) -> tuple[str, int]:
    text_parts: list[str] = []
    image_count = 0

    for message in messages_to_dicts(messages):
        role = message.get("role")
        if role is not None and role != "user":
            continue

        content = message.get("content", "")
        if isinstance(content, str):
            stripped = strip_bagel_think_prompt(content)
            if stripped:
                text_parts.append(stripped)
            continue

        if not isinstance(content, list):
            continue

        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "text":
                text = part.get("text") or ""
                stripped = strip_bagel_think_prompt(text)
                if stripped:
                    text_parts.append(stripped)
            elif part_type in {"image", "image_url"}:
                image_count += 1
                text_parts.append(f"<img><|image_{image_count}|></img>")

    return " ".join(text_parts), image_count


def build_bagel_vqa_reference_prompt_token_ids(
    messages: list[Any],
    tokenizer: Any,
) -> tuple[list[int], str, int]:
    final_text, image_count = build_bagel_reference_text_and_image_count(messages)
    if image_count <= 0:
        return [], final_text, image_count

    image_pad = bagel_token_id(tokenizer, "<|image_pad|>")
    im_start = bagel_token_id(tokenizer, "<|im_start|>")

    prompt_token_ids: list[int] = []
    prompt_token_ids.extend(
        bagel_encode_text_segment(tokenizer, BAGEL_VLM_THINK_SYSTEM_PROMPT)
    )
    prompt_token_ids.extend([image_pad] * image_count)
    if final_text:
        prompt_token_ids.extend(bagel_encode_text_segment(tokenizer, final_text))
    prompt_token_ids.append(im_start)
    return prompt_token_ids, final_text, image_count


def build_bagel_vqa_rope_positions(
    *,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor | None,
    req_ids: list[str],
    num_computed_tokens: list[int],
    num_scheduled_tokens: list[int],
    rope_states: dict[str, dict[str, Any]],
    start_of_image_id: int,
    end_of_image_id: int,
) -> torch.Tensor | None:
    """Compute BAGEL logical RoPE positions without touching vLLM slots."""
    if input_ids is None or positions is None or not req_ids:
        return None
    if positions.ndim != 1:
        return None

    total_scheduled = sum(int(n) for n in num_scheduled_tokens)
    if total_scheduled <= 0:
        return None

    active_req_ids = set(req_ids)
    for rid in list(rope_states):
        if rid not in active_req_ids:
            rope_states.pop(rid, None)

    rope_positions = positions.clone()
    any_changed = False
    offset = 0

    for req_idx, req_id in enumerate(req_ids):
        sched = int(num_scheduled_tokens[req_idx])
        if sched <= 0:
            continue

        start = offset
        end = offset + sched
        offset = end

        token_slice = input_ids[start:end]
        if token_slice.numel() == 0:
            continue

        state = rope_states.get(req_id)
        has_vision_marker = bool(
            ((token_slice == start_of_image_id) | (token_slice == end_of_image_id))
            .any()
            .item()
        )
        if state is None and not has_vision_marker:
            continue

        if state is None:
            state = {
                "enabled": True,
                "next": int(num_computed_tokens[req_idx]),
                "in_vision": False,
                "block": None,
            }
        else:
            state["enabled"] = True

        logical = []
        next_pos = int(state.get("next", num_computed_tokens[req_idx]))
        in_vision = bool(state.get("in_vision", False))
        block_pos = state.get("block")
        if block_pos is None:
            block_pos = next_pos
        block_pos = int(block_pos)

        for tok in token_slice.tolist():
            tok = int(tok)
            if tok == start_of_image_id and not in_vision:
                in_vision = True
                block_pos = next_pos
                logical.append(block_pos)
            elif in_vision:
                logical.append(block_pos)
                if tok == end_of_image_id:
                    in_vision = False
                    next_pos = block_pos + 1
            else:
                logical.append(next_pos)
                next_pos += 1

        new_vals = torch.tensor(logical, device=positions.device, dtype=positions.dtype)
        rope_positions[start:end] = new_vals
        if not torch.equal(new_vals, positions[start:end]):
            any_changed = True

        state["next"] = next_pos
        state["in_vision"] = in_vision
        state["block"] = block_pos if in_vision else None
        rope_states[req_id] = state

    return rope_positions if any_changed else None


def build_bagel_vqa_image_spans(
    *,
    input_ids: torch.Tensor | None,
    req_ids: list[str],
    num_computed_tokens: list[int],
    num_scheduled_tokens: list[int],
    start_of_image_id: int,
    end_of_image_id: int,
) -> list[dict[str, int]]:
    """Find complete BAGEL VQA image blocks in the scheduled token batch.

    Returned offsets are flat query rows for the current vLLM batch. `kv_end`
    is the per-request sequence length immediately after the image block, which
    prevents the non-causal image recompute from attending to future text.
    """
    if input_ids is None or not req_ids:
        return []
    if input_ids.ndim != 1:
        return []

    spans: list[dict[str, int]] = []
    offset = 0
    for req_idx, _req_id in enumerate(req_ids):
        sched = int(num_scheduled_tokens[req_idx])
        if sched <= 0:
            continue

        start = offset
        end = offset + sched
        offset = end

        token_slice = input_ids[start:end]
        block_start: int | None = None
        for local_idx, tok in enumerate(token_slice.tolist()):
            tok = int(tok)
            if tok == start_of_image_id:
                block_start = local_idx
            elif tok == end_of_image_id and block_start is not None:
                block_end = local_idx + 1
                spans.append(
                    {
                        "req_idx": req_idx,
                        "request_start": start,
                        "num_computed_tokens": int(num_computed_tokens[req_idx]),
                        "q_start": start + block_start,
                        "q_end": start + block_end,
                        "kv_local_end": block_end,
                        "kv_end": int(num_computed_tokens[req_idx]) + block_end,
                    }
                )
                block_start = None

    return spans
