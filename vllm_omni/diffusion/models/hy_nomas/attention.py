# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from Tencent-Hunyuan/Hy-Nomas (Copyright 2026 Tencent).


def action_ranges(history_mask, text_mask, image_tokens, offsets, windows, tokens_per_frame):
    result = []
    history_length = history_mask.shape[1]
    for batch in range(text_mask.shape[0]):
        visible_history = history_mask[batch].nonzero().flatten().tolist()
        visible_text = text_mask[batch].nonzero().flatten().tolist()
        positions_q = list(range(image_tokens)) + [image_tokens + index for index in visible_text]
        positions_k = visible_history + [history_length + index for index in positions_q]
        off = offsets[batch] if batch < len(offsets) else None
        if off is None:
            result.append(
                [(0, image_tokens + text_mask.shape[1], 0, history_length + image_tokens + text_mask.shape[1])]
            )
            continue
        off = [min(int(index), len(visible_text)) for index in off]
        rectangles = []

        def emit(start, end, own_action):
            if end <= start:
                return
            key_ranges = [(0, len(visible_history)), (len(visible_history), len(visible_history) + image_tokens)]
            text_start = len(visible_history) + image_tokens
            key_ranges.append((text_start, text_start + off[0]))
            if own_action is not None and 0 <= own_action < len(off) - 1:
                key_ranges.append((text_start + off[own_action], text_start + off[own_action + 1]))
            for left, right in key_ranges:
                if right > left:
                    rectangles.append(
                        (positions_q[start], positions_q[end - 1] + 1, positions_k[left], positions_k[right - 1] + 1)
                    )

        for start, end, action in windows:
            emit(
                max(0, min(int(start) * tokens_per_frame, image_tokens)),
                max(0, min(int(end) * tokens_per_frame, image_tokens)),
                int(action),
            )
        emit(image_tokens, image_tokens + off[0], None)
        for index in range(len(off) - 1):
            emit(image_tokens + off[index], image_tokens + off[index + 1], index)
        result.append(rectangles)
    return result
