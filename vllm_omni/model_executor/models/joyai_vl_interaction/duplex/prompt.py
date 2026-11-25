# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Visual turn ordering shared by direct JoyAI prompts and ASR handoffs."""

import regex as re

from vllm_omni.errors import OmniClientError
from vllm_omni.experimental.fullduplex.joyvl.decision.prompts import USER_QUERY_HEADER


def validate_working_messages(messages: object, history: list[dict[str, object]], image_count: int) -> None:
    """Validate the serialized visual/history associations before rendering."""
    error = "joyai_prompt_context working_messages must preserve visual turns and history item references"

    def valid_key(value: object) -> bool:
        return type(value) in (str, int)

    if not isinstance(messages, list) or not messages or any(not valid_key(item.get("_item_key")) for item in history):
        raise OmniClientError(error)
    placeholders = 0
    for index, message in enumerate(messages):
        if (
            not isinstance(message, dict)
            or message.get("role") not in ("user", "assistant")
            or (message.get("_query_key") is not None and not valid_key(message["_query_key"]))
            or (message.get("_response_item_id") is not None and not isinstance(message["_response_item_id"], str))
            or not isinstance(message.get("_history_anchors", []), list)
            or any(not valid_key(key) for key in message.get("_history_anchors", []))
            or message.get("_current", False) is not (index == len(messages) - 1)
        ):
            raise OmniClientError(error)
        content = message.get("content")
        if message["role"] == "assistant":
            if not isinstance(content, str) or index == len(messages) - 1:
                raise OmniClientError(error)
            continue
        if not isinstance(content, list):
            raise OmniClientError(error)
        for part in content:
            if not isinstance(part, dict) or part.get("type") not in ("image", "text"):
                raise OmniClientError(error)
            if part["type"] == "image":
                placeholders += 1
            elif not isinstance(part.get("text"), str):
                raise OmniClientError(error)
    if placeholders != image_count:
        raise OmniClientError(error)


def visual_response_refs(messages: list[dict[str, object]]) -> set[str]:
    """Identify replies whose visual source is still present in this prompt."""
    return {
        message["_response_item_id"]
        for message in messages
        if message.get("_response_item_id")
        and isinstance(message["content"], list)
        and any(part.get("type") == "image" for part in message["content"])
    }


def interleave_history(
    history: list[dict[str, object]],
    working_messages: list[dict[str, object]],
    prefix_keys: set,
    *,
    history_query_headers: bool = True,
) -> list[dict[str, object]]:
    """Place committed Q&A at its visual source, resolving item references anew."""
    response_refs = visual_response_refs(working_messages)
    history = [message for message in history if message["content"] or message["_item_key"] in response_refs]
    by_key = {message["_item_key"]: message for message in history}
    positions = {message["_item_key"]: index for index, message in enumerate(history)}
    current_query_key = working_messages[-1].get("_query_key")
    query_has_position = current_query_key in by_key and any(
        current_query_key in message.get("_history_anchors", ()) for message in working_messages[:-1]
    )
    claimed = {
        key
        for message in working_messages
        for key in (message.get("_query_key"), message.get("_response_item_id"))
        if key is not None and not (message.get("_current") and query_has_position and key == current_query_key)
    }
    messages = []
    cursor = 0

    def append_history(end: int) -> None:
        nonlocal cursor
        while cursor < end:
            message = history[cursor]
            if message["_item_key"] not in claimed:
                content = message["content"]
                if message["role"] == "user" and history_query_headers:
                    content = [{"type": "text", "text": f"{USER_QUERY_HEADER}\n{content}"}]
                messages.append({"role": message["role"], "content": content})
            cursor += 1

    for source in working_messages:
        if source.get("_current"):
            append_history(len(history))
        else:
            anchor = next(
                (positions[key] for key in reversed(source.get("_history_anchors", ())) if key in positions), -1
            )
            append_history(anchor + 1)
        content = source["content"]
        query_key = source.get("_query_key")
        if isinstance(content, list) and query_key is not None:
            # Budget-trimmed or deleted historical queries stay removed.
            # The current instruction remains available with its images.
            if (
                query_key in prefix_keys
                or (source.get("_current") and query_has_position)
                or (query_key not in by_key and query_key != current_query_key)
            ):
                content = [
                    part
                    for part in content
                    if part.get("type") != "text" or not part["text"].startswith(USER_QUERY_HEADER)
                ]
        messages.append({"role": source["role"], "content": content})
        response = by_key.get(source.get("_response_item_id"))
        if response is not None and response["role"] == "assistant":
            messages.append({"role": "assistant", "content": response["content"]})
    return messages


def drop_oldest_visual_frame(messages: list[dict[str, object]]) -> None:
    """Trim one frame and its timestamp at an explicit model-context boundary."""
    for index, message in enumerate(messages):
        content = message["content"]
        if message["role"] != "user" or not isinstance(content, list):
            continue
        for part_index, part in enumerate(content):
            if part.get("type") == "image":
                start = part_index
                # Captions may end in a timestamp without being a frame timestamp.
                if part_index and re.fullmatch(r"<-?[0-9]+\.[0-9] seconds>", content[part_index - 1].get("text", "")):
                    start -= 1
                del content[start : part_index + 1]
                if not content:
                    del messages[index]
                    if index < len(messages) and messages[index]["role"] == "assistant":
                        del messages[index]
                return
