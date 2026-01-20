from collections.abc import Awaitable, Iterable
from typing import Any, cast

import numpy as np
from openai.types.chat import ChatCompletionContentPartTextParam
from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import (
    AsyncMultiModalContentParser,
    AsyncMultiModalItemTracker,
    BaseMultiModalContentParser,
    BaseMultiModalItemTracker,
    ChatCompletionContentPartParam,
    ChatCompletionMessageParam,
    ConversationMessage,
    MultiModalDataDict,
    MultiModalUUIDDict,
    _AssistantParser,
    _ChatTemplateContentFormat,
    _ContentPart,
    _get_full_multimodal_text_prompt,
    _parse_chat_message_content_part,
    _postprocess_messages,
    _ToolParser,
)


class OmniAsyncMultiModalItemTracker(AsyncMultiModalItemTracker):
    def create_parser(self) -> "BaseMultiModalContentParser":
        return OmniAsyncMultiModalContentParser(self)


class OmniAsyncMultiModalContentParser(AsyncMultiModalContentParser):
    def __init__(self, tracker: AsyncMultiModalItemTracker) -> None:
        super().__init__(tracker=tracker)
        self._mm_processor_kwargs: dict[str, Any] | None = None

    def set_mm_processor_kwargs(self, mm_processor_kwargs: dict[str, Any] | None) -> None:
        """Set mm_processor_kwargs for use in parsing."""
        self._mm_processor_kwargs = mm_processor_kwargs

    def parse_video(self, video_url: str | None, uuid: str | None = None) -> None:
        video = self._connector.fetch_video_async(video_url=video_url) if video_url else None

        placeholder = self._tracker.add("video", video, uuid)
        self._add_placeholder("video", placeholder)

        # Extract audio from video if use_audio_in_video is True
        if video_url and self._mm_processor_kwargs and self._mm_processor_kwargs.get("use_audio_in_video", False):
            audio_coro = self._extract_audio_from_video_async(video_url)
            audio_placeholder = self._tracker.add("audio", audio_coro, uuid)
            self._add_placeholder("audio", audio_placeholder)

    async def _extract_audio_from_video_async(self, video_url: str) -> tuple[np.ndarray, int | float]:
        """
        Extract audio from video URL using librosa.
        Returns tuple of (audio_array, sample_rate) compatible with audio format.

        All blocking I/O operations are run in a thread pool to avoid blocking the event loop.
        """
        import asyncio
        import os
        import tempfile
        from urllib.parse import urlparse

        # Parse URL to determine type
        parsed_url = urlparse(video_url)
        temp_video_file_path = None

        def _download_video_sync(url: str) -> bytes:
            """Synchronous video download - runs in thread pool."""
            from urllib.request import urlopen

            return urlopen(url).read()

        def _write_temp_file_sync(data: bytes, suffix: str) -> str:
            """Synchronous temp file write - runs in thread pool."""
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(data)
                return temp_file.name

        def _load_audio_sync(file_path: str) -> tuple[np.ndarray, int | float]:
            """Synchronous audio loading with librosa - runs in thread pool."""
            import librosa

            return librosa.load(file_path, sr=16000)

        def _cleanup_file_sync(file_path: str) -> None:
            """Synchronous file deletion - runs in thread pool."""
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except OSError:
                pass

        try:
            if parsed_url.scheme in ("http", "https"):
                # Download video from HTTP/HTTPS URL asynchronously
                video_data = await asyncio.to_thread(_download_video_sync, video_url)
                # Write temp file asynchronously
                temp_video_file_path = await asyncio.to_thread(_write_temp_file_sync, video_data, ".mp4")
            elif parsed_url.scheme == "file":
                # Use file path directly (handle Windows paths)
                from urllib.request import url2pathname

                temp_video_file_path = url2pathname(parsed_url.path)
            elif parsed_url.scheme == "data":
                # Handle data URL (base64 encoded video)
                import base64

                header, data = video_url.split(",", 1)
                video_data = base64.b64decode(data)
                # Write temp file asynchronously
                temp_video_file_path = await asyncio.to_thread(_write_temp_file_sync, video_data, ".mp4")
            else:
                # Assume it's a local file path
                temp_video_file_path = video_url

            # Extract audio using librosa asynchronously (CPU-intensive, runs in thread pool)
            audio_array, sample_rate = await asyncio.to_thread(_load_audio_sync, temp_video_file_path)

            return audio_array, sample_rate
        finally:
            # Clean up temporary file if we created one (asynchronously)
            if temp_video_file_path and parsed_url.scheme in ("http", "https", "data"):
                await asyncio.to_thread(_cleanup_file_sync, temp_video_file_path)


def parse_chat_messages_futures(
    messages: list[ChatCompletionMessageParam],
    model_config: ModelConfig,
    content_format: _ChatTemplateContentFormat,
    mm_processor_kwargs: dict[str, Any] | None = None,
) -> tuple[
    list[ConversationMessage],
    Awaitable[MultiModalDataDict | None],
    MultiModalUUIDDict | None,
]:
    conversation: list[ConversationMessage] = []
    mm_tracker = OmniAsyncMultiModalItemTracker(model_config)

    for msg in messages:
        sub_messages = _parse_chat_message_content(
            msg,
            mm_tracker,
            content_format,
            interleave_strings=(
                content_format == "string"
                and model_config.multimodal_config is not None
                and model_config.multimodal_config.interleave_mm_strings
            ),
            mm_processor_kwargs=mm_processor_kwargs,
        )

        conversation.extend(sub_messages)

    _postprocess_messages(conversation)

    return conversation, mm_tracker.all_mm_data(), mm_tracker.all_mm_uuids()


def _parse_chat_message_content(
    message: ChatCompletionMessageParam,
    mm_tracker: BaseMultiModalItemTracker,
    content_format: _ChatTemplateContentFormat,
    interleave_strings: bool,
    mm_processor_kwargs: dict[str, Any] | None = None,
) -> list[ConversationMessage]:
    role = message["role"]
    content = message.get("content")

    if content is None:
        content = []
    elif isinstance(content, str):
        content = [ChatCompletionContentPartTextParam(type="text", text=content)]
    result = _parse_chat_message_content_parts(
        role,
        content,  # type: ignore
        mm_tracker,
        wrap_dicts=(content_format == "openai"),
        interleave_strings=interleave_strings,
        mm_processor_kwargs=mm_processor_kwargs,
    )

    for result_msg in result:
        if role == "assistant":
            parsed_msg = _AssistantParser(message)

            # The 'tool_calls' is not None check ensures compatibility.
            # It's needed only if downstream code doesn't strictly
            # follow the OpenAI spec.
            if "tool_calls" in parsed_msg and parsed_msg["tool_calls"] is not None:
                result_msg["tool_calls"] = list(parsed_msg["tool_calls"])
        elif role == "tool":
            parsed_msg = _ToolParser(message)
            if "tool_call_id" in parsed_msg:
                result_msg["tool_call_id"] = parsed_msg["tool_call_id"]

        if "name" in message and isinstance(message["name"], str):
            result_msg["name"] = message["name"]

    return result


def _parse_chat_message_content_parts(
    role: str,
    parts: Iterable[ChatCompletionContentPartParam],
    mm_tracker: BaseMultiModalItemTracker,
    *,
    wrap_dicts: bool,
    interleave_strings: bool,
    mm_processor_kwargs: dict[str, Any] | None = None,
) -> list[ConversationMessage]:
    content = list[_ContentPart]()

    mm_parser = mm_tracker.create_parser()
    # Set mm_processor_kwargs if parser supports it
    if hasattr(mm_parser, "set_mm_processor_kwargs"):
        mm_parser.set_mm_processor_kwargs(mm_processor_kwargs)

    for part in parts:
        parse_res = _parse_chat_message_content_part(
            part,
            mm_parser,
            wrap_dicts=wrap_dicts,
            interleave_strings=interleave_strings,
        )
        if parse_res:
            content.append(parse_res)

    if wrap_dicts:
        # Parsing wraps images and texts as interleaved dictionaries
        return [ConversationMessage(role=role, content=content)]  # type: ignore
    texts = cast(list[str], content)
    mm_placeholder_storage = mm_parser.mm_placeholder_storage()
    if mm_placeholder_storage:
        text_prompt = _get_full_multimodal_text_prompt(mm_placeholder_storage, texts, interleave_strings)
    else:
        text_prompt = "\n".join(texts)

    return [ConversationMessage(role=role, content=text_prompt)]
