# Copyright 2025 vLLM-Omni Team
"""Kimi Audio-specific request preprocessor.

Handles the dual-path data flow for Kimi Audio's upstream-aligned architecture:
1. Convert OpenAI-style ``input_audio`` items into ``audio_url`` data URLs for
   the chat template, so it generates audio markers
   (<|im_media_begin|><|im_kimia_text_blank|><|im_media_end|>).
2. Pass raw audio bytes through unchanged for Whisper feature extraction.
"""

from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


class KimiAudioRequestPreprocessor:
    """Request preprocessor for Kimi Audio models.

    Kimi Audio uses an upstream-aligned architecture (Whisper-Large-v3 for audio
    comprehension) that requires special handling at the request preprocessing stage.

    The chat template needs to see audio URLs in messages to generate audio markers,
    which create BLANK tokens that get replaced with Whisper features.  OpenAI-style
    ``input_audio`` items are therefore converted to inline ``audio_url`` data URLs
    before the template is applied.  Meanwhile, the raw audio bytes must be
    extracted and passed to the multimodal processor for Whisper feature extraction.
    """

    @staticmethod
    def prepare_messages(
        messages: list[dict[str, Any]],
        deferred_multi_modal_data: dict[str, Any] | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        """Prepare messages for Kimi Audio chat template.

        The Kimi Audio chat template only recognises ``audio_url`` items when
        generating audio markers.  OpenAI-style ``input_audio`` items are
        converted to inline ``audio_url`` data URLs so the template produces the
        placeholder BLANK tokens that get replaced with Whisper features.

        Args:
            messages: Request messages (may contain ``input_audio`` items).
            deferred_multi_modal_data: Raw audio bytes extracted by the generic
                preprocessor; passed through unchanged.

        Returns:
            Tuple of (messages_for_template, deferred_data):
            - messages_for_template: Messages with ``input_audio`` converted to
              ``audio_url`` data URLs.
            - deferred_data: Raw audio bytes for Whisper processing.
        """
        converted_messages = []
        for message in messages:
            new_message = dict(message)
            content = new_message.get("content", [])
            if isinstance(content, list):
                new_content = []
                for item in content:
                    if (
                        isinstance(item, dict)
                        and item.get("type") == "input_audio"
                        and isinstance(item.get("input_audio"), dict)
                    ):
                        input_audio = item["input_audio"]
                        audio_format = input_audio.get("format", "wav")
                        data = input_audio.get("data", "")
                        new_item = {
                            "type": "audio_url",
                            "audio_url": {"url": f"data:audio/{audio_format};base64,{data}"},
                        }
                        new_content.append(new_item)
                    else:
                        new_content.append(item)
                new_message["content"] = new_content
            converted_messages.append(new_message)

        return converted_messages, deferred_multi_modal_data
