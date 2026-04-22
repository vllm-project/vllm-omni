"""OpenAI-compatible multi-modal API."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


class ContentType(Enum):
    """Content types for multi-modal input."""

    TEXT = "text"
    IMAGE_URL = "image_url"
    IMAGE_BASE64 = "image_base64"
    AUDIO_URL = "audio_url"
    AUDIO_BASE64 = "audio_base64"
    VIDEO_URL = "video_url"
    VIDEO_BASE64 = "video_base64"


@dataclass
class MultiModalContent:
    """Multi-modal content item."""

    content_type: ContentType
    text: str | None = None
    url: str | None = None
    base64_data: str | None = None
    detail: str = "auto"


@dataclass
class ChatMessage:
    """Chat message."""

    role: str
    content: str | list[MultiModalContent]


@dataclass
class ChatCompletionRequest:
    """Chat completion request."""

    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int | None = None
    stream: bool = False
    stop: str | list[str] | None = None


class MultiModalAPIServer:
    """Extended OpenAI-compatible API server."""

    def __init__(self, omni_engine=None):
        self._engine = omni_engine

    def parse_multimodal_content(self, messages: list[ChatMessage]) -> dict[str, Any]:
        """Parse multi-modal content from messages."""
        multi_modal_data: dict[str, Any] = {}

        for message in messages:
            if isinstance(message.content, list):
                for content in message.content:
                    if content.content_type == ContentType.IMAGE_URL:
                        multi_modal_data["image"] = content.url
                    elif content.content_type == ContentType.IMAGE_BASE64:
                        multi_modal_data["image"] = content.base64_data
                    elif content.content_type == ContentType.AUDIO_URL:
                        multi_modal_data["audio"] = content.url
                    elif content.content_type == ContentType.AUDIO_BASE64:
                        multi_modal_data["audio"] = content.base64_data
                    elif content.content_type == ContentType.VIDEO_URL:
                        multi_modal_data["video"] = content.url
                    elif content.content_type == ContentType.VIDEO_BASE64:
                        multi_modal_data["video"] = content.base64_data

        return multi_modal_data

    def extract_text_prompt(self, messages: list[ChatMessage]) -> str:
        """Extract text prompt from messages."""
        prompts = []

        for message in messages:
            if isinstance(message.content, str):
                prompts.append(message.content)
            elif isinstance(message.content, list):
                for content in message.content:
                    if content.text:
                        prompts.append(content.text)

        return "\n".join(prompts)

    def build_sampling_params(self, request: ChatCompletionRequest) -> dict[str, Any]:
        """Build sampling parameters from request."""
        params = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_tokens": request.max_tokens,
        }

        if request.stop:
            params["stop"] = [request.stop] if isinstance(request.stop, str) else request.stop

        return params

    def format_completion(self, outputs: Any) -> dict[str, Any]:
        """Format completion response."""
        return {
            "id": f"chatcmpl-{id(outputs)}",
            "object": "chat.completion",
            "created": 0,
            "model": outputs.model_name if hasattr(outputs, "model_name") else "unknown",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": outputs.text if hasattr(outputs, "text") else str(outputs),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": getattr(outputs, "prompt_tokens", 0),
                "completion_tokens": getattr(outputs, "completion_tokens", 0),
                "total_tokens": getattr(outputs, "total_tokens", 0),
            },
        }

    def list_models(self) -> list[dict[str, Any]]:
        """List available models."""
        return [
            {"id": "qwen-omni", "object": "model", "created": 0, "permission": [], "root": "qwen-omni"},
        ]
