"""Unit tests for multi-modal API."""

from vllm_omni.entrypoints.openai.multi_modal_api import (
    ChatCompletionRequest,
    ChatMessage,
    ContentType,
    MultiModalAPIServer,
    MultiModalContent,
)


class TestMultiModalAPIServer:
    """Tests for MultiModalAPIServer."""

    def test_parse_image_content(self):
        """Test parsing image content."""
        server = MultiModalAPIServer()
        messages = [
            ChatMessage(
                role="user",
                content=[MultiModalContent(content_type=ContentType.IMAGE_URL, url="https://example.com/image.jpg")],
            )
        ]

        result = server.parse_multimodal_content(messages)
        assert "image" in result
        assert result["image"] == "https://example.com/image.jpg"

    def test_parse_text_content(self):
        """Test parsing text content."""
        server = MultiModalAPIServer()
        messages = [ChatMessage(role="user", content="Hello world")]

        prompt = server.extract_text_prompt(messages)
        assert prompt == "Hello world"

    def test_extract_text_from_multimodal(self):
        """Test extracting text from multi-modal content."""
        server = MultiModalAPIServer()
        messages = [
            ChatMessage(
                role="user",
                content=[
                    MultiModalContent(content_type=ContentType.TEXT, text="Describe this image"),
                    MultiModalContent(content_type=ContentType.IMAGE_URL, url="https://example.com/img.png"),
                ],
            )
        ]

        prompt = server.extract_text_prompt(messages)
        assert "Describe this image" in prompt

    def test_build_sampling_params(self):
        """Test building sampling parameters."""
        server = MultiModalAPIServer()
        request = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="test")],
            temperature=0.8,
            top_p=0.9,
            max_tokens=100,
        )

        params = server.build_sampling_params(request)
        assert params["temperature"] == 0.8
        assert params["top_p"] == 0.9
        assert params["max_tokens"] == 100

    def test_build_sampling_params_with_stop(self):
        """Test building sampling params with stop."""
        server = MultiModalAPIServer()
        request = ChatCompletionRequest(
            model="test", messages=[ChatMessage(role="user", content="test")], stop=["STOP", "END"]
        )

        params = server.build_sampling_params(request)
        assert params["stop"] == ["STOP", "END"]

    def test_list_models(self):
        """Test listing models."""
        server = MultiModalAPIServer()
        models = server.list_models()

        assert len(models) > 0
        assert "id" in models[0]


class TestMultiModalContent:
    """Tests for MultiModalContent."""

    def test_image_url_content(self):
        """Test image URL content."""
        content = MultiModalContent(
            content_type=ContentType.IMAGE_URL, url="https://example.com/image.jpg", detail="high"
        )

        assert content.url == "https://example.com/image.jpg"
        assert content.detail == "high"

    def test_audio_base64_content(self):
        """Test audio base64 content."""
        content = MultiModalContent(content_type=ContentType.AUDIO_BASE64, base64_data="dGVzdA==")

        assert content.base64_data == "dGVzdA=="


class TestChatMessage:
    """Tests for ChatMessage."""

    def test_string_content(self):
        """Test string content."""
        message = ChatMessage(role="user", content="Hello")
        assert message.content == "Hello"

    def test_list_content(self):
        """Test list content."""
        content = [MultiModalContent(content_type=ContentType.TEXT, text="Hi")]
        message = ChatMessage(role="user", content=content)
        assert len(message.content) == 1
