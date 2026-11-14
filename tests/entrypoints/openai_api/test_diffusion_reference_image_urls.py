# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""URL-format reference image inputs for diffusion requests.

Regression tests for the /v1/chat/completions ``image_url`` http(s) input
support: URLs must survive prompt extraction, be fetched through
MediaConnector (SSRF-safe), and reach the generation engine as PIL images
instead of being silently dropped by the base64-only consumers.
"""

from __future__ import annotations

import base64
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.serve.engine.protocol import ErrorResponse

from tests.helpers.serving_chat import build_serving_chat
from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

CONNECTOR_TARGET = "vllm_omni.entrypoints.openai.serving_chat.MediaConnector"
TEST_URL = "https://example.com/images/pic.png"


def _make_pil(color: tuple[int, int, int] = (255, 0, 0)) -> Image.Image:
    return Image.new("RGB", (16, 16), color)


def _make_b64(color: tuple[int, int, int] = (0, 255, 0)) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _make_data_url() -> str:
    return f"data:image/png;base64,{_make_b64((0, 0, 255))}"


def _build_serving() -> OmniOpenAIServingChat:
    serving = build_serving_chat()
    serving.model_config = SimpleNamespace(
        allowed_local_media_path="",
        allowed_media_domains=None,
    )
    return serving


def _user_messages(url: str) -> list[dict[str, object]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "edit this image"},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        }
    ]


class TestExtractDiffusionPromptAndMedia:
    """URL handling at the extraction layer (_extract_diffusion_prompt_and_media)."""

    def test_http_url_image_is_kept_as_raw_url(self):
        serving = _build_serving()
        prompt, images, videos, audios = serving._extract_diffusion_prompt_and_media(_user_messages(TEST_URL))

        assert prompt == "edit this image"
        assert images == [TEST_URL]
        assert videos == []
        assert audios == []

    def test_https_url_image_is_kept_as_raw_url(self):
        serving = _build_serving()
        _, images, _, _ = serving._extract_diffusion_prompt_and_media(
            _user_messages("https://example.com/images/other.png")
        )

        assert images == ["https://example.com/images/other.png"]

    def test_data_url_still_strips_to_b64_payload(self):
        serving = _build_serving()
        data_url = _make_data_url()
        _, images, _, _ = serving._extract_diffusion_prompt_and_media(_user_messages(data_url))

        assert images == [data_url.split(",", 1)[1]]

    def test_plain_base64_image_item_still_collected(self):
        serving = _build_serving()
        b64 = _make_b64()
        messages = [{"role": "user", "content": [{"image": b64}]}]
        _, images, _, _ = serving._extract_diffusion_prompt_and_media(messages)

        assert images == [b64]

    def test_non_http_scheme_url_is_ignored(self):
        serving = _build_serving()
        _, images, _, _ = serving._extract_diffusion_prompt_and_media(_user_messages("ftp://example.com/pic.png"))

        assert images == []

    def test_empty_url_is_ignored(self):
        serving = _build_serving()
        _, images, _, _ = serving._extract_diffusion_prompt_and_media(_user_messages(""))

        assert images == []


class TestResolveReferenceImagesToPil:
    """Unit tests for the shared base64/URL reference-image resolver."""

    @pytest.mark.asyncio
    async def test_url_is_fetched_via_media_connector_and_appended(self):
        serving = _build_serving()
        pil = _make_pil()
        with patch(CONNECTOR_TARGET) as connector_cls:
            connector = connector_cls.return_value
            connector.fetch_image_async = AsyncMock(return_value=pil)

            resolved = await serving._resolve_reference_images_to_pil([TEST_URL])

        connector.fetch_image_async.assert_awaited_once_with(TEST_URL)
        assert resolved == [pil]
        assert isinstance(resolved[0], Image.Image)

    @pytest.mark.asyncio
    async def test_connector_receives_model_config_media_domains(self):
        serving = _build_serving()
        serving.model_config = SimpleNamespace(
            allowed_local_media_path="/data/media",
            allowed_media_domains=["example.com"],
        )
        with patch(CONNECTOR_TARGET) as connector_cls:
            connector_cls.return_value.fetch_image_async = AsyncMock(return_value=_make_pil())

            await serving._resolve_reference_images_to_pil([TEST_URL])

        connector_cls.assert_called_once_with(
            media_io_kwargs=None,
            allowed_local_media_path="/data/media",
            allowed_media_domains=["example.com"],
        )

    @pytest.mark.asyncio
    async def test_base64_is_decoded_without_remote_fetch(self):
        serving = _build_serving()
        with patch(CONNECTOR_TARGET) as connector_cls:
            connector = connector_cls.return_value
            connector.fetch_image_async = AsyncMock()

            resolved = await serving._resolve_reference_images_to_pil([_make_b64()])

        connector.fetch_image_async.assert_not_awaited()
        assert len(resolved) == 1
        assert isinstance(resolved[0], Image.Image)

    @pytest.mark.asyncio
    async def test_invalid_base64_is_skipped_without_raising(self):
        serving = _build_serving()
        with patch(CONNECTOR_TARGET):
            resolved = await serving._resolve_reference_images_to_pil(["!!!not-base64!!!"])

        assert resolved == []

    @pytest.mark.asyncio
    async def test_failed_url_fetch_is_skipped_without_raising(self):
        serving = _build_serving()
        with patch(CONNECTOR_TARGET) as connector_cls:
            connector = connector_cls.return_value
            connector.fetch_image_async = AsyncMock(side_effect=RuntimeError("blocked domain"))

            resolved = await serving._resolve_reference_images_to_pil([TEST_URL])

        assert resolved == []

    @pytest.mark.asyncio
    async def test_mixed_inputs_resolve_in_order(self):
        serving = _build_serving()
        url_pil = _make_pil((255, 0, 0))
        with patch(CONNECTOR_TARGET) as connector_cls:
            connector = connector_cls.return_value
            connector.fetch_image_async = AsyncMock(return_value=url_pil)

            resolved = await serving._resolve_reference_images_to_pil([TEST_URL, _make_b64((0, 255, 0))])

        assert len(resolved) == 2
        assert resolved[0] is url_pil
        assert isinstance(resolved[1], Image.Image)

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty_list(self):
        serving = _build_serving()
        with patch(CONNECTOR_TARGET) as connector_cls:
            resolved = await serving._resolve_reference_images_to_pil([])

        connector_cls.assert_not_called()
        assert resolved == []


class TestGenerateDiffusionImagesFromUrl:
    """API regression: a mocked connector returns a real PIL image and it
    must reach the generation engine as multi-modal input."""

    def _make_engine(self, output_pil: Image.Image) -> MagicMock:
        engine = MagicMock()
        engine.generate = AsyncMock(
            return_value=SimpleNamespace(
                images=[output_pil],
                stage_durations=None,
                peak_memory_mb=None,
                metrics=None,
                prompt=None,
            )
        )
        return engine

    @pytest.mark.asyncio
    async def test_url_reference_reaches_generation_as_pil(self):
        serving = _build_serving()
        url_pil = _make_pil((255, 0, 0))
        output_pil = _make_pil((0, 255, 0))
        engine = self._make_engine(output_pil)
        serving._diffusion_engine = engine

        with patch(CONNECTOR_TARGET) as connector_cls:
            connector = connector_cls.return_value
            connector.fetch_image_async = AsyncMock(return_value=url_pil)

            result = await serving.generate_diffusion_images(
                prompt="a red square",
                reference_images=[TEST_URL],
            )

        images, _stage_durations, _peak_memory, _cot, _metrics = result
        assert images == [output_pil]

        # The URL image reached generation as PIL multi-modal input.
        gen_prompt = engine.generate.call_args.kwargs["prompt"]
        assert gen_prompt["multi_modal_data"]["image"] is url_pil
        assert isinstance(gen_prompt["multi_modal_data"]["image"], Image.Image)
        connector.fetch_image_async.assert_awaited_once_with(TEST_URL)

    @pytest.mark.asyncio
    async def test_failed_url_fetch_still_generates_without_reference(self):
        serving = _build_serving()
        output_pil = _make_pil((0, 255, 0))
        engine = self._make_engine(output_pil)
        serving._diffusion_engine = engine

        with patch(CONNECTOR_TARGET) as connector_cls:
            connector = connector_cls.return_value
            connector.fetch_image_async = AsyncMock(side_effect=RuntimeError("blocked domain"))

            result = await serving.generate_diffusion_images(
                prompt="a red square",
                reference_images=[TEST_URL],
            )

        images = result[0]
        assert images == [output_pil]
        # No reference image reached generation: best-effort behavior.
        gen_prompt = engine.generate.call_args.kwargs["prompt"]
        assert "multi_modal_data" not in gen_prompt

    @pytest.mark.asyncio
    async def test_base64_reference_still_reaches_generation(self):
        serving = _build_serving()
        output_pil = _make_pil((0, 255, 0))
        engine = self._make_engine(output_pil)
        serving._diffusion_engine = engine

        with patch(CONNECTOR_TARGET) as connector_cls:
            connector = connector_cls.return_value
            connector.fetch_image_async = AsyncMock()

            result = await serving.generate_diffusion_images(
                prompt="a red square",
                reference_images=[_make_b64((255, 0, 0))],
            )

        images = result[0]
        assert images == [output_pil]
        connector.fetch_image_async.assert_not_awaited()
        gen_prompt = engine.generate.call_args.kwargs["prompt"]
        assert isinstance(gen_prompt["multi_modal_data"]["image"], Image.Image)


class _AwaitableAsyncIterable:
    """Result wrapper usable as both ``await x`` and ``async for ... in x``."""

    def __init__(self, value: object) -> None:
        self._value = value

    def __await__(self):
        if False:  # pragma: no cover - generator marker
            yield
        return self._value

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._value is None:
            raise StopAsyncIteration
        value, self._value = self._value, None
        return value


class _FakeDiffusionEngine:
    """Pure-diffusion engine stand-in covering both call forms of generate()
    (``await`` on the images path, ``async for`` on the chat path)."""

    def __init__(self, output_pil: Image.Image) -> None:
        self.result = SimpleNamespace(
            final_output_type="image",
            images=[output_pil],
            multimodal_output={},
            stage_durations=None,
            peak_memory_mb=None,
            metrics=None,
            prompt=None,
        )
        self.stage_configs: list[object] = []
        self.calls: list[dict[str, object]] = []

    def generate(self, prompt, sampling_params_list=None, request_id=None, **kwargs):
        self.calls.append({"prompt": prompt, "kwargs": kwargs})
        return _AwaitableAsyncIterable(self.result)


def _build_for_diffusion(
    output_pil: Image.Image | None = None,
    *,
    allowed_local_media_path: str = "",
    allowed_media_domains: list[str] | None = None,
) -> tuple[OmniOpenAIServingChat, _FakeDiffusionEngine]:
    """Build the handler the way pure-diffusion servers do (api_server.py):
    via for_diffusion(), which bypasses __init__ and has no model_config."""
    engine = _FakeDiffusionEngine(output_pil or _make_pil((0, 255, 0)))
    serving = OmniOpenAIServingChat.for_diffusion(
        diffusion_engine=engine,  # type: ignore[arg-type]
        model_name="test-diffusion-model",
        allowed_local_media_path=allowed_local_media_path,
        allowed_media_domains=allowed_media_domains,
    )
    return serving, engine


def _chat_request(content_url: str | None) -> ChatCompletionRequest:
    content: list[dict[str, object]] = [{"type": "text", "text": "edit this image"}]
    if content_url is not None:
        content.append({"type": "image_url", "image_url": {"url": content_url}})
    return ChatCompletionRequest(
        model="test-diffusion-model",
        messages=[{"role": "user", "content": content}],
    )


class TestForDiffusionReferenceImages:
    """Regression for the #9829 review finding: for_diffusion() instances
    never set model_config, so reference-image resolution must not depend
    on it. Base64 inputs decode locally; http(s) URLs go through
    MediaConnector with the SSRF args threaded via for_diffusion()."""

    @pytest.mark.asyncio
    async def test_base64_resolves_without_model_config(self):
        serving, _engine = _build_for_diffusion()
        assert "model_config" not in vars(serving)

        resolved = await serving._resolve_reference_images_to_pil([_make_b64()])

        assert len(resolved) == 1
        assert isinstance(resolved[0], Image.Image)

    @pytest.mark.asyncio
    async def test_pure_base64_never_constructs_media_connector(self):
        serving, _engine = _build_for_diffusion()
        with patch(CONNECTOR_TARGET) as connector_cls:
            resolved = await serving._resolve_reference_images_to_pil([_make_b64()])

        connector_cls.assert_not_called()
        assert len(resolved) == 1

    @pytest.mark.asyncio
    async def test_url_uses_for_diffusion_media_domains(self):
        serving, _engine = _build_for_diffusion(
            allowed_local_media_path="/data/media",
            allowed_media_domains=["example.com"],
        )
        with patch(CONNECTOR_TARGET) as connector_cls:
            connector_cls.return_value.fetch_image_async = AsyncMock(return_value=_make_pil())

            resolved = await serving._resolve_reference_images_to_pil([TEST_URL])

        connector_cls.assert_called_once_with(
            media_io_kwargs=None,
            allowed_local_media_path="/data/media",
            allowed_media_domains=["example.com"],
        )
        assert len(resolved) == 1

    @pytest.mark.asyncio
    async def test_url_reference_reaches_generation_as_pil(self):
        url_pil = _make_pil((255, 0, 0))
        output_pil = _make_pil((0, 255, 0))
        serving, engine = _build_for_diffusion(output_pil)

        with patch(CONNECTOR_TARGET) as connector_cls:
            connector_cls.return_value.fetch_image_async = AsyncMock(return_value=url_pil)

            result = await serving.generate_diffusion_images(
                prompt="a red square",
                reference_images=[TEST_URL],
            )

        images = result[0]
        assert images == [output_pil]
        gen_prompt = engine.calls[0]["prompt"]
        assert gen_prompt["multi_modal_data"]["image"] is url_pil

    @pytest.mark.asyncio
    async def test_base64_chat_completion_succeeds_without_model_config(self):
        # Before the fix this returned 500: "Image generation failed:
        # 'OmniOpenAIServingChat' object has no attribute 'model_config'".
        output_pil = _make_pil((0, 255, 0))
        serving, engine = _build_for_diffusion(output_pil)

        response = await serving.create_chat_completion(_chat_request(_make_data_url()), raw_request=None)

        assert not isinstance(response, ErrorResponse)
        gen_prompt = engine.calls[0]["prompt"]
        assert isinstance(gen_prompt["multi_modal_data"]["image"], Image.Image)

    @pytest.mark.asyncio
    async def test_url_chat_completion_succeeds_without_model_config(self):
        output_pil = _make_pil((0, 255, 0))
        serving, engine = _build_for_diffusion(output_pil)

        with patch(CONNECTOR_TARGET) as connector_cls:
            connector_cls.return_value.fetch_image_async = AsyncMock(return_value=_make_pil((255, 0, 0)))

            response = await serving.create_chat_completion(_chat_request(TEST_URL), raw_request=None)

        assert not isinstance(response, ErrorResponse)
        gen_prompt = engine.calls[0]["prompt"]
        assert isinstance(gen_prompt["multi_modal_data"]["image"], Image.Image)
        connector_cls.return_value.fetch_image_async.assert_awaited_once_with(TEST_URL)

    @pytest.mark.asyncio
    async def test_text_only_chat_completion_unaffected(self):
        output_pil = _make_pil((0, 255, 0))
        serving, engine = _build_for_diffusion(output_pil)

        response = await serving.create_chat_completion(_chat_request(None), raw_request=None)

        assert not isinstance(response, ErrorResponse)
        gen_prompt = engine.calls[0]["prompt"]
        assert "multi_modal_data" not in gen_prompt
