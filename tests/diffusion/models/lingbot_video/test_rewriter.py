# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json

import pytest
from PIL import Image

from vllm_omni.diffusion.models.lingbot_video.auto_negative import (
    categorized_negative,
    prune_negative,
)
from vllm_omni.diffusion.models.lingbot_video.request_utils import LingBotGenerationMode
from vllm_omni.diffusion.models.lingbot_video.rewriter import (
    LingBotVideoRewriter,
    build_expand_prompt,
    build_map_prompt,
    needs_rewrite,
    parse_caption,
)
from vllm_omni.diffusion.models.lingbot_video.rewriter_backends import (
    HTTPRewriterBackend,
    TransformersRewriterBackend,
    _chat_completions_url,
    build_lingbot_rewriter,
)
from vllm_omni.diffusion.models.lingbot_video.rewriter_prompts import (
    VIDEO_DURATION_EN,
    VIDEO_DURATION_ZH,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class FakeBackend:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def generate(self, text, image, use_lora):
        self.calls.append((text, image, use_lora))
        return self.replies.pop(0)


def _rewrite(rewriter, prompt, *, mode=LingBotGenerationMode.T2V, image=None, negative="negative"):
    return rewriter.rewrite_request(
        prompt=prompt,
        negative_prompt=negative,
        mode=mode,
        num_frames=121 if mode is not LingBotGenerationMode.T2I else 1,
        fps=24,
        input_image=image,
    )


def test_rewriter_uses_two_turns_and_enables_adapter_only_for_mapping():
    backend = FakeBackend(["A red fox trots through snow.", '{"b": 2, "a": 1}'])
    rewriter = LingBotVideoRewriter(backend)

    prompt, negative = _rewrite(rewriter, "a red fox")

    assert prompt == '{"b":2,"a":1}'
    assert negative == "negative"
    assert [use_lora for _, _, use_lora in backend.calls] == [False, True]
    assert all(image is None for _, image, _ in backend.calls)
    assert backend.calls[0][0].endswith(f"a red fox\n\n{VIDEO_DURATION_EN.format(duration=5)}")
    assert "DETAILED CAPTION:\nA red fox trots through snow." in backend.calls[1][0]


def test_ti2v_rewriter_passes_the_condition_image_to_both_turns():
    image = Image.new("RGB", (16, 8), color="blue")
    backend = FakeBackend(["The subject moves.", '{"caption":"mapped"}'])
    rewriter = LingBotVideoRewriter(backend)

    prompt, _ = _rewrite(
        rewriter,
        "the subject moves",
        mode=LingBotGenerationMode.TI2V,
        image=image,
    )

    assert prompt == '{"caption":"mapped"}'
    assert [call[1] for call in backend.calls] == [image, image]


def test_structured_caption_skips_rewriting():
    backend = FakeBackend([])
    rewriter = LingBotVideoRewriter(backend)
    caption = '{"comprehensive_description":"a red fox"}'

    prompt, negative = _rewrite(rewriter, caption)

    assert prompt == caption
    assert negative == "negative"
    assert backend.calls == []


def test_non_primary_rank_uses_broadcast_result_without_calling_backend(monkeypatch):
    from vllm_omni.diffusion.models.lingbot_video import rewriter as module

    backend = FakeBackend([])
    rewriter = LingBotVideoRewriter(backend)
    monkeypatch.setattr(module.dist, "is_available", lambda: True)
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_rank", lambda: 1)

    def fake_broadcast(objects, src):
        assert src == 0
        objects[0] = (True, ('{"caption":"broadcast"}', "broadcast negative"))

    monkeypatch.setattr(module.dist, "broadcast_object_list", fake_broadcast)

    prompt, negative = _rewrite(rewriter, "a red fox")

    assert prompt == '{"caption":"broadcast"}'
    assert negative == "broadcast negative"
    assert backend.calls == []


def test_rewriter_does_not_fall_back_when_mapping_is_not_json():
    backend = FakeBackend(["expanded", "sorry, I cannot map that"])
    rewriter = LingBotVideoRewriter(backend)

    with pytest.raises(RuntimeError, match="no parseable structured caption"):
        _rewrite(rewriter, "a red fox")


def test_prompt_builders_select_image_and_caption_language_contracts():
    image_expand = build_expand_prompt("t2i", "a glass of tea", 5)
    assert "User image prompt:\na glass of tea" in image_expand
    assert "Video Duration" not in image_expand
    assert "DETAILED CAPTION:\nexpanded" in build_map_prompt("t2i", "expanded", 5)

    chinese = build_expand_prompt("t2v", "一只狐狸在雪地里奔跑", 3)
    assert VIDEO_DURATION_ZH.format(duration=3) in chinese
    assert VIDEO_DURATION_EN.format(duration=3) not in chinese


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('Here you go: {"a": 1} hope that helps', {"a": 1}),
        ('["not", "an", "object"]', None),
        ("no json here", None),
    ],
)
def test_parse_caption_tolerates_fences_and_prose(raw, expected):
    assert parse_caption(raw) == expected


def test_needs_rewrite_follows_structured_caption_shape():
    assert needs_rewrite("a red fox")
    assert not needs_rewrite('{"caption":"a red fox"}')
    assert not needs_rewrite('  {"caption":"a red fox"}')


def test_auto_negative_keeps_default_order_and_only_deletes_terms():
    default = {
        "universal_negative": {
            "visual_quality": ["low quality", "underexposed", "crushed blacks"],
            "artistic_style": ["painting", "cartoon"],
            "temporal_and_motion_stability": ["motion blur", "warping"],
        }
    }
    model_output = {
        "universal_negative": {
            "visual_quality": ["invented", "low quality"],
            "artistic_style": ["cartoon", "painting"],
            "temporal_and_motion_stability": ["warping", "motion blur"],
        }
    }

    result = prune_negative(default, model_output, "a moody night scene with deep shadows")

    assert result["universal_negative"]["visual_quality"] == ["low quality"]
    assert result["universal_negative"]["artistic_style"] == ["painting", "cartoon"]
    assert result["universal_negative"]["temporal_and_motion_stability"] == [
        "motion blur",
        "warping",
    ]


def test_auto_negative_runs_after_prompt_mapping_and_uses_base_model():
    default_negative = json.dumps(
        {
            "universal_negative": {
                "visual_quality": ["low quality", "underexposed"],
                "artistic_style": ["painting"],
            }
        }
    )
    backend = FakeBackend(
        [
            "A night scene.",
            '{"comprehensive_description":"a dim night scene"}',
            '{"universal_negative":{"visual_quality":["low quality"],"artistic_style":["painting"]}}',
        ]
    )
    rewriter = LingBotVideoRewriter(backend, auto_negative=True)

    prompt, negative = _rewrite(
        rewriter,
        "a night scene",
        negative=default_negative,
    )

    assert json.loads(prompt)["comprehensive_description"] == "a dim night scene"
    assert json.loads(negative)["universal_negative"]["visual_quality"] == ["low quality"]
    assert [use_lora for _, _, use_lora in backend.calls] == [False, True, False]


@pytest.mark.parametrize(
    "negative",
    [
        "blurry, low quality",
        '{"universal_negative":[]}',
        '{"universal_negative":{"visual_quality":"not a list"}}',
        '{"universal_negative":{"visual_quality":[1,2]}}',
    ],
)
def test_categorized_negative_rejects_unprunable_shapes(negative):
    assert categorized_negative(negative) is None


@pytest.mark.parametrize(
    ("base", "expected"),
    [
        ("http://host:8000", "http://host:8000/v1/chat/completions"),
        ("http://host:8000/v1", "http://host:8000/v1/chat/completions"),
        (
            "http://host:8000/v1/chat/completions",
            "http://host:8000/v1/chat/completions",
        ),
    ],
)
def test_chat_completions_url_accepts_server_or_endpoint_urls(base, expected):
    assert _chat_completions_url(base) == expected


def test_http_backend_selects_endpoint_model_and_encodes_image(mocker):
    response = mocker.Mock()
    response.json.return_value = {"choices": [{"message": {"content": " mapped "}}]}
    post = mocker.patch(
        "vllm_omni.diffusion.models.lingbot_video.rewriter_backends.httpx.post",
        return_value=response,
    )
    backend = HTTPRewriterBackend(
        url="http://expand:8000",
        map_url="http://map:8001/v1",
        expand_model="base-model",
        map_model="map-model",
        timeout=12.0,
    )

    result = backend.generate("map this", Image.new("RGB", (2, 2)), use_lora=True)

    assert result == "mapped"
    url = post.call_args.args[0]
    payload = post.call_args.kwargs["json"]
    assert url == "http://map:8001/v1/chat/completions"
    assert payload["model"] == "map-model"
    assert payload["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert post.call_args.kwargs["timeout"] == 12.0
    response.raise_for_status.assert_called_once_with()


def test_build_rewriter_selects_backend_and_is_off_by_default():
    assert build_lingbot_rewriter({}) is None

    remote = build_lingbot_rewriter(
        {
            "rewriter_url": "http://host:8000",
            "rewriter_expand_model": "base",
            "rewriter_map_model": "mapped",
            "rewriter_auto_negative": True,
        }
    )
    assert isinstance(remote, LingBotVideoRewriter)
    assert isinstance(remote.backend, HTTPRewriterBackend)
    assert remote.auto_negative is True

    local = build_lingbot_rewriter(
        {
            "rewriter_model_path": "Qwen/Qwen3.6-27B",
            "rewriter_adapter_path": "robbyant/lingbot-video-rewriter-lora",
        }
    )
    assert isinstance(local.backend, TransformersRewriterBackend)
    assert local.backend.model is None


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            {"rewriter_url": "http://host", "rewriter_model_path": "model"},
            "mutually exclusive",
        ),
        ({"rewriter_map_url": "http://map"}, "requires"),
        ({"rewriter_model_path": "model"}, "requires"),
        ({"rewriter_adapter_path": "adapter"}, "requires"),
        ({"rewriter_auto_negative": True}, "requires"),
        ({"rewriter_timeout": 0}, "positive"),
        ({"rewriter_timeout": float("nan")}, "positive"),
        ({"rewriter_max_new_tokens": 0}, "positive"),
        ({"rewriter_max_new_tokens": 1.5}, "positive integer"),
        ({"rewriter_auto_negative": "true"}, "boolean"),
    ],
)
def test_build_rewriter_rejects_invalid_configuration(config, message):
    with pytest.raises(ValueError, match=message):
        build_lingbot_rewriter(config)
