# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise Omni routing through the active upstream rendering pipeline."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from vllm.inputs import tokens_input
from vllm.pooling_params import PoolingParams
from vllm.renderers import BaseRenderer
from vllm.renderers.params import TokenizeParams
from vllm.v1.engine.input_processor import InputProcessor

from vllm_omni.inputs import preprocess as preprocess_mod
from vllm_omni.inputs.preprocess import OmniRenderer, build_omni_renderer, omni_renderer_cls

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Renderer(BaseRenderer):
    def __init__(self):
        self.model_config = SimpleNamespace(is_encoder_decoder=False, enable_prompt_embeds=True)
        self.tokenizer = None
        self.default_cmpl_tok_params = TokenizeParams(max_total_tokens=128)
        self._tokenize_prompt = lambda prompt, params: {**prompt, "prompt_token_ids": [1, 2, 3]}
        self._tokenize_prompt_async = AsyncMock(side_effect=self._tokenize_prompt)
        self._process_multimodal = Mock(return_value=tokens_input([1, 2, 3, 99]))
        self._process_multimodal_async = AsyncMock(side_effect=self._process_multimodal)

    def render_messages(self, messages, params):
        return messages, {"prompt": "hello"}


@pytest.fixture
def renderer():
    return omni_renderer_cls(_Renderer)()


@pytest.mark.parametrize("kwargs", [{}, {"target_h": 512, "target_w": 768}])
@pytest.mark.parametrize("tokenized", [False, True])
def test_process_inputs_routes_no_media_processor_kwargs(renderer, kwargs, tokenized):
    # Keep real process_inputs/render_cmpl; stub unrelated config validation.
    processor = object.__new__(InputProcessor)
    processor.renderer = renderer
    processor.model_config = renderer.model_config
    processor.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_size_local=1,
            local_engines_only=False,
        )
    )
    processor._validate_params = Mock()
    processor._validate_lora = Mock()
    processor._validate_model_inputs = Mock()
    prompt: dict[str, object] = {"prompt_token_ids": [1, 2, 3]} if tokenized else {"prompt": "hello"}
    prompt.update(mm_processor_kwargs=kwargs, cache_salt="salt")
    request = processor.process_inputs("image-request", prompt, PoolingParams(), ("embed",))
    assert request.prompt_token_ids == [1, 2, 3, 99]
    assert request.cache_salt == "salt"
    renderer._process_multimodal.assert_called_once_with(
        [1, 2, 3],
        {},
        mm_processor_kwargs=kwargs,
        mm_uuids=None,
        skip_mm_cache=False,
    )


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("kind", ["text", "tokens", "media", "kwargs", "embeds"])
def test_renderer_preserves_routes_extras_and_cache_policy(renderer, async_mode, kind):
    prompt: dict[str, object] = {
        "prompt": "hello",
        "additional_information": {"speaker": 1},
        "model_intermediate_buffer": {"x": 2},
    }
    if kind == "tokens":
        prompt["prompt_token_ids"] = [1, 2, 3]
    elif kind == "media":
        prompt["multi_modal_data"] = {"image": "image"}
    elif kind == "kwargs":
        prompt["mm_processor_kwargs"] = {}
    elif kind == "embeds":
        prompt["prompt_embeds"] = torch.ones(3, 4)
    if async_mode:
        (result,) = asyncio.run(renderer.render_cmpl_async([prompt], skip_mm_cache=True))
    else:
        (result,) = renderer.render_cmpl([prompt], skip_mm_cache=True)
    assert result["additional_information"] == {"speaker": 1}
    assert result["model_intermediate_buffer"] == {"x": 2}
    assert result["prompt"] == "hello"
    if kind in ("media", "kwargs"):
        assert result["prompt_token_ids"] == [1, 2, 3, 99]
        assert renderer._process_multimodal.call_args.kwargs["skip_mm_cache"] is True
    else:
        renderer._process_multimodal.assert_not_called()
        if kind == "embeds":
            torch.testing.assert_close(result["prompt_embeds"], prompt["prompt_embeds"])
        else:
            assert result["prompt_token_ids"] == [1, 2, 3]


def test_omni_renderer_is_a_real_subclass_created_once():
    cls = omni_renderer_cls(_Renderer)
    assert issubclass(cls, OmniRenderer) and issubclass(cls, _Renderer)
    assert omni_renderer_cls(_Renderer) is cls
    assert omni_renderer_cls(cls) is cls
    assert cls.__name__ == "Omni_Renderer"
    instance = cls()
    assert isinstance(instance, _Renderer)
    assert not hasattr(instance, "_renderer")


def test_concrete_renderer_overrides_are_not_suppressed():
    """The old proxy bound BaseRenderer.render_cmpl itself and hid subclass overrides."""

    class _Custom(_Renderer):
        def render_cmpl(self, prompts, *args, **kwargs):
            return ["custom"]

    assert omni_renderer_cls(_Custom)().render_cmpl([{"prompt": "x"}]) == ["custom"]


class _CtorRenderer(BaseRenderer):
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def render_messages(self, messages, params):
        raise NotImplementedError


def test_build_omni_renderer_with_explicit_class_skips_registry(monkeypatch):
    monkeypatch.setattr(preprocess_mod, "cached_tokenizer_from_config", Mock(side_effect=AssertionError("resolved")))
    config = SimpleNamespace(model_config=SimpleNamespace())
    built = build_omni_renderer(config, renderer_cls=_CtorRenderer, tokenizer=None)
    assert isinstance(built, OmniRenderer) and isinstance(built, _CtorRenderer)
    assert built.config is config and built.tokenizer is None


def test_build_omni_renderer_resolves_like_upstream(monkeypatch):
    config = SimpleNamespace(model_config=SimpleNamespace())
    tokenizer = object()
    monkeypatch.setattr(
        preprocess_mod, "cached_tokenizer_from_config", lambda mc: tokenizer if mc is config.model_config else None
    )
    monkeypatch.setattr(preprocess_mod, "tokenizer_args_from_config", lambda mc: ("fake-mode", None))
    load_cls = Mock(return_value=_CtorRenderer)
    monkeypatch.setattr(preprocess_mod.RENDERER_REGISTRY, "load_renderer_cls", load_cls)
    built = build_omni_renderer(config)
    load_cls.assert_called_once_with("fake-mode")
    assert isinstance(built, OmniRenderer) and isinstance(built, _CtorRenderer)
    assert built.tokenizer is tokenizer
