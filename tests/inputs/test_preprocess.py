# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise Omni routing through the active upstream rendering pipeline."""

import asyncio
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from vllm.exceptions import VLLMValidationError
from vllm.inputs import mm_input, tokens_input
from vllm.multimodal.inputs import MultiModalKwargsItems, PlaceholderRange
from vllm.multimodal.parse import MultiModalDataParser
from vllm.pooling_params import PoolingParams
from vllm.renderers import BaseRenderer
from vllm.renderers.params import TokenizeParams
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from tests.helpers.fixtures import ipc
from vllm_omni.core.sched.output import OmniNewRequestData
from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.async_engine_utils import apply_omni_final_stage_metadata, upgrade_to_omni_request
from vllm_omni.inputs import preprocess as preprocess_mod
from vllm_omni.inputs.input_processor import OmniInputProcessor
from vllm_omni.inputs.preprocess import OmniRenderer, build_omni_renderer, omni_renderer_cls
from vllm_omni.inputs.processed_media import ProcessedMediaIdentity, ProcessedMediaProvenance
from vllm_omni.request import OmniRequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
executor_roundtrip = ipc.executor_roundtrip


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
@pytest.mark.parametrize("processor_cls", [InputProcessor, OmniInputProcessor])
def test_process_inputs_routes_no_media_processor_kwargs(renderer, kwargs, tokenized, processor_cls):
    # Keep real process_inputs/render_cmpl; stub unrelated config validation.
    processor = object.__new__(processor_cls)
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


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("size", [123, "bad", [], None, {"shortest_edge": 4096}])
def test_qwen_video_size_validation_precedes_multimodal_processing(renderer, mocker, async_mode, size):
    renderer.model_config.hf_config = SimpleNamespace(model_type="qwen3_vl")
    renderer.get_mm_processor = mocker.Mock(
        return_value=SimpleNamespace(info=SimpleNamespace(ctx=SimpleNamespace(get_merged_mm_kwargs=lambda kw, **_: kw)))
    )
    prompt = {"prompt": "video", "multi_modal_data": {"video": ["frames"]}, "mm_processor_kwargs": {"size": size}}

    def render():
        if async_mode:
            return asyncio.run(renderer.render_cmpl_async([prompt]))
        return renderer.render_cmpl([prompt])

    if size is not None and not isinstance(size, dict):
        with pytest.raises(VLLMValidationError, match="size must be a dictionary"):
            render()
        renderer._process_multimodal.assert_not_called()
    else:
        render()
        renderer._process_multimodal.assert_called_once()


def test_other_video_processors_keep_their_own_size_contract(renderer):
    renderer.model_config.hf_config = SimpleNamespace(model_type="other")
    renderer.render_cmpl(
        [
            {
                "prompt": "video",
                "multi_modal_data": {"video": ["frames"]},
                "mm_processor_kwargs": {"size": 224, "fps": [1], "num_frames": []},
            }
        ]
    )
    renderer._process_multimodal.assert_called_once()


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("model_type", ["qwen3_vl", "qwen3_vl_moe"])
@pytest.mark.parametrize("from_config", [False, True])
@pytest.mark.parametrize(
    "options,valid",
    [
        ({"fps": []}, False),
        ({"num_frames": []}, False),
        ({"fps": [1.0]}, False),
        ({"num_frames": [2]}, False),
        ({"fps": [1.0, "bad"]}, False),
        ({"num_frames": [2, 2.5]}, False),
        ({"fps": [1, 2]}, False),
        ({"fps": "bad"}, False),
        ({"num_frames": {}}, False),
        ({"fps": True}, False),
        ({"num_frames": [2, True]}, False),
        ({"fps": float("inf")}, False),
        ({"fps": [1.0, float("nan")]}, False),
        ({"num_frames": 0}, False),
        ({"fps": -1.0}, False),
        ({"fps": None, "num_frames": None}, True),
        ({"fps": 1}, True),
        ({"fps": 2.5}, True),
        ({"num_frames": 2}, True),
        ({"fps": [1.0, 2.5], "num_frames": [2, 4]}, True),
    ],
)
def test_qwen_video_sampling_options(renderer, async_mode, model_type, from_config, options, valid):
    parser = MultiModalDataParser()
    renderer.model_config.hf_config = SimpleNamespace(model_type=model_type)
    renderer.get_mm_processor = lambda: SimpleNamespace(
        info=SimpleNamespace(
            ctx=SimpleNamespace(get_merged_mm_kwargs=lambda kw, **_: {**(options if from_config else {}), **kw}),
            parse_mm_data=lambda data, **_: parser.parse_mm_data(data),
        )
    )
    video = torch.zeros(2, 4, 4, 3, dtype=torch.uint8)
    prompt = {
        "prompt": "video",
        "multi_modal_data": {"video": [video, video]},
        "mm_processor_kwargs": {} if from_config else options,
    }

    def render():
        if async_mode:
            return asyncio.run(renderer.render_cmpl_async([prompt]))
        return renderer.render_cmpl([prompt])

    if valid:
        render()
        renderer._process_multimodal.assert_called_once()
    else:
        with pytest.raises(VLLMValidationError, match="Video processor"):
            render()
        renderer._process_multimodal.assert_not_called()


@pytest.mark.parametrize("video_form", ["array", "frames", "metadata", "cached"])
def test_qwen_video_sampling_lists_count_videos_not_frames(renderer, video_form):
    video = torch.zeros(3, 4, 4, 3, dtype=torch.uint8)
    data = {"array": video, "frames": list(video), "metadata": (video, {}), "cached": [None]}[video_form]
    parser = MultiModalDataParser()
    renderer.model_config.hf_config = SimpleNamespace(model_type="qwen3_vl")
    renderer.get_mm_processor = lambda: SimpleNamespace(
        info=SimpleNamespace(
            ctx=SimpleNamespace(get_merged_mm_kwargs=lambda kw, **_: {"fps": [1.0], **kw}),
            parse_mm_data=lambda data, **_: parser.parse_mm_data(data),
        )
    )
    renderer.render_cmpl([{"prompt": "video", "multi_modal_data": {"video": data}}])
    renderer._process_multimodal.assert_called_once()


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("no_media_kwargs", [False, True])
@pytest.mark.parametrize("limit", [2, 4])
def test_renderer_applies_expanded_prompt_bound_without_splitting_media(renderer, async_mode, no_media_kwargs, limit):
    renderer._process_multimodal.return_value = mm_input(
        list(range(6)), {}, {"image": ["hash"]}, {"image": [PlaceholderRange(offset=1, length=2)]}
    )
    prompt = {"prompt_token_ids": [1, 2, 3], "additional_information": {"speaker": 1}}
    if no_media_kwargs:
        prompt["mm_processor_kwargs"] = {"target_h": 512}
    else:
        prompt["multi_modal_data"] = {"image": "image"}
    params = TokenizeParams(max_total_tokens=128, truncate_prompt_tokens=limit, truncation_side="right")

    def render():
        if async_mode:
            return asyncio.run(renderer.render_cmpl_async([prompt], params))
        return renderer.render_cmpl([prompt], params)

    if limit == 2:
        with pytest.raises(VLLMValidationError, match="would split or drop"):
            render()
    else:
        (result,) = render()
        assert result["prompt_token_ids"] == [0, 1, 2, 3]
        assert result["mm_placeholders"]["image"] == [PlaceholderRange(offset=1, length=2)]
        assert result["additional_information"] == {"speaker": 1}


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


def _provenance():
    return ProcessedMediaProvenance(
        (1, 90, 90, 2),
        (ProcessedMediaIdentity("image", "a" * 64, "b" * 64, 1, 2),),
        ("untrusted-caller-routing-uuid",),
    )


def _media_input(proof):
    result = mm_input(
        prompt_token_ids=[1, 90, 90, 2],
        mm_kwargs=MultiModalKwargsItems({"image": [None]}),
        mm_hashes={"image": ["untrusted-caller-routing-uuid"]},
        mm_placeholders={"image": [PlaceholderRange(offset=1, length=2)]},
    )
    result["_omni_processed_media"] = proof
    return result


def _omni_processor(renderer):
    processor = object.__new__(OmniInputProcessor)
    processor.renderer = renderer
    processor.model_config = renderer.model_config
    processor.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=1, data_parallel_size_local=1, local_engines_only=False)
    )
    processor._validate_params = Mock()
    processor._validate_lora = Mock()
    processor._validate_model_inputs = Mock()
    processor.lora_config = None
    return processor


@pytest.mark.parametrize("already_rendered", [False, True])
def test_provenance_survives_input_engine_and_scheduler_wires(renderer, already_rendered, executor_roundtrip):
    proof = _provenance()
    rendered = _media_input(proof)
    renderer._process_multimodal.return_value = rendered
    processor = _omni_processor(renderer)
    raw = {
        "prompt_token_ids": [1, 9, 2],
        "multi_modal_data": {"image": "source"},
        "additional_information": {"_omni_processed_media": {"fake": True}},
    }
    request = processor.process_inputs(
        "current", rendered if already_rendered else raw, PoolingParams(), ("embed",), session_id="session"
    )
    assert isinstance(request, OmniEngineCoreRequest)
    assert request.processed_media_provenance is proof
    request = apply_omni_final_stage_metadata(upgrade_to_omni_request(request, raw), 1)
    encoded = MsgpackEncoder().encode(request)
    decoded = MsgpackDecoder(OmniEngineCoreRequest).decode(encoded)
    assert decoded.processed_media_provenance == proof
    assert decoded.session_id == "session"
    admitted = OmniRequest.from_engine_core_request(decoded, None)
    for scheduled in (
        OmniNewRequestData.from_request(admitted, ([1],)),
        OmniNewRequestData.from_base(OmniNewRequestData.from_request(admitted, ([1],)), admitted),
    ):
        # Exercise the scheduler broadcast, separately from frontend msgpack.
        received = executor_roundtrip(scheduled)
        assert received.processed_media_provenance == proof
        received.processed_media_provenance.validate(received.prompt_token_ids, received.mm_features)


def test_plain_json_cannot_mint_processor_provenance(renderer):
    processor = _omni_processor(renderer)
    fake = asdict(_provenance())
    request = processor.process_inputs("fake", _media_input(fake), PoolingParams(), ("embed",))
    assert getattr(request, "processed_media_provenance", None) is None
    raw = {
        "prompt_token_ids": [1, 90, 90, 2],
        "_omni_processed_media": _provenance(),
        "additional_information": {"processed_media_provenance": fake},
    }
    # Even a typed marker on a raw prompt is not an output of the renderer.
    request = processor.process_inputs("raw", raw, PoolingParams(), ("embed",))
    request = upgrade_to_omni_request(request, raw)
    assert request.processed_media_provenance is None


@pytest.mark.parametrize("tamper", ["prompt", "hash", "position", "mask", "missing", "extra_hash"])
def test_stale_or_mismatched_processor_proof_is_rejected(renderer, tamper):
    processor = _omni_processor(renderer)
    rendered = _media_input(_provenance())
    if tamper == "prompt":
        rendered["prompt_token_ids"][0] = 999
    elif tamper == "hash":
        rendered["mm_hashes"]["image"][0] = "different"
    elif tamper == "position":
        rendered["mm_placeholders"]["image"][0] = PlaceholderRange(offset=0, length=2)
    elif tamper == "mask":
        rendered["mm_placeholders"]["image"][0] = PlaceholderRange(
            offset=1, length=2, is_embed=torch.tensor([True, False])
        )
    elif tamper == "missing":
        rendered["_omni_processed_media"] = replace(_provenance(), media=())
    else:
        rendered["_omni_processed_media"] = replace(_provenance(), routing_hashes=("extra", "hash"))
    with pytest.raises(ValueError, match="provenance"):
        processor.process_inputs("stale", rendered, PoolingParams(), ("embed",))
