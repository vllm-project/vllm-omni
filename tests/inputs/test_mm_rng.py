# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise request RNG identity and transport with the real vLLM cache path."""

from dataclasses import replace
from types import SimpleNamespace

import msgspec
import pytest
import torch
from PIL import Image
from transformers import BatchFeature
from vllm.config import ModelConfig, ParallelConfig, VllmConfig
from vllm.config.multimodal import MultiModalConfig
from vllm.exceptions import VLLMValidationError
from vllm.inputs import mm_input
from vllm.lora.request import LoRARequest
from vllm.multimodal.cache.lru import LruKeyReplicatedReceiverCache, LruKeyReplicatedSenderCache
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.multimodal.parse import MultiModalDataParser
from vllm.multimodal.processing import InputProcessingContext, ProcessorInputs
from vllm.multimodal.processing.context import TimingContext
from vllm.multimodal.processing.processor import PlaceholderFeaturesInfo
from vllm.pooling_params import PoolingParams
from vllm.renderers import BaseRenderer
from vllm.renderers.params import TokenizeParams
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from vllm_omni.core.sched.output import OmniNewRequestData
from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.inputs.input_processor import OmniInputProcessor
from vllm_omni.inputs.mm_rng import (
    HUNYUAN_VAE_RNG_POLICY,
    MM_RNG_CONTEXT_KEY,
    MMRNGContext,
    contextualize_mm_features,
)
from vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3 import (
    HunyuanImage3DummyInputsBuilder,
    HunyuanImage3MultiModalProcessor,
    HunyuanImage3ProcessingInfo,
)
from vllm_omni.request import OmniRequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _context(seed=0):
    return MMRNGContext(HUNYUAN_VAE_RNG_POLICY, seed, ("vae_generator_seed", "vae_token_grid_hw"))


def _item(seed=0):
    data = {
        "vae_token_grid_hw": MultiModalFieldElem(torch.tensor([2, 3]), MultiModalBatchedField(keep_on_cpu=True)),
        "vae_pixel_values": MultiModalFieldElem(torch.ones(18), MultiModalBatchedField()),
    }
    if seed is not None:
        data["vae_generator_seed"] = MultiModalFieldElem(torch.tensor(seed), MultiModalBatchedField(keep_on_cpu=True))
    return MultiModalKwargsItem(data)


def _feature(identity, index=0, seed=0):
    return MultiModalFeatureSpec(
        data=_item(seed),
        modality="image",
        identifier=identity,
        mm_hash=f"base-{identity}",
        mm_position=PlaceholderRange(offset=index * 10, length=4),
    )


def _engine_input(context, items=None):
    result = mm_input(
        prompt_token_ids=[99, 99],
        mm_kwargs={"image": [_item()] if items is None else items},
        mm_hashes={"image": ["base-A"]},
        mm_placeholders={"image": [PlaceholderRange(offset=0, length=2)]},
    )
    if context is not None:
        result[MM_RNG_CONTEXT_KEY] = context
    return result


def _input_processor(engine_input, mocker):
    processor = object.__new__(OmniInputProcessor)
    model_config = mocker.Mock(spec=ModelConfig, is_encoder_decoder=False, enable_prompt_embeds=True)
    parallel = mocker.Mock(
        spec=ParallelConfig, data_parallel_size=1, data_parallel_size_local=1, local_engines_only=False
    )
    processor.model_config = model_config
    processor.vllm_config = mocker.Mock(spec=VllmConfig, parallel_config=parallel)
    processor.lora_config = None
    processor.generation_config_fields = {}
    renderer = mocker.Mock(spec=BaseRenderer, tokenizer=None)
    renderer.default_cmpl_tok_params = TokenizeParams(max_total_tokens=128)
    renderer.render_cmpl.return_value = (engine_input,)
    renderer.get_eos_token_id.return_value = None
    renderer.get_mm_processor.return_value = HunyuanImage3MultiModalProcessor
    processor.renderer = renderer
    processor._validate_params = mocker.Mock(return_value=False)
    processor._validate_lora = mocker.Mock()
    processor._validate_model_inputs = mocker.Mock()
    processor.apply_watermarking = mocker.Mock()
    return processor


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("seed", [None, 0])
def test_raw_and_rendered_inputs_keep_actual_vae_seed(raw, seed, mocker):
    context = None if seed is None else _context(seed)
    engine_input = _engine_input(context, [_item(seed)])
    processor = _input_processor(engine_input, mocker)
    request = processor.process_inputs(
        "request", {"prompt": "image"} if raw else engine_input, SamplingParams(seed=999), ("generate",)
    )
    assert processor.renderer.render_cmpl.call_count == int(raw)
    assert request.mm_features[0].mm_hash == "base-A"
    assert request.mm_features[0].data is engine_input["mm_kwargs"]["image"][0]
    if seed is None:
        assert getattr(request, MM_RNG_CONTEXT_KEY, None) is None
        assert request.mm_features[0].identifier == "base-A"
    else:
        assert request.mm_rng_context == context
        assert request.mm_rng_context.seed != request.sampling_params.seed
        assert request.mm_features[0].identifier != "base-A"


def test_seeded_manual_engine_input_without_context_is_rejected(mocker):
    engine_input = _engine_input(None)
    processor = _input_processor(engine_input, mocker)
    with pytest.raises(ValueError, match="processor-produced mm_rng_context"):
        processor.process_inputs("manual", engine_input, PoolingParams(), ("embed",))


def test_reference_prefix_identity_shares_only_matching_rng_context():
    a, b = _feature("lora:A"), _feature("lora:B", 1)
    ab = contextualize_mm_features([a, b], _context(0))
    aa = contextualize_mm_features([a, a], _context(0))
    repeat = contextualize_mm_features([a, b], _context(0))
    assert [f.identifier for f in ab] == [f.identifier for f in repeat]
    assert ab[0].identifier == aa[0].identifier
    assert aa[0].identifier != aa[1].identifier
    assert ab[1].identifier != contextualize_mm_features([b], _context(0))[0].identifier
    assert ab[0].identifier != contextualize_mm_features([a], _context(1))[0].identifier
    assert (
        ab[0].identifier
        != contextualize_mm_features([replace(a, identifier="other-lora:A")], _context(0))[0].identifier
    )
    assert a.identifier == "lora:A"
    assert ab[0].mm_hash == a.mm_hash
    assert ab[0].data is a.data


@pytest.mark.parametrize("uses_mrope", [False, True])
def test_prefix_stripping_retains_rng_metadata_without_pixels(uses_mrope):
    features = [_feature("A"), _feature("B", 1)]
    request = OmniRequest("req", [1] * 20, SamplingParams(), None, mm_features=features)
    request.mm_rng_context = _context()
    request.num_computed_tokens = 4
    upstream = NewRequestData.from_request(request, ([0],), uses_mrope=uses_mrope)
    result = OmniNewRequestData.from_base(upstream, request)
    assert result.mm_rng_context == request.mm_rng_context
    assert set(result.mm_features[0].data) == set(request.mm_rng_context.retain_fields)
    assert result.mm_features[0].data["vae_token_grid_hw"] is features[0].data["vae_token_grid_hw"]
    assert result.mm_features[1].data is features[1].data
    assert "vae_pixel_values" in request.mm_features[0].data
    assert "vae_pixel_values" not in (upstream.mm_features[0].data or {})


def test_rng_context_wire_and_request_clone_round_trip():
    base = EngineCoreRequest(
        request_id="wire",
        prompt_token_ids=[1],
        mm_features=None,
        sampling_params=SamplingParams(),
        pooling_params=None,
        arrival_time=1.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    request = OmniEngineCoreRequest.from_request(base, mm_rng_context=_context())
    received = MsgpackDecoder(OmniEngineCoreRequest).decode(MsgpackEncoder().encode(request))
    assert received.mm_rng_context == _context()
    assert OmniEngineCoreRequest.from_request(received).mm_rng_context == _context()
    assert msgspec.msgpack.decode(msgspec.msgpack.encode(_context()), type=MMRNGContext) == _context()


@pytest.fixture
def real_processor(mocker):
    # Keep the production apply, hash, cache merge, field splitting and seed
    # injection. Only pixel preprocessing and prompt expansion are lightweight.
    model_config = mocker.Mock(spec=ModelConfig, model="tiny-hunyuan", max_model_len=20)
    mm_config = MultiModalConfig(mm_processor_cache_gb=0.01)
    model_config.multimodal_config = mm_config
    model_config.get_multimodal_config.return_value = mm_config
    info = HunyuanImage3ProcessingInfo(InputProcessingContext(model_config, tokenizer=None))
    parser = MultiModalDataParser()
    info.get_data_parser = mocker.Mock(return_value=parser)
    info.get_hf_config = mocker.Mock(return_value=SimpleNamespace(image_base_size=4))
    image_counts = []

    def process_pixels(prompt, images, **kwargs):
        image_counts.append(len(images))
        return BatchFeature({"vae_token_grid_hw": torch.tensor([[2, 3]] * len(images))}) if images else BatchFeature()

    info.get_image_processor = mocker.Mock(return_value=process_pixels)
    processor = object.__new__(HunyuanImage3MultiModalProcessor)
    processor.info = info
    processor.dummy_inputs = HunyuanImage3DummyInputsBuilder(info)
    processor._get_mm_prompt_updates = lambda items, *_: {
        "image": [[] for _ in range(items.get_count("image", strict=False))]
    }
    processor._maybe_apply_prompt_updates = lambda **_: (
        [99, 99],
        {"image": [PlaceholderFeaturesInfo("image", 0, 0, [99, 99], None)]},
    )
    return processor, image_counts


@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        ("dp_global", "out of range"),
        ("dp_local", "out of range"),
        ("lora", "LoRA is not enabled"),
        ("task", "does not support generation"),
    ],
)
def test_invalid_raw_request_does_not_poison_lru_for_next_request(invalid, message, real_processor, mocker):
    processor, image_counts = real_processor
    model_config = processor.info.ctx.model_config
    sender = LruKeyReplicatedSenderCache(model_config)
    receiver = LruKeyReplicatedReceiverCache(model_config)
    image = Image.new("RGB", (4, 4), "red")
    inputs = ProcessorInputs(
        [99],
        processor.info.get_data_parser().parse_mm_data({"image": image}),
        {"image": ["image-uuid"]},
        hf_processor_mm_kwargs={"vae_generator_seed": 0},
        cache=sender,
    )
    input_processor = _input_processor(None, mocker)
    input_processor._validate_params = InputProcessor._validate_params.__get__(input_processor)
    input_processor._validate_lora = InputProcessor._validate_lora.__get__(input_processor)
    input_processor.speculative_config = input_processor.structured_outputs_config = None
    input_processor.validate_logits_processors_params = mocker.Mock()
    input_processor.model_config.is_diffusion = input_processor.model_config.return_sampling_mask = False
    input_processor.model_config.get_vocab_size.return_value = 128
    input_processor.vllm_config._check_supports_watermarking.return_value = False
    input_processor.renderer.render_cmpl.side_effect = lambda *_: (
        processor.apply(inputs, TimingContext(enabled=False)),
    )
    raw = {
        "prompt": "image",
        "multi_modal_data": {"image": image},
        "mm_processor_kwargs": {"vae_generator_seed": 0},
    }
    kwargs = {}
    if invalid.startswith("dp_"):
        parallel = input_processor.vllm_config.parallel_config
        parallel.local_engines_only = invalid == "dp_local"
        parallel.data_parallel_size = 4 if parallel.local_engines_only else 1
        kwargs["data_parallel_rank"] = 1
    elif invalid == "lora":
        kwargs["lora_request"] = LoRARequest("adapter", 1, "/unused/adapter")
    tasks = ("embed",) if invalid == "task" else ("generate",)
    with pytest.raises(VLLMValidationError, match=message):
        input_processor.process_inputs("invalid", raw, SamplingParams(), tasks, **kwargs)
    input_processor.renderer.render_cmpl.assert_not_called()
    assert image_counts == []

    # A rejected request never reaches the receiver. Its sender entry must not
    # make this valid request omit the image and fail with a cache miss.
    request = input_processor.process_inputs("valid", raw, SamplingParams(), ("generate",))
    assert image_counts == [1]
    assert request.mm_rng_context == _context(0)
    assert request.mm_features[0].data is not None
    restored = receiver.get_and_update_features(request.mm_features)
    assert restored[0].data["vae_generator_seed"].data.item() == 0


@pytest.mark.parametrize("seed", [None, 0])
def test_real_processor_lru_and_uuid_only_cache_hits_keep_request_context(seed, real_processor, mocker):
    processor, image_counts = real_processor
    model_config = processor.info.ctx.model_config
    parser = processor.info.get_data_parser()
    cache = LruKeyReplicatedSenderCache(model_config)
    receiver = LruKeyReplicatedReceiverCache(model_config)
    kwargs = {} if seed is None else {"vae_generator_seed": seed}
    image = Image.new("RGB", (4, 4), "red")
    inputs = ProcessorInputs(
        [99],
        parser.parse_mm_data({"image": image}),
        {"image": ["image-uuid"]},
        hf_processor_mm_kwargs=kwargs,
        cache=cache,
    )
    first = processor.apply(inputs, TimingContext(enabled=False))
    second = processor.apply(
        replace(inputs, mm_data_items=parser.parse_mm_data({"image": [None]})), TimingContext(enabled=False)
    )
    assert image_counts == [1, 0]
    assert first["mm_hashes"] == second["mm_hashes"]
    assert second["mm_kwargs"]["image"][0] is None
    expected = None if seed is None else _context(seed)
    assert first.get(MM_RNG_CONTEXT_KEY) == second.get(MM_RNG_CONTEXT_KEY) == expected
    first_request = _input_processor(first, mocker).process_inputs("first", first, PoolingParams(), ("embed",))
    second_request = _input_processor(second, mocker).process_inputs("second", second, PoolingParams(), ("embed",))
    first_features = receiver.get_and_update_features(first_request.mm_features)
    second_features = receiver.get_and_update_features(second_request.mm_features)
    assert first_features[0].identifier == second_features[0].identifier
    assert first_features[0].mm_hash == first["mm_hashes"]["image"][0]
    assert second_features[0].data is first_features[0].data


def test_profile_dummy_inputs_do_not_inherit_configured_request_seed(real_processor):
    processor, image_counts = real_processor
    processor.info.ctx.model_config.multimodal_config.mm_processor_kwargs = {"vae_generator_seed": 42}
    result = processor.get_dummy_mm_inputs({"image": 1})
    assert image_counts == [1]
    assert MM_RNG_CONTEXT_KEY not in result
    assert "vae_generator_seed" not in result["mm_kwargs"]["image"][0]
