# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
Test multimodal processing correctness for Omni models.

Tests that:
1. Cached vs non-cached processor outputs are identical.
2. Text prompt vs token prompt inputs produce identical results after processing.

Adapted from vllm/tests/models/multimodal/processing/test_common.py
"""

from functools import partial

import numpy as np
import pytest
from PIL import Image
from vllm.config.multimodal import (
    AudioDummyOptions,
    BaseDummyOptions,
    ImageDummyOptions,
    VideoDummyOptions,
)
from vllm.inputs import MultiModalDataDict, MultiModalInput
from vllm.multimodal.cache import MultiModalProcessorOnlyCache
from vllm.multimodal.inputs import batched_tensors_equal
from vllm.multimodal.processing import BaseMultiModalProcessor, InputProcessingContext
from vllm.tokenizers import TokenizerLike, cached_tokenizer_from_config

from tests.model_executor.helpers import bootstrap_vllm_layer_custom_op_modules
from tests.model_executor.models.registry import (
    _MULTIMODAL_OMNI_EXAMPLE_MODELS,
    _OmniExamplesInfo,
)
from vllm_omni.config import OmniModelConfig
from vllm_omni.model_executor.models.registry import OmniModelRegistry


def random_image(rng: np.random.RandomState, min_wh: int, max_wh: int):
    w, h = rng.randint(min_wh, max_wh, size=(2,))
    arr = rng.randint(0, 255, size=(w, h, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def random_video(
    rng: np.random.RandomState,
    min_frames: int,
    max_frames: int,
    min_wh: int,
    max_wh: int,
):
    num_frames = rng.randint(min_frames, max_frames)
    w, h = rng.randint(min_wh, max_wh, size=(2,))
    return rng.randint(0, 255, size=(num_frames, w, h, 3), dtype=np.uint8)


def random_audio(
    rng: np.random.RandomState,
    min_len: int,
    max_len: int,
    sr: int,
):
    audio_len = rng.randint(min_len, max_len)
    return rng.rand(audio_len), sr


_IGNORE_MM_KEYS: dict[str, set[str]] = {}


def _get_model_archs_to_test() -> list[str]:
    """Return architecture strings for parametrization."""
    return list(_MULTIMODAL_OMNI_EXAMPLE_MODELS.keys())


def get_text_token_prompts(
    processor: BaseMultiModalProcessor,
    mm_data: MultiModalDataDict,
):
    """Return ``(text_prompt, token_prompt)`` for the given multimodal data."""
    dummy_inputs = processor.dummy_inputs
    tokenizer: TokenizerLike = processor.info.get_tokenizer()
    model_config = processor.info.ctx.model_config

    parsed_data = processor.info.parse_mm_data(mm_data)
    mm_counts = {k: len(vs) for k, vs in parsed_data.items()}

    inputs = dummy_inputs.get_dummy_processor_inputs(
        model_config.max_model_len,
        mm_counts,
        mm_options={},
    )

    text_prompt: str | None
    token_prompt: list[int]
    if isinstance(inputs.prompt, list):
        text_prompt = None
        token_prompt = inputs.prompt
    elif isinstance(inputs.prompt, str):
        text_prompt = inputs.prompt
        token_prompt = tokenizer.encode(
            text_prompt,
            **processor.info.get_default_tok_params().get_encode_kwargs(),
        )
    else:
        raise TypeError(type(inputs.prompt))

    return text_prompt, token_prompt


def _build_model_config(model_arch: str, info: _OmniExamplesInfo) -> OmniModelConfig:
    """Create an ``OmniModelConfig`` suitable for processor testing."""
    kwargs: dict = dict(
        model=info.default,
        tokenizer=info.default,
        tokenizer_mode="auto",
        trust_remote_code=info.trust_remote_code,
        model_arch=model_arch,
        model_stage=info.model_stage,
        max_model_len=info.max_model_len,
        enforce_eager=True,
        dtype="auto",
    )
    if info.hf_config_name is not None:
        kwargs["hf_config_name"] = info.hf_config_name

    model_config = OmniModelConfig(**kwargs)
    # Ensure cache is effectively unlimited to prevent eviction during test
    model_config.multimodal_config.mm_processor_cache_gb = 2048
    return model_config


def _get_model_class_for_omni_processing(
    model_config: OmniModelConfig,
):
    """Resolve the multimodal model class for processor tests.

    vLLM's ``MULTIMODAL_REGISTRY._get_model_cls`` uses
    :func:`get_model_architecture`, which re-imports
    :mod:`vllm.model_executor.model_loader` and can re-register the same
    custom Torch / CustomOp names when tests run in isolation
    (``tests/model_executor/``).  Omni-registered classes are loaded
    unambiguously via :meth:`OmniModelRegistry._try_load_model_cls`.

    :func:`bootstrap_vllm_layer_custom_op_modules` also runs (see
    :file:`../helpers.py` and :file:`../conftest.py`) so vLLM layer modules are in :data:`sys.modules`
    before shims like ``qwen2_5_omni_thinker`` import
    :mod:`vllm.model_executor.models.qwen2_5_omni_thinker` (avoids duplicate
    ``@CustomOp.register`` e.g. ``fatrelu_and_mul``).
    """
    bootstrap_vllm_layer_custom_op_modules()
    if not model_config.model_arch:
        raise ValueError("OmniModelConfig.model_arch is required for processing tests")
    model_cls = OmniModelRegistry._try_load_model_cls(model_config.model_arch)
    if model_cls is None:
        raise RuntimeError(
            f"OmniModelRegistry has no class registered for {model_config.model_arch!r}; "
            "add it to vllm_omni.model_executor.models or fix the test matrix."
        )
    return model_cls


def _test_processing_correctness(
    model_arch: str,
    hit_rate: float,
    num_batches: int,
    simplify_rate: float,
):
    """Run the full processing-correctness test for one Omni model."""
    info = _MULTIMODAL_OMNI_EXAMPLE_MODELS[model_arch]
    info.check_transformers_version(on_fail="skip")
    info.check_available_online(on_fail="skip")

    model_config = _build_model_config(model_arch, info)

    model_cls = _get_model_class_for_omni_processing(model_config)
    factories = model_cls._processor_factory
    ctx = InputProcessingContext(
        model_config,
        tokenizer=cached_tokenizer_from_config(model_config),
    )
    cache = MultiModalProcessorOnlyCache(model_config)

    processing_info = factories.info(ctx)
    supported_mm_limits = processing_info.get_supported_mm_limits()
    limit_mm_per_prompt_ints = {
        modality: 3 if limit is None else limit for modality, limit in supported_mm_limits.items()
    }

    def _to_dummy_options(modality: str, count: int) -> BaseDummyOptions:
        if modality == "video":
            return VideoDummyOptions(count=count)
        if modality == "image":
            return ImageDummyOptions(count=count)
        if modality == "audio":
            return AudioDummyOptions(count=count)
        return BaseDummyOptions(count=count)

    model_config.get_multimodal_config().limit_per_prompt = {
        modality: _to_dummy_options(modality, count) for modality, count in limit_mm_per_prompt_ints.items()
    }

    baseline_processor = factories.build_processor(ctx, cache=None)
    cached_processor = factories.build_processor(ctx, cache=cache)

    rng = np.random.RandomState(0)

    input_to_hit = {
        "image": Image.new("RGB", size=(128, 128)),
        "video": np.zeros((4, 128, 128, 3), dtype=np.uint8),
        "audio": (np.zeros((512,)), 16000),
    }
    input_factory = {
        "image": partial(random_image, rng, min_wh=128, max_wh=256),
        "video": partial(random_video, rng, min_frames=2, max_frames=16, min_wh=128, max_wh=256),
        "audio": partial(random_audio, rng, min_len=512, max_len=1024, sr=16000),
    }

    for batch_idx in range(num_batches):
        mm_data = {
            k: [
                (input_to_hit[k] if rng.rand() < hit_rate else input_factory[k]())
                for _ in range(rng.randint(limit + 1))
            ]
            for k, limit in limit_mm_per_prompt_ints.items()
        }

        # Drop unnecessary keys and test single -> multi conversion
        if rng.rand() < simplify_rate:
            for k in list(mm_data.keys()):
                if not mm_data[k]:
                    del mm_data[k]
                elif len(mm_data[k]) == 1:
                    mm_data[k] = mm_data[k][0]

        _test_processing_correctness_one(
            model_config,
            mm_data,
            baseline_processor,
            cached_processor,
            batch_idx,
        )


def _test_processing_correctness_one(
    model_config: OmniModelConfig,
    mm_data: MultiModalDataDict,
    baseline_processor: BaseMultiModalProcessor,
    cached_processor: BaseMultiModalProcessor,
    batch_idx: int,
):
    model_type = model_config.hf_config.model_type

    text_prompt, token_prompt = get_text_token_prompts(baseline_processor, mm_data)
    mm_items = baseline_processor.info.parse_mm_data(mm_data)
    ignore_mm_keys = _IGNORE_MM_KEYS.get(model_type, set[str]())

    baseline_tokenized_result = baseline_processor(
        token_prompt,
        mm_items=mm_items,
        hf_processor_mm_kwargs={},
    )

    cached_tokenized_result = cached_processor(
        token_prompt,
        mm_items=mm_items,
        hf_processor_mm_kwargs={},
    )

    _assert_inputs_equal(
        baseline_tokenized_result,
        cached_tokenized_result,
        ignore_mm_keys=ignore_mm_keys,
        msg=f"Failed ({batch_idx=}, {token_prompt=}, {mm_data=})",
    )

    if text_prompt is not None:
        baseline_text_result = baseline_processor(
            text_prompt,
            mm_items=mm_items,
            hf_processor_mm_kwargs={},
        )
        cached_text_result = cached_processor(
            text_prompt,
            mm_items=mm_items,
            hf_processor_mm_kwargs={},
        )

        _assert_inputs_equal(
            baseline_text_result,
            cached_text_result,
            ignore_mm_keys=ignore_mm_keys,
            msg=f"Failed ({batch_idx=}, {text_prompt=}, {mm_data=})",
        )

        _assert_inputs_equal(
            baseline_text_result,
            baseline_tokenized_result,
            ignore_mm_keys=ignore_mm_keys,
            msg=f"Failed ({batch_idx=}, {text_prompt=}, {token_prompt=}, {mm_data=})",
        )

        _assert_inputs_equal(
            cached_text_result,
            cached_tokenized_result,
            ignore_mm_keys=ignore_mm_keys,
            msg=f"Failed ({batch_idx=}, {text_prompt=}, {token_prompt=}, {mm_data=})",
        )


@pytest.mark.core_model
@pytest.mark.omni
@pytest.mark.cpu
@pytest.mark.parametrize("model_arch", _get_model_archs_to_test())
@pytest.mark.parametrize("hit_rate", [0.3, 0.5, 1.0])
@pytest.mark.parametrize("num_batches", [32])
@pytest.mark.parametrize("simplify_rate", [1.0])
def test_omni_processing_correctness(
    model_arch: str,
    hit_rate: float,
    num_batches: int,
    simplify_rate: float,
):
    """
    For each Omni model verify that:
    - cached and non-cached processor outputs are identical.
    - text-prompt and token-prompt inputs yield the same result.
    """
    _test_processing_correctness(
        model_arch,
        hit_rate=hit_rate,
        num_batches=num_batches,
        simplify_rate=simplify_rate,
    )


def _assert_inputs_equal(
    a: MultiModalInput,
    b: MultiModalInput,
    *,
    ignore_mm_keys: set[str] | None = None,
    msg: str = "",
):
    if ignore_mm_keys is None:
        ignore_mm_keys = set()

    ignore_prompt_keys = ("prompt", "mm_kwargs")
    a_rest = {k: v for k, v in a.items() if k not in ignore_prompt_keys}
    b_rest = {k: v for k, v in b.items() if k not in ignore_prompt_keys}

    assert a_rest == b_rest, msg

    a_data = a["mm_kwargs"].get_data()
    b_data = b["mm_kwargs"].get_data()

    for key in ignore_mm_keys:
        a_data.pop(key, None)
        b_data.pop(key, None)

    assert batched_tensors_equal(a_data, b_data), msg
