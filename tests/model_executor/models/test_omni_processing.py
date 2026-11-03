# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
Test multimodal processing correctness for Omni models.

Tests that:
1. Cached vs non-cached processor outputs are identical.
2. Text prompt vs token prompt inputs produce identical results after processing.

Adapted from vllm/tests/models/multimodal/processing/test_common.py
"""

import json
import os
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache, partial
from pathlib import Path

import numpy as np
import pytest
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from PIL import Image
from vllm.config.multimodal import (
    AudioDummyOptions,
    BaseDummyOptions,
    ImageDummyOptions,
    MultiModalConfig,
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
from vllm_omni.platforms import current_omni_platform
from vllm_omni.transformers_utils.repo_utils import hf_api

_AMD_TIMING_ENV = "VLLM_OMNI_AMD_PROCESSOR_TIMING"
_TIMING_PREFIX = "OMNI_PROCESSOR_PHASE "
_TIMING_PHASES = (
    "config_tokenizer_setup",
    "processor_construction",
    "prompt_input_preparation",
    "baseline_token_processing",
    "cached_token_processing",
    "baseline_text_processing",
    "cached_text_processing",
    "assertions",
)


class _PhaseTimer:
    def __init__(self, nodeid: str):
        self.enabled = os.getenv(_AMD_TIMING_ENV) == "1"
        self.nodeid = nodeid
        self.elapsed: defaultdict[str, float] = defaultdict(float)

    def _emit(self, payload: dict[str, object]) -> None:
        print(
            f"{_TIMING_PREFIX}{json.dumps(payload, sort_keys=True, separators=(',', ':'))}",
            flush=True,
        )

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        started = time.perf_counter()
        self._emit({"event": "start", "nodeid": self.nodeid, "phase": name})
        status = "error"
        try:
            yield
            status = "success"
        finally:
            seconds = time.perf_counter() - started
            self.elapsed[name] += seconds
            self._emit(
                {
                    "event": "end",
                    "nodeid": self.nodeid,
                    "phase": name,
                    "seconds": round(seconds, 9),
                    "status": status,
                }
            )

    def report(self) -> None:
        if not self.enabled:
            return

        for phase in _TIMING_PHASES:
            if phase not in self.elapsed:
                continue
            self._emit(
                {
                    "event": "summary",
                    "nodeid": self.nodeid,
                    "phase": phase,
                    "seconds": round(self.elapsed[phase], 9),
                }
            )


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

_PROCESSOR_METADATA_PATTERNS = ("*.json", "*.txt")


def _get_model_archs_to_test() -> list[str]:
    """Return architecture strings for parametrization."""
    return list(_MULTIMODAL_OMNI_EXAMPLE_MODELS.keys())


def _get_processor_snapshot_smoke_archs() -> list[str]:
    """Select one processor architecture for each backing model repository."""
    representatives: dict[str, str] = {}
    for model_arch, info in _MULTIMODAL_OMNI_EXAMPLE_MODELS.items():
        representatives.setdefault(info.default, model_arch)
    return list(representatives.values())


def get_text_token_prompts(
    processor: BaseMultiModalProcessor,
    mm_data: MultiModalDataDict,
):
    """Return ``(text_prompt, token_prompt)`` for the given multimodal data."""
    dummy_inputs = processor.dummy_inputs
    tokenizer: TokenizerLike = processor.info.get_tokenizer()

    parsed_data = processor.info.parse_mm_data(mm_data)
    mm_counts = {k: len(vs) for k, vs in parsed_data.items()}

    text_prompt = dummy_inputs.get_dummy_text(mm_counts)
    token_prompt = tokenizer.encode(
        text_prompt,
        **processor.info.get_default_tok_params().get_encode_kwargs(),
    )

    return text_prompt, token_prompt


@cache
def _resolve_processor_snapshot(model: str) -> str:
    """Resolve processor metadata once while excluding model weights."""
    snapshot_path = hf_api().snapshot_download(
        repo_id=model,
        allow_patterns=list(_PROCESSOR_METADATA_PATTERNS),
    )
    if not isinstance(snapshot_path, str):
        raise TypeError(f"Expected a snapshot path for {model}, got {snapshot_path!r}")
    return snapshot_path


@pytest.fixture
def clear_processor_snapshot_cache():
    _resolve_processor_snapshot.cache_clear()
    yield
    _resolve_processor_snapshot.cache_clear()


def _build_model_config(model_arch: str, info: _OmniExamplesInfo) -> OmniModelConfig:
    """Create an ``OmniModelConfig`` suitable for processor testing."""
    # Only ROCm CI needs metadata prefetch. Keep the original Hub-ID path on
    # other platforms so their processor lookup behavior remains covered.
    model_path = _resolve_processor_snapshot(info.default) if current_omni_platform.is_rocm() else info.default
    kwargs: dict = dict(
        model=model_path,
        tokenizer=model_path,
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
    timing: _PhaseTimer,
):
    """Run the full processing-correctness test for one Omni model."""
    info = _MULTIMODAL_OMNI_EXAMPLE_MODELS[model_arch]
    info.check_transformers_version(on_fail="skip")
    info.check_available_online(on_fail="skip")

    with timing.phase("config_tokenizer_setup"):
        model_config = _build_model_config(model_arch, info)

    with timing.phase("processor_construction"):
        model_cls = _get_model_class_for_omni_processing(model_config)
        factories = model_cls._processor_factory

    with timing.phase("config_tokenizer_setup"):
        ctx = InputProcessingContext(
            model_config,
            tokenizer=cached_tokenizer_from_config(model_config),
        )

    with timing.phase("processor_construction"):
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

        baseline_processor = factories.build_processor(ctx)
        cached_processor = factories.build_processor(ctx)

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
        with timing.phase("prompt_input_preparation"):
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
            cache,
            timing,
        )


def _test_processing_correctness_one(
    model_config: OmniModelConfig,
    mm_data: MultiModalDataDict,
    baseline_processor: BaseMultiModalProcessor,
    cached_processor: BaseMultiModalProcessor,
    batch_idx: int,
    cache: MultiModalProcessorOnlyCache,
    timing: _PhaseTimer,
):
    model_type = model_config.hf_config.model_type

    with timing.phase("prompt_input_preparation"):
        text_prompt, token_prompt = get_text_token_prompts(baseline_processor, mm_data)
        mm_items = baseline_processor.info.parse_mm_data(mm_data)
        ignore_mm_keys = _IGNORE_MM_KEYS.get(model_type, set[str]())

    with timing.phase("baseline_token_processing"):
        baseline_tokenized_result = baseline_processor(
            token_prompt,
            mm_items=mm_items,
            hf_processor_mm_kwargs={},
            cache=None,
        )

    with timing.phase("cached_token_processing"):
        cached_tokenized_result = cached_processor(
            token_prompt,
            mm_items=mm_items,
            hf_processor_mm_kwargs={},
            cache=cache,
        )

    with timing.phase("assertions"):
        _assert_inputs_equal(
            baseline_tokenized_result,
            cached_tokenized_result,
            ignore_mm_keys=ignore_mm_keys,
            msg=f"Failed ({batch_idx=}, {token_prompt=}, {mm_data=})",
        )

    if text_prompt is not None:
        with timing.phase("baseline_text_processing"):
            baseline_text_result = baseline_processor(
                text_prompt,
                mm_items=mm_items,
                hf_processor_mm_kwargs={},
                cache=None,
            )
        with timing.phase("cached_text_processing"):
            cached_text_result = cached_processor(
                text_prompt,
                mm_items=mm_items,
                hf_processor_mm_kwargs={},
                cache=cache,
            )

        with timing.phase("assertions"):
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
                msg=(f"Failed ({batch_idx=}, {text_prompt=}, {token_prompt=}, {mm_data=})"),
            )

            _assert_inputs_equal(
                cached_text_result,
                cached_tokenized_result,
                ignore_mm_keys=ignore_mm_keys,
                msg=(f"Failed ({batch_idx=}, {text_prompt=}, {token_prompt=}, {mm_data=})"),
            )


@pytest.mark.core_model
@pytest.mark.omni
@pytest.mark.cpu
@pytest.mark.parametrize("model_arch", _get_model_archs_to_test())
@pytest.mark.parametrize("hit_rate", [0.3, 0.5, 1.0])
@pytest.mark.parametrize("num_batches", [32])
@pytest.mark.parametrize("simplify_rate", [1.0])
def test_omni_processing_correctness(
    request: pytest.FixtureRequest,
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
    timing = _PhaseTimer(request.node.nodeid)
    try:
        _test_processing_correctness(
            model_arch,
            hit_rate=hit_rate,
            num_batches=num_batches,
            simplify_rate=simplify_rate,
            timing=timing,
        )
    finally:
        timing.report()


@pytest.mark.core_model
@pytest.mark.omni
@pytest.mark.cpu
@pytest.mark.parametrize(
    "model_arch",
    _get_processor_snapshot_smoke_archs(),
    ids=lambda model_arch: _MULTIMODAL_OMNI_EXAMPLE_MODELS[model_arch].default,
)
def test_rocm_processor_snapshot_smoke(
    model_arch: str,
    clear_processor_snapshot_cache,
):
    """Exercise one real local-metadata snapshot per Omni model repository."""
    if not current_omni_platform.is_rocm():
        pytest.skip("ROCm-only processor metadata integration smoke")

    info = _MULTIMODAL_OMNI_EXAMPLE_MODELS[model_arch]
    info.check_transformers_version(on_fail="skip")
    info.check_available_online(on_fail="skip")

    before = _resolve_processor_snapshot.cache_info()
    first = _build_model_config(model_arch, info)
    tokenizer = cached_tokenizer_from_config(first)
    after_first = _resolve_processor_snapshot.cache_info()

    assert tokenizer is not None
    assert first.model == first.tokenizer
    assert Path(first.model).is_dir()
    assert after_first.misses == before.misses + 1
    assert after_first.hits == before.hits

    second = _build_model_config(model_arch, info)
    after_second = _resolve_processor_snapshot.cache_info()

    assert second is not first
    assert second.model == first.model
    assert second.tokenizer == first.tokenizer
    assert after_second.misses == after_first.misses
    assert after_second.hits == after_first.hits + 1


@pytest.mark.core_model
@pytest.mark.omni
@pytest.mark.cpu
def test_phase_timer_disabled_has_no_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.delenv(_AMD_TIMING_ENV, raising=False)

    def fail_if_called() -> float:
        raise AssertionError("disabled timing must not read the clock")

    monkeypatch.setattr(time, "perf_counter", fail_if_called)
    timing = _PhaseTimer("test-node")
    with timing.phase("assertions"):
        pass
    timing.report()

    assert capsys.readouterr().out == ""


@pytest.mark.core_model
@pytest.mark.omni
@pytest.mark.cpu
def test_phase_timer_streams_ordered_flushed_json(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(_AMD_TIMING_ENV, "1")
    readings = iter((10.0, 10.125))
    monkeypatch.setattr(time, "perf_counter", lambda: next(readings))
    records: list[tuple[str, bool]] = []

    def record_print(value: str, *, flush: bool) -> None:
        records.append((value, flush))

    monkeypatch.setattr("builtins.print", record_print)
    timing = _PhaseTimer("test-node")
    with timing.phase("config_tokenizer_setup"):
        assert len(records) == 1
    timing.report()

    payloads = [json.loads(record.removeprefix(_TIMING_PREFIX)) for record, _ in records]
    assert payloads == [
        {
            "event": "start",
            "nodeid": "test-node",
            "phase": "config_tokenizer_setup",
        },
        {
            "event": "end",
            "nodeid": "test-node",
            "phase": "config_tokenizer_setup",
            "seconds": 0.125,
            "status": "success",
        },
        {
            "event": "summary",
            "nodeid": "test-node",
            "phase": "config_tokenizer_setup",
            "seconds": 0.125,
        },
    ]
    assert all(flush for _, flush in records)


@pytest.mark.core_model
@pytest.mark.omni
@pytest.mark.cpu
def test_phase_timer_streams_error_and_preserves_exception(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.setenv(_AMD_TIMING_ENV, "1")
    readings = iter((20.0, 20.5))
    monkeypatch.setattr(time, "perf_counter", lambda: next(readings))
    error = RuntimeError("sensitive exception value")

    timing = _PhaseTimer("test-node")
    with pytest.raises(RuntimeError) as exc_info:
        with timing.phase("assertions"):
            raise error

    assert exc_info.value is error
    payloads = [json.loads(line.removeprefix(_TIMING_PREFIX)) for line in capsys.readouterr().out.splitlines()]
    assert payloads[-1] == {
        "event": "end",
        "nodeid": "test-node",
        "phase": "assertions",
        "seconds": 0.5,
        "status": "error",
    }
    assert "sensitive exception value" not in json.dumps(payloads)


@pytest.mark.core_model
@pytest.mark.omni
@pytest.mark.cpu
def test_processor_snapshot_is_reused_with_fresh_model_configs(
    mocker,
    clear_processor_snapshot_cache,
):
    mocker.patch.object(current_omni_platform, "is_rocm", return_value=True)
    info = next(iter(_MULTIMODAL_OMNI_EXAMPLE_MODELS.values()))
    snapshot_download = mocker.patch.object(
        hf_api(),
        "snapshot_download",
        return_value="/cache/model",
    )
    configs = [mocker.Mock(spec=OmniModelConfig) for _ in range(2)]
    for config in configs:
        config.multimodal_config = MultiModalConfig(mm_processor_cache_gb=0)
    model_config_cls = mocker.patch(
        f"{__name__}.OmniModelConfig",
        side_effect=configs,
    )
    first = _build_model_config("arch", info)
    second = _build_model_config("arch", info)

    snapshot_download.assert_called_once_with(
        repo_id=info.default,
        allow_patterns=list(_PROCESSOR_METADATA_PATTERNS),
    )
    assert model_config_cls.call_count == 2
    assert first is not second
    assert first.multimodal_config is not second.multimodal_config
    assert model_config_cls.call_args_list[0].kwargs["model"] == "/cache/model"
    assert model_config_cls.call_args_list[0].kwargs["tokenizer"] == "/cache/model"

    first.multimodal_config.mm_processor_cache_gb = 1
    assert second.multimodal_config.mm_processor_cache_gb == 2048


@pytest.mark.core_model
@pytest.mark.omni
@pytest.mark.cpu
@pytest.mark.parametrize("model_arch", _get_model_archs_to_test())
def test_non_rocm_processor_config_keeps_original_hub_ids(mocker, model_arch):
    mocker.patch.object(current_omni_platform, "is_rocm", return_value=False)
    snapshot_download = mocker.patch.object(hf_api(), "snapshot_download")
    config = mocker.Mock(spec=OmniModelConfig)
    config.multimodal_config = MultiModalConfig(mm_processor_cache_gb=0)
    model_config_cls = mocker.patch(f"{__name__}.OmniModelConfig", return_value=config)
    info = _MULTIMODAL_OMNI_EXAMPLE_MODELS[model_arch]

    assert _build_model_config(model_arch, info) is config

    snapshot_download.assert_not_called()
    expected_kwargs = dict(
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
        expected_kwargs["hf_config_name"] = info.hf_config_name
    model_config_cls.assert_called_once_with(**expected_kwargs)
    assert config.multimodal_config.mm_processor_cache_gb == 2048


@pytest.mark.core_model
@pytest.mark.omni
@pytest.mark.cpu
@pytest.mark.parametrize("failure", ["rate-limit", "offline-cache-miss"])
def test_processor_snapshot_failures_are_not_swallowed_or_cached(
    mocker,
    clear_processor_snapshot_cache,
    failure,
):
    if failure == "rate-limit":
        error = HfHubHTTPError(
            "429 Too Many Requests",
            response=mocker.Mock(headers={}, request=mocker.Mock()),
        )
    else:
        error = LocalEntryNotFoundError("processor metadata is not cached")
    snapshot_download = mocker.patch.object(
        hf_api(),
        "snapshot_download",
        side_effect=error,
    )

    for _ in range(2):
        with pytest.raises(type(error)) as exc_info:
            _resolve_processor_snapshot("org/model")
        assert exc_info.value is error

    assert snapshot_download.call_count == 2


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
