# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for vllm_omni.sample_rate: canonical key, strict payload reading,
deprecation warnings for legacy spellings, and the lenient find scan."""

import warnings
from types import SimpleNamespace, TracebackType

import pytest
import torch

from vllm_omni.outputs.mm_outputs import MultimodalPayload
from vllm_omni.sample_rate import (
    ALL_SAMPLE_RATE_KEYS,
    DEPRECATED_SAMPLE_RATE_KEYS,
    SAMPLE_RATE_KEY,
    SAMPLE_RATE_REMOVAL_VERSION,
    DeprecatedSampleRateKeyWarning,
    InvalidSampleRateError,
    MissingSampleRateError,
    coerce_sample_rate,
    find_sample_rate,
    get_sampling_rate,
    get_sampling_rate_or_none,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_canonical_key_contract() -> None:
    assert SAMPLE_RATE_KEY == "sr"
    assert SAMPLE_RATE_KEY not in DEPRECATED_SAMPLE_RATE_KEYS
    assert ALL_SAMPLE_RATE_KEYS[0] == SAMPLE_RATE_KEY
    assert SAMPLE_RATE_REMOVAL_VERSION == "0.36"


# ---------------------------------------------------------------------------
# get_sampling_rate: canonical key, all observed value shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (24000, 24000),
        (torch.tensor(24000, dtype=torch.int32), 24000),
        (torch.tensor(24000.0), 24000),
        ([torch.tensor(24000, dtype=torch.int32), torch.tensor(24000, dtype=torch.int32)], 24000),
        # Whole-batch emission shape (worker_v2/streaming_audio.py).
        (torch.tensor([24000, 24000], dtype=torch.int32), 24000),
        (24000.0, 24000),
    ],
)
def test_get_sampling_rate_canonical_shapes(value: object, expected: int) -> None:
    assert get_sampling_rate({SAMPLE_RATE_KEY: value}) == expected


# ---------------------------------------------------------------------------
# get_sampling_rate: rejection of invalid values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        True,  # bool must not pass as int 1
        "24k",
        0,
        -24000,
        24000.5,  # non-integral float
        torch.zeros(2, 2),  # conflicting elements
        torch.tensor([24000, 16000]),  # conflicting per-request rates
        [],  # empty list
    ],
)
def test_get_sampling_rate_rejects_invalid_values(value: object) -> None:
    with pytest.raises(InvalidSampleRateError):
        get_sampling_rate({SAMPLE_RATE_KEY: value})


def test_get_sampling_rate_rejects_non_mapping() -> None:
    with pytest.raises(TypeError, match="Mapping"):
        get_sampling_rate("not a payload")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# get_sampling_rate: missing key
# ---------------------------------------------------------------------------


def test_get_sampling_rate_missing_raises() -> None:
    with pytest.raises(MissingSampleRateError, match="must"):
        get_sampling_rate({"audio": torch.zeros(1, 100)})


def test_get_sampling_rate_or_none_missing_returns_none() -> None:
    assert get_sampling_rate_or_none({"audio": torch.zeros(1, 100)}) is None


def test_missing_error_lists_probed_keys() -> None:
    with pytest.raises(MissingSampleRateError, match="sr"):
        get_sampling_rate({})


def test_none_valued_canonical_key_falls_through_to_alias() -> None:
    # Legacy watermark bridge produced {"sr": None} when only an alias exists.
    with pytest.warns(DeprecatedSampleRateKeyWarning):
        assert get_sampling_rate({"sr": None, "audio_sample_rate": 44100}) == 44100


# ---------------------------------------------------------------------------
# get_sampling_rate: deprecated spellings warn and resolve
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", DEPRECATED_SAMPLE_RATE_KEYS)
def test_deprecated_flat_keys_resolve_with_warning(key: str) -> None:
    with pytest.warns(DeprecatedSampleRateKeyWarning, match=f"{key}.*{SAMPLE_RATE_REMOVAL_VERSION}"):
        assert get_sampling_rate({key: 22050}) == 22050


def test_nested_diffusion_metadata_resolves_with_warning() -> None:
    payload = {"audio": torch.zeros(1, 100), "metadata": {"audio": {"sample_rate": 44100}}}
    with pytest.warns(DeprecatedSampleRateKeyWarning, match="metadata.audio.sample_rate"):
        assert get_sampling_rate(payload) == 44100


def test_nested_diffusion_metadata_or_none() -> None:
    payload = {"metadata": {"audio": {"sample_rate": 32000}}}
    with pytest.warns(DeprecatedSampleRateKeyWarning):
        assert get_sampling_rate_or_none(payload) == 32000


def test_canonical_key_wins_without_warning() -> None:
    payload = {"sr": 24000, "sample_rate": 16000, "audio_sample_rate": 44100}
    with warnings_as_errors():
        assert get_sampling_rate(payload) == 24000


def test_unrelated_nested_metadata_is_missing() -> None:
    payload = {"metadata": {"video": {"fps": 16}}}
    assert get_sampling_rate_or_none(payload) is None


# ---------------------------------------------------------------------------
# Warning hygiene: dedup per callsite
# ---------------------------------------------------------------------------


def test_deprecation_warning_fires_once_per_callsite() -> None:
    payload = {"sample_rate": 24000}
    with pytest.warns(DeprecatedSampleRateKeyWarning) as record:
        for _ in range(50):
            assert get_sampling_rate(payload) == 24000
    # Streaming paths read many chunks; only the first read at a callsite warns.
    assert len(record) == 1
    assert SAMPLE_RATE_REMOVAL_VERSION in str(record[0].message)


def test_distinct_callsites_warn_independently(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni import sample_rate as sr_module

    monkeypatch.setattr(sr_module, "_warned_callsites", set())
    payload = {"sample_rate": 24000}
    with pytest.warns(DeprecatedSampleRateKeyWarning) as record:
        # Two distinct source lines = two callsites.
        assert get_sampling_rate(payload) == 24000
        assert get_sampling_rate(payload) == 24000
        # One more line, repeated in a loop, stays at one additional warning.
        for _ in range(10):
            assert get_sampling_rate_or_none(payload) == 24000
    assert len(record) == 3


# ---------------------------------------------------------------------------
# MultimodalPayload inputs (Mapping protocol; sr may sit in .tensors)
# ---------------------------------------------------------------------------


def test_multimodal_payload_with_sr_in_tensors() -> None:
    payload = MultimodalPayload.from_dict({"sr": torch.tensor(24000), "audio": torch.zeros(1, 100)})
    assert payload is not None
    with warnings_as_errors():
        assert get_sampling_rate(payload) == 24000


def test_multimodal_payload_with_sr_in_metadata() -> None:
    payload = MultimodalPayload(metadata={"sr": 24000})
    with warnings_as_errors():
        assert get_sampling_rate(payload) == 24000


def test_multimodal_payload_missing_rate() -> None:
    payload = MultimodalPayload.from_dict({"audio": torch.zeros(1, 100)})
    assert payload is not None
    assert get_sampling_rate_or_none(payload) is None


# ---------------------------------------------------------------------------
# find_sample_rate: lenient generic scan (metrics-compatible, no warnings)
# ---------------------------------------------------------------------------


def test_find_sample_rate_mapping() -> None:
    with warnings_as_errors():
        assert find_sample_rate({"sr": 24000}) == 24000
        assert find_sample_rate({"output_sample_rate": 44100}) == 44100


def test_find_sample_rate_config_object() -> None:
    config = SimpleNamespace(output_sample_rate=44100)
    with warnings_as_errors():
        assert find_sample_rate(config) == 44100


def test_find_sample_rate_sequence_first_hit_wins() -> None:
    with warnings_as_errors():
        assert find_sample_rate([{}, {"sr": 24000}, {"sr": 48000}]) == 24000


def test_find_sample_rate_absent() -> None:
    with warnings_as_errors():
        assert find_sample_rate({}) is None
        assert find_sample_rate({"fps": 16}) is None
        assert find_sample_rate(None) is None
        assert find_sample_rate(SimpleNamespace(other=1)) is None


def test_find_sample_rate_string_is_not_a_sequence() -> None:
    with warnings_as_errors():
        assert find_sample_rate("sr") is None


def test_find_sample_rate_lenient_coercion() -> None:
    # Lenient scan keeps the historical metrics behavior: unusable values
    # are skipped rather than raised, 0-d tensors unwrap, first list entry wins.
    with warnings_as_errors():
        assert find_sample_rate({"sr": [torch.tensor(22050), torch.tensor(9999)]}) == 22050
        assert find_sample_rate({"sr": "bad", "sample_rate": 16000}) == 16000
        assert find_sample_rate({"sr": 0}) is None


# ---------------------------------------------------------------------------
# coerce_sample_rate: lenient core
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (24000, 24000),
        (torch.tensor(24000), 24000),
        ([24000], 24000),
        ((torch.tensor(44100),), 44100),
        (None, None),
        (0, None),
        (-1, None),
        ("24k", None),
        (True, None),  # bool is rejected even in the lenient path
    ],
)
def test_coerce_sample_rate(value: object, expected: int | None) -> None:
    assert coerce_sample_rate(value) == expected


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class warnings_as_errors:
    """Turn only our deprecation warning into errors to assert no-warn paths."""

    def __enter__(self) -> "warnings_as_errors":
        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("error", DeprecatedSampleRateKeyWarning)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._ctx.__exit__(exc_type, exc_value, traceback)
