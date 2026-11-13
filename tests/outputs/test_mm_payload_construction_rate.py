# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MultimodalPayload construction requires a rate on payloads that ship audio.

The construction-time half of the producer contract: a payload whose audio
samples will reach a consumer must carry a sample rate. Warns during the
transition release; zero-audio sparse steps stay legal without one."""

import warnings
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import torch

from vllm_omni.outputs.mm_outputs import MultimodalPayload
from vllm_omni.sample_rate import SAMPLE_RATE_KEY, MissingSampleRateAtConstructionWarning

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _fresh_warning_callsites(monkeypatch: pytest.MonkeyPatch) -> None:
    """The construction warning is deduplicated per callsite process-wide.

    Reset the dedup set so each test observes the warning independently.
    """
    from vllm_omni import sample_rate as sr_module

    monkeypatch.setattr(sr_module, "_warned_callsites", set())


@contextmanager
def _no_construction_warnings() -> Iterator[None]:
    warnings.simplefilter("error", MissingSampleRateAtConstructionWarning)
    try:
        yield
    finally:
        warnings.simplefilter("default", MissingSampleRateAtConstructionWarning)


def test_audio_samples_without_a_rate_warn_at_construction() -> None:
    with pytest.warns(MissingSampleRateAtConstructionWarning, match="must emit 'sr'"):
        MultimodalPayload.from_raw({"model_outputs": [torch.zeros(8)]}, "audio")


def test_bare_audio_tensor_without_a_rate_warns() -> None:
    with pytest.warns(MissingSampleRateAtConstructionWarning):
        MultimodalPayload.from_raw(torch.zeros(8), "audio")


def test_canonical_rate_silences_the_warning() -> None:
    with _no_construction_warnings():
        payload = MultimodalPayload.from_raw(
            {"model_outputs": [torch.zeros(8)], SAMPLE_RATE_KEY: [torch.tensor(24000)]}, "audio"
        )
    assert payload is not None
    assert [rate.item() for rate in payload[SAMPLE_RATE_KEY]] == [24000]


def test_legacy_rate_spelling_is_tolerated_at_construction() -> None:
    # Which spelling a payload uses is the reader's deprecation business;
    # construction only checks presence.
    with _no_construction_warnings():
        MultimodalPayload.from_raw({"model_outputs": [torch.zeros(8)], "audio_sample_rate": 24000}, "audio")


@pytest.mark.parametrize(
    "audio_value",
    [
        None,
        [],
        torch.zeros(0),
    ],
    ids=["none", "empty-list", "empty-tensor"],
)
def test_zero_audio_steps_stay_legal_without_a_rate(audio_value) -> None:
    with _no_construction_warnings():
        MultimodalPayload.from_raw({"model_outputs": audio_value}, "audio")


def test_empty_rate_list_with_samples_still_warns() -> None:
    # Samples shipping with a placeholder empty rate list are unlabeled.
    with pytest.warns(MissingSampleRateAtConstructionWarning):
        MultimodalPayload.from_raw({"model_outputs": [torch.zeros(8)], SAMPLE_RATE_KEY: []}, "audio")


def test_non_audio_modalities_never_warn() -> None:
    with _no_construction_warnings():
        MultimodalPayload.from_raw({"model_outputs": [torch.zeros(8)]}, "hidden")
        MultimodalPayload.from_raw({"model_outputs": [torch.zeros(8)]}, "text")
        MultimodalPayload.from_raw({"model_outputs": torch.zeros(8)}, "latent")


def test_the_construction_warning_is_deduplicated() -> None:
    with pytest.warns(MissingSampleRateAtConstructionWarning):
        MultimodalPayload.from_raw({"model_outputs": [torch.zeros(8)]}, "audio")
    # The second unlabeled payload in the same process must not warn again.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        MultimodalPayload.from_raw({"model_outputs": [torch.zeros(8)]}, "audio")
    assert not [w for w in caught if issubclass(w.category, MissingSampleRateAtConstructionWarning)]
