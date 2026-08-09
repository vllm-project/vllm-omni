# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import os
from collections.abc import Iterator

import pytest
import torch
import vllm.envs as envs
from pytest_mock import MockerFixture

from vllm_omni.diffusion.batch_invariance import (
    DIFFUSION_BATCH_INVARIANT_ENV,
    MAX_TORCH_MANUAL_SEED,
    MIN_TORCH_MANUAL_SEED,
    diffusion_batch_invariant_enabled,
    validate_batch_invariant_diffusion_request,
    validate_batch_invariant_diffusion_seed,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker import diffusion_worker as diffusion_worker_module
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _restore_native_batch_invariant_env() -> Iterator[None]:
    """Undo the bootstrap's process-level env write.

    _initialize_batch_invariance() sets VLLM_BATCH_INVARIANT so upstream's own
    re-check sees it. That is permanent by design in a worker process, but in a
    test process it would leak into later tests, where an unset diffusion switch
    follows the global one and silently arms the seed contract.
    """
    sentinel = object()
    before = os.environ.get("VLLM_BATCH_INVARIANT", sentinel)
    try:
        yield
    finally:
        if before is sentinel:
            os.environ.pop("VLLM_BATCH_INVARIANT", None)
        else:
            os.environ["VLLM_BATCH_INVARIANT"] = before
        # drop any value pinned into envs.__dict__ by monkeypatch.setattr, whose
        # undo writes the previously computed value back and defeats the lazy read
        envs.__dict__.pop("VLLM_BATCH_INVARIANT", None)


@pytest.mark.parametrize("seed", [MIN_TORCH_MANUAL_SEED, -2, 0, 42, MAX_TORCH_MANUAL_SEED])
def test_batch_invariant_mode_accepts_full_torch_seed_range(monkeypatch, seed):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)

    validate_batch_invariant_diffusion_seed(
        OmniDiffusionSamplingParams(seed=seed),
        request_id="request-test",
    )


@pytest.mark.parametrize(
    "seed",
    [None, True, False, 1.5, "1", MIN_TORCH_MANUAL_SEED - 1, MAX_TORCH_MANUAL_SEED + 1],
)
def test_batch_invariant_mode_rejects_missing_invalid_or_out_of_range_seed(monkeypatch, seed):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)

    with pytest.raises(ValueError, match="seed"):
        validate_batch_invariant_diffusion_seed(
            OmniDiffusionSamplingParams(seed=seed),
            request_id="request-test",
        )


@pytest.mark.parametrize("seed", [None, 7])
def test_batch_invariant_mode_rejects_generator_input(monkeypatch, seed):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    generator = torch.Generator(device="cpu").manual_seed(7)

    with pytest.raises(ValueError, match="does not accept generator"):
        validate_batch_invariant_diffusion_seed(
            OmniDiffusionSamplingParams(seed=seed, generator=generator),
            request_id="request-test",
        )


def test_feature_off_preserves_generator_and_missing_seed_compatibility(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)

    validate_batch_invariant_diffusion_seed(
        OmniDiffusionSamplingParams(generator=torch.Generator(device="cpu")),
        request_id="request-test",
    )
    validate_batch_invariant_diffusion_seed(
        OmniDiffusionSamplingParams(),
        request_id="request-test",
    )


def test_diffusion_switch_unset_follows_global_batch_invariant(monkeypatch):
    monkeypatch.delenv(DIFFUSION_BATCH_INVARIANT_ENV, raising=False)

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    assert diffusion_batch_invariant_enabled() is True

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    assert diffusion_batch_invariant_enabled() is False


@pytest.mark.parametrize("raw", ["1", "TRUE", " On "])
def test_diffusion_switch_enables_while_global_is_off(monkeypatch, raw):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setenv(DIFFUSION_BATCH_INVARIANT_ENV, raw)

    assert diffusion_batch_invariant_enabled() is True


@pytest.mark.parametrize("raw", ["0", "FALSE", " Off "])
def test_diffusion_switch_disables_while_global_is_on(monkeypatch, raw):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setenv(DIFFUSION_BATCH_INVARIANT_ENV, raw)

    assert diffusion_batch_invariant_enabled() is False


@pytest.mark.parametrize("raw", ["", "2"])
def test_diffusion_switch_rejects_unparsable_values(monkeypatch, raw):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setenv(DIFFUSION_BATCH_INVARIANT_ENV, raw)

    with pytest.raises(ValueError, match=DIFFUSION_BATCH_INVARIANT_ENV):
        diffusion_batch_invariant_enabled()


def test_seed_contract_follows_the_diffusion_switch_not_the_global_one(monkeypatch):
    """The switch must gate the seed validator itself, not just the helper."""
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setenv(DIFFUSION_BATCH_INVARIANT_ENV, "1")

    with pytest.raises(ValueError, match="seed"):
        validate_batch_invariant_diffusion_seed(
            OmniDiffusionSamplingParams(),
            request_id="request-test",
        )

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setenv(DIFFUSION_BATCH_INVARIANT_ENV, "0")
    validate_batch_invariant_diffusion_seed(
        OmniDiffusionSamplingParams(),
        request_id="request-test",
    )


def _request(**params) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompt={"prompt": "a cup of coffee on a table"},
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1, **params),
        request_id="request-test",
    )


def test_request_gate_rejects_implicit_seed(monkeypatch):
    """Construction must fail before __post_init__ hides the gap behind a random seed.

    The raise observed here comes from the seed validator, which runs first; the
    seed_was_explicit check is covered separately below.
    """
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)

    with pytest.raises(ValueError, match="requires an explicit integer seed"):
        _request()


def test_request_gate_rejects_a_seed_the_fallback_supplied(monkeypatch):
    """The seed_was_explicit check is what the seed validator alone cannot make.

    A request built while the switch was off keeps the fallback seed __post_init__
    assigned it, so by the time the validator is called directly the params look
    indistinguishable from an explicitly seeded request. Only the recorded flag
    separates them.
    """
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    request = _request()
    assert request.seed_was_explicit is False
    assert request.sampling_params.seed is not None

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    with pytest.raises(ValueError, match="seed must be explicit"):
        validate_batch_invariant_diffusion_request(request)


def test_request_gate_rejects_generator_device(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)

    with pytest.raises(ValueError, match="generator_device"):
        _request(seed=1234, generator_device="cpu")


@pytest.mark.parametrize(
    ("field_name", "value"),
    [("latents", torch.zeros(1)), ("sigmas", [1.0, 0.5])],
)
def test_request_gate_rejects_externally_supplied_rng_inputs(monkeypatch, field_name, value):
    """These bypass the seed: latents replace the initial noise, sigmas the schedule."""
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)

    with pytest.raises(ValueError, match=field_name):
        _request(seed=1234, **{field_name: value})


def test_request_gate_accepts_configurations_outside_the_evidence_table(monkeypatch):
    """Off-table recipes run rather than being rejected.

    Batch invariance is documented, not enforced: determinism holds for the operators
    vLLM replaces, and everything else is unverified rather than unsupported. This
    pins that behaviour so the configuration gate cannot be reintroduced silently.
    """
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)

    request = _request(
        seed=1234,
        height=1024,
        width=768,
        num_outputs_per_prompt=2,
        max_sequence_length=512,
        guidance_scale=7.5,
        output_type="pil",
    )

    assert request.seed_was_explicit is True
    assert request.sampling_params.height == 1024


def test_request_gate_is_a_noop_when_batch_invariance_is_disabled(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)

    validate_batch_invariant_diffusion_request(_request(generator_device="cpu", sigmas=[1.0]))


def test_worker_bootstrap_is_noop_when_batch_invariance_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: pytest.fail())

    diffusion_worker_module._initialize_batch_invariance(torch.device("cpu"))


@pytest.mark.parametrize(("device", "hip"), [("cpu", None), ("cuda", "6.0")])
def test_worker_bootstrap_skips_devices_without_operator_coverage_silently(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    device: str,
    hip: str | None,
) -> None:
    """Non-CUDA and ROCm/HIP both return before installing anything, and say nothing."""
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: pytest.fail())

    with caplog.at_level(logging.DEBUG):
        diffusion_worker_module._initialize_batch_invariance(torch.device(device, 0))

    assert caplog.records == []


@pytest.mark.parametrize("capability", [(7, 5), (8, 0), (12, 0)])
def test_worker_bootstrap_admits_every_cuda_capability(
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
    capability: tuple[int, int],
) -> None:
    """Unmeasured capability is unverified, not unsupported: run, do not raise.

    The capabilities here sit below, at and above upstream's SM80 override family and
    span the measured 8.9, so re-adding a bound in either direction fails this test
    rather than surfacing as a startup crash on a user's GPU. Determinism on such a
    device is scoped by docs/features/batch_invariance.md, not by a gate.
    """
    from vllm.model_executor.layers import batch_invariant

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    init_batch_invariance = mocker.patch.object(batch_invariant, "init_batch_invariance")

    diffusion_worker_module._initialize_batch_invariance(torch.device("cuda", 0))

    init_batch_invariance.assert_called_once_with()


def test_worker_bootstrap_aligns_native_env_for_diffusion_only_switch(
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """The diffusion-only switch must reach upstream, which re-reads the env itself.

    init_batch_invariance() opens with ``if envs.VLLM_BATCH_INVARIANT``, so without
    the alignment every gate passes and the op replacement is a silent no-op.
    """
    from vllm.model_executor.layers import batch_invariant

    monkeypatch.setenv(DIFFUSION_BATCH_INVARIANT_ENV, "1")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))

    # vllm.envs serves this name from a module-level __getattr__. An earlier
    # monkeypatch.setattr(envs, ...) in this file pins the computed value into
    # envs.__dict__ and its undo leaves it there, which would silently turn the
    # assertion below into a check against a frozen constant. Drop the pin and
    # assert the lazy path is live, so ordering damage fails loudly here.
    envs.__dict__.pop("VLLM_BATCH_INVARIANT", None)
    assert envs.VLLM_BATCH_INVARIANT is False, "lazy env read is not live"

    seen: list[bool] = []
    mocker.patch.object(
        batch_invariant,
        "init_batch_invariance",
        side_effect=lambda: seen.append(envs.VLLM_BATCH_INVARIANT),
    )

    diffusion_worker_module._initialize_batch_invariance(torch.device("cuda", 0))

    # upstream's own re-check has to observe an enabled switch at call time
    assert seen == [True]
    # bool(int(...)) is upstream's parser, so a non-numeric truthy value would raise
    assert int(os.environ["VLLM_BATCH_INVARIANT"]) == 1
