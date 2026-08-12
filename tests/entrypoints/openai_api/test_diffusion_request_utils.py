# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest

from vllm_omni.entrypoints.openai.diffusion_request_utils import (
    DiffusionRequestOptionSpec,
    apply_normalized_diffusion_request_extra_args,
    normalize_diffusion_request_args,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

REQUEST_SOURCES = (
    "root.declared",
    "root.extra_args",
    "root.extra_params",
    "nested.declared",
    "nested.extra_args",
    "nested.extra_params",
)


def _source_args(source: str, value: object) -> dict[str, object]:
    scope, container = source.split(".")
    payload = {"solver": value} if container == "declared" else {container: {"solver": value}}
    kwargs: dict[str, object] = {scope: payload}
    registered = {"solver"} if container == "declared" else set()
    kwargs["request_options"] = DiffusionRequestOptionSpec.from_fields(
        serving_root_fields=set(),
        registered_extra_fields=registered,
        root_field_aliases=None,
    )
    return kwargs


@pytest.mark.parametrize("source", REQUEST_SOURCES)
@pytest.mark.parametrize(("value", "expected"), [("ddim", "ddim"), (None, "euler")])
def test_normalizer_routes_values_without_erasing_defaults(
    source: str,
    value: object,
    expected: str,
) -> None:
    normalized, _ = normalize_diffusion_request_args(**_source_args(source, value))
    sampling_params = OmniDiffusionSamplingParams(
        extra_args={"solver": "euler", "stage_default": True},
    )
    apply_normalized_diffusion_request_extra_args(sampling_params, normalized)

    assert sampling_params.extra_args == {"solver": expected, "stage_default": True}


def test_non_overlapping_legacy_and_canonical_values_are_merged(mocker) -> None:
    warning_once = mocker.patch("vllm_omni.entrypoints.openai.diffusion_request_utils.logger.warning_once")

    normalized, _ = normalize_diffusion_request_args(
        root={"extra_args": {"cfg_text_scale": 7.0}},
        nested={"extra_params": {"sample_solver": "euler"}},
        request_options=DiffusionRequestOptionSpec(),
    )

    assert normalized == {"cfg_text_scale": 7.0, "sample_solver": "euler"}
    warning_once.assert_called_once()


@pytest.mark.parametrize(
    "kwargs, spec_fields, expected_sources",
    [
        (
            {
                "root": {"seed": 1, "extra_args": {"seed": 2}},
            },
            {"serving_root_fields": {"seed"}},
            "request.seed, request.extra_args.seed",
        ),
        (
            {
                "root": {"cfg_scale": 7.0},
                "nested": {"true_cfg_scale": 2.0},
            },
            {
                "serving_root_fields": {"true_cfg_scale"},
                "root_field_aliases": {"cfg_scale": "true_cfg_scale"},
            },
            "request.cfg_scale, request.extra_body.true_cfg_scale",
        ),
        (
            {
                "root": {
                    "extra_args": {"solver": "euler"},
                    "extra_params": {"solver": "ddim"},
                }
            },
            {},
            "request.extra_args.solver, request.extra_params.solver",
        ),
        (
            {
                "root": {"cfg_text_scale": 7.0},
                "nested": {"extra_args": {"cfg_text_scale": 8.0}},
            },
            {"registered_extra_fields": {"cfg_text_scale"}},
            "request.cfg_text_scale, request.extra_body.extra_args.cfg_text_scale",
        ),
    ],
)
def test_duplicate_sources_are_rejected(
    kwargs: dict[str, object],
    spec_fields: dict[str, object],
    expected_sources: str,
    mocker,
) -> None:
    warning_once = mocker.patch("vllm_omni.entrypoints.openai.diffusion_request_utils.logger.warning_once")
    request_options = DiffusionRequestOptionSpec.from_fields(
        serving_root_fields=spec_fields.get("serving_root_fields", set()),
        registered_extra_fields=spec_fields.get("registered_extra_fields", set()),
        root_field_aliases=spec_fields.get("root_field_aliases"),
    )

    with pytest.raises(ValueError, match=expected_sources):
        normalize_diffusion_request_args(request_options=request_options, **kwargs)

    warning_once.assert_not_called()


@pytest.mark.parametrize(
    ("scope", "name"),
    (
        ("root", "extra_args"),
        ("root", "extra_params"),
        ("nested", "extra_args"),
        ("nested", "extra_params"),
    ),
)
def test_public_containers_must_be_objects(scope: str, name: str) -> None:
    with pytest.raises(ValueError, match="must be a JSON object"):
        normalize_diffusion_request_args(
            **{scope: {name: ["not", "an", "object"]}},
            request_options=DiffusionRequestOptionSpec(),
        )


def test_container_keys_must_be_strings() -> None:
    with pytest.raises(ValueError, match="must be a JSON object with string keys"):
        normalize_diffusion_request_args(
            root={"extra_args": {1: "invalid"}},
            request_options=DiffusionRequestOptionSpec(),
        )


@pytest.mark.parametrize(
    "source, kwargs",
    [
        ("request", {"root": {"cfg_text_scale": 7.0, "solver": "euler"}}),
        ("request.extra_body", {"nested": {"cfg_text_scale": 7.0}}),
    ],
)
def test_unknown_free_form_fields_warn_and_are_not_routed(
    source: str,
    kwargs: dict[str, object],
    mocker,
) -> None:
    warning = mocker.patch("vllm_omni.entrypoints.openai.diffusion_request_utils.logger.warning")

    normalized_extra_args, request_args = normalize_diffusion_request_args(
        request_options=DiffusionRequestOptionSpec(),
        **kwargs,
    )

    assert "cfg_text_scale" not in normalized_extra_args
    assert "cfg_text_scale" not in request_args
    warning.assert_called_once()
    assert warning.call_args.args[1] == source


def test_registered_declared_fields_do_not_warn(mocker) -> None:
    warning = mocker.patch("vllm_omni.entrypoints.openai.diffusion_request_utils.logger.warning")
    request_options = DiffusionRequestOptionSpec.from_fields(
        serving_root_fields=set(),
        registered_extra_fields={"cfg_text_scale"},
        root_field_aliases=None,
    )

    normalized_extra_args, _ = normalize_diffusion_request_args(
        root={"cfg_text_scale": 7.0},
        request_options=request_options,
    )

    assert normalized_extra_args == {"cfg_text_scale": 7.0}
    warning.assert_not_called()
