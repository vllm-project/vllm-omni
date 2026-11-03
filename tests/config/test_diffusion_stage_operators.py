# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.config.stage_config import (
    DiffusionStageOperator,
    DiffusionStageRole,
    resolve_diffusion_stage_role,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize(
    ("role", "request_operators", "step_operators"),
    [
        (DiffusionStageRole.ENCODE, {"encode"}, set()),
        (DiffusionStageRole.DECODE, {"decode"}, {"post_decode"}),
        (
            DiffusionStageRole.DENOISE,
            {"diffuse"},
            {
                "prepare_denoise",
                "denoise_step",
                "step_scheduler",
                "step_output",
            },
        ),
        (
            DiffusionStageRole.DENOISE_DECODE,
            {"diffuse", "diffuse_decode", "decode"},
            {
                "prepare_denoise",
                "denoise_step",
                "step_scheduler",
                "step_output",
                "post_decode",
            },
        ),
        (
            DiffusionStageRole.FULL,
            {"forward", "encode", "diffuse", "diffuse_decode", "decode"},
            {
                "prepare_encode",
                "prepare_denoise",
                "denoise_step",
                "step_scheduler",
                "step_output",
                "post_decode",
            },
        ),
    ],
)
def test_role_authorizes_each_operation_independently(role, request_operators, step_operators):
    assert {operator.value for operator in role.allowed_operators} == request_operators | step_operators
    assert {operator.value for operator in role.request_operators} == request_operators
    assert {operator.value for operator in role.step_operators} == step_operators


@pytest.mark.parametrize(
    ("role", "default_operator"),
    [
        (DiffusionStageRole.ENCODE, DiffusionStageOperator.ENCODE),
        (DiffusionStageRole.DENOISE, DiffusionStageOperator.DIFFUSE),
        (DiffusionStageRole.DECODE, DiffusionStageOperator.DECODE),
        (DiffusionStageRole.DENOISE_DECODE, DiffusionStageOperator.DIFFUSE_DECODE),
        (DiffusionStageRole.FULL, DiffusionStageOperator.FORWARD),
    ],
)
def test_default_request_contract_does_not_replace_operator_permissions(role, default_operator):
    assert role.default_request_operator is default_operator
    assert role.required_operators(step_execution=False) == {default_operator}
    assert default_operator in role.request_operators


@pytest.mark.parametrize(
    ("role", "required_operators"),
    [
        (DiffusionStageRole.ENCODE, {"encode"}),
        (DiffusionStageRole.DECODE, {"decode"}),
        (
            DiffusionStageRole.DENOISE,
            {"prepare_denoise", "denoise_step", "step_scheduler", "step_output"},
        ),
        (
            DiffusionStageRole.DENOISE_DECODE,
            {"prepare_denoise", "denoise_step", "step_scheduler", "post_decode"},
        ),
        (
            DiffusionStageRole.FULL,
            {"prepare_encode", "denoise_step", "step_scheduler", "post_decode"},
        ),
    ],
)
def test_step_deployment_requires_only_its_default_preparation_and_terminal_operation(role, required_operators):
    required = role.required_operators(step_execution=True)

    assert {operator.value for operator in required} == required_operators
    assert required <= role.allowed_operators


def test_explicit_diffuse_has_the_same_permissions_on_generation_and_full_roles():
    diffuse = DiffusionStageOperator.DIFFUSE
    generate_decode = DiffusionStageRole.DENOISE_DECODE

    assert diffuse in generate_decode.allowed_operators
    assert diffuse in DiffusionStageRole.FULL.allowed_operators
    assert DiffusionStageOperator.FORWARD not in generate_decode.allowed_operators
    assert DiffusionStageOperator.PREPARE_ENCODE not in generate_decode.allowed_operators
    assert diffuse.required_stages < DiffusionStageOperator.DIFFUSE_DECODE.required_stages
    assert DiffusionStageOperator.DIFFUSE_DECODE.required_stages < DiffusionStageOperator.FORWARD.required_stages


@pytest.mark.parametrize("role", [DiffusionStageRole.FULL, DiffusionStageRole.DENOISE_DECODE])
def test_decode_capable_roles_may_export_latents_before_decoding(role):
    assert DiffusionStageOperator.STEP_OUTPUT in role.step_operators
    assert DiffusionStageOperator.POST_DECODE in role.step_operators
    assert DiffusionStageOperator.STEP_OUTPUT not in role.required_operators(step_execution=True)


@pytest.mark.parametrize(
    ("stage_role", "model_stage", "expected"),
    [
        (None, "dit", DiffusionStageRole.FULL),
        (None, "text_encode", DiffusionStageRole.ENCODE),
        ("vae_decode", None, DiffusionStageRole.DECODE),
        ("denoise_decode", "dit", DiffusionStageRole.DENOISE_DECODE),
        ("denoise", "text_encode", DiffusionStageRole.DENOISE),
        (None, "unregistered_legacy_stage", DiffusionStageRole.FULL),
    ],
)
def test_legacy_role_resolution_is_preserved(stage_role, model_stage, expected):
    assert resolve_diffusion_stage_role(stage_role, model_stage) is expected


@pytest.mark.parametrize("operator", [DiffusionStageOperator.STEP_PREPARE, DiffusionStageOperator.STEP_FINALIZE])
def test_boundary_selector_has_no_concrete_stage_permissions(operator):
    with pytest.raises(ValueError, match="Resolve boundary selector"):
        _ = operator.required_stages
    assert all(operator not in role.allowed_operators for role in DiffusionStageRole)


@pytest.mark.parametrize("role", list(DiffusionStageRole))
def test_role_enumeration_and_repr_do_not_read_boundary_permissions(monkeypatch, role):
    original = DiffusionStageOperator.required_stages.fget
    observed = set()

    def required_stages(operator):
        assert not operator.is_boundary_selector
        observed.add(operator)
        return original(operator)

    monkeypatch.setattr(DiffusionStageOperator, "required_stages", property(required_stages))
    assert role.required_operators(step_execution=False) <= role.allowed_operators
    assert role.required_operators(step_execution=True) <= role.allowed_operators
    assert role.step_operators <= role.allowed_operators
    for operator in DiffusionStageOperator:
        assert operator.value in repr(operator)
    assert observed == {operator for operator in DiffusionStageOperator if not operator.is_boundary_selector}
