# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for stage-role vocabulary, component spec, and topology validation."""

from __future__ import annotations

from dataclasses import dataclass



@dataclass
class _Stage:
    """Duck-typed stand-in for StagePipelineConfig used by topology validation."""

    stage_id: int
    model_stage: str
    input_sources: tuple
    final_output: bool = False


# --- role vocabulary -------------------------------------------------------


def test_normalize_role(stage_roles):
    assert stage_roles.normalize_stage_role(None) == "diffusion"
    assert stage_roles.normalize_stage_role("") == "diffusion"
    assert stage_roles.normalize_stage_role("  ") == "diffusion"
    assert stage_roles.normalize_stage_role("encode") == "encode"
    assert stage_roles.normalize_stage_role(" Denoise ") == "Denoise"


def test_role_predicates(stage_roles):
    assert stage_roles.is_monolithic_role(None)
    assert stage_roles.is_monolithic_role("diffusion")
    assert not stage_roles.is_monolithic_role("encode")

    assert stage_roles.is_disaggregated_role("encode")
    assert stage_roles.is_disaggregated_role("denoise")
    assert stage_roles.is_disaggregated_role("decode")
    assert not stage_roles.is_disaggregated_role("diffusion")
    assert not stage_roles.is_disaggregated_role("custom_role")

    assert stage_roles.is_known_role("decode")
    assert not stage_roles.is_known_role("custom_role")


# --- component spec --------------------------------------------------------


def test_component_spec_enabled_and_requires(stage_roles):
    spec = stage_roles.StageComponentSpec(tokenizer=True, text_encoder=True, dit=False)
    assert spec.requires("tokenizer")
    assert not spec.requires("dit")
    assert not spec.requires("nonexistent")
    assert "tokenizer" in spec.enabled_components()
    assert "dit" not in spec.enabled_components()


def test_component_spec_union(stage_roles):
    a = stage_roles.StageComponentSpec(tokenizer=True, text_encoder=True)
    b = stage_roles.StageComponentSpec(dit=True, scheduler=True)
    u = a.union(b)
    assert u.tokenizer and u.text_encoder and u.dit and u.scheduler
    assert not u.vae_decoder


def test_all_components_spec(stage_roles):
    assert stage_roles.ALL_COMPONENTS.tokenizer
    assert stage_roles.ALL_COMPONENTS.vae_decoder
    assert stage_roles.ALL_COMPONENTS.dit
    assert len(stage_roles.ALL_COMPONENTS.enabled_components()) == 8


def test_component_spec_describe(stage_roles):
    assert stage_roles.StageComponentSpec().describe() == "none"
    assert "tokenizer" in stage_roles.StageComponentSpec(tokenizer=True).describe()


# --- topology validation ---------------------------------------------------


def _linear_topology():
    return [
        _Stage(0, "encode", ()),
        _Stage(1, "denoise", (0,)),
        _Stage(2, "decode", (1,), final_output=True),
    ]


def test_valid_linear_topology(stage_roles):
    assert stage_roles.validate_linear_diffusion_topology(_linear_topology()) == []


def test_empty_topology(stage_roles):
    errs = stage_roles.validate_linear_diffusion_topology([])
    assert errs and "no stages" in errs[0]


def test_duplicate_stage_ids(stage_roles):
    stages = [_Stage(0, "encode", ()), _Stage(0, "denoise", (0,), final_output=True)]
    errs = stage_roles.validate_linear_diffusion_topology(stages)
    assert any("Duplicate stage id" in e for e in errs)


def test_denoise_without_upstream(stage_roles):
    stages = [_Stage(1, "denoise", ()), _Stage(2, "decode", (1,), final_output=True)]
    errs = stage_roles.validate_linear_diffusion_topology(stages)
    assert any("Denoise stage 1 has no upstream source" in e for e in errs)


def test_decode_without_denoise_source(stage_roles):
    # decode points at an encode stage, not a denoise stage
    stages = [_Stage(0, "encode", ()), _Stage(2, "decode", (0,), final_output=True)]
    errs = stage_roles.validate_linear_diffusion_topology(stages)
    assert any("no upstream denoise source" in e for e in errs)


def test_nonexistent_input_source(stage_roles):
    stages = [
        _Stage(0, "encode", ()),
        _Stage(1, "denoise", (9,)),
        _Stage(2, "decode", (1,), final_output=True),
    ]
    errs = stage_roles.validate_linear_diffusion_topology(stages)
    assert any("non-existent input source 9" in e for e in errs)


def test_self_reference(stage_roles):
    stages = [
        _Stage(0, "encode", ()),
        _Stage(1, "denoise", (1,)),
        _Stage(2, "decode", (1,), final_output=True),
    ]
    errs = stage_roles.validate_linear_diffusion_topology(stages)
    assert any("references itself" in e for e in errs)


def test_multiple_final_outputs(stage_roles):
    stages = [
        _Stage(0, "encode", ()),
        _Stage(1, "denoise", (0,), final_output=True),
        _Stage(2, "decode", (1,), final_output=True),
    ]
    errs = stage_roles.validate_linear_diffusion_topology(stages)
    assert any("exactly one final_output" in e for e in errs)


def test_final_output_not_on_decode(stage_roles):
    stages = [
        _Stage(0, "encode", ()),
        _Stage(1, "denoise", (0,), final_output=True),
        _Stage(2, "decode", (1,)),
    ]
    errs = stage_roles.validate_linear_diffusion_topology(stages)
    assert any("final_output must be placed on the decode stage" in e for e in errs)


def test_denoise_multiple_upstream(stage_roles):
    stages = [
        _Stage(0, "encode", ()),
        _Stage(3, "encode", ()),
        _Stage(1, "denoise", (0, 3)),
        _Stage(2, "decode", (1,), final_output=True),
    ]
    errs = stage_roles.validate_linear_diffusion_topology(stages)
    assert any("upstream sources; the linear" in e for e in errs)


def test_monolithic_single_stage_valid(stage_roles):
    stages = [_Stage(0, "diffusion", (), final_output=True)]
    # single monolithic stage: no disaggregated adjacency requirements
    assert stage_roles.validate_linear_diffusion_topology(stages) == []


def test_custom_role_accepted(stage_roles):
    # unknown roles participate only in id/reference checks
    stages = [
        _Stage(0, "encode", ()),
        _Stage(1, "denoise", (0,)),
        _Stage(2, "decode", (1,)),
        _Stage(3, "kv_update", (1,), final_output=False),
        _Stage(4, "custom_final", (2,), final_output=True),
    ]
    errs = stage_roles.validate_linear_diffusion_topology(stages)
    # final on a custom (non-decode) stage while a decode stage exists -> error,
    # but the custom kv_update role itself must not produce a role error.
    assert not any("kv_update" in e for e in errs)
