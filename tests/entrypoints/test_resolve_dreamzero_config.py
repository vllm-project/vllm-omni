# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.config.resolver import resolve_omni_config
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.stage_diffusion_proc import StageDiffusionProc

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_dreamzero_resolves_through_registry_with_model_defaults(monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.config.config_factory.StageConfigFactory._try_infer_model_type",
        classmethod(lambda _cls, model, trust_remote_code=True: "vla"),
    )
    monkeypatch.setattr(
        "vllm_omni.config.config_factory.StageConfigFactory.get_hf_config",
        classmethod(lambda _cls, model, trust_remote_code=True: None),
    )
    monkeypatch.setattr(
        "vllm_omni.config.config_factory._looks_like_dreamzero",
        lambda _model: True,
    )

    resolved = resolve_omni_config(
        "GEAR-Dreams/DreamZero-DROID",
        trust_remote_code=False,
        deploy_config_path=None,
        cli_overrides=None,
        stage_overrides=None,
        strategy_config_path=None,
    )
    engine_args = resolved.stage_configs[0].engine_args

    assert resolved.config_path is not None
    assert resolved.config_path.endswith("vllm_omni/deploy/dreamzero.yaml")
    assert engine_args.model_class_name == "DreamZeroPipeline"
    assert engine_args.model_config.policy_server_config.action_space == "joint_position"


def test_dreamzero_enrich_config_preserves_explicit_model_class_name(monkeypatch):
    monkeypatch.setattr(
        "vllm.transformers_utils.config.get_hf_file_to_dict",
        lambda path, _model, **_kwargs: (
            None if path == "model_index.json" else {"model_type": "vla", "architectures": ["VLA"]}
        ),
    )

    od_config = OmniDiffusionConfig(
        model="GEAR-Dreams/DreamZero-DROID",
        model_class_name="DreamZeroPipeline",
    )
    proc = StageDiffusionProc(od_config.model, od_config)

    proc._enrich_config()

    assert od_config.model_class_name == "DreamZeroPipeline"
