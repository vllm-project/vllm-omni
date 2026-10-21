# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
Unit tests for StageConfigFactory and related classes.
"""

import importlib
from pathlib import Path
from unittest.mock import patch

import pytest
from transformers import PretrainedConfig, Qwen3OmniMoeConfig
from vllm.config import CUDAGraphMode

from tests.helpers.stage_config import get_deploy_config_path, get_deploy_config_stage
from vllm_omni.config import config_factory as config_factory_module
from vllm_omni.config.config_factory import StageConfigFactory, _materialize_object_storage_configs
from vllm_omni.config.endpoint_policy import EndpointRestriction, OmniServingCapability
from vllm_omni.config.omni_config import VllmOmniConfig
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES, register_pipeline, resolve_pipeline_config
from vllm_omni.config.stage_config import (
    _DEPLOY_DIR,
    DeployConfig,
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
    StageType,
    _apply_platform_overrides,
    _deep_merge_stage,
    _resolve_scheduler,
    build_stage_runtime_overrides,
    load_deploy_config,
    normalize_pipeline_cli_overrides,
    pipeline_cfg_resolver,
)
from vllm_omni.engine.arg_utils import SHARED_FIELDS, internal_blacklist_keys
from vllm_omni.engine.stage_init_utils import project_engine_args

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SINGLE_STAGE_PIPE_CFG = (StagePipelineConfig(stage_id=0, model_stage="a", final_output=True),)
# Validation strings to match against in post init
NO_STAGES_MATCH_STR = "no stages"
NO_TERMINAL_STAGE_MATCH_STR = "No terminal stage"


@pytest.fixture(autouse=True)
def _stable_test_platform(monkeypatch):
    from vllm_omni import platforms

    platform = platforms.current_omni_platform
    monkeypatch.setattr(platform, "device_name", "cpu", raising=False)
    monkeypatch.setattr(platform, "device_type", "cpu", raising=False)


@pytest.fixture(autouse=True)
def clear_config_factory_caches():
    """Clear cached classmethods from the StageConfigFactory to prevent test pollution."""
    yield
    StageConfigFactory.get_hf_config.cache_clear()
    StageConfigFactory.try_infer_model_type.cache_clear()
    _materialize_object_storage_configs.cache_clear()


Q3_OMNI_ALL_STAGES_HF_CONFIG = Qwen3OmniMoeConfig(enable_audio_output=True)
Q3_OMNI_THINKER_HF_CONFIG = Qwen3OmniMoeConfig(enable_audio_output=False)


class TestStageType:
    """Tests for StageType enum."""

    def test_stage_type_values(self):
        """Test StageType enum values."""
        assert StageType.LLM.value == "llm"
        assert StageType.DIFFUSION.value == "diffusion"

    def test_stage_type_from_string(self):
        """Test creating StageType from string."""
        assert StageType("llm") == StageType.LLM
        assert StageType("diffusion") == StageType.DIFFUSION


class TestStageResolutionHelpers:
    """Tests for shared stage override / filtering helpers."""

    def test_build_stage_runtime_overrides_ignores_other_stage_and_internal_keys(self):
        # Pass the same filter set the function uses by default
        # (orchestrator-only fields plus SHARED_FIELDS so ``model`` is
        # treated as not-per-stage-overridable).
        overrides = build_stage_runtime_overrides(
            0,
            {
                "gpu_memory_utilization": 0.5,
                "stage_0_gpu_memory_utilization": 0.9,
                "stage_1_gpu_memory_utilization": 0.1,
                "stage_0_model": "should_be_ignored",
                "parallel_config": {"world_size": 2},
            },
            internal_keys=internal_blacklist_keys() | SHARED_FIELDS,
        )

        assert overrides["gpu_memory_utilization"] == 0.9
        assert "model" not in overrides
        assert "parallel_config" not in overrides


class TestPipelineDiscovery:
    """Tests for the central pipeline registry (``OMNI_PIPELINES``)."""

    def test_registry_has_known_models(self):
        """Check that specific models are in OMNI_PIPELINES."""
        assert "qwen2_5_omni" in OMNI_PIPELINES
        assert "qwen3_omni_moe" in OMNI_PIPELINES
        assert "qwen3_omni_moe_thinker_only" in OMNI_PIPELINES
        assert "qwen3_tts" in OMNI_PIPELINES

    @pytest.mark.parametrize(
        "cuda,major,memory_gib,mps,optimized",
        [
            (False, None, 0, True, False),
            (True, None, 141, True, False),
            (True, 8, 141, True, False),
            (True, 9, 80, True, False),
            (True, 10, 180, True, False),
            (True, 9, 141, False, False),
            (True, 9, 141, True, True),
        ],
    )
    def test_cosyvoice3_device_default_and_explicit_overrides(
        self, monkeypatch, cuda, major, memory_gib, mps, optimized
    ):
        from vllm.platforms import current_platform
        from vllm.platforms.interface import DeviceCapability

        from vllm_omni.platforms import current_omni_platform

        monkeypatch.setattr(current_platform, "is_cuda", lambda: cuda)
        monkeypatch.setattr(current_omni_platform, "device_name", "cuda" if cuda else "cpu")
        capability = DeviceCapability(major, 0) if major is not None else None
        monkeypatch.setattr(current_platform, "get_device_capability", lambda: capability)
        monkeypatch.setattr(current_platform, "get_device_total_memory", lambda: memory_gib * 1024**3)
        monkeypatch.setattr(
            "vllm_omni.model_executor.models.cosyvoice3.pipeline.shutil.which",
            lambda command: "/usr/bin/nvidia-cuda-mps-control" if mps else None,
        )
        pipeline = resolve_pipeline_config("cosyvoice3")
        assert pipeline is not None
        config = VllmOmniConfig.from_pipeline_config(pipeline)
        talker, codec = config.stage_configs
        assert talker.scheduler_config.max_num_seqs == codec.scheduler_config.max_num_seqs == (32 if optimized else 8)
        assert talker.runtime_config.cuda_mps == codec.runtime_config.cuda_mps == optimized
        assert talker.model_config.hf_overrides == ({"cosyvoice3_sampling_mode": "standard"} if optimized else None)
        assert pipeline.stages[0].sampling_constraints["stop_token_ids"] == list(range(6561, 6761))
        if optimized:
            assert codec.runtime_config.env["COSYVOICE3_CACHED_ISTFT"] == "1"
            assert codec.runtime_config.env["COSYVOICE3_HIFT_GRAPH"] == "1"
            override = VllmOmniConfig.from_pipeline_config(
                pipeline, cli_overrides={"hf_overrides": {"cosyvoice3_sampling_mode": "ras", "seed": 7}}
            )
            assert override.stage_by_id(0).model_config.hf_overrides == {"cosyvoice3_sampling_mode": "ras", "seed": 7}
        explicit = VllmOmniConfig.from_pipeline_config(
            pipeline, deploy_config_path=get_deploy_config_path("cosyvoice3.yaml")
        )
        assert explicit.stage_by_id(0).scheduler_config.max_num_seqs == 8
        assert explicit.stage_by_id(0).model_config.hf_overrides is None

    def test_registry_resolver_qwen3_omni_all_stages(self):
        """Test that providing the HF config for qwen3 omni with audio enabled uses all stages."""
        pipeline = resolve_pipeline_config(
            "qwen3_omni_moe",
            Q3_OMNI_ALL_STAGES_HF_CONFIG,
        )
        assert isinstance(pipeline, PipelineConfig)
        assert pipeline.model_type == "qwen3_omni_moe"
        assert pipeline.default_deploy_config_name == "qwen3_omni_moe.yaml"
        assert len(pipeline.stages) == 3  # thinker + talker + code2wav

    def test_registry_resolver_qwen3_omni_thinker_only(self):
        """Test that providing the HF config for qwen3 omni without audio is thinker only."""
        pipeline = resolve_pipeline_config(
            "qwen3_omni_moe",
            Q3_OMNI_THINKER_HF_CONFIG,
        )
        assert isinstance(pipeline, PipelineConfig)
        assert pipeline.model_type == "qwen3_omni_moe_thinker_only"
        assert pipeline.default_deploy_config_name is None
        assert len(pipeline.stages) == 1  # thinker only

    def test_registry_qwen3_omni_thinker_only_key_is_static(self):
        """Explicit registry key must not re-enter the HF-config resolver."""
        pipeline = resolve_pipeline_config(
            "qwen3_omni_moe_thinker_only",
            Q3_OMNI_ALL_STAGES_HF_CONFIG,
        )
        assert pipeline is OMNI_PIPELINES["qwen3_omni_moe_thinker_only"]
        assert pipeline.model_type == "qwen3_omni_moe_thinker_only"
        assert len(pipeline.stages) == 1
        assert pipeline.stages[0].engine_output_type == "text"
        assert pipeline.stages[0].final_output_type == "text"

    @pytest.mark.parametrize(
        "pipeline",
        [registered for registered in OMNI_PIPELINES.values() if isinstance(registered, PipelineConfig)],
        ids=lambda pipeline: pipeline.model_type,
    )
    def test_registered_default_deploy_config_exists(self, pipeline):
        if pipeline.default_deploy_config_name is None:
            return
        assert Path(get_deploy_config_path(pipeline.default_deploy_config_name)).is_file()

    @pytest.mark.parametrize(
        "deploy_path",
        sorted(_DEPLOY_DIR.glob("*.yaml")),
        ids=lambda deploy_path: deploy_path.name,
    )
    def test_explicit_deploy_pipeline_is_registered(self, deploy_path):
        deploy = load_deploy_config(deploy_path)
        if deploy.pipeline is not None:
            assert deploy.pipeline in OMNI_PIPELINES

    def test_registry_returns_none_for_unknown(self):
        """Unknown model_types aren't found and resolve to `None`."""
        assert "definitely_not_a_real_model" not in OMNI_PIPELINES
        assert OMNI_PIPELINES.get("definitely_not_a_real_model") is None
        assert resolve_pipeline_config("definitely_not_a_real_model") is None

    def test_pipeline_config_supports_hf_architectures(self):
        """PipelineConfig accepts hf_architectures for HF-arch fallback
        (replaces the old _ARCHITECTURE_MODELS dict)."""
        p = PipelineConfig(
            model_type="custom_collide",
            hf_architectures=("SomeCollidingArch",),
            stages=SINGLE_STAGE_PIPE_CFG,
        )
        assert p.hf_architectures == ("SomeCollidingArch",)


class TestHunyuanImage3DeployPipeline:
    def test_mooncake_profile_resolves_typed_topology_and_kv_transfer(self):
        pipeline = resolve_pipeline_config("hunyuan_image_3_moe")
        config = VllmOmniConfig.from_pipeline_config(
            pipeline,
            deploy_config_path=get_deploy_config_path("hunyuan-image-3-mooncakeconnector.yaml"),
        )
        ar, dit = config.stage_configs

        assert ar.stage_pipeline_config.execution_type == StageExecutionType.LLM_AR
        assert ar.is_comprehension is True
        assert ar.final_output is True
        assert ar.final_output_type == "text"
        assert dit.stage_pipeline_config.execution_type == StageExecutionType.DIFFUSION
        assert dit.input_sources == [0]
        assert dit.final_output is True
        assert dit.final_output_type == "image"

        for stage, devices, role, engine_id in (
            (ar, "0,1,2,3", "kv_producer", "hunyuan-image3-ar"),
            (dit, "4,5,6,7", "kv_consumer", "hunyuan-image3-dit"),
        ):
            assert stage.runtime_config.devices == devices
            assert stage.runtime_config.num_replicas == 2
            assert stage.parallel_config.tensor_parallel_size == 2
            kv_transfer = stage.connector_config.kv_transfer_config
            assert kv_transfer.kv_connector == "MooncakeConnector"
            assert kv_transfer.kv_role == role
            assert kv_transfer.engine_id == engine_id
            assert kv_transfer.kv_connector_extra_config["mooncake_protocol"] == "tcp"

        assert ar.connector_config.kv_transfer_config.kv_connector_extra_config["bootstrap_port"] == 8998
        assert dit.diffusion_config.kv_transfer_config.kv_connector_extra_config["transfer_timeout"] == 60.0
        assert dit.parallel_config.enable_expert_parallel is True


class TestCosmos3PolicyPipeline:
    """Cosmos3 policy serving resolves via deploy yaml selection (π0 precedent)."""

    def test_registered_without_capturing_base_cosmos3_checkpoints(self):
        assert "cosmos3_policy" in OMNI_PIPELINES
        # T2I/video Cosmos3 checkpoints report model_type=cosmos3_omni and the
        # same model_index.json _class_name as policy checkpoints; they must
        # keep resolving through the single-stage diffusion fallback, so the
        # policy pipeline must not be reachable by auto-detection.
        assert "cosmos3_omni" not in OMNI_PIPELINES
        pipeline = OMNI_PIPELINES["cosmos3_policy"]
        assert pipeline.hf_architectures == ()
        assert pipeline.diffusers_class_name is None


class TestCosmos3OmniDeployPipeline:
    """Omni deploy yaml selects an opt-in topology so --deploy-config applies (#6874)."""

    def test_registered_without_auto_capturing_cosmos3_omni(self):
        assert "cosmos3_omni_deploy" in OMNI_PIPELINES
        # Same shared HF metadata as policy: must not register cosmos3_omni for
        # auto-detect, or T2I/video/policy would collide.
        assert "cosmos3_omni" not in OMNI_PIPELINES
        pipeline = OMNI_PIPELINES["cosmos3_omni_deploy"]
        assert pipeline.hf_architectures == ()
        assert pipeline.diffusers_class_name is None
        # Align with CLI Cosmos3OmniDiffusersPipeline default (video), not T2I-only.
        assert pipeline.stages[0].final_output_type == "video"

    def test_deploy_yaml_applies_guardrails_false(self):
        deploy = load_deploy_config(get_deploy_config_path("cosmos3_omni.yaml"))
        assert deploy.pipeline == "cosmos3_omni_deploy"
        assert deploy.stages[0].devices is None

        stages = VllmOmniConfig.from_pipeline_config(
            OMNI_PIPELINES["cosmos3_omni_deploy"], user_deploy_config=deploy
        ).stage_configs
        assert len(stages) == 1
        stage = stages[0]

        assert stage.stage_type == "diffusion"
        assert stage.final_output_type == "video"
        assert stage.diffusion_config.model_class_name == "Cosmos3OmniDiffusersPipeline"
        assert stage.diffusion_config.model_config["guardrails"] is False
        assert stage.runtime_config.devices is None

    def test_deploy_yaml_accepts_documented_multi_gpu_cli(self):
        from vllm_omni.engine.stage_init_utils import _check_stage_device_layout, project_engine_args

        deploy = load_deploy_config(get_deploy_config_path("cosmos3_omni.yaml"))
        config = VllmOmniConfig.from_pipeline_config(
            OMNI_PIPELINES["cosmos3_omni_deploy"],
            cli_overrides={
                "cfg_parallel_size": 2,
                "use_hsdp": True,
                "hsdp_shard_size": 2,
            },
            user_deploy_config=deploy,
        )
        assert len(config.stage_configs) == 1
        stage = config.stage_by_id(0)
        assert stage.diffusion_config.model_config["guardrails"] is False
        assert stage.parallel_config.cfg_parallel_size == 2
        assert stage.parallel_config.use_hsdp is True
        assert stage.parallel_config.hsdp_shard_size == 2
        assert stage.runtime_config.devices is None
        _check_stage_device_layout(stage, project_engine_args(stage, "test-model"))

    def test_get_pipeline_config_honours_deploy_yaml_pipeline_key(self):
        deploy_path = get_deploy_config_path("cosmos3_omni.yaml")
        # No HF download: explicit pipeline: key is highest priority.
        pipeline = StageConfigFactory.get_pipeline_config(
            model="nvidia/Cosmos3-Super-Text2Image",
            trust_remote_code=True,
            deploy_config_path=deploy_path,
        )
        assert pipeline is not None
        assert pipeline.model_type == "cosmos3_omni_deploy"

    def test_without_deploy_yaml_stays_on_unregistered_fallback(self):
        # Shared HF metadata must not auto-select a pipeline; only the deploy
        # yaml ``pipeline:`` key selects cosmos3_omni_deploy and merges guardrails.
        assert "cosmos3_omni" not in OMNI_PIPELINES

        class FakeCosmos3Config(PretrainedConfig):
            model_type = "cosmos3_omni"

        fake_config = FakeCosmos3Config()
        fake_config.architectures = ["Cosmos3OmniDiffusersPipeline"]
        model = "nvidia/Cosmos3-Super-Text2Image"
        deploy_path = get_deploy_config_path("cosmos3_omni.yaml")

        StageConfigFactory.get_hf_config.cache_clear()
        StageConfigFactory.try_infer_model_type.cache_clear()
        with patch("vllm_omni.config.config_factory.get_config", return_value=fake_config):
            assert (
                StageConfigFactory.get_pipeline_config(
                    model=model,
                    trust_remote_code=True,
                    deploy_config_path=None,
                )
                is None
            )

            pipeline = StageConfigFactory.get_pipeline_config(
                model=model,
                trust_remote_code=True,
                deploy_config_path=deploy_path,
            )
        assert pipeline is not None
        assert pipeline.model_type == "cosmos3_omni_deploy"

        config = VllmOmniConfig.from_pipeline_config(pipeline, deploy_config_path=str(deploy_path))
        assert config.stage_by_id(0).diffusion_config.model_config["guardrails"] is False


class TestStagePipelineConfig:
    def test_frozen(self):
        s = StagePipelineConfig(stage_id=0, model_stage="a")
        with pytest.raises(AttributeError):
            s.model_stage = "changed"

    def test_defaults(self):
        s = StagePipelineConfig(stage_id=0, model_stage="a")
        assert s.execution_type == StageExecutionType.LLM_AR
        assert s.input_sources == ()
        assert s.final_output is False
        assert s.sampling_constraints == {}
        assert s.engine_output_type is None
        assert s.scheduler_cls is None


class TestPipelineConfigNew:
    def test_simple_init(self):
        # Ensure pipeline config validates in post init.
        PipelineConfig(
            model_type="t",
            model_arch="A",
            stages=SINGLE_STAGE_PIPE_CFG,
        )

    def test_frozen(self):
        p = PipelineConfig(
            model_type="t",
            model_arch="A",
            stages=SINGLE_STAGE_PIPE_CFG,
        )
        with pytest.raises(AttributeError):
            p.model_type = "changed"

    def test_validate_no_stages(self):
        with pytest.raises(ValueError, match=NO_STAGES_MATCH_STR):
            PipelineConfig(model_type="t", model_arch="A")

    def test_validate_no_terminal_stage(self):
        """A pipeline with no ``final_output`` stage can never emit a result."""
        with pytest.raises(ValueError, match=NO_TERMINAL_STAGE_MATCH_STR):
            PipelineConfig(
                model_type="t",
                model_arch="A",
                stages=(
                    StagePipelineConfig(stage_id=0, model_stage="a"),
                    StagePipelineConfig(stage_id=1, model_stage="b", input_sources=(0,)),
                ),
            )

    def test_pipeline_can_have_final_output_in_any_stage(self):
        """Ensure any stage may carry ``final_output``."""
        PipelineConfig(
            model_type="t",
            model_arch="A",
            stages=(
                StagePipelineConfig(stage_id=0, model_stage="a", final_output=True),
                StagePipelineConfig(stage_id=1, model_stage="b", input_sources=(0,)),
            ),
        )


class TestPipelineRegistration:
    def test_resolve_pipeline_prefers_deploy_pipeline_key(self, clean_pipeline_registry, tmp_path):
        deploy_key = "deploy_selected_pipeline"
        model_type_key = "hf_model_type_pipeline"
        deploy_pipe = PipelineConfig(
            model_type=deploy_key,
            stages=SINGLE_STAGE_PIPE_CFG,
        )
        model_type_pipe = PipelineConfig(
            model_type=model_type_key,
            stages=SINGLE_STAGE_PIPE_CFG,
        )
        register_pipeline(deploy_pipe)
        register_pipeline(model_type_pipe)

        deploy_path = tmp_path / "deploy.yaml"
        deploy_path.write_text(f"pipeline: {deploy_key}\n", encoding="utf-8")

        class FakeConfig(PretrainedConfig):
            model_type = model_type_key

        fake_config = FakeConfig()
        with patch("vllm_omni.config.config_factory.get_config", return_value=fake_config):
            pipeline_cfg = StageConfigFactory.get_pipeline_config(
                "fake/model",
                trust_remote_code=True,
                deploy_config_path=str(deploy_path),
            )

        assert pipeline_cfg is deploy_pipe

    def test_resolve_pipeline_matches_hf_architecture_fallback(self, clean_pipeline_registry):
        pipeline_key = "architecture_fallback_pipeline"
        pipe_cfg = PipelineConfig(
            model_type=pipeline_key,
            hf_architectures=("ArchitectureFallbackForTest",),
            stages=SINGLE_STAGE_PIPE_CFG,
        )
        register_pipeline(pipe_cfg)

        class FakeConfig(PretrainedConfig):
            model_type = "unregistered_model_type"

        fake_config = FakeConfig()
        fake_config.architectures = ["ArchitectureFallbackForTest"]
        with patch("vllm_omni.config.config_factory.get_config", return_value=fake_config):
            pipeline_cfg = StageConfigFactory.get_pipeline_config(
                "fake/model",
                trust_remote_code=True,
                deploy_config_path=None,
            )

        assert pipeline_cfg is pipe_cfg

    def test_resolve_pipeline_architecture_fallback_respects_predicates(self, clean_pipeline_registry):
        shared_arch = "SharedPredicateArchitectureForTest"

        def rejecting_predicate(_hf_config):
            return False

        def raising_predicate(_hf_config):
            raise ValueError("predicate failure")

        reject_cfg = PipelineConfig(
            model_type="rejected_by_predicate",
            hf_architectures=(shared_arch,),
            hf_config_predicate=rejecting_predicate,
            stages=SINGLE_STAGE_PIPE_CFG,
        )
        raises_cfg = PipelineConfig(
            model_type="raises_in_predicate",
            hf_architectures=(shared_arch,),
            hf_config_predicate=raising_predicate,
            stages=SINGLE_STAGE_PIPE_CFG,
        )
        accept_cfg = PipelineConfig(
            model_type="accepted_by_predicate",
            hf_architectures=(shared_arch,),
            stages=SINGLE_STAGE_PIPE_CFG,
        )
        register_pipeline(reject_cfg)
        register_pipeline(raises_cfg)
        register_pipeline(accept_cfg)

        class FakeConfig(PretrainedConfig):
            model_type = "unregistered_model_type"

        fake_config = FakeConfig()
        fake_config.architectures = [shared_arch]
        with patch("vllm_omni.config.config_factory.get_config", return_value=fake_config):
            pipeline_cfg = StageConfigFactory.get_pipeline_config(
                "fake/model",
                trust_remote_code=True,
                deploy_config_path=None,
            )

        assert pipeline_cfg is accept_cfg

    def test_resolve_pipeline_architecture_fallback_rejects_non_matching_architecture(
        self,
        clean_pipeline_registry,
    ):
        predicate_calls = []

        def predicate(_hf_config):
            predicate_calls.append(_hf_config)
            return True

        pipe_cfg = PipelineConfig(
            model_type="predicate_without_arch_match",
            hf_architectures=("DifferentArchitectureForTest",),
            hf_config_predicate=predicate,
            stages=SINGLE_STAGE_PIPE_CFG,
        )
        register_pipeline(pipe_cfg)

        class FakeConfig(PretrainedConfig):
            model_type = "unregistered_model_type"

        fake_config = FakeConfig()
        fake_config.architectures = ["RequestedArchitectureForTest"]
        with patch("vllm_omni.config.config_factory.get_config", return_value=fake_config):
            pipeline_cfg = StageConfigFactory.get_pipeline_config(
                "fake/model",
                trust_remote_code=True,
                deploy_config_path=None,
            )

        assert pipeline_cfg is None
        assert predicate_calls == [fake_config]

    def test_resolve_pipeline_architecture_fallback_resolves_callable_pipeline(self, clean_pipeline_registry):
        pipeline_key = "callable_architecture_fallback"
        resolved_cfg = PipelineConfig(
            model_type="callable_resolved_pipeline",
            hf_architectures=("CallableArchitectureForTest",),
            stages=SINGLE_STAGE_PIPE_CFG,
        )
        seen_hf_configs = []

        def resolver(hf_config):
            seen_hf_configs.append(hf_config)
            return resolved_cfg

        register_pipeline(resolver, model_type=pipeline_key)

        class FakeConfig(PretrainedConfig):
            model_type = "unregistered_model_type"

        fake_config = FakeConfig()
        fake_config.architectures = ["CallableArchitectureForTest"]
        with patch("vllm_omni.config.config_factory.get_config", return_value=fake_config):
            pipeline_cfg = StageConfigFactory.get_pipeline_config(
                "fake/model",
                trust_remote_code=True,
                deploy_config_path=None,
            )

        assert pipeline_cfg is resolved_cfg
        assert seen_hf_configs == [fake_config]

    def test_resolve_pipeline_returns_none_without_registry_or_architecture_match(self, clean_pipeline_registry):
        class FakeConfig(PretrainedConfig):
            model_type = "unregistered_model_type"

        fake_config = FakeConfig()
        fake_config.architectures = ["UnmatchedArchitectureForTest"]
        with patch("vllm_omni.config.config_factory.get_config", return_value=fake_config):
            pipeline_cfg = StageConfigFactory.get_pipeline_config(
                "fake/model",
                trust_remote_code=True,
                deploy_config_path=None,
            )

        assert pipeline_cfg is None

    def test_create_from_model_returns_structured_omni_config(self):
        class FakeConfig(PretrainedConfig):
            model_type = "qwen3_tts"

        with patch("vllm_omni.config.config_factory.get_config", return_value=FakeConfig()):
            omni_config = StageConfigFactory.create_from_model(
                "fake/model",
                trust_remote_code=False,
                cli_overrides={},
                deploy_config_path=None,
            )

        assert isinstance(omni_config, VllmOmniConfig)
        assert omni_config.pipeline_config is OMNI_PIPELINES["qwen3_tts"]
        assert len(omni_config.stage_configs) == 2

    def test_create_from_model_preserves_model_on_structured_diffusion_stage(self):
        class FakeConfig(PretrainedConfig):
            model_type = "dreamzero"

        with patch("vllm_omni.config.config_factory.get_config", return_value=FakeConfig()):
            omni_config = StageConfigFactory.create_from_model(
                "fake/model",
                trust_remote_code=False,
                cli_overrides={},
                deploy_config_path=None,
            )

        assert isinstance(omni_config, VllmOmniConfig)
        assert omni_config.stage_by_id(0).diffusion_config.model == "fake/model"

    def test_pipeline_registration(self, clean_pipeline_registry):
        """Ensure that we can register and create a custom pipeline config."""
        new_model_type = "new_model_type"
        pipe_cfg = PipelineConfig(
            model_type=new_model_type,
            stages=SINGLE_STAGE_PIPE_CFG,
        )

        # Register the new PipelineConfig
        assert new_model_type not in OMNI_PIPELINES
        register_pipeline(pipe_cfg)
        assert new_model_type in OMNI_PIPELINES
        assert OMNI_PIPELINES[new_model_type] is pipe_cfg

        class FakeConfig(PretrainedConfig):
            model_type = new_model_type

        # Create the model
        fake_config = FakeConfig()
        with (
            patch("vllm_omni.config.config_factory.get_config", return_value=fake_config),
            patch.object(VllmOmniConfig, "from_pipeline_config") as mock_create,
        ):
            StageConfigFactory.create_from_model(
                "fake/model",
                trust_remote_code=False,
                cli_overrides={},
                deploy_config_path=None,
            )
            mock_create.assert_called_once()
            assert mock_create.call_args.args == (pipe_cfg,)
            assert mock_create.call_args.kwargs["cli_overrides"] == {
                "trust_remote_code": False,
                "model": "fake/model",
            }
        assert pipe_cfg.model_type == new_model_type

    def test_resolver_registration(self, clean_pipeline_registry, mocker):
        """A registered HF-config resolver must construct the typed pipeline."""
        new_model_type = "new_model_type"
        resolved_type = "resolved_type"

        class FakeConfig(PretrainedConfig):
            model_type = new_model_type

        @pipeline_cfg_resolver(config_type=FakeConfig)
        def custom_resolver(hf_config: FakeConfig) -> PipelineConfig:
            return PipelineConfig(model_type=resolved_type, stages=SINGLE_STAGE_PIPE_CFG)

        assert new_model_type not in OMNI_PIPELINES
        register_pipeline(custom_resolver, model_type=new_model_type)
        assert OMNI_PIPELINES[new_model_type] is custom_resolver

        mock_get_config = mocker.patch("vllm_omni.config.config_factory.get_config", return_value=FakeConfig())
        mock_create = mocker.patch.object(VllmOmniConfig, "from_pipeline_config")
        StageConfigFactory.create_from_model(
            "fake/model",
            trust_remote_code=True,
            cli_overrides={},
            deploy_config_path=None,
        )
        mock_create.assert_called_once()
        assert mock_create.call_args.args[0].model_type == resolved_type
        assert mock_create.call_args.kwargs["cli_overrides"] == {
            "trust_remote_code": True,
            "model": "fake/model",
        }
        mock_get_config.assert_called_once_with("fake/model", trust_remote_code=True)

    def test_resolve_when_autodetect_resolves_none(self):
        """Regression test for: https://github.com/vllm-project/vllm-omni/issues/4726"""
        deploy_path = get_deploy_config_path("ming_tts.yaml")
        resolved_config = StageConfigFactory.create_from_model(
            model="inclusionAI/Ming-omni-tts-0.5B",
            trust_remote_code=False,
            cli_overrides={},
            deploy_config_path=deploy_path,
        )
        assert resolved_config is not None
        assert len(resolved_config.stage_configs) > 0

    def test_structured_path_loads_explicit_deploy_config_once(self, clean_pipeline_registry, tmp_path):
        pipeline_key = "single_load_pipeline"
        pipeline_cfg = PipelineConfig(
            model_type=pipeline_key,
            stages=SINGLE_STAGE_PIPE_CFG,
        )
        register_pipeline(pipeline_cfg)
        deploy_path = tmp_path / "single_load.yaml"
        deploy_path.write_text(f"pipeline: {pipeline_key}\n", encoding="utf-8")

        class FakeConfig(PretrainedConfig):
            model_type = "unregistered"

        with (
            patch("vllm_omni.config.config_factory.get_config", return_value=FakeConfig()),
            patch(
                "vllm_omni.config.config_factory.load_deploy_config",
                wraps=load_deploy_config,
            ) as mock_load,
        ):
            StageConfigFactory.create_from_model(
                "fake/model",
                trust_remote_code=False,
                cli_overrides={},
                deploy_config_path=str(deploy_path),
            )

        mock_load.assert_called_once_with(deploy_path)

    def test_deploy_override_uses_correct_endpoint_restrictions(self, clean_pipeline_registry, tmp_path):
        """Ensure endpoint restrictions must come from the final pipeline
        after deploy config overrides, not the auto-detected pipeline.
        """
        # Register two pipeline configs, where one has an endpoint restriction, and one doesn't
        restriction = EndpointRestriction(
            OmniServingCapability.COMPLETIONS,
            "pipeline_a blocks completions",
        )
        pipe_a = PipelineConfig(
            model_type="detect_type",
            endpoint_restrictions=(restriction,),
            stages=SINGLE_STAGE_PIPE_CFG,
        )
        pipe_b = PipelineConfig(
            model_type="override_type",
            endpoint_restrictions=(),
            stages=SINGLE_STAGE_PIPE_CFG,
        )
        register_pipeline(pipe_a)
        register_pipeline(pipe_b)

        # Create a config with the autodetected type, and write the
        # deploy config specifying the override type to a temp path
        class FakeConfig(PretrainedConfig):
            model_type = "detect_type"

        deploy_yaml = tmp_path / "override.yaml"
        deploy_yaml.write_text("pipeline: override_type\n")

        # Get the endpoint restrictions, passing the deploy config with the override
        # type + patching the config for the detected type. Ensure that the endpoint
        # restrictions correspond to the type in the deploy config.
        with patch(
            "vllm_omni.config.config_factory.get_config",
            return_value=FakeConfig(),
        ):
            restrictions = StageConfigFactory.get_pipeline_endpoint_restrictions(
                model="fake/model",
                trust_remote_code=False,
                deploy_config_path=str(deploy_yaml),
            )

        assert restrictions == ()


class TestResolveScheduler:
    def test_all_execution_types_handled(self):
        for et in StageExecutionType:
            _resolve_scheduler(et)

    def test_ar_sync_when_false(self):
        cls = _resolve_scheduler(StageExecutionType.LLM_AR, async_scheduling=False)
        assert cls is not None
        assert "Async" not in cls.__name__

    def test_ar_async_when_true(self):
        cls = _resolve_scheduler(StageExecutionType.LLM_AR, async_scheduling=True)
        assert cls is not None
        assert "Async" in cls.__name__

    def test_generation(self):
        cls = _resolve_scheduler(StageExecutionType.LLM_GENERATION)
        assert cls is not None
        assert "Generation" in cls.__name__

    def test_diffusion_returns_none(self):
        assert _resolve_scheduler(StageExecutionType.DIFFUSION) is None


class TestDeployConfigLoading:
    def test_rejects_legacy_stage_args_schema(self, tmp_path):
        deploy_path = tmp_path / "legacy.yaml"
        deploy_path.write_text("stage_args:\n  - stage_id: 0\n", encoding="utf-8")

        with pytest.raises(ValueError, match=r"stage_args.*PipelineConfig.*stages"):
            load_deploy_config(deploy_path)

    @pytest.mark.parametrize("platform", ["cuda", "npu"])
    def test_minicpmo_talker_multi_frame_is_npu_scoped(self, platform: str):
        """Stage 1's K-frame decode needs the NPU worker to inject constant
        continue drafts, so the deploy block that arms it must not sit in the
        platform-shared stage.

        On NPU the ``speculative_config`` is a fingerprint: the worker replays
        the decode graph and vLLM's n-gram proposer is never consulted. On CUDA
        the same block hands the Talker to the real proposer plus the rejection
        sampler, which asserts on its codec ids (device-side assert out of
        ``get_token_bin_counts_and_mask``).

        Stage 0's prefix caching is NPU-scoped for the same reason: a CUDA
        stage-0 prefix hit hands the Thinker an empty tts handoff, and the
        Talker then indexes past the codec vocab.
        """
        deploy = _apply_platform_overrides(
            load_deploy_config(Path(get_deploy_config_path("minicpmo_4_5.yaml"))),
            platform=platform,
        )
        # Check the deploy-side stop id before the merge: afterwards it is
        # unioned with the pipeline's codec-EOS default.
        deploy_stage1 = next(stage for stage in deploy.stages if stage.stage_id == 1)
        stages = VllmOmniConfig.from_pipeline_config(
            resolve_pipeline_config("minicpmo_4_5"), user_deploy_config=deploy
        ).stage_configs

        if platform == "npu":
            assert deploy_stage1.default_sampling_params["stop_token_ids"] == [1]
            # The armed head ends on 1; the union keeps the codec EOS inert
            # (the collapsed head cannot sample it).
            assert stages[1].model_config.default_sampling_params["stop_token_ids"] == [1, 6561]
            # The duplex KV-window contract requires prefix caching off, and the
            # NPU overlay states that explicitly so a stage-level value can
            # never turn it back on for this stage.
            assert stages[0].cache_config.enable_prefix_caching is not True
            assert stages[1].speculative_config == {
                "method": "ngram",
                "num_speculative_tokens": 7,
                "prompt_lookup_min": 1,
                "prompt_lookup_max": 1,
            }
            assert stages[1].scheduler_config.async_scheduling is False
            assert [stage.scheduler_config.max_num_seqs for stage in stages] == [8, 8, 8]
        else:
            assert "stop_token_ids" not in (deploy_stage1.default_sampling_params or {})
            # Without the block the head is the full codec vocabulary, so the
            # codec EOS is the only stop id -- the marker would end a request
            # on an ordinary codec id.
            assert stages[1].model_config.default_sampling_params["stop_token_ids"] == [6561]
            # Either absent (falls back to the top-level false) or explicitly
            # false; the regression this guards against is true.
            assert stages[0].cache_config.enable_prefix_caching is not True
            assert stages[1].speculative_config is None
            assert stages[1].scheduler_config.async_scheduling is True
            # Base concurrency stays 16; only the NPU overlay lowers it to 8,
            # so a CUDA deployment keeps the pipeline default.
            assert [stage.scheduler_config.max_num_seqs for stage in stages] == [16, 16, 16]

    def test_load_qwen3_omni_moe_deploy_config(self):
        deploy_path = Path(get_deploy_config_path("qwen3_omni_moe.yaml"))
        deploy = load_deploy_config(deploy_path)
        assert len(deploy.stages) == 3
        assert deploy.async_chunk is True
        assert deploy.connectors is not None
        assert deploy.platforms is not None

    def test_load_voxtral_tts_deploy_config_schema_fields(self):
        deploy_path = Path(get_deploy_config_path("voxtral_tts.yaml"))
        deploy = load_deploy_config(deploy_path)
        assert deploy.stages[0].config_format == "mistral"
        assert deploy.stages[0].load_format == "mistral"
        assert deploy.stages[0].tokenizer_mode == "mistral"
        assert not any(
            name in deploy.stages[0].engine_extras for name in ("config_format", "load_format", "tokenizer_mode")
        )

    def test_load_ming_flash_omni_deploy_config_schema_fields(self):
        deploy_path = Path(get_deploy_config_path("ming_flash_omni.yaml"))
        deploy = load_deploy_config(deploy_path)
        assert deploy.stages[0].compilation_config == {"pass_config": {"fuse_allreduce_rms": False}}
        assert "compilation_config" not in deploy.stages[0].engine_extras

    def test_load_voxcpm2_deploy_config_preserves_engine_extras(self):
        deploy_path = get_deploy_config_path("voxcpm2.yaml")
        raw_stage = get_deploy_config_stage("voxcpm2.yaml", 0)
        expected_runtime_config = raw_stage["engine_extras"]["hf_overrides"]["voxcpm2_runtime_config"]

        deploy = load_deploy_config(deploy_path)
        runtime_config = deploy.stages[0].engine_extras["hf_overrides"]["voxcpm2_runtime_config"]
        assert runtime_config == expected_runtime_config

    @pytest.mark.parametrize(
        "deploy_name",
        ["minimax_h3_disaggregated.yaml", "minimax_h3_disaggregated_turbo.yaml"],
    )
    @pytest.mark.parametrize("shared_memory", [False, True], ids=["nixl", "shared_memory"])
    def test_minimax_h3_deploy_connector_wiring_and_roles(self, deploy_name, shared_memory):
        from vllm_omni.config.stage_config import resolve_deploy_yaml
        from vllm_omni.distributed.omni_connectors.utils.initialization import load_omni_transfer_config
        from vllm_omni.engine.stage_init_utils import get_stage_connector_spec

        base_config = resolve_deploy_yaml(get_deploy_config_path(deploy_name))
        if not shared_memory:
            deploy_name = deploy_name.replace(".yaml", "_nixl.yaml")
        deploy_path = get_deploy_config_path(deploy_name)
        config = resolve_deploy_yaml(deploy_path)
        assert config["async_chunk"] is False
        connector_name = "shared_memory_connector" if shared_memory else "nixl_connector"
        assert connector_name in config["connectors"]
        assert config["stages"][0]["output_connectors"] == {"to_stage_1": connector_name}
        assert config["stages"][1]["input_connectors"] == {"from_stage_0": connector_name}
        load_deploy_config(deploy_path)
        for base_stage, stage in zip(base_config["stages"], config["stages"], strict=True):
            edge_keys = {"input_connectors", "output_connectors"}
            assert {key: value for key, value in stage.items() if key not in edge_keys} == {
                key: value for key, value in base_stage.items() if key not in edge_keys
            }
        transfer = load_omni_transfer_config(config_dict=config)
        assert transfer is not None
        assert set(transfer.connectors) == {("0", "1")}
        edge = transfer.connectors[("0", "1")]
        expected_name = "SharedMemoryConnector" if shared_memory else "NixlConnector"
        assert edge.name == expected_name
        assert "role" not in edge.extra
        for stage_id, role in ((0, "sender"), (1, "receiver")):
            spec = get_stage_connector_spec(transfer, stage_id, async_chunk=False)
            assert spec["name"] == expected_name
            assert spec["extra"]["role"] == role
            if shared_memory:
                assert "from_stage" not in spec["extra"]
                assert "to_stage" not in spec["extra"]
            else:
                assert spec["extra"]["from_stage"] == 0
                assert spec["extra"]["to_stage"] == 1
                assert spec["extra"]["host"] == "auto"
                assert spec["extra"]["zmq_port"] == 50071
                assert spec["extra"]["backends"] == ["UCX"]
        assert "role" not in edge.extra

    @pytest.mark.parametrize(
        "deploy_name",
        ["minimax_h3_disaggregated.yaml", "minimax_h3_disaggregated_turbo.yaml"],
    )
    def test_minimax_h3_builtin_stage_one_tp4_ulysses1_override(self, deploy_name):
        config = VllmOmniConfig.from_pipeline_config(
            OMNI_PIPELINES["minimax_h3_disaggregated"],
            deploy_config_path=get_deploy_config_path(deploy_name),
            cli_overrides={"stage_1_tensor_parallel_size": 4, "stage_1_ulysses_degree": 1},
        )
        encoder = config.stage_by_id(0)
        diffusion = config.stage_by_id(1)
        assert encoder.parallel_config.tensor_parallel_size == 2
        assert diffusion.parallel_config.tensor_parallel_size == 4
        assert diffusion.parallel_config.ulysses_degree == 1
        assert diffusion.parallel_config.vae_patch_parallel_size == 4
        assert tuple(diffusion.diffusion_config.stage_input_payload_keys) == ("encoder_output",)

    def test_minimax_h3_text_encoder_tp_alias_targets_stage_zero(self):
        pipeline = OMNI_PIPELINES["minimax_h3_disaggregated"]

        assert normalize_pipeline_cli_overrides(pipeline, {}) == {}
        assert normalize_pipeline_cli_overrides(pipeline, {"text_encoder_tp_size": 4}) == {
            "stage_0_tensor_parallel_size": 4
        }

    def test_minimax_h3_stage_zero_tp_override_wins_alias_conflict(self):
        pipeline = OMNI_PIPELINES["minimax_h3_disaggregated"]

        with pytest.warns(UserWarning, match="stage_0_tensor_parallel_size=2 takes precedence"):
            normalized = normalize_pipeline_cli_overrides(
                pipeline,
                {"text_encoder_tp_size": 4, "stage_0_tensor_parallel_size": 2},
            )

        assert normalized == {"stage_0_tensor_parallel_size": 2}

    def test_minimax_h3_rejects_text_encoder_tp_on_diffusion_stage(self):
        pipeline = OMNI_PIPELINES["minimax_h3_disaggregated"]

        with pytest.raises(ValueError, match="stage_1_text_encoder_tp_size cannot be set"):
            normalize_pipeline_cli_overrides(pipeline, {"stage_1_text_encoder_tp_size": 4})

    @pytest.mark.parametrize(
        ("config_json", "model_index", "expected_pipeline"),
        [
            ({"model_type": "step_audio_2"}, None, "step_audio_2"),
            (None, {"_class_name": "HunyuanVideo15Pipeline"}, "hunyuan_video_15"),
            (None, {"_class_name": "WanPipeline"}, "wan2_2_ti2v"),
            (None, {"_class_name": "WanDMDPipeline"}, "wan2_2_ti2v"),
        ],
    )
    def test_migrated_models_are_discovered_without_explicit_deploy(
        self,
        config_json,
        model_index,
        expected_pipeline,
    ):
        def get_model_file(filename, _model, revision=None):
            del revision
            if filename == "config.json":
                return config_json
            if filename == "model_index.json":
                return model_index
            return None

        with (
            patch.object(StageConfigFactory, "get_hf_config", return_value=None),
            patch("vllm_omni.config.config_factory.get_hf_file_to_dict", side_effect=get_model_file),
        ):
            pipeline = StageConfigFactory.get_pipeline_config(
                model="/models/unrelated-checkpoint-name",
                trust_remote_code=False,
            )

        assert pipeline is not None
        assert pipeline.model_type == expected_pipeline

    def test_minimax_h3_disaggregation_is_not_auto_discovered(self):
        def get_model_file(filename, _model, revision=None):
            del revision
            if filename == "config.json":
                return None
            if filename == "model_index.json":
                return {"_class_name": "MiniMaxH3Pipeline"}
            return None

        with (
            patch.object(StageConfigFactory, "get_hf_config", return_value=None),
            patch(
                "vllm_omni.config.config_factory.get_hf_file_to_dict",
                side_effect=get_model_file,
            ),
        ):
            pipeline = StageConfigFactory.get_pipeline_config(
                model="/models/unrelated-checkpoint-name",
                trust_remote_code=False,
            )

        assert pipeline is None

    @pytest.mark.parametrize("deploy_name", ["step_audio_2.yaml", "step_audio_2_async_chunk.yaml"])
    def test_step_audio2_deploy_configs_fit_two_gpus(self, deploy_name):
        deploy = load_deploy_config(Path(get_deploy_config_path(deploy_name)))

        assert deploy.stages[0].devices == "0,1"
        assert deploy.stages[0].tensor_parallel_size == 2
        assert deploy.stages[1].devices == "1"

    def test_no_bundled_legacy_stage_config_yamls(self):
        repo_root = Path(__file__).resolve().parents[2]
        stage_config_dir = repo_root / "vllm_omni" / "model_executor" / "stage_configs"
        assert not list(stage_config_dir.glob("*.yaml"))

    def test_mixed_schema_preserves_flat_fields(self):
        """Ensure flat fields are not dropped when engine_args are present."""
        fake_config = {
            "stages": [
                {
                    "stage_id": 0,
                    "gpu_memory_utilization": 0.7,
                    "max_num_seqs": 8,
                    "engine_args": {
                        "tensor_parallel_size": 4,
                        "enforce_eager": True,
                    },
                },
            ]
        }

        with patch("vllm_omni.config.stage_config.resolve_deploy_yaml", return_value=fake_config):
            deploy = load_deploy_config("dummy.yaml")

        assert len(deploy.stages) == 1
        stage = deploy.stages[0]
        # Check that the engine args are set
        assert stage.tensor_parallel_size == 4
        assert stage.enforce_eager is True
        # Check that the other settings are also preserved
        assert stage.gpu_memory_utilization == 0.7
        assert stage.max_num_seqs == 8

    def test_engine_parse_engine_fields(self):
        """Test that we correctly parse & recursively merge stage deploy fields."""
        fake_config = {
            "stages": [
                {
                    "stage_id": 0,
                    "compilation_config": {
                        "encoder_cudagraph_token_budgets": [1024, 2048],
                        "pass_config": {"fuse_norm_quant": True},
                    },
                    "engine_args": {
                        "compilation_config": {
                            "cudagraph_mm_encoder": True,
                            "pass_config": {"fuse_allreduce_rms": False},
                        },
                    },
                },
            ]
        }

        with patch("vllm_omni.config.stage_config.resolve_deploy_yaml", return_value=fake_config):
            deploy = load_deploy_config("dummy.yaml")

        assert len(deploy.stages) == 1
        stage = deploy.stages[0]
        assert stage.compilation_config == {
            "encoder_cudagraph_token_budgets": [1024, 2048],
            "pass_config": {
                "fuse_norm_quant": True,
                "fuse_allreduce_rms": False,
            },
            "cudagraph_mm_encoder": True,
        }

    def test_engine_extras_deep_merges_dicts_simple(self):
        """Ensure dictionary valued keys merge properly for top level dicts."""
        fake_config = {
            "stages": [
                {
                    "stage_id": 0,
                    "foo": {"a": 111, "b": 1},
                    "engine_args": {
                        "foo": {"b": 2, "c": 3},
                    },
                },
            ]
        }

        with patch("vllm_omni.config.stage_config.resolve_deploy_yaml", return_value=fake_config):
            deploy = load_deploy_config("dummy.yaml")

        assert len(deploy.stages) == 1
        stage = deploy.stages[0]
        assert "foo" in stage.engine_extras
        assert stage.engine_extras["foo"] == {"a": 111, "b": 2, "c": 3}

    def test_engine_extras_dict_type_mismatch(self):
        """Ensure that we handle type mismatches with nested dicts correctly."""
        fake_config = {
            "stages": [
                {
                    "stage_id": 0,
                    "foo": {"b": {1: 1}},
                    "engine_args": {
                        "foo": {"b": 2},
                    },
                },
            ]
        }

        with patch("vllm_omni.config.stage_config.resolve_deploy_yaml", return_value=fake_config):
            deploy = load_deploy_config("dummy.yaml")

        assert len(deploy.stages) == 1
        stage = deploy.stages[0]
        assert "foo" in stage.engine_extras
        assert stage.engine_extras["foo"] == {"b": 2}

    def test_mixed_engine_extras_deep_merges_dicts(self):
        """Ensure dictionary valued keys merge properly for nested dicts."""
        fake_config = {
            "stages": [
                {
                    "stage_id": 0,
                    "foo": {"a": 111, "b": {"e": 199}},
                    "engine_args": {
                        "foo": {"b": {"d": 9}, "c": 3},
                    },
                },
            ]
        }

        with patch("vllm_omni.config.stage_config.resolve_deploy_yaml", return_value=fake_config):
            deploy = load_deploy_config("dummy.yaml")

        assert len(deploy.stages) == 1
        stage = deploy.stages[0]
        assert "foo" in stage.engine_extras
        assert stage.engine_extras["foo"] == {"a": 111, "b": {"d": 9, "e": 199}, "c": 3}

    def test_explicit_engine_extras_merge_with_flat_and_engine_args(self):
        """Explicit engine_extras should be preserved with existing pass-through styles."""
        fake_config = {
            "stages": [
                {
                    "stage_id": 0,
                    "engine_extras": {
                        "hf_overrides": {
                            "runtime_config": {
                                "explicit_only": True,
                                "shared": "explicit",
                                "nested": {"a": 1},
                            }
                        }
                    },
                    "hf_overrides": {
                        "runtime_config": {
                            "flat_only": True,
                            "shared": "flat",
                            "nested": {"b": 2},
                        }
                    },
                    "engine_args": {
                        "hf_overrides": {
                            "runtime_config": {
                                "engine_args_only": True,
                                "shared": "engine_args",
                                "nested": {"c": 3},
                            }
                        }
                    },
                },
            ]
        }

        with patch("vllm_omni.config.stage_config.resolve_deploy_yaml", return_value=fake_config):
            deploy = load_deploy_config("dummy.yaml")

        assert deploy.stages[0].engine_extras["hf_overrides"] == {
            "runtime_config": {
                "engine_args_only": True,
                "explicit_only": True,
                "flat_only": True,
                "nested": {"a": 1, "b": 2, "c": 3},
                "shared": "engine_args",
            }
        }

    def test_deep_merge_does_not_mutate_inputs(self):
        """Merging engine_args must not mutate the base stage dict."""
        fake_config = {
            "stages": [
                {
                    "stage_id": 0,
                    "foo": {"a": 1, "b": {"x": 10, "y": 20}},
                    "engine_args": {
                        "foo": {"b": {"y": 99, "z": 30}, "c": 3},
                    },
                },
            ]
        }
        original_foo = fake_config["stages"][0]["foo"]
        assert isinstance(original_foo, dict)
        original_b = original_foo["b"]

        with patch("vllm_omni.config.stage_config.resolve_deploy_yaml", return_value=fake_config):
            deploy = load_deploy_config("dummy.yaml")

        # Values should merge into the new dict correctly
        assert deploy.stages[0].engine_extras["foo"] == {
            "a": 1,
            "b": {"x": 10, "y": 99, "z": 30},
            "c": 3,
        }
        # But the original nested dicts foo / b should be unchanged, since
        # we recursively shallow copy to avoid mutating in place.
        assert original_foo == {"a": 1, "b": {"x": 10, "y": 20}}
        assert original_b == {"x": 10, "y": 20}

    def test_mixed_schema_engine_args_wins_scalars(self):
        """engine_args takes precedence over flat fields for scalar conflicts."""
        fake_config = {
            "stages": [
                {
                    "stage_id": 0,
                    "gpu_memory_utilization": 0.7,
                    "engine_args": {
                        "gpu_memory_utilization": 0.5,
                    },
                },
            ]
        }

        with patch("vllm_omni.config.stage_config.resolve_deploy_yaml", return_value=fake_config):
            deploy = load_deploy_config("dummy.yaml")

        assert deploy.stages[0].gpu_memory_utilization == 0.5

    @pytest.mark.parametrize(
        ("yaml_val", "expected"),
        [
            ('"2"', 2),
            ('""', 0),
            ("null", 0),
        ],
    )
    def test_coerces_active_stream_window_to_int(self, tmp_path, yaml_val, expected):
        """Ensure active stream window coerces str to int."""
        deploy_path = tmp_path / "deploy.yaml"
        deploy_path.write_text(f"active_stream_window: {yaml_val}\n", encoding="utf-8")
        deploy = load_deploy_config(deploy_path)
        assert deploy.active_stream_window == expected


class TestQwen3OmniPipeline:
    def test_registered(self):
        p = resolve_pipeline_config(
            "qwen3_omni_moe",
            Q3_OMNI_ALL_STAGES_HF_CONFIG,
        )
        assert isinstance(p, PipelineConfig)
        assert p.model_arch == "Qwen3OmniMoeForConditionalGeneration"
        assert len(p.stages) == 3

    def test_thinker(self):
        p = resolve_pipeline_config(
            "qwen3_omni_moe",
            Q3_OMNI_ALL_STAGES_HF_CONFIG,
        )
        assert isinstance(p, PipelineConfig)

        s = p.get_stage(0)
        assert isinstance(s, StagePipelineConfig)
        assert s.model_stage == "thinker"
        assert s.execution_type == StageExecutionType.LLM_AR
        assert s.owns_tokenizer is True
        assert s.engine_output_type == "latent"
        assert s.sampling_constraints["detokenize"] is True

    def test_talker(self):
        p = resolve_pipeline_config(
            "qwen3_omni_moe",
            Q3_OMNI_ALL_STAGES_HF_CONFIG,
        )
        assert isinstance(p, PipelineConfig)

        s = p.get_stage(1)
        assert isinstance(s, StagePipelineConfig)
        assert s.input_sources == (0,)
        assert s.sampling_constraints["stop_token_ids"] == [2150]
        # thinker2talker was removed: sync_process_input_func always wins.
        assert s.custom_process_input_func is None
        assert s.sync_process_input_func is not None
        assert s.sync_process_input_func.endswith("thinker2talker_token_only")
        assert s.custom_process_next_stage_input_func is not None

    def test_code2wav(self):
        p = resolve_pipeline_config(
            "qwen3_omni_moe",
            Q3_OMNI_ALL_STAGES_HF_CONFIG,
        )
        assert isinstance(p, PipelineConfig)

        s = p.get_stage(2)
        assert isinstance(s, StagePipelineConfig)
        assert s.execution_type == StageExecutionType.LLM_GENERATION
        assert s.final_output_type == "audio"
        assert s.engine_output_type == "audio"


class TestQwen2_5OmniPipeline:
    def test_registered(self):
        p = resolve_pipeline_config("qwen2_5_omni")
        assert isinstance(p, PipelineConfig)
        assert p.model_arch == "Qwen2_5OmniForConditionalGeneration"
        assert len(p.stages) == 3

    def test_thinker(self):
        p = resolve_pipeline_config("qwen2_5_omni")
        assert isinstance(p, PipelineConfig)

        s = p.get_stage(0)
        assert isinstance(s, StagePipelineConfig)
        assert s.model_stage == "thinker"
        assert s.execution_type == StageExecutionType.LLM_AR
        assert s.owns_tokenizer is True
        assert s.engine_output_type == "latent"
        assert s.requires_multimodal_data is True
        assert s.hf_config_name == "thinker_config"

    def test_talker(self):
        p = resolve_pipeline_config("qwen2_5_omni")
        assert isinstance(p, PipelineConfig)

        s = p.get_stage(1)
        assert isinstance(s, StagePipelineConfig)
        assert s.input_sources == (0,)
        assert s.hf_config_name == "talker_config"
        assert s.sampling_constraints["stop_token_ids"] == [8294]
        # thinker2talker was removed: qwen2_5_omni has no async_chunk support,
        # so sync_process_input_func always wins and custom_process_input_func
        # was dead code.
        assert s.custom_process_input_func is None
        assert s.sync_process_input_func is not None

    def test_code2wav(self):
        p = resolve_pipeline_config("qwen2_5_omni")
        assert isinstance(p, PipelineConfig)

        s = p.get_stage(2)
        assert isinstance(s, StagePipelineConfig)
        assert s.execution_type == StageExecutionType.LLM_GENERATION
        assert s.final_output_type == "audio"
        assert s.engine_output_type == "audio"
        assert s.hf_config_name == "thinker_config"


class TestQwen3TTSPipeline:
    def test_registered(self):
        p = resolve_pipeline_config("qwen3_tts")
        assert isinstance(p, PipelineConfig)
        assert p is not None
        assert p.model_arch == "Qwen3TTSTalkerForConditionalGeneration"
        assert len(p.stages) == 2

    def test_talker_stage(self):
        p = resolve_pipeline_config("qwen3_tts")
        assert isinstance(p, PipelineConfig)

        s = p.get_stage(0)
        assert isinstance(s, StagePipelineConfig)
        assert s.model_stage == "qwen3_tts"
        assert s.execution_type == StageExecutionType.LLM_AR
        assert s.owns_tokenizer is True
        assert s.engine_output_type == "latent"
        assert s.sampling_constraints["stop_token_ids"] == [2150]
        # Stage 0 inherits the pipeline-level model_arch
        assert s.model_arch is None

    def test_code2wav_stage_has_per_stage_model_arch(self):
        p = resolve_pipeline_config("qwen3_tts")
        assert isinstance(p, PipelineConfig)

        s = p.get_stage(1)
        assert isinstance(s, StagePipelineConfig)
        assert s.execution_type == StageExecutionType.LLM_GENERATION
        assert s.final_output_type == "audio"
        assert s.engine_output_type == "audio"
        # Per-stage model_arch override (different from pipeline-level talker)
        assert s.model_arch == "Qwen3TTSCode2Wav"
        # tts_args is passed through via extras
        assert s.extras["tts_args"]["max_instructions_length"] == 500

    def test_subtalker_sampling_params_deep_merge_preserves_base_keys(self):
        """Verify subtalker sampling params participate in stage deep-merge."""
        base = {
            "stage_id": 0,
            "subtalker_sampling_params": {
                "do_sample": True,
                "temperature": 0.9,
                "top_k": 50,
                "top_p": 1.0,
            },
        }
        overlay = {
            "stage_id": 0,
            "subtalker_sampling_params": {
                "temperature": 0.7,
                "top_k": 32,
            },
        }

        merged = _deep_merge_stage(base, overlay)

        assert merged["subtalker_sampling_params"] == {
            "do_sample": True,
            "temperature": 0.7,
            "top_k": 32,
            "top_p": 1.0,
        }


class TestMingFlashOmniPipeline:
    def test_registered(self):
        p = resolve_pipeline_config("ming_flash_omni")
        assert isinstance(p, PipelineConfig)
        assert p.model_arch == "MingFlashOmniForConditionalGeneration"
        assert len(p.stages) == 2

    def test_thinker_stage(self):
        p = resolve_pipeline_config("ming_flash_omni")
        assert isinstance(p, PipelineConfig)

        s = p.get_stage(0)
        assert isinstance(s, StagePipelineConfig)
        assert s.model_stage == "thinker"
        assert s.execution_type == StageExecutionType.LLM_AR
        assert s.owns_tokenizer is True
        assert s.requires_multimodal_data is True
        assert s.engine_output_type == "text"
        assert s.hf_config_name == "llm_config"
        assert s.sampling_constraints["detokenize"] is True

    def test_talker_stage(self):
        p = resolve_pipeline_config("ming_flash_omni")
        assert isinstance(p, PipelineConfig)

        s = p.get_stage(1)
        assert isinstance(s, StagePipelineConfig)
        assert s.model_stage == "ming_tts"
        assert s.execution_type == StageExecutionType.LLM_GENERATION
        assert s.input_sources == (0,)
        assert s.final_output_type == "audio"
        assert s.engine_output_type == "audio"
        assert s.hf_config_name == "talker_config"
        # Per-stage model_arch override (Ming talker has its own self-contained LLM)
        assert s.model_arch == "MingFlashOmniTalkerForConditionalGeneration"
        assert s.tokenizer_subdir == "talker/llm"
        # thinker2talker was removed: ming_flash_omni has no async_chunk support
        # and both thinker2talker / thinker2talker_token_only called _build_talker_inputs
        # identically, so custom_process_input_func was dead code.
        assert s.custom_process_input_func is None
        assert s.sync_process_input_func is not None

    def test_talker_stage_processor_wiring_resolves(self):
        """The sync_process_input_func string must point to a real callable.

        Lazy string references only fail at first inference otherwise — this
        catches typos in the pipeline declaration at import / registration time.
        """
        p = resolve_pipeline_config("ming_flash_omni")
        assert isinstance(p, PipelineConfig)

        s = p.get_stage(1)
        assert isinstance(s, StagePipelineConfig)
        module_path, _, attr = s.sync_process_input_func.rpartition(".")
        module = importlib.import_module(module_path)
        assert callable(getattr(module, attr))

    def test_tts_pipeline_registered(self):
        p = resolve_pipeline_config("ming_flash_omni_tts")
        assert isinstance(p, PipelineConfig)
        assert p.model_arch == "MingFlashOmniTalkerForConditionalGeneration"
        assert len(p.stages) == 1

    def test_tts_stage(self):
        p = resolve_pipeline_config("ming_flash_omni_tts")
        assert isinstance(p, PipelineConfig)

        s = p.get_stage(0)
        assert isinstance(s, StagePipelineConfig)
        assert s.model_stage == "ming_tts"
        assert s.execution_type == StageExecutionType.LLM_GENERATION
        assert s.input_sources == ()
        assert s.owns_tokenizer is True
        assert s.final_output_type == "audio"
        assert s.engine_output_type == "audio"
        assert s.hf_config_name == "talker_config"
        assert s.tokenizer_subdir == "talker/llm"

    def test_thinker_only_pipeline_registered(self):
        p = resolve_pipeline_config("ming_flash_omni_thinker_only")
        assert isinstance(p, PipelineConfig)
        assert p.model_arch == "MingFlashOmniForConditionalGeneration"
        assert len(p.stages) == 1

    def test_thinker_only_stage(self):
        p = resolve_pipeline_config("ming_flash_omni_thinker_only")
        assert isinstance(p, PipelineConfig)

        s = p.get_stage(0)
        assert isinstance(s, StagePipelineConfig)
        assert s.model_stage == "thinker"
        assert s.execution_type == StageExecutionType.LLM_AR
        assert s.input_sources == ()
        assert s.owns_tokenizer is True
        assert s.requires_multimodal_data is True
        assert s.final_output_type == "text"
        assert s.engine_output_type == "text"
        assert s.hf_config_name == "llm_config"
        assert s.sampling_constraints["detokenize"] is True

    def test_image_pipeline_registered(self):
        p = OMNI_PIPELINES.get("ming_flash_omni_image")
        assert p is not None
        assert p.model_arch == "MingFlashOmniForConditionalGeneration"
        assert len(p.stages) == 2

    def test_image_thinker_stage(self):
        s = resolve_pipeline_config("ming_flash_omni_image").get_stage(0)
        assert s.model_stage == "thinker"
        assert s.execution_type == StageExecutionType.LLM_AR
        assert s.input_sources == ()
        assert s.final_output is False
        assert s.owns_tokenizer is True
        assert s.requires_multimodal_data is True
        # Image variant exports hidden states for the diffusion stage.
        assert s.engine_output_type == "latent"
        assert s.hf_config_name == "thinker_config"
        assert s.sampling_constraints["detokenize"] is False
        assert s.prompt_expand_func is not None

    def test_image_dit_stage(self):
        s = resolve_pipeline_config("ming_flash_omni_image").get_stage(1)
        assert s.model_stage == "dit"
        assert s.execution_type == StageExecutionType.DIFFUSION
        assert s.input_sources == (0,)
        assert s.final_output is True
        assert s.final_output_type == "image"
        assert s.hf_config_name == "image_gen_config"
        assert s.model_arch == "MingImagePipeline"
        assert s.custom_process_input_func is not None

    def test_image_processor_wiring_resolves(self):
        """The prompt_expand_func and custom_process_input_func strings must point to real callables."""
        pipeline = resolve_pipeline_config("ming_flash_omni_image")
        assert isinstance(pipeline, PipelineConfig)

        thinker = pipeline.get_stage(0)
        dit = pipeline.get_stage(1)
        for ref in (thinker.prompt_expand_func, dit.custom_process_input_func):
            module_path, _, attr = ref.rpartition(".")
            module = importlib.import_module(module_path)
            assert callable(getattr(module, attr))


class TestBaseConfigInheritance:
    """Test deploy YAML base_config inheritance."""

    @pytest.mark.parametrize(
        ("platform", "expected_devices", "talker_memory"),
        [
            ("rocm", ["0", "1", "1"], 0.3),
            ("cuda", ["0", "1", "2"], 0.4),
            ("xpu", ["0", "1", "2"], 0.5),
        ],
    )
    def test_qwen2_5_omni_ci_placement(self, monkeypatch, platform, expected_devices, talker_memory):
        """The two-card ROCm examples must colocate Code2Wav without changing other platforms."""
        from vllm_omni import platforms

        monkeypatch.setattr(platforms.current_omni_platform, "device_name", platform)
        deploy = load_deploy_config(get_deploy_config_path("ci/qwen2_5_omni.yaml"))
        stages = VllmOmniConfig.from_pipeline_config(
            resolve_pipeline_config("qwen2_5_omni"), user_deploy_config=deploy
        ).stage_configs

        assert [stage.runtime_config.devices for stage in stages] == expected_devices
        assert stages[1].cache_config.gpu_memory_utilization == talker_memory

    def test_minicpmo_overlays_inherit_talker_sampling_params(self):
        """Overlays must keep the base Talker codec Sampler knobs."""
        for filename in (
            "minicpmo_4_5_2gpu.yaml",
            "minicpmo_4_5_3gpu.yaml",
            "minicpmo_4_5_3gpu_stage1_replicas.yaml",
            "minicpmo_4_5_4gpu_stage1_replicas.yaml",
            "minicpmo_4_5_8x4090_stage1_replicas.yaml",
        ):
            path = Path(get_deploy_config_path(filename))
            if not path.exists():
                pytest.skip(f"{filename} not found")
            deploy = load_deploy_config(path)
            sampling = deploy.stages[1].default_sampling_params
            assert sampling is not None, f"{filename} stage 1 lost default_sampling_params"
            assert sampling["temperature"] == 0.8, filename
            assert sampling["top_k"] == 25, filename
            assert sampling["top_p"] == 0.85, filename
            assert sampling["repetition_penalty"] == 1.05, filename

    def test_ci_inherits_from_main(self):
        ci_path = Path(get_deploy_config_path("ci/qwen3_omni_moe.yaml"))
        if not ci_path.exists():
            pytest.skip("CI deploy config not found")

        deploy = load_deploy_config(ci_path)
        assert len(deploy.stages) == 3
        # CI overrides
        assert deploy.stages[0].load_format is None
        assert "load_format" not in deploy.stages[0].engine_extras
        assert deploy.stages[0].max_num_seqs == 5
        # Inherited from base
        assert deploy.stages[0].gpu_memory_utilization == 0.9
        assert deploy.connectors is not None
        assert "connector_of_shared_memory" in deploy.connectors
        # CI overlay explicitly sets async_chunk: False (see
        # tests.helpers.stage_config._CI_OVERLAYS and PR #2383 discussion). Overlay
        # bool overrides base even when the base yaml has async_chunk: true.
        assert deploy.async_chunk is False

    def test_ci_sampling_merge(self):
        ci_path = Path(get_deploy_config_path("ci/qwen3_omni_moe.yaml"))
        if not ci_path.exists():
            pytest.skip("CI deploy config not found")

        deploy = load_deploy_config(ci_path)
        s0 = deploy.stages[0].default_sampling_params
        # CI overrides max_tokens
        assert s0["max_tokens"] == 150

    def test_pure_inheritance_overlay(self, tmp_path):
        """An overlay with only ``base_config`` inherits everything."""
        base = Path(get_deploy_config_path("qwen3_omni_moe.yaml"))
        if not base.exists():
            pytest.skip("Base deploy config not found")

        overlay = tmp_path / "overlay.yaml"
        overlay.write_text(f"base_config: {base}\n")

        deploy = load_deploy_config(overlay)
        assert len(deploy.stages) == 3
        assert deploy.stages[0].gpu_memory_utilization == 0.9

    def test_single_field_overlay(self, tmp_path):
        """An overlay overriding one stage field merges with the base."""
        base = Path(get_deploy_config_path("qwen3_omni_moe.yaml"))
        if not base.exists():
            pytest.skip("Base deploy config not found")

        overlay = tmp_path / "overlay.yaml"
        overlay.write_text(f"base_config: {base}\nstages:\n  - stage_id: 2\n    max_num_batched_tokens: 1000000\n")

        deploy = load_deploy_config(overlay)
        assert deploy.stages[2].max_num_batched_tokens == 1000000
        # Rest inherited
        assert deploy.stages[0].gpu_memory_utilization == 0.9


class TestPlatformOverrides:
    """Test platform-specific deploy config overrides."""

    def test_qwen3_tts_rocm_disables_code2wav_outer_cudagraph(self):
        deploy_path = Path(get_deploy_config_path("qwen3_tts.yaml"))

        base = load_deploy_config(deploy_path)
        assert base.stages[0].enforce_eager is None
        assert base.stages[1].enforce_eager is False

        rocm = _apply_platform_overrides(base, platform="rocm")
        assert rocm.stages[0].enforce_eager is None
        assert rocm.stages[1].enforce_eager is True

    @pytest.mark.parametrize("platform", ["cuda", "npu"])
    def test_minicpmo_4_5_caps_talker_kv_cache(self, platform):
        pipeline = resolve_pipeline_config("minicpmo_4_5")
        assert isinstance(pipeline, PipelineConfig)
        deploy_path = Path(get_deploy_config_path("minicpmo_4_5.yaml"))

        deploy = _apply_platform_overrides(load_deploy_config(deploy_path), platform=platform)
        stages = VllmOmniConfig.from_pipeline_config(pipeline, user_deploy_config=deploy).stage_configs
        # Both single-device profiles reserve headroom for colocated Code2Wav.
        assert stages[1].cache_config.kv_cache_memory_bytes == 4 * 1024**3

    def test_minicpmo_4_5_cuda_replicas_use_automatic_talker_kv_cache(self):
        pipeline = resolve_pipeline_config("minicpmo_4_5")
        assert isinstance(pipeline, PipelineConfig)
        # Multi-GPU replica profiles place Talker on dedicated devices and
        # preserve their previous automatic KV-cache sizing.
        for filename in (
            "minicpmo_4_5_3gpu_stage1_replicas.yaml",
            "minicpmo_4_5_4gpu_stage1_replicas.yaml",
        ):
            replica = _apply_platform_overrides(
                load_deploy_config(Path(get_deploy_config_path(filename))), platform="cuda"
            )
            replica_stages = VllmOmniConfig.from_pipeline_config(pipeline, user_deploy_config=replica).stage_configs
            # Explicit null clears the inherited single-GPU 4 GiB CUDA cap.
            assert replica_stages[1].cache_config.kv_cache_memory_bytes is None

    def test_fish_speech_npu_uses_ascend_kv_block_size(self):
        deploy_path = Path(get_deploy_config_path("fish_qwen3_omni.yaml"))

        deploy = _apply_platform_overrides(load_deploy_config(deploy_path), platform="npu")

        assert deploy.stages[0].engine_extras["block_size"] == 128

    def test_npu_overrides(self):
        deploy_path = Path(get_deploy_config_path("qwen3_omni_moe.yaml"))
        if not deploy_path.exists():
            pytest.skip("Deploy config not found")

        deploy = load_deploy_config(deploy_path)
        deploy = _apply_platform_overrides(deploy, platform="npu")

        assert deploy.stages[0].gpu_memory_utilization == 0.6
        assert deploy.stages[0].tensor_parallel_size == 2
        assert deploy.stages[0].devices == "0,1"
        # Stage 2 unaffected fields stay at base
        assert deploy.stages[2].enforce_eager is False

    def test_qwen2_5_omni_xpu_uses_eager_ar_stages(self):
        deploy_path = Path(get_deploy_config_path("qwen2_5_omni.yaml"))

        deploy = load_deploy_config(deploy_path)
        deploy = _apply_platform_overrides(deploy, platform="xpu")

        assert deploy.stages[0].enforce_eager is True
        assert deploy.stages[1].enforce_eager is True
        assert deploy.stages[2].enforce_eager is True

    def test_xpu_overrides(self):
        deploy_path = Path(get_deploy_config_path("qwen3_omni_moe.yaml"))
        if not deploy_path.exists():
            pytest.skip("Deploy config not found")

        deploy = load_deploy_config(deploy_path)
        deploy = _apply_platform_overrides(deploy, platform="xpu")

        assert deploy.stages[0].tensor_parallel_size == 4
        assert deploy.stages[0].devices == "0,1,2,3"
        assert deploy.stages[0].engine_extras.get("max_cudagraph_capture_size") == 0

    def test_unknown_platform_noop(self):
        deploy_path = Path(get_deploy_config_path("qwen3_omni_moe.yaml"))
        if not deploy_path.exists():
            pytest.skip("Deploy config not found")

        deploy = load_deploy_config(deploy_path)
        original_mem = deploy.stages[0].gpu_memory_utilization
        deploy = _apply_platform_overrides(deploy, platform="unknown_hw")
        assert deploy.stages[0].gpu_memory_utilization == original_mem

    def test_platforms_deep_merge_inheritance(self, tmp_path):
        """Overlay's platforms: block layers onto base's, per-stage."""
        base = tmp_path / "base.yaml"
        base.write_text(
            "stages:\n"
            "  - stage_id: 0\n"
            "    gpu_memory_utilization: 0.9\n"
            "platforms:\n"
            "  rocm:\n"
            "    stages:\n"
            "      - stage_id: 0\n"
            "        enforce_eager: true\n"
        )
        overlay = tmp_path / "overlay.yaml"
        overlay.write_text(
            f"base_config: {base.name}\n"
            "platforms:\n"
            "  rocm:\n"
            "    stages:\n"
            "      - stage_id: 0\n"
            "        max_num_seqs: 1\n"
        )

        deploy = load_deploy_config(overlay)
        deploy = _apply_platform_overrides(deploy, platform="rocm")
        # Both base's enforce_eager and overlay's max_num_seqs should apply.
        assert deploy.stages[0].enforce_eager is True
        assert deploy.stages[0].max_num_seqs == 1
        # Inherited stage default not touched by overlay platforms section.
        assert deploy.stages[0].gpu_memory_utilization == 0.9

    def test_qwen3_omni_talker_sampling_is_seeded(self):
        """The only stochastic Qwen3-Omni stage must stay reproducible.

        Dropping this seed (as #4986 did) leaves the talker sampling codec
        tokens unseeded at temperature 0.9, which reopened the audio-vs-text
        nightly failures in #6090. Pin it so a cleanup cannot remove it again
        without failing here.
        """
        deploy_path = Path(get_deploy_config_path("qwen3_omni_moe.yaml"))
        pipeline = resolve_pipeline_config(
            "qwen3_omni_moe",
            Q3_OMNI_ALL_STAGES_HF_CONFIG,
        )
        assert isinstance(pipeline, PipelineConfig)

        for platform in ("cpu", "cuda", "musa", "npu", "rocm", "xpu"):
            deploy = _apply_platform_overrides(load_deploy_config(deploy_path), platform=platform)
            stages = VllmOmniConfig.from_pipeline_config(pipeline, user_deploy_config=deploy).stage_configs
            talker_sampling = stages[1].model_config.default_sampling_params
            assert talker_sampling["temperature"] > 0.0, "talker is expected to sample, not decode greedily"
            assert talker_sampling.get("seed") == 42, f"talker sampling lost its seed on {platform}"

    def test_moss_local_cuda_default_uses_one_profile_without_hardware_dispatch(self, monkeypatch):
        from vllm.platforms import current_platform

        from vllm_omni.platforms import current_omni_platform

        monkeypatch.setattr(current_omni_platform, "device_name", "cuda")

        def unexpected_probe(*args):
            pytest.fail("Pipeline selection must not probe CUDA memory or MPS")

        monkeypatch.setattr(current_platform, "get_device_total_memory", unexpected_probe)
        monkeypatch.setattr("shutil.which", unexpected_probe)
        pipeline = resolve_pipeline_config("moss_tts_local")
        assert pipeline.default_deploy_config_name == "moss_tts_local.yaml"
        config = VllmOmniConfig.from_pipeline_config(pipeline)
        talker, codec = config.stage_configs
        assert talker.scheduler_config.max_num_seqs == codec.scheduler_config.max_num_seqs == 128
        assert talker.cache_config.kv_cache_memory_bytes == 32 * 1024**3
        assert talker.cache_config.enable_prefix_caching
        assert not codec.cache_config.enable_prefix_caching
        for stage in config.stage_configs:
            assert stage.model_config.use_v2_model_runner
            assert stage.runtime_config.cuda_mps

    @pytest.mark.parametrize("platform", ["cpu", "npu", "xpu", "rocm", "musa"])
    def test_moss_local_non_cuda_default_preserves_v1(self, monkeypatch, platform):
        from vllm.platforms import current_platform

        from vllm_omni.platforms import current_omni_platform

        monkeypatch.setattr(current_omni_platform, "device_name", platform)

        def unexpected_probe(*args):
            pytest.fail("Non-CUDA selection must not probe CUDA memory or MPS")

        monkeypatch.setattr(current_platform, "get_device_total_memory", unexpected_probe)
        monkeypatch.setattr("shutil.which", unexpected_probe)
        pipeline = resolve_pipeline_config("moss_tts_local")
        config = VllmOmniConfig.from_pipeline_config(pipeline)
        for stage in config.stage_configs:
            assert not stage.model_config.use_v2_model_runner
            assert not stage.runtime_config.cuda_mps
            assert stage.scheduler_config.max_num_seqs == 64
        assert config.stage_configs[0].cache_config.kv_cache_memory_bytes is None

    def test_moss_local_custom_deploy_overrides_default(self, monkeypatch, tmp_path):
        from vllm_omni.platforms import current_omni_platform

        monkeypatch.setattr(current_omni_platform, "device_name", "cuda")
        base_path = get_deploy_config_path("moss_tts_local.yaml")
        path = tmp_path / "custom_moss.yaml"
        path.write_text(
            f"base_config: {base_path}\n"
            "platforms:\n  cuda:\n    model_runner: v1\n    cuda_mps: false\n"
            "    stages:\n      - stage_id: 0\n        max_num_seqs: 32\n"
            "      - stage_id: 1\n        max_num_seqs: 32\n"
        )
        pipeline = resolve_pipeline_config("moss_tts_local")
        config = VllmOmniConfig.from_pipeline_config(pipeline, deploy_config_path=path)
        for stage in config.stage_configs:
            assert not stage.model_config.use_v2_model_runner
            assert not stage.runtime_config.cuda_mps
            assert stage.scheduler_config.max_num_seqs == 32

    @pytest.mark.parametrize("platform", ["cuda", "npu", "xpu", "rocm", "musa"])
    def test_moss_local_system_profile_reaches_native_runner_and_cuda_only_mps(self, platform):
        pipeline = resolve_pipeline_config("moss_tts_local")
        path = Path(get_deploy_config_path("moss_tts_local.yaml"))
        deploy = _apply_platform_overrides(load_deploy_config(path), platform=platform)
        stages = VllmOmniConfig.from_pipeline_config(pipeline, user_deploy_config=deploy).stage_configs
        is_cuda = platform == "cuda"
        assert deploy.cuda_mps is is_cuda
        assert all(stage.runtime_config.cuda_mps is is_cuda for stage in stages)
        assert all(stage.model_config.use_v2_model_runner is is_cuda for stage in stages)
        if is_cuda:
            args = project_engine_args(stages[0], "test-model")
            assert args["enable_prefix_caching"] is True
            assert args["hf_overrides"]["mrv2_gpu_slot_state"] is True
            assert args["hf_overrides"]["mrv2_batch_prefill"] is True
            assert args["hf_overrides"]["mrv2_direct_tokens"] is True
            assert "local_compile_audio_sampler" not in args["hf_overrides"]
            extra = deploy.connectors["shm"]["extra"]
            assert extra["generation_min_batch_size"] == 16
            assert extra["generation_max_wait_ms"] == 6
            assert not stages[0].runtime_config.env
        else:
            assert all(not stage.cache_config.enable_prefix_caching for stage in stages)
            assert all(not stage.runtime_config.env for stage in stages)

    def test_moss_local_first_audio_preserves_prefix_cache_and_runner_options(self):
        pipeline = resolve_pipeline_config("moss_tts_local")
        path = get_deploy_config_path("moss_tts_local.yaml")
        deploy = _apply_platform_overrides(load_deploy_config(path), platform="cuda")
        talker, codec = VllmOmniConfig.from_pipeline_config(pipeline, user_deploy_config=deploy).stage_configs
        args = project_engine_args(talker, "test-model")
        assert args["enable_prefix_caching"] is True
        assert args["use_v2_model_runner"] is True
        assert args["max_num_seqs"] == 128
        assert args["max_num_batched_tokens"] == 512
        assert args["hf_overrides"] == {
            "mrv2_gpu_slot_state": True,
            "mrv2_batch_prefill": True,
            "mrv2_direct_tokens": True,
            "mrv2_eager_mtp": True,
        }
        assert talker.compilation_config.mode == 0
        assert talker.compilation_config.cudagraph_mode == CUDAGraphMode.FULL
        raw_compilation = deploy.stages[0].compilation_config
        assert raw_compilation is not None and "inductor_compile_config" not in raw_compilation
        extra = deploy.connectors["shm"]["extra"]
        assert extra["initial_codec_chunk_frames"] == 1
        assert extra["codec_chunk_frames"] == 15
        assert extra["generation_min_batch_size"] == 16
        assert extra["generation_max_wait_ms"] == 6
        assert codec.model_config.use_v2_model_runner is True
        assert deploy.cuda_mps is True
        assert args["kv_cache_memory_bytes"] == 32 * 1024**3
        codec_args = project_engine_args(codec, "test-model")
        assert codec_args["max_num_seqs"] == 128
        assert codec_args["hf_overrides"]["codec_attention_backend"] == "triton_slot"
        compilation = codec.compilation_config
        assert compilation.mode == 0
        assert compilation.cudagraph_mode == CUDAGraphMode.FULL
        assert max(compilation.cudagraph_capture_sizes) == 128
        raw_compilation = deploy.stages[1].compilation_config
        assert raw_compilation is not None and "inductor_compile_config" not in raw_compilation
        for variant in ("moss_tts_delay", "moss_tts_realtime"):
            assert not any(stage.supports_native_mrv2_data_plane for stage in resolve_pipeline_config(variant).stages)

    def test_platform_mps_rejects_non_boolean(self):
        deploy = load_deploy_config(get_deploy_config_path("moss_tts_local.yaml"))
        deploy.platforms["cuda"]["cuda_mps"] = "true"
        with pytest.raises(ValueError, match="platform cuda_mps must be a boolean"):
            _apply_platform_overrides(deploy, platform="cuda")


class TestAuraOmniDeploy:
    def test_aura_omni_deploy_forces_pipeline_override(self):
        deploy_path = Path(get_deploy_config_path("aura_omni.yaml"))
        deploy = load_deploy_config(deploy_path)

        assert deploy.pipeline == "aura_omni"


class TestDeployCliOverrideFlow:
    """Test deploy-YAML baselines overridden by CLI runtime overrides."""


class TestSamplingConstraintsPrecedence:
    """Test scalar constraint precedence and additive required stop tokens."""


class TestPipelineConfigResolvers:
    @pytest.mark.parametrize("resolver", [obj for obj in OMNI_PIPELINES.values() if callable(obj)])
    def test_all_resolvers_reject_bad_types(self, resolver):
        """Ensure that all resolvers registered reject incorrect config types."""

        class NotTheRightHfConfig(PretrainedConfig):
            pass

        assert resolver(NotTheRightHfConfig()) is None


class TestObjectStorageConfigResolution:
    """Object-storage URIs must be materialized locally before any HF-style reads.

    Regression coverage for review on the Run:AI PR: ``get_config`` and the
    config.json/model_index.json fallbacks crashed with ``HFValidationError``
    for ``s3://`` URIs, and the name-match fallback scanned the whole URI, so
    a bucket named after another pipeline could hijack pipeline selection.
    """

    @pytest.fixture
    def fake_object_storage(self, monkeypatch, tmp_path):
        """Replace ObjectStorageModel with a fake that "pulls" into a tmp dir.

        Returns a recorder: calls to the returned callable write files into
        the materialized directory, and ``recorder.pulls`` records every
        ``pull_files`` invocation.
        """
        pulls: list[tuple[str, list[str] | None, list[str] | None]] = []
        materialized_files: list[tuple[str, str]] = []

        class FakeObjectStorageModel:
            def __init__(self, url: str):
                self.dir = str(tmp_path)

            def pull_files(self, model_path, allow_pattern=None, ignore_pattern=None):
                pulls.append((model_path, allow_pattern, ignore_pattern))
                for name, content in materialized_files:
                    (tmp_path / name).write_text(content)

        monkeypatch.setattr(config_factory_module, "ObjectStorageModel", FakeObjectStorageModel)

        class Recorder:
            def __init__(self) -> None:
                self.pulls = pulls

            def set_files(self, files: list[tuple[str, str]]) -> None:
                materialized_files.clear()
                materialized_files.extend(files)

        return Recorder()

    def test_passthrough_for_non_uri(self, fake_object_storage):
        assert _materialize_object_storage_configs("org/model") == "org/model"
        assert _materialize_object_storage_configs("/local/model") == "/local/model"
        assert fake_object_storage.pulls == []

    def test_materialize_pulls_configs_once_per_uri(self, fake_object_storage):
        uri = "s3://bucket/model"

        first = _materialize_object_storage_configs(uri)
        second = _materialize_object_storage_configs(uri)

        assert first == second
        assert len(fake_object_storage.pulls) == 1
        pulled_uri, allow_pattern, ignore_pattern = fake_object_storage.pulls[0]
        assert pulled_uri == uri
        assert "*.json" in allow_pattern
        assert ignore_pattern is None

    def test_try_infer_model_type_reads_materialized_config(self, fake_object_storage):
        """A misleading bucket name must not beat the real config.json content."""
        fake_object_storage.set_files(
            [
                (
                    "config.json",
                    '{"model_type": "qwen3_omni_moe", "architectures": ["Qwen3OmniMoeForConditionalGeneration"]}',
                )
            ]
        )
        uri = "s3://qwen3-tts-models/Qwen3-Omni-30B-A3B-Instruct"

        model = StageConfigFactory.try_infer_model_type(model=uri, trust_remote_code=False)

        assert model == "qwen3_omni_moe"
        assert fake_object_storage.pulls  # materialization happened

    def test_name_match_fallback_scans_basename_only(self, fake_object_storage):
        # Empty config.json (CosyVoice3 style) forces the name-match fallback.
        fake_object_storage.set_files([("config.json", "{}")])

        deceptive = "s3://qwen3-tts-models/plain-checkpoint"
        assert StageConfigFactory.try_infer_model_type(model=deceptive, trust_remote_code=False) is None

        matching = "s3://any-bucket/my-cosyvoice3-model"
        assert StageConfigFactory.try_infer_model_type(model=matching, trust_remote_code=False) == "cosyvoice3"


class TestAsyncChunkDefaults:
    def test_async_chunk_auto_disabled_without_processor(self):
        """Ensure a multi-stage model that doesn't support async chunk turns it off by default."""
        pipeline = PipelineConfig(
            model_type="test_no_async",
            model_arch="TestNoAsync",
            stages=(
                StagePipelineConfig(
                    stage_id=0,
                    model_stage="ar",
                    execution_type=StageExecutionType.LLM_AR,
                    final_output=True,
                ),
                StagePipelineConfig(
                    stage_id=1,
                    model_stage="generation",
                    execution_type=StageExecutionType.LLM_GENERATION,
                    input_sources=(0,),
                ),
            ),
        )

        deploy = DeployConfig()
        # async chunk should not try to default to True in this case,
        # since doing so will just raise a ValueError in validation.
        config = VllmOmniConfig.from_pipeline_config(pipeline, user_deploy_config=deploy)
        assert not any(stage.connector_config.async_chunk for stage in config.stage_configs)

    def test_async_chunk_auto_disabled_when_yaml_omits_key(self, tmp_path):
        """Ensure a multi-stage model that doesn't support async chunk turns it off by default
        when a deploy config is provided that doesn't explicitly set it."""

        deploy_path = tmp_path / "no_async_chunk.yaml"
        deploy_path.write_text(
            """
stages:
  - stage_id: 0
    devices: "0"
  - stage_id: 1
    devices: "0"
""",
            encoding="utf-8",
        )
        deploy = load_deploy_config(deploy_path)

        pipeline = PipelineConfig(
            model_type="test_no_async",
            model_arch="TestNoAsync",
            stages=(
                StagePipelineConfig(
                    stage_id=0,
                    model_stage="ar",
                    execution_type=StageExecutionType.LLM_AR,
                    final_output=True,
                ),
                StagePipelineConfig(
                    stage_id=1,
                    model_stage="generation",
                    execution_type=StageExecutionType.LLM_GENERATION,
                    input_sources=(0,),
                ),
            ),
        )

        config = VllmOmniConfig.from_pipeline_config(pipeline, user_deploy_config=deploy)
        assert not any(stage.connector_config.async_chunk for stage in config.stage_configs)
