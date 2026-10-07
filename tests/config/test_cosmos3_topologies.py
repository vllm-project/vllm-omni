# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cosmos3 disaggregated topology registration, deploy YAML and device mapping.

``cosmos3_omni_disagg`` (one stage per Mixture-of-Transformers tower) is a second
topology over the *same* checkpoint as the co-located ``cosmos3_omni_deploy``. The
invariant these tests defend is that it is not auto-detectable: every Cosmos3
checkpoint -- T2I, T2V/I2V/V2V and policy alike -- reports
``model_type=cosmos3_omni`` with ``model_index.json``
``_class_name=Cosmos3OmniDiffusersPipeline``, so a topology that claimed those
would hijack every other Cosmos3 deployment. It is reachable only through an
explicit ``pipeline:`` key in a deploy YAML, and a Cosmos3 deployment that names
no pipeline keeps resolving through the single-stage diffusion fallback.
"""

import json
from pathlib import Path

import pytest

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.config_factory import StageConfigFactory, _materialize_object_storage_configs
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES, resolve_pipeline_config
from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    load_deploy_config,
    merge_pipeline_deploy,
)
from vllm_omni.diffusion.models.cosmos3_pipeline_config import (
    COSMOS3_ARCH,
    COSMOS3_DISAGG_PIPELINE,
    COSMOS3_GENERATOR_ARCH,
    COSMOS3_REASONER_ARCH,
    COSMOS3_UND_KV_KEY,
    COSMOS3_UND_META_KEY,
)
from vllm_omni.entrypoints.stage_utils import resolve_stage_physical_devices

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

DISAGG_YAML = "cosmos3_super_t2i_disagg.yaml"

COSMOS3_HF_ARCH = "Cosmos3ForConditionalGeneration"


@pytest.fixture(autouse=True)
def clear_config_factory_caches():
    """Cached classmethods leak resolved model types between tests otherwise."""
    yield
    StageConfigFactory.get_hf_config.cache_clear()
    StageConfigFactory.try_infer_model_type.cache_clear()
    _materialize_object_storage_configs.cache_clear()


def _deploy(name: str):
    return load_deploy_config(Path(get_deploy_config_path(name)))


def _stage(deploy, stage_id: int):
    return next(s for s in deploy.stages if s.stage_id == stage_id)


class TestTopologyRegistration:
    def test_disagg_topology_registered(self):
        assert resolve_pipeline_config("cosmos3_omni_disagg") is COSMOS3_DISAGG_PIPELINE

    def test_does_not_claim_the_bare_checkpoint_model_type(self):
        """``cosmos3_omni`` is the HF ``model_type`` of *every* Cosmos3 checkpoint.

        Registering it would make the disagg topology the auto-detected answer for
        T2I, video and policy checkpoints alike.
        """
        assert "cosmos3_omni" not in OMNI_PIPELINES

    def test_disagg_topology_is_one_stage_per_tower(self):
        reasoner, generator = COSMOS3_DISAGG_PIPELINE.stages
        # Safe to name: this topology is unreachable without a deploy config that
        # selects it by name, so by the time it resolves the caller has already
        # supplied one.
        assert COSMOS3_DISAGG_PIPELINE.default_deploy_config_name == DISAGG_YAML

        assert (reasoner.stage_id, reasoner.model_stage) == (0, "reasoner")
        assert reasoner.execution_type is StageExecutionType.DIFFUSION
        assert reasoner.input_sources == ()
        assert reasoner.owns_tokenizer is True
        assert reasoner.final_output is False
        assert reasoner.model_arch == COSMOS3_REASONER_ARCH

        assert (generator.stage_id, generator.model_stage) == (1, "generator")
        assert generator.input_sources == (0,)
        assert generator.final_output is True
        assert generator.final_output_type == "image"
        assert generator.model_arch == COSMOS3_GENERATOR_ARCH
        # The handoff travels in the stage payload, not the AR KV-transfer path.
        assert generator.omni_kv_config == {"need_recv_cache": False}

    def test_generator_input_processor_is_importable(self):
        """The bridge is declared as a string; make sure it actually resolves."""
        import importlib

        target = COSMOS3_DISAGG_PIPELINE.stages[1].custom_process_input_func
        module_path, _, func_name = target.rpartition(".")
        func = getattr(importlib.import_module(module_path), func_name)
        assert callable(func)
        assert func.__name__ == "reasoner2generator"

    def test_payload_keys_are_shared_with_the_bridge(self):
        """The tower pipelines and the bridge must not drift apart on key names."""
        from vllm_omni.model_executor.stage_input_processors import cosmos3 as bridge

        assert bridge.KV_KEY == COSMOS3_UND_KV_KEY
        assert bridge.META_KEY == COSMOS3_UND_META_KEY


class TestDisaggTopologyIsOptInOnly:
    def test_declares_no_auto_detect_hooks(self):
        assert COSMOS3_DISAGG_PIPELINE.hf_architectures == ()
        assert COSMOS3_DISAGG_PIPELINE.diffusers_class_name is None

    def test_no_pipeline_claims_the_cosmos3_architecture(self):
        """The arch fallback scans every registered pipeline; none may match.

        A claimant here would capture every Cosmos3 checkpoint, because they all
        ship ``architectures=["Cosmos3ForConditionalGeneration"]`` -- video and
        policy checkpoints included.
        """
        claimants = [
            key
            for key, entry in OMNI_PIPELINES.items()
            if isinstance(entry, PipelineConfig) and COSMOS3_HF_ARCH in entry.hf_architectures
        ]
        assert claimants == []

    def test_no_pipeline_claims_the_diffusers_class_name(self):
        """Same for the model_index.json fallback."""
        claimants = [
            key
            for key, entry in OMNI_PIPELINES.items()
            if isinstance(entry, PipelineConfig) and entry.diffusers_class_name == COSMOS3_ARCH
        ]
        assert claimants == []

    def test_model_index_autodetect_reaches_no_registered_pipeline(self, tmp_path):
        """A Cosmos3 checkpoint that names no pipeline stays on the fallback.

        ``get_pipeline_config`` returning None is what makes the engine build the
        default single-stage diffusion config, where ``final_output_type`` is
        resolved dynamically per model class ("video" for
        ``Cosmos3OmniDiffusersPipeline``) instead of pinned by a registry entry.
        """
        (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": COSMOS3_ARCH}), encoding="utf-8")

        assert StageConfigFactory.try_infer_model_type(model=str(tmp_path), trust_remote_code=False) is None
        assert (
            StageConfigFactory.get_pipeline_config(
                model=str(tmp_path),
                trust_remote_code=False,
            )
            is None
        )

    def test_deploy_pipeline_key_selects_the_topology(self, tmp_path):
        """The one and only route into the topology."""
        deploy_path = tmp_path / "deploy.yaml"
        deploy_path.write_text("pipeline: cosmos3_omni_disagg\n", encoding="utf-8")

        pipeline = StageConfigFactory.get_pipeline_config(
            model=str(tmp_path),
            trust_remote_code=False,
            deploy_config_path=str(deploy_path),
        )

        assert pipeline is COSMOS3_DISAGG_PIPELINE


class TestDisaggDeployConfig:
    def test_shipped_yaml_selects_the_disagg_topology(self):
        deploy = _deploy(DISAGG_YAML)

        assert deploy.pipeline == "cosmos3_omni_disagg"
        # merge_pipeline_deploy raises for an async_chunk pipeline whose stages
        # have input_sources but no async_chunk producer.
        assert deploy.async_chunk is False
        assert [s.stage_id for s in deploy.stages] == [0, 1]

    def test_one_card_per_stage(self):
        """`devices` are logical indexes into a *shared* visible set, so the two
        stages must name different indexes or both towers land on one GPU."""
        devices = [s.devices for s in _deploy(DISAGG_YAML).stages]

        assert devices == ["0", "1"]

    @pytest.mark.parametrize("stage_id", [0, 1])
    def test_no_intra_stage_collectives(self, stage_id: int):
        """Each tower fits uncut on one H200 (141 GB): every degree is 1, HSDP off."""
        parallel = _stage(_deploy(DISAGG_YAML), stage_id).engine_extras["parallel_config"]

        assert parallel["use_hsdp"] is False
        assert {
            parallel["tensor_parallel_size"],
            parallel["data_parallel_size"],
            parallel["pipeline_parallel_size"],
            parallel["ulysses_degree"],
            parallel["ring_degree"],
            parallel["cfg_parallel_size"],
        } == {1}

    @pytest.mark.parametrize("stage_id", [0, 1])
    def test_both_stages_disable_guardrails(self, stage_id: int):
        """Either stage left with guardrails on hard-fails the whole pipeline at
        build time wherever `cosmos-guardrail` is absent."""
        stage = _stage(_deploy(DISAGG_YAML), stage_id)

        assert stage.engine_extras["model_config"]["guardrails"] is False

    def test_neither_stage_pins_generation_parameters(self):
        """Per-stage generation defaults are actively dangerous in this topology.

        The two towers must resolve the text conditioning identically -- the
        reasoner encodes the unconditional branch only when guidance_scale > 1, and
        the generator looks its branches up by a fingerprint of the tokenized
        prompt, which depends on max_sequence_length, use_system_prompt and the
        geometry. A default set on one stage and not the other makes them disagree
        and the request fails with a replay miss. Shipping none on either side is
        the only configuration that cannot diverge.
        """
        stage0, stage1 = (_stage(_deploy(DISAGG_YAML), i) for i in (0, 1))

        assert not stage0.default_sampling_params
        assert not stage1.default_sampling_params

    def test_tensor_parallel_size_is_stage_local(self):
        """Every parallel degree in this topology is the stage's own business.

        UND K/V is born TP-sharded, but the reasoner all-gathers the KV-head
        dimension before the payload leaves the tower and each generator rank slices
        back out the range its own cross-attention owns, so the handoff is
        TP-independent. Nothing in the merged config may therefore tie the two
        stages' degrees together -- each stage carries its own ``parallel_config``,
        and the shipped values are 1 because each tower fits on one card.
        """
        stages = [_stage(_deploy(DISAGG_YAML), i) for i in (0, 1)]

        for stage in stages:
            assert stage.engine_extras["parallel_config"]["tensor_parallel_size"] == 1
        # Distinct dicts, not one shared object: a change to one stage's degrees
        # cannot leak into the other's.
        assert stages[0].engine_extras["parallel_config"] is not stages[1].engine_extras["parallel_config"]

    def test_merges_into_reasoner_plus_generator(self):
        stages = merge_pipeline_deploy(COSMOS3_DISAGG_PIPELINE, _deploy(DISAGG_YAML))

        assert [s.model_stage for s in stages] == ["reasoner", "generator"]
        assert [s.yaml_runtime["devices"] for s in stages] == ["0", "1"]
        assert stages[0].final_output is False
        assert stages[1].final_output is True
        assert stages[1].final_output_type == "image"
        assert stages[1].custom_process_input_func.endswith(".reasoner2generator")
        assert all(s.yaml_engine_args["model_config"]["guardrails"] is False for s in stages)


class TestDisaggDeviceMapping:
    """The shipped YAML plus one shared CUDA_VISIBLE_DEVICES per stage worker."""

    @pytest.mark.parametrize(
        ("visible", "expected"),
        [
            ("0,1", ["0", "1"]),
            # The pair does not have to be contiguous, or start at 0.
            ("2,6", ["2", "6"]),
            ("4,5,6,7", ["4", "5"]),
            # Logical, not physical: reordering the visible set swaps the towers.
            ("6,2", ["6", "2"]),
        ],
    )
    def test_stages_resolve_to_distinct_physical_gpus(self, visible: str, expected: list[str]):
        deploy = _deploy(DISAGG_YAML)

        resolved = [
            resolve_stage_physical_devices(stage.stage_id, stage.devices, visible_baseline=visible)
            for stage in deploy.stages
        ]

        assert resolved == expected
