# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cosmos3 topology registration, deploy YAMLs and per-stage device mapping.

Cosmos3 ships two topologies over the *same* checkpoint: the co-located
``cosmos3_omni`` (both Mixture-of-Transformers towers in one diffusion stage) and
the disaggregated ``cosmos3_omni_disagg`` (one stage per tower). The invariant
these tests defend is that the second one is reachable *only* through an explicit
``pipeline:`` key in a deploy YAML -- if it ever became auto-detectable it would
hijack every co-located Cosmos3 deployment merely by being registered.
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
    COSMOS3_PIPELINE,
    COSMOS3_REASONER_ARCH,
    COSMOS3_UND_KV_KEY,
    COSMOS3_UND_META_KEY,
)
from vllm_omni.entrypoints.stage_utils import resolve_stage_physical_devices

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

COLOCATED_YAML = "cosmos3_super_t2i.yaml"
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
    def test_both_topologies_registered(self):
        assert resolve_pipeline_config("cosmos3_omni") is COSMOS3_PIPELINE
        assert resolve_pipeline_config("cosmos3_omni_disagg") is COSMOS3_DISAGG_PIPELINE

    def test_colocated_topology_is_a_single_diffusion_stage(self):
        assert len(COSMOS3_PIPELINE.stages) == 1
        stage = COSMOS3_PIPELINE.stages[0]
        assert stage.model_stage == "diffusion"
        assert stage.execution_type is StageExecutionType.DIFFUSION
        assert stage.input_sources == ()
        assert stage.final_output is True
        assert stage.final_output_type == "image"
        assert stage.model_arch == COSMOS3_ARCH

    def test_colocated_topology_declares_no_default_deploy_config(self):
        """Registering the topology must not change any existing deployment.

        ``_get_deploy_config`` auto-loads a pipeline's default deploy YAML for
        every caller that passes no ``--deploy-config``, so naming one here would
        push device count, ``max_num_seqs``, ``enforce_eager``,
        ``gpu_memory_utilization`` and the ``guardrails`` gate onto co-located
        deployments that never asked for a deploy config, purely as a side effect
        of this registration. ``COLOCATED_YAML`` is opt-in; it still has to exist
        and parse, which ``TestColocatedDeployConfig`` covers.
        """
        assert COSMOS3_PIPELINE.default_deploy_config_name is None

    def test_disagg_topology_is_one_stage_per_tower(self):
        reasoner, generator = COSMOS3_DISAGG_PIPELINE.stages
        # Safe to name here, unlike on the co-located config: this topology is
        # unreachable without a deploy config that selects it by name, so by the
        # time it resolves the caller has already supplied one.
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


class TestDisaggIsOptInOnly:
    def test_disagg_declares_no_auto_detect_hooks(self):
        assert COSMOS3_DISAGG_PIPELINE.hf_architectures == ()
        assert COSMOS3_DISAGG_PIPELINE.diffusers_class_name is None

    def test_only_the_colocated_topology_claims_the_cosmos3_architecture(self):
        """The arch fallback scans every registered pipeline; exactly one may match."""
        claimants = [
            key
            for key, entry in OMNI_PIPELINES.items()
            if isinstance(entry, PipelineConfig) and COSMOS3_HF_ARCH in entry.hf_architectures
        ]
        assert claimants == ["cosmos3_omni"]

    def test_only_the_colocated_topology_claims_the_diffusers_class_name(self):
        """Same for the model_index.json fallback."""
        claimants = [
            key
            for key, entry in OMNI_PIPELINES.items()
            if isinstance(entry, PipelineConfig) and entry.diffusers_class_name == COSMOS3_ARCH
        ]
        assert claimants == ["cosmos3_omni"]

    def test_model_index_autodetect_selects_the_colocated_topology(self, tmp_path):
        (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": COSMOS3_ARCH}), encoding="utf-8")

        model_type = StageConfigFactory.try_infer_model_type(model=str(tmp_path), trust_remote_code=False)

        assert model_type == "cosmos3_omni"

    def test_deploy_pipeline_key_selects_the_disagg_topology(self, tmp_path):
        """The one and only route into the 2-stage topology."""
        deploy_path = tmp_path / "deploy.yaml"
        deploy_path.write_text("pipeline: cosmos3_omni_disagg\n", encoding="utf-8")

        pipeline = StageConfigFactory.get_pipeline_config(
            model=str(tmp_path),
            trust_remote_code=False,
            deploy_config_path=str(deploy_path),
        )

        assert pipeline is COSMOS3_DISAGG_PIPELINE


class TestColocatedDeployConfig:
    def test_shipped_yaml_shape(self):
        deploy = _deploy(COLOCATED_YAML)

        # No `pipeline:` key -- the co-located topology is found from the
        # checkpoint's own model_type, exactly as it was before it was registered.
        assert deploy.pipeline is None
        assert deploy.async_chunk is False
        assert len(deploy.stages) == 1

    def test_parallel_degrees_agree_with_the_device_count(self):
        """The layout must satisfy the constraints its own degrees imply, whatever
        it is scaled to.

        The product of the parallel degrees has to equal WORLD (= len(devices)),
        and `apply_hsdp` additionally raises unless hsdp_replicate_size x
        hsdp_shard_size == WORLD. Asserting the invariants rather than the shipped
        numbers means rescaling the YAML needs no test edit, but a YAML that would
        raise at startup still fails here, without a GPU.
        """
        stage = _stage(_deploy(COLOCATED_YAML), 0)
        parallel = stage.engine_extras["parallel_config"]
        world = len(stage.devices.split(","))

        # Degrees omitted from the YAML keep their DiffusionParallelConfig default of 1.
        degree = 1
        for key in ("cfg_parallel_size", "ulysses_degree", "ring_degree", "tensor_parallel_size"):
            degree *= parallel.get(key, 1)
        assert degree == world

        if parallel.get("use_hsdp", False):
            assert parallel.get("hsdp_replicate_size", 1) * parallel["hsdp_shard_size"] == world

    def test_default_needs_no_collectives(self):
        """One card holds all 120.91 GiB of both towers, and HSDP on two cards is
        *slower* at 1024x1024 (439.4 vs 253.8 ms/step) because it all-gathers every
        layer from the peer each step. So the default ships without collectives;
        scaling out is for headroom (guardrails on, batching, >2048x2048)."""
        stage = _stage(_deploy(COLOCATED_YAML), 0)

        assert stage.devices == "0"
        assert stage.engine_extras["parallel_config"]["use_hsdp"] is False

    def test_yaml_keeps_the_guardrail_models_out_of_the_default_path(self):
        """Cosmos3's pre-process hook eager-loads the *gated* guardrail models at
        pipeline build time, so leaving them on makes this layout hard-fail
        wherever `cosmos-guardrail` is not installed. `serve --no-guardrails` and
        the offline example's `--extra-body` reach the same setting; `serve
        --guardrails` overrides this line back on."""
        stage = _stage(_deploy(COLOCATED_YAML), 0)
        assert stage.engine_extras["model_config"]["guardrails"] is False

    def test_yaml_pins_no_generation_parameters(self):
        """A deployment file must not decide generation behaviour.

        A pinned seed in particular would make every request that names no seed
        return the same image for the life of the server; the rest
        (num_inference_steps, guidance_scale, height, width) would shadow
        Cosmos3's own T2I defaults while looking like deployment config.
        """
        stage = _stage(_deploy(COLOCATED_YAML), 0)
        assert not stage.default_sampling_params

    def test_merges_into_a_single_image_stage(self):
        deploy = _deploy(COLOCATED_YAML)
        stages = merge_pipeline_deploy(COSMOS3_PIPELINE, deploy)

        assert len(stages) == 1
        assert stages[0].model_stage == "diffusion"
        assert stages[0].final_output_type == "image"
        # The merge must carry every YAML knob through to the stage: a dropped
        # `devices` is what silently collapses the layout onto one GPU.
        assert stages[0].yaml_runtime["devices"] == _stage(deploy, 0).devices
        assert stages[0].yaml_engine_args["model_config"]["guardrails"] is False


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

    def test_both_stages_run_the_same_tensor_parallel_size(self):
        """UND K/V is TP-sharded, so this is a correctness constraint, not a
        preference: the reasoner emits [B, S, num_kv_heads // tp, head_dim] and the
        generator's cross-attention consumes the shape *its own* TP size implies."""
        stages = [_stage(_deploy(DISAGG_YAML), i) for i in (0, 1)]
        tp_sizes = {s.engine_extras["parallel_config"]["tensor_parallel_size"] for s in stages}

        assert len(tp_sizes) == 1

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
