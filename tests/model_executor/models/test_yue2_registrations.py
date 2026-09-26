# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Wiring tests: everything a fresh checkout must agree on for YuE2 to serve.

The integration spans four hand-edited registration points (pipeline registry,
model registry, HF-config registry, deploy YAML) plus one config class that
synthesizes Qwen3 backbone fields the checkpoint omits. None of the previous
test files can see a typo between two files; these tests assert the files
agree with each other:

* the pipeline's declared arch exists in the model registry and is importable;
* the deploy YAML pins that same arch, skips the tokenizer build the
  checkpoint cannot satisfy, and carries the extra_args keys whose presence
  switches on per-request sampling args;
* ``Yue2Config`` fills the Qwen3 fields the AR backbone reads (a gap that
  previously crashed engine init) while keeping the acoustic fields.

These import ``vllm_omni`` and ``vllm``: they run wherever the package is
installed (CI / dev boxes), not standalone.
"""

from pathlib import Path

import pytest
import yaml

import vllm_omni
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.config.stage_config import StageExecutionType
from vllm_omni.model_executor.models.registry import _OMNI_MODELS
from vllm_omni.model_executor.models.yue2.constants import (
    ABC_END,
    CONTEXT,
    KEY_PHASE,
    MUSIC_END,
)
from vllm_omni.transformers_utils.configs.yue2 import Yue2Config

ARCH = "Yue2ForCausalLM"

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_pipeline_is_registered_as_single_stage_ar():
    pipeline = OMNI_PIPELINES["yue2"]
    assert len(pipeline.stages) == 1
    (stage,) = pipeline.stages
    assert stage.stage_id == 0
    assert stage.execution_type == StageExecutionType.LLM_AR
    assert stage.final_output is True
    assert stage.final_output_type == "audio"
    assert stage.owns_tokenizer is True
    # Both phase ends, so a request can never run past its own phase's stop.
    assert stage.sampling_constraints["stop_token_ids"] == [ABC_END, MUSIC_END]
    assert stage.sampling_constraints["detokenize"] is False


def test_declared_arch_resolves_through_the_model_registry():
    assert OMNI_PIPELINES["yue2"].model_arch == ARCH
    assert _OMNI_MODELS[ARCH] == ("yue2", "yue2", ARCH)


def test_model_module_and_class_exist():
    import importlib

    module = importlib.import_module("vllm_omni.model_executor.models.yue2.yue2")
    assert hasattr(module, ARCH)


def test_hf_config_is_registered_for_autoconfig():
    from transformers import AutoConfig

    config = AutoConfig.for_model("yue2")
    assert isinstance(config, Yue2Config)


def test_config_synthesizes_the_qwen3_backbone_fields():
    # The checkpoint's config.json omits these; Qwen3Model construction reads
    # every one of them (engine init crashed on hidden_act before this).
    config = Yue2Config()
    assert config.hidden_act == "silu"
    assert config.attention_bias is False
    assert config.attention_dropout == 0.0
    assert config.use_sliding_window is False
    assert config.sliding_window is None
    assert config.max_window_layers == 0


def test_config_keeps_the_acoustic_fields():
    config = Yue2Config(latent_dim=64, max_latent_frames=24576, timestep_shift=1.0)
    assert config.latent_dim == 64
    assert config.max_latent_frames == 24576
    assert config.timestep_shift == 1.0
    assert config.model_type == "yue2"


class TestDeployYaml:
    doc: dict

    @classmethod
    def setup_class(cls):
        path = Path(vllm_omni.__file__).parent / "deploy" / "yue2.yaml"
        cls.doc = yaml.safe_load(path.read_text())

    def test_pipeline_key_and_single_stage(self):
        assert self.doc["pipeline"] == "yue2"
        (stage,) = self.doc["stages"]
        assert stage["stage_id"] == 0

    def test_architectures_pin_matches_the_registered_arch(self):
        (stage,) = self.doc["stages"]
        assert stage["hf_overrides"]["architectures"] == [ARCH]

    def test_tokenizer_build_is_skipped(self):
        # The checkpoint ships only qwen.tiktoken; an HF tokenizer build on it
        # fails. Every prompt arrives as token ids.
        (stage,) = self.doc["stages"]
        assert stage["skip_tokenizer_init"] is True
        assert stage.get("tokenizer")

    def test_default_sampling_params_carry_extra_args(self):
        # The presence of the extra_args keys is what turns on
        # has_sampling_extra_args and threads per-request yue2_* args into the
        # model's forward. Without them the sampler runs on preset defaults
        # and every song ignores its request's phase and seed.
        (stage,) = self.doc["stages"]
        extra_args = stage["default_sampling_params"]["extra_args"]
        assert KEY_PHASE in extra_args

    def test_context_window_matches_the_checkpoint(self):
        (stage,) = self.doc["stages"]
        assert stage["max_model_len"] == CONTEXT
