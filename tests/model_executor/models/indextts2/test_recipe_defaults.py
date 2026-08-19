# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import yaml


def test_indextts25_default_recipe_enables_validated_cfm_batching():
    recipe_path = Path(__file__).parents[4] / "vllm_omni" / "deploy" / "indextts2_5.yaml"
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    stage1 = next(stage for stage in recipe["stages"] if stage["stage_id"] == 1)

    assert stage1["hf_overrides"]["s2mel_cfm_batch_size"] == 4


def test_indextts25_continuous_recipe_enables_stepwise_cfm():
    recipe_path = Path(__file__).parents[4] / "vllm_omni" / "deploy" / "indextts2_5_continuous.yaml"
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    stage0 = next(stage for stage in recipe["stages"] if stage["stage_id"] == 0)
    stage1 = next(stage for stage in recipe["stages"] if stage["stage_id"] == 1)
    overrides = stage1["hf_overrides"]
    stage0_overrides = stage0["hf_overrides"]

    assert stage0["max_num_seqs"] == 32
    assert stage1["max_num_seqs"] == 32
    assert stage1["worker_cls"] == ("vllm_omni.model_executor.models.indextts2.runner.IndexTTS2GenerationWorker")
    assert overrides["s2mel_cfm_batch_size"] == 32
    assert overrides["stepwise_generation"] is True
    assert overrides["s2mel_continuous_max_padding_ratio"] == 1.15
    assert stage0_overrides["stage0_conditioning_prefix_cache"] is True
    assert stage0_overrides["stage0_conditioning_prefix_cache_max_bytes"] == 64 * 1024**2
    assert overrides["s2mel_vocoder_fused_activation"] is True
