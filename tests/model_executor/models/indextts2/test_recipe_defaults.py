# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_indextts25_default_recipe_enables_validated_cfm_batching():
    recipe_path = Path(__file__).parents[4] / "vllm_omni" / "deploy" / "indextts2_5.yaml"
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    stage1 = next(stage for stage in recipe["stages"] if stage["stage_id"] == 1)

    assert stage1["hf_overrides"]["s2mel_cfm_batch_size"] == 4


@pytest.mark.parametrize("recipe_name", ["indextts2.yaml", "indextts2_low_latency.yaml", "indextts2_5.yaml"])
def test_indextts_stage0_recipes_disable_chunked_prefill(recipe_name):
    # IndexTTS prefill needs the whole prompt in one step; a chunked prompt
    # crashes the stage engine with "prefill embedding span mismatch".
    recipe_path = Path(__file__).parents[4] / "vllm_omni" / "deploy" / recipe_name
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    stage0 = next(stage for stage in recipe["stages"] if stage["stage_id"] == 0)

    assert stage0["enable_chunked_prefill"] is False
    assert stage0["max_num_batched_tokens"] >= stage0["max_model_len"]
