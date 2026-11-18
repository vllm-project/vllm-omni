# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-omni project
"""Duo deploy presets must reach the runtime config they appear to configure.

Two failures motivated this test, and neither is visible by reading the YAML:

1. A stage's Realization weights were pinned to one card while the preset said
   `tensor_parallel_size: 4`. The preset value was fine; what actually decides
   it is the resolved diffusion config, so only asserting the resolved value
   catches a regression.
2. The Intent stage's `vocab_size` disagreed with the released tokenizer, which
   `intent.py` rejects at load time.

These tests read the real presets and resolve them through the same builders the
engine uses, so a preset that "looks right" but resolves differently fails.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from vllm_omni.config import omni_config as omni_config_module
from vllm_omni.config.omni_config import extract_diffusion_stage_config_kwargs
from vllm_omni.config.stage_config import load_deploy_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

DEPLOY_DIR = Path("vllm_omni/deploy")
DUO_PRESETS = ("duo_causal_4gpu.yaml", "duo_dmd.yaml", "duo_tp8_shared.yaml")


def _resolved_stage_engine(stage_id: int, preset: str) -> dict:
    """Deploy YAML -> stage engine args -> diffusion config (config layer only)."""
    deploy = load_deploy_config(str(DEPLOY_DIR / preset))
    stage = next(s for s in deploy.stages if s.stage_id == stage_id)
    engine = omni_config_module._stage_engine_overrides(stage)
    return extract_diffusion_stage_config_kwargs(engine, stage_id=stage_id, include_engine_adapter_metadata=True)


def _resolved_diffusion_tp(stage_id: int, preset: str) -> int | None:
    """TP as the diffusion runtime sees it: via the parallel-config builder.

    Reading the raw engine args is not enough: a flat `tensor_parallel_size:`
    and a nested `parallel_config:` block are normalised by different code, and
    only this builder applies both.
    """
    deploy = load_deploy_config(str(DEPLOY_DIR / preset))
    stage = next(s for s in deploy.stages if s.stage_id == stage_id)
    engine = omni_config_module._stage_engine_overrides(stage)
    parallel = omni_config_module._build_parallel_config(
        deploy, engine, omni_config_module.OmniStageDiffusionParallelConfig
    )
    return parallel.tensor_parallel_size


def _duo_diffusion_stages(preset: str) -> list[int]:
    raw = yaml.safe_load((DEPLOY_DIR / preset).read_text(encoding="utf-8"))
    return [
        entry["stage_id"]
        for entry in raw["stages"]
        if "DuoModelRunner" in str(entry.get("diffusion_model_runner_cls", ""))
    ]


@pytest.mark.parametrize("preset", DUO_PRESETS)
def test_preset_declares_the_documented_diffusion_tp(preset):
    # The config layer must agree with the YAML. Scope: it does NOT prove the
    # engine builds that config. Measured on 4xA100 with the released
    # causal-base bundle, the `Omni(model=..., deploy_config_path=...,
    # model_class_name=..., step_execution=...)` path resolved a single stage
    # with tensor_parallel_size=1 (66 GB on device 0) even though both the YAML
    # and this config layer said 4; only passing `tensor_parallel_size=4` to
    # `Omni(...)` produced four 16.6 GiB workers. The presets therefore keep the
    # value AND callers must pass it explicitly. Pinning it here still catches
    # a preset drifting away from the value the config layer would resolve.
    raw = yaml.safe_load((DEPLOY_DIR / preset).read_text(encoding="utf-8"))
    raw_by_id = {entry["stage_id"]: entry for entry in raw["stages"]}

    stage_ids = _duo_diffusion_stages(preset)
    assert stage_ids, f"{preset} should declare at least one Duo diffusion stage"

    for stage_id in stage_ids:
        entry = raw_by_id[stage_id]
        declared = entry.get("tensor_parallel_size")
        if declared is None:
            declared = (entry.get("parallel_config") or {}).get("tensor_parallel_size")
        if declared is None:
            # Omitted means 1; nothing to cross-check.
            continue
        resolved = _resolved_diffusion_tp(stage_id, preset)
        assert resolved == declared, (
            f"{preset} stage {stage_id} declares tensor_parallel_size={declared} "
            f"but the diffusion runtime resolves {resolved!r}"
        )


@pytest.mark.parametrize("preset", DUO_PRESETS)
def test_duo_stage_options_reach_the_stage_model_config(preset):
    # `duo` / `duo_codec` options are read from `model_config`; a preset that
    # declares them somewhere else would silently deploy with defaults.
    raw = yaml.safe_load((DEPLOY_DIR / preset).read_text(encoding="utf-8"))
    for entry in raw["stages"]:
        for key in ("duo", "duo_codec"):
            block = (entry.get("model_config") or {}).get(key)
            if not block:
                continue
            resolved = _resolved_stage_engine(entry["stage_id"], preset)
            assert resolved.get("model_config", {}).get(key) == block, (
                f"{preset} stage {entry['stage_id']} declares model_config.{key} "
                f"but it does not reach the diffusion config"
            )


@pytest.mark.parametrize("preset", DUO_PRESETS)
def test_presets_do_not_pin_a_vocab_size_the_tokenizer_cannot_supply(preset):
    # Released intent_adapter/tokenizer.json has 151643 base entries plus 38
    # added tokens, i.e. 151681 usable ids. `intent.py` rejects a mismatch, so a
    # preset declaring a larger vocab fails at load.
    raw = yaml.safe_load((DEPLOY_DIR / preset).read_text(encoding="utf-8"))
    expected = 151643 + 38
    for entry in raw["stages"]:
        declared = ((entry.get("hf_overrides") or {}).get("text_config") or {}).get("vocab_size")
        if declared is None:
            continue
        assert declared == expected, (
            f"{preset} stage {entry['stage_id']} declares vocab_size={declared}, "
            f"but the released tokenizer only supplies {expected} ids"
        )


def test_presets_are_loadable_yaml_with_unique_stage_ids():
    # Cheap structural guard: the parametrised tests above assume readable YAML
    # with one entry per stage id.
    for preset in DUO_PRESETS:
        raw = yaml.safe_load((DEPLOY_DIR / preset).read_text(encoding="utf-8"))
        ids = [entry["stage_id"] for entry in raw["stages"]]
        assert ids == sorted(set(ids)), f"{preset} has duplicate or unsorted stage ids: {ids}"


def test_causal_4gpu_preset_asks_for_four_way_realization():
    # The smallest topology that fits the 66 GB backbone on 4xA100-80GB; the
    # number is the point of the preset, so pin it rather than infer it.
    raw = yaml.safe_load((DEPLOY_DIR / "duo_causal_4gpu.yaml").read_text(encoding="utf-8"))
    realizations = [
        entry for entry in raw["stages"] if "DuoModelRunner" in str(entry.get("diffusion_model_runner_cls", ""))
    ]
    assert len(realizations) == 1
    assert realizations[0]["tensor_parallel_size"] == 4
    assert realizations[0]["devices"] == "0,1,2,3"


def test_manifest_adapter_modules_handles_all_released_variants(tmp_path):
    # causal-base records its LoRA under models.student, the DMD release under
    # models.generator, and the quantized release ships a merged model.
    from vllm_omni.diffusion.models.duo.pipeline import _manifest_adapter_modules

    def write(manifest: dict) -> Path:
        path = tmp_path / "complete.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    student = write({"models": {"student": {"mode": "lora", "lora": {"modules": ["proj_in"]}}}})
    assert _manifest_adapter_modules(student) == ["proj_in"]

    generator = write({"models": {"generator": {"stage": "dmd", "lora": {"modules": ["ff.net.2"]}}}})
    assert _manifest_adapter_modules(generator) == ["ff.net.2"]

    merged = write({"models": {"realization": {"mode": "full"}}})
    assert _manifest_adapter_modules(merged) is None
