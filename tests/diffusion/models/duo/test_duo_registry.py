# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contract tests for the Duo 1.0 integration wiring.

Duo reuses the MiniMax-H3 Realization blocks, so the value of these tests is
that the checkpoint keeps resolving to the H3-derived classes while Duo's own
stages (Intent / Realization / Codec) stay individually addressable.
"""

import re
from pathlib import Path

import pytest

from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata
from vllm_omni.diffusion.registry import (
    _DIFFUSION_MODELS,
    _DIFFUSION_PRE_PROCESS_FUNCS,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_duo_diffusion_models_are_registered():
    # The Realization and Codec stages are separate pipelines on one checkpoint.
    assert _DIFFUSION_MODELS["DuoPipeline"] == ("duo", "pipeline", "DuoPipeline")
    assert _DIFFUSION_MODELS["DuoCodecPipeline"] == ("duo", "codec", "DuoCodecPipeline")


def test_duo_both_stages_publish_video():
    # A chunk publishes an MP4 carrying both tracks, so the container is video.
    for arch in ("DuoPipeline", "DuoCodecPipeline"):
        assert get_diffusion_model_metadata(arch).final_output_type == "video"


def test_duo_pre_process_hooks_are_wired():
    # The hooks must be registered on the same arch keys as the pipelines, or a
    # Duo checkpoint would load a pipeline whose inputs are never prepared.
    assert _DIFFUSION_PRE_PROCESS_FUNCS["DuoPipeline"] == "get_duo_pre_process_func"
    assert _DIFFUSION_PRE_PROCESS_FUNCS["DuoCodecPipeline"] == "get_duo_codec_pre_process_func"


def test_duo_pipelines_expose_intent_realization_codec():
    duo = OMNI_PIPELINES["duo"]
    assert duo.model_arch == "DuoIntentForConditionalGeneration"
    stages = {s.model_stage: s for s in duo.stages}
    assert set(stages) == {"intent", "realization", "codec"}
    # Intent drives Realization, and Realization drives the codec: a stage may
    # only read sources with a lower id, so the order is part of the contract.
    assert stages["intent"].input_sources == ()
    assert stages["realization"].input_sources == (0,)
    assert stages["codec"].input_sources == (1,)
    # Exactly one stage publishes; selecting two would duplicate the response.
    assert [s.stage_id for s in duo.stages if s.final_output] == [2]


def test_duo_asr_pipeline_prepends_asr_stage():
    asr = OMNI_PIPELINES["duo_asr"]
    assert asr.duplex_plugin == OMNI_PIPELINES["duo"].duplex_plugin
    stages = sorted(asr.stages, key=lambda s: s.stage_id)
    assert stages[0].model_stage == "asr"
    # The ASR stage shifts the Duo stages down by one and feeds Intent from it.
    assert [s.model_stage for s in stages[1:]] == ["intent", "realization", "codec"]
    assert stages[1].input_sources == (0,)
    assert stages[2].input_sources == (1,)
    assert stages[3].input_sources == (2,)


def test_duo_registry_entries_resolve_to_real_modules():
    # The literal comparisons above cannot catch a tuple that is wrong in the
    # registry and the test in the same way. Resolve each entry the way the
    # registry does, so a typo in mod_folder/mod_relname/cls_name fails here.
    for arch in ("DuoPipeline", "DuoCodecPipeline"):
        mod_folder, mod_relname, cls_name = _DIFFUSION_MODELS[arch]
        module_path = Path("vllm_omni/diffusion/models") / mod_folder / f"{mod_relname}.py"
        assert module_path.is_file(), f"{arch} points at a missing module: {module_path}"
        assert re.search(rf"^class {cls_name}\b", module_path.read_text(), re.MULTILINE), (
            f"{arch} points at a class {cls_name} that the module does not define"
        )
    # The pre-process hooks must be real module-level functions in those modules.
    for arch, func_name in _DIFFUSION_PRE_PROCESS_FUNCS.items():
        if not arch.startswith("Duo"):
            continue
        _, mod_relname, _ = _DIFFUSION_MODELS[arch]
        module_path = Path("vllm_omni/diffusion/models/duo") / f"{mod_relname}.py"
        assert re.search(rf"^def {func_name}\b", module_path.read_text(), re.MULTILINE), (
            f"{arch} pre-process hook {func_name} is not defined in {module_path}"
        )


def test_duo_stages_reference_registered_pipelines():
    # A stage whose model_arch is not registered would only fail at engine
    # construction, i.e. after the checkpoint has been loaded. The ASR stage
    # deliberately reuses vLLM's own Qwen3-ASR arch, so both registries count.
    from vllm.model_executor.models.registry import _VLLM_MODELS

    from vllm_omni.model_executor.models.registry import _OMNI_MODELS

    registered = set(_DIFFUSION_MODELS) | set(_OMNI_MODELS) | set(_VLLM_MODELS)
    for name in ("duo", "duo_asr"):
        for stage in OMNI_PIPELINES[name].stages:
            assert stage.model_arch in registered, (
                f"{name} stage {stage.stage_id} ({stage.model_stage}) uses unregistered arch {stage.model_arch}"
            )
