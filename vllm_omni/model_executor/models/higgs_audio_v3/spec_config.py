# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Speculative method config for the Higgs Audio v3 INT8 drafter.

vLLM loads this in every process, so it must not import the speculator.
"""

from vllm.v1.worker.gpu.spec_decode.speculator_config import SpeculatorConfig

HIGGS_AUDIO_SPEC_METHOD = "higgs_audio_int8"


class HiggsAudioSpeculatorConfig(SpeculatorConfig):
    speculator_class = "vllm_omni.model_executor.models.higgs_audio_v3.spec_decode.HiggsAudioSpeculator"
