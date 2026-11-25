# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Realtime-Venus reuses MiniCPM-o 4.5's Thinker, Talker and Code2Wav."""

from dataclasses import replace

from vllm_omni.model_executor.models.minicpmo_4_5.pipeline import MINICPMO_4_5_PIPELINE

REALTIME_VENUS_OMNI_PIPELINE = replace(
    MINICPMO_4_5_PIPELINE,
    model_type="realtime_venus_omni",
    # Use the shared execution classes; do not duplicate checkpoint kernels.
    default_deploy_config_name="realtime_venus_omni.yaml",
    hf_architectures=("RealtimeVenusOmni",),
    duplex_plugin="vllm_omni.model_executor.models.realtime_venus.duplex.RealtimeVenusDuplexPlugin",
)
