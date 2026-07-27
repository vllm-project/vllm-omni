# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

# MagiHuman is a text -> video+audio model. Its model-specific knobs are read
# from sampling_params.extra_args in pipeline_magi_human.py; declaring them here
# routes request `extra_body` fields into OmniDiffusionSamplingParams.extra_args
# so the model can be driven through the standard task example.
MAGI_HUMAN_EXTRA_BODY_PARAMS = frozenset(
    {
        "seconds",
        "audio_path",
        "image_path",
        "sr_height",
        "sr_width",
        "sr_num_inference_steps",
    }
)
MAGI_HUMAN_EXTRA_OUTPUT_PARAMS = frozenset()
