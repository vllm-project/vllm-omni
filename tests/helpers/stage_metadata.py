# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""StageMetadata builder for unit tests."""

from __future__ import annotations

import copy
import dataclasses

from vllm.sampling_params import SamplingParams

from vllm_omni.engine.stage_client import StageClient
from vllm_omni.engine.stage_init_utils import StageMetadata

# StageMetadata defaults for tests, whose values may be overridden on per-test deep copies.
_DEFAULT_STAGE_METADATA = StageMetadata(
    stage_id=0,
    stage_type="llm",
    engine_output_type=None,
    is_comprehension=False,
    requires_multimodal_data=False,
    engine_input_source=[],
    final_output=False,
    final_output_type=None,
    default_sampling_params=SamplingParams(),
    custom_process_input_func=None,
    model_stage=None,
    runtime_cfg=None,
)


def make_stage_metadata(**kwargs) -> StageMetadata:
    """Create a StageMetadata object for testing."""
    metadata = copy.deepcopy(_DEFAULT_STAGE_METADATA)
    return dataclasses.replace(metadata, **kwargs)


def stage_metadata_for_client(stage_id: int, client: StageClient) -> StageMetadata:
    """Return the metadata of the stage a client is a replica of."""
    return make_stage_metadata(
        stage_id=stage_id,
        stage_type=client.stage_type,
        final_output=client.final_output,
        final_output_type=client.final_output_type,
        requires_multimodal_data=client.requires_multimodal_data,
        engine_input_source=client.engine_input_source,
        custom_process_input_func=client.custom_process_input_func,
        model_stage=client.model_stage,
    )
