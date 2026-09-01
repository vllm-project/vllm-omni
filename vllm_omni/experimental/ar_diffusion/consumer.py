# SPDX-License-Identifier: Apache-2.0
"""AsyncOmni consumer for typed realtime AR-Diffusion ticks."""

from __future__ import annotations

import copy
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import Any, Protocol

from vllm_omni.experimental.ar_diffusion.tick_protocol import (
    ARDiffusionChunkMetadata,
    ARDiffusionTickRequest,
)
from vllm_omni.inputs.data import (
    OmniDiffusionSamplingParams,
    OmniPromptType,
    OmniSamplingParams,
)
from vllm_omni.outputs import OmniRequestOutput

AR_DIFFUSION_OUTPUT_METADATA_KEY = "ar_diffusion"


class ARDiffusionGenerateClient(Protocol):
    """Internal subset of AsyncOmni used to execute one tick."""

    def generate(
        self,
        prompt: OmniPromptType,
        *,
        request_id: str,
        sampling_params_list: Sequence[OmniSamplingParams],
        output_modalities: list[str] | None = None,
    ) -> AsyncIterator[OmniRequestOutput]: ...


def _metadata_mapping(output: OmniRequestOutput) -> Mapping[str, Any]:
    multimodal_output = output.multimodal_output
    if not isinstance(multimodal_output, Mapping):
        raise ValueError("AR-Diffusion output must contain a multimodal output mapping.")
    metadata = multimodal_output.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("AR-Diffusion output must contain multimodal_output.metadata.")
    value = metadata.get(AR_DIFFUSION_OUTPUT_METADATA_KEY)
    if not isinstance(value, Mapping):
        raise ValueError(f"AR-Diffusion output metadata must contain {AR_DIFFUSION_OUTPUT_METADATA_KEY!r}.")
    return value


def _parse_chunk_metadata(output: OmniRequestOutput) -> ARDiffusionChunkMetadata:
    value = _metadata_mapping(output)
    session_id = value.get("session_id")
    request_id = value.get("request_id")
    chunk_index = value.get("chunk_index")
    applied_event_ids = value.get("applied_event_ids", ())
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("AR-Diffusion output metadata session_id must be a non-empty string.")
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("AR-Diffusion output metadata request_id must be a non-empty string.")
    if isinstance(chunk_index, bool) or not isinstance(chunk_index, int) or chunk_index < 0:
        raise ValueError("AR-Diffusion output metadata chunk_index must be a non-negative integer.")
    if not isinstance(applied_event_ids, Sequence) or isinstance(applied_event_ids, (str, bytes, bytearray)):
        raise ValueError("AR-Diffusion output metadata applied_event_ids must be a sequence.")
    event_ids = tuple(applied_event_ids)
    if any(isinstance(event_id, bool) or not isinstance(event_id, int) or event_id < 0 for event_id in event_ids):
        raise ValueError("AR-Diffusion output metadata applied_event_ids must contain non-negative integers.")
    if event_ids != tuple(sorted(set(event_ids))):
        raise ValueError("AR-Diffusion output metadata applied_event_ids must be unique and strictly increasing.")
    return ARDiffusionChunkMetadata(
        session_id=session_id,
        request_id=request_id,
        chunk_index=chunk_index,
        applied_event_ids=event_ids,
    )


class ARDiffusionOmniTickConsumer:
    """Converts a typed tick into one AsyncOmni diffusion request.

    ``prompt_provider`` packages session initialization data such as an initial
    image into the normal Omni prompt. Control ``track/schema/data`` remains in
    the typed tick and is not interpreted by this class.
    """

    def __init__(
        self,
        engine_client: ARDiffusionGenerateClient,
        *,
        prompt_provider: Callable[[ARDiffusionTickRequest], OmniPromptType],
        sampling_params_list: Sequence[OmniSamplingParams],
        diffusion_stage_id: int,
        output_modalities: Sequence[str] | None = None,
    ) -> None:
        templates = tuple(sampling_params_list)
        if not templates:
            raise ValueError("sampling_params_list must not be empty.")
        if (
            isinstance(diffusion_stage_id, bool)
            or not isinstance(diffusion_stage_id, int)
            or not 0 <= diffusion_stage_id < len(templates)
        ):
            raise ValueError("diffusion_stage_id must index sampling_params_list.")
        if not isinstance(templates[diffusion_stage_id], OmniDiffusionSamplingParams):
            raise TypeError("The selected diffusion stage must use OmniDiffusionSamplingParams.")
        stage_pools = getattr(getattr(engine_client, "engine", None), "stage_pools", None)
        if stage_pools is None or not 0 <= diffusion_stage_id < len(stage_pools):
            raise ValueError(
                "AR-Diffusion realtime execution requires an initialized AsyncOmni "
                "engine with a matching diffusion stage pool."
            )
        replica_count = getattr(stage_pools[diffusion_stage_id], "num_replicas", None)
        if isinstance(replica_count, bool) or not isinstance(replica_count, int) or replica_count < 1:
            raise ValueError("The selected AR-Diffusion stage must expose a positive num_replicas.")
        if replica_count != 1:
            raise ValueError(
                "AR-Diffusion realtime sessions currently require exactly one "
                "replica for the selected diffusion stage; session-affine "
                f"routing is not implemented (got num_replicas={replica_count})."
            )
        self._engine_client = engine_client
        self._prompt_provider = prompt_provider
        self._sampling_params_templates = templates
        self._diffusion_stage_id = diffusion_stage_id
        self._output_modalities = list(output_modalities) if output_modalities is not None else None

    def _sampling_params_for_tick(
        self,
        tick: ARDiffusionTickRequest,
    ) -> list[OmniSamplingParams]:
        params = [copy.copy(template) for template in self._sampling_params_templates]
        diffusion_params = params[self._diffusion_stage_id]
        assert isinstance(diffusion_params, OmniDiffusionSamplingParams)
        diffusion_params.extra_args = {
            **(diffusion_params.extra_args or {}),
            **tick.to_extra_args(),
        }
        return params

    async def execute_tick(self, tick: ARDiffusionTickRequest) -> OmniRequestOutput:
        prompt = self._prompt_provider(tick)
        final_output: OmniRequestOutput | None = None
        async for output in self._engine_client.generate(
            prompt,
            request_id=tick.request_id,
            sampling_params_list=self._sampling_params_for_tick(tick),
            output_modalities=self._output_modalities,
        ):
            if output.stage_id != self._diffusion_stage_id:
                continue
            if output.finished:
                final_output = output

        if final_output is None:
            raise RuntimeError(
                f"AR-Diffusion tick {tick.request_id!r} completed without a final "
                f"output from stage {self._diffusion_stage_id}."
            )
        if final_output.error:
            raise RuntimeError(f"AR-Diffusion tick {tick.request_id!r} failed: {final_output.error}")
        if final_output.request_id != tick.request_id:
            raise ValueError("AR-Diffusion output request_id does not match the submitted tick request_id.")
        # Parse here so malformed/missing standard metadata fails at the engine
        # boundary, before the session attempts to commit accepted events.
        self.chunk_metadata(final_output)
        return final_output

    def chunk_metadata(self, output: OmniRequestOutput) -> ARDiffusionChunkMetadata:
        return _parse_chunk_metadata(output)
