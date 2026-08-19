# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generation scheduling policy for IndexTTS continuous S2Mel decoding."""

from vllm.v1.request import Request, RequestStatus

from vllm_omni.core.sched.omni_generation_scheduler import (
    OmniGenerationScheduler,
)


class IndexTTS2GenerationScheduler(OmniGenerationScheduler):
    """Keep recurrent CFM state alive until the model reports completion."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        model_config = self.vllm_config.model_config
        hf_config = getattr(model_config, "hf_config", model_config)
        self._stepwise_generation = bool(getattr(hf_config, "stepwise_generation", False))

    def _cached_additional_information(self, request: Request) -> dict | None:
        if self._stepwise_generation:
            return None
        return super()._cached_additional_information(request)

    def _should_finish_generation_request(
        self,
        request: Request,
        model_finished_req_ids: set[str],
    ) -> bool:
        if not self._stepwise_generation:
            return super()._should_finish_generation_request(
                request,
                model_finished_req_ids,
            )
        return (
            request.status
            in (
                RequestStatus.FINISHED_ERROR,
                RequestStatus.FINISHED_STOPPED,
            )
            or request.request_id in model_finished_req_ids
        )
