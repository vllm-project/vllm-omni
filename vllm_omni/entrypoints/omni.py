from __future__ import annotations

import copy
import time
import uuid
from collections.abc import Callable, Generator, Iterable, Sequence
from typing import TYPE_CHECKING, Literal, overload

from tqdm.auto import tqdm
from vllm.logger import init_logger
from vllm.sampling_params import RequestOutputKind

from vllm_omni.entrypoints.client_request_state import ClientRequestState
from vllm_omni.entrypoints.omni_base import OmniBase
from vllm_omni.metrics.stats import OrchestratorAggregator as OrchestratorMetrics
from vllm_omni.outputs import OmniRequestOutput

if TYPE_CHECKING:
    from vllm_omni.inputs.data import OmniPromptType, OmniSamplingParams

logger = init_logger(__name__)


class Omni(OmniBase):
    """Synchronous entrypoint for offline generation."""

    def _set_final_only_for_llm_stages(
        self,
        sampling_params_list: Sequence[OmniSamplingParams],
    ) -> list[OmniSamplingParams]:
        """Return per-stage params with LLM stages forced to FINAL_ONLY."""
        effective_params: list[OmniSamplingParams] = []
        for stage_id, params in enumerate(sampling_params_list):
            sp = copy.deepcopy(params)
            stage_meta = self.engine.get_stage_metadata(stage_id)
            if stage_meta.get("stage_type") != "diffusion" and hasattr(sp, "output_kind"):
                sp.output_kind = RequestOutputKind.FINAL_ONLY
            effective_params.append(sp)
        return effective_params

    @overload
    def generate(
        self,
        prompts: OmniPromptType | Sequence[OmniPromptType],
        sampling_params_list: OmniSamplingParams | Sequence[OmniSamplingParams] | None = None,
        *,
        py_generator: Literal[True],
        use_tqdm: bool | Callable[..., tqdm] = True,
    ) -> Generator[OmniRequestOutput, None, None]: ...

    @overload
    def generate(
        self,
        prompts: OmniPromptType | Sequence[OmniPromptType],
        sampling_params_list: OmniSamplingParams | Sequence[OmniSamplingParams] | None = None,
        *,
        py_generator: Literal[False] = False,
        use_tqdm: bool | Callable[..., tqdm] = True,
    ) -> list[OmniRequestOutput]: ...

    def generate(
        self,
        prompts: OmniPromptType | Sequence[OmniPromptType],
        sampling_params_list: OmniSamplingParams | Sequence[OmniSamplingParams] | None = None,
        *,
        py_generator: bool = False,
        use_tqdm: bool | Callable[..., tqdm] = True,
    ) -> Generator[OmniRequestOutput, None, None] | list[OmniRequestOutput]:
        sampling_params_list = self.resolve_sampling_params_list(sampling_params_list)
        try:
            if py_generator:
                return self._run_generation_with_generator(prompts, sampling_params_list, use_tqdm)
            return list(self._run_generation(prompts, sampling_params_list, use_tqdm))
        except Exception as e:
            logger.exception("[Omni] Failed to run generation: %s", e)
            self.close()
            raise

    def _run_generation_with_generator(
        self,
        prompts: OmniPromptType | Sequence[OmniPromptType],
        sampling_params_list: Sequence[OmniSamplingParams],
        use_tqdm: bool | Callable[..., tqdm] = True,
    ) -> Generator[OmniRequestOutput, None, None]:
        gen = self._run_generation(prompts, sampling_params_list, use_tqdm)
        try:
            yield from gen
        finally:
            self.close()

    def _run_generation(
        self,
        prompts: OmniPromptType | Sequence[OmniPromptType],
        sampling_params_list: Sequence[OmniSamplingParams],
        use_tqdm: bool | Callable[..., tqdm] = True,
    ) -> Generator[OmniRequestOutput, None, None]:
        try:
            sampling_params_list = self._set_final_only_for_llm_stages(sampling_params_list)

            if isinstance(prompts, str) or not isinstance(prompts, Sequence):
                request_prompts: list[OmniPromptType] = [prompts]
            else:
                request_prompts = list(prompts)

            if not request_prompts:
                return

            request_ids = [f"{i}_{uuid.uuid4()}" for i in range(len(request_prompts))]
            req_start_ts: dict[str, float] = {}
            wall_start_ts = time.time()
            req_final_stage_ids: dict[str, int] = {}

            for req_id, prompt in zip(request_ids, request_prompts):
                prompt_modalities = prompt.get("modalities", None) if isinstance(prompt, dict) else None
                final_stage_id = self._compute_final_stage_id(prompt_modalities)
                req_final_stage_ids[req_id] = final_stage_id

                metrics = OrchestratorMetrics(
                    self.num_stages,
                    self.log_stats,
                    wall_start_ts,
                    final_stage_id,
                )
                req_state = ClientRequestState(req_id)
                req_state.metrics = metrics
                self.request_states[req_id] = req_state

                self.engine.add_request(
                    request_id=req_id,
                    prompt=prompt,
                    sampling_params_list=sampling_params_list,
                    final_stage_id=final_stage_id,
                )
                submit_ts = time.time()
                req_state.metrics.stage_first_ts[0] = submit_ts
                req_start_ts[req_id] = submit_ts

            active_reqs = set(request_ids)
            pbar = None
            if use_tqdm:
                tqdm_func = use_tqdm if callable(use_tqdm) else tqdm
                pbar = tqdm_func(total=len(request_ids), desc="Processed prompts", dynamic_ncols=True)

            while active_reqs:
                msg = self.engine.try_get_output()

                should_continue, req_id, stage_id, req_state = self._handle_output_message(msg)
                if should_continue:
                    continue

                if req_id not in active_reqs:
                    logger.warning("[Omni] Received output for unknown/finished request_id=%s", req_id)
                    continue

                if req_state.metrics is None:
                    continue
                output_to_yield = self._process_single_result(
                    result=msg,
                    stage_id=stage_id,
                    metrics=req_state.metrics,
                    req_start_ts=req_start_ts,
                    wall_start_ts=wall_start_ts,
                    final_stage_id_for_e2e=req_final_stage_ids[req_id],
                )
                if output_to_yield is not None:
                    yield output_to_yield

                if msg.get("finished"):
                    active_reqs.discard(req_id)
                    if pbar is not None:
                        pbar.update(1)
                    self._log_summary_and_cleanup(req_id)
        except Exception:
            if "active_reqs" in locals() and active_reqs:
                self.abort(list(active_reqs))
            raise
        finally:
            if "pbar" in locals() and pbar is not None:
                pbar.close()

    def abort(self, request_id: str | Iterable[str]) -> None:
        request_ids = [request_id] if isinstance(request_id, str) else list(request_id)
        self.engine.abort(request_ids)
        for req_id in request_ids:
            self.request_states.pop(req_id, None)
        if self.log_stats:
            logger.info("[Omni] Aborted request(s) %s", ",".join(request_ids))
