"""Unified stage-local runtime abstraction for vLLM-Omni."""

from __future__ import annotations

import asyncio
import time as _time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreOutputs

from vllm_omni.engine.stage_client import (
    StagePoolClient,
    StagePoolDiffusionClient,
    StagePoolLLMClient,
)
from vllm_omni.metrics.stats import StageRequestStats as StageRequestMetrics
from vllm_omni.metrics.stats import StageStats
from vllm_omni.metrics.utils import count_tokens_from_outputs

if TYPE_CHECKING:
    from vllm_omni.engine.orchestrator import OrchestratorRequestState

logger = init_logger(__name__)


@dataclass
class _ReplicaMetrics:
    """Per-replica metrics accumulators owned by a stage pool."""

    batch_seq: int = 0
    agg_total_tokens: int = 0
    agg_total_gen_time_ms: float = 0.0


class StagePool:
    """Replicas of one logical stage with RR + affinity selection."""

    def __init__(
        self,
        stage_id: int,
        clients: StagePoolClient | list[StagePoolClient],
        *,
        output_processor: Any = None,
        stage_vllm_config: Any = None,
    ) -> None:
        if isinstance(clients, list):
            normalized_clients: list[StagePoolClient] = list(clients)
        else:
            normalized_clients = [clients]

        if not normalized_clients:
            raise ValueError(f"StagePool for stage {stage_id} has no replicas")
        self.stage_id = stage_id
        self.clients: list[StagePoolClient] = normalized_clients
        self._output_processor = output_processor
        self._stage_vllm_config = stage_vllm_config
        self._next_replica_id = 0
        self._request_bindings: dict[str, int] = {}
        self._replica_metrics: list[_ReplicaMetrics] = [_ReplicaMetrics() for _ in self.clients]

    # ---- Stage-level properties ----

    @property
    def num_replicas(self) -> int:
        return len(self.clients)

    @property
    def stage_type(self) -> str | None:
        return self.stage_client.stage_type

    @property
    def final_output(self) -> bool:
        return self.clients[0].final_output

    @property
    def stage_client(self) -> StagePoolClient:
        return self.clients[0]

    @property
    def llm_stage_client(self) -> StagePoolLLMClient:
        return cast(StagePoolLLMClient, self.stage_client)

    @property
    def stage_vllm_config(self) -> Any:
        return self._stage_vllm_config

    @property
    def output_processor(self) -> Any:
        return self._output_processor

    # ---- Route binding lifecycle ----

    def get_bound_replica_id(self, request_id: str) -> int | None:
        """Return the currently bound replica id for *request_id* if present."""
        return self._request_bindings.get(request_id)

    def get_bound_client(self, request_id: str) -> StagePoolClient | None:
        """Return the currently bound client for *request_id* if present."""
        replica_id = self.get_bound_replica_id(request_id)
        if replica_id is None:
            return None
        return self.clients[replica_id]

    def get_bound_llm_client(self, request_id: str) -> StagePoolLLMClient | None:
        """Return the currently bound LLM client for *request_id* if present."""
        client = self.get_bound_client(request_id)
        if client is None:
            return None
        return cast(StagePoolLLMClient, client)

    def release_binding(self, request_id: str) -> None:
        """Drop the route binding for *request_id* in this stage."""
        self._request_bindings.pop(request_id, None)

    def release_bindings(self, request_ids: list[str]) -> None:
        """Drop route bindings for the given request ids in this stage."""
        for request_id in request_ids:
            self.release_binding(request_id)

    def select_replica_id(
        self,
        request_id: str,
        *,
        affinity_request_id: str | None = None,
    ) -> int:
        """Pick a replica id for *request_id* and cache the choice."""
        cached = self.get_bound_replica_id(request_id)
        if cached is not None:
            return cached

        chosen = self.get_bound_replica_id(affinity_request_id) if affinity_request_id is not None else None
        if chosen is None:
            if self.num_replicas == 1:
                chosen = 0
            else:
                chosen = self._next_replica_id
                self._next_replica_id = (self._next_replica_id + 1) % self.num_replicas

        self._request_bindings[request_id] = chosen
        return chosen

    def _llm_client(self, replica_id: int) -> StagePoolLLMClient:
        return cast(StagePoolLLMClient, self.clients[replica_id])

    def _diffusion_client(self, replica_id: int) -> StagePoolDiffusionClient:
        return cast(StagePoolDiffusionClient, self.clients[replica_id])

    # ---- Metrics ----

    def build_stage_metrics(
        self,
        request_outputs: list[Any],
        *,
        submit_ts: float,
        replica_id: int,
    ) -> StageRequestMetrics:
        """Build stage metrics for outputs produced on one replica."""
        now = _time.time()
        stage_gen_time_ms = (now - submit_ts) * 1000.0

        num_tokens_out = count_tokens_from_outputs(request_outputs)
        num_tokens_in = 0
        if self.stage_id == 0:
            for ro in request_outputs:
                ptids = getattr(ro, "prompt_token_ids", None)
                if ptids is not None:
                    num_tokens_in += len(ptids)

        metrics = self._replica_metrics[replica_id]
        metrics.batch_seq += 1
        batch_id = metrics.batch_seq
        metrics.agg_total_tokens += num_tokens_out
        metrics.agg_total_gen_time_ms += stage_gen_time_ms

        return StageRequestMetrics(
            num_tokens_in=num_tokens_in,
            num_tokens_out=num_tokens_out,
            stage_gen_time_ms=stage_gen_time_ms,
            batch_id=batch_id,
            batch_size=1,
            rx_decode_time_ms=0.0,
            rx_transfer_bytes=0,
            rx_in_flight_time_ms=0.0,
            stage_stats=StageStats(
                total_token=metrics.agg_total_tokens,
                total_gen_time_ms=metrics.agg_total_gen_time_ms,
            ),
        )

    # ---- Stage-local admission ----

    async def submit_initial(
        self,
        request_id: str,
        req_state: OrchestratorRequestState,
        request: Any,
        *,
        prompt_text: Any = None,
        affinity_request_id: str | None = None,
        submit_kwargs: dict[str, Any] | None = None,
        params_override: Any = None,
    ) -> int:
        """Submit a stage-entry request into this pool."""
        params = params_override if params_override is not None else req_state.sampling_params_list[self.stage_id]
        submit_kwargs = dict(submit_kwargs or {})
        if self.stage_type == "diffusion":
            replica_id = self.select_replica_id(
                request_id,
                affinity_request_id=affinity_request_id,
            )
            client = self._diffusion_client(replica_id)
            if isinstance(request, list):
                await client.add_batch_request_async(request_id, request, params, **submit_kwargs)
            else:
                await client.add_request_async(request_id, request, params, **submit_kwargs)
            return replica_id

        replica_id = self.select_replica_id(
            request_id,
            affinity_request_id=affinity_request_id,
        )
        try:
            self.output_processor.add_request(
                request=request,
                prompt=prompt_text,
                parent_req=None,
                request_index=0,
                queue=None,
            )
        except Exception:
            self.release_binding(request_id)
            raise

        try:
            await self._llm_client(replica_id).add_request_async(request, **submit_kwargs)
        except Exception:
            self.release_binding(request_id)
            rollback = getattr(self.output_processor, "remove_request", None)
            if callable(rollback):
                try:
                    rollback(request_id)
                except Exception as rollback_error:
                    logger.warning(
                        "[StagePool] Failed to rollback output processor state for req=%s stage-%s: %s",
                        request_id,
                        self.stage_id,
                        rollback_error,
                    )
            raise
        return replica_id

    async def submit_update(
        self,
        request_id: str,
        req_state: OrchestratorRequestState,
        request: Any,
        *,
        prompt_text: Any = None,
    ) -> int:
        """Submit a streaming update to an already admitted request."""
        params = req_state.sampling_params_list[self.stage_id]
        replica_id = self.get_bound_replica_id(request_id)
        if replica_id is None:
            replica_id = self.select_replica_id(request_id)

        if self.stage_type == "diffusion":
            await self._diffusion_client(replica_id).add_request_async(request_id, request, params)
        else:
            # Refresh the shared output-processor state before yielding to the
            # stage client so streaming segments are merged against the latest
            # prompt/token metadata.
            self.output_processor.add_request(
                request=request,
                prompt=prompt_text,
                parent_req=None,
                request_index=0,
                queue=None,
            )
            await self._llm_client(replica_id).add_request_async(request)
        return replica_id

    # ---- Stage-local polling ----

    async def _poll_stage_raw(self, client: StagePoolLLMClient) -> EngineCoreOutputs | None:
        """Pull raw EngineCoreOutputs from a stage replica without processing."""
        outputs = await client.get_output_async()
        if not outputs.outputs:
            return None
        return outputs

    async def process_llm_raw_outputs(
        self,
        replica_id: int,
        raw_outputs: EngineCoreOutputs,
    ) -> list[Any]:
        """Run the shared LLM output processor on one raw poll result."""
        client = self._llm_client(replica_id)
        processor = self.output_processor
        processed = processor.process_outputs(
            raw_outputs.outputs,
            raw_outputs.timestamp,
            None,
        )

        if processed.reqs_to_abort:
            await client.abort_requests_async(processed.reqs_to_abort)

        if raw_outputs.scheduler_stats is not None:
            processor.update_scheduler_stats(raw_outputs.scheduler_stats)

        return processed.request_outputs

    async def poll_llm_raw_output(
        self,
        replica_id: int,
        *,
        timeout_s: float = 0.001,
    ) -> EngineCoreOutputs | None:
        """Poll raw EngineCore outputs from one LLM replica once."""
        client = self._llm_client(replica_id)
        try:
            return await asyncio.wait_for(
                self._poll_stage_raw(client),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            return None
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "[StagePool] _poll_stage_raw failed for stage-%s replica-%s",
                self.stage_id,
                replica_id,
            )
            raise

    def poll_diffusion_output(self, replica_id: int) -> Any | None:
        """Drain one ready diffusion output from the given replica if present."""
        return self._diffusion_client(replica_id).get_diffusion_output_nowait()

    # ---- Stage-local control plane ----

    async def abort_requests(self, request_ids: list[str]) -> None:
        """Abort the given requests in this stage pool.

        Request-bound abort routing stays inside the pool because route affinity
        (`request_id -> replica_id`) is pool-owned.
        """
        if not request_ids:
            return

        request_ids_by_replica: dict[int, list[str]] = {}
        for request_id in request_ids:
            replica_id = self.get_bound_replica_id(request_id)
            if replica_id is None:
                logger.debug("[StagePool] abort: no binding for req=%s in stage-%s", request_id, self.stage_id)
                continue
            request_ids_by_replica.setdefault(replica_id, []).append(request_id)

        for replica_id, replica_request_ids in request_ids_by_replica.items():
            await self.clients[replica_id].abort_requests_async(replica_request_ids)

        # Clean up OutputProcessor state (e.g. mm_accumulated tensors) that
        # would otherwise leak — aborted requests never produce a final
        # EngineCoreOutput, so process_outputs() never fires its cleanup path.
        all_aborted = [rid for ids in request_ids_by_replica.values() for rid in ids]
        if all_aborted and self._output_processor is not None:
            self._output_processor.abort_requests(all_aborted, internal=True)

    async def collective_rpc(
        self,
        replica_id: int,
        method: str,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Dispatch a stage-scoped control-plane RPC to one physical route."""
        kwargs = dict(kwargs or {})
        client = self.clients[replica_id]
        try:
            return await client.collective_rpc_async(
                method=method,
                timeout=timeout,
                args=args,
                kwargs=kwargs,
            )
        except Exception as exc:
            logger.exception(
                "[StagePool] collective_rpc failed: stage=%s replica=%s method=%s",
                self.stage_id,
                replica_id,
                method,
            )
            return {
                "supported": False,
                "error": str(exc),
            }

    def shutdown_replica(self, replica_id: int) -> None:
        """Shutdown one backend handle in this stage pool."""
        client = self.clients[replica_id]
        try:
            client.shutdown()
            logger.info(
                "[StagePool] Stage %d replica %d shut down",
                self.stage_id,
                replica_id,
            )
        except Exception as e:
            logger.warning(
                "[StagePool] Failed to shutdown stage %d replica %d: %s",
                self.stage_id,
                replica_id,
                e,
            )
