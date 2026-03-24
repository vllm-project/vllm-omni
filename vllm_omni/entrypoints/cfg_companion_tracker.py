"""CFG companion request tracker for the Omni orchestrator.

Encapsulates all bookkeeping for Classifier-Free Guidance companion
requests (prompt expansion, parent/companion ID mapping, completion
tracking, deferred forwarding, failure propagation, and timeouts)
so that ``Omni._run_generation`` stays clean.
"""

from __future__ import annotations

import copy
import logging
import os
import time
from collections.abc import Callable, Sequence
from typing import Any

from vllm_omni.distributed.omni_connectors.adapter import try_send_via_connector
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniSamplingParams

logger = logging.getLogger(__name__)


class CfgCompanionTracker:
    """Manages CFG companion request lifecycle in the orchestrator scheduling loop."""

    def __init__(
        self,
        prompt_expand_func: Callable[..., Any] | None,
        stage0_sampling_params: Any,
        timeout_s: float | None = None,
    ) -> None:
        self._expand_func = prompt_expand_func
        self._sp0 = stage0_sampling_params
        self._timeout_s = (
            timeout_s if timeout_s is not None else float(os.environ.get("VLLM_CFG_PENDING_TIMEOUT_S", "120"))
        )

        self._companion_map: dict[str, dict[str, str]] = {}  # parent -> {role: companion_id}
        self._companion_ids: set[str] = set()
        self._companion_to_parent: dict[str, str] = {}  # companion -> parent
        self._done: dict[str, set[str]] = {}  # parent -> completed companion ids
        self._pending_parents: dict[str, dict[str, Any]] = {}  # parent -> deferred result
        self._failed_parents: set[str] = set()

    @property
    def is_active(self) -> bool:
        return bool(self._companion_ids)

    @property
    def num_companions(self) -> int:
        return len(self._companion_ids)

    @property
    def stage0_sampling_params(self) -> Any:
        return self._sp0

    def expand_prompts(
        self,
        request_id_to_prompt: dict[str, Any],
    ) -> list[tuple[str, Any]]:
        """Expand user prompts into ``(companion_id, prompt)`` pairs via model-specific func."""
        if not self._expand_func:
            return []

        pairs: list[tuple[str, Any]] = []
        for rid, prompt in request_id_to_prompt.items():
            expanded = self._expand_func(prompt, self._sp0)
            if not expanded:
                continue
            role_map: dict[str, str] = {}
            for ep in expanded:
                cid = f"{rid}{ep.request_id_suffix}"
                role_map[ep.role] = cid
                self._companion_ids.add(cid)
                self._companion_to_parent[cid] = rid
                pairs.append((cid, ep.prompt))
            self._companion_map[rid] = role_map
            self._done[rid] = set()

        logger.debug(
            "CFG expansion: %d parent(s) -> %d companion(s)",
            len(self._companion_map),
            len(self._companion_ids),
        )
        return pairs

    def is_companion(self, req_id: str) -> bool:
        return req_id in self._companion_ids

    def has_companions(self, parent_id: str) -> bool:
        return parent_id in self._companion_map

    def all_companions_done(self, parent_id: str) -> bool:
        role_map = self._companion_map.get(parent_id, {})
        done_set = self._done.get(parent_id, set())
        return all(cid in done_set for cid in role_map.values())

    def get_companion_request_ids(self, parent_id: str) -> dict[str, str]:
        """Return ``{role: companion_request_id}`` for a parent."""
        return self._companion_map.get(parent_id, {})

    def is_parent_failed(self, parent_id: str) -> bool:
        return parent_id in self._failed_parents

    # -- Lifecycle events --

    def on_companion_error(self, companion_id: str) -> tuple[str | None, bool]:
        """Record failure. Returns ``(parent_id, parent_was_aborted)``."""
        parent_id = self._companion_to_parent.get(companion_id)
        if parent_id is None:
            return None, False
        self._failed_parents.add(parent_id)
        logger.error("CFG companion %s failed; marking parent %s as failed", companion_id, parent_id)
        aborted = parent_id in self._pending_parents
        if aborted:
            self._pending_parents.pop(parent_id, None)
        return parent_id, aborted

    def on_companion_completed(self, companion_id: str) -> str | None:
        """Mark done. Returns parent_id only if parent is pending and all companions finished."""
        parent_id = self._companion_to_parent.get(companion_id)
        if parent_id is None:
            return None
        self._done[parent_id].add(companion_id)
        logger.debug("CFG companion %s completed (parent=%s)", companion_id, parent_id)
        if parent_id in self._pending_parents and self.all_companions_done(parent_id):
            return parent_id
        return None

    def consume_parent_failure(self, parent_id: str) -> None:
        self._failed_parents.discard(parent_id)

    # -- Deferred parent management --

    def defer_parent(self, parent_id: str, engine_outputs: Any, stage_id: int) -> None:
        """Hold parent result while waiting for companions to finish."""
        self._pending_parents[parent_id] = {
            "engine_outputs": engine_outputs,
            "stage_id": stage_id,
            "pending_since": time.time(),
        }
        logger.debug("Parent %s deferred, waiting for CFG companions", parent_id)

    def pop_pending_parent(self, parent_id: str) -> dict[str, Any] | None:
        return self._pending_parents.pop(parent_id, None)

    def check_timeouts(self) -> list[str]:
        """Return and remove parent IDs that exceeded the pending timeout."""
        if not self._pending_parents:
            return []
        now = time.time()
        timed_out: list[str] = []
        for pid in list(self._pending_parents):
            pending_since = self._pending_parents[pid].get("pending_since", now)
            if now - pending_since > self._timeout_s:
                self._pending_parents.pop(pid)
                self._failed_parents.discard(pid)
                timed_out.append(pid)
                logger.error("Parent %s timed out waiting for CFG companions (>%.0fs)", pid, self._timeout_s)
        return timed_out

    # -- Forward parent with CFG KV --

    def forward_parent_with_cfg(
        self,
        req_id: str,
        parent_result: dict[str, Any],
        stage_list: Sequence[Any],
        connectors: dict[tuple[str, str], Any],
        sampling_params_list: Sequence[OmniSamplingParams],
        request_id_to_prompt: dict[str, Any],
        final_stage_id_to_prompt: dict[str, int],
        metrics: Any,
        remaining_by_stage: list[int],
    ) -> bool:
        """Forward a parent request to the next stage with CFG KV request IDs attached."""
        stage_id = parent_result["stage_id"]
        next_stage_id = stage_id + 1
        if next_stage_id > final_stage_id_to_prompt.get(req_id, 0):
            return True

        next_stage = stage_list[next_stage_id]
        try:
            with metrics.stage_postprocess_timer(stage_id, req_id):
                next_inputs = next_stage.process_engine_inputs(
                    stage_list,
                    [request_id_to_prompt[req_id]],
                    source_outputs_override=parent_result["engine_outputs"],
                )
        except Exception as e:
            logger.exception(
                "Process engine inputs error for req %s at stage %d: %s",
                req_id,
                next_stage_id,
                e,
            )
            return False

        sp_next = copy.deepcopy(sampling_params_list[next_stage_id])
        if isinstance(sp_next, OmniDiffusionSamplingParams):
            sp_next.cfg_kv_request_ids = self.get_companion_request_ids(req_id)
            logger.info(
                "Attaching cfg_kv_request_ids=%s to request %s",
                sp_next.cfg_kv_request_ids,
                req_id,
            )

        connector_key = (str(stage_id), str(next_stage_id))
        connector = connectors.get(connector_key)
        sent_via_connector = False
        if connector:
            sent_via_connector = try_send_via_connector(
                connector=connector,
                stage_id=stage_id,
                next_stage_id=next_stage_id,
                req_id=req_id,
                next_inputs=next_inputs,
                sampling_params=sp_next,
                original_prompt=request_id_to_prompt[req_id],
                next_stage_queue_submit_fn=stage_list[next_stage_id].submit,
                metrics=metrics,
            )

        if not sent_via_connector:
            raise RuntimeError(
                f"Failed to send CFG request {req_id} to stage-{next_stage_id} via connector. "
                "Configure a connector for this edge or inspect connector logs for details."
            )

        logger.debug("Forwarded CFG-enabled request %s to stage-%d", req_id, next_stage_id)
        remaining_by_stage[next_stage_id] += 1
        return True
