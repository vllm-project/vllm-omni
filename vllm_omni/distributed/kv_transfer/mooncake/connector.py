# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bulk KV/HS loading for Qwen3-Omni turn serving on MRv2.

The native Mooncake connector owns KV copies and their GPU block references.
This adapter adds a joint lookup on its worker, request-attempt identities,
bounded CPU capture storage, and delayed receive completion. Native connector
integration targets vLLM revision 4ba787ec9959e566d3b587a796dff07673324da5.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from copy import copy
from dataclasses import dataclass, field, replace
from typing import Any, cast

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorTransferResults,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.connector import MooncakeStoreConnector
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    MooncakeLookupResult,
    MooncakeStoreConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler import MooncakeStoreScheduler
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import MooncakeStoreWorker
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus

from .config import MooncakeConfig
from .hs import HSBlockSpec, RequestHS, decode_hs, encode_hs, hs_key, prefix_digest

logger = init_logger(__name__)
_INIT_LOCK = threading.Lock()
_HS_HEADER_ALLOWANCE = 16392


class JointStoreWorker(MooncakeStoreWorker):
    """Use the existing lookup RPC to return joint, contiguous availability."""

    def __init__(self, config: VllmConfig, kv_config: KVCacheConfig, namespace: str, max_tokens: int) -> None:
        self.hs_namespace = namespace
        self.max_hs_tokens = max_tokens
        super().__init__(config, kv_config)
        if self.block_size != self.hash_block_size:
            raise ValueError("Mooncake KV/HS currently requires equal KV and hash block sizes")

    def lookup(self, num_tokens: int, block_hashes: Sequence[BlockHash]) -> MooncakeLookupResult:
        result = super().lookup(num_tokens, block_hashes)
        end = min(result.hit_length, self.max_hs_tokens)
        end -= end % self.block_size
        hashes = block_hashes[: end // self.block_size]
        keys: list[str] = []
        for block_hash in hashes:
            key = hs_key(self.hs_namespace, block_hash)
            keys.extend((key, key + ":ready"))
        if not keys:
            return MooncakeLookupResult(0)
        try:
            exists = self.store.batch_is_exist(keys)
        except (RuntimeError, OSError) as exc:
            logger.warning("Mooncake HS lookup failed: %s", exc)
            return MooncakeLookupResult(0)
        matched = 0
        for index in range(len(hashes)):
            pair = exists[2 * index : 2 * index + 2]
            if len(pair) != 2 or not all(value == 1 for value in pair):
                break
            matched += self.block_size
        return MooncakeLookupResult(
            matched,
            tuple(replace(boundary, num_tokens=matched) for boundary in result.tail_key_boundaries) if matched else (),
        )


@dataclass
class Attempt:
    """An engine request ID plus a unique retry/preemption generation.

    Ordinary serving already assigns unique internal IDs when a client reuses
    an external ID. Native completion messages must keep that engine identity;
    ``token`` additionally fences successive loads within one engine request.
    """

    req_id: str
    token: str
    prompt_len: int
    hashes: list[bytes] = field(default_factory=list)
    rows: RequestHS | None = None
    cancelled: bool = False


@dataclass
class PendingRead:
    attempt: Attempt
    reserved_bytes: int
    deadline: float
    future: Future[dict[str, torch.Tensor]] | None = None
    kv_done: bool = False
    failed: bool = False
    failure_reported: bool = False
    completion_reported: bool = False
    block_ids: frozenset[int] = frozenset()


class JointMetadata(MooncakeStoreConnectorMetadata):
    """Native KV transfer metadata plus Omni request-attempt ownership."""

    def __init__(
        self,
        native: MooncakeStoreConnectorMetadata,
        attempts: dict[str, str],
        finished: set[str],
        cancelled: set[str],
    ) -> None:
        super().__init__(native.unfinished_request_ids, native.preempted_req_ids)
        self.requests = native.requests
        self.omni_attempts = attempts
        self.omni_finished = finished
        self.omni_cancelled = cancelled


class OmniMooncakeStoreConnector(MooncakeStoreConnector):
    """Opt-in TP=1 full-attention adapter. All reads precede computation."""

    def __init__(
        self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config: KVCacheConfig | None = None
    ) -> None:
        self._closed = False
        self._lock = threading.RLock()
        self._executor: ThreadPoolExecutor | None = None
        self._attempts: dict[str, Attempt] = {}
        self._delivered_until: dict[str, int] = {}
        self._reads: dict[str, PendingRead] = {}
        self._saves: dict[Future[None], int] = {}
        self._reserved_bytes = 0
        self._scheduler_attempts: dict[str, str] = {}
        self._bypass: set[str] = set()
        self._retired: set[str] = set()
        self._cancelled: set[str] = set()
        self.connector_worker = None
        self.connector_scheduler = None
        self.settings = MooncakeConfig.from_vllm_config(vllm_config)
        self._validate_supported(vllm_config, kv_cache_config)
        assert kv_cache_config is not None
        assert vllm_config.kv_transfer_config is not None
        model = vllm_config.model_config
        self.hidden_layer = model.hf_config.talker_config.accept_hidden_layer
        self.hidden_size = model.hf_text_config.hidden_size
        self.hs_dtype = model.dtype
        spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
        self.block_size = spec.block_size
        self._row_bytes = 2 * self.hidden_size * torch.empty((), dtype=self.hs_dtype).element_size()
        identity = {
            "namespace": self.settings.namespace,
            "revision": self.settings.model_revision,
            "stage": "thinker",
            "format": 1,
            "layer": self.hidden_layer,
            "config": model.hf_config.to_dict(),
            "dtype": str(self.hs_dtype),
            "kv_spec": repr(spec),
        }
        self.namespace = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
        # Isolate native KV as well as HS across incompatible checkpoints.
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config
        extra.update(cache_prefix=self.namespace, load_async=True, save_decode_cache=False)
        KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self._capacity_only = False
        self._kv_cache_config = kv_cache_config
        self._kv_cache_events = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = MooncakeStoreScheduler(vllm_config, kv_cache_config)
        else:
            # The pinned upstream constructor reads a process environment
            # variable. Keep this scoped to construction and restore it.
            with _INIT_LOCK:
                previous = os.environ.get("MOONCAKE_CONFIG_PATH")
                os.environ["MOONCAKE_CONFIG_PATH"] = self.settings.config_file
                try:
                    self.connector_worker = JointStoreWorker(
                        vllm_config,
                        kv_cache_config,
                        self.namespace,
                        self.settings.max_hs_bytes // (3 * self._row_bytes),
                    )
                finally:
                    if previous is None:
                        os.environ.pop("MOONCAKE_CONFIG_PATH", None)
                    else:
                        os.environ["MOONCAKE_CONFIG_PATH"] = previous
            self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="omni-mooncake-hs")

    @staticmethod
    def _validate_supported(config: VllmConfig, kv_config: KVCacheConfig | None) -> None:
        model = config.model_config
        parallel = config.parallel_config
        if not current_platform.is_cuda() or not config.use_v2_model_runner:
            raise ValueError("Mooncake KV/HS requires CUDA and Model Runner V2")
        if any(
            size != 1
            for size in (
                parallel.tensor_parallel_size,
                parallel.pipeline_parallel_size,
                parallel.data_parallel_size,
                parallel.decode_context_parallel_size,
                parallel.prefill_context_parallel_size,
            )
        ):
            raise ValueError("Mooncake KV/HS requires a single TP=1 serving replica")
        if getattr(model, "model_stage", None) != "thinker" or model.hf_config.model_type != "qwen3_omni_moe":
            raise ValueError("Mooncake KV/HS supports the Qwen3-Omni Thinker only")
        if config.cache_config.enable_prefix_caching or config.speculative_config or config.lora_config:
            raise ValueError("Mooncake KV/HS does not support local prefix caching, speculation, or LoRA")
        if getattr(model, "session_mode", "turn") not in ("turn", "turn_based"):
            raise ValueError("Mooncake KV/HS duplex serving is not yet supported")
        transfer = config.kv_transfer_config
        if transfer is None or transfer.kv_load_failure_policy != "recompute":
            raise ValueError("Mooncake KV/HS requires kv_load_failure_policy='recompute'")
        if transfer.kv_role != "kv_both":
            raise ValueError("Mooncake KV/HS requires kv_role='kv_both'")
        if getattr(model, "engine_output_type", "text") == "text":
            raise ValueError("Mooncake KV/HS requires Thinker capture outputs for Talker")
        if kv_config is None or len(kv_config.kv_cache_groups) != 1:
            raise ValueError("Mooncake KV/HS requires one full-attention KV group")
        if type(kv_config.kv_cache_groups[0].kv_cache_spec) is not FullAttentionSpec:
            raise ValueError("Mooncake KV/HS requires ordinary full attention")

    def get_num_new_matched_tokens(self, request: Request, num_computed_tokens: int) -> tuple[int | None, bool]:
        assert self.connector_scheduler is not None
        if request.request_id in self._bypass:
            self.connector_scheduler.load_specs.pop(request.request_id, None)
            return 0, False
        return super().get_num_new_matched_tokens(request, num_computed_tokens)

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        super().update_connector_output(connector_output)
        for req_id in connector_output.failed_recving or ():
            self._bypass.add(req_id)
        for req_id in connector_output.finished_recving or ():
            if req_id not in self._bypass:
                continue
            # Rotate only after native copies drain and recovery can run.
            self._scheduler_attempts.pop(req_id, None)

    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        if request.status in (RequestStatus.FINISHED_ABORTED, RequestStatus.FINISHED_ERROR):
            self._cancelled.add(request.request_id)
        return super().request_finished_all_groups(request, block_ids)

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> JointMetadata:
        retired = set(scheduler_output.finished_req_ids) | set(scheduler_output.preempted_req_ids or ())
        for req_id in retired:
            self._scheduler_attempts.pop(req_id, None)
            self._bypass.discard(req_id)
        native = cast(MooncakeStoreConnectorMetadata, super().build_connector_meta(scheduler_output))
        req_ids = {req.req_id for req in native.requests} | set(scheduler_output.num_scheduled_tokens)
        for req_id in req_ids:
            if req_id not in self._scheduler_attempts:
                self._scheduler_attempts[req_id] = uuid.uuid4().hex
        metadata = JointMetadata(
            native,
            dict(self._scheduler_attempts),
            set(scheduler_output.finished_req_ids),
            set(self._cancelled) | set(scheduler_output.preempted_req_ids or ()),
        )
        self._cancelled.clear()
        return metadata

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        # Bind before the forward: native async load submission itself may be
        # deferred until post_forward, after the runner snapshots the attempt.
        connector_metadata = cast(JointMetadata, connector_metadata)
        metadata = copy(connector_metadata)
        with self._lock:
            cancelled_tokens = set()
            for req_id in connector_metadata.omni_finished:
                self._delivered_until.pop(req_id, None)
            for req_id in connector_metadata.omni_finished | connector_metadata.preempted_req_ids:
                old = self._attempts.pop(req_id, None)
                if old:
                    # A normal response may finish before its best-effort
                    # save drains. Only abort/preemption invalidates writes.
                    old.cancelled = req_id in connector_metadata.omni_cancelled
                    old.rows = None
                    self._retired.add(old.token)
                    if old.cancelled:
                        cancelled_tokens.add(old.token)
            for req_id, token in connector_metadata.omni_attempts.items():
                old = self._attempts.get(req_id)
                if old is not None and old.token != token:
                    old.cancelled = True
                    old.rows = None
                    self._retired.add(old.token)
                    cancelled_tokens.add(old.token)
                if old is None or old.token != token:
                    self._attempts[req_id] = Attempt(req_id, token, 0)
            metadata.requests = []
            for req in connector_metadata.requests:
                attempt = self._attempts[req.req_id]
                attempt.prompt_len = req.num_prompt_tokens
                attempt.hashes = list(req.block_hashes)
                native = copy(req)
                native.req_id = attempt.token
                metadata.requests.append(native)
            # Native preemption immediately invalidates queued store jobs.
            # Normal finishes instead retire through get_transfer_results,
            # which preserves their ledger until all pinned jobs drain.
            metadata.preempted_req_ids = cancelled_tokens
            metadata.unfinished_request_ids = {a.token for a in self._attempts.values()}
        super().bind_connector_metadata(metadata)

    def current_attempt(self, req_id: str) -> str | None:
        with self._lock:
            attempt = self._attempts.get(req_id)
            return attempt.token if attempt is not None and not attempt.cancelled else None

    def _spec(self, block_hash: bytes, index: int) -> HSBlockSpec:
        return HSBlockSpec(
            self.namespace,
            prefix_digest(block_hash),
            index * self.block_size,
            (index + 1) * self.block_size,
            self.hidden_layer,
            self.hidden_size,
            self.hs_dtype,
        )

    def _read_hs(self, attempt: Attempt, tokens: int) -> dict[str, torch.Tensor]:
        blocks = []
        assert self.connector_worker is not None
        store = self.connector_worker.store
        for index, block_hash in enumerate(attempt.hashes[: tokens // self.block_size]):
            if attempt.cancelled:
                raise ValueError("cancelled HS load")
            key = hs_key(self.namespace, block_hash)
            payload = store.get(key)
            if not isinstance(payload, bytes) or not payload:
                raise ValueError("missing HS object")
            if store.get(key + ":ready") != hashlib.sha256(payload).digest():
                raise ValueError("HS object has no matching completion record")
            blocks.append(decode_hs(payload, self._spec(block_hash, index)))
        if len(blocks) * self.block_size != tokens:
            raise ValueError("incomplete HS prefix")
        return {name: torch.cat([block[name] for block in blocks]) for name in ("embedding", "hidden")}

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        metadata = cast(JointMetadata, self._get_connector_metadata())
        with self._lock:
            by_token = {a.token: a for a in self._attempts.values()}
            for req in metadata.requests:
                spec = req.load_spec
                if spec is None or not spec.can_load:
                    continue
                attempt = by_token[req.req_id]
                tokens = spec.kvpool_cached_tokens
                reserve = 3 * tokens * self._row_bytes + _HS_HEADER_ALLOWANCE
                pending = PendingRead(attempt, reserve, time.monotonic() + self.settings.load_timeout_seconds)
                pending.block_ids = frozenset(req.block_ids[0][: tokens // self.block_size])
                if (
                    len(self._reads) + len(self._saves) >= self.settings.max_pending_transfers
                    or self._memory_used() + reserve > self.settings.max_hs_bytes
                ):
                    pending.failed = True
                    pending.reserved_bytes = 0
                else:
                    self._reserved_bytes += reserve
                    assert self._executor is not None
                    pending.future = self._executor.submit(self._read_hs, attempt, tokens)
                self._reads[attempt.token] = pending
        # Even an HS admission failure drains the native KV operation before
        # allowing the scheduler to free its destination blocks.
        super().start_load_kv(forward_context, **kwargs)

    def _memory_used(self) -> int:
        return self._reserved_bytes + sum(a.rows.buffered_bytes for a in self._attempts.values() if a.rows)

    def get_transfer_results(self, finished_req_ids: set[str]) -> KVConnectorTransferResults:
        with self._lock:
            retired = set(self._retired)
            self._retired.clear()
        native = super().get_transfer_results(retired)
        failed_blocks = super().get_block_ids_with_load_errors()
        finished, failed = set(), set()
        with self._lock:
            for token in native.failed_recving:
                if token in self._reads:
                    self._reads[token].failed = True
            for token in native.finished_recving:
                if token in self._reads:
                    self._reads[token].kv_done = True
            for token, pending in list(self._reads.items()):
                attempt = pending.attempt
                if pending.block_ids & failed_blocks:
                    pending.failed = True
                if time.monotonic() >= pending.deadline:
                    pending.failed = True
                hs_done = pending.future is None or pending.future.done()
                tensors = None
                if not pending.failed and pending.future is not None and hs_done:
                    try:
                        tensors = pending.future.result()
                    except (ValueError, RuntimeError, OSError) as exc:
                        logger.warning("Mooncake HS read failed for %s: %s", attempt.req_id, exc)
                        pending.failed = True
                if pending.failed and not pending.failure_reported and not attempt.cancelled:
                    failed.add(attempt.req_id)
                    pending.failure_reported = True
                if pending.failed and pending.kv_done and not pending.completion_reported:
                    # HS reads touch CPU buffers only. A timed-out read may
                    # drain in the background while this request recomputes;
                    # keep its reservation until the future really finishes.
                    current = self._attempts.get(attempt.req_id)
                    if current is None or current.token == token:
                        finished.add(attempt.req_id)
                    pending.completion_reported = True
                if not pending.kv_done or not hs_done:
                    continue
                self._reserved_bytes -= pending.reserved_bytes
                if not pending.failed and not attempt.cancelled:
                    assert tensors is not None
                    attempt.rows = RequestHS(attempt.prompt_len, self.block_size, self.settings.max_hs_bytes)
                    attempt.rows.restore(tensors)
                current = self._attempts.get(attempt.req_id)
                if not pending.completion_reported and (current is None or current.token == token):
                    finished.add(attempt.req_id)
                del self._reads[token]
            for future, reserve in list(self._saves.items()):
                if future.done():
                    try:
                        future.result()
                    except (ValueError, RuntimeError, OSError) as exc:
                        logger.warning("Mooncake HS save skipped: %s", exc)
                    self._reserved_bytes -= reserve
                    del self._saves[future]
        return KVConnectorTransferResults(finished_sending=set(), finished_recving=finished, failed_recving=failed)

    def get_block_ids_with_load_errors(self) -> set[int]:
        # get_transfer_results consumes block-level failures and upgrades them
        # to whole-attempt failure so HS state and retry bypass stay coherent.
        return set()

    def _save_hs(self, attempt: Attempt, blocks: list[tuple[int, int, dict[str, torch.Tensor]]]) -> None:
        assert self.connector_worker is not None
        store = self.connector_worker.store
        deadline = time.monotonic() + self.settings.load_timeout_seconds
        for start, _, tensors in blocks:
            if attempt.cancelled:
                return
            index = start // self.block_size
            block_hash = attempt.hashes[index]
            key = hs_key(self.namespace, block_hash)
            payload = encode_hs(self._spec(block_hash, index), tensors)
            if store.put(key, payload) != 0:
                return
            del payload
            # put(existing_key) is immutable in the pinned Store. Read the
            # actual winner so concurrent duplicate writers publish its digest.
            actual = store.get(key)
            if not isinstance(actual, bytes) or not actual:
                return
            decode_hs(actual, self._spec(block_hash, index))
            kv_key = self.connector_worker.token_dbs[0].key_for(block_hash)
            while store.is_exist(kv_key) != 1:
                if attempt.cancelled or time.monotonic() >= deadline:
                    return
                time.sleep(0.005)
            if not attempt.cancelled:
                digest = hashlib.sha256(actual).digest()
                ready_key = key + ":ready"
                previous = store.get(ready_key)
                # HS can be evicted independently of its completion record.
                # A refilled object needs a new record; immutable put alone
                # would preserve the old digest and poison every later hit.
                if previous and previous != digest:
                    store.remove(ready_key, force=True)
                store.put(ready_key, digest)

    def capture_step(
        self, req_id: str, attempt_id: str, start_token: int, prompt_len: int, tensors: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor] | None:
        with self._lock:
            attempt = self._attempts.get(req_id)
            if attempt is None or attempt.cancelled or attempt.token != attempt_id:
                return None
            if attempt.rows is None:
                attempt.rows = RequestHS(prompt_len, self.block_size, self.settings.max_hs_bytes)
            # Saving is best effort; output assembly is mandatory. Budget the
            # block copies before RequestHS stages this step's captures.
            fresh_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
            if self._memory_used() + 3 * fresh_bytes > self.settings.max_hs_bytes:
                attempt.rows.save_enabled = False
            merged = attempt.rows.append(start_token, tensors)
            blocks = attempt.rows.pop_blocks()
            reserve = (
                3 * sum(t.numel() * t.element_size() for _, _, group in blocks for t in group.values())
                + _HS_HEADER_ALLOWANCE
            )
            if (
                blocks
                and len(self._reads) + len(self._saves) < self.settings.max_pending_transfers
                and self._memory_used() + reserve <= self.settings.max_hs_bytes
            ):
                self._reserved_bytes += reserve
                assert self._executor is not None
                future = self._executor.submit(self._save_hs, attempt, blocks)
                self._saves[future] = reserve
            # Preemption rebuilds KV/HS, but the downstream accumulator still
            # owns already published rows. Keep its watermark across attempts.
            end = start_token + tensors["embedding"].shape[0]
            merged_start = end - merged["embedding"].shape[0]
            delivered = self._delivered_until.get(req_id, 0)
            trim = min(merged["embedding"].shape[0], max(0, delivered - merged_start))
            if merged_start > delivered:
                raise ValueError("Mooncake handoff would omit prompt rows")
            self._delivered_until[req_id] = max(delivered, end)
            if trim:
                merged = {name: tensor[trim:] for name, tensor in merged.items()}
            return merged

    def shutdown(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        with self._lock:
            for attempt in self._attempts.values():
                attempt.cancelled = True
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
        super().shutdown()
        with self._lock:
            self._attempts.clear()
            self._delivered_until.clear()
            self._reads.clear()
            self._saves.clear()
            self._reserved_bytes = 0
