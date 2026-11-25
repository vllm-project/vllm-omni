# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Adapter-only IPC updates using vLLM's existing four-phase session.

Owns both halves of the ``omni_lora_ipc`` backend:
- Worker side: :class:`MergedLoRAIPCWeightTransferEngine`
- Trainer side: :class:`MergedLoRAIPCTrainerWeightTransferEngine`
"""

from __future__ import annotations

import inspect
import os
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, ClassVar, Self

import torch
from torch.multiprocessing.reductions import reduce_tensor
from vllm.config import VllmConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    ParamMeta,
    VLLMWeightSyncClient,
    WeightSource,
    materialize_full_tensor,
)
from vllm.distributed.weight_transfer.clients import HTTPVLLMWeightSyncClient
from vllm.distributed.weight_transfer.factory import (
    WeightTransferEngineFactory,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.ipc_engine import (
    IPCTrainerInitInfo,
    IPCTrainerWeightTransferEngine,
    IPCWeightTransferEngine,
    IPCWeightTransferUpdateInfo,
)
from vllm.logger import init_logger

from vllm_omni.lora.binding import normalize_target_modules
from vllm_omni.lora.ipc_wire import decode_ipc_handles, encode_ipc_handles
from vllm_omni.lora.manager import LoRAManager
from vllm_omni.lora.merged_weights import check_merge_error
from vllm_omni.lora.request import TensorLoRARequest

logger = init_logger(__name__)

_WARNED_EXPANDABLE_SEGMENTS = False


def _reset_expandable_segments_warning() -> None:
    """Reset the once-per-process allocator warning guard (for tests)."""
    global _WARNED_EXPANDABLE_SEGMENTS
    _WARNED_EXPANDABLE_SEGMENTS = False


def has_expandable_segments_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return True if ``expandable_segments:True`` is active in PyTorch alloc conf."""
    environ = os.environ if env is None else env
    for key in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        raw = environ.get(key, "")
        if not raw:
            continue
        for token in raw.split(","):
            entry = token.strip()
            if ":" not in entry:
                continue
            opt, val = entry.split(":", 1)
            if opt.strip().lower() == "expandable_segments" and val.strip().lower() == "true":
                return True
    return False


def warn_if_expandable_segments_disabled(
    device: torch.device,
    env: Mapping[str, str] | None = None,
) -> bool:
    """Warn once per worker process when CUDA expandable_segments is inactive."""
    global _WARNED_EXPANDABLE_SEGMENTS
    if device.type != "cuda":
        return False
    if _WARNED_EXPANDABLE_SEGMENTS or has_expandable_segments_enabled(env):
        return False
    _WARNED_EXPANDABLE_SEGMENTS = True
    logger.warning(
        "omni_lora_ipc worker constructed without expandable_segments enabled; set "
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True in the worker process environment "
        "to avoid CUDA caching-allocator fragmentation OOMs during colocated merged-LoRA syncs."
    )
    return True


class MergedLoRASessionAbortedError(RuntimeError):
    """Raised on the worker when the trainer explicitly aborts a staged session."""


class MergedLoRAUpdateIDMismatchError(ValueError):
    """Raised when a weight-update chunk carries an ``update_id`` that does not match the active session."""


_ABORT_UPDATE_PAYLOAD: dict[str, Any] = {
    "names": [],
    "dtype_names": [],
    "shapes": [],
    "abort": True,
}


def _extract_error_text(exc: BaseException) -> str:
    """Return exception text plus any attached HTTP response body or reason."""
    parts = [str(exc)]
    response = getattr(exc, "response", None)
    if response is not None:
        reason = getattr(response, "reason", None) or getattr(response, "reason_phrase", None)
        if isinstance(reason, str) and reason:
            parts.append(reason)
        try:
            body = getattr(response, "text", None)
        except Exception:
            body = None
        if isinstance(body, str) and body:
            parts.append(body)
        else:
            raw_content = getattr(response, "content", None)
            if isinstance(raw_content, (bytes, bytearray)) and raw_content:
                parts.append(raw_content.decode("utf-8", errors="replace"))
    return " ".join(part for part in parts if part)


def _validate_update_id(update_id: str | None) -> str | None:
    if update_id is None:
        return None
    if not isinstance(update_id, str) or not update_id.strip():
        raise ValueError("update_id must be a non-empty string")
    return update_id


def _accepts_update_id(fn: Any) -> bool:
    """Return True if ``fn`` explicitly accepts keyword argument ``update_id``."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    param = sig.parameters.get("update_id")
    if param is None:
        return False
    return param.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def _uses_http_pickle_client(client: Any) -> bool:
    """Return True for built-in HTTP sync clients that pickle ``ipc_handles``."""
    for cls in type(client).__mro__:
        if cls.__module__ == "vllm.distributed.weight_transfer.clients" and cls.__name__ == "HTTPVLLMWeightSyncClient":
            return True
    return False


def _default_release_ipc_counter(ipc_args: tuple[Any, ...]) -> None:
    """Release one sender-side PyTorch CUDA IPC ref counter for an unpublished export."""
    if len(ipc_args) < 13:
        return
    storage_cls = ipc_args[4]
    storage_device = ipc_args[6]
    storage_handle = ipc_args[7]
    storage_size_bytes = ipc_args[8]
    ref_counter_handle = ipc_args[11]
    ref_counter_offset = ipc_args[12]
    if storage_handle is None or storage_size_bytes == 0 or ref_counter_handle is None:
        return
    release_fn = getattr(storage_cls, "_release_ipc_counter", None)
    if callable(release_fn):
        release_fn(ref_counter_handle, ref_counter_offset, device=storage_device)


@dataclass
class MergedLoRAIPCUpdateInfo(IPCWeightTransferUpdateInfo):
    """IPC tensor metadata plus PEFT config; omit config for a full base update."""

    peft_config: dict | None = None
    lora_int_id: int = 1
    abort: bool = False
    ipc_handles_wire: list[dict[str, Any]] | dict[str, Any] | None = None
    update_id: str | None = None

    def __post_init__(self) -> None:
        self.update_id = _validate_update_id(self.update_id)
        if self.abort:
            if (
                self.names
                or self.dtype_names
                or self.shapes
                or self.ipc_handles
                or self.ipc_handles_pickled
                or self.ipc_handles_wire
            ):
                raise ValueError("Abort update info must not carry tensor payloads")
            self.ipc_handles = []
        elif self.ipc_handles_wire is not None:
            if self.ipc_handles is not None or self.ipc_handles_pickled is not None:
                raise ValueError(
                    "Cannot specify `ipc_handles_wire` together with `ipc_handles` or `ipc_handles_pickled`"
                )
            self.ipc_handles = decode_ipc_handles(self.ipc_handles_wire)
            self.ipc_handles_wire = None
        super().__post_init__()
        if self.ipc_handles is not None:
            self.ipc_handles = decode_ipc_handles(self.ipc_handles)


class _StagedAdapter(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tensors: dict[str, torch.Tensor] = {}

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = set()
        for name, tensor in weights:
            if name in self.tensors:
                raise ValueError(f"Duplicate LoRA tensor in weight-update session: {name}")
            # IPC exporters may reuse their buffers as soon as this chunk returns.
            self.tensors[name] = tensor.detach().clone()
            loaded.add(name)
        return loaded


class MergedLoRAIPCWeightTransferEngine(IPCWeightTransferEngine):
    """Stage adapter chunks, then merge at finish while inference is paused.

    The diffusion worker binds ``lora_manager`` after construction. Dense sessions
    retain upstream reload semantics; adapter sessions preserve the previous
    policy on preparation or recoverable copy failures across the worker group.
    """

    update_info_cls = MergedLoRAIPCUpdateInfo
    supports_draft_weight_update = False
    model: torch.nn.Module
    lora_manager: LoRAManager
    cpu_group: torch.distributed.ProcessGroup | None = None
    _last_abort_was_non_mutating: bool = False
    _update_id: str | None = None
    _skip_next_end_weight_update: bool = False

    def __init__(
        self,
        config: WeightTransferConfig,
        vllm_config: VllmConfig,
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(config, vllm_config, device, model)
        warn_if_expandable_segments_disabled(device)

    def _cleanup_session(self, *, suppress_errors: bool) -> None:
        """Release staged tensors, packed importer handles, and target overrides.

        When ``suppress_errors`` is True (cleaning up either a failed session
        or a session whose new weights have already committed), cleanup
        exceptions are logged at warning with ``exc_info`` so a root failure is
        never masked and a committed policy is never falsely reported as aborted.
        """
        cleanup_error: Exception | None = None
        self._update_id = None
        if hasattr(self, "_staged"):
            self._staged.tensors.clear()
        for label, step in (
            ("close packed IPC importer", self._packed_importer.close),
            ("reset weight update target", self.reset_weight_update_target),
        ):
            try:
                step()
            except Exception as exc:
                logger.warning(
                    "Failed to %s during merged LoRA session cleanup: %s",
                    label,
                    exc,
                    exc_info=True,
                )
                if cleanup_error is None:
                    cleanup_error = exc
        if not suppress_errors and cleanup_error is not None:
            raise cleanup_error

    def start_weight_update(self, update_id: str | None = None) -> None:
        """Begin a session; its first chunk selects dense or adapter loading."""
        validated_id = _validate_update_id(update_id)
        if not hasattr(self, "lora_manager"):
            from vllm_omni.lora.merged_manager import ARMergedLoRAManager

            if self.vllm_config.lora_config is not None:
                raise ValueError("AR merged LoRA requires enable_lora=False; runtime adapters cannot share the policy")
            self.lora_manager = ARMergedLoRAManager(self.model, self.device, torch.float32, lora_application="merged")
            if torch.distributed.is_initialized():
                from vllm.distributed import get_world_group

                self.cpu_group = get_world_group().cpu_group
        self._staged = _StagedAdapter()
        self._kind: str | None = None
        self._peft_config: dict | None = None
        self._adapter_id = 1
        self._error: Exception | None = None
        self._last_abort_was_non_mutating = False
        self._skip_next_end_weight_update = False
        self._update_id = validated_id

    def receive_weights(self, update_info: MergedLoRAIPCUpdateInfo) -> None:
        """Receive unchanged IPC tensors with optional adapter metadata.

        Args:
            update_info: IPC handles and PEFT configuration repeated per chunk.

        Raises:
            ValueError: A session mixes adapter identities, configs, update IDs, or update kinds.
        """
        if update_info.update_id is not None and self._update_id != update_info.update_id:
            raise MergedLoRAUpdateIDMismatchError("Weight-update chunk update_id does not match the active session")
        kind = "dense" if update_info.peft_config is None else "adapter"
        if self._kind is None:
            self._kind = kind
            self._peft_config = update_info.peft_config
            self._adapter_id = update_info.lora_int_id
            if kind == "dense":
                self.lora_manager.reset_merged_adapter()
                super().start_weight_update()
        elif (kind, update_info.peft_config, update_info.lora_int_id) != (
            self._kind,
            self._peft_config,
            self._adapter_id,
        ):
            raise ValueError("A weight-update session must use one update kind, PEFT config and adapter ID")
        if kind == "dense":
            super().receive_weights(update_info)
        else:
            self.model = self._staged
            try:
                super().receive_weights(update_info)
            finally:
                self.reset_weight_update_target()

    def matches_update_id(self, update_id: str | None) -> bool:
        """Return True when ``update_id`` matches the active session (or is untagged)."""
        return update_id is None or self._update_id == update_id

    def abort_weight_update(self, update_id: str | None = None) -> bool:
        """Discard staged adapter tensors without preparing or committing.

        When ``update_id`` is provided, only aborts if it matches the active
        session's ``_update_id``.

        Returns:
            ``True`` when live model weights were not mutated during the
            aborted session (empty or staged-adapter session), ``False`` when
            a dense base update may already have modified live weights or when
            ``update_id`` does not match the active session.
        """
        validated_id = _validate_update_id(update_id)
        if not self.matches_update_id(validated_id):
            return False
        non_mutating = self._kind != "dense"
        self._last_abort_was_non_mutating = non_mutating
        self._skip_next_end_weight_update = False
        self._kind = None
        self._peft_config = None
        self._error = None
        self._cleanup_session(suppress_errors=True)
        return non_mutating

    def update_weights(self, update_info: dict) -> None:
        """Receive a chunk, discarding staged tensors if receiving fails.

        Args:
            update_info: Serialized ``MergedLoRAIPCUpdateInfo``.
        """
        self._skip_next_end_weight_update = False
        if isinstance(update_info, dict) and update_info.get("abort"):
            parsed = self.parse_update_info(update_info)
            if not self.matches_update_id(parsed.update_id):
                # Do not raise: another caller owns the active session, so worker
                # update_weights error cleanup must leave that session untouched.
                return
            self.abort_weight_update()
            raise MergedLoRASessionAbortedError("Merged LoRA weight-update session aborted by trainer")
        if isinstance(update_info, dict) and "update_id" in update_info and update_info["update_id"] is not None:
            incoming_id = _validate_update_id(update_info["update_id"])
            if not self.matches_update_id(incoming_id):
                # Reject stale/foreign chunks without running session failure cleanup
                # or closing the active caller's session on Diffusion or AR workers.
                self._skip_next_end_weight_update = True
                raise MergedLoRAUpdateIDMismatchError("Weight-update chunk update_id does not match the active session")
        try:
            super().update_weights(update_info)
        except Exception as error:
            self._error = error
        try:
            check_merge_error(self._error, self.cpu_group)
        except Exception:
            self._cleanup_session(suppress_errors=True)
            raise

    def finish_weight_update(self) -> None:
        """Validate all ranks before copying a new policy into live storage."""
        self._skip_next_end_weight_update = False
        failed = False
        committed = False
        try:
            if self._kind == "dense":
                super().finish_weight_update()
                committed = True
                return
            error = self._error
            prepared = None
            if error is None:
                try:
                    request = TensorLoRARequest(
                        lora_name="merged",
                        lora_int_id=self._adapter_id,
                        lora_path="tensor",
                        lora_tensors=self._staged.tensors,
                        peft_config=self._peft_config,
                    )
                    prepared = self.lora_manager.prepare_merged_adapter(request)
                except Exception as exc:
                    error = exc
            check_merge_error(error, self.cpu_group)
            assert prepared is not None
            self.lora_manager.commit_merged_adapter(*prepared, group=self.cpu_group)
            committed = True
        except BaseException:
            failed = True
            raise
        finally:
            self._cleanup_session(suppress_errors=failed or committed)

    def shutdown(self) -> None:
        """Disconnect the trainer while retaining the live merged policy."""
        if hasattr(self, "_staged"):
            self._staged.tensors.clear()
        super().shutdown()


class LoRAAdapterWeightSource(WeightSource):
    """``WeightSource`` view over gathered PEFT adapter tensors and config.

    Yields ``(name, tensor)`` pairs in sorted-name order so every trainer rank
    builds and all-gathers IPC handles in identical positional alignment.
    """

    def __init__(
        self,
        tensors: Mapping[str, torch.Tensor] | Callable[[], Mapping[str, torch.Tensor]],
        peft_config: dict[str, Any],
        *,
        lora_int_id: int = 1,
    ) -> None:
        self._tensors_or_fn = tensors
        self._peft_config = _normalize_peft_config(peft_config)
        self.lora_int_id = lora_int_id

    @property
    def peft_config(self) -> dict[str, Any]:
        return dict(self._peft_config)

    def _current_tensors(self) -> Mapping[str, torch.Tensor]:
        raw = self._tensors_or_fn() if callable(self._tensors_or_fn) else self._tensors_or_fn
        if not isinstance(raw, Mapping) or not raw:
            raise ValueError("LoRAAdapterWeightSource requires a non-empty mapping of adapter tensors")
        return raw

    def update(
        self,
        tensors: Mapping[str, torch.Tensor] | Callable[[], Mapping[str, torch.Tensor]],
        *,
        peft_config: dict[str, Any] | None = None,
        lora_int_id: int | None = None,
    ) -> None:
        """Swap in new adapter tensors/config for the next RL sync round."""
        self._tensors_or_fn = tensors
        if peft_config is not None:
            self._peft_config = _normalize_peft_config(peft_config)
        if lora_int_id is not None:
            self.lora_int_id = lora_int_id
        self.__dict__.pop("metadata", None)

    @cached_property
    def metadata(self) -> list[ParamMeta]:
        tensors = self._current_tensors()
        return [ParamMeta(name, tensors[name].dtype, tuple(tensors[name].shape)) for name in sorted(tensors)]

    def iter_full(self) -> Iterator[tuple[str, torch.Tensor]]:
        tensors = self._current_tensors()
        for name in sorted(tensors):
            yield name, materialize_full_tensor(tensors[name])


def _normalize_peft_config(peft_config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(peft_config, dict) or not peft_config:
        raise ValueError(f"peft_config must be a non-empty dict, got {type(peft_config).__name__}")
    normalized = dict(peft_config)
    normalized["target_modules"] = normalize_target_modules(
        normalized.get("target_modules"),
        allow_none=False,
    )
    return normalized


@dataclass
class MergedLoRAIPCTrainerInitInfo(IPCTrainerInitInfo):
    """Trainer-side init info for the ``omni_lora_ipc`` backend.

    Uses per-tensor IPC staging (``packed=False``); packed-buffer producer
    rework is explicitly out of scope per RFC #10444 §3.
    """

    backend: ClassVar[str] = "omni_lora_ipc"
    packed: bool = False
    process_group: torch.distributed.ProcessGroup | None = field(default=None, repr=False)
    encode_handles: bool = True
    lora_int_id: int = 1
    device: torch.device | int | None = None
    # Injectable device/IPC seams for CPU unit tests or custom transports.
    gpu_uuid: str | None = field(default=None, repr=False)
    reduce_tensor_fn: Callable[[torch.Tensor], tuple[Any, tuple[Any, ...]]] | None = field(default=None, repr=False)
    stage_tensor_fn: Callable[[torch.Tensor], torch.Tensor] | None = field(default=None, repr=False)
    ipc_collect_fn: Callable[[], None] | None = field(default=None, repr=False)
    synchronize_fn: Callable[[], None] | None = field(default=None, repr=False)
    release_ipc_counter_fn: Callable[[tuple[Any, ...]], None] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.packed:
            raise ValueError("omni_lora_ipc trainer engine supports per-tensor staging only (packed=False)")


class MergedLoRAIPCTrainerWeightTransferEngine(IPCTrainerWeightTransferEngine):
    """Trainer-side ``omni_lora_ipc`` engine composing upstream IPC machinery.

    Owns sorted-name cross-rank alignment, per-tensor device staging,
    ``reduce_tensor`` export, UUID all-gather merge over a caller-supplied
    process group, optional msgspec-safe handle encoding, four-phase session
    sequencing through ``VLLMWeightSyncClient``, and staging-reference lifetime
    past the post-ack barrier and ``ipc_collect``.
    """

    init_info_cls = MergedLoRAIPCTrainerInitInfo  # type: ignore[assignment]

    def __init__(
        self,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource,
        is_sender: bool = True,
        rank: int = 0,
        process_group: torch.distributed.ProcessGroup | None = None,
        encode_handles: bool = True,
        lora_int_id: int = 1,
        device: torch.device | int | None = None,
        gpu_uuid: str | None = None,
        reduce_tensor_fn: Callable[[torch.Tensor], tuple[Any, tuple[Any, ...]]] | None = None,
        stage_tensor_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        ipc_collect_fn: Callable[[], None] | None = None,
        synchronize_fn: Callable[[], None] | None = None,
        release_ipc_counter_fn: Callable[[tuple[Any, ...]], None] | None = None,
    ) -> None:
        # Bypass IPCTrainerWeightTransferEngine.__init__ only when resolving
        # device/UUID so CPU unit tests can supply stubbed seams without CUDA.
        super(IPCTrainerWeightTransferEngine, self).__init__(
            client=client,
            source=source,
            is_sender=is_sender,
            rank=rank,
        )
        self.packed = False
        self.process_group = process_group
        self.encode_handles = encode_handles
        self.default_lora_int_id = lora_int_id
        self._reduce_tensor_fn = reduce_tensor_fn or reduce_tensor
        self._ipc_collect_fn = ipc_collect_fn or torch.cuda.ipc_collect
        self._synchronize_fn = synchronize_fn or torch.accelerator.synchronize
        if release_ipc_counter_fn is not None:
            self._release_ipc_counter_fn: Callable[[tuple[Any, ...]], None] | None = release_ipc_counter_fn
        elif reduce_tensor_fn is None:
            self._release_ipc_counter_fn = _default_release_ipc_counter
        else:
            self._release_ipc_counter_fn = None

        if device is None:
            if gpu_uuid is None:
                self.device_index = torch.accelerator.current_device_index()
                self.staging_device = torch.device("cuda", self.device_index)
            else:
                self.device_index = 0
                self.staging_device = torch.device("cpu")
        elif isinstance(device, int):
            self.device_index = device
            self.staging_device = torch.device("cuda", device)
        else:
            self.staging_device = device
            self.device_index = 0 if device.index is None else device.index

        if gpu_uuid is not None:
            self.gpu_uuid = gpu_uuid
        else:
            self.gpu_uuid = str(torch.cuda.get_device_properties(self.device_index).uuid)

        if stage_tensor_fn is not None:
            self._stage_tensor_fn = stage_tensor_fn
        else:
            self._stage_tensor_fn = self._default_stage_tensor

    def _default_stage_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().to(device=self.staging_device).contiguous()

    @classmethod
    def trainer_init(
        cls,
        init_info: MergedLoRAIPCTrainerInitInfo,  # type: ignore[override]
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource | None = None,
    ) -> Self:
        if init_info.packed:
            raise ValueError("omni_lora_ipc trainer engine supports per-tensor staging only (packed=False)")
        if source is None:
            raise ValueError("omni_lora_ipc trainer weight transfer requires a WeightSource.")
        if not source.implements_iter_full():
            raise TypeError(
                f"omni_lora_ipc sends full adapter tensors, but {type(source).__name__} does not implement iter_full()."
            )
        if len(source.owned_names(init_info.rank)) != len(source.metadata):
            raise ValueError(
                f"rank {init_info.rank} does not own every adapter tensor, but omni_lora_ipc "
                "needs every trainer rank to own every adapter tensor."
            )
        engine = cls(
            client=client,
            source=source,
            is_sender=init_info.is_sender,
            rank=init_info.rank,
            process_group=init_info.process_group,
            encode_handles=init_info.encode_handles,
            lora_int_id=init_info.lora_int_id,
            device=init_info.device,
            gpu_uuid=init_info.gpu_uuid,
            reduce_tensor_fn=init_info.reduce_tensor_fn,
            stage_tensor_fn=init_info.stage_tensor_fn,
            ipc_collect_fn=init_info.ipc_collect_fn,
            synchronize_fn=init_info.synchronize_fn,
            release_ipc_counter_fn=init_info.release_ipc_counter_fn,
        )
        if engine.is_sender:
            engine.client.init_weight_transfer_engine({"packed": False})
        return engine

    def _group_world_size(self) -> int:
        if not torch.distributed.is_initialized():
            return 1
        return torch.distributed.get_world_size(group=self.process_group)

    def _coordinate_preparation(
        self,
        *,
        names: list[str],
        local_error: BaseException | None = None,
    ) -> None:
        """Coordinate local staging/sync status and parameter names before IPC export.

        Running this exchange before ``reduce_tensor`` ensures that a staging
        error, asynchronous device sync error, or cross-rank name mismatch never
        creates unopened CUDA IPC exports or strands peer ranks in a later
        collective.
        """
        world_size = self._group_world_size()
        if world_size == 1:
            if local_error is not None:
                raise local_error
            return

        gathered: list[Any] = [None] * world_size
        local_err_msg = repr(local_error) if local_error is not None else None
        torch.distributed.all_gather_object(gathered, (names, local_err_msg), group=self.process_group)

        peer_errors: list[tuple[int, str]] = []
        for rank_idx, item in enumerate(gathered):
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError(f"Invalid gathered preparation payload from rank {rank_idx}")
            _, rank_err = item
            if rank_err is not None:
                peer_errors.append((rank_idx, rank_err))

        if local_error is not None:
            raise local_error
        if peer_errors:
            err_rank, err_msg = peer_errors[0]
            raise RuntimeError(f"omni_lora_ipc trainer preparation failed on rank {err_rank}: {err_msg}")

        for rank_idx, (rank_names, _) in enumerate(gathered):
            if rank_names != names:
                raise ValueError(
                    f"omni_lora_ipc parameter name mismatch across trainer ranks: "
                    f"rank {rank_idx} exported {rank_names}, expected {names}"
                )

    def _all_gather_and_merge_handles(
        self,
        handles: list[dict[str, tuple]],
        names: list[str] | None = None,
        local_error: BaseException | None = None,
    ) -> list[dict[str, tuple]]:
        """All-gather and merge exported per-tensor GPU UUID handles.

        Staging and local device synchronization already completed in
        ``_coordinate_preparation``; this collective coordinates any export
        failure and merges handle dicts across ranks.
        """
        world_size = self._group_world_size()
        if world_size == 1:
            if local_error is not None:
                raise local_error
            return handles

        gathered: list[Any] = [None] * world_size
        local_err_msg = repr(local_error) if local_error is not None else None
        local_item: Any = (names, handles, local_err_msg)
        torch.distributed.all_gather_object(gathered, local_item, group=self.process_group)
        torch.distributed.barrier(group=self.process_group)

        peer_errors: list[tuple[int, str]] = []
        for rank_idx, item in enumerate(gathered):
            if not isinstance(item, tuple) or len(item) != 3:
                raise ValueError(f"Invalid gathered IPC payload from rank {rank_idx}")
            _, _, rank_err = item
            if rank_err is not None:
                peer_errors.append((rank_idx, rank_err))

        if local_error is not None:
            raise local_error
        if peer_errors:
            err_rank, err_msg = peer_errors[0]
            raise RuntimeError(f"omni_lora_ipc handle export failed on rank {err_rank}: {err_msg}")

        expected = len(handles)
        rank_handle_lists: list[list[dict[str, tuple]]] = []
        for rank_idx, (rank_names, rank_handles, _) in enumerate(gathered):
            if names is not None and rank_names != names:
                raise ValueError(
                    f"omni_lora_ipc parameter name mismatch across trainer ranks: "
                    f"rank {rank_idx} exported {rank_names}, expected {names}"
                )
            if rank_handles is None or len(rank_handles) != expected:
                raise ValueError(
                    f"omni_lora_ipc handle count mismatch across trainer ranks: "
                    f"rank {rank_idx} exported {None if rank_handles is None else len(rank_handles)}, "
                    f"expected {expected}"
                )
            rank_handle_lists.append(rank_handles)

        if self.is_sender:
            merged: list[dict[str, tuple]] = []
            for param_idx in range(expected):
                combined: dict[str, tuple] = {}
                for rank_handles in rank_handle_lists:
                    combined.update(rank_handles[param_idx])
                merged.append(combined)
            return merged
        return [{} for _ in handles]

    def _exchange_session_outcome(
        self,
        local_error: BaseException | None,
        *,
        handles_published: bool = False,
    ) -> tuple[tuple[int, str] | None, bool]:
        """Barrier and broadcast the session outcome and publish flag across ranks."""
        world_size = self._group_world_size()
        if world_size == 1:
            return None, handles_published
        outcomes: list[Any] = [None] * world_size
        local_msg = repr(local_error) if local_error is not None else None
        torch.distributed.all_gather_object(outcomes, (local_msg, handles_published), group=self.process_group)
        any_published = False
        first_failure: tuple[int, str] | None = None
        for rank_idx, item in enumerate(outcomes):
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError(f"Invalid session outcome payload from rank {rank_idx}")
            msg, published = item
            any_published = any_published or bool(published)
            if msg is not None and first_failure is None:
                first_failure = (rank_idx, msg)
        return first_failure, any_published

    def _release_unpublished_exports(self, local_handles: list[dict[str, tuple]]) -> None:
        """Decrement sender IPC counters for exports confirmed never sent to any worker."""
        if self._release_ipc_counter_fn is None:
            return
        for handle_map in local_handles:
            for ipc_args in handle_map.values():
                try:
                    self._release_ipc_counter_fn(ipc_args)
                except Exception as exc:
                    logger.warning(
                        "Failed to release unpublished CUDA IPC counter during omni_lora_ipc cleanup: %s",
                        exc,
                        exc_info=True,
                    )

    def _release_staged_refs(
        self,
        weight_refs: list[torch.Tensor],
        *,
        unpublished_handles: list[dict[str, tuple]] | None = None,
        run_ipc_collect: bool = True,
        suppress_errors: bool,
    ) -> None:
        if unpublished_handles:
            self._release_unpublished_exports(unpublished_handles)
        weight_refs.clear()
        if not run_ipc_collect:
            return
        try:
            self._ipc_collect_fn()
        except Exception as exc:
            logger.warning(
                "ipc_collect failed during omni_lora_ipc trainer cleanup: %s",
                exc,
                exc_info=True,
            )
            if not suppress_errors:
                raise

    def _abort_remote_session(
        self,
        *,
        finish_attempted: bool = False,
        update_id: str | None = None,
    ) -> None:
        """Drop staged weights on the worker without committing when a session fails.

        Uses ``client.abort_weight_update()`` when the client exposes it (and
        accepts ``update_id`` whenever ``update_id`` is scoped to a session),
        and otherwise sends the abort-sentinel payload through the standard
        4-method ``VLLMWeightSyncClient.update_weights`` route so both
        diffusion and AR workers discard staged tensors without committing and
        clear ``_weight_update_active``.
        """
        abort_fn = getattr(self.client, "abort_weight_update", None)
        if callable(abort_fn) and (update_id is None or _accepts_update_id(abort_fn)):
            try:
                if update_id is not None:
                    aborted = abort_fn(update_id=update_id)
                    if aborted is False:
                        logger.debug(
                            "Skipped remote omni_lora_ipc abort for update_id=%s because another session is active.",
                            update_id,
                        )
                        return
                else:
                    abort_fn()
                if finish_attempted:
                    logger.warning(
                        "Remote omni_lora_ipc session aborted after an interrupted finish_weight_update; "
                        "if only the finish reply was lost, the worker may already have committed."
                    )
                else:
                    logger.debug("Remote omni_lora_ipc session aborted via client.abort_weight_update")
            except Exception as abort_exc:
                logger.warning(
                    "Remote omni_lora_ipc abort_weight_update failed before releasing trainer IPC references: %s",
                    abort_exc,
                    exc_info=True,
                )
            return

        payload = dict(_ABORT_UPDATE_PAYLOAD)
        if update_id is not None:
            payload["update_id"] = update_id
        try:
            self.client.update_weights(payload)
            if update_id is not None:
                logger.debug(
                    "Remote omni_lora_ipc abort for update_id=%s was ignored because another session is active.",
                    update_id,
                )
        except Exception as abort_exc:
            msg = _extract_error_text(abort_exc)
            if isinstance(abort_exc, MergedLoRASessionAbortedError) or "aborted by trainer" in msg:
                logger.debug("Remote omni_lora_ipc session aborted cleanly without committing: %s", abort_exc)
            elif "start_weight_update must be called before update_weights" in msg:
                if finish_attempted:
                    logger.warning(
                        "Remote omni_lora_ipc session was already closed after an interrupted "
                        "finish_weight_update; worker policy state may reflect the committed update: %s",
                        abort_exc,
                    )
                else:
                    logger.debug("Remote omni_lora_ipc session was already closed on the worker: %s", abort_exc)
            else:
                logger.warning(
                    "Remote omni_lora_ipc abort could not confirm worker session closure before "
                    "releasing trainer IPC references: %s",
                    abort_exc,
                    exc_info=True,
                )

    def _resolve_adapter_metadata(
        self,
        peft_config: dict[str, Any] | None,
        lora_int_id: int | None,
    ) -> tuple[dict[str, Any], int]:
        resolved_cfg = peft_config
        if resolved_cfg is None:
            resolved_cfg = getattr(self.source, "peft_config", None)
        if resolved_cfg is None:
            raise ValueError("omni_lora_ipc send_weights requires peft_config on the source or as an argument")
        normalized_cfg = _normalize_peft_config(resolved_cfg)
        resolved_id = (
            lora_int_id if lora_int_id is not None else getattr(self.source, "lora_int_id", self.default_lora_int_id)
        )
        return normalized_cfg, int(resolved_id)

    def send_weights(
        self,
        *,
        peft_config: dict[str, Any] | None = None,
        lora_int_id: int | None = None,
    ) -> None:
        """Stage, export, all-gather, and publish one merged-LoRA adapter session."""
        assert self.source is not None

        names: list[str] = []
        dtype_names: list[str] = []
        shapes: list[list[int]] = []
        weight_refs: list[torch.Tensor] = []
        normalized_cfg: dict[str, Any] | None = None
        resolved_id = self.default_lora_int_id
        prep_error: BaseException | None = None

        # Phase 1a: stage tensors locally and synchronize before exporting any
        # CUDA IPC handles so peer staging/sync failures never leave unopened
        # exports sitting in IPC limbo.
        raw_items: dict[str, torch.Tensor] = {}
        try:
            normalized_cfg, resolved_id = self._resolve_adapter_metadata(peft_config, lora_int_id)
            for name, tensor in self.source.iter_full():
                if name in raw_items:
                    raise ValueError(f"Duplicate LoRA tensor name from weight source: {name}")
                raw_items[name] = tensor
            if not raw_items:
                raise ValueError("omni_lora_ipc cannot send an empty adapter")

            for name in sorted(raw_items):
                staged = self._stage_tensor_fn(raw_items[name])
                weight_refs.append(staged)
                names.append(name)
                dtype_names.append(str(staged.dtype).split(".")[-1])
                shapes.append(list(staged.shape))
                del staged
            self._synchronize_fn()
        except BaseException as exc:
            prep_error = exc
        finally:
            raw_items.clear()

        # Phase 1b: coordinate staging/sync status and sorted names across ranks.
        try:
            self._coordinate_preparation(names=names, local_error=prep_error)
        except BaseException:
            self._release_staged_refs(weight_refs, run_ipc_collect=False, suppress_errors=True)
            raise

        # Phase 1c: export IPC handles only after every rank is ready, then all-gather.
        ipc_handles: list[dict[str, tuple]] = []
        export_error: BaseException | None = None
        try:
            for idx in range(len(weight_refs)):
                _, ipc_args = self._reduce_tensor_fn(weight_refs[idx])
                ipc_handles.append({self.gpu_uuid: ipc_args})
        except BaseException as exc:
            export_error = exc

        try:
            merged_handles = self._all_gather_and_merge_handles(
                ipc_handles,
                names=names,
                local_error=export_error,
            )
        except BaseException:
            self._release_staged_refs(
                weight_refs,
                unpublished_handles=ipc_handles,
                run_ipc_collect=bool(ipc_handles),
                suppress_errors=True,
            )
            raise

        # Phase 2: sender encoding + RPC session + collective outcome broadcast.
        rpc_error: BaseException | None = None
        peer_failure: tuple[int, str] | None = None
        session_started = False
        start_with_id_attempted = False
        update_attempted = False
        session_finished = False
        finish_attempted = False
        session_update_id: str | None = None
        try:
            if self.is_sender:
                assert normalized_cfg is not None
                use_http_wire = self.encode_handles and _uses_http_pickle_client(self.client)
                wire_handles: list[dict[str, Any]] = (
                    encode_ipc_handles(merged_handles, json_safe=use_http_wire)  # type: ignore[assignment]
                    if self.encode_handles
                    else merged_handles
                )
                handle_key = "ipc_handles_wire" if use_http_wire else "ipc_handles"
                payload: dict[str, Any] = {
                    "names": names,
                    "dtype_names": dtype_names,
                    "shapes": shapes,
                    handle_key: wire_handles,
                    "peft_config": normalized_cfg,
                    "lora_int_id": resolved_id,
                }
                if _accepts_update_id(self.client.start_weight_update):
                    session_update_id = uuid.uuid4().hex
                    payload["update_id"] = session_update_id
                    start_with_id_attempted = True
                    self.client.start_weight_update(update_id=session_update_id)
                else:
                    self.client.start_weight_update()
                session_started = True
                update_attempted = True
                self.client.update_weights(payload)
                finish_attempted = True
                self.client.finish_weight_update()
                session_finished = True
        except BaseException as exc:
            rpc_error = exc
            # Abort after a confirmed start, or after an unconfirmed start only
            # when an update_id token was sent so the worker can verify session
            # ownership and leave any other caller's session untouched.
            if self.is_sender and (session_started or start_with_id_attempted) and not session_finished:
                self._abort_remote_session(
                    finish_attempted=finish_attempted,
                    update_id=session_update_id,
                )
        finally:
            # Keep staging references strongly alive until after the worker ack
            # and collective outcome exchange across all trainer ranks.
            any_published = True
            try:
                peer_failure, any_published = self._exchange_session_outcome(
                    rpc_error,
                    handles_published=update_attempted,
                )
            except Exception as exc:
                logger.warning(
                    "Post-send outcome exchange failed during omni_lora_ipc trainer cleanup: %s",
                    exc,
                    exc_info=True,
                )
                if rpc_error is None:
                    self._release_staged_refs(weight_refs, suppress_errors=True)
                    raise
            unpublished = (
                ipc_handles if (rpc_error is not None or peer_failure is not None) and not any_published else None
            )
            self._release_staged_refs(
                weight_refs,
                unpublished_handles=unpublished,
                suppress_errors=(rpc_error is not None or peer_failure is not None),
            )

        if rpc_error is not None:
            raise rpc_error
        if peer_failure is not None:
            err_rank, err_msg = peer_failure
            raise RuntimeError(f"omni_lora_ipc weight sync failed on rank {err_rank}: {err_msg}")


class OmniHTTPWeightSyncClient(HTTPVLLMWeightSyncClient):
    """HTTP weight-sync client passing ``update_id`` to Omni's ``/start_weight_update``."""

    def start_weight_update(self, update_id: str | None = None) -> None:
        validated_id = _validate_update_id(update_id)
        body = {"update_id": validated_id} if validated_id is not None else None
        self._post("start_weight_update", body)


WeightTransferEngineFactory.register_engine("omni_lora_ipc", MergedLoRAIPCWeightTransferEngine)
WeightTransferTrainerFactory.register_engine("omni_lora_ipc", MergedLoRAIPCTrainerWeightTransferEngine)
