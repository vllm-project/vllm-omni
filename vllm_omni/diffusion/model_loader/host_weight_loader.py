# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Host Weight Runtime integration for diffusion model loaders.

This mixin keeps HWR identity, restore, publication, and startup-recovery
policy out of the ordinary diffusers loader.  The host loader supplies the
canonical source-preparation and weight-loading hooks; this module owns only
the optional final-layout transaction.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast

import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.model_loader.host_weight_plan import (
    HostWeightPlan,
    TensorBinding,
)

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm_omni.diffusion.model_loader.host_weights.identity_adapter import FinalLayoutIdentityContext
    from vllm_omni.diffusion.model_loader.host_weights.source_identity import (
        NodeSourceDigestCache,
        PreparedWeightSource,
    )
    from vllm_omni.host_weight_runtime import (
        HostWeightLease,
        HostWeightLeaseCarrier,
        HostWeightRuntime,
        ResolutionOutcome,
        RuntimeMode,
    )

_T = TypeVar("_T")


class _WeightSource(Protocol):
    model_or_path: str
    subfolder: str | None
    revision: str | None
    prefix: str
    fall_back_to_pt: bool
    allow_patterns_overrides: list[str] | None


class _AllGatherCoordinator(Protocol):
    world_size: int
    rank_in_group: int
    ranks: Sequence[int]
    cpu_group: Any
    device_group: Any


class _HWRCommitError(RuntimeError):
    """A committed warm restore made the current model disposable."""


class _HWRIneligibleError(ValueError):
    pass


@dataclass
class _HWRState:
    mode: RuntimeMode
    context: FinalLayoutIdentityContext
    runtime: HostWeightRuntime
    outcome: ResolutionOutcome
    coordinator: _AllGatherCoordinator | None = None
    artifact_content_digest: str | None = None
    plan: HostWeightPlan | None = None
    allow_checkpoint_production: bool = True


class HWRLoaderMixin:
    """Optional final-layout HWR behavior shared by diffusion loaders."""

    if TYPE_CHECKING:
        od_config: Any
        parallel_config: Any
        quant_config: Any
        host_weight_plan: HostWeightPlan | None
        _hwr_state: _HWRState | None
        _last_load_request: dict[str, object] | None
        _force_canonical_load: bool
        _checkpoint_publication_waiter: Callable[[], None] | None

        def _prepare_weights(
            self,
            model_name_or_path: Path | str,
            subfolder: str | None,
            revision: str | None,
            fall_back_to_pt: bool,
            allow_patterns_overrides: list[str] | None,
        ) -> tuple[Path | str, list[str], bool]: ...

        def _get_checkpoint_adapter(
            self,
            model: nn.Module,
            source: Any,
            use_safetensors: bool,
        ) -> Any: ...

        def load_model(self, **kwargs: object) -> nn.Module: ...

    @staticmethod
    def _identity_value(value: object) -> object:
        """Convert config objects into deterministic identity metadata."""
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, torch.dtype):
            return str(value)
        if isinstance(value, dict):
            return {str(key): HWRLoaderMixin._identity_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [HWRLoaderMixin._identity_value(item) for item in value]
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            try:
                return HWRLoaderMixin._identity_value(to_dict())
            except TypeError:
                pass
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return HWRLoaderMixin._identity_value(dataclasses.asdict(cast(Any, value)))
        return f"{type(value).__module__}.{type(value).__qualname__}:{value!r}"

    @staticmethod
    def _identity_fingerprint(value: object) -> str:
        from vllm_omni.host_weight_runtime import CanonicalJson

        return hashlib.sha256(CanonicalJson.from_value(value).encoded).hexdigest()

    def _hwr_eligibility_mode(
        self,
        model: nn.Module,
        modules: object,
        *,
        dist_offload: bool,
        use_allgather: bool,
        load_format: str,
    ) -> RuntimeMode | None:
        """Return the enabled HWR mode only after all zero-interaction gates."""
        from vllm_omni.host_weight_runtime import RuntimeMode

        raw_mode = getattr(self.od_config, "host_weight_runtime_mode", "disabled")
        try:
            mode = RuntimeMode(raw_mode)
        except ValueError as exc:
            raise ValueError("host_weight_runtime_mode must be disabled, preferred, or required") from exc

        # These gates intentionally precede HWR imports, source preparation,
        # identity construction, and store creation.
        if mode is RuntimeMode.DISABLED or not dist_offload:
            return None

        parallel = self.parallel_config
        reasons: list[str] = []
        if load_format != "default":
            reasons.append("load_format must be 'default'")
        if bool(getattr(parallel, "use_hsdp", False)):
            reasons.append("HSDP layouts are not supported by the final-layout BF16 consumer")
        if self.quant_config is not None:
            reasons.append("quantized layouts require a representation-specific HWR producer")
        if getattr(self.od_config, "lora_path", None):
            reasons.append("adapted weights are not reusable base-model artifacts")

        dit_modules = tuple(zip(getattr(modules, "dit_names", ()), getattr(modules, "dits", ())))
        if not dit_modules:
            reasons.append("no DiT modules were discovered")
        elif any(
            getattr(dit, "host_weight_restore_contract", None) is None
            or not callable(getattr(dit, "validate_restored_host_weights", None))
            for _, dit in dit_modules
        ):
            reasons.append("every owned DiT must declare the final-layout restore contract")
        elif use_allgather and any(
            dit.host_weight_restore_contract.implementation_id != "minimax-h3-dit" for _, dit in dit_modules
        ):
            reasons.append("AllGather HWR is promoted only for MiniMax H3 BF16")

        if reasons:
            message = "; ".join(reasons)
            if mode is RuntimeMode.REQUIRED:
                raise ValueError(f"required Host Weight Runtime path is ineligible: {message}")
            logger.info("Host Weight Runtime is ineligible; using the canonical DLO path: %s", message)
            return None
        return mode

    def _prepare_hwr_sources(
        self,
        model: nn.Module,
        modules: object,
        sources: Sequence[_WeightSource],
    ) -> tuple[PreparedWeightSource, ...]:
        from vllm_omni.diffusion.model_loader.host_weights import (
            ImplementationIdentity,
            PreparedWeightSource,
            WeightSourceKind,
        )

        dit_prefixes = tuple(f"{name}." for name in getattr(modules, "dit_names", ()))
        selected_sources = tuple(
            source
            for source in sources
            if not source.prefix or any(prefix.startswith(source.prefix) for prefix in dit_prefixes)
        )
        if not selected_sources:
            raise ValueError("final-layout HWR requires canonical weight sources covering every DiT")

        prepared: list[PreparedWeightSource] = []
        for source in selected_sources:
            resolved_root, weight_files, use_safetensors = self._prepare_weights(
                source.model_or_path,
                source.subfolder,
                source.revision,
                source.fall_back_to_pt,
                source.allow_patterns_overrides,
            )
            adapter = self._get_checkpoint_adapter(model, source, use_safetensors)
            adapter_identity = None
            if adapter is not None:
                adapter_name = f"{type(adapter).__module__}.{type(adapter).__qualname__}"
                adapter_identity = ImplementationIdentity(
                    implementation_id=adapter_name,
                    version="1",
                    fingerprint=self._identity_fingerprint({"adapter": adapter_name}),
                )
            source_kind = (
                WeightSourceKind.LOCAL_PATH
                if os.path.isdir(os.fspath(source.model_or_path))
                else WeightSourceKind.HUGGING_FACE_HUB
            )
            prepared.append(
                PreparedWeightSource(
                    model_or_path=os.fspath(source.model_or_path),
                    subfolder=source.subfolder,
                    requested_revision=source.revision,
                    prefix=source.prefix,
                    resolved_root=Path(os.fspath(resolved_root)),
                    weight_files=tuple(Path(os.fspath(path)) for path in weight_files),
                    use_safetensors=use_safetensors,
                    checkpoint_adapter=adapter_identity,
                    source_kind=source_kind,
                )
            )
        return tuple(prepared)

    def _resolve_hwr_allgather_coordinator(self, use_allgather: bool) -> _AllGatherCoordinator | None:
        """Resolve the same DP-or-SP cohort used later by DLO AllGather."""
        parallel = self.parallel_config
        dp_size = int(getattr(parallel, "data_parallel_size", 1) or 1)
        sp_size = int(getattr(parallel, "sequence_parallel_size", 1) or 1)
        expected_size = dp_size if dp_size > 1 else sp_size
        if not use_allgather or expected_size <= 1:
            return None
        if not torch.distributed.is_initialized():
            raise RuntimeError("BF16 HWR AllGather resolution requires initialized distributed groups")

        from vllm_omni.diffusion.distributed.parallel_state import (
            get_data_parallel_world_size,
            get_dp_group,
            get_sp_group,
        )

        coordinator = get_dp_group() if get_data_parallel_world_size() > 1 else get_sp_group()
        return cast(_AllGatherCoordinator, coordinator)

    @staticmethod
    def _coordinate_hwr_group(
        coordinator: _AllGatherCoordinator | None,
        phase: str,
        local_record: dict[str, object],
        *,
        require_identity: bool = True,
    ) -> list[dict[str, object]]:
        """Gather one startup record from every rank in the DLO cohort."""
        if coordinator is None:
            return [
                {
                    **local_record,
                    "phase": phase,
                    "group_size": 1,
                    "group_rank": 0,
                    "group_ranks": [0],
                }
            ]

        world_size = coordinator.world_size
        record = {
            **local_record,
            "phase": phase,
            "group_size": world_size,
            "group_rank": coordinator.rank_in_group,
            "group_ranks": list(coordinator.ranks),
        }
        gathered: list[object] = [None] * world_size
        group = coordinator.cpu_group or coordinator.device_group
        torch.distributed.all_gather_object(gathered, record, group=group)
        records = cast(list[dict[str, object]], gathered)
        if require_identity and len({candidate["identity_digest"] for candidate in records}) != 1:
            raise RuntimeError(f"BF16 HWR AllGather weight identity differs during {phase}")
        return records

    def _build_hwr_context(
        self,
        model: nn.Module,
        modules: object,
        *,
        load_format: str,
        sources: Sequence[_WeightSource],
        source_digest_cache: NodeSourceDigestCache | None = None,
    ) -> FinalLayoutIdentityContext:
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        from vllm_omni.diffusion.model_loader.host_weights import (
            FINAL_LAYOUT_BF16_POLICY,
            FinalLayoutLoaderIdentity,
            FinalLayoutParallelIdentity,
            FinalLayoutRequest,
            ImplementationIdentity,
            build_final_layout_identity,
        )

        parallel = self.parallel_config
        tp_size = int(getattr(parallel, "tensor_parallel_size", 1))
        try:
            tp_rank = int(get_tensor_model_parallel_rank()) if tp_size > 1 else 0
        except Exception:
            tp_rank = int(getattr(parallel, "tensor_parallel_rank", 0))
        sp_size = int(getattr(parallel, "sequence_parallel_size", 1) or 1)
        loader_config = {
            "dtype": str(self.od_config.dtype),
            "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
            "model_config": self._identity_value(getattr(self.od_config, "tf_model_config", None)),
            "load_format": load_format,
            "quantization": self._identity_value(self.quant_config),
        }
        contracts = [
            self._identity_value(getattr(dit, "host_weight_restore_contract")) for dit in getattr(modules, "dits", ())
        ]
        loader_identity = FinalLayoutLoaderIdentity(
            implementation=ImplementationIdentity(
                implementation_id="vllm-omni.diffusion.diffusers-loader",
                version="final-layout-v1",
                fingerprint=self._identity_fingerprint(
                    {
                        "loader": "diffusers-loader-final-layout-v1",
                        "pipeline": f"{type(model).__module__}.{type(model).__qualname__}",
                    }
                ),
            ),
            model_config_fingerprint=self._identity_fingerprint(loader_config),
            weight_transform_fingerprint=self._identity_fingerprint(
                {
                    "contracts": contracts,
                    "transform": "diffusion-final-layout-loader-transforms-v1",
                }
            ),
        )
        semantic_parallel = FinalLayoutParallelIdentity(
            tensor_parallel_size=tp_size,
            tensor_parallel_rank=tp_rank,
            sequence_parallel_size=sp_size,
            ulysses_degree=int(getattr(parallel, "ulysses_degree", 1)),
            ring_degree=int(getattr(parallel, "ring_degree", 1)),
            allgather_degree=int(getattr(parallel, "allgather_degree", 1)),
            ulysses_mode=str(getattr(parallel, "ulysses_mode", "strict")),
            pipeline_parallel_size=int(getattr(parallel, "pipeline_parallel_size", 1)),
            cfg_parallel_size=int(getattr(parallel, "cfg_parallel_size", 1)),
            use_hsdp=bool(getattr(parallel, "use_hsdp", False)),
            enable_expert_parallel=bool(getattr(parallel, "enable_expert_parallel", False)),
        )
        model_id = str(getattr(self.od_config, "model", "") or "")
        if not model_id:
            raise ValueError("final-layout HWR requires a canonical model identifier")
        request = FinalLayoutRequest(
            model_id=model_id,
            loader=loader_identity,
            parallel=semantic_parallel,
            load_format=load_format,
        )
        prepared_sources = self._prepare_hwr_sources(model, modules, sources)
        dit_modules = tuple(zip(getattr(modules, "dit_names", ()), getattr(modules, "dits", ())))
        return build_final_layout_identity(
            model,
            dit_modules=dit_modules,
            prepared_sources=prepared_sources,
            request=request,
            policy=FINAL_LAYOUT_BF16_POLICY,
            source_digest_cache=source_digest_cache,
        )

    @staticmethod
    def _phase_ranks(records: Sequence[dict[str, object]], status: str) -> list[int]:
        return [cast(int, record["group_rank"]) for record in records if record["status"] == status]

    def _coordinate_hwr_eligibility(
        self,
        model: nn.Module,
        modules: object,
        *,
        coordinator: _AllGatherCoordinator | None,
        dist_offload: bool,
        use_allgather: bool,
        load_format: str,
    ) -> RuntimeMode | None:
        mode = None
        error = None
        try:
            mode = self._hwr_eligibility_mode(
                model,
                modules,
                dist_offload=dist_offload,
                use_allgather=use_allgather,
                load_format=load_format,
            )
        except Exception as exc:
            error = exc

        status = "error" if error else "ready" if mode else "ineligible"
        records = self._coordinate_hwr_group(
            coordinator,
            "eligibility",
            {
                "identity_digest": "bf16-final-layout-eligibility-v1",
                "status": status,
                "error_type": type(error).__name__ if error else None,
            },
            require_identity=False,
        )
        failed = self._phase_ranks(records, "error")
        if failed:
            if coordinator is None:
                raise cast(Exception, error)
            raise RuntimeError(f"BF16 HWR AllGather eligibility failed on group ranks {failed}") from error

        ineligible = self._phase_ranks(records, "ineligible")
        if not ineligible:
            return mode
        if len(ineligible) == len(records):
            return None
        raise RuntimeError(f"BF16 HWR AllGather eligibility differs across ranks: ineligible={ineligible}")

    @staticmethod
    def _validate_hwr_source_prefixes(modules: object, sources: Sequence[_WeightSource]) -> None:
        expected = frozenset(f"{name}." for name in getattr(modules, "dit_names", ()))
        available = frozenset(source.prefix for source in sources)
        if not expected <= available:
            raise _HWRIneligibleError("final-layout HWR requires one dedicated source prefix per owned DiT")
        overlapping = {prefix for prefix in available if any(name.startswith(prefix) for name in expected)}
        if not overlapping <= expected:
            raise _HWRIneligibleError("final-layout HWR requires dedicated DiT sources")

    def _resolve_local_hwr_artifact(
        self,
        model: nn.Module,
        modules: object,
        *,
        mode: RuntimeMode,
        load_format: str,
        sources: Sequence[_WeightSource],
    ) -> tuple[_HWRState, HostWeightLease | None]:
        from vllm_omni.diffusion.model_loader.host_weights import NodeSourceDigestCache
        from vllm_omni.host_weight_runtime import (
            HostWeightRuntime,
            HostWeightRuntimeConfig,
            ProductionPolicy,
            ResolutionOutcome,
            RuntimeMode,
            StorageDomainPolicy,
        )
        from vllm_omni.host_weight_runtime.filesystem import detect_storage_class

        self._validate_hwr_source_prefixes(modules, sources)
        root = Path(self.od_config.host_weight_runtime_root).expanduser()
        runtime = HostWeightRuntime.from_config(
            HostWeightRuntimeConfig(
                mode=mode,
                domain=StorageDomainPolicy(root=root, storage_class=detect_storage_class(root)),
                production=ProductionPolicy(
                    allow_local_build=mode is RuntimeMode.PREFERRED,
                    allow_post_load_publish=True,
                ),
            )
        )
        context = self._build_hwr_context(
            model,
            modules,
            load_format=load_format,
            sources=sources,
            source_digest_cache=NodeSourceDigestCache(
                root,
                timeout_seconds=runtime.config.wait.coordination_timeout_seconds,
            ),
        )
        resolution = runtime.resolve(context.identity)
        if resolution.report.outcome is ResolutionOutcome.FAILED:
            failure = next(attempt.failure for attempt in reversed(resolution.report.attempts) if attempt.failure)
            raise RuntimeError(f"Host Weight Runtime resolution failed: {failure.message}")
        state = _HWRState(mode, context, runtime, resolution.report.outcome)
        return state, cast("HostWeightLease | None", resolution.lease)

    def _coordinate_hwr_artifact(
        self,
        model: nn.Module,
        modules: object,
        *,
        coordinator: _AllGatherCoordinator | None,
        mode: RuntimeMode,
        load_format: str,
        sources: Sequence[_WeightSource],
    ) -> tuple[_HWRState | None, HostWeightLease | None]:
        from vllm_omni.host_weight_runtime import ResolutionOutcome, RuntimeMode

        state = None
        lease = None
        error = None
        ineligible_reason = None
        try:
            state, lease = self._resolve_local_hwr_artifact(
                model,
                modules,
                mode=mode,
                load_format=load_format,
                sources=sources,
            )
        except _HWRIneligibleError as exc:
            ineligible_reason = str(exc)
        except Exception as exc:
            error = exc

        if error:
            status = "error"
        elif ineligible_reason:
            status = "ineligible"
        elif lease is None:
            status = "fallback"
        else:
            status = "ready"
        records = self._coordinate_hwr_group(
            coordinator,
            "artifact_resolution",
            {
                "identity_digest": state.context.identity.key if state else None,
                "content_digest": lease.provenance.artifact_content_sha256 if lease else None,
                "status": status,
                "error_type": type(error).__name__ if error else None,
                "outcome": state.outcome.value if state else None,
            },
            require_identity=False,
        )

        failed = self._phase_ranks(records, "error")
        if failed:
            if lease:
                lease.close()
            if coordinator is None:
                raise cast(Exception, error)
            raise RuntimeError(f"BF16 HWR AllGather artifact resolution failed on group ranks {failed}") from error

        ineligible = self._phase_ranks(records, "ineligible")
        if ineligible:
            if len(ineligible) != len(records):
                if lease:
                    lease.close()
                raise RuntimeError(f"BF16 HWR AllGather artifact eligibility differs across ranks: {ineligible}")
            if mode is RuntimeMode.REQUIRED:
                raise ValueError(f"required Host Weight Runtime path is ineligible: {ineligible_reason}")
            logger.info("Host Weight Runtime is ineligible; using canonical DLO: %s", ineligible_reason)
            return None, None

        identities = {record["identity_digest"] for record in records}
        if len(identities) != 1:
            if lease:
                lease.close()
            raise RuntimeError("BF16 HWR AllGather artifact identity differs across ranks")

        missing = [cast(int, record["group_rank"]) for record in records if record["status"] != "ready"]
        if missing:
            if lease:
                lease.close()
            if mode is RuntimeMode.REQUIRED:
                raise RuntimeError(f"required Host Weight Runtime artifact was unavailable on group ranks {missing}")
            state = cast(_HWRState, state)
            state.outcome = ResolutionOutcome.CANONICAL_FALLBACK
            logger.info("BF16 HWR artifact unavailable on ranks %s; using canonical DLO", missing)
            return state, None

        contents = {record["content_digest"] for record in records}
        if len(contents) != 1:
            cast("HostWeightLease", lease).close()
            raise RuntimeError("BF16 HWR AllGather artifact content differs across ranks")
        return cast(_HWRState, state), cast("HostWeightLease", lease)

    def _run_hwr_phase(
        self,
        state: _HWRState,
        phase: str,
        action: Callable[[], _T],
    ) -> tuple[_T | None, Exception | None, list[int]]:
        value = None
        error = None
        try:
            value = action()
        except Exception as exc:
            error = exc
        records = self._coordinate_hwr_group(
            state.coordinator,
            phase,
            {
                "identity_digest": state.context.identity.key,
                "content_digest": state.artifact_content_digest,
                "status": "error" if error else "ready",
                "error_type": type(error).__name__ if error else None,
            },
        )
        failed = [cast(int, record["group_rank"]) for record in records if record["status"] != "ready"]
        return value, error, failed

    def _commit_hwr_restore(
        self,
        state: _HWRState,
        lease: HostWeightLease,
        restore_plan: object,
    ) -> HostWeightPlan:
        from vllm_omni.host_weight_runtime import HostWeightLeaseCarrier

        cast(Any, restore_plan).commit()
        metadata = cast(dict[str, object], state.context.identity.source.metadata.to_value())
        bindings = cast(list[dict[str, str]], metadata["target_bindings"])
        return HostWeightPlan(
            backing_kind="host_weight_runtime",
            bindings={name: TensorBinding(name, "") for name in state.context.tensor_names},
            planned_source_prefixes=frozenset(binding["source_prefix"] for binding in bindings),
            lease_carrier=HostWeightLeaseCarrier(lease),
        )

    def _restore_hwr_artifact(
        self,
        model: nn.Module,
        state: _HWRState,
        lease: HostWeightLease,
    ) -> _HWRState:
        from vllm_omni.diffusion.model_loader.host_weights import FinalLayoutTensorRestorer
        from vllm_omni.host_weight_runtime import ResolutionOutcome, RuntimeMode

        restore_plan, error, failed = self._run_hwr_phase(
            state,
            "restore_plan",
            lambda: FinalLayoutTensorRestorer(state.context).plan_restore(model, lease),
        )
        if failed:
            lease.close()
            if state.mode is RuntimeMode.PREFERRED:
                state.outcome = ResolutionOutcome.CANONICAL_FALLBACK
                state.allow_checkpoint_production = False
                logger.warning("HWR restore planning failed on ranks %s; using canonical loading", failed)
                return state
            if state.coordinator is None:
                raise cast(Exception, error)
            raise RuntimeError(f"required Host Weight Runtime restore planning failed on group ranks {failed}")

        committed, error, failed = self._run_hwr_phase(
            state,
            "restore_commit",
            lambda: self._commit_hwr_restore(state, lease, restore_plan),
        )
        if failed:
            lease.close()
            raise _HWRCommitError(
                f"Host Weight Runtime restore commit failed on group ranks {failed}; "
                "the restored model must be discarded"
            ) from error

        state.plan = cast(HostWeightPlan, committed)
        return state

    def _checkpoint_plan_for_group(
        self,
        state: _HWRState,
        plan: HostWeightPlan | None,
        fallback_reason: str | None,
    ) -> HostWeightPlan | None:
        records = self._coordinate_hwr_group(
            state.coordinator,
            "checkpoint_plan",
            {
                "identity_digest": state.context.identity.key,
                "status": "ready" if plan is not None else "unavailable",
                "error_type": None,
            },
        )
        unavailable = self._phase_ranks(records, "unavailable")
        if not unavailable:
            return cast(HostWeightPlan, plan)
        if len(unavailable) != len(records):
            raise RuntimeError(f"BF16 HWR checkpoint plan availability differs across ranks: {unavailable}")
        logger.info("Direct checkpoint HWR production unavailable; using canonical DLO: %s", fallback_reason)
        return None

    def _start_checkpoint_publication(
        self,
        model: nn.Module,
        modules: object,
        state: _HWRState,
        plan: HostWeightPlan,
    ) -> Callable[[], None] | None:
        from vllm_omni.diffusion.model_loader.host_weights import CheckpointPlanBF16Producer

        try:
            dit_modules = tuple(zip(getattr(modules, "dit_names", ()), getattr(modules, "dits", ())))
            producer = CheckpointPlanBF16Producer(state.context, model, dit_modules, plan)
        except Exception:
            logger.warning("Direct checkpoint HWR publication setup failed", exc_info=True)
            return None

        def publish() -> None:
            resolution = state.runtime.resolve(state.context.identity, producer=producer)
            if resolution.lease is None:
                raise RuntimeError(f"direct checkpoint HWR production returned {resolution.report.outcome.value}")
            resolution.lease.close()
            logger.info("BF16 HWR artifact published in parallel with checkpoint-backed DLO setup")

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hwr-publish")
        future = executor.submit(publish)

        def wait() -> None:
            try:
                future.result()
            except Exception:
                logger.warning("Direct checkpoint HWR publication failed", exc_info=True)
            finally:
                executor.shutdown()

        return wait

    def _select_checkpoint_plan(
        self,
        model: nn.Module,
        modules: object,
        state: _HWRState | None,
        plan: HostWeightPlan | None,
        fallback_reason: str | None,
    ) -> tuple[_HWRState | None, HostWeightPlan | None]:
        if state is None:
            return None, plan
        if not state.allow_checkpoint_production:
            return None, None
        plan = self._checkpoint_plan_for_group(state, plan, fallback_reason)
        if plan is None:
            return state, None
        self._checkpoint_publication_waiter = self._start_checkpoint_publication(model, modules, state, plan)
        return None, plan

    def _resolve_hwr(
        self,
        model: nn.Module,
        modules: object,
        *,
        dist_offload: bool,
        use_allgather: bool,
        load_format: str,
        sources: Sequence[_WeightSource],
    ) -> _HWRState | None:
        """Resolve one rank-symmetric final-layout artifact transaction."""
        from vllm_omni.host_weight_runtime import RuntimeMode

        raw_mode = getattr(self.od_config, "host_weight_runtime_mode", "disabled")
        if raw_mode == RuntimeMode.DISABLED or not dist_offload:
            self._hwr_eligibility_mode(
                model,
                modules,
                dist_offload=dist_offload,
                use_allgather=use_allgather,
                load_format=load_format,
            )
            return None

        coordinator = self._resolve_hwr_allgather_coordinator(use_allgather)
        mode = self._coordinate_hwr_eligibility(
            model,
            modules,
            coordinator=coordinator,
            dist_offload=dist_offload,
            use_allgather=use_allgather,
            load_format=load_format,
        )
        if mode is None:
            return None

        state, lease = self._coordinate_hwr_artifact(
            model,
            modules,
            coordinator=coordinator,
            mode=mode,
            load_format=load_format,
            sources=sources,
        )
        if state is not None:
            state.coordinator = coordinator
        if state is None or lease is None:
            return state
        state.artifact_content_digest = lease.provenance.artifact_content_sha256
        return self._restore_hwr_artifact(model, state, lease)

    def _run_hwr_warm_phase(
        self,
        state: _HWRState,
        phase: str,
        action: Callable[[], None],
    ) -> Exception | None:
        error = None
        try:
            action()
        except Exception as exc:
            error = exc
        if state.plan is None:
            return error
        records = self._coordinate_hwr_group(
            state.coordinator,
            phase,
            {
                "identity_digest": state.context.identity.key,
                "content_digest": state.artifact_content_digest,
                "status": "error" if error else "ready",
                "error_type": type(error).__name__ if error else None,
            },
        )
        failed = [cast(int, record["group_rank"]) for record in records if record["status"] != "ready"]
        if failed:
            return RuntimeError(f"BF16 HWR AllGather {phase} failed on group ranks {failed}")
        return error

    @staticmethod
    def _prepare_hwr_warm_retry(state: _HWRState, error: Exception, phase: str) -> None:
        from vllm_omni.host_weight_runtime import RuntimeMode

        plan = cast(HostWeightPlan, state.plan)
        cast("HostWeightLeaseCarrier", plan.lease_carrier).close()
        if state.mode is RuntimeMode.REQUIRED:
            raise error
        logger.warning("HWR %s failed; retrying with a fresh canonical model: %s", phase, error)

    def load_fresh_canonical_model(self) -> nn.Module:
        """Reload a disposable startup model without HWR or checkpoint mmap."""
        if self._last_load_request is None:
            raise RuntimeError("cannot construct a fresh canonical model before the initial load")
        request = dict(self._last_load_request)
        self._force_canonical_load = True
        try:
            return self.load_model(**request)
        finally:
            self._force_canonical_load = False

    def _publish_hwr_after_load(self, model: nn.Module, modules: object, state: _HWRState | None) -> None:
        from vllm_omni.diffusion.model_loader.host_weights import FinalLayoutBF16Producer
        from vllm_omni.host_weight_runtime import PostLoadPublicationOutcome, ResolutionOutcome

        if state is None or state.outcome is not ResolutionOutcome.CANONICAL_FALLBACK:
            return
        dit_modules = tuple(zip(getattr(modules, "dit_names", ()), getattr(modules, "dits", ())))
        try:
            producer = FinalLayoutBF16Producer(state.context, model, dit_modules)
            report = state.runtime.publish_after_load(state.context.identity, producer=producer)
        except Exception:
            logger.warning("Host Weight Runtime post-load publication failed", exc_info=True)
            return
        if report.outcome is PostLoadPublicationOutcome.FAILED:
            logger.warning("Host Weight Runtime post-load publication failed: %s", report.failure)

    def _attach_offload_startup_state(self, model: nn.Module) -> None:
        """Publish generic offload startup state on the loaded pipeline."""
        if self.host_weight_plan is None:
            return
        from vllm_omni.diffusion.offloader.startup import OffloadStartupState, attach_offload_startup_state

        hwr_state = self._hwr_state
        from vllm_omni.host_weight_runtime import RuntimeMode

        allow_retry = hwr_state is not None and hwr_state.plan is not None and hwr_state.mode is RuntimeMode.PREFERRED
        attach_offload_startup_state(
            model,
            OffloadStartupState(
                host_weight_plan=self.host_weight_plan,
                fresh_model_loader=self.load_fresh_canonical_model if allow_retry else None,
                allow_fresh_retry=allow_retry,
                after_backend_enable=self._checkpoint_publication_waiter,
            ),
        )
