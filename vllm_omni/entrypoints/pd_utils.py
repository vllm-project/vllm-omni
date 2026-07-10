# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PD (Prefill-Decode) disaggregation helpers.

Mixin for OmniBase — keeps omni.py focused on orchestration.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm import SamplingParams

_DEFAULT_MOONCAKE_BOOTSTRAP_PORT = 25201
_VLLM_MOONCAKE_BOOTSTRAP_PORT_ENV = "VLLM_MOONCAKE_BOOTSTRAP_PORT"
_PD_PREFILL_PICK_STRATEGY_ENV = "VLLM_OMNI_PD_PREFILL_PICK_STRATEGY"
_PD_PREFILL_PICK_STRATEGIES = ("round_robin", "least_inflight")
_DEFAULT_PD_PREFILL_PICK_STRATEGY = "round_robin"

logger = logging.getLogger(__name__)


class PDDisaggregationMixin:
    def _get_pd_prefill_ids(self) -> list[int]:
        return list(getattr(self, "_pd_prefill_ids", []) or [])

    def _get_pd_decode_id(self) -> int | None:
        ids = self._get_pd_decode_ids()
        return ids[0] if ids else None

    def _get_pd_decode_ids(self) -> list[int]:
        return list(getattr(self, "_pd_decode_ids", []) or [])

    def _init_pd_state(self) -> None:
        topology = self.detect_pd_separation_topology(self.stage_configs)
        self._pd_prefill_ids = list(topology["prefill_ids"]) if topology else []
        self._pd_decode_ids = list(topology.get("decode_ids") or []) if topology else []
        self._pd_decode_to_prefill = dict(topology.get("decode_to_prefill") or {}) if topology else {}
        self._pd_decode_id = self._pd_decode_ids[0] if self._pd_decode_ids else None

        if self._pd_decode_id is not None and self._pd_prefill_ids:
            try:
                from ..distributed.kv_transfer.mooncake_pd_patch import apply_mooncake_connector_patch

                apply_mooncake_connector_patch()
            except Exception as e:
                logger.warning(
                    "[%s] Failed to apply MooncakeConnector patch: %s. PD KV transfer may fail.",
                    self._name,
                    e,
                )
            self._validate_pd_separation_config()

        self._pd_prefill_pick_strategy = self._resolve_pd_prefill_pick_strategy()
        self._pd_prefill_inflight = {p: 0 for p in self._pd_prefill_ids}
        self._pd_prefill_inflight_lock = threading.Lock()
        self._pd_prefill_round_robin_idx = 0

    @staticmethod
    def detect_pd_separation_topology(stage_configs: list[Any]) -> dict[str, Any] | None:
        prefill_by_id: dict[int, int] = {}
        decode_indices: list[int] = []
        for i, stage in enumerate(stage_configs):
            if getattr(stage, "is_prefill_only", False):
                prefill_by_id[i] = i
                sid = getattr(stage, "stage_id", i)
                if sid != i:
                    prefill_by_id[sid] = i
            if getattr(stage, "is_decode_only", False):
                decode_indices.append(i)

        if not decode_indices:
            return None

        prefill_ids: list[int] = []
        decode_to_prefill: dict[int, list[int]] = {}
        for d_id in decode_indices:
            decode_stage = stage_configs[d_id]
            source_ids = list(getattr(decode_stage, "engine_input_source", []) or [])
            this_decode_prefills: list[int] = []
            for src in source_ids:
                if src not in prefill_by_id:
                    continue
                p_idx = prefill_by_id[src]
                if p_idx not in this_decode_prefills:
                    this_decode_prefills.append(p_idx)
                if p_idx not in prefill_ids:
                    prefill_ids.append(p_idx)
            decode_to_prefill[d_id] = this_decode_prefills

        if not prefill_ids:
            return None

        return {
            "prefill_ids": prefill_ids,
            "decode_id": decode_indices[0],
            "decode_ids": list(decode_indices),
            "decode_to_prefill": decode_to_prefill,
        }

    @staticmethod
    def _to_dict(obj: Any, default: Any = None) -> dict[str, Any] | None:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj
        if is_dataclass(obj):
            try:
                return asdict(obj)
            except Exception:
                return default
        for attr in ("model_dump", "dict"):
            if hasattr(obj, attr):
                try:
                    return getattr(obj, attr)()
                except Exception:
                    pass
        if hasattr(obj, "items"):
            try:
                return dict(obj)
            except Exception:
                pass
        try:
            return dict(obj)
        except Exception:
            try:
                return vars(obj)
            except Exception:
                return default

    def _kv_cfg_to_dict(self, kv_cfg: Any) -> dict[str, Any]:
        return self._to_dict(kv_cfg, default={}) or {}

    def _normalize_kv_transfer_params(self, kv_params: Any) -> dict[str, Any] | None:
        return self._to_dict(kv_params)

    def _validate_pd_separation_config(self) -> None:
        prefill_ids = self._get_pd_prefill_ids()
        decode_ids = self._get_pd_decode_ids()
        assert prefill_ids and decode_ids

        def _get_kv_cfg(stage: Any) -> dict[str, Any]:
            ea = stage.engine_args
            cfg = getattr(ea, "kv_transfer_config", None)
            if cfg is None and hasattr(ea, "get"):
                cfg = ea.get("kv_transfer_config")
            if cfg is None:
                raise ValueError(f"Stage-{stage.stage_id} is marked for PD but has no kv_transfer_config")
            cfg_dict = self._kv_cfg_to_dict(cfg)
            if not cfg_dict:
                raise ValueError(f"Stage-{stage.stage_id} kv_transfer_config could not be parsed")
            return cfg_dict

        decode_cfgs: dict[int, dict[str, Any]] = {}
        for d_id in decode_ids:
            d_cfg = _get_kv_cfg(self.stage_configs[d_id])
            d_role = d_cfg.get("kv_role")
            if d_role not in ("kv_consumer", "kv_both"):
                raise ValueError(f"Decode stage-{d_id} kv_role must be kv_consumer or kv_both, got {d_role!r}")
            d_conn = d_cfg.get("kv_connector")
            if not d_conn:
                raise ValueError("PD requires kv_connector in decode kv_transfer_config")
            decode_cfgs[d_id] = d_cfg

        def _validate_pair(p_id: int, d_id: int) -> None:
            d_stage = self.stage_configs[d_id]
            d_cfg = decode_cfgs[d_id]
            d_sources = list(getattr(d_stage, "engine_input_source", []) or [])
            d_tp = getattr(getattr(d_stage, "engine_args", None), "tensor_parallel_size", 1)

            p_stage = self.stage_configs[p_id]
            p_cfg = _get_kv_cfg(p_stage)
            p_role = p_cfg.get("kv_role")
            if p_role not in ("kv_producer", "kv_both"):
                raise ValueError(f"Prefill stage-{p_id} kv_role must be kv_producer or kv_both, got {p_role!r}")
            if p_id not in d_sources and p_stage.stage_id not in d_sources:
                raise ValueError(f"Decode stage-{d_id} must list prefill stage-{p_id} in engine_input_source")
            if p_cfg.get("kv_connector") != d_cfg.get("kv_connector"):
                raise ValueError(
                    f"PD connector mismatch: prefill stage-{p_id} uses {p_cfg.get('kv_connector')!r}, "
                    f"decode stage-{d_id} uses {d_cfg.get('kv_connector')!r}"
                )
            for key in ("kv_buffer_device", "kv_buffer_size"):
                p_val = p_cfg.get(key)
                d_val = d_cfg.get(key)
                if p_val is not None and d_val is not None and p_val != d_val:
                    raise ValueError(
                        f"PD {key} mismatch: prefill stage-{p_id}={p_val!r}, decode stage-{d_id}={d_val!r}"
                    )
            p_tp = getattr(getattr(p_stage, "engine_args", None), "tensor_parallel_size", 1)
            if p_tp != d_tp:
                raise ValueError(
                    f"PD stages must have matching tensor_parallel_size: "
                    f"prefill stage-{p_id}={p_tp}, decode stage-{d_id}={d_tp}"
                )

        for d_id in decode_ids:
            for p_id in self._pd_decode_to_prefill.get(d_id, []):
                _validate_pair(p_id, d_id)

        bootstrap_ports: dict[int, int] = {}
        all_pd_stages = [(p_id, _get_kv_cfg(self.stage_configs[p_id])) for p_id in prefill_ids]
        all_pd_stages.extend((d_id, decode_cfgs[d_id]) for d_id in decode_ids)
        for stage_id, cfg in all_pd_stages:
            if "mooncake" not in str(cfg.get("kv_connector")).lower():
                continue
            stage = self.stage_configs[stage_id]
            env_port = self._read_runtime_env_bootstrap_port(stage)
            extra = cfg.get("kv_connector_extra_config", {}) or {}
            if not isinstance(extra, dict):
                extra = self._kv_cfg_to_dict(extra)
            extra_port = extra.get("mooncake_bootstrap_port")
            if env_port is not None and extra_port is not None and int(env_port) != int(extra_port):
                raise ValueError(
                    f"Stage-{stage_id} mooncake bootstrap port mismatch: "
                    f"runtime.env={env_port}, extra_config={extra_port}"
                )
            port = (
                int(env_port)
                if env_port is not None
                else int(extra_port)
                if extra_port is not None
                else _DEFAULT_MOONCAKE_BOOTSTRAP_PORT
            )
            if port in bootstrap_ports:
                raise ValueError(
                    f"Mooncake bootstrap port collision: stage-{bootstrap_ports[port]} "
                    f"and stage-{stage_id} both use port {port}"
                )
            bootstrap_ports[port] = stage_id

    @staticmethod
    def _read_runtime_env_bootstrap_port(stage: Any) -> int | None:
        runtime_cfg = getattr(stage, "runtime", None)
        if runtime_cfg is None and hasattr(stage, "get"):
            runtime_cfg = stage.get("runtime")
        if runtime_cfg is None:
            return None
        env_section = runtime_cfg.get("env") if hasattr(runtime_cfg, "get") else getattr(runtime_cfg, "env", None)
        if env_section is None:
            return None
        try:
            raw = (
                env_section.get(_VLLM_MOONCAKE_BOOTSTRAP_PORT_ENV)
                if hasattr(env_section, "get")
                else env_section[_VLLM_MOONCAKE_BOOTSTRAP_PORT_ENV]
            )
        except Exception:
            return None
        if raw is None:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            logger.warning(
                "[PD] runtime.env.%s=%r on stage-%s is not an int; ignoring.",
                _VLLM_MOONCAKE_BOOTSTRAP_PORT_ENV,
                raw,
                getattr(stage, "stage_id", "?"),
            )
            return None

    def _resolve_pd_prefill_pick_strategy(self) -> str:
        candidate = getattr(self, "_pd_prefill_pick_strategy_override", None)
        if not candidate:
            candidate = os.getenv(_PD_PREFILL_PICK_STRATEGY_ENV) or None
        if not candidate:
            return _DEFAULT_PD_PREFILL_PICK_STRATEGY
        candidate = str(candidate).strip().lower()
        if candidate not in _PD_PREFILL_PICK_STRATEGIES:
            logger.warning(
                "[PD] Unknown pd_prefill_pick_strategy=%r; falling back to %r. Allowed: %s",
                candidate,
                _DEFAULT_PD_PREFILL_PICK_STRATEGY,
                list(_PD_PREFILL_PICK_STRATEGIES),
            )
            return _DEFAULT_PD_PREFILL_PICK_STRATEGY
        return candidate

    def _pick_prefill_stage(self, *_args: Any) -> int:
        del _args
        prefill_ids = self._get_pd_prefill_ids()
        if not prefill_ids:
            raise RuntimeError("_pick_prefill_stage called but no PD prefill stages are configured")
        if len(prefill_ids) == 1:
            return prefill_ids[0]

        strategy = getattr(self, "_pd_prefill_pick_strategy", _DEFAULT_PD_PREFILL_PICK_STRATEGY)
        with self._pd_prefill_inflight_lock:
            if strategy == "least_inflight":
                chosen = min(prefill_ids, key=lambda p: (self._pd_prefill_inflight.get(p, 0), p))
            else:
                idx = self._pd_prefill_round_robin_idx % len(prefill_ids)
                chosen = prefill_ids[idx]
                self._pd_prefill_round_robin_idx = (idx + 1) % len(prefill_ids)
            self._pd_prefill_inflight[chosen] = self._pd_prefill_inflight.get(chosen, 0) + 1
        return chosen

    def _prepare_prefill_sampling_params(self, req_id: str, sp: SamplingParams) -> SamplingParams:
        sp = sp.clone()
        sp.max_tokens = 1
        if hasattr(sp, "min_tokens"):
            try:
                sp.min_tokens = 0
            except Exception:
                pass
        sp.stop = []
        sp.stop_token_ids = []
        sp.include_stop_str_in_output = False
        if sp.extra_args is None:
            sp.extra_args = {}
        kv_params = self._normalize_kv_transfer_params(sp.extra_args.get("kv_transfer_params"))
        merged = dict(kv_params or {})
        merged.update({"do_remote_decode": True, "do_remote_prefill": False, "transfer_id": f"xfer-{req_id}"})
        sp.extra_args["kv_transfer_params"] = merged
        return sp

    def _maybe_expand_sampling_params(self, sampling_params_list: list) -> list:
        prefill_ids = self._get_pd_prefill_ids()
        decode_ids = self._get_pd_decode_ids()
        if not prefill_ids or not decode_ids:
            return sampling_params_list

        if len(prefill_ids) == 1 and len(decode_ids) == 1:
            decode_id = decode_ids[0]
            if len(sampling_params_list) != len(self.stage_configs) - 1:
                return sampling_params_list
            sp_list = list(sampling_params_list)
            sp_list.insert(decode_id, sp_list[prefill_ids[0]])
            return sp_list

        pd_slots = set(prefill_ids) | set(decode_ids)
        expected_caller_len = 1 + len(self.stage_configs) - len(pd_slots)
        if len(sampling_params_list) != expected_caller_len:
            return sampling_params_list

        sp_list = list(sampling_params_list)
        prefill_template = sp_list[0]
        non_pd_iter = iter(sp_list[1:])
        result = []
        for stage_idx in range(len(self.stage_configs)):
            if stage_idx in pd_slots:
                result.append(prefill_template)
            else:
                result.append(next(non_pd_iter, prefill_template))
        return result
