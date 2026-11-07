# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""PD (Prefill-Decode) disaggregation helpers.

Mixin for OmniBase — keeps omni.py focused on orchestration.
"""

import logging
import threading
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm import SamplingParams

# vLLM's VLLM_MOONCAKE_BOOTSTRAP_PORT default, also the stage runtime's default
# for kv_connector_extra_config.bootstrap_port.
_DEFAULT_MOONCAKE_BOOTSTRAP_PORT = 8998

logger = logging.getLogger(__name__)


def mooncake_prefill_bootstrap_addr(kv_cfg: Mapping[str, Any], replica_id: int = 0) -> str | None:
    """Bootstrap URL of a MooncakeConnector prefill replica, or ``None``.

    Mirrors ``StageRuntime._prepare_replica_vllm_config``: replica ``r`` of a
    Mooncake ``kv_producer`` listens on
    ``kv_connector_extra_config.bootstrap_port`` (default 8998) + ``r``.
    """
    if kv_cfg.get("kv_connector") != "MooncakeConnector":
        return None
    extra = kv_cfg.get("kv_connector_extra_config") or {}
    port = int(extra.get("bootstrap_port", _DEFAULT_MOONCAKE_BOOTSTRAP_PORT)) + replica_id
    host = kv_cfg.get("kv_ip") or "127.0.0.1"
    return f"http://{host}:{port}"


def reject_pd_until_runtime_ready(stage_configs: Sequence[Any]) -> None:
    """Refuse to launch a detected PD pair: its prefill->decode handoff is not implemented.

    The orchestrator needs ``remote_request_id`` from the prefill output, which
    the pinned MooncakeConnector never returns, so every request would fail at
    the handoff after both engines loaded; no other connector has been run on
    this path. Each launch path calls this right after its stage configs are
    resolved, before it launches any stage: ``OmniEngineBase.__init__``
    (``Omni``/``AsyncOmni`` and ``vllm serve`` with one API server), the
    ``--api-server-count`` > 1 parent in ``_build_multi_api_stage_runtime`` and
    ``run_headless``.
    """
    # TODO: remove once the PD runtime handoff lands (Mooncake transfer_id
    # contract, decode request rebuilt from the processed prefill request),
    # tracked in https://github.com/vllm-project/vllm-omni/pull/2947.
    pair = PDDisaggregationMixin.detect_pd_separation_from_stage_configs(list(stage_configs))
    if pair is None:
        return
    raise NotImplementedError(
        f"Stages declared with pd_role form a PD pair (prefill stage {pair[0]}, decode stage {pair[1]}), "
        "but PD disaggregation cannot run yet: the prefill->decode KV handoff is not implemented, so "
        "requests would fail at the handoff. Remove pd_role from these stages to launch the pipeline."
    )


class PDDisaggregationMixin:
    """Mixin supplying PD disaggregation helpers to OmniBase."""

    def _get_pd_separation_pair(self) -> tuple[int, int] | None:
        """PD prefill/decode indices when ``_init_pd_state`` ran; else ``None``.

        Partial test doubles may skip ``OmniBase.__init__``; treat missing state as
        no PD disaggregation instead of raising ``AttributeError``.
        """
        return getattr(self, "_pd_separation_pair", None)

    def _init_pd_state(self) -> None:
        """Initialise PD disaggregation state."""
        self._pd_separation_pair: tuple[int, int] | None = self.detect_pd_separation_from_stage_configs(
            self.stage_configs
        )
        self._pd_connector_info: dict[str, Any] | None = None
        self._pd_kv_params_by_req: dict[str, dict[str, Any]] = {}
        self._pd_kv_params_lock = threading.Lock()
        if self._pd_separation_pair is not None:
            self._validate_pd_separation_config()
            self._pd_connector_info = self._get_pd_connector_info()
            p_id, d_id = self._pd_separation_pair
            logger.info(
                "[%s] PD disaggregation detected: prefill=stage-%d, decode=stage-%d",
                self._name,
                p_id,
                d_id,
            )

    @staticmethod
    def detect_pd_separation_from_stage_configs(stage_configs: list[Any]) -> tuple[int, int] | None:
        """Scan stage configs for a prefill/decode pair.

        Returns:
            (prefill_idx, decode_idx) if one pair exists, None if not found.

        Raises:
            ValueError: if multiple candidate PD pairs are found.
        """
        prefill_by_id: dict[int, int] = {}
        decode_indices: list[int] = []
        for i, stage in enumerate(stage_configs):
            pd_role = getattr(stage, "pd_role", None)
            pd_role_value = getattr(pd_role, "value", pd_role)
            is_prefill = pd_role_value == "prefill"
            is_decode = pd_role_value == "decode"

            # Keep the legacy boolean markers as a fallback for hand-built
            # stage configs that do not carry ``pd_role``.
            if pd_role_value is None:
                is_prefill = bool(getattr(stage, "is_prefill_only", False))
                is_decode = bool(getattr(stage, "is_decode_only", False))

            if is_prefill:
                prefill_by_id[i] = i
                sid = getattr(stage, "stage_id", i)
                if sid != i:
                    prefill_by_id[sid] = i
            if is_decode:
                decode_indices.append(i)

        pd_pairs: list[tuple[int, int]] = []
        for j in decode_indices:
            source_ids = PDDisaggregationMixin._get_stage_input_sources(stage_configs[j])
            for src in source_ids:
                if src in prefill_by_id:
                    pd_pairs.append((prefill_by_id[src], j))
                    break

        if len(pd_pairs) > 1:
            raise ValueError(
                f"Multiple PD pairs detected ({pd_pairs}); only a single PD pair per pipeline is supported"
            )
        return pd_pairs[0] if pd_pairs else None

    @staticmethod
    def _to_dict(obj: Any, default: Any = None) -> dict[str, Any] | None:
        """Convert obj to a plain dict, trying several strategies."""
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
                logger.debug("Unable to convert %s to dict", type(obj))
                return default

    @staticmethod
    def _get_stage_input_sources(stage: Any) -> list[int]:
        """Upstream stage ids: legacy ``engine_input_source`` or typed ``input_sources``."""
        source_ids = getattr(stage, "engine_input_source", None)
        if source_ids is None:
            source_ids = getattr(stage, "input_sources", None)
        return list(source_ids or [])

    @staticmethod
    def _get_stage_kv_transfer_config(stage: Any) -> Any:
        """KV transfer config from legacy ``engine_args`` or typed ``connector_config``."""
        engine_args = getattr(stage, "engine_args", None)
        if engine_args is not None:
            cfg = getattr(engine_args, "kv_transfer_config", None)
            if cfg is None and hasattr(engine_args, "get"):
                cfg = engine_args.get("kv_transfer_config", None)
            if cfg is not None:
                return cfg
        return getattr(getattr(stage, "connector_config", None), "kv_transfer_config", None)

    @staticmethod
    def _get_stage_tensor_parallel_size(stage: Any) -> int:
        engine_args = getattr(stage, "engine_args", None)
        tp = getattr(engine_args, "tensor_parallel_size", None) if engine_args is not None else None
        if tp is None:
            tp = getattr(getattr(stage, "parallel_config", None), "tensor_parallel_size", None)
        return int(tp or 1)

    def _kv_cfg_to_dict(self, kv_cfg: Any) -> dict[str, Any]:
        return self._to_dict(kv_cfg, default={}) or {}

    def _normalize_kv_transfer_params(self, kv_params: Any) -> dict[str, Any] | None:
        return self._to_dict(kv_params)

    def _validate_pd_separation_config(self) -> None:
        """Validate PD stage configurations are consistent."""
        pair = self._get_pd_separation_pair()
        assert pair is not None
        p_id, d_id = pair
        p_stage = self.stage_configs[p_id]
        d_stage = self.stage_configs[d_id]

        def _get_kv_cfg(stage: Any) -> dict[str, Any]:
            cfg = self._get_stage_kv_transfer_config(stage)
            if cfg is None:
                raise ValueError(f"Stage-{stage.stage_id} is marked for PD but has no 'kv_transfer_config'")
            cfg_dict = self._kv_cfg_to_dict(cfg)
            if not cfg_dict:
                raise ValueError(f"Stage-{stage.stage_id} kv_transfer_config could not be parsed")
            return cfg_dict

        p_cfg = _get_kv_cfg(p_stage)
        d_cfg = _get_kv_cfg(d_stage)

        p_role = p_cfg.get("kv_role")
        d_role = d_cfg.get("kv_role")
        if p_role not in ("kv_producer", "kv_both"):
            raise ValueError(f"Prefill stage-{p_id} kv_role must be 'kv_producer' or 'kv_both', got '{p_role}'")
        if d_role not in ("kv_consumer", "kv_both"):
            raise ValueError(f"Decode stage-{d_id} kv_role must be 'kv_consumer' or 'kv_both', got '{d_role}'")

        d_sources = self._get_stage_input_sources(d_stage)
        if p_id not in d_sources and p_stage.stage_id not in d_sources:
            raise ValueError(f"Decode stage-{d_id} must list prefill stage-{p_id} in engine_input_source")

        p_conn = p_cfg.get("kv_connector")
        d_conn = d_cfg.get("kv_connector")
        if p_conn != d_conn:
            raise ValueError(f"PD connector mismatch: prefill uses '{p_conn}', decode uses '{d_conn}'")
        if not p_conn:
            raise ValueError("PD requires kv_connector in kv_transfer_config")

        for key in ("kv_buffer_device", "kv_buffer_size"):
            p_val = p_cfg.get(key)
            d_val = d_cfg.get(key)
            if p_val is not None and d_val is not None and p_val != d_val:
                raise ValueError(f"PD {key} mismatch: prefill='{p_val}', decode='{d_val}'")

        p_tp = self._get_stage_tensor_parallel_size(p_stage)
        d_tp = self._get_stage_tensor_parallel_size(d_stage)
        if p_tp != d_tp:
            raise ValueError(f"PD stages must have matching tensor_parallel_size: prefill={p_tp}, decode={d_tp}")

    def _get_pd_connector_info(self) -> dict[str, Any] | None:
        """Extract prefill engine KV connector info."""
        pair = self._get_pd_separation_pair()
        if pair is None:
            return None

        p_id, _ = pair
        p_stage = self.stage_configs[p_id]

        kv_cfg = self._get_stage_kv_transfer_config(p_stage)
        if kv_cfg is None:
            return None

        kv_cfg_dict = self._kv_cfg_to_dict(kv_cfg)
        if not kv_cfg_dict:
            return None
        extra_cfg = kv_cfg_dict.get("kv_connector_extra_config")
        if extra_cfg is not None and not isinstance(extra_cfg, dict):
            kv_cfg_dict = {**kv_cfg_dict, "kv_connector_extra_config": self._kv_cfg_to_dict(extra_cfg)}

        info: dict[str, Any] = {}
        # Replica 0's address; the orchestrator reads the per-replica value
        # from the prefill client's KV transfer config.
        bootstrap_addr = mooncake_prefill_bootstrap_addr(kv_cfg_dict)
        if bootstrap_addr is not None:
            info["prefill_bootstrap_addr"] = bootstrap_addr
        return info

    def _prepare_prefill_sampling_params(self, req_id: str, sp: "SamplingParams") -> "SamplingParams":
        sp = sp.clone()
        sp.max_tokens = 1
        if hasattr(sp, "min_tokens"):
            try:
                sp.min_tokens = 1
            except Exception:
                pass
        # MooncakeConnector cancels KV transfer unless finish_reason='length'
        sp.stop = []
        sp.stop_token_ids = []
        sp.include_stop_str_in_output = False
        if sp.extra_args is None:
            sp.extra_args = {}
        kv_params = self._normalize_kv_transfer_params(sp.extra_args.get("kv_transfer_params"))
        merged: dict[str, Any] = {}
        if kv_params:
            merged.update(kv_params)
        merged.update(
            {
                "do_remote_decode": True,
                "do_remote_prefill": False,
                "transfer_id": f"xfer-{req_id}",
            }
        )
        sp.extra_args["kv_transfer_params"] = merged
        logger.debug("[PD] prefill SP: req=%s kv_transfer_params=%s", req_id, merged)
        return sp

    def _pop_pd_kv_params(self, req_id: str, fallback: Any | None = None) -> dict[str, Any] | None:
        kv_params = self._normalize_kv_transfer_params(fallback)
        with self._pd_kv_params_lock:
            stored = self._pd_kv_params_by_req.pop(req_id, None)
        if kv_params is None:
            kv_params = stored
        return kv_params

    def _drop_pd_kv_params(self, req_id: str) -> None:
        with self._pd_kv_params_lock:
            self._pd_kv_params_by_req.pop(req_id, None)

    def _extract_kv_transfer_params(self, engine_outputs: Any) -> dict[str, Any] | None:
        """Extract kv_transfer_params from engine outputs (if available)."""
        outputs = engine_outputs if isinstance(engine_outputs, list) else [engine_outputs]
        for output in outputs:
            kv_params = getattr(output, "kv_transfer_params", None)
            if kv_params is not None:
                return self._normalize_kv_transfer_params(kv_params)
        return None

    def _is_pd_routing(self, stage_id: int, next_stage_id: int) -> bool:
        """True when edge stage_id → next_stage_id is the prefill→decode boundary."""
        pair = self._get_pd_separation_pair()
        return pair is not None and pair == (stage_id, next_stage_id)

    def _maybe_expand_sampling_params(self, sampling_params_list: list) -> list:
        """Auto-duplicate thinker SP for decode stage when user provides N-1 params."""
        pair = self._get_pd_separation_pair()
        if pair is None:
            return sampling_params_list
        if len(sampling_params_list) != len(self.stage_configs) - 1:
            return sampling_params_list
        p_id, d_id = pair
        sp_list = list(sampling_params_list)
        sp_list.insert(d_id, sp_list[p_id])
        return sp_list

    def _build_decode_kv_params(
        self,
        req_id: str,
        sp_next: "SamplingParams",
        prefill_kv_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build decode-side kv_transfer_params dict."""
        decode_kv_params: dict[str, Any] = {
            "do_remote_decode": False,
            "do_remote_prefill": True,
            "transfer_id": f"xfer-{req_id}",
        }

        if sp_next.extra_args:
            existing = self._normalize_kv_transfer_params(sp_next.extra_args.get("kv_transfer_params"))
            if existing:
                decode_kv_params.update(existing)

        if self._pd_connector_info:
            baddr = self._pd_connector_info.get("prefill_bootstrap_addr")
            if baddr is not None and "remote_bootstrap_addr" not in decode_kv_params:
                decode_kv_params["remote_bootstrap_addr"] = baddr

        if prefill_kv_params:
            decode_kv_params.update(prefill_kv_params)

        decode_kv_params["do_remote_prefill"] = True
        decode_kv_params["do_remote_decode"] = False
        if not decode_kv_params.get("transfer_id"):
            decode_kv_params["transfer_id"] = f"xfer-{req_id}"

        return decode_kv_params

    def _prepare_pd_decode_routing(
        self,
        req_id: str,
        prompt: Any,
        sp_template: "SamplingParams",
        prefill_kv_fallback: Any | None = None,
        engine_outputs: Any | None = None,
    ) -> tuple[list, "SamplingParams"]:
        """Prepare next_inputs and sampling params for prefill→decode routing.

        Used by both sync (omni.py) and async (async_omni.py) paths.

        Returns:
            (next_inputs, sp_next) ready to submit to the decode stage.
        """
        next_inputs = [prompt] if not isinstance(prompt, list) else prompt

        sp_next = sp_template.clone()
        if sp_next.extra_args is None:
            sp_next.extra_args = {}

        if engine_outputs is not None:
            prefill_kv = self._extract_kv_transfer_params(engine_outputs)
        else:
            prefill_kv = self._pop_pd_kv_params(req_id, prefill_kv_fallback)

        decode_kv_params = self._build_decode_kv_params(req_id, sp_next, prefill_kv)
        sp_next.extra_args["kv_transfer_params"] = decode_kv_params

        logger.debug(
            "[PD] routing req=%s, remote_request_id=%s",
            req_id,
            decode_kv_params.get("remote_request_id", "NOT SET"),
        )
        if "remote_request_id" not in decode_kv_params:
            logger.warning(
                "[PD] remote_request_id NOT SET for req %s. Apply mooncake_connector.py patch to fix.",
                req_id,
            )

        return next_inputs, sp_next
