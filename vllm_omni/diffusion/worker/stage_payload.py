from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.worker.omni_connector_model_runner_mixin import OmniConnectorModelRunnerMixin

logger = init_logger(__name__)


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device) if value.device != device else value
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        moved = [_to_device(item, device) for item in value]
        return moved if isinstance(value, list) else tuple(moved)
    return value


class DiffusionStagePayloadMixin(OmniConnectorModelRunnerMixin):
    """Adapt diffusion prompts and outputs to shared connector transport."""

    _STAGE_PAYLOAD_HANDLE_KEY = "_stage_payload_transfer"

    def _stage_payload_broadcast_groups(self) -> tuple[Any, ...]:
        from vllm_omni.diffusion.distributed.parallel_state import get_fs_group, get_hsdp_replicate_group

        hsdp_groups = []
        for get_group in (get_fs_group, get_hsdp_replicate_group):
            try:
                group = get_group()
            except AssertionError:
                continue
            if group is not None and getattr(group, "world_size", 1) > 1:
                hsdp_groups.append(group)
        if hsdp_groups:
            return tuple(hsdp_groups)

        groups = list(super()._stage_payload_broadcast_groups())
        try:
            from vllm_omni.diffusion.distributed.parallel_state import get_sp_group

            sp_group = get_sp_group()
        except (AssertionError, ImportError):
            sp_group = None
        if sp_group is not None and getattr(sp_group, "world_size", 1) > 1:
            groups.append(sp_group)
        return tuple(groups)

    def _maybe_recv_stage_payload(self, req: OmniDiffusionRequest) -> None:
        if OmniDiffusionRequest.is_dummy_run_request_id(req.request_id):
            return
        prompt = getattr(req, "prompt", None)
        if not isinstance(prompt, dict):
            return
        handle = prompt.pop(self._STAGE_PAYLOAD_HANDLE_KEY, None)
        expected_keys = tuple(getattr(self.od_config, "stage_input_payload_keys", ()) or ())
        if not isinstance(handle, dict) and not expected_keys:
            return
        from_stage, to_stage = self._kv_transfer_manager.recv_stages
        if not isinstance(handle, dict) and (from_stage is None or to_stage is None):
            raise RuntimeError(f"Stage {self.od_config.stage_id} expects a payload but has no incoming edge")
        sender_info = getattr(req, "payload_sender_info", None) or getattr(req, "kv_sender_info", None)
        if isinstance(sender_info, dict):
            sender_stage = handle.get("from_stage", from_stage) if isinstance(handle, dict) else from_stage
            sender_info = self._kv_transfer_manager._resolve_sender_info(sender_info, sender_stage)
        payload = self.recv_stage_payload(
            getattr(req, "external_req_id", None) or req.request_id,
            str(from_stage),
            str(to_stage),
            sender_info=sender_info,
            handle=handle if isinstance(handle, dict) else None,
        )
        additional = prompt.setdefault("additional_information", {})
        if isinstance(payload, dict):
            target_device = self._target_device or self.device
            for name, value in payload.items():
                if not expected_keys or name in expected_keys:
                    additional[name] = _to_device(value, target_device)
        required_keys = expected_keys or (handle.get("payload_keys", ()) if isinstance(handle, dict) else ())
        missing = [name for name in required_keys if additional.get(name) is None]
        if missing or (isinstance(handle, dict) and not required_keys and not isinstance(payload, dict)):
            raise RuntimeError(f"Stage payload unavailable for {req.request_id}; missing keys: {missing}")

    def _maybe_send_stage_payload(
        self,
        reqs: list[OmniDiffusionRequest],
        outputs: list[DiffusionOutput],
    ) -> None:
        """Publish complete leader outputs; keep inline values if the put fails.

        Pipelines must gather sharded outputs before calling this adapter.
        Every rank publishes the same handle and drops only transferred keys.
        """
        if all(OmniDiffusionRequest.is_dummy_run_request_id(req.request_id) for req in reqs):
            return
        payload_keys = tuple(getattr(self.od_config, "stage_output_payload_keys", ()) or ())
        if not payload_keys:
            return
        from_stage, to_stage = self._kv_transfer_manager.send_stages
        if not from_stage or not to_stage:
            logger.warning("Stage %s declares payload keys but has no outgoing edge", self.od_config.stage_id)
            return
        handles: dict[str, dict[str, Any]] = {}
        if self.is_data_transfer_rank():
            connector = self._stage_payload_connector()
            if connector is not None:
                for req, output in zip(reqs, outputs):
                    if OmniDiffusionRequest.is_dummy_run_request_id(req.request_id):
                        continue
                    custom = getattr(output, "custom_output", None)
                    if not isinstance(custom, dict):
                        continue
                    payload = {key: custom[key] for key in payload_keys if custom.get(key) is not None}
                    if not payload:
                        continue
                    external_req_id = getattr(req, "external_req_id", None) or req.request_id
                    _, _, put_key, _ = self._stage_payload_recv_spec(external_req_id, from_stage, to_stage)
                    try:
                        success, size, metadata = connector.put(from_stage, to_stage, put_key, payload)
                    except Exception as exc:
                        logger.warning("Stage payload put failed for %s: %s", put_key, exc)
                        continue
                    if not success:
                        logger.warning("Stage payload %s was rejected; keeping the inline payload", put_key)
                        continue
                    handles[req.request_id] = {
                        "key": put_key,
                        "from_stage": from_stage,
                        "to_stage": to_stage,
                        "size_bytes": int(size),
                        "metadata": metadata,
                        "payload_keys": list(payload),
                    }
                    logger.debug("Stage payload put %s size=%s keys=%s", put_key, size, list(payload))
        handles = self._broadcast_tp_payload_packet(handles) or {}
        for req, output in zip(reqs, outputs):
            handle = handles.get(req.request_id)
            custom = getattr(output, "custom_output", None)
            if handle is None or not isinstance(custom, dict):
                continue
            custom[self._STAGE_PAYLOAD_HANDLE_KEY] = handle
            for name in handle["payload_keys"]:
                custom.pop(name, None)
