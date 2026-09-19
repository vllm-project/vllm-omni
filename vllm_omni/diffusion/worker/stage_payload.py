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
        prompt = getattr(req, "prompt", None)
        if not isinstance(prompt, dict):
            return
        handle = prompt.pop(self._STAGE_PAYLOAD_HANDLE_KEY, None)
        expected_keys = tuple(getattr(self.od_config, "stage_input_payload_keys", ()) or ())
        if not isinstance(handle, dict) and not expected_keys:
            return
        from_stage, to_stage = self.kv_transfer_manager.recv_stages
        if not isinstance(handle, dict) and (from_stage is None or to_stage is None):
            logger.warning("Stage %s expects a payload but has no incoming edge", self.od_config.stage_id)
            return
        sender_info = getattr(req, "payload_sender_info", None) or getattr(req, "kv_sender_info", None)
        if isinstance(sender_info, dict) and "host" not in sender_info:
            sender_info = sender_info.get(0, sender_info.get("0"))
        payload = self.recv_stage_payload(
            getattr(req, "external_req_id", None) or req.request_id,
            str(from_stage),
            str(to_stage),
            sender_info=sender_info,
            handle=handle if isinstance(handle, dict) else None,
        )
        if not isinstance(payload, dict):
            return
        target_device = self._target_device or self.device
        additional = prompt.setdefault("additional_information", {})
        for name, value in payload.items():
            if not expected_keys or name in expected_keys:
                additional[name] = _to_device(value, target_device)

    def _maybe_send_stage_payload(
        self,
        reqs: list[OmniDiffusionRequest],
        outputs: list[DiffusionOutput],
    ) -> None:
        """Publish complete leader outputs; keep inline values if the put fails.

        Pipelines must gather sharded outputs before calling this adapter.
        Every rank publishes the same handle and drops only transferred keys.
        """
        payload_keys = tuple(getattr(self.od_config, "stage_output_payload_keys", ()) or ())
        if not payload_keys:
            return
        from_stage, to_stage = self.kv_transfer_manager.send_stages
        if not from_stage or not to_stage:
            logger.warning("Stage %s declares payload keys but has no outgoing edge", self.od_config.stage_id)
            return
        handles: dict[str, dict[str, Any]] = {}
        if self.is_data_transfer_rank():
            connector = self._stage_payload_connector()
            if connector is not None:
                for req, output in zip(reqs, outputs):
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
