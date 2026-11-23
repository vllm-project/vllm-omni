# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Adapter-only IPC updates using vLLM's existing four-phase session."""

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory
from vllm.distributed.weight_transfer.ipc_engine import IPCWeightTransferEngine, IPCWeightTransferUpdateInfo

from vllm_omni.lora.manager import LoRAManager
from vllm_omni.lora.merged_weights import check_merge_error
from vllm_omni.lora.request import TensorLoRARequest


@dataclass
class MergedLoRAIPCUpdateInfo(IPCWeightTransferUpdateInfo):
    """IPC tensor metadata plus PEFT config; omit config for a full base update."""

    peft_config: dict | None = None
    lora_int_id: int = 1


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

    def start_weight_update(self) -> None:
        """Begin a session; its first chunk selects dense or adapter loading."""
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

    def receive_weights(self, update_info: MergedLoRAIPCUpdateInfo) -> None:
        """Receive unchanged IPC tensors with optional adapter metadata.

        Args:
            update_info: IPC handles and PEFT configuration repeated per chunk.

        Raises:
            ValueError: A session mixes adapter identities, configs or update kinds.
        """
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

    def update_weights(self, update_info: dict) -> None:
        """Receive a chunk, discarding staged tensors if receiving fails.

        Args:
            update_info: Serialized ``MergedLoRAIPCUpdateInfo``.
        """
        try:
            super().update_weights(update_info)
        except Exception as error:
            self._error = error
        try:
            check_merge_error(self._error, self.cpu_group)
        except Exception:
            self._staged.tensors.clear()
            self._packed_importer.close()
            raise

    def finish_weight_update(self) -> None:
        """Validate all ranks before copying a new policy into live storage."""
        try:
            if self._kind == "dense":
                super().finish_weight_update()
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
        finally:
            self._staged.tensors.clear()
            self._packed_importer.close()
            self.reset_weight_update_target()

    def shutdown(self) -> None:
        """Disconnect the trainer while retaining the live merged policy."""
        if hasattr(self, "_staged"):
            self._staged.tensors.clear()
        super().shutdown()


WeightTransferEngineFactory.register_engine("omni_lora_ipc", MergedLoRAIPCWeightTransferEngine)
