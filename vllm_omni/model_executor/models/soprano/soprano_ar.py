# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Omni ABI adaptation; all AR computation/loading/KV remain native Qwen3."""

import os
from collections.abc import Iterable

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)


class SopranoForCausalLM(Qwen3ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        cfg = vllm_config.model_config.hf_config
        if (cfg.hidden_size, cfg.vocab_size, cfg.eos_token_id) != (512, 8192, 3):
            raise ValueError("Explicit Soprano pipeline requires the Soprano Qwen3 checkpoint")
        if vllm_config.model_config.async_chunk and not vllm_config.model_config.enforce_eager:
            raise ValueError("Soprano streaming requires eager execution and owned CPU hidden-state handoff")
        if vllm_config.model_config.async_chunk and vllm_config.speculative_config is not None:
            raise ValueError("Soprano streaming does not support speculative multi-token steps")
        if vllm_config.model_config.dtype != torch.float32:
            raise ValueError("Soprano's validated offline AR baseline requires dtype=float32; BF16 is not validated")
        if os.environ.get("TRITON_F32_DEFAULT") != "ieee":
            raise ValueError(
                "Soprano's validated CUDA FP32 profiles (soprano.yaml / soprano_streaming.yaml) require "
                "TRITON_F32_DEFAULT=ieee before starting vllm serve or the offline Omni process. "
                "Export it in the launcher environment so worker processes inherit full-precision attention."
            )
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **omni_metadata: object,
    ) -> torch.Tensor | IntermediateTensors:
        # Omni's request metadata is consumed by the runner/processor, not the
        # Qwen3 layers. Do not pass it through to the native strict signature.
        return super().forward(input_ids, positions, intermediate_tensors, inputs_embeds)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        audit = []

        def observed_weights():
            parameters = dict(self.named_parameters())
            for name, tensor in weights:
                destination = name
                for packed, sources in self.packed_modules_mapping.items():
                    for source in sources:
                        destination = destination.replace(f".{source}.", f".{packed}.")
                if destination not in parameters:
                    raise ValueError(f"Unaccounted Soprano AR checkpoint key: {name}")
                audit.append({"key": name, "shape": list(tensor.shape), "destination": destination})
                yield name, tensor

        loaded = super().load_weights(observed_weights())
        self.checkpoint_key_audit = audit
        for row in audit:
            if row["destination"] not in loaded:
                raise ValueError(f"Soprano key not loaded: {row}")
            logger.info("Soprano AR key %s -> %s shape=%s", row["key"], row["destination"], row["shape"])
        return loaded
