from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_omni.model_executor.models.omni_diffusion.utils import (
    OMNI_DIFFUSION_END_OF_TEXT_TOKEN,
    OMNI_DIFFUSION_TEXT_ADAPTER_TOKEN_IDS_KEY,
    normalize_token_id_sequence,
)


def _runtime_info_list(runtime_additional_information: Any) -> list[dict[str, Any]]:
    if isinstance(runtime_additional_information, list):
        return [info if isinstance(info, dict) else {} for info in runtime_additional_information]
    if isinstance(runtime_additional_information, dict):
        return [runtime_additional_information]
    return [{}]


def _has_target_token_ids(runtime_infos: list[dict[str, Any]]) -> bool:
    return any(
        normalize_token_id_sequence(
            info.get(OMNI_DIFFUSION_TEXT_ADAPTER_TOKEN_IDS_KEY),
            source="Omni-Diffusion text adapter target",
        )
        for info in runtime_infos
    )


class OmniDiffusionTextAdapterForConditionalGeneration(nn.Module):
    """Emit Omni-Diffusion one-shot text tokens through the AR text path.

    Omni-Diffusion's official ``DreamModel.generate`` returns the whole text
    answer in one call. The vLLM OpenAI text path, however, consumes sampled
    AR tokens from a final LLM stage. This adapter bridges those contracts:
    stage 0 passes target text token IDs via per-request runtime information,
    and this stage forces the normal sampler to emit the target sequence one
    token per decode step.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        del prefix
        self.vllm_config = vllm_config
        model_config = vllm_config.model_config
        hf_config = model_config.hf_config
        self.dtype = model_config.dtype
        self.hidden_size = int(getattr(hf_config, "hidden_size", 1) or 1)
        self.model_path = model_config.model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=model_config.trust_remote_code,
        )
        self.vocab_size = max(
            int(getattr(hf_config, "vocab_size", 0) or 0),
            len(self.tokenizer),
        )
        self.eos_token_id = self._resolve_eos_token_id()
        self._next_token_ids: list[int] = [self.eos_token_id]

    def _resolve_eos_token_id(self) -> int:
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is not None:
            return int(eos_token_id)
        token_id = self.tokenizer.convert_tokens_to_ids(OMNI_DIFFUSION_END_OF_TEXT_TOKEN)
        if token_id is None or int(token_id) < 0:
            raise ValueError("Omni-Diffusion text adapter could not resolve an EOS token.")
        return int(token_id)

    def _next_token_id_for_request(self, runtime_info: dict[str, Any]) -> int:
        token_ids = normalize_token_id_sequence(
            runtime_info.get(OMNI_DIFFUSION_TEXT_ADAPTER_TOKEN_IDS_KEY),
            source="Omni-Diffusion text adapter target",
        )
        if not token_ids:
            return self.eos_token_id

        # The runner injects generated_len per request before every AR step,
        # so no model-global text cache is needed for concurrent requests.
        generated_len = int(runtime_info.get("generated_len", 0) or 0)
        if generated_len < len(token_ids):
            return int(token_ids[generated_len])
        return self.eos_token_id

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del positions, intermediate_tensors, inputs_embeds
        if input_ids is None:
            raise ValueError("Omni-Diffusion text adapter requires input_ids.")

        runtime_infos = _runtime_info_list(kwargs.get("runtime_additional_information"))
        self._next_token_ids = [self._next_token_id_for_request(info) for info in runtime_infos]
        if not _has_target_token_ids(runtime_infos):
            # During profiling/warmup this adapter has no upstream stage-0
            # text token payload yet. Returning CPU hidden states lets the
            # Omni runner skip the default sampler warmup, avoiding FlashInfer
            # sampling kernels on machines that cannot run them.
            return torch.zeros(
                (input_ids.numel(), self.hidden_size),
                dtype=self.dtype,
                device="cpu",
            )

        return torch.zeros(
            (input_ids.numel(), self.hidden_size),
            dtype=self.dtype,
            device=input_ids.device,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return placeholder embeddings for vLLM's generation runner.

        The adapter does not use real token embeddings. It drives decoding
        from per-request target token IDs in ``runtime_additional_information``
        and forces the next token through ``compute_logits``. vLLM still
        requires generative model wrappers to expose ``embed_input_ids``.
        """
        return torch.zeros(
            (*input_ids.shape, self.hidden_size),
            dtype=self.dtype,
            device=input_ids.device,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata | None = None,
    ) -> torch.Tensor:
        del sampling_metadata
        row_count = int(hidden_states.shape[0])
        next_token_ids = self._next_token_ids
        if len(next_token_ids) != row_count:
            raise ValueError(
                "Expected one Omni-Diffusion next token ID per logits row, "
                f"got token_ids={len(next_token_ids)}, rows={row_count}."
            )

        invalid_token_ids = [token_id for token_id in next_token_ids if not 0 <= token_id < self.vocab_size]
        if invalid_token_ids:
            raise ValueError(
                "Omni-Diffusion text adapter received token IDs outside its vocabulary: "
                f"token_ids={invalid_token_ids}, vocab_size={self.vocab_size}."
            )

        logits = torch.full(
            (row_count, self.vocab_size),
            -1.0e4,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        for row, token_id in enumerate(next_token_ids):
            logits[row, int(token_id)] = 0.0
        return logits

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        del batch_size, dtype, device
        return IntermediateTensors({})

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # This adapter has no checkpoint parameters; do not iterate over the
        # Dream checkpoint just to discard it.
        del weights
        return set()

    def get_dummy_runtime_additional_information(self, num_reqs: int) -> list[dict[str, Sequence[int]]]:
        return [{OMNI_DIFFUSION_TEXT_ADAPTER_TOKEN_IDS_KEY: [self.eos_token_id]} for _ in range(num_reqs)]
