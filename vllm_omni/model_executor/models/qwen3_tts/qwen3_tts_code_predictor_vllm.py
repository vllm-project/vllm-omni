# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Qwen3-TTS predictor: per-call sampling and opt-in CUDA frame-local KV reuse."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import torch
from vllm.config import VllmConfig
from vllm.config.vllm import set_current_vllm_config
from vllm.logger import init_logger

from vllm_omni.model_executor.models.common.qwen3_code_predictor import (
    CodePredictorBaseModel,
    CodePredictorWrapper,
    CodePredictorWrapperConfig,
)
from vllm_omni.platforms import current_omni_platform

from .configuration_qwen3_tts import Qwen3TTSTalkerCodePredictorConfig, Qwen3TTSTalkerConfig

if TYPE_CHECKING:
    from .cached_code_predictor import FrameLocalKVCache

logger = init_logger(__name__)

# Backward-compat alias used by tests
Qwen3TTSTalkerCodePredictorModelVLLM = CodePredictorBaseModel


class Qwen3TTSTalkerCodePredictorForConditionalGenerationVLLM(CodePredictorWrapper):
    """Qwen3-TTS code predictor (per-call sampling, projection)."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen3TTSTalkerCodePredictorConfig,
        talker_config: Qwen3TTSTalkerConfig,
        prefix: str = "code_predictor",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            cp_config=config,
            wrapper_config=CodePredictorWrapperConfig(
                use_cuda_graphs=current_omni_platform.is_npu(),
                use_parallel_embedding=False,
                use_projection=(config.hidden_size != talker_config.hidden_size),
                return_proj_buf=False,
                sampling_mode="per_call",
            ),
            talker_hidden_size=int(talker_config.hidden_size),
            prefix=prefix,
        )
        # Store talker_config for backward compat (accessed by some callers)
        self.talker_config = talker_config
        self._vllm_config = vllm_config
        extra = self._stage_connector_extra_config(vllm_config)
        self._kv_requested = self._parse_bool_config(extra.get("code_predictor_kv_cache"))
        self._fused_sampling = self._parse_bool_config(extra.get("code_predictor_fused_sampling"))
        self._frame_cache: FrameLocalKVCache | None = None

    def _setup_compile(self) -> None:
        if self._compiled_model_fwd is not None:
            return
        weight = next(self.model.parameters())
        supported = (
            current_omni_platform.is_cuda()
            and weight.device.type == "cuda"
            and weight.dtype == torch.bfloat16
            and current_omni_platform.supports_torch_inductor()
        )
        if not self._kv_requested or not supported:
            if self._kv_requested:
                logger.warning("Qwen3-TTS frame-local KV requires CUDA BF16; using re-prefill")
            return super()._setup_compile()

        # Import CUDA-only kernels only when this model explicitly opts in.
        from .cached_code_predictor import FrameLocalKVCache

        self._model_dtype = weight.dtype
        self._lm_heads_list = list(self.lm_head)
        self._codec_embeds_list = list(self.model.codec_embedding)
        self._bucket_sizes = self._batch_bucket_sizes()
        max_batch = max(self._bucket_sizes)
        self._ensure_buffers(weight.device, weight.dtype, max_batch)
        cache = FrameLocalKVCache(self.model, max_batch)
        with torch._dynamo.config.patch(
            cache_size_limit=max(torch._dynamo.config.cache_size_limit, 2 * len(self._bucket_sizes))
        ):
            for batch in self._bucket_sizes:
                for step in range(1, min(3, self._num_groups)):
                    for _ in range(2):
                        cache(self._proj_buf, batch, step)
        self._synchronize_warmup(weight.device)
        self._frame_cache = cache
        # Preserve the shared wrapper's initialization sentinel. Its full
        # re-prefill callable is unused while _frame_cache is active.
        self._compiled_model_fwd = self.model.forward
        logger.info("Qwen3-TTS frame-local KV warmed for batches %s", self._bucket_sizes)

    def _predict_step_logits(
        self,
        proj_buf: torch.Tensor,
        bsz: int,
        padded_bsz: int,
        step: int,
        is_npu_capturing: bool,
    ) -> torch.Tensor:
        if self._frame_cache is None:
            return super()._predict_step_logits(proj_buf, bsz, padded_bsz, step, is_npu_capturing)
        hidden = self._frame_cache(proj_buf, padded_bsz, step)
        return self._lm_heads_list[step - 1](hidden[:bsz, -1])

    def _sample_per_call(
        self,
        logits: torch.Tensor,
        inv_temperature: float,
        top_k: int,
        generator: torch.Generator | Sequence[torch.Generator | None] | None,
        uniforms: torch.Tensor | None,
    ) -> torch.Tensor:
        if (
            self._fused_sampling
            and current_omni_platform.is_cuda()
            and logits.is_cuda
            and uniforms is not None
            and uniforms.dtype == torch.float32
        ):
            from .code_predictor_sampling import sample_code_topk_gumbel

            return sample_code_topk_gumbel(logits, uniforms, top_k, inv_temperature)
        return super()._sample_per_call(logits, inv_temperature, top_k, generator, uniforms)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with vllm config context (required for VocabParallelEmbedding)."""
        with set_current_vllm_config(self._vllm_config):
            return super().load_weights(weights)
