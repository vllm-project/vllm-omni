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
        # TTS owns its frame-local cache and overrides _predict_step_logits.
        # Do not also activate the shared wrapper's Omni incremental loop.
        self._kv_cache_enabled = False
        # Store talker_config for backward compat (accessed by some callers)
        self.talker_config = talker_config
        self._vllm_config = vllm_config
        extra = self._stage_connector_extra_config(vllm_config)
        self._kv_requested = self._parse_bool_config(extra.get("code_predictor_kv_cache"))
        self._fused_sampling = self._parse_bool_config(extra.get("code_predictor_fused_sampling"))
        self._frame_graph_requested = self._parse_bool_config(extra.get("code_predictor_frame_graph"))
        self._frame_graphs: dict = {}
        self._frame_graph_top_k = int(self._subtalker_top_k(vllm_config))
        self._frame_graph_temperature = float(self._subtalker_temperature(vllm_config))
        self._frame_cache: FrameLocalKVCache | None = None
        self._fused_requested = self._parse_bool_config(extra.get("code_predictor_fused"))
        self._fused = None

    @staticmethod
    def _subtalker_top_k(vllm_config: VllmConfig) -> int:
        params = getattr(vllm_config.model_config, "subtalker_sampling_params", None)
        if isinstance(params, dict) and params.get("top_k") is not None:
            return int(params["top_k"])
        return 50

    @staticmethod
    def _subtalker_temperature(vllm_config: VllmConfig) -> float:
        params = getattr(vllm_config.model_config, "subtalker_sampling_params", None)
        if isinstance(params, dict) and params.get("temperature") is not None:
            return float(params["temperature"])
        return 0.9

    def _capture_frame_graphs(self) -> None:
        if not self._frame_graph_requested or not current_omni_platform.is_npu():
            return
        if self._compiled_model_fwd is None or not self._bucket_sizes:
            return
        from vllm_omni.platforms.npu.layers.predictor_frame_graph import capture_predictor_frame_graphs

        self._frame_graphs = capture_predictor_frame_graphs(
            self, self._frame_graph_top_k, self._frame_graph_temperature
        )

    def _run_frame_graph(
        self,
        layer0_code: torch.Tensor,
        layer0_embed: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        *,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        sample_uniforms: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if not self._frame_graph_requested or sample_uniforms is None:
            return None
        if not do_sample or temperature <= 0 or top_p != 1.0:
            return None
        if int(top_k) != self._frame_graph_top_k or float(temperature) != self._frame_graph_temperature:
            return None
        if current_omni_platform.is_npu() and torch.npu.is_current_stream_capturing():
            return None
        self._setup_compile()
        bsz = int(layer0_code.shape[0])
        padded = self._padded_bsz(bsz)
        graph = self._frame_graphs.get(padded)
        if graph is None or sample_uniforms.dtype != torch.float32 or not sample_uniforms.is_contiguous():
            return None
        dtype = self._model_dtype
        device = layer0_code.device
        self._ensure_buffers(device, dtype, padded)
        proj = self._proj_buf
        proj[:padded].zero_()
        projection = self.small_to_mtp_projection
        proj[:bsz, 0, :] = projection(last_talker_hidden.reshape(bsz, 1, -1).to(dtype)).reshape(bsz, -1)
        proj[:bsz, 1, :] = projection(layer0_embed.reshape(bsz, 1, -1).to(dtype)).reshape(bsz, -1)
        graph.codes[:, 0] = 0
        graph.codes[:bsz, 0] = layer0_code.reshape(bsz).to(dtype=torch.long)
        if padded != bsz:
            graph.uniforms[bsz:].fill_(0.5)
        return graph.run(sample_uniforms)[:bsz].clone()

    def _setup_compile(self) -> None:
        if self._compiled_model_fwd is not None:
            return
        weight = next(self.model.parameters())
        if self._kv_requested and current_omni_platform.is_npu() and weight.dtype == torch.bfloat16:
            # The incremental body is eager here: this branch returns before
            # NPU graph capture, so each codec frame runs 15 uncaptured
            # forwards. Graphed re-prefill is one replay per codebook step and
            # was about 3x faster on a 1.7B CustomVoice serving run. Keep it.
            logger.info("Qwen3-TTS NPU code predictor keeps graphed re-prefill instead of eager incremental KV")
            self._kv_requested = False
        supported = (
            current_omni_platform.is_cuda()
            and weight.device.type == "cuda"
            and weight.dtype == torch.bfloat16
            and current_omni_platform.supports_torch_inductor()
        )
        if not self._kv_requested or not supported:
            if self._kv_requested:
                logger.warning("Qwen3-TTS frame-local KV requires CUDA BF16; using re-prefill")
            super()._setup_compile()
            self._capture_frame_graphs()
            return

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
        if self._fused_requested:
            from .fused_code_predictor import FusedCodePredictor

            self._fused = FusedCodePredictor(self, max_batch)
            # Compile every kernel variant outside any graph capture.
            hidden = int(self.talker_config.hidden_size)
            for batch in self._fused.warmup_batches():
                zeros = torch.zeros(batch, 1, hidden, device=weight.device, dtype=weight.dtype)
                uniforms = torch.full(
                    (batch, self._num_groups - 1, int(self.config.vocab_size)), 0.5, device=weight.device
                )
                codes = torch.zeros(batch, 1, device=weight.device, dtype=torch.long)
                self._fused(codes, zeros, zeros, 1.0, 50, uniforms)
            self._synchronize_warmup(weight.device)
            logger.info("Qwen3-TTS fused code predictor enabled (capacity %d)", max_batch)
        # Preserve the shared wrapper's initialization sentinel. Its full
        # re-prefill callable is unused while _frame_cache is active.
        self._compiled_model_fwd = self.model.forward
        logger.info("Qwen3-TTS frame-local KV warmed for batches %s", self._bucket_sizes)

    @torch.inference_mode()
    def forward(
        self,
        layer0_code: torch.Tensor,
        layer0_embed: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        do_sample: bool = True,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        generator: torch.Generator | None = None,
        generators: Sequence[torch.Generator | None] | None = None,
        sample_uniforms: torch.Tensor | None = None,
    ):
        self._validate_sampling_inputs(int(layer0_code.shape[0]), generators, sample_uniforms)
        framed = self._run_frame_graph(
            layer0_code,
            layer0_embed,
            last_talker_hidden,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            sample_uniforms=sample_uniforms,
        )
        if framed is not None:
            return framed
        if self._fused_requested:
            self._setup_compile()
        if (
            self._fused is not None
            and do_sample
            and temperature > 0
            and top_p == 1.0
            and sample_uniforms is not None
            and sample_uniforms.dtype == torch.float32
            and sample_uniforms.is_contiguous()
        ):
            return self._fused(
                layer0_code,
                layer0_embed,
                last_talker_hidden,
                1.0 / temperature,
                int(top_k),
                sample_uniforms,
            )
        return super().forward(
            layer0_code,
            layer0_embed,
            last_talker_hidden,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
            generators=generators,
            sample_uniforms=sample_uniforms,
        )

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
            and current_omni_platform.is_npu()
            and logits.device.type == "npu"
            and uniforms is not None
            and uniforms.dtype == torch.float32
            and logits.dtype in (torch.float32, torch.bfloat16)
            and logits.ndim == 2
            and logits.shape[0] > 0
            and 0 < logits.shape[1] <= 4096
            and 0 <= top_k <= logits.shape[1]
        ):
            from vllm_omni.platforms.npu.layers.code_predictor_sampling import sample_topk_gumbel

            return sample_topk_gumbel(logits, uniforms, top_k, inv_temperature)
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
