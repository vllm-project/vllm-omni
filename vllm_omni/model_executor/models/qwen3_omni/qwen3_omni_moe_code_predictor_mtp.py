"""Qwen3-Omni Code Predictor -- thin wrapper over CodePredictorWrapper."""

from vllm.config import VllmConfig

from vllm_omni.model_executor.models.common.qwen3_code_predictor import (
    CodePredictorWrapper,
    CodePredictorWrapperConfig,
)


class Qwen3OmniMoeTalkerCodePredictor(CodePredictorWrapper):
    """Qwen3-Omni code predictor (no CUDA graphs, VocabParallelEmbedding)."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        cp_config = vllm_config.model_config.hf_config.code_predictor_config
        super().__init__(
            vllm_config=vllm_config,
            cp_config=cp_config,
            wrapper_config=CodePredictorWrapperConfig(
                use_cuda_graphs=False,
                use_parallel_embedding=True,
                use_projection=False,
                return_proj_buf=True,
                sampling_mode="stored",
            ),
            talker_hidden_size=cp_config.hidden_size,
            prefix=prefix,
        )
