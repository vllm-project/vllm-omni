# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Standalone talker configuration, independent of thinker/vision imports."""

from typing import Any

from transformers import AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedTokenizerFast, Qwen2Config


class MingFlashOmniTalkerConfig(PretrainedConfig):
    """Configuration class for Ming-flash-omni-2.0 talker (TTS) stage.

    The talker uses a Qwen2 LLM backbone with CFM (Conditional Flow Matching)
    via a DiT diffusion transformer, plus an Aggregator that maps generated
    audio latents back to the LLM embedding space for autoregressive generation.
    """

    model_type = "ming_flash_omni_talker"

    def __init__(
        self,
        llm_config: dict[str, Any] | None = None,
        flowmodel: dict[str, Any] | None = None,
        aggregator: dict[str, Any] | None = None,
        steps: int = 10,
        patch_size: int = 4,
        history_patch_size: int = 32,
        latent_dim: int = 64,
        cfg_strength: float = 2.0,
        audio_vae_path: str | None = None,
        campplus_model: str | None = None,
        max_decode_steps: int = 200,
        max_cache_len: int = 2048,
        enable_llm_cuda_graph: bool = False,
        compile_flow: bool = False,
        low_latency_streaming: bool = False,
        strict_weight_loading: bool = False,
        normalize_text: bool = True,
        scrub_markup: bool = False,
        tn_mode: str = "legacy",
        duration_limit: bool = True,
        stream_decode: bool = True,
        trim_trailing_silence: bool | None = None,
        **kwargs,
    ):
        self.llm_config = llm_config
        self.flowmodel = flowmodel or {}
        self.aggregator = aggregator or {}
        self.steps = steps
        self.patch_size = patch_size
        self.history_patch_size = history_patch_size
        self.latent_dim = latent_dim
        self.cfg_strength = cfg_strength
        self.audio_vae_path = audio_vae_path
        self.campplus_model = campplus_model
        self.max_decode_steps = max_decode_steps
        self.max_cache_len = max_cache_len
        self.enable_llm_cuda_graph = enable_llm_cuda_graph
        # Batched Talker only: torch.compile the DiT forward before capture.
        self.compile_flow = compile_flow
        self.low_latency_streaming = low_latency_streaming
        self.strict_weight_loading = strict_weight_loading
        self.normalize_text = normalize_text
        self.scrub_markup = scrub_markup
        self.tn_mode = tn_mode
        self.duration_limit = duration_limit
        self.stream_decode = stream_decode
        self.trim_trailing_silence = trim_trailing_silence
        if tn_mode not in {"legacy", "upstream"}:
            raise ValueError("tn_mode must be 'legacy' or 'upstream'")
        if patch_size <= 0 or history_patch_size < patch_size or max_decode_steps <= 0 or max_cache_len <= 0:
            raise ValueError("Invalid talker patch/history sizes or decode/cache limits")
        super().__init__(**kwargs)

    def get_text_config(self, decoder: bool = False) -> PretrainedConfig:  # noqa: ARG002
        # Transformers v5 also calls this during base-config initialization.
        # Return the real nested Qwen config for tokenizer and KV geometry.
        llm_config = getattr(self, "llm_config", None)
        if isinstance(llm_config, dict):
            return Qwen2Config.from_dict(llm_config)

        return llm_config if llm_config is not None else self


AutoConfig.register(MingFlashOmniTalkerConfig.model_type, MingFlashOmniTalkerConfig)
AutoTokenizer.register(MingFlashOmniTalkerConfig, fast_tokenizer_class=PreTrainedTokenizerFast)
