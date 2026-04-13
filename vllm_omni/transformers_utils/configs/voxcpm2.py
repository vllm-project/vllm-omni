# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation


class VoxCPM2Config(PretrainedConfig):
    """Configuration for VoxCPM2 native AR integration.

    The HuggingFace checkpoint stores LM parameters inside a nested
    ``lm_config`` dict.  This class hoists them to top-level attributes
    so that vllm's ``MiniCPMModel`` can consume them directly.

    vllm's MiniCPM **always** applies muP scaling (scale_emb, scale_depth,
    dim_model_base).  VoxCPM2 was trained with ``use_mup=false``, so we
    neutralise the scalings:
      * ``scale_emb = 1.0``
      * ``scale_depth = sqrt(num_hidden_layers)``  (cancels the division)
      * ``dim_model_base = hidden_size``  (makes scale_width = 1.0)
    """

    model_type = "voxcpm2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        # -- top-level VoxCPM2 params --
        architecture: str = "voxcpm2",
        lm_config: dict | None = None,
        encoder_config: dict | None = None,
        dit_config: dict | None = None,
        audio_vae_config: dict | None = None,
        patch_size: int = 4,
        feat_dim: int = 64,
        residual_lm_num_layers: int = 8,
        residual_lm_no_rope: bool = True,
        scalar_quantization_latent_dim: int = 512,
        scalar_quantization_scale: int = 9,
        max_length: int = 8192,
        device: str = "cuda",
        dtype: str = "bfloat16",
        # -- LM defaults (overridden by lm_config if present) --
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        vocab_size: int = 73448,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 16,
        num_hidden_layers: int = 28,
        num_key_value_heads: int = 2,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.architecture = architecture

        # -- VoxCPM2-specific fields --
        self.lm_config = lm_config or {}
        self.encoder_config = encoder_config or {}
        self.dit_config = dit_config or {}
        self.audio_vae_config = audio_vae_config or {}
        self.patch_size = patch_size
        self.feat_dim = feat_dim
        self.residual_lm_num_layers = residual_lm_num_layers
        self.residual_lm_no_rope = residual_lm_no_rope
        self.scalar_quantization_latent_dim = scalar_quantization_latent_dim
        self.scalar_quantization_scale = scalar_quantization_scale
        self.max_length = max_length
        self.device = device
        self.dtype = dtype

        # -- Hoist LM parameters to top-level for MiniCPMModel --
        lm = self.lm_config
        self.vocab_size = lm.get("vocab_size", vocab_size)
        self.hidden_size = lm.get("hidden_size", hidden_size)
        self.intermediate_size = lm.get("intermediate_size", intermediate_size)
        self.max_position_embeddings = lm.get("max_position_embeddings", max_position_embeddings)
        self.num_attention_heads = lm.get("num_attention_heads", num_attention_heads)
        self.num_hidden_layers = lm.get("num_hidden_layers", num_hidden_layers)
        self.num_key_value_heads = lm.get("num_key_value_heads", num_key_value_heads)
        self.rms_norm_eps = lm.get("rms_norm_eps", rms_norm_eps)
        self.rope_theta = lm.get("rope_theta", rope_theta)

        # MiniCPM-specific: kv_channels overrides head_dim when set.
        kv_channels = lm.get("kv_channels")
        if kv_channels is not None:
            self.head_dim = kv_channels
        else:
            self.head_dim = self.hidden_size // self.num_attention_heads

        # MiniCPM requires hidden_act; VoxCPM2 uses SiLU.
        self.hidden_act = "silu"
        self.hidden_act_param = 0.0
        self.tie_word_embeddings = False
        self.num_experts = 0

        # -- muP scaling --
        # Native VoxCPM2 MiniCPM gates scale_depth behind use_mup:
        #   use_mup=True  → residual += h * (scale_depth / sqrt(N))
        #   use_mup=False → residual += h  (plain add, no scaling)
        # But vllm's MiniCPMModel ALWAYS applies scale_depth / sqrt(N).
        # Native applies scale_emb externally; vllm applies it in embed_input_ids.
        use_mup = lm.get("use_mup", False)
        self.scale_emb = lm.get("scale_emb", 1.0)
        if use_mup:
            self.scale_depth = lm.get("scale_depth", 1.0)
            self.dim_model_base = lm.get("dim_model_base", self.hidden_size)
        else:
            # Neutralize: scale_depth/sqrt(N) = 1.0, scale_width = 1.0
            self.scale_depth = math.sqrt(self.num_hidden_layers)
            self.dim_model_base = self.hidden_size

        # -- RoPE scaling (longrope) --
        raw_rope = lm.get("rope_scaling", rope_scaling)
        if raw_rope is not None:
            self.rope_scaling = dict(raw_rope)
            # HF expects "rope_type" not "type"
            if "type" in self.rope_scaling:
                self.rope_scaling["rope_type"] = self.rope_scaling.pop("type")
            # longrope requires "factor" (used by HF validation)
            if "factor" not in self.rope_scaling:
                self.rope_scaling["factor"] = 1.0
            rope_config_validation(self)

            # vllm's MiniCPMAttention reads config.rope_parameters (a dict
            # with rope_type, theta, scaling factors, etc.).  HF transformers
            # only auto-computes this for known model_types; for custom
            # types we must build it manually.
            if not getattr(self, "rope_parameters", None):
                rp = dict(self.rope_scaling)
                rp["rope_theta"] = self.rope_theta
                self.rope_parameters = rp
        else:
            self.rope_scaling = None

    def get_text_config(self, **kwargs):
        """Return self as the text config — LM attributes are top-level."""
        return self


AutoConfig.register("voxcpm2", VoxCPM2Config)

__all__ = ["VoxCPM2Config"]
