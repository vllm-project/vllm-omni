# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
"""MiniMind-O configuration classes for vLLM-Omni integration."""

from transformers import PretrainedConfig

from vllm_omni.model_executor.models.minimind_o.minimind_llm import MiniMindConfig as MiniMindTextConfig


def _build_text_config(
    *,
    vocab_size: int,
    hidden_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    intermediate_size: int,
    max_position_embeddings: int,
    rms_norm_eps: float,
    rope_theta: float,
    hidden_act: str,
    tie_word_embeddings: bool = True,
    use_moe: bool = False,
    num_experts: int = 4,
    num_experts_per_tok: int = 1,
    moe_intermediate_size: int = 2432,
    norm_topk_prob: bool = True,
) -> MiniMindTextConfig:
    return MiniMindTextConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
        rms_norm_eps=rms_norm_eps,
        rope_theta=rope_theta,
        hidden_act=hidden_act,
        tie_word_embeddings=tie_word_embeddings,
        use_moe=use_moe,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_intermediate_size=moe_intermediate_size,
        norm_topk_prob=norm_topk_prob,
    )


class MiniMindOThinkerConfig(PretrainedConfig):
    model_type = "minimind_o_thinker"

    def __init__(
        self,
        *,
        vocab_size: int = 6400,
        hidden_size: int = 768,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 4,
        head_dim: int = 96,
        intermediate_size: int = 2432,
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1e6,
        rope_scaling=None,
        hidden_act: str = "silu",
        audio_hidden_size: int = 512,
        audio_ids: list[int] | None = None,
        image_hidden_size: int = 768,
        image_token_len: int = 64,
        image_ids: list[int] | None = None,
        bridge_layer: int = 3,
        audio_token_index: int | None = None,
        image_token_index: int | None = None,
        video_token_index: int = 13,
        use_moe: bool = False,
        num_experts: int = 4,
        num_experts_per_tok: int = 1,
        moe_intermediate_size: int = 2432,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        audio_ids = audio_ids or [16]
        image_ids = image_ids or [12]
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.hidden_act = hidden_act
        self.audio_hidden_size = audio_hidden_size
        self.audio_ids = audio_ids
        self.image_hidden_size = image_hidden_size
        self.image_token_len = image_token_len
        self.image_ids = image_ids
        self.bridge_layer = bridge_layer
        self.audio_token_index = audio_token_index if audio_token_index is not None else audio_ids[0]
        self.image_token_index = image_token_index if image_token_index is not None else image_ids[0]
        self.video_token_index = video_token_index
        self.audio_start_token_id = None
        self.audio_end_token_id = None
        self.vision_start_token_id = None
        self.vision_end_token_id = None
        self.seconds_per_chunk = 1.0
        self.spatial_merge_size = 2
        self.vision_config = type(
            "VisionCfg",
            (),
            {"spatial_merge_size": 2, "tokens_per_second": 25},
        )()
        self.text_config = _build_text_config(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            hidden_act=hidden_act,
            tie_word_embeddings=True,
            use_moe=use_moe,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            norm_topk_prob=norm_topk_prob,
        )


class MiniMindOTalkerConfig(PretrainedConfig):
    model_type = "minimind_o_talker"

    def __init__(
        self,
        *,
        vocab_size: int = 6400,
        hidden_size: int = 768,
        talker_hidden_size: int = 768,
        num_talker_hidden_layers: int = 4,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 4,
        head_dim: int = 96,
        intermediate_size: int = 2432,
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1e6,
        hidden_act: str = "silu",
        audio_vocab_size: int = 2112,
        audio_pad_token: int = 2049,
        audio_stop_token: int = 2050,
        audio_spk_token: int = 2051,
        spk_emb_size: int = 192,
        mtp_num_layers: int = 8,
        mtp_rank: int = 256,
        use_moe: bool = False,
        num_experts: int = 4,
        num_experts_per_tok: int = 1,
        moe_intermediate_size: int = 2432,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.talker_hidden_size = talker_hidden_size
        self.num_talker_hidden_layers = num_talker_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.hidden_act = hidden_act
        self.audio_vocab_size = audio_vocab_size
        self.audio_pad_token = audio_pad_token
        self.audio_stop_token = audio_stop_token
        self.audio_spk_token = audio_spk_token
        self.spk_emb_size = spk_emb_size
        self.mtp_num_layers = mtp_num_layers
        self.mtp_rank = mtp_rank
        self.tts_codec_start_token_id = audio_pad_token
        self.tts_codec_end_token_id = audio_stop_token
        self.tts_codec_pad_token_id = audio_pad_token
        self.tts_codec_mask_token_id = audio_pad_token
        self.tts_text_start_token_id = 1
        self.tts_text_end_token_id = 2
        self.tts_text_pad_token_id = 0
        self.text_config = _build_text_config(
            vocab_size=audio_vocab_size,
            hidden_size=talker_hidden_size,
            num_hidden_layers=num_talker_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            hidden_act=hidden_act,
            tie_word_embeddings=False,
            use_moe=use_moe,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            norm_topk_prob=norm_topk_prob,
        )


class MiniMindOCode2WavConfig(PretrainedConfig):
    model_type = "minimind_o_code2wav"

    def __init__(
        self,
        *,
        codebook_size: int = 2112,
        num_quantizers: int = 8,
        hidden_size: int = 512,
        frame_rate: float = 12.5,
        sample_rate: int = 24000,
        num_decoder_layers: int = 8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers
        self.hidden_size = hidden_size
        self.frame_rate = frame_rate
        self.sample_rate = sample_rate
        self.num_decoder_layers = num_decoder_layers


class MiniMindOConfig(PretrainedConfig):
    """Top-level HF config for jingyaogong/minimind-3o (flat checkpoint)."""

    model_type = "minimind-o"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        architectures=None,
        model_type="minimind-o",
        vocab_size=6400,
        hidden_size=768,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=96,
        intermediate_size=2432,
        max_position_embeddings=32768,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        rope_scaling=None,
        hidden_act="silu",
        dropout=0.0,
        dtype="bfloat16",
        flash_attn=True,
        inference_rope_scaling=False,
        use_moe=False,
        num_experts=4,
        num_experts_per_tok=1,
        moe_intermediate_size=2432,
        router_aux_loss_coef=0.0005,
        norm_topk_prob=True,
        audio_hidden_size=512,
        audio_vocab_size=2112,
        audio_pad_token=2049,
        audio_stop_token=2050,
        audio_spk_token=2051,
        audio_ids=None,
        audio_special_token="<|audio_pad|>",
        image_hidden_size=768,
        image_token_len=64,
        image_ids=None,
        image_special_token="<|image_pad|>",
        num_talker_hidden_layers=4,
        talker_hidden_size=768,
        spk_emb_size=192,
        bridge_layer=3,
        think_end_ids=None,
        bos_token_id=1,
        eos_token_id=2,
        auto_map=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        audio_ids = audio_ids if audio_ids is not None else [16]
        image_ids = image_ids if image_ids is not None else [12]
        think_end_ids = think_end_ids if think_end_ids is not None else [26, 234, 234]

        default_arch = "MiniMindOMoeForConditionalGeneration" if use_moe else "MiniMindOForConditionalGeneration"
        self.architectures = architectures or [default_arch]
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.hidden_act = hidden_act
        self.dropout = dropout
        self.dtype = dtype
        self.flash_attn = flash_attn
        self.inference_rope_scaling = inference_rope_scaling
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.router_aux_loss_coef = router_aux_loss_coef
        self.norm_topk_prob = norm_topk_prob
        self.audio_hidden_size = audio_hidden_size
        self.audio_vocab_size = audio_vocab_size
        self.audio_pad_token = audio_pad_token
        self.audio_stop_token = audio_stop_token
        self.audio_spk_token = audio_spk_token
        self.audio_ids = audio_ids
        self.audio_special_token = audio_special_token
        self.image_hidden_size = image_hidden_size
        self.image_token_len = image_token_len
        self.image_ids = image_ids
        self.image_special_token = image_special_token
        self.num_talker_hidden_layers = num_talker_hidden_layers
        self.talker_hidden_size = talker_hidden_size
        self.spk_emb_size = spk_emb_size
        self.bridge_layer = bridge_layer
        self.think_end_ids = think_end_ids
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.auto_map = auto_map

        self.thinker_config = MiniMindOThinkerConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            hidden_act=hidden_act,
            audio_hidden_size=audio_hidden_size,
            audio_ids=audio_ids,
            image_hidden_size=image_hidden_size,
            image_token_len=image_token_len,
            image_ids=image_ids,
            bridge_layer=bridge_layer,
            use_moe=use_moe,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            norm_topk_prob=norm_topk_prob,
        )
        self.talker_config = MiniMindOTalkerConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            talker_hidden_size=talker_hidden_size,
            num_talker_hidden_layers=num_talker_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            hidden_act=hidden_act,
            audio_vocab_size=audio_vocab_size,
            audio_pad_token=audio_pad_token,
            audio_stop_token=audio_stop_token,
            audio_spk_token=audio_spk_token,
            spk_emb_size=spk_emb_size,
            use_moe=use_moe,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            norm_topk_prob=norm_topk_prob,
        )
        self.code2wav_config = MiniMindOCode2WavConfig(
            codebook_size=audio_vocab_size,
            hidden_size=audio_hidden_size,
        )


# HF AutoConfig alias (model_omni.OmniConfig)
MiniMindOmniConfig = MiniMindOConfig
