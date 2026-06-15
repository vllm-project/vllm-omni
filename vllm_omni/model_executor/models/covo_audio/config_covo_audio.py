# Copyright 2026 Tencent.
from transformers.configuration_utils import PretrainedConfig

# audio_token_index: token IDs >= this value are audio codes.
# Matches config.audio_token_index and len(tokenizer) = 151671.
# Note: the LLM total vocab_size is 168055 (includes audio tokens), which
# is different from this boundary value.
COVO_AUDIO_TOKEN_INDEX = 151671


class CovoAudioCode2WavConfig(PretrainedConfig):
    model_type = "covo_audio_code2wav"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wav_input_sr = 24000
        self.n_speakers = 1
        self.n_styles = 1

        self.token2latent = {
            "upsample_factor": 4,
            "token_vocab_size": 16384,
            "token_input_dim": 512,
            "z_dim": 64,
            "spkr_embed_dim": 1024,
            "spkr_mask_ratio": 0.0,
            "bert_mask_rate0": 0.7,
            "bert_mask_rate1": 1.0,
            "random_maskrate": 0.3,
            "cfg_dropout": 0.2,
            "sigma": 1e-05,
            "transformer": {
                "num_layers": 12,
                "hidden_size": 1024,
                "ffn_hidden_size": 4096,
                "num_heads": 16,
                "modulation": True,
                "alibi_bias": False,
                "rotary_bias": True,
                "qk_norm": False,
                "max_position_embeddings": 4096,
                "attn_dropout": 0.0,
                "dropout": 0.1,
            },
        }

        self.wavegan = {
            "type": "BigVGANFlowVAE",
            "upsample_rates": [5, 3, 2, 2, 2, 2],
            "upsample_kernel_sizes": [10, 6, 4, 4, 4, 4],
            "upsample_initial_channel": 1536,
            "resblock": "1",
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "downsample_rates": [2, 2, 2, 2, 3, 5],
            "downsample_channels": [12, 24, 48, 96, 192, 384, 768],
            "activation": "snakebeta",
            "snake_logscale": True,
            "latent_dim": 64,
            "use_flow": True,
            "use_vae": True,
            "kl_weight": 5,
            "causal": True,
            "flow_hidden_channels": 256,
        }

        self.inference = {
            "s_steps": 10,
            "cfg_alpha": 1.0,
            "dynamic_the": False,
            "rescale_logits": False,
        }
