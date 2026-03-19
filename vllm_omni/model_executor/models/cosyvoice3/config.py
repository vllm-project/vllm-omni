# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from transformers.configuration_utils import PretrainedConfig


class CosyVoice3Config(PretrainedConfig):
    model_type = "cosyvoice3"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sample_rate = 24000
        self.llm_input_size = 896
        self.llm_output_size = 896
        self.hidden_size = self.llm_output_size
        self.num_attention_heads = 14
        self.num_hidden_layers = 24
        self.spk_embed_dim = 192
        self.token_frame_rate = 25
        self.token_mel_ratio = 2
        self.vocab_size = 151923
        self.min_token_text_ratio = 2
        self.max_token_text_ratio = 20
        self.allowed_special = "all"
        self.skip_special_tokens = True
        self.target_sr = 24000
        self.feat_extractor = {
            "n_fft": 1920,
            "num_mels": 80,
            "sampling_rate": self.sample_rate,
            "hop_size": 480,
            "win_size": 1920,
            "fmin": 0,
            "fmax": None,
            "center": False,
        }
        self.speculative_config = None
        self.qwen_pretrain_path = "CosyVoice-BlankEN"
        self.campplus_onxx_path = "campplus.onnx"
        self.speech_tokenizer_path = "speech_tokenizer_v3.onnx"
        self.spk2info_path = "spk2info.pt"
        self.version = "cosyvoice3"
        self.llm = {
            "llm_input_size": self.llm_input_size,
            "llm_output_size": self.llm_output_size,
            "speech_token_size": 6561,
            "eos_token_id": 6561 + 1,
            "length_normalized_loss": True,
            "lsm_weight": 0,
            "mix_ratio": [5, 15],
            "llm": {
                "pretrain_path": self.qwen_pretrain_path,
            },
            "sampling": {
                "top_p": 0.8,
                "top_k": 25,
                "win_size": 10,
                "tau_r": 0.1,
            },
            "spk_embed_dim": self.spk_embed_dim,
        }
        self.flow = {
            "input_size": 80,
            "output_size": 80,
            "spk_embed_dim": self.spk_embed_dim,
            "output_type": "mel",
            "vocab_size": 6561,
            "input_frame_rate": self.token_frame_rate,
            "only_mask_loss": True,
            "token_mel_ratio": self.token_mel_ratio,
            "pre_lookahead_len": 3,
            "pre_lookahead_layer": {
                "in_channels": 80,
                "channels": 1024,
                "pre_lookahead_len": 3,
            },
            "decoder": {
                "in_channels": 240,
                "n_spks": 1,
                "spk_emb_dim": 80,
                "cfm_params": {
                    "sigma_min": 1e-06,
                    "solver": "euler",
                    "t_scheduler": "cosine",
                    "training_cfg_rate": 0.2,
                    "inference_cfg_rate": 0.7,
                    "reg_loss_type": "l1",
                },
                "estimator": {
                    "dim": 1024,
                    "depth": 22,
                    "heads": 16,
                    "dim_head": 64,
                    "ff_mult": 2,
                    "mel_dim": 80,
                    "mu_dim": 80,
                    "spk_dim": 80,
                    "out_channels": 80,
                    "static_chunk_size": self.token_frame_rate * self.token_mel_ratio,
                    "num_decoding_left_chunks": -1,
                },
            },
        }
        self.hift = {
            "in_channels": 80,
            "base_channels": 512,
            "nb_harmonics": 8,
            "sampling_rate": self.sample_rate,
            "nsf_alpha": 0.1,
            "nsf_sigma": 0.003,
            "nsf_voiced_threshold": 10,
            "upsample_rates": [8, 5, 3],
            "upsample_kernel_sizes": [16, 11, 7],
            "istft_params": {
                "n_fft": 16,
                "hop_len": 4,
            },
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "source_resblock_kernel_sizes": [7, 7, 11],
            "source_resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "lrelu_slope": 0.1,
            "audio_limit": 0.99,
            "conv_pre_look_right": 4,
            "f0_predictor": {
                "num_class": 1,
                "in_channels": 80,
                "cond_channels": 512,
            },
        }
