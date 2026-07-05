# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy

from transformers import PretrainedConfig


class IndexTTS2Config(PretrainedConfig):
    model_type = "indextts2"

    # Defaults for the official IndexTTS2 checkpoint.  Values from HF
    # config.json (passed via ``kwargs``) take precedence over these.
    _DEFAULTS = {
        "num_attention_heads": 20,
        "hidden_size": 1280,
        "dataset": {
            "bpe_model": "bpe.model",
            "sample_rate": 24000,
            "squeeze": False,
            "mel": {
                "sample_rate": 24000,
                "n_fft": 1024,
                "hop_length": 256,
                "win_length": 1024,
                "n_mels": 100,
                "mel_fmin": 0,
                "normalize": False,
            },
        },
        "gpt": {
            "model_dim": 1280,
            "max_mel_tokens": 1815,
            "max_text_tokens": 600,
            "heads": 20,
            "use_mel_codes_as_input": True,
            "mel_length_compression": 1024,
            "layers": 24,
            "number_text_tokens": 12000,
            "number_mel_codes": 8194,
            "start_mel_token": 8192,
            "stop_mel_token": 8193,
            "start_text_token": 0,
            "stop_text_token": 1,
            "train_solo_embeddings": False,
            "condition_num_latent": 32,
            "condition_type": "conformer_perceiver",
            "condition_module": {
                "output_size": 512,
                "linear_units": 2048,
                "attention_heads": 8,
                "num_blocks": 6,
                "input_layer": "conv2d2",
                "perceiver_mult": 2,
            },
            "emo_condition_module": {
                "output_size": 512,
                "linear_units": 1024,
                "attention_heads": 4,
                "num_blocks": 4,
                "input_layer": "conv2d2",
                "perceiver_mult": 2,
            },
        },
        "semantic_codec": {
            "codebook_size": 8192,
            "hidden_size": 1024,
            "codebook_dim": 8,
            "vocos_dim": 384,
            "vocos_intermediate_dim": 2048,
            "vocos_num_layers": 12,
        },
        "s2mel": {
            "preprocess_params": {
                "sr": 22050,
                "spect_params": {
                    "n_fft": 1024,
                    "win_length": 1024,
                    "hop_length": 256,
                    "n_mels": 80,
                    "fmin": 0,
                    "fmax": "None",
                },
            },
            "dit_type": "DiT",
            "reg_loss_type": "l1",
            "style_encoder": {"dim": 192},
            "length_regulator": {
                "channels": 512,
                "is_discrete": False,
                "in_channels": 1024,
                "content_codebook_size": 2048,
                "sampling_ratios": [1, 1, 1, 1],
                "vector_quantize": False,
                "n_codebooks": 1,
                "quantizer_dropout": 0.0,
                "f0_condition": False,
                "n_f0_bins": 512,
            },
            "DiT": {
                "hidden_dim": 512,
                "num_heads": 8,
                "depth": 13,
                "class_dropout_prob": 0.1,
                "block_size": 8192,
                "in_channels": 80,
                "style_condition": True,
                "final_layer_type": "wavenet",
                "target": "mel",
                "content_dim": 512,
                "content_codebook_size": 1024,
                "content_type": "discrete",
                "f0_condition": False,
                "n_f0_bins": 512,
                "content_codebooks": 1,
                "is_causal": False,
                "long_skip_connection": True,
                "zero_prompt_speech_token": False,
                "time_as_token": False,
                "style_as_token": False,
                "uvit_skip_connection": True,
                "add_resblock_in_transformer": False,
            },
            "wavenet": {
                "hidden_dim": 512,
                "num_layers": 8,
                "kernel_size": 5,
                "dilation_rate": 1,
                "p_dropout": 0.2,
                "style_condition": True,
            },
        },
        "gpt_checkpoint": "gpt.pth",
        "w2v_stat": "wav2vec2bert_stats.pt",
        "s2mel_checkpoint": "s2mel.pth",
        "emo_matrix": "feat2.pt",
        "spk_matrix": "feat1.pt",
        "emo_num": [3, 17, 2, 8, 4, 5, 10, 24],
        "qwen_emo_path": "qwen0.6bemo4-merge/",
        "vocoder": {"type": "bigvgan", "name": "nvidia/bigvgan_v2_22khz_80band_256x"},
        "version": 2.0,
        # Default-off performance tracing for IndexTTS2 overhead analysis.
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        defaults = copy.deepcopy(self._DEFAULTS)
        for attr, default in defaults.items():
            if not hasattr(self, attr):
                setattr(self, attr, default)
        # Keep gpt.model_dim in sync with top-level hidden_size.
        if isinstance(self.gpt, dict) and self.gpt.get("model_dim") != self.hidden_size:
            self.gpt["model_dim"] = self.hidden_size
        self.output_sample_rate = int(
            self.s2mel["preprocess_params"].get("sr", 22050) if isinstance(self.s2mel, dict) else 22050
        )
