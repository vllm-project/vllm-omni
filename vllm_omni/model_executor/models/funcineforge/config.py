# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FunCineForge model configuration."""

from __future__ import annotations

from typing import Any

from transformers.configuration_utils import PretrainedConfig


class FunCineForgeConfig(PretrainedConfig):
    """Configuration for the FunCineForge dubbing model.

    The model has three stages: LM (talker) -> Flow Matching -> Causal HiFiGAN.
    Within vLLM-Omni the last two stages are bundled as "code2wav".

    Follows the ``PretrainedConfig`` pattern used by CosyVoice3, OmniVoice,
    FishSpeech, Qwen3-TTS, and VoxtralTTS in this repository.
    """

    model_type = "funcineforge"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("eos_token_id", 6562)
        super().__init__(**kwargs)

        # vLLM requires hidden_size on the top-level config for the
        # multimodal pipeline; it comes from the Qwen2-0.5B backbone.
        self.hidden_size: int = kwargs.get("hidden_size", 896)

        # vocab_size must cover the Qwen2 tokenizer (151936) plus any
        # FunCineForge special tokens (startofclue=151646, endofclue=151647).
        # Use Qwen2-0.5B's native vocab_size to satisfy vLLM's input
        # validation (InputProcessor._validate_model_input).
        self.vocab_size: int = kwargs.get("vocab_size", 151936)

        # Qwen2-0.5B attention parameters — vLLM uses these for KV cache
        # allocation (num_key_value_heads, num_hidden_layers) and profiling.
        # Must match the inner Qwen2 backbone.
        self.num_attention_heads: int = kwargs.get("num_attention_heads", 14)
        self.num_key_value_heads: int = kwargs.get("num_key_value_heads", 2)
        self.num_hidden_layers: int = kwargs.get("num_hidden_layers", 24)

        # -------------------------------------------------------------- LM
        self.llm: dict[str, Any] = kwargs.get(
            "llm",
            {
                "llm": {
                    "pretrain_path": "Qwen2-0.5B-CosyVoice-BlankEN",
                },
            },
        )

        # Codec vocabulary size used by the talker (LLM embedding).
        # Flow matching uses a smaller codebook_size (6561) for mel embedding.
        self.codec_unit: int = kwargs.get("codec_unit", 6761)
        # Timespk vocabulary size (gender/age/speaker-id tags)
        self.timespk_unit: int = kwargs.get("timespk_unit", 1550)
        # Face embedding dimension (input to face_linear)
        self.face_size: int = kwargs.get("face_size", 512)

        # Special token IDs (from decode.yaml — fixed positions within codec_head range)
        self.sos: int = kwargs.get("sos", 6561)
        self.eos: int = kwargs.get("eos", 6562)
        self.turn_of_speech: int = kwargs.get("turn_of_speech", 6563)
        self.fill_token: int = kwargs.get("fill_token", 6564)

        # Timespeaker tag IDs (from decode.yaml)
        self.pangbai: int = kwargs.get("pangbai", 1500)
        self.dubai: int = kwargs.get("dubai", 1501)
        self.duihua: int = kwargs.get("duihua", 1502)
        self.duoren: int = kwargs.get("duoren", 1503)
        self.male: int = kwargs.get("male", 1504)
        self.female: int = kwargs.get("female", 1505)
        self.child: int = kwargs.get("child", 1506)
        self.youth: int = kwargs.get("youth", 1507)
        self.adult: int = kwargs.get("adult", 1508)
        self.middle: int = kwargs.get("middle", 1509)
        self.elderly: int = kwargs.get("elderly", 1510)
        self.speaker_id_start: int = kwargs.get("speaker_id_start", 1511)

        # Qwen special tokens used by FunCineForge
        self.startofclue_token: int = kwargs.get("startofclue_token", 151646)
        self.endofclue_token: int = kwargs.get("endofclue_token", 151647)

        # Generation limits
        self.max_length: int = kwargs.get("max_length", 1500)  # 60 s * 25 fps
        self.min_length: int = kwargs.get("min_length", 50)  # 2 s * 25 fps
        self.token_rate: int = kwargs.get("token_rate", 25)  # codec tokens per second
        self.sample_rate: int = kwargs.get("sample_rate", 24000)

        # Sampling method: "ras" (Repetition Aware Sampling)
        self.sampling: str = kwargs.get("sampling", "ras")

        # -------------------------------------------------------------- FM
        # Defaults match the official funcineforge_zh_en checkpoint
        self.flow: dict[str, Any] = kwargs.get(
            "flow",
            {
                "codebook_size": 6561,
                "model_size": 1024,
                "xvec_size": 192,
                "dit_conf": {
                    "dim": 1024,
                    "depth": 22,
                    "heads": 16,
                    "dim_head": 64,
                    "ff_mult": 2,
                    "mel_dim": 80,
                    "mu_dim": 80,
                    "spk_dim": 80,
                    "causal_mask_type": [
                        {"prob_min": 0.0, "prob_max": 0.25, "block_size": -1, "ratio": 2},
                        {"prob_min": 0.25, "prob_max": 0.5, "block_size": 1, "ratio": 2},
                        {"prob_min": 0.5, "prob_max": 0.75, "block_size": 15, "ratio": 2},
                        {"prob_min": 0.75, "prob_max": 1.0, "block_size": 30, "ratio": 2},
                    ],
                },
                "mel_feat_conf": {
                    "n_fft": 1920,
                    "hop_length": 480,
                    "win_length": 1920,
                    "sampling_rate": 24000,
                    "n_mel_channels": 80,
                    "mel_fmin": 0,
                    "mel_fmax": 8000,
                    "center": False,
                    "feat_type": "power_log",
                },
                "prompt_conf": {
                    "prompt_type": "prefix",
                    "prompt_width_ratio_range": [0.7, 1.0],
                },
                "inference_cfg_rate": 0.7,
                "lookahead_length": 3,
                "feat_token_ratio": 2,  # mel frames per codec token (50 Hz / 25 Hz)
                "token_rate": 25,
                "n_timesteps": 10,
            },
        )

        # -------------------------------------------------------------- Vocoder
        # Defaults match the official funcineforge_zh_en checkpoint
        self.vocoder: dict[str, Any] = kwargs.get(
            "vocoder",
            {
                "CausalHiFTGenerator_conf": {
                    "in_channels": 80,
                    "base_channels": 512,
                    "nb_harmonics": 8,
                    "sampling_rate": 24000,
                    "nsf_alpha": 0.1,
                    "nsf_sigma": 0.003,
                    "nsf_voiced_threshold": 10,
                    "upsample_rates": [8, 5, 3],
                    "upsample_kernel_sizes": [16, 11, 7],
                    "istft_params": {"n_fft": 16, "hop_len": 4},
                    "resblock_kernel_sizes": [3, 7, 11],
                    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    "source_resblock_kernel_sizes": [7, 7, 11],
                    "source_resblock_dilation_sizes": [
                        [1, 3, 5],
                        [1, 3, 5],
                        [1, 3, 5],
                    ],
                    "lrelu_slope": 0.1,
                    "audio_limit": 0.99,
                },
                "CausalConvRNNF0Predictor_conf": {
                    "num_class": 1,
                    "in_channels": 80,
                    "cond_channels": 512,
                },
                "sample_rate": 24000,
            },
        )

        # -------------------------------------------------------------- Paths
        # Relative to model_dir; matches FunAudioLLM/Fun-CineForge HF layout
        self.llm_ckpt: str = kwargs.get(
            "llm_ckpt",
            "funcineforge_zh_en/llm/ds-model.pt.best/mp_rank_00_model_states.pt",
        )
        self.flow_ckpt: str = kwargs.get(
            "flow_ckpt",
            "funcineforge_zh_en/flow/ds-model.pt.best/mp_rank_00_model_states.pt",
        )
        self.vocoder_ckpt: str = kwargs.get(
            "vocoder_ckpt",
            "funcineforge_zh_en/vocoder/ds-model.pt.best/avg_5_removewn.pt",
        )
        self.campplus_onnx: str = kwargs.get(
            "campplus_onnx",
            "funcineforge_zh_en/camplus.onnx",
        )
        self.speech_tokenizer_path: str = kwargs.get("speech_tokenizer_path", "speech_tokenizer_v3.onnx")

        # -------------------------------------------------------------- Misc
        # Dynamic token ratio for online serving (chars -> min/max tokens)
        self.min_token_text_ratio: int = kwargs.get("min_token_text_ratio", 2)
        self.max_token_text_ratio: int = kwargs.get("max_token_text_ratio", 20)
