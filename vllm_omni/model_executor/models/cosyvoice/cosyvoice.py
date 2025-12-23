import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from transformers.configuration_utils import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalDataDict, MultiModalFieldConfig
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptIndexTargets,
    PromptInsertion,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs

# from vllm.model_executor.models.qwen2 import
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import OmniOutput
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    MultiModalDataItems,
    MultiModalKwargsItems,
)

logger = init_logger(__name__)


class CosyVoiceConfig(PretrainedConfig):
    model_type = "cosyvoice"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sample_rate = 24000
        self.llm_input_size = 896
        self.llm_output_size = 896
        self.hidden_size = self.llm_output_size
        self.spk_embed_dim = 192
        self.token_frame_rate = 25
        self.token_mel_ratio = 2
        self.allowed_special = "all"
        self.skip_special_tokens = True
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
        self.qwen_pretrain_path = "CosyVoice-BlankEN"
        self.campplus_onxx_path = "campplus.onnx"
        self.speech_tokenizer_path = "speech_tokenizer_v3.onnx"
        self.spk2info_path = "spk2info.pt"
        self.version = "cosyvoice3"
        self.llm = {
            "llm_input_size": self.llm_input_size,
            "llm_output_size": self.llm_output_size,
            "speech_token_size": 6561,
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
        self.hf_cache_location = "/mnt/d/.cache/huggingface/hub/models--FunAudioLLM--Fun-CosyVoice3-0.5B-2512/"
        self.model_path = "snapshots/5646a54a6bea9eb1ec64b3ded068fdcf5a65f9ae/"
        self.model_dir = self.hf_cache_location + self.model_path
        self.tokenizer_path = self.hf_cache_location + self.model_path + "CosyVoice-BlankEN"


class CosyVoiceMultiModalProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(CosyVoiceConfig)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None}


class CosyVoiceMultiModalProcessor(BaseMultiModalProcessor[CosyVoiceMultiModalProcessingInfo]):
    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def _get_frontend(self):
        if hasattr(self, "_frontend"):
            return self._frontend

        from cosyvoice.cli.frontend import CosyVoiceFrontEnd
        from cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer
        from matcha.utils.audio import mel_spectrogram

        config = self.info.ctx.model_config.hf_config
        model_dir = self.info.ctx.model_config.model
        model_dir = config.model_dir
        if not os.path.isdir(model_dir):
            raise ValueError(f"CosyVoice model_dir not found: {model_dir}")

        yaml_name = None
        for name in ("cosyvoice3.yaml", "cosyvoice2.yaml", "cosyvoice.yaml"):
            if os.path.exists(os.path.join(model_dir, name)):
                yaml_name = name
                break
        if yaml_name is None:
            raise FileNotFoundError("Cannot find cosyvoice*.yaml in model dir")

        feat_cfg = getattr(config, "feat_extractor", {})
        feat_extractor = partial(mel_spectrogram, **feat_cfg)
        tokenizer = partial(
            get_qwen_tokenizer,
            token_path=os.path.join(model_dir, config.qwen_pretrain_path),
            skip_special_tokens=config.skip_special_tokens,
            version=config.version,
        )

        print(config)
        self._frontend = CosyVoiceFrontEnd(
            tokenizer,
            feat_extractor,
            os.path.join(model_dir, config.campplus_onxx_path),
            os.path.join(model_dir, config.speech_tokenizer_path),
            os.path.join(model_dir, config.spk2info_path),
            config.allowed_special,
        )
        return self._frontend

    def _ensure_wav_path(self, audio_item: object, mm_kwargs: Mapping[str, object]) -> str:
        if isinstance(audio_item, tuple):
            audio, sr = audio_item
        else:
            audio, sr = audio_item, None

        if isinstance(audio, torch.Tensor):
            audio_tensor = audio.detach().cpu()
            if audio_tensor.ndim == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
        elif isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
            if audio_tensor.ndim == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported audio type: {type(audio)}")

        if sr is None:
            sr = int(mm_kwargs.get("sample_rate", 24000))

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        torchaudio.save(tmp.name, audio_tensor, sr)
        return tmp.name

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        frontend = self._get_frontend()
        print(f"mm_data: {mm_data}")
        print(f"mm_kwargs: {mm_kwargs}")
        print(f"tok_kwargs: {tok_kwargs}")

        audio = mm_data.get("audio")
        if audio is None:
            audio = mm_data.get("audios")
            if audio is not None:
                audio = audio[0]

        if audio is None:
            # Text-only path for profiling/cache
            text_token, _ = frontend._extract_text_token(prompt)
            return BatchFeature({"input_ids": text_token})

        prompt_text = mm_kwargs.get("prompt_text")
        if not isinstance(prompt_text, str):
            raise ValueError("CosyVoice zero-shot requires mm_kwargs['prompt_text'] as a string.")

        wav_path = self._ensure_wav_path(audio, mm_kwargs)
        try:
            model_input = frontend.frontend_zero_shot(
                prompt,
                prompt_text,
                wav_path,
                24000,
                mm_kwargs.get("zero_shot_spk_id", ""),
            )
            model_input["input_ids"] = model_input["llm_prompt_speech_token"]
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

        # TODO: support cross_lingual, instruct2, sft, and vc modes.
        return BatchFeature(model_input)

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        logger.info("Getting CosyVoice MM fields config.")
        logger.info(f"hf_processor_mm_kwargs: {hf_processor_mm_kwargs}")
        # logger.info(f"hf_inputs: {hf_inputs}")
        return {
            "prompt_speech_feat": MultiModalFieldConfig.batched("audio"),
            "prompt_speech_feat_len": MultiModalFieldConfig.batched("audio"),
            "llm_prompt_speech_token": MultiModalFieldConfig.batched("audio"),
            "llm_prompt_speech_token_len": MultiModalFieldConfig.batched("audio"),
            "flow_prompt_speech_token": MultiModalFieldConfig.batched("audio"),
            "flow_prompt_speech_token_len": MultiModalFieldConfig.batched("audio"),
            "llm_embedding": MultiModalFieldConfig.batched("audio"),
            "flow_embedding": MultiModalFieldConfig.batched("audio"),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        logger.info("Getting CosyVoice prompt updates.")
        logger.info(
            f"mm_items: {mm_items} hf_processor_mm_kwargs: {hf_processor_mm_kwargs} out_mm_kwargs: {out_mm_kwargs}"
        )
        out_mm_data = out_mm_kwargs.get_data()
        prompt_feat_lens = out_mm_data["prompt_speech_feat_len"]
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()
        placeholder = None
        placeholder_token_id = getattr(tokenizer, "eos_token_id", None)
        if placeholder_token_id is None:
            placeholder_token_id = next(iter(vocab.values()))

        def insertion(item_idx: int):
            n = int(prompt_feat_lens[item_idx].item())
            if placeholder is not None:
                return PromptUpdateDetails.from_seq(placeholder * n)
            return PromptUpdateDetails.from_seq([placeholder_token_id] * n)

        return [
            PromptInsertion(
                modality="audio",
                target=PromptIndexTargets.start(),
                insertion=insertion,
            )
        ]


class CosyVoiceDummyInputsBuilder(BaseDummyInputsBuilder[CosyVoiceMultiModalProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "This is a dummy text input for CosyVoice model."

    def get_dummy_mm_data(
        self, seq_len: int, mm_counts: Mapping[str, int], mm_options: Mapping[str, BaseDummyOptions] | None = None
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio")

        max_prompt_seconds = 30
        prompt_sample_rate = 24000
        target_audio_length = max_prompt_seconds * prompt_sample_rate

        audio_overrides = mm_options.get("audio") if mm_options else None
        mm_data = {
            "audio": self._get_dummy_audios(
                length=target_audio_length,
                num_audios=num_audios,
                overrides=audio_overrides,
            )
        }
        return mm_data

    def get_dummy_processor_inputs(
        self, seq_len: int, mm_counts: Mapping[str, int], mm_options: Mapping[str, BaseDummyOptions] | None = None
    ) -> ProcessorInputs:
        logger.info("Building dummy processor inputs for CosyVoice.")
        logger.info(f"mm_counts: {mm_counts}, mm_options: {mm_options}")
        logger.info(f"seq_len: {seq_len}")
        inputs = super().get_dummy_processor_inputs(seq_len, mm_counts, mm_options)
        # print(self.ctx.get_hf_config())
        inputs.hf_processor_mm_kwargs = {"prompt_text": "This is a dummy prompt text for CosyVoice."}
        logger.info(f"Built dummy processor inputs for CosyVoice. {inputs}")
        return inputs


@MULTIMODAL_REGISTRY.register_processor(
    CosyVoiceMultiModalProcessor,
    info=CosyVoiceMultiModalProcessingInfo,
    dummy_inputs=CosyVoiceDummyInputsBuilder,
)
class CosyVoiceModel(
    nn.Module,
    SupportsMultiModal,
):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        print(self.config)
        print(vllm_config.model_config)
        self.model_stage = vllm_config.model_config.model_stage
        self.model_dir = vllm_config.model_config.model
        self.model = None
        if self.model_stage == "text_speech_lm":
            # Initialize text to speech LM stage
            from cosyvoice.llm.llm import CosyVoice3LM, Qwen2Encoder
            from cosyvoice.utils.common import ras_sampling

            # os.path.join(model_dir,
            llm = Qwen2Encoder(os.path.join(self.model_dir, self.config.llm["llm"]["pretrain_path"]))
            self.text_speech_lm_model = CosyVoice3LM(
                llm_input_size=self.config.llm["llm_input_size"],
                llm_output_size=self.config.llm["llm_output_size"],
                speech_token_size=self.config.llm["speech_token_size"],
                llm=llm,
                sampling=partial(ras_sampling, **self.config.llm["sampling"]),
                length_normalized_loss=self.config.llm["length_normalized_loss"],
                lsm_weight=self.config.llm["lsm_weight"],
                mix_ratio=self.config.llm["mix_ratio"],
            )
            self.model = self.text_speech_lm_model

        elif self.model_stage == "chunk_aware_flow_matching":
            # Initialize chunk aware flow matching stage
            from cosyvoice.flow.DiT.dit import DiT
            from cosyvoice.flow.flow import CausalMaskedDiffWithDiT
            from cosyvoice.flow.flow_matching import CausalConditionalCFM
            from cosyvoice.transformer.upsample_encoder import PreLookaheadLayer
            from omegaconf import DictConfig

            pre_lookahead_layer = PreLookaheadLayer(**self.config.flow["pre_lookahead_layer"])

            decoder_cfg = self.config.flow["decoder"]
            cfm_params = DictConfig(decoder_cfg["cfm_params"])
            estimator = DiT(**decoder_cfg["estimator"])
            decoder = CausalConditionalCFM(
                in_channels=decoder_cfg["in_channels"],
                estimator=estimator,
                cfm_params=cfm_params,
                n_spks=decoder_cfg["n_spks"],
                spk_emb_dim=decoder_cfg["spk_emb_dim"],
            )
            self.chunk_aware_flow_matching_model = CausalMaskedDiffWithDiT(
                input_size=self.config.flow["input_size"],
                output_size=self.config.flow["output_size"],
                spk_embed_dim=self.config.flow["spk_embed_dim"],
                output_type=self.config.flow["output_type"],
                vocab_size=self.config.flow["vocab_size"],
                input_frame_rate=self.config.flow["input_frame_rate"],
                only_mask_loss=self.config.flow["only_mask_loss"],
                token_mel_ratio=self.config.flow["token_mel_ratio"],
                pre_lookahead_len=self.config.flow["pre_lookahead_len"],
                pre_lookahead_layer=pre_lookahead_layer,
                decoder=decoder,
            )
            self.model = self.chunk_aware_flow_matching_model
        elif self.model_stage == "acoustic_features_to_waveform":
            # Initialize acoustic features to waveform stage
            from cosyvoice.hifigan.f0_predictor import CausalConvRNNF0Predictor
            from cosyvoice.hifigan.generator import CausalHiFTGenerator

            f0_predictor = CausalConvRNNF0Predictor(
                num_class=self.config.hift["f0_predictor"]["num_class"],
                in_channels=self.config.hift["f0_predictor"]["in_channels"],
                cond_channels=self.config.hift["f0_predictor"]["cond_channels"],
            )
            self.acoustic_features_to_waveform_model = CausalHiFTGenerator(
                in_channels=self.config.hift["in_channels"],
                base_channels=self.config.hift["base_channels"],
                nb_harmonics=self.config.hift["nb_harmonics"],
                sampling_rate=self.config.hift["sampling_rate"],
                nsf_alpha=self.config.hift["nsf_alpha"],
                nsf_sigma=self.config.hift["nsf_sigma"],
                nsf_voiced_threshold=self.config.hift["nsf_voiced_threshold"],
                upsample_rates=self.config.hift["upsample_rates"],
                upsample_kernel_sizes=self.config.hift["upsample_kernel_sizes"],
                istft_params=self.config.hift["istft_params"],
                resblock_kernel_sizes=self.config.hift["resblock_kernel_sizes"],
                resblock_dilation_sizes=self.config.hift["resblock_dilation_sizes"],
                source_resblock_kernel_sizes=self.config.hift["source_resblock_kernel_sizes"],
                source_resblock_dilation_sizes=self.config.hift["source_resblock_dilation_sizes"],
                lrelu_slope=self.config.hift["lrelu_slope"],
                audio_limit=self.config.hift["audio_limit"],
                conv_pre_look_right=self.config.hift["conv_pre_look_right"],
                f0_predictor=f0_predictor,
            )
            self.model = self.acoustic_features_to_waveform_model
            pass
        else:
            raise ValueError(f"Unknown model stage: {self.model_stage}")

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.model_stage == "text_speech_lm":
            llm = getattr(self.model, "llm", None)
            embed_tokens = getattr(getattr(getattr(llm, "model", None), "model", None), "embed_tokens", None)
            if callable(embed_tokens):
                return embed_tokens(input_ids)
            hidden = int(self.config.llm["llm_input_size"])
        elif self.model_stage == "chunk_aware_flow_matching":
            hidden = int(self.config.flow["input_size"])
        else:
            hidden = int(self.config.hift["in_channels"])
        return torch.zeros(
            (input_ids.shape[0], input_ids.shape[1], hidden),
            device=input_ids.device,
            dtype=torch.float32,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.model_stage == "text_speech_lm":
            logger.info(
                "Forward pass for text to speech LM stage"
                f"input_ids: {input_ids},  intermediate_tensors: {intermediate_tensors}, inputs_embeds: {inputs_embeds}"
                f"inputs_embeds {inputs_embeds} kwargs: {kwargs.keys()}"
                f"kwargs: {kwargs}"
            )
            if not kwargs:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                dtype = inputs_embeds.dtype if inputs_embeds is not None else torch.float32
                batch = inputs_embeds.shape[0] if inputs_embeds is not None else input_ids.shape[0]
                hidden = int(self.config.llm["llm_output_size"])
                return torch.zeros((batch, 1, hidden), device=device, dtype=dtype)
            hidden_states = self.model.inference()
            return OmniOutput(text_hidden_states=hidden_states)
        elif self.model_stage == "chunk_aware_flow_matching":
            logger.info("Forward pass for chunk aware flow matching stage")
            if not kwargs:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                dtype = inputs_embeds.dtype if inputs_embeds is not None else torch.float32
                batch = inputs_embeds.shape[0] if inputs_embeds is not None else input_ids.shape[0]
                time_steps = 1
                feature_dim = int(self.config.flow["output_size"])
                return torch.zeros((batch, time_steps, feature_dim), device=device, dtype=dtype)
            hidden_states = self.model.inference()
            return OmniOutput(text_hidden_states=hidden_states)
        elif self.model_stage == "acoustic_features_to_waveform":
            logger.info("Forward pass for acoustic features to waveform stage")
            if not kwargs:
                device = input_ids.device if input_ids is not None else inputs_embeds.device
                dtype = inputs_embeds.dtype if inputs_embeds is not None else torch.float32
                batch = inputs_embeds.shape[0] if inputs_embeds is not None else input_ids.shape[0]
                time_steps = 1
                return torch.zeros((batch, time_steps), device=device, dtype=dtype)
            hidden_states = self.model.inference()
            return OmniOutput(text_hidden_states=hidden_states)
        else:
            logger.error(f"Unknown model stage during forward: {self.model_stage}")
            raise ValueError(f"Unknown model stage: {self.model_stage}")
            # Forward pass for text to speech LM stage
        pass

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None
        if self.model_stage == "text_speech_lm" and hasattr(self.model, "llm_decoder"):
            return self.model.llm_decoder(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        logger.info(f"Loading weights for CosyVoice model stage: {self.model_stage}")
        loaded_names = {name for name, _ in self.named_parameters()}
        if self.model_stage == "text_speech_lm":
            # Load weights for text to speech LM stage
            llm_weight_path = os.path.join(self.model_dir, "llm.pt")
            logger.info(f"Loading LLM weights from {llm_weight_path}")
            device = next(self.parameters()).device
            self.model.load_state_dict(torch.load(llm_weight_path, map_location=device), strict=True)
            self.model.to(device).eval()
        elif self.model_stage == "chunk_aware_flow_matching":
            # Load weights for chunk aware flow matching stage
            flow_weight_path = os.path.join(self.model_dir, "flow.pt")
            logger.info(f"Loading Flow weights from {flow_weight_path}")
            device = next(self.parameters()).device
            self.model.load_state_dict(torch.load(flow_weight_path, map_location=device), strict=True)
            self.model.to(device).eval()
        elif self.model_stage == "acoustic_features_to_waveform":
            # Load weights for acoustic features to waveform stage
            hift_weight_path = os.path.join(self.model_dir, "hift.pt")
            logger.info(f"Loading HIFT weights from {hift_weight_path}")
            device = "cpu"
            self.model.load_state_dict(torch.load(hift_weight_path, map_location=device), strict=True)
            self.model.to(device).eval()
        else:
            raise ValueError(f"Unknown model stage: {self.model_stage}")
        return loaded_names
