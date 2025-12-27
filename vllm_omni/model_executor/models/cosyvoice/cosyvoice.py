import json
import os
from collections.abc import Iterable, Mapping, Sequence
from functools import partial

import librosa
import numpy as np
import onnxruntime
import torch
import torch.nn as nn
import torchaudio.compliance.kaldi as kaldi
import whisper
from transformers.configuration_utils import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalDataDict, MultiModalFieldConfig
from vllm.multimodal.parse import MultiModalDataParser
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptIndexTargets,
    PromptInsertion,
    PromptUpdate,
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
        self.min_token_text_ratio = 2
        self.max_token_text_ratio = 20
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
        """If the config is not already present pass it
        as a class and it will try to find it in your
        model directory just copy the config class there also.
        """
        return self.ctx.get_hf_config(CosyVoiceConfig)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        """How many audio can you pass. I think I should keep it as 1
        For now I have kept it None.
        """
        return {"audio": None}


############## Util functions
@staticmethod
def _concat_text_with_prompt_ids(
    text: torch.Tensor,
    text_len: torch.Tensor,
    prompt_text: torch.Tensor,
    prompt_text_len: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # logger.info("Concatenating prompt_text with main text input")
    # logger.info(f"text shape: {text.shape}, text_len: {text_len}")
    # logger.info(f"prompt_text shape: {prompt_text.shape}, prompt_text_len: {prompt_text_len}")
    text = torch.concat([prompt_text, text], dim=1)
    text_len = text_len + prompt_text_len
    # logger.info(f"After concat, text shape: {text.shape}, text_len shape: {text_len}")
    return text, text_len


def extract_text_token(text, tokenizer, allowed_special):
    text_token = tokenizer.tokenizer(text, return_tensors="pt")["input_ids"]
    # logger.info(text_token)
    # logger.info(text_token.shape)
    text_token_len = text_token.shape[1]
    return text_token, text_token_len


def load_wav(wav, target_sr, min_sr=16000):
    # logger.info(type(wav))
    if not isinstance(wav, tuple):
        speech, sample_rate = librosa.load(wav)
    else:
        speech, sample_rate = wav
        speech = speech

    if sample_rate != target_sr:
        assert sample_rate >= min_sr, f"wav sample rate {sample_rate} must be greater than {target_sr}"
        speech = librosa.resample(speech, orig_sr=sample_rate, target_sr=target_sr)

    speech = torch.tensor([speech], dtype=torch.float32)
    # logger.info(type(speech))
    return speech


def _extract_speech_feat(prompt_wav, feat_extractor, device):
    speech = load_wav(prompt_wav, 24000)
    speech_feat = feat_extractor(speech).squeeze(dim=0).transpose(0, 1).to(device)
    speech_feat = speech_feat.unsqueeze(dim=0)
    speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32).to(device)
    return speech_feat, speech_feat_len


def _extract_speech_token(prompt_wav, speech_tokenizer_session, device):
    speech = load_wav(prompt_wav, 16000)
    assert speech.shape[1] / 16000 <= 30, "do not support extract speech token for audio longer than 30s"

    feat = whisper.log_mel_spectrogram(speech, n_mels=128)
    speech_token = (
        speech_tokenizer_session.run(
            None,
            {
                speech_tokenizer_session.get_inputs()[0].name: feat.detach().cpu().numpy(),
                speech_tokenizer_session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32),
            },
        )[0]
        .flatten()
        .tolist()
    )
    speech_token = torch.tensor([speech_token], dtype=torch.int32).to(device)
    speech_token_len = torch.tensor([speech_token.shape[1]], dtype=torch.int32).to(device)
    return speech_token, speech_token_len


def _extract_spk_embedding(prompt_wav, campplus_session, device):
    speech = load_wav(prompt_wav, 16000)
    feat = kaldi.fbank(speech, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    embedding = (
        campplus_session.run(None, {campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0]
        .flatten()
        .tolist()
    )
    embedding = torch.tensor([embedding]).to(device)
    return embedding


def _make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


def _fade_in_out(fade_in_mel, fade_out_mel, window):
    device = fade_in_mel.device
    fade_in_mel, fade_out_mel = fade_in_mel.cpu(), fade_out_mel.cpu()
    mel_overlap_len = int(window.shape[0] / 2)
    if fade_in_mel.device == torch.device("cpu"):
        fade_in_mel = fade_in_mel.clone()
    fade_in_mel[..., :mel_overlap_len] = (
        fade_in_mel[..., :mel_overlap_len] * window[:mel_overlap_len]
        + fade_out_mel[..., -mel_overlap_len:] * window[mel_overlap_len:]
    )
    return fade_in_mel.to(device)


def _compute_min_max_len(
    text_len: torch.Tensor,
    prompt_text_len: torch.Tensor,
    min_token_text_ratio: float,
    max_token_text_ratio: float,
) -> tuple[int, int]:
    base_len = text_len - prompt_text_len
    return int(base_len * min_token_text_ratio), int(base_len * max_token_text_ratio)


###############################

"""
1. step get input prompt and mm_data to return input_ids and tensors.
()
2.
"""


class CosyVoiceMultiModalProcessor(BaseMultiModalProcessor[CosyVoiceMultiModalProcessingInfo]):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """
        apply-> cached_apply_hf_processor -> apply_hf_processor_mm ->
        _call_hf_processor.
        _call_hf_processor takes input prompt and mm_data and returns
        token ids and tensors
        """
        from cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer

        # logger.info(f"{prompt} {mm_data} {mm_kwargs} {tok_kwargs}")
        config = self.info.ctx.get_hf_config()
        # mc = self.info.ctx.model_config
        # logger.info("HF processor stage_id=%s model_stage=%s", mc.stage_id, mc.model_stage)
        model_dir = self.info.ctx.model_config.model
        self.tokenizer = get_qwen_tokenizer(
            token_path=os.path.join(model_dir, config.qwen_pretrain_path),
            skip_special_tokens=config.skip_special_tokens,
            version=config.version,
        )

        option = onnxruntime.SessionOptions()
        self.speech_tokenizer = onnxruntime.InferenceSession(
            os.path.join(model_dir, config.speech_tokenizer_path),
            sess_options=option,
            providers=["CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"],
        )

        from matcha.utils.audio import mel_spectrogram

        feat_cfg = getattr(config, "feat_extractor", {})
        self.feat_extractor = partial(mel_spectrogram, **feat_cfg)
        campplus_full_path = os.path.join(model_dir, config.campplus_onxx_path)
        self.campplus_session = onnxruntime.InferenceSession(
            campplus_full_path, sess_options=option, providers=["CPUExecutionProvider"]
        )

        audio = mm_data.get("audio", None)

        # TODO: See where audios is coming from
        if audio is None:
            audio = mm_data.get("audios")
            if audio is not None:
                # logger.info(f"audios : {audio}")
                audio = audio[0], 24000

        text_token, text_token_len = extract_text_token(prompt, self.tokenizer, config.allowed_special)
        if audio is None:
            # Text-only path for profiling/cache
            return BatchFeature({"input_ids": text_token, "input_len": [text_token_len]})

        prompt_text = mm_kwargs.get("prompt_text")
        ## Unsure how to pass prompt text for profiling'
        # cannot pass text in mm_data so need to have a workaround.
        # For now if audio is present but prompt text is not I can
        # return at above only?
        # logger.info(f"prompt_text {prompt_text}")

        if not isinstance(prompt_text, str):
            raise ValueError(f"prompt text is None : {prompt_text}")

        prompt_text_token, prompt_text_token_len = extract_text_token(
            prompt_text, self.tokenizer, config.allowed_special
        )

        input_ids, input_len = _concat_text_with_prompt_ids(
            text_token,
            text_token_len,
            prompt_text_token,
            prompt_text_token_len,
        )
        logger.info(
            "cosyvoice _call_hf_processor: prompt_text_token=%s text_token=%s input_ids=%s "
            "prompt_text_len=%s text_len=%s input_len=%s",
            prompt_text_token.tolist(),
            text_token.tolist(),
            input_ids.tolist(),
            int(prompt_text_token_len),
            int(text_token_len),
            int(input_len),
        )
        try:
            os.makedirs("/mnt/d/testing", exist_ok=True)
            with open("/mnt/d/testing/cosyvoice_len.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "text_len": int(text_token_len),
                        "prompt_text_len": int(prompt_text_token_len),
                        "expected_total_len": int(input_len),
                    },
                    f,
                )
        except Exception:
            pass
        device = "cpu"

        speech_feat, speech_feat_len = _extract_speech_feat(audio, self.feat_extractor, device)
        speech_token, speech_token_len = _extract_speech_token(audio, self.speech_tokenizer, device)
        embedding = _extract_spk_embedding(audio, self.campplus_session, device)

        # input_ids = input_ids
        return BatchFeature(
            {
                # "tts_prompt_text":[[prompt]], "speech_prompt_text": [[prompt_text]],
                "input_ids": input_ids,
                "input_len": [input_len],
                "text_len": [text_token_len],
                "prompt_text_token": prompt_text_token,
                "prompt_text_len": [prompt_text_token_len],
                "speech_feat": speech_feat,
                "speech_feat_len": [speech_feat_len],
                "speech_token": speech_token,
                "speech_token_len": [speech_token_len],
                "embedding": embedding,
            }
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: "BatchFeature",
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        # logger.info(f"hf_inputs: {hf_inputs}")
        # logger.info(f"hf_processor {hf_processor_mm_kwargs}")

        return {
            # "tts_prompt_text": MultiModalFieldConfig.batched("audio"),
            # "speech_prompt_text": MultiModalFieldConfig.batched("audio"),
            "input_len": MultiModalFieldConfig.batched("audio"),
            "text_len": MultiModalFieldConfig.batched("audio"),
            "prompt_text_len": MultiModalFieldConfig.batched("audio"),
            "prompt_text_token": MultiModalFieldConfig.batched("audio"),
            "speech_feat": MultiModalFieldConfig.batched("audio"),
            "speech_feat_len": MultiModalFieldConfig.batched("audio"),
            "speech_token": MultiModalFieldConfig.batched("audio"),
            "speech_token_len": MultiModalFieldConfig.batched("audio"),
            "embedding": MultiModalFieldConfig.batched("audio"),
        }

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        # Weird but here prompt_text is actual text
        # logger.info("hf_processor_implies_updates")
        # logger.info(f"text {prompt_text}")
        # logger.info(f"mm_items {mm_items}")
        # logger.info(f"hf_processor_mm_kwargs {hf_processor_mm_kwargs}")
        # logger.info(f"tokenization_kwargs {tokenization_kwargs}")
        return False

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        """
        TODO: Think if this is the correct?
        """
        # out_mm_data = out_mm_kwargs.get_data()
        # logger.info(f"mm_items {mm_items}")
        # logger.info(f"hf_processor_mm_kwargs {hf_processor_mm_kwargs}")
        # logger.info(f"out_mm_kwargs {out_mm_kwargs.get_data().keys()}")

        # def replacement(item_idx: int):
        #     return [1]

        # return [
        #     PromptReplacement(
        #         modality="audio",
        #         target=PromptIndexTargets.start(),
        #         replacement=replacement
        #     )
        # ]

        # return []

        def insertion(item_idx):
            # length = mm_items.get("audio", AudioProcessorItems).get_audio_length(item_idx)
            return [1, 1]

        return [
            PromptInsertion(
                modality="audio",
                target=PromptIndexTargets.start(),
                insertion=insertion,
            )
        ]

    def _get_data_parser(self) -> MultiModalDataParser:
        """For audio you need to define target_sr;
        so need to create this data parser to avoid those errors
        """
        return MultiModalDataParser(target_sr=16000)


class CosyVoiceDummyInputsBuilder(BaseDummyInputsBuilder[CosyVoiceMultiModalProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "Hello, this is a test of the CosyVoice system capability."

    def get_dummy_mm_data(
        self, seq_len: int, mm_counts: Mapping[str, int], mm_options: Mapping[str, BaseDummyOptions] | None = None
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio")
        max_prompt_seconds = 30
        prompt_sample_rate = 24000
        target_audio_length = max_prompt_seconds * prompt_sample_rate

        audio_overrides = mm_options.get("audio") if mm_options else None
        mm_data = {
            "audio": (
                self._get_dummy_audios(
                    length=target_audio_length,
                    num_audios=num_audios,
                    overrides=audio_overrides,
                )[0],
                24000,
            ),
        }
        # logger.info(f"mm_data: {mm_data}")
        return mm_data

    def get_dummy_processor_inputs(
        self, seq_len: int, mm_counts: Mapping[str, int], mm_options: Mapping[str, BaseDummyOptions] | None = None
    ) -> ProcessorInputs:
        inputs = super().get_dummy_processor_inputs(seq_len, mm_counts, mm_options)
        # logger.info(f"mm_counts: {mm_counts}")
        # logger.info(f"seq_len: {seq_len}")
        # logger.info(f"mm_options: {mm_options}")
        inputs.hf_processor_mm_kwargs = {"prompt_text": "Testing my voices. Why should I not?"}
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
    supports_multimodal_raw_input_only = True
    supports_multimodal = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.have_multimodal_outputs = True
        self.config = vllm_config.model_config.hf_config
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

            # Initialize acoustic features to waveform stage
            from cosyvoice.hifigan.f0_predictor import CausalConvRNNF0Predictor
            from cosyvoice.hifigan.generator import CausalHiFTGenerator

            f0_predictor = CausalConvRNNF0Predictor(
                num_class=self.config.hift["f0_predictor"]["num_class"],
                in_channels=self.config.hift["f0_predictor"]["in_channels"],
                cond_channels=self.config.hift["f0_predictor"]["cond_channels"],
            )
            self.hift = CausalHiFTGenerator(
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
            # Run hift in float32 to avoid dtype mismatches in internal ops.
            self.hift = self.hift.float()
            self.token_overlap_len = 20
            self.mel_overlap_len = int(
                self.token_overlap_len / self.chunk_aware_flow_matching_model.input_frame_rate * 22050 / 256
            )
            self.mel_window = np.hamming(2 * self.mel_overlap_len)
            self.mel_cache_len = 20
            self.source_cache_len = int(self.mel_cache_len * 256)
            self.speech_window = np.hamming(2 * self.source_cache_len)
            self.mel_overlap_dict: dict[str, torch.Tensor] = {}
            self.flow_cache_dict: dict[str, torch.Tensor] = {}
            self.hift_cache_dict: dict[str, dict[str, torch.Tensor] | None] = {}
            # self.model = self.hift
        else:
            raise ValueError(f"Stop it! {self.model_stage}")

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if self.model_stage == "text_speech_lm":
            logits = self.model.llm_decoder(hidden_states)
            return logits
        else:
            raise RuntimeError(f"embed_input_ids is only valid for {self.model_stage}.")

    def embed_multimodal(self, **kwargs: object) -> torch.Tensor:
        # logger.info(f"embed_multimodal kwargs {kwargs}")
        if self.model_stage == "text_speech_lm":
            logger.info(f"-------------------- {kwargs['speech_token']}")
            self.speech_token = kwargs["speech_token"]
            self.embedding = kwargs["embedding"]
            # self.prompt_text_token = kwargs["prompt_text_token"]
            self.speech_feat = kwargs["speech_feat"]
            # logger.info(f"tokens {self.speech_token.shape}")
            return self.speech_token
        else:
            raise RuntimeError(f"embed_input_ids is only valid for {self.model_stage}.")

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal=None,
    ) -> torch.Tensor:
        # logger.info(f"multimodal_embeddings: {multimodal_embeddings}")
        # logger.info(f"is_multimodal: {is_multimodal}")
        # logger.info(f"input_ids {input_ids}")
        if self.model_stage == "text_speech_lm":
            embed_tokens = self.model.llm.model.model.embed_tokens(input_ids)

            # TODO: This is also same random trick because I couldn't figure out how to
            # add it in processor.
            if len(input_ids) >= 2 and input_ids[0] == 1 and input_ids[1] == 1:
                sos = self.model.speech_embedding.weight[self.model.sos].reshape(1, -1)
                task_id = self.model.speech_embedding.weight[self.model.task_id].reshape(1, -1)
                embed_tokens = torch.cat([sos, embed_tokens[2:], task_id], dim=0)
            return embed_tokens
        elif self.model_stage == "chunk_aware_flow_matching":
            assert input_ids.dim() == 1
            # logger.info(f"input_ids: {input_ids} multi emb {multimodal_embeddings}")
            # logger.info(f"is_multimodal: {is_multimodal}")
            hidden = int(self.config.hidden_size)
            return torch.zeros(
                (input_ids.shape[0], hidden),
            )
        else:
            # logger.info(f"input_ids: {input_ids} multi emb {multimodal_embeddings}")
            # logger.info(f"is_multimodal: {is_multimodal}")
            raise RuntimeError(f"embed_input_ids is not valid for {self.model_stage}.")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        additional_information: dict[str, object] | None = None,
        **kwargs: object,
    ) -> OmniOutput:
        if self.model_stage == "text_speech_lm":
            # logger.info(f"Forward pass for text {self.model_stage}")
            # logger.info(f"input_ids {input_ids}")
            # logger.info(f"positions {positions} {positions.shape}")
            # logger.info(f"intermediate_tensors {intermediate_tensors}")
            # logger.info(f"inputs_embeds {inputs_embeds.shape if inputs_embeds
            # is not None else None} ")
            # logger.info(f"kwargs {kwargs.keys()}")
            # logger.info(f"kwargs {kwargs['runtime_additional_information'] if
            # 'runtime_additional_information' in kwargs else None}")
            # logger.info(f"model_stage {self.model_stage}")
            # logger.info(f"additional_information {additional_information}")
            if input_ids is not None:
                #### TODO: This is random trick I added because I didn't know how to pass it without
                # adding any prompt updates
                if len(input_ids) > 2 and input_ids.tolist()[:2] == [1, 1]:
                    input_ids = input_ids[2:]
                    inputs_embeds = self.embed_input_ids(input_ids)
                    prompt_speech_token = kwargs.get("speech_token")
                    print(f"kwargs {kwargs.keys()}")
                    input_len = kwargs.get("input_len")
                    logger.info(f"input_len {input_len.shape}")
                    prompt_text_len = kwargs.get("prompt_text_len")
                    self.min_len, self.max_len = _compute_min_max_len(
                        input_len[0],
                        prompt_text_len[0],
                        self.config.min_token_text_ratio,
                        self.config.max_token_text_ratio,
                    )

                    print(input_ids.shape)
                    pstoken = prompt_speech_token[0][0]
                    prompt_speech_token_emb = self.model.speech_embedding(pstoken)
                    print(prompt_speech_token_emb.shape)
                    print(inputs_embeds.shape)
                    logger.info(
                        f" {prompt_speech_token.shape} prompt_speech_token {prompt_speech_token} "
                        f"prompt_speech_token_emb {prompt_speech_token_emb}"
                    )
                    inputs_embeds = torch.concat([inputs_embeds, prompt_speech_token_emb], dim=0)

            if inputs_embeds is None:
                inputs_embeds = self.embed_input_ids(input_ids)
            # Ensure [B, T, C]
            if inputs_embeds.dim() == 2:
                inputs_embeds = inputs_embeds.unsqueeze(0)
            batch, seq_len, _ = inputs_embeds.shape
            seq_lens = torch.full((batch,), seq_len, dtype=torch.int32, device=inputs_embeds.device)
            hidden_states, _ = self.model.llm(inputs_embeds, seq_lens)
            # logger.info(f"hidden_states {hidden_states.shape}")
            hidden_states = hidden_states.squeeze(0)

            # logger.info(f"kwargs {kwargs}")
            multimodal_outputs = {}

            if hasattr(self, "speech_token"):
                multimodal_outputs = {
                    "speech_token": self.speech_token,
                    "embedding": self.embedding,
                    "speech_feat": self.speech_feat,
                }

                # logger.info(f"multimodal_outputs {multimodal_outputs}")

            return OmniOutput(text_hidden_states=hidden_states, multimodal_outputs=multimodal_outputs)
        elif self.model_stage == "chunk_aware_flow_matching":
            # logger.info(f"Forward pass for text {self.model_stage}")
            # logger.info(f"input_ids {input_ids}")
            # logger.info(f"positions {positions} {positions.shape}")
            # logger.info(f"intermediate_tensors {intermediate_tensors}")
            # logger.info(f"inputs_embeds {inputs_embeds.shape}")
            # logger.info(f"kwargs {kwargs.keys()}")
            # logger.info(f"model_stage {self.model_stage}")
            # logger.info(f"additional_information {additional_information}")
            from torch.nn import functional as F

            req_ids = kwargs.get("request_ids", [])

            runtime_info = kwargs.get("runtime_additional_information", [])
            # addi_by_req = kwargs.get("additional_information_by_req_id")  # prefill only

            # logger.info(f"req_ids {req_ids} runtime_info {runtime_info} addi_by_req{addi_by_req}")
            if not req_ids:
                return OmniOutput(text_hidden_states=None, multimodal_outputs=None)

            d = next(self.parameters())
            device, dtype = d.device, d.dtype
            embedding = F.normalize(runtime_info[0]["embedding"][0].to(device=device, dtype=dtype), dim=1)
            embedding = self.model.spk_embed_affine_layer(embedding)

            prompt_token = runtime_info[0]["speech_token"][0].to(device=device)
            token = input_ids.unsqueeze(0).to(device=device)
            token_len1, token_len2 = prompt_token.shape[1], token.shape[1]
            # Build length tensors for pad mask logic.
            prompt_token_len = torch.tensor([token_len1], device=token.device, dtype=torch.int32)
            token_len = torch.tensor([token_len2], device=token.device, dtype=torch.int32)
            token = torch.concat([prompt_token, token], dim=1)
            token_len = prompt_token_len + token_len
            mask = (~_make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
            vocab_size = self.model.input_embedding.num_embeddings
            token = self.model.input_embedding(torch.clamp(token, min=0, max=vocab_size - 1)) * mask

            # text encode
            prompt_feat = runtime_info[0]["speech_feat"][0]

            h = self.model.pre_lookahead_layer(token)
            h = h.repeat_interleave(self.model.token_mel_ratio, dim=1)
            mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]

            # get conditions
            conds = torch.zeros([1, mel_len1 + mel_len2, self.model.output_size], device=token.device).to(h.dtype)
            conds[:, :mel_len1] = prompt_feat
            conds = conds.transpose(1, 2)

            mask = (~_make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)
            feat, _ = self.model.decoder(
                mu=h.transpose(1, 2).contiguous(),
                mask=mask.unsqueeze(1),
                spks=embedding,
                cond=conds,
                n_timesteps=10,
                streaming=False,
            )
            feat = feat[:, :, mel_len1:]

            tts_mel = feat

            req_id = kwargs["request_ids"][0]
            if req_id not in self.mel_overlap_dict:
                self.mel_overlap_dict[req_id] = torch.zeros(1, 80, 0, device=device)
                self.flow_cache_dict[req_id] = torch.zeros(1, 80, 0, 2, device=device)
                self.hift_cache_dict[req_id] = None

            token_offset = 0
            tts_mel = tts_mel[:, :, token_offset * self.model.token_mel_ratio :]

            if self.mel_overlap_dict[req_id].shape[2] != 0:
                tts_mel = _fade_in_out(tts_mel, self.mel_overlap_dict[req_id], self.mel_window)

            hift_cache = self.hift_cache_dict[req_id]
            if hift_cache is not None:
                hift_cache_mel = hift_cache["mel"]
                tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
                self.hift_cache_dict[req_id]["mel"] = tts_mel
            else:
                self.hift_cache_dict[req_id] = {"mel": tts_mel, "speech_offset": 0}

            # TODO Add speed control later
            # if speed != 1.0:
            #     tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode="linear")
            hift_weight = self.hift.m_source.l_linear.weight
            tts_mel = tts_mel.to(device=hift_weight.device, dtype=hift_weight.dtype)
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel)
            tts_speech = tts_speech[:, self.hift_cache_dict[req_id]["speech_offset"] :]
            self.hift_cache_dict[req_id]["speech_offset"] += tts_speech.shape[1]

            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"audio": tts_speech},
            )
        else:
            raise ValueError(f"Stop it! {input_ids}")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # logger.info(f"Loading weights for CosyVoice model stage: {self.model_stage} {weights}")
        # loaded_names = {name for name, _ in self.named_parameters()}
        if self.model_stage == "text_speech_lm":
            # Load weights for text to speech LM stage
            llm_weight_path = os.path.join(self.model_dir, "llm.pt")
            # logger.info(f"Loading LLM weights from {llm_weight_path}")
            device = next(self.parameters()).device
            self.model.load_state_dict(torch.load(llm_weight_path, map_location=device), strict=True)
            self.model.to(device).eval()
        elif self.model_stage == "chunk_aware_flow_matching":
            # Load weights for chunk aware flow matching stage
            flow_weight_path = os.path.join(self.model_dir, "flow.pt")
            # logger.info(f"Loading Flow weights from {flow_weight_path}")
            device = next(self.parameters()).device
            self.model.load_state_dict(torch.load(flow_weight_path, map_location=device), strict=True)
            self.model.to(device).eval()
        else:
            raise ValueError("Stop it!")
