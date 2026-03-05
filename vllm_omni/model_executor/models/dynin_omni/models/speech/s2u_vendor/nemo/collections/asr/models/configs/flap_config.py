from dataclasses import dataclass, field
from typing import List, Optional, Any

from omegaconf import MISSING

import nemo
from nemo.collections.asr.models.configs.common_config import TextEmbed, ConvBlock
from nemo.collections.asr.models.configs.diffusion_config import SeqUNet, Diffusion, Sampler
from nemo.collections.asr.models.wav2vec.wav2vec_config import Wav2VecMaskingConfig
from nemo.collections.asr.modules.audio_preprocessing import (
    AudioToMelSpectrogramPreprocessorConfig,
    SpectrogramAugmentationConfig,
)
from nemo.core.config import modelPT as model_cfg

from .common_config import *
from .transducer_config import *
from .conv_transformer_config import *


@dataclass
class Text2TextDatasetConfig(nemo.core.classes.dataset.DatasetConfig):
    manifest_dir: str = MISSING
    manifest_filepath: str = MISSING

    tokenizer: str = MISSING

    parse_online: bool = False
    mask_id: int = MISSING
    mask_prob: float = MISSING
    word_mask: bool = False
    space_id: int = MISSING
    space_num_min: int = MISSING
    space_num_max: int = MISSING
    in_word_space_num_min: int = 0
    in_word_space_num_max: int = 0
    replace_prob: float = 0.0
    replace_ids: Optional[List[int]] = None
    min_text_len: int = 1
    max_text_len: int = 10240
    max_crop_words: int = 0
    max_crop_chars: int = 0
    file_format: str = 'txt'
    encoding: str = 'utf-8'


@dataclass
class UpsamplingConfig:
    variance: float = 1.0

@dataclass
class TextConvTransformerEncoder:
    _target_: str = 'nemo.collections.asr.modules.TextConvTransformerEncoder'
    text_embed: TextEmbed = MISSING
    encoder: ConvTransformerBlock = MISSING
    use_conv_mask: bool = MISSING
    use_tf_pad: bool = True
    ln_eps: float = 1e-5
    init_mode: str = 'xavier_uniform'
    bias_init_mode: str = 'zero'
    embedding_init_mode: str = 'xavier_uniform'
    pad_to: int = 0
    pad_value: float = 0.0
    upsampling_args: Optional[UpsamplingConfig] = None


@dataclass
class LatentAutoencoder:
    encoder: ConvBlock = MISSING
    decoder: ConvBlock = MISSING
    loss_type: str = 'l2'


@dataclass
class TextUpsampling:
    min_ratio: float = MISSING
    max_ratio: float = MISSING
    max_len: int = 0


@dataclass
class DiffusionTextProjector:
    dpm_type: str = 'ddpm'
    diffusion: Any = MISSING
    conditioning_key: Optional[str] = None
    condition_dropout: float = 0.0
    latent_denoiser_type: str = 'unet'
    latent_denoiser: Any = MISSING
    text_encoder: TextConvTransformerEncoder = MISSING
    sampler: Sampler = field(default_factory=Sampler)
    text_upsampling: Optional[TextUpsampling] = None
    latent_autoencoder: Optional[LatentAutoencoder] = None


@dataclass
class FLAPTextProjectorTransducerModel(model_cfg.ModelConfig):
    labels: List[str] = MISSING
    tokenizer: Optional[Tokenizer] = None
    input_tokenizer: Optional[Tokenizer] = None

    train_ds: Text2TextDatasetConfig = MISSING
    validation_ds: Text2TextDatasetConfig = MISSING
    test_ds: Text2TextDatasetConfig = MISSING

    expected_gpu_num: int = MISSING
    optim: Optional[OptimConfig] = MISSING

    text_projector: TextConvTransformerEncoder = MISSING

    pretrained_chkpt: str = MISSING
    pretrained_chkpt_converter: str = MISSING

    speech_pre_encoder: ConvTransformerEncoder = MISSING
    speech_backbone_encoder: ConvTransformerEncoder = MISSING
    decoder: TransformerTDecoder = MISSING
    joint: RNNTJoint = MISSING
    model_defaults: ModelDefaults = MISSING

    decoding: Decoding = MISSING


@dataclass
class FLAPTextProjectorTransducerConfig(model_cfg.ModelPTConfig):
    model: FLAPTextProjectorTransducerModel = field(default_factory=FLAPTextProjectorTransducerModel)


@dataclass
class FLAPLatentMaskConfig:
    mask_emb_dim: int = MISSING
    text_masking: Optional[Wav2VecMaskingConfig] = None
    speech_masking: Optional[Wav2VecMaskingConfig] = None


@dataclass
class FLAPMultiTaskTransducerModel(model_cfg.ModelConfig):
    labels: List[str] = MISSING
    tokenizer: Optional[Tokenizer] = None
    input_tokenizer: Optional[Tokenizer] = None

    speech2text_train_ds: DatasetConfig = MISSING
    speech2text_validation_ds: DatasetConfig = MISSING
    speech2text_test_ds: DatasetConfig = MISSING

    text2text_train_ds: Text2TextDatasetConfig = MISSING
    text2text_validation_ds: Text2TextDatasetConfig = MISSING
    text2text_test_ds: Text2TextDatasetConfig = MISSING

    pretrained_chkpt: Optional[str] = None

    expected_gpu_num: int = MISSING
    optim: Optional[OptimConfig] = MISSING

    preprocessor: AudioToMelSpectrogramPreprocessorConfig = MISSING
    spec_augment: Optional[SpectrogramAugmentationConfig] = MISSING
    latent_masking: Optional[FLAPLatentMaskConfig] = None

    freeze_speech_pre_encoder: bool = True
    freeze_text_projector: bool = True

    freeze_speech_backbone_transducer: bool = False
    freeze_auxiliary_decoder: bool = False

    text_projector: Any = MISSING
    speech_pre_encoder: ConvTransformerEncoder = MISSING
    speech_backbone_encoder: ConvTransformerEncoder = MISSING
    decoder: TransformerTDecoder = MISSING
    joint: RNNTJoint = MISSING
    model_defaults: ModelDefaults = MISSING

    auxiliary_decoder: Any = None
    freeze_text_auxiliary_decoder: bool = True

    speech2text_loss_weight: float = MISSING
    text2text_loss_weight: float = MISSING
    text2text_grad_accum_batches: int = 1
    auxiliary_loss_weight: float = 0.0

    decoding: Decoding = MISSING


@dataclass
class FLAPMultiTaskTransducerConfig(model_cfg.ModelPTConfig):
    model: FLAPMultiTaskTransducerModel = field(default_factory=FLAPMultiTaskTransducerModel)
