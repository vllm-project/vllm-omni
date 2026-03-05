from typing import Optional, List, Any

from dataclasses import field, dataclass
from omegaconf import MISSING

from nemo.collections.asr.models.configs.common_config import ConvBlock
from nemo.collections.asr.models.spec2vec.spec2vec_config import FeatureEncoderConfig, ProjectorConfig, \
    NoisePerturbConfig
from nemo.collections.asr.models.st2vec.st2vec_config import ShiftPerturbConfig
from nemo.collections.asr.models.wav2vec.wav2vec_config import LossConfig, Wav2VecTransformerConfig, \
    Wav2VecMaskingConfig, QuantizerConfig
from nemo.collections.asr.modules.audio_preprocessing import AudioToMelSpectrogramPreprocessorConfig
from nemo.core.config.modelPT import ModelConfig


@dataclass
class StyleFusionConfig:
    # content_dim: int = MISSING
    # style_dim: int = MISSING
    content_upsampling_method: str = 'conv_proj'
    content_upsampling_layers: List[Any] = None
    fusion_method: str = 'sum'


@dataclass
class DecoderConfig:
    content_dim: int = MISSING
    style_dim: int = MISSING
    style_fusion: StyleFusionConfig = MISSING
    conv_block: ConvBlock = MISSING
    content_quantizer: Optional[QuantizerConfig] = None
    style_quantizer: Optional[QuantizerConfig] = None
    use_tf_pad: bool = False
    ln_eps: float = 1e-5


@dataclass
class SPIRALConfig:
    target_shifting: Optional[ShiftPerturbConfig] = None
    target_compute_perturb: bool = False

    target_momentum: float = 0.99
    target_momentum_final: Optional[float] = None
    target_momentum_steps: Optional[int] = None
    target_momentum_type: Optional[str] = None

    projector: Optional[ProjectorConfig] = None
    predictor: Optional[ProjectorConfig] = None

    n_negatives: int = field(
        default=100, metadata={'help': 'Number of negatives to sample from the same audio sample'}
    )
    cross_sample_negatives: int = field(
        default=0, metadata={'help': 'Number of negatives to sample from any sample in the batch'}
    )
    codebook_negatives: int = field(default=0, metadata={'help': 'Number of negative examples in codebook'})
    negatives_from_everywhere: bool = field(
        default=False, metadata={'help': 'Sample negatives from everywhere, not just masked states'}
    )
    negatives_from_noisy_features: bool = False


@dataclass
class ConvStyleEncoder(ConvBlock):
    _target_: str = 'nemo.collections.asr.parts.convolution_layers.ConvBlock'


@dataclass
class SPIRAL2PretrainConfig(ModelConfig):
    preprocessor: AudioToMelSpectrogramPreprocessorConfig = MISSING

    content_encoder: FeatureEncoderConfig = field(default_factory=FeatureEncoderConfig)
    # pretrained_encoder_path: Optional[str] = None
    decoder: Optional[DecoderConfig] = None
    freeze_feature_encoder: bool = False
    noise_mix_ratio: Optional[float] = None

    content_masking: Optional[Wav2VecMaskingConfig] = None

    style_encoder: Optional[Any] = None

    style_masking: Optional[Wav2VecMaskingConfig] = None

    noise_perturb: Optional[NoisePerturbConfig] = None

    recon_loss_type: str = 'l2'
    recon_time_mask_only: bool = False

    content_quant_ppl_loss_weight: float = 0.0
    style_quant_ppl_loss_weight: float = 0.0

    spiral_config: Optional[SPIRALConfig] = None
    spiral_loss_weight: float = 0.0
    spiral_loss_type: str = 'wav2vec'
    spiral_logit_temp: float = field(default=0.1, metadata={'help': 'Temperature to divide logits by'})
    # loss: LossConfig = LossConfig()

    expected_gpu_num: int = 1
