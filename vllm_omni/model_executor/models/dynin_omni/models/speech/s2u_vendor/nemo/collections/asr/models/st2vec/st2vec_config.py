from typing import Optional, Any

from dataclasses import field, dataclass
from omegaconf import MISSING

from nemo.collections.asr.models.spec2vec.spec2vec_config import FeatureEncoderConfig, ProjectorConfig, \
    NoisePerturbConfig
from nemo.collections.asr.models.wav2vec.wav2vec_config import LossConfig, Wav2VecTransformerConfig, \
    Wav2VecMaskingConfig, QuantizerConfig
from nemo.collections.asr.modules.audio_preprocessing import AudioToMelSpectrogramPreprocessorConfig
from nemo.core.config.modelPT import ModelConfig


@dataclass
class ShiftPerturbConfig:
    dist: str = 'uniform'
    shift_prob: float = MISSING
    max_ratio: float = 0.5
    unit: int = MISSING
    max: Optional[int] = None
    min: Optional[int] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    truncate: bool = True


@dataclass
class PitchEstimationTask:
    pitch_estimator: Optional[ProjectorConfig] = None
    sample_rate: int = 16000
    pitch_min: int = 80
    pitch_max: int = 400
    reduction_rate: int = MISSING
    loss_type: str = 'l2'


@dataclass
class ReconstructionTask:
    quantizer: QuantizerConfig = MISSING
    reconstructor: ProjectorConfig = MISSING
    global_cond_extractor: Optional[ProjectorConfig] = None
    reduction_rate: int = MISSING
    use_teacher_feat_prob: float = 0.0
    loss_type: str = 'l2'


@dataclass
class ST2VecEncoderConfig:
    preprocessor: AudioToMelSpectrogramPreprocessorConfig = MISSING

    feature_encoder: FeatureEncoderConfig = field(default_factory=FeatureEncoderConfig)
    pretrained_encoder_path: Optional[str] = None
    freeze_feature_encoder: bool = False
    freeze_student: bool = False
    noise_mix_ratio: Optional[float] = None
    masking: Optional[Wav2VecMaskingConfig] = None
    shifting: Optional[ShiftPerturbConfig] = None
    target_shifting: Optional[ShiftPerturbConfig] = None
    target_masking: Optional[Wav2VecMaskingConfig] = None
    target_compute_perturb: bool = False

    target_momentum: float = 0.99
    target_momentum_final: Optional[float] = None
    target_momentum_steps: Optional[int] = None
    target_momentum_type: Optional[str] = None
    projector: Optional[ProjectorConfig] = None
    predictor: Optional[ProjectorConfig] = None

    quantizer: Optional[QuantizerConfig] = None

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

    pitch_estimation_task: Optional[PitchEstimationTask] = None
    pitch_loss_weight: float = 0.0

    reconstruction_task: Optional[ReconstructionTask] = None
    reconstruction_loss_weight: float = 0.0
    reconstruction_quant_ppl_loss_weight: float = 0.0

@dataclass
class FeatST2VecEncoderConfig:
    preprocessor: AudioToMelSpectrogramPreprocessorConfig = MISSING

    feature_encoder: FeatureEncoderConfig = field(default_factory=FeatureEncoderConfig)
    context_net: Wav2VecTransformerConfig = MISSING
    masking: Optional[Wav2VecMaskingConfig] = None
    target_masking: Optional[Wav2VecMaskingConfig] = None

    target_momentum: float = 0.99
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
class ST2VecPretrainModelConfig(ModelConfig):
    encoder_type: str = 'st'
    st2vec_encoder: Any = MISSING

    noise_perturb: Optional[NoisePerturbConfig] = None

    loss_type: str = 'wav2vec'
    logit_temp: float = field(default=0.1, metadata={'help': 'Temperature to divide logits by'})
    loss: LossConfig = field(default_factory=LossConfig)

    expected_gpu_num: int = 1
