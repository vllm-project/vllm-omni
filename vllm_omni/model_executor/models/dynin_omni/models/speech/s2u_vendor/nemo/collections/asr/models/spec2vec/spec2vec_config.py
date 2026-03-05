from typing import Optional, List, Any

from dataclasses import field, dataclass
from omegaconf import MISSING

from nemo.collections.asr.models.configs.common_config import Conv2dBlock, Conv1dNormAct, DatasetConfig, Tokenizer
from nemo.collections.asr.models.wav2vec.wav2vec_config import LossConfig, QuantizerConfig, Wav2VecTransformerConfig, \
    Wav2VecMaskingConfig
from nemo.collections.asr.modules.audio_preprocessing import AudioToMelSpectrogramPreprocessorConfig
from nemo.core.config.modelPT import ModelConfig


@dataclass
class ConvTransformerBlock:
    conv_layers: List[Conv1dNormAct] = MISSING
    transformer_block: Optional[Wav2VecTransformerConfig] = None


@dataclass
class FeatureEncoderConfig:
    _target_: str = 'nemo.collections.asr.parts.spec2vec.FeatureEncoder'
    feat_in: int = MISSING
    use_conv_mask: bool = MISSING
    conv2d_block: Optional[Conv2dBlock] = MISSING
    conv_transformer_blocks: List[ConvTransformerBlock] = MISSING
    use_tf_pad: bool = True
    ln_eps: float = 1e-5


@dataclass
class ProjectorConfig:
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None
    use_conv_mask: bool = True
    use_tf_pad: bool = True
    ln_eps: float = 1e-5
    conv_layers: Optional[List[Conv1dNormAct]] = None
    transformer: Optional[Wav2VecTransformerConfig] = None


@dataclass
class Spec2VecEncoderConfig:
    noisy_spec2vec: bool = False

    preprocessor: AudioToMelSpectrogramPreprocessorConfig = MISSING

    quantizer: QuantizerConfig = field(default_factory=QuantizerConfig)
    feature_encoder: FeatureEncoderConfig = field(default_factory=FeatureEncoderConfig)
    freeze_feature_encoder: bool = False
    targets_grad_update_inverval: int = 1
    transformer_encoder: Wav2VecTransformerConfig = field(default_factory=Wav2VecTransformerConfig)
    masking: Optional[Wav2VecMaskingConfig] = None
    learnable_mask: bool = True

    dropout_input: float = field(default=0.1, metadata={'help': 'Dropout applied to input raw features'})
    dropout_features: float = field(
        default=0.1, metadata={'help': 'Dropout applied to the features generator by convolutions'}
    )
    final_hidden_dim: Optional[int] = None
    dropout_final: float = 0.1
    final_dim: int = field(default=0, metadata={'help': 'Project final representations and targets to this dimension'})
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


@dataclass
class Spec2VecPretrainModelConfig(ModelConfig):
    spec2vec_encoder: Spec2VecEncoderConfig = MISSING

    logit_temp: float = field(default=0.1, metadata={'help': 'Temperature to divide logits by'})
    loss: LossConfig = field(default_factory=LossConfig)

    expected_gpu_num: int = 1


@dataclass
class NoisePerturbConfig:
    manifest_path: List[str]
    min_snr_db: float
    max_snr_db: float
    max_gain_db: float = 300.0
    ratio: float = 1.0
    target_sr: int = 16000
    data_dir: str = ''
    cache_noise: bool = False


@dataclass
class Spec2VecCTCFinetuneModelConfig(ModelConfig):
    pretrain_chkpt_path: Optional[str] = MISSING

    encoder_type: str = 'spec2vec'
    encoder: Any = MISSING
    decoder: Any = MISSING

    labels: Optional[List[str]] = None
    tokenizer: Optional[Tokenizer] = None
    add_end_space: bool = False
    lang: str = 'en'

    freeze_finetune_updates: int = 0

    noise_perturb: Optional[NoisePerturbConfig] = None

    # Dataset configs
    train_ds: DatasetConfig = MISSING
    validation_ds: DatasetConfig = MISSING
    test_ds: DatasetConfig = MISSING

    expected_gpu_num: int = MISSING


@dataclass
class Wav2VecCTCFinetuneModelConfig(Spec2VecCTCFinetuneModelConfig):
    encoder_type: str = 'wav2vec'


@dataclass
class ST2VecCTCFinetuneModelConfig(Spec2VecCTCFinetuneModelConfig):
    label_type: Optional[str] = None  # ['char', 'phone','unit', 'bpe']
    encoder_type: str = 'st'
    use_teacher_encoder: bool = False


@dataclass
class ST2VecVQCTCFinetuneModelConfig(Spec2VecCTCFinetuneModelConfig):
    label_type: str = None  # ['char', 'phone','bpe']
    encoder_type: str = 'st'
    quantizer: QuantizerConfig = MISSING
    quant_ppl_loss_weight: float = MISSING
    use_teacher_encoder: bool = False


@dataclass
class FeatST2VecCTCFinetuneModelConfig(Spec2VecCTCFinetuneModelConfig):
    encoder_type: str = 'feat_st'
