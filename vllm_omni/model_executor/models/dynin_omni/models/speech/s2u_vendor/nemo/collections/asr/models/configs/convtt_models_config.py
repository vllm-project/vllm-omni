from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import MISSING

from nemo.collections.asr.modules.audio_preprocessing import (
    AudioToMelSpectrogramPreprocessorConfig,
    SpectrogramAugmentationConfig,
)
from nemo.core.config import modelPT as model_cfg

from .common_config import *
from .transducer_config import *
from .conv_transformer_config import *


@dataclass
class ConvTTModel(model_cfg.ModelConfig):
    labels: List[str] = MISSING
    tokenizer: Optional[Tokenizer] = None

    train_ds: DatasetConfig = MISSING
    validation_ds: DatasetConfig = MISSING
    test_ds: DatasetConfig = MISSING

    expected_gpu_num: int = MISSING
    optim: Optional[OptimConfig] = MISSING

    preprocessor: AudioToMelSpectrogramPreprocessorConfig = MISSING
    spec_augment: Optional[SpectrogramAugmentationConfig] = MISSING

    encoder: ConvTransformerEncoder = MISSING
    decoder: TransformerTDecoder = MISSING
    joint: RNNTJoint = MISSING
    model_defaults: ModelDefaults = MISSING

    decoding: Decoding = MISSING


@dataclass
class ConvTTConfig(model_cfg.ModelPTConfig):
    model: ConvTTModel = field(default_factory=ConvTTModel)
