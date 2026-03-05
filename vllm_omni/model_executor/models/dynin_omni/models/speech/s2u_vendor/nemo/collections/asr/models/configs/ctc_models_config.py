# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Any, List, Optional

from omegaconf import MISSING

from nemo.collections.asr.models.configs.common_config import Tokenizer, DatasetConfig, OptimConfig, Conv1dNormAct, \
    ProjUpsampling
from nemo.collections.asr.models.spec2vec.spec2vec_config import ProjectorConfig
from nemo.collections.asr.modules.audio_preprocessing import (
    AudioToMelSpectrogramPreprocessorConfig,
    SpectrogramAugmentationConfig,
)
from nemo.core.config import modelPT as model_cfg


@dataclass
class ConvASRDecoderConfig:
    _target_: str = 'nemo.collections.asr.modules.ConvASRDecoder'
    feat_in: int = MISSING
    num_classes: int = 0
    proj_upsampling: Optional[ProjUpsampling] = None
    conv_layers: Optional[List[Conv1dNormAct]] = None
    projector: Optional[ProjectorConfig] = None
    use_conv_mask: bool = True
    use_tf_pad: bool = True
    ln_eps: float = 1e-5
    blank_pos: str = 'after_vocab_last'
    init_mode: str = 'xavier_uniform'
    vocabulary: Optional[List[str]] = field(default_factory=list)


@dataclass
class EncDecCTCConfig(model_cfg.ModelConfig):
    # Model global arguments
    sample_rate: int = 16000
    labels: List[str] = MISSING
    tokenizer: Optional[Tokenizer] = None

    # Dataset configs
    train_ds: DatasetConfig = MISSING
    validation_ds: DatasetConfig = MISSING
    test_ds: DatasetConfig = MISSING

    # Optimizer / Scheduler config
    optim: OptimConfig = MISSING

    # Model component configs
    preprocessor: AudioToMelSpectrogramPreprocessorConfig = MISSING
    spec_augment: Optional[SpectrogramAugmentationConfig] = MISSING
    encoder: Any = MISSING
    decoder: Any = MISSING


@dataclass
class EncDecCTCModelConfig(model_cfg.ModelPTConfig):
    model: EncDecCTCConfig = field(default_factory=EncDecCTCConfig)
