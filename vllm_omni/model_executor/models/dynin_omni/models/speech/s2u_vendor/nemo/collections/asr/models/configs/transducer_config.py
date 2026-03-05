from dataclasses import dataclass
from typing import Any, Optional

from omegaconf import MISSING

from nemo.collections.asr.models.configs.conv_transformer_config import RelTransformerBlock

__all__ = ['TransformerTDecoder', 'JointNet', 'RNNTJoint', 'Greedy', 'Beam', 'Decoding', 'ModelDefaults']

@dataclass
class TransformerTDecoder:
    _target_: str = 'nemo.collections.asr.modules.TransformerTDecoder'
    vocab_size: int = MISSING
    embed_size: int = MISSING
    embed_dropout: float = MISSING
    embed_proj_size: int = MISSING
    norm_embed: bool = False
    norm_embed_proj: bool = False
    sos_idx: Optional[int] = MISSING
    transformer_block: RelTransformerBlock = MISSING
    blank_pos: str = MISSING
    blank_as_pad: bool = True
    mask_label_prob: float = 0.0
    mask_label_id: Optional[int] = None
    ln_eps: float = 1e-5
    init_mode: str = 'xavier_uniform'
    bias_init_mode: str = 'zero'
    embedding_init_mode = 'xavier_uniform'


@dataclass
class JointNet:
    joint_hidden: int = MISSING
    activation: str = 'relu'
    dropout: float = 0.0
    single_bias: bool = True
    encoder_hidden: int = MISSING
    pred_hidden: int = MISSING


@dataclass
class RNNTJoint:
    _target_: str = 'nemo.collections.asr.modules.RNNTJoint'

    jointnet: JointNet = MISSING

    blank_pos: str = MISSING

    log_softmax: Any = None

    experimental_fuse_loss_wer: bool = False
    fused_batch_size: int = 1

    init_mode: str = 'xavier_uniform'
    bias_init_mode: str = 'zero'


@dataclass
class Greedy:
    max_symbols: int = MISSING


@dataclass
class Beam:
    beam_size: int = MISSING
    score_norm: bool = True
    return_best_hypothesis: bool = True

    tsd_max_sym_exp_per_step: Optional[int] = 50
    alsd_max_target_len: Any = 1.0

    nsc_max_timesteps_expansion: int = 1
    nsc_prefix_alpha: int = 1

    beam_temperature: float = 1.0
    beam_combine_path: bool = True
    beam_max_exp_step: int = 4
    beam_prune_exp: bool = True
    beam_prune_exp_full: bool = True
    beam_word_reward_ratio: float = 0.0


@dataclass
class Decoding:
    strategy: str = MISSING
    greedy: Optional[Greedy] = None
    beam: Optional[Beam] = None


@dataclass
class ModelDefaults:
    enc_hidden: int = MISSING
    pred_hidden: int = MISSING
