# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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

from nemo.collections.asr.models.configs.common_config import AdamWParams, DatasetConfig, Conv1dNormAct, \
    PolynomialHoldDecayAnnealingParams, ProjUpsampling, Tokenizer
from nemo.collections.asr.models.configs.ctc_models_config import ConvASRDecoderConfig
from nemo.collections.asr.models.wav2vec.wav2vec_config import QuantizerConfig
from nemo.core.config import TrainerConfig
from nemo.core.config.modelPT import ModelPTConfig
from nemo.utils.exp_manager import ExpManagerConfig, CallbackParams
from nemo.collections.asr.models.spec2vec.spec2vec_config import ST2VecVQCTCFinetuneModelConfig

config_name = 'vq_ctc_finetune'

sample_rate = 16000
num_features = 128

model = ST2VecVQCTCFinetuneModelConfig()

# # English labels
# LABELS = ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2',
#           'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2', 'F',
#           'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2',
#           'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y',
#           'Z', 'ZH']
# # tonal  phone number: 69

# # Chinese labels
# LABELS = ['a1', 'a2', 'a3', 'a4', 'a5', 'ai1', 'ai2', 'ai3', 'ai4', 'an1', 'an2', 'an3', 'an4', 'ang1', 'ang2', 'ang3',
#           'ang4', 'ao1', 'ao2', 'ao3', 'ao4', 'b', 'c', 'ch', 'd', 'e1', 'e2', 'e3', 'e4', 'e5', 'ei1', 'ei2', 'ei3',
#           'ei4', 'en1', 'en2', 'en3', 'en4', 'en5', 'eng1', 'eng2', 'eng3', 'eng4', 'f', 'g', 'h', 'i1', 'i2', 'i3',
#           'i4', 'ia1', 'ia2', 'ia3', 'ia4', 'ian1', 'ian2', 'ian3', 'ian4', 'iang1', 'iang2', 'iang3', 'iang4', 'iao1',
#           'iao2', 'iao3', 'iao4', 'ie1', 'ie2', 'ie3', 'ie4', 'ii1', 'ii2', 'ii3', 'ii4', 'ii5', 'iii1', 'iii2', 'iii3',
#           'iii4', 'in1', 'in2', 'in3', 'in4', 'ing1', 'ing2', 'ing3', 'ing4', 'iong1', 'iong2', 'iong3', 'iong4',
#           'iou1', 'iou2', 'iou3', 'iou4', 'j', 'k', 'l', 'm', 'n', 'o1', 'o2', 'o3', 'o4', 'ong1', 'ong2', 'ong3',
#           'ong4', 'ou1', 'ou2', 'ou3', 'ou4', 'p', 'q', 'r', 'rr', 's', 'sh', 't', 'u1', 'u2', 'u3', 'u4', 'ua1', 'ua2',
#           'ua3', 'ua4', 'uai1', 'uai2', 'uai3', 'uai4', 'uan1', 'uan2', 'uan3', 'uan4', 'uang1', 'uang2', 'uang3',
#           'uang4', 'ui1', 'ui2', 'ui3', 'ui4', 'un1', 'un2', 'un3', 'un4', 'uo1', 'uo2', 'uo3', 'uo4', 'uo5', 'v1',
#           'v2', 'v3', 'v4', 'van1', 'van2', 'van3', 'van4', 've1', 've2', 've3', 've4', 'vn1', 'vn2', 'vn3', 'vn4', 'x',
#           'z', 'zh']
# # tonal  phone number: 171

LABELS = ['en_AA0', 'en_AA1', 'en_AA2', 'en_AE0', 'en_AE1', 'en_AE2', 'en_AH0', 'en_AH1', 'en_AH2', 'en_AO0', 'en_AO1',
          'en_AO2', 'en_AW0', 'en_AW1', 'en_AW2', 'en_AY0', 'en_AY1', 'en_AY2', 'en_B', 'en_CH', 'en_D', 'en_DH',
          'en_EH0',
          'en_EH1', 'en_EH2', 'en_ER0', 'en_ER1', 'en_ER2', 'en_EY0', 'en_EY1', 'en_EY2', 'en_F', 'en_G', 'en_HH',
          'en_IH0',
          'en_IH1', 'en_IH2', 'en_IY0', 'en_IY1', 'en_IY2', 'en_JH', 'en_K', 'en_L', 'en_M', 'en_N', 'en_NG', 'en_OW0',
          'en_OW1', 'en_OW2', 'en_OY0', 'en_OY1', 'en_OY2', 'en_P', 'en_R', 'en_S', 'en_SH', 'en_T', 'en_TH', 'en_UH0',
          'en_UH1', 'en_UH2', 'en_UW0', 'en_UW1', 'en_UW2', 'en_V', 'en_W', 'en_Y', 'en_Z', 'en_ZH',
          'CN_a1', 'CN_a2', 'CN_a3', 'CN_a4', 'CN_a5', 'CN_ai1', 'CN_ai2', 'CN_ai3', 'CN_ai4', 'CN_an1', 'CN_an2',
          'CN_an3',
          'CN_an4', 'CN_ang1', 'CN_ang2', 'CN_ang3', 'CN_ang4', 'CN_ao1', 'CN_ao2', 'CN_ao3', 'CN_ao4', 'CN_b', 'CN_c',
          'CN_ch', 'CN_d', 'CN_e1', 'CN_e2', 'CN_e3', 'CN_e4', 'CN_e5', 'CN_ei1', 'CN_ei2', 'CN_ei3', 'CN_ei4',
          'CN_en1',
          'CN_en2', 'CN_en3', 'CN_en4', 'CN_en5', 'CN_eng1', 'CN_eng2', 'CN_eng3', 'CN_eng4', 'CN_f', 'CN_g', 'CN_h',
          'CN_i1', 'CN_i2', 'CN_i3', 'CN_i4', 'CN_ia1', 'CN_ia2', 'CN_ia3', 'CN_ia4', 'CN_ian1', 'CN_ian2', 'CN_ian3',
          'CN_ian4', 'CN_iang1', 'CN_iang2', 'CN_iang3', 'CN_iang4', 'CN_iao1', 'CN_iao2', 'CN_iao3', 'CN_iao4',
          'CN_ie1',
          'CN_ie2', 'CN_ie3', 'CN_ie4', 'CN_ii1', 'CN_ii2', 'CN_ii3', 'CN_ii4', 'CN_ii5', 'CN_iii1', 'CN_iii2',
          'CN_iii3',
          'CN_iii4', 'CN_in1', 'CN_in2', 'CN_in3', 'CN_in4', 'CN_ing1', 'CN_ing2', 'CN_ing3', 'CN_ing4', 'CN_iong1',
          'CN_iong2', 'CN_iong3', 'CN_iong4', 'CN_iou1', 'CN_iou2', 'CN_iou3', 'CN_iou4', 'CN_j', 'CN_k', 'CN_l',
          'CN_m',
          'CN_n', 'CN_o1', 'CN_o2', 'CN_o3', 'CN_o4', 'CN_ong1', 'CN_ong2', 'CN_ong3', 'CN_ong4', 'CN_ou1', 'CN_ou2',
          'CN_ou3', 'CN_ou4', 'CN_p', 'CN_q', 'CN_r', 'CN_rr', 'CN_s', 'CN_sh', 'CN_t', 'CN_u1', 'CN_u2', 'CN_u3',
          'CN_u4',
          'CN_ua1', 'CN_ua2', 'CN_ua3', 'CN_ua4', 'CN_uai1', 'CN_uai2', 'CN_uai3', 'CN_uai4', 'CN_uan1', 'CN_uan2',
          'CN_uan3',
          'CN_uan4', 'CN_uang1', 'CN_uang2', 'CN_uang3', 'CN_uang4', 'CN_ui1', 'CN_ui2', 'CN_ui3', 'CN_ui4', 'CN_un1',
          'CN_un2', 'CN_un3', 'CN_un4', 'CN_uo1', 'CN_uo2', 'CN_uo3', 'CN_uo4', 'CN_uo5', 'CN_v1', 'CN_v2', 'CN_v3',
          'CN_v4',
          'CN_van1', 'CN_van2', 'CN_van3', 'CN_van4', 'CN_ve1', 'CN_ve2', 'CN_ve3', 'CN_ve4', 'CN_vn1', 'CN_vn2',
          'CN_vn3',
          'CN_vn4', 'CN_x', 'CN_z', 'CN_zh']
# tonal  phone number: 69 + 171 = 240


model.labels = LABELS
model.label_type = 'phone'  # one of ['char', 'phone','bpe']
model.add_end_space = False
model.tokenizer = None  # if tokenizer is not None, use BPE

from my_conf.st2vec_lfr_pretrain_maskp5cp4gaus_tp3_tgtshift16_preln_lr3e3_40ms_fp16_init80ms_multilingual_2 import \
    st2vec_encoder

encoder = st2vec_encoder
encoder.masking.mask_prob = 0  # actually no mask
encoder.masking.mask_channel_prob = 0

transformer0 = encoder.feature_encoder.conv_transformer_blocks[-2].transformer_block
transformer0.encoder.activation_dropout = 0.1
transformer0.encoder.dropout = 0.1
transformer = encoder.feature_encoder.conv_transformer_blocks[-1].transformer_block
transformer.encoder.encoder_layerdrop = 0.1
transformer.encoder.activation_dropout = 0.1
transformer.encoder.dropout = 0.1

model.encoder = encoder

# encoder->quantizer->decoder->CTC

# For finite scalar quantizer
# Quantization level of each dimension
quant_level_per_dim = [8, 8, 8, 8]
model.quantizer = QuantizerConfig(
    levels=quant_level_per_dim,
    l2_norm=False,
    batch_norm=False,
)

# enc_output_dim = transformer.encoder.embedding_dim # 768

# feat_in=quantizer_latent_dim,
model.decoder = ConvASRDecoderConfig(
    feat_in=len(quant_level_per_dim),
    # proj_upsampling=ProjUpsampling(rate=1, filters=512, kernel_size=(5,), norm_type='ln', act_func='relu', dropout=0.1),
    conv_layers=[Conv1dNormAct(filters=512, kernel_size=(5,), stride=(1,),
                               norm_type=None, dropout=0.1,
                               act_func='relu'),
                 Conv1dNormAct(filters=512, kernel_size=(5,), stride=(1,),
                               norm_type=None, dropout=0.1,
                               act_func='relu'),
                 Conv1dNormAct(filters=512, kernel_size=(5,), stride=(1,),
                               norm_type=None, dropout=0.1,
                               act_func='relu'),
                 Conv1dNormAct(filters=512, kernel_size=(5,), stride=(1,),
                               norm_type=None, dropout=0.1,
                               act_func='relu'),
                 ],
    vocabulary=LABELS,
    blank_pos='after_vocab_last'
)

model.quant_ppl_loss_weight = 1
_batch_size = 14  # 4

model.expected_gpu_num = 8
lr = 0.00003
model.optim = AdamWParams(
    lr=lr,
    eps=1e-6,
    betas=[0.9, 0.98],
    weight_decay=0.01,
    sched=PolynomialHoldDecayAnnealingParams(
        min_lr=lr * 0.05,
        warmup_ratio=0.1,
        hold_ratio=0.4,
        max_steps=80000,
    ),
)

trainer = TrainerConfig(
    gpus=8,
    max_epochs=320,
    accelerator='ddp',
    accumulate_grad_batches=1,
    checkpoint_callback=False,  # Provided by exp_manager
    logger=False,  # Provided by exp_manager
    log_every_n_steps=50,
    progress_bar_refresh_rate=50,
    num_sanity_val_steps=0,
    check_val_every_n_epoch=1
)

exp_manager = ExpManagerConfig(
    name=config_name,
    create_checkpoint_callback=True,
    checkpoint_callback_params=CallbackParams(
        monitor="val_wer",
        mode="min",
        save_top_k=5
    )
)

cfg = ModelPTConfig(
    name=config_name,
    model=model,
    trainer=trainer,
    exp_manager=exp_manager
)
