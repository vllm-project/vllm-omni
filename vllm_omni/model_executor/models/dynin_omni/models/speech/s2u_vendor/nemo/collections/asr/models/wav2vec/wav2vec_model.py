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


# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Facebook, Inc. and its affiliates.
#
import logging
from math import ceil
from typing import Dict, Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch import nn

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.losses.wav2vecloss import Wav2VecLoss
from nemo.collections.asr.models.wav2vec.wav2vec_config import Wav2VecEncoderModelConfig
from nemo.collections.asr.modules.wav2vec_modules import GumbelVectorQuantizer, compute_mask_indices
from nemo.collections.asr.parts.perturb import process_augmentations
from nemo.collections.asr.parts.wav2vec import ConvFeatureEncoder, GradMultiply, Wav2VecTransformerEncoder, \
    TransformerEncoder
from nemo.core import ModelPT
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import AudioSignal, EncodedRepresentation, LossType, MaskType, NeuralType
from nemo.core.neural_types.elements import BoolType, FloatType


def buffered_arange(max):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]


class Wav2VecEncoderModel(ModelPT):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        self.global_rank = 0
        self.world_size = 1
        self.local_rank = 0
        if trainer is not None:
            self.global_rank = (trainer.node_rank * trainer.num_gpus) + trainer.local_rank
            self.world_size = trainer.num_nodes * trainer.num_gpus
            self.local_rank = trainer.local_rank

        super().__init__(cfg=cfg, trainer=trainer)

        schema = OmegaConf.structured(Wav2VecEncoderModelConfig)
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        elif not isinstance(cfg, DictConfig):
            raise ValueError(f"cfg was type: {type(cfg)}. Expected either a dict or a DictConfig")

        cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg = OmegaConf.merge(schema, cfg)

        feature_enc_layers = cfg.conv_feature_encoder.conv_feature_layers
        self.embed = feature_enc_layers[-1][0]  # Select last conv output layer dimension

        self.feature_extractor = ConvFeatureEncoder(
            conv_layers=feature_enc_layers,
            mode=cfg.conv_feature_encoder.extractor_mode,
            conv_bias=cfg.conv_feature_encoder.conv_bias,
        )

        encoder_embed_dim = cfg.transformer_encoder.encoder.embedding_dim
        self.post_extract_proj = (
            nn.Linear(self.embed, encoder_embed_dim)
            if self.embed != encoder_embed_dim and not cfg.quantizer.quantize_input
            else None
        )
        assert not cfg.quantizer.quantize_input  # finetune expect this

        self.mask_cfg = cfg.masking

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult

        self.quantizer = None
        self.input_quantizer = None

        self.n_negatives = cfg.n_negatives
        self.cross_sample_negatives = cfg.cross_sample_negatives
        self.codebook_negatives = cfg.codebook_negatives
        self.negatives_from_everywhere = cfg.negatives_from_everywhere

        final_dim = cfg.final_dim if cfg.final_dim > 0 else encoder_embed_dim
        self.final_dim = final_dim
        self.quantize_targets = cfg.quantizer.quantize_targets
        if self.quantize_targets:
            assert cfg.quantizer.targets_bottleneck_dim is None
            vq_dim = cfg.quantizer.latent_dim if cfg.quantizer.latent_dim > 0 else final_dim
            self.quantizer = GumbelVectorQuantizer(
                dim=self.embed,
                num_vars=cfg.quantizer.latent_vars,
                temp=cfg.quantizer.latent_temp,
                groups=cfg.quantizer.latent_groups,
                combine_groups=False,
                vq_dim=vq_dim,
                time_first=True,
            )
            self.project_q = nn.Linear(vq_dim, final_dim)
        else:
            assert cfg.loss.prob_ppl_weight == 0
            targets_bottleneck_dim = cfg.quantizer.targets_bottleneck_dim
            if targets_bottleneck_dim is None:
                self.project_q = nn.Linear(self.embed, final_dim)
            else:
                act_fn_dic = {'relu': nn.ReLU, 'gelu': nn.GELU}
                targets_proj_act_fn = cfg.quantizer.targets_bottleneck_act_fn
                targets_proj_layers = (
                    [nn.Linear(self.embed, targets_bottleneck_dim)]
                    + ([] if targets_proj_act_fn is None else [act_fn_dic[targets_proj_act_fn]])
                    + [nn.Linear(targets_bottleneck_dim, final_dim)]

                )
                self.project_q = torch.nn.Sequential(*targets_proj_layers)

        if cfg.quantizer.quantize_input:
            if cfg.quantizer.same_quantizer and self.quantizer is not None:
                vq_dim = final_dim
                self.input_quantizer = self.quantizer
            else:
                vq_dim = cfg.quantizer.latent_dim if cfg.quantizer.latent_dim > 0 else encoder_embed_dim
                self.input_quantizer = GumbelVectorQuantizer(
                    dim=self.embed,
                    num_vars=cfg.quantizer.latent_vars,
                    temp=cfg.quantizer.latent_temp,
                    groups=cfg.quantizer.latent_groups,
                    combine_groups=False,
                    vq_dim=vq_dim,
                    time_first=True,
                )
            self.project_inp = nn.Linear(vq_dim, encoder_embed_dim)

        self.mask_emb = nn.Parameter(torch.FloatTensor(encoder_embed_dim).uniform_())

        if cfg.transformer_encoder.use_pytorch_transformer:
            self.encoder = Wav2VecTransformerEncoder(cfg.transformer_encoder)
        else:
            self.encoder = TransformerEncoder(cfg.transformer_encoder)
        self.layer_norm = nn.LayerNorm(self.embed)

        self.target_glu = None
        if cfg.target_glu:
            self.target_glu = nn.Sequential(nn.Linear(final_dim, final_dim * 2), nn.GLU())

        self.final_proj = nn.Linear(encoder_embed_dim, final_dim)
        self.loss = Wav2VecLoss(
            feature_loss_weight=cfg.loss.feature_loss_weight,
            prob_ppl_weight=cfg.loss.prob_ppl_weight,
            logit_temp=cfg.logit_temp,
        )
        self._prev_log_step = -1

    def training_step(self, batch, batch_idx):
        loss, contrastive_loss, feature_loss, prob_ppl_loss, cur_temp, prob_ppl, _ = self._step(batch)

        if self.global_step > self._prev_log_step:
            self._prev_log_step = self.global_step
            tensorboard = self.logger.experiment
            tensorboard.add_scalar('loss', loss, self.global_step)
            tensorboard.add_scalar('contrastive_loss', contrastive_loss, self.global_step)
            tensorboard.add_scalar('feature_loss', feature_loss, self.global_step)
            if self.quantize_targets:
                tensorboard.add_scalar('prob_ppl_loss', prob_ppl_loss, self.global_step)
                tensorboard.add_scalar('temp', cur_temp, self.global_step)
                tensorboard.add_scalar('prob_ppl', prob_ppl, self.global_step)
            tensorboard.add_scalar('learning_rate', self._optimizer.param_groups[0]['lr'], self.global_step)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, contrastive_loss, feature_loss, prob_ppl_loss, _, _, accuracy = self._step(batch)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('val_accuracy', accuracy, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, contrastive_loss, feature_loss, prob_ppl_loss, _, _, accuracy = self._step(batch)
        self.log('test_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('test_accuracy', accuracy, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False)

    def _step(self, batch):
        audio_signal, audio_lengths = batch

        self._update_quantizer_temp()
        logits, targets, sampled_negatives, _, features_penalty, prob_ppl_loss, cur_temp, prob_ppl = self(
            source=audio_signal, source_len=audio_lengths
        )
        loss, contrastive_loss, feature_loss, prob_ppl_loss, accuracy = self.loss(
            logits=logits,
            targets=targets,
            negatives=sampled_negatives,
            prob_ppl_loss=prob_ppl_loss,
            feature_loss=features_penalty,
            compute_accuracy=not self.training
        )
        return loss, contrastive_loss, feature_loss, prob_ppl_loss, cur_temp, prob_ppl, accuracy

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        return None

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "source": NeuralType(('B', 'T'), AudioSignal()),
            "padding_mask": NeuralType(('B', 'T'), MaskType(), optional=True),
            "mask": NeuralType(elements_type=BoolType(), optional=True),
            "features_only": NeuralType(elements_type=BoolType(), optional=True),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "logits": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
            "targets": NeuralType(('B', 'T', 'D'), EncodedRepresentation(), optional=True),
            "sampled_negatives": NeuralType(('N', 'B', 'T', 'D'), EncodedRepresentation(), optional=True),
            "padding_mask": NeuralType(('B', 'T'), MaskType(), optional=True),
            "features_penalty": NeuralType(elements_type=LossType(), optional=True),
            "prob_ppl_loss": NeuralType(elements_type=LossType(), optional=True),
            "cur_codebook_temp": NeuralType(elements_type=FloatType(), optional=True),
        }

    def forward(self, source, source_len, *, mask=True, features_only=False) -> tuple:
        prob_ppl_loss, cur_temp = None, None

        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)
        feature_lens = self.feature_extractor.get_subsampled_lens(source_len)
        padding_mask = self._create_padding_mask(feature_lens)
        assert feature_lens.max() == features.shape[2] == padding_mask.shape[1]

        features = features.transpose(1, 2)

        features_penalty = features[~padding_mask].float().pow(2).mean()  # L2 Norm on features

        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        assert self.input_quantizer is None
        # if self.input_quantizer:
        #     features, prob_ppl_loss, cur_codebook_temp = self.input_quantizer(features)
        #     features = self.project_inp(features)
        if mask:
            logits, mask_indices, mask_num = self.apply_mask(features, padding_mask)
            if mask_indices is not None:
                targets = unmasked_features[mask_indices]
                if self.mask_cfg.mask_shrink_to_batch_min:
                    targets = targets.view(
                        unmasked_features.size(0), -1, unmasked_features.size(-1)
                    )
                else:
                    # fake batch dim 1
                    targets = targets.view(
                        1, -1, unmasked_features.size(-1)
                    )
                    assert targets.shape[1] == sum(mask_num)
            else:
                targets = unmasked_features
        else:
            logits = features
            targets = unmasked_features
            mask_indices = None
            mask_num = None

        logits = self.encoder(logits, padding_mask=padding_mask)

        if features_only:
            return logits, padding_mask

        if self.quantize_targets:
            targets, prob_ppl_loss, cur_temp, prob_ppl = self.quantizer(targets)
            targets = self.project_q(targets)

            if self.negatives_from_everywhere:
                assert self.mask_cfg.mask_shrink_to_batch_min
                neg_cands, *_ = self.quantizer(unmasked_features)
                sampled_negatives, _ = self.sample_negatives(neg_cands, targets.size(1))
                sampled_negatives = self.project_q(sampled_negatives)
            else:
                if self.mask_cfg.mask_shrink_to_batch_min:
                    sampled_negatives, _ = self.sample_negatives(targets, targets.size(1))
                else:
                    sampled_negatives, _ = self.sample_negatives_flat(targets, mask_num)

            if self.codebook_negatives > 0:
                assert self.mask_cfg.mask_shrink_to_batch_min
                cb_negs = self.quantizer.sample_from_codebook(
                    targets.size(0) * targets.size(1), self.codebook_negatives
                )
                cb_negs = cb_negs.view(
                    self.codebook_negatives, targets.size(0), targets.size(1), -1
                )  # order doesnt matter
                cb_negs = self.project_q(cb_negs)
                sampled_negatives = torch.cat([sampled_negatives, cb_negs], dim=0)
        else:
            targets = self.project_q(targets)
            prob_ppl = None

            if self.negatives_from_everywhere:
                assert self.mask_cfg.mask_shrink_to_batch_min
                sampled_negatives, _ = self.sample_negatives(unmasked_features, targets.size(1))
                sampled_negatives = self.project_q(sampled_negatives)
            else:
                if self.mask_cfg.mask_shrink_to_batch_min:
                    sampled_negatives, _ = self.sample_negatives(targets, targets.size(1))
                else:
                    sampled_negatives, _ = self.sample_negatives_flat(targets, mask_num)

        mask_logits = logits[mask_indices]
        if self.mask_cfg.mask_shrink_to_batch_min:
            mask_logits = mask_logits.view(logits.size(0), -1, logits.size(-1))
        else:
            # fake batch dim to 1
            mask_logits = mask_logits.view(1, -1, logits.size(-1))

        if self.target_glu:
            targets = self.target_glu(targets)
            sampled_negatives = self.target_glu(sampled_negatives)

        mask_logits = self.final_proj(mask_logits)

        return mask_logits, targets, sampled_negatives, padding_mask, features_penalty, prob_ppl_loss, cur_temp, prob_ppl

    def extract_features(self, source, audio_lengths, mask=False):
        padding_mask = self._create_padding_mask(audio_lengths)
        return self(source=source, padding_mask=padding_mask, mask=mask, features_only=True)

    def remove_pretraining_modules(self):
        self.quantizer = None
        self.project_q = None
        self.target_glu = None
        self.final_proj = None

    def _update_quantizer_temp(self):
        if self.quantizer:
            self.quantizer.set_num_updates(self.trainer.global_step)
        if self.input_quantizer:
            self.input_quantizer.set_num_updates(self.trainer.global_step)

    def apply_mask(self, x, padding_mask):
        B, T, C = x.shape
        if self.mask_cfg.mask_prob > 0:
            mask_indices, mask_num = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_cfg.mask_prob,
                self.mask_cfg.mask_length,
                self.mask_cfg.mask_type,
                self.mask_cfg.mask_other,
                min_masks=2,
                no_overlap=self.mask_cfg.no_mask_overlap,
                min_space=self.mask_cfg.mask_min_space,
                shrink_to_batch_min=self.mask_cfg.mask_shrink_to_batch_min,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            mask_emb = self.mask_emb.type_as(x)
            x[mask_indices] = mask_emb
        else:
            mask_indices = None

        if self.mask_cfg.mask_channel_prob > 0:
            # assert self.mask_cfg.mask_shrink_to_batch_min
            mask_channel_indices, _ = compute_mask_indices(
                (B, C),
                None,
                self.mask_cfg.mask_channel_prob,
                self.mask_cfg.mask_channel_length,
                self.mask_cfg.mask_channel_type,
                self.mask_cfg.mask_channel_other,
                no_overlap=self.mask_cfg.no_mask_channel_overlap,
                min_space=self.mask_cfg.mask_channel_min_space,
                shrink_to_batch_min=self.mask_cfg.mask_channel_shrink_to_batch_min,
            )
            mask_channel_indices = torch.from_numpy(mask_channel_indices).to(x.device).unsqueeze(1).expand(-1, T, -1)
            x[mask_channel_indices] = 0

        assert len(mask_num) == B
        return x, mask_indices, mask_num

    def sample_negatives(self, y, num):

        if self.n_negatives == 0 and self.cross_sample_negatives == 0:
            return y.new(0)

        bsz, tsz, fsz = y.shape
        y = y.view(-1, fsz)  # BTC => (BxT)C

        cross_high = tsz * bsz
        high = tsz
        with torch.no_grad():
            assert high > 1, f"{bsz, tsz, fsz}"

            if self.n_negatives > 0:
                tszs = buffered_arange(num).unsqueeze(-1).expand(-1, self.n_negatives).flatten()

                neg_idxs = torch.randint(low=0, high=high - 1, size=(bsz, self.n_negatives * num))
                neg_idxs[neg_idxs >= tszs] += 1

            if self.cross_sample_negatives > 0:
                tszs = buffered_arange(num).unsqueeze(-1).expand(-1, self.cross_sample_negatives).flatten()

                cross_neg_idxs = torch.randint(
                    low=0, high=cross_high - 1, size=(bsz, self.cross_sample_negatives * num),
                )
                cross_neg_idxs[cross_neg_idxs >= tszs] += 1

        if self.n_negatives > 0:
            for i in range(1, bsz):
                neg_idxs[i] += i * high
        else:
            neg_idxs = cross_neg_idxs

        if self.cross_sample_negatives > 0 and self.n_negatives > 0:
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

        negs = y[neg_idxs.view(-1)]
        negs = negs.view(bsz, num, self.n_negatives + self.cross_sample_negatives, fsz).permute(
            2, 0, 1, 3
        )  # to NxBxTxC
        return negs, neg_idxs

    def sample_negatives_flat(self, y, nums):

        if self.n_negatives == 0 and self.cross_sample_negatives == 0:
            return y.new(0)

        bsz, tsz, fsz = y.shape
        assert bsz == 1 and tsz == sum(nums)  # fake batch dim
        y = y.view(-1, fsz)  # BTC => (BxT)C

        # cross_high = tsz * bsz

        neg_idxs_l = []
        idx_start = 0
        with torch.no_grad():
            for i, num_i in enumerate(nums):
                assert num_i > 1, f"{bsz, tsz, fsz}"

                assert self.n_negatives > 0
                tszs_i = buffered_arange(num_i).unsqueeze(-1).expand(-1, self.n_negatives).flatten()

                high_i = num_i
                neg_idxs_i = torch.randint(low=0, high=high_i - 1, size=(self.n_negatives * num_i,))
                neg_idxs_i[neg_idxs_i >= tszs_i] += 1

                neg_idxs_i += idx_start
                idx_start += num_i

                neg_idxs_l.append(neg_idxs_i)

                assert self.cross_sample_negatives == 0
                # if self.cross_sample_negatives > 0:
                #     tszs = buffered_arange(num_i).unsqueeze(-1).expand(-1, self.cross_sample_negatives).flatten()
                #
                #     cross_neg_idxs = torch.randint(
                #         low=0, high=cross_high - 1, size=(self.cross_sample_negatives * num_i),
                #     )
                #     cross_neg_idxs[cross_neg_idxs >= tszs] += 1

                # if self.n_negatives <= 0:
                #     neg_idxs = cross_neg_idxs

                # if self.cross_sample_negatives > 0 and self.n_negatives > 0:
                #     neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

        neg_idxs = torch.cat(neg_idxs_l)
        assert neg_idxs.ndim == 1

        negs = y[neg_idxs]
        negs = negs.view(bsz, sum(nums), self.n_negatives + self.cross_sample_negatives, fsz).permute(
            2, 0, 1, 3
        )  # to NxBxTxC
        return negs, neg_idxs

    def _create_padding_mask(self, audio_lengths):
        # Broadcast to vectorize creating the padding mask
        max_len = max(audio_lengths)
        padding_mask = torch.arange(max_len, device=audio_lengths.device)
        padding_mask = padding_mask.expand(len(audio_lengths), max_len) < audio_lengths.unsqueeze(1)
        # Negate to false where no padding
        padding_mask = ~padding_mask
        return padding_mask

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        if 'shuffle' not in train_data_config:
            train_data_config['shuffle'] = True

        # preserve config
        self._update_dataset_config(dataset_name='train', config=train_data_config)

        self._train_dl = self._setup_dataloader_from_config(config=train_data_config)

        # Need to set this because if using an IterableDataset, the length of the dataloader is the total number
        # of samples rather than the number of batches, and this messes up the tqdm progress bar.
        # So we set the number of steps manually (to the correct number) to fix this.
        if 'is_tarred' in train_data_config and train_data_config['is_tarred']:
            # We also need to check if limit_train_batches is already set.
            # If it's an int, we assume that the user has set it to something sane, i.e. <= # training batches,
            # and don't change it. Otherwise, adjust batches accordingly if it's a float (including 1.0).
            if isinstance(self._trainer.limit_train_batches, float):
                self._trainer.limit_train_batches = int(
                    self._trainer.limit_train_batches
                    * ceil((len(self._train_dl.dataset) / self.world_size) / train_data_config['batch_size'])
                )

    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        if 'shuffle' not in val_data_config:
            val_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='validation', config=val_data_config)

        self._validation_dl = self._setup_dataloader_from_config(config=val_data_config)

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        if 'shuffle' not in test_data_config:
            test_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='test', config=test_data_config)

        self._test_dl = self._setup_dataloader_from_config(config=test_data_config)

    def _setup_dataloader_from_config(self, config: Optional[Dict]):

        if 'augmentor' in config:
            augmentor = process_augmentations(config['augmentor'])
        else:
            augmentor = None

        shuffle = config['shuffle']

        if 'manifest_filepath' in config and config['manifest_filepath'] is None:
            logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
            return None

        dataset = audio_to_text_dataset.get_audio_dataset(config=config, augmentor=augmentor)

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=dataset.collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )
