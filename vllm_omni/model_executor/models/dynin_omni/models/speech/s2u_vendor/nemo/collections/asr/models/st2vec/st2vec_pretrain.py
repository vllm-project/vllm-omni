import logging
from math import ceil
from typing import Dict, Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.losses.similarityloss import NegativeCosineSimilarityLoss
from nemo.collections.asr.losses.wav2vecloss import Wav2VecLoss
from nemo.collections.asr.models.st2vec.st2vec_model import ST2VecEncoder, FeatST2VecEncoder
from nemo.collections.asr.parts.perturb import process_augmentations, RandomNoisePerturbation, AudioAugmentor
from nemo.core import ModelPT
from nemo.core.classes.common import PretrainedModelInfo
import pickle
import numpy as np

def buffered_arange(max):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]


class ST2VecPretrainModel(ModelPT):
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

        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        elif not isinstance(cfg, DictConfig):
            raise ValueError(f"cfg was type: {type(cfg)}. Expected either a dict or a DictConfig")

        if cfg.encoder_type == 'st':
            self.st2vec_encoder = ST2VecEncoder(cfg.st2vec_encoder)
        elif cfg.encoder_type == 'feat_st':
            self.st2vec_encoder = FeatST2VecEncoder(cfg.st2vec_encoder)
        else:
            raise ValueError('unknown encoder type: {}'.format(cfg.encoder_type))

        self.loss_type = cfg.loss_type
        if self.loss_type == 'neg_cos_sim':
            self.loss = NegativeCosineSimilarityLoss()
        else:
            assert self.loss_type == 'wav2vec'
            self.loss = Wav2VecLoss(
                feature_loss_weight=0.0,
                prob_ppl_weight=cfg.loss.prob_ppl_weight,
                logit_temp=cfg.logit_temp,
            )

        self.pitch_loss_weight = cfg.st2vec_encoder.pitch_loss_weight
        self.reconstruction_loss_weight = cfg.st2vec_encoder.reconstruction_loss_weight
        self.reconstruction_quant_ppl_loss_weight = cfg.st2vec_encoder.reconstruction_quant_ppl_loss_weight

        self._prev_log_step = -1

    def training_step(self, batch, batch_idx):
        loss, contrastive_loss, prob_ppl_loss, cur_temp, prob_ppl, _, pitch_loss, recon = self._step(batch)

        if self.global_step > self._prev_log_step:
            self._prev_log_step = self.global_step
            tensorboard = self.logger.experiment
            tensorboard.add_scalar('loss', loss, self.global_step)
            if prob_ppl_loss is not None:
                tensorboard.add_scalar('contrastive_loss', contrastive_loss, self.global_step)
                tensorboard.add_scalar('prob_ppl_loss', prob_ppl_loss, self.global_step)
                tensorboard.add_scalar('temp', cur_temp, self.global_step)
                tensorboard.add_scalar('prob_ppl', prob_ppl, self.global_step)
            if self.pitch_loss_weight:
                tensorboard.add_scalar('pitch_loss', pitch_loss, self.global_step)
            if self.reconstruction_loss_weight:
                tensorboard.add_scalar('contrastive_loss', contrastive_loss, self.global_step)
                tensorboard.add_scalar('recon_loss', recon['loss'], self.global_step)
                tensorboard.add_scalar('recon_quant_ppl_loss', recon['quant_ppl_loss'], self.global_step)
                tensorboard.add_scalar('recon_quant_ppl', recon['quant_ppl'], self.global_step)
                tensorboard.add_scalar('recon_quant_temp', recon['quant_temp'], self.global_step)
            tensorboard.add_scalar('learning_rate', self._optimizer.param_groups[0]['lr'], self.global_step)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, contrastive_loss, prob_ppl_loss, _, prob_ppl, accuracy, pitch_loss, recon = self._step(batch)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        if prob_ppl is not None:
            self.log('val_contrastive_loss', contrastive_loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=False)
            self.log('val_prob_ppl', prob_ppl, prog_bar=False, on_step=False, on_epoch=True, sync_dist=False)
        if accuracy is not None:
            self.log('val_accuracy', accuracy, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False)
        if self.pitch_loss_weight:
            self.log('val_pitch_loss', pitch_loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=False)
        if self.reconstruction_loss_weight:
            self.log('val_contrastive_loss', contrastive_loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=False)
            self.log('val_recon_loss', recon['loss'], prog_bar=False, on_step=False, on_epoch=True, sync_dist=False)
            self.log('val_recon_quant_ppl_loss', recon['quant_ppl_loss'], prog_bar=False, on_step=False, on_epoch=True, sync_dist=False)
            self.log('val_recon_quant_ppl', recon['quant_ppl'], prog_bar=False, on_step=False, on_epoch=True, sync_dist=False)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, contrastive_loss, prob_ppl_loss, _, _, accuracy, _, recon = self._step(batch)
        self.log('test_loss', loss, prog_bar=True, on_epoch=True, sync_dist=True)
        if accuracy is not None:
            self.log('test_accuracy', accuracy, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False)
        if recon is not None and batch_idx < 4:
            recon, recon_lens, target, target_len = recon['recon']
            torch.set_printoptions(profile="full")
            print('\n recon sample: {}'.format(batch_idx))
            print('tgtt:', target[0][120:240])
            print('pred:', recon[0][120:240])
            torch.set_printoptions(profile="default")  # reset

    def _step(self, batch):
        if len(batch) == 4:
            audio_signal, audio_lengths, p_audio_signal, p_audio_lengths = batch
        else:
            audio_signal, audio_lengths = batch
            p_audio_signal, p_audio_lengths = None, None

        logits, targets, sampled_negatives, _, prob_ppl_loss, cur_temp, prob_ppl, pitch_loss, recon = self(
            source=audio_signal, source_lens=audio_lengths, p_source=p_audio_signal, p_source_lens=p_audio_lengths
        )
        if self.loss_type == 'neg_cos_sim':
            loss = self.loss(predictions=logits, targets=targets)
            contrastive_loss, prob_ppl_loss, accuracy = None, None, None
        else:
            assert self.loss_type == 'wav2vec'
            loss, contrastive_loss, _, prob_ppl_loss, accuracy = self.loss(
                logits=logits,
                targets=targets,
                negatives=sampled_negatives,
                prob_ppl_loss=prob_ppl_loss,
                feature_loss=None,
                compute_accuracy=not self.training
            )

        if self.pitch_loss_weight:
            loss = loss + self.pitch_loss_weight * pitch_loss

        if self.reconstruction_loss_weight:
            loss = loss + self.reconstruction_loss_weight * recon['loss'] + self.reconstruction_quant_ppl_loss_weight * recon['quant_ppl_loss']

        return loss, contrastive_loss, prob_ppl_loss, cur_temp, prob_ppl, accuracy, pitch_loss, recon

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        return None

    def forward(self, source, source_lens, p_source, p_source_lens, mask=True, features_only=False) -> tuple:
        return self.st2vec_encoder(source, source_lens, p_source, p_source_lens, mask=mask, features_only=features_only,
                                   global_step=self.global_step)

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        if 'shuffle' not in train_data_config:
            train_data_config['shuffle'] = True

        # preserve config
        self._update_dataset_config(dataset_name='train', config=train_data_config)

        self._train_dl = self._setup_dataloader_from_config(config=train_data_config, noise_perturb_config=self._cfg['noise_perturb'])

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

        self._validation_dl = self._setup_dataloader_from_config(config=val_data_config, noise_perturb_config=None)

    def setup_test_data(self, test_data_config: Optional[Union[DictConfig, Dict]]):
        if 'shuffle' not in test_data_config:
            test_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='test', config=test_data_config)

        self._test_dl = self._setup_dataloader_from_config(config=test_data_config, noise_perturb_config=None)

    def _setup_dataloader_from_config(self, config: Optional[Dict], noise_perturb_config):

        if noise_perturb_config is not None:
            noise_perturb = RandomNoisePerturbation(**noise_perturb_config)
            augmentor = AudioAugmentor(perturbations=[(1.0, noise_perturb)])
            return_both = True
        else:
            augmentor = None
            return_both = False

        shuffle = config['shuffle']

        if 'manifest_filepath' in config and config['manifest_filepath'] is None:
            logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
            return None

        dataset = audio_to_text_dataset.get_audio_dataset(config=config, augmentor=augmentor, return_both=return_both)

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=dataset.collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    @torch.no_grad()
    def extract_feature(self, output_dir):
        # Model's mode and device
        mode = self.training
        device = next(self.parameters()).device
        preprocessor = self.st2vec_encoder.wav2spec
        pad_to_value = preprocessor.featurizer.pad_to

        extracted_feat = []
        extracted_feat_cnt = 1
        feat_file = None
        if self.st2vec_encoder.reconstruction_cfg:
           feat_path = output_dir / 'feat.txt'
           feat_file = open(feat_path, 'w')
            
        try:
            # preprocessor.featurizer.pad_to = 0
            # Switch model to evaluation mode
            self.eval()
            # Freeze the encoder and decoder modules
            # Work in tmp directory - will store manifest file there
            start = extracted_feat_cnt
            
            for test_batch in self._test_dl:

                batch = [d_i.to(device) for d_i in test_batch]

                if len(batch) == 4:
                    audio_signal, audio_lengths, _, _ = batch
                else:
                    audio_signal, audio_lengths = batch

                feat, feat_len = self.st2vec_encoder(audio_signal, audio_lengths, None, None, mask=False,
                                    features_only=True, global_step=self.global_step)

#                 res = {'feat': feat.detach().cpu().numpy(),
#                        'feat_len': feat_len.detach().cpu().numpy()}
                if self.st2vec_encoder.reconstruction_cfg:
                    q_feat, q_feat_ids = self.st2vec_encoder.reconstructor.get_quantized_feat(feat, feat_len) 
                    q_feat = q_feat.detach().cpu().numpy()
                    q_feat_ids = q_feat_ids.detach().cpu().numpy()
                    feat_len = feat_len.detach().cpu().numpy()
                    #print("=========", np.shape(q_feat_ids), np.shape(feat_len))
                    for i in range(len(q_feat_ids)):
                        feat_str = " ".join([str(x[0]) for x in q_feat_ids[i][:feat_len[i]]])
                        feat_file.write(f"{feat_str}\n")
                else:
                     res = {'feat': feat.detach().cpu().numpy(),
                        'feat_len': feat_len.detach().cpu().numpy()}
                     extracted_feat.append({res})
                     if extracted_feat_cnt % 200 == 0 or extracted_feat_cnt == len(self._test_dl) :
                        feat_fp = output_dir / 'feat_{}-{}.pkl'.format(start, extracted_feat_cnt)
                        with feat_fp.open(mode='wb') as output_file:
                            print('save features to: {}'.format(feat_fp))
                            pickle.dump(extracted_feat, output_file)
                        extracted_feat = [] # clear the list
                        start = extracted_feat_cnt + 1 # set the chunk start index 
                extracted_feat_cnt += 1
                #
                print('extract feat: {}/{}'.format(extracted_feat_cnt, len(self._test_dl)))
            if feat_file:
              feat_file.close()
        finally:
            # set mode back to its original value
            self.train(mode=mode)
            preprocessor.featurizer.pad_to = pad_to_value
        return extracted_feat
