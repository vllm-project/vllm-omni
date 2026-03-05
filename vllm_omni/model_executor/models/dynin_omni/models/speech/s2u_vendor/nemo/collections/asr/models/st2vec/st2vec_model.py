import contextlib
import math

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn

from nemo.collections.asr.models.st2vec.st2vec_config import ShiftPerturbConfig
# from nemo.collections.asr.modules import yin
from nemo.collections.asr.modules.wav2vec_modules import compute_mask_indices, GumbelVectorQuantizer
from nemo.collections.asr.parts.spec2vec import Projector
from nemo.collections.asr.parts.spectr_augment import GAUSSIAN_MASK
from nemo.collections.asr.parts.wav2vec import TransformerEncoder
from nemo.core.classes.common import Serialization


def buffered_arange(max):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]


class ST2VecEncoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.wav2spec = Serialization.from_config_dict(cfg.preprocessor)

        self.freeze_student = cfg.freeze_student
        self.feature_encoder = Serialization.from_config_dict(cfg.feature_encoder)
        if cfg.pretrained_encoder_path:
            print('load feature_encoder from:', cfg.pretrained_encoder_path)
            self.feature_encoder.load_state_dict(
                get_state_dict('st2vec_encoder.feature_encoder.', cfg.pretrained_encoder_path))
        if self.freeze_student:
            stop_grad(self.feature_encoder)

        self.mask_cfg = cfg.masking
        self.target_mask_cfg = cfg.target_masking

        self.n_negatives = cfg.n_negatives
        self.cross_sample_negatives = cfg.cross_sample_negatives
        self.codebook_negatives = cfg.codebook_negatives
        self.negatives_from_everywhere = cfg.negatives_from_everywhere
        self.negatives_from_noisy_features = cfg.negatives_from_noisy_features

        if self.mask_cfg.mask_emb_type == 'zero':
            self.mask_emb = 0.0
        else:
            assert self.mask_cfg.mask_emb_type == 'gaussian'
            num_features = cfg.preprocessor.features
            self.register_buffer("mask_emb", torch.tensor(GAUSSIAN_MASK[:num_features]))

        if cfg.shifting is not None:
            self.random_shift = RandomShift(cfg.shifting)
        else:
            self.random_shift = None

        if cfg.target_shifting is not None:
            assert not cfg.target_shifting.truncate
            assert cfg.target_shifting.dist == 'uniform' and cfg.target_shifting.min >= 0
            self.target_shifting = RandomShift(cfg.target_shifting)
        else:
            self.target_shifting = None

        cfg.projector.input_dim = self.feature_encoder.output_dim
        self.projector = Projector(cfg.projector)
        if cfg.pretrained_encoder_path:
            print('load projector from:', cfg.pretrained_encoder_path)
            self.projector.load_state_dict(get_state_dict('st2vec_encoder.projector.', cfg.pretrained_encoder_path))
        if self.freeze_student:
            stop_grad(self.projector)

        self.target_compute_perturb = cfg.target_compute_perturb

        self.target_update_step = 0
        if cfg.target_momentum > 0:
            self.target_feature_encoder = Serialization.from_config_dict(cfg.feature_encoder)
            self.target_feature_encoder.load_state_dict(self.feature_encoder.state_dict())
            for p in self.target_feature_encoder.parameters():
                p.requires_grad = False

            self.target_projector = Projector(cfg.projector)
            self.target_projector.load_state_dict(self.projector.state_dict())
            for p in self.target_projector.parameters():
                p.requires_grad = False

            if cfg.target_momentum_final is not None:
                assert cfg.target_momentum_steps is not None
                self.momentum_schedule = momentum_scheduler(cfg.target_momentum, cfg.target_momentum_final,
                                                            cfg.target_momentum_steps, type=cfg.target_momentum_type)
            else:
                self.momentum_schedule = lambda _: cfg.target_momentum
        else:
            self.target_feature_encoder = None
            self.target_projector = None
            self.momentum_schedule = None

        if cfg.predictor is not None:
            cfg.predictor.input_dim = self.projector.output_dim
            self.predictor = Projector(cfg.predictor)
            if cfg.pretrained_encoder_path:
                print('load predictor from:', cfg.pretrained_encoder_path)
                self.predictor.load_state_dict(get_state_dict('st2vec_encoder.predictor.', cfg.pretrained_encoder_path))
            final_dim = self.predictor.output_dim
            if self.freeze_student:
                stop_grad(self.predictor)
        else:
            self.predictor = None
            final_dim = self.projector.output_dim

        if cfg.quantizer is not None:
            vq_dim = cfg.quantizer.latent_dim if cfg.quantizer.latent_dim > 0 else final_dim
            self.quantizer = GumbelVectorQuantizer(
                dim=self.projector.output_dim,
                num_vars=cfg.quantizer.latent_vars,
                temp=cfg.quantizer.latent_temp,
                groups=cfg.quantizer.latent_groups,
                combine_groups=False,
                vq_dim=vq_dim,
                time_first=True,
            )
            self.project_q = nn.Linear(vq_dim, final_dim)  # Todo: can utilize later
        else:
            self.quantizer = None
            self.project_q = None

        self.pitch_pred_cfg = cfg.pitch_estimation_task
        if self.pitch_pred_cfg:
            self.pitch_pred_cfg.pitch_estimator.input_dim = self.feature_encoder.output_dim
            self.pitch_estimator = Projector(self.pitch_pred_cfg.pitch_estimator)

        self.reconstruction_cfg = cfg.reconstruction_task
        if self.reconstruction_cfg:
            self.reconstruction_cfg.reconstructor.input_dim = self.feature_encoder.output_dim
            if self.reconstruction_cfg.global_cond_extractor:
                self.reconstruction_cfg.global_cond_extractor.input_dim = num_features
            self.reconstructor = Reconstrutor(self.reconstruction_cfg)

    def forward(self, wavs, wav_lens, p_wavs, p_wav_lens, *, mask=True, features_only=False, global_step=None) -> tuple:
        # specs: [B, C, T]
        unmasked_specs, unmasked_specs_len = self.wav2spec(input_signal=wavs, length=wav_lens)

        if self.reconstruction_cfg:
            recon_target = unmasked_specs.clone()
            recon_target_len = unmasked_specs_len.clone()

        if p_wavs is None:
            if features_only:
                specs = unmasked_specs
                specs_len = unmasked_specs_len
            else:
                specs = unmasked_specs.clone()
                specs_len = unmasked_specs_len.clone()
        else:
            specs, specs_len = self.wav2spec(input_signal=p_wavs, length=p_wav_lens)

        specs_mask = create_padding_mask(unmasked_specs_len, unmasked_specs.shape[2]) if mask else None

        if not features_only:
            if self.training and self.target_shifting is not None:
                unmasked_specs = unmasked_specs.transpose(1, 2)
                unmasked_specs, unmasked_specs_len, target_shift_num, _, target_r_shift_num = (
                    self.target_shifting.shift(unmasked_specs, unmasked_specs_len, self.mask_emb)
                )
                unmasked_specs = unmasked_specs.transpose(1, 2)
            else:
                target_shift_num = 0
                target_r_shift_num = 0

            if self.training and self.target_mask_cfg is not None:
                unmasked_specs = unmasked_specs.transpose(1, 2)
                unmasked_specs, _, _ = apply_mask(self.target_mask_cfg, unmasked_specs, specs_mask, self.mask_emb)
                unmasked_specs = unmasked_specs.transpose(1, 2)

            if self.momentum_schedule is not None:
                assert global_step is not None
                target_momentum = self.momentum_schedule(global_step)
                if global_step > self.target_update_step:
                    ema_update(self.target_feature_encoder, self.feature_encoder, target_momentum)
                    ema_update(self.target_projector, self.projector, target_momentum)
                    self.target_update_step = global_step
                target_feature_encoder = self.target_feature_encoder
                target_projector = self.target_projector
            else:
                target_feature_encoder = self.feature_encoder
                target_projector = self.projector
            with torch.no_grad():
                with as_eval(target_feature_encoder,
                             target_projector) if not self.target_compute_perturb else contextlib.suppress():
                    unmasked_features, unmasked_feature_lens, _ = target_feature_encoder(unmasked_specs,
                                                                                         unmasked_specs_len)
                    # [B, D, T] => [B, T, D]
                    unmasked_features = unmasked_features.transpose(1, 2)

                    teacher_encoder_features = unmasked_features
                    teacher_encoder_features_len = unmasked_feature_lens

                    unmasked_features = target_projector(unmasked_features, length=unmasked_feature_lens)

            if target_shift_num > 0:
                unmasked_features = unmasked_features[:, target_shift_num:]
                unmasked_feature_lens = unmasked_feature_lens - target_shift_num
                if self.reconstruction_cfg and self.reconstruction_cfg.use_teacher_feat_prob:
                    teacher_encoder_features = teacher_encoder_features[:, target_shift_num:]
                    teacher_encoder_features_len = teacher_encoder_features_len - target_shift_num

            else:
                assert target_shift_num == 0

            if target_r_shift_num > 0:
                unmasked_features = unmasked_features[:, :-target_r_shift_num]
                unmasked_feature_lens = unmasked_feature_lens - target_r_shift_num
                if self.reconstruction_cfg and self.reconstruction_cfg.use_teacher_feat_prob:
                    teacher_encoder_features = teacher_encoder_features[:, :-target_r_shift_num]
                    teacher_encoder_features_len = teacher_encoder_features_len - target_r_shift_num
            else:
                assert target_r_shift_num == 0

        if self.random_shift is not None and not features_only:
            specs = specs.transpose(1, 2)
            specs, specs_len, shift_num, _, r_shift_num = self.random_shift.shift(specs, specs_len, self.mask_emb)
            specs = specs.transpose(1, 2)
        else:
            shift_num = 0
            r_shift_num = 0

        if mask:
            specs = specs.transpose(1, 2)
            specs, _, _ = apply_mask(self.mask_cfg, specs, specs_mask, self.mask_emb)
            specs = specs.transpose(1, 2)

        features, feature_lens, _ = self.feature_encoder(specs, specs_len)
        # [B, D, T] => [B, T, D]
        features = features.transpose(1, 2)
        if features_only:
            return features, feature_lens

        assert mask

        encoder_features = features
        encoder_features_len = feature_lens

        features = self.projector(features, length=feature_lens)

        if self.predictor is not None:
            pred_features = self.predictor(features, length=feature_lens)
        else:
            pred_features = features

        if shift_num > 0:
            # remove paddings introduced by shift
            pred_features = pred_features[:, shift_num:]
            feature_lens = feature_lens - shift_num
        elif shift_num < 0:
            assert self.reconstruction_cfg is None
            unmasked_features = unmasked_features[:, abs(shift_num):]
            unmasked_feature_lens -= abs(shift_num)

        if r_shift_num > 0:
            pred_features = pred_features[:, :-r_shift_num]
            feature_lens = feature_lens - r_shift_num
        elif r_shift_num < 0:
            assert self.reconstruction_cfg is None
            unmasked_features = unmasked_features[:, :-abs(r_shift_num)]
            unmasked_feature_lens -= abs(r_shift_num)

        assert pred_features.shape[1] == unmasked_features.shape[1]
        assert torch.equal(feature_lens, unmasked_feature_lens)

        padding_mask = create_padding_mask(feature_lens, pred_features.shape[1])
        features_mask = ~padding_mask

        pred_features = pred_features[features_mask]
        # fake batch dim to 1
        pred_features = pred_features.view(1, -1, pred_features.size(-1))

        unmasked_features = unmasked_features[features_mask]
        # fake batch dim 1
        unmasked_features = unmasked_features.view(1, -1, unmasked_features.size(-1))

        assert not unmasked_features.requires_grad
        if self.quantizer is not None:
            self.quantizer.set_num_updates(global_step)
            unmasked_features, prob_ppl_loss, cur_temp, prob_ppl = self.quantizer(unmasked_features)
            unmasked_features = self.project_q(unmasked_features)
        else:
            prob_ppl_loss, cur_temp, prob_ppl = None, None, None

        sampled_negatives, _ = self.sample_negatives_flat(unmasked_features, feature_lens.tolist())

        if self.pitch_pred_cfg:
            pitch_loss = self.get_pitch_pred_loss(encoder_features, encoder_features_len, wavs, wav_lens)
        else:
            pitch_loss = None

        if self.reconstruction_cfg:
            recon = self.reconstructor(encoder_features, encoder_features_len, teacher_encoder_features,
                                       teacher_encoder_features_len,
                                       recon_target, recon_target_len, global_step)
        else:
            recon = None

        return pred_features, unmasked_features, sampled_negatives, padding_mask, prob_ppl_loss, cur_temp, prob_ppl, pitch_loss, recon

    def get_pitch_pred_loss(self, features, feature_lens, wavs, wav_lens, mean=True):
        target = yin.estimate(wavs, sample_rate=self.pitch_pred_cfg.sample_rate,
                              pitch_min=self.pitch_pred_cfg.pitch_min,
                              pitch_max=self.pitch_pred_cfg.pitch_max)
        # target = wavs[:, :int(wavs.shape[1] / 160)]
        target = target / self.pitch_pred_cfg.pitch_max

        pred = self.pitch_estimator(features, length=feature_lens)

        B, T, C = pred.size()
        assert C == self.pitch_pred_cfg.reduction_rate
        pred = pred.reshape(B, T * C)
        pred_lens = feature_lens * self.pitch_pred_cfg.reduction_rate

        # import random
        # if random.random() < 0.001:
        #     torch.set_printoptions(profile="full")
        #     print('\ntgtt:', target[0][:300])
        #     print('pred:', pred[0][:300])
        #     torch.set_printoptions(profile="default")  # reset

        if pred.shape[1] > target.shape[1]:
            pred = pred.narrow(dim=1, start=0, length=target.shape[1]).contiguous()
        elif pred.shape[1] < target.shape[1]:
            target = target.narrow(dim=1, start=0, length=pred.shape[1]).contiguous()

        loss_mask = create_padding_mask(pred_lens, pred.shape[1])
        loss_mask = ~loss_mask

        pred = pred[loss_mask]
        target = target[loss_mask]
        if self.pitch_pred_cfg.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.pitch_pred_cfg.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")
        return loss

    def check_collapse(self, features, feature_lens, unmasked_features, proj_features):
        import torch.nn.functional as F
        trunc_len = min(feature_lens.min(), 80)

        feat_0_trunc = features[0, :trunc_len]
        # [T, 1, C]
        feat_0_trunc_src = feat_0_trunc.unsqueeze(1)
        # [1, T, C]
        feat_0_trunc_tgt = feat_0_trunc.unsqueeze(0)
        feat_self_sim = F.cosine_similarity(feat_0_trunc_src, feat_0_trunc_tgt, dim=-1)
        print('feat_self_sim: \n', feat_self_sim)

        proj_feat_0_trunc = proj_features[0, :trunc_len]
        feat_proj_sim = F.cosine_similarity(feat_0_trunc, proj_feat_0_trunc, dim=-1)
        print('feat_proj_sim: \n', feat_proj_sim)

        um_feat_0_trunc = unmasked_features[0, :trunc_len]
        feat_um_sim = F.cosine_similarity(feat_0_trunc, um_feat_0_trunc, dim=-1)
        print('feat_um_sim: \n', feat_um_sim)

        proj_um_sim = F.cosine_similarity(proj_feat_0_trunc, um_feat_0_trunc, dim=-1)
        print('proj_um_sim: \n', proj_um_sim)

        feat_1_trunc = features[1, :trunc_len]
        feat_cross_sim = F.cosine_similarity(feat_0_trunc, feat_1_trunc, dim=-1)
        print('feat_cross_sim: \n', feat_cross_sim)

    def extract_features(self, source, audio_lengths, mask=False):
        padding_mask = create_padding_mask(audio_lengths, max_len=source.shape[1])
        return self(source=source, padding_mask=padding_mask, mask=mask, features_only=True)

    def remove_pretraining_modules(self, use_teacher_encoder=False):
        self.projector = None
        if use_teacher_encoder:
            print('use target feature encoder!', flush=True)
            self.feature_encoder.load_state_dict(self.target_feature_encoder.state_dict())
        self.target_feature_encoder = None
        self.target_projector = None
        self.predictor = None
        self.quantizer = None
        self.project_q = None

        self.pitch_estimator = None
        self.reconstructor = None

    def _update_quantizer_temp(self, global_step):
        if self.quantize_targets:
            self.quantizer.set_num_updates(global_step)

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

        neg_idxs = torch.cat(neg_idxs_l)
        assert neg_idxs.ndim == 1

        negs = y[neg_idxs]
        negs = negs.view(bsz, sum(nums), self.n_negatives + self.cross_sample_negatives, fsz).permute(
            2, 0, 1, 3
        )  # to NxBxTxC
        return negs, neg_idxs


def create_padding_mask(audio_lengths, max_len):
    # Broadcast to vectorize creating the padding mask
    padding_mask = torch.arange(max_len, device=audio_lengths.device)
    padding_mask = padding_mask.expand(len(audio_lengths), max_len) < audio_lengths.unsqueeze(1)
    # Negate to false where no padding
    padding_mask = ~padding_mask
    return padding_mask


class RandomShift:
    def __init__(self, cfg: ShiftPerturbConfig):
        self.dist = cfg.dist
        if self.dist == 'uniform':
            assert isinstance(cfg.max, int) and isinstance(cfg.min, int)
            self.min = cfg.min
            self.max = cfg.max
        else:
            assert cfg.dist == 'rounded_normal'
            assert isinstance(cfg.mean, float) and isinstance(cfg.std, float)
            self.mean = cfg.mean
            self.std = cfg.std
        self.max_ratio = cfg.max_ratio
        assert isinstance(cfg.unit, int)
        self.unit = cfg.unit
        self.shift_prob = cfg.shift_prob
        self.truncate = cfg.truncate

    def shift(self, inputs, inputs_len, mask_emb):
        if np.random.random() >= self.shift_prob:
            return inputs, inputs_len, 0, 0, 0

        shift_num, shift_num_units, r_shift_num, r_shift_num_units = self.get_shift_num(inputs_len.min())

        if self.truncate and shift_num > 0 and r_shift_num > 0:
            r_shift_num = 0
            r_shift_num_units = 0

        orig_inputs_t = inputs.shape[1]

        if shift_num_units > 0:
            inputs = torch.nn.functional.pad(inputs, (0, 0, shift_num_units, 0))
            inputs[:, :shift_num_units] = mask_emb
            inputs_len = inputs_len + shift_num_units
        elif shift_num_units < 0:
            abs_shift_num_units = abs(shift_num_units)
            inputs = inputs[:, abs_shift_num_units:]
            inputs_len = inputs_len - abs_shift_num_units

        if r_shift_num_units > 0:
            inputs = torch.nn.functional.pad(inputs, (0, 0, 0, r_shift_num_units))
            shift_padding_mask = create_shift_padding_mask(inputs_len, inputs.shape[1], r_shift_num_units)
            inputs[shift_padding_mask] = mask_emb
            inputs_len = inputs_len + r_shift_num_units
        elif r_shift_num_units < 0:
            shift_padding_mask = create_shift_padding_mask(inputs_len, inputs.shape[1], r_shift_num_units)
            inputs[shift_padding_mask] = 0.0
            abs_shift_num_units = abs(r_shift_num_units)
            inputs_len = inputs_len - abs_shift_num_units
            inputs = inputs[:, :-abs_shift_num_units]

        inputs_t_diff = inputs.shape[1] - orig_inputs_t
        if self.truncate and inputs_t_diff > 0:
            truncated_r_shift_num = r_shift_num - int(inputs_t_diff / self.unit)
            assert truncated_r_shift_num == -shift_num
            inputs = inputs[:, :-inputs_t_diff]
            inputs_len = inputs_len - inputs_t_diff
        else:
            truncated_r_shift_num = r_shift_num

        return inputs, inputs_len, shift_num, r_shift_num, truncated_r_shift_num

    def get_shift_num(self, total_units_num):
        if self.dist == 'uniform':
            shift_num = np.random.randint(self.min, self.max + 1)
            r_shift_num = np.random.randint(self.min, self.max + 1)
        else:
            shift_num = np.random.normal(loc=self.mean, scale=self.std)
            shift_num = int(round(shift_num))
            r_shift_num = np.random.normal(loc=self.mean, scale=self.std)
            r_shift_num = int(round(r_shift_num))

        max_num = int(total_units_num * self.max_ratio / self.unit)
        if shift_num > max_num:
            if self.truncate:
                shift_num = max_num
        elif shift_num < -max_num:
            shift_num = -max_num

        if r_shift_num < 0:
            if shift_num > 0:
                r_shift_num = max(-max_num, r_shift_num)
            else:
                r_shift_num = max(-(max_num - abs(shift_num)), r_shift_num)

        return shift_num, shift_num * self.unit, r_shift_num, r_shift_num * self.unit


def create_shift_padding_mask(lengths, max_len, shift_num_units):
    positions = torch.arange(max_len, device=lengths.device)
    positions.expand(len(lengths), max_len)
    shift_audio_lengths = lengths + shift_num_units
    if shift_num_units > 0:
        padding_mask = (positions >= lengths.unsqueeze(1)) & (positions < shift_audio_lengths.unsqueeze(1))
    else:
        padding_mask = (positions >= shift_audio_lengths.unsqueeze(1)) & (positions < lengths.unsqueeze(1))
    return padding_mask


class FeatST2VecEncoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.wav2spec = Serialization.from_config_dict(cfg.preprocessor)

        self.feature_encoder = Serialization.from_config_dict(cfg.feature_encoder)
        self.target_momentum = cfg.target_momentum
        self.target_update_step = 0
        if self.target_momentum > 0:
            self.target_feature_encoder = Serialization.from_config_dict(cfg.feature_encoder)
            self.target_feature_encoder.load_state_dict(self.feature_encoder.state_dict())
            for p in self.target_feature_encoder.parameters():
                p.requires_grad = False
        else:
            self.target_feature_encoder = None

        context_net_dim = cfg.context_net.encoder.embedding_dim
        self.feature_proj = nn.Linear(self.feature_encoder.output_dim, context_net_dim)

        self.context_net = TransformerEncoder(cfg.context_net)

        self.mask_cfg = cfg.masking
        self.target_mask_cfg = cfg.target_masking

        if self.mask_cfg.mask_emb_type == 'zero':
            self.mask_emb = 0.0
        else:
            assert self.mask_cfg.mask_emb_type == 'gaussian'
            num_features = cfg.preprocessor.features
            self.register_buffer("mask_emb", torch.tensor(GAUSSIAN_MASK[:num_features]))

        self.n_negatives = cfg.n_negatives
        self.cross_sample_negatives = cfg.cross_sample_negatives
        self.codebook_negatives = cfg.codebook_negatives
        self.negatives_from_everywhere = cfg.negatives_from_everywhere
        self.negatives_from_noisy_features = cfg.negatives_from_noisy_features

        cfg.predictor.input_dim = context_net_dim
        self.predictor = None if cfg.predictor is None else Projector(cfg.predictor)

    def forward(self, wavs, wav_lens, *, mask=True, features_only=False, global_step=None) -> tuple:
        specs, specs_len = self.wav2spec(
            input_signal=wavs, length=wav_lens,
        )

        specs_mask = self._create_padding_mask(specs_len, specs.shape[1]) if mask else None

        if features_only:
            unmasked_features = None
            padding_mask = None
            features_mask = None
        else:
            unmasked_specs = specs.clone()
            if self.training and self.target_mask_cfg is not None:
                unmasked_specs = unmasked_specs.transpose(1, 2)
                unmasked_specs, _, _ = apply_mask(self.target_mask_cfg, unmasked_specs, specs_mask, self.mask_emb)
                unmasked_specs = unmasked_specs.transpose(1, 2)
            if self.target_momentum > 0:
                assert global_step is not None
                if global_step > self.target_update_step:
                    ema_update(self.target_feature_encoder, self.feature_encoder, self.target_momentum)
                    self.target_update_step = global_step
                target_feature_encoder = self.target_feature_encoder
            else:
                target_feature_encoder = self.feature_encoder
            with torch.no_grad():
                with as_eval(target_feature_encoder):
                    unmasked_features, unmasked_feature_lens, _ = target_feature_encoder(unmasked_specs, specs_len)
                    # [B, D, T] => [B, T, D]
                    unmasked_features = unmasked_features.transpose(1, 2)

                    padding_mask = self._create_padding_mask(unmasked_feature_lens, unmasked_features.shape[1])
                    features_mask = ~padding_mask
                    unmasked_features = unmasked_features[features_mask]
                    # fake batch dim 1
                    unmasked_features = unmasked_features.view(1, -1, unmasked_features.size(-1))

        if mask:
            specs = specs.transpose(1, 2)
            specs, _, _ = apply_mask(self.mask_cfg, specs, specs_mask, self.mask_emb)
            specs = specs.transpose(1, 2)

        features, feature_lens, _ = self.feature_encoder(specs, specs_len)
        # [B, D, T] => [B, T, D]
        features = features.transpose(1, 2)

        if self.feature_proj is not None:
            features = self.feature_proj(features)

        features = self.context_net(features, padding_mask=padding_mask)

        if features_only:
            return features, feature_lens

        assert mask

        if self.predictor is not None:
            features = self.predictor(features, length=feature_lens)

        features = features[features_mask]
        # fake batch dim to 1
        features = features.view(1, -1, features.size(-1))

        assert not self.negatives_from_everywhere
        if self.negatives_from_noisy_features:
            sampled_negatives, _ = self.sample_negatives_flat(features, feature_lens.tolist())
        else:
            with torch.no_grad():
                sampled_negatives, _ = self.sample_negatives_flat(unmasked_features, feature_lens.tolist())

        prob_ppl_loss, cur_temp, prob_ppl = None, None, None
        return features, unmasked_features, sampled_negatives, padding_mask, prob_ppl_loss, cur_temp, prob_ppl

    def extract_features(self, source, audio_lengths, mask=False):
        padding_mask = self._create_padding_mask(audio_lengths, max_len=source.shape[1])
        return self(source=source, padding_mask=padding_mask, mask=mask, features_only=True)

    def remove_pretraining_modules(self):
        self.target_feature_encoder = None
        self.predictor = None

    def _update_quantizer_temp(self, global_step):
        if self.quantize_targets:
            self.quantizer.set_num_updates(global_step)

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

        neg_idxs = torch.cat(neg_idxs_l)
        assert neg_idxs.ndim == 1

        negs = y[neg_idxs]
        negs = negs.view(bsz, sum(nums), self.n_negatives + self.cross_sample_negatives, fsz).permute(
            2, 0, 1, 3
        )  # to NxBxTxC
        return negs, neg_idxs

    def _create_padding_mask(self, audio_lengths, max_len):
        # Broadcast to vectorize creating the padding mask
        padding_mask = torch.arange(max_len, device=audio_lengths.device)
        padding_mask = padding_mask.expand(len(audio_lengths), max_len) < audio_lengths.unsqueeze(1)
        # Negate to false where no padding
        padding_mask = ~padding_mask
        return padding_mask


def apply_mask(mask_cfg, x, padding_mask, mask_emb, mask_positions=None):
    B, T, C = x.shape
    if mask_cfg.mask_prob > 0:
        mask_indices, mask_num = compute_mask_indices(
            (B, T),
            padding_mask,
            mask_cfg.mask_prob,
            mask_cfg.mask_length,
            mask_cfg.mask_type,
            mask_cfg.mask_other,
            min_masks=2,
            no_overlap=mask_cfg.no_mask_overlap,
            min_space=mask_cfg.mask_min_space,
            shrink_to_batch_min=mask_cfg.mask_shrink_to_batch_min,
            mask_positions=mask_positions
        )
        mask_indices = torch.from_numpy(mask_indices).to(x.device)
        if isinstance(mask_emb, torch.Tensor):
            mask_emb = mask_emb.type_as(x)
        x[mask_indices] = mask_emb
        assert len(mask_num) == B
    else:
        mask_indices = None
        mask_num = None

    if mask_cfg.mask_channel_prob > 0:
        # assert mask_cfg.mask_shrink_to_batch_min
        mask_channel_indices, _ = compute_mask_indices(
            (B, C),
            None,
            mask_cfg.mask_channel_prob,
            mask_cfg.mask_channel_length,
            mask_cfg.mask_channel_type,
            mask_cfg.mask_channel_other,
            no_overlap=mask_cfg.no_mask_channel_overlap,
            min_space=mask_cfg.mask_channel_min_space,
            shrink_to_batch_min=mask_cfg.mask_channel_shrink_to_batch_min,
        )
        mask_channel_indices = torch.from_numpy(mask_channel_indices).to(x.device).unsqueeze(1).expand(-1, T, -1)
        x[mask_channel_indices] = 0

    return x, mask_indices, mask_num


def ema_update(ema_module, new_module, m):
    with torch.no_grad():
        for param_q, param_k in zip(new_module.parameters(), ema_module.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)


@contextlib.contextmanager
def as_eval(*modules):
    training_states = []
    for module_i in modules:
        training_states.append(module_i.training)
        module_i.eval()

    try:
        yield
    finally:
        for module_i, training_state_i in zip(modules, training_states):
            module_i.train(training_state_i)


def momentum_scheduler(base_value, final_value, max_steps, *, type):
    if type == 'linear':
        def linear_scheduler(step):
            if step <= max_steps:
                cur_value = base_value + (final_value - base_value) * (step / max_steps)
            else:
                cur_value = final_value
            return cur_value

        return linear_scheduler
    elif type == 'cosine':
        def cosine_scheduler(step):
            if step <= max_steps:
                cur_value = final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * step / max_steps))
            else:
                cur_value = final_value
            return cur_value

        return cosine_scheduler
    else:
        raise ValueError('unknown scheduler type: {}'.format(type))


def get_state_dict(
        module_param_prefix,
        checkpoint_path,
        *,
        map_location=None,
        strict: bool = True):
    from pytorch_lightning.utilities.cloud_io import load as pl_load
    if map_location is not None:
        checkpoint = pl_load(checkpoint_path, map_location=map_location)
    else:
        checkpoint = pl_load(checkpoint_path, map_location=lambda storage, loc: storage)

    # load the state_dict on the model automatically
    if module_param_prefix is not None:
        module_state_dict = {k[len(module_param_prefix):]: v for k, v in checkpoint['state_dict'].items() if
                             k.startswith(module_param_prefix)}
    else:
        module_state_dict = checkpoint['state_dict']
    return module_state_dict


class Reconstrutor(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.use_teacher_feat_prob = cfg.use_teacher_feat_prob

        self.quantizer = GumbelVectorQuantizer(
            dim=cfg.reconstructor.input_dim,
            num_vars=cfg.quantizer.latent_vars,
            temp=cfg.quantizer.latent_temp,
            groups=cfg.quantizer.latent_groups,
            combine_groups=False,
            vq_dim=cfg.quantizer.latent_dim,
            time_first=True,
        )
        cfg.reconstructor.input_dim = cfg.quantizer.latent_dim
        self.decoder = Projector(cfg.reconstructor)
        self.reduction_rate = cfg.reduction_rate
        self.loss_type = cfg.loss_type

        if cfg.global_cond_extractor is not None:
            self.global_cond_extractor = Projector(cfg.global_cond_extractor)
        else:
            self.global_cond_extractor = None

    def _update_quantizer_temp(self, global_step):
        self.quantizer.set_num_updates(global_step)

    def forward(self, h, h_len, t_h, t_h_len, target, target_len, global_step):
        use_teacher_feat = False
        if self.training and self.use_teacher_feat_prob > 0 and np.random.random() < self.use_teacher_feat_prob:
            h = t_h
            h_len = t_h_len
            use_teacher_feat = True

        # [B, D, T] => [B, T, D]
        target = target.transpose(1, 2)

        if self.global_cond_extractor is not None:
            global_cond = self.get_global_cond(target, target_len)
        else:
            global_cond = None

        self.quantizer.set_num_updates(global_step)
        h, prob_ppl_loss, cur_temp, prob_ppl = self.quantizer(h)

        if self.global_cond_extractor is not None:
            h = h + global_cond.unsqueeze(1)

        pred = self.decoder(h, length=h_len)

        B, T, C = pred.size()
        assert C // self.reduction_rate == target.shape[2]
        pred = pred.reshape(B, T * self.reduction_rate, C // self.reduction_rate)
        pred_lens = h_len * self.reduction_rate

        recon = pred
        recon_lens = pred_lens
        sample = recon, recon_lens, target, target_len

        import random
        if random.random() < 0.0005:
            torch.set_printoptions(profile="full")
            print('\n use teacher feat: {}'.format(use_teacher_feat))
            print('tgtt:', target[0][180:260])
            print('pred:', pred[0][180:260])
            torch.set_printoptions(profile="default")  # reset

        if pred.shape[1] > target.shape[1]:
            pred = pred.narrow(dim=1, start=0, length=target.shape[1]).contiguous()
        elif pred.shape[1] < target.shape[1]:
            target = target.narrow(dim=1, start=0, length=pred.shape[1]).contiguous()

        loss_mask = create_padding_mask(pred_lens, pred.shape[1])
        loss_mask = ~loss_mask

        pred = pred[loss_mask]
        target = target[loss_mask]
        mean = True
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")
        return {'loss': loss,
                'recon': sample,
                'quant_ppl_loss': prob_ppl_loss,
                'quant_temp': cur_temp,
                'quant_ppl': prob_ppl}

    def get_quantized_feat(self, feat, feat_len):
        q, q_ids = self.quantizer(feat, quant_only=True)
        return q, q_ids

    def get_global_cond(self, input, input_len):
        cond = self.global_cond_extractor(input, length=input_len)
        pad_mask = create_padding_mask(input_len, input.shape[1])
        cond = cond.masked_fill(pad_mask.unsqueeze(2), 0.0)
        cond = torch.sum(cond, axis=1)
        cond = cond / input_len.unsqueeze(1)
        return cond


def stop_grad(module):
    for p in module.parameters():
        p.requires_grad = False
