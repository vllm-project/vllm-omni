import contextlib

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn

from nemo.collections.asr.modules.wav2vec_modules import GumbelVectorQuantizer, compute_mask_indices
from nemo.collections.asr.parts.wav2vec import Wav2VecTransformerEncoder, TransformerEncoder
from nemo.core.classes.common import Serialization


def buffered_arange(max):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]


class Spec2VecEncoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.noisy_spec2vec = cfg.noisy_spec2vec
        self.wav2spec = Serialization.from_config_dict(cfg.preprocessor)
        self.feature_encoder = Serialization.from_config_dict(cfg.feature_encoder)
        self.freeze_feature_encoder = cfg.freeze_feature_encoder

        self.mask_cfg = cfg.masking

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.quantizer = None

        self.n_negatives = cfg.n_negatives
        self.cross_sample_negatives = cfg.cross_sample_negatives
        self.codebook_negatives = cfg.codebook_negatives
        self.negatives_from_everywhere = cfg.negatives_from_everywhere

        self.final_dim = cfg.final_dim
        assert self.final_dim > 0
        self.quantize_targets = cfg.quantizer.quantize_targets
        if self.quantize_targets:
            assert cfg.quantizer.targets_bottleneck_dim is None
            vq_dim = cfg.quantizer.latent_dim if cfg.quantizer.latent_dim > 0 else self.final_dim
            self.quantizer = GumbelVectorQuantizer(
                dim=self.feature_encoder.output_dim,
                num_vars=cfg.quantizer.latent_vars,
                temp=cfg.quantizer.latent_temp,
                groups=cfg.quantizer.latent_groups,
                combine_groups=False,
                vq_dim=vq_dim,
                time_first=True,
            )
            self.project_q = nn.Linear(vq_dim, self.final_dim)
        else:
            targets_bottleneck_dim = cfg.quantizer.targets_bottleneck_dim
            if targets_bottleneck_dim is None:
                self.project_q = nn.Linear(self.feature_encoder.output_dim, self.final_dim)
            else:
                act_fn_dic = {'relu': nn.ReLU, 'gelu': nn.GELU}
                targets_proj_act_fn = cfg.quantizer.targets_bottleneck_act_fn
                targets_proj_layers = (
                    [nn.Linear(self.feature_encoder.output_dim, targets_bottleneck_dim)]
                    + ([] if targets_proj_act_fn is None else [act_fn_dic[targets_proj_act_fn]()])
                    + [nn.Dropout(cfg.quantizer.targets_bottleneck_dropout)]
                    + [nn.Linear(targets_bottleneck_dim, self.final_dim)]

                )
                self.project_q = torch.nn.Sequential(*targets_proj_layers)

        self.targets_grad_update_inverval = cfg.targets_grad_update_inverval
        self.grad_step_count = 0

        encoder_embed_dim = cfg.transformer_encoder.encoder.embedding_dim
        if cfg.learnable_mask:
            if self.noisy_spec2vec:
                mask_emb_dim = cfg.preprocessor.features
            else:
                mask_emb_dim = encoder_embed_dim
            self.mask_emb = nn.Parameter(torch.FloatTensor(mask_emb_dim).uniform_())
        else:
            self.mask_emb = 0.0

        if cfg.transformer_encoder.use_pytorch_transformer:
            self.encoder = Wav2VecTransformerEncoder(cfg.transformer_encoder)
        else:
            self.encoder = TransformerEncoder(cfg.transformer_encoder)

        if cfg.final_hidden_dim is None:
            self.final_proj = nn.Linear(encoder_embed_dim, self.final_dim)
        else:
            self.final_proj = torch.nn.Sequential(
                nn.Linear(encoder_embed_dim, cfg.final_hidden_dim),
                nn.Dropout(cfg.dropout_final),
                nn.ReLU(),
                nn.Linear(cfg.final_hidden_dim, self.final_dim)
            )

    def forward(self, wavs, wav_lens, *, mask=True, features_only=False) -> tuple:
        if self.noisy_spec2vec:
            return self.noisy_spec2vec_forward(wavs, wav_lens, mask=mask, features_only=features_only)
        else:
            return self.spec2vec_forward(wavs, wav_lens, mask=mask, features_only=features_only)

    def spec2vec_forward(self, wavs, wav_lens, *, mask=True, features_only=False) -> tuple:
        specs, specs_len = self.wav2spec(
            input_signal=wavs, length=wav_lens,
        )

        if self.freeze_feature_encoder:
            self.feature_encoder.bn_eval()
        with torch.no_grad() if self.freeze_feature_encoder else contextlib.suppress():
            features, feature_lens, _ = self.feature_encoder(specs, specs_len)
        # [B, D, T] => [B, T, D]
        features = features.transpose(1, 2)

        unmasked_features = None if features_only else features.clone()

        padding_mask = self._create_padding_mask(feature_lens, features.shape[1])
        assert padding_mask.size(1) == features.size(1)
        assert padding_mask.ndim == 2

        features = self.dropout_input(features)
        if not features_only:
            unmasked_features = self.dropout_features(unmasked_features)

        if mask and self.mask_cfg is not None:
            logits, mask_indices, mask_num = self.apply_mask(features, padding_mask)
            if features_only:
                targets = None
            elif mask_indices is not None:
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
            targets = None if features_only else unmasked_features
            mask_indices = None
            mask_num = None

        logits = self.encoder(logits, padding_mask=padding_mask)

        if features_only:
            return logits, feature_lens

        prob_ppl_loss, cur_temp = None, None
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

        mask_logits = self.final_proj(mask_logits)

        return mask_logits, targets, sampled_negatives, padding_mask, prob_ppl_loss, cur_temp, prob_ppl

    def noisy_spec2vec_forward(self, wavs, wav_lens, *, mask=True, features_only=False) -> tuple:
        specs, specs_len = self.wav2spec(
            input_signal=wavs, length=wav_lens,
        )

        unmasked_specs = None if features_only else specs.clone()

        if mask:
            specs = specs.transpose(1, 2)
            specs_mask = self._create_padding_mask(specs_len, specs.shape[1])
            mask_positions = []
            specs, _, _ = self.apply_mask(specs, specs_mask, mask_positions=mask_positions)
            specs = specs.transpose(1, 2)
        else:
            mask_positions = None

        if self.freeze_feature_encoder:
            self.feature_encoder.bn_eval()
        with torch.no_grad() if self.freeze_feature_encoder else contextlib.suppress():
            features, feature_lens, _ = self.feature_encoder(specs, specs_len)
        # [B, D, T] => [B, T, D]
        features = features.transpose(1, 2)

        if features_only:
            unmasked_features = None
        else:
            no_targets_grad = False
            if self.training:
                self.grad_step_count += 1
                if self.targets_grad_update_inverval == 0 or (self.grad_step_count % self.targets_grad_update_inverval != 0):
                    no_targets_grad = True
            if self.targets_grad_update_inverval == 1:
                assert not no_targets_grad

            with torch.no_grad() if no_targets_grad else contextlib.suppress():
                with as_eval(self.feature_encoder):
                    unmasked_features, _, _ = self.feature_encoder(unmasked_specs, specs_len)
            unmasked_features = unmasked_features.transpose(1, 2)

        padding_mask = self._create_padding_mask(feature_lens, features.shape[1])
        assert padding_mask.size(1) == features.size(1)
        assert padding_mask.ndim == 2

        features = self.dropout_input(features)
        if not features_only:
            unmasked_features = self.dropout_features(unmasked_features)

        if mask and not features_only:
            # positions to lens
            mask_positions = np.array(mask_positions) + 1
            mask_positions = self.feature_encoder.get_subsampled_lens(mask_positions)
            # lens to positions
            mask_positions = mask_positions - 1
            mask_indices = np.full((unmasked_features.shape[0], unmasked_features.shape[1]), False)
            mask_num = []
            for i, mask_position_i in enumerate(mask_positions):
                mask_position_i = np.unique(mask_position_i)
                mask_num.append(mask_position_i.shape[0])
                mask_indices[i, mask_position_i] = True
            mask_indices = torch.from_numpy(mask_indices).to(unmasked_features.device)
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
        else:
            mask_indices = None
            mask_num = None
            targets = None

        logits = self.encoder(features, padding_mask=padding_mask)

        if features_only:
            return logits, feature_lens

        prob_ppl_loss, cur_temp = None, None
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

        mask_logits = self.final_proj(mask_logits)

        return mask_logits, targets, sampled_negatives, padding_mask, prob_ppl_loss, cur_temp, prob_ppl

    def extract_features(self, source, audio_lengths, mask=False):
        padding_mask = self._create_padding_mask(audio_lengths, max_len=source.shape[1])
        return self(source=source, padding_mask=padding_mask, mask=mask, features_only=True)

    def remove_pretraining_modules(self):
        self.quantizer = None
        self.project_q = None
        self.final_proj = None
        self.dropout_features = None

    def _update_quantizer_temp(self, global_step):
        if self.quantize_targets:
            self.quantizer.set_num_updates(global_step)

    def apply_mask(self, x, padding_mask, mask_positions=None):
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
                mask_positions=mask_positions
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            mask_emb = self.mask_emb
            if isinstance(mask_emb, torch.Tensor):
                mask_emb = mask_emb.type_as(x)
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
                shrink_to_batch_min=self.mask_cfg.mask_shrink_to_batch_min,
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


@contextlib.contextmanager
def as_eval(module):
    training_state = module.training
    module.eval()

    try:
        yield
    finally:
        module.train(training_state)
