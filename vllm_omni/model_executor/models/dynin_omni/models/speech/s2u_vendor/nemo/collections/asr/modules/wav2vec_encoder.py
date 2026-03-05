import torch
from torch import nn

from nemo.collections.asr.modules.wav2vec_modules import GumbelVectorQuantizer, compute_mask_indices
from nemo.collections.asr.parts.wav2vec import ConvFeatureEncoder, Wav2VecTransformerEncoder, GradMultiply, \
    TransformerEncoder


def buffered_arange(max):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]


class Wav2VecEncoderModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

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
            assert cfg.ctc_loss.prob_ppl_weight == 0
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

        features_penalty = None if features_only else features[~padding_mask].float().pow(2).mean()  # L2 Norm on features

        features = self.layer_norm(features)
        unmasked_features = None if features_only else features.clone()

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        if not features_only:
            unmasked_features = self.dropout_features(unmasked_features)

        assert self.input_quantizer is None
        # if self.input_quantizer:
        #     features, prob_ppl_loss, cur_codebook_temp = self.input_quantizer(features)
        #     features = self.project_inp(features)
        if mask:
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
        self.dropout_features = None
        self.input_quantizer = None
        self.project_q = None
        self.project_inp = None
        self.target_glu = None

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
