import contextlib

import torch

from nemo.collections.asr.models.st2vec.st2vec_model import apply_mask, create_padding_mask


class FLAPTransducerModel(torch.nn.Module):

    def __init__(self, text_projector, speech_pre_encoder, speech_backbone_transducer, *, auxiliary_decoder=None, latent_masking_cfg=None,
                 use_dpm_text_projector=False):
        super().__init__()

        self.use_dpm_text_projector = use_dpm_text_projector
        self.text_projector = text_projector
        self.speech_pre_encoder = speech_pre_encoder
        self.speech_backbone_transducer = speech_backbone_transducer
        self.auxiliary_decoder = auxiliary_decoder
        if latent_masking_cfg:
            self.mask_emb = torch.nn.Parameter(torch.FloatTensor(latent_masking_cfg.mask_emb_dim).uniform_())
            self.text_masking = latent_masking_cfg.text_masking
            self.speech_masking = latent_masking_cfg.speech_masking
        else:
            self.mask_emb = None
            self.text_masking = None
            self.speech_masking = None

    def forward(self, *, text, text_len, speech, speech_len, transcript, transcript_len, mask=True, freeze_auxiliary_decoder=False):
        if text is not None:
            assert speech is None
            if self.use_dpm_text_projector:
                latent, latent_len = self.text_projector(text, text_len)
            else:
                latent, latent_len, _ = self.text_projector(text, text_len, match_output_len=False)
            unmasked_latent = latent

            if self.text_masking and mask:
                if self.auxiliary_decoder is not None:
                    unmasked_latent = latent.clone()

                latent = latent.transpose(1, 2)
                latent_mask = create_padding_mask(latent_len, latent.shape[1])
                latent, _, _ = apply_mask(self.text_masking, latent, latent_mask, self.mask_emb)
                latent = latent.transpose(1, 2)
        else:
            latent, latent_len, _ = self.speech_pre_encoder(audio_signal=speech, length=speech_len, match_output_len=False)
            unmasked_latent = latent

            if self.speech_masking and mask:
                if self.auxiliary_decoder is not None:
                    unmasked_latent = latent.clone()

                latent = latent.transpose(1, 2)
                latent_mask = create_padding_mask(latent_len, latent.shape[1])
                latent, _, _ = apply_mask(self.speech_masking, latent, latent_mask, self.mask_emb)
                latent = latent.transpose(1, 2)

        results = self.speech_backbone_transducer(latent, latent_len, transcript, transcript_len)

        extra = {}
        if self.auxiliary_decoder is not None:
            with torch.no_grad() if freeze_auxiliary_decoder else contextlib.suppress():
                aux_log_prob, aux_log_prob_len = self.auxiliary_decoder(encoder_output=unmasked_latent, lens=latent_len)
            extra['aux_decoder_log_probs'] = aux_log_prob
            extra['aux_decoder_log_probs_len'] = aux_log_prob_len

        return results + (extra,)
