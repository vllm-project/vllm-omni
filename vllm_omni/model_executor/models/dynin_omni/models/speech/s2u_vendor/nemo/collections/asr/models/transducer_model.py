import torch

from htrain.util import instantiate_from_config


class TransducerModel(torch.nn.Module):

    def __init__(self, encoder, decoder, joint):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.joint = joint
        assert not self.joint.fuse_loss_wer

    @classmethod
    def from_cfg(cls, cfg):
        encoder = instantiate_from_config(cfg.encoder)
        decoder = instantiate_from_config(cfg.decoder)
        joint = instantiate_from_config(cfg.joint)
        return cls(encoder, decoder, joint)

    def forward(self, signal, signal_len, transcript, transcript_len):
        encoded, encoded_len, extra = self.encoder(audio_signal=signal, length=signal_len)

        decoder_h, target_length = self.decoder(targets=transcript, target_length=transcript_len)

        if not self.joint.fuse_loss_wer:
            joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder_h)
            return joint, target_length, encoded, encoded_len
        else:
            # Fused joint step
            loss_value, _, _, _ = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoder_h,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=transcript_len,
                compute_wer=False,
            )
            return loss_value
