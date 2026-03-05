import torch

from nemo.collections.asr.parts.convolution_layers import ConvBlock


class ConvAutoencoder(torch.nn.Module):

    def __init__(self, encoder, decoder, loss_type='l2'):
        super().__init__()

        self.encoder = ConvBlock(**encoder)
        self.decoder = ConvBlock(**decoder)
        assert loss_type == 'l2'
        self.loss_type = loss_type

    def forward(self, inputs, inputs_len):
        # inputs: [B, C, T]
        encoded = self.encode(inputs, inputs_len)
        reconstructed = self.decode(encoded, inputs_len)

        loss_mask = seq_mask(inputs_len, inputs.shape[2])
        mse_loss = torch.nn.functional.mse_loss(inputs.transpose(1, 2)[loss_mask],
                                                reconstructed.transpose(1, 2)[loss_mask])

        return mse_loss, reconstructed

    def encode(self, inputs, inputs_len):
        encoded, _ = self.encoder(inputs, inputs_len)
        return encoded

    def decode(self, encoded, inputs_len):
        decoded, _ = self.decoder(encoded, inputs_len)
        return decoded


def seq_mask(audio_lengths, max_len):
    # Broadcast to vectorize creating the padding mask
    padding_mask = torch.arange(max_len, device=audio_lengths.device)
    padding_mask = padding_mask.expand(len(audio_lengths), max_len) < audio_lengths.unsqueeze(1)
    return padding_mask
