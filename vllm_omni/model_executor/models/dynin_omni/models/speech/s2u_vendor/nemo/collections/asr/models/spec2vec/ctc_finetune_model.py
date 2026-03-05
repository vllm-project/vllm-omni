import contextlib
import torch

from nemo.core import Serialization


class CTCFinetuneModel(torch.nn.Module):
    def __init__(self, encoder, cfg):
        super().__init__()

        self.encoder = encoder

        self.decoder = Serialization.from_config_dict(cfg.decoder)

        self.freeze_finetune_updates = cfg.freeze_finetune_updates

    def forward(self, input_signal, input_signal_length, global_step):
        ft = False if global_step is None else self.freeze_finetune_updates <= global_step
        with torch.no_grad() if not ft else contextlib.suppress():
            encoded, encoded_len = self.encoder(input_signal, input_signal_length, None, None,
                                                   mask=self.training, features_only=True)

        # [B, T, D] => [B, D, T]
        encoded = encoded.transpose(1, 2)

        # Ensure that shape mismatch does not occur due to padding
        # Due to padding and subsequent downsampling, it may be possible that
        # max sequence length computed does not match the actual max sequence length
        max_output_len = encoded_len.max()
        if encoded.shape[2] != max_output_len:
            encoded = encoded.narrow(dim=2, start=0, length=max_output_len).contiguous()

        logits, encoded_len = self.decoder(encoder_output=encoded, lens=encoded_len, log_prob=False)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # with torch.no_grad():
        #     greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)

        return log_probs, encoded_len, logits

