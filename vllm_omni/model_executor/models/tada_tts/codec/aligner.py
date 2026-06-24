"""Vendored TADA aligner (from MIT-licensed hume-tada), adapted for OFFLINE local use.

Forced-alignment of text tokens to reference audio via a wav2vec2-large CTC encoder
+ a DP alignment. Used ONLY offline by the example to build a voice-cloning prompt
(it is never imported by the serving worker). Differences vs. upstream:
  * the wav2vec2 config is reconstructed in-code (no ``facebook/wav2vec2-large`` fetch);
  * the tokenizer is loaded from a caller-supplied local path (no ``meta-llama`` fetch);
  * weights are loaded from a local ``aligner/`` safetensors directory.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torchaudio
from transformers import Wav2Vec2Config, Wav2Vec2ForCTC


def _align_text_tokens(probs: torch.Tensor, text_tokens: torch.Tensor) -> list[int]:
    """DP forced-alignment: assign each text token the audio frame maximising total prob.

    Verbatim from upstream ``tada/modules/aligner.py``. ``probs`` is [L, V]
    (L time frames, V vocab); ``text_tokens`` is [T]. Returns T frame positions.
    """
    L, V = probs.shape
    T = len(text_tokens)
    device = probs.device

    F = torch.full((L, T), -float("inf"), device=device)
    backpointer = torch.zeros(L, T, dtype=torch.long, device=device)

    token_probs = probs[:, text_tokens]  # (L, T)

    cummax_first = torch.cummax(token_probs[:, 0], dim=0)
    F[:, 0] = cummax_first.values
    backpointer[:, 0] = cummax_first.indices

    if T <= L:
        diag_indices = torch.arange(T, device=device)
        F[diag_indices, diag_indices] = torch.cumsum(token_probs[diag_indices, diag_indices], dim=0)
        backpointer[diag_indices, diag_indices] = diag_indices

    for i in range(1, L):
        max_j = min(i, T)
        if max_j <= 1:
            continue
        j_range = torch.arange(1, max_j, device=device)
        skip_scores = F[i - 1, j_range]
        use_scores = F[i - 1, j_range - 1] + token_probs[i, j_range]
        use_better = use_scores >= skip_scores
        F[i, j_range] = torch.where(use_better, use_scores, skip_scores)
        backpointer[i, j_range] = torch.where(use_better, i, -1)

    positions = torch.zeros(T, dtype=torch.long, device=device)
    i, j = L - 1, T - 1
    pos_idx = T - 1
    while j >= 0:
        if j == 0:
            positions[pos_idx] = backpointer[i, j]
            break
        elif backpointer[i, j] == -1:
            i -= 1
        else:
            positions[pos_idx] = backpointer[i, j]
            pos_idx -= 1
            i -= 1
            j -= 1
    return positions.tolist()


@dataclass
class AlignOutput:
    token_positions: torch.Tensor
    token_masks: torch.Tensor
    logits: torch.Tensor | None = None


def _wav2vec2_large_config(vocab_size: int) -> Wav2Vec2Config:
    """Reconstruct the ``facebook/wav2vec2-large`` config (derived from the local
    aligner weight shapes), so no network fetch is needed offline."""
    return Wav2Vec2Config(
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        feat_extract_norm="group",
        do_stable_layer_norm=False,
        conv_dim=[512, 512, 512, 512, 512, 512, 512],
        conv_stride=[5, 2, 2, 2, 2, 2, 2],
        conv_kernel=[10, 3, 3, 3, 3, 2, 2],
        num_conv_pos_embeddings=128,
        num_conv_pos_embedding_groups=16,
        vocab_size=vocab_size,
    )


class Aligner(torch.nn.Module):
    """wav2vec2-large CTC encoder + DP forced-alignment (offline, local weights)."""

    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        cfg = _wav2vec2_large_config(vocab_size=len(tokenizer))
        self.encoder_config = cfg
        self.encoder = Wav2Vec2ForCTC(cfg)

    @classmethod
    def from_local(cls, codec_path: str, tokenizer, device: torch.device | str = "cpu", dtype=torch.float32):
        """Build the aligner and load weights from ``<codec_path>/aligner``."""
        import os

        from safetensors.torch import load_file

        self = cls(tokenizer)
        weights = load_file(os.path.join(codec_path, "aligner", "model.safetensors"))
        missing, unexpected = self.load_state_dict(weights, strict=False)
        if missing:
            # weight_norm parametrization may rename pos_conv; surface anything substantive.
            real_missing = [k for k in missing if "pos_conv" not in k]
            if real_missing:
                raise RuntimeError(f"Aligner: missing weights: {real_missing[:8]} ...")
        return self.to(device=device, dtype=dtype).eval()

    @torch.no_grad()
    def forward(
        self,
        audio: torch.Tensor,
        text_tokens: torch.Tensor,
        audio_length: torch.Tensor | None = None,
        sample_rate: int = 24000,
    ) -> AlignOutput:
        audio = torchaudio.functional.resample(audio, sample_rate, 16000)
        attention_mask = None
        if audio_length is not None:
            al16 = (audio_length.float() * 16000 / sample_rate).long()
            attention_mask = torch.arange(audio.shape[1], device=audio.device).unsqueeze(0) < al16.unsqueeze(1)
        logits = self.encoder(audio, attention_mask=attention_mask).logits  # [B, L, V]
        if audio_length is None:
            audio_length = torch.tensor([audio.shape[1] * sample_rate / 16000], device=audio.device)
        input_lengths = (audio_length.float() / sample_rate * 50).ceil().long()
        token_positions, token_masks = self._align_text_tokens(logits, text_tokens, input_lengths)
        return AlignOutput(token_positions=token_positions, token_masks=token_masks)

    @torch.no_grad()
    def _align_text_tokens(
        self, logits: torch.Tensor, text_tokens: torch.Tensor, input_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eos_id = self.tokenizer.eos_token_id

        def process_single_item(_logits, _text_tokens):
            valid_tokens = torch.nn.functional.pad(_text_tokens, (1, 0))
            new_logits = torch.ones_like(_logits, device="cpu", dtype=torch.float32) * -float("inf")
            new_logits[:, valid_tokens] = _logits[:, valid_tokens].float().cpu()
            _tt_cpu = _text_tokens.cpu()
            selected_positions = _align_text_tokens(new_logits, _tt_cpu[_tt_cpu != eos_id])
            pos_emb = torch.zeros(int(input_lengths.max()), dtype=torch.long, device=logits.device)
            pos_emb[selected_positions] = 1
            selected = 1 + torch.tensor(selected_positions, dtype=torch.long, device=logits.device)
            return selected, pos_emb

        results = [process_single_item(lg, tt) for lg, tt in zip(logits, text_tokens)]
        all_positions = [r[0] for r in results]
        all_masks = torch.stack([r[1] for r in results], dim=0)
        all_positions = torch.nn.utils.rnn.pad_sequence(all_positions, batch_first=True, padding_value=0)
        return all_positions, all_masks
