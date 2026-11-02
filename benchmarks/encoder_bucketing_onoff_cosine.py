# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-row feature-off vs feature-on cosine for the encoder length bucketing.

The bucketed path is lossy for unmasked encoders (feature-off attends over the
full extractor padding, the bucketed path over the smaller rung window), so the
drift must be measured against OFF, not against per-item solo. Run with:

    COVO_AUDIO_CHAT_CKPT=/path/to/Covo-Audio-Chat \
    VLLM_OMNI_AUDIO_BUCKET_SECONDS=2,4,8,16,30 \
    python benchmarks/encoder_bucketing_onoff_cosine.py
"""

import json
import os

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn
from transformers.models.whisper.configuration_whisper import WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoder

COVO = os.environ.get("COVO_AUDIO_CHAT_CKPT", "/mnt/zhangzhihao3/models/Covo-Audio-Chat")
LENGTHS = [100, 200, 800, 2000, 500, 3000, 300, 1000]  # mel frames (PR_REPORT mix)
RUNGS = (200, 400, 800, 1600, 3000)


def load_encoder():
    ec = json.load(open(os.path.join(COVO, "config.json")))["encoder_config"]
    cfg = WhisperConfig(
        d_model=ec["d_model"],
        encoder_layers=ec["encoder_layers"],
        encoder_attention_heads=ec["encoder_attention_heads"],
        encoder_ffn_dim=ec["encoder_ffn_dim"],
        num_mel_bins=ec["num_mel_bins"],
        max_source_positions=ec["max_source_positions"],
    )
    enc = WhisperEncoder(cfg).to(torch.bfloat16)
    with safe_open(os.path.join(COVO, "model-00004-of-00004.safetensors"), framework="pt") as f:
        state = {k[len("encoder.") :]: f.get_tensor(k) for k in f.keys() if k.startswith("encoder.")}
    enc.load_state_dict(state, strict=False)
    return enc.cuda().eval()


class CovoModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def embed_multimodal(self, **kwargs):
        return self.encoder(kwargs["audio_features"]).last_hidden_state


def make_batch(lengths, mel=128):
    feats = torch.randn(len(lengths), mel, max(lengths), dtype=torch.bfloat16, device="cuda")
    for row, length in enumerate(lengths):
        feats[row, :, length:] = 0
    return feats


def main():
    torch.manual_seed(0)
    encoder = load_encoder()
    feats = make_batch(LENGTHS)

    with torch.inference_mode():
        off = encoder(feats).last_hidden_state  # untouched forward, 30s padding

        model = CovoModel(encoder)
        from vllm_omni.worker.bucketed_encoder import install_bucketed_encoders

        install_bucketed_encoders(model)
        model.embed_multimodal(audio_features=feats, encoder_bucket_frames=torch.tensor(LENGTHS))
        on = model.embed_multimodal(audio_features=feats, encoder_bucket_frames=torch.tensor(LENGTHS))

    print(f"{'len':>5} {'rung':>5} {'cos(on,off)':>12}")
    for row, length in enumerate(LENGTHS):
        rung = min(r for r in RUNGS if r >= length)
        width = length // 2  # output frames over the item's true length
        a = on[row, :width].float().flatten()
        b = off[row, :width].float().flatten()
        cos = F.cosine_similarity(a, b, dim=0).item()
        print(f"{length:>5} {rung:>5} {cos:>12.4f}")


if __name__ == "__main__":
    main()
