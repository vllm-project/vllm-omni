# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encoder-level on/off cosine + latency for the encoder length bucketing.

Two reference models, selected by ``--model-type``:

* ``covo``      — Covo-Audio-Chat's unmasked HF ``WhisperEncoder`` (the lossy case).
* ``stepaudio2``— Step-Audio-2-mini's mask-aware ``AudioEncoder`` (the lossless case).

Runs the bucketed wrapper on a synthetic batch of mixed lengths and reports the
per-row cosine (on vs off) plus the encoder-only latency and speedup. Checkpoint
paths come from env vars only (``COVO_AUDIO_CHAT_CKPT`` / ``STEP_AUDIO2_CKPT``);
no data paths are committed.

Prerequisites
-------------
* Run from the repo root, or set ``PYTHONPATH=<vllm-omni checkout>`` if the
  package is not installed (the dev checkout is not editable).
* Hardware: a single NVIDIA GPU (verified on RTX 5090 32 GB, bf16).
* Checkpoints (public Hugging Face): ``Step-Audio-2-mini``, ``Covo-Audio-Chat``.
  Set the env var to your local checkpoint dir, e.g.
  ``STEP_AUDIO2_CKPT=/path/to/Step-Audio-2-mini``.
* No audio dataset is required — it uses synthetic mixed-length batches.

Bucketing env vars (set to turn the feature on)
-----------------------------------------------
* ``VLLM_OMNI_AUDIO_BUCKET_SECONDS=2,4,8,16,30`` — the rung ladder in seconds
  (rung mel frames = seconds x 100).
* ``VLLM_OMNI_AUDIO_BUCKET_MODULES=encoder`` — which encoder modules to bucket.
* ``VLLM_OMNI_AUDIO_BUCKET_TIME_AXIS=2`` — the time axis of the feature tensor.
* ``VLLM_OMNI_AUDIO_BUCKET_CONTEXT_MARGIN=2`` — extra conv-context frames (StepAudio2 = 2).

Usage (lossy case, unmasked Whisper):
    COVO_AUDIO_CHAT_CKPT=/path/to/Covo-Audio-Chat \
    VLLM_OMNI_AUDIO_BUCKET_SECONDS=2,4,8,16,30 \
    VLLM_OMNI_AUDIO_BUCKET_MODULES=encoder VLLM_OMNI_AUDIO_BUCKET_TIME_AXIS=2 \
    python benchmarks/encoder_bucketing_onoff_cosine.py --model-type covo

Usage (lossless case, mask-aware):
    STEP_AUDIO2_CKPT=/path/to/Step-Audio-2-mini \
    VLLM_OMNI_AUDIO_BUCKET_SECONDS=2,4,8,16,30 \
    VLLM_OMNI_AUDIO_BUCKET_MODULES=encoder VLLM_OMNI_AUDIO_BUCKET_TIME_AXIS=2 \
    VLLM_OMNI_AUDIO_BUCKET_CONTEXT_MARGIN=2 \
    python benchmarks/encoder_bucketing_onoff_cosine.py --model-type stepaudio2
"""

import glob
import json
import os
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn

# The "on" path installs the bucketed encoder wrapper, which reads these ladder
# env vars. Default them so the script works without the caller setting them
# (override via the environment if desired).
os.environ.setdefault("VLLM_OMNI_AUDIO_BUCKET_SECONDS", "2,4,8,16,30")
os.environ.setdefault("VLLM_OMNI_AUDIO_BUCKET_MODULES", "encoder")
os.environ.setdefault("VLLM_OMNI_AUDIO_BUCKET_TIME_AXIS", "2")
os.environ.setdefault("VLLM_OMNI_AUDIO_BUCKET_CONTEXT_MARGIN", "0")

RUNGS = (200, 400, 800, 1600, 3000)
N_ITERS = 20

# Mixed-length batches: many short items + one long item so short items visibly
# stop paying for the longest window. Covo pads to 30 s (3000 frames); StepAudio2's
# positional window caps at ~15 s (1500 frames).
_LENGTHS = {
    "covo": [100, 200, 800, 2000, 500, 3000, 300, 1000],
    "stepaudio2": [100] * 16 + [1500],
}

# Extra speedup batches, reported alongside ``_LENGTHS`` so the realistic spread
# (LibriSpeech-length distribution) is visible next to the "best case"
# (many short + one long) batch in ``_LENGTHS``.
_REALISTIC_BATCHES = {
    "stepaudio2": [
        (
            "realistic (LibriSpeech dist)",
            [400, 500, 650, 750, 800, 900, 1000, 1100, 1300, 1500, 1700, 2000, 2300, 2600, 2900, 2980],
        ),
    ],
}


def load_encoder(model_type: str, checkpoint: str) -> tuple[nn.Module, bool]:
    """Load the encoder and return ``(encoder, has_length_arg)``."""
    if model_type == "covo":
        from transformers.models.whisper.configuration_whisper import WhisperConfig
        from transformers.models.whisper.modeling_whisper import WhisperEncoder

        ec = json.load(open(os.path.join(checkpoint, "config.json")))["encoder_config"]
        cfg = WhisperConfig(
            d_model=ec["d_model"],
            encoder_layers=ec["encoder_layers"],
            encoder_attention_heads=ec["encoder_attention_heads"],
            encoder_ffn_dim=ec["encoder_ffn_dim"],
            num_mel_bins=ec["num_mel_bins"],
            max_source_positions=ec["max_source_positions"],
        )
        enc = WhisperEncoder(cfg).to(torch.bfloat16)
        shard = os.path.join(checkpoint, "model-00004-of-00004.safetensors")
        with safe_open(shard, framework="pt") as f:
            state = {k[len("encoder.") :]: f.get_tensor(k) for k in f.keys() if k.startswith("encoder.")}
        enc.load_state_dict(state, strict=False)
        return enc.cuda().eval(), False

    if model_type == "stepaudio2":
        from vllm_omni.model_executor.models.step_audio2.step_audio2_thinker import AudioEncoder

        cfg = json.load(open(os.path.join(checkpoint, "config.json")))["audio_encoder_config"]
        enc = AudioEncoder(
            n_mels=cfg["n_mels"],
            n_ctx=cfg["n_audio_ctx"],
            n_state=cfg["n_audio_state"],
            n_head=cfg["n_audio_head"],
            n_layer=cfg["n_audio_layer"],
        ).to(torch.bfloat16)
        state = {}
        for shard in sorted(glob.glob(os.path.join(checkpoint, "model-*.safetensors"))):
            with safe_open(shard, framework="pt") as f:
                for k in f.keys():
                    if k.startswith("encoder."):
                        state[k[len("encoder.") :]] = f.get_tensor(k)
        missing, _ = enc.load_state_dict(state, strict=False)
        assert not missing, f"missing encoder tensors: {missing}"
        return enc.cuda().eval(), True

    raise ValueError(f"unknown --model-type {model_type!r}")


class Model(nn.Module):
    """Embed path that calls the encoder in the model's native contract."""

    def __init__(self, encoder: nn.Module, has_length_arg: bool):
        super().__init__()
        self.encoder = encoder
        self.has_length_arg = has_length_arg

    def embed_multimodal(self, **kwargs):
        if self.has_length_arg:
            return self.encoder(kwargs["audio_features"], kwargs["audio_lens"])
        return self.encoder(kwargs["audio_features"]).last_hidden_state


def make_batch(lengths, mel=128, margin=0):
    # Pad to the longest FULL input (positional length + conv context margin),
    # mirroring pad_sequence in the real processor. Rows carry real content in
    # [0, length + margin) and silence (zeros) after.
    feats = torch.randn(len(lengths), mel, max(lengths) + margin, dtype=torch.bfloat16, device="cuda")
    for row, length in enumerate(lengths):
        feats[row, :, length + margin :] = 0
    return feats


def _features(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, tuple) and torch.is_tensor(output[0]):
        return output[0]
    return getattr(output, "last_hidden_state", None)


def _measure_speedup(encoder, model, has_length, lengths, margin=0):
    """Return ``(off_ms, on_ms)`` for one batch of the given mel-frame lengths."""
    feats = make_batch(lengths, margin=margin)
    lens = torch.tensor(lengths, dtype=torch.long)

    def call_off():
        return encoder(feats, lens.cuda()) if has_length else encoder(feats)

    def call_on():
        if has_length:
            return model.embed_multimodal(audio_features=feats, audio_lens=lens)
        return model.embed_multimodal(audio_features=feats, encoder_bucket_frames=lens)

    with torch.inference_mode():
        for _ in range(3):
            call_off()
            call_on()
        torch.accelerator.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_ITERS):
            call_off()
        torch.accelerator.synchronize()
        off_ms = (time.perf_counter() - t0) / N_ITERS * 1000
        t0 = time.perf_counter()
        for _ in range(N_ITERS):
            call_on()
        torch.accelerator.synchronize()
        on_ms = (time.perf_counter() - t0) / N_ITERS * 1000
    return off_ms, on_ms


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=("covo", "stepaudio2"), required=True)
    args = parser.parse_args()

    # StepAudio2's front conv reads two mel frames beyond the reported length;
    # set the deployment-wide margin so the bucketed wrapper copies them.
    if args.model_type == "stepaudio2":
        os.environ["VLLM_OMNI_AUDIO_BUCKET_CONTEXT_MARGIN"] = "2"
    margin = int(os.environ.get("VLLM_OMNI_AUDIO_BUCKET_CONTEXT_MARGIN", "0"))

    checkpoint = os.environ["COVO_AUDIO_CHAT_CKPT" if args.model_type == "covo" else "STEP_AUDIO2_CKPT"]
    lengths = _LENGTHS[args.model_type]

    torch.manual_seed(0)
    encoder, has_length = load_encoder(args.model_type, checkpoint)
    feats = make_batch(lengths, margin=margin)
    lens = torch.tensor(lengths, dtype=torch.long)

    with torch.inference_mode():
        if has_length:
            off = _features(encoder(feats, lens.cuda()))
        else:
            off = _features(encoder(feats))

        model = Model(encoder, has_length)
        from vllm_omni.worker import bucketed_encoder
        from vllm_omni.worker.bucketed_encoder import install_bucketed_encoders

        # This benchmark wraps the raw encoder in a synthetic ``Model``, not a
        # real stage model, so bypass the verified-model whitelist and drive the
        # wrapper with the resolved knobs directly.
        bucketed_encoder._verified_knobs = lambda vllm_config: {"context_margin": margin, "time_axis": 2}

        assert install_bucketed_encoders(model) == ["encoder"]
        if has_length:
            model.embed_multimodal(audio_features=feats, audio_lens=lens)  # prime
            on = _features(model.embed_multimodal(audio_features=feats, audio_lens=lens))
        else:
            model.embed_multimodal(audio_features=feats, encoder_bucket_frames=lens)  # prime
            on = _features(model.embed_multimodal(audio_features=feats, encoder_bucket_frames=lens))

    # Per-row cosine over each item's true output length.
    out_widths = [length // 2 if args.model_type == "covo" else (length + 1) // 2 // 2 for length in lengths]
    print(f"{'len':>5} {'rung':>5} {'cos(on,off)':>13}")
    min_cos = 1.0
    for row, length in enumerate(lengths):
        rung = min(r for r in RUNGS if r >= length)
        width = out_widths[row]
        a = on[row, :width].float().flatten()
        b = off[row, :width].float().flatten()
        cos = F.cosine_similarity(a, b, dim=0).item()
        min_cos = min(min_cos, cos)
        print(f"{length:>5} {rung:>5} {cos:>13.4f}")
    print(f"min cosine: {min_cos:.4f}")

    batches = [("best case", lengths)] + _REALISTIC_BATCHES.get(args.model_type, [])
    for name, batch_lengths in batches:
        off_ms, on_ms = _measure_speedup(encoder, model, has_length, batch_lengths, margin)
        print(f"\n{name}: off {off_ms:.2f} ms   on {on_ms:.2f} ms   speedup {off_ms / on_ms:.2f}x")


if __name__ == "__main__":
    main()
