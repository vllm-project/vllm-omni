#!/usr/bin/env python3
"""Analyze bosonai/higgs-audio-v3-tts-4b checkpoint metadata.

Downloads only config.json, tokenizer.json, tokenizer_config.json, and
model.safetensors.index.json (no multi-GB weight files). Prints a structured
analysis to stdout. Redirect to a file for durable evidence.

Usage:
    export HF_HOME=/path/to/hf/cache
    python scripts/analyze_higgs_v3_checkpoint.py > results/higgs_v3_checkpoint_analysis.txt
"""

import json
import os


def main():
    repo_id = "bosonai/higgs-audio-v3-tts-4b"
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

    print("=== Higgs-Audio-V3 Checkpoint Analysis ===")
    print(f"Repo: {repo_id}")
    print(f"HF_HOME: {hf_home}")
    print()

    # Download metadata files only
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id,
        allow_patterns=["config.json", "tokenizer*", "special_tokens*", "*.json"],
        ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.onnx", "*.msgpack"],
    )
    print(f"Snapshot path: {path}")
    print(f"Files: {sorted(os.listdir(path))}")
    print()

    # === config.json ===
    print("=== config.json ===")
    with open(os.path.join(path, "config.json")) as f:
        config = json.load(f)

    print(f"model_type: {config.get('model_type')}")
    print(f"architectures: {config.get('architectures')}")
    print(f"audio_token_id: {config.get('audio_token_id')}")

    aec = config.get("audio_encoder_config", {})
    print("\naudio_encoder_config:")
    print(f"  encoder_type: {aec.get('encoder_type')}")
    print(f"  num_codebooks: {aec.get('num_codebooks')}")
    print(f"  vocab_size: {aec.get('vocab_size')}")
    print(f"  out_dim: {aec.get('out_dim')}")
    print(f"  tie_word_embeddings: {aec.get('tie_word_embeddings')}")
    print(f"  use_delay_pattern: {aec.get('use_delay_pattern')}")

    tc = config.get("text_config", {})
    print("\ntext_config:")
    print(f"  model_type: {tc.get('model_type')}")
    print(f"  hidden_size: {tc.get('hidden_size')}")
    print(f"  num_hidden_layers: {tc.get('num_hidden_layers')}")
    print(f"  num_attention_heads: {tc.get('num_attention_heads')}")
    print(f"  num_key_value_heads: {tc.get('num_key_value_heads')}")
    print(f"  head_dim: {tc.get('head_dim')}")
    print(f"  intermediate_size: {tc.get('intermediate_size')}")
    print(f"  vocab_size: {tc.get('vocab_size')}")
    print(f"  bos_token_id: {tc.get('bos_token_id')}")
    print(f"  eos_token_id: {tc.get('eos_token_id')}")
    print(f"  tie_word_embeddings: {tc.get('tie_word_embeddings')}")
    print(f"  rms_norm_eps: {tc.get('rms_norm_eps')}")
    print(f"  max_position_embeddings: {tc.get('max_position_embeddings')}")
    print()

    # === tokenizer.json ===
    print("=== tokenizer.json special tokens ===")
    with open(os.path.join(path, "tokenizer.json")) as f:
        tok = json.load(f)

    added = tok.get("added_tokens", [])
    tts_specials = [
        t
        for t in added
        if any(kw in t.get("content", "") for kw in ["tts", "audio", "text", "ref_", "streaming", "await"])
    ]
    for t in sorted(tts_specials, key=lambda x: x["id"]):
        print(f"  {t['id']:>8d}  {t['content']!r}  special={t.get('special')}")

    # Also show eos-related tokens
    eos_tokens = [t for t in added if "end" in t.get("content", "").lower()]
    print("\nEOS-related tokens:")
    for t in sorted(eos_tokens, key=lambda x: x["id"]):
        print(f"  {t['id']:>8d}  {t['content']!r}  special={t.get('special')}")
    print()

    # === model.safetensors.index.json ===
    index_path = os.path.join(path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        print("=== model.safetensors.index.json ===")
        with open(index_path) as f:
            idx = json.load(f)
        wm = idx.get("weight_map", {})
        print(f"Total keys: {len(wm)}")

        # Group by top-level prefix
        groups: dict[str, list[str]] = {}
        for key in sorted(wm.keys()):
            parts = key.split(".")
            if parts[0] == "body" and parts[1] == "layers":
                group = f"body.layers.{parts[2]}"
            elif parts[0] == "tied":
                group = ".".join(parts[:4]) if len(parts) > 4 else ".".join(parts[:3])
            else:
                group = ".".join(parts[:2])
            groups.setdefault(group, []).append(key)

        print(f"\nKey groups ({len(groups)}):")
        for g in sorted(groups.keys()):
            keys = groups[g]
            print(f"  {len(keys):>4d}  {g}")
            if len(keys) <= 5:
                for k in keys:
                    print(f"         {k}")

        # Show one full layer
        layer0 = [k for k in sorted(wm.keys()) if k.startswith("body.layers.0.")]
        print(f"\nbody.layers.0 keys ({len(layer0)}):")
        for k in layer0:
            print(f"  {k}")

        # Show codec key prefixes
        codec_keys = [k for k in sorted(wm.keys()) if "modality_embeddings.0.model" in k]
        codec_prefixes: dict[str, int] = {}
        for k in codec_keys:
            stripped = k.replace("tied.embedding.modality_embeddings.0.model.", "")
            prefix = ".".join(stripped.split(".")[:2])
            codec_prefixes[prefix] = codec_prefixes.get(prefix, 0) + 1
        print(f"\nCodec key prefixes ({len(codec_keys)} total):")
        for p in sorted(codec_prefixes.keys()):
            print(f"  {codec_prefixes[p]:>4d}  {p}")
    else:
        print("model.safetensors.index.json not found")

    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()
