#!/usr/bin/env python
"""Convert TTS-1.6b-en_fr weights from Kyutai format to HF MoshiConfig format.

Usage (from local checkout):
    uv run --with safetensors --with torch \\
        python convert_tts_weights.py \\
        --local-dir /path/to/tts-1.6b-en_fr \\
        --output /path/to/hf-tts-1.6b

Usage (download from HF Hub):
    uv run --with safetensors --with torch --with huggingface_hub \\
        python convert_tts_weights.py \\
        --repo kyutai/tts-1.6b-en_fr-pytorch-bf16 \\
        --output /path/to/hf-tts-1.6b
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def convert_kyutai_to_hf(
    kyutai_weights: dict[str, torch.Tensor],
    kyutai_config: dict,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Convert Kyutai TTS LMModel weights to HF MoshiForConditionalGeneration naming.

    Key TTS-specific handling:
    - text_emb uses demux_second_stream: pre-multiply out1 for TTS single-stream path.
    - depformer_emb uses low_rank=128: pre-multiply to expand to depth_dim.
    - Cross-attention and condition provider weights.
    """
    hf_weights: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()

    def consume(key: str) -> torch.Tensor | None:
        if key in kyutai_weights:
            consumed.add(key)
            return kyutai_weights[key]
        return None

    def require(key: str) -> torch.Tensor:
        t = consume(key)
        assert t is not None, f"Required Kyutai weight missing: {key}"
        return t

    dep_q = kyutai_config["dep_q"]
    n_q = kyutai_config["n_q"]
    dim = kyutai_config["dim"]
    num_heads = kyutai_config["num_heads"]
    kv_repeat = kyutai_config.get("kv_repeat", 1)
    num_kv = num_heads // kv_repeat
    head_dim = dim // num_heads
    hidden_scale = kyutai_config["hidden_scale"]
    ffn_dim = int(hidden_scale * dim)

    schedule = kyutai_config.get("depformer_weights_per_step_schedule")
    num_weight_sets = max(schedule) + 1 if schedule else dep_q
    num_depth_layers = kyutai_config.get("depformer_num_layers", 4)
    depth_dim = kyutai_config.get("depformer_dim", 1024)
    depth_heads = kyutai_config.get("depformer_num_heads", 16)
    depth_kv_repeat = kyutai_config.get("depformer_kv_repeat", 1)
    depth_num_kv = depth_heads // depth_kv_repeat
    depth_head_dim = depth_dim // depth_heads

    # Text embedding: demux_second_stream=True → out1 + out2 projections.
    # Pre-multiply both into separate lookup tables:
    #   embed_tokens.weight              = weight @ out1.T  (first stream)
    #   embed_tokens_second_stream.weight = weight @ out2.T  (second stream)
    text_emb_w = require("text_emb.weight")  # [text_card+1, dim]
    text_out1 = require("text_emb.out1.weight")  # [dim, dim]
    text_out2 = require("text_emb.out2.weight")  # [dim, dim]
    hf_weights["decoder.model.embed_tokens.weight"] = text_emb_w @ text_out1.T
    hf_weights["decoder.model.embed_tokens_second_stream.weight"] = text_emb_w @ text_out2.T

    for i in range(n_q):
        hf_weights[f"embed_tokens.{i}.weight"] = require(f"emb.{i}.weight")

    num_layers = kyutai_config["num_layers"]
    for layer_idx in range(num_layers):
        pfx_src = f"transformer.layers.{layer_idx}"
        pfx_dst = f"decoder.model.layers.{layer_idx}"

        in_proj_w = require(f"{pfx_src}.self_attn.in_proj_weight")
        q_dim = num_heads * head_dim
        kv_dim = num_kv * head_dim
        hf_weights[f"{pfx_dst}.self_attn.q_proj.weight"] = in_proj_w[:q_dim]
        hf_weights[f"{pfx_dst}.self_attn.k_proj.weight"] = in_proj_w[q_dim : q_dim + kv_dim]
        hf_weights[f"{pfx_dst}.self_attn.v_proj.weight"] = in_proj_w[q_dim + kv_dim : q_dim + 2 * kv_dim]
        hf_weights[f"{pfx_dst}.self_attn.o_proj.weight"] = require(f"{pfx_src}.self_attn.out_proj.weight")

        linear_in_w = require(f"{pfx_src}.gating.linear_in.weight")
        half = linear_in_w.shape[0] // 2
        hf_weights[f"{pfx_dst}.mlp.gate_proj.weight"] = linear_in_w[:half]
        hf_weights[f"{pfx_dst}.mlp.up_proj.weight"] = linear_in_w[half:]
        hf_weights[f"{pfx_dst}.mlp.down_proj.weight"] = require(f"{pfx_src}.gating.linear_out.weight")

        for norm_src, norm_dst in [
            ("norm1", "input_layernorm"),
            ("norm2", "post_attention_layernorm"),
        ]:
            w = require(f"{pfx_src}.{norm_src}.alpha")
            hf_weights[f"{pfx_dst}.{norm_dst}.weight"] = w.reshape(-1)

        # Cross-attention (speaker conditioning) — only present when cross_attention=True.
        if kyutai_config.get("cross_attention", False):
            cross_in = require(f"{pfx_src}.cross_attention.in_proj_weight")
            hf_weights[f"{pfx_dst}.cross_attn.q_proj.weight"] = cross_in[:q_dim]
            hf_weights[f"{pfx_dst}.cross_attn.k_proj.weight"] = cross_in[q_dim : q_dim + kv_dim]
            hf_weights[f"{pfx_dst}.cross_attn.v_proj.weight"] = cross_in[q_dim + kv_dim : q_dim + 2 * kv_dim]
            hf_weights[f"{pfx_dst}.cross_attn.o_proj.weight"] = require(f"{pfx_src}.cross_attention.out_proj.weight")
            hf_weights[f"{pfx_dst}.cross_attn_layernorm.weight"] = require(f"{pfx_src}.norm_cross.weight")
            hf_weights[f"{pfx_dst}.cross_attn_layernorm.bias"] = require(f"{pfx_src}.norm_cross.bias")

    out_norm = consume("out_norm.weight")
    if out_norm is None:
        out_norm = require("out_norm.alpha")
    hf_weights["decoder.model.norm.weight"] = out_norm.reshape(-1)

    hf_weights["decoder.lm_head.weight"] = require("text_linear.weight")

    cond_prefix = "condition_provider.conditioners"
    for cond_name in ("cfg", "control", "speaker_wavs"):
        for sub in ("embed.weight", "learnt_padding", "output_proj.weight"):
            key = f"{cond_prefix}.{cond_name}.{sub}"
            t = consume(key)
            if t is not None:
                hf_weights[f"{cond_prefix}.{cond_name}.{sub}"] = t

    for layer_idx in range(num_depth_layers):
        pfx = f"depformer.layers.{layer_idx}"

        fc1_parts = [require(f"{pfx}.gating.{s}.linear_in.weight") for s in range(num_weight_sets)]
        fc2_parts = [require(f"{pfx}.gating.{s}.linear_out.weight") for s in range(num_weight_sets)]
        hf_weights[f"depth_decoder.layers.{layer_idx}.mlp.fc1.weight"] = torch.stack(fc1_parts, dim=0)
        hf_weights[f"depth_decoder.layers.{layer_idx}.mlp.fc2.weight"] = torch.stack(fc2_parts, dim=0)

        in_proj = require(f"{pfx}.self_attn.in_proj_weight")
        q_dim_d = depth_heads * depth_head_dim
        kv_dim_d = depth_num_kv * depth_head_dim
        chunk_size = q_dim_d + 2 * kv_dim_d
        q_parts, k_parts, v_parts = [], [], []
        for step in range(num_weight_sets):
            chunk = in_proj[step * chunk_size : (step + 1) * chunk_size]
            q_parts.append(chunk[:q_dim_d])
            k_parts.append(chunk[q_dim_d : q_dim_d + kv_dim_d])
            v_parts.append(chunk[q_dim_d + kv_dim_d :])
        hf_weights[f"depth_decoder.layers.{layer_idx}.self_attn.q_proj.weight"] = torch.stack(q_parts, dim=0)
        hf_weights[f"depth_decoder.layers.{layer_idx}.self_attn.k_proj.weight"] = torch.stack(k_parts, dim=0)
        hf_weights[f"depth_decoder.layers.{layer_idx}.self_attn.v_proj.weight"] = torch.stack(v_parts, dim=0)

        out_proj = require(f"{pfx}.self_attn.out_proj.weight")
        o_parts = [out_proj[s * depth_dim : (s + 1) * depth_dim] for s in range(num_weight_sets)]
        hf_weights[f"depth_decoder.layers.{layer_idx}.self_attn.o_proj.weight"] = torch.stack(o_parts, dim=0)

        for norm_idx, norm_name in [(1, "input_layernorm"), (2, "post_attention_layernorm")]:
            alpha = require(f"{pfx}.norm{norm_idx}.alpha")
            hf_weights[f"depth_decoder.layers.{layer_idx}.{norm_name}.weight"] = alpha.reshape(-1)
            beta = consume(f"{pfx}.norm{norm_idx}.beta")
            if beta is not None:
                hf_weights[f"depth_decoder.layers.{layer_idx}.{norm_name}.bias"] = beta.reshape(-1)

    for i in range(dep_q - 1):
        emb_w = require(f"depformer_emb.{i}.weight")  # [card+1, low_rank_dim]
        lr_w = require(f"depformer_emb.{i}.low_rank.weight")  # [depth_dim, low_rank_dim]
        hf_weights[f"depth_decoder.embed_tokens.{i}.weight"] = emb_w @ lr_w.T

    dep_text_w = require("depformer_text_emb.weight")  # [text_card+1, low_rank_dim]
    dep_text_out1 = require("depformer_text_emb.out1.weight")  # [depth_dim, low_rank_dim]
    dep_text_out2 = require("depformer_text_emb.out2.weight")  # [depth_dim, low_rank_dim]
    require("depformer_text_emb.low_rank.weight")  # consume; dead in demux path
    hf_weights["depth_decoder.text_embed_tokens.weight"] = dep_text_w @ dep_text_out1.T
    hf_weights["depth_decoder.text_embed_tokens_second_stream.weight"] = dep_text_w @ dep_text_out2.T

    lm_head_parts = [require(f"linears.{s}.weight") for s in range(dep_q)]
    hf_weights["depth_decoder.lm_heads.weight"] = torch.stack(lm_head_parts, dim=0)

    proj_parts = [require(f"depformer_in.{s}.weight") for s in range(num_weight_sets)]
    hf_weights["depth_decoder.input_projections.weight"] = torch.stack(proj_parts, dim=0)

    unconsumed = set(kyutai_weights.keys()) - consumed
    if unconsumed:
        print(f"\nERROR: {len(unconsumed)} Kyutai weights were NOT converted:")
        for k in sorted(unconsumed):
            print(f"  {k}: {kyutai_weights[k].shape}")
        raise RuntimeError(f"Conversion incomplete: {len(unconsumed)} Kyutai weights not consumed.")

    audio_vocab = kyutai_config.get("card", 2048)
    text_card = kyutai_config["text_card"]
    _assert_shape(hf_weights, "decoder.model.embed_tokens.weight", (text_card + 1, dim))
    _assert_shape(hf_weights, "decoder.model.embed_tokens_second_stream.weight", (text_card + 1, dim))
    _assert_shape(hf_weights, "depth_decoder.lm_heads.weight", (dep_q, audio_vocab, depth_dim))
    _assert_shape(hf_weights, "depth_decoder.input_projections.weight", (num_weight_sets, depth_dim, dim))
    for i in range(dep_q - 1):
        _assert_shape(hf_weights, f"depth_decoder.embed_tokens.{i}.weight", (audio_vocab + 1, depth_dim))
    for layer_idx in range(num_depth_layers):
        _assert_shape(
            hf_weights,
            f"depth_decoder.layers.{layer_idx}.self_attn.q_proj.weight",
            (num_weight_sets, depth_heads * depth_head_dim, depth_dim),
        )

    sample_main_fc1 = hf_weights["decoder.model.layers.0.mlp.gate_proj.weight"]
    intermediate_size = sample_main_fc1.shape[0]

    depformer_ffn = kyutai_config.get("depformer_dim_feedforward")
    if depformer_ffn is None:
        sample_fc1 = kyutai_weights.get("depformer.layers.0.gating.0.linear_in.weight")
        depformer_ffn = sample_fc1.shape[0] if sample_fc1 is not None else int(hidden_scale * depth_dim)
    sample_depth_fc1 = hf_weights.get("depth_decoder.layers.0.mlp.fc1.weight")
    if sample_depth_fc1 is not None:
        depformer_intermediate = sample_depth_fc1.shape[1] // 2  # [num_ws, ffn_dim, hidden]
    else:
        depformer_intermediate = depformer_ffn // 2

    hf_config = {
        "architectures": ["MoshiForConditionalGeneration"],
        "model_type": "moshi",
        "torch_dtype": "bfloat16",
        "vocab_size": kyutai_config["text_card"],
        "lm_head_vocab_size": hf_weights["decoder.lm_head.weight"].shape[0],
        "hidden_size": dim,
        "ffn_dim": ffn_dim,
        "intermediate_size": intermediate_size,
        "head_dim": head_dim,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv,
        "max_position_embeddings": kyutai_config.get("context", 500),
        "rms_norm_eps": 1e-8,
        "rope_theta": kyutai_config.get("max_period", 10000.0),
        "sliding_window": kyutai_config.get("context", 500),
        "hidden_act": "silu",
        "num_codebooks": dep_q,
        "audio_vocab_size": audio_vocab,
        "depth_hidden_size": depth_dim,
        "depth_intermediate_size": depformer_intermediate,
        "depth_num_hidden_layers": kyutai_config.get("depformer_num_layers", 4),
        "depth_num_attention_heads": depth_heads,
        "depth_num_key_value_heads": depth_num_kv,
        "depth_max_position_embeddings": dep_q + 1,
        "depth_sliding_window": dep_q,
        "n_q": n_q,
        "dep_q": dep_q,
        "delays": kyutai_config.get("delays"),
        "depformer_weights_per_step_schedule": kyutai_config.get("depformer_weights_per_step_schedule"),
        "depformer_norm": None,  # No per-codebook output norms in TTS
        "norm": kyutai_config.get("norm"),
        "kv_repeat": kv_repeat,
        "depformer_low_rank_embeddings": kyutai_config.get("depformer_low_rank_embeddings"),
        "demux_second_stream": kyutai_config.get("demux_second_stream", False),
        "tts_config": kyutai_config.get("tts_config"),
        "cross_attention": kyutai_config.get("cross_attention", False),
        "conditioners": kyutai_config.get("conditioners"),
        "fuser": kyutai_config.get("fuser"),
        "audio_encoder_config": {
            "model_type": "mimi",
            "sampling_rate": 24000,
            "audio_channels": 1,
            "hidden_size": 512,
            "num_filters": 64,
            "num_residual_layers": 1,
            "upsampling_ratios": [8, 6, 5, 4],
            "kernel_size": 7,
            "last_kernel_size": 3,
            "residual_kernel_size": 3,
            "dilation_growth_rate": 2,
            "use_causal_conv": True,
            "pad_mode": "constant",
            "compress": 2,
            "trim_right_ratio": 1.0,
            "codebook_size": 2048,
            "codebook_dim": 256,
            "num_quantizers": 32,
            "use_conv_shortcut": False,
            "vector_quantization_hidden_dimension": 256,
            "upsample_groups": 512,
            "num_hidden_layers": 8,
            "intermediate_size": 2048,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "hidden_act": "gelu",
            "max_position_embeddings": 8000,
            "norm_eps": 1e-05,
            "rope_theta": 10000.0,
            "sliding_window": 250,
            "head_dim": 64,
            "layer_scale_initial_scale": 0.01,
            "num_semantic_quantizers": 1,
            "_frame_rate": 12.5,
        },
    }

    print(hf_config)

    return hf_weights, hf_config


def _assert_shape(weights: dict[str, torch.Tensor], key: str, expected: tuple[int, ...]) -> None:
    if key not in weights:
        raise RuntimeError(f"Expected weight {key} not found in converted weights")
    actual = tuple(weights[key].shape)
    if actual != expected:
        raise RuntimeError(f"Shape mismatch for {key}: expected {expected}, got {actual}")


def main():
    parser = argparse.ArgumentParser(description="Convert TTS-1.6b-en_fr weights to HF format")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--repo",
        default=None,
        help="HF Hub repo (e.g. kyutai/tts-1.6b-en_fr-pytorch-bf16)",
    )
    group.add_argument(
        "--local-dir",
        default=None,
        help="Path to a local Kyutai TTS checkout (must contain config.json and *.safetensors)",
    )
    parser.add_argument("--revision", default=None, help="HF repo revision/commit")
    parser.add_argument("--output", required=True, help="Output directory for HF checkpoint")
    args = parser.parse_args()

    if args.local_dir:
        local_dir = Path(args.local_dir)
        config_path = local_dir / "config.json"
        assert config_path.exists(), f"config.json not found in {local_dir}"
        with open(config_path) as f:
            kyutai_config = json.load(f)

        moshi_name = kyutai_config.get("moshi_name")
        if moshi_name:
            weights_path = local_dir / moshi_name
        else:
            candidates = list(local_dir.glob("*.safetensors"))
            candidates = [p for p in candidates if "mimi" not in p.name and "tokenizer" not in p.name]
            assert candidates, f"No LM safetensors found in {local_dir}"
            weights_path = candidates[0]

        mimi_weights_path = None
        mimi_name = kyutai_config.get("mimi_name")
        if mimi_name:
            mp = local_dir / mimi_name
            if mp.exists():
                mimi_weights_path = mp

        tokenizer_path = None
        tok_name = kyutai_config.get("tokenizer_name")
        if tok_name:
            tp = local_dir / tok_name
            if tp.exists():
                tokenizer_path = tp

    else:
        from huggingface_hub import hf_hub_download

        print(f"Loading config from {args.repo}...")
        config_file = hf_hub_download(repo_id=args.repo, filename="config.json", revision=args.revision)
        with open(config_file) as f:
            kyutai_config = json.load(f)

        moshi_name = kyutai_config.get("moshi_name")
        assert moshi_name, "config.json does not contain 'moshi_name'"
        weights_path = Path(hf_hub_download(repo_id=args.repo, filename=moshi_name, revision=args.revision))

        mimi_weights_path = None
        mimi_name = kyutai_config.get("mimi_name")
        if mimi_name:
            mimi_weights_path = Path(hf_hub_download(repo_id=args.repo, filename=mimi_name, revision=args.revision))

        tokenizer_path = None
        tok_name = kyutai_config.get("tokenizer_name")
        if tok_name:
            tokenizer_path = Path(hf_hub_download(repo_id=args.repo, filename=tok_name, revision=args.revision))

    print(
        f"Kyutai config: dim={kyutai_config.get('dim')}, "
        f"n_q={kyutai_config.get('n_q')}, dep_q={kyutai_config.get('dep_q')}, "
        f"text_card={kyutai_config.get('text_card')}, "
        f"norm={kyutai_config.get('norm')}, "
        f"depformer_low_rank_embeddings={kyutai_config.get('depformer_low_rank_embeddings')}, "
        f"demux_second_stream={kyutai_config.get('demux_second_stream')}, "
        f"cross_attention={kyutai_config.get('cross_attention')}"
    )

    print(f"Loading weights from {weights_path}...")
    kyutai_weights = load_file(str(weights_path))
    print(f"Loaded {len(kyutai_weights)} weight tensors")

    print("Converting...")
    hf_weights, hf_config = convert_kyutai_to_hf(kyutai_weights, kyutai_config)

    # Summary
    depth_weights = [k for k in hf_weights if k.startswith("depth_decoder.")]
    cross_weights = [k for k in hf_weights if "cross_attn" in k]
    cond_weights = [k for k in hf_weights if k.startswith("condition_provider.")]
    main_weights = [
        k
        for k in hf_weights
        if not k.startswith("depth_decoder.") and "cross_attn" not in k and not k.startswith("condition_provider.")
    ]
    print(f"Converted to {len(hf_weights)} HF weight tensors")
    print(f"  Main transformer:   {len(main_weights)} weights")
    print(f"  Depth decoder:      {len(depth_weights)} weights")
    if cross_weights:
        print(f"  Cross-attention:    {len(cross_weights)} weights (stored for future use)")
    print(f"  Condition provider: {len(cond_weights)} weights (stored for future use)")
    print(f"  All {len(kyutai_weights)} Kyutai weights consumed ✓")

    print("\nKey tensor shapes:")
    for key in [
        "decoder.model.embed_tokens.weight",
        "depth_decoder.lm_heads.weight",
        "depth_decoder.input_projections.weight",
        "depth_decoder.text_embed_tokens.weight",
        "depth_decoder.embed_tokens.0.weight",
        "depth_decoder.layers.0.self_attn.q_proj.weight",
        "depth_decoder.layers.0.mlp.fc1.weight",
    ]:
        if key in hf_weights:
            print(f"  {key}: {list(hf_weights[key].shape)}")

    os.makedirs(args.output, exist_ok=True)

    config_path_out = os.path.join(args.output, "config.json")
    with open(config_path_out, "w") as f:
        json.dump(hf_config, f, indent=2)
    print(f"\nSaved config to {config_path_out}")

    weights_path_out = os.path.join(args.output, "model.safetensors")
    save_file(hf_weights, weights_path_out)
    print(f"Saved weights to {weights_path_out}")

    if mimi_weights_path and mimi_weights_path.exists():
        import shutil

        mimi_dst = os.path.join(args.output, "mimi.safetensors")
        shutil.copy2(str(mimi_weights_path), mimi_dst)
        print(f"Copied Mimi weights to {mimi_dst}")

    if tokenizer_path and tokenizer_path.exists():
        import shutil

        tok_ext = tokenizer_path.suffix
        tok_dst = os.path.join(args.output, f"tokenizer{tok_ext}")
        shutil.copy2(str(tokenizer_path), tok_dst)
        print(f"Copied tokenizer to {tok_dst}")

        tok_config = {
            "tokenizer_class": "LlamaTokenizer",
            "model_max_length": kyutai_config.get("context", 500),
        }
        tok_config_path = os.path.join(args.output, "tokenizer_config.json")
        with open(tok_config_path, "w") as f:
            json.dump(tok_config, f, indent=2)
        print(f"Saved tokenizer config to {tok_config_path}")

    print(f"\nDone! HF checkpoint saved to {args.output}")
    print(
        f"Key config: vocab={hf_config['vocab_size']}, hidden={hf_config['hidden_size']}, "
        f"layers={hf_config['num_hidden_layers']}, codebooks={hf_config['num_codebooks']}, "
        f"n_q={hf_config['n_q']}, dep_q={hf_config['dep_q']}, "
        f"depth_layers={hf_config['depth_num_hidden_layers']}"
    )


if __name__ == "__main__":
    main()
