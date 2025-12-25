#!/usr/bin/env python3
"""
Fun-Audio-Chat Deep Unit Test Suite
Compares consistency between official HuggingFace and vllm-omni implementations.

Contains the following test modules:
1. Processor Consistency: Verify preprocessing results (Mel features, Token IDs, etc.)
2. Continuous Encoder: Verify Whisper-like encoder outputs
3. Discrete Encoder: Verify discrete encoder and feature fusion logic
4. Full Embedding: Verify final embeddings fed to LLM

Usage:
    python tests/fun_audio_chat/test_unit_comparison.py --test all
    python tests/fun_audio_chat/test_unit_comparison.py --test processor
    python tests/fun_audio_chat/test_unit_comparison.py --test continuous
"""

import argparse
import gc
import os
import sys

import librosa
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoProcessor

# Add repository root to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
fun_audio_chat_path = os.path.join(REPO_ROOT, "Fun-Audio-Chat")
if os.path.exists(fun_audio_chat_path):
    sys.path.insert(0, fun_audio_chat_path)
    print(f"[Setup] Added {fun_audio_chat_path} to sys.path")
else:
    print(f"[Setup] Warning: {fun_audio_chat_path} not found. Trying relative path.")
    sys.path.insert(0, os.path.abspath("Fun-Audio-Chat"))

# Register HF model
try:
    from funaudiochat.register import register_funaudiochat

    register_funaudiochat()
except ImportError as e:
    print(f"[Setup] Error importing funaudiochat: {e}")
    print(f"[Setup] sys.path: {sys.path}")
    sys.exit(1)

# Configuration
MODEL_PATH = os.path.join(REPO_ROOT, "pretrained_models", "Fun-Audio-Chat-8B")
AUDIO_PATH = os.path.join(REPO_ROOT, "Fun-Audio-Chat", "examples", "ck7vv9ag.wav")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# Strict mode tolerance (bfloat16 accumulates ~0.1% error over 32 layers)
ATOL = 0.1  # Allow 0.1 absolute tolerance for bfloat16
RTOL = 0.01  # 1% relative tolerance


def cleanup():
    """Clear GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_header(msg):
    print("\n" + "=" * 80)
    print(f" {msg}")
    print("=" * 80)


def load_audio():
    """Load test audio"""
    audio_data, sr = librosa.load(AUDIO_PATH, sr=16000)
    print(f"[Audio] Path: {AUDIO_PATH}")
    print(f"[Audio] Shape: {audio_data.shape}, Duration: {len(audio_data) / sr:.2f}s, SR: {sr}")
    return audio_data


def get_hf_components():
    """Load HF components (Model, Processor, Config)"""
    print_header("Loading HF Official Model")
    config = AutoConfig.from_pretrained(MODEL_PATH)
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # Memory optimization: Load model on-demand
    # We load the full model here since we need its weights
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, config=config, torch_dtype=DTYPE, device_map=DEVICE)
    model.eval()
    return model, processor, config


def get_vllm_encoders(hf_config):
    """Load vllm-omni encoders"""
    print_header("Loading vLLM-Omni Encoders")
    from vllm_omni.model_executor.models.fun_audio_chat.audio_encoder import (
        FunAudioChatAudioEncoder,
        FunAudioChatDiscreteEncoder,
    )

    audio_config = hf_config.audio_config

    # Instantiate
    vllm_continuous = FunAudioChatAudioEncoder(audio_config).to(DEVICE, DTYPE)
    vllm_discrete = FunAudioChatDiscreteEncoder(audio_config).to(DEVICE, DTYPE)

    return vllm_continuous, vllm_discrete


def copy_weights(src_module, dst_module):
    """Strict weight copying"""
    src_state = src_module.state_dict()
    dst_state = dst_module.state_dict()

    missing_keys = []
    shape_mismatch = []

    for key in dst_state:
        if key in src_state:
            if src_state[key].shape == dst_state[key].shape:
                dst_state[key] = src_state[key].clone()
            else:
                shape_mismatch.append(f"{key}: src={src_state[key].shape} != dst={dst_state[key].shape}")
        else:
            missing_keys.append(key)

    if missing_keys:
        print(f"⚠️ Missing keys in source: {missing_keys}")
    if shape_mismatch:
        print(f"⚠️ Shape mismatches: {shape_mismatch}")

    dst_module.load_state_dict(dst_state)
    print(f"✅ Weights copied for {type(dst_module).__name__}")


def compare_tensors(name, hf_t, vllm_t, atol=ATOL, rtol=RTOL):
    """Detailed tensor comparison"""
    print(f"\n>>> Comparing {name}")

    if hf_t.shape != vllm_t.shape:
        print(f"❌ Shape Mismatch: HF={hf_t.shape} vs vLLM={vllm_t.shape}")
        return False

    hf_t = hf_t.float()
    vllm_t = vllm_t.float()

    diff = (hf_t - vllm_t).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Check for NaN/Inf
    if torch.isnan(hf_t).any() or torch.isnan(vllm_t).any():
        print("❌ NaN detected in outputs")
        return False

    print(f"  Max Diff: {max_diff:.6f}")
    print(f"  Mean Diff: {mean_diff:.6f}")

    # Statistics
    print(f"  HF Stats:   mean={hf_t.mean():.4f}, std={hf_t.std():.4f}, min={hf_t.min():.4f}, max={hf_t.max():.4f}")
    print(
        f"  vLLM Stats: mean={vllm_t.mean():.4f}, std={vllm_t.std():.4f}, min={vllm_t.min():.4f}, max={vllm_t.max():.4f}"
    )

    is_close = torch.allclose(hf_t, vllm_t, atol=atol, rtol=rtol)
    if is_close:
        print(f"✅ PASSED (atol={atol}, rtol={rtol})")
    else:
        print(f"❌ FAILED (atol={atol}, rtol={rtol})")
        # Print top mismatches
        flat_indices = torch.argsort(diff.flatten(), descending=True)[:5]
        print("  Top 5 mismatches:")
        for idx in flat_indices:
            i = np.unravel_index(idx.item(), hf_t.shape)
            print(f"    idx={i}: HF={hf_t[i].item():.6f}, vLLM={vllm_t[i].item():.6f}, Diff={diff[i].item():.6f}")

    return is_close


# ==============================================================================
# Test Cases
# ==============================================================================


def get_default_input(processor, audio_data):
    """Prepare standard inputs using HF Processor"""
    DEFAULT_S2T_PROMPT = "You are asked to generate text tokens."
    AUDIO_TEMPLATE = "<|AUDIO|>"
    conversation = [
        {"role": "system", "content": DEFAULT_S2T_PROMPT},
        {"role": "user", "content": AUDIO_TEMPLATE},
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    return processor(text=text, audio=[audio_data], return_tensors="pt", return_token_type_ids=False)


def test_processor():
    print_header("TEST: Processor Consistency")
    audio_data = load_audio()
    _, processor, _ = get_hf_components()

    inputs = get_default_input(processor, audio_data)

    print("\n[Processor Outputs]")
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            if k == "input_features":
                print(f"    stats: mean={v.float().mean():.4f}, std={v.float().std():.4f}")

    # Validation points:
    # 1. Check if input_features shape matches expected (batch, 128, frames)
    # 2. Check if speech_ids exists
    assert "input_features" in inputs
    assert "speech_ids" in inputs

    cleanup()
    return True


def test_continuous_encoder():
    print_header("TEST: Continuous Encoder (Whisper-like)")
    audio_data = load_audio()
    hf_model, processor, hf_config = get_hf_components()
    vllm_continuous, _ = get_vllm_encoders(hf_config)

    # Synchronize weights
    copy_weights(hf_model.continuous_audio_tower, vllm_continuous)

    # Prepare inputs
    inputs = get_default_input(processor, audio_data)
    input_features = inputs.input_features.to(DEVICE)
    feature_attention_mask = inputs.feature_attention_mask.to(DEVICE)
    speech_maxlen = inputs.speech_ids.shape[-1]

    # 1. Calculate audio feature lengths
    audio_feature_lengths = feature_attention_mask.sum(-1)
    print(f"[Debug] input_features shape: {input_features.shape}")  # [B, D, T]
    print(f"[Debug] audio_feature_lengths: {audio_feature_lengths}")

    # 2. Pack features like HF does in get_audio_features
    # HF: input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
    # This gives shape [D, TotalT] (packed, 2D)
    packed_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
    print(f"[Debug] packed_features shape: {packed_features.shape}")  # [D, TotalT]

    # 3. Compute internal lengths
    aftercnn_lens, output_lens = vllm_continuous._get_feat_extract_output_lengths(audio_feature_lengths)
    print(f"[Debug] aftercnn_lens: {aftercnn_lens}, output_lens: {output_lens}")

    # --- HF Forward (uses packed internally via get_audio_features) ---
    with torch.no_grad():
        # Call encoder directly with packed features (same as HF internally does)
        hf_encoder_out = hf_model.continuous_audio_tower(
            packed_features.to(DTYPE),
            feature_lens=audio_feature_lengths,
            aftercnn_lens=aftercnn_lens,
            speech_maxlen=speech_maxlen,
        )
        hf_out = hf_encoder_out.last_hidden_state

    # --- vLLM Forward (with packed features) ---
    with torch.no_grad():
        vllm_out = vllm_continuous(
            input_features=packed_features.to(DTYPE),  # [D, TotalT] packed
            feature_lens=audio_feature_lengths,
            aftercnn_lens=aftercnn_lens,
            speech_maxlen=speech_maxlen,
        )

    # Compare
    res = compare_tensors("Continuous Encoder Output", hf_out, vllm_out)

    cleanup()
    return res


def test_discrete_encoder():
    print_header("TEST: Discrete Encoder & Fusion")
    audio_data = load_audio()
    hf_model, processor, hf_config = get_hf_components()
    _, vllm_discrete = get_vllm_encoders(hf_config)

    # Synchronize weights
    copy_weights(hf_model.audio_tower, vllm_discrete)

    # Prepare inputs
    inputs = get_default_input(processor, audio_data)
    speech_ids = inputs.speech_ids.to(DEVICE)
    input_features = inputs.input_features.to(DEVICE)
    feature_attention_mask = inputs.feature_attention_mask.to(DEVICE)
    speech_maxlen = speech_ids.shape[-1]

    # Get Continuous Features (as input to Discrete Encoder)
    with torch.no_grad():
        continuous_feats, cont_lens = hf_model.get_audio_features(
            input_features, feature_attention_mask=feature_attention_mask, speech_maxlen=speech_maxlen
        )

    # --- HF Forward ---
    with torch.no_grad():
        feature_exist_mask = torch.tensor([True], device=DEVICE)
        hf_out = hf_model.audio_tower(
            speech_ids,
            continuous_audio_features=continuous_feats,
            continuous_audio_output_lengths=cont_lens,
            feature_exist_mask=feature_exist_mask,
            return_dict=True,
        ).last_hidden_state

    # --- vLLM Forward ---
    with torch.no_grad():
        vllm_out = vllm_discrete(
            audio_ids=speech_ids,
            continuous_audio_features=continuous_feats,
            continuous_audio_output_lengths=cont_lens,
            feature_exist_mask=feature_exist_mask,
        )

    # Compare
    res = compare_tensors("Discrete Encoder Output", hf_out, vllm_out)

    cleanup()
    return res


def test_full_pipeline():
    print_header("TEST: Full Pipeline (Embeddings)")
    # This test mainly checks the value range of final embeddings fed to LLM
    audio_data = load_audio()
    hf_model, processor, _ = get_hf_components()

    inputs = get_default_input(processor, audio_data)
    inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        # 1. Text Embeddings
        text_embeds = hf_model.get_input_embeddings()(inputs["input_ids"])

        # 2. Audio Embeddings (Full Process)
        continuous_feats, cont_lens = hf_model.get_audio_features(
            inputs["input_features"],
            feature_attention_mask=inputs["feature_attention_mask"],
            speech_maxlen=inputs["speech_ids"].shape[-1],
        )

        feature_exist_mask = torch.tensor([True], device=DEVICE)
        audio_embeds = hf_model.audio_tower(
            inputs["speech_ids"],
            continuous_audio_features=continuous_feats,
            continuous_audio_output_lengths=cont_lens,
            feature_exist_mask=feature_exist_mask,
            return_dict=True,
        ).last_hidden_state

    print("\n[Embedding Statistics]")
    print(
        f"Text Embeddings:  mean={text_embeds.mean():.4f}, std={text_embeds.std():.4f}, range=[{text_embeds.min():.4f}, {text_embeds.max():.4f}]"
    )
    print(
        f"Audio Embeddings: mean={audio_embeds.mean():.4f}, std={audio_embeds.std():.4f}, range=[{audio_embeds.min():.4f}, {audio_embeds.max():.4f}]"
    )

    ratio = audio_embeds.std() / text_embeds.std()
    print(f"Std Ratio (Audio/Text): {ratio:.2f}")

    if ratio > 10:
        print("⚠️ WARNING: Audio embeddings have significantly larger variance than text embeddings!")

    cleanup()
    return True


def test_continuous_encoder_detailed():
    """Layer-by-layer diagnostic for the continuous encoder."""
    print_header("TEST: Continuous Encoder DETAILED (Layer-by-Layer)")
    audio_data = load_audio()
    hf_model, processor, hf_config = get_hf_components()
    vllm_continuous, _ = get_vllm_encoders(hf_config)

    # Copy weights
    copy_weights(hf_model.continuous_audio_tower, vllm_continuous)

    # Prepare input
    inputs = get_default_input(processor, audio_data)
    input_features = inputs.input_features.to(DEVICE)
    feature_attention_mask = inputs.feature_attention_mask.to(DEVICE)

    # Pack features
    packed_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
    packed_features = packed_features.to(DTYPE)

    hf_encoder = hf_model.continuous_audio_tower
    vllm_encoder = vllm_continuous

    # ====== Step 1: Conv1 ======
    print("\n[Step 1] Conv1 output:")
    with torch.no_grad():
        # For packed [D, T] we need to add batch dim for conv: [1, D, T]
        hf_conv1_in = packed_features.unsqueeze(0)  # [1, 128, 600]
        vllm_conv1_in = packed_features.unsqueeze(0)

        hf_conv1_out = torch.nn.functional.gelu(hf_encoder.conv1(hf_conv1_in))
        vllm_conv1_out = torch.nn.functional.gelu(vllm_encoder.conv1(vllm_conv1_in))

        diff = (hf_conv1_out.float() - vllm_conv1_out.float()).abs()
        print(f"  HF shape: {hf_conv1_out.shape}, vLLM shape: {vllm_conv1_out.shape}")
        print(f"  Max Diff: {diff.max():.6f}, Mean Diff: {diff.mean():.6f}")
        print(f"  HF stats: mean={hf_conv1_out.float().mean():.4f}, std={hf_conv1_out.float().std():.4f}")
        print(f"  vLLM stats: mean={vllm_conv1_out.float().mean():.4f}, std={vllm_conv1_out.float().std():.4f}")
        conv1_match = diff.max().item() < 0.01
        print(f"  {'✅ MATCH' if conv1_match else '❌ MISMATCH'}")

    # ====== Step 2: Conv2 ======
    print("\n[Step 2] Conv2 output:")
    with torch.no_grad():
        hf_conv2_out = torch.nn.functional.gelu(hf_encoder.conv2(hf_conv1_out))
        vllm_conv2_out = torch.nn.functional.gelu(vllm_encoder.conv2(vllm_conv1_out))

        diff = (hf_conv2_out.float() - vllm_conv2_out.float()).abs()
        print(f"  HF shape: {hf_conv2_out.shape}, vLLM shape: {vllm_conv2_out.shape}")
        print(f"  Max Diff: {diff.max():.6f}, Mean Diff: {diff.mean():.6f}")
        conv2_match = diff.max().item() < 0.01
        print(f"  {'✅ MATCH' if conv2_match else '❌ MISMATCH'}")

    # ====== Step 3: Positional Embedding ======
    print("\n[Step 3] After positional embedding:")
    with torch.no_grad():
        hf_embed = hf_conv2_out.transpose(1, 2)  # [1, T', D]
        vllm_embed = vllm_conv2_out.transpose(1, 2)  # [1, T', D]

        seq_len = hf_embed.shape[1]
        hf_pos = hf_encoder.positional_embedding.positional_embedding[:seq_len, :].unsqueeze(0).to(DTYPE)
        vllm_pos = vllm_encoder.positional_embedding.positional_embedding[:seq_len, :].unsqueeze(0).to(DTYPE)

        # Check position embeddings match
        pos_diff = (hf_pos.float() - vllm_pos.float()).abs()
        print(f"  Position embedding diff: max={pos_diff.max():.6f}")

        hf_with_pos = hf_embed + hf_pos
        vllm_with_pos = vllm_embed + vllm_pos

        diff = (hf_with_pos.float() - vllm_with_pos.float()).abs()
        print(f"  Max Diff: {diff.max():.6f}, Mean Diff: {diff.mean():.6f}")
        pos_match = diff.max().item() < 0.01
        print(f"  {'✅ MATCH' if pos_match else '❌ MISMATCH'}")

    # ====== Step 4: First Transformer Layer ======
    print("\n[Step 4] After first transformer layer:")
    with torch.no_grad():
        # Prepare attention mask (no masking for single sequence)
        # HF uses packed format and cu_seqlens
        seq_len = hf_with_pos.shape[1]
        hidden_states = hf_with_pos.squeeze(0)  # [T', D]
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=DEVICE)

        # Create attention mask
        attention_mask = torch.zeros(1, 1, seq_len, seq_len, device=DEVICE, dtype=DTYPE)

        hf_layer0_out = hf_encoder.layers[0](hidden_states, cu_seqlens=cu_seqlens, attention_mask=attention_mask)
        hf_layer0_hidden = hf_layer0_out[0] if isinstance(hf_layer0_out, tuple) else hf_layer0_out

        vllm_layer0_out = vllm_encoder.layers[0](hidden_states, cu_seqlens=cu_seqlens, attention_mask=attention_mask)
        vllm_layer0_hidden = vllm_layer0_out if not isinstance(vllm_layer0_out, tuple) else vllm_layer0_out[0]

        diff = (hf_layer0_hidden.float() - vllm_layer0_hidden.float()).abs()
        print(f"  HF shape: {hf_layer0_hidden.shape}, vLLM shape: {vllm_layer0_hidden.shape}")
        print(f"  Max Diff: {diff.max():.6f}, Mean Diff: {diff.mean():.6f}")
        print(f"  HF stats: mean={hf_layer0_hidden.float().mean():.4f}, std={hf_layer0_hidden.float().std():.4f}")
        print(f"  vLLM stats: mean={vllm_layer0_hidden.float().mean():.4f}, std={vllm_layer0_hidden.float().std():.4f}")
        layer0_match = diff.max().item() < 0.01
        print(f"  {'✅ MATCH' if layer0_match else '❌ MISMATCH'}")

        if not layer0_match:
            # Further diagnose within the layer
            print("\n  [Step 4a] Diagnosing first layer internals:")
            # Self-attn layer norm
            ln_out_hf = hf_encoder.layers[0].self_attn_layer_norm(hidden_states)
            ln_out_vllm = vllm_encoder.layers[0].self_attn_layer_norm(hidden_states)
            ln_diff = (ln_out_hf.float() - ln_out_vllm.float()).abs()
            print(f"    self_attn_layer_norm: max_diff={ln_diff.max():.6f}")

            # Self-attn output
            attn_out_hf = hf_encoder.layers[0].self_attn(
                ln_out_hf, cu_seqlens=cu_seqlens, attention_mask=attention_mask
            )
            attn_out_vllm = vllm_encoder.layers[0].self_attn(
                ln_out_vllm, cu_seqlens=cu_seqlens, attention_mask=attention_mask
            )
            attn_diff = (attn_out_hf.float() - attn_out_vllm.float()).abs()
            print(f"    self_attn: max_diff={attn_diff.max():.6f}")
            print(f"      HF attn out: mean={attn_out_hf.float().mean():.4f}, std={attn_out_hf.float().std():.4f}")
            print(
                f"      vLLM attn out: mean={attn_out_vllm.float().mean():.4f}, std={attn_out_vllm.float().std():.4f}"
            )

            if attn_diff.max() > 0.01:
                # Check Q, K, V projections
                print("\n    [Step 4b] Diagnosing attention projections:")
                q_hf = hf_encoder.layers[0].self_attn.q_proj(ln_out_hf)
                q_vllm = vllm_encoder.layers[0].self_attn.q_proj(ln_out_vllm)
                print(f"      q_proj diff: {(q_hf.float() - q_vllm.float()).abs().max():.6f}")

                k_hf = hf_encoder.layers[0].self_attn.k_proj(ln_out_hf)
                k_vllm = vllm_encoder.layers[0].self_attn.k_proj(ln_out_vllm)
                print(f"      k_proj diff: {(k_hf.float() - k_vllm.float()).abs().max():.6f}")

                v_hf = hf_encoder.layers[0].self_attn.v_proj(ln_out_hf)
                v_vllm = vllm_encoder.layers[0].self_attn.v_proj(ln_out_vllm)
                print(f"      v_proj diff: {(v_hf.float() - v_vllm.float()).abs().max():.6f}")

    cleanup()
    return conv1_match and conv2_match and pos_match and layer0_match


def test_llm_forward():
    """Test full LLM forward pass with merged audio embeddings."""
    print_header("TEST: LLM Forward Pass (Audio + Text -> Logits)")
    audio_data = load_audio()
    hf_model, processor, hf_config = get_hf_components()

    # Prepare full inputs
    inputs = get_default_input(processor, audio_data)
    inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # ====== Step 1: Prepare embeddings via HF model ======
    print("\n[Step 1] Getting audio embeddings from HF...")
    with torch.no_grad():
        # Get audio features
        continuous_feats, cont_lens = hf_model.get_audio_features(
            inputs["input_features"],
            feature_attention_mask=inputs["feature_attention_mask"],
            speech_maxlen=inputs["speech_ids"].shape[-1],
        )
        print(f"  continuous_feats: shape={continuous_feats.shape}")

        # Get discrete encoder output
        feature_exist_mask = torch.tensor([True], device=DEVICE)
        audio_embeds = hf_model.audio_tower(
            inputs["speech_ids"],
            continuous_audio_features=continuous_feats,
            continuous_audio_output_lengths=cont_lens,
            feature_exist_mask=feature_exist_mask,
            return_dict=True,
        ).last_hidden_state
        print(
            f"  audio_embeds: shape={audio_embeds.shape}, stats: min={audio_embeds.min():.4f}, max={audio_embeds.max():.4f}"
        )

    # ====== Step 2: Merge embeddings ======
    print("\n[Step 2] Merging embeddings...")
    with torch.no_grad():
        input_ids = inputs["input_ids"]

        # Get text embeddings
        text_embeds = hf_model.get_input_embeddings()(input_ids)
        print(
            f"  text_embeds: shape={text_embeds.shape}, stats: min={text_embeds.min():.4f}, max={text_embeds.max():.4f}"
        )

        # Find audio token positions
        # AUDIO_TOKEN_INDEX from config
        audio_token_idx = 151669  # <|audio_start|> / <|AUDIO|>
        audio_mask = input_ids == audio_token_idx
        n_audio_tokens = audio_mask.sum().item()
        print(f"  Audio token positions: {audio_mask.sum().item()} tokens at index {audio_token_idx}")

        # Debug: print all unique token IDs to find the audio token
        unique_ids = input_ids.unique().tolist()
        print(f"  Unique token IDs in input: {unique_ids}")

        # Flatten audio embeds for merging
        flat_audio = audio_embeds.reshape(-1, audio_embeds.shape[-1])
        n_audio_embeds = flat_audio.shape[0]
        print(f"  Audio embeddings to merge: {n_audio_embeds}")

        if n_audio_tokens != n_audio_embeds:
            print(f"  ⚠️ MISMATCH: {n_audio_tokens} audio tokens vs {n_audio_embeds} audio embeddings!")

        # Merge
        merged_embeds = text_embeds.clone()
        if n_audio_tokens > 0 and n_audio_embeds > 0:
            flat_merged = merged_embeds.reshape(-1, merged_embeds.shape[-1])
            flat_mask = audio_mask.reshape(-1)
            # Adjust if counts differ
            if n_audio_embeds > n_audio_tokens:
                flat_audio = flat_audio[:n_audio_tokens]
            flat_merged[flat_mask] = flat_audio.to(flat_merged.dtype)
            merged_embeds = flat_merged.reshape(text_embeds.shape)

        print(
            f"  merged_embeds: shape={merged_embeds.shape}, stats: min={merged_embeds.min():.4f}, max={merged_embeds.max():.4f}"
        )

    # ====== Step 3: Forward through HF LLM ======
    print("\n[Step 3] Forward through HF language model...")
    with torch.no_grad():
        # Use HF model's forward with inputs_embeds
        hf_outputs = hf_model.language_model(inputs_embeds=merged_embeds, output_hidden_states=True, return_dict=True)
        hf_logits = hf_outputs.logits
        print(f"  HF logits: shape={hf_logits.shape}")
        print(f"  HF logits stats: mean={hf_logits.float().mean():.4f}, std={hf_logits.float().std():.4f}")

        # Get predicted token
        last_logits = hf_logits[0, -1, :]
        predicted_token = last_logits.argmax().item()
        predicted_text = processor.tokenizer.decode([predicted_token])
        print(f"  HF predicted next token: {predicted_token} = '{predicted_text}'")

        # Top 5 predictions
        top_k = 5
        top_vals, top_ids = torch.topk(last_logits, top_k)
        print(f"  HF Top-{top_k} predictions:")
        for i in range(top_k):
            tok = processor.tokenizer.decode([top_ids[i].item()])
            print(f"    {i + 1}. {top_ids[i].item():6d} ({top_vals[i]:.2f}): '{tok}'")

    # ====== Step 4: Compare with HF's native forward ======
    print("\n[Step 4] Compare with HF's native forward (full pipeline)...")
    with torch.no_grad():
        # Use HF model's built-in forward which handles everything
        hf_native_outputs = hf_model(
            input_ids=inputs["input_ids"],
            input_features=inputs["input_features"],
            speech_ids=inputs["speech_ids"],
            feature_attention_mask=inputs["feature_attention_mask"],
            speech_attention_mask=inputs["speech_attention_mask"],
            feature_exist_mask=inputs["feature_exist_mask"],
            return_dict=True,
        )
        hf_native_logits = hf_native_outputs.logits
        print(f"  HF native logits: shape={hf_native_logits.shape}")

        # Compare logits
        logits_diff = (hf_logits.float() - hf_native_logits.float()).abs()
        print(f"  Logits diff: max={logits_diff.max():.6f}, mean={logits_diff.mean():.6f}")

        native_last_logits = hf_native_logits[0, -1, :]
        native_predicted = native_last_logits.argmax().item()
        native_text = processor.tokenizer.decode([native_predicted])
        print(f"  HF native predicted: {native_predicted} = '{native_text}'")

        if predicted_token != native_predicted:
            print("  ⚠️ MISMATCH: manual vs native predictions differ!")

    cleanup()
    return True


def test_vllm_forward():
    """
    Test vLLM model forward pass directly.
    This mimics what happens in the actual inference pipeline.
    """
    print_header("TEST: vLLM Model Forward Pass")

    audio_data = load_audio()
    hf_model, processor, hf_config = get_hf_components()

    # ====== Step 1: Prepare inputs like vLLM does ======
    print("\n[Step 1] Prepare inputs...")
    inputs = get_default_input(processor, audio_data)
    inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    input_ids = inputs["input_ids"]  # [1, 57] or [1, 53]
    input_features = inputs["input_features"]  # [1, 128, 30000]
    feature_attention_mask = inputs["feature_attention_mask"]  # [1, 30000]
    speech_ids = inputs["speech_ids"]  # [1, 150]

    print(f"  input_ids: {input_ids.shape}")
    print(f"  input_features: {input_features.shape}")
    print(f"  speech_ids: {speech_ids.shape}")

    # ====== Step 2: Load vLLM encoders with HF weights ======
    print("\n[Step 2] Load vLLM encoders...")
    vllm_continuous, vllm_discrete = get_vllm_encoders(hf_config)
    copy_weights(hf_model.continuous_audio_tower, vllm_continuous)
    copy_weights(hf_model.audio_tower, vllm_discrete)

    # ====== Step 3: Run vLLM continuous encoder ======
    print("\n[Step 3] vLLM Continuous Encoder...")
    audio_feature_lengths = feature_attention_mask.sum(-1)  # [1]

    # Pack features like HF does
    packed_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
    packed_features = packed_features.to(DTYPE)

    aftercnn_lens, output_lens = vllm_continuous._get_feat_extract_output_lengths(audio_feature_lengths)
    speech_maxlen = speech_ids.shape[-1]  # 150

    with torch.no_grad():
        vllm_continuous_out = vllm_continuous(
            input_features=packed_features,
            feature_lens=audio_feature_lengths,
            aftercnn_lens=aftercnn_lens,
            speech_maxlen=speech_maxlen,
        )
    print(f"  vllm_continuous_out: {vllm_continuous_out.shape}")
    print(f"    stats: min={vllm_continuous_out.min():.4f}, max={vllm_continuous_out.max():.4f}")

    # Compare with HF
    with torch.no_grad():
        hf_continuous_out, hf_cont_lens = hf_model.get_audio_features(
            input_features, feature_attention_mask=feature_attention_mask, speech_maxlen=speech_maxlen
        )
    continuous_diff = (vllm_continuous_out.float() - hf_continuous_out.float()).abs()
    print(f"  Diff vs HF: max={continuous_diff.max():.6f}, mean={continuous_diff.mean():.6f}")
    continuous_match = continuous_diff.max().item() < 0.1
    print(f"  {'✅ MATCH' if continuous_match else '❌ MISMATCH'}")

    # ====== Step 4: Run vLLM discrete encoder ======
    print("\n[Step 4] vLLM Discrete Encoder...")
    feature_exist_mask = torch.tensor([[True]], device=DEVICE)  # [1, 1]

    with torch.no_grad():
        # vLLM discrete encoder uses 'audio_ids' not 'speech_ids'
        vllm_discrete_out = vllm_discrete(
            audio_ids=speech_ids.unsqueeze(1) if speech_ids.ndim == 2 else speech_ids,  # [1, 1, 150]
            continuous_audio_features=vllm_continuous_out,  # [1, 150, 4096]
            continuous_audio_output_lengths=output_lens,  # [150]
            feature_exist_mask=feature_exist_mask,
        )
    print(f"  vllm_discrete_out: {vllm_discrete_out.shape}")
    print(f"    stats: min={vllm_discrete_out.min():.4f}, max={vllm_discrete_out.max():.4f}")

    # Compare with HF
    with torch.no_grad():
        hf_discrete_out = hf_model.audio_tower(
            speech_ids,
            continuous_audio_features=hf_continuous_out,
            continuous_audio_output_lengths=hf_cont_lens,
            feature_exist_mask=torch.tensor([True], device=DEVICE),
            return_dict=True,
        ).last_hidden_state
    discrete_diff = (vllm_discrete_out.float() - hf_discrete_out.float()).abs()
    print(f"  Diff vs HF: max={discrete_diff.max():.6f}, mean={discrete_diff.mean():.6f}")
    discrete_match = discrete_diff.max().item() < 0.1
    print(f"  {'✅ MATCH' if discrete_match else '❌ MISMATCH'}")

    # ====== Step 5: Merge embeddings ======
    print("\n[Step 5] Merge embeddings...")
    with torch.no_grad():
        # Get text embeddings using HF's embed_tokens (we'd use vLLM's in real inference)
        text_embeds = hf_model.get_input_embeddings()(input_ids)

        # Find audio tokens
        audio_token_idx = 151669
        audio_mask = input_ids == audio_token_idx
        n_audio_tokens = audio_mask.sum().item()

        # Flatten audio embeddings
        flat_audio = vllm_discrete_out.reshape(-1, vllm_discrete_out.shape[-1])
        n_audio_embeds = flat_audio.shape[0]

        print(f"  Audio tokens: {n_audio_tokens}, Audio embeddings: {n_audio_embeds}")

        # Merge
        merged_embeds = text_embeds.clone()
        if n_audio_tokens > 0:
            flat_merged = merged_embeds.reshape(-1, merged_embeds.shape[-1])
            flat_mask = audio_mask.reshape(-1)
            if n_audio_embeds > n_audio_tokens:
                flat_audio = flat_audio[:n_audio_tokens]
            flat_merged[flat_mask] = flat_audio.to(flat_merged.dtype)
            merged_embeds = flat_merged.reshape(text_embeds.shape)

        print(f"  merged_embeds: {merged_embeds.shape}")
        print(f"    stats: min={merged_embeds.min():.4f}, max={merged_embeds.max():.4f}")

    # ====== Step 6: Forward through LLM and compare ======
    print("\n[Step 6] Forward through LLM...")
    with torch.no_grad():
        # HF forward
        hf_outputs = hf_model.language_model(inputs_embeds=merged_embeds, return_dict=True)
        hf_logits = hf_outputs.logits

        # Predictions
        last_logits = hf_logits[0, -1, :]
        predicted = last_logits.argmax().item()
        predicted_text = processor.tokenizer.decode([predicted])

        print(f"  HF logits shape: {hf_logits.shape}")
        print(f"  HF prediction: {predicted} = '{predicted_text}'")

        # Top 5
        top_k = 5
        top_vals, top_ids = torch.topk(last_logits, top_k)
        print(f"  HF Top-{top_k} predictions:")
        for i in range(top_k):
            tok = processor.tokenizer.decode([top_ids[i].item()])
            print(f"    {i + 1}. {top_ids[i].item():6d} ({top_vals[i]:.2f}): '{tok}'")

    # ====== Step 7: Now test what vLLM's inference produces ======
    print("\n[Step 7] Test vLLM's positions handling...")
    # The key difference might be in how positions are handled
    # vLLM uses positions tensor, HF uses position_ids or infers from input

    seq_len = merged_embeds.shape[1]

    # Test: what if we call HF LLM with explicit position_ids?
    position_ids = torch.arange(seq_len, device=DEVICE).unsqueeze(0)
    hf_outputs_with_pos = hf_model.language_model(
        inputs_embeds=merged_embeds, position_ids=position_ids, return_dict=True
    )
    hf_logits_with_pos = hf_outputs_with_pos.logits

    logits_diff = (hf_logits.float() - hf_logits_with_pos.float()).abs()
    print(f"  Logits diff (with vs without explicit positions): max={logits_diff.max():.6f}")

    pred_with_pos = hf_logits_with_pos[0, -1, :].argmax().item()
    print(f"  Prediction with explicit positions: {pred_with_pos} = '{processor.tokenizer.decode([pred_with_pos])}'")

    if predicted != pred_with_pos:
        print("  ❌ Predictions differ with explicit positions!")
    else:
        print("  ✅ Predictions match with explicit positions")

    cleanup()
    return predicted == 77045  # Should predict "Absolutely"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        choices=["all", "processor", "continuous", "discrete", "full", "detailed", "llm", "vllm"],
    )
    args = parser.parse_args()

    results = {}

    if args.test in ["all", "processor"]:
        try:
            results["processor"] = test_processor()
        except Exception as e:
            print(f"❌ Processor Test Error: {e}")
            import traceback

            traceback.print_exc()
            results["processor"] = False

    if args.test in ["all", "continuous"]:
        try:
            results["continuous"] = test_continuous_encoder()
        except Exception as e:
            print(f"❌ Continuous Test Error: {e}")
            import traceback

            traceback.print_exc()
            results["continuous"] = False

    if args.test == "detailed":
        try:
            results["detailed"] = test_continuous_encoder_detailed()
        except Exception as e:
            print(f"❌ Detailed Test Error: {e}")
            import traceback

            traceback.print_exc()
            results["detailed"] = False

    if args.test in ["all", "discrete"]:
        try:
            results["discrete"] = test_discrete_encoder()
        except Exception as e:
            print(f"❌ Discrete Test Error: {e}")
            import traceback

            traceback.print_exc()
            results["discrete"] = False

    if args.test in ["all", "full"]:
        try:
            results["full"] = test_full_pipeline()
        except Exception as e:
            print(f"❌ Full Pipeline Test Error: {e}")
            import traceback

            traceback.print_exc()
            results["full"] = False

    if args.test == "llm":
        try:
            results["llm"] = test_llm_forward()
        except Exception as e:
            print(f"❌ LLM Forward Test Error: {e}")
            import traceback

            traceback.print_exc()
            results["llm"] = False

    if args.test == "vllm":
        try:
            results["vllm"] = test_vllm_forward()
        except Exception as e:
            print(f"❌ vLLM Forward Test Error: {e}")
            import traceback

            traceback.print_exc()
            results["vllm"] = False

    print_header("Summary")
    for k, v in results.items():
        print(f"{k:<15}: {'✅ PASS' if v else '❌ FAIL'}")


if __name__ == "__main__":
    main()
