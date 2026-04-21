#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Native S2S end-to-end driver.

Loads Fun-Audio-Chat-8B weights into our native modules
(vllm_omni.model_executor.models.fun_audio_chat.encoder / crq_decoder) plus a
plain transformers Qwen3 language_model, runs prefill + decode matching the
reference _sample algorithm, and feeds the resulting CRQ tokens through the
reference CosyVoice3 token2wav helper to synthesize a WAV.

This bypasses vllm entirely; it's a correctness check for the ported modules.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

THIS = Path(__file__).resolve()
REPO = THIS.parents[1]
REF = Path("/home/jovyan/ye/vllm-omni/src/funaudiochat")
CKPT_DEFAULT = Path("/home/jovyan/ye/vllm-omni/pretrained_models/Fun-Audio-Chat-8B")
AUDIO_DEFAULT = REF / "examples" / "ck7vv9ag.wav"
SAVES = REPO / "saves" / "native"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=str(CKPT_DEFAULT))
    ap.add_argument("--audio", default=str(AUDIO_DEFAULT))
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    torch.manual_seed(args.seed)

    # Register our config (so AutoConfig knows funaudiochat types).
    sys.path.insert(0, str(REPO))
    from vllm_omni.transformers_utils.configs.fun_audio_chat import (  # noqa: F401
        FunAudioChatAudioEncoderConfig, FunAudioChatConfig,
    )
    from vllm_omni.model_executor.models.fun_audio_chat.encoder import (
        FunAudioChatAudioEncoder, FunAudioChatDiscreteEncoder,
    )
    from vllm_omni.model_executor.models.fun_audio_chat.crq_decoder import (
        CRQState, FunAudioChatDecoder,
    )

    import librosa
    import torchaudio
    from transformers import (
        AutoConfig, AutoModelForCausalLM, AutoTokenizer, WhisperFeatureExtractor,
    )

    device = "cuda:0"
    dtype = torch.bfloat16

    print(f"[nat] device={device} dtype={dtype} model={args.model_path}")
    cfg = AutoConfig.from_pretrained(args.model_path)
    ac = cfg.audio_config
    tc = cfg.text_config

    # ── Build modules ────────────────────────────────────────────────────────
    cont_enc = FunAudioChatAudioEncoder(ac).to(device=device, dtype=dtype).eval()
    disc_enc = FunAudioChatDiscreteEncoder(ac).to(device=device, dtype=dtype).eval()
    crq = FunAudioChatDecoder(ac).to(device=device, dtype=dtype).eval()
    # Tied weight: audio_tower.embed_tokens.weight == audio_invert_tower.lm_head.weight
    crq.lm_head.weight = disc_enc.embed_tokens.weight

    # Load all three module sets from checkpoint.
    import json as _json
    idx_path = Path(args.model_path) / "model.safetensors.index.json"
    with idx_path.open() as f:
        idx = _json.load(f)
    weight_map = idx["weight_map"]

    from safetensors.torch import load_file as load_safe
    shards_needed = {
        shard for name, shard in weight_map.items()
        if name.startswith((
            "continuous_audio_tower.", "audio_tower.",
            "audio_invert_tower.", "language_model.",
        ))
    }
    shard_tensors: dict[str, dict[str, torch.Tensor]] = {}
    for shard in shards_needed:
        shard_tensors[shard] = load_safe(str(Path(args.model_path) / shard))

    def collect(prefix: str, skip_prefixes: tuple[str, ...] = ()) -> dict[str, torch.Tensor]:
        out = {}
        for name, shard in weight_map.items():
            if not name.startswith(prefix):
                continue
            if any(name.startswith(sp) for sp in skip_prefixes):
                continue
            out[name[len(prefix):]] = shard_tensors[shard][name]
        return out

    cont_state = collect("continuous_audio_tower.")
    disc_state = collect("audio_tower.")
    crq_state_ckpt = collect(
        "audio_invert_tower.",
        skip_prefixes=("audio_invert_tower.crq_transformer.embed_tokens.",),
    )
    # lm_head is tied to audio_tower.embed_tokens; that comes through disc_enc load.
    m, u = cont_enc.load_state_dict(cont_state, strict=False)
    assert not [x for x in m if "positional" not in x] and not u, (m, u)
    m, u = disc_enc.load_state_dict(disc_state, strict=False)
    assert not m and not u, (m, u)
    # Ensure crq.lm_head is wired to disc embed_tokens BEFORE loading.
    crq.lm_head.weight = disc_enc.embed_tokens.weight
    m, u = crq.load_state_dict(crq_state_ckpt, strict=False)
    # lm_head is reported missing because it's shared; that's expected.
    assert not [x for x in m if x != "lm_head.weight"] and not u, (m, u)
    print(f"[nat] encoders + CRQ loaded")

    # ── Language model (Qwen3) ───────────────────────────────────────────────
    # Build the text model from text_config alone.
    print("[nat] loading Qwen3 language model ...")
    # Extract only language_model.* weights into a temp dir? Simpler: let HF
    # load the whole checkpoint and filter in-memory via state_dict hook. Even
    # simpler: pass Fun-Audio-Chat-8B to AutoModelForCausalLM with the Qwen3
    # text_config and a conversion map.
    # For speed, construct Qwen3 from text_config and load language_model.* directly.
    lm = AutoModelForCausalLM.from_config(tc)
    lm = lm.to(device=device, dtype=dtype).eval()
    lm_state = collect("language_model.")
    m, u = lm.load_state_dict(lm_state, strict=False)
    # Qwen3 may report lm_head.weight missing if tied to embed_tokens; ok.
    missing_meaningful = [x for x in m if x != "lm_head.weight"]
    if missing_meaningful:
        print(f"[nat] lm missing (non-lm_head): {missing_meaningful[:5]}")
    if u:
        print(f"[nat] lm unexpected: {u[:5]}")
    if hasattr(lm, "tie_weights"):
        try:
            lm.tie_weights()
        except Exception:
            lm.lm_head.weight = lm.model.embed_tokens.weight
    print("[nat] language model loaded")

    # ── Prefill: build inputs_embeds ─────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    feat_extractor = WhisperFeatureExtractor(
        feature_size=128, sampling_rate=16000, hop_length=160,
        chunk_length=30, n_fft=400, padding_value=0.0, return_attention_mask=True,
    )

    audio_wav, _ = librosa.load(args.audio, sr=16000)
    duration_s = len(audio_wav) / 16000.0
    num_frames_25hz = int(duration_s * 25)
    num_audio_tokens = math.ceil(num_frames_25hz / ac.group_size)
    print(f"[nat] input audio: {duration_s:.2f}s -> {num_audio_tokens} <|AUDIO|> placeholders")

    # Prompt (mirror reference infer_s2s.py).
    try:
        from utils.constant import SPOKEN_S2M_PROMPT  # type: ignore
    except ImportError:
        sys.path.insert(0, str(REF))
        from utils.constant import SPOKEN_S2M_PROMPT  # type: ignore
    audio_placeholder = (
        "<|audio_bos|>" + ("<|AUDIO|>" * num_audio_tokens) + "<|audio_eos|>"
    )
    conversation = [
        {"role": "system", "content": SPOKEN_S2M_PROMPT},
        {"role": "user", "content": audio_placeholder},
    ]
    prompt = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False,
    )
    print(f"[nat] prompt: {prompt[:120]!r}...")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Audio features (Whisper mel, padded to max_length of 30s).
    feat = feat_extractor(
        [audio_wav], sampling_rate=16000, return_tensors="pt", padding="max_length",
        return_attention_mask=True,
    )
    input_features = feat["input_features"].to(device=device, dtype=dtype)  # [1, 128, 3000]
    attn_mask = feat["attention_mask"].to(device=device)                     # [1, 3000]
    feature_lens = attn_mask.sum(-1)                                         # [1]

    # Encode continuous audio tower.
    aftercnn_lens, output_lens = cont_enc._get_feat_extract_output_lengths(feature_lens)
    speech_maxlen = int(output_lens.item())
    # Pad to multiple of group_size per reference prefill convention.
    speech_maxlen_padded = (
        (speech_maxlen + ac.group_size - 1) // ac.group_size
    ) * ac.group_size
    # continuous_audio_tower wants flat mel [num_mel_bins, sum(feature_lens)].
    flat_mel = input_features.permute(0, 2, 1)[attn_mask.bool()].permute(1, 0)
    cont_out = cont_enc(
        flat_mel, feature_lens=feature_lens, aftercnn_lens=aftercnn_lens,
        speech_maxlen=speech_maxlen_padded,
    ).last_hidden_state  # [1, speech_maxlen_padded, output_dim]

    # Discrete encoder: all-pad speech_ids, fuse continuous features.
    speech_ids_prefill = torch.full(
        (1, speech_maxlen_padded), ac.pad_token_id, dtype=torch.long, device=device,
    )
    fem = torch.ones(1, dtype=torch.bool, device=device)
    audio_features = disc_enc(
        speech_ids_prefill, continuous_audio_features=cont_out, feature_exist_mask=fem,
        return_dict=True,
    ).last_hidden_state  # [1, speech_maxlen_padded/group_size, output_dim]
    audio_features = audio_features[0, :num_audio_tokens]  # [num_audio_tokens, H]

    # Build inputs_embeds by masked_scatter at <|AUDIO|> positions.
    inputs_embeds = lm.model.embed_tokens(input_ids)
    audio_token_id = cfg.audio_token_index
    audio_mask = (input_ids == audio_token_id)
    n_placeholders = int(audio_mask.sum().item())
    print(f"[nat] n_placeholders in prompt: {n_placeholders}")
    assert n_placeholders == audio_features.shape[0], (
        n_placeholders, audio_features.shape
    )
    inputs_embeds = inputs_embeds.masked_scatter(
        audio_mask.unsqueeze(-1).expand_as(inputs_embeds), audio_features.to(inputs_embeds.dtype)
    )

    # ── Decode loop ──────────────────────────────────────────────────────────
    print("[nat] decode loop ...")
    past_kv = None
    generated_text_ids: list[int] = []
    generated_crq: list[int] = []
    generate_speech = False
    speech_finished = False
    pending_force_text_abos = True
    eos_text_id = tc.eos_token_id
    audio_bos_index = tc.audio_bos_index
    audio_eos_index = tc.audio_eos_index
    crq_eos = ac.eos_token_id

    # First step uses inputs_embeds (no input_ids to LM).
    step_embeds = inputs_embeds
    step_input_ids = None

    # Per-request CRQ state.
    # Reference DEFAULT_S2M_GEN_KWARGS: do_sample=True, temperature=0.8,
    # top_p=0.9, repetition_penalty=1.2. CRQ needs these or it gets stuck
    # on a single codebook index under greedy sampling.
    from transformers.generation.logits_process import (
        LogitsProcessorList as LPL,
        RepetitionPenaltyLogitsProcessor,
        TemperatureLogitsWarper,
        TopPLogitsWarper,
    )
    crq_lp = LPL([
        RepetitionPenaltyLogitsProcessor(1.2),
        TemperatureLogitsWarper(0.8),
        TopPLogitsWarper(0.9),
    ])
    crq_state = CRQState(
        logits_processor=crq_lp,
        do_sample=True,
        speech_ids=torch.empty(1, 0, dtype=torch.long, device=device),
    )

    import time
    t0 = time.time()

    for step in range(args.max_new_tokens):
        with torch.no_grad():
            if past_kv is None:
                # Prefill: LM forward + warm CRQ KV cache over the whole prompt
                # (ref L1309 monkey-patches audio_invert_tower.forward to
                # crq_generate_forward so this happens implicitly).
                out = lm.model(inputs_embeds=step_embeds, use_cache=True, return_dict=True)
                past_kv = out.past_key_values
                prefill_hidden = out.last_hidden_state
                prefill_text_emb = lm.model.embed_tokens(input_ids).to(dtype)
                crq_input_prefill = prefill_hidden + prefill_text_emb.detach()
                _, crq_state = crq.crq_generate_forward(crq_input_prefill, crq_state)
                # Discard `audio_embeds` (and `generate_tokens`) from warmup:
                # at the first REAL speech step we must reseed from BOS, not
                # from the prefill's (discarded) last-position token.
                from dataclasses import replace as _replace
                crq_state = _replace(crq_state, audio_embeds=None)
                last_hidden = prefill_hidden[:, -1:, :]
            else:
                # Decode step: use inputs_embeds derived from previous text
                # token (and speech feedback if speech is active).
                text_emb = lm.model.embed_tokens(step_input_ids.unsqueeze(0)).to(dtype)
                if generate_speech and crq_state.speech_ids.shape[-1] >= ac.group_size:
                    recent = crq_state.speech_ids[:, -ac.group_size:]
                    audio_fb = disc_enc(recent, return_dict=True).last_hidden_state
                    step_embeds_d = (text_emb + audio_fb.to(dtype)) / 2
                else:
                    step_embeds_d = text_emb
                out = lm.model(
                    inputs_embeds=step_embeds_d, past_key_values=past_kv,
                    use_cache=True, return_dict=True,
                )
                past_kv = out.past_key_values
                last_hidden = out.last_hidden_state[:, -1:, :]  # [1, 1, H]
            logits = lm.lm_head(last_hidden).float()  # [1, 1, V]

            # Ref ordering:
            #   (1) Before sampling, CRQ runs inside forward() (using the step's
            #       last_hidden + text_embeds) but speech tokens are only
            #       consumed if `generate_speech` is already True at entry.
            #   (2) After sampling, the model decides whether the NEXT step
            #       enters speech mode (via `generate_speech |= (last_tok==
            #       audio_bos_index)`).
            # So CRQ for this step must use `generate_speech` *before* we flip
            # it; that prevents the off-by-one where force_text_abos would
            # immediately consume CRQ tokens from the prefill context.
            speech_active_this_step = generate_speech and not speech_finished
            if speech_active_this_step:
                text_emb_crq = (
                    lm.model.embed_tokens(step_input_ids.unsqueeze(0)).to(dtype)
                    if step_input_ids is not None
                    else lm.model.embed_tokens(input_ids[:, -1:]).to(dtype)
                )
                crq_input = last_hidden + text_emb_crq.detach()
                new_tokens, crq_state = crq.crq_generate_forward(crq_input, crq_state)
                new_tokens = new_tokens.long()
                group = new_tokens[0].tolist()
                if crq_eos in group:
                    idx_eos = group.index(crq_eos)
                    group = group[:idx_eos]
                    speech_finished = True
                generated_crq.extend(group)
                if not speech_finished:
                    crq_state.speech_ids = torch.cat(
                        [crq_state.speech_ids, new_tokens], dim=-1
                    )

            # Force audio_bos on the very first decoded token if requested.
            if pending_force_text_abos:
                next_token = torch.tensor([audio_bos_index], device=device)
                pending_force_text_abos = False
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)  # [1]

            tok_int = int(next_token.item())
            generated_text_ids.append(tok_int)

            # Flip speech mode if audio_bos sampled (affects NEXT step's CRQ).
            if tok_int == audio_bos_index and not generate_speech:
                generate_speech = True
            if tok_int == audio_eos_index:
                speech_finished = True
            if tok_int == eos_text_id:
                break

            step_input_ids = next_token

    dt = time.time() - t0
    print(f"[nat] decode done: {len(generated_text_ids)} text tokens, "
          f"{len(generated_crq)} crq tokens in {dt:.1f}s")
    text_out = tokenizer.decode(generated_text_ids, skip_special_tokens=True)
    print(f"[nat] text: {text_out!r}")
    print(f"[nat] crq head: {generated_crq[:24]}")

    # ── Stage 1: CRQ tokens -> WAV via reference cosyvoice ───────────────────
    SAVES.mkdir(parents=True, exist_ok=True)
    tokens_out = SAVES / f"nat_{Path(args.audio).stem}.tokens.json"
    tokens_out.write_text(
        json.dumps(
            {
                "text_tokens": generated_text_ids,
                "crq_tokens": generated_crq,
                "text": text_out,
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    print(f"[nat] tokens -> {tokens_out}")

    valid_crq = [int(t) for t in generated_crq if 0 <= int(t) < ac.bos_token_id]
    if not valid_crq:
        print("[nat] WARNING: no valid CRQ tokens; skipping vocoder")
        return 1

    # Use reference CosyVoice detokenizer for Stage 1 (matches plan O7(a)).
    # get_audio_detokenizer() hard-codes a relative path; call CosyVoice3
    # directly with the absolute HF-downloaded path instead.
    os.chdir(REF)  # still needed so token2wav() finds utils/new_spk2info.pt
    sys.path.insert(0, str(REF))
    sys.path.insert(0, str(REF / "third_party" / "CosyVoice"))
    sys.path.insert(0, str(REF / "third_party" / "CosyVoice" / "third_party" / "Matcha-TTS"))
    from cosyvoice.cli.cosyvoice import CosyVoice3
    from utils.cosyvoice_detokenizer import token2wav
    vocoder_path = "/home/jovyan/ye/vllm-omni/pretrained_models/Fun-CosyVoice3-0.5B-2512"
    cosy = CosyVoice3(vocoder_path, load_trt=False, load_vllm=False, fp16=False)
    cosy.model.flow.decoder.estimator.static_chunk_size = 2 * 25 * 30
    speech = token2wav(
        cosy, valid_crq, embedding=None,
        token_hop_len=25 * 30, pre_lookahead_len=3,
    )
    out = SAVES / f"nat_{Path(args.audio).stem}.wav"
    # torchaudio.save requires torchcodec in torch 2.10; use soundfile instead.
    import soundfile as sf
    wav_np = speech.cpu().float().numpy()
    if wav_np.ndim == 2:
        wav_np = wav_np.T  # soundfile wants [T, C]
    sf.write(str(out), wav_np, cosy.sample_rate, subtype="PCM_16")
    print(f"[nat] wav -> {out}  sr={cosy.sample_rate}  len={speech.shape[-1]}")

    try:
        import whisper
        os.chdir(REPO)
        w = whisper.load_model("small")
        res = w.transcribe(str(out), language="zh")
        print(f"[whisper] {res['text']}")
    except Exception as exc:  # noqa: BLE001
        print(f"[whisper] skipped: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
