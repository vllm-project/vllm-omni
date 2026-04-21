# Plan: Fun-Audio-Chat-8B native S2S in vllm-omni

**Branch:** `fun-audio-chat-s2s-native-pipeline`
**Goal:** Replace the current broken port with a correct, fully native S2S pipeline for `FunAudioLLM/Fun-Audio-Chat-8B` in vllm-omni — no external `src/funaudiochat` imports, no external CosyVoice repo, no `/wbl-fast/…` paths. Support BS≥1 with per-request CRQ state. Produce clean audio verifiable by Whisper ASR transcription of the synthesized wav.

**User answers (interview, 2026-04-21):**
- Scope: **Full native, no external deps**.
- Reference: cloned from `https://github.com/FunAudioLLM/Fun-Audio-Chat.git` into `src/funaudiochat/` (reference-only; not shipped as a runtime dep).
- Concurrency: **BS≥1 sync, per-request CRQ state**.
- Acceptance: **Manual listen + Whisper ASR transcription** (Whisper available in the local uv env).
- Checkpoint: **download from HF** (`FunAudioLLM/Fun-Audio-Chat-8B`, `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`).
- Env: **uv** (`uv venv` + `uv pip install`).
- Symptom: "pure trash" — do not trust the current implementation.

---

## Root-cause analysis (reference vs current port)

Every bug below was found by direct diff of `src/funaudiochat/funaudiochat/modeling_funaudiochat.py` (reference) against `vllm_omni/model_executor/models/fun_audio_chat/` (current port). Each cites specific lines.

1. **Prefill audio-token count is 2× off.** Ref (processing: `processing_funaudiochat.py` L196–223) uses a 25 Hz speech-token rate and groups by `group_size=5`, giving `ceil(duration × 5)` `<|AUDIO|>` placeholders. Port (`fun_audio_chat.py:231–236` `get_num_audio_tokens`) derives from mel frames: `ceil(50 × duration / 5) = ceil(10 × duration)`. For 1 s → port produces 10 slots where the model was trained on 5. LM prefill alignment between `<|AUDIO|>` placeholders and audio features is broken from token 0.

2. **Audio encoder is a simplified misfit.** Ref `FunAudioChatAudioEncoder` (modeling L252–452) does: chunked (`n_window*2=200`-frame) features → `conv1(k=3,s=1)+conv2(k=3,s=2)` → sinusoid pos emb → `encoder_layers` custom attention w/ `cu_seqlens` → `avg_pool1d(k=2,s=2)` → `ln_post` → `proj`. Net downsample 4×. Port `FunAudioEncoder` (`fun_audio_chat.py:104–206`) uses stock `WhisperEncoderLayer`, skips the avg_pool, then does an *extra* `mean` over `group_size=5` to compensate — shape ends up half-right. **Architecture and weight names diverge** (e.g. ref has `layers.*.self_attn_layer_norm/final_layer_norm`, port uses Whisper's `self_attn_layer_norm/encoder_attn_layer_norm` — `AutoWeightsLoader` silently mismatches).

3. **`FunAudioChatDiscreteEncoder` (`audio_tower`) is missing entirely.** Ref (L502–552) embeds the speech-id stream (pad tokens for user audio; real ids during decode feedback), group-mean-pools by 5, linear-proj, and **adds** the continuous encoder output (in `continuous_features_mode="add"`) to produce the final audio features masked-scattered into `inputs_embeds`. Port has no `audio_tower` at all. Weights under `audio_tower.embed_tokens.*` are *remapped to `audio_invert_tower.lm_head.*`* (port L644), which is wrong: these are a different tensor in the checkpoint, not tied.

4. **Decode-time audio-feature feedback is missing.** Ref `prepare_inputs_for_generation` (L1196–1208): once `generate_speech` is set, LM input at each step is replaced by `audio_features = (text_emb + audio_tower(speech_ids[-group_size:])) / 2`. This is the **autoregressive audio conditioning** that lets the LM emit coherent CRQ tokens step after step. Port has no override path — the LM sees only the previously-sampled text token, so it hallucinates blindly after `audio_bos`.

5. **No `audio_eos` detection.** Ref (L1420–1436) sets `finish_speech = (next_speech_tokens == audio_config.eos_token_id).any(-1)`, freezes the tail to eos, and stops appending. Port accumulates CRQ tokens forever until `max_tokens`, so the tail is garbage fed to the vocoder.

6. **`sp_gen_kwargs` control knobs are dropped.** Ref `_sample` (L1269–1271, L1381–1383, L1399–1401): `text_greedy`, `only_crq_sampling`, `force_text_abos`, `disable_speech`. Infer-s2s uses `text_greedy=True` + `force_text_abos=True`. Without `force_text_abos`, the model may not emit `audio_bos` at all and speech generation never starts (silence). Port ignores all of these.

7. **Hard-coded codebook size 6561.** Port `token2wav.py:102` and assumptions elsewhere hard-code `0 ≤ t < 6561`. Should come from `audio_config.codebook_size`.

8. **External deps pin the runtime to a missing machine.** `fun_audio_chat.py:40–58` imports `FunAudioChatDecoder` from `src/funaudiochat` at model init — fatal if not checked out. `token2wav.py:41` hard-codes `/wbl-fast/usrs/shl/speech-data-synthesis-new/CosyVoice`. `token2wav.py:36–39` hard-codes `benchmarks/_data/_hf_cache/hub/models--FunAudioLLM--Fun-CosyVoice3-0.5B-2512`. User chose full-native; these go away.

9. **BS=1 only.** Port stashes CRQ state on `self.audio_invert_tower` (`fun_audio_chat.py:506–515, 550–553`). Any concurrent request corrupts the other. User wants BS≥1, so CRQ state must live in vllm-omni's `model_intermediate_buffer[req_id]` and be threaded into/out of a stateless `audio_invert_tower.crq_generate_forward` call per request per step.

10. **No acceptance harness.** There is no example or test. User wants Whisper-ASR verification of the synthesized wav.

---

## done/

### D1. Interview + reference clone
- **Goal:** gather scope/acceptance/env answers; make reference source readable on-box.
- **Acceptance criteria:** AskUserQuestion answers captured above; `src/funaudiochat/` present.
- **Dependencies:** none.
- **Next action:** n/a (done).
- **Affected paths:** `src/funaudiochat/` (reference, git-cloned, not shipped).

---

## blocked/

Blocked items must be resolved *before* their corresponding `ongoing/` items start. Each has a dedicated unblock sub-plan with bidirectional backlinks.

### B1. Reference code + CosyVoice submodules + env + weights not available as a runnable stack on this box
- **Goal:** get a working runnable baseline: (a) reference repo + its `third_party/CosyVoice` + Matcha-TTS submodules; (b) `uv` venv with torch/transformers/vllm/vllm-omni + whisper + funaudiochat deps; (c) HF checkpoints for `Fun-Audio-Chat-8B` and `Fun-CosyVoice3-0.5B-2512`; (d) a one-command reference-inference smoke run that produces a known-good wav to A/B against.
- **Acceptance criteria:** `uv run python src/funaudiochat/examples/infer_s2s.py` (pointed at a local ckpt) produces a wav that Whisper ASR transcribes to a plausible Chinese utterance; no `/wbl-fast/…` paths in the run.
- **Dependencies:** network access for `git submodule update --init` and `huggingface-cli download`.
- **Next action:** see unblock plan → [unblock-reference-and-env.md](./unblock-reference-and-env.md).
- **Affected paths:** `.venv/`, `src/funaudiochat/third_party/CosyVoice/**`, `pretrained_models/**` (or `HF_HOME`), `design/unblock-reference-and-env.md`.
- **Blocks:** O1–O8 (we cannot A/B any port without a reference run, and cannot run vllm-omni without env + weights).

---

## ongoing/

All items gated on B1 resolving. Order is load-bearing — later items depend on state wired by earlier items.

### O1. Port configs: FunAudioChatConfig + FunAudioChatAudioEncoderConfig (completion)
- **Goal:** replace `transformers_utils/configs/fun_audio_chat.py` with a config that carries *all* reference fields (`codebook_size`, `bos_token_id`, `eos_token_id`, `pad_token_id`, `continuous_features_mode`, `crq_transformer_config`, `group_size`, `n_window`, `enable_audio_invert_tower`) and text-config audio indices (`audio_bos_index`, `audio_eos_index`, `sil_index`). Register via `AutoConfig.register`.
- **Acceptance criteria:** `AutoConfig.from_pretrained(Fun-Audio-Chat-8B)` returns an object whose `.audio_config.codebook_size`, `.audio_config.crq_transformer_config`, `.text_config.audio_bos_index` all match the checkpoint `config.json`.
- **Dependencies:** B1.
- **Next action:** dump `config.json` from the HF checkpoint and mirror every field.
- **Affected paths:** `vllm_omni/transformers_utils/configs/fun_audio_chat.py`.

### O2. Native `FunAudioChatAudioEncoder` (continuous_audio_tower)
- **Goal:** port ref L100–452 faithfully: `FunAudioChatAudioAttention` (w/ `cu_seqlens` packed attention), `FunAudioChatAudioEncoderLayer`, `SinusoidsPositionEmbedding`, `FunAudioChatAudioEncoder.forward` including chunking by `n_window*2`, `padded_and_mask_function`, `avg_pool1d(k=2,s=2)`, `ln_post`, `proj`, `audio_bos_eos_token` embedding, `_get_feat_extract_output_lengths`. Keep weight names identical to ref so `AutoWeightsLoader` (with the existing `continuous_audio_tower.*` prefix) just works.
- **Acceptance criteria:** given the same 1 s input, vllm-omni encoder's `last_hidden_state` is numerically close (atol=1e-3) to the ref encoder loaded from the same checkpoint. Output length = `((mel-1)//2+1 -2)//2+1` (= 25 for 1 s).
- **Dependencies:** O1.
- **Next action:** line-by-line transcribe; write a tiny parity test that loads both encoders from the same weights and diffs outputs.
- **Affected paths:** `vllm_omni/model_executor/models/fun_audio_chat/encoder.py` (new), `fun_audio_chat.py` (delete current `FunAudioEncoder`).

### O3. Native `FunAudioChatDiscreteEncoder` (audio_tower)
- **Goal:** port ref L502–552. Embed (`nn.Embedding(codebook_size, output_dim)`), reshape → group-mean-pool by `group_size`, `output_matching` linear, plus `continual_output_matching` for fused continuous features (both `"add"` and `"replace"` modes), `_get_feat_extract_output_lengths`.
- **Acceptance criteria:** given speech_ids and the continuous encoder's output from O2, produces the exact `inputs_embeds` slice that replaces `<|AUDIO|>` placeholders in ref's `forward` (atol=1e-3).
- **Dependencies:** O2.
- **Next action:** transcribe; add to parity test.
- **Affected paths:** `vllm_omni/model_executor/models/fun_audio_chat/discrete_encoder.py` (new).

### O4. Native `FunAudioChatDecoder` (audio_invert_tower / CRQ)
- **Goal:** port ref L563–768 in full: `pre_matching`, `input_matching`, `output_matching`, `lm_head`, `crq_transformer` (AutoModel from `crq_transformer_config`; delete its `embed_tokens` as ref does L584), and **both** `forward` (training — used in our `forward` path if ever) and `crq_generate_forward` (generation). Keep a stateless signature: accept `crq_state` (past_key_values, audio_embeds) as *arguments* and return the updated state, rather than stashing on `self`. This is the foundation for per-request state in O6.
- **Acceptance criteria:** given the same `speech_inputs_embeds = last_hidden + text_embeds` input and the same checkpoint, produces the same `crq_generate_tokens` (group_size=5 per step) as ref's CRQ decoder (atol=0 for argmax, or exact match on greedy).
- **Dependencies:** O1.
- **Next action:** transcribe; parity-test against reference with a fixed random-seeded hidden-state input.
- **Affected paths:** `vllm_omni/model_executor/models/fun_audio_chat/crq_decoder.py` (new), deletes `_try_import_decoder` in `fun_audio_chat.py`.

### O5. Weight loading (`load_weights`) matches ref checkpoint layout
- **Goal:** rewrite the remap table. Verified targets: `continuous_audio_tower.*` → `continuous_audio_tower.*` (O2 names); `audio_tower.*` → `audio_tower.*` (O3 names, **not** into `audio_invert_tower.lm_head` as port does now); `audio_invert_tower.*` → `audio_invert_tower.*` minus `audio_invert_tower.crq_transformer.embed_tokens.*` (ref deletes it, we must skip); `language_model.*` → `language_model.*`. Assert no unused keys and no missing keys at the end.
- **Acceptance criteria:** loading the real Fun-Audio-Chat-8B checkpoint emits no "unexpected/missing" warnings; every Parameter's norm matches ref's post-load norm.
- **Dependencies:** O2, O3, O4.
- **Next action:** after O2–O4, snapshot `list(state_dict.keys())` from ref and diff against our module tree.
- **Affected paths:** `vllm_omni/model_executor/models/fun_audio_chat/fun_audio_chat.py` (`load_weights`).

### O6. Decode-time state machine + per-request CRQ state (BS≥1)
- **Goal:** implement the reference `_sample` control flow inside vllm-omni's `forward` / `compute_logits` / `postprocess` hooks, with **all per-request state kept in `model_intermediate_buffer[req_id]`**, not on the module. State per request (`CRQState` dataclass stored under a single key, never mutated in place outside the owning hook): `generate_speech: bool`, `speech_finished: bool`, `speech_ids: LongTensor[T] on CPU`, `crq_past_key_values` (per-layer K/V cache for the CRQ transformer, on-GPU), `crq_audio_embeds`, `force_text_abos_pending: bool`.

**Flow per step per request (strict gating, no CRQ advance before `audio_bos`):**
  1. `prepare_inputs` hook — if `generate_speech[req]` and `speech_ids[req].shape[-1] >= group_size`: compute `audio_features = audio_tower(speech_ids[-group_size:])`, `text_emb = embed(last_sampled_text_token)`, `inputs_embeds = (text_emb + audio_features) / 2`, override that request's slice in the batched `inputs_embeds` before LM forward. Else normal text embedding.
  2. LM `forward` — unchanged path, returns `hidden_states`.
  3. `compute_logits` — text `logits = lm_head(hidden_states)` (unchanged). Then **for each request, strictly gated on `generate_speech[req] and not speech_finished[req]` — `text_greedy` does NOT gate CRQ**: call `crq_generate_forward(hidden[req] + embed(last_text[req]), state=req_state)` → 5 CRQ tokens; detect `audio_eos` (`next_speech_tokens == audio_config.eos_token_id`), mask the tail to eos, set `speech_finished[req]=True`; append non-finished portion to `speech_ids[req]`. Requests that are still in text-only mode do not call `crq_generate_forward` at all, so `crq_past_key_values` / `crq_audio_embeds` remain `None` until `audio_bos` fires.
  4. `postprocess` — persist updated `CRQState` into `model_intermediate_buffer[req_id]`.
  5. Sampling / post-sample hook — after text token is sampled: if `force_text_abos_pending[req]` and this is the first post-prefill token, force it to `audio_bos_index` (ref L1381-1383); clear `force_text_abos_pending[req]`. Otherwise, if sampled token is `audio_bos_index`, flip `generate_speech[req] = True`. Note: `generate_speech` is set **after** sampling, so step 3 for *this* step cannot have used CRQ — CRQ first runs on the step *following* `audio_bos`, matching ref L1196-1208 semantics.

**Per-request state lifecycle (explicit edges):**
  - *Request admit / prefill*: engine initializes `CRQState(generate_speech=False, speech_finished=False, speech_ids=empty, crq_past_key_values=None, crq_audio_embeds=None, force_text_abos_pending=config.force_text_abos)`. Initialization lives in a `make_omni_output` / engine entry hook keyed by `req_id`.
  - *audio_eos reached*: set `speech_finished=True`; `crq_past_key_values`/`crq_audio_embeds` released (`None`) to free GPU memory. No new CRQ work scheduled for this req.
  - *Request completes normally (text EOS after speech_finished)*: engine pops `model_intermediate_buffer[req_id]`; `CRQState.__del__` / explicit `.release()` nulls the tensors. Must not hold references in Stage-0 module attributes.
  - *Client abort / scheduler eviction / timeout*: engine calls `on_request_cancelled(req_id)` which pops and releases the same way. Hook runs even if mid-CRQ-step (no partial state persists).
  - *Exception in `crq_generate_forward` for one request*: the per-request try/except marks `speech_finished=True` and logs; other requests in the batch are unaffected.
  - *`req_id` reuse*: engine guarantees req_id uniqueness within a run; on fresh admit, buffer key must not already exist — assert this at admit time.
- **Acceptance criteria:**
  - *Parity (BS=1, greedy):* on the reference smoke input (O10) with `temperature=0, text_greedy=True, force_text_abos=True`, Stage-0 emits the **identical** text-token sequence and the **identical** CRQ token sequence as the reference `model.generate(**inputs, **gen_kwargs)` run captured in UO4. Exact tensor equality.
  - *BS≥2 independence:* two concurrent requests with different audio inputs produce the same outputs as running them serially; shuffling batch order does not change either output (seeded greedy).
  - *Pre-BOS state invariant:* assert in a test that before the first `audio_bos` token is sampled, `CRQState.crq_past_key_values is None` and `CRQState.speech_ids.numel() == 0` for every step.
  - *Lifecycle:* unit test simulating (a) `audio_eos` mid-generation, (b) client abort, (c) exception raised inside `crq_generate_forward`, each asserts the `model_intermediate_buffer[req_id]` entry is released and the module holds no per-request tensors afterward.
- **Dependencies:** O4, O5.
- **Next action:** first implement `CRQState` + the per-request lifecycle hooks (admit/release/abort), then the flow. BS=1 parity test → BS=2 independence test → lifecycle tests.
- **Affected paths:** `vllm_omni/model_executor/models/fun_audio_chat/fun_audio_chat.py` (rewrite `forward`/`compute_logits`/`postprocess`/`make_omni_output`), `vllm_omni/model_executor/models/fun_audio_chat/crq_state.py` (new: `CRQState` dataclass + lifecycle helpers), `tests/model_executor/models/fun_audio_chat/test_crq_state_lifecycle.py` (new).

### O7. Native Stage-1 token2wav using the existing vllm-omni CosyVoice3 code
- **Goal:** replace `token2wav.py`'s external CosyVoice import with an in-tree bridge that uses `vllm_omni/model_executor/models/cosyvoice3/cosyvoice3_code2wav.py` (already native: flow_model + HiFT vocoder). Port the reference `token2wav()` helper from `src/funaudiochat/utils/cosyvoice_detokenizer.py` (chunked 30 s segments, merge-tail-if-short, `token_hop_len=25*30`, `pre_lookahead_len=3`) against the native code2wav API (not the external `CosyVoice3.model`). Default speaker embedding: ship `new_spk2info.pt`'s `"中文女".embedding` as a Parameter/Buffer loaded from the native Stage-1 module (weight file shipped alongside Stage-1 config, since it is 588 KB and already in the cloned reference). Remove `/wbl-fast/…` and HF-cache paths.
- **Acceptance criteria:** given the exact CRQ token list produced by reference S2S on a fixed input, Stage-1 produces a wav that (a) is valid (non-NaN, proper sample rate 24 kHz) and (b) Whisper ASR transcribes to the same text as reference-repo `token2wav()` output on the same tokens, modulo punctuation.
- **Dependencies:** B1; separate from O2–O6 (can proceed in parallel once weights are downloaded).
- **Next action:** read `vllm_omni/.../cosyvoice3/cosyvoice3_code2wav.py` full API surface; write a small diff script: feed ref's token list through both the reference detokenizer and the native one, compare waveforms (PESQ or simple waveform RMS-difference thresholds).
- **Affected paths:** `vllm_omni/model_executor/models/fun_audio_chat/token2wav.py` (rewrite), `vllm_omni/model_executor/models/fun_audio_chat/spk_embeddings.py` (new, loads `new_spk2info.pt` → default embedding).

### O8. Prompt replacement: `<|AUDIO|>` count aligned with reference (5 per sec)
- **Goal:** fix `FunAudioChatProcessingInfo.get_num_audio_tokens` and `FunAudioChatMultiModalProcessor._get_prompt_updates` so the number of `<|AUDIO|>` tokens matches the reference processor: compute from 25 Hz speech-token rate, `ceil(duration × 25 / group_size) = ceil(duration × 5)`. Also produce and cache `speech_ids` (all `audio_pad_token` per reference processor L196–200) in the mm kwargs so O3's audio_tower has an input path at prefill.
- **Acceptance criteria:** for a 3.2 s user audio (same as `examples/ck7vv9ag.wav`), vllm-omni places the same number of `<|AUDIO|>` placeholders and the same `input_ids` sequence as the reference processor for the same chat_template output.
- **Dependencies:** O1, O2 (for `_get_feat_extract_output_lengths`), O3.
- **Next action:** run ref processor on the example wav, dump `input_ids`; run our processor; diff.
- **Affected paths:** `vllm_omni/model_executor/models/fun_audio_chat/fun_audio_chat.py` (`FunAudioChatMultiModalProcessor`, `FunAudioChatProcessingInfo`).

### O9. Pipeline + deploy yaml adjustments
- **Goal:** update `vllm_omni/model_executor/models/fun_audio_chat/pipeline.py` and `vllm_omni/deploy/funaudiochat.yaml` to raise `max_num_seqs` above 1 (per user's BS≥1 requirement), drop `--max-num-seqs 1` hints in comments, and wire the new Stage-1 class. Keep `async_chunk: false` (no streaming this round per user's earlier "sync only" deploy yaml, which is the safe default — will re-check during review).
- **Acceptance criteria:** `vllm-omni serve`–style smoke starts Stage-0 + Stage-1 with `max_num_seqs=4` without asserting BS=1; a two-concurrent-request test returns two independent wavs.
- **Dependencies:** O6, O7.
- **Next action:** edit yaml + pipeline.py; smoke-run.
- **Affected paths:** `vllm_omni/deploy/funaudiochat.yaml`, `vllm_omni/model_executor/models/fun_audio_chat/pipeline.py`, `vllm_omni/model_executor/stage_input_processors/funaudiochat.py`.

### O10. Example + multi-layer acceptance test (parity-first, ASR supplemental)
- **Goal:** ship gates that actually catch the documented failure classes (truncation, repetition, EOS corruption, wrong content, wrong alignment). ASR alone cannot — it will happily transcribe garbage that happens to contain Chinese characters.
- **Deliverables:**
  1. `examples/offline_inference/fun_audio_chat/infer_s2s.py` — vllm-omni port of `src/funaudiochat/examples/infer_s2s.py`; writes wav + `tokens.json` (generated text tokens and CRQ tokens) so a human can hear it and a test can diff it.
  2. `tests/e2e/offline_inference/test_fun_audio_chat_s2s.py` with these assertions against the **reference baseline frozen in UO4** (`saves/reference/ck7vv9ag.tokens.json` + `saves/reference/ck7vv9ag.wav`):
     - *Token parity (primary gate):* with `temperature=0, text_greedy=True, force_text_abos=True, seed=42`, vllm-omni's text-token sequence is **exactly equal** to the reference's, and the CRQ token sequence is **exactly equal** up to the first audio_eos (tail after EOS is don't-care). Failure fails the build.
     - *Termination:* exactly one `audio_eos` is emitted; CRQ stream length is positive and within `[0.5×, 2×]` of the reference's length (catches silent-tail and runaway-repeat).
     - *No-repetition budget:* longest run of an identical CRQ token ≤ 10 (catches CRQ decoder getting stuck on one codebook index). The reference's longest run is measured during UO4 and set as the upper bound plus a small margin.
     - *Waveform sanity:* output wav is non-NaN, non-all-zero, 24 kHz, duration within `[0.5×, 2×]` of the reference wav.
     - *ASR supplemental (non-blocking, logged only):* Whisper transcribes both the native wav and the reference wav; report character-level edit distance and log to the test output. Not a gate on its own, but a regression signal.
  3. Skip the whole suite if the reference baseline or checkpoints aren't present (env-gated fixture, matching `tests/e2e/offline_inference/test_cosyvoice3.py` conventions).
- **Acceptance criteria:** token-parity + termination + no-repetition + waveform-sanity all pass on a clean run against the frozen reference baseline; skips cleanly when weights/baseline are absent.
- **Dependencies:** O6, O7, O8, O9, UO4 (baseline must be frozen and tracked).
- **Next action:** write the vllm-omni example first so it produces `tokens.json`; then land the four assertions; Whisper last.
- **Affected paths:** `examples/offline_inference/fun_audio_chat/infer_s2s.py` (new), `tests/e2e/offline_inference/test_fun_audio_chat_s2s.py` (new), `tests/e2e/offline_inference/fun_audio_chat_reference/` (new, tracks reference tokens.json + small wav).

### O11. Cleanup: delete external-path dependencies & stale code
- **Goal:** remove any lingering external-path fallbacks after O2–O7 land. Grep for `FUN_AUDIO_REF_PATH`, `/wbl-fast`, `src/funaudiochat` under `vllm_omni/` and delete those branches. The reference clone under `src/funaudiochat/` is kept *only* for side-by-side parity tests and is `.gitignore`d.
- **Acceptance criteria:** `rg -n 'FUN_AUDIO_REF_PATH|/wbl-fast|src/funaudiochat' vllm_omni/` returns nothing.
- **Dependencies:** O7.
- **Next action:** final sweep.
- **Affected paths:** `vllm_omni/model_executor/models/fun_audio_chat/*.py`, `.gitignore`.

---

## Open questions to confirm during implementation (not blockers)

- **`continuous_features_mode`** — default in ref config is `"replace"` but training-code config uses `"add"`; confirmed against the actual HF `config.json` once downloaded.
- **`force_text_abos` default** — user's current yaml is BS≥1 sync, but ref `infer_s2s.py` uses `force_text_abos=True`. Plan sets it True for S2S turns; re-evaluate if multi-turn hits cases where the first turn should *not* force abos.
- **`text_greedy` for BS>1** — ref infers with `text_greedy=True` for S2S. Works with temperature=0 + seed. Keep as default until acceptance check says otherwise.

---

## Changelog

- 2026-04-21 v1 — initial draft after interview + reference clone + root-cause diff against ref L100–1454.
- 2026-04-21 v1-review — `/codex:review` ran clean on the plan + .gitignore diff (sandbox had to be set to `danger-full-access` to escape bwrap; broker restarted). No major concerns flagged — but the built-in reviewer only looks at code diffs, not plan content.
- 2026-04-21 v2 — `/codex:adversarial-review` surfaced four concerns. Addressed in plan:
  1. [high] *CRQ state lifecycle undefined* → O6 now defines explicit init/audio_eos/cancel/eviction/exception/req_id-reuse edges and a `CRQState` dataclass; added lifecycle test file.
  2. [high] *CRQ would advance before `audio_bos`* (`text_greedy` was wrongly gating CRQ) → O6 step 3 now gates strictly on `generate_speech[req] and not speech_finished[req]`; added explicit pre-BOS invariant test.
  3. [high] *O10 ASR gate too weak* → O10 replaced with token-parity (primary gate) + termination + no-repetition + waveform-sanity; ASR demoted to non-blocking signal. Requires UO4 baseline to be frozen and tracked.
  4. [medium] *Reference baseline unreproducible* → addressed in the unblock plan v2 (lockfile + tracked smoke wrapper + frozen baseline artifact).
