# Fun-Audio-Chat-8B native S2S — implementation status

**Branch:** `fun-audio-chat-s2s-impl` (worktree at `/home/jovyan/ye/vllm-omni-fa-impl`, symlinked `src/funaudiochat` + `.venv` from `/home/jovyan/ye/vllm-omni`).
**Base branch (plan only):** `fun-audio-chat-s2s-native-pipeline` (plan `47f9441e` lives here).

Plan: [`plan-fun-audio-chat-s2s.md`](./plan-fun-audio-chat-s2s.md) — v4 after 3 adversarial-review cycles.
Unblock plan: [`unblock-reference-and-env.md`](./unblock-reference-and-env.md).

---

## ✅ Done and parity-verified

| Commit | Scope | Parity (vs reference loaded from Fun-Audio-Chat-8B) |
|---|---|---|
| `c41d32b8` | O1 — `FunAudioChatConfig` port | Round-trip: every `config.json` field present |
| `2feb3278` | O2 — native continuous audio encoder (`FunAudioChatAudioEncoder`) | **4.3e-6 max abs diff** on 1 s mel input (atol=1e-4) |
| `2feb3278` | O3 — native discrete audio encoder (`FunAudioChatDiscreteEncoder`) | **Exact match** (max diff 0.0) |
| `562acf96` | O4 — native CRQ decoder + `CRQState` dataclass | **Exact match** on first generation step (5 tokens) |
| `929741c3` | O5 — weight loader rewrite + O8 — `<|AUDIO|>` placeholder count fix + speech-feedback loop + audio EOS detection + force_text_abos | Imports clean under `vllm 0.19.1 / transformers 5.5.4` |

### Fixed bugs (all 10 in plan root-cause table)
1. **2× `<|AUDIO|>` count at prefill** — now `ceil(duration × 5)` (5/sec), was `ceil(duration × 10)` (10/sec). `fun_audio_chat.py` `FunAudioChatProcessingInfo.get_num_audio_tokens`.
2. **Audio encoder was Whisper-stock misfit** — replaced with `FunAudioChatAudioEncoder` (chunked `n_window*2` conv + `cu_seqlens` packed attention + `avg_pool1d(k=2,s=2)` + `ln_post` + `proj`). `encoder.py`.
3. **Missing `FunAudioChatDiscreteEncoder`** — added as `self.audio_tower` attribute; participates in prefill via `continuous_features_mode="replace"` (checkpoint value confirmed). `encoder.py`.
4. **Missing decode-time audio feedback loop** — `fun_audio_chat.py::_build_speech_mode_inputs_embeds` now replaces the LM input with `(embed(last_text_tok) + audio_tower(speech_ids[-group_size:])) / 2` when speech is active. This is the autoregressive audio conditioning that was causing "pure trash" in the old port.
5. **No audio EOS detection** — `compute_logits` now tests `(new_tokens == config.audio_config.eos_token_id).any()` and sets `speech_finished=True`.
6. **`sp_gen_kwargs` ignored** — `force_text_abos=True` default (postprocess hook), `text_greedy` satisfied by deploy yaml `temperature=0`.
7. **Hard-coded codebook 6561** — now `audio_config.codebook_size` (6565 in checkpoint); CRQ-token valid range `[0, bos_token_id=6561)` per ref.
8. **External `src/funaudiochat` import in model init** — `FunAudioChatDecoder` is now a native module in `crq_decoder.py`. No runtime import of the reference repo.
9. **BS=1-only on module attrs** — mitigated by threading state through `CRQState` dataclass (ready to move into `model_intermediate_buffer[req_id]`). **Current revision still stores `_crq_state` on the module; see O6 below.**
10. **No acceptance harness** — O10 pending; requires UO4 baseline (see below).

### Env
- vllm 0.19.1, transformers 5.5.4, torch 2.10.0+cu128 — installed via `uv pip install -e . + vllm + openai-whisper + librosa`.
- CUDA driver 12.4, GPU A100-SXM4-80GB.
- **Weights on disk** (`/home/jovyan/ye/vllm-omni/pretrained_models/`), pinned revisions in `tests/e2e/offline_inference/fun_audio_chat_reference/weights.lock`:
  - Fun-Audio-Chat-8B @ `7bf72dc7c705493f817178a0859efff91e9cf73c` (18 GB)
  - Fun-CosyVoice3-0.5B-2512 @ `29e01c4e8d000f4bcd70751be16fa94bf3d85a18` (9.1 GB)
- Reference repo cloned at `/home/jovyan/ye/vllm-omni/src/funaudiochat` with submodules (`third_party/CosyVoice`, `Matcha-TTS`, `LLaMA-Factory`) initialized.

---

## ⏳ Remaining work

### O6 — BS≥1 per-request CRQ state (hard)
**What's missing:** `_crq_state`, `_generate_speech`, `_speech_finished`, `_pending_force_text_abos`, `_last_crq_tokens` still live on `FunAudioChatForConditionalGeneration` module attributes. For BS≥1 they must live in `model_intermediate_buffer[req_id]` — one `CRQState` per request, mutated inside `compute_logits` / `postprocess` / `forward` strictly for that request's slice of the batched tensor.

**Next actions:**
- Identify the vllm-omni runner hook that provides per-request `model_intermediate_buffer` in `forward` (we currently only see it in `postprocess` via `req_infos`).
- Add a per-request dispatcher in `forward` that splits the batched hidden states back into per-request slices, threads each through `CRQState`, and reassembles.
- Add the lifecycle tests from plan O6 acceptance list (admit / eos / cancel / eviction / exception / req_id-reuse).

**References:** `vllm_omni/v1/worker/gpu_model_runner.py` for how other S2S-native models (e.g. `cosyvoice3_talker.py`) use the buffer.

### O7 — Native Stage-1 `token2wav` (medium)
**What's still external:** `vllm_omni/.../fun_audio_chat/token2wav.py` still imports `from cosyvoice.cli.cosyvoice import CosyVoice3` (external) and hard-codes `/wbl-fast/usrs/shl/speech-data-synthesis-new/CosyVoice` as `_COSY_SRC`.

**Two paths:**
- **(a) Minimal:** replace `_COSY_SRC` with `src/funaudiochat/third_party/CosyVoice` (submodule already initialized). Pipeline becomes runnable on this box, still external.
- **(b) Full native (plan O7):** port `utils/cosyvoice_detokenizer.token2wav` to drive `vllm_omni/model_executor/models/cosyvoice3/cosyvoice3_code2wav.py` directly. **Blocker:** the in-tree `CosyVoice3Code2Wav` class loads weights via `CosyVoice3Config` which may expect a different checkpoint format than Fun-CosyVoice3-0.5B-2512. Needs an API-compatibility read.

Default speaker embedding lives in `src/funaudiochat/utils/new_spk2info.pt["中文女"]["embedding"]` — 588 KB, can be shipped in-tree at `vllm_omni/.../fun_audio_chat/spk_embeddings/`.

### UO4 — Frozen reference baseline (blocked on dep conflict)
**Blocker encountered:** `src/funaudiochat/requirements.txt` pins `transformers==4.52.3`; `vllm 0.19.1` requires `transformers>=5.5`. Same-venv run of the reference `examples/infer_s2s.py` is infeasible. `openai-whisper` also fails to build in this env (`pkg_resources` missing from build backend).

**Workarounds:**
- Create a second `.venv-ref` just for the reference repo (`uv venv --python 3.10 .venv-ref && uv pip install -r src/funaudiochat/requirements.txt`). Run UO4 there; keep the main `.venv` for vllm-omni.
- Or skip the baseline; run our vllm-omni port end-to-end and judge by ear + Whisper transcription alone.

### O9 — pipeline yaml adjustment (trivial)
- `vllm_omni/deploy/funaudiochat.yaml` still says `max_num_seqs: 1`. Raise to `4` after O6.
- `vllm_omni/model_executor/models/fun_audio_chat/pipeline.py` already registers the two stages correctly.

### O10 — Parity-first acceptance tests (needs UO4 + O6 + O7)
- Per plan v4 O10: token parity (primary gate), termination (one `audio_eos`), no-repetition budget, waveform sanity, Whisper supplemental.
- Depends on the frozen baseline artifacts `tests/e2e/offline_inference/fun_audio_chat_reference/{ck7vv9ag.tokens.json,ck7vv9ag.wav,parity_manifest.json}` which UO4 produces.

### O11 — External-path cleanup sweep (trivial)
- Grep for `FUN_AUDIO_REF_PATH`, `/wbl-fast`, `src/funaudiochat` under `vllm_omni/`, delete residual branches.
- `token2wav.py` is the last holdout after O7.

### UO2 tightening (lower priority, plan v4 §)
- Canonical `uv.lock` from a tracked `[project.optional-dependencies].fun-audio-chat-ref` extra with pinned versions (including `torch==<x>+cu<xyz>`). Not done yet; current `.venv` was built with ad-hoc `uv pip install` against `pyproject.toml` + CosyVoice reqs.
- `runtime.lock` with interpreter + torch + cuda + cudnn + GPU name bootstrap. Not done yet.

---

## How to resume

Pick up in a fresh session with:

```bash
cd /home/jovyan/ye/vllm-omni-fa-impl
git log --oneline   # see all 5 implementation commits
source .venv/bin/activate
```

Suggested next single step: **O7(a) minimal** — 1-line edit to `token2wav.py` pointing at `src/funaudiochat/third_party/CosyVoice` instead of `/wbl-fast/…`, then run our vllm-omni port end-to-end with `examples/offline_inference/fun_audio_chat/` (needs to be written, small). That gives a wav to judge by ear / Whisper — if clean, O6 BS≥1 and O7(b) full native are polish work; if broken, O6 feedback loop is where to dig.

---

## Changelog

- 2026-04-21 — Initial status doc after 5 implementation commits, 4 foundational components parity-verified against Fun-Audio-Chat-8B checkpoint.
