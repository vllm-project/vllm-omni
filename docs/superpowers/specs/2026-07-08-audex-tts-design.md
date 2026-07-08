# Audex (Nemotron-Labs-Audex-2B) TTS integration — design

Date: 2026-07-08
Status: approved (brainstorming); pending implementation plan

## Goal

Support `nvidia/Nemotron-Labs-Audex-2B` in vLLM-Omni for the TTS task, offline
and online (with chunk streaming), as a standard 2-stage pipeline. Other Audex
tasks (TTA, audio QA, S2S) are out of scope for v1 but must not be blocked by
this design. CFG (classifier-free guidance) is planned as phase 2; v1 runs
without it.

Environment: vLLM-Omni's pinned vLLM/transformers versions are authoritative.
The official `nemotron_dense_vllm_plugin` is NOT installed; its model code is
ported into this repo instead. No new pip dependencies.

## Model facts (from the HF repo)

- HF repo root is a manifest (`config.json`: `model_type:
  "nemotron_labs_audex"`) with task checkpoints in subfolders:
  - `checkpoint_folder_audiogen/` — TTS/TTA LLM. `architectures:
    ["NemotronDenseForCausalLM"]`, `model_type: nemotron_dense`, 2B dense
    (28 layers, hidden 2048, relu2 MLP, GQA 16/8, rope_theta 1e8), vocab
    205312 including `<speechcodec_N>` (single codebook) and
    `<speechgen_start>`/`<speechgen_end>` markers.
  - `audex_causal_speech_decoder/` — bundled HF trust_remote_code streaming
    codec decoder (fp32, 2.4 GB). `session = decoder.create_session(
    chunk_frames=N)`; `session.push([[code], ...])` yields incremental
    (sample_rate, samples) chunks; `session.flush()` drains the 4-frame
    lookahead tail. hop 320 @ 16 kHz → 50 codec fps.
- Official TTS prompt (ChatML), generation primed with `<speechgen_start>`:

  ```
  <|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n
  <|im_start|>user\n<|text to speech|> Generate speech for this transcription. {text}<|im_end|>\n
  <|im_start|>assistant\n<think></think><speechgen_start>
  ```

- Official sampling defaults for TTS: temperature 0.1, top_k 80, max_tokens
  2048, cfg_scale 1.5 (cond/uncond prompt pairs + logits fusion via a vLLM
  patch). cfg_scale 1.0 == plain single-sequence generation.
- No voice cloning / prompt-audio support exists in the model.

## Architecture (2 stages)

```
text ── ChatML prompt ──► Stage 0: audex thinker (LLM_AR)
                          NemotronDenseForCausalLM (ported), bf16
                          emits <speechcodec_N>, stops at <speechgen_end>
                             │ SharedMemoryConnector, async_chunk
                             ▼ payload: flat code ids (token_id − offset), Q=1
                          Stage 1: audex code2wav (LLM_GENERATION)
                          AudexCausalSpeechDecoder (fp32), per-request
                          streaming session, delta audio out, flush at EOS
                             ▼
                          16 kHz waveform (streamed chunks or full clip)
```

Naming: the AR stage is called **thinker** everywhere (files, processor
functions, `model_stage`) — not "talker". The vLLM model class keeps the name
`NemotronDenseForCausalLM` because it must match the checkpoint's
`architectures` field.

### Model path resolution

Users pass the repo root (`nvidia/Nemotron-Labs-Audex-2B` or a local
snapshot). The root manifest's `model_type: "nemotron_labs_audex"` selects the
pipeline; per-stage model paths resolve to subfolders:
stage 0 → `<root>/checkpoint_folder_audiogen`,
stage 1 → `<root>/audex_causal_speech_decoder`.
Open detail (resolve during planning): whether `StagePipelineConfig` already
carries a per-stage path/subfolder field or the redirection lives in the
config-loading layer. Both are viable.

### Precision & sampling

- Thinker: bf16 (official script default). Fallbacks if ASR quality fails:
  fp16 (official `--fp16`), then fp32. Explicit validation checkpoint —
  a bf16-only collapse happened before in this repo (cosyvoice3 HF-talker
  path, different mechanism, but cheap to check).
- Decoder: fp32 (native weights; fits single GPU alongside stage 0).
- Deploy-yaml sampling defaults: temperature 0.1, top_k 80, max_tokens 2048.
- v1 has no CFG (equivalent to cfg_scale 1.0).

## Components

### New files

| Path | Content |
|---|---|
| `vllm_omni/model_executor/models/audex/audex_thinker.py` | `NemotronDenseForCausalLM` ported from the HF repo's vLLM plugin (~450 lines, standard decoder-only: relu2 MLP, RMSNorm, partial-rotary RoPE), adapted to this repo's pinned vLLM interfaces, plus the thin omni-stage wrapper (`engine_output_type="latent"` output protocol). |
| `vllm_omni/model_executor/models/audex/audex_code2wav.py` | Stage-1 wrapper. Loads the bundled decoder via `AutoModel.from_pretrained(<root>/audex_causal_speech_decoder, trust_remote_code=True)`; if that proves brittle, vendor the 3 decoder .py files (~25 KB) instead. Per-request streaming session keyed by `_omni_req_id`, freed on finish/abort. `forward()` pushes codec frames, returns **delta** audio as `OmniOutput(multimodal_output={"audio": ...})`; on EOS calls `session.flush()` for the 4-frame lookahead tail. |
| `vllm_omni/model_executor/models/audex/pipeline.py` | `AUDEX_PIPELINE = PipelineConfig(model_type="nemotron_labs_audex", ...)` mirroring `qwen3_tts/pipeline.py`: stage 0 `LLM_AR`, `model_stage="audex_thinker"`, `sampling_constraints={"detokenize": False, "stop_token_ids": [...]}` (the `<speechgen_end>` token id, resolved from the tokenizer at init alongside the codec offset); stage 1 `LLM_GENERATION`, `model_arch="AudexCode2Wav"`, `final_output_type="audio"`. |
| `vllm_omni/model_executor/stage_input_processors/audex.py` | ChatML prompt build (`build_cond_prompt`; `build_null_prompt` stub reserved for CFG phase 2). `thinker2code2wav_async_chunk` / `thinker2code2wav_full_payload`: filter `<speechcodec_N>` tokens, subtract vocab offset, ship flat single-codebook frame payload to stage 1. |
| `vllm_omni/deploy/audex.yaml` | `async_chunk: true`; SharedMemoryConnector with `codec_streaming: true`, `codec_chunk_frames: 25`, `initial_codec_chunk_frames: 1`. Stage 0: bf16, `max_num_seqs ≥ 4`, `max_model_len 4096`, official sampling defaults. Stage 1: session-mode decoder (no KV usage); memory knobs copied from qwen3_tts structure, values tuned empirically. |
| `examples/offline_inference/text_to_speech/audex/end2end.py` | Offline example (text in → WAV out), shaped like higgs_v3/qwen3_tts examples; drives ASR validation. |

### Edits to existing files

1. `vllm_omni/model_executor/models/registry.py` — register
   `NemotronDenseForCausalLM` → `audex.audex_thinker` and `AudexCode2Wav` →
   `audex.audex_code2wav` (lazy module paths).
2. `vllm_omni/config/pipeline_registry.py` —
   `OMNI_PIPELINES["nemotron_labs_audex"] = AUDEX_PIPELINE`; root-manifest →
   subfolder path resolution hook (here or in config loading).
3. `vllm_omni/entrypoints/openai/serving_speech.py` — standard 5-point
   integration: stage constant, `_TTS_MODEL_STAGES` union,
   `_detect_tts_model_type()`, request validation (non-empty `input`; `voice`
   set → 400 "voice cloning not supported"), param builder.
4. `.gitignore` — add the currently untracked local files (`.humanize/`,
   `CLAUDE.local.md`, local benchmark scripts) so they stay untouched.

### Data-flow invariants (repo conventions)

- Token→code mapping: `<speechcodec_N>` occupy a contiguous vocab block; the
  offset is discovered once at init by scanning the tokenizer (official
  `build_codec_token_maps` approach), never hardcoded.
- Streaming returns delta audio only (invariant I1); per-request state keyed
  by `_omni_req_id` (I5); no `.item()`/CPU branching in hot loops (I3).
- End-of-stream: whether the thinker stops at `<speechgen_end>` or at
  max_tokens, stage 1 must receive the EOS signal and flush — otherwise the
  final ~80 ms (4 lookahead frames) is silently lost. Integration-specific
  hazard; test explicitly.

## Error handling

| Scenario | Behavior |
|---|---|
| No codec tokens generated | Request-level error (online: error response; offline: sample marked failed). No silent blank-WAV fallback. |
| Hit max_tokens without `<speechgen_end>` | Normal finish: decode accumulated codes + flush; log a warning (expected for over-long text). |
| Decoder `push()`/`flush()` raises | Request-level failure; session force-freed; other in-flight requests unaffected. |
| Client abort / request cancel | Stage-1 session cleaned up with `_omni_req_id`; no leak. |
| `voice` or other unsupported params | 400 at the serving layer. |

## Testing & validation

1. **Port-correctness anchor**: run the official `run_audio_gen_vllm.py` with
   `cfg_scale=1.0` in this environment first; its WAVs + ASR results are the
   comparison baseline and answer "is no-CFG quality acceptable?".
2. **Offline e2e**: `end2end.py` WAVs → `transcribe-tts-output` skill, all OK
   (0 NOISE/FAIL); validate batch=1 AND batch>1 (cosyvoice3 precedent of
   batch>1-only corruption).
3. **Online smoke**: `vllm-omni serve <root> --omni` → `/v1/audio/speech`
   non-streaming + streaming; streaming must deliver an early first chunk and
   total audio length equal to non-streaming (flush not lost).
4. **Concurrency & perf**: `vllm-omni-online-tts-benchmark` concurrency sweep
   (TTFP/RTF/underrun); register `audex` in
   `vllm-omni-offline-tts-benchmark`.
5. **Repo tests**: `tests/e2e/offline_inference/test_audex.py` (L3 smoke) +
   online L2/L3 per the `vllm-omni-test` CI levels.

## CFG — phase 2 (not in v1)

v1 interfaces must not block CFG, and must not be restructured for it:

- Reserved now: serving request + sampling params accept `cfg_scale`; v1
  allows only 1.0 and returns "not yet supported" for >1.0 (never silently
  ignores). Prompt building is split into `build_cond_prompt` /
  `build_null_prompt` (the null variant = official `<unk>` length-matched
  padding); v1 uses only the former.
- Phase-2 sketch: stage 0 expands one user request into a cond/uncond pair,
  pair-aware scheduling keeps both in lockstep prefill/decode (the official
  `vllm_cfg_patch` idea, implemented in vllm-omni's stage scheduling rather
  than patching vLLM), a logits processor fuses
  `uncond + scale·(cond−uncond)`, and only the cond sequence's tokens flow to
  stage 1. Known cost: ~2× KV, ~half throughput.
- Trigger: if the step-1 baseline shows cfg 1.0 ASR quality clearly worse
  than cfg 1.5, phase 2 is pulled forward.

## Out of scope (v1)

- Voice cloning / prompt audio (the model does not support it at all).
- TTA, audio QA, S2S tasks (future; the pipeline registry keying by root
  manifest `model_type` leaves room for task-specific stage graphs later).
- CFG execution (phase 2, above).
