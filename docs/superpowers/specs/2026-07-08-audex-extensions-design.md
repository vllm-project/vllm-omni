# Audex Extensions Design: CFG Phase 2, TTA, Speech2Text, Speech2Speech

Date: 2026-07-08
Status: Approved for design; implementation planning deferred (per user request).
Predecessor: `2026-07-08-audex-tts-design.md` (TTS v1, branch `audex-tts`, HEAD ecb06cb3).

## 1. Scope and decomposition

Four sub-projects extending the existing Audex TTS integration
(`nvidia/Nemotron-Labs-Audex-2B` in vllm-omni). Each gets its own
spec-sized slice; recommended build order and dependencies:

| # | Sub-project | Depends on | New checkpoint parts |
|---|---|---|---|
| 1 | CFG phase 2 (TTS guidance) | TTS v1 (done) | none |
| 2 | TTA (text-to-audio, caption → sound) | #1 (CFG mandatory, scale 3.0) | external XCodec1 decoder |
| 3 | S2T thinker-only (ASR / audio QA) | none (parallel to #1/#2) | `checkpoint_folder_full` (audio encoder + projector) |
| 4 | S2S (cascaded speech-in speech-out) | #3 + TTS v1 (#1 recommended) | `checkpoint_folder_full` + existing speech decoder |

All sub-projects reuse the v1 foundations: `ensure_audex_snapshot`
(repo-root path UX), the `audex/` model folder, the stage-input-processor
module, the `AudexAdapter` serving adapter, and the e2e test conventions
(no download at collection, `--run-level` gated quality asserts).

Out of scope (unchanged from v1): Chinese/multilingual support (empirically
unsupported by the checkpoint, even with CFG), voice cloning (fixed voice),
training/finetuning.

## 2. Sub-project 1 — CFG phase 2

### 2.1 What the official implementation does (reference)

`inference_scripts_vllm/audiogen_scripts/` implements AR-token CFG as three
cooperating pieces:

1. **Paired requests**: every generation submits a `cond` prompt and an
   `uncond` (null) prompt. The null prompt replaces the transcription with
   `<unk>` repeated, iteratively adjusted so the tokenized length exactly
   matches the cond prompt. Both carry
   `SamplingParams.extra_args = {cfg_scale, cfg_role, cfg_pair_id}`.
2. **`CFGLogitsProcessor`** (vLLM v1 `LogitsProcessor`): each step blends
   `uncond + scale * (cond - uncond)` and writes the result to both rows;
   it also patches `GPUModelRunner._sample` (once per worker process) to
   copy the cond sampled token into the uncond slot, so both sequences stay
   token-identical.
3. **Scheduler patches** (`vllm_cfg_patch.apply_cfg_patches()`): make the
   v1 `Scheduler` pair-aware — hold a lone pair member in `waiting` until
   its partner arrives, keep partners adjacent, equalize
   `num_computed_tokens` after each schedule (prefix-cache hits / chunked
   prefill can desync a pair), and finish both members together.

### 2.2 Gap in vllm-omni

vllm-omni's existing CFG machinery (`prompt_expand_func` +
`cfg_kv_collect_func`, used by bagel and ming_flash_omni) creates
**companion requests whose KV caches feed a diffusion stage**. There is no
per-step AR logits blending anywhere in the tree. We therefore combine both
worlds:

- reuse the **engine-layer companion mechanism** (`prompt_expand_func` →
  `_enqueue_cfg_companions` → `AddCompanionRequestMessage` with affinity
  routing) to create and admit the uncond request into the *same* stage-0
  engine as the cond request;
- vendor the **official logits processor + scheduler patches** to do the
  actual guidance inside that engine.

### 2.3 Design

New module `vllm_omni/model_executor/models/audex/cfg.py` (vendored +
adapted to vLLM 0.24):

- `AudexCFGLogitsProcessor` — the official `CFGLogitsProcessor`, minus the
  debug probe logging. Registered per-stage through the deploy yaml's
  stage `engine_args.logits_processors` (vLLM 0.24 `EngineArgs` accepts a
  list of import paths; vllm-omni stage engine-args dicts pass through).
- `apply_cfg_patches()` — the official scheduler patch. Constraint: it must
  run **in the stage-0 engine-core process before `Scheduler` is
  constructed**. Hook: module import side effect when vLLM resolves the
  `logits_processors` import path happens in the engine-core process during
  config construction; if that ordering proves fragile on 0.24, fall back
  to an explicit call in the stage-init path that vllm-omni already runs in
  the engine subprocess (`stage_init_utils`), gated on the audex stage
  name. The chosen hook is an implementation detail; the invariant to test
  is "patched before first `schedule()`".

Pair creation:

- `prompt.py`'s `build_null_prompt` (currently `NotImplementedError`) gets
  the official iterative `<unk>` length-matching implementation, sharing
  the ChatML template with `build_cond_prompt`.
- New `expand_cfg_prompts` in
  `vllm_omni/model_executor/stage_input_processors/audex.py`, wired as
  stage-0 `prompt_expand_func` in the pipeline. It emits one companion
  (`role="uncond"`, `request_id_suffix="__cfg_uncond"`) **only when** the
  request's sampling params carry `extra_args.cfg_scale > 1.0`; otherwise
  no companion, and behavior is byte-identical to v1 (this keeps
  `cfg_scale` absent/1.0 as the zero-cost default).
- The cond request's `extra_args` (`cfg_role="cond"`,
  `cfg_pair_id=<request id>`) are injected where the request is built: the
  serving adapter for online, the example scripts for offline.

Streaming interaction (async_chunk):

- The uncond companion also flows through the thinker's output stream, so
  `thinker2code2wav_async_chunk` / `_full_payload` must **skip companion
  requests** (external id endswith `__cfg_uncond`) — otherwise a second,
  duplicate codec stream reaches code2wav. This is a two-line guard plus a
  unit test.
- Token sync guarantees cond/uncond emit identical tokens, so the cond
  stream's timing behavior (first-chunk latency, holdback logic) is
  unchanged; throughput cost is the expected ~2× thinker compute per
  request.

Serving surface:

- `AudexAdapter` currently 400s on `cfg_scale != 1.0`; phase 2 lifts that:
  accepted range `[1.0, 10.0]`, default 1.0 (off). When >1.0 the adapter
  returns both the cond prompt and the cfg extra_args; the null prompt is
  built adapter-side (it owns the tokenizer-independent template; the
  length-matching needs the tokenizer, available via the serving context).
- Offline: `end2end.py` / `offline_benchmark_audex.py` gain `--cfg-scale`
  (default 1.0). Official TTS default is 1.5; we document 1.5 as the
  recommended quality setting but keep 1.0 the default so v1 baselines
  stay reproducible.

Validation / acceptance:

- Unit: null-prompt length parity across varied inputs; companion skip in
  chunk producers; logits blend math on synthetic rows.
- e2e offline at `full_model`: en-24 corpus at cfg 1.5 — CER ≤ v1 baseline
  (7.24% sequential / 6.58% batched) and pairwise A/B listenability spot
  check; identical-token invariant asserted between cond output and a
  probe of the uncond stream.
- e2e online: one request at cfg 1.5 streaming + non-streaming; cfg 1.0
  regression pass of the existing 4-test suite.
- Scheduler-patch stress: concurrency 4 with mixed cfg on/off requests —
  no pair desync (equalize counters exposed via debug logs), clean aborts.

## 3. Sub-project 2 — TTA (text-to-audio)

Caption → general audio (sound effects, music-ish). Same audiogen thinker
checkpoint; different token space, prompt, guidance strength, and decoder.

### 3.1 Token space and decoding

- Output tokens are `<audiocodec_N>` (4-codebook RVQ, codebook size 1024,
  **interleaved**: position p in the generated span belongs to phase
  `p mod 4`), bracketed by `<audiogen_start>` (131073) / `<audiogen_end>`
  (131074). Note: the tokenizer defines 8192 `<audiocodec_*>` symbols but
  the RVQ mapping uses 4×1024; the exact id→(phase, code) mapping is
  derived from the tokenizer at startup exactly as official
  `build_tta_phase_token_ids` does — verify the 8192-vs-4096 layout against
  the real tokenizer during implementation (a unit test pins it).
- **RVQ phase masking is required for validity**: vendored
  `TTARVQPhaseMaskLogitsProcessor` (official, ~217 lines) masks each step's
  logits to the current phase's codebook (phase 0 additionally allows
  `<audiogen_end>` on frame boundaries; an optional `codec_cap` forces
  end). Registered via `logits_processors` like the CFG processor; driven
  by `extra_args["tta_rvq"]` (phase token ids, start/end tids, cap).
- **CFG is mandatory** for TTA quality: default scale 3.0 (official). The
  null prompt uses the TTA template with `<unk>` filler
  (`<|text to audio|> Generate audio for this caption. ...`). Reuses all of
  sub-project 1.
- Waveform decoding uses **XCodec1**
  (`hf-audio/xcodec-hubert-general-balanced`) — an external checkpoint NOT
  inside the Audex repo. Path supplied via yaml stage `model` /
  `XCODEC1_PATH` env; `ensure_*` helper resolves repo-id → snapshot like
  `ensure_audex_snapshot` does.

### 3.2 Pipeline shape

New pipeline `nemotron_labs_audex_tta` + `deploy/nemotron_labs_audex_tta.yaml`
(selected explicitly, same mechanism as `ming_flash_omni_thinker_only.yaml`'s
`pipeline:` key):

- Stage 0: the existing `audex_thinker` (same `checkpoint_folder_audiogen`
  weights), stop token `<audiogen_end>`, TTA sampling defaults from the
  official script, `prompt_expand_func` for CFG, `logits_processors`
  = [CFG, TTA-RVQ-mask].
- Stage 1: new `AudexXCodec1` model (GenerationModelRunner, mirrors
  `AudexCode2Wav`'s interface: `input_modalities="audio"`,
  `have_multimodal_outputs`, `requires_raw_input_tokens`,
  per-request handling, `_OMNI_CONNECTOR_INIT_ARCHS` entries). **Sync
  full-payload only in v1 of TTA** (`async_chunk: false`): XCodec1 is a
  CNN codec the official flow decodes over the full sequence; chunked
  decode has boundary-artifact risk and no official reference. Streaming
  TTA is future work.
- New stage-input processor `thinker2xcodec_full_payload`: filters
  `<audiocodec_*>` ids from `output_token_ids` (offset/size from yaml, like
  the speechcodec path), de-interleaves into `[frames, 4]` payload, raises
  on zero codes (same serving-layer error contract as TTS).

### 3.3 Surfaces and validation

- Offline example `examples/offline_inference/text_to_audio/audex/end2end.py`
  with a small vendored caption list; registered in the offline benchmark
  skill as `audex_tta` (RTF/throughput only — ASR verification does not
  apply to non-speech audio).
- Online: reuse `/v1/audio/speech` with a distinct adapter
  (`name="audex_tta"`, stage_keys keyed off the TTA pipeline's stage
  names) — the endpoint shape (input text → audio bytes) fits; document
  that `voice` is ignored. This is the cheapest surface; if it turns out
  awkward it can ship offline-only first.
- Acceptance: N=8 captions decode to non-silent, RVQ-phase-valid audio
  (validity asserted on the token stream — official `validate_rvq_phase`
  logic as a test helper); A/B parity vs the official
  `run_audio_gen_vllm_rvq_logit_mask.py` on identical seeds/captions
  (token-level comparison, since waveform equality is decoder-identical).

## 4. Sub-project 3 — S2T thinker-only (ASR / audio understanding)

Speech (or general audio) + text instruction → text. Uses
`checkpoint_folder_full` (`NemotronDenseAudexForConditionalGeneration`,
model_type `nemotron_dense_audex`): the same 2B NemotronDense LM body plus a
Qwen2-Audio-architecture whisper encoder (32 layers, d_model 1280, 128 mel
bins, NV-Whisper weights) and a relu2 MLP projector; audio is chunked into
30 s clips, each expanding to 750 embedding positions replacing
`<so_embedding>`, wrapped in `<so_start>`/`<so_end>`.

### 4.1 Model port

Port the official `audex_2b_vllm` plugin (~700 lines total:
`modeling_audex_vllm.py`, `processing_audex_vllm.py`, `audio_features.py`,
`audio_encoder.py`) into `vllm_omni/model_executor/models/audex/`:

- `audex_omni.py` — `AudexForConditionalGeneration`
  (`SupportsMultiModal`): composes the **existing** `audex_thinker` LM body
  (weight-compatible; the full checkpoint's LM weights load into it), the
  audio encoder, and the projector. Prefer reusing vLLM upstream
  `Qwen2AudioEncoder` if the config maps cleanly (the checkpoint declares
  `qwen2_audio_encoder`); otherwise port the official 75-line encoder
  wrapper. Registered in `_OMNI_MODELS`.
- Multimodal processor — port `processing_audex_vllm.py` (mel features,
  30 s chunking, `<so_embedding>` → 750-token expansion), following the
  registry conventions vllm-omni uses for qwen2_5_omni's thinker.
- `ensure_audex_snapshot` grows `checkpoint_folder_full/*` patterns
  (config, tokenizer, **both** weight shards — v1 only pulls shard 1 of 2
  for the dedup symlink trick) when the requested pipeline needs it;
  pattern selection keyed by pipeline/model_stage so TTS-only deployments
  don't download the extra 1.3 GB shard (shard 1 is 4.2 GB, already pulled
  by v1 for the audiogen dedup symlink).

### 4.2 Pipeline (the part the user pointed at)

`NEMOTRON_LABS_AUDEX_THINKER_ONLY_PIPELINE`
(model_type `nemotron_labs_audex_thinker_only`) +
`deploy/nemotron_labs_audex_thinker_only.yaml`, modeled directly on
`ming_flash_omni_thinker_only.yaml` / `qwen2_5_omni_thinker_only`:

- Single stage 0: `audex_omni`, `LLM_AR`, `requires_multimodal_data: true`,
  `final_output_type: "text"`, `owns_tokenizer: true`, detokenize on,
  `model_subdir="checkpoint_folder_full"`.
- Sampling defaults from the official audioqa script (ASR uses
  temperature 0).

### 4.3 Surfaces and validation

- Online: OpenAI chat completions with `audio_url` content parts (the
  existing multimodal chat path; no new adapter needed — this is plain
  audio-in text-out).
- Offline example `examples/offline_inference/audio_language/audex/`
  mirroring the official `run_audioqa_vllm.py` prompts ("Transcribe the
  input speech." etc.).
- Acceptance: ASR WER on the en seed-tts references (transcribe our own
  v1 TTS oracle WAVs, compare to their texts — assets already vendored);
  parity spot-check against the official plugin on 5 clips; audioqa smoke
  (non-ASR instruction answered coherently at `full_model` level).

## 5. Sub-project 4 — S2S (cascaded speech-to-speech)

The official `unified_s2s_scripts` demo is an explicit **three-pass cascade
over one full-checkpoint engine**: (1) ASR pass — audio +
"Transcribe the input speech." → transcript (temp 0); (2) chat pass —
transcript in a chat template → text answer; (3) TTS pass —
`<|text to speech|> ...` + answer → `<speechcodec_*>` tokens → causal
speech decoder → waveform. We keep that cascade shape.

### 5.1 Pipeline

`NEMOTRON_LABS_AUDEX_FULL_PIPELINE` (model_type
`nemotron_labs_audex_full`) + `deploy/nemotron_labs_audex_full.yaml`:

- Stage 0: `audex_omni` on `checkpoint_folder_full` (audio-capable, and its
  LM head covers the speechcodec vocab, so it does all three passes),
  CFG `prompt_expand_func` attached (TTS pass benefits, default 1.5 like
  sub-project 1; guarded to only fire when `cfg_scale` present).
- Stage 1: the **existing** `AudexCode2Wav` + the existing
  `thinker2code2wav_*` processors, async_chunk on — unchanged.
- Per-request routing: ASR/chat passes are text-final and must not
  traverse stage 1. vllm-omni supports per-request final-stage selection
  (`final_output_stage_ids` on submission); text passes set final stage 0,
  the TTS pass runs the full 2-stage path. The existing zero-codec
  ValueError in the chunk producer therefore never sees a text pass.
  (If per-request routing turns out not to be exposed on the chosen
  serving path, fallback: the chunk producer skips requests whose prompt
  lacks `<speechgen_start>` — decided at implementation with a test either
  way.)

### 5.2 Orchestration and surfaces

- The cascade lives at the **client/example layer**, mirroring the
  official demo: `examples/online_serving/speech_to_speech/audex/client.py`
  issues pass 1→2→3 against one server (chat completions for passes 1–2,
  `/v1/audio/speech` with the audex adapter for pass 3 — the adapter
  accepts the answer text like any TTS input). No new server endpoint in
  v1; a fused single-request S2S endpoint is future work.
- Offline example: one script running the three passes via the Omni
  offline API on the full pipeline.

### 5.3 Validation

- e2e (full_model): spoken en question (vendored WAV) → transcript matches
  reference (WER gate) → answer non-empty → output WAV ASR-transcribes to
  the answer text (existing transcribe-tts-output skill), first-chunk
  latency recorded.
- Regression: TTS-only requests against the full pipeline match
  sub-project 1 quality gates (the full checkpoint's TTS pass vs the
  audiogen checkpoint's — small CER delta tolerated, threshold set from a
  first measurement).

## 6. Cross-cutting decisions

- **Naming**: stays "thinker" everywhere (established v1 convention).
- **Checkpoint layout**: all pipelines accept the repo root
  (`nvidia/Nemotron-Labs-Audex-2B`) and resolve subdirs via
  `ensure_audex_snapshot`; download patterns are pipeline-scoped so each
  deployment pulls only what it needs.
- **Vendoring**: CFG processor, scheduler patch, and RVQ mask processor are
  vendored under `models/audex/` with SPDX headers (Apache-2.0, same as the
  speech decoder), adapted to vLLM 0.24 APIs; divergences from the official
  files are commented at the divergence site only.
- **Testing conventions**: identical to v1 — unit tests colocated per
  module, e2e parametrized by repo id with no collection-time downloads,
  quality asserts gated on `--run-level`, OmniRunner teardown hygiene.
- **Env**: vllm024_venv; no new Python deps except whatever XCodec1 loading
  needs (checked at implementation; prefer transformers-native loading).

## 7. Open questions (deferred to implementation planning)

1. Exact hook for `apply_cfg_patches()` in the stage-0 engine-core process
   (import side effect vs stage-init call) — decide by testing patch/
   scheduler construction ordering on vLLM 0.24.
2. `<audiocodec_*>` 8192-vs-4096 vocab layout — pin with a tokenizer unit
   test before building the RVQ masks.
3. Whether upstream `Qwen2AudioEncoder` loads NV-Whisper weights directly
   or the official encoder wrapper must be ported.
4. Per-request `final_output_stage_ids` availability on the serving path
   used by the S2S cascade (fallback documented in §5.1).
5. TTA online surface: ship with the speech endpoint adapter or
   offline-only first (both acceptable; decide by effort at plan time).
