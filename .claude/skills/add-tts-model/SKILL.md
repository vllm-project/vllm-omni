---
name: add-tts-model
description: "Integrate a new text-to-speech model into vLLM-Omni from HuggingFace reference implementation through production-ready serving with streaming and CUDA graph acceleration. Use when adding a new TTS model, wiring stage separation for speech synthesis, enabling online voice generation serving, debugging TTS integration behavior, or building audio output pipelines."
---

# TTS Model Integration Workflow

## Overview

```
HF Reference -> Stage Separation -> Online Serving -> Async Chunk -> CUDA Graph -> Pre-commit/DCO
   (Phase 1)      (Phase 2)          (Phase 3)        (Phase 4)     (Phase 5)      (Phase 6)
```

Three architecture patterns are supported:

- **Two-stage pipeline** (e.g. Qwen3-TTS, Fish Speech, CosyVoice3): AR
  code-predictor → audio decoder, connected via async_chunk for low-latency
  streaming. Use this for maximum performance.
- **Single-stage AR via generator** (e.g. MOSS-TTS-Nano): entire model runs
  inside one AR worker, streaming audio chunks through a per-request
  `inference_stream()` generator. Use this when the upstream model bundles AR
  + codec inseparably. See [references/single-stage-ar.md](references/single-stage-ar.md).
- **Single-stage, vLLM-native base LM + side computation** (e.g. VoxCPM2):
  the base language model runs under vLLM's PagedAttention as a normal AR
  model; diffusion / VAE / side computations run outside vLLM and are
  attached via the runner post-processing hook. This is a distinct pattern
  from the generator approach above — do not confuse the two.

The single-stage variants skip Phase 4 (async_chunk) but Phase 5 (CUDA graph)
is still encouraged for the inner AR loop.

## Cross-Cutting Invariants

These rules apply to every TTS model regardless of architecture (AR vs AR+diffusion, single-stage vs two-stage, codec-based vs VAE-based). They surface repeatedly across PRs — check them at the end of every phase.

### I1. Streaming output contract

Pick exactly one per-step semantics for `forward()` and document it in the docstring:

- **Delta**: yield only new audio samples produced this step. Preferred — linear cost, low memory.
- **Cumulative**: re-decode from step 0 every call. O(N²); only acceptable if the codec has no streaming decode path.

If you choose **delta**, verify the full emit→consolidate→consume chain:

1. `forward()` returns `{"model_outputs": <new_chunk_only>, ...}`
2. `_consolidate_multimodal_tensors()` in `vllm_omni/engine/output_processor.py` concatenates the audio key into one tensor at finish. If it skips the key (`continue`), offline consumers receive only the final chunk. See `output_processor.py` for the concrete list of handled modality keys.
3. Streaming consumers (SSE, Gradio) receive per-step deltas; offline consumers (`engine.generate()`) receive a single concatenated tensor.

Cumulative-vs-delta mismatch is the most common silent bug — offline RTF benchmarks pass, but users hear replays or truncation.

### I2. Multimodal output consumer hygiene

`outputs[0].outputs[0].multimodal_output[<key>]` can be any of `Tensor`, `list[Tensor]` (pre-consolidation snapshot), `np.ndarray`, or scalar. When writing tests, examples, and benchmarks:

- **Never** use `dict.get("a") or dict.get("b")` on tensor values — Python evaluates the tensor's boolean, raising `RuntimeError: Boolean value of Tensor with more than one value is ambiguous`. Use explicit `if x is None` chains.
- Always defensively handle the list form: `if isinstance(x, list): x = torch.cat([t.reshape(-1) for t in x], dim=0)`.
- Assert `shape` / `dtype` / `duration` explicitly; do not rely on truthiness for presence checks.

### I3. Hot-loop GPU discipline

Inside any per-step model loop (AR decode, diffusion solver, CFM Euler, vocoder block loop):

- No `tensor.item()`, `.cpu()`, or `.tolist()` — each triggers a GPU→CPU sync; at 10 steps × 60 frames × 4 ops that is 2400 syncs per request.
- Prefer `dst.copy_(src)` over `dst.fill_(src.item())` when writing a scalar tensor into a buffer.
- Prefer `torch.compile(Model.forward, fullgraph=False)` on the whole forward over per-submodule compile — fewer dispatch boundaries, larger fusion regions. Measure before choosing granularity.
- No Python-side control flow that depends on tensor values; use `torch.where` / masking instead.

Profile first, optimize second. See the profiling docs / project memory for the trace-analysis workflow.

### I4. Validation pyramid

Offline RTF alone is necessary but not sufficient. Every new TTS model must pass all three:

| Layer | Catches | Tool |
|-------|---------|------|
| Offline RTF / duration check | Throughput regressions, missing audio, wrong sample rate | `end2end.py`, pytest e2e |
| Browser streaming playback | Delta/cumulative bugs, chunk boundary glitches, TTFP regressions | Gradio demo over `/v1/audio/speech?stream=true` |
| Concurrent requests | Per-request state leaks, codec window round-robin gaps | `max_num_seqs>1` smoke test with 4+ parallel prompts |

Declaring a model "done" without all three has shipped regressions more than once.

### I5. Per-request state is owned by the request, not the model

If the model caches *anything* across `forward()` calls (streaming generators, codec buffers, sliding-window pads, CUDA graph state), key it by request ID:

```python
self._state: dict[str, YourState] = {}    # request_key → state
# fetch: request_key = str(info.get("_omni_req_id", "0"))
# free on finish: del self._state[request_key]
```

A shared buffer silently corrupts audio across concurrent requests — the symptom is crosstalk or truncation only under load.

## Phase 1: HuggingFace Reference

**Goal**: Understand the reference implementation and verify it produces correct audio.

### Steps

1. **Run the reference model** end-to-end using the official HuggingFace / GitHub code
2. **Document the architecture**:
   - What are the sub-models? (AR decoder, codec decoder, vocoder, etc.)
   - What is the token vocabulary? (semantic codes, RVQ codebooks, special tokens)
   - What is the output format? (sample rate, channels, codec type)
3. **Capture reference outputs** for comparison during integration
4. **Identify the config structure**: `config.json` fields, `model_type`, sub-model configs

### Key Questions

- How many codebooks? What are the codebook sizes?
- What special tokens exist? (`<|voice|>`, `<|audio_start|>`, `<|im_end|>`, etc.)
- What is the token-to-ID mapping for codec codes?
- What is the hop length / frame rate of the codec?
- Does the model support voice cloning? How? (reference audio encoding, speaker embeddings, etc.)

### Deliverables

- Working reference script that produces audio
- Architecture diagram / notes
- Token vocabulary mapping
- Reference audio samples for regression testing

## Phase 2: Stage Separation (Offline Inference)

**Goal**: Split the model into vLLM-Omni stages and get offline inference working.

### Steps

1. **Register the model** in `vllm_omni/model_executor/models/registry.py`
2. **Create config classes** (`configuration_<model>.py`) with `model_type` registration
3. **Implement Stage 0** (AR model):
   - Subclass appropriate base (e.g., wrap Qwen3 decoder layers)
   - Implement `forward()` for autoregressive token generation
   - Handle special token logic (start/stop tokens, codec token mapping)
   - If dual-AR (like Fish Speech), implement Fast AR as a nested module
4. **Implement Stage 1** (Decoder):
   - Load codec weights (may need lazy loading from separate checkpoint)
   - Implement `forward()`: codec codes -> audio waveform
   - Return `OmniOutput` with `multimodal_outputs`
5. **Create stage config YAML** defining both stages, memory allocation, and model paths
6. **Create stage input processor** for prompt building
7. **Write end2end.py** test script

### Critical Parameters to Get Right

| Parameter | Impact if Wrong |
|-----------|----------------|
| Hop length | Audio duration wrong, streaming noise |
| Token ID mapping | Garbage codes -> noise output |
| Codebook count/size | Shape mismatch crashes |
| Stop token | Generation never stops or stops too early |
| dtype / autocast | Numerical issues, silent quality degradation |
| Repetition penalty | Must match reference (often 1.0 for TTS) |

### Debugging Priority (from experience)

When audio output is wrong, check in this order:

1. **RoPE / attention**: Are position encodings correct? Is the attention mask right?
2. **Normalization**: RMSNorm epsilon, layer norm placement (pre vs post)
3. **Hop length**: Product of all upsample rates in the codec decoder
4. **Token mapping**: Are codec IDs correctly offset from the vocabulary base?
5. **Sampling parameters**: Temperature, top_k, top_p, repetition_penalty
6. **Tensor layout**: Codebook-major vs frame-major ordering
7. **dtype**: Float32 for codec decoders (autocast can corrupt audio)

### Streaming Correctness Rules (single-stage and two-stage)

These bugs appear in almost every new TTS PR. Check all before the first push. See also the cross-cutting invariants I1 (output contract) and I5 (per-request state) above — the rules below are the Phase 2-specific instances of those invariants:

- **Accumulate codes across AR steps** — each `forward()` appends new codes; do not reset between steps or audio will be truncated (fish speech: `fix: accumulate audio_codes across steps`)
- **Emit delta audio, not full waveform** — in streaming mode yield only the new chunk per step, not the re-decoded full waveform from step 0 (fish speech: `fix: emit delta audio not full waveform`)
- **All return paths must emit `model_outputs`** — if any early-return branch skips setting `model_outputs`, the serving layer silently drops that step's audio (fish speech: `fix: ensure ALL return paths emit model_outputs`)
- **Per-request state isolation** — for batched concurrent requests, key all state by request ID; a shared buffer corrupts audio across requests (fish speech: `fix: per-request vocode + delta emission`)
- **Codec tensor device** — move codec codes to the codec decoder's device before calling decode; mismatches cause silent CPU fallback or crashes (fish speech: `fix: use model device for CUDA stream`)
- **AR stage `max_num_seqs`** — set to at least 4 in production configs; for single-stage models this is the only stage. For two-stage models, Stage 0 (AR) needs `max_num_seqs ≥ 4` to pipeline concurrent requests; Stage 1 (codec decoder) typically uses `max_num_seqs: 1` intentionally. Default of 1 everywhere causes audio gaps under concurrency because the codec window round-robins across requests (RFC #2568)

### Optional Dependency Handling

Patch optional dependencies (`torchaudio` / `torchcodec` / `soundfile`) at
the top of `load_weights()`, not at module import. Failures to do so cause
cryptic errors only on environments missing the optional package — after
the model is already deployed. See
[references/optional-deps.md](references/optional-deps.md) for the full
pattern, signature constraints, and MOSS-TTS-Nano reference.

### Single-Stage AR Pattern (alternative to two-stage)

When the upstream model cannot be cleanly split into an AR stage and a
separate decoder, run the full pipeline inside a single AR worker and
stream audio through a per-request `inference_stream()` generator keyed by
`_omni_req_id`. Stage config must set `worker_type: ar`,
`engine_output_type: audio`, `final_output: true`, `is_comprehension: true`,
and `async_chunk: false` at the top level. Only extract params from
`additional_information` that you actually forward, or pre-commit fails
`ruff F841`.

Full walkthrough with the complete `forward()` / `_create_stream_gen()`
skeleton and stage-config fields:
[references/single-stage-ar.md](references/single-stage-ar.md). For an
in-tree reference, look for any single-stage AR model under
`vllm_omni/model_executor/models/` — e.g. the MOSS-TTS-Nano integration when
it lands.

**VoxCPM2 is a different pattern** and should not reuse this skeleton — it
runs the base LM under vLLM PagedAttention with external side-computation.
See `plan/voxcpm2_native_ar_design.md`.

### Deliverables

- Model files in `vllm_omni/model_executor/models/<model_name>/`
- Stage config YAML
- Working `end2end.py` at `examples/offline_inference/text_to_speech/<model>/end2end.py`
- New section in `examples/offline_inference/text_to_speech/README.md` (table row + per-model section). Do **not** create a top-level `examples/offline_inference/<model>/` dir or a per-model `README.md` inside `text_to_speech/<model>/` — the hub README is the documented surface and the mkdocs `generate_examples` hook only descends one level into `examples/<category>/`.

## Phase 3: Online Serving

**Goal**: Expose the model via `/v1/audio/speech` API endpoint.

### Steps

1. **Register in `serving_speech.py`** — add all 5 points in a **single commit**;
   partial integration causes hard-to-debug failures. This file is modified by every
   model PR and is the most common source of rebase conflicts — see conflict note below.

   **Point 1** — stage constant (near the top, alongside the other `_*_TTS_MODEL_STAGES` sets):
   ```python
   _YOUR_MODEL_TTS_MODEL_STAGES = {"your_stage_key"}
   ```

   **Point 2** — union into `_TTS_MODEL_STAGES`:
   ```python
   _TTS_MODEL_STAGES: set[str] = (
       ...
       | _YOUR_MODEL_TTS_MODEL_STAGES
   )
   ```

   **Point 3** — model type detection in `_detect_tts_model_type()`:
   ```python
   if model_stage in _YOUR_MODEL_TTS_MODEL_STAGES:
       return "your_model"
   ```

   **Point 4** — validation dispatch in `_validate_tts_request()`:
   ```python
   if self._tts_model_type == "your_model":
       return self._validate_your_model_request(request)
   ```

   **Point 5** — validation + parameter-builder methods:
   ```python
   def _validate_your_model_request(self, request) -> str | None:
       if not request.input or not request.input.strip():
           return "Input text cannot be empty"
       return None

   def _build_your_model_params(self, request) -> dict:
       params = {"text": [request.input]}
       if request.voice is not None:
           params["voice"] = [request.voice]
       return params
   ```
   Wire `_build_your_model_params` into `_create_tts_request()` alongside the other
   model-specific param builders.

   > **Two dispatch patterns coexist**: Fish Speech uses a `self._is_fish_speech` boolean
   > instance attribute checked before `elif self._is_tts`, while all newer models
   > (CosyVoice3, MOSS-TTS-Nano) use the `_tts_model_type` string returned by
   > `_detect_tts_model_type()`. For new models, always use the `_tts_model_type` string
   > pattern — do not add new `_is_*` flags.

   > **Unused variable rule**: only extract fields in `_build_your_model_params` that
   > are actually forwarded to the model. Unused extractions fail `ruff F841`.
   > For voice-cloning fields (`ref_audio` → `prompt_audio_path`, `ref_text` →
   > `prompt_text`), add them to the param builder and verify they reach the model call.

   **Rebase conflict note**: when rebasing onto `main` after another model was merged,
   `serving_speech.py` will conflict. Resolution: always keep *both* the upstream
   model's additions and your own — never discard either side.

2. **Handle model-specific parameters**:
   - Voice cloning: `ref_audio` encoding and prompt injection
   - `max_new_tokens` override in sampling params
   - Model-specific default values
3. **Create client scripts**: `speech_client.py`, `run_server.sh`
4. **Test all response formats**: wav, mp3, flac, pcm
5. **Add Gradio demo**: Interactive web UI with streaming support

### Voice Cloning Pattern

```python
import base64
from pathlib import Path

def build_voice_clone_prompt(ref_audio_path: str, text: str, codec) -> list:
    """Build prompt with reference audio for voice cloning in serving_speech.py."""
    audio_bytes = Path(ref_audio_path).read_bytes()
    codes = codec.encode(audio_bytes)  # Encode on CPU using model's codec (e.g., DAC)
    token_ids = [code + codec.vocab_offset for code in codes.flatten().tolist()]
    return [
        {"role": "system", "content": f"<|voice|>{''.join(chr(t) for t in token_ids)}"},
        {"role": "user", "content": text},
    ]
```

### Deliverables

- Updated `serving_speech.py` with all 5 integration points (single commit)
- Client scripts and server launcher under `examples/online_serving/text_to_speech/<model>/`
- Gradio demo with streaming and voice cloning UI in the same dir
- E2E online serving test (`tests/e2e/online_serving/test_<model>.py`)
- Buildkite CI entry in `.buildkite/test-merge.yml`
- New section in `examples/online_serving/text_to_speech/README.md` (table row + per-model section). Do **not** create a top-level `examples/online_serving/<model>/` dir or a per-model `README.md` inside `text_to_speech/<model>/`.

### E2E test pitfalls to avoid

- **One `OmniServerParams` set per file.** `omni_server` is module-scoped; a second
  id in the same file forces mid-module teardown/restart and exposes startup
  races (`APIConnectionError` on the first request post-restart). Split variants
  into separate files instead.
- **No external URL fetches from the server.** CI and some dev hosts can't
  reach `raw.githubusercontent.com` over TLS. Inline ref audio as
  `data:audio/wav;base64,...`; the serving layer accepts both URL and data URL.
- **Use the harness readiness gate.** The fixture waits for HTTP 200 on
  `/health`; don't add `time.sleep` in tests. If warmup is incomplete, make
  `/health` return non-200 until you're actually ready.
- **Mark with `@pytest.mark.core_model` + `hardware_test(res={"cuda": "H100"})`**
  so the test lands in `test-ready.yml` (triggered by the `ready` label) rather
  than only nightly.

## Phase 4: Async Chunk (Streaming)

**Goal**: Enable inter-stage streaming so audio chunks are produced while AR generation continues.

### Steps

1. **Update stage config YAML**:
   ```yaml
   async_chunk: true
   codec_chunk_frames: 25      # frames per chunk
   codec_left_context_frames: 25  # overlap for smooth boundaries
   ```
2. **Implement chunk handling in Stage 1**:
   - Accept partial input (chunk of codec codes)
   - Handle left context for smooth audio boundaries
   - Return partial audio in `OmniOutput`
3. **Test streaming**:
   - Verify audio quality matches non-streaming output
   - Check for artifacts at chunk boundaries
   - Measure TTFA (time to first audio)
4. **Update online serving** to support `stream=true` with PCM output

### Streaming Architecture

```
Stage 0 (AR)                    Stage 1 (Decoder)
  |                                |
  |-- chunk 0 (25 frames) ------> decode -> audio chunk 0 -> client
  |-- chunk 1 (25 frames) ------> decode -> audio chunk 1 -> client
  |-- chunk 2 (25 frames) ------> decode -> audio chunk 2 -> client
  ...
```

### Key Considerations

- **Left context overlap**: Prevents audible artifacts at chunk boundaries
- **Hop length matters**: `context_audio_samples = context_frames * hop_length`
- **First chunk latency**: Can use larger initial chunk for better quality, then smaller chunks

### Deliverables

- Updated stage config with async_chunk enabled
- Smooth streaming audio without boundary artifacts
- TTFA metrics

## Phase 5: CUDA Graph Acceleration

**Goal**: Capture the AR loop as a CUDA graph for significant speedup.

### Steps

1. **Identify the hot loop**: The AR decoding loop that runs N steps per token
2. **Create static buffers**:
   - KV caches with fixed max sequence length
   - Pre-built causal masks and position tensors per step
   - Static input/output tensors
3. **Implement graph capture**:
   - Warm up with real data
   - Capture the forward pass
   - Replay with updated inputs
4. **Handle constraints**:
   - Use `torch.argmax` instead of `torch.multinomial` (graph-safe)
   - Fixed batch size (fall back to eager for other sizes)
   - No dynamic control flow inside the graph

See [references/cuda-graph-example.md](references/cuda-graph-example.md) for
a worked skeleton (Qwen3-TTS code predictor, 16-step AR loop), performance
expectations (3–5× on the graphed component for fixed batch_size=1), and the
graph-safety constraints you must honor inside the captured region.

### Deliverables

- CUDA graph implementation for the AR hot loop
- Benchmark script comparing eager vs graph performance
- Documentation of constraints and fallback behavior

## Phase 6: Pre-commit and DCO

**Goal**: Every commit passes `pre-commit` lint and carries a DCO
`Signed-off-by` line that matches the author email.

- Install hooks once: `pre-commit install`.
- Run `pre-commit run --files <changed-files>` before every push; accept any
  auto-fixes, stage, re-commit.
- Sign every commit with `git commit -s`. DCO checks that author email and
  `Signed-off-by` email match — `git config user.email` must match your
  GitHub account email.

Common pre-commit failures, recovery commands for missing sign-off, and the
full `pre-commit run` invocation for a TTS model:
[references/precommit-dco.md](references/precommit-dco.md).

## Integration Checklist

Use this checklist when integrating a new TTS model:

### Cross-Cutting Invariants (verify at end of every phase)
- [ ] I1: `forward()` docstring states cumulative vs delta; consolidation path audited end-to-end
- [ ] I2: Tests / examples / benchmarks never use `dict.get(a) or dict.get(b)` on tensor values; list form handled
- [ ] I3: No `.item()` / `.cpu()` / Python branch on tensor values inside per-step loops
- [ ] I4: Offline RTF, browser streaming playback, and concurrent-request smoke test all pass
- [ ] I5: Any cross-step cache keyed by `_omni_req_id`; entries freed when the request finishes

### Phase 1: HF Reference
- [ ] Reference model runs and produces correct audio
- [ ] Architecture documented (stages, codebooks, tokens, sample rate)
- [ ] Reference audio samples saved for comparison

### Phase 2: Stage Separation
- [ ] Model registered in `registry.py`
- [ ] Config classes created with `model_type` registration
- [ ] Stage 0 (AR) implemented and generates correct tokens
- [ ] Stage 1 (Decoder) produces correct audio from tokens — dtype float32 for codec decoder
- [ ] Stage 1 `max_num_seqs` ≥ 4 in production config (default 1 causes gaps under concurrency)
- [ ] Optional dependency fallbacks handled at `load_weights()` time (torchaudio/soundfile/etc.)
- [ ] Streaming: codec codes accumulated across AR steps (not reset per step)
- [ ] Streaming: delta audio emitted per chunk, not full re-decoded waveform
- [ ] Streaming: all `forward()` return paths emit `model_outputs`
- [ ] Streaming: per-request state keyed by request ID (not shared across requests)
- [ ] Streaming: codec tensors moved to codec decoder device before decode
- [ ] Stage config YAML created
- [ ] `end2end.py` produces audio matching reference quality
- [ ] README.md written

### Phase 3: Online Serving
- [ ] All 5 `serving_speech.py` integration points added in one commit
- [ ] Only extract params in `_build_*_params` that are forwarded to the model call (ruff F841)
- [ ] Prompt builder handles text input correctly
- [ ] Voice cloning works (if supported)
- [ ] All response formats work (wav, mp3, flac, pcm)
- [ ] Client scripts and server launcher created
- [ ] E2E online serving test written (`tests/e2e/online_serving/test_<model>.py`)
- [ ] Buildkite CI entry added to `.buildkite/test-merge.yml`
- [ ] Gradio demo working
- [ ] Documentation added (offline + online docs, nav, supported models)

### Phase 4: Async Chunk
- [ ] Stage config updated with `async_chunk: true`
- [ ] Stage 1 handles partial chunks correctly
- [ ] No audio artifacts at chunk boundaries
- [ ] Streaming via API (`stream=true`) works
- [ ] TTFA measured and acceptable

### Phase 5: CUDA Graph
- [ ] Hot loop identified and profiled
- [ ] Static buffers allocated
- [ ] Graph captured and replays correctly
- [ ] Benchmark shows meaningful speedup
- [ ] Fallback to eager works for unsupported configs

### Phase 6: Pre-commit and DCO
- [ ] `pre-commit run --files <changed>` passes before every push
- [ ] Every commit has `Signed-off-by` matching the author email (`git commit -s`)
- [ ] `git config user.email` matches the email registered on your GitHub account
- [ ] Details and failure-recovery commands: [references/precommit-dco.md](references/precommit-dco.md)

## References

In-skill references (details split out of the main body):

- [references/single-stage-ar.md](references/single-stage-ar.md) — full `forward()` / generator skeleton for the MOSS-TTS-Nano-style pattern
- [references/optional-deps.md](references/optional-deps.md) — torchaudio / torchcodec fallback pattern
- [references/cuda-graph-example.md](references/cuda-graph-example.md) — Qwen3-TTS code-predictor CUDA graph skeleton
- [references/precommit-dco.md](references/precommit-dco.md) — full pre-commit invocation, failure table, DCO recovery

Project docs and adjacent skills:

- [TTS audio skill](../vllm-omni-audio-tts/SKILL.md) — supported models and usage
- [Fish Speech integration](../vllm-omni-audio-tts/references/fish-speech.md) — complete example of Phases 1–3
- [Qwen3-TTS reference](../vllm-omni-audio-tts/references/qwen-tts.md) — complete example of all 5 phases
- [Adding a TTS model (developer guide)](https://github.com/vllm-project/vllm-omni/blob/main/docs/contributing/model/adding_tts_model.md)
- `plan/voxcpm2_native_ar_design.md` — VoxCPM2's vLLM-native AR + side-computation pattern (distinct from the generator-based single-stage described above)
