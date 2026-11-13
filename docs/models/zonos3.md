# ZONOS3-Mini implementation report

Implementation is in progress. The complete two-stage service is not yet
available. Reference-prefix audio encoding, service integration validation, GPU acceptance
and benchmarks remain outstanding. Pipeline registration, deployment YAML and
the text/precomputed-conditioning adapter are implemented, but have not passed
an end-to-end GPU serving run.

## Implemented paths

Source organization follows the integration's model boundaries:

- `modeling_zonos3_mini.py`: Backbone, embeddings, FFNs, routing and head.
- `gdn.py`: GDN kernels and persistent windows, kept separate for file size.
- `codec.py`: Codec neural layers, decoder definition and release tensor mapping.
- `model_state.py`: native request ModelState implementations.
- `sampler.py`: the native ten-codebook sampling interface.

Codec synthesis, checkpoint I/O, request-slot management and the serving model
adapter remain focused companion modules. `pipeline.py` defines the two stages;
`vllm_omni/deploy/zonos3_mini.yaml` selects MRV2 and their connector. The production
configuration resolver is tested to preserve both stages' custom release parser,
native data-plane flags, disabled prefix caching and AR FULL decode graph mode.
`cuda_graph_wrapper.py` captures the complete Codec neural/synthesis path;
`codec_synthesis.py` owns its batched fixed-buffer arithmetic. Actual CUDA
capture/replay acceptance remains pending an NVIDIA GPU run.

The AR core implements MultiEmbedding, Full AttnRes with a reusable source
workspace, nine GDN layers, two native vLLM global Attention bindings, dense
and latent MoE FFNs, and MultiOutputHead. Backbone forward returns hidden states
and never invokes the head. CPU comparison of the entire eleven-layer stack
covers hidden states and logits, chunked/mixed prefill and decode, batch reorder,
and speaker/condition row substitution, using a test-only CPU KV backend.
Native paged-KV execution has not been validated on CUDA.

`Zonos3ForCausalLM` is registered and loads the flat release checkpoint through
the native loader. Its custom ModelState constructs/caches conditioned embeddings
once per request, slices them by absolute prefill offset, and supplies GDN slots,
next-step embeddings and retained graph-bucket workspaces. It accepts prepared
frame IDs and speaker/condition metadata through EngineCore serialization;
the text frontend now builds exact placeholder lengths for unconditioned text,
precomputed speaker embeddings and precomputed reference codes. It uses the
release language registry, Rust text normalizer and byte/SuperBPE tokenization.
Silence prefill is passed to the producer as delayed priming rows; it counts only
new generated frames toward chunk thresholds and requests one initial PCM crop.
Raw-reference speaker cloning now resolves audio through the existing Speech
media loader and reserves exactly one speaker row. The AR model registers the
ownmect_s2 MECT-B2 encoder, verifies its pinned checkpoint SHA256 and loads it
strictly in FP32. ModelState computes Kaldi fbank and the speaker vector once per
request, then caches the projected prompt across prefill chunks. Its four-stage
MECT/soft-MoE/ASTP arithmetic matches the official source with random weights on
CPU; real speaker-checkpoint and end-to-end cloning acceptance remain unverified.
Waveform-to-reference-prefix codes (the alternative cloning mode) is still
outstanding; the existing reference-prefix path accepts precomputed codes only.

`Zonos3Sampler.sample_step()` calls MultiOutputHead, samples all ten codebooks,
updates GPU EOA/drain/history/feedback state, and returns one logical scheduler
token. Incomplete prefill rows do not consume physical progress. CUDA RNG uses
native request-seeded Gumbel draws with an AR-position/codebook nonce; CPU checks
use a request-local Torch generator. Sampling settings are independent per
request, rather than inheriting the reference sampler's batch-wide min/max
reductions for heterogeneous top-k/top-p/min-p values. Uniform-setting filter
probabilities are compared against the official sampling function.

Custom-head warmup uses the native OmniSampler startup hook. Generic text-only
warmup constraints and grammar are bypassed only for internal warmup IDs;
serving requests retain strict validation. Eager startup warmup is permitted
before capture, while steady-state decode rejects non-FULL execution. CPU
sampler tests select the upstream non-UVA buffer allocator on macOS.

The native Attention factory rejects KV-only prefix caching, eager execution,
graph modes other than `FULL_DECODE_ONLY`, and unsupported tensor/pipeline/expert
parallel configurations. These are startup guards, not evidence of successful
native FULL capture. The flat AR checkpoint loader requires every parameter and
rejects unknown or mismatched tensors while retaining backend runtime buffers.

GDN retains committed conv/recurrent state at multiples of 64 and replays only
the unfinished window. Decode uses fixed 64-row windows with device-selected
positions and masked commits. CUDA calls the official `fla-core==0.5.2` kernels
with explicit `FLA_TILELANG=0`; vLLM's differently-versioned GDN implementation
is not substituted. Native vLLM fused-experts computation is used on CUDA.
These paths are implemented but their numerical parity/capture is not yet
verified on a GPU. Install the `zonos3` optional dependency for that path.

The producer processor is
`vllm_omni/model_executor/stage_input_processors/zonos3.py`. It consumes
CPU-materialized delayed RVQ rows, reconstructs frames, applies chunk thresholds
and the partial-codebook first-chunk policy, and flushes the final complete frames.
Missing residual codes are `-1`; frames emitted early are not decoded again.
Its state lives in the existing transport-owned per-request payload cache.

The tested native publication path is:

```text
NativeOutputWorker._materialize
  -> OmniRunnerDataPlane.publish_output / complete_outputs
  -> producer zonos3_rvq_async_chunk
  -> existing connector save worker / connector.put
```

The normal asynchronous runner output materialization owns the D2H boundary.
The producer rejects device-resident codes and adds no EOA-related D2H copy.
It declares one scheduling placeholder through `meta.next_stage_prompt_ids`;
RVQ values stay in `codes.audio` instead of becoming scheduler token IDs.

`Zonos3CodecModel` is registered as a native generation-stage architecture.
It accepts runner-ordered request IDs and payloads through the existing
`model_intermediate_buffer` contract and returns `OmniOutput` with PCM and
44.1 kHz sample rates. `CodecManager` groups already-scheduled requests by new
chunk length and executes one B>1 neural forward per compatible group.
Requests may have different absolute positions and chunk sequence numbers.
No padding advances neural state. With graphs disabled, synthesis runs per
request after the batched neural execution. With graphs enabled, the complete
path is batched inside one graph, including final flush.

### Complete Codec graph

Stage 1 `additional_config` accepts `zonos3_codec_cuda_graph` (default false),
`zonos3_codec_graph_batch_sizes` and `zonos3_codec_graph_frame_sizes`. The supplied
deploy YAML enables it with B=1/2/4/8/16/32 and RVQ frames=1/2/4/8/15. The native
Generation runner remains eager: the model owns these graphs independently.

Keys are `(batch_bucket, RVQ_frame_bucket)`. Both axes round upward. Per-row
valid frame counts govern KV/conv commits, spectral envelopes and PCM coverage;
padding values are never treated as actual RVQ frames. Scratch slots are
separate from live request slots. A device `final` vector selects the centered
iSTFT/OLA, FIR and resampler final boundaries in the same graph, including an
empty final input padded to the smallest RVQ bucket. First/continuation/final
rows may coexist. Only a finite spectral overlap is retained/reconstructed;
neural prefixes are not decoded again. Valid PCM is copied before a subsequent
replay can overwrite bucket storage.

Capture/warmup touches scratch slots only and runs during model loading. Capture
failure aborts startup. Shapes beyond configured buckets execute the same
complete stateful arithmetic eagerly; `replay_count` and `eager_count` expose
which path ran. Disable the option to use the original eager implementation.

CPU tests compare the complete padded arithmetic against the original stream
at B=1/4/8/16/32, including mixed lengths/final rows, partial codebooks, empty
tails, initial PCM cropping, cancellation and reuse. These are not CUDA replay
results. Run actual capture/replay on NVIDIA hardware with:

```bash
python -m pytest tests/model_executor/models/test_zonos3_codec_graph_cuda.py -q
```

## State and lifecycle

| Owner | State |
| --- | --- |
| Stage 0 sampling state | Device tensors for position, EOA, drain, history, validity, next RVQ codes and embeddings; inactive prefill and post-EOS steps are fenced |
| Stage 0 ModelState | Request identities, conditioning cache, independent GDN windows/committed states and retained bucket workspaces, including dedicated padding/capture slots |
| Producer processor | Delayed rows, received/emitted frame cursors, first EOA bound, chunk sequence and terminal state |
| Stage 1 CodecManager | Independent request slots, semantic/head convolution history, bounded per-layer KV, absolute frame progress, iSTFT overlap, FIR and resampler history |
| Stage 1 model | Mapping between native scheduler IDs and stable external payload identities |

Stage 1 validates chunk sequence and frame offsets before mutating state.
The materialized sampler output retains its EOA/Tail Drain completion bit.
The native data plane publishes the step that first reaches EOS or the token
limit, and fences only subsequent in-flight outputs. A completed Tail Drain
does not close the transport: the producer waits for the scheduler's terminal
notification, after outstanding outputs drain, then emits exactly one final
Codec chunk with request identity, sequence, offset and possibly empty codes.
Length truncation flushes complete frames through the same path without claiming
a completed Tail Drain. No anonymous finish sentinel is used for these requests.
Final chunks flush synthesis and reclaim neural slots; completion cleanup clears
identity mappings and terminal tombstones. Cancellation discards pending history
without flushing audio. Preemption/state recomputation is explicitly rejected.
The producer's `meta.aborted` control marker distinguishes cancellation from a
normal final chunk, so a cancelled request does not flush stale Codec tails or
prevent another request in the scheduled cohort from decoding.
Output PCM owns storage independent of slots subsequently reused by other requests.

## Weights and reference evidence

The decoder builds the retained causal stages and multi-resolution iSTFT head
from the selected release configuration (`keep2_plain` or `full_mr`). Loading
checks the registry's generator SHA-256, generator step 600000, matching model
configuration, head step 50000 and complete mapped state keys. Stage pruning
uses the original layer indices before applying the released trainable override.
The codec keeps FP32 parameters; CUDA transformer GEMMs use BF16 autocast and
the Stage 1 worker disables TF32 at weight loading.

CPU arithmetic checks use unmodified definitions from official source revision
`4c66ceac5a5004dafd852f42674a6c38f657c7cd`. They cover ConvNeXt,
incremental MOSS attention at batch sizes 1/4/8/16/32, iSTFT, FIR and resampling.
Synthesis and resampling match the official device-stream implementation exactly
in these CPU checks. Neural checks use small random weights and tolerance-based
comparison. These are not official release-weight, CUDA or audio-quality results.

## Local validation

### Server validation commands

The following commands are provided for GPU validation; they have not been
executed against the complete service. Install the `zonos3` optional dependencies
in the server environment first. Keep the full release bundle, including the
speaker encoder and autoencoder directories, when using a local model path.

```bash
FLA_TILELANG=0 vllm serve Zyphra/ZONOS3-Mini --omni \
  --deploy-config vllm_omni/deploy/zonos3_mini.yaml --port 8000
```

After startup, request unconditioned speech through the standard Speech API:

```bash
curl --fail-with-body http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"Zyphra/ZONOS3-Mini","input":"Hello from Zonos3.","response_format":"wav"}' \
  --output zonos3.wav
```

Use the existing universal TTS driver for the concurrency matrix. `default_voice`
means unconditioned synthesis here; Zonos3 has no built-in preset voice. For
`voice_clone`, provide the Seed-TTS evaluation dataset through `--dataset-path`.
The sample-rate override is required for correct streamed PCM duration and RTF.

```bash
VLLM_OMNI_BENCH_AUDIO_SAMPLE_RATE=44100 python benchmarks/tts/bench_tts.py \
  --model Zyphra/ZONOS3-Mini --task default_voice \
  --dataset-path benchmarks/build_dataset/seed_tts_smoke \
  --concurrency 1 4 8 16 32 --num-prompts 64 --num-warmups 2 \
  --output-dir results/zonos3
```

These client results do not establish the single FULL Backbone replay, decoder
batch shapes or absence of EOA synchronization. Capture server traces separately
and correlate them with the matching model/config/revision and workload.

### CPU checks

The local environment is Python 3.12, PyTorch 2.14.0 and editable
vLLM 0.33.0+cpu. Run the model and producer tests from the repository root:

```bash
.venv/bin/python -m pytest tests/model_executor/models/test_zonos3_*.py \
  tests/model_executor/stage_input_processors/test_zonos3.py \
  tests/worker/test_scheduling_metadata_adapter.py
```

The lifecycle regression run passed 512 tests across the Zonos3, metadata
adapter, native AR/generation runner, native data plane, scheduling coordinator
and payload schema suites, with one Linux shared-memory test excluded on macOS.
Subsequent focused Codec tests also cover truncation before the first complete
frame. The native finalizer/output-worker/producer/connector-to-Codec test uses
synthetic release weights and verifies PCM coverage/parity, one final chunk,
cancellation before/after output, late-output fencing, slot and identity cleanup,
and identity reuse. It does not replace a real server run.

Set `ZONOS3_REFERENCE_ROOT` to an official source checkout to enable the optional
official-source CPU checks. Native generation/output-worker tests are under
`tests/worker_v2/`. On macOS, the existing
`test_release_reclaims_segments_no_downstream_stage_consumed` requires Linux
`/dev/shm` and is excluded from the recorded native data-plane test run.

No official model-weight parity run, end-to-end service run, CUDA Graph trace,
GPU concurrency benchmark or quality comparison has been completed. Server
validation remains required; no performance numbers are claimed.
