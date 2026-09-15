# Cosmos3 multiview/LiDAR validation report and reproduction

Status: implementation present and selected CPU checks pass; full runtime
verification, GPU execution, and visual acceptance pending. No RGB comparisons,
latency measurements on GPUs, or GPU memory results are claimed.
The development host is macOS without CUDA, the
repository virtualenv cannot start, and the full vLLM serving stack is unavailable.
The iteration-4200 weights and the example records' `/lustre` media must be
available on the validation host.

## Implemented coverage

The two-stage exporter retains `lidar2llm`/`llm2lidar` as
`lidar_proj_in`/`lidar_proj_out`, including biases. The exporter at imaginaire4
`101e4f89b92beca1e99e0bca9df8a42083557d58` packages the complete reference V1.2
VAE in `lidar_vae/config.json` and `lidar_vae/diffusion_pytorch_model.safetensors`.
The component includes coordinates, resolved architecture defaults, streaming
settings, physical projection, encoder/decoder weights, and latent mean/std.
Runtime loading reads encoder and decoder weights independently from the same
artifact in FP32. It checks shared metadata and every explicit transformer network setting,
accepts the VAE's additional defaults, and constructs from the saved VAE config.
Complete projection weights and strict encoder/decoder state dictionaries are
required. Older unversioned WSM
artifacts remain loadable.

The runtime uses independent camera/LiDAR shapes, per-camera causal caption
encoding, sensor-specific sparse attention, and a common UniPC schedule. LiDAR
positions rewind to the camera origin and advance by 0.75 at the production
rates. The packed cursor includes the furthest temporal or spatial endpoint
for reference parity, but is inert in the current single-sample request path:
no subsequent sample or modality consumes it.
The numerical encoder dependency is vendored from imaginaire4 `e55e4fad16a9`;
the decoder is ported from `9ca7bd6adfe` using identical shared blocks.
`lidar.return_output=true` enables numeric output through offline inference and
asynchronous video jobs. Default requests retain RGB-only output.

Unified VAE loader tests cover selective tensor reads, exact FP32 preservation
under a BF16 default, local/Hub component resolution, configuration agreement,
missing files and encoder tensors, invalid precision/statistics, and construction
of a small real encoder without attention execution. The saved
`apply_validity_mask` setting does not change encoder input masking.

### Unified VAE loader verification (2026-09-15)

The encoder-only results below predate decoder integration.

- 275 checks passed through the existing CPU adapters across
  `test_cosmos3_lidar.py` and `test_cosmos3_multiview_pipeline.py`.
- Three existing integration checks (generic warmup, lazy runtime imports, and
  pipeline registration) failed on unavailable runtime imports and were
  deselected in the focused rerun. Ordinary pytest collection also fails because
  this host lacks vLLM; the repository virtualenv cannot start.
- A synthetic small instance of the actual reference `TransformerVAE` was
  exported with imaginaire4's `export_lidar_vae`, then loaded by the updated
  vLLM-Omni encoder under a BF16 default dtype. All 58 consumed tensors matched
  the reference exactly in FP32. The check substituted synthetic tokenizer
  weights for checkpoint loading and did not execute attention kernels.

Production-checkpoint latent parity, CUDA inference, and offline/HTTP smoke
validation remain pending. These CPU results do not validate GPU kernels or
serving behavior.

## Existing runtime validation

Selected CPU checks across the runtime/client and reference suites cover
metadata/path validation and real FFmpeg chunk trimming. CPU checks exercise:

- Numeric files, sweep selection, malformed data, circular padding, normalization,
  FP32 weight loading and streaming, and missing artifact/statistic failures.
- Packed transformer boundaries, independent causal captions, cached text,
  target-only timestep embeddings, camera conditions, and control-CFG removal.
- Mixed sensor UniPC updates against independently shaped states using the same
  schedule; full synthetic pipeline requests return RGB only.
- All request modes, camera subsets and ordering, client upload mapping, and
  existing 480p/720p geometry and sparse-attention cases.
- Exported rigs must be ordered subsets of one active training rig. Request
  subsets/reordering require schema version 2 and `variable_view_count=true`;
  fixed and legacy contracts retain their exact camera order.
- Artifact resolution/FPS/emphasis defaults and explicit request precedence;
  clients defer omitted resolution to the checkpoint. Per-camera JSON objects
  fail validation before prompt formatting.
- The actual reference directory loader → inverse width transform → safetensors
  → server padding/normalization, including nearest-return pooling, validity,
  and occupied HD-map unit intensity. This roundtrip matches exactly.
- Reference export metadata and preparation utility behavior.
- Header-only LiDAR admission, value validation after truncation, projection-derived
  circular padding, scalar CFG delegation, and checkpoint-specific encoder discovery.
- Shared base/multiview GEN execution, including grouped cache wrappers and
  sequence-parallel call placement. Caption lengths are computed during prompt
  preparation; cached packed forwards perform no per-caption `.item()` reductions.

Local verification uses an isolated Python 3.12 environment. GPU runtime imports
were replaced with small CPU adapters; tested transformer/pipeline definitions,
sparse attention, scheduler, reference loaders, and tensor operations run from
the source files. This is narrower than initializing a deployed vLLM engine.
The HTTP synchronous/asynchronous upload, invalid-payload cleanup, and cancellation
cases have been extended; the complete endpoint suite must run in the serving
environment (local collection lacks PyAV/vLLM).

The previously reported 505 checks were an aggregate of selected CPU-adapter
runs, not successful collection or execution of all changed test modules.
`test_cosmos3_transformer.py` was outside that aggregate; review found and
removed a stray path before its SPDX header that prevented pytest collection.
The latest cleanup verification and earlier selected-suite results are:

- Cleanup verification: 53 tests in `test_cosmos3_lidar.py`, 185 in
  `test_cosmos3_multiview_pipeline.py`, 23 in `test_cosmos3_output_projection.py`,
  24 in `test_cosmos3_gen_mlp.py`, and 61 in `test_multiview_flex_attention.py`
  passed through the CPU adapters.
  Three runtime import/registration/warmup tests were deselected.
- Cleanup verification: 4 reference preparation tests passed, including rerunning
  FFmpeg over an existing clip with a changed frame window and FPS.
- Earlier verification: 167 upload/manifest/client tests and 44 offline-client
  tests passed, along with 34 reference export tests covering compatible subsets
  and incompatible rigs.
- During the header repair, all 47 tests in `test_cosmos3_transformer.py`
  collected through the CPU adapter. Their execution against the full runtime
  remains pending.
- Ruff lint and formatting checks passed for the changed Python files in both
  repositories.

Joint noise regression checks compare initial camera/LiDAR tensors with
successive draws from one generator, including a caller-provided generator,
an already-advanced generator, and injected camera latents. All eight joint
cases failed before the RNG fix and passed afterward. Camera conditioning and
the final caller-generator state are also checked.

LiDAR now continues the live request generator after the camera draw. This
avoids restarting its random stream and respects caller-provided RNG state.
It is not exact reference-inference RNG parity: the local reference
`OmniMoTModel._prepare_inference_data` calls `misc.arch_invariant_rand` separately
for each sensor with the sample seed, and that helper constructs a fresh NumPy
`RandomState` per call. The reference training path shares a noise generator.

## Export the iteration-4200 checkpoint

Run in the installed imaginaire4 Cosmos3 environment, from the imaginaire4 root.
Set these paths to the supplied checkpoint and its resolved training config;
do not substitute a teacher-forcing experiment.

```bash
export COSMOS3_JOINT_DCP=/models/source/iter_000004200
export COSMOS3_JOINT_CONFIG=/models/source/config.yaml
export COSMOS3_JOINT_HF=/models/cosmos3-joint-4200-hf
export COSMOS3_JOINT_DIFFUSERS=/models/cosmos3-joint-4200-diffusers

python -m cosmos3.scripts.export_model \
  --checkpoint-path "$COSMOS3_JOINT_DCP" --config-file "$COSMOS3_JOINT_CONFIG" \
  --backbone-type cosmos3_multiview -o "$COSMOS3_JOINT_HF"

python -m cosmos3.scripts.convert_model_to_diffusers \
  --checkpoint-path "$COSMOS3_JOINT_HF" -o "$COSMOS3_JOINT_DIFFUSERS"
```

Export requires access to the reference V1.2 tokenizer checkpoint and its latent
statistics. A config-only conversion is not a deployable joint artifact.
The encoder folder is loaded separately from the transformer's weight index.
Validate `schema_version=2`, `separate_view_text_tokenization=true`, variable-view
metadata, the production 0.4-second attention window, and the two encoder files.

## Prepare and run the supplied five records

Preparation reads caption files and HD-map archives using reference helpers.
The `/lustre` dataset paths must be mounted; the runtime receives ordinary local
camera files and numeric safetensors. Both clients resolve relative local paths
against their request file's directory.

```bash
# In imaginaire4; chunk zero, camera conditions retained, seeds 42 through 46.
python -m cosmos3.scripts.prepare_multiview_lidar \
  packages/cosmos3/inputs/omni_multiview/wsm_lidar_transfer_i2v.jsonl \
  --output /data/prepared/joint-i2v --limit 5 --chunk 0 --seed 42

# T2V from the same records, dropping RGB conditions.
python -m cosmos3.scripts.prepare_multiview_lidar \
  packages/cosmos3/inputs/omni_multiview/wsm_lidar_transfer_i2v.jsonl \
  --output /data/prepared/joint-t2v --limit 5 --chunk 0 --seed 42 --t2v

# Explicitly ordered smaller rig; use the identical camera list in reference runs.
python -m cosmos3.scripts.prepare_multiview_lidar \
  packages/cosmos3/inputs/omni_multiview/wsm_lidar_transfer_i2v.jsonl \
  --output /data/prepared/joint-720-3view --limit 5 --seed 42 --resolution 720 \
  --cameras camera_rear_tele_30fov camera_front_wide_120fov camera_cross_left_120fov

# In vllm-omni, initially on one GPU in eager mode.
python examples/offline_inference/multiview_video/cosmos3_multiview.py \
  --model "$COSMOS3_JOINT_DIFFUSERS" --input /data/prepared/joint-i2v/requests.jsonl \
  --output-dir /data/results/omni-i2v-eager --enforce-eager

# HTTP uses the same prepared request, including its per-camera captions.
vllm serve "$COSMOS3_JOINT_DIFFUSERS" --omni \
  --model-class-name Cosmos3MultiviewPipeline --enforce-eager --port 8091

python examples/online_serving/multiview_video/cosmos3_multiview_client.py \
  /data/prepared/joint-i2v/request_0000.json --server http://localhost:8091 \
  --output /data/results/http-async.mp4
python examples/online_serving/multiview_video/cosmos3_multiview_client.py \
  /data/prepared/joint-i2v/request_0000.json --server http://localhost:8091 \
  --sync --output /data/results/http-sync.mp4
```

The preparation utility writes one `request_NNNN.json` per sample plus
`requests.jsonl`. Frame counts come from the selected caption span on the camera
VAE grid; FPS is explicit. Nonzero chunks trim camera videos to the same frame
start and select matching LiDAR sweeps. Rerunning preparation in the same output
directory overwrites generated clips, numeric files, and request manifests.
It calls `load_lidar_transfer_control`
with `V1P2_TRANSFER_RANGE_PROJECTION`, then `undo_model_width_transform` before
saving the numeric file. It does not reproduce archive preprocessing itself.
Do not pass reference `load_caption_from_data` or archive paths to the server.
Continuation remains client-side, using successive requests and generated RGB
conditions.

For reference comparison, copy the original five-record JSONL and set each
record's `seed` to `42 + record_index`, `multiview.load_caption_from_data=true`,
and `multiview.use_first_chunk_only=true`. Apply the same camera subset/order,
resolution, and RGB-condition selection as the prepared requests. Keep the
original reference LiDAR archive path. Then run:

```bash
# In imaginaire4, against the supplied checkpoint's own configuration.
python -m cosmos3.scripts.inference \
  --checkpoint-path "$COSMOS3_JOINT_DCP" --config-file "$COSMOS3_JOINT_CONFIG" \
  -i /data/reference-five-records.jsonl -o /data/results/reference-i2v \
  --load-caption-from-data
```

Both sides must use 35 steps, guidance 6, shift 10, control guidance 1, the same
caption chunk/frame count/FPS, and the same emphasis setting. Seeds are fixed
within each runtime; acceptance does not require identical RNG implementations
or an end-to-end numerical similarity threshold. For legacy WSM comparison use
its original artifact, top-level captions, and the caller-supplied negative
prompt from the reference request.

## GPU and visual acceptance matrix

Run eager first; only after reviewing it, repeat supported configurations with
regional compilation (omit `--enforce-eager`), CFGP2, strict Ulysses CP2, TP2,
and HSDP. HSDP and TP are alternatives. Triton is the default; test FA4 on its
supported Blackwell build with `VLLM_OMNI_COSMOS3_MULTIVIEW_BACKEND=fa4`.
Do not treat CPU results as validation of these kernels or collectives.

| Checkpoint / input | Cases | RGB comparison | Latency / peak memory |
|---|---|---|---|
| Joint 4200, 480p, 11 views | I2V and T2V; all five records | Pending | Not measured |
| Joint 4200, 480p, subsets | 1, 3, 4, 5, 7, 9 views; reordered rig | Pending | Not measured |
| Joint 4200, 720p | 1, 3, 6 views; I2V and T2V | Pending | Not measured |
| Joint checkpoint, camera only | Ordinary, transfer, view completion | Pending | Not measured |
| Existing unversioned WSM | Original and subset rigs | Pending | Not measured |
| Serving | Sync/async RGB, cancel, timeout, malformed numeric upload | Pending | Not measured |
| Distributed / compiled | Supported topology and backend combinations | Pending | Not measured |

Review conditioning fidelity, WSM adherence, temporal stability, camera order,
and cross-camera consistency. Store paired RGB outputs and written observations
per record. No LiDAR outputs are needed for this review.

The offline client records `generation_seconds` before video export in each
`sample_outputs.json`. Keep cold and warm measurements separate. Record GPU
model/count, CUDA/PyTorch/attention versions, topology, request dimensions,
steps, and compiled/eager mode. Capture device-memory samples during both runs,
for example in a separate terminal:

```bash
nvidia-smi --query-gpu=timestamp,index,memory.used \
  --format=csv --loop-ms=200 > /data/results/gpu-memory.csv
```

Report the maximum sampled memory per GPU and identify this as device-used
memory, including runtime overhead and any other GPU processes. Preserve the
server's reported stage durations and peak-memory fields when available.
There are no numerical-similarity or performance pass thresholds.

## Checks in the supported development environments

```bash
# vllm-omni
pytest tests/model_extras/test_cosmos3_multiview_uploads.py \
  tests/examples/offline_inference/test_cosmos3_multiview.py \
  tests/diffusion/models/cosmos3/test_cosmos3_transformer.py \
  tests/diffusion/models/cosmos3/test_cosmos3_lidar.py \
  tests/diffusion/models/cosmos3/test_cosmos3_multiview_pipeline.py \
  tests/diffusion/models/cosmos3/test_cosmos3_output_projection.py \
  tests/diffusion/models/cosmos3/test_cosmos3_gen_mlp.py \
  tests/diffusion/models/cosmos3/test_multiview_flex_attention.py
pytest tests/entrypoints/openai_api/test_video_server.py -k 'multiview or joint_invalid'

# imaginaire4
pytest packages/cosmos3/cosmos3/scripts/multiview_export_test.py \
  packages/cosmos3/cosmos3/scripts/prepare_multiview_lidar_test.py
```

The cross-repository normalization check expects sibling `imaginaire4` and
`vllm-omni` directories; it skips explicitly when the runtime checkout is absent.

## Decoder integration validation (2026-09-15)

The decoder integration adds strict FP32 decoder loading, reference streaming
cache behavior, physical output conversion, opt-in pipeline output, and
safetensors persistence/downloads for asynchronous video jobs. It reuses the
existing unified artifact without changing the exporter.

Local CPU validation uses the existing isolated environment and source adapters
for unavailable vLLM imports. The checked tensor operations, decoder, pipeline,
postprocessor, storage, and job lifecycle functions execute repository source.
These results do **not** establish CUDA attention parity or full engine/HTTP serving:

- Across the selected CPU-adapter runs, 579 decoder, encoder, pipeline,
  offline, and upload/client checks passed, along with 11 serving-handler and
  storage checks. Three runtime import/registration/warmup checks were excluded.
  Ruff lint and formatting passed for all 22 changed Python files, and
  `git diff --check` passed.
- Decoder checks cover exact FP32 loading under a BF16 default, missing/invalid
  weights, asymmetric topology resolution, streaming support validation,
  latent affine conversion, chunk/context boundaries, single-sweep output,
  request cache isolation, mask thresholding, width cropping, and safetensors.
- A small real causal decoder without local attention executes on CPU and
  matches its full-sequence decode at `rtol=1e-4, atol=1e-4`.
- Pipeline checks exercise the opt-in output and preserve generated camera
  latents and request RNG state. Offline/client checks preserve the flag and
  output metadata through file writing and upload resolution.
- Storage checks exercise both downloads, partial-save failures, cancellation
  during a write (including repeated cancellation), deletion, expiration, and
  synchronous rejection with uploaded-file cleanup.

Cancellation follow-up: all 29 serving/storage checks passed through the CPU
source adapter. The module now covers fixed cancellation
deadlines for RGB-only and joint outputs, DELETE returning HTTP 409 while a save
is stalled, cancellation of the DELETE request itself, bounded cleanup waits,
and late-write cleanup after success or failure. A blocked filesystem-thread
case verifies cleanup after the cancelled job has already returned. Deferred
operations remain owned until completion, including cleanup through their
original storage manager. These checks use the same CPU source adapter; full
server shutdown and filesystem failure recovery remain unverified.

Normal pytest collection on this host fails because vLLM is unavailable. CUDA,
production-checkpoint decoder parity, full HTTP smoke tests, and distributed/
offload execution remain pending. No GPU performance results are claimed.

### FlexAttention replacement checks (2026-09-15)

Local validation ran on macOS arm64 with PyTorch 2.14.0 and no NATTEN installed,
using the CPU source adapters described above:

- 783 selected operator, encoder, decoder, multiview, numeric output,
  offline, and upload checks passed. All 29 serving/storage checks passed.
- Tests now execute local attention with nonzero projections, symmetric and
  asymmetric decoder layouts, alternating dilation, and both factorized and
  joint 3D bottlenecks. Streaming checks cover partial chunks, bounded/unbounded
  context, batch sizes 1/2, repeated requests, and RNG preservation.
- Operator results match independently enumerated FP64 neighborhoods at
  `rtol=1e-4, atol=1e-4`. CPU allocation inspection covers a 58,112-token spatial
  mask without a token-level square allocation. Cache eviction/reuse and real
  Dynamo graph isolation are checked separately.
- Twelve CUDA cases skipped. Three multiview runtime import/registration/warmup
  checks were excluded after failing on unavailable runtime dependencies.
  Full native pytest/HTTP execution remains unavailable on this host.
- Ruff lint/formatting, Python compilation, and `git diff --check` passed.

CUDA numerical parity, real-checkpoint output quality, GPU allocator peaks,
offload/reload execution, cold-start time, and warmed encoder/decoder latency
remain **unqualified**. The following utility and commands prepare those checks;
CPU results do not satisfy the GPU acceptance criteria.

### Run on the CUDA validation host

Install the supported vLLM-Omni runtime with `.[cosmos3-lidar]`. Run:

```bash
pytest tests/diffusion/models/cosmos3/test_cosmos3_lidar_decoder.py \
  tests/diffusion/models/cosmos3/test_lidar_neighborhood_attention.py \
  tests/diffusion/models/cosmos3/test_cosmos3_lidar.py \
  tests/diffusion/models/cosmos3/test_cosmos3_multiview_pipeline.py \
  tests/diffusion/test_diffusion_output_formatter.py \
  tests/model_extras/test_cosmos3_multiview_uploads.py \
  tests/examples/offline_inference/test_cosmos3_multiview.py \
  tests/entrypoints/openai_api/test_cosmos3_lidar_output.py \
  tests/entrypoints/openai_api/test_video_server.py
```

Run the runtime regression suites without NATTEN installed. The encoder and
decoder now share PyTorch FlexAttention, including the decoder's local blocks
at each upsampling level. The CPU operator tests use an independent FP64
neighborhood oracle. CUDA cases exercise the compiled operator and multiple
devices when available.

For reference qualification, use the imaginaire4 checkout and its tokenizer
dependencies, including its original NATTEN build, in a separate validation
environment. Compare identical inputs using a real exported checkpoint:

```bash
python tools/validate_cosmos3_lidar_decoder.py \
  --model /models/cosmos3-multiview \
  --reference-root /workspace/imaginaire4 \
  --frames 19 --seed 42 --report outputs/lidar_decoder_synthetic_parity.json
```

The default synthetic case crosses the production 9-sweep chunk boundary and
ends with a partial chunk. For chunk length 9, run positive lengths
`--frames 1`, `8`, `9`, `10`, and `19`, and repeat with `--batch-size 2`.
Use artifacts with bounded and unbounded context and symmetric/asymmetric
decoder topology. Synthetic tensors establish execution and streaming behavior;
they do not qualify production output quality.
For production parity, save the generated normalized LiDAR target from the
multiview pipeline's `final_targets[1]` to safetensors under `latents`, then run:

```bash
python tools/validate_cosmos3_lidar_decoder.py \
  --model /models/cosmos3-multiview \
  --reference-root /workspace/imaginaire4 \
  --latents outputs/generated_lidar_latents.safetensors \
  --encoder-input outputs/prepared_lidar_control.safetensors \
  --report outputs/lidar_decoder_generated_parity.json
```

The encoder input must contain metric FP32 `frames` shaped `[3,T,128,1800]`.
The utility checks raw range/intensity and validity probabilities at
`rtol=1e-4, atol=1e-4`, then metric output after reference width cropping. Binary
validity must match exactly. It also checks the encoder's normalized latents
against reference encoding at the same tolerances. Decoder comparison precedes
optional smoothing in the reference artifact writer.

Each implementation's first call is measured separately from three warm-ups
and ten timed runs. Timed runs have no capture hooks. Only one implementation's
encoder or decoder resides on CUDA during each measurement; the report records
resident inputs/cache memory and peak allocation separately. Persistent compiler
disk caches are not cleared, and the report identifies this limit on cold-start
measurements. Repeated requests must reuse cached attention geometry/code,
preserve RNG state, and produce identical output. On production spatial grids,
the warmed memory increment must remain below one dense FP32 spatial score
matrix. Cache sizes, actual local shapes, numeric errors, configuration, and
per-run latency are recorded even when a later parity check fails.

`passed: true` records numerical and memory checks for the supplied inputs;
production qualification additionally requires `production_inputs: true` and
review of results on the real checkpoint. Performance regressions are reported
and can be tuned separately.

Finally, add `return_output: true` to the prepared iteration-4200 examples' LiDAR
entries and run the offline and asynchronous HTTP clients documented in
[the multiview guide](cosmos3_multiview.md#generated-numeric-lidar). Verify valid
numeric files at 10 Hz alongside the camera outputs, exact RGB agreement with
the same requests without LiDAR output, and cleanup after job deletion. Run
eager single-GPU execution first, followed by supported distributed and offload
configurations. Record actual latency/memory and parity results here.
