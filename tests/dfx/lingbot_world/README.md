# LingBot World regression gates

The suite drives the real `WS /v1/realtime/video` endpoint. CPU tests validate
gate logic separately and are not model performance evidence.

## vLLM 0.33 reference and qualification status

The inference source is frozen at
`3094f0ea747cb2563d99d049da25725cc45f0d2e` in
`https://github.com/tzhouam/vllm-omni.git`. Every profile records both the
repository and full revision, so provisioning can fetch the immutable source
without relying on an upstream branch containing the reference commit.
This reference starts from accepted main
`02286829737a54167e890185bcbb6948ed9afbcf` and carries exactly the three established
LingBot compiler and offload files from
`48cebe5a248d4b0160f20c7fc18b5d1c46883872`, plus the same small
`vllm_omni/diffusion/output_formatter.py` compatibility fix used by the
candidate. Configured streaming must retain the model-provided generation index
even when a request finishes in one block. Real local diagnostic requests proved
that the preceding `61d19b6094189136ad047bf8b060d4180c3c60bf` emitted the correct
nine frames but omitted that index; the unchanged strict benchmark therefore
rejected single-block requests and single-block prewarming diagnostics. The fix preserves
ordinary nonstreaming output and changes no generation or numerical behavior.
The reference and candidate formatter file bytes are identical.

The final reference also applies the identical `core/sched/output.py` and
`core/sched/omni_generation_scheduler.py` compatibility fixes used by the
candidate. Ascend dynamically adds a default `replay_start` field to the base
cached-request dataclass. Both preceding references fail when Omni declares
required positional fields after that inherited default. Required keyword-only
Omni fields fix declaration order, and conversion preserves all actual base
field values and object references. This changes no model generation or
numerical settings. No replay corruption was observed; forwarding populated
upstream fields is a compatibility contract covered by regression tests.
The reference contains no candidate harness or new performance optimization.
Updating a candidate must not advance this pin.

No training cohorts had been collected when this reference was replaced. The
preceding frozen protocol is preserved byte for byte in
[calibration/history/v033_reference_protocol_before_single_block_streaming.json](calibration/history/v033_reference_protocol_before_single_block_streaming.json).
The active protocol records the source fix and former protocol hash; all fitting
rules, workload settings and candidate allowances remain unchanged. Minimal
local functionality diagnostics are excluded from calibration and hardware
qualification.

The intermediate streaming-compatible protocol is likewise preserved byte for
byte in
[calibration/history/v033_reference_protocol_before_ascend_cached_requests.json](calibration/history/v033_reference_protocol_before_ascend_cached_requests.json)
(SHA256 `37c63fc54e1d47358e703481e0d246aaa612b54cfd3d8e7600a3939a134f091c`).
The final protocol records both compatibility updates; no training cohort was
collected before either update, and their statistical rules are identical.

The runtime migration is necessary: the previous reference `48cebe5a` fails
actual serving startup on vLLM 0.33 because
`vllm.entrypoints.launchers.cli_args` was removed. The new reference and candidate
both pass explicit LingBot pipeline imports and `serve --omni --help` on vLLM
0.33.0 / PyTorch 2.14.0. These CPU preflights do not qualify model inference,
performance, numerical precision or video quality on either platform.

The retained vLLM 0.32 calibration records were frozen from reference roles of
CUDA builds [17633](https://buildkite.com/vllm/vllm-omni/builds/17633) and
[17640](https://buildkite.com/vllm/vllm-omni/builds/17640). They and
[calibration/history](calibration/history/README.md) remain byte-identical.
Their old source/runtime contracts deliberately reject the new profiles.
CUDA nightly, weekly, 600-chunk and NPU qualification on the new reference remain
pending until fresh reference-only calibration and independent acceptance run.
No earlier green build qualifies this new source/runtime combination.

The predeclared replacement method is retained in
[calibration/v033_reference_protocol.json](calibration/v033_reference_protocol.json).
Collect two independent three-start reference cohorts for each exact platform
and workload. Reference collection never starts a candidate server, never reads
or rewrites calibration thresholds, and writes `passed: false` alongside
`reference_only: true`, `measurement_only: true` and `complete: true` when
collection succeeds. Separate reference-quality collection retains native raw
scores and within-cohort pixel comparisons with the same completion semantics.
Calibration must combine all six actual reference videos per case and recompute
all 15 unordered pairs in the pinned evaluator environment. Only after freezing
the replacement records may a fresh paired campaign establish acceptance.
Candidate observations never select calibration values, and historical
candidate videos cannot become reference videos.

For a Buildkite construction campaign, set `LINGBOT_REFERENCE_ONLY=1` at build
creation. Existing `generate` and `evaluate` commands select `reference` and
`reference-quality`; split NPU `collect` and `aggregate` commands select
`reference-collect` and `reference-aggregate`, preserving the three existing
matrix repeats and their complete artifact checks. These tests collect evidence
and cannot establish candidate acceptance. The launcher records its requested stage, effective stage and phase
in `ci-configuration.json` and prints the phase in the job log. Use a build
message identifying reference-only calibration. Unset the variable, or set it to
`0`, for the later independent acceptance campaign. The ready pipeline's smoke
job retains its ordinary acceptance phase even when a construction build also
selects reference collectors; smoke artifacts are never calibration inputs.
Invalid switch values and unknown stages fail before provisioning.

Before collecting new values, use this reference-only protocol:

1. Collect two complete fresh three-start reference cohorts per platform and
   profile, covering every weekly case and the exact nightly/600-chunk lengths.
   Record source, deployment, inputs, hardware, packages and artifact hashes.
2. Freeze calibration from those six reference starts. Native continuous
   reference-spread ceilings are per case and dimension: the larger of the
   existing conservative ceiling and the six-value range plus 25%, rounded up
   to 0.005. The earlier proposal sharing ceilings across weekly cases failed
   independent controls and is not adopted. Keep candidate score tolerances
   and binary-activity limits unchanged.
3. Use all 15 unordered reference-video pairs for pixel calibration. Retain
   full-frame/window coverage, the canonical pinned decoder, and existing
   candidate PSNR/SSIM allowances.
4. For performance, retain the 5% candidate allowance and existing statistics.
   Derive startup-stall/TTFC noise floors from the maximum excess between the
   two reference cohorts, add 25%, and round up to 1/5 ms respectively. Derive
   each platform/profile's P95 reference-CV ceiling from its largest training
   case/cohort CV plus 25%, rounded up to 0.005 and at least 3%. Mean/P50 CV and
   steady-stall limits remain unchanged. These are proposed reference-noise
   limits until independent validation passes.
5. Freeze records before a third independent reference cohort and final-source
   candidate acceptance. Require every held-out case to pass, retain all
   negative controls, and investigate failures without refitting to them.
   Promote qualified records and archive their predecessors byte for byte.

Reference collection and scoring use separate output roots for each cohort:

```bash
PROFILE=tests/dfx/lingbot_world/configs/cuda_nightly.json
for COHORT in 1 2; do
  bash tests/dfx/lingbot_world/run_ci.sh reference "$PROFILE" \
    "lingbot-artifacts/cohort-$COHORT/generation"
  bash tests/dfx/lingbot_world/run_ci.sh reference-quality "$PROFILE" \
    "lingbot-artifacts/cohort-$COHORT/quality" \
    "lingbot-artifacts/cohort-$COHORT/generation"
done
python -m tests.dfx.lingbot_world.calibrate \
  --generation lingbot-artifacts/cohort-1/generation \
  --generation lingbot-artifacts/cohort-2/generation \
  --quality lingbot-artifacts/cohort-1/quality \
  --quality lingbot-artifacts/cohort-2/quality \
  --protocol tests/dfx/lingbot_world/calibration/v033_reference_protocol.json \
  --python "$LINGBOT_EVALUATOR_PYTHON" --output lingbot-artifacts/frozen-v033
```

Generation requires the profile's full device reservation; reference quality
and calibration pair evaluation use the pinned one-GPU evaluator environment.
Export `LINGBOT_EVALUATOR_PYTHON` in the caller before invoking the builder.
The builder refuses to overwrite its frozen calibration and population files.
Promote those immutable records only after their independent reference/candidate
qualification passes; a completed collection or builder command is not acceptance.

Weekly CI scores each generation shard in three separate evaluator jobs. After
downloading one cohort, combine its three completed per-case quality roots before
passing that cohort to the builder:

```bash
python -m tests.dfx.lingbot_world.combine_reference_quality \
  --generation lingbot-artifacts/cuda-weekly/shard-0 \
  --quality lingbot-artifacts/cuda-weekly-quality/shard-0/case-0 \
  --quality lingbot-artifacts/cuda-weekly-quality/shard-0/case-1 \
  --quality lingbot-artifacts/cuda-weekly-quality/shard-0/case-2 \
  --output lingbot-artifacts/cohort-1/quality-shard-0
```

The combiner preserves and hashes all original reports and native outputs,
requires identical generation, protocol and evaluator provenance, and rejects
missing cases, reused evaluator producers or candidate observations. It keeps
`passed: false`; it creates no calibration or acceptance result. Repeat for the
second cohort and all six shards, using the NPU artifact prefixes for NPU.
Run the builder separately for each pair of matching shard cohorts. Each shard
produces its own immutable `*_shardNof6.json` calibration and quality population;
an unsharded profile or another shard's records cannot qualify that shard.
All six shards then require independent reference/candidate acceptance.

## Prerequisites and commands

Use the inference environment matching `docker/Dockerfile.ci` (vLLM 0.33.0 / PyTorch 2.14.0)
and development dependencies. Reserve devices with the host scheduler before
inference. Full CUDA profiles use four GPUs while reserving all eight GPU slots
of a node in the H100 pool; NPU reserves eight Ascend A2 devices and uses four. The smoke
uses the real 14B checkpoint at 192×320 on one GPU. Full performance/quality
profiles remain 480×832. CI provides exclusive devices.

Reference preparation requires final vLLM 0.33.0; CUDA also requires final
PyTorch 2.14.0. It imports each source's LingBot pipeline and executes its actual
`serve --omni --help` entrypoint in a separate process before model downloads.
The CPU gate also compiles the real attention block with a cross-attention
graph break, checks exact eager/compiled outputs and K/V reuse, and verifies
leased-output lifetimes and native fallback. These checks cover the observed
`nullcontext(None)` failure without disabling compilation or changing workloads.
Each process verifies that it loaded the requested source checkout. A package
or import mismatch fails setup without starting model inference. NPU also
requires an integrated vLLM 0.33 Ascend stack, including `torch-npu` and
`vllm-ascend`, that passes the same two-source preflight. Pin the vendor image
by digest and retain its actual package versions: a `v0.33.0` image tag can still
contain a development `vllm-ascend` package and vendor-specific Torch version.
Source preflight and package metadata do not establish NPU model qualification.
Do not replace the integrated vendor stack with a CUDA wheel.

The pinned official Ascend image contains an unstaged Dockerfile download-URL
patch. NPU profiles bind
[runtime_sources.lock.json](runtime_sources.lock.json), which records the exact
vendor HEADs, trees, index entries, Git flags, submodule and original file
status, modes and hashes. The attestor reads this state without modifying the
vendor checkout. The Ascend checkout remains recorded as dirty; any other
tracked, staged, untracked, concealed-index or file-content change fails setup.
Generation verifies this state again before model inference and includes its
actual attestation in the hardware contract. Both reference cohorts and the
frozen performance and quality records must carry the same attestation.
Profiles bind both the lock-file hash and the canonical package-record hash;
each frozen verifier checks that package hash independently for both records,
so matching malformed records cannot bypass source validation.
Supplier source/native file inventories are provenance evidence; they do not
establish loaded-library, kernel, performance or video-quality qualification.
The preceding protocol is archived byte-for-byte in
[the source-snapshot history](calibration/history/v033_reference_protocol_before_vendor_source_snapshot.json).
No training cohort was collected before this provenance update; fitting rules,
workloads, numerical allowances and the frozen reference revision are unchanged.

For PR builds, LingBot NPU jobs require `npu-test` together with `nightly-test`
or `weekly-test`. A shared nightly/weekly label alone must not allocate NPU
benchmark resources. Scheduled non-PR runs still use `NIGHTLY=1` or `WEEKLY=1`.

The explicit `h100_8` preset requests eight `nvidia.com/gpu` scheduler slots
on `mithril-h100-pool`, preventing another GPU job from sharing a current
eight-GPU node during paired measurements. The name describes reserved
capacity; each full profile still sets `dies: 4`. This reservation is excluded
from implicit marker inference, so existing `not cards_1` jobs retain their
four-GPU allocation. The launcher verifies all eight reserved
devices, records their UUIDs in `allocation.json`, then preserves the original
device mapping while exposing only four to inference. This retains SP4, memory
and precision settings. It changes scheduling isolation rather than regression
thresholds. A four-GPU allocation cannot silently run an exclusive profile.
For a local full CUDA run, reserve eight devices with `gpu run --gpus 8 -- ...`
before invoking the launcher. The public nightly/weekly scheduling remains
configured; maintainer confirmation of shared-pool versus local-runner
placement is pending. A local run uses the same launcher, profiles, full
measurements and gates, after its scheduler grants all eight devices.

NPU generation and long-pair collection use the existing `a2b3_npu_8`
scheduler preset. Before model preparation, a separate preflight process
requires eight runtime-visible devices and records their names, memory sizes
and CANN runtime identifiers in `allocation.json`. The launcher then exports
the first four identifiers through `ASCEND_RT_VISIBLE_DEVICES`; inference
rechecks the device count, mapping and properties. These identifiers are not
physical UUIDs. The eight-slot reservation reduces device co-location while
retaining SP4, the workload and the reference/candidate regression limits.
It does not prove exclusive access to host CPU, storage or networking.
The launcher reads the selected identifiers from `allocation.json`, so backend
startup messages on stdout cannot become part of the device mask.

```bash
pytest -q --confcutdir=tests/benchmarks \
  tests/benchmarks/test_lingbot_world_realtime.py \
  tests/benchmarks/test_lingbot_world_regression.py \
  tests/benchmarks/test_lingbot_world_precision.py \
  tests/benchmarks/test_lingbot_world_video_quality.py \
  tests/benchmarks/test_lingbot_world_ci.py
```

The CPU regression job belongs to the existing Simple Test group and runs only
`tests/benchmarks/test_lingbot_world_*.py`, without loading a checkpoint. The
ordinary Other Test sweep excludes those exact files. Shared uploader, AMD,
helper, diffusion-model and offloader contracts retain their existing Simple
jobs and are no longer replayed in the dedicated model job. The scheduled
nightly CPU coverage job still collects the whole suite for its distinct
coverage purpose. Selected model-contract skips remain failures through the
explicit DFX pytest plugin.

The shared `run-pytest-ci.sh` wrapper handles report initialization, package and
source metadata, setup/interruption errors and exit propagation. Both the
ordinary Other Test job and model jobs use it. Only paired-checkout, device-mask
and evaluator preparation remains in `tests/dfx/lingbot_world/run_ci.sh` beside
the workload tests; hardware expansion, job admission and artifact upload use
the existing shared Buildkite mechanisms.
The model stages download the pinned checkpoint if `LINGBOT_MODEL_PATH` is unset.
Only that explicit provisioning subprocess enables Hub access; inference keeps
the runner's offline setting.
For an existing local or mirrored checkpoint, export that path first. All model
files and images must match `assets.lock.json`; directory names do not matter.

```bash
BUILDKITE=true CUDA_VISIBLE_DEVICES="" \
  CI_POLICY_FILE=tests/dfx/lingbot_world/quality.json \
  bash .buildkite/common/scripts/run-pytest-ci.sh lingbot-artifacts/unit \
  python -m pytest -sv -p tests.dfx.lingbot_world.conftest \
  tests/benchmarks/test_lingbot_world_*.py --lingbot-output=lingbot-artifacts/unit \
  -m 'core_model and cpu' --run-level core_model --junitxml=lingbot-artifacts/unit/pytest.xml
bash tests/dfx/lingbot_world/run_ci.sh smoke \
  tests/dfx/lingbot_world/configs/cuda_smoke.json lingbot-artifacts/cuda-smoke
bash tests/dfx/lingbot_world/run_ci.sh generate \
  tests/dfx/lingbot_world/configs/cuda_nightly.json lingbot-artifacts/cuda-nightly
```

Source the CANN and ATB environments and substitute `npu_nightly.json` for Ascend.
The script prepares a clean, separately checked-out reference revision and
runs pytest with the correct platform/card selection and `--run-level full_model`.
Direct equivalent after preparing `/tmp/lingbot-reference`:

```bash
export LINGBOT_ALLOCATION_FILE="$(pwd)/lingbot-artifacts/cuda-nightly/allocation.json"
CUDA_VISIBLE_DEVICES="$(python -m tests.dfx.lingbot_world.prepare devices \
  --profile tests/dfx/lingbot_world/configs/cuda_nightly.json \
  --root lingbot-artifacts/cuda-nightly)"
export CUDA_VISIBLE_DEVICES
pytest -sv tests/dfx/lingbot_world/test_generate.py \
  --lingbot-profile tests/dfx/lingbot_world/configs/cuda_nightly.json \
  --lingbot-reference-root=/tmp/lingbot-reference \
  --lingbot-output lingbot-artifacts/cuda-nightly \
  -m 'full_model and diffusion and cuda and cards_4' --run-level full_model \
  --junitxml=lingbot-artifacts/cuda-nightly/pytest.xml
```

NPU profiles use four A2 cards with eager execution and component-level CPU
text-encoder offload; DiT/VAE stay on the NPU. The all-resident configuration
OOMed on the target runner at the first warmup (64GB cards expose 60.96GiB).
The CI deployment retains the full resolution, KV window and rollout lengths;
`gpu_memory_fraction=0.1` avoids allocating extra pages above the one-session
minimum after offloading frees device memory. This placement is identical for
reference/candidate and is included in the hashed deployment contract.
The same deployment disables eager multithreaded safetensors loading for both
roles: four ranks otherwise read up to sixteen whole shards concurrently from
the shared checkpoint filesystem. The serial iterator preserves the checkpoint
tensors and leaves the 1,200-second initialization budget unchanged. Startup
success must still be established on the target runner.
The NPU profiles pin `cpu_threads_per_worker=1`. Both reference and candidate
servers and fixed-input captures receive `OMP_NUM_THREADS`, `MKL_NUM_THREADS`
and `OPENBLAS_NUM_THREADS` before importing Torch, while retaining CANN's
backend import paths. This bounds host-side parallelism across the four ranks.
The setting is part of the hashed profile; changing it requires fresh
independent reference calibration and target-hardware validation.

Both server roles, numerical captures and the PR smoke explicitly disable
`VLLM_OMNI_AR_DIFFUSION_KV_GATHER` and set
`AR_DIFFUSION_KV_SCRATCH_BLOCKS_PER_BRANCH=0`, retaining the model-declared
scratch minimum. Host overrides cannot change these optimization settings.
The effective values are retained in `contract.json` as `engine_environment`
before capture or inference starts. The smoke retains its profile, deployment
hash and the same switch whitelist in `configuration.json` before provisioning.
The smoke passes the profile's chunk count, warmup exclusion, mux/target frame
rates and receive deadlines explicitly to the benchmark. It recomputes raw
chunk/session metrics with the shared validator and retains `metrics.json`;
incomplete, malformed or policy-mismatched output fails before cancellation
checks, while the server context still cleans up. Its short timings are
functional diagnostics and do not establish a calibrated performance baseline.
Historical campaigns did not record these environment switches; final-source
target runs must validate the pinned setup.
The NPU aggregate rejects missing or changed values even when all three
measurement pairs agree with each other.

CUDA nightly, weekly and long generation jobs retry a lost Buildkite agent
(`exit_status=-1`) at most once. The original attempt and its artifacts remain
retained. Pytest failures, OOM, cancellation and timeout are excluded from this
retry policy; required gates still fail. Artifact consumers use the current
attempt and verify the complete matching manifest before acceptance.

The NPU bootstrap reference is `8d7c354674c7e1dc57544f248689574fccb71af9`,
which places unselected DiT weights before direct AR projections and releases
selected text weights before video conditioning. Those placement fixes are
required for this component-offload deployment. CUDA keeps its original reference;
every platform still calibrates independent reference starts before checking candidates.

## Performance, precision and lifecycle

Three reference and three candidate server starts run in alternating AB/BA
order. Each reference/candidate pair shares its device reservation and dependencies.
Raw benchmark JSON retains unmeasured warmup chunk arrivals and startup timings
under `warmups`, separately from measured `sessions` and aggregate `metrics`.
These records are serialized after measurement and add no requests or settling
delays. They help distinguish warmup behavior from subsequent TTFC failures.
If a warmup or measured session fails, the client writes a separate
`benchmark-{warmup|session}-N-failure.json` containing received chunk timings,
protocol events and the original error. This diagnostic has `complete: false`
and no aggregate metrics; it cannot establish acceptance. Trace writes occur
after failure and preserve the original exception even if the filesystem fails.
Lifecycle checks use a one-second playback window: the initial nine-frame
block fits, but admitting the following twelve-frame block requires playback
progress. The held-playhead assertion therefore exercises backpressure before
second-block inference, independently of model speed. These instrumented
requests remain outside performance measurements.
Each case receives an unmeasured
eight-chunk warmup session. The first six and terminal chunks stay in raw results and
playback/FPS accounting, but are excluded from steady cadence by `workload.py`.

Reference mean/P50 CV above 3% invalidates calibration. NPU P95 retains the
same 3% limit. The CUDA nightly P95 reference-CV ceiling is
**3%**, derived from the largest of six
case/cohort CVs across builds 17633/17640, adding 25% and rounding upward to
0.005 with a 3% minimum. Each group contains three independent reference starts.
The complete measured profile is bound by `qualified_profile_sha256`; weekly
and long runs require their own reference cohorts. Raw hashes and values remain
in the calibration record, and `p95-cv-calibration.json` records the applied rule
and record hash. Each candidate retains the profile's 5% regression allowance.

For TTFC and accumulated playback stalls only, the allowance is the greater of
5% of the worst reference and three reference population standard deviations.
The profile pins this multiplier to at most three. CUDA nightly additionally
uses reference-only jitter floors of **35 ms for TTFC** and
**248 ms for total stalls**. The two reference cohorts
supply the maximum two-way excess; adding 25% and rounding upward to 5 ms and
1 ms respectively selects these floors. All 18 reference measurements, raw
artifact hashes and fitted null controls are retained. Candidate observations
never select either floor. Fitted controls are not independent validation. The complete two-way reference
comparison retains one failed steady-stall control (street-42, build 17633,
reference start 0): a 1,916.7 ms arrival interval at chunk 10 causes 277.9 ms of
post-startup stalls. Its underlying cause is not established. This observation
is retained, receives no new steady-stall floor, and keeps construction from
being described as fully passing. A subsequent independent campaign must pass
the unchanged steady-stall gate before acceptance.
`ttfc-calibration.json` and `startup-stall-calibration.json` retain the applied
rule and record hash. NPU uses neither constant floor.

The total-stall floor requires an independent post-startup stall gate without
that floor, so later stalls cannot hide behind startup jitter. The startup
boundary is `max(2, warmup_chunks)` media blocks (six in CI); both components
remain in total-stall accounting. The one-chunk playback buffer is explicitly
passed and checked. TTFC floors do not change cadence, FPS, P99 or stall limits.
Independent final-source target validation remains required for every profile.
Candidate measurements never calibrate their own limits. This is a relative
regression gate, not a promise of 12 FPS. The MP4 label and playback FPS basis
are recorded separately. P99 is always reported but gates only when each
independent start has at least 1000 steady observations; long profiles use
two measured 600-chunk sessions per start.

CI partitions weekly's 18 cases into six CUDA and six NPU groups of three cases. Each case still receives all three reference and three
candidate starts. Generated effective profiles retain the source profile SHA256;
generation and evaluation use the same `LINGBOT_CASE_SHARD=INDEX/COUNT` selection.
No scene, seed, rollout length or measurement is removed.
Weekly scoring splits each generation shard into three required evaluator jobs,
selected by `LINGBOT_QUALITY_CASE_INDEX=0`, `1` or `2` (pytest option
`--lingbot-quality-case-index`). Each evaluator scores one case's six videos:
three independent reference starts and three candidate starts. All 18 cases
remain mandatory for both CUDA and NPU, and each result records its selected
case and the complete, unchanged generation profile and manifest hash. Every
producer artifact is still checksum-verified; a failed generation gate keeps
its evaluator failed even when native quality passes. Evaluator outputs use
separate `shard-N/case-M` directories.
The CUDA evaluator keeps its 240-minute job limit. Three-case/18-video jobs
[timed out in build 17498](https://buildkite.com/vllm/vllm-omni/builds/17498#01a14c3f-4df0-425a-98f4-05f93e00f899),
after the earlier six-case layout also exceeded the effective CI limit. These
failed runs remain evidence; fresh target runs must establish that the per-case
layout finishes. Generation profiles, warmups, starts, rollout lengths and
regression budgets are unchanged.
The quality coordinator checkpoints evaluator versions, hardware, input manifest
and policy hashes before launching native workers. A forced job termination can
bypass Python cleanup, so this checkpoint stays failed until the complete gate
finishes; it also retains provenance when only one native suite completed.
The initial failed generation report records the actual candidate source revision
before asset checks, numerical capture or server startup. Initialization failures
retain that revision even when no complete measurement manifest can be written.
Aggregation and quality also record their own actual revision before validating
inputs. A partial handoff must not force diagnosis to infer the consumer's
version from copied producer files.

NPU nightly and long collection each run as three independent reference/candidate
pairs, followed by a blocking CPU aggregation job. Nightly retains its three
scenes, 40-chunk rollouts and 477-frame quality clips, with seven measured
sessions per start. Its 231 steady samples exceed the P95 sampling guard of
200 (ten expected observations in the upper 5% tail). Three-start reference
P95 CV remains capped at 3%, and candidate cadence degradation at 5%.
This follows [recorded zero-switch reference-only measurements](calibration/npu_cadence_sampling.json):
33-sample nightly reference P95 CV reaches 7.141%, while mean/P50 stay below
1.21%. Nonoverlapping windows within one long reference start also show
5.743% P95 CV at 33 samples, versus 1.396% at 198 samples. Those cropped
windows diagnose sample dispersion; independent seven-session target runs
still must validate this protocol. This sampling guard is not a confidence
bound. P99 remains report-only below 1,000 steady samples.
The three pairs reuse the existing collector and aggregate, retain all
independent starts and keep each nightly collection timeout at 240 minutes.
The same guard gives NPU weekly two 160-chunk sessions (306 steady samples)
per start; all 18 cases and six shards remain. Weekly also collects three
independent pairs per shard, with a 240-minute collection deadline and a
separate 60-minute CPU gate per shard. The original serial weekly job was
[timed out after 300 minutes](https://buildkite.com/vllm/vllm-omni-npu-ci/builds/9202#01a1497f-717f-4416-8f01-8e323f49d042)
despite requesting 1,440 minutes, retaining only two complete starts per role.
Splitting collection preserves the full population and all measurements; it
does not increase deadlines or resolve unstable reference P95 calibration.
The long profile still has two 600-chunk sessions
and at least 1,186 steady observations. The first pair also runs both complete
600-chunk memory/control diagnostics. Pairs may run on different nodes, but the
aggregate rejects different hardware/dependency contracts and unstable reference
populations. It verifies all three complete manifests, revisions and checksums,
rejects duplicated sessions, recomputes performance from raw chunks, and applies
the same performance and seven VAE precision checks. Collection success alone is
not regression acceptance. Quality jobs depend on the aggregate, retain scores
for complete failed populations, and require both aggregate and quality gates
to pass for acceptance.

These partitions keep individual jobs below the effective timeout observed in
the CI environment. A configured step timeout does not override a smaller
[pipeline/organization maximum](https://buildkite.com/docs/pipelines/configure/build-timeouts).

Fixed-input VAE diagnostics run outside performance windows. Both checkouts
load the real Wan VAE and decode the same CPU-seeded BF16 latent. Independent
same-code repeats must be exact before reference/candidate tolerances apply.
This is VAE numerical evidence, not a full DiT replay or GPU/NPU bitwise-parity
claim. Generated videos additionally cover full-pipeline behavior.

Existing public endpoint tests check chunk order, pacing, stop and successful
session reuse. An additional instrumented rollout sends a live SE3 camera update
(with the fixed action script disabled, as required by the pipeline),
requires its event ID in streamed started/active/completed metadata, and samples
device-wide memory on the allocated devices. The post-warmup growth budget is
256 MiB per die. Diagnostic timings never enter performance limits. Server
child processes must exit after teardown.

## GPU evaluation, including NPU videos

Selected CI pixel comparisons use the same verified `LINGBOT_EVALUATOR_PYTHON`
as native scoring. Separate isolated Python workers compute PSNR and sampled
SSIM with the locked NumPy, OpenCV and scikit-image versions. Each worker retains
its inputs, log, result, actual dependency versions and OpenCV build hash;
timeouts terminate its process group and leave an incomplete result. The
comparison window, frame population, SSIM stride and candidate budgets stay fixed.

The evaluator uses a separate Python 3.12 environment with resolved dependencies
in `requirements/lingbot-quality.txt`. `evaluators.lock.json` pins code revisions,
submodules and asset checksums. Provisioning validates downloads before scoring.
An optional `LINGBOT_EVALUATOR_CACHE` supplies files at the lock's relative
paths; cached files still undergo checksum verification.
Upstream dependencies install both `opencv-python` and `opencv-python-headless`,
which share the `cv2` namespace. Matching package versions alone cannot identify
the imported implementation. Fresh provisioning reinstalls the pinned headless
wheel after dependency sync, then verifies its imported loader, native module
and bundled files against the headless distribution's SHA256 `RECORD` entries.
The selected interpreter and binary/build fingerprints are retained in
`opencv-provider.json` and the quality report. Pixel workers must match that
verified build before their results can pass.
The Buildkite cache key includes the evaluator/weight lock, resolved dependency
lock and provider-verification code. A completed interpreter is never
resynchronized underneath an active scorer. Cached interpreters are reverified;
changed ownership or provenance fails without repair. If any cache input changes,
use a fresh custom evaluator root.

```bash
export LINGBOT_EVALUATOR_ROOT=/path/to/evaluator-cache
python -m tests.dfx.lingbot_world.prepare evaluators \
  --root "$LINGBOT_EVALUATOR_ROOT"
export LINGBOT_EVALUATOR_PYTHON="$LINGBOT_EVALUATOR_ROOT/venv/bin/python"
pytest -sv tests/dfx/lingbot_world/test_quality.py \
  --lingbot-profile tests/dfx/lingbot_world/configs/npu_nightly.json \
  --lingbot-generation lingbot-artifacts/npu-nightly \
  --lingbot-output lingbot-artifacts/npu-nightly-quality \
  -m 'full_model and diffusion and cuda and H100 and cards_1' \
  --run-level full_model --junitxml=lingbot-artifacts/npu-nightly-quality/pytest.xml
```

Transfer the complete generation directory to a GPU evaluator. Candidate commit,
profile and artifact checksums must match. Seven VBench custom-input dimensions
and six WBench video-quality dimensions are checked per case. Continuous scores
are checked for every candidate repeat. Both pinned dynamic-degree evaluators
return a binary classification for each complete video. Activity therefore uses
a two-sided difference in active counts across three independent starts: the
unchanged 0.34 fractional budget permits one changed classification, while two
or three changes fail. All six individual classifications remain in the report.
The active [reference-only record](calibration/cuda_nightly_reference.json)
uses independent three-start H100 reference cohorts from builds
[17633](https://buildkite.com/vllm/vllm-omni/builds/17633) and
[17640](https://buildkite.com/vllm/vllm-omni/builds/17640), with matching model,
deployment, runtime, playback/warmup policy and recorded optimization switches.
Both suites execute on H100 with all 127 pinned dependencies. The record retains
six native reference scores per scene/dimension, public Buildkite identities and
raw hashes. Binary activity preserves the unchanged 0.34 count budget.

Pixel calibration computes all 15 unordered pairs of those six reference videos,
including nine cross-cohort pairs. The locked decoder checks every one of the
477 frames and every window. Version 3 requires two complete three-start
reference cohorts, every pair exactly once, six native reference scores per
dimension and complete per-case reference-spread limits. Candidate videos and
scores do not enter this population. The [pre-collection protocol](calibration/cuda_nightly_v032_protocol.json) was
recorded before collecting the second cohort and is embedded with its source
revision and hash.

Excessive continuous reference spread fails calibration; scores are never
averaged across dimensions or scenes. Each scene/dimension receives the larger
of its six-reference range plus 25%, rounded upward to 0.005, and the unchanged
conservative global limit. Fresh-reference and candidate checks use that case's
limits. The candidate floor remains the worst fixed reference minus its unchanged
tolerance. Weekly and NPU retain their separate global policies.

Both collection campaigns retain their original failed calibration gates. The
new records use reference-role inputs only, retain fitted construction checks,
and preserve candidate performance, score, binary-activity and pixel allowances.
Third-cohort acceptance must use the already frozen records, retain failed
observations and investigate failures without refitting. Earlier pixel-method
reencode, localized-degradation, truncation and held-out controls remain in the
[historical records](calibration/history/README.md); those controls do not qualify
the new reference epoch or replace current target-hardware CI.

The separate [600-chunk population](calibration/cuda_long_quality_population.json)
still contains historical measurements without recorded optimization switches
or decoder provenance. The strict guard rejects it. Its replacement requires
independent complete long reference/native cohorts; nightly measurements cannot
establish long acceptance. Weekly and NPU profiles likewise need their own target
validation. Local diagnostic scores do not establish hardware acceptance.
Candidate tolerances are also keyed by generation platform. CUDA VBench
aesthetic uses 0.025: independent same-code reference deficits reached
0.018552, exceeding the initial 0.01 budget, before adding 25% reserve and rounding
up to 0.005. Other native candidate budgets stay at 0.01 (HPSv3: 0.02).
NPU retains its initial native budgets. Historical candidate-budget derivations
remain in the archived record; this nightly recalibration leaves them unchanged.
They are separate from reference-spread ceilings and still require fresh
target-runner validation; candidate observations never set their own limits.
Standard VBench background consistency encodes every native-selected frame
through CLIP in batches of 32 with gradients disabled. The pinned native
implementation otherwise retains a full-rollout gradient graph and exceeds
even 140 GiB on a 600-chunk reference video. This adapter changes only that
metric's image encoding; native transforms, normalization, cosine aggregation
and frame selection are preserved. Other metrics keep their native paths and
the evaluator checkout stays immutable. Reports bind the adapter source hash
and retain per-video encoded-frame/batch counters. CI requires every expected
frame and rejects a missing counter, changed batch size or enabled gradients.

Both native score populations are saved before calibration checks. A reference
calibration error is recorded as a failed case check; remaining cases and pixel
checks still run and retain their evidence. The final pytest assertion fails
if any native calibration, candidate score or pixel check fails.
All selected scores from these pinned evaluator revisions use a 0–1 scale.

Video diagnostics validate full frame/geometry/FPS coverage, compute PSNR on
every frame and SSIM every 24 frames, and compare candidate divergence against
all three pairs of independent same-code reference repeats. Candidate PSNR
must stay within 3 dB of the worst reference distance. Sampled SSIM may drop by
0.04 on CUDA or 0.03 on NPU. Independent CUDA reference SSIM deficits reached
0.031969, exceeding the initial 0.03 margin; 25% reserve rounded upward to 0.005
gives 0.04. Their maximum PSNR deficit was 2.535 dB, so the existing 3 dB margin
is retained. Only failed reference-only budgets are adjusted; the CUDA values
are not copied to NPU. These are approximate rollout-divergence budgets; the separate fixed-input
VAE numerical gate still requires exact repeats. PSNR is also gated independently in every two-second
window. Perfect repeats use
an explicit minimum PSNR budget rather than arithmetic on infinity. Failed pixel
checks retain the reference noise floors and every candidate result.

The fixed-photograph CI profiles additionally reject any decoded frame whose
pixels all have the same RGB value, in both reference and candidate outputs.
A real 600-chunk reference population admitted an inserted 32-frame black
window under relative pixel budgets (5.45 dB against a 4.69 dB floor). The
independent spatial-content check catches that collapse without widening or
tightening the measured noise budgets. Every decoded frame is checked; reports
retain `uniform_frame_indices`, and missing coverage fails selected CI tests.
Standalone comparison diagnostics retain support for intentionally uniform
videos unless this CI policy is explicitly selected.

These are fixed-input CI video-quality evaluations, not official leaderboard
totals. WBench navigation, reconstruction and API/VLM dimensions are outside
this video-quality suite.

## Buildkite and baseline maintenance

- READY: CPU benchmark contracts in Simple Test and a one-GPU real-checkpoint
  stream/cancellation smoke in the existing E2E Test group. The CPU job uses the
  standard L4 CI image with accelerators masked, and retains JUnit, the quality
  policy and package versions. Selected skips fail through the same pytest hook
  as model jobs. The smoke also survives the uploader's `--e2e` weekly sweep.
- NIGHTLY: independent CUDA/NPU generation in the standard
  `:card_index_dividers: World Model Test` group, followed by one-GPU evaluation.
  The NPU build provisions its matching CUDA evaluator image in the same build.
- WEEKLY/manual (`weekly-test`): the same World Model Test group, three scenes,
  three camera patterns and two
  seeds (18 cases), 160 chunks each across the groups above, plus the complete
  600-chunk long profile. NPU long quality waits for all three collected pairs
  and their CPU performance/precision aggregation gate.

The regression suite uses organization-wide Buildkite concurrency groups shared
across PR, nightly, weekly and manually triggered builds. Matrix shards and
retries consume the same slots; group names must never include a build, branch,
schedule or matrix identifier.

| Work | Shared group (`vllm-omni/lingbot-world/` prefix) | Maximum jobs | Reserved devices at maximum |
| --- | --- | --- | --- |
| CUDA generation and PR smoke | `cuda-generation` | 1 | 8 H100 (smoke uses 1) |
| NPU collection | `npu-generation` | 1 | 8 NPU |
| CUDA and NPU video quality together | `quality` | 2 | 2 H100 |
| CPU unit and aggregation on L4 workers | `l4-cpu` | 1 | 1 L4, masked during tests |

All regression jobs use `priority: 0`, the same as normal PR tests. All four
groups use `concurrency_method: eager`, allowing ready jobs to claim available
slots without waiting behind earlier dependency-blocked jobs.
Concurrency limits do not preempt running jobs or reserve capacity for other PRs.
These limits bound this suite's peak reservation, not its total device-hours:
all cases, starts and quality dimensions remain required. They apply to newly
uploaded jobs; already queued jobs and their retries retain their old settings.
Direct local shell invocations are outside Buildkite's concurrency control.
See Buildkite's [concurrency](https://buildkite.com/docs/pipelines/configure/workflows/controlling-concurrency)
and [priority](https://buildkite.com/docs/pipelines/configure/workflows/job-priority)
semantics.

Selected jobs are blocking. Missing prerequisites and skipped CI tests fail.
NPU generation uses the existing infrastructure retry codes (`-1`, `128`),
with one retry per code. Pytest failures and OOM are never automatically retried.
The generation steps preserve `retry.manual.permit_on_passed: true` alongside
their narrow automatic infrastructure retries. Numerical, quality, OOM and
cancellation failures do not become automatic reruns.
JUnit, raw chunks, measured limits, precision captures, memory/events, videos,
scores and server/capture/evaluator logs are retained through `artifact_paths`,
including failed jobs. Buildkite retains job console output, so the shared
wrapper and Other Test no longer duplicate it as `run.log` or `pytest.log`.
A failed placeholder JUnit report is written before provisioning so a forced
SIGKILL still leaves failure evidence. Pytest replaces it on completion; normal
failure and SIGTERM also retain structured launcher errors. A zero exit with
missing, empty, incomplete, failed or entirely skipped JUnit is rejected.
Provisioning records host available/total RAM, cgroup membership, raw memory
limits/peaks/pressure counters, available v1/v2 CPU quota/throttling/pressure
counters and CPU-set masks,
and the resolved checkpoint filesystem in `model.json`.
Pytest failure JSON captures fresh kernel resource evidence
after cleanup; peak counters can still expose load-time pressure. Kernel paths
remain explicit because cgroup namespace layouts differ. Unavailable interfaces
remain absent/null; read errors are retained without replacing the original
test failure.
These read-only snapshots run before serving or after failure, outside timing.
Failed server startup also logs up to 32 process records before cleanup,
including status, CPU time, memory, available I/O counters and Linux wait channels.
Unavailable processes or interfaces retain errors/null values; omitted process
counts remain explicit. Command lines and environment variables are not captured.
This failure-only snapshot preserves the original startup error and cleanup;
it supplies diagnostic evidence without identifying a stall's cause by itself.
After a successful server context exits, smoke and full collectors explicitly
assert the shared cleanup helper's 5% engine-memory condition on every selected
logical device. This check uses a zero wait budget after normal cleanup and sits
outside performance measurements. The shared helper uses a monotonic deadline,
so clock corrections cannot add a wait. A failed `cleanup.json` checkpoint is
written before the probe; residual allocations, unavailable devices and invalid
device populations fail pytest and retain that checkpoint. This strict call
preserves the shared helper's existing Whisper-worker accounting; LingBot does
not load a Whisper worker.
An unstable reference remains a failed calibration in the complete generation
manifest. Other cases, all seven VAE checks and video checksums are still
retained, so independent scoring can inspect the failed population.
Quality steps also run after a failed generation gate: when all videos and
their checksums are retained, both native suites and pixel checks still record
their evidence. `quality.json` separates `generation_passed` and
`quality_passed`; overall `passed` requires both. Missing, unhashed, corrupt or
incompatible generation populations fail before scoring. The generation job
remains blocking, and evaluator-image failures still prevent scoring.

To reproduce one CUDA weekly group, set `LINGBOT_CASE_SHARD=0/3` when
invoking both `generate` and `evaluate`. For one NPU weekly group:

```bash
for repeat in 0 1 2; do
  LINGBOT_CASE_SHARD=0/6 LINGBOT_MEASUREMENT_REPEAT="$repeat" \
    bash tests/dfx/lingbot_world/run_ci.sh collect \
    tests/dfx/lingbot_world/configs/npu_weekly.json \
    "lingbot-artifacts/npu-weekly-measurements/shard-0/pair-$repeat"
done
LINGBOT_CASE_SHARD=0/6 bash tests/dfx/lingbot_world/run_ci.sh aggregate \
  tests/dfx/lingbot_world/configs/npu_weekly.json lingbot-artifacts/npu-weekly/shard-0 \
  lingbot-artifacts/npu-weekly-measurements/shard-0
```

Use `LINGBOT_CASE_SHARD=0/6` for its subsequent `evaluate` invocation too.
Each CPU gate consumes only its shard's three pair prefixes. A failed pair
keeps that shard failed; complete shards still retain their own results.
To reproduce NPU long
collection, invoke `collect` three times with `LINGBOT_MEASUREMENT_REPEAT=0`,
`1` and `2`, writing to `lingbot-artifacts/npu-long-measurements/pair-0`,
`pair-1` and `pair-2`. Each invocation reserves eight NPU slots and uses four
visible devices.
Then run the CPU gate without loading a checkpoint:

```bash
bash tests/dfx/lingbot_world/run_ci.sh aggregate \
  tests/dfx/lingbot_world/configs/npu_long.json lingbot-artifacts/npu-long \
  lingbot-artifacts/npu-long-measurements
```

For a local run, leave `BUILDKITE` unset so the already collected directories
are used directly. On CI the launcher downloads the unique generation prefix
from the current build, so every matrix pair is available. Quality jobs use
their matching platform/profile/shard prefix. Pipeline dependencies and
source/contract/checksum validation still bind the downloaded data to its
producer. See the [agent artifact download reference](https://buildkite.com/docs/agent/cli/reference/artifact).
Use the aggregate's output directory for the subsequent GPU evaluator.

For source attribution, use `candidate_revision` from the retained
`generation.json`. Buildkite can test a synthetic merge containing upstream
changes absent from the PR branch. Compare that merge with `reference_revision`
and its upstream parent before attributing failures to the PR or host noise.
Comparing only the unmerged branch cannot establish source equivalence.
Upstream differences alone do not justify replacing the reference or widening
limits; preserve the failed observations while investigating their cause.

Advance a reference only by changing its full SHA and inspecting three new
independent reference starts on every target runner. Model, input, evaluator,
topology or precision changes require reviewed lock/profile updates and fresh
measurements. Include matching-head Buildkite URLs in the PR. Never drop failed
observations, automatically widen budgets, or treat CPU tests and older hardware
experiments as acceptance of the new commit.

For a quality calibration update, retain two independent three-start reference
cohorts, their raw-score hashes and matching numerical contracts. Evaluate
each cohort as held-out reference data against the other before adjusting any
failed budget. Candidate outputs do not enter this calculation. Update the
reference-only record and its profile-owned population policy together, retaining
original failed observations and unchanged candidate budgets, then run
`pytest tests/benchmarks/test_lingbot_world_regression.py -m 'core_model and cpu'`
and a fresh target-runner campaign. CUDA quality jobs reject calibration reuse
after deployment, hardware, runtime package, precision or model/reference changes;
they retain the applied policy, generation profile and calibration record.
The CUDA nightly record does not establish NPU or weekly/long acceptance.

The CUDA 40-chunk quality profile also retains a fixed reference population in
`calibration/cuda_nightly_quality_population.json`: six reference starts from
two target-H100 cohorts, native continuous scores and all 15 decoded reference-pair
summaries per scene. Its provenance records the Buildkite jobs, raw artifact and
reference-video hashes. Candidate scores do not enter this population. Three
fresh references still run in every campaign: their native spread must remain
bounded, and their native and pixel results must pass the fixed population's
limits before candidate quality can pass. Binary activity still compares the
fresh three-start counts. Candidate HPS, PSNR and SSIM budgets are unchanged.
This prevents an unusually favorable three-reference cohort from silently
tightening the candidate gate. Model, deployment, inputs, hardware/runtime,
asset or evaluator changes reject population reuse. CUDA 600-chunk runs use the
separate population described above; NPU and CUDA weekly profiles retain their
own reference comparisons. The retained vLLM 0.32 nightly record binds both historical reference
cohorts and freshly recomputed pixel summaries from their actual reference
videos, using NumPy 1.26.4, OpenCV 4.11.0.86 and scikit-image 0.25.2; the OpenCV
build hash is retained. No local interpreter paths are required by the record.
To replay the 15 pairs for a case, first provision the pinned evaluator as above
and download its six recorded reference videos from their public Buildkite jobs.
Place them under `reference-videos/<build-number>/<recorded-reference-path>`.
The following example verifies their recorded hashes and retains the complete
comparison outputs; review the summaries and independent controls before updating
the active record. Collect new cohorts with three independent reference starts,
never by copying or relabeling an existing video.

```bash
export LINGBOT_CALIBRATION_CASE=street-42
python - <<'PY'
import itertools
import os
from pathlib import Path
from benchmarks.lingbot_world.regression import read_json, sha256, verify_environment, write_json
from benchmarks.lingbot_world.video_quality import compare_video_pair_in_environment

root = Path("tests/dfx/lingbot_world")
population = read_json(root / "calibration/cuda_nightly_quality_population.json")
case = population["cases"][os.environ["LINGBOT_CALIBRATION_CASE"]]
policy = read_json(root / "quality.json")
python = Path(os.environ["LINGBOT_EVALUATOR_PYTHON"])
packages = verify_environment(python, Path("requirements/lingbot-quality.txt"))
videos = []
for video in case["reference_videos"]:
    path = Path("reference-videos") / video["build_url"].rsplit("/", 1)[-1] / video["path"]
    assert sha256(path) == video["sha256"]
    videos.append(path)
rows = []
for first, second in itertools.combinations(range(6), 2):
    pixels = compare_video_pair_in_environment(
        videos[first], videos[second], Path("reference-pixel-replay"),
        python=python, expected_packages=packages,
        window_seconds=policy["pixel_window_seconds"], ssim_stride=24,
        timeout_s=policy["timeout_s"],
    )
    rows.append({"pair": [first, second], "pixels": pixels})
write_json(Path("reference-pair-replay.json"), {"comparisons": rows})
PY
```

The original unrecorded nightly records remain byte-identical under
[calibration/history](calibration/history/README.md) for historical method and
rejection tests. They are not loaded by CI. The long record still requires
replacement. Independent final-source target runs remain required; neither
fitted data nor replaying old scores establishes acceptance.
Both calibration validators require a nonempty `engine_environment` mapping
with string keys and values matching the measured contract. Missing or changed
settings reject reuse, including when both records omit the field. Historical
records without recorded switches therefore remain diagnostic inputs until
replacement reference cohorts establish their provenance.
Reference-noise calibration also requires recorded and matching `target_fps`,
`mux_fps`, `warmup_chunks` and `warmup_num_chunks`. Changing playback or either
warmup policy invalidates calibration; missing fields cannot establish a match.
Legacy shared reference-noise records support held-out scene/seed cases and
rollout lengths under the same runtime and playback/warmup policy.
Fixed quality populations also require every reference pixel summary to record
`pixel_environment.packages` matching the verified evaluator's NumPy, OpenCV and
scikit-image versions. Missing or changed decoder metadata rejects reuse.
Fresh performance records include `qualified_profile_sha256`, the canonical
digest of the complete measured profile. Weekly and long workloads require
their own reference cohorts instead of inheriting nightly qualification.
Version 3 quality populations retain the complete six-start, 15-pair protocol
and record `max_reference_spreads` in each case. Both fresh-reference and
candidate native checks use that case's calibrated limits. Candidate score,
binary-activity and pixel allowances remain unchanged.

Recompute reference summaries from the actual reference videos in the locked
evaluator environment; do not add current versions to historical summaries whose
decoder environment was unrecorded. Native-score dependency verification alone
does not establish the environment used for earlier pixel calculations.
Run the same generation and GPU evaluation commands when collecting replacement
calibration data. A compatibility failure keeps the gates failed while retaining
complete generation manifests, numerical comparisons, raw native scores and
pixel comparisons. Candidate checks are marked unjudged until calibration is
valid. Preserve those failed results, derive replacement records from the
reference inputs only, and run a fresh independent campaign after review.

The pinned `startup_timeout_s` bounds both the outer server-readiness wait and
the server's `--init-timeout` / `--stage-init-timeout`. This avoids silently using
a shorter internal default while cold checkpoints load. Initialization timeouts
still fail pytest and retain their evidence; measurement thresholds are separate.
