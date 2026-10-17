# LingBot World regression gates

The suite drives the real `WS /v1/realtime/video` endpoint. CPU tests validate
gate logic separately and are not model performance evidence.

## Prerequisites and commands

Use the inference environment matching `docker/Dockerfile.ci` (vLLM 0.31.0)
and development dependencies. Reserve devices with the host scheduler before
inference. Full CUDA profiles use four GPUs while reserving all eight GPU slots
of a node in the H100 pool; NPU reserves eight Ascend A2 devices and uses four. The smoke
uses the real 14B checkpoint at 192×320 on one GPU. Full performance/quality
profiles remain 480×832. CI provides exclusive devices.

The CUDA performance preset prevents other GPU jobs from sharing the eight-GPU
node during paired measurements. The launcher verifies all eight reserved
devices, records their UUIDs in `allocation.json`, then preserves the original
device mapping while exposing only four to inference. This retains SP4, memory
and precision settings. It changes scheduling isolation rather than regression
thresholds. A four-GPU allocation cannot silently run an exclusive profile.
For a local full CUDA run, reserve eight devices with `gpu run --gpus 8 -- ...`
before invoking the launcher.

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

The `unit` stage captures CPU test artifacts without loading a checkpoint.
The model stages download the pinned checkpoint if `LINGBOT_MODEL_PATH` is unset.
Only that explicit provisioning subprocess enables Hub access; inference keeps
the runner's offline setting.
For an existing local or mirrored checkpoint, export that path first. All model
files and images must match `assets.lock.json`; directory names do not matter.

```bash
BUILDKITE=true bash .buildkite/common/scripts/run-lingbot-ci.sh unit \
  tests/dfx/lingbot_world/quality.json lingbot-artifacts/unit
bash .buildkite/common/scripts/run-lingbot-ci.sh smoke \
  tests/dfx/lingbot_world/configs/cuda_smoke.json lingbot-artifacts/cuda-smoke
bash .buildkite/common/scripts/run-lingbot-ci.sh generate \
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
same 3% limit. H100 P95 uses a 3.5% reference-CV ceiling fitted from six
predeclared weekly hold cases with three complete reference starts. Their
maximum P95 CV is 2.6194%; the existing 25% reserve rounded up to 0.005 gives
3.5%. The original job was interrupted during candidate collection and remains
failed. Twelve unseen weekly cases also fit that ceiling; these case controls
are not a second independent complete weekly cohort or hardware acceptance.
Raw reference hashes and values are retained in the calibration record;
`p95-cv-calibration.json` records its rule and hash in CUDA results.
Each candidate still receives the profile's 5% regression allowance.

For TTFC and accumulated playback stalls only, the allowance is the greater of
5% of the worst reference and three reference population standard deviations.
The profile pins this multiplier to at most three. H100 additionally uses
reference-only jitter floors of **85 ms for TTFC** and **106 ms for total stalls**.
Two recorded-zero nightly reference cohorts supply the maximum cross-cohort
excess; the existing 25% reserve rounds upward to 5 ms and 1 ms respectively.
The record retains all 18 reference observations, raw artifact hashes and
fitted null controls. Candidate observations never select either floor.
Those fitted controls are not independent validation after selecting the floors.
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
CUDA weekly scoring now evaluates 18 videos per shard, with a 240-minute job
limit. The earlier six-case/36-video orbit shard
[timed out after approximately 300 minutes](https://buildkite.com/vllm/vllm-omni/builds/17464#01a1485e-e82e-4b98-8f8a-b4893af2d460)
despite requesting 1,440 minutes. Its failed JUnit and partial native scores
remain failure evidence. The smaller shards retain the same generation and
quality profiles, warmups, starts and regression budgets; fresh target runs
must confirm that the complete evaluation finishes within the job limit.
The quality coordinator checkpoints evaluator versions, hardware, input manifest
and policy hashes before launching native workers. A forced job termination can
bypass Python cleanup, so this checkpoint stays failed until the complete gate
finishes; it also retains provenance when only one native suite completed.

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
The Buildkite cache key includes both the evaluator/weight lock and the resolved
dependency lock. A completed interpreter is never resynchronized underneath an
active scorer. If either lock changes, use a fresh custom evaluator root.

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
uses independent three-start H100 cohorts from builds
[17464](https://buildkite.com/vllm/vllm-omni/builds/17464) and
[17478](https://buildkite.com/vllm/vllm-omni/builds/17478), with matching models,
deployment, runtime, playback/warmup policy and recorded zero optimization
switches. Both suites execute on H100 with all 127 pinned dependencies.
The record retains all six native reference scores per scene/dimension, public
Buildkite identities and raw hashes. Binary reference activity differs by at
most one classification; its unchanged 0.34 candidate budget still rejects two.

Excessive continuous reference spread fails calibration; scores are never
averaged across dimensions or scenes. The 40-chunk fixed population owns its
reference ceilings: six-reference per-scene ranges receive the existing 25%
reserve, rounded up to 0.005, retaining any larger global ceiling. The resulting
nightly subject-consistency ceiling is 0.065 and WBench HPSv3 is 0.1. These are
reference-stability limits; candidate tolerances do not change. Fresh reference
spread checks and fixed-population comparisons use the same profile-owned
ceilings. Weekly and NPU policies retain their separate global limits.
The candidate floor remains the worst reference minus its unchanged tolerance.

Both original nightly gates remain failed. Before fitting the new population,
two-way reference-only controls fail 8/36 native continuous checks and 4/18
pixel checks; all 12 binary controls pass. The new six-reference construction
passes 54 fitted checks and rejects 87 individual native/pixel boundary
perturbations. These checks are not independent acceptance: a fresh final-source
H100 campaign must pass the gates and retain its complete evidence.

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

- READY: a dedicated CPU unit gate and a one-GPU real-checkpoint
  stream/cancellation smoke. The unit job uses the standard L4 CI image with
  accelerators masked, and retains JUnit, logs, the quality policy and package
  versions. Selected skips fail through the same pytest hook as model jobs.
- NIGHTLY: independent CUDA/NPU generation followed by one-GPU evaluation.
  The NPU build provisions its matching CUDA evaluator image in the same build.
- WEEKLY/manual (`weekly-test`): three scenes, three camera patterns and two
  seeds (18 cases), 160 chunks each across the groups above, plus the complete
  600-chunk long profile. NPU long quality waits for all three collected pairs
  and their CPU performance/precision aggregation gate.

Selected jobs are blocking. Missing prerequisites and skipped CI tests fail.
NPU generation uses the existing infrastructure retry codes (`-1`, `128`),
with one retry per code. Pytest failures and OOM are never automatically retried.
JUnit, raw chunks, measured limits, precision captures, memory/events, videos,
scores and logs are retained through `artifact_paths`, including failed jobs.
A failed placeholder JUnit report is written before provisioning so a forced
SIGKILL still leaves failure evidence. Pytest replaces it on completion; normal
failure and SIGTERM also retain structured launcher errors.
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
    bash .buildkite/common/scripts/run-lingbot-ci.sh collect \
    tests/dfx/lingbot_world/configs/npu_weekly.json \
    "lingbot-artifacts/npu-weekly-measurements/shard-0/pair-$repeat"
done
LINGBOT_CASE_SHARD=0/6 bash .buildkite/common/scripts/run-lingbot-ci.sh aggregate \
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
bash .buildkite/common/scripts/run-lingbot-ci.sh aggregate \
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
The preliminary CUDA record does not establish NPU or weekly/long acceptance.

The CUDA 40-chunk quality profile also retains a fixed reference population in
`calibration/cuda_nightly_quality_population.json`: six reference starts from
two target-H100 cohorts, native continuous scores and six decoded reference-pair
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
own reference comparisons. The active nightly record binds both recorded-zero
cohorts and freshly recomputed pixel summaries from their actual reference
videos, using NumPy 1.26.4, OpenCV 4.11.0.86 and scikit-image 0.25.2; the OpenCV
build hash is retained. No local interpreter paths are required by the record.
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
Scene/seed cases and rollout lengths remain held-out inputs under the same
runtime and playback/warmup policy.
Fixed quality populations also require every reference pixel summary to record
`pixel_environment.packages` matching the verified evaluator's NumPy, OpenCV and
scikit-image versions. Missing or changed decoder metadata rejects reuse.
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
