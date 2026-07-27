# MiniCPM-o 4.5 Ascend Competition Suite

This directory provides the reproducible local proxy suite for the Ascend
high-performance inference track. It does not claim to implement the official
score while the organizer's starter kit, request schema, metric definitions,
hardware image, and pass thresholds remain unpublished.

## 1. Resolve the environment

Update `environment_manifest.yaml` from the newest official announcement.
Every `UNRESOLVED` value must be resolved before a formal run. Capture the
actual machine separately:

```bash
.venv/bin/python -m benchmarks.competition.minicpmo_ascend.collect_environment \
  --output artifacts/minicpmo_ascend/environment.json \
  --starter-kit /path/to/starter-kit.tar.gz \
  --model-path /path/to/MiniCPM-o-4_5 \
  --model-manifest /path/to/model-sha256.txt
```

The collector records the Git SHA and dirty diff, package versions, CANN
version files, physical-card and logical-chip inventory from `npu-smi`, and
checksums of supplied manifests/artifacts. It deliberately does not hash a
multi-gigabyte model tree implicitly. When the model directory has no revision
metadata, create a relative-path manifest explicitly:

```bash
cd /path/to/MiniCPM-o-4_5
find . -type f -print0 | sort -z | xargs -0 sha256sum \
  > /path/to/model-sha256.txt
```

## 2. Start the server

```bash
MODEL=/path/to/MiniCPM-o-4_5 \
MODEL_REVISION=<fixed-revision> \
bash benchmarks/competition/minicpmo_ascend/start_server.sh
```

The default deployment is
`vllm_omni/deploy/minicpmo_4_5_ascend_910c_1card.yaml`. It uses both logical
chips exposed by one physical Ascend 910C card; device IDs `0` and `1` do not
mean two cards. Override `DEPLOY_CONFIG` only with a checked-in candidate
configuration.

## 3. Run the gated proxy suite

Provide a deterministic local video. Image and audio fixtures are generated
offline by the suite.

```bash
VIDEO_INPUT=/path/to/official-or-local-fixture.mp4 \
MODEL=/path/to/MiniCPM-o-4_5 \
MODEL_PATH=/path/to/MiniCPM-o-4_5 \
MODEL_MANIFEST=/path/to/model-sha256.txt \
bash benchmarks/competition/minicpmo_ascend/run_suite.sh \
  --concurrency 1 2 4 --num-requests 20 --warmups 2
```

The command runs multimodal smoke validation first, then separate text-only
and text-plus-audio benchmarks, a text-plus-audio stability run, raw NPU/host
resource collection, the machine-readable correctness gate, and a baseline
report with an artifact checksum manifest. A failed, timed-out, truncated,
empty, or invalid-audio request is excluded from metrics and makes the command
fail.

Raw per-request output includes first SSE event, first text, first audio,
audio chunk arrival/inter-chunk times, E2E, finish reasons, error details, WAV
format, chunk hashes, and reconstructed audio. Results are labeled
`local_proxy`; no unofficial composite score is emitted.

## 4. Run the Daily-Omni proxy effect check

```bash
DAILY_OMNI_QA_JSON=/data/Daily-Omni/qa.json \
DAILY_OMNI_VIDEO_DIR=/data/Daily-Omni/videos \
bash benchmarks/competition/minicpmo_ascend/run_daily_omni.sh
```

This requests text only and disables thinking so A-D answer extraction is
stable. Replace it with the official effect suite as soon as one is released.

## 5. Capture an NPU profile

Keep profiling separate from score measurements. The profile runner generates
a temporary deploy config, starts a clean server, warms it outside the capture
window, profiles a fixed request, stops the server, and emits a unified JSON
and Markdown summary:

```bash
MODEL=/path/to/MiniCPM-o-4_5 \
PROFILE_ID=stage2-baseline \
PROFILE_STAGES=2 \
bash benchmarks/competition/minicpmo_ascend/run_profile.sh
```

Use a unique `PROFILE_ID` for every capture; the runner refuses to mix a new
capture into a non-empty artifact directory.

Profile all stages only when stage ownership is unclear:

```bash
PROFILE_ID=all-stages-baseline PROFILE_STAGES=0,1,2 \
bash benchmarks/competition/minicpmo_ascend/run_profile.sh
```

Artifacts are written under
`artifacts/minicpmo_ascend/profiles/<profile-id>/`. Compare the same workload,
stage selection, profiler configuration, and environment with:

```bash
.venv/bin/python -m benchmarks.competition.minicpmo_ascend.profile_analysis compare \
  artifacts/minicpmo_ascend/profiles/baseline/profile_analysis.json \
  artifacts/minicpmo_ascend/profiles/candidate/profile_analysis.json \
  --output artifacts/minicpmo_ascend/profiles/comparison.json
```

Profiler timing is diagnostic evidence and must never replace the unprofiled
baseline/candidate benchmark.

## Formal-run rules

- Use clean server restarts and record warmup separately from measurements.
- Keep profiler runs separate from score runs.
- Never compare text-only and text-plus-audio as the same workload.
- Preserve raw JSON, server logs, resource samples, WAV artifacts, exact
  commands, Git diff, model manifest, and starter-kit checksum.
- Do not report a formal score until the official script and definitions are
  checked in and `UNRESOLVED` manifest fields are resolved.
