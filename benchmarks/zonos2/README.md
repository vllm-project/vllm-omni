# ZONOS2 P6 measurements

This directory contains evaluation/benchmark drivers. It does not enable new
production compile, CUDA graph or scheduling modes. Use an otherwise idle
single GPU, local frozen model assets and an owned output directory.

## Protocol

`protocol.py` fixes ten texts: three English, three Chinese, two English
number/date cases and two cloning cases. `prepare.py` runs NeMo once and
saves the exact base/full prompt frames and reference embedding. Both
backends consume the same bundles. Official scheduler speaker markers
are supplied through its normal speaker interface.

Effective generation parameters are T=1.15, k=106, top-p disabled,
min-p=.18, repetition=1.2/window50/CB0–7, seed42, max1024, CFG1.
Native top-p=1 and official top-p=0 both mean disabled. Their RNG
implementations differ; this comparison does not promise equal random draws.

Each performance operating point has two single-request warmups and three
repeats of the fixed set, giving 30 measured requests. C1, C4 and C8 remain
separate. The last partial wave is retained. Initialization/frontend time is
excluded. RTF is latency/audio duration; throughput reports requests/s,
audio seconds/s and raw generated frames/s. Memory is sampled by nvidia-smi
every 0.5s and includes weights, KV reserve and codec/runtime allocations.

TTFC timestamps a CPU-ready code frame. TTFP uses CPU-ready PCM at the
decoder; native consumer timestamps are retained separately. Native includes
the production process/IPC path. Official uses its frozen offline scheduler
and public streaming vocoder as an observer, not an official HTTP deployment.
Driver boundaries and framework/library differences are recorded; timings
are not a kernel-only backend comparison.

P50/P95 confidence intervals use 2000 request-level bootstrap draws, seed42.
They describe this small fixed corpus, not an independent production workload
or SLA. CUDA event intervals include launch/submission gaps. Profiler kernel
busy intervals are unioned separately to avoid double counting CPU operation
GPU attribution. Profile runs are excluded from performance summaries.

## Reproduce

Install optional metric packages from `requirements.txt`, with torch/torchaudio
matching the native environment. Set the local asset environment variables
documented by the ZONOS2 model README. Use the frozen official checkout and
its own validated environment for the reference backend.

```bash
export CUDA_VISIBLE_DEVICES=2
export VLLM_ZONOS2_DAC_PATH=/owned/assets/weights_44khz_8kbps_0.0.1.pth
export VLLM_ZONOS2_SPEAKER_PATH=/owned/assets/qwen3-speaker-snapshot
export VLLM_ZONOS2_TN_CACHE_DIR=/owned/tn-cache
export HF_MODULES_CACHE=/owned/eval-assets/hf-modules

# Explicit one-time evaluator preparation; verifies pinned content hashes.
python -m benchmarks.zonos2.assets --out /owned/eval-assets
CUDA_VISIBLE_DEVICES='' python -m benchmarks.zonos2.prepare \
  --model /owned/zonos2-safetensors --reference tests/assets/qwen3_tts/clone_2.wav \
  --out /owned/p6/inputs

# Repeat independently for C=1,4,8; controller refuses a busy/multi-visible GPU.
python -m benchmarks.zonos2.run --gpu 2 --out /owned/p6/native_c1 -- \
  python -m benchmarks.zonos2.native --model /owned/zonos2-safetensors \
  --inputs /owned/p6/inputs --out /owned/p6/native_c1 --concurrency 1 --rounds 3

# Add the frozen official python directory to PYTHONPATH. Same bundles/knobs.
PYTHONPATH=.:/owned/ZONOS2-official/python python -m benchmarks.zonos2.run \
  --gpu 2 --out /owned/p6/official_c1 -- \
  /owned/ZONOS2-official/.venv/bin/python -m benchmarks.zonos2.official \
  --model /owned/original-hf-snapshot --dac "$VLLM_ZONOS2_DAC_PATH" \
  --inputs /owned/p6/inputs --out /owned/p6/official_c1 --concurrency 1 --rounds 3

python -m benchmarks.zonos2.run --gpu 2 --out /owned/p6/quality_job -- \
  python -m benchmarks.zonos2.quality --inputs /owned/p6/inputs \
  --assets /owned/eval-assets --runs /owned/p6/native_c1 /owned/p6/official_c1 \
  --out /owned/p6/quality_baseline.json

python -m benchmarks.zonos2.summarize /owned/p6/native_c1 /owned/p6/official_c1 \
  --out /owned/p6/performance.json
```

The native driver accepts `--deploy-config vllm_omni/deploy/zonos2.yaml`
to qualify the shipped B1 profile. Without this option it retains the P6
benchmark stage overrides (`max_num_seqs=8`, 128-token chunked prefill).
The effective configuration is saved in `result.json`; do not mix profiles
when comparing timings.

Quality uses local OpenAI Whisper large-v3 (T0, beam5) for English WER and
Chinese CER. ASR output and spoken truth use NeMo TN, NFKC/punctuation rules;
Chinese hypotheses are converted to simplified Chinese. This is not the
Seed-TTS paraformer-ZH protocol. Speaker cosine uses the native Qwen3 2048D
encoder on the two cloning cases; it is a proxy, not an independent WavLM SV
score. UTMOS uses pinned balacoon TorchScript on 16kHz float32.
Per-sample hypotheses, edit counts, duration, RMS, cap flags and failures
are retained. Thresholds are fixed before scoring (WER/CER .15, cosine .5,
UTMOS 3, duration .25–15s); failure samples are not removed from aggregates.

## Profile and independent A/B

Use `run --profile` with `native --profile --rounds 1` to collect a warmed
32-forward AR trace and four actual DAC calls. `profile_report.py` extracts
kernel counts, busy intervals and launch/CPU costs. Export time and profiler
overhead invalidate trace-run latency, so they are never mixed into baselines.

For full pipeline streaming A/B, run native C1 again with only `--sync`
changed, then score its round-zero audio. This toggles the existing handoff
and incremental DAC transport, not asynchronous AR scheduling.

For isolated DAC A/B, run `dac_ab.py --variant eager`, `compile`, `graph`
separately against `native_c1`. It replays the actual saved codes with
unchanged shear/EOS/OLA and saves ten WAVs per variant. Warmup/capture costs
are recorded separately, and 100 timed calls are measured at 16/20 frames.
It performs no sampler substitution. Numerical gates require maximum wave
error <=1e-4 and SNR >=60dB. Rerun `quality.py` over each result directory.

`gates.py` records speed, numerical and quality noninferiority decisions:
WER increase <=.02, CER increase <=.01, UTMOS drop <=.05, cosine drop <=.02.
Successful component timing does not establish an end-to-end AR speedup.
`feasibility.py` records that the existing full-runtime AR graph guard rejects
non-eager mode; the benchmark never bypasses that request-lifecycle guard.
No candidate is silently replaced with eager fallback when it fails.

Outputs include original logs, controller/contended status, GPU samples,
per-request timings/codes/WAVs, profile traces, quality failures and CI/gates.
The controller only signals the process group it creates. A job that becomes
GPU-contended is excluded. CPU metric tests live in
`tests/benchmarks/test_zonos2_protocol.py`. P7 documentation/PR work is separate.
