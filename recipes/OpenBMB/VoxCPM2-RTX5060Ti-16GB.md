# VoxCPM2 on RTX 5060 Ti 16GB

A measured [A1](https://github.com/vllm-project/vllm-omni/issues/10008) baseline
for BF16 synthesis and reference-audio cloning on one native-Linux card.
Both the 6 GiB reference KV pool and optional 2 GiB pool fit these workloads.
The smaller pool leaves more headroom at the cost of token capacity.

## Environment and profiles

| Component | Measured value |
| --- | --- |
| GPU / host limits | RTX 5060 Ti, SM120, 16,311 MiB; 11 CPUs / 29 GiB RAM |
| OS / driver | Ubuntu 24.04.4 LTS / 580.142 |
| Omni / vLLM | `7467d4b59f4290334aae8f84777e989e1cb1bbeb` / 0.33.0 (wheel `4ba787ec9959e566d3b587a796dff07673324da5`) |
| Model | `openbmb/VoxCPM2` revision `7936080b41163874f38610f46f438e2c0757f9fd` |
| Runtime | Python 3.12.13; Torch 2.14.0+cu130; loaded cuDNN 9.25.1.1 |
| Other libraries | VoxCPM 2.0.3; Transformers 5.19.0; FlashInfer 0.7.2.post1; JIT NVCC 12.8.93 |
| Generation | BF16, 10 CFM steps, CFG 2, seed 42; VAE decode / audio emit cadence 2/1 |

[A: reference](VoxCPM2-RTX5060Ti-16GB-reference.yaml) reserves 6 GiB KV;
[B: smaller pool](VoxCPM2-RTX5060Ti-16GB.yaml) reserves 2 GiB. Both inherit
admission 8, engine eager mode and model-owned unified graphs. Both disable
legacy batched VAE to avoid the concurrent-speech conflict in
[#9971](https://github.com/vllm-project/vllm-omni/pull/9971).
The two YAMLs differ only in KV budget. No quantization or offload was used.

## Reproduce

Install the [matching runtime](../../docs/getting_started/installation/README.md)
and pinned model snapshot. Copy the two YAMLs into the same paths in the
pinned source checkout; copy `benchmarks/worker_memory.py` for diagnostics.
These are measurements of that revision, not current-main performance.
The host preloaded cuDNN 9.25.1.1 separately from its installed 9.24 metadata:

~~~bash
uv pip install --no-deps --target /data/voxcpm2-a1-16g/omni-cudnn-runtime nvidia-cudnn-cu13==9.25.1.1
export LD_PRELOAD=/data/voxcpm2-a1-16g/omni-cudnn-runtime/nvidia/cudnn/lib/libcudnn.so.9
export LD_LIBRARY_PATH=/data/voxcpm2-a1-16g/omni-cudnn-runtime/nvidia/cudnn/lib
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MAX_JOBS=2 TORCHINDUCTOR_COMPILE_THREADS=2
vllm serve /data/voxcpm2-a1-16g/models/VoxCPM2 --omni \
  --served-model-name voxcpm2 --host 127.0.0.1 --port 18080 \
  --deploy-config recipes/OpenBMB/VoxCPM2-RTX5060Ti-16GB.yaml
~~~

Adapt local paths and preserve other required library paths. Use the reference
YAML for A. After `/health`, verify a 48 kHz mono PCM16 WAV through the
[shared speech API](../../docs/serving/speech_api.md#voxcpm2), including cloning.

~~~bash
VLLM_OMNI_BENCH_AUDIO_SAMPLE_RATE=48000 \
python benchmarks/tts/bench_tts.py \
  --model openbmb/VoxCPM2 --served-model-name voxcpm2 --task default_voice \
  --dataset-path benchmarks/build_dataset/seed_tts_smoke --locale en \
  --host 127.0.0.1 --port 18080 --concurrency 1 2 4 \
  --num-prompts 30 --num-warmups 2 --request-seed 42 \
  --metric-percentiles 50,95,99 --output-dir /data/voxcpm2-measured \
  -- --tokenizer /data/voxcpm2-a1-16g/models/VoxCPM2 --seed 0 --save-detailed --disable-tqdm
~~~

First run n=8 in a separate output directory for shape prewarm. The KV study
used three AB/BA/AB rounds, a fresh service per profile/round and retained disk
caches: 540 measured requests, 144 explicit prewarm and 54 manual WAVs.
CLI readiness checks and warmups are additional. Request order and executed
scripts are in the [evidence package](https://github.com/RongHuang14/vllm-omni/releases/download/a1-voxcpm2-16gb-evidence-v1/voxcpm2-a1-16gb-evidence-v1.zip).

## Memory and performance

Device memory was sampled every 200 ms across startup, serving and shutdown.
All three rounds peaked at **12,465 MiB (A)** / **9,478 MiB (B)**: a 2,987 MiB
reduction, about 24%. B peaked before observed health and A afterward; the
difference between these maxima is not the KV allocation difference.

| KV | Clients | Per-round p95 first audio | Per-round p95 RTF | Audio seconds / second |
| --- | ---: | ---: | ---: | ---: |
| 6 GiB | 1 | 124.1–128.0 ms | 0.196–0.204 | 5.182–5.278 |
| 6 GiB | 2 | 346.9–388.2 ms | 0.272–0.290 | 8.206–8.594 |
| 6 GiB | 4 | 477.4–490.7 ms | 0.334–0.338 | 13.063–13.137 |
| 2 GiB | 1 | 120.0–126.2 ms | 0.193–0.197 | 5.231–5.388 |
| 2 GiB | 2 | 374.7–381.1 ms | 0.284–0.285 | 8.262–8.385 |
| 2 GiB | 4 | 477.2–495.8 ms | 0.327–0.346 | 12.992–13.488 |

Ranges cover three n=30 percentiles, not pooled n=90. RTF is request elapsed
time / audio duration. No statistically established speedup is claimed.

Actual-worker storage accounting ran separately from latency. At the drained
post-benchmark boundary, B had 4,547.471 MiB parameters, 316.003 MiB registered
buffers (including fused weights), 2,047.500 MiB KV backing storage and
0.289 MiB selected audio-path tensors, deduplicated across storage aliases.
Another 425.574 MiB allocated was unassigned: opaque graph pools, temporary
CFM/VAE workspaces, unselected tensors and allocator rounding are not split.
Current allocated / reserved was 7,336.837 / 7,834 MiB. A separate mixed-load
interval peaked at 7.434 / 8.008 GiB allocated / reserved. Reserved includes
allocated; full-startup allocator peaks remain unmeasured.

## Admission and unified-graph comparison

The 2 GiB comparison used B (admission/capture ceiling 8, unified on), C
(4, on), and D (4, off). C changes `max_num_seqs` and
`unified_decode_graph_max_batch_size` to 4; D additionally sets
`enable_unified_decode_graph: false`. Other graph settings stay shared.
Each fresh profile process ran n=8 prewarm, then two n=20 English short/long
sweeps at clients 1/2/4; the second reversed input/concurrency order.
The two sweeps share a process. All 720 measured requests completed.

| Profile / input, c4 | Per-sweep p95 first audio | Per-sweep p95 RTF | Audio seconds / second |
| --- | ---: | ---: | ---: |
| B / short | 201.4–208.3 ms | 0.273–0.280 | 14.645–14.915 |
| B / long | 204.5–205.7 ms | 0.245–0.249 | 16.216–16.538 |
| C / short | 202.9–204.4 ms | 0.275–0.276 | 15.003–15.144 |
| C / long | 208.2–212.8 ms | 0.247–0.253 | 15.966–16.331 |
| D / short | 223.2–227.6 ms | 0.277–0.280 | 14.512–14.550 |
| D / long | 217.7–219.0 ms | 0.254–0.258 | 15.749–15.864 |

The [unrounded records](https://github.com/RongHuang14/vllm-omni/releases/download/a1-voxcpm2-16gb-evidence-v1/voxcpm2_rtx5060ti_16gb_reference.json)
also contain c1/2 cells, inputs, transcripts, artifact pins and source hashes.
Fixtures and full commands are in the package. B→C lowered admission and the
graph capture ceiling together from 8 to 4; current allocated fell by ~1.2 MiB.
All profiles peaked at 9,478 MiB whole-device; the optional YAML keeps admission 8 and unified graphs on.
Clients never exceeded 4, so this does not characterize admission queuing.
Fresh cached processes reached health in 66.1–69.1 s; first external audio
was 2.255–2.362 s. Health includes built-in speech warmup. The first diagnostic
startup after VM restart took 439.3 s while populating caches; no controlled
cache eviction was performed. These observations are separate from steady state.

## Quality and limits

The original c1/c4 KV cohorts produced 320 outputs; the B/C/D c4 cohort added
240. Each profile/concurrency cohort used 30 default-voice and 10 cloning inputs per language,
using two fixed synthetic references. Paired gates were declared before
synthesis: EN WER / ZH CER increase ≤1 percentage point, mean WavLM-proxy drop
≤0.02, individual drops >0.05 investigated. All passed. WavLM is a proxy, not
a trained speaker verifier. The records retain every transcript/edit count.
The author heard six original KV pairs; both English-clone files had background
noise and were byte-identical across KV budgets. Source unknown; no other
issues reported. This non-blind check does not cover changed new-cohort PCM.

- A separate c4 mixed stream completed 991 measured requests over 904.81 s.
  Its p95 first audio was **2.24–3.60 s**, with short-task p95 RTF >1: the
  study's interactive targets failed. All 54 captured clips had zero ASR error;
  remaining outputs were not individually ASR-evaluated.
- Three idle level-1 sleep/wake cycles matched prewarm PCM, reclaimed
  7,318–7,328 MiB and took 1.409–1.415 s wake-to-first-audio. Backend peaks
  decreased across sleep, invalidating that full-cycle allocator interval.
- Earlier legacy-VAE concurrent speech omitted/repeated words despite HTTP
  success. Those timings are excluded; the existing configuration workaround
  above is used, without another engine fix.
- No saturated 4,096-token context, eight-client, 8/12GB, WSL2, other-GPU or
  production-duration qualification. See the [shared recipe](VoxCPM2.md) for
  the model contract beyond this measured hardware profile.
