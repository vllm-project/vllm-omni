# MOSS-TTS-Realtime on RTX 5060 Ti 16GB

The stock profile ran out of memory on this card. This profile uses a 1 GiB
Talker KV cache and depends on the codec residency fix in
[#10345](https://github.com/vllm-project/vllm-omni/pull/10345), with the
reference-input device fix now present upstream. It completes synthesis, but
concurrent requests have substantial first-audio delays and some streams fall
behind playback.

## Supported model contract

| Item | Tested contract |
| --- | --- |
| Endpoint | `POST /v1/audio/speech` |
| Input | Complete English or Chinese text and a fixed synthetic reference WAV |
| Output | Streaming PCM, 24 kHz, mono, signed 16-bit |
| Profile | Both stages on GPU 0; stage request capacities 8 / 1; HTTP concurrency 1 / 2 / 4 |
| Precision | BF16 Talker, FP32 codec; no quantization or CPU offload |
| Generation | 15-frame codec chunks; 4,096 context; 2,048 new-frame cap; sampling seed 42 |

The measurements cover full-text input with streaming audio output. They do
not qualify native incremental text input, full-duplex conversation, other
GPU models, WSL2, or a minimum viable memory tier. Sleep/wake is disabled and
was not measured.

## Hardware and software

One desktop GPU, 16,311MiB; native Linux, 28GiB container RAM limit.

| Component | Measured version |
| --- | --- |
| GPU architecture / driver | SM120 / 580.142 |
| OS / CUDA | Ubuntu 24.04.4 / 13.0 |
| Python / PyTorch | 3.12.13 / 2.14.0+cu130 |
| vLLM / Transformers | 0.33.0 / 5.19.0 |
| TorchAudio / TorchVision | 2.11.0+cu130 / 0.29.0+cu130 |
| Omni source | `0c6328ed4f5512cb1bf48d5bcba1f7634e47cbe7` plus the frozen evidence patch |

## Checkpoints and deployment

Apply #10345 before using this configuration until that prerequisite is merged.
The results below belong to the pinned source and evidence patch; GPU inference
has not been rerun on the current rebased PR head.

Run from the repository root in a matching vLLM environment. Pin the Talker
and codec so the API reference encoder and worker use identical weights:

```bash
export MOSS_A7_MODELS="$PWD/moss-a7-models"
hf download OpenMOSS-Team/MOSS-TTS-Realtime \
  --revision 75682787d8e2fcc73faca37ba2931453ca9c4022 \
  --local-dir "$MOSS_A7_MODELS/MOSS-TTS-Realtime"
hf download OpenMOSS-Team/MOSS-Audio-Tokenizer \
  --revision 3cd226ba2947efa357ef453bcad111b6eafba782 \
  --local-dir "$MOSS_A7_MODELS/codec"

python - <<'PY'
import json
import os
from pathlib import Path

import yaml

models = Path(os.environ["MOSS_A7_MODELS"]).resolve()
path = models / "MOSS-TTS-Realtime/config.json"
config = json.loads(path.read_text())
config["codec_model_name_or_path"] = str(models / "codec")
path.write_text(json.dumps(config, indent=2) + "\n")
deploy = yaml.safe_load(Path("vllm_omni/deploy/moss_tts_realtime.yaml").read_text())
talker = next(stage for stage in deploy["stages"] if stage["stage_id"] == 0)
talker["kv_cache_memory_bytes"] = 1073741824
Path("moss-realtime-16gb.yaml").write_text(yaml.safe_dump(deploy, sort_keys=False))
PY

vllm serve "$MOSS_A7_MODELS/MOSS-TTS-Realtime" --omni \
  --served-model-name OpenMOSS-Team/MOSS-TTS-Realtime \
  --deploy-config moss-realtime-16gb.yaml \
  --host 127.0.0.1 --port 18081
```

Other settings match `moss_tts_realtime.yaml`. KV reduction alone still OOMed
at API reference-codec loading. Stage 1 validates the full checkpoint before
dropping its encoder, so startup peak remains relevant.

## Qualification results

Three fresh server processes ran the same 20 English/Chinese texts at each
concurrency. The concurrency orders were 1/2/4, 2/4/1 and 4/1/2. Each language
batch alternated short and long inputs. Before timing, each concurrency limit
was exercised with two requests per language; those separate calls reached
at most two in-flight requests, including the c4 warmup. The Talker captured
its CUDA graphs during startup, and inference JIT warnings occurred before
the timed batches.

| HTTP concurrency | Completed | First PCM read p50 / p95 (s) | EOF RTF p50 / p95 | Transport interval p95 (s) | Within 100ms playback deficit |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 60/60 | 1.421 / 1.469 | 1.068 / 1.117 | 1.275 | 31/60 |
| 2 | 60/60 | 4.012 / 22.632 | 1.661 / 7.149 | 2.267 | 24/60 |
| 4 | 60/60 | 32.573 / 45.136 | 3.249 / 13.688 | 3.407 | 40/60 |

Each server contributed 20 requests per concurrency. Across the three servers,
first-audio p95 ranged from c1 1.447–1.482s, c2 21.736–23.259s, c4 38.809–45.756s.
These ranges describe per-round variation; the table pools all 60 requests.

First audio is the first nonempty client read, not the first audible sample.
HTTP reads can split/merge codec chunks. EOF RTF is request-to-EOF time / PCM
duration; newer benchmarks use the last audio payload, so recompute this
boundary from `audio_stream_details` when comparing. Continuity starts at the
first read with no added buffer; late bursts can pass despite high latency.

With retained compiler caches, health was observed after 76.29–77.22s; the
first API requests then took 5.13–5.30s to first PCM. A prior
successful startup had a 53.08s first request. These are separate observations,
not a controlled cold-cache comparison or wake latency.

Extract `inputs-v1` from the evidence bundle and set `MOSS_A7_INPUTS` to it.
Its manifest records 20 test texts, two synthetic VoxCPM2 reference WAVs,
selection and hashes; this is a small reproducibility corpus. Run per
locale/concurrency with separate result files. Warmups use two prompts.

```bash
vllm bench serve --omni --host 127.0.0.1 --port 18081 \
  --model OpenMOSS-Team/MOSS-TTS-Realtime \
  --tokenizer "$MOSS_A7_MODELS/MOSS-TTS-Realtime" \
  --backend openai-audio-speech --endpoint /v1/audio/speech \
  --dataset-name seed-tts --dataset-path "$MOSS_A7_INPUTS" \
  --seed-tts-locale en --disable-shuffle --seed 42 \
  --num-prompts 10 --num-warmups 0 --output-len 2048 \
  --ready-check-timeout-sec 0 --max-concurrency 1 --request-rate inf \
  --omni-request-timeout-s 120 --percentile-metrics audio_ttfp,e2el,audio_rtf \
  --metric-percentiles 50,95,99 --save-result --save-detailed \
  --result-dir results --result-filename timing-en-c1.json
```

## Memory and interruption

Independent batches used allocator reset/read windows, with worker PIDs
matched to stage logs. Allocated/reserved values below are peaks in GiB;
device values are sampled MiB and include the API reference encoder.

| HTTP concurrency | Talker allocated / reserved | Codec allocated / reserved | Device peak (MiB) |
| --- | ---: | ---: | ---: |
| 1 | 5.516 / 5.650 | 3.837 / 3.949 | 14,655.4 |
| 2 | 5.516 / 5.650 | 3.837 / 3.949 | 14,655.4 |
| 4 | 5.529 / 5.650 | 3.837 / 3.949 | 14,655.4 |

Reserved includes allocated; independent peaks must not be summed. API NVML
residency peaked at 3,854MiB; its allocator was not instrumented. NVML intervals
were median 101.10ms, max 810.84ms, so brief peaks may be missed. These batches
are separate from timing. Allocator collection used the
[pinned external helper](https://github.com/RongHuang14/vllm-omni/blob/a1-voxcpm2-16gb-evidence-v1/benchmarks/worker_memory.py),
SHA256 `10e1c9064960a2b9cd60d91aa91b40eefc89cd788ad2e8b8e53b48d546bb270d`;
serving does not depend on it.

A long-request disconnect triggered server abort; a following c1 probe
completed. This checks HTTP cancellation/recovery, not native barge-in.

## Quality and verification

An independent c4 run saved all 20 outputs. After serving exited, the existing
Seed-TTS evaluator scored EN with Whisper-large-v3 and ZH with Paraformer:

| Language | Samples | Corpus content error | Mean WavLM proxy |
| --- | ---: | ---: | ---: |
| EN | 10 | WER 0/296 words, 0.00% | 0.910 |
| ZH | 10 | CER 2/487 characters, 0.41% | 0.903 |

ZH differences were homophones (得/的, 它/他). The author heard six EN/ZH
samples with no listed content/noise/voice issue. This is nonblind, not MOS;
WavLM is an embedding proxy, not speaker verification. File playback does not
reproduce transport stalls. Real-checkpoint reference codes and 15-frame PCM
matched exactly before/after codec module removal. The stock OOM profile
produced no audio for a relative ASR comparison.

## Sustained workload

One independent server ran six windows of 26 EN requests (13 short/13 long
from ten texts), ordered c1/c2/c2/c1/c1/c2. Import gaps are excluded.

| HTTP concurrency | Completed | Window duration range (s) | Audio s / wall s range | Within 100ms deficit |
| --- | ---: | ---: | ---: | ---: |
| 1 | 78/78 | 250.75–256.55 | 0.945–0.957 | 40/78 |
| 2 | 78/78 | 242.70–247.00 | 0.965–0.984 | 36/78 |

All windows completed without OOM/HTTP failure, but generation stayed below
one audio second per wall second; this does not establish continuous realtime.

## Configuration screens

Later control-first screens used another RTX 5060 Ti 16GB instance with the
same source/model, fresh servers, four warmups and eight timed EN short/long
requests at each of c1/c4. These small samples are separate from the baseline.

| Change | Observation | Decision |
| --- | --- | --- |
| Codec chunk frames 15 → 5 | c1 first-audio p50 1.412 → 0.603s; c4 p50 10.698 → 30.907s and throughput 1.051 → 0.983 audio-s/wall-s | Keep 15 frames |
| Stage 1 capacity 1 → 2 | c4 first-audio p50 30.013 → 6.729s; 100ms deficit passes 5/8 → 2/8 | Keep capacity 1 |
| Stage 1 CUDA Graph enabled | Streaming graph captured; API reference-codec GPU loading then OOM, 0/4 warmup requests completed | Keep eager codec |

No candidate replaced the recipe or entered repeated/quality qualification.
Control c4 medians varied; no repeatable speedup is claimed.

## Evidence

The [versioned release](https://github.com/RongHuang14/vllm-omni/releases/tag/a7-moss-realtime-16gb-evidence-v1)
contains the baseline inputs, raw results, measured source patch, quality WAVs
and separate tuning/failure logs. Start with `REPRODUCE.md` and `RESULTS.md`
in the baseline bundle, or `TUNING.md` in the tuning bundle. Download both
archives and `SHA256SUMS`; run `shasum -a 256 -c SHA256SUMS` before extraction.
The bundled patch reconstructs the measured revision and its EOF RTF boundary.
