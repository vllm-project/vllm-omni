# Soprano-1.1-80M text-to-speech — RTX 4090 Laptop 16GB

## Summary

- Vendor/model: `ekwek/Soprano-1.1-80M`, English fixed-voice TTS.
- Runtime: native vLLM Qwen3 AR/KV execution followed by the Soprano decoder;
  no Transformers generation or LMDeploy inside the workers.
- Modes: offline complete utterances; online WAV and incremental PCM/SSE.
- Hardware: one NVIDIA RTX 4090 **Laptop** GPU, 16GB; eager FP32, TP1.
- Maintainer: Community.

## Supported model contract

| Property | Qualified scope |
| --- | --- |
| Input | English text; no reference audio, voice cloning, voice design, speaker selection or Qwen3-TTS `task_type` |
| Voice | Checkpoint's single voice; omit `voice` or use `default` |
| Output | 32000 Hz mono; WAV, or signed 16-bit little-endian PCM when streaming |
| Text units | Official cleaning/splitting in the production frontend; 1–256 prompt tokens per unit, no truncation |
| Offline | Public `Omni` engine plus production `offline.synthesize`; splits and concatenates complete utterances in sentence order |
| Speech API | One synthesis unit per request; split multi-unit text before sending, retain order; no automatic request fanout |
| Generation | 1–512 new tokens per unit; default seed 44, temperature 0.3, top-p 0.95, top-k 50, repetition penalty 1.2 |
| Complete profile | `soprano.yaml`: two eager FP32 stages on GPU0, TP1, `max_num_seqs: 1` per stage |
| Streaming profile | `soprano_streaming.yaml`: same topology, async handoff, default chunk size 4 |

The original checkpoint identifies as ordinary Qwen3. **Always pass the explicit
deploy YAML**; a checkpoint path or `--served-model-name soprano` alone does not
select the Soprano pipeline. Do not edit `config.json` to imitate another model.

## References and attribution

- [Pinned model card](https://huggingface.co/ekwek/Soprano-1.1-80M/tree/27b5a5f5f541a1db3a51d6fd1b0fc7147b92cd01).
- [Original implementation](https://github.com/ekwek1/soprano/tree/12fac06eb8fa53bad8b3941d3cb11e9c869477c4)
  by the Soprano authors, Apache-2.0. Retained upstream attributions, including
  the text utilities and decoder sources, are in the vendored
  [NOTICE](../../vllm_omni/model_executor/models/soprano/NOTICE) and
  [LICENSE](../../vllm_omni/model_executor/models/soprano/LICENSE).
- Shared [offline TTS](../../examples/offline_inference/text_to_speech/README.md),
  [online TTS](../../examples/online_serving/text_to_speech/README.md),
  [Speech API](../../docs/serving/speech_api.md),
  [supported models](../../docs/models/supported_models.md), and
  [native implementation/testing notes](../../docs/models/soprano.md).

## Hardware and software

| Hardware | Local qualification |
| --- | --- |
| Accelerator | 1× RTX 4090 Laptop GPU, 16376 MiB reported; not a desktop 4090, H100 or L4 |
| Interconnect | Single device; no inter-device link or multi-GPU profile exercised |
| Placement | Both stages on device 0; no CPU weight offload; AR KV cache 64 MiB |
| Memory budget | `gpu_memory_utilization: 0.20` per stage on the same GPU; these budgets add, not separate cards |

| Software | Validated environment |
| --- | --- |
| OS | Ubuntu 26.04 under WSL2, Linux `6.18.33.2-microsoft-standard-WSL2` |
| Python | 3.12.15 |
| NVIDIA driver | 610.47 |
| PyTorch / runtime | 2.13.0+cu130 / CUDA 13.0; driver capability is not the PyTorch runtime version |
| vLLM / Transformers | 0.32.0 / 5.19.0 |
| Runtime implementation | Post-integration runtime validated at `888671bd744ac7e0d2482b306035b1259307712f`; the qualification-only documentation update preserves its runtime, benchmark and test trees |
| Checkpoint revision | `27b5a5f5f541a1db3a51d6fd1b0fc7147b92cd01` |

## Setup

Run from the repository root in the compatible vLLM-Omni environment, with
the `soprano` optional extra installed (`uv pip install -e '.[soprano]'`).
Obtain the unmodified pinned checkpoint, including tokenizer, AR weights and
`decoder.pth`. For a fresh download use `hf download ekwek/Soprano-1.1-80M
--revision 27b5a5f5f541a1db3a51d6fd1b0fc7147b92cd01 --local-dir <directory>`.
The commands below used an already cached snapshot; installation/download is
setup guidance, not a claim of another local model run.

<!-- soprano-command: setup -->
```bash
export SOPRANO_MODEL_DIR="$PWD/.local-soprano/models/Soprano-1.1-80M"
export SOPRANO_OUTPUT_DIR="$PWD/.local-soprano/artifacts/12-docs/output"
mkdir -p "$SOPRANO_OUTPUT_DIR"
export TRITON_F32_DEFAULT=ieee HF_HUB_OFFLINE=1
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=2
```

The FP32 native attention contract requires `TRITON_F32_DEFAULT=ieee` before
Python starts. The YAML selects the v1 runner and disables graphs, prefix cache
and chunked prefill. These settings do not establish cross-engine bitwise parity.

## Offline WAV

There is no model-neutral offline AR-TTS CLI in this revision. Reuse the public
`Omni` engine; model-specific prompt construction, defaults and output validation
remain in the production Soprano frontend. The diffusion-only `text_to_audio.py`
runner is not an AR-TTS entrypoint. No per-model Python example is required.

<!-- soprano-command: offline -->
```bash
python -c '
import os
import soundfile as sf
from transformers import AutoTokenizer
from vllm_omni import Omni
from vllm_omni.model_executor.models.soprano.offline import synthesize

checkpoint = os.environ["SOPRANO_MODEL_DIR"]
tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
engine = Omni(model=checkpoint, deploy_config="vllm_omni/deploy/soprano.yaml")
try:
    result = synthesize(engine, "Hello, this is a simple speech synthesis test.", tokenizer)
    assert result.sample_rate == 32000 and not result.truncated
    sf.write(os.environ["SOPRANO_OUTPUT_DIR"] + "/offline.wav",
             result.audio.numpy(), result.sample_rate, subtype="PCM_16")
finally:
    engine.close()
'
```

## Online WAV, raw PCM and SSE

Start the shared server in a separate terminal after applying the setup exports.
Use `soprano_streaming.yaml` for all three response modes below; `soprano.yaml`
is the complete-WAV-only alternative. `--served-model-name` sets the public API
alias, so each JSON `model` must match it.

<!-- soprano-command: serve -->
```bash
vllm serve "$SOPRANO_MODEL_DIR" --omni \
  --deploy-config vllm_omni/deploy/soprano_streaming.yaml \
  --served-model-name soprano --host 127.0.0.1 --port 8002
```

Wait for readiness; a successful TCP connection alone is insufficient.

<!-- soprano-command: health -->
```bash
curl --fail http://127.0.0.1:8002/health
curl --fail http://127.0.0.1:8002/v1/models
```

<!-- soprano-command: wav -->
```bash
curl --fail-with-body http://127.0.0.1:8002/v1/audio/speech \
  -H 'Content-Type: application/json' \
  --data-raw '{"model":"soprano","input":"Hello, this is a simple speech synthesis test.","voice":"default","seed":44,"response_format":"wav","max_new_tokens":512}' \
  --output "$SOPRANO_OUTPUT_DIR/online.wav"
```

<!-- soprano-command: pcm -->
```bash
curl --fail-with-body --no-buffer http://127.0.0.1:8002/v1/audio/speech \
  -H 'Content-Type: application/json' \
  --data-raw '{"model":"soprano","input":"Hello, this is a simple speech synthesis test.","voice":"default","seed":44,"response_format":"pcm","stream":true,"stream_format":"audio","max_new_tokens":512}' \
  --dump-header "$SOPRANO_OUTPUT_DIR/pcm.headers" \
  --output "$SOPRANO_OUTPUT_DIR/stream.pcm"
```

<!-- soprano-command: sse -->
```bash
curl --fail-with-body --no-buffer http://127.0.0.1:8002/v1/audio/speech \
  -H 'Content-Type: application/json' \
  --data-raw '{"model":"soprano","input":"Hello, this is a simple speech synthesis test.","voice":"default","seed":44,"response_format":"pcm","stream":true,"stream_format":"sse","max_new_tokens":512}' \
  --output "$SOPRANO_OUTPUT_DIR/stream.sse"
```

SSE is text, not an audio file: decode the base64 `audio` values from
`speech.audio.delta` events in order; require `speech.audio.done` and check for
error events. Raw PCM has no header: use `X-Audio-Sample-Rate: 32000` and
`X-Audio-Channels: 1`, not a player's 24 kHz default. WAV responses from the
streaming profile collect the same emitted deltas; they are not a different
full-history decoder. Numerical equality with `soprano.yaml` is not promised.

## Playback and verification

The following playback path was exercised on the named WSL2/Windows host. It
requires `ffmpeg` on PATH and Windows PowerShell; it does not qualify native
Windows inference. Other players must also use 32000 Hz, mono, PCM16.

<!-- soprano-command: pcm-to-wav -->
```bash
ffmpeg -hide_banner -y -f s16le -ar 32000 -ac 1 \
  -i "$SOPRANO_OUTPUT_DIR/stream.pcm" "$SOPRANO_OUTPUT_DIR/stream.wav"
```

<!-- soprano-command: play -->
```bash
powershell.exe -NoProfile -NonInteractive -Command \
  "(New-Object System.Media.SoundPlayer '$(wslpath -w "$SOPRANO_OUTPUT_DIR/stream.wav")').PlaySync()"
```

Playback invocation is not a human quality verdict. Before accepting outputs,
check HTTP status, WAV metadata, nonempty finite samples and truncation status.
The complete response reports length stops through `X-Audio-Truncated`;
`synthesize` returns `truncated`. A zero/one-predictor generation is an explicit
synthesis failure, not a successful empty WAV. Incremental failure can occur
after headers have been sent; check SSE termination/errors or raw transport
completion rather than relying only on the initial HTTP status.

## Benchmark

Use the [shared TTS benchmark](../../benchmarks/tts/README.md), registered task
`default_voice`, with its existing 20-item smoke dataset. Keep the server above
running. No reference audio, Qwen task type or other voice name is needed.

<!-- soprano-command: benchmark -->
```bash
VLLM_OMNI_BENCH_AUDIO_SAMPLE_RATE=32000 python benchmarks/tts/bench_tts.py \
  --model ekwek/Soprano-1.1-80M --served-model-name soprano \
  --host 127.0.0.1 --port 8002 --task default_voice --locale en \
  --dataset-path benchmarks/build_dataset/seed_tts_smoke \
  --concurrency 1 --num-prompts 20 --num-warmups 3 \
  --request-seed 44 --output-len 512 --metric-percentiles 50,95 \
  --output-dir "$SOPRANO_OUTPUT_DIR/benchmark" \
  -- --tokenizer "$SOPRANO_MODEL_DIR" --seed 0 --save-detailed
```

The response format headers override the benchmark's raw-PCM fallback.
At 32 kHz mono PCM16, 64000 bytes are one audio second. `--output-len` is a
dataset estimate, not an HTTP generation-limit override; this deploy fixes the
effective default cap at 512. For performance comparisons retain actual audio
duration, failures, complete tails, warmups, power and request settings. Score
saved audio after stopping services, avoiding ASR GPU contention.

## Supported features and qualification limits

| Feature | Status on this hardware |
| --- | --- |
| Native eager FP32, offline and Speech API | Locally executed with the pinned checkpoint |
| [Async chunk streaming](../../docs/design/feature/async_chunk.md) | PCM/SSE deltas, chunk 4; complete WAV also supported on this profile |
| Concurrent HTTP | Historical C1/C2/C4/C8 measurements; integrated runner rechecked for cancellation with a peer; still one sequence per stage, not GPU batch 8 |
| Continuous playback | Not guaranteed: the pre-integration matched-55W experiment had 4/3000 requests with >100 ms deficits, maximum 362 ms; not remeasured on the integrated runner |
| [Quantization](../../docs/user_guide/quantization/overview.md), BF16 AR | Unqualified; no INT8/INT4/FP8 or BF16 combination tested here |
| Multiple GPUs, TP>1, DP/PP, GPU batching>1 | Not qualified; no provided multi-device profile |
| CUDA graphs / speculative multi-token decoding | Streaming explicitly rejects these combinations; complete-response combinations are unqualified |
| Prefix caching / chunked prefill | Disabled; unqualified combinations |
| Other hardware/platforms | CPU-only inference, desktop 4090, L4/H100, AMD/ROCm, Ascend, Intel, macOS/MPS and native Windows not qualified by this recipe |
| Quality | Historical English ASR and manual-review checklist; fresh online content assertions; human listening/MOS pending, no cloning SIM acceptance |

The post-integration command replay at `888671bd744ac7e0d2482b306035b1259307712f` produced 98304 samples
(3.072 seconds) per fixed-text WAV at 32000 Hz mono. The streaming profile's
WAV, raw PCM and 12 SSE audio deltas were sample-identical; the SSE terminal
event was present. The documented benchmark smoke completed 20/20 requests,
zero failures, and 77.056 actual audio seconds. These are structural/functional
qualification results, not subjective quality or a speedup claim. Playback
returned exit code zero; human listening remains pending. Sample counts are
observations for these settings, not guaranteed across engines or precision modes.

The earlier, pre-integration 100-unit performance experiment separately disabled PyTorch matmul
and cuDNN TF32 in both real workers, and used three interleaved backend pairs
per C at an observed 55W limit. Those additional startup controls are not applied
by the quick-start commands above; its performance figures are not reproduced
by this smoke or established for the subsequently integrated upstream runner.
The power limit was observed, not software-locked. Its continuity
failures remain visible in the feature table. Transformers was a complete-response
reference, and LMDeploy was not run because the optional package was absent.

Raw commands, environments, WAVs, JSONL, worker traces, rejected precision/power
runs and failure lists are retained locally under
`.local-soprano/artifacts/11-experiments/`; this documentation's command replay
is under `.local-soprano/artifacts/12-docs/`. These ignored local files are not
shipped with the repository. Without those artifacts, the numbers above are a
reported local qualification, not independently reproduced evidence.
