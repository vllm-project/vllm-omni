# VibeVoice-1.5B text-to-speech — H100

> Offline and HTTP TTS with optional references and up to four speakers,
> running as one eager `LLM_AR` stage on one H100.

## Summary

- Vendor: Microsoft
- Model: `vibevoice/VibeVoice-1.5B-hf` (native Transformers checkpoint)
- Task: text-to-speech, up to four speakers, optional voice cloning
- Mode: offline `Omni` and `/v1/audio/speech` (non-streaming or streaming)
- Hardware: 1x H100 80GB
- Maintainer: Community

## When to use this recipe

Use VibeVoice-1.5B for plain text or speaker-labelled dialogue, with optional
reference voices. Omit references for native no-reference generation;
there are no bundled default voices.

## Supported model contract

| Property        | Value                                                                                                                                                                                                                                                           |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Checkpoint      | `vibevoice/VibeVoice-1.5B-hf` only; the original `microsoft/VibeVoice-1.5B` schema and its forks are rejected at load                                                                                                                                           |
| Text            | non-empty plain text for speaker 0, or `Speaker N: text` lines with up to four distinct IDs; special-token literals and mixed plain/script lines are rejected                                                                                                   |
| Reference audio | optional, up to one clip per speaker (four total); any channel layout and sample rate (mixed to mono, resampled to 24 kHz), up to 60 s per clip                                                                                                                 |
| No reference    | `render_vibevoice_prompt(text, reference=False)` renders the native no-voice prompt. As in Transformers, there is no default voice: the voice is sampled per request from the diffusion noise. A reference prompt without audio is rejected before scheduling   |
| Output          | mono float32 at 24 kHz, 3,200 samples per audio token; offline FINAL_ONLY returns the concatenated waveform in `multimodal_output["audio"]`                                                                                                                     |
| Generation      | CFG 1.3, 10 DPM-Solver diffusion steps (checkpoint defaults); length capped by `max_tokens`                                                                                                                                                                     |
| Deploy profile  | `vllm_omni/deploy/vibevoice.yaml`: BF16, eager, `max_num_seqs: 4`, `max_model_len: 16384`, FlashAttention 3, `kv_cache_memory_bytes: 2 GiB`                                                                                                                     |

## References

- Model card: [https://huggingface.co/vibevoice/VibeVoice-1.5B-hf](https://huggingface.co/vibevoice/VibeVoice-1.5B-hf)
- Technical report: [https://arxiv.org/abs/2508.19205](https://arxiv.org/abs/2508.19205)
- Shared TTS examples: [offline](../../examples/offline_inference/text_to_speech/README.md#vibevoice), [online](../../examples/online_serving/text_to_speech/README.md#vibevoice)
- Supported models: [`docs/models/supported_models.md`](../../docs/models/supported_models.md)
- Pipeline: `vllm_omni/model_executor/models/vibevoice/pipeline.py`; test: `tests/e2e/offline_inference/test_vibevoice_tts.py`

## Hardware

- Accelerator model and per-device memory: NVIDIA H100 80GB HBM3
- Number of devices: 1
- Device interconnect: not applicable
- Host memory: not material (no CPU offload)
- Qualification scope: offline `Omni`, deploy defaults, concurrency 1 and 4;
  HTTP functional tests with 2/4 references, streaming and queued requests.
  HTTP checks do not establish per-speaker perceptual quality or serving performance.

## Software environment

The following environment and quality results describe the historical offline qualification.

- OS: Linux
- Python: 3.12
- Driver / runtime: NVIDIA driver 580.65.06, CUDA 13.0 (torch 2.13.0+cu130)
- vLLM version: 0.31.0 (Transformers 5.14.1, diffusers 0.40.0, flashinfer-python 0.7.0.post1)
- HF reference: Transformers 5.19.0, torch 2.13.0, diffusers 0.40.0 in a separate environment
- Reviewed vLLM-Omni implementation: `375fb7cfa0f68711c3b8f38bb0d81fb38122f35a`
  (production code unchanged from `0d3a6ffa`; the later amend strengthens an E2E assertion)

## Command

```python
import soundfile as sf
import torch
from vllm import SamplingParams

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.model_executor.models.vibevoice.prompt import render_vibevoice_prompt

prompt = {
    "prompt": render_vibevoice_prompt("Hello from VibeVoice."),
    "multi_modal_data": {"audio": sf.read("reference.wav", dtype="float32")},  # (waveform, sample_rate)
}
# Without a reference: {"prompt": render_vibevoice_prompt(text, reference=False)}
omni = Omni(model="vibevoice/VibeVoice-1.5B-hf")  # loads vllm_omni/deploy/vibevoice.yaml
try:
    output = omni.generate([prompt], [SamplingParams(max_tokens=2048)])[0].outputs[0]
    mm = output.multimodal_output
    audio = mm.get("audio")
    if audio is None:
        audio = mm.get("model_outputs")
    if audio is None:
        raise RuntimeError("VibeVoice returned no audio.")
    if isinstance(audio, list):
        if not audio:
            raise RuntimeError("VibeVoice returned empty audio.")
        audio = torch.cat([torch.as_tensor(chunk).reshape(-1) for chunk in audio])
    else:
        audio = torch.as_tensor(audio).reshape(-1)
    sf.write("out.wav", audio.detach().cpu().numpy(), 24000)
finally:
    omni.close()
```

## Online serving

```bash
vllm serve vibevoice/VibeVoice-1.5B-hf --omni --port 8091

# Native no-reference generation; no preset voice is required.
curl http://localhost:8091/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"vibevoice/VibeVoice-1.5B-hf","input":"Hello from VibeVoice.","response_format":"wav","max_new_tokens":512}' \
  --output out.wav

# SSE: speech.audio.delta (base64 PCM) followed by speech.audio.done with usage.
curl -N http://localhost:8091/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"vibevoice/VibeVoice-1.5B-hf","input":"Hello from VibeVoice.","stream":true,"response_format":"pcm","max_new_tokens":512}'
```

For raw PCM/WAV chunks, set `stream_format: "audio"` and `response_format: "pcm"` or `"wav"`. PCM is signed 16-bit little-endian, 24 kHz mono. Streaming
WAV has a header with unknown final length. Network chunks need not align with
audio-token boundaries. This single-stage pipeline retains `async_chunk: false`.

For dialogue, pass `"input": "Speaker 3: Hello.\nSpeaker 1: Welcome.\nSpeaker 3: Thank you."`.
IDs are preserved; repeated turns do not count as new speakers. A `ref_audio`
string conditions the first distinct speaker. A list of K sources conditions
the first K distinct speakers in first-appearance order; K may be smaller than
the speaker count. Sources can be HTTP(S) URLs, base64 audio data URLs, or
`file://` URIs (requiring `--allowed-local-media-path`). Remaining speakers
have no reference; no reference audio is inserted implicitly.

For a reusable single-speaker voice, upload audio via
[`POST /v1/audio/voices`](../../docs/serving/speech_api.md), then send its name
as `voice`. Uploads have a 20 MiB size limit in addition to the 60 s duration
cap; use mono or compressed audio for long references that exceed the size
limit (e.g. 60 s of 96 kHz stereo PCM16). Uploads with only a speaker embedding
are not supported.
`voice` and `ref_audio` are mutually exclusive. The shared API treats
`voice="default"` as omitted unless an uploaded voice has that name; it does
not insert reference audio. Other unregistered voice names are rejected.
Stored upload transcripts are not used as conditioning. For multi-speaker requests, use `ref_audio` instead
of an uploaded `voice` name.

Only `speed=1.0` and output `sample_rate=24000` are supported. Request-level
seed, instructions, language, ref_text, task_type, ref_audio_2,
speaker_embedding, word_timestamps and non-empty extra_params are rejected.
Model-specific controls belong in `extra_params`; the shared request schema
ignores unknown top-level JSON keys rather than forwarding them to the adapter.
CFG=1.3 and ten diffusion steps are fixed.
`max_new_tokens` must be 1–16384; prompt plus output must fit the 16384 context.
At most four requests run concurrently; additional requests queue. A bounded
startup warmup uses no reference and does not provide a default voice.
WebSocket incremental-text input is outside this recipe's qualification.

## Verification

```bash
VIBEVOICE_MODEL=vibevoice/VibeVoice-1.5B-hf \
    pytest -s -v tests/e2e/offline_inference/test_vibevoice_tts.py \
    tests/e2e/online_serving/test_vibevoice_tts.py \
    -m 'advanced_model and cuda and H100' --run-level=advanced_model
```

Expected offline behavior: four requests of different lengths stop naturally
with `samples == audio_tokens × 3200`, a request stopped at the length cap
still returns its last chunk, and a no-reference request produces finite,
non-empty audio. All cases verify the public `audio` field and 24 kHz sample rate.

Online tests cover optional references, 2/4-speaker scripts, PCM/WAV/SSE
framing, uploads, queuing beyond four requests and recovery after disconnects.
These are functional checks, not ASR/SIM quality gates. CI routes both modules
to a single-H100 step in `.buildkite/cuda/test-merge.yml` (L3).

### Full-dataset evaluation (2026-10-15)

Eight local runs on H100 80GB: Seed-TTS-Eval EN (1,088 items) and ZH (2,020
items), Omni/HF, B1/B4. Launcher startup records identify source `2e558341b`.
Both backends used the native checkpoint and the same ordered datasets, BF16,
seed 0, `max_tokens=2048`, CFG 1.3 and 10 diffusion steps. Omni used deploy
defaults (TP1/PP1). Each configuration ran one same-size warmup batch followed
by one full-dataset pass. Matching seeds do not imply identical waveforms.
GPU exclusivity was declared by the operator; retained start/end snapshots
show idle GPUs but do not establish uninterrupted isolation.

Aggregate RTF = summed generation batch wall time / all generated audio duration.
Both timers exclude startup, warmup, reference-file reads, WAV writes, shutdown
and quality evaluation. Boundaries are not strictly identical: Omni prompt-string
rendering precedes its timer, while HF template processing is timed; HF also
times CPU extraction and stop/duration bookkeeping.

WER/SIM use the shared `seed_tts_eval.py` evaluator: Whisper-large-v3 for EN,
Paraformer-zh for ZH, and WavLM-base-plus embedding cosine for SIM. Saved WAVs
are FLOAT before the common PCM16 evaluation conversion. WER/SIM are per-item
means; SIM is a proxy, not a perceptual-quality or speaker-verification guarantee.

| Corpus | Backend | Batch | WER    | SIM    | Evaluated | Natural stop | RTF    |
| ------ | ------- | ----- | ------ | ------ | --------- | ------------ | ------ |
| EN     | Omni    | 1     | 0.0501 | 0.8623 | 1088      | 1088/1088    | 0.3171 |
| EN     | HF      | 1     | 0.0599 | 0.8649 | 1088      | 1088/1088    | 0.4464 |
| EN     | Omni    | 4     | 0.0745 | 0.8618 | 1088      | 1088/1088    | 0.1161 |
| EN     | HF      | 4     | 0.0563 | 0.8631 | 1088      | 1088/1088    | 0.1630 |
| ZH     | Omni    | 1     | 0.0249 | 0.8742 | 2019      | 2020/2020    | 0.3153 |
| ZH     | HF      | 1     | 0.0263 | 0.8718 | 2020      | 2020/2020    | 0.4419 |
| ZH     | Omni    | 4     | 0.0263 | 0.8731 | 2020      | 2020/2020    | 0.1141 |
| ZH     | HF      | 4     | 0.0286 | 0.8720 | 2020      | 2020/2020    | 0.1651 |

- ZH Omni B1 excludes one `empty_asr` from its means. ZH Omni B4 quality uses
  a full **CPU re-evaluation of the same 2,020 saved WAVs**, replacing an
  incomplete CUDA evaluation with 27 ASR initialization exceptions. It has
  zero evaluation failures; generation RTF and 2020/2020 natural stops remain
  from the original run. ASR success and natural termination are separate.
- Observed HF/Omni RTF ratios are **1.40–1.45x**, not exact matched-boundary
  or identical-output speedups. EN Omni B4 WER is about **1.81 percentage
  points higher** than HF B4. These single passes do not establish statistical
  equivalence or explain cross-run differences.
- [Fixed-version evidence: scripts, reports, logs and audit](https://github.com/YouHengfei/vllm-omni/blob/2b404800150c3355b45ef333983acf388c09edb4/batches/ab_std_20261015_080817/EVIDENCE.md)
  documents the launcher metadata defect and remaining provenance limitations.
  Startup hashes do not cover all weights/reference audio; the immutable dataset
  revision is unavailable and evaluation model aliases were not forcibly pinned.
  This is not a new peak-memory measurement. Serving benchmarks remain a follow-up.

### Online full-dataset evaluation (2026-10-19)

Head `93166c21b`, Seed-TTS-Eval, single-speaker reference cloning (B1, serial).
vLLM-Omni online on GPU4 (H100 80GB, BF16, eager, FA3, seed=0, CFG 1.3,
10 diffusion steps, `max_new_tokens=2048`). HF reference in a separate env
(Transformers 5.19, torch 2.13.0, same checkpoint). WER uses Whisper-large-v3
(EN) / Paraformer (ZH); SIM uses WavLM-base-plus embedding cosine (proxy,
not speaker verification). RTF = per-request wall time / generated audio
duration. Both runs are `status=complete`, `contaminated=False`, with GPU
snapshot before/after and env freeze.

| Locale | Backend      | Transport       | Items | WER (mean) | SIM (mean) | RTF   | TTFP p50 / p99 (ms) |
| ------ | ------------ | --------------- | ----- | ---------- | ---------- | ----- | ------------------- |
| EN     | Omni online  | non-streaming   | 1088  | 0.0571     | 0.8617     | 0.310 | —                   |
| EN     | Omni online  | streaming (PCM) | 1088  | 0.0869     | 0.8623     | 0.306 | 55 / 62             |
| EN     | HF reference | serial generate | 1088  | 0.0599     | 0.8649     | 0.433 | n/a                 |
| ZH     | Omni online  | non-streaming   | 2020  | 0.0275     | 0.8723     | 0.320 | —                   |
| ZH     | Omni online  | streaming (PCM) | 2020  | 0.0310     | 0.8708     | 0.315 | 57 / 63             |
| ZH     | HF reference | serial generate | 2019  | 0.0281     | 0.8722     | 0.437 | n/a                 |

- All backends produced natural stops for all items. 0 request failures;
  2 ASR failures in ZH non-streaming (ASR noise, not generation failure).
- WER/SIM are single-pass observations; statistical equivalence and absence
  of regression have not been established. Noise attribution would require
  controlled paired/repeated measurements with an explicit acceptance tolerance.
- Omni online RTF (~0.31) is well below 1.0 (real-time capable); HF serial
  RTF is ~0.43. Timer boundaries differ (Omni includes HTTP overhead; HF is
  serial `generate`), so the ~1.4× ratio is not a matched-boundary speedup
  claim.
- EN streaming mean WER is 0.0869 versus 0.0571 non-streaming (+2.98 percentage
  points), despite both medians being zero. The cause is unestablished; this
  difference alone does not establish a runtime regression. Observed streaming
  TTFP p50 is 55 ms (EN) and 57 ms (ZH).

## Notes

- Memory (historical observation): 12.25 GiB peak with deploy defaults
  (weights 5.16 GiB, positive KV 2.0 GiB, model-owned CFG negative KV 1.75 GiB),
  reserved at startup and flat during inference. With `max_model_len=4096` and
  `kv_cache_memory_bytes=512 MiB` the peak drops to 8.94 GiB.
- Memory (online serving, 2026-10-20, H100): the `vllm serve` process tree
  reserves 12.24 GiB at startup and stays flat within +0.02 GiB across serial,
  4-concurrent, and 4×~2-minute workloads (process-tree NVML sampled every
  50 ms, foreign processes excluded). GPU memory returned to 0 MiB after
  shutdown.
- Key flags: `enforce_eager: true` and `async_scheduling: false` are required.
  Startup fails if the KV budget cannot hold `max_num_seqs × max_model_len`,
  because preemption is not supported.
- Known limitations: no reference voice gives a different voice per request,
  and occasional bad samples, as in Transformers. Offline `Omni.generate`
  closes the engine when a request is rejected. Long-duration soak and other
  accelerators are not qualified.

## Supported features

| Feature                                                   | Status                                                                                 | Guide                                                                                                                         |
| --------------------------------------------------------- | -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Offline `Omni.generate`                                   | supported, up to 4 concurrent requests                                                 | [`examples/offline_inference/text_to_speech/README.md`](../../examples/offline_inference/text_to_speech/README.md#vibevoice)  |
| Voice clone                                               | optional; one reference per conditioned speaker, at most four                          | —                                                                                                                             |
| Multi-speaker                                             | up to four distinct speakers, partial references allowed                               | —                                                                                                                             |
| Default voices                                            | none; omitted references use native no-reference generation                            | —                                                                                                                             |
| `/v1/audio/speech` / streaming                            | non-streaming, SSE, PCM/WAV streams; uploaded audio voices for single-speaker requests | [`docs/serving/speech_api.md`](../../docs/serving/speech_api.md)                                                              |
| CUDA graphs / async scheduling                            | not supported (eager, sync scheduling)                                                 | —                                                                                                                             |
| Chunked prefill                                           | supported                                                                              | —                                                                                                                             |
| Prefix caching, speculative decoding, TP/PP, quantized KV | not supported (rejected at startup where applicable)                                   | —                                                                                                                             |
