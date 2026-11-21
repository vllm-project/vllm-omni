# Qwen3-TTS 1.7B on Ascend 910B3 64GB

## Summary

- Vendor: Qwen
- Model: Qwen3-TTS-12Hz-1.7B (CustomVoice, VoiceDesign, Base)
- Task: Text-to-speech
- Mode: Online serving
- Hardware: 1x Ascend 910B3, 65536 MB HBM as reported by `npu-smi`
- Maintainer: Community

## When to use this recipe

Serve the three Qwen3-TTS 1.7B variants, one at a time, on a single Ascend
910B3 using vLLM-Omni 0.28.0 and the vLLM-Ascend plugin. This is a bounded
functional qualification with short prompts and sequential requests.

## Supported model contract

See the [shared Qwen3-TTS recipe](./Qwen3-TTS.md) and
[canonical model documentation](https://github.com/QwenLM/Qwen3-TTS) for the
voice and language contract. The tested inputs and outputs are:

| Variant | Tested input | Observed output |
| --- | --- | --- |
| CustomVoice | English text with `ryan`; Chinese text with `vivian`; an English style instruction | 24 kHz mono PCM16 WAV; raw PCM streaming also tested in English |
| VoiceDesign | English text and a natural-language voice description | 24 kHz mono PCM16 WAV |
| Base | English text, a 4.16 s synthetic reference WAV, and its transcript | 24 kHz mono PCM16 WAV |

Other languages, long inputs, and voice similarity were not qualified here.

## References

- [Qwen3-TTS model card](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
- [NPU installation guide](../../docs/getting_started/installation/npu.md)
- [Shared online TTS examples](../../examples/online_serving/text_to_speech/README.md#qwen3-tts)
- [Tested deployment configuration](https://github.com/vllm-project/vllm-omni/blob/eb11446b7f2e30ca582f8aff3afe12e9a2e66f6c/vllm_omni/deploy/qwen3_tts.yaml)
- [Recipe tracking issue #2645](https://github.com/vllm-project/vllm-omni/issues/2645)

## Hardware

- Accelerator: 1x Ascend 910B3, 65536 MB HBM; logical device 0 inside the container
- Architecture: aarch64
- Container resources: 32 CPU cores, 32 GiB RAM, 16 GiB `/dev/shm`
- Storage: 300 GB workspace volume; checkpoints and caches stored there
- Device interconnect: no multi-device communication tested
- Qualification scope: one model server at a time, concurrency 1

## Software environment

The platform image was named **vLLM-Omni**, with platform version label
**0.28.0** and publication page
[ascend/vllm-omni](https://quay.io/repository/ascend/vllm-omni).
An exact registry tag and image digest were not recorded; the platform label
alone is not an immutable image identifier.

| Component | Observed version |
| --- | --- |
| Container OS | Ubuntu 22.04.5 LTS |
| Python | 3.12.13 |
| Ascend driver / CANN | 26.1.1 / 9.1.0 |
| PyTorch / torch-npu | `2.10.0+cpu` / `2.10.0.post4` |
| vLLM | `0.28.0+empty`, commit `2cf0a6915ce544dc493a0990f2ea38d81601128a` |
| vLLM-Omni | `0.28.0+cpu`, commit `eb11446b7f2e30ca582f8aff3afe12e9a2e66f6c` (`v0.28.0`) |
| vLLM-Ascend | Image-provided source; package metadata reports `0.0.0`; no Git revision available |
| Transformers / Triton-Ascend | 5.14.1 / 3.2.2 |
| ModelScope / SoundFile / NumPy | 1.39.1 / 0.14.0 / 2.5.2 |

The `+cpu` package suffix does not describe the execution device: these runs
used `torch_npu` on the NPU. No runtime source patches, driver/CANN upgrades,
or deployment YAML edits were applied. MemFabric 1.2.0 and CANN registration
were left unchanged. The image had pre-existing `pip check` dependency
conflicts, so these results do not establish a clean dependency resolution.

## Command

### Prepare the tested image environment

These paths refer to the image above. For a different image, follow the NPU
installation guide and record its actual versions before comparing results.

```bash
export PATH=/usr/local/python3.12.13/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/python3.12.13/lib:/usr/local/lib:$LD_LIBRARY_PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann-9.1.0/share/info/ascendnpu-ir/bin/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export ASCEND_RT_VISIBLE_DEVICES=0
```

Download the checkpoints, including their `speech_tokenizer` subdirectories,
to the workspace volume. The qualification used ModelScope snapshots:

```bash
python - <<'PY'
from modelscope import snapshot_download

for variant in ("CustomVoice", "VoiceDesign", "Base"):
    name = f"Qwen3-TTS-12Hz-1.7B-{variant}"
    snapshot_download(f"Qwen/{name}", local_dir=f"/workspace/qwen3-tts-910b3/models/{name}")
PY
```

The downloaded weights, configuration, and tokenizer files were checked
against the following official Hugging Face revisions. ModelScope's default
revision can change; these hashes identify the qualified model contents,
not a pinned ModelScope download command.

| Checkpoint | Matching Hugging Face revision |
| --- | --- |
| CustomVoice | `0c0e3051f131929182e2c023b9537f8b1c68adfe` |
| VoiceDesign | `5ecdb67327fd37bb2e042aab12ff7391903235d3` |
| Base | `fd4b254389122332181a7c3db7f27e918eec64e3` |

### Launch

Use the checkout at the **tested vLLM-Omni commit**, not a deployment YAML
copied from current main:

```bash
cd /vllm-workspace/vllm-omni
VARIANT=CustomVoice
PORT=8091
vllm serve /workspace/qwen3-tts-910b3/models/Qwen3-TTS-12Hz-1.7B-${VARIANT} \
    --omni --deploy-config vllm_omni/deploy/qwen3_tts.yaml \
    --host 127.0.0.1 --port "$PORT" \
    --served-model-name Qwen/Qwen3-TTS-12Hz-1.7B-${VARIANT}
```

For VoiceDesign or Base, stop this model server and repeat with
`VARIANT=VoiceDesign` or `VARIANT=Base` and `PORT=8092`.
The stock configuration uses the V1 runner, both stages on device 0,
`gpu_memory_utilization: 0.3` per stage, `async_chunk: true`, and
`SharedMemoryConnector`. The stage 0 engine reported BF16, no quantization,
and tensor/pipeline/data parallel sizes of 1. Stage 0 has `max_num_batched_tokens: 32768` and
`max_model_len: 4096`. Its configured `max_num_seqs: 64` is not evidence of
64-request concurrency. Current main was not runtime-tested for this recipe.

## Verification

With CustomVoice running, check discovery and synthesize the English reference:

```bash
curl --fail http://127.0.0.1:8091/health
curl --fail http://127.0.0.1:8091/v1/models
curl --fail http://127.0.0.1:8091/v1/audio/voices
curl --fail http://127.0.0.1:8091/v1/audio/speech \
    -H 'Content-Type: application/json' \
    -d '{"model":"Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice","input":"Hello, this is a speech synthesis test. The weather is sunny today.","voice":"ryan","language":"English","response_format":"wav","seed":42}' \
    --output english.wav
curl --fail http://127.0.0.1:8091/v1/audio/speech \
    -H 'Content-Type: application/json' \
    -d '{"model":"Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice","input":"你好，这是语音合成测试。今天天气很好，欢迎使用我们的服务。","voice":"vivian","language":"Chinese","response_format":"wav","seed":42}' \
    --output chinese.wav
curl --fail --no-buffer http://127.0.0.1:8091/v1/audio/speech \
    -H 'Content-Type: application/json' \
    -d '{"model":"Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice","input":"Hello, this audio is being streamed from the server.","voice":"ryan","language":"English","response_format":"pcm","stream":true,"stream_format":"audio","seed":42}' \
    --output stream.pcm
```

After switching to VoiceDesign on port 8092:

```bash
curl --fail http://127.0.0.1:8092/v1/audio/speech \
    -H 'Content-Type: application/json' \
    -d '{"model":"Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign","task_type":"VoiceDesign","input":"Hello, this is a test of speech synthesis on an Ascend processor.","language":"English","instructions":"A warm, friendly female voice speaking clearly and calmly.","response_format":"wav","seed":42}' \
    --output voice-design.wav
```

After switching to Base on port 8092, reuse `english.wav` from CustomVoice
and its transcript as the synthetic reference:

```bash
REF_AUDIO=$(base64 -w0 english.wav)
curl --fail http://127.0.0.1:8092/v1/audio/speech \
    -H 'Content-Type: application/json' \
    --data-binary @- --output base.wav <<JSON
{"model":"Qwen/Qwen3-TTS-12Hz-1.7B-Base","task_type":"Base","input":"Hello, this is a test of speech synthesis on an Ascend processor.","language":"English","ref_audio":"data:audio/wav;base64,${REF_AUDIO}","ref_text":"Hello, this is a speech synthesis test. The weather is sunny today.","response_format":"wav","seed":42}
JSON
```

### Observed results

Qualification date: 2026-11-12. All discovery endpoints and synthesis cases
succeeded with HTTP 200. Each model received one warmup request before
sequential measured requests, with `seed: 42`. CustomVoice's warmup used
`Hello, this is a speech synthesis test.` with `ryan` in English; VoiceDesign
and Base used their verification payloads for warmup and then repeated them.
The table records **one request per case**, not a latency benchmark.
RTF is request wall time divided by generated audio duration.

| Case | Audio duration (s) | Request wall time (s) | RTF |
| --- | ---: | ---: | ---: |
| CustomVoice English | 4.16 | 1.985 | 0.477 |
| CustomVoice Chinese | 6.00 | 2.767 | 0.461 |
| CustomVoice English style instruction | 4.56 | 2.159 | 0.473 |
| CustomVoice English streaming | 3.20 | 1.527 | — |
| VoiceDesign English | 3.76 | 2.231 | 0.593 |
| Base English with synthetic reference | 4.32 | 2.208 | 0.511 |

The style case used input `I am so excited to share this wonderful news with you!`
and instruction `Speak with great enthusiasm and happiness.` with `ryan`.
Streaming delivered its first nonempty audio chunk after 0.149 s; three HTTP
chunks were observed. This is client-observed first audio, not model-only latency.
Warmup request times were 7.557 s (CustomVoice), 4.663 s (VoiceDesign), and
46.919 s (Base); these are not server startup times.

All six measured clips were nonempty, finite, nonsilent, 24 kHz mono PCM16.
CPU Whisper-small transcription recovered the expected words, allowing for
punctuation, case, and Chinese script differences. This checks content only;
listening quality, emotion fidelity, and speaker similarity were not assessed.

## Notes

- Sampled device-wide HBM maxima were 22443 MB (CustomVoice), 22437 MB
  (VoiceDesign), and 22723 MB (Base), versus a 3439 MB pre-server baseline.
  Samples used a one-second sleep plus query time (12, 5, and 36 samples).
  These totals include reservations and other device use; they are neither
  instantaneous peaks nor minimum memory requirements.
- The instance reported HCCS warning `819B8605` (`port lane drop`) before
  and after testing. An NPU matrix multiplication check passed and reported
  HBM ECC counters were zero. This is an instance condition, not a general
  910B3 limitation or a multi-device health qualification.
- No concurrency, endurance, long-prompt, or cross-hardware performance
  comparison was performed. Results apply to the recorded environment and
  model revisions; they do not qualify other images or current main.

## Supported features

| Feature | Qualification on this profile | Shared guide |
| --- | --- | --- |
| Online speech API | All three 1.7B variants, individually on one NPU | [TTS examples](../../examples/online_serving/text_to_speech/README.md#qwen3-tts) |
| Streaming output | CustomVoice English raw PCM only | [Streaming examples](../../examples/online_serving/text_to_speech/README.md#qwen3-tts) |
| Voice design / reference audio | English synthesis passed; perceptual quality unassessed | [Model contract](./Qwen3-TTS.md) |
| Multi-device or concurrent serving | Not tested | [NPU guide](../../docs/getting_started/installation/npu.md) |
