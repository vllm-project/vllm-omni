# Qwen3-TTS 1.7B on Ascend 310P (Atlas 300I Duo)

## Summary

- Vendor: Qwen
- Model: Qwen3-TTS-12Hz-1.7B (Base voice-clone variant)
- Task: Text-to-speech
- Mode: Online serving
- Hardware: 1x Ascend 310P3 chip (one Atlas 300I Duo card), 44 GB
- Maintainer: Community

## When to use this recipe

Serve the Qwen3-TTS Base voice-clone variant on a single Ascend 310P3 chip
using vLLM-Omni with the vLLM-Ascend 310P image. This is a bounded
functional qualification: one cold-start reference-audio request plus
steady-state repeats, sequential requests, concurrency 1.

The mel front-end runs on the NPU through the pure-PyTorch FFT/STFT
re-implementation introduced in PR #11134 — 310P ships no FFT-family aclnn
kernels, so without that change the reference-audio mel spectrogram falls
back to CPU with a per-request CPU<->NPU round-trip.

## Supported model contract

See the [shared Qwen3-TTS recipe](./Qwen3-TTS.md) and the
[canonical model documentation](https://github.com/QwenLM/Qwen3-TTS). The
tested input and output:

| Variant | Tested input | Observed output |
| --- | --- | --- |
| Base | 10 s Chinese reference WAV + request text `你好，这是昇腾三一零P上的Qwen3语音合成服务测试。` (`x_vector_only_mode`) | 24 kHz mono 16-bit WAV, 5.12 s |

Other variants, other languages, long inputs, and voice similarity were not
qualified here.

## References

- [Qwen3-TTS model card](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)
- [NPU installation guide](../../docs/getting_started/installation/npu.md)
- [Shared online TTS examples](../../examples/online_serving/text_to_speech/README.md#qwen3-tts)
- [Tested deployment configuration](../../vllm_omni/deploy/qwen3_tts.yaml)
- [310P optimization roadmap](https://github.com/vllm-project/vllm-omni/issues/10538)
- [FFT/STFT implementation PR](https://github.com/vllm-project/vllm-omni/pull/11134)

## Hardware

- Accelerator: 1x Ascend 310P3 (Atlas 300I Duo card, single chip used),
  44 GB; host device `davinci2`, container logical device 0
- Architecture: aarch64
- Host: 96 CPU cores, 1 TiB RAM
- Device interconnect: no multi-device communication tested
- Qualification scope: one model server at a time, concurrency 1

## Software environment

| Component | Observed version |
| --- | --- |
| Container image | `quay.io/ascend/vllm-ascend:v0.30.0rc2-310p` |
| Container OS | Ubuntu (aarch64) |
| Python | 3.12.13 |
| Ascend driver / CANN | 25.5.1 / 9.2.0 |
| PyTorch / torch-npu | `2.10.0+cpu` / `2.10.0.post7` |
| vLLM | `0.30.0+empty` |
| vLLM-Ascend | `0.30.0rc2` |
| vLLM-Omni | `v0.30.0` + PR #11134 (`44a6b13`) |
| Transformers | 5.14.0 |

Version notes that matter on 310P:

- `transformers` must be `< 5.15` (5.22 breaks `Qwen3TTSTokenizerV2Encoder`
  with an `upsample` property error); 5.14.0 verified.
- `--dtype float16` is required: BF16 weight loading hits a 310P TransData
  kernel gap at embedding load (`Optype [TransData] ... unsupported`).
  FP16 loads clean in 148 s.
- 310P has no Triton (vLLM-Ascend RFC #15577); the MRV2
  `compute_slot_mapping` Triton kernel needs a native fallback
  (vLLM-Ascend #21381). The qualification used a local test-only fallback.
- The `_310p` patch wiring fixes from PR #11134 are required: without them
  `is_310p()` silently returns `False` on the worker path and the model
  loads the base builder (CPU mel) instead of the NPU-native one.

## Command

### Environment

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/opt/vllm-omni:$PYTHONPATH
export ASCEND_RT_VISIBLE_DEVICES=0
export VLLM_BATCH_INVARIANT=1
```

### Download the checkpoint

```bash
# weights: model.safetensors (3.86 GB) + speech_tokenizer/model.safetensors (0.68 GB)
# matching Hugging Face revision: fd4b254389122332181a7c3db7f27e918eec64e3
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir /workspace/qwen3-tts-model
```

### Launch

```bash
cd /opt/vllm-omni
vllm serve /workspace/qwen3-tts-model \
    --served-model-name qwen3-tts-base \
    --deploy-config vllm_omni/deploy/qwen3_tts.yaml \
    --omni --port 8091 --trust-remote-code \
    --enforce-eager --dtype float16
```

`--enforce-eager` was used for the qualification; graph capture was not
tested on 310P. Startup to `health: 200` takes about 6 minutes including
weight loading (148 s) and stage-1 initialization.

## Verification

```bash
curl --fail http://127.0.0.1:8091/health

REF_AUDIO=$(base64 -w0 /workspace/ref_zh.wav)   # 3~15 s clean speech
curl --fail http://127.0.0.1:8091/v1/audio/speech \
    -H 'Content-Type: application/json' \
    --data-binary @- --output clone.wav <<JSON
{"model":"qwen3-tts-base","task_type":"Base",
 "input":"你好，这是昇腾三一零P上的Qwen3语音合成服务测试。",
 "language":"Chinese",
 "ref_audio":"data:audio/wav;base64,${REF_AUDIO}",
 "x_vector_only_mode":true,
 "response_format":"wav","seed":42}
JSON
```

### Observed results

Qualification date: 2026-11-26. The cold-start reference-audio request
(first request after server startup) returned HTTP 200 in 36.98 s; a
steady-state repeat returned HTTP 200 in 20.74 s for 5.12 s of audio
(RTF ≈ 4.05). The output was a nonempty, finite, 24 kHz mono PCM16 WAV.

The NPU mel front-end was confirmed active via the worker log:

```text
[310P-PATCH-APPLIED] talker/codepredictor/builder swaps installed
[FFT-OPS-MEL] NPU mel via npu_fft (310P builder override)
```

This is a single-request functional qualification, not a latency or
throughput benchmark. 310P3 is roughly 1/4 the compute of an Atlas 800I A2
module; RTF is correspondingly higher than the 910B3 profile.

## Notes

- The mel front-end FFT/STFT re-implementation and the two 310P patch-wiring
  fixes (`is_310p` import fallback, worker-base model-patch gate) come from
  PR #11134; serving without them either crashes at the Triton
  `compute_slot_mapping` kernel or silently runs the CPU mel path.
- `torch.stft`/`torch.fft.*`/`torch.hann_window` have no aclnn kernels on
  310P; direct calls fail with `NPU function error` / `ERR01007` on complex
  transfers. Use the `fft_ops` module from PR #11134.
- No concurrency, endurance, long-prompt, streaming, or cross-hardware
  comparison was performed. Results apply to the recorded environment and
  model revision; they do not qualify other images or current main.

## Supported features

| Feature | Qualification on this profile | Shared guide |
| --- | --- | --- |
| Online speech API | Base voice-clone variant, one 310P3 chip | [TTS examples](../../examples/online_serving/text_to_speech/README.md#qwen3-tts) |
| Streaming output | Not tested | [Streaming examples](../../examples/online_serving/text_to_speech/README.md#qwen3-tts) |
| Reference audio | Chinese reference passed; quality/similarity unassessed | [Model contract](./Qwen3-TTS.md) |
| CustomVoice / VoiceDesign | Not tested on 310P | [Model contract](./Qwen3-TTS.md) |
| Multi-device or concurrent serving | Not tested | [NPU guide](../../docs/getting_started/installation/npu.md) |
