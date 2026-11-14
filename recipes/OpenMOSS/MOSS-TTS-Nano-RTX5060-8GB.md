# MOSS-TTS-Nano — RTX 5060 Laptop 8 GB

## Summary

- Vendor: OpenMOSS
- Model: `OpenMOSS-Team/MOSS-TTS-Nano-100M`
- Task: text-to-speech
- Mode: online serving, one request at a time
- Hardware: 1x NVIDIA GeForce RTX 5060 Laptop, 8151 MiB
- Maintainer: community

## When to use this recipe

The stock profile already fits this GPU. This one only lowers
`gpu_memory_utilization` from 0.3 to 0.2, so the stage claims less of the card
when something else needs the rest. The measured peak did not change.

## Supported model contract

| Task | Input | Output | Profile |
| --- | --- | --- | --- |
| Text-to-speech | one text prompt | 48 kHz WAV | `moss_tts_nano_8gb.yaml` |

Requests must stay serial. The model and its audio tokenizer keep process-wide
RNG and streaming decode state, so concurrent generation corrupts audio.

## References

- Model card: [OpenMOSS-Team/MOSS-TTS-Nano-100M](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M), commit `44502f80dbf9743528fa921cc544d662c685ebec`
- Audio tokenizer: `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano`, commit `6aa02b01e445cc585582cf0ba480bc3ea6c8dd68`
- Deploy profile: [`vllm_omni/deploy/moss_tts_nano_8gb.yaml`](../../vllm_omni/deploy/moss_tts_nano_8gb.yaml), overlaying [`moss_tts_nano.yaml`](../../vllm_omni/deploy/moss_tts_nano.yaml)
- Tracking issue: [vllm-omni#10008](https://github.com/vllm-project/vllm-omni/issues/10008)

## Hardware

- Accelerator: NVIDIA GeForce RTX 5060 Laptop, 8151 MiB, SM120 (Blackwell).
- Devices: 1. No interconnect.
- Host memory: 30 GB. Not on the critical path; the model is resident on GPU.
- Qualification: one machine, measured 2026-11-10.

## Software environment

- OS: Linux, container `vllm/vllm-omni:nightly`
- Python: 3.12
- Driver / runtime: NVIDIA driver 595.99.02, CUDA 13.0
- vLLM: 0.32.0
- vLLM-Omni: 0.32.0rc2.dev352+g670aa7ec6, source `670aa7ec6043571e0a8a16448685322e41d93828`

## Command

```bash
vllm serve OpenMOSS-Team/MOSS-TTS-Nano-100M --omni \
  --deploy-config vllm_omni/deploy/moss_tts_nano_8gb.yaml --port 8091
```

## Verification

Send one speech request and check the WAV is 48 kHz and not silent. The
qualified run used three fixed prompts and seed 42.

## Notes

| `gpu_memory_utilization` | Cold start | Device peak | Host RSS | Steady |
| --- | --- | --- | --- | --- |
| 0.30 (stock) | 38.0 s | 845 MiB | 4474 MiB | 2.02 / 1.71 / 2.07 s |
| 0.20 (this profile) | 35.7 s | 845 MiB | 4427 MiB | 2.07 / 1.71 / 2.15 s |

Both runs produced byte-identical WAV at 48 kHz, durations 3.76 / 3.04 / 3.92 s,
RMS 0.154 / 0.127 / 0.113, all finite. 0.20 does not reduce the peak. It only
stops the stage from reserving memory it never uses.

Decoding reference audio needs TorchCodec and the FFmpeg shared libraries.
Without them the model returns a silent WAV instead of failing.

## Supported features

| Feature | Status on this hardware |
| --- | --- |
| Single GPU, serial requests | Qualified. See the table above. |
| Concurrent requests | Not supported. The model keeps process-wide decode state. |
| Quantization, sleep/wake | Not measured. |
