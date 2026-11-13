# OmniVoice

> Offline and online text to speech with automatic voice, voice design, and reference voice cloning

## Summary

- Vendor: k2-fsa
- Model: `k2-fsa/OmniVoice`
- Task: Multilingual text to speech
- Mode: Offline inference for all modes and online serving for automatic voice
- Maintainer: Community

## References

- [Hugging Face model](https://huggingface.co/k2-fsa/OmniVoice)
- [Offline text to speech examples](../../docs/user_guide/examples/offline_inference/text_to_speech.md#omnivoice)
- [Online text to speech examples](../../docs/user_guide/examples/online_serving/text_to_speech.md#omnivoice)
- [Deploy config](../../vllm_omni/deploy/omnivoice.yaml)

## Requirements

Download the checkpoint before the first run:

```bash
huggingface-cli download k2-fsa/OmniVoice
```

Automatic voice and voice design require `transformers>=4.57.0`. Reference voice cloning requires `transformers>=5.3.0`.

The vLLM Omni deploy config runs the OmniVoice generator and decoder inside one float32 diffusion stage.

## ROCm

### 1x AMD MI300X

#### Environment

- GPU: one AMD Instinct MI300X with 191.69 GiB visible HBM and gfx942
- Kernel: Linux 6.8.0-134-generic, x86_64
- Python: 3.12.13
- PyTorch: 2.11.0+gitd0c8b1f
- ROCm or HIP: 7.2.53211
- vLLM: 0.27.0+rocm723
- vLLM Omni checkout: `73e1368c7bb940efe1a025859c9d6c8eeeb2e3f0`
- transformers: 5.15.0

The run used the official ROCm image built from `docker/Dockerfile.rocm`.

#### Automatic voice

Run the offline example from the repository root:

```bash
python3 examples/offline_inference/text_to_speech/omnivoice/end2end.py \
    --model k2-fsa/OmniVoice \
    --deploy-config vllm_omni/deploy/omnivoice.yaml \
    --text "Hello, this is OmniVoice running on one AMD MI300X." \
    --seed 42 \
    --output omnivoice_mi300x.wav
```

The checked run produced a valid 24 kHz mono WAV with 3.60 seconds of audio, RMS 0.1194, and peak absolute amplitude 0.9129. The first request took 0.685 seconds, which gives a real time factor of 0.19. Model loading used 3.5739 GiB and took 9.564 seconds. The request used 4.56 GB reserved and 3.64 GB allocated according to the internal profiler, while the largest one second whole device sample was 8.33 GiB. The log states that the OmniVoice Triton kernels loaded.

#### Reference voice cloning

The offline interface also accepts a reference WAV and its transcript:

```bash
python3 examples/offline_inference/text_to_speech/omnivoice/end2end.py \
    --model k2-fsa/OmniVoice \
    --text "Hello, this is a cloned voice." \
    --ref-audio ref.wav \
    --ref-text "This is the reference transcription." \
    --output cloned.wav
```

#### Voice design

```bash
python3 examples/offline_inference/text_to_speech/omnivoice/end2end.py \
    --model k2-fsa/OmniVoice \
    --text "Hello, this is a designed voice." \
    --instruct "female, low pitch, British accent" \
    --output designed.wav
```

## Online serving

Online serving currently supports automatic voice:

```bash
vllm serve k2-fsa/OmniVoice --omni --port 8091 --trust-remote-code
```

```bash
cd examples/online_serving/text_to_speech/omnivoice
python speech_client.py --text "Hello, how are you?"
```
