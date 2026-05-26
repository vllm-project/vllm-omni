# higgs-audio v2 online example

This directory contains the online-serving entry points for boson-ai's higgs-audio v2 as integrated by vllm-omni: a 2-stage TTS pipeline (Llama-3.2-3B talker with DualFFN audio expert + HiggsAudio codec decoder) emitting 24 kHz mono speech.

## Prerequisites

Voice clone uses HF's `HiggsAudioV2TokenizerModel` loaded from `k2-fsa/OmniVoice/audio_tokenizer/` (the boson-ai standalone tokenizer Hub repo's `model.safetensors` is the 3B talker LM, not the codec). Only that ~806 MB subdir is downloaded.

```bash
pip install -U "transformers>=5.3.0"
```

## Files

- `run_server.sh` — launch the vllm-omni server with the bundled `vllm_omni/deploy/higgs_audio_v2.yaml` deploy config.
- `batch_speech_client.py` — send a list of prompts to `/v1/audio/speech` and save the returned WAV / PCM bytes to a directory; optionally passes `--ref-audio` + `--ref-text` for shallow voice clone.

## Launching the server

```bash
GPUS=6,7 PORT=8094 bash examples/online_serving/text_to_speech/higgs_audio_v2/run_server.sh
```

Environment overrides:

- `MODEL` — HF id of the talker (default `bosonai/higgs-audio-v2-generation-3B-base`).
- `PORT` — server port (default `8094`).
- `GPUS` — `CUDA_VISIBLE_DEVICES` value (default `6,7`).
- `GPU_UTIL` — `--gpu-memory-utilization` (default `0.4`).

The script also exports `VLLM_USE_DEEP_GEMM=0` / `VLLM_MOE_USE_DEEP_GEMM=0` so the example works on images without the optional `deep_gemm` backend.

The deploy YAML ships with `async_chunk: false` and `codec_streaming: true`, i.e. Stage 0 finishes its codec frames before Stage 1 starts decoding, and Stage 1 streams WAV/PCM bytes to the client chunk-by-chunk.

## Driving the server

Plain TTS:

```bash
python examples/online_serving/text_to_speech/higgs_audio_v2/batch_speech_client.py \
    --base-url http://localhost:8094 \
    --model bosonai/higgs-audio-v2-generation-3B-base \
    --output-dir /tmp/higgs_audio_v2_batch \
    --prompts "Hello world." \
              "The quick brown fox jumps over the lazy dog."
```

Voice clone — pass a reference clip and its transcript (both required together):

```bash
python examples/online_serving/text_to_speech/higgs_audio_v2/batch_speech_client.py \
    --base-url http://localhost:8094 \
    --model bosonai/higgs-audio-v2-generation-3B-base \
    --output-dir /tmp/higgs_audio_v2_clone \
    --ref-audio /path/to/reference.wav \
    --ref-text  "Exact transcript spoken in reference.wav." \
    --prompts "Hello, this is a cloned voice."
```

## Notes

- `--ref-text` must be the real transcript of `--ref-audio`; mismatched text degrades cloned-voice quality.
- Out of scope (rejected with explicit 4xx by the request validator): multi-speaker `[SPEAKERn]` tags inside `input`, `profile:` text-only speaker descriptions, the `ref_audio_in_system_message` system-block variant, chunked long-form generation, and per-request `voice` / `instructions` / `task_type` / `language` / `speed != 1.0` / `x_vector_only_mode` / `speaker_embedding`.
