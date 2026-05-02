# VoxCPM

## Prerequisites

Install VoxCPM in one of these ways:

```bash
pip install voxcpm
```

or point vLLM-Omni to a local VoxCPM source tree:

```bash
export VLLM_OMNI_VOXCPM_CODE_PATH=/path/to/VoxCPM/src
```

If the native VoxCPM `config.json` lacks HF metadata such as `model_type`,
prepare a persistent HF-compatible config directory and export:

```bash
export VLLM_OMNI_VOXCPM_HF_CONFIG_PATH=/tmp/voxcpm_hf_config
mkdir -p "$VLLM_OMNI_VOXCPM_HF_CONFIG_PATH"
cp "$VOXCPM_MODEL/config.json" "$VLLM_OMNI_VOXCPM_HF_CONFIG_PATH/config.json"
cp "$VOXCPM_MODEL/generation_config.json" "$VLLM_OMNI_VOXCPM_HF_CONFIG_PATH/generation_config.json" 2>/dev/null || true
python3 -c 'import json, os; p=os.path.join(os.environ["VLLM_OMNI_VOXCPM_HF_CONFIG_PATH"], "config.json"); cfg=json.load(open(p, "r", encoding="utf-8")); cfg["model_type"]="voxcpm"; cfg.setdefault("architectures", ["VoxCPMForConditionalGeneration"]); json.dump(cfg, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)'
```

The VoxCPM stage configs read `VLLM_OMNI_VOXCPM_HF_CONFIG_PATH` directly. The `python3 -c` form above avoids heredoc/indentation issues in interactive shells.

## Launch the Server

Use the async-chunk stage config by default:

```bash
export VOXCPM_MODEL=/path/to/voxcpm-model
cd examples/online_serving/voxcpm
./run_server.sh
```

Use the non-streaming stage config:

```bash
./run_server.sh sync
```

You can also launch the server directly:

```bash
vllm serve "$VOXCPM_MODEL" \
    --stage-configs-path vllm_omni/model_executor/stage_configs/voxcpm_async_chunk.yaml \
    --trust-remote-code \
    --enforce-eager \
    --omni \
    --port 8091
```

## Send Requests

### Basic text-to-speech

```bash
python openai_speech_client.py \
    --model "$VOXCPM_MODEL" \
    --text "This is a VoxCPM online text-to-speech example."
```

### Voice cloning

```bash
python openai_speech_client.py \
    --model "$VOXCPM_MODEL" \
    --text "This sentence is synthesized with a cloned voice." \
    --ref-audio /path/to/reference.wav \
    --ref-text "The exact transcript spoken in reference.wav."
```

`ref_text` must be the real transcript of the reference audio. Placeholder text or mismatched text will usually degrade quality badly.

### Streaming PCM output

```bash
python openai_speech_client.py \
    --model "$VOXCPM_MODEL" \
    --text "This is a streaming VoxCPM request." \
    --stream \
    --output voxcpm_stream.pcm
```

### Using curl

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "model": "OpenBMB/VoxCPM1.5",
        "input": "Hello from VoxCPM online serving.",
        "response_format": "wav"
    }' --output output.wav
```

Voice cloning:

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "model": "OpenBMB/VoxCPM1.5",
        "input": "This sentence uses a cloned voice.",
        "ref_audio": "https://example.com/reference.wav",
        "ref_text": "The exact transcript spoken in the reference audio.",
        "response_format": "wav"
    }' --output cloned.wav
```

Streaming PCM:

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "model": "OpenBMB/VoxCPM1.5",
        "input": "This is a streaming VoxCPM request.",
        "stream": true,
        "response_format": "pcm"
    }' --output output.pcm
```

## Supported Request Shape

VoxCPM online serving currently supports:

- plain text-to-speech
- voice cloning with `ref_audio` + `ref_text`
- `stream=true` with `response_format=pcm` or `wav`

VoxCPM online serving does not use these generic TTS fields:

- `voice`
- `instructions`
- `language`
- `speaker_embedding`
- `x_vector_only_mode`

## Streaming vs Non-Streaming

- `voxcpm_async_chunk.yaml` enables async-chunk streaming and is best for single-request streaming latency.
- `voxcpm.yaml` performs one-shot latent generation then VAE decode.

Like native VoxCPM, the async streaming path should be treated as single-request. If you need stable throughput benchmarking, prefer `voxcpm.yaml`.

Do not use `voxcpm_async_chunk.yaml` for concurrent online streaming or `/v1/audio/speech/batch`. For multiple requests, prefer `voxcpm.yaml`.

## Benchmark

The serving benchmark reports TTFP and RTF:

```bash
python benchmarks/voxcpm/vllm_omni/bench_tts_serve.py \
    --host 127.0.0.1 \
    --port 8091 \
    --num-prompts 10 \
    --max-concurrency 1 \
    --result-dir /tmp/voxcpm_bench
```

For the async-chunk server, keep `--max-concurrency 1`.
