# Composable Text-LLM → TTS Pipeline

This directory demonstrates **RFC Theme 2: TTS as a Composable Layer**,
enabling any vLLM text model to be paired with any TTS decoder without
requiring a built-in talker stage in the base model.

## General Pattern
```
[Any vLLM text LLM]  ->  [text_tts_bridge]  ->  [Any TTS decoder]  ->  audio
      Stage 0              async_chunk hook          Stage 1
```

The bridge (`text_tts_bridge.py`) is a `custom_process_next_stage_input_func`
hook that:
1. Buffers incremental decoded tokens from Stage 0 into sentences
2. Forwards complete sentence chunks to Stage 1 as TTS inputs
3. Injects voice/language defaults when Stage 0 has no speaker concept

Any LLM + TTS combination is supported. The pipeline topology is defined
once in `models/text_tts/pipeline.yaml` and is model-agnostic. Users only
need to specify which models to use in the deploy config:

| Stage 0 (text LLM) | Stage 1 (TTS decoder) | Deploy config |
|---|---|---|
| Llama-3.1-8B | Qwen3-TTS-1.7B | `text_tts.yaml` (reference) |
| Mistral-7B | CosyVoice3 | edit `text_tts.yaml` |
| Domain-finetuned LLM | FishSpeech | edit `text_tts.yaml` |

## Config Structure

The configuration follows the two-tier pattern used across vLLM-Omni:

- **`vllm_omni/model_executor/models/text_tts/pipeline.yaml`**
  Static DAG topology: bridge hook, SharedMemoryConnector, edges.
  Defined once by the framework — users never need to touch this.

- **`vllm_omni/model_executor/stage_configs/text_tts.yaml`**
  User-facing deploy config: specifies which LLM and TTS model to use,
  plus runtime parameters (devices, GPU memory utilization).

## Reference Example: Llama-3.1-8B + Qwen3-TTS

- **Stage 0**: `meta-llama/Llama-3.1-8B-Instruct`
- **Stage 1**: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- **Pipeline**: `vllm_omni/model_executor/models/text_tts/pipeline.yaml`
- **Deploy config**: `vllm_omni/model_executor/stage_configs/text_tts.yaml`

### Setup
```bash
pip install vllm-omni soundfile
pip install onnxruntime sox   # required by Qwen3-TTS
```

### Usage
```bash
cd examples/offline_inference/llama3_qwen3tts

# Basic (2-GPU machine)
python end2end.py --prompt "Explain how transformers work."

# Custom voice and language
python end2end.py --prompt "Bonjour." --voice serena --language French

# Streaming (low latency, audio chunks arrive progressively)
python end2end.py --prompt "Tell me a story." --streaming
```

Available voices: `aiden`, `dylan`, `eric`, `ono_anna`, `ryan`,
`serena`, `sohee`, `uncle_fu`, `vivian`

## Adding a New LLM + TTS Combination

Only two fields in the deploy config need to change:

1. Set `stage_args[0].engine_args.model` to your text LLM
2. Set `stage_args[1].engine_args.model` to your TTS model
3. Adjust `runtime.devices` and `gpu_memory_utilization` as needed

The pipeline topology, bridge hook, and connector config are inherited
from `pipeline.yaml` with no changes required.

## Design Considerations

** Bridge as stage-level processor, not a new stage type.**
The bridge is a `custom_process_next_stage_input_func` hook in Stage 0
`engine_args`, reusing the existing `OmniChunkTransferAdapter` async_chunk
framework with zero changes to `OmniStage`.

** Latency / buffering.**
`SentenceChunker` buffers tokens until a sentence boundary and at least
`min_sentence_chars` characters. Tunable in `pipeline.yaml` connector config:
lower value = less Time-To-First-Audio, higher value = smoother audio.

**Voice/speaker parameter routing.**
`default_voice` and `default_language` are injected from the connector
config when Stage 0 has no concept of speaker. Per-request override is
passed via `request.additional_information`.

## Known Limitations / Follow-up

- Online serving (`/v1/audio/speech`) integration is a follow-up
- Richer per-turn voice routing (e.g. LLM-emitted speaker tags) is out of scope
