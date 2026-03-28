# Composable Text-LLM → TTS Pipeline

This directory demonstrates **RFC Theme 2: TTS as a Composable Layer**,
enabling any vLLM text model to be paired with any TTS decoder without
requiring a built-in talker stage.

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

Any combination is supported by providing the appropriate stage config YAML:

| Stage 0 (text LLM) | Stage 1 (TTS decoder) | Config |
|---|---|---|
| Llama-3.1-8B | Qwen3-TTS-1.7B | `llama3_qwen3tts.yaml` |
| Mistral-7B | CosyVoice3 | (add your own YAML) |
| Domain-finetuned LLM | FishSpeech | (add your own YAML) |

## Reference Example: Llama-3.1-8B + Qwen3-TTS

This example uses:
- **Stage 0**: `meta-llama/Llama-3.1-8B-Instruct`
- **Stage 1**: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- **Config**: `vllm_omni/model_executor/stage_configs/llama3_qwen3tts.yaml`

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

# Streaming
python end2end.py --prompt "Tell me a story." --streaming
```

Available voices: `aiden`, `dylan`, `eric`, `ono_anna`, `ryan`,
`serena`, `sohee`, `uncle_fu`, `vivian`

## Adding a New LLM + TTS Combination

To wire a different pair, create a new YAML based on `llama3_qwen3tts.yaml`:

1. Set `engine_args.model` in Stage 0 to your text LLM
2. Set `engine_args.model` in Stage 1 to your TTS model
3. Keep `custom_process_next_stage_input_func: vllm_omni.model_executor.stage_input_processors.text_tts_bridge.text2tts`
4. Tune `min_sentence_chars` in `connectors.connector_of_shared_memory.extra`

## Design (RFC answers)

**Q1 — Bridge as stage-level processor, not a new stage type.**
The bridge is a `custom_process_next_stage_input_func` hook on Stage 0,
reusing the existing `async_chunk` / `OmniChunkTransferAdapter` framework.

**Q2 — Latency / buffering.**
`SentenceChunker` buffers tokens until a sentence boundary and
`min_sentence_chars` characters. Tunable in the YAML connector config.

**Q3 — Voice/speaker parameter routing.**
`default_voice` / `default_language` injected from YAML connector config.
Per-request override via `request.additional_information`.

## Known Limitations / Follow-up

- Online serving (`/v1/audio/speech`) integration is a follow-up
- Richer per-turn voice routing (LLM-emitted tags) is out of scope
