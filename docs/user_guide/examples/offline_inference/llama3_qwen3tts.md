# Llama-3.1 + Qwen3-TTS: Composable Text-LLM → TTS Pipeline

This example demonstrates **RFC Theme 2: TTS as a Composable Layer**,
pairing any vLLM text model with any TTS decoder without requiring a
built-in talker stage.
```
[Llama-3.1-8B]  →  [SentenceChunker bridge]  →  [Qwen3-TTS-1.7B]  →  WAV
   Stage 0              async_chunk                  Stage 1
```

## Usage
```bash
cd examples/offline_inference/llama3_qwen3tts

# Basic
python end2end.py --prompt "Explain how transformers work."

# Custom voice and language
python end2end.py --prompt "Bonjour." --voice serena --language French

# Streaming
python end2end.py --prompt "Tell me a story." --streaming
```

## Stage config
```
vllm_omni/model_executor/stage_configs/llama3_qwen3tts.yaml
```

See the [example README](../../../examples/offline_inference/llama3_qwen3tts/README.md)
for full design rationale and voice options.
