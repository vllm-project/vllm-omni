# Llama-3.1 + Qwen3-TTS: Composable Text-LLM → TTS Pipeline

This example demonstrates **RFC Theme 2: TTS as a Composable Layer** from the
[vllm-omni TTS Development Roadmap](https://github.com/vllm-project/vllm-omni/issues/XXX).

The key idea: **any vLLM text model can be paired with any TTS decoder**,
without requiring a built-in "talker" stage in the base model.

```
[Llama-3.1-8B]  →  [SentenceChunker bridge]  →  [Qwen3-TTS-1.7B]  →  audio
   Stage 0              async_chunk                  Stage 1
```

---

## Models

| Stage | Model | Role |
|-------|-------|------|
| 0 | `meta-llama/Llama-3.1-8B-Instruct` | Text generation (any vLLM text model works) |
| 1 | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Speech synthesis |

---

## Setup

```bash
pip install vllm-omni soundfile
# Qwen3-TTS also needs:
pip install onnxruntime sox
```

---

## Usage

```bash
cd examples/offline_inference/llama3_qwen3tts

# Basic
python end2end.py --prompt "Explain how transformers work."

# Custom voice and language
python end2end.py \
  --prompt "Bonjour, comment allez-vous?" \
  --voice serena \
  --language French

# Streaming (low-latency, audio chunks arrive progressively)
python end2end.py \
  --prompt "Tell me a short story." \
  --streaming

# Override model paths (local cache)
python end2end.py \
  --llm-model /path/to/Llama-3.1-8B-Instruct \
  --tts-model /path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --prompt "Hello from a local model."
```

Output WAV files are written to `output_audio/` by default (`--output-dir` to change).

Available voices for `CustomVoice`: `aiden`, `dylan`, `eric`, `ono_anna`, `ryan`,
`serena`, `sohee`, `uncle_fu`, `vivian`.

---

## Design

### RFC Q1: Bridge as a stage-level processor

The bridge (`text_tts_bridge.py`) is implemented as a `custom_process_input_func`
hook reusing the existing `async_chunk` framework — not a new stage type. This
keeps the pipeline config system unchanged.

### RFC Q2: Latency / buffering

`SentenceChunker` accumulates tokens until a sentence boundary **and** at least
`min_sentence_chars` characters have been buffered, then flushes a chunk to Stage 1.
This is configurable in the YAML:

```yaml
bridge:
  min_sentence_chars: 40   # tune for latency vs. audio quality
  sentence_delimiters: [".", "!", "?", "。", "！", "？"]
```

Lower `min_sentence_chars` → lower Time-To-First-Audio (TTFA), more TTS cold-starts.
Higher → smoother audio, higher TTFA.

### RFC Q3: Voice/speaker parameter routing

Stage 0 (Llama) has no concept of TTS voices. The bridge injects `default_voice`
and `default_language` from the YAML config into every Stage 1 chunk. Per-request
overrides are passed via `extra_body`:

```python
# Via API
{"tts_voice": "serena", "tts_language": "French"}
# passed as extra in the Stage 0 input dict; threaded through by the orchestrator
```

This is a **simple default-inject strategy** and is explicitly a starting point.
Future work (tracked separately) includes a richer voice-routing API.

---

## Files

```
llama3_qwen3tts/
├── end2end.py                    # Offline + streaming demo script
├── README.md                     # This file

vllm_omni/model_executor/
├── stage_configs/
│   └── llama3_qwen3tts.yaml      # Pipeline stage config
└── stage_input_processors/
    └── text_tts_bridge.py        # SentenceChunker + text2tts hook
```

---

## Known Limitations / Future Work

- **Voice routing API**: Currently voice/speaker params are injected as static
  defaults. A proper per-turn voice-routing API (e.g. letting the LLM output
  `<voice>serena</voice>` tags) is out of scope for this PR.
- **Online serving**: This example covers offline inference only. Hooking the
  bridge into the `/v1/audio/speech` online endpoint is a follow-up.
- **Other TTS backends**: The bridge format is designed to be TTS-agnostic;
  adapting to `CosyVoice3` or `FishSpeech` requires only a new `build_tts_input()`
  variant and a matching stage config.
