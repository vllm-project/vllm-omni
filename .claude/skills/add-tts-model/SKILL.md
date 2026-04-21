---
name: add-tts-model
description: "Integrate a new text-to-speech model into vLLM-Omni from HuggingFace reference implementation through production-ready serving with streaming and CUDA graph acceleration. Use when adding a new TTS model, wiring stage separation for speech synthesis, enabling online voice generation serving, debugging TTS integration behavior, or building audio output pipelines."
---

# TTS Model Integration Workflow

## Overview

```
HF Reference -> Stage Separation -> Online Serving -> Async Chunk -> CUDA Graph
   (Phase 1)      (Phase 2)          (Phase 3)        (Phase 4)     (Phase 5)
```

## Phase 1: HuggingFace Reference

**Goal**: Understand the reference implementation and verify it produces correct audio.

### Steps

1. **Run the reference model** end-to-end using the official HuggingFace / GitHub code
2. **Document the architecture**:
   - What are the sub-models? (AR decoder, codec decoder, vocoder, etc.)
   - What is the token vocabulary? (semantic codes, RVQ codebooks, special tokens)
   - What is the output format? (sample rate, channels, codec type)
3. **Capture reference outputs** for comparison during integration
4. **Identify the config structure**: `config.json` fields, `model_type`, sub-model configs

### Key Questions

- How many codebooks? What are the codebook sizes?
- What special tokens exist? (`<|voice|>`, `<|audio_start|>`, `<|im_end|>`, etc.)
- What is the token-to-ID mapping for codec codes?
- What is the hop length / frame rate of the codec?
- Does the model support voice cloning? How? (reference audio encoding, speaker embeddings, etc.)

### Deliverables

- Working reference script that produces audio
- Architecture diagram / notes
- Token vocabulary mapping
- Reference audio samples for regression testing

## Phase 2: Stage Separation (Offline Inference)

**Goal**: Split the model into vLLM-Omni stages and get offline inference working.

### Steps

1. **Register the model** in `vllm_omni/model_executor/models/registry.py`
2. **Create config classes** (`configuration_<model>.py`) with `model_type` registration
3. **Implement Stage 0** (AR model):
   - Subclass appropriate base (e.g., wrap Qwen3 decoder layers)
   - Implement `forward()` for autoregressive token generation
   - Handle special token logic (start/stop tokens, codec token mapping)
   - If dual-AR (like Fish Speech), implement Fast AR as a nested module
4. **Implement Stage 1** (Decoder):
   - Load codec weights (may need lazy loading from separate checkpoint)
   - Implement `forward()`: codec codes -> audio waveform
   - Return `OmniOutput` with `multimodal_outputs`
5. **Create stage config YAML** defining both stages, memory allocation, and model paths
6. **Create stage input processor** for prompt building
7. **Write end2end.py** test script

### Critical Parameters to Get Right

| Parameter | Impact if Wrong |
|-----------|----------------|
| Hop length | Audio duration wrong, streaming noise |
| Token ID mapping | Garbage codes -> noise output |
| Codebook count/size | Shape mismatch crashes |
| Stop token | Generation never stops or stops too early |
| dtype / autocast | Numerical issues, silent quality degradation |
| Repetition penalty | Must match reference (often 1.0 for TTS) |

### Debugging Priority (from experience)

When audio output is wrong, check in this order:

1. **RoPE / attention**: Are position encodings correct? Is the attention mask right?
2. **Normalization**: RMSNorm epsilon, layer norm placement (pre vs post)
3. **Hop length**: Product of all upsample rates in the codec decoder
4. **Token mapping**: Are codec IDs correctly offset from the vocabulary base?
5. **Sampling parameters**: Temperature, top_k, top_p, repetition_penalty
6. **Tensor layout**: Codebook-major vs frame-major ordering
7. **dtype**: Float32 for codec decoders (autocast can corrupt audio)

### Deliverables

- Model files in `vllm_omni/model_executor/models/<model_name>/`
- Stage config YAML
- Working `end2end.py` with correct audio output
- README.md in the example directory

## Phase 3: Online Serving

**Goal**: Expose the model via `/v1/audio/speech` API endpoint.

### Steps

1. **Register in `serving_speech.py`**:
   - Add model stage name to `_TTS_MODEL_STAGES` set
   - Add model detection flag (e.g., `_is_fish_speech`)
   - Implement prompt builder method (e.g., `_build_fish_speech_prompt()`)
2. **Handle model-specific parameters**:
   - Voice cloning: `ref_audio` encoding and prompt injection
   - `max_new_tokens` override in sampling params
   - Model-specific default values
3. **Create client scripts**: `speech_client.py`, `run_server.sh`
4. **Test all response formats**: wav, mp3, flac, pcm
5. **Add Gradio demo**: Interactive web UI with streaming support

### Voice Cloning Pattern

```python
import base64
from pathlib import Path

def build_voice_clone_prompt(ref_audio_path: str, text: str, codec) -> list:
    """Build prompt with reference audio for voice cloning in serving_speech.py."""
    audio_bytes = Path(ref_audio_path).read_bytes()
    codes = codec.encode(audio_bytes)  # Encode on CPU using model's codec (e.g., DAC)
    token_ids = [code + codec.vocab_offset for code in codes.flatten().tolist()]
    return [
        {"role": "system", "content": f"<|voice|>{''.join(chr(t) for t in token_ids)}"},
        {"role": "user", "content": text},
    ]
```

### Deliverables

- Updated `serving_speech.py` with model-specific prompt builder
- Client scripts and server launcher
- Gradio demo with streaming and voice cloning UI
- Documentation (offline + online serving docs)

## Phase 4: Async Chunk (Streaming)

**Goal**: Enable inter-stage streaming so audio chunks are produced while AR generation continues.

### Steps

1. **Update stage config YAML**:
   ```yaml
   async_chunk: true
   codec_chunk_frames: 25      # frames per chunk
   codec_left_context_frames: 25  # overlap for smooth boundaries
   ```
2. **Implement chunk handling in Stage 1**:
   - Accept partial input (chunk of codec codes)
   - Handle left context for smooth audio boundaries
   - Return partial audio in `OmniOutput`
3. **Test streaming**:
   - Verify audio quality matches non-streaming output
   - Check for artifacts at chunk boundaries
   - Measure TTFA (time to first audio)
4. **Update online serving** to support `stream=true` with PCM output

### Streaming Architecture

```
Stage 0 (AR)                    Stage 1 (Decoder)
  |                                |
  |-- chunk 0 (25 frames) ------> decode -> audio chunk 0 -> client
  |-- chunk 1 (25 frames) ------> decode -> audio chunk 1 -> client
  |-- chunk 2 (25 frames) ------> decode -> audio chunk 2 -> client
  ...
```

### Key Considerations

- **Left context overlap**: Prevents audible artifacts at chunk boundaries
- **Hop length matters**: `context_audio_samples = context_frames * hop_length`
- **First chunk latency**: Can use larger initial chunk for better quality, then smaller chunks

### Deliverables

- Updated stage config with async_chunk enabled
- Smooth streaming audio without boundary artifacts
- TTFA metrics

## Phase 5: CUDA Graph Acceleration

**Goal**: Capture the AR loop as a CUDA graph for significant speedup.

### Steps

1. **Identify the hot loop**: The AR decoding loop that runs N steps per token
2. **Create static buffers**:
   - KV caches with fixed max sequence length
   - Pre-built causal masks and position tensors per step
   - Static input/output tensors
3. **Implement graph capture**:
   - Warm up with real data
   - Capture the forward pass
   - Replay with updated inputs
4. **Handle constraints**:
   - Use `torch.argmax` instead of `torch.multinomial` (graph-safe)
   - Fixed batch size (fall back to eager for other sizes)
   - No dynamic control flow inside the graph

### Example: Code Predictor CUDA Graph (Qwen3-TTS)

```python
import torch

class CodePredictorGraph:
    """Captures the 16-step code predictor AR loop as a single CUDA graph."""

    def setup_graph(self, device: torch.device, kv_heads: int = 4, head_dim: int = 64):
        self.num_steps = 16
        self.kv_cache = torch.zeros(1, kv_heads, self.num_steps, head_dim, device=device)
        self.positions = torch.arange(self.num_steps, device=device)
        self.causal_mask = torch.tril(torch.ones(self.num_steps, self.num_steps, device=device))
        self.input_buf = torch.zeros(1, 1, kv_heads * head_dim, device=device)
        self.output_buf = torch.zeros(1, self.num_steps, device=device, dtype=torch.long)
        # Warm up, then: self.graph = torch.cuda.CUDAGraph(); self.graph.capture(...)

    def run_graph(self, initial_input: torch.Tensor) -> torch.Tensor:
        self.input_buf.copy_(initial_input)
        self.graph.replay()
        return self.output_buf.clone()
```

### Performance Expectations

Based on Qwen3-TTS code predictor experience:
- **3-5x speedup** for the graphed component
- Only effective for fixed batch sizes (typically batch_size=1)
- Falls back to eager mode for unsupported configurations

### Deliverables

- CUDA graph implementation for the AR hot loop
- Benchmark script comparing eager vs graph performance
- Documentation of constraints and fallback behavior

## Integration Checklist

Use this checklist when integrating a new TTS model:

### Phase 1: HF Reference
- [ ] Reference model runs and produces correct audio
- [ ] Architecture documented (stages, codebooks, tokens, sample rate)
- [ ] Reference audio samples saved for comparison

### Phase 2: Stage Separation
- [ ] Model registered in `registry.py`
- [ ] Config classes created with `model_type` registration
- [ ] Stage 0 (AR) implemented and generates correct tokens
- [ ] Stage 1 (Decoder) produces correct audio from tokens
- [ ] Stage config YAML created
- [ ] `end2end.py` produces audio matching reference quality
- [ ] README.md written

### Phase 3: Online Serving
- [ ] Model added to `serving_speech.py`
- [ ] Prompt builder handles text input correctly
- [ ] Voice cloning works (if supported)
- [ ] All response formats work (wav, mp3, flac, pcm)
- [ ] Client scripts and server launcher created
- [ ] Gradio demo working
- [ ] Documentation added (offline + online docs, nav, supported models)

### Phase 4: Async Chunk
- [ ] Stage config updated with `async_chunk: true`
- [ ] Stage 1 handles partial chunks correctly
- [ ] No audio artifacts at chunk boundaries
- [ ] Streaming via API (`stream=true`) works
- [ ] TTFA measured and acceptable

### Phase 5: CUDA Graph
- [ ] Hot loop identified and profiled
- [ ] Static buffers allocated
- [ ] Graph captured and replays correctly
- [ ] Benchmark shows meaningful speedup
- [ ] Fallback to eager works for unsupported configs

## References

- [TTS audio skill](../vllm-omni-audio-tts/SKILL.md) -- supported models and usage
- [Fish Speech integration](../vllm-omni-audio-tts/references/fish-speech.md) -- complete example of Phases 1-3
- [Qwen3-TTS reference](../vllm-omni-audio-tts/references/qwen-tts.md) -- complete example of all 5 phases
- [Adding a TTS model (developer guide)](https://github.com/vllm-project/vllm-omni/blob/main/docs/contributing/model/adding_tts_model.md)
