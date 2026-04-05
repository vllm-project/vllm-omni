# VibeVoice TTS — vLLM-Omni Integration Plan

## 1. Architecture Comparison: Voxtral TTS vs VibeVoice TTS

Both models follow a **two-stage pipeline** where an autoregressive LLM generates intermediate audio representations that are then decoded into PCM waveforms. However, they differ in critical implementation details.

### 1.1 Structural Similarities

| Aspect | Voxtral TTS | VibeVoice TTS |
|--------|-------------|---------------|
| Pipeline | 2-stage (LLM → audio decoder) | 2-stage (LLM → audio decoder) |
| LLM role | AR next-token prediction, hidden states fed to audio head | AR next-token prediction, hidden states fed to diffusion head |
| Audio head | Generates audio representations per decode step from LLM hidden state | Generates audio representations per decode step from LLM hidden state |
| Stage 0 output | Intermediate audio codes/latents streamed to Stage 1 | Intermediate acoustic latents streamed to Stage 1 |
| Stage 1 role | Decode intermediate representation → PCM waveform | Decode intermediate representation → PCM waveform |
| vLLM trick | `fake_logits` to force sampler to emit EOS or audio-placeholder token | Same `fake_logits` approach needed |
| Multimodal output | `OmniOutput` with `multimodal_outputs` per step | Same pattern needed |

### 1.2 Key Differences

| Aspect | Voxtral TTS | VibeVoice TTS |
|--------|-------------|---------------|
| **LLM backbone** | Mistral (custom causal LM) | Qwen2.5 (1.5B or larger) |
| **Tokenizer** | `mistral_common` (`tokenizer_mode: mistral`) | Qwen2.5 tokenizer (standard HF `tokenizer_mode: auto`) |
| **Config format** | Mistral-style `params.json` (`config_format: mistral`) | Standard HF `config.json` (`config_format: auto`) |
| **Weight format** | Mistral format (`load_format: mistral`) | Standard safetensors (`load_format: auto`) |
| **Audio head type** | Flow-matching ODE (Euler, ~8 steps, CFG) | DDPM diffusion + DPM-Solver (~20 inference steps) |
| **Audio representation** | **Discrete** multi-codebook codes (semantic code + FSQ acoustic codes) | **Continuous** latent vectors (acoustic_vae_dim=64, semantic_vae_dim=128) |
| **Audio codec** | Single custom neural tokenizer with `SemanticCodebook` + `AcousticCodebook` | Two separate tokenizers: `AcousticTokenizer` (σ-VAE, ~680M params) + `SemanticTokenizer` (~340M params) |
| **Codec architecture** | Transformer-based with ALiBi, flash attention | ConvNeXt-like (7 stages, depthwise causal conv, RMSNorm) — no attention |
| **Codec frame rate** | Model-dependent | 7.5 Hz (3200x downsample from 24kHz) |
| **LLM→audio projection** | Direct: LLM hidden → acoustic transformer → codes | Via `SpeechConnector` (Linear→RMSNorm→Linear): acoustic + semantic connectors project latents to LLM hidden size |
| **Multi-speaker** | Voice cloning via voice embeddings or reference audio | Up to 4 built-in speakers, speaker tags in text |
| **HF `model_type`** | `"voxtral_tts"` | `"vibevoice"` |
| **HF architecture** | `VoxtralTTSForConditionalGeneration` | `VibeVoiceForConditionalGeneration` |
| **Sampling rate** | Model-dependent | 24,000 Hz |

### 1.3 VibeVoice Inference Flow (Conceptual)

```
Text input
    ↓
[Qwen2.5 tokenizer] → token_ids
    ↓
[Qwen2.5 LLM] → hidden_states (per AR step)
    ↓
[DiffusionHead + DPMSolver] → continuous acoustic latent (dim=64) per frame
    ↓  (streamed to Stage 1)
[AcousticTokenizer.decode()] → PCM waveform (24kHz)
```

During training, the LLM also receives **semantic connector** outputs (from the semantic tokenizer encoding of speech) and **acoustic connector** outputs (from the acoustic tokenizer encoding of speech) as input embeddings for speech tokens. At inference, each AR step produces an LLM hidden state that is passed through the diffusion head to generate an acoustic latent vector, which is then decoded by the acoustic tokenizer's decoder into waveform audio.

---

## 2. Files to Create / Modify

All new files live under `vllm_omni/model_executor/models/vibevoice_tts/` unless otherwise noted.

### 2.1 New Files

| # | File | Purpose |
|---|------|---------|
| 1 | `vibevoice_tts/__init__.py` | Package init |
| 2 | `vibevoice_tts/configuration_vibevoice_tts.py` | `VibeVoiceTTSConfig` + optional config parser |
| 3 | `vibevoice_tts/vibevoice_tts.py` | Top-level model wrapper (`VibeVoiceTTSForConditionalGeneration`): routes to generation or decoder sub-model based on `model_stage` |
| 4 | `vibevoice_tts/vibevoice_tts_audio_generation.py` | Stage 0: Qwen2.5 LLM + DiffusionHead + `SpeechConnector`s + `fake_logits` + `compute_mm_logits` |
| 5 | `vibevoice_tts/vibevoice_tts_audio_tokenizer.py` | Stage 1: Acoustic tokenizer decoder that converts continuous latents → PCM |
| 6 | `vibevoice_tts/vibevoice_tts_diffusion_head.py` | Diffusion head implementation (adaLN FFN blocks, timestep embedder, DPM-Solver integration) |
| 7 | `model_executor/stage_configs/vibevoice_tts.yaml` | Two-stage pipeline YAML |
| 8 | `model_executor/stage_input_processors/vibevoice_tts.py` | `generator2tokenizer` + `generator2tokenizer_async_chunk` for inter-stage latent streaming |
| 9 | *(optional)* `vibevoice_tts/cuda_graph_diffusion_head_wrapper.py` | CUDA graph wrapper for the diffusion head (performance optimization) |

### 2.2 Files to Modify

| # | File | Change |
|---|------|--------|
| 1 | `model_executor/models/registry.py` | Register `VibeVoiceForConditionalGeneration`, `VibeVoiceTTSAudioGeneration`, `VibeVoiceTTSAudioTokenizer` in `_OMNI_MODELS` |
| 2 | `engine/arg_utils.py` | Import `VibeVoiceTTSConfig`, register in `_register_omni_hf_configs` under `model_type="vibevoice"` |
| 3 | `entrypoints/openai/serving_speech.py` | Add `_VIBEVOICE_TTS_MODEL_STAGES`, validation, prompt builder, speaker loading |
| 4 | `config/stage_config.py` | Map pipeline type `"vibevoice_tts"` → subdirectory `vibevoice_tts` (if not auto-discovered) |

---

## 3. Detailed Implementation Steps

### Step 1: Configuration (`configuration_vibevoice_tts.py`)

**Goal:** Parse the HuggingFace `config.json` from `microsoft/VibeVoice-1.5B` into a vLLM-compatible config object.

**Key design decisions:**
- VibeVoice uses **standard HF `config.json`** (not Mistral-style `params.json`), so no custom `config_format` parser is needed. Use `config_format: auto`.
- The config contains 4 sub-configs: `decoder_config` (Qwen2), `acoustic_tokenizer_config`, `semantic_tokenizer_config`, `diffusion_head_config`.

**Implementation:**

```python
class VibeVoiceTTSConfig(PretrainedConfig):
    model_type = "vibevoice"

    def __init__(self, decoder_config=None, acoustic_tokenizer_config=None,
                 semantic_tokenizer_config=None, diffusion_head_config=None,
                 acoustic_vae_dim=64, semantic_vae_dim=128, **kwargs):
        # Parse sub-configs from dicts or config objects
        # decoder_config → Qwen2Config
        # acoustic_tokenizer_config → VibeVoiceAcousticTokenizerConfig
        # semantic_tokenizer_config → VibeVoiceSemanticTokenizerConfig
        # diffusion_head_config → VibeVoiceDiffusionHeadConfig
        ...

    def get_text_config(self, decoder=False):
        """Required by vLLM for memory profiling — return the Qwen2 decoder config."""
        return self.decoder_config
```

**Notes:**
- Reuse the sub-config classes from the upstream VibeVoice repo (`VibeVoiceAcousticTokenizerConfig`, `VibeVoiceSemanticTokenizerConfig`, `VibeVoiceDiffusionHeadConfig`) or define simplified versions.
- `get_text_config()` is critical for vLLM's memory profiler to read `hidden_size`, `num_attention_heads`, etc.
- Store `audio_config` as a flat dict with: `sampling_rate` (24000), `frame_rate` (7.5), `acoustic_vae_dim` (64), `semantic_vae_dim` (128), `ddpm_num_inference_steps` (20), supported speakers, etc.

### Step 2: Model Registry (`registry.py`)

Add three architecture entries to `_OMNI_MODELS`:

```python
## VibeVoice TTS
"VibeVoiceForConditionalGeneration": (
    "vibevoice_tts",
    "vibevoice_tts",
    "VibeVoiceTTSForConditionalGeneration",
),
"VibeVoiceTTSAudioGeneration": (
    "vibevoice_tts",
    "vibevoice_tts_audio_generation",
    "VibeVoiceTTSAudioGenerationForConditionalGeneration",
),
"VibeVoiceTTSAudioTokenizer": (
    "vibevoice_tts",
    "vibevoice_tts_audio_tokenizer",
    "VibeVoiceTTSAudioTokenizer",
),
```

**Why three entries:**
- `VibeVoiceForConditionalGeneration` — the HF architecture string in `config.json`. The top-level wrapper routes to one of the two below based on `model_stage`.
- `VibeVoiceTTSAudioGeneration` — Stage 0 (LLM + diffusion head).
- `VibeVoiceTTSAudioTokenizer` — Stage 1 (acoustic decoder → PCM).

### Step 3: Config Registration (`engine/arg_utils.py`)

In `_register_omni_hf_configs()`:

```python
from vllm_omni.model_executor.models.vibevoice_tts.configuration_vibevoice_tts import (
    VibeVoiceTTSConfig,
)

# Register with AutoConfig and vLLM's _CONFIG_REGISTRY
("vibevoice", VibeVoiceTTSConfig),
```

### Step 4: Stage 0 — Audio Generation (`vibevoice_tts_audio_generation.py`)

This is the largest and most complex file. It implements the LLM + diffusion head stage.

#### 4.1 Core Class: `VibeVoiceTTSAudioGenerationForConditionalGeneration`

```python
class VibeVoiceTTSAudioGenerationForConditionalGeneration(nn.Module):
    def __init__(self, *, vllm_config, prefix=""):
        # 1. Language model: init Qwen2 via vLLM's init_vllm_registered_model()
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model.language_model")
        )

        # 2. LM head for text logits (tied weights with embeddings)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # 3. Diffusion head (custom, not from vLLM registry)
        self.diffusion_head = VibeVoiceDiffusionHead(config.diffusion_head_config)

        # 4. DPM-Solver noise scheduler for inference
        self.noise_scheduler = DPMSolverMultistepScheduler(...)

        # 5. Speech connectors
        self.acoustic_connector = SpeechConnector(acoustic_vae_dim, hidden_size)
        self.semantic_connector = SpeechConnector(semantic_vae_dim, hidden_size)

        # 6. Acoustic tokenizer (encoder only — for voice cloning / ref audio)
        self.acoustic_tokenizer = ...  # Optional: only if voice cloning is supported

        # 7. Scaling/bias buffers for latent normalization
        self.speech_scaling_factor = ...
        self.speech_bias_factor = ...
```

#### 4.2 `compute_mm_logits()` — The Diffusion Decode Step

This is the core per-step function. For each AR decode step:

1. Extract the last hidden state from the LLM.
2. Run the **diffusion head** with DPM-Solver to generate an acoustic latent vector:
   ```
   noise → [DPM-Solver, N steps] → acoustic_latent (dim=64)
   ```
3. Determine if this is an end-of-audio step (e.g., LLM emitted an EOS-like token).
4. Return the acoustic latent as multimodal output.

**Key difference from Voxtral:** Voxtral's flow-matching uses 8 Euler steps and produces *discrete* codes. VibeVoice's DDPM uses ~20 DPM-Solver steps and produces *continuous* latents. The continuous latents are float tensors, not integer code indices.

```python
def compute_mm_logits(self, hidden_states, ...):
    # hidden_states: [batch, hidden_size] — last LLM hidden for current step
    condition = hidden_states  # The LLM output conditions the diffusion

    # Run diffusion reverse process
    latent = self._run_diffusion_inference(condition)
    # latent: [batch, acoustic_vae_dim=64]

    # Check for end-of-audio
    fake_eos = self._check_eos(...)

    return latent, fake_eos
```

#### 4.3 `_run_diffusion_inference()` — DPM-Solver Loop

```python
def _run_diffusion_inference(self, condition):
    batch_size = condition.shape[0]
    latent_size = self.config.diffusion_head_config.latent_size  # 64

    # Start from pure noise
    sample = torch.randn(batch_size, latent_size, device=condition.device, dtype=condition.dtype)

    # Set up scheduler timesteps
    self.noise_scheduler.set_timesteps(self.config.diffusion_head_config.ddpm_num_inference_steps)

    for t in self.noise_scheduler.timesteps:
        timesteps = t.expand(batch_size).to(condition.device)
        model_output = self.diffusion_head(sample, timesteps, condition)
        sample = self.noise_scheduler.step(model_output, t, sample).prev_sample

    # Undo scaling/bias normalization
    sample = sample / self.speech_scaling_factor - self.speech_bias_factor

    return sample
```

#### 4.4 `fake_logits_for_audio_tokens()` — Sampler Control

Same pattern as Voxtral: produce fake logits that force the vLLM sampler to pick either the **audio placeholder token** (to continue generation) or **EOS** (to stop).

```python
def fake_logits_for_audio_tokens(self, fake_eos_mask, vocab_size, audio_token_id):
    logits = torch.full((batch_size, vocab_size), -float('inf'), ...)
    # For sequences that should continue: set audio_token_id to high logit
    logits[~fake_eos_mask, audio_token_id] = 0.0
    # For sequences that should end: set EOS to high logit
    logits[fake_eos_mask, eos_token_id] = 0.0
    return logits
```

#### 4.5 `load_weights()` — Weight Remapping

Handle the weight name mapping between HF checkpoint and the vLLM model structure:
- `model.language_model.*` → `language_model.*` (Qwen2 weights)
- `model.prediction_head.*` → `diffusion_head.*`
- `model.acoustic_connector.*` → `acoustic_connector.*`
- `model.semantic_connector.*` → `semantic_connector.*`
- `model.acoustic_tokenizer.*` → `acoustic_tokenizer.*` (encoder weights, if loading)
- `model.semantic_tokenizer.*` → `semantic_tokenizer.*` (encoder weights, if loading)
- `lm_head.weight` → tied to embeddings

### Step 5: Diffusion Head (`vibevoice_tts_diffusion_head.py`)

Port the diffusion head from the upstream VibeVoice repo. This is relatively straightforward:

```python
class VibeVoiceDiffusionHead(nn.Module):
    """
    AdaLN-modulated FFN blocks conditioned on timestep + LLM hidden state.

    Architecture per layer:
        x = x + gate * FFN(modulate(norm(x), shift, scale))
    where (shift, scale, gate) = adaLN_modulation(condition + timestep_emb)
    """
    def __init__(self, config):
        self.noisy_images_proj = nn.Linear(latent_size, hidden_size)
        self.cond_proj = nn.Linear(hidden_size, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.layers = nn.ModuleList([HeadLayer(...) for _ in range(config.head_layers)])  # 4 layers
        self.final_layer = FinalLayer(...)

    def forward(self, noisy_latent, timesteps, condition):
        x = self.noisy_images_proj(noisy_latent)
        t = self.t_embedder(timesteps)
        c = self.cond_proj(condition) + t
        for layer in self.layers:
            x = layer(x, c)
        return self.final_layer(x, c)
```

Also port the `DPMSolverMultistepScheduler` or use the one from the `diffusers` library (it's a standard component).

### Step 6: Stage 1 — Audio Tokenizer Decoder (`vibevoice_tts_audio_tokenizer.py`)

**Goal:** Receive continuous acoustic latents from Stage 0, decode them to PCM waveforms.

```python
class VibeVoiceTTSAudioTokenizer(nn.Module):
    """
    Stage 1: Acoustic tokenizer decoder.
    Receives continuous latent frames [batch, num_frames, 64] from Stage 0,
    runs the acoustic tokenizer decoder to produce PCM waveforms.
    """
    def __init__(self, *, vllm_config, prefix=""):
        # Load only the DECODER half of the acoustic tokenizer
        self.acoustic_decoder = TokenizerDecoder(config)
        self.sampling_rate = 24000
        self.downsample_factor = 3200  # 24000 / 7.5 Hz

    def forward(self, latents, ...):
        """
        latents: [batch, num_frames, acoustic_vae_dim]
        returns: list of 1-D float tensors (PCM waveforms)
        """
        # latents come in as [B, T, D], decoder expects [B, D, T]
        latents = latents.transpose(1, 2)
        waveforms = self.acoustic_decoder(latents)
        # waveforms: [B, 1, num_samples]
        return [w.squeeze(0).squeeze(0) for w in waveforms]
```

**Key difference from Voxtral Stage 1:**
- Voxtral Stage 1 receives **discrete integer codes** (multi-codebook tokens) that need to be parsed, unshifted, and fed through a codec decoder with codebook embeddings.
- VibeVoice Stage 1 receives **continuous float latents** that are directly fed to the ConvNeXt-based decoder. No codebook lookup is needed.
- The decoder architecture is purely convolutional (no attention), so it should be simpler and potentially faster.

**Streaming support:**
- The VibeVoice acoustic tokenizer supports streaming via `VibeVoiceTokenizerStreamingCache` (caches causal convolution states).
- For chunked decode in Stage 1, maintain a `VibeVoiceTokenizerStreamingCache` per request and feed latent chunks incrementally.
- Context frame handling is simpler than Voxtral because there are no discrete code boundaries — just overlap the latent frames.

### Step 7: Top-Level Wrapper (`vibevoice_tts.py`)

Same pattern as Voxtral's `VoxtralTTSForConditionalGeneration`: a routing class that loads the appropriate sub-model based on `model_stage`.

```python
class VibeVoiceTTSForConditionalGeneration(nn.Module):
    def __init__(self, *, vllm_config, prefix=""):
        model_stage = vllm_config.model_config.model_stage

        if model_stage == "audio_generation":
            self.model = VibeVoiceTTSAudioGenerationForConditionalGeneration(
                vllm_config=vllm_config, prefix=prefix
            )
        elif model_stage == "audio_tokenizer":
            self.model = VibeVoiceTTSAudioTokenizer(
                vllm_config=vllm_config, prefix=prefix
            )

    def forward(self, ...):
        return self.model(...)

    def compute_logits(self, ...):
        # Stage 0 only: delegate to audio generation model
        return self.model.compute_logits(...)

    def make_omni_output(self, ...):
        return self.model.make_omni_output(...)

    def load_weights(self, weights):
        # Route weights to the appropriate sub-model
        self.model.load_weights(weights)
```

### Step 8: Stage Input Processor (`stage_input_processors/vibevoice_tts.py`)

Bridges Stage 0 output (continuous latents) to Stage 1 input.

**Key differences from Voxtral:**
- Voxtral streams **discrete integer codes** with multiple codebooks, requiring flatten/unflatten and token offset arithmetic.
- VibeVoice streams **continuous float tensors** (shape `[num_frames, 64]`), which is simpler — just concatenate frames and chunk.

```python
def generator2tokenizer(stage0_output):
    """Non-streaming: collect all latent frames, pass to Stage 1."""
    all_latents = torch.stack(stage0_output.multimodal_frames)  # [T, 64]
    return {"latents": all_latents}

async def generator2tokenizer_async_chunk(stage0_output_stream, connector_extra):
    """Streaming: accumulate latent frames, chunk, and emit."""
    chunk_frames = connector_extra.get("codec_chunk_frames", 25)
    left_context = connector_extra.get("codec_left_context_frames", 25)

    buffer = []
    async for frame in stage0_output_stream:
        buffer.append(frame.multimodal_latent)  # [64]
        if len(buffer) >= chunk_frames:
            # Emit chunk with left context for streaming decode
            start = max(0, len(buffer) - chunk_frames - left_context)
            chunk = torch.stack(buffer[start:])
            yield {"latents": chunk, "context_frames": min(left_context, len(buffer) - chunk_frames)}
            buffer = buffer[-left_context:]  # Keep context for next chunk
```

### Step 9: Stage Config YAML (`stage_configs/vibevoice_tts.yaml`)

```yaml
async_chunk: true
stage_args:
  - stage_id: 0
    stage_type: llm
    runtime:
      process: true
      devices: "0"
    engine_args:
      max_num_seqs: 32
      model_stage: audio_generation
      model_arch: VibeVoiceForConditionalGeneration
      worker_type: ar
      worker_cls: vllm_omni.worker.gpu_ar_worker.GPUARWorker
      scheduler_cls: vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler
      gpu_memory_utilization: 0.8
      enforce_eager: false
      trust_remote_code: true
      async_scheduling: true
      engine_output_type: latent
      enable_prefix_caching: false
      tokenizer_mode: auto          # Qwen2.5 uses standard HF tokenizer
      config_format: auto            # Standard HF config.json
      load_format: auto              # Standard safetensors
      skip_mm_profiling: true
      enable_chunked_prefill: false
      max_model_len: 4096
      custom_process_next_stage_input_func: vllm_omni.model_executor.stage_input_processors.vibevoice_tts.generator2tokenizer_async_chunk
    output_connectors:
      to_stage_1: connector_of_shared_memory
    is_comprehension: true
    final_output: false
    final_output_type: text
    default_sampling_params:
      temperature: 0.0
      top_p: 1.0
      top_k: -1
      max_tokens: 4096
      seed: 42
      detokenize: true
      repetition_penalty: 1.1

  - stage_id: 1
    stage_type: llm
    runtime:
      process: true
      devices: "0"
    engine_args:
      max_num_seqs: 32
      model_stage: audio_tokenizer
      model_arch: VibeVoiceForConditionalGeneration
      worker_type: generation
      worker_cls: vllm_omni.worker.gpu_generation_worker.GPUGenerationWorker
      scheduler_cls: vllm_omni.core.sched.omni_generation_scheduler.OmniGenerationScheduler
      async_scheduling: false
      gpu_memory_utilization: 0.2
      enforce_eager: true
      trust_remote_code: true
      enable_prefix_caching: false
      skip_mm_profiling: true
      engine_output_type: audio
      tokenizer_mode: auto
      config_format: auto
      load_format: auto
      max_num_batched_tokens: 65536
      max_model_len: 65536
    engine_input_source: [0]
    is_comprehension: false
    final_output: true
    final_output_type: audio
    input_connectors:
      from_stage_0: connector_of_shared_memory
    tts_args:
      max_instructions_length: 500
    default_sampling_params:
      temperature: 0.9
      top_p: 0.8
      top_k: 40
      max_tokens: 4096
      seed: 42
      detokenize: true
      repetition_penalty: 1.05

runtime:
  enabled: true
  defaults:
    window_size: -1
    max_inflight: 1
  connectors:
    connector_of_shared_memory:
      name: SharedMemoryConnector
      extra:
        shm_threshold_bytes: 65536
        codec_streaming: true
        connector_get_sleep_s: 0.01
        connector_get_max_wait_first_chunk: 3000
        connector_get_max_wait: 300
        codec_chunk_frames: 25
        codec_chunk_frames_at_begin: 5
        codec_left_context_frames: 25
  edges:
    - from: 0
      to: 1
      window_size: -1
```

**Key differences from Voxtral YAML:**
- `tokenizer_mode: auto` (not `mistral`)
- `config_format: auto` (not `mistral`)
- `load_format: auto` (not `mistral`)
- `gpu_memory_utilization: 0.2` for Stage 1 (VibeVoice decoder is ~340M params, lighter than Voxtral's full codec)
- Stage 1 `max_tokens: 4096` (generous, may need tuning based on frame rate)

### Step 10: OpenAI Serving Layer (`serving_speech.py`)

#### 10.1 Add VibeVoice to TTS model stage sets

```python
_VIBEVOICE_TTS_MODEL_STAGES = {"vibevoice_audio_generation"}

_TTS_MODEL_STAGES = (
    _VOXTRAL_TTS_MODEL_STAGES
    | _QWEN3_TTS_MODEL_STAGES
    | _FISH_TTS_MODEL_STAGES
    | _COSYVOICE3_TTS_MODEL_STAGES
    | _OMNIVOICE_TTS_MODEL_STAGES
    | _VIBEVOICE_TTS_MODEL_STAGES      # <-- NEW
)
```

#### 10.2 Add TTS type detection

```python
if model_stage in _VIBEVOICE_TTS_MODEL_STAGES:
    return "vibevoice_tts"
```

#### 10.3 Add validation

```python
def _validate_vibevoice_tts_request(self, request):
    if not request.input or not request.input.strip():
        return "Input text cannot be empty"
    # VibeVoice supports up to 4 speakers via text markup
    # Validate speaker format if provided
    return None
```

#### 10.4 Add prompt builder

VibeVoice uses standard Qwen2.5 tokenization with special speaker tags embedded in the text. The prompt format depends on the upstream VibeVoice convention (e.g., `[Speaker1]`, `[Speaker2]`, etc.).

```python
async def _build_vibevoice_prompt(self, request):
    text = request.input
    voice = request.voice  # e.g., "Alice", "Bob"
    # Format: prepend speaker tag or use VibeVoice's expected prompt format
    prompt = f"<|speaker:{voice}|>{text}"
    return prompt
```

#### 10.5 Add speaker loading

```python
def _load_supported_speakers(self):
    if self._tts_model_type == "vibevoice_tts":
        # VibeVoice has built-in speakers
        return {"alice", "bob", "carol", "dave"}  # Up to 4
```

---

## 4. Implementation Order and Dependencies

```
Phase 1 — Foundation (can be done in parallel):
  ├── Step 1: configuration_vibevoice_tts.py
  ├── Step 5: vibevoice_tts_diffusion_head.py
  └── Step 6: vibevoice_tts_audio_tokenizer.py

Phase 2 — Core model (depends on Phase 1):
  ├── Step 4: vibevoice_tts_audio_generation.py
  └── Step 7: vibevoice_tts.py (top-level wrapper)

Phase 3 — Pipeline wiring (depends on Phase 2):
  ├── Step 2: registry.py modifications
  ├── Step 3: arg_utils.py registration
  ├── Step 8: stage_input_processors/vibevoice_tts.py
  └── Step 9: vibevoice_tts.yaml

Phase 4 — API integration (depends on Phase 3):
  └── Step 10: serving_speech.py

Phase 5 — Optimization (optional, after basic functionality works):
  └── Step 9 (optional): cuda_graph_diffusion_head_wrapper.py
```

---

## 5. Key Technical Challenges and Considerations

### 5.1 Continuous vs Discrete Latents

The most fundamental difference. Voxtral streams **integer code indices** between stages; VibeVoice streams **float tensors**. This affects:
- **SharedMemoryConnector:** Must handle float tensor serialization (not just int lists). Verify that the existing `connector_of_shared_memory` can transport arbitrary tensors, not just token IDs.
- **Stage input processor:** Simpler (no codebook offset math), but must handle float precision and potential numerical issues.
- **Batching in Stage 1:** Voxtral pads discrete code sequences; VibeVoice must pad continuous latent sequences (easier with standard tensor padding).

### 5.2 Diffusion Inference Speed

The diffusion head runs **20 DPM-Solver steps per frame** (vs Voxtral's 8 Euler steps for flow-matching). Each step is lightweight (4 FFN layers, no attention), but 20 steps could be a bottleneck. Mitigation:
- **Reduce inference steps:** DPM-Solver++ can often achieve good quality with 5-10 steps. Make `ddpm_num_inference_steps` configurable in the YAML.
- **CUDA graphs:** Wrap the diffusion head in a CUDA graph (analogous to `CUDAGraphAcousticTransformerWrapper` in Voxtral). The diffusion head has fixed input/output shapes per batch size, making it graph-friendly.
- **Batch diffusion steps:** If multiple sequences are being generated simultaneously, batch the diffusion head calls.

### 5.3 Memory Management

VibeVoice has more parameters than Voxtral's acoustic head:
- Acoustic tokenizer decoder: ~340M params
- Semantic tokenizer encoder: ~340M params (only needed for voice cloning)
- Diffusion head: relatively small (~50M params)

**Strategy:**
- Stage 0 loads: Qwen2.5 LM (~1.5B) + diffusion head (~50M) + connectors. The acoustic/semantic tokenizer encoders are only needed if voice cloning from raw audio is supported — otherwise skip them.
- Stage 1 loads: acoustic tokenizer decoder only (~340M). Set `gpu_memory_utilization` accordingly.

### 5.4 Latent Normalization (Scaling/Bias)

VibeVoice uses `speech_scaling_factor` and `speech_bias_factor` buffers that are computed from training data statistics. These are stored in the checkpoint as registered buffers. Ensure `load_weights()` correctly loads them, and that the diffusion head output is un-normalized before sending to Stage 1:
```python
acoustic_latent = diffusion_output / speech_scaling_factor - speech_bias_factor
```

### 5.5 End-of-Audio Detection

Voxtral uses a dedicated `[END_AUDIO]` semantic token from the flow-matching head. VibeVoice doesn't have a discrete semantic code — the LLM itself must signal end-of-audio via its standard EOS token or a special token in the Qwen2.5 vocabulary. The `fake_logits` mechanism must correctly detect when the LLM hidden state corresponds to an end-of-speech decision. Options:
- Use the LLM's own text logits: if the argmax of the *real* logits points to EOS or a speech-end token, set `fake_eos = True`.
- Train/fine-tune a small binary classifier on the hidden state to detect end-of-speech.
- Use a maximum frame count as a hard stop.

### 5.6 Text Tokenization and Prompt Format

VibeVoice uses the **Qwen2.5 tokenizer** (standard HF tokenizer, `vocab_size=151936`). The prompt format for multi-speaker TTS may use special tokens or text markup. Need to determine:
- How speakers are designated in the input (special tokens? text tags?)
- Whether any preprocessing (e.g., phonemization) is needed
- How the text/speech boundary is marked in the token sequence

Refer to the upstream VibeVoice inference code for the exact prompt format.

### 5.7 Streaming Decode in Stage 1

The ConvNeXt-based acoustic tokenizer decoder supports **causal streaming** via `VibeVoiceTokenizerStreamingCache`. For low-latency streaming:
- Process the first small chunk (e.g., 5 frames at codec_chunk_frames_at_begin) as soon as available.
- Then stream chunks of 25 frames with 25 frames of left context.
- The cache maintains causal convolution state across chunks, ensuring seamless concatenation.
- **Context cutting:** Similar to Voxtral's `apply_ctx_frames_cutting`, strip the audio samples corresponding to left-context frames from each chunk's output to avoid duplication.

---

## 6. Testing Plan

### 6.1 Unit Tests

| Test | File | What it verifies |
|------|------|------------------|
| Config parsing | `tests/model_executor/models/vibevoice_tts/test_config.py` | `VibeVoiceTTSConfig` correctly parses `microsoft/VibeVoice-1.5B` config.json |
| Diffusion head | `tests/model_executor/models/vibevoice_tts/test_diffusion_head.py` | Forward pass shape, DPM-Solver convergence |
| Stage input processor | `tests/model_executor/stage_input_processors/test_vibevoice_tts.py` | Latent chunking, context frame handling |
| Audio tokenizer decode | `tests/model_executor/models/vibevoice_tts/test_audio_tokenizer.py` | Latent → PCM shape, streaming cache consistency |

### 6.2 Integration / E2E Tests

| Test | File | What it verifies |
|------|------|------------------|
| Offline inference | `tests/e2e/offline_inference/test_vibevoice_tts.py` | End-to-end text → audio with `microsoft/VibeVoice-1.5B` |
| Online serving | `tests/e2e/online_serving/test_vibevoice_tts.py` | OpenAI TTS API endpoint, streaming WAV |

---

## 7. Reference Files (Voxtral TTS Counterparts)

For each new VibeVoice file, consult the corresponding Voxtral file as a template:

| VibeVoice file | Voxtral reference |
|----------------|-------------------|
| `configuration_vibevoice_tts.py` | `voxtral_tts/configuration_voxtral_tts.py` |
| `vibevoice_tts.py` | `voxtral_tts/voxtral_tts.py` |
| `vibevoice_tts_audio_generation.py` | `voxtral_tts/voxtral_tts_audio_generation.py` |
| `vibevoice_tts_audio_tokenizer.py` | `voxtral_tts/voxtral_tts_audio_tokenizer.py` |
| `vibevoice_tts_diffusion_head.py` | `voxtral_tts/cuda_graph_acoustic_transformer_wrapper.py` (different arch, same role) |
| `stage_input_processors/vibevoice_tts.py` | `stage_input_processors/voxtral_tts.py` |
| `stage_configs/vibevoice_tts.yaml` | `stage_configs/voxtral_tts.yaml` |

---

## 8. Summary

Adding VibeVoice TTS support follows the **same two-stage pipeline pattern** as Voxtral TTS, with these key adaptations:

1. **Replace Mistral with Qwen2.5** as the LLM backbone (simpler — standard HF config/tokenizer/weights).
2. **Replace flow-matching with DDPM diffusion** for the audio generation head (more inference steps, but same conceptual role).
3. **Replace discrete multi-codebook codes with continuous VAE latents** for the inter-stage representation (simpler streaming, no codebook arithmetic).
4. **Replace the custom neural codec with the ConvNeXt-based acoustic tokenizer decoder** for Stage 1 (convolutional only, with streaming cache support).
5. **Adapt the prompt format** for VibeVoice's multi-speaker text markup instead of Voxtral's voice embeddings.

The overall integration effort is **moderate** — the pipeline infrastructure (stages, connectors, fake_logits, OmniOutput) already exists and can be reused. The main work is implementing the diffusion head inference loop, the acoustic tokenizer decoder, and the weight loading logic.

---

## 9. Running VibeVoice TTS with vLLM-Omni

### 9.1 Prerequisites

- **GPU:** 1× NVIDIA H100 (80 GB) recommended. An A100 (80 GB) also works. The Qwen2.5-1.5B LLM + diffusion head fit in ~6 GB; the acoustic tokenizer decoder adds ~1.5 GB. Total VRAM usage is approximately 8 GB, so smaller GPUs may work with reduced `gpu_memory_utilization`.
- **Python:** 3.10–3.13.
- **Packages:** `vllm-omni` installed with all dependencies:

```bash
pip install -e ".[dev]"
```

- **Model weights:** The VibeVoice model is hosted on HuggingFace as `microsoft/VibeVoice-1.5B`. Weights are downloaded automatically on first use. To pre-download:

```bash
huggingface-cli download microsoft/VibeVoice-1.5B
```

### 9.2 Start the Server (Online Serving)

Launch the vLLM-Omni server with the VibeVoice stage config:

```bash
vllm serve microsoft/VibeVoice-1.5B \
    --stage-configs-path vllm_omni/model_executor/stage_configs/vibevoice_tts.yaml \
    --omni \
    --port 8091 \
    --trust-remote-code \
    --enforce-eager
```

Wait for the server to print `Application startup complete` (typically 30–60 seconds for weight loading).

### 9.3 Generate Speech via the API

#### Using curl

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, how are you doing today?",
        "voice": "alice"
    }' --output output.wav
```

#### Using Python (httpx)

```python
import httpx

response = httpx.post(
    "http://localhost:8091/v1/audio/speech",
    json={
        "input": "Hello, how are you doing today?",
        "voice": "alice",
    },
    timeout=120.0,
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

#### Using the OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8091/v1", api_key="none")

response = client.audio.speech.create(
    model="microsoft/VibeVoice-1.5B",
    input="Hello, how are you doing today?",
    voice="alice",
    response_format="wav",
)

response.stream_to_file("output.wav")
```

### 9.4 Available Voices

VibeVoice ships with 4 built-in speaker presets:

| Voice | Description |
|-------|-------------|
| `alice` | Female speaker |
| `bob` | Male speaker |
| `carol` | Female speaker |
| `dave` | Male speaker |

You can query available voices at runtime:

```bash
curl http://localhost:8091/v1/audio/voices
```

### 9.5 Offline Inference (Programmatic)

For batch processing without an HTTP server, use the `Omni` / `AsyncOmni` API directly.

#### Blocking (Omni)

```python
import torch
from vllm import SamplingParams
from vllm_omni.entrypoints.omni import Omni

MODEL = "microsoft/VibeVoice-1.5B"
STAGE_CONFIG = "vllm_omni/model_executor/stage_configs/vibevoice_tts.yaml"

omni = Omni(
    model=MODEL,
    stage_configs_path=STAGE_CONFIG,
    enforce_eager=True,
)

sampling_params = SamplingParams(max_tokens=4096)

# VibeVoice uses Qwen2.5 tokenizer — pass text directly as a prompt
inputs = {"prompt": "[alice]: Hello, how are you doing today?"}

outputs = list(omni.generate([inputs], [sampling_params, sampling_params]))

for i, output in enumerate(outputs):
    audio_data = output.multimodal_output.get("audio")
    if audio_data:
        audio_tensor = torch.cat(audio_data) if isinstance(audio_data, list) else audio_data
        audio_array = audio_tensor.float().cpu().numpy()

        import soundfile as sf
        sf.write(f"output_{i}.wav", audio_array, 24000)
        print(f"Saved output_{i}.wav ({len(audio_array) / 24000:.2f}s)")

omni.close()
```

#### Streaming (AsyncOmni)

```python
import asyncio
import uuid

import numpy as np
import soundfile as sf
import torch
from vllm import SamplingParams

from vllm_omni import AsyncOmni

MODEL = "microsoft/VibeVoice-1.5B"
STAGE_CONFIG = "vllm_omni/model_executor/stage_configs/vibevoice_tts.yaml"


async def main():
    async_omni = AsyncOmni(
        model=MODEL,
        stage_configs_path=STAGE_CONFIG,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(max_tokens=4096)
    inputs = {"prompt": "[bob]: This is a streaming test."}

    chunks = []
    accumulated_samples = 0

    async for stage_output in async_omni.generate(
        prompt=inputs,
        request_id=str(uuid.uuid4()),
        sampling_params_list=[sampling_params, sampling_params],
    ):
        mm = getattr(stage_output, "multimodal_output", None)
        if not mm or "audio" not in mm:
            continue

        audio_chunk = mm["audio"]
        finished = stage_output.finished

        if isinstance(audio_chunk, torch.Tensor):
            if finished:
                audio_np = audio_chunk[accumulated_samples:].float().cpu().numpy()
            else:
                audio_np = audio_chunk.float().cpu().numpy()
        else:
            audio_np = np.array(audio_chunk, dtype=np.float32)

        accumulated_samples += len(audio_np)
        chunks.append(audio_np)

    full_audio = np.concatenate(chunks)
    sf.write("output_streaming.wav", full_audio, 24000)
    print(f"Saved output_streaming.wav ({len(full_audio) / 24000:.2f}s)")

    async_omni.shutdown()


asyncio.run(main())
```

### 9.6 Prompt Format

VibeVoice uses the standard Qwen2.5 tokenizer. The speaker is specified inline as a text tag:

```
[speaker_name]: Text to synthesize.
```

For example:
- `[alice]: Good morning, everyone.`
- `[bob]: The weather looks great today.`

When using the OpenAI-compatible API (`/v1/audio/speech`), the `voice` field is automatically prepended as a speaker tag — you only need to provide the plain text in `input`.

### 9.7 Key Configuration Options

The stage config YAML at `vllm_omni/model_executor/stage_configs/vibevoice_tts.yaml` can be customized:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gpu_memory_utilization` (Stage 0) | `0.8` | VRAM fraction for the LLM + diffusion head |
| `gpu_memory_utilization` (Stage 1) | `0.2` | VRAM fraction for the acoustic decoder |
| `max_tokens` (Stage 0) | `4096` | Maximum AR decode steps (frames at 7.5 Hz ≈ 546 seconds max) |
| `enforce_eager` | `false` / `true` | Set to `true` to disable CUDA graphs (recommended initially) |
| `codec_chunk_frames` | `25` | Frames per streaming chunk (~3.3s of audio) |
| `codec_chunk_frames_at_begin` | `5` | Frames for the first chunk (~0.67s, lower latency) |
| `codec_left_context_frames` | `25` | Context overlap for streaming decode |
| `temperature` (Stage 0) | `0.0` | LLM sampling temperature (0.0 = greedy) |
| `repetition_penalty` (Stage 0) | `1.1` | Penalize repeated tokens |

#### Multi-GPU

To split the two stages across GPUs, edit the `devices` field:

```yaml
stage_args:
  - stage_id: 0
    runtime:
      devices: "0"    # Stage 0 on GPU 0
  - stage_id: 1
    runtime:
      devices: "1"    # Stage 1 on GPU 1
```

### 9.8 Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `ModuleNotFoundError: No module named 'vllm_omni'` | Package not installed | Run `pip install -e .` from the repo root |
| `ValueError: Invalid model_stage` | Wrong stage config | Ensure `--stage-configs-path` points to `vibevoice_tts.yaml` |
| `CUDA out of memory` | GPU too small | Lower `gpu_memory_utilization` in both stages, or use `--tensor-parallel-size 2` |
| Audio is silent or very short | Too few decode steps | Increase `max_tokens` in the stage config default_sampling_params |
| `Invalid speaker 'xyz'` | Unsupported voice name | Use one of: `alice`, `bob`, `carol`, `dave` |
| Server never prints "startup complete" | Model download stuck | Pre-download with `huggingface-cli download microsoft/VibeVoice-1.5B` |

### 9.9 Running Unit Tests

```bash
pytest tests/model_executor/models/vibevoice_tts/ \
       tests/model_executor/stage_input_processors/test_vibevoice_tts_async_chunk.py \
       -v -m "core_model and cpu"
```

This runs all 46 VibeVoice unit tests (config parsing, diffusion head, latent parsing, stage input processor) without requiring a GPU.
