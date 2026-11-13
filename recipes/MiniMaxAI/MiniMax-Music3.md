# MiniMax Music 3

> Text-to-music: lyrics plus a style caption in, a 32 kHz stereo song out

## Summary

- Vendor: MiniMax
- Model: `MiniMaxAI/MiniMax-Music3`
- Task: Text-to-music generation (vocals and instrumental together)
- Mode: Online serving through the OpenAI-compatible `/v1/audio/speech` API
- Maintainer: @linyueqian

## When to use this recipe

Use it to serve `MiniMaxAI/MiniMax-Music3` for full-song generation up to six
minutes. This is not a TTS model with a musical voice: there is no reference
speaker, no `voice`, and no `temperature`. What you control is the lyrics, the
caption, the seed, and the length.

## References

- Model card: [MiniMaxAI/MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3)
- Demo samples: [MiniMax Music 3 demo](https://minimax-ai.github.io/music3-demo/)
- Deploy config: `vllm_omni/deploy/minimax_music3.yaml` (single GPU),
  `vllm_omni/deploy/minimax_music3_2gpu.yaml` (split across two)

## Architecture

| Component | Spec |
|---|---|
| Backbone | Qwen3 decoder, 36 layers, hidden 4096, GQA 32/8, vocab 200k |
| Depth decoder | 4 layers, hidden 4096, 16 heads; 7 residual codebooks of 1024 |
| Frame | 8 codebooks; `c0` in the backbone vocabulary, `c1..c7` in the depth decoder |
| AR guidance | Classifier-free, scale 1.5, `c0` masked to the conditioned branch's top 50 |
| Acoustic stage | Flow-matching transformer (36 layers, dim 2048) plus a DAC-style decoder, 512x upsampling |
| Solver | Euler, 30 steps, acoustic guidance scale 1.7 |
| Window | 200 frames per acoustic chunk, 100-frame hop |
| Frame rate | 25 frames per second |
| Context length | 10,240 tokens |
| Output | 32 kHz stereo WAV |

Generation is two-stage. The backbone predicts one audio frame per decode step,
each frame eight RVQ codebooks deep. Every 200 frames are handed to the
flow-matching transformer, which solves a latent that the decoder turns into
waveform.

**Guidance doubles the rows.** Every request decodes twice: one row sees the
real prompt, the other sees the same prompt with the caption-and-lyrics span
replaced by `<|audio_cfg|>`. Both rows hold their own KV cache for the whole
song, so a request costs twice the KV its length suggests. Size the pool for
rows, not requests.

## Hardware Support

## GPU

### 1x H200 141GB

#### Environment

- OS: Linux
- Python: 3.12
- CUDA: 12.9
- vLLM version: 0.27.0
- vLLM-Omni version or commit: branch `feat/minimax-music3`

#### Command

```bash
vllm serve MiniMaxAI/MiniMax-Music3 \
    --host 0.0.0.0 --port 8000 \
    --trust-remote-code --omni
```

The default deploy config `vllm_omni/deploy/minimax_music3.yaml` is discovered
automatically from the checkpoint's `model_type=minimax_music3`. It colocates
both stages on device 0.

#### Verification

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniMaxAI/MiniMax-Music3",
    "input": "[Verse]\nWalking down the empty street at midnight\nStreetlights flicker like a broken dream\n[Chorus]\nAnd I keep on walking\nTill the morning finds me",
    "instructions": "A melancholic lo-fi hip-hop track at 85 BPM in F minor: mellow Rhodes piano riff, soft vinyl crackle, dusty boom-bap drums with a laid-back swing, warm upright bass. Intimate bedroom production, gentle tape saturation, no bright cymbals.",
    "seed": 1,
    "max_new_tokens": 750
  }' \
  --output song.wav
```

Expected: a 30 second 32 kHz stereo WAV. `max_new_tokens` counts audio frames
at 25 per second, so 750 frames is at most 30 seconds. It is a cap, not a
target: the model ends the song itself when it emits the audio-end token, and a
response shorter than the cap is the model finishing rather than a truncation.

#### Notes

- **Memory:** stage 0 takes `gpu_memory_utilization: 0.6` for the bf16 backbone
  and its KV pool; stage 1 takes 0.2 for the transformer and decoder weights,
  which sit outside any KV budget.
- **Key flags:** none required. Guidance, the solver step count and the window
  geometry are fixed by the checkpoint and are not request parameters.
- **Known limitations:**
  - The external API is non-streaming; `stream: true` is rejected.
  - `max_model_len` is 10,240, so a maximum-length prompt (5,000 tokens) and a
    maximum-length song (9,000 frames) cannot both fit. Long captions shorten
    the maximum song. This is a property of the checkpoint.
  - The acoustic stage runs in float32. bfloat16 is accepted but measurably
    degrades the solver.

### 2x H200 141GB

Use `minimax_music3_2gpu.yaml` to put the acoustic stage on the second device:

```bash
vllm serve MiniMaxAI/MiniMax-Music3 \
    --host 0.0.0.0 --port 8000 \
    --trust-remote-code --omni \
    --deploy-config minimax_music3_2gpu.yaml
```

Only the placement differs; both layouts run the acoustic stage in float32.

## Writing the request

Two fields carry the request, and both are required.

- `input` is the **lyrics**. Structure tags (`[Verse]`, `[Chorus]`, `[Bridge]`,
  `[Outro]`) steer the arrangement and are part of the prompt contract.
  **Put a tag on its own line.** Normalization keeps only the tags on a line
  that starts with one and drops the rest of that line, so lyrics written next
  to a tag are silently lost:

  ```text
  "[Verse]\nWalking down the street"   ->   [start] [verse] Walking down the street
  "[Verse] Walking down the street"    ->   [start] [verse]
  ```

- `instructions` is the **caption** describing genre, instrumentation, tempo,
  mood and production. It is the strongest control available: vague captions
  give generic arrangements. Long structured captions work well, since the
  model was trained on descriptions covering global attributes, emotional
  progression, vocal detail and arrangement.

A request is deterministic in its seed: the same lyrics, caption, seed and
length return the same audio. Changing only the seed gives another take of the
same song.

## Request parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input` | string | required | Lyrics, non-empty. Tags on their own lines |
| `instructions` | string | required | Caption, non-empty |
| `seed` | int | `0` | Non-negative. Fixes the output for a given request |
| `max_new_tokens` | int | `9000` | Cap on audio frames, 25 per second. Maximum 9,000 |
| `response_format` | string | `"wav"` | Output container |

## Parameters the model rejects

The request is refused rather than silently ignored, so a mistake is visible
immediately.

| Parameter | Reason |
|---|---|
| `temperature`, `top_p`, `top_k`, `repetition_penalty` | Sampling is fixed: guidance at 1.5, then a seeded top-k 50 draw |
| `voice` | There is no speaker to select; the vocal comes from the caption |
| `ref_audio`, `ref_text`, `language`, `task_type` | No reference-audio conditioning and no language tag |
| `speed` | Only `1.0`. Tempo belongs in the caption, e.g. "at 92 BPM" |
| `stream: true` | The external API is non-streaming |
