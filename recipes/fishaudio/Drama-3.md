# Fish Audio Drama 3

> Online serving for TTS

## Summary

- Vendor: FishAudio
- Model: `fishaudio/drama-3`
- Task: Text-to-speech with voice cloning, speaker tags and inline event tags
- Mode: Online serving with the OpenAI-compatible `/v1/audio/speech` API, or offline with `Omni`
- Maintainer: Community

## When to use this recipe

Use this recipe as a baseline for running `fishaudio/drama-3`. Drama 3 is a
35B-total / 3B-active MoE text backbone (Qwen3.5 MoE: hybrid gated-delta-net
and full attention, block-FP8 routed experts) with a codebook tower that
decodes 10 codebooks per frame. It outputs 44.1 kHz mono audio through the
same Fish DAC codec as Fish Speech S2 Pro.

## Pipeline

```text
text + reference (wav or pre-encoded codes, transcript)
  │
  ▼
Stage 0  fish_omni (LLM AR, Model Runner V2; one forward samples a whole frame)
  ├─ backbone (Qwen3.5 MoE) samples the frame's main token
  └─ codebook tower (13 layers, paged KV beside the backbone's) samples codes 1..9
  │   10-codebook frames, streamed in chunks (shared-memory connector)
  ▼
Stage 1  dac_decoder (FishSpeechDACDecoder) → 44.1 kHz audio
```

## References

- Model card: <https://huggingface.co/fishaudio/drama-3>
- Fish Speech S2 Pro recipe (same codec stage): [`Fish-Speech-S2-Pro.md`](./Fish-Speech-S2-Pro.md)

## Hardware Support

## GPU

### 1x H200 141GB

#### Environment

- OS: Linux
- Python: 3.12
- vLLM version: 0.31.0
- vLLM-Omni version or commit: main + this change

#### Command

```bash
vllm serve fishaudio/drama-3 --omni --port 8091
```

Notes:

- `--omni` is required.
- The default deploy config `vllm_omni/deploy/fish_omni.yaml` is loaded
  automatically (HF `model_type=fish_omni`). Both stages share one GPU
  through a private MPS daemon; stage 0 runs on Model Runner V2 with CUDA
  graphs (FULL for decode steps, PIECEWISE for steps with prompt rows).
- The checkpoint's `bsgemm-moe` quantization maps to vLLM's block-FP8 MoE
  kernels for the routed experts; all other weights stay bf16.

#### Verification

Voice cloning:

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "The quarterly report is due on Thursday, so please finish it by noon.",
        "voice": "default",
        "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
        "ref_text": "We asked over twenty different people, and they all said it was his.",
        "response_format": "wav"
    }' --output cloned.wav
```

#### Pre-encoded reference

A reference can be sent already encoded, as in the model card's production
serving setup: `ref_audio_codes` holds `[codebooks, frames]` raw codebook
indices of the checkpoint's codec (`codec.pth`), with `ref_text`. Nothing is
fetched or encoded per request; the prompt is the same as for the audio.

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"input": "...", "ref_audio_codes": [[...], ...], "ref_text": "...", "response_format": "wav"}' \
    --output cloned.wav
```

#### Event and emotion tags

Non-verbal events and delivery are controlled only through the input text.
Write a short natural-language descriptor in square brackets where the event
should happen:

```text
[sighs] Fine. If we miss it, it's on you. [laughs] Well, maybe that's fair.
```

- There is no fixed tag list. Tags are ordinary text (no special tokens), so
  any short descriptor is accepted; wordings close to the ones Fish Audio
  documents work best: `[sighs]`, `[laughs]`, `[laughing]`, `[whisper]`,
  `[angry]`, `[excited]`, `[pause]`.
- Speaker tags `<|speaker:0|>` … `<|speaker:15|>` are the only special control
  tokens: `"<|speaker:0|> Hi. <|speaker:1|> Hello."`. Text without a leading
  speaker tag is spoken by speaker 0.
- The reference clip also shapes delivery implicitly (an expressive or pause-heavy
  reference carries over), but it is not a targeted control.
- A tag or filler word at the very start of a line ("[sighs] …", "Yeah. …")
  more often renders the first word or event and then only silence until
  `max_new_tokens`; mid-sentence tags are more reliable.

Offline (`Omni`): pass `additional_information` with `text`, `ref_text`,
`ref_audio_wav`, `ref_audio_sr` (and optionally `voice_name` to cache the
encoded reference), and size `prompt_token_ids` with
`FishOmniForConditionalGeneration.estimate_prompt_len_from_additional_information`,
which returns the exact prompt row count.

#### Notes

- Memory: stage 0 weights take 36.9 GiB; the default deploy config uses
  0.78 of the GPU for stage 0 and 0.1 for the codec stage.
- Default sampling (model card): temperature 1.0, top_p 0.9, top_k 30,
  max_tokens 500. The codebook heads use the same values. Sampling runs in
  fused kernels that cap top_k at 64 (`hf_overrides`
  `fish_omni_max_top_k`, a power of two from 32 to 4096, raises it). A request's
  `seed` drives every draw of its frames, the codebook heads included.
- The main token uses repetition-aware sampling: a semantic token that
  repeats one of the last 16 main tokens is re-drawn with temperature 1.0,
  top_k 30, top_p 0.95.
- The default deploy keeps the gated-delta-net state in bfloat16 (the
  checkpoint's `mamba_ssm_dtype` is float32): +7% / +13% audio-s/s at 64 /
  128 streams with the same Seed-TTS-Eval WER and speaker similarity. Set
  `mamba_ssm_cache_dtype: float32` to keep the checkpoint's dtype.
- On CUDA the codec stage decodes channels-last from its own CUDA graphs
  (`fish_speech_dac_cuda_graphs`) and keeps each stream's convolution history
  and window transformer keys / values between windows
  (`fish_speech_dac_stateful`), so a window decodes only its new frames and
  windows splice as in decoding the whole utterance. Elsewhere it uses the
  module decoder, re-decoding each window's left context.

#### Measured performance (1x H200, both stages on one GPU)

Default `vllm_omni/deploy/fish_omni.yaml` (vLLM 0.31.0). Seed-TTS-Eval en
(1088 lines, each cloning its own reference, sent pre-encoded as
`ref_audio_codes`), streaming `/v1/audio/speech` (raw pcm), the sampling
above, a fresh server per concurrency, 8 warmup requests excluded.
Throughput is seconds of audio generated per wall-clock second.

| Concurrency | Audio s/s | TTFA p50 | TTFA p95 | RTF (mean) | Streams that stall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 163.8 | 195 ms | 382 ms | 0.374 | 0% |
| 128 | 232.5 | 252 ms | 649 ms | 0.515 | 0% |

#### Known limitations

- One tensor and pipeline rank only.
- Prefix caching is off: prompt rows carry ten codes per audio row that
  token-id hashing does not see.
- Some takes stop speaking after the first word or event and emit silence
  until `max_tokens`. This is model behaviour; it is more frequent for lines
  that start with a filler word or a tag, and with pause-heavy references.
- `previous_takes` (editing as continuation) and long-text chunking are not
  exposed by the speech API yet.
