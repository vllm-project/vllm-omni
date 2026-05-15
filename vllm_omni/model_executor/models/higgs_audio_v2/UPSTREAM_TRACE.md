# higgs-audio-v2 Upstream Trace

This memo records the upstream HF reference behavior that the vllm-omni
`higgs_audio_v2` integration must reproduce. All facts here are derived from
the official `transformers` source (`transformers.models.higgs_audio_v2`) and
the boson-ai checkpoints `bosonai/higgs-audio-v2-generation-3B-base` and
`bosonai/higgs-audio-v2-tokenizer`. Use this as the contract for AC-1, AC-2,
and AC-3.

## Fixed model constants (from `config.json`)

```json
{
  "model_type": "higgs_audio_v2",
  "architectures": [
    "HiggsAudioV2ForConditionalGeneration"
  ],
  "hidden_size": 3072,
  "num_hidden_layers": 28,
  "num_attention_heads": 24,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "vocab_size": 128256,
  "max_position_embeddings": 2048,
  "num_codebooks": 8,
  "codebook_size": 1026,
  "audio_token_id": 128016,
  "audio_bos_token_id": 128013,
  "audio_delay_token_id": 128014,
  "audio_stream_bos_id": 1024,
  "audio_stream_eos_id": 1025,
  "eos_token_id": 128009,
  "bos_token_id": 1,
  "pad_token_id": 128001,
  "tie_word_embeddings": false
}
```

## DualFFN routing rule (from `HiggsAudioV2DecoderLayer.forward`)

Each transformer block contains two parallel pre-norm + MLP paths:
- text path: `input_layernorm`, `post_attention_layernorm`, `mlp`
- audio path: `audio_input_layernorm`, `audio_post_attention_layernorm`, `audio_mlp`

A per-position `audio_token_mask: BoolTensor[B, S]` selects between them.

- Pre-attention norm: `audio_input_layernorm` on audio positions; `input_layernorm` on text positions. The two outputs are stitched together with `masked_scatter` and then fed through a single (shared) self-attention.
- Post-attention FFN: text positions are passed through `mlp(post_attention_layernorm(.))`; audio positions are passed through `audio_mlp(audio_post_attention_layernorm(.))`. Both deltas are ADDED to the residual (not replacing).
- Edge case: when `audio_token_mask is None` (pure-audio inference), the audio path is applied to ALL positions.

## audio_token_mask construction (from `HiggsAudioV2Model.get_placeholder_mask`)

```python
special_audio_mask = (input_ids == audio_token_id) | (input_ids == audio_delay_token_id)
```

i.e., a position is "audio" iff its token id is `audio_token_id=128016` (the audio placeholder) OR `audio_delay_token_id=128014` (the delay filler).

## Audio embedding rule (from `HiggsAudioV2Embeddings`)

- `embed_audio_tokens`: `nn.Embedding(num_codebooks * codebook_size, hidden_size)`, i.e. 8 * 1026 = 8208 rows.
- `audio_tokens_offsets = arange(num_codebooks) * codebook_size = [0, 1026, 2052, 3078, 4104, 5130, 6156, 7182]`.
- For `audio_input_ids` of shape `(B, num_audio_frames, num_codebooks)` with values in `[0, codebook_size)`:
    - `inputs_embeds = embed_audio_tokens(audio_input_ids + audio_tokens_offsets)`
    - `inputs_embeds.sum(dim=-2)` collapses across codebooks.
- The text prompt's `inputs_embeds = embed_tokens(input_ids)`. Then a `masked_scatter` substitutes audio frames at positions where `audio_token_mask` is True.

## Delay-pattern handling (from `HiggsAudioV2DelayPatternLogitsProcessor`)

- A `delay_pattern: list[int]` controls the per-codebook offset for the audio-stream BOS and EOS tokens. The processor masks the codebook vocab so each codebook `k` is forced to emit `audio_stream_bos_id=1024` at the start and `audio_stream_eos_id=1025` at the end until its delay counter reaches 0.
- The 8-codebook canonical delay pattern is the MusicGen-style monotonic sequence `[0, 1, 2, 3, 4, 5, 6, 7]` (codebook k starts emitting real codes only after k frames). This is consistent with the `HiggsAudioV2DelayPatternLogitsProcessor.__call__` math (`scores.reshape(-1, num_codebooks, codebook_size)` and the per-row `vocab_mask_bos`/`vocab_mask_eos` masking).
- Real audio code IDs are in `[0, 1024)`. Codes `1024` and `1025` are the stream-BOS / stream-EOS markers and must NOT reach the codec decoder. The vllm-omni Stage 1 must reject any value `>= 1024` with an explicit `ValueError`.

## Stream BOS/EOS emission rule

- `audio_stream_bos_id=1024` is emitted at the boundary that opens an audio stream; it is consumed by the LM during the codebook-output build-up phase and is filtered out before sending codes to the codec.
- `audio_stream_eos_id=1025` is emitted at the boundary that closes an audio stream; the LM uses it to learn end-of-audio, and it is filtered out before decode.
- `audio_bos_token_id=128013` is the *prompt-level* audio bos in the LM vocabulary (text-space token id) that marks the position in `input_ids` where the audio frames begin.
- `audio_delay_token_id=128014` is the *prompt-level* delay placeholder used to fill positions where a codebook has not yet started emitting real codes (post-delay-pattern construction). These positions still go through the audio path of DualFFN.

## Plain-text prompt template (from upstream docs / `HiggsAudioV2Processor.apply_chat_template`)

Conversation form for a plain-text TTS request:

```python
conversation = [
    {"role": "system", "content": [{"type": "text", "text": "Generate audio following instruction."}]},
    {"role": "user",   "content": [{"type": "text", "text": "<USER TEXT HERE>"}]},
]
processor.apply_chat_template(
    conversation, add_generation_prompt=True, tokenize=True,
    return_dict=True, sampling_rate=24000, return_tensors="pt",
)
```

The captured `rendered_chat_template` from the reference run is persisted under
`text_template_*` keys in the fixture files alongside the `input_ids` tensor.
vllm-omni's `higgs_audio_v2_tokenizer.build_plain_text_prompt(...)` must produce
the same `input_ids` for the same `<USER TEXT>` (AC-1 positive test).

## Fixture inventory

Each `tests/fixtures/higgs_audio_v2/reference_<slug>.pt` holds the per-prompt
record described at the top of `reference_hf.py`. The captured fields satisfy:

- AC-1 (input-token parity): exact `input_ids` from upstream tokenizer.
- AC-2 (per-codebook parity): `audio_codes` is the post-revert real-code tensor
  with shape `[1, num_codebooks=8, T]` and values in `[0, 1023]`.
- AC-3 (DualFFN routing): `audio_token_mask` is the per-position routing mask
  recorded from the live forward pass on the first decoder layer (matches
  `HiggsAudioV2Model.get_placeholder_mask`).
- AC-4 (Stage-1 decode parity): `reference_pcm` is the upstream-decoded waveform
  as int16, mono, 24 kHz. Normalized-float RMS comparison with vllm-omni Stage 1
  must be `<= 1e-4` (see plan AC-4).

## Pinned prompt list

- 'Hello world.'
- 'The quick brown fox jumps over the lazy dog.'
- 'It was the night before my birthday.'
- 'She sells seashells by the seashore.'
- 'Innovation distinguishes between a leader and a follower.'
- 'Mary had a little lamb whose fleece was white as snow.'
- 'Time flies like an arrow; fruit flies like a banana.'
- 'All that glitters is not gold.'
- 'An apple a day keeps the doctor away.'
- 'May the force be with you, always.'
- 'To be or not to be, that is the question.'

## Notes for the vllm-omni implementation

- The Stage-0 talker must implement DualFFN by subclassing
  `vllm.model_executor.models.llama.LlamaDecoderLayer` and replacing the
  `mlp` member with a routed `DualFFNLayer` that consults the per-position
  audio mask precomputed at model input time.
- The HF→vLLM weight mapping must transcribe both the text MLP weights
  (`model.layers.<L>.mlp.{gate_proj,up_proj,down_proj}`) and the audio MLP
  weights (`model.layers.<L>.audio_mlp.{gate_proj,up_proj,down_proj}`), plus
  the parallel layernorm pairs (`{input_layernorm, audio_input_layernorm,
  post_attention_layernorm, audio_post_attention_layernorm}.weight`).
- The fused QKV projection uses GQA with 24 Q heads / 8 KV heads / head_dim=128;
  pack as `[hidden + 2 * kv_head_dim * head_dim, hidden]` mirroring how
  `vllm.model_executor.models.llama.LlamaAttention.load_weights` consumes
  separated `q_proj/k_proj/v_proj`.
- The RoPE config is `rope_type="llama3"` with `factor=32.0`,
  `low_freq_factor=0.125`, `high_freq_factor=0.5`, `original_max_position_embeddings=1024`.
- Multi-codebook output head: the model has `(num_codebooks * codebook_size) = 8 * 1026 = 8208`-wide audio output (via `embed_audio_tokens` lookups) AND a 128256-wide text head (the standard Llama `lm_head`). Stage-0 emits codebook 0 via the audio head plus the residual codebooks 1..7 via a per-step fast-AR head; see plan task3 for the structure.
