# LongCat-Next — HuggingFace reference vs. vLLM-Omni branch port

Side-by-side comparison of the reference implementation in
[`meituan-longcat/LongCat-Next`](https://huggingface.co/meituan-longcat/LongCat-Next)
(the checkpoint's own `trust_remote_code=True` modules, `main` @ commit
`3e934f1a` / `0cf0631`) and the in-progress native port on branch
`feat/longcat-next-integration` (HEAD `bc4bac9c`, 47 commits ahead of `main`).

Source files pulled for this comparison (code only, **no weights** were
downloaded):

- HF: `modeling_longcat_next.py`, `modeling_longcat_ngram.py`,
  `configuration_longcat_next.py`, `configuration_longcat_ngram.py`,
  `modular_longcat_next.py`, `modular_longcat_next_visual.py`,
  `modular_longcat_next_audio.py`, `cosy24k_vocoder.py`, `image_refiner.py`,
  `refiner_modules.py`, `processing_longcat_next.py`, `parse_model_response.py`,
  plus `config.json` / `generation_config.json`.
- vLLM-Omni: `vllm_omni/model_executor/models/longcat_next/*.py`,
  `vllm_omni/transformers_utils/configs/longcat_next.py`,
  `vllm_omni/transformers_utils/processors/longcat_next.py`,
  `vllm_omni/model_executor/stage_input_processors/longcat_next.py`,
  `vllm_omni/config/pipeline_registry.py`, `vllm_omni/model_executor/models/registry.py`.

> Note on `vllm_omni/diffusion/models/longcat_image/`: that package is a
> separate Flux-style text-to-image diffusion model (already on `main`, from a
> different repo), **not** part of this branch's LongCat-Next integration, and
> is out of scope here. The LongCat-Next image **decoder** stage
> (`LongcatNextImageDecoder`) is what corresponds to the HF repo's
> `image_decoder/` + `image_refiner.py` + `refiner_modules.py` path.

> Section 9 (appended on request) additionally cross-checks the port and the HF
> reference against Meituan's own SGLang serving harness,
> `LongCat-Next-inference/` (local copy under `opensource_stuff/`).

---

## 1. Repository layout mapping

| Role | HF reference file | vLLM-Omni branch file |
|---|---|---|
| Backbone + multimodal model | `modeling_longcat_next.py` (`LongcatNextModel`, `LongcatNextForCausalLM`) | `modeling_longcat_next.py` (`LongcatNextForCausalLM`) |
| N-gram embeddings / cache | `modeling_longcat_ngram.py` (`NgramEmbedding`, `NgramCache`, `LongcatFlashNgramModel`) | reused vLLM native `vllm/model_executor/models/longcat_flash_ngram.py` (`FlashNgramModel`, `NgramEmbedding`) |
| Configs | `configuration_longcat_next.py`, `configuration_longcat_ngram.py` | `vllm_omni/transformers_utils/configs/longcat_next.py` (shim) + remote `load_remote_hf_config` |
| Depth heads (visual/audio heads) | `modular_longcat_next.py` (`CasualDepthTransformerHead`) | loaded **unchanged** from checkpoint remote code (`get_remote_attr`) |
| Vision encoder + VQ bridge | `modular_longcat_next_visual.py` (`VisualEncoder`, `VisualVQBridge`, `LongcatNextVisualTokenizer`) | loaded **unchanged** from checkpoint remote code |
| Audio encoder + VQ bridge + decoder + CFM | `modular_longcat_next_audio.py` (`LongcatNextAudioTokenizer`, …) | loaded **unchanged** from checkpoint remote code |
| HiFT vocoder | `cosy24k_vocoder.py` (`Cosy24kVocoder`) | loaded **unchanged** from checkpoint remote code |
| Image refiner (DiT + VAE) | `image_refiner.py`, `refiner_modules.py` | loaded **unchanged** via `lazy_decode_and_save` |
| Image decoder stage | (`lazy_decode_and_save` path) | `modeling_longcat_next_image_decoder.py` (`LongcatNextImageDecoder`) |
| Audio decoder stage | `decode_wave_vocoder2` / `decode_save_concat2` path | `modeling_longcat_next_audio_decoder.py` (`LongcatNextAudioDecoder`) |
| Combined decoder stage | (N/A — single process dispatch) | `modeling_longcat_next_multi_decoder.py` (`LongcatNextMultiDecoder`) |
| Processor (text/image/audio) | `processing_longcat_next.py` (`LongcatNextProcessor`, `LongcatNextAudioProcessor`) | `longcat_next_processor.py` (vLLM multimodal proc) + vendored `transformers_utils/processors/longcat_next.py` |
| Response post-parse (tool calls/thinking) | `parse_model_response.py` | not ported (out of scope — client-side text post-processing) |
| Weight loading / pipeline glue | (transformers internals) | `pipeline.py`, `stage_input_processors/longcat_next.py`, `deploy/longcat_next_*.yaml` |

---

## 2. Architecture at a glance

Both implement the same any-to-any model:

```
text + image + audio  ──►  LongCat Flash backbone (MoE + MLA, 14 layers × 2 sublayers)
                              ▲ n-gram fused embeddings (emb_neighbor_num=4, emb_split_num=4)
                              │ + visual/audio code embeddings merged at <img_pad>/<audio_pad>
                              ▼
   text branch: lm_head (131125) ──► text tokens
   image branch: visual_head (CasualDepthTransformerHead) ──► 8-level visual codes ──► VisionTransformerDecoder + refiner ──► RGB
   audio branch: audio_head  (CasualDepthTransformerHead) ──► 8-level audio codes  ──► audio decoder + flow-matching ──► Cosy24k HiFT ──► waveform
```

Backbone dimensions (checkpoint `config.json`, not the class defaults):

| Param | Checkpoint value |
|---|---|
| `vocab_size` | 282624 |
| `text_vocab_size` (n-gram hash space) | 131072 |
| `text_vocab_plus_multimodal_special_token_size` (lm_head) | 131125 |
| `hidden_size` | 3072 |
| `num_layers` (2 sublayers each) | 14 |
| `num_attention_heads` | 32 |
| `q_lora_rank` / `kv_lora_rank` (MLA) | 1536 / 512 |
| `moe_topk` / `n_routed_experts` / `zero_expert_num` | 12 / 256 / 128 |
| `ffn_hidden_size` | 6144 |
| `emb_neighbor_num` / `emb_split_num` / `ngram_vocab_size_ratio` | 4 / 4 / 78 |
| `visual_offset` / `audio_offset` | 150581 / 131125 |
| `visual codebook` | 8 × 16384 (flat) |
| `audio codebook` | [8192, 4096, 2048, 1024×5] |
| visual head depth-transformer | dim 2048, 4 layers, ffn_scale 16 |
| audio head depth-transformer | dim 3072, 4 layers, ffn_scale 16 |

---

## 3. Component-by-component comparison

### 3.1 Backbone (`LongcatFlashNgramModel`)

| | HF | vLLM-Omni |
|---|---|---|
| Class | `LongcatFlashNgramModel` in `modeling_longcat_ngram.py` (extends transformers `LongcatFlashModel`) | vLLM-native `FlashNgramModel` from `vllm/model_executor/models/longcat_flash_ngram.py` |
| Attention | transformers `LongcatFlashDecoderLayer` (MLA + MoE), full KV cache | vLLM fused MLA / MoE kernels, **paged KV cache, TP/PP sharding** |
| Layer count | `config.num_layers` (14) × 2-sublayer flash blocks | same (native `FlashModel` reads `num_layers`) |
| Notes | pure-PyTorch, single GPU | the whole reason for the port — native TP/MoE/MLA kernels |

**Verdict:** intentionally different (native vLLM kernels), but should be
numerically equivalent to the transformers flash layer given identical weights.
The branch's MLA input-scale fold (§3.8) is the known correction that keeps the
two numerically aligned.

### 3.2 N-gram embeddings

| | HF | vLLM-Omni |
|---|---|---|
| `m` (hash vocab) | `ngram_vocab_size_ratio * text_vocab_size` = 78 × 131072 | same constant (kernel uses `text_vocab_size=131072`, not the 282624 full vocab — see §3.4) |
| Hash function | Python: `ngram_ids = ids + Σ shifted[order] * power_mod`; `new_ids = ngram_ids % (m + index*2 + 1)`; per-(order,split) embedders + post-projs, normalize `x[~ignored] /= (1 + k*(n-1))` | CUDA kernel `ops.ngram_compute_n_gram_ids` (same algorithm vLLM's LongCat Flash uses); `NgramEmbedding.embed_batched` does base + n-gram + normalize |
| Context | stateless: `NgramCache` keeps last `n-1` tokens per request (`update_ngram_context`) | per-request rolling `_ctx_len = n-1` context dict keyed by `request_id` (state lives on the model, fed through the omni `preprocess` hook) |
| EOS/special handling | `_shift_right_ignore_eos`: EOS = segment boundary, id `0` = no-op, `oe_ignored` = [131072, 131125) zeroed to 0 | `_neg_eos` (EOS → −EOS), boundary = id ≥ 131072 **or** id == 0 → kernel `-1` boundary entry |
| Streaming vs batch | one full call per step over the whole sequence | streamed span-by-span with a rolling left context (must produce identical hashes) |

**Verdict:** same intent, different mechanism. The hash arithmetic is the
vLLM-native kernel, so parity depends on the kernel's convention matching HF's
`_shift_right_ignore_eos` + modular-hash math. The branch carries an explicit
audit (`[longcat-ngram]` log of first-span `oe_ids`) and an A/B bypass
(`LONGCAT_NGRAM_DISABLE=1`) precisely because this is a top divergence suspect —
worth a pod-side diff of `oe_ids` vs the HF kernel's output for one prompt.

### 3.3 Config

| | HF | vLLM-Omni |
|---|---|---|
| `LongcatNextConfig` | full nested config (`LongcatNextVisualConfig` ⊂ Qwen2_5_VL, `LongcatNextAudioConfig` ⊂ Whisper) | `vllm_omni/.../configs/longcat_next.py` — a **minimal `PretrainedConfig` shim** (no nested `visual_config`/`audio_config`) |
| Nested configs | parsed into real config objects | the shim is bypassed on purpose: `load_remote_hf_config` loads the **checkpoint's own** `LongcatNextConfig` so remote code gets real nested objects (docstring in `longcat_next_utils.py:111`) |
| `pad_token_id` | omitted in config.json; generation_config = 3 | `_DEFAULT_PAD_TOKEN_ID = 3` injected if `None` |
| `intermediate_size` | checkpoint uses `ffn_hidden_size` | `hf.intermediate_size = ffn_hidden_size` aligned before building the backbone |

**Verdict:** functionally equivalent — the shim is only a registry
placeholder; all real config work goes through the checkpoint's own config
class.

### 3.4 N-gram table sizing

- HF `NgramEmbedding` explicitly uses `config.text_vocab_size` (the class even
  comments out the `vocab_size` variant): `self.m = ngram_vocab_size_ratio * text_vocab_size`.
- vLLM-Omni: native `NgramEmbedding` would use `config.vocab_size` (282624 →
  ~135 GB table + broken hashes), so the port **temporarily nils
  `ngram_vocab_size_ratio`**, builds the backbone, then attaches its own
  `NgramEmbedding` with `ngram_cfg.vocab_size = text_vocab_size` (131072).

**Verdict:** aligned with the HF math; workaround is contained and commented.

### 3.5 `lm_head`

| | HF | vLLM-Omni |
|---|---|---|
| Size | `nn.Linear(hidden_size, text_vocab_plus_multimodal_special_token_size)` = 131125 | `ParallelLMHead(131125, hidden_size)` (TP-sharded) |
| Bias | none | none |

**Verdict:** match. Note 131125 < 131072 is **not** true — 131125 > 131072; the
lm_head covers text vocab + the 53 multimodal special tokens ([131072, 131125)),
but not the code ranges (audio @131125+, visual @150581+).

### 3.6 Multimodal encoders (understanding direction)

| | HF | vLLM-Omni |
|---|---|---|
| Visual encode | `visual_tokenizer.encode` → ids → `+= visual_offset_vals` → `embed_tokens(ids).sum(1)` → `visual_embedding_layer` | identical (`_encode_images`, same remote `LongcatNextVisualTokenizer`) |
| Audio encode | `audio_tokenizer.encode` → ids → `+= audio_offset_vals` → `embed_tokens(ids).sum(1)` | identical (`_encode_audios`, same remote `LongcatNextAudioTokenizer`) |
| Replication | runs inside the single process | runs **per rank** (replicated, not TP-sharded), consistent with the remote tokenizers' non-TP nature |
| Batching | bs=1 | whole flat batch through the encoder once, output split per item (`flat_from_sizes`) |

**Verdict:** same code, same math; only the batching wrapper differs.

### 3.7 Depth heads (`CasualDepthTransformerHead`)

| | HF | vLLM-Omni |
|---|---|---|
| Class | checkpoint `modular_longcat_next.CasualDepthTransformerHead` | **same checkpoint class**, instantiated via remote code |
| `cuda:0` hardcode | `forward` moves `visual_tokens` / `visual_emb_layers` to `"cuda:0"` | run on **rank 0 only** + `tp_group.broadcast(codes_row, src=0)` to avoid the hardcoded `cuda:0` under TP |
| Code embedding layer | the full `model.embed_tokens` (indexed by full offset ids) | a compact replicated `nn.Embedding` over the code range rows ([offset, offset+Σsizes)), indexed by relative ids — avoids TP all-reduce inside the rank-0 loop |
| Autoregressive depth loop | `cumsum(emb(prev levels))` feeding level `L` | identical (`_sample_depth_head`), plus sentinel masking (§3.10) |

**Verdict:** same class + same math; only the device/TP workaround differs.

### 3.8 MLA input-scale folding (LongCat-specific)

- HF `LongcatFlashMLA` scales `q_pass/q_rot *= (hidden/q_lora_rank)^0.5` and
  `k_pass *= (hidden/kv_lora_rank)^0.5` **before** the loRA projections.
- vLLM's generic `DeepseekV2MLAAttention` omits these scales → re-weights every
  head's nope/rope score terms (logit re-rank + compression vs reference).
- Port folds the constant scalars into `q_b_proj.weight` / `kv_b_proj.weight`
  (`_apply_mla_scale_fold`, `modeling_longcat_next.py:1955`), mathematically
  exact, idempotent, backbone layers only.

**Verdict:** deliberate, correctness-preserving port fix — this is the port's
own known correction for a real numerical gap.

### 3.9 Generation state machine

The state machine is the biggest architectural divergence (reference is a
single monolithic `generate()` loop; the port is a step-driven runner hook).

| Concept | HF | vLLM-Omni |
|---|---|---|
| State holder | `LongcatNextForCausalLMGenerationStatus` (mode, `current_image_token_num`, `is_audio_text_end`, `is_audio_start`) | per-request dicts `_audio_gen` / `_visual_gen` on the model, driven from `preprocess` / `talker_mtp` / `compute_logits` |
| Mode switch → visual | last token == `image_start` → `switch_to("visual")`, **auto-injects `anyres_prefix`** = `<longcat_img_token_size>{h} {w}</longcat_img_token_size>` into the sequence | last token == `IMG_START` → creates `_visual_gen`; **prefix NOT injected** — caller must put the anyres prefix in the prompt and pass `token_w/token_h` via `additional_information` (fallback 37×37) |
| Mode switch → audio | last token == `audiogen_start` → `switch_to("audio")` | last token == `AUDIOGEN_START` → creates `_audio_gen` |
| `is_audio_start` | set when the **model emits `audiotext_start`** | `audio_start = gen_step > delay` with fixed `delay = 0` → always true from step 1; `ext_id = audiotext_start` forced at `gen_step == delay` |
| Audio text end | `is_audio_text_end` set when sampled token == `audiotext_pad` | `text_end` set when last visible token == `AUDIOTEXT_PAD`, then compute_logits pins the stream to pad |
| Text sampling | standard HF loop with `logits_processor(input_ids, logits)` | `compute_logits` + vLLM sampler (only the gen-mode forcing is custom) |
| After image end | `switch_to("text")`; next step `mode=="text" and last_step_mode=="visual"` → **emit EOS and terminate generation** | state pops on `IMG_END`, then **normal text generation continues** |
| After audio end | `switch_to("text")`; continues generating text | state pops on `AUDIOGEN_END`; continues text |
| `audio_parallel_decoding` | supported (config option, `False` in checkpoint) | not implemented (sequential only — matches checkpoint default) |

**Verdict:** behaviorally aligned for the default checkpoint settings, but with
three documented divergences:
1. **anyres prefix injection** — HF auto-injects it; the port requires it in the
   prompt / per-request canvas. (Port warns loudly when `token_w/token_h` are
   missing.)
2. **post-image continuation** — HF hard-terminates after `<longcat_img_end>`;
   the port keeps generating text.
3. **audio start trigger** — HF keys off the model's emitted `audiotext_start`;
   the port forces it deterministically at `gen_step == 0`.

### 3.10 Depth-head sampling & sentinel handling

| | HF | vLLM-Omni |
|---|---|---|
| Visual level-0 end-of-image class | masked by the SGLang reference (`output_processor.py:312`); **not** masked in the HF generate loop (relies on grid `is_img_end`) | masked in `_sample_depth_head(mask_sentinel=True)` so the image can only end at the grid bound |
| Audio non-zero-level sentinels | no masking (raw head output) | `mask_audio_sentinels=True`: level ≥ 1 sentinel (`codebook_sizes[i]`) → −inf, preventing OOB VQ-codebook gathers at decode (a real crash fix) |
| Audio chunk-end marker | level-0 `codebook_sizes[0]` (16384) row kept for `decode_wave_vocoder2` | kept and emitted as an explicit boundary row (§3.12) |
| Repetition penalty | HF `_get_logits_processor` applied to past `multimodal_ids` (code-level) | explicit `_sample_audio_code` penalty over `past_codes[:, level]` (score<0 → ×p, else ÷p) — same math |
| Sampling defaults | audio: temp 0.5 / top_k 5 / top_p 0.85 / rep 1.3; visual: temp 0.5 / top_k 1024 / top_p 0.75 / rep 1.0 (generation_config) | identical coalesced defaults in `talker_mtp` |

**Verdict:** match on math; the port adds two protective masks (visual
sentinel + audio non-zero-level sentinels) that make it *safer* than the HF
path.

### 3.11 Visual CFG

| | HF | vLLM-Omni |
|---|---|---|
| Mechanism | `input_ids.repeat((2, 1))` at bs=1 when `cfg_scale != 1.0`; uncond copy's text ids zeroed (keeps anyres + image tokens); per-level `combined = cfg * (cond − uncond) + uncond` in `inner_sample` | twin **engine requests** (`request_id + "__cfg_visual"`), uncond stream built at string level by `expand_longcat_cfg_prompts`; same `combined` formula in `_sample_cfg_visual_codes` |
| Lockstep | both streams in one batch, same step | two independently-scheduled requests linked only by engine affinity; `talker_mtp` falls back to non-CFG sampling on any desync step (logged) |
| Default scale | `cfg_scale` 3.0 (generation_config custom_params) | `_DEFAULT_CFG_SCALE = 3.0` |
| Uncond blanking | token-level (guaranteed same length) | string-level (cannot guarantee equal tokenization; model-side twin-sync keeps them aligned from first decode step) |

**Verdict:** same math, different execution substrate. The port's desync
fallback is a graceful degradation the HF path (single batch) never needs.

### 3.12 Audio generation loop (`talker_mtp` vs `get_multimodal_logits_and_ids`)

| | HF | vLLM-Omni |
|---|---|---|
| Next-step embedding | next `forward` call rebuilds inputs_embeds from masks + `audio_text_ids` + `audio_ids[-mask.sum():]` (embed_tokens sum) | `talker_mtp` builds it inline per step as a 3-stream **sum**: `ext_id_emb` (zeroed if `audiotext_pad`) + visible-token emb (zeroed if `audiotext_pad`) + audio-code emb (kept frames, level-0 != 0) |
| Visible token during audio | `audio_pad` after text-end; `audiotext_start` at the pre-audio step; `audiogen_end` when `audio_ids[-1,0] == offset_vals[1]` (chunk-end marker) | compute_logits forces the same tokens from `_audio_gen` state (`text_end` → pin pad; `terminal` → force `audiogen_end`; else ban EOS) |
| Chunk-end / max-gen termination | `audiogen_end` emitted when the marker code is sampled; no max-gen cap (relies on caller `max_new_tokens`) | `eoc_terminal = deoff >= codebook_sizes[0]` **or** `max_gen` cap (`LONGCAT_MAX_GEN`, default 30 s × 25 fps); terminal row becomes an explicit boundary marker row |
| EOS suppression | HF generate loop can stop on EOS/stopping criteria; mode logic overrides tokens | EOS is explicitly `−inf`ed whenever a gen state is active |

**Verdict:** same observable behavior on the default path (marker-chunked
audio, `audiotext_pad` text pinning, `audiogen_end` closing), with the port
adding a `max_gen` safety cap and cleaner boundary-marker semantics so one
request can produce multiple audio segments without decoder overflow (see
`modeling_longcat_next.py:1490-1512` for the multi-segment rationale).

### 3.13 Image generation loop

| | HF | vLLM-Omni |
|---|---|---|
| Grid math | `is_img_newline = (i+1) % (w+1) == 0`, `is_img_end = (i+1)/(w+1) == h`, `i` = `current_image_token_num` | `gen_step % (token_w+1) == 0` → newline; `gen_step >= token_h*(token_w+1)` → terminal (forces IMG_END) |
| Codes at newline steps | **skipped** (no depth-head call) | depth-head still sampled but row discarded (`frame_kept = not is_row_boundary`) |
| Visible tokens | `image_pad` / `image_newline` / `image_end` written by the loop | same three forced by compute_logits from `_visual_gen.ext_id` |
| Next-step embedding | next `forward` masks visual positions and embeds `visual_ids` | masked-replace in `talker_mtp`: `vision_emb = _code_embeddings(offset_codes)` + `visual_embedding_layer`, else the visible token's own embedding |

**Verdict:** same grid semantics and visible-token stream. The port wastes a
depth-head call on newline steps (cosmetic). Grid-bound termination matches
HF's `is_img_end`.

### 3.14 Audio decoder stage

| | HF | vLLM-Omni |
|---|---|---|
| Chunking | `lazy_decode_and_save`: split at level-0 marker rows; each chunk **includes** its marker row; `decode_wave_vocoder2` re-truncates at the marker and flattens all valid segments into one `audio_tokenizer.decode` call | `_split_chunks`: marker row **excluded**; last chunk padded with a synthetic marker if missing; each chunk decoded independently |
| Vocoder | `Cosy24kVocoder.decode(mel.transpose(0,1).float().unsqueeze(0))` after CFM | identical call, but only after `audio_tokenizer.decode(chunk)` (same remote tokenizer, includes audio-decoder + flow-matching) |
| Cross-fade | `decode_save_concat2`: appends blend **and** the full next wave → seam emitted twice (potential stutter) | documented **divergence**: seam emitted once (`prev[:, :-overlap]` + blended overlap), trimmed heads — fixes chunk-boundary stutter |
| Sample rate / overlap | 24000 / 1200 (generation_config custom_params) | same defaults, read from the checkpoint's generation_config |
| Batch | bs=1 | warns + decodes only first request if `max_num_seqs > 1` |

**Verdict:** same net result for a single chunk; the port's chunk reassembly is
a deliberate, commented improvement for long/multi-chunk audio.

### 3.15 Image decoder stage

| | HF | vLLM-Omni |
|---|---|---|
| Core | `LongcatNextVisualTokenizer.lazy_decode_and_save` (codebook gather → VisionTransformerDecoder → RefinerPipeline) | **same remote call**, run from `LongcatNextImageDecoder.forward` |
| Codes | raw per-level indices (no offsets — offsets are subtracted by the HF wrapper `decode_visual_ids_and_save`) | raw indices passed straight through (an earlier version wrongly added offsets; fixed) |
| Grid handling | exact grid assumed | **defensive truncation** if overrun (`codes[:token_h*token_w]`) and clean empty-output (instead of a device assert crash) if short |
| Output | saves PNG to disk (`refined_path`) | reads the first saved image back to a `[1, 3, H, W]` float tensor for `OmniOutput` |

**Verdict:** identical core; the port hardens grid-shape errors instead of
crashing the GPU worker.

### 3.16 Combined multi-decoder stage

The port adds `LongcatNextMultiDecoder` + `LONGCAT_NEXT_THINKER_MULTI_DECODER_PIPELINE`.
Rationale (documented in `modeling_longcat_next_multi_decoder.py`): the omni
stage orchestrator only forwards `src_stage_id + 1` output, so a 3-stage
thinker→image→audio chain would feed the image decoder's output to the audio
decoder (audio always broken). This mirrors the reference's own
`PostProcessor.decode_multi` (single process, dispatch on `gen_image`/`gen_audio`).
HF has no analogue (its decode is not staged).

### 3.17 Processor (tokenization / feature extraction)

| | HF | vLLM-Omni |
|---|---|---|
| Image | `Qwen2VLImageProcessor` over **file paths** regex-scanned from the prompt | same `Qwen2VLImageProcessor` over **in-memory PIL/arrays**; placeholder expanded by `_get_prompt_updates` |
| Audio | `LongcatNextAudioProcessor.process` (librosa path load, resample, fbank, split_with_overlap, `inference_output_length`) | same fbank pipeline vendored, driven **on in-memory waveforms** (`MultiModalDataParser(target_sr=16000)`) |
| Placeholder grammar | `<start>path<end>` replaced with `pad*N` | `<start><pad><end>` → `pad*N` via `PromptReplacement` |
| Tokenization | `add_special_tokens=True` default in `_call_hf_processor`? | port uses `add_special_tokens=False` in `_call_hf_processor` (`longcat_next_processor.py:177`) — flagged as an audit point in `preprocess` |

**Verdict:** same feature math, adapted to in-memory data flow. The
`add_special_tokens=False` choice vs the HF default is called out in the code
as a potential 1-token shift that changes n-gram hashes — the branch logs first
span ids for a pod diff.

### 3.18 Response post-processing (`parse_model_response.py`)

HF ships a client-side helper that turns `<longcat_think>…</longcat_think>` /
`<longcat_tool_call>…</longcat_tool_call>` markup into a chat message
(reasoning_content / content / tool_calls). This is **not part of the model** —
the port (correctly) does not vendor it; it's a consumer-side concern.

---

## 4. Weight loading

| | HF | vLLM-Omni |
|---|---|---|
| Backbone | `from_pretrained` via `_keys_to_ignore_on_load_unexpected = [r"model\.mtp.*"]` | `AutoWeightsLoader` skipping `model.mtp.*`; `visual/audio_head` and tokenizer subtrees routed to the replicated remote modules |
| Side modules | part of the same state dict | split out by `_SIDE_MODULE_PREFIXES` and `load_state_dict`ed into the remote classes (strict=False, missing/unexpected logged) |
| Decoder stages | lazy inside `lazy_decode_and_save` | `load_weight_subtree` (index-map driven, only pulls the `model.audio_tokenizer.*` / `model.visual_tokenizer.*` shards — never materialises the 74B backbone) + remote `from_pretrained` for `cosy24k_vocoder/hift.pt` and `image_decoder/image_decoder.safetensors` |
| Path placeholders | users must edit `WEIGHT_PATH_TO_LONGCAT_NEXT` | `resolve_checkpoint_relative_path` resolves the placeholder against the local model dir |
| `track_weights_loading` | N/A | remote-code submodules explicitly marked loaded to satisfy vLLM's init audit |

**Verdict:** different machinery (as required by the engine), same weights, with
the port being more careful about memory (subtree-only decoder loading).

---

## 5. Verified alignment table (values cross-checked against the checkpoint)

| Item | Checkpoint / HF | Port | Match |
|---|---|---|---|
| Token ids (`<longcat_img_start/end/pad/newline>`) | 131106 / 131107 / 131108 / 131109 | same (`longcat_next_utils.py`) | ✅ |
| Token ids (`<longcat_audio_start/end/pad>`) | 131103 / 131104 / 131105 | same | ✅ |
| Token ids (`<longcat_audiotext_start/end/pad>`) | 131120 / 131121 / 131122 | same | ✅ |
| Token ids (`<longcat_audiogen_start/end>`) | 131123 / 131124 | same | ✅ |
| `visual_offset` / `audio_offset` | 150581 / 131125 | same | ✅ |
| Visual codebook sizes | 8 × 16384 | same | ✅ |
| Audio codebook sizes | [8192, 4096, 2048, 1024×5] | same (`AUDIO_CODEBOOK_SIZES`) | ✅ |
| lm_head width | 131125 | 131125 | ✅ |
| n-gram hash vocab `m` | 78 × 131072 | `text_vocab_hash_size = 131072`, ratio 78 | ✅ |
| Audio sampling defaults | temp .5 / k 5 / p .85 / rep 1.3 | same | ✅ |
| Visual sampling defaults | temp .5 / k 1024 / p .75 / rep 1.0 | same | ✅ |
| cfg_scale | 3.0 | 3.0 | ✅ |
| audio sr / overlap | 24000 / 1200 | 24000 / 1200 | ✅ |
| audio chunk-end marker | `codebook_sizes[0]` = 16384 | 16384 | ✅ |
| visual head dims | 2048 / 4 layers / ffn 16 | same (remote class) | ✅ |
| audio head dims | 3072 / 4 layers / ffn 16 | same (remote class) | ✅ |

---

## 6. Documented divergences (port is intentionally different)

1. **Post-image text continuation** — HF emits EOS and ends generation right
   after `<longcat_img_end>`; the port resumes normal text generation.
2. **anyres prefix** — HF auto-injects `<longcat_img_token_size>{h} {w}</...>`;
   the port requires the caller to include it and to pass `token_w/token_h`.
3. **Audio start trigger** — HF waits for the model's `audiotext_start`; the
   port forces it at `gen_step == 0` (delay=0), so audio always starts at step 1.
4. **Audio chunk cross-fade** — HF replays the overlap seam (stutter); the port
   blends once.
5. **`audio_parallel_decoding`** — supported by HF, unimplemented in the port
   (matches checkpoint default `False`).
6. **max_gen cap** — port-only safety cap on audio length.
7. **CFG twin construction** — token-level repeat+blank (HF) vs string-level
   uncond build (port).
8. **Sentinel masking** — port masks visual level-0 end-of-image and all
   audio non-zero-level sentinels (HF masks none of these in the generate
   loop); a robustness improvement.
9. **Decoder grid overrun/short** — port truncates or returns empty output
   instead of asserting/crashing.

---

## 7. Open items / things to verify (suggested follow-ups)

- **N-gram hash parity**: confirm the vLLM `ngram_compute_n_gram_ids` kernel
  + streaming context reproduces the HF `NgramEmbedding` hashes for one prompt
  (the branch's `[longcat-ngram]` audit log exists for exactly this).
- **Tokenizer/BOS parity**: `add_special_tokens=False` in the port vs the HF
  default — confirm no 1-token shift that would perturb n-gram contexts
  (`[longcat-text]` audit log).
- **Visual logit span**: the branch logs a ~0.6× compressed final-logit span +
  re-ranked top-10 vs reference; the per-layer RMS audit hooks (`LONGCAT_AUDIO_DEBUG`)
  are wired to bisect where the divergence accrues. Post-MLA-fold this is the
  main remaining numerical question.
- **Image overrun behaviour in HF loop**: the port's comments reference the
  SGLang `output_processor.py`/`state_machine.py` for the "unreachable
  IMAGE_END" claim; the HF `generate` loop itself terminates at the grid bound
  (`is_img_end`), which the port matches — confirm the port's grid-bound
  termination is exercised by the multi-decoder deploy path.
- **Multi-segment audio**: the port's boundary-marker rows rely on
  `_extract_codes_from_output` keeping marker rows and
  `LongcatNextAudioDecoder._split_chunks` splitting on them — add a
  multi-chunk e2e test if not already covered.

---

## 8. File-by-file delta (quick reference)

| HF file | Port equivalent | Status |
|---|---|---|
| `modeling_longcat_next.py` | `modeling_longcat_next.py` | re-implemented (native backbone + runner hooks) |
| `modeling_longcat_ngram.py` | native `longcat_flash_ngram.py` + port glue | reused vLLM-native + custom sizing |
| `configuration_longcat_next.py` / `configuration_longcat_ngram.py` | config shim + `load_remote_hf_config` | shim + remote config |
| `modular_longcat_next.py` | `_sample_depth_head` driver | remote class reused |
| `modular_longcat_next_visual.py` | `_encode_images`, `LongcatNextImageDecoder` | remote class reused |
| `modular_longcat_next_audio.py` | `_encode_audios`, `LongcatNextAudioDecoder`, `talker_mtp` | remote class reused |
| `cosy24k_vocoder.py` | `LongcatNextAudioDecoder` vocoder step | remote class reused |
| `image_refiner.py` / `refiner_modules.py` | `LongcatNextImageDecoder` (via `lazy_decode_and_save`) | remote class reused |
| `processing_longcat_next.py` | `longcat_next_processor.py` + vendored processor | re-implemented for in-memory data |
| `parse_model_response.py` | — | not ported (client-side) |
| *(no analogue)* | `pipeline.py`, `stage_input_processors/longcat_next.py`, `LongcatNextMultiDecoder`, deploy YAMLs | port-only orchestration |

---

## 9. Meituan's `LongCat-Next-inference` repo (SGLang serving) vs HF reference and vLLM-Omni port

> Second pass (added on request). This section cross-checks the **inference
> repo** — `opensource_stuff/LongCat-Next-inference/`, i.e.
> `github.com/meituan-longcat/LongCat-Next-inference`, branch `main`, 3 commits
> (`af1a9a6` → `da1b3cd` → `70ab100`) — against the HF reference (Sections 1–8)
> and the vLLM-Omni port. The inference repo is Meituan's own **SGLang-based**
> serving harness for the LongCat-Next checkpoint; the vLLM-Omni port is an
> independent **vLLM-native** reimplementation of the same checkpoint. Code only
> was reviewed; **no weights** were downloaded. All line references below are to
> files inside `LongCat-Next-inference/` unless prefixed.

### 9.1 What the inference repo actually is

| Aspect | Detail |
|---|---|
| Serving engine | **SGLang** (`sglang.srt.*`), not vLLM. The LLM core is SGLang's own `sglang.srt.models.longcat_flash.FLASHForCausalLM` (MLA + MoE + `FusedOverEmbedding` n-gram + `flashmla` attention + `sgl_kernel.compute_n_gram_ids`), which the vLLM-Omni port re-implements natively as `FlashNgramModel` (paged KV, TP/PP, fused MLA/MoE kernels). |
| Model entry point | `modules/nmm_flash.py::NmmFlashForCausalLM` — subclasses `FLASHForCausalLM`, overrides `load_weights` (skips `audio_head.`, `model.audio_tokenizer.`, `visual_head.`, `model.visual_tokenizer.`, `model.audio_embed_layers.`; truncates `model.embed_tokens.weight`/`lm_head.weight` to `[:131125]`), overrides `forward` to feed `forward_batch.request_cache_input["input_embedding"]` as `input_embeds` (`CaptureHiddenMode.LAST`), and delegates sampling to `NmmSample`. |
| Checkpoint requirement | Needs the LongCat-Next checkpoint **plus a repackaged `nmm_infer/` config dir** inside the model root (`context.py:20`, `demo.py:68`), the `image_decoder/image_decoder.safetensors` + vocoder file (`postprocessor.py:29-33`), and the deployed `config.json`. Not verifiable offline — see 9.6. |
| Orchestration | `framework/fluentllm.py` (FluentLlmBackend) wraps the engine + an optional shared-memory **request cache** (`framework/request_cache/`) that caches `output_multi_ids(int64,8)` and `input_embedding(bfloat16,3072)` per round — Meituan's "Smart Prefix Caching / Hidden State caching" from the README. |
| Scope of the harness | Test driver (`demo.py`, `example/test_cases.yaml`) for `img_gen`, `img_und`, `aud_2_txt`, `spk_syn`, `aud_2_aud`. |

### 9.2 Architecture at a glance

The three implementations share the same math but differ in where the
multimodal decode loop runs.

```
HF reference (generate loop, single process):
  encoders -> n-gram emb -> backbone -> lm_head/depth-heads -> decoders

LongCat-Next-inference (SGLang, single model, INLINE depth heads):
  PreProcessor (flash_omni LongcatModel) computes the WHOLE prompt's hidden
  states -> SGLang engine decodes from those embeddings (mm-mode
  input_embeds+multi_ids). Each decode step: backbone forward -> NmmSample ->
  OmniImageHead/OmniAudioHead sample all 8 code levels off the last hidden state
  INLINE, state machine post-processes text/multi ids -> next-step input
  embedding computed by NmmSample (text oe + audio/visual code embeds). Depth
  heads + state machine live in the SGLang model worker.

vLLM-Omni port (pipelined, decoupled stages):
  Thinker stage (LongcatNextForCausalLM) -> LongcatNextAudioDecoder /
  LongcatNextImageDecoder / LongcatNextMultiDecoder stages. Multimodal tokens are
  generated by dedicated decoder stages that consume thinker hidden states, and
  fed back into the thinker as embeddings across the pipeline.
```

The inference repo keeps everything in one SGLang decode loop (LLM + depth heads
+ state machine co-located, CUDA-graphed on separate audio/image streams,
`output_processor.py:399-565`); the port splits the same interleaved semantics
into separate vLLM pipeline stages. Functionally equivalent, architecturally
different plumbing.

### 9.3 Component-by-component mapping (inference repo ↔ port)

| Inference-repo file | Role | Port equivalent |
|---|---|---|
| `modules/nmm_flash.py` | SGLang model entry (embedding prefill, weight-filter load) | `modeling_longcat_next.py::LongcatNextForCausalLM` |
| `sglang.srt.models.longcat_flash` (external) | Flash backbone + `FusedOverEmbedding` n-gram | native vLLM `longcat_flash_ngram.py::FlashNgramModel` + port n-gram glue |
| `modules/mllm_over_embedding.py` | n-gram lookup wrapper (sgl-kernel + Python fallback) | `_build_ngram_oe_ids` / `embed_batched` (`modeling_longcat_next.py:486-532`) |
| `modules/context.py` | per-req state machines, codebook embs (visual/audio), cuda-graph buffers, gen-type grouping | `_visual_gen`/`_audio_gen` state (`modeling_longcat_next.py:638-745`) |
| `modules/input_processor.py` | decode-step embedding mix (text/audio/visual) | `get_multimodal_embeddings` + `_ensure_*_code_embed_module` (`modeling_longcat_next.py:1336-1582`) |
| `modules/output_processor.py` | depth-head sampling + state-machine post-process + cuda graphs | `_sample_depth_head`/`_sample_cfg_visual_codes`/`talker_mtp` + decoder stages |
| `modules/image_head.py` | `OmniImageHead`/`OmniAudioHead` (re-impl of HF `CasualDepthTransformerHead`) | HF remote `CasualDepthTransformerHead` reused unchanged |
| `modules/visual_emb.py` | serving-side visual codebook emb + `pre_buffer` MLP | `_code_embeddings` + `visual_embedding_layer` (HF remote) |
| `modules/state_machine.py` | routing state machine | port's `_visual_gen`/`_audio_gen` + decoder stage machines |
| `modules/special_token.py` | token-id constants (from `nmm_infer` config) | `longcat_next_utils.py` constants |
| `processor/preprocessor.py` | hidden-state prefill encoder (flash_omni `LongcatModel`) | `longcat_next_processor.py` + vendored HF processor |
| `processor/flash_omni/modeling_longcat_oe.py` | HF-style training/encode model (`LongcatModel`, `NgramEmbedding`, `LongcatAudioTokenizer`) | (reference math only; port reuses checkpoint remote classes) |
| `processor/postprocessor.py` + `processor/decoder/*` | image/audio decode (refiner, vocoder, cross-fade) | `LongcatNextImageDecoder` / `LongcatNextAudioDecoder` |
| `nmm_pf.yaml` | deploy config (vocab 131125, tp4/ep, flashmla, request cache) | `deploy/longcat_next_*.yaml` |

### 9.4 Verified-aligned behavior (inference repo ≡ HF ≡ port)

Confirmed by reading the inference repo against the HF checkpoint sources:

- **Token ids / offsets**: `special_token.py` reads `visual_config.image_start/end/pad/line_token_id` (131106–131109), `audio_config.audiotext_*`/`audiogen_*` (131120–131124), EOS 2; hardcodes `IMAGE_TOKEN_SIZE_START/END` = 131090/131091; `AUDIO_END_FLAG_ID = audio_config.vq_config.codebook_sizes[0]` = 8192. Matches `longcat_next_utils.py`.
- **lm_head / embedding width = 131125**: `nmm_flash.py:44` truncates `embed_tokens`/`lm_head` to `[:131125]` and `nmm_pf.yaml:18` sets `vocab_size: 131125`. Matches the port's 131125-wide `LogitsProcessor` (`modeling_longcat_next.py:180-191`) and the checkpoint's `text_vocab_plus_multimodal_special_token_size=131125`.
- **N-gram math**: `mllm_over_embedding.py` + sgl-kernel compute `id * power_mod % mod` per order/split with `mod = m + index*2 + 1` (m = `ngram_vocab_size_ratio` × `text_vocab_size`), `+ exclusive_oe_embeder_size_sums[index]`, mean over 13 components (`word_emb + 12`). Identical to HF `NgramEmbedding` (`modeling_longcat_oe.py:753-918`, `precompute_vocab_mods`) and the port's kernel. The port's `text_vocab_size=131072` hash space + `ratio` is what the inference repo's `FusedOverEmbedding` is sized on.
- **Special tokens excluded from n-gram context**: `input_processor.py:216-233` zeroes special tokens in the tail window of `oe_token_table`; HF zeroes them before n-gram; port's `oe_ignored` ids do the same (`modeling_longcat_next.py:782`).
- **MLA input scaling**: the training model applies `mla_scale_q_lora`/`mla_scale_kv_lora` = `(hidden/q_lora_rank)^0.5`, `(hidden/kv_lora_rank)^0.5` on `q_pass/q_rot/k_pass` (`modeling_longcat_oe.py:572-606`) — this is exactly the scale the port folds into `q_b_proj`/`kv_b_proj` (`_apply_mla_scale_fold`). Confirms the fold is necessary and matches.
- **Depth head**: `OmniImageHead`/`OmniAudioHead` (`image_head.py`) reproduce HF `CasualDepthTransformerHead` (`modular_longcat_next.py:81-157`): `[LLM hidden; cumsum(level embeds)]` → `hidden_norm`+`hidden_proj` → N× `CasualDepthTransformerLayer` (causal flash-attn over the 8 depth positions + depth-mixed SwiGLU einsum) → `headnorm` → `heads[vq_size+1]`. Same weights layout (plus TP sharding helpers and a `hidden_in_proj`→`hidden_proj` rename for the audio head under `use_oe`).
- **Visual embedding**: serving-side `VisualEmbeddingBridge` = sum of 8 per-level codebook embeds sliced from `model.embed_tokens` at `visual_offset` (each `codedim+1`) + `pre_buffer` (LayerNorm + SwiGLU MLP, no attention — matches HF `VisualEmbeddingBridge`). Equivalent to HF `get_visual_embeddings` and the port's `_code_embeddings`+`visual_embedding_layer`.
- **Audio embedding**: 8× `nn.Embedding(codedim+1)` sliced from `model.embed_tokens` at `audio_offset`; decode-step mix = `ext_ids_emb + text_emb + audio_embs` with rows masked where level-0 id ∈ {0, 8192} (`input_processor.py:116-148`). The port documents the same 3-stream construction as equivalent to this (`modeling_longcat_next.py:1347`).
- **Sampling family**: rep-penalty over per-level past codes → `/temperature` → softmax → top-k/top-p (no min-p), sequential codebook autoregression (`output_processor.py:369-397`). Matches HF/port defaults (audio 0.5/5/0.85/1.3, visual 0.5/1024/0.75/1.0 from `generation_config.json`).
- **Marker masking**: `logits[:, codebook_sizes[i]] = -inf` for image sampling (`output_processor.py:312`) — the chunk-end class is never sampled for images; audio ends only on the level-0 marker (8192). Consistent with HF/port sentinel handling.
- **Anyres / grid**: `<longcat_img_token_size>{h} {w}</longcat_img_token_size>` prefix + `token_w` newline cadence every `(token_w+1)`-th step (`output_processor.py:204-216`; demo `img_gen` uses 18×18 → 324 frames, matching the port's `token_h*token_w` grid bound).
- **Image decode stage**: `VisionTransformerDecoder` + `ImageRefinerContainer`/`RefinerPipeline` + `FlowMatchEulerDiscreteScheduler` and the `quantize.codebooks[idx].embed` feature path are the **same components** HF's `LongcatNextVisualTokenizer.lazy_decode_and_save` uses (`omni_gen2_new/modular_longcat_next_visual.py` ≡ HF visual module). The port reuses the same checkpoint remote classes.
- **Audio cross-fade**: `decode_save_concat` (`audio_decode.py`) = HF `decode_save_concat2` formula (overlap ramp, default 1200 here, sr 24000); the port matches the 1200 overlap.

### 9.5 Divergences (inference repo vs HF / vs port)

1. **Engine**: SGLang (`FLASHForCausalLM` + sgl-kernel ngram + flashmla) vs vLLM-native port. Numerical expectations should be identical per 9.4, but the inference repo's exact `FusedOverEmbedding` internals are external to this repo (see 9.6).
2. **Orchestration model**: single SGLang decode loop with inline depth heads + state machine + request-cache hidden-state prefill, vs the port's pipelined thinker→decoder stages. The port also adds a **multi-decoder** stage for combined audio+visual; the inference repo interleaves audio+visual in one loop via `forward_batch.input_multi_ids`.
3. **Per-request single-purpose mode**: the inference repo derives `gen_image`/`gen_audio` from the **last prompt token** (`demo.py:91-102`) and the state machine enters image/audio stage directly. **Image-end (`131107`) → ABORT** (`state_machine.py:84-87`) — the request is killed, matching HF's hard stop but **not** the port, which continues text after image. After audio, the repo only loops chunks or aborts; it never returns to text (`state_machine.py:108-119`). The port supports arbitrary text→image→text→audio→text sequences. This is the biggest behavioral difference and is an **orchestration** choice, not a weight-level one.
4. **Audio start delay**: the repo supports `delay ∈ {0, inf}`; the demo default is `inf` (audio starts the step after text-end: `delay=min(delay, gen_step)` at first `audiotext_pad`, then `ext_ids=audiotext_start` when `gen_step==delay`), with `spk_syn` explicitly setting `delay: 0` (`test_cases.yaml:58`). The port hard-forces audio-start at `gen_step==0` (delay=0). Both semantics exist in the checkpoint; the port only wires delay=0.
5. **Audio max steps**: state machine cap `max_gen=1000` (`input_processor.py:79`) vs the port's `LONGCAT_MAX_GEN` (default 30 s × 25 fps ≈ 750). Both are safety caps; normal termination is the 8192 marker.
6. **Audio decode path (older-stack)**: the repo decodes audio via the omni `LongcatAudioTokenizer.decode` (OmniAudioDecoder + flow-matching mel) then `Cosy24kVocoder` on the mel (`postprocessor.py:63-73`, `audio_decode.py`), whereas HF/port decode with `Cosy24kVocoder` **directly on the level-0 codes** (`modular_longcat_next_audio.py` `lazy_decode_and_save`). The repo's `processor/flash_omni/audio_modeling_omni.py` is an older omni-gen2 audio stack; strict `load_state_dict` from `model.audio_tokenizer.` weights needs checkpoint verification (see 9.6).
7. **CFG construction**: the repo issues **two SGLang requests** paired by `cfg_pair_id` (cond = full prompt, uncond = regex-stripped `<longcat_img_token_size>…<longcat_img_start>` only, `demo.py:224-239`), with cond/uncond as even/odd rows; the port builds the cond/uncond twin **inside one request**. Both yield a 2× image batch, `cfg_scale*(cond−uncond)+uncond`, forced-identical sampled tokens.
8. **Hidden-state prefill**: the repo computes the **whole prompt** hidden-state sequence in the preprocessor (`flash_omni LongcatModel.forward` returns `inputs_embeds`; the transformer layers are commented out) and hands it to the engine (`input_embeds` mode), skipping the serving model's own `embed_tokens` on prefill. The port feeds token ids and merges n-gram/multimodal embeddings per-span in the vLLM runner. Same math, different plumbing.
9. **`shift_right_ignore_eos`**: the repo's *encoder* `NgramEmbedding` segments n-gram context at EOS/special tokens (`modeling_longcat_oe.py:780-830`); the serving kernel approximates it via ignore-token zeroing. The port's `NgramCache` replicates HF's stateless rolling context — the §7 n-gram parity test is exactly what reconciles these.

### 9.6 Caveats / open items for the inference repo

- **External dependency**: `sglang.srt.models.longcat_flash` is not vendored here; its `FLASHConfig.onmi_extra_info` (`context`/`input_processor`/`output_processor`) contract and `FusedOverEmbedding` exact math are assumed equivalent to HF's `LongcatFlashNgramModel`.
- **Checkpoint-side files required**: `nmm_infer/` config dir, `image_decoder/image_decoder.safetensors`, and the vocoder weight path are not in the repo; strict weight-loading compatibility with the LongCat-Next checkpoint (esp. the older omni audio encoder/decoder stack, and `image_decoder`/refiner key names) is unverified without the checkpoint.
- **Default-vs-request params**: the repo's *demo* uses `token_w=18`, `cfg_scale=1.8` and demo-level multi-sampling defaults (0.2/20/0.85/1.1); the checkpoint `generation_config.json` defaults are `token_h/w=37`, `cfg_scale=3.0` (used by the port). These are per-request choices, not weight differences.
- **`pad_token_id`**: `generation_config.json` sets pad 3 while `config.json` has pad `None`; the repo's embedding mix masks audio rows by id, so this only matters if a request emits id 3 (low risk).

---

*End of Section 9 (added in a second pass: LongCat-Next-inference comparison).*
