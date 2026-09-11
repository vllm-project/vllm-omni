# Model-local KV caches

Some models keep their attention KV as HuggingFace `transformers` cache objects
instead of using the engine's paged KV manager. Each such model declares what it
holds by implementing `model_local_kv_specs()`, and
`OmniGPUModelRunner.load_model` sums and logs the result after load.

## What is and is not already accounted for

This distinction decides what the protocol is for, and an earlier revision of
this page got it wrong by claiming none of this memory was accounted for.

CUDA-graph capture happens *during* `load_model` -- MiMo in its constructor,
Qwen3-TTS in `load_weights`. `determine_available_memory()` then samples
per-process GPU memory through NVML and subtracts it from the requested budget
(`vllm_omni/worker/base.py`). So the **graph-resident** portion is already
charged. Subtracting the declared total from the KV budget would double-count
it.

What is *not* charged is the **per-request** portion, which is allocated after
profiling. That half is not small: Qwen3-TTS's codec decoder is 4.44 MiB per
request, so a server at `max_num_seqs=256` reaches roughly 1.1 GiB that no
per-stage number predicted.

The consumer therefore reports and does not subtract.

## What the four models declare

| Model | Cache | Bounded by | Per row | Rows |
|---|---|---|---|---|
| Qwen3-TTS codec decoder (async-chunk only) | sliding `DynamicCache` | `sliding_window - 1 = 71` | 4.44 MiB | `max_num_seqs`, x2 for the working copy |
| Qwen3-TTS graph pool | retained captures | same | 4.44 MiB | fixed, from captured shapes |
| MiMo-Audio local transformer | `DynamicCache` | `group_size + max(delay_pattern) = 11` | 704 KiB | `max_num_seqs` |
| MiMo-Audio graph pool | captured | same | 704 KiB | fixed 261 (bucket sum) |
| MiniCPM-o Whisper encoder (streaming only) | `EncoderDecoderCache` | `embed_positions` rows = 1500 | 140.62 MiB | fixed 1, per session |
| ming_flash_omni talker | `StaticCache` | hardcoded `max_cache_len = 2048` | 24.00 MiB | fixed 1 |

**No cache is bounded by `max_model_len`.** Each has its own mechanism: a
sliding window, a decode-loop trip count, an encoder frame limit, a constant.

**ming's geometry does not exist in this repo.** `self._llm_config` is a
`Qwen2Config` from the checkpoint, so layers, kv-heads, head-dim and dtype are
only knowable after load. That is what makes this a runtime query rather than a
table.

## Lifetime is not multiplicity

`scope` records when memory appears and goes. It does **not** decide how wide
the allocation is, and an earlier revision that derived scaling from it was
wrong in two different directions at once:

- ming's talker is per-call but serialized -- `forward` takes
  `runtime_additional_information[0]` and runs the whole AR loop inline -- so
  scaling it by `max_num_seqs` over-reported it by that factor.
- MiniCPM-o's encoder is per *session*, not per sequence, so `max_num_seqs` is
  simply the wrong number for it.

So width is declared explicitly, by naming the driver. `RowDriver` has two
values, because two are all the known caches need:

- `FIXED` -- a count the model controls, with `rows_fixed` and a required
  `rows_reason`. `rows_fixed=1` on a cache that looks per-request is a claim a
  reviewer must be able to check.
- `MAX_NUM_SEQS` -- one row per in-flight sequence; only the engine knows the
  value.

A third value belongs here when a model actually has a cache the engine widens
by some other number. An earlier revision added one speculatively, for a single
declarer, and paid for it with a class hierarchy, an engine-capacity object and
a conditional-annotation field.

`rows` counts row-equivalents at peak and deliberately does not distinguish
"one allocation N rows wide" from "N allocations of one row". Those cost the
same, and a revision that tried to carry both got the object topology wrong in
three of four declarers: MiniCPM-o keeps one cache object per session while
asserting batch size 1, and both graph pools keep one distinct object per
captured bucket. The layout belongs in `allocation_note`, where it cannot be
mistaken for arithmetic.

## Declaring nothing is a valid declaration

A declaration must be inert when its path is. Qwen3-TTS holds KV only in
async-chunk mode -- stateless decoding runs `_forward_exact`, which calls the
transformer without `use_cache` -- so the stateless path declares nothing.
Getting this wrong is easy in both directions, and both have happened here: one
revision hung the declaration off the CUDA-graph wrapper and reported zero under
`enforce_eager`, where the eager async-chunk path does allocate; the next
declared unconditionally and invented roughly a gigabyte for stateless
deployments.

Where the model cannot see the deciding setting, it declares the honest unit
and says so in `allocation_note`. MiniCPM-o's encoder cache exists only on the
streaming path and is one object per session; the session cap is not visible
from the model, so it declares one session rather than scaling by a number it
would have to guess at.

## Finding the declarers

`collect_model_local_kv_specs()` walks `named_modules()`. Two wrinkles:

- The runner's `self.model` may be a `CUDAGraphWrapper`, which is not an
  `nn.Module` but forwards attribute access to its runnable. The walk therefore
  works either way, but entering through both the wrapper and its runnable would
  count a root-level declarer twice. The collector calls the supported
  `unwrap()` accessor first.
- An owner that is not a submodule at all (Qwen3-TTS's graph wrapper, ming's
  `MingAudioGenerator`) is reached by one explicit hop from the module that
  holds it.

## Adding a model

Return one entry per distinct allocation. Prefer `spec_from_hf_config()` and
pass the same config object the cache is built from; it raises rather than
guessing when an attribute is missing. `capacity_source` is diagnostic text --
never branch on it.

## Known limits

Row counts taken from live state after warmup miss caches `transformers` has
not yet materialized, which makes Qwen3-TTS's graph-pool figure a floor for the
xvec path.

MiMo's graph-pool cache has no reachable Python object -- it is a local inside
the captured frame -- so its width comes from the captured bucket list rather
than from tensors. Its two entries are also mutually exclusive per call: a call
either replays a bucket or runs eagerly. The sum is the right ceiling for the
process and the wrong number for a single request.

MiMo captures every bucket gated only on `torch.cuda.is_available()`
(`mimo_audio_llm.py:670`), so those 179.44 MiB stay resident even when the
deploy config sets `enforce_eager: true`. Gating it is a one-line change that
needs a model boot to verify, so it is noted rather than bundled here.
