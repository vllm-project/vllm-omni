# B1 — CosyVoice3 RAS sampler: shared notes

Working notes for splitting RFC #6870 B1 between two contributors. Everything
below is pinned to `vllm_omni/model_executor/models/cosyvoice3/cosyvoice3.py`
on upstream `main` @ `e284d907`, whose sampler is identical to the 0.28.0
release. vLLM's `random_sample` is identical in 0.28.0 and 0.29.0.

## Status

| Item | State |
|---|---|
| Feasibility spike | Branch [`spike/b1-ras-sync`](https://github.com/suyanli220/vllm-omni/commit/1e78da3a). Not mergeable. Stage 0 on H100: **4.175 → 3.775 ms/token** (−9.6%, t = −11.2, n = 10 per arm), token count unchanged. |
| PR1, device-resident primary sampling | Branch [`b1/device-primary-sampling`](https://github.com/suyanli220/vllm-omni/commit/2fbc941e). Passes the equivalence harness below. Stage 0 on H100: **4.212 → 4.059 ms/token** (−3.6%, paired t = −9.9, 10/10 rows faster), E2E −3.1% (paired t = −6.8), token count identical row by row. |
| PR2, device-resident RAS policy | Not started. Owner to be agreed. |

The spike and PR1 remove different sets of syncs, so the spike's 9.6% is not
PR1's number. See the inventory.

## Current semantics

Per decode step, for the whole batch:

1. **Routing.** The RAS path runs only when the stage is the talker, no
   logprobs are requested, `temperature` is not `None`, there are no bad
   words, and both frequency and presence penalties are zero. An all-greedy
   batch has `temperature is None` in vLLM, so it goes to vLLM's own `Sampler`
   and never reaches RAS.
2. **Logit prep.** Cast to float32, apply `allowed_token_ids_mask` (True means
   masked out), apply non-argmax-invariant logits processors, raise if any row
   has no finite logit, fill non-finite entries with −inf.
3. **Per request.** Read `temperature`. Below `1e-5` the row takes
   `argmax(logits)` and skips everything else, including RAS.
4. **Primary draw.** `ws = log_softmax(logits / temperature)`. Sort
   `softmax(ws)` descending with a stable sort. Keep token *i* iff
   `cumsum_before[i] < top_p` and `i < top_k`. A non-positive `top_k` means no
   limit. Draw with vLLM `random_sample` over the full sorted vector, zeroed
   outside the keep-set.
5. **Repetition check.** If `win_size > 0` and the request has history, count
   occurrences of the drawn id in the last `win_size` outputs.
6. **Conditional resample.** If `count >= win_size * tau_r`, set that id's
   score to −inf, restore it if nothing finite remains, and draw again from
   `softmax(ws)` over the **whole vocabulary**. No top-p or top-k on this draw.

Facts worth knowing before anyone writes tests:

- **With the shipped config, one repeat is enough to trigger RAS.**
  `win_size = 10` and `tau_r = 0.1`, so the threshold is 1: the fallback fires
  whenever the drawn id appears anywhere in the last ten tokens. On sticky
  logits the harness sees it fire 73 times in 80 draws. The real trigger rate
  on speech tokens has not been measured and should be.
- **Defaults depend on batch composition.** vLLM passes `top_p = None` when no
  request in the batch has `top_p < 1`, and likewise for `top_k`. The model
  then substitutes its config default of 0.8 and 25. A request asking for
  `top_p = 1.0` gets 0.8 when batched alone and 1.0 when batched with a request
  that sets `top_p < 1`. This is pre-existing. PR1 preserves it deliberately.
- **`random_sample` divides its input in place.** Nothing may read `weights`
  after the call.
- **The repo has three RAS implementations with three RNG schemes.**
  CosyVoice3 uses vLLM's exponential-noise `random_sample`.
  `models/common/nucleus_ras_sampling.py`, used by GLM-TTS, uses
  `torch.multinomial` with boolean-mask indexing, which syncs; its docstring
  says CosyVoice3 uses it, and CosyVoice3 does not. `audio8_tts/sampling.py`
  already has a sync-free batched RAS that draws both candidates every step
  and selects with `torch.where`. That last one is prior art for PR2, though
  its RAS rule differs: it resamples from a flatter distribution rather than
  blocking the repeated id.

## RNG contract

This is the decision that has to be made before PR2 is written.

**Seeded requests today.** Each request with a seed owns a `torch.Generator`,
keyed by batch slot. Per step it advances by one full-vocabulary row of
exponentials for the primary draw, and by one more row **only if** the
repetition check fires. Unseeded requests draw from the global stream and have
no reproducibility today.

**PR1 keeps seeded trajectories exact.** `random_sample` overwrites each seeded
row's noise from that row's own generator over the same width, and the legacy
RAS boundary still draws only when it fires. Per generator, the sequence of
draws is unchanged. The harness confirms token-for-token identity over 40-step
trajectories in every case, including one where RAS fires 73 times.

**PR2 cannot keep that for free.** A sync-free fallback needs the fallback
candidate before the predicate is known on the host. Three options:

| Option | Seeded trajectory | Syncs left in RAS |
|---|---|---|
| A. Draw the fallback for every row, select with `where` | Changes, but is still deterministic per seed | 0 |
| B. Compute the predicate for the whole batch, read it with one `tolist()`, draw fallbacks only for rows that fired | Exact | 1 per step, instead of 1 per request |
| C. As today | Exact | 1 per request |

Option A is what Audio8 does. Option B is a middle ground nobody has measured.
CosyVoice3's shipped deploy config sets no seed, so option A changes no
default behaviour, but it does change what a user-supplied seed reproduces.
**This is a maintainer call, not ours.**

## The boundary

```
logits
  ↓  routing, logit prep                       unchanged
  ↓
_sample_primary()                              PR1: device-resident
  → token_ids [N], weighted_scores [N,V], greedy [N]
  ↓
─────────────── _apply_ras_legacy() ───────────────
  per request: history → predicate → fallback  PR2 replaces this function
  ↓
SamplerOutput(token_ids.int32)
```

PR2's contract with PR1 is the signature of `_apply_ras_legacy`: it receives
the primary ids, the scores to resample from, and the greedy mask, and returns
final ids. PR2 should not need to touch `_sample_primary`.

## Sync inventory

Per decode step. N is the number of requests on the RAS path.

| Blocking read | Count | Removed by |
|---|---|---|
| `temperature`, `top_p`, `top_k` via `_req_scalar` | 3N | **PR1** |
| Primary token `.item()` | N sampled | **PR1** |
| Greedy `argmax().item()` | per greedy row | **PR1** |
| Fallback token `.item()` | per firing | **PR1**, because ids are tensors now |
| Greedy-row mask `tolist()` | 1, mixed batches only | Added by PR1; gone once PR2 lands |
| Repetition predicate `.item()` | N sampled with history | **PR2** |
| History upload `torch.as_tensor(list)` | N sampled, H2D | **PR2** |
| No-finite-logits check | 1 | Open |
| Frequency and presence penalty probes | 2 | Open |

The harness counts blocking reads in the sampler:

| | Legacy | PR1 | Spike |
|---|---|---|---|
| Batch 1 | 8 | 4 | 1 |
| Batch 4, one greedy row | 20 | 7 | — |

The H100 trace shows 14.2 `aten::item` and 3.0 `aten::is_nonzero` per step at
batch 1. The sampler owns 6 and 2 of those. The other ~8 `item` calls are
outside the sampler and outside B1.

Measured on H100, and priced with the per-call cost from the stage-0 trace
(`aten::item` ≈ 85 µs, `aten::is_nonzero` ≈ 222 µs):

| | Reads removed | Host time removed, nominal | Stage-0 gain | Realised |
|---|---|---|---|---|
| PR1 | 4 `item` | ≈ 0.34 ms | 0.153 ms/token | 45% |
| Spike | 5 `item` + 2 `is_nonzero` | ≈ 0.87 ms | 0.400 ms/token | 46% |

Both land at about the same realisation rate. **Do not use that rate to price
individual syncs.** A direct test of the two penalty probes did not follow it:

| Arm, same session, in this order | Stage-0 ms/token | Paired diff vs first |
|---|---|---|
| Both probes (as shipped) | 4.1262 | |
| Folded into one read | 4.1256 | −0.0006, t = −0.04 |
| No probe (upper bound, not mergeable) | 3.9915 | −0.1347, t = −3.86 |
| Both probes again | 4.2196 | **+0.0933, t = +16.7** |

The repeat of the first arm came out slower in every row, so the session drifted
and the middle arms cannot be read at face value. Two things still stand. Folding
two reads into one bought nothing measurable. And the pattern fits a simpler
mechanism than per-sync pricing: the first blocking read after the model forward
absorbs the wait for the whole GPU queue, and later reads in the same step find
the queue already drained. On that reading `is_nonzero` looks 2.6× as expensive
as `item` in the trace only because it is the first read to run, and removing a
read helps only if no other read takes its place as the drain point. A rerun in
ABCCBA order, with a trace of the no-probe arm, is pending.

PR1's gain at batch 1 is smaller than the spike's. The spike also skipped the
three validation reads that run first in the step, so under the drain-point
reading it moved the first blocking read much further along. PR1's advantage
grows with batch size: per sampled request it cuts five reads to one, and the
one left is PR2's.

## Correctness matrix

"Exact" means token-identical over a multi-step trajectory with seeded
generators. "Support" means the drawn ids stay inside the keep-set, which is
all that can be checked for unseeded rows.

| Case | Contract | PR1 harness | PR2 owner | H100 | 4090 |
|---|---|---|---|---|---|
| Benchmark config, batch 1 | exact | ✅ | | ✅ PR1, token counts identical run by run | |
| Batch > 1, mixed temperature | exact | ✅ | | | |
| Greedy row inside a random batch | argmax | ✅ | | | |
| Mixed `top_p` / `top_k` per request | exact | ✅ | | | |
| `top_p` / `top_k` absent → config defaults | exact | ✅ | | | |
| No history | exact | ✅ | | | |
| Partial history, shorter than `win_size` | exact | ✅ | | | |
| Repeated history, RAS fires | exact under A/B/C choice | ✅ legacy RAS | | | |
| Single valid token | that token | ✅ | | | |
| `allowed_token_ids_mask` | exact | ✅ | | | |
| Per-request generator, mixed seeded and unseeded | seeded exact, unseeded support | ✅ | | | |
| Every candidate blocked by RAS | original score restored | not covered | | | |
| Logits processors present | exact | not covered | | | |
| Real speech tokens: trigger rate, audio sanity | measured | — | | | |

The harness has a negative control: an off-by-one in the top-k limit is
detected. Without that, "identical" would not mean much.

## Acceptance

**PR1.** Sampler `item` and `_local_scalar_dense` per step clearly lower in a
stage-0 trace. Host gap lower. Stage-0 ms/token independently better, measured
against a baseline in the same session. Seeded trajectories exact.

**PR2.** Remaining RAS reads at or near zero. Distribution semantics preserved
under whichever RNG option is chosen. Positive stage-0 result on both H100 and
RTX 4090.

Whether PR1 and PR2 ship as one PR or two stacked PRs is decided after both
have numbers.

## Open questions

1. RNG contract for PR2: option A, B or C.
2. The two penalty probes. Folding them into one read gave no measurable gain.
   Removing both may, but that run drifted. `no_penalties` cannot replace them,
   because the deploy config's `repetition_penalty: 1.0001` makes it false. If
   they matter, the fix is to decide routing where sampling params are still on
   the host, not to make the read cheaper.
3. The no-finite-logits check. Keep it as a sync, move it behind a debug flag,
   or repair on the device?
4. Should CosyVoice3 and the common helper converge, given they use different
   RNG schemes?
5. The composition-dependent `top_p` / `top_k` default. Fix it, or document it?
