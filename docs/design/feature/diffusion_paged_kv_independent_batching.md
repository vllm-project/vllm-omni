# Paged Independent-Request Batching and Step Execution

This document is the implementation contract for enabling HunyuanImage-3.0
independent-request batching and step execution under
`diffusion_kv_mode=paged_scheduler`. It is a design contract, not a user-guide
support claim. HunyuanImage3 now implements the three slices below for local
paged KV; native AR-KV multi-request and the user guide remain out of scope.

Terminology follows the
[diffusion continuous batching design](diffusion_continuous_batching.md).
Paged KV ownership, page lifecycle, and native attention stay in the
[scheduler-managed paged KV design](diffusion_paged_kv_cache.md). This page
only changes how Hunyuan groups public requests and how it hands spans to the
paged planner.

The workstream is tracked by [#9192](https://github.com/vllm-project/vllm-omni/issues/9192)
and [#9663](https://github.com/vllm-project/vllm-omni/issues/9663).

## Table of Contents

1. [Overview](#overview)
2. [Current Blockers](#current-blockers)
3. [Span Representation](#span-representation)
4. [Execution Slices](#execution-slices)
5. [Grouping Rule](#grouping-rule)
6. [Image-to-Image Spans](#image-to-image-spans)
7. [Out of Scope](#out-of-scope)
8. [Related Files](#related-files)

## Overview

[#9192](https://github.com/vllm-project/vllm-omni/issues/9192) asks the Hunyuan pipeline to consume multiple request rows under paged
KV, and to group so that one mismatched boundary cannot fail the whole batch.
That is one feature and three execution paths. CFG branches of one public
request are already two paged rows; they are not independent-request batching.

`max_num_seqs` still counts public requests.
`diffusion_kv_max_rows_per_request` still sizes internal CFG rows of one
request (1 without CFG, 2 for standard cond/uncond).

## Current Blockers

These were the Hunyuan paged blockers this contract removed. They stay in the
table so the slice split remains reviewable.

| Path | Configuration | Former Hunyuan paged blocker |
| --- | --- | --- |
| Single-request step | `step_execution=true`, `max_num_seqs=1` | `prepare_encode` / `denoise_step` raise when paged is active |
| Step continuous batching | `step_execution=true`, `max_num_seqs>1` | The same raise, plus `TORCH_SDPA` via `_ensure_grouped_attention_backend_supported`, plus later-step span shifting |
| Request-level batching | `step_execution=false`, `max_num_seqs>1` | `supports_request_batch = False`; `prepare_model_inputs` rejects `prepared_layout` when `batch_size != 1` |

The dense grouped-step merge pads later-step rows onto a uniform max-prefix
grid (`_merge_attention_masks` / `_shift_full_attn_spans` inside
`_merge_step_model_inputs`) so a 4D mask stays well-formed. That merge is the
step-path representation problem. Skipping it does not implement request-level
batching.

## Span Representation

The paged planner already accepts grouped-step geometry where rows share the
image-token count but keep different per-row sequence lengths. This is an
adapter/planner fact, not Hunyuan E2E: the pipeline still raises before
attention.

With `prefix_lens=[9, 5]` (`seq_lens=[17, 13]`, `query_lens=[8, 8]`):

| Spans given to paged | Dense `_check_homogeneous` | Paged `build_paged_piecewise_plan` |
| --- | --- | --- |
| `[[(9,17)], [(9,17)]]` — output of `_shift_full_attn_spans` | Pass | Rejected: `(9,17)` is out of range for `seq_len=13` |
| `[[(9,17)], [(5,13)]]` — true per-row coordinates | Rejected | Pass (`homogeneous_batch_shape=(2, 8)`) |

`_shift_full_attn_spans(spans, prefix=5, max_prefix=9)` produces the first
row. Dense grouped merge needs that uniform grid. The paged planner needs the
second. The two representations are mutually exclusive.

`_check_homogeneous` is the dense FLASH piecewise helper, not the TORCH_SDPA
4D-mask merge. The `TORCH_SDPA`-only guard is therefore not the paged blocker:
paged builds no dense mask and does no padding. Dense grouped steps keep
`TORCH_SDPA`. Paged grouped steps run under paged / `FLASH_ATTN`.

The shift only runs on later denoise steps (`_prompt_kv_prefix_lens` is `None`
on step 0). First-step spans are already per-row. CFG interleave from
`_step_row_order` (cond rows, then uncond rows) uses the same unshifted
contract: paged accepts
`[[(9,17)], [(5,13)], [(9,17)], [(5,13)]]` with matching per-row `seq_lens`.

When paged is active on the step path:

- Do not call `_shift_full_attn_spans`.
- Keep per-row `seq_len` and true `full_attn_spans`.
- Do not force uniform lengths through `_merge_attention_masks`.

## Execution Slices

Implement in this order. Each slice is independently reviewable.

### 1. Single-request step

Lift the paged reject in `_validate_step_request` / `denoise_step` for
`max_num_seqs=1`. CFG rows of one request share a prefix, so the later-step
shift is a no-op. No span-representation change.

### 2. Step continuous batching

Apply the paged span contract above. Drop the `TORCH_SDPA`-only requirement
for a paged group. Keep the current dense grouped-step guard. Group with the
rule below so one mismatched row is split out instead of failing the wave.

### 3. Request-level batching

Enable `supports_request_batch` and a multi-request `prepare_model_inputs` /
row-pack path. This is a different merge than `_merge_step_model_inputs`.
The first request-batch slice is local paged KV only. The native AR-KV
connector stays `max_num_seqs=1` until that workstream lands.

[#9192](https://github.com/vllm-project/vllm-omni/issues/9192) "consume multiple request rows" is slice 2 plus slice 3. Slice 1 is the
smallest paged step enablement.

## Grouping Rule

[#9192](https://github.com/vllm-project/vllm-omni/issues/9192) said "CFG shape, sequence length, and prefix-hit boundary". Sequence
length here means image-token / latent geometry, not full `prefix+image`
`seq_len`. The span table above is two rows with different `seq_len` in one
paged group. Current dense `_step_group_key` already groups by CFG, latent
shape, `num_image_tokens`, and `ar_kv_reuse_len`, not by full `seq_len`.
Grouping by full `seq_len` would split geometry the paged planner accepts, and
would be stricter than today's dense grouped step.

A paged group may share one forward only when all of these match:

- CFG factor (`1` or `2`)
- latent spatial shape and `num_image_tokens`
- first-step versus later-step
- `ar_kv_reuse_len` and whether imported AR KV is present
- piecewise segment structure after prefix-cache / `query_offset` slicing
  (same number of causal/full segments and the same mode sequence)

Prefix-hit boundary means: after the cached prefix is sliced off, the
remaining query/segment structure must match. Different hit lengths that leave
different remaining segment counts split the group. Different AR prefix
lengths that still leave one image-tail full-attn segment may stay together.

One mismatched row is split into another group. It must not fail the rest of
the batch.

Local prefix hits in step execution are carried from admission metadata to each
request state separately from native AR-KV computed tokens. The first forward
slices each CFG row at its own cached boundary; later image-only steps do not
apply that cut again. Group signatures use the planner's segment construction.

The initial step implementation still groups first-step requests by their full
query width. Different prompt lengths can therefore prefill in separate groups
and join compatible image-only groups on later steps. Request-level packing
supports heterogeneous prompt lengths within a compatible group.

## Image-to-Image Spans

Do not treat "reference spans at different prefix offsets" as a planner
reject. Well-formed later-step i2i query is the image tail only: prefix-region
reference spans do not intersect the query, `build_segments` drops them, and
both rows plan as a single full-attn segment. First-step / full-sequence query
also passes when the causal/full pattern aligns, for example `[(1, 4), (9, 17)]`
versus `[(2, 4), (5, 13)]`.

Split the group only when segment structure differs (different segment count
or mode sequence), including some first-step layouts and partial prefix-hits
that leave a different remaining pattern. The first implementation slice does
not extend `build_paged_piecewise_plan`; grouping isolates the mismatch.

Overlapping spans such as `[(3, 6), (5, 13)]` with `prefix=5` are invalid
Hunyuan geometry and are not evidence that different reference offsets fail.

## Out of Scope

- Ring / AllGather-KV / Ulysses+Ring ([#9396](https://github.com/vllm-project/vllm-omni/pull/9396)).
- Default-paged flip and dense-path removal
  ([#9542](https://github.com/vllm-project/vllm-omni/pull/9542)). Completing
  slices 1 and 2 unblocks the "unset mode + `step_execution`" question; it
  does not flip the default.
- Combining AR KV import with local prefix hits. Grouping still keys on
  `ar_kv_reuse_len` so mixed AR-KV rows cannot share a group.
- KV-page offload.
- User-guide support claims. The user guide stays request-level until the
  corresponding runtime slice lands.

## Related Files

- Hunyuan step merge and grouping: [`pipeline_hunyuan_image3.py`](gh-file:vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py)
  (`_validate_step_request`, `_merge_step_model_inputs`, `_shift_full_attn_spans`,
  `_step_group_key`, `_ensure_grouped_attention_backend_supported`,
  `supports_request_batch`)
- Hunyuan row layout: [`request_layout.py`](gh-file:vllm_omni/diffusion/models/hunyuan_image3/request_layout.py)
- Paged planner: [`piecewise_attn.py`](gh-file:vllm_omni/diffusion/attention/backends/utils/piecewise_attn.py)
  (`build_paged_piecewise_plan`, `build_segments`, `_check_homogeneous`)
- Paged adapter: [`paged_attention_adapter.py`](gh-file:vllm_omni/diffusion/diffusion_kv/paged_attention_adapter.py)
- Baseline paged KV contract: [Scheduler-Managed Paged KV Cache](diffusion_paged_kv_cache.md)
- Engine mode names: [Diffusion Continuous Batching](diffusion_continuous_batching.md)
