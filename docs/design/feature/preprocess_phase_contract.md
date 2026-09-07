# Runner-to-model prefill/decode phase contract

## Overview

Autoregressive models with a custom `preprocess` hook need to distinguish
prompt processing from decoding. A one-token span can be either a complete
one-token prompt, the last chunk of a longer prompt, or a decode step. Using
span length as the phase signal can replace a prompt embedding with a generated
audio embedding and advance teacher-forced text too early without an exception.

[Issue #4382](https://github.com/vllm-project/vllm-omni/issues/4382) requests a
stable contract for the metadata and routing guards introduced by
[PR #3662](https://github.com/vllm-project/vllm-omni/pull/3662).
The shared runner already implements those guards. This design documents the
existing interface and adds runner-level regression coverage rather than
introducing a second phase detector or changing scheduling policy.

## Scope and objectives

- Make the three existing phase keys safe to depend on in out-of-tree models.
- Preserve prompt embeddings and model state on a one-token prefill tail.
- Keep normal and batched decode routing consistent in mixed batches.
- Explain the correctness workaround required on older releases.

This contract does not enable chunked prefill for models whose other state
handling does not support it. It does not change the scheduler, introduce new
kwargs, or standardize other `_omni_*` fields. The implementation adds
documentation and regression coverage; no hot-path allocation is added.

## Design

### Ownership and data flow

```text
Scheduler / current input batch
  prompt_token_ids + num_computed_tokens_cpu (before this step)
                         |
            OmniGPUModelRunner._preprocess
                         |
            per-request phase metadata
                  /                 \
       prefill / other spans     single-token decode
          preprocess             preprocess or preprocess_decode_batch
              |                              |
       prompt embedding              runner-managed talker_mtp
                  \                 /
                    model forward
```

The producer is `OmniGPUModelRunner._preprocess` in
`vllm_omni/worker/gpu_model_runner.py`. `GPUARModelRunner` and
`GPUGenerationModelRunner` inherit it. The current `OmniNPUModelRunner` also
inherits this method; device-specific execution still needs its own hardware
validation. A plugin runner that overrides preprocessing is responsible for
preserving this interface.

### Stable model-facing fields

The following names, types, and meanings are a supported interface for in-tree
and out-of-tree model hooks despite the leading underscore. Models should treat
them as read-only runner-owned values. Changes to their meanings or removal
require an explicit compatibility/migration plan.

| Key | Python type | Meaning |
| --- | --- | --- |
| `_omni_prompt_len` | `int` | Length of the current request state's `prompt_token_ids`. This is the stage's prompt length, not the current chunk size or raw text length. |
| `_omni_num_computed_tokens` | `int` | The row's `input_batch.num_computed_tokens_cpu` value **before** processing the scheduled span. It includes already-computed/cached progress, not just tokens computed in the current call. |
| `_omni_is_prefill` | `bool` | Exactly `_omni_num_computed_tokens < _omni_prompt_len`. Equality and greater progress mean decode. |

The runner refreshes the fields after current request state has been gathered,
immediately before selecting the per-row hook. Existing values in intermediate
model state cannot override the current phase. Row indices follow the current
input batch (including any reordering), not request arrival order. A resumed
or streaming request is classified from its **current** prompt/progress values;
the phase is not latched for the lifetime of the request.

The normal hook receives these fields as keyword arguments:

```python
def preprocess(self, input_ids, input_embeds, **info):
    if info["_omni_is_prefill"]:
        offset = info["_omni_num_computed_tokens"]
        # Select the scheduled slice of the model's prompt embeddings.
        ...
    else:
        # Advance decode-only text/audio state.
        ...
```

For `preprocess_decode_batch(input_ids=..., req_infos=...)`, each `req_infos`
entry contains the same fields, aligned with its corresponding input row.
Every such row has `_omni_is_prefill == False`. The optional earlier
`preprocess_batch` hook is a separate whole-batch preparation hook; these freshly
computed per-row fields are not promised at that earlier boundary.

### Routing guarantee

With `has_preprocess=True`, the batched decode hook is eligible only when it is
callable, the runner has `talker_mtp`, the scheduled span is one token, and
`_omni_is_prefill` is false. Otherwise the normal `preprocess` hook runs.

Runner-managed MTP is likewise restricted to single-token decode rows. A prefill
row, including a one-token tail, is never added to that MTP batch: the runner
does not consume its `mtp_inputs`, generate codes for it, or overwrite its
preprocessed embedding with an MTP embedding. Decode rows in the same step
continue to receive their MTP outputs at their own token offsets. Models may
still implement their own model-specific work inside `preprocess`.

| Prompt length | Computed before step | Scheduled span | Phase | Runner MTP eligible |
| --- | --- | --- | --- | --- |
| 5 | 0 | 4 | Prefill | No |
| 5 | 4 | 1 | Prefill tail | No |
| 5 | 5 | 1 | Decode | Yes |
| 5 | 6 | 1 | Decode | Yes |
| 1 | 0 | 1 | Prefill | No |
| 5 | 5 | 2 | Decode | No (not a single-token span) |

### Compatibility

!!! warning "vLLM-Omni 0.20.x and earlier"
    AR talker models with custom preprocessing that infer phase from span length
    must use `enable_chunked_prefill: false` for correctness on these releases.
    They lack the phase metadata and the runner's prefill-tail MTP protection.
    A model-only heuristic cannot prevent the older runner from overwriting a
    one-token prompt embedding. Disabling chunking avoids split tails; it does
    not make span length a reliable phase detector for every possible prompt.

The mechanism landed in #3662 (v0.21.0rc2 / v0.22.0 and later); this document
formalizes that existing behavior. Prefer upgrading the matching vLLM and
vLLM-Omni release pair and consuming the explicit phase keys. Do not silently
fall back to `input_ids.numel() > 1` when the model requires chunking correctness.
For a custom or backported runtime, verify both metadata and MTP guards.

## Correctness and testing plan

Runner tests call the production `_preprocess` and `_talker_mtp_forward` methods
with CPU buffers and a deterministic MTP model. They inspect kwargs at the model
boundary, resulting embeddings, and per-request generated codes. Coverage must
include a multi-step prompt split with a one-token tail followed by decode,
mixed prefill/decode batches in different orders, normal/batched decode hooks,
adjacent decode requests in the same hook call with exact per-request codes,
single-token prompts, cached progress, and decode spans longer than one token.
No test should pass merely because MTP was disabled for every row.

Real-weight validation uses Qwen3-TTS CustomVoice through the full talker and
code2wav pipeline. Record the actual scheduled prompt progress, assert that at
least one one-token prefill tail was observed, check its hook metadata and MTP
exclusion, and require finite non-empty output audio. Compare to an unchunked
run with the same real input, greedy sampling, and `VLLM_BATCH_INVARIANT=1`.
Greedy sampling alone does not guarantee identical codes across different
prefill shapes: floating-point reduction differences can change an argmax and
diverge in subsequent autoregressive steps. Invariant kernels keep this exact
comparison focused on the phase contract. The two engines must be initialized
separately because their chunking settings and token budgets differ. GPU tests belong under
`tests/e2e/features/`; existing model tests remain useful adjacent regression
coverage. Functional audio checks do not establish throughput or all-model
semantic parity; this change makes no performance claim.

## Open questions and discussions

There are no new field names or execution modes to choose. Future runner
implementations, including Model Runner V2, should preserve these semantics or
provide an explicit migration path. Hardware validation beyond the tested
platform remains separate from the shared Python routing tests.

## References

- [Issue #4382](https://github.com/vllm-project/vllm-omni/issues/4382)
- [Original metadata and guards, #3662](https://github.com/vllm-project/vllm-omni/pull/3662)
- [Qwen2.5-Omni consumer fix, #5019](https://github.com/vllm-project/vllm-omni/pull/5019)
- [Adding an Omni model](../../contributing/model/adding_omni_model.md)
- [Test writing guide](../../contributing/ci/test_writing_guide.md)
