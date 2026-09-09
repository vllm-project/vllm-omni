# Reusing Omni model contracts in Model Runner V2

MRV2 should reuse a model's existing Omni lifecycle rather than require a
second model implementation. `OmniModelState` is selected when the model
explicitly declares `has_preprocess`, `has_postprocess`, or
`have_multimodal_outputs`. Existing architecture dispatch remains as a legacy
fallback; an unlisted upstream model without these capabilities continues to
use the upstream factory.

| Concern | Shared runner responsibility | Model responsibility |
| --- | --- | --- |
| Preprocessing | Schedule prefill and decode; gather request state; write results back in batch order | Prompt/reference encoding and token embeddings |
| Decode batching | Reuse `preprocess_decode_batch(input_ids, req_infos)`; accept optional `input_embeds`; resolve signature once | Return IDs, embeddings, MTP hidden/control tensors, and one state update per request |
| MTP and sampling | Pack tensors, preserve request RNG state, dispatch declared graph support, preserve sampler settings | Codebook algorithm, local sampling, logits and EOS semantics |
| State | Own per-request intermediate buffers independently of execution slots | Declare GPU-resident state keys; maintain any algorithm-specific codec state |
| Output | Supply `(start, end)` hidden-row spans and state in current batch order; unwrap `OmniOutput` | Map model results to declared output keys |
| Termination | Call `on_requests_finished` on actual finish/abort, including requests whose chunk slot was released | Release request-owned streaming state |

An existing `preprocess_decode_batch_mrv2` takes precedence for compatibility
with optimized implementations. New integrations can use the original shared
hook. Hook exceptions propagate: the runner does not retry another signature
and risk sampling or updating state twice.

Chunk completion is not request completion. Generation slots may be recycled
after each chunk while codec state stays keyed by request ID. Preemption is
also not reported as a terminal callback.

These contracts do not automatically make every model graph-safe or eligible
for the native data plane. Models still have to declare payload ownership,
streaming delta/cumulative semantics, and graph constraints. Those properties
cannot be inferred from an audio output flag. Explicit per-request sampling
continues to use the existing safe fallback when a model cannot consume a
batch of independent generators.

MOSS Local v1.5 is the integration validation case. No changes to its model
implementation, sampling algorithm, deployment defaults, or native transport
capability declarations are required by this contract bridge. This establishes
execution compatibility, not a performance guarantee or full native-transport
support. Models using independent request generators still take the existing
per-row fallback unless they declare support for batched generators.
