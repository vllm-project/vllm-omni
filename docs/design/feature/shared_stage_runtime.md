# Shared stage runtime capabilities

These interfaces let model adapters use the same stage scheduler, connector,
and output lifetime machinery. They do not by themselves enable a model's
MRv2 implementation or establish a serving performance improvement.

## Runner selection and sessions

`model_runner: v1` or `v2` can be set on an individual deploy stage. An omitted
stage value inherits the deployment value. An explicit platform-wide V1
fallback also clears stage V2 overrides; unsupported native worker platforms
fail at configuration time.

A downstream native MRv2 receiver supports turn requests. Non-turn deployment
modes and resumable input requests are rejected before stage submission. V1
continues to provide the streaming-session path. Selecting MRv2 does not make
turn and duplex performance interchangeable.

## Model contracts

| Declaration or hook | Contract |
| --- | --- |
| `logits_vocab_size` | Positive output-head vocabulary size; adopted before sampler initialization and used to sanitize stop IDs. |
| `mrv2_custom_sampler(sampler)` | Uses the upstream model-state custom-sampler return contract to replace or wrap sampling. A sampler may declare `omni_static_staged_writes` only if no per-step staged state changes without new requests. |
| `mrv2_decode_preprocess_is_identity` | Decode needs only the token embedding and has no per-request preprocessing updates. Prefill and eager MTP keep their existing paths. |
| `make_omni_output_mrv2` | Builds an `OmniOutput` using the current `input_batch`, `req_states`, and gathered intermediate buffers. |
| `supports_mrv2_full_graph_aux_outputs` | Tuple-returning models may opt into FULL graphs only with a nonempty, stable tensor pytree whose leaves have the token axis first. Capture validates structure and shape. Real-token slicing remains separate from graph padding. |
| `publishes_sampled_embeddings` | Publishes `embed.sampled` for each kept single-token sample in the sampling step. Unfinished prefill rows and speculative steps publish no sampled embedding. Payloads have per-step ownership and join the existing output-copy completion event. |
| `requires_request_ids` | Generation stages receive actual request IDs so model-owned state can follow request lifetimes. |

The native data plane preserves scalar metadata as the latest value, exposes
the last output token to stage processors, and sends a terminal marker even
when a processor has no remaining chunk at EOS. Sender-only stages do not
wait for nonexistent connector input. Request initialization accepts nested
model intermediate buffers through the existing storage rules.

## Transport and scheduling

SHM wakeups are advisory and enabled by default. Each receiver owns a UUID FIFO
under a user/parent-process/stage prefix; a sender broadcasts nonblocking hints
to that stage's receivers. Closing or restarting a receiver cannot unlink a
peer's FIFO. Shutdown rechecks descriptor ownership before draining. The
receiver retains timed polling, including for unsupported namespaces and
process layouts that do not share the launching parent.

`VLLM_OMNI_SHM_WAKEUP=0` disables hints. Sender and receiver worker threads use
separate work events. `VLLM_OMNI_CONNECTOR_RECV_POLL_MS` controls the receive
fallback interval. Broadcast discovery costs scale with the number of matching
FIFO paths. Hard process termination may leave stale FIFO files until external
cleanup; senders ignore them. Multi-replica serving performance needs separate
measurement after changing this transport.

`VLLM_OMNI_CODEC_FIRST_CHUNK_EXPRESS=1` opts native generation stages into
first-chunk-only scheduling steps. Consecutive express steps are disallowed.
`VLLM_OMNI_CODEC_FIRST_CHUNK_EXPRESS_SLACK_S` optionally requires estimated
playback credit before delaying a ready continuation. Credit is measured from
emitted mono sample vectors and server time; it is not client playback telemetry.
Unknown or multichannel layouts receive no credit. This policy is off by default
and requires workload-specific latency, throughput, and underrun validation.

The V1 generation runner batches pinned D2H copies behind one host wait per
step, retaining blocking copies where pinned CUDA copies are unavailable.
Stage wrapper diagnostics identify child hooks that the top-level model does
not expose.

## Validation and PR boundaries

CPU tests cover config inheritance, session rejection, FIFO replica/restart and
shutdown lifetimes, scheduling fairness, terminal delivery, and output contracts.
Small CUDA tests cover pinned copies and auxiliary tensor graph replay. Existing
Buildkite CPU sweeps and the `tests/worker_v2/` CUDA sweep collect these tests.
The chat benchmark also records WAV chunk continuity using the same underrun
calculation as the speech benchmark.

First-audio direct delivery, eager MTP, and managed MPS are supplied by the
upstream implementation. Model kernels, model adapters, deployment profiles,
and end-to-end quality/performance results belong in the separate model PRs.
This shared change makes no Qwen3-Omni or MiniCPM-o speedup claim. JIT launch,
buffer-view experiments, and diagnostic monkey patches are not included.
