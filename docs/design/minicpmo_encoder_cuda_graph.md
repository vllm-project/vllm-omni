# MiniCPM-o 4.5 input encoder CUDA graphs

The vision transformer and stateless audio encoder reuse vLLM's encoder graph
manager for exact input shapes. Each encoder retains at most four graphs by
default, and captures on the second call with the same shape and CUDA stream.
Startup profiling can consume slots. Graphs are never evicted: once the cache
is full, other shapes run eagerly for the lifetime of the model.

Use `--hf-overrides` to configure both encoders:

```json
{
  "encoder_cuda_graph": true,
  "encoder_cuda_graph_max_graphs": 4,
  "encoder_cuda_graph_min_capture_calls": 2,
  "encoder_cuda_graph_min_free_bytes": 1073741824,
  "encoder_cuda_graph_share_pools": true
}
```

`max_graphs` is a nonnegative integer per encoder; zero disables capture.
`min_capture_calls` is an integer of at least two. Increasing it reduces capture
of short-lived shapes. Admission history is bounded to four times `max_graphs`,
so an evicted history entry must accumulate its calls again. Increasing the
graph cap can improve coverage but retains more GPU memory. Existing graphs
are not replaced, avoiding repeated capture costs when traffic changes.

By default, graphs in the same encoder on the same device and replay stream
share a memory pool and capture stream. Different encoders and replay streams
remain isolated. Graph inputs and output buffers are allocated outside capture;
only transient capture allocations share the pool. Returned outputs remain
cloned. A late capture waits for earlier replay work and output clones before
reusing the pool. Set `encoder_cuda_graph_share_pools` to `false` for an A/B
comparison with a separate pool and capture stream per graph.

The graph count is **not a byte limit**. Sharing pools reuses intermediate
storage but does not eliminate per-shape static input and output buffers.
Before a new capture, the adapter checks device-free memory against
`min_free_bytes`, a nonnegative integer (default: 1 GiB; zero disables the check).
Below this floor the call runs eagerly, and a later call may try admission
again. Existing graphs can still replay. This check is conservative about
reusable allocator memory and does not predict capture allocations, reserve
memory, or guarantee that capture fits. Size the floor for the deployment;
disable encoder graphs when their additional memory cannot be accommodated.

Each adapter exposes `get_cumulative_stats()`: graph hits, total misses, hit
rate, graph count, and separate ineligible, capacity, warmup and memory misses.
These include calls rejected before the upstream manager, so a full cache's
lost coverage is visible. Vision uses `vpm._encoder_graph`; audio uses
`_audio_encoder_graph` on the model. The counters are per adapter, not global.

Capture failures propagate the original exception and make that adapter
unusable. Subsequent calls raise without retrying capture or running CUDA eager
work. Restart the worker, optionally with encoder graphs disabled. This is a
fatal contract, not automatic recovery: a failed CUDA capture can leave the
device context unusable. `capture_failures` records this terminal state.

`--enforce-eager` always disables these graphs. CPU, grad/autocast, stateful
streaming audio and the other model-specific eager fallbacks remain unchanged.
The adapter clones graph outputs through upstream postprocessing so retained
embeddings survive later replays.
