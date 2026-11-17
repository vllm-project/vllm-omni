# LFM3-Omni integration case

Use this case when a model has tightly coupled temporal and depth decoders or a
stateful streaming codec. It illustrates the generic contracts; its constants
are not defaults for other models.

Source: [LFM3-Omni integration PR #9827](https://github.com/vllm-project/vllm-omni/pull/9827),
implementation snapshot `12db351fbbb55083b991032351185a871ed0bc0c`, with checkpoint
`LiquidAI/LFM3-Omni-2B` at `21d75123f86c1f7ffb4bef1ea202710c4e7ebe67`.
This is a contribution snapshot, not a claim that it is merged or universally
validated. The pinned snapshot remains reproducible if the PR changes; it does
not automatically describe the eventual merged implementation. When #9827 merges,
recheck the token IDs, reveal schedule, chunking and graph controls below against
the merged code before updating this case's source pointer.

For the pinned snapshot, the depth-head graph switch is
`hf_overrides.lfm3_depth_graph` (set it to `false` for the control), consumed by
[`lfm3_omni/model.py`](https://github.com/vllm-project/vllm-omni/blob/12db351fbbb55083b991032351185a871ed0bc0c/vllm_omni/model_executor/models/lfm3_omni/model.py).
This is a snapshot-specific model override, not a generic in-tree stage field.

| Observed integration constraint | Decision and reusable check |
|---------------------------------|-----------------------------|
| The generated 12-codebook audio frame feeds the next temporal step | Keep temporal decoding and the MaskGIT depth head in one AR stage; put the codec downstream. Split by dependency, not module name. |
| Nonpersistent rotary frequencies differed when constructed directly on NPU | Preserve reference CPU construction followed by device transfer; compare buffers at the first divergent frame. Loading identical weights is insufficient. |
| Seeded MaskGIT uses a reveal schedule and ordered random operations | Preserve per-request generators and random draw order, including the `[4, 4, 4]` reveal schedule. Validate greedy and stochastic paths independently. |
| Text end can transition into the audio tail | Distinguish text-end `125014`, audio sentinel `2016`, and global text EOS `124900`; verify which markers are emitted, fed back, decoded, or terminate generation. |
| Transport can combine producer chunks | Restore four-frame codec calls for this reference streaming profile; flush only the final subchunk and support an empty final payload. Compare against HF streaming with the same schedule. |
| Depth-head NPU graphs can remain enabled with outer eager execution | Toggle the component graph explicitly for graph parity and performance controls. Capture only supported operations and keep the reference sampling semantics. |
| A metadata-only event reached the benchmark consumer | Read audio optionally while retaining completion handling; first-audio time requires nonempty samples. Rerun the affected comparison after repairing the consumer. |
| Stage workers needed more than the default shutdown grace | Validate configurable shutdown grace and wait for worker exits; request success alone did not prove a clean lifecycle. |

Finite token/code/waveform fixtures and repeated copies of one sentence are
useful diagnostics, but do not satisfy the accuracy-dataset and different-prompt
concurrency gates. Keep those evidence classes separate, and report untested
modalities, workloads, and backends explicitly. This case does not establish a
global performance optimum.
