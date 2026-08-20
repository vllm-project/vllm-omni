# Qwen3-Omni Worked Example

This example fills the edge contract record for the Qwen3-Omni
Thinker-to-Talker boundary with `async_chunk=false`. It covers one edge so the
ownership and readiness decisions remain visible.

Current source anchors:

- topology: `model_executor/models/qwen3_omni/pipeline.py`;
- scaffold processor: `thinker2talker_token_only`;
- payload builder: `thinker2talker_full_payload`;
- consumer binding: `Qwen3OmniMoeForConditionalGeneration.talker_preprocess_prefill`;
- receive gate: `uses_full_payload_input_coordinator` and
  `OmniSchedulingCoordinator`.

## Boundary Record

| Boundary item | Decision and evidence |
|---|---|
| Producer and execution type | Stage 0 Thinker, `LLM_AR`; handoff follows Thinker completion |
| Consumer and execution type | Stage 1 Talker, `LLM_AR`; reconstructed input drives Talker prefill and subsequent decode |
| Identity and cardinality | Scheduler and local task state use the internal request ID; connector keys use the mapped external request ID; one Thinker request maps to one Talker request |
| Reconstruction claim | Talker input is reconstructed from an orchestrator scaffold, the terminal worker full payload, and Talker-local model state |
| Deployment objective | Independent Thinker/Talker placement and scaling; performance remains a measured gate |

## Dependency Ledger

| Dependency | Owner and stable handoff | Transfer plane | Consumer binding | Requirement and validation |
|---|---|---|---|---|
| Talker allocation length | Thinker completed token IDs | Completed output/control | Placeholder `prompt_token_ids` from `thinker2talker_token_only` | Required for Talker admission and KV allocation; validate computed prompt length |
| Thinker sequence embeddings | Thinker terminal accumulated output | Full payload | `embed.prefill` in `talker_preprocess_prefill` | Required; rows align with the reconstructed Thinker sequence |
| Thinker hidden states | Thinker terminal accumulated output | Full payload | `hidden_states.output` | Required; rows align with Talker conditioning |
| Thinker token identity | Thinker request at completion | Full payload | `ids.all` and `ids.prompt` | Required for ChatML and generated-token alignment |
| TTS special embeddings | Thinker terminal output | Full payload | `embed.tts_bos`, `embed.tts_eos`, `embed.tts_pad` | Required by Thinker-to-Talker prefill construction |
| Voice metadata | Original request | Full payload with matching scaffold fallback | `speaker` | Optional per-request voice override; missing or invalid values use the model default, and both representations must agree when present |

The payload always carries `meta.finished=True` and may carry `language` when
provided. Talker prefill reads neither field, and full-payload receive
completion is determined by the transfer mode rather than `meta.finished`.

## Selected Full-Payload Details

| Detail | Source-proven behavior |
|---|---|
| Accumulation | Qwen declares no replace keys, so rank-2-or-higher tensors with matching trailing dimensions use `CONCAT` on dimension 0; scalar and global values keep the latest value |
| Materialization | `flush_full_payload_outputs()` removes and materializes eligible accumulated entries after Thinker completion, then attempts enqueue; it reports neither enqueue nor transport success |
| Terminal transform | `thinker2talker_full_payload` removes the final stop-token row from Thinker embeddings and hidden states when more than one row exists; it withholds a payload that has token IDs but empty conditioning |
| Wire representation | Conditioning and TTS special tensors are detached to CPU; token IDs remain Python lists |
| Consumer conversion | Talker moves conditioning and TTS special tensors to its module device as `bfloat16`, and converts token ID lists to device tensors |
| Identity join | `_resolve_external_req_id()` selects the connector-key identity; scheduler state and pending task bookkeeping retain the internal request ID |
| Failure bounds | Background send retries at most `_MAX_SEND_RETRIES`; payload wait uses `VLLM_OMNI_INPUT_WAIT_TIMEOUT_S` only when its value is positive |

## Lifecycle

| Concern | Decision and evidence |
|---|---|
| Readiness and join | Talker scaffold exists and the matching full payload has completed receive, TP fanout, and scheduler feedback |
| Completion and output | The boundary is consumed when the payload is installed for Talker prefill; Talker then follows its own AR lifecycle. Thinker text and final Code2Wav audio retain their declared output ordering |
| Cleanup | Remove Thinker accumulation on flush, send state after background completion, receive cache after runtime injection, and all request state on finish, timeout, or abort |
| Scaffold failure | Processor exceptions propagate from the orchestrator. An empty result after terminal Thinker output produces a terminal empty output and request cleanup before a Talker request is created |
| Payload failure | After Talker creation, a rejected payload, connector failure, or ID mismatch leaves Talker waiting for input. Send retry is bounded; the scheduler marks the request `FINISHED_ERROR` on input timeout when `VLLM_OMNI_INPUT_WAIT_TIMEOUT_S > 0`. Required payload has no semantic fallback |

## Implementation Mapping

| Responsibility | Current mapping |
|---|---|
| Topology | Thinker stage 0 and Talker stage 1 in `QWEN3_OMNI_PIPELINE` |
| Control scaffold | Orchestrator calls `thinker2talker_token_only` for the completed Thinker output |
| Full-payload producer | Thinker runner accumulates output, then `thinker2talker_full_payload` builds the terminal payload |
| Transport | Runner mixin enqueues the payload; its background save loop performs connector `put` and retry |
| Receive gate | Talker scheduler parks the request in `WAITING_FOR_INPUT` and emits receive registration |
| Runtime injection | Data-transfer rank receives and fans out the payload; scheduler feedback restores the request; runner moves it into `model_intermediate_buffer` |
| Model consumer | `talker_preprocess_prefill` binds embeddings, hidden states, token IDs, special embeddings, and `speaker` |

```text
Thinker completion
  -> orchestrator builds Talker placeholder scaffold
  -> Talker request enters WAITING_FOR_INPUT

Thinker runner accumulation
  -> terminal materialization and payload hook
  -> connector task enqueue
  -> background put/retry
  -> Talker receive and TP fanout
  -> scheduler receive feedback

scaffold + payload
  -> coordinator restores Talker request
  -> runner installs model_intermediate_buffer
  -> Talker prefill
  -> Talker AR decode
```

Control and payload may arrive in either order. Connector retention, request
keys, cleanup, and the configured positive input-wait timeout bound the join.

## Verification

1. Confirm the pipeline selects `thinker2talker_token_only` and
   `thinker2talker_full_payload` when `async_chunk=false`.
2. Validate payload keys, terminal row alignment, and prompt-length agreement.
3. Validate `WAITING_FOR_INPUT`, receive registration, restore, timeout, and
   cleanup transitions.
4. Validate payload installation before Talker prefill and local TP fanout.
5. Run a fresh Qwen3-Omni full-model request with `async_chunk=false` and compare
   text/audio semantics against the declared baseline.

The Talker-to-Code2Wav edge uses the same review method with a different
payload: terminal codec rows are aligned to Talker output IDs, flattened for
Code2Wav, and consumed by an `LLM_GENERATION` stage. Analyze it as a separate
edge contract rather than extending this record.
