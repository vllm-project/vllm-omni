# Current Runtime Mapping

Use this reference while mapping an accepted edge contract onto the current
vLLM-Omni checkout. These symbols and constraints may change; re-resolve them
after every rebase.

## Topology And Processor Selection

- `StagePipelineConfig` defines structural stage metadata and processor hooks.
- `_select_processor_funcs()` chooses `sync_process_input_func` as the input
  processor when `async_chunk=false` and the field is configured.
- Current stage planning requires zero-based stage IDs matching their position
  in `stage_configs`.
- Ordinary orchestrator forwarding targets `src_stage_id + 1`; runner receive
  registration resolves the preceding stage. Treat skipped, branching, or
  merged edges as framework work until their routing is implemented and tested.
- Per-stage deploy fields control devices, replicas, and TP. DP and PP are
  pipeline-wide engine fields.

## Completed-Output Path

`StageEngineCoreClient.process_engine_inputs()` receives completed
`source_outputs` from the orchestrator and calls the selected input processor.
This path can satisfy an edge without a worker full-payload hook or KV transfer.

Representative anchor:

- `model_executor/stage_input_processors/moss_tts.py:talker2codec`

The processor must return the target stage's exact request type and cardinality.

## Full-Payload Path

The current producer lifecycle is:

```text
accumulate_full_payload_output()
  -> flush_full_payload_outputs() after request completion
  -> materialize and remove accumulator state
  -> custom_process_next_stage_input_func
  -> enqueue connector task
  -> _save_loop()
  -> _send_single_request()
  -> connector.put() with bounded retry
```

Current accumulation behavior:

- rank-2-or-higher tensors concatenate on dim 0 when trailing shapes match;
- keys declared by the processor module's `_FULL_PAYLOAD_REPLACE_KEYS` retain
  only the latest value;
- scalar, rank-1, and global values retain the latest emission.

Validate EOS and max-token terminal row counts. A flush materializes and removes
eligible entries, then attempts enqueue; it reports neither enqueue nor
transport success. The background thread owns transport and retry. Verify the
required edge's timeout, failure propagation, and cleanup because exhausted
retries may not by themselves terminate the consumer.

## Full-Payload Consumer Gate

`uses_full_payload_input_coordinator()` currently enables request parking for a
model/stage whitelist in `_FULL_PAYLOAD_INPUT_STAGES`. Adding a producer hook or
connector configuration does not automatically enable the consumer gate.

The current consumer lifecycle is:

```text
downstream scaffold
  -> WAITING_FOR_INPUT
  -> pending receive registration
  -> data-transfer rank connector.get()
  -> local TP fanout
  -> stage receive feedback
  -> coordinator restores request to WAITING
  -> runner injects payload into runtime state
```

An AR consumer becomes runnable and then follows its normal iterative schedule.
A generation consumer runs according to its one-shot execution contract.

## Connector Compatibility

- Startup compatibility checks compare connector names. Compare endpoint
  `extra` settings explicitly when both ends configure an edge.
- An intermediate payload stage needs a role-neutral connector instance that
  supports both receive and send operations.
- Backend retention and TTL determine whether payload publication can precede
  scaffold creation. Verify both arrival orders with the selected backend.

## KV And Companion Requests

`OmniKVTransferManager` owns KV extraction, publication, receive, and metadata.
For each KV edge, verify:

1. producer expansion and stable parent/companion IDs;
2. role order and cardinality;
3. per-role transfer criterion and cache extraction;
4. sender and receiver configuration compatibility;
5. metadata used to reconstruct positions and cache bindings;
6. collector ordering at the consumer;
7. timeout, abort propagation, and cleanup for every role.

Representative anchors include `prompt_expand_func`, `cfg_kv_collect_func`,
`omni_kv_config`, and `distributed/omni_connectors/kv_transfer_manager.py`.

## Diffusion Transition

The current diffusion orchestrator branch calls the target processor once. It
treats `None` and an empty list as terminal input errors, unwraps a one-element
list, and submits one prompt. `StagePool.submit_initial()` rejects a remaining
list prompt. Put request expansion and CFG companion collection in their
declared hooks, then return one assembled diffusion prompt.

## Additional Model Examples

| Model | Example boundary | Contract focus |
|---|---|---|
| HunyuanImage3 | Existing AR to DiT edge | `HUNYUAN_IMAGE3_PIPELINE` and `ar2diffusion` convert completed AR text and token-derived resolution, together with original multimodal input, into one diffusion request |
| MiniMax-H3 | Candidate DiT to video/audio VAE decode edge | Transfer terminal `video_latent`, `audio_latent`, `height`, and `width`; prove independent VAE construction and acceptable transfer cost before implementation |

## Verification Entry Points

Resolve these paths and test names from the current checkout. Use the
`vllm-omni-test` skill to select CI markers, hardware, and the final full-model
command.

| Contract area | Current entry point |
|---|---|
| Pipeline and processor selection | `tests/config/test_config_factory.py` |
| Qwen3-Omni scaffold and payload construction | `tests/model_executor/stage_input_processors/test_qwen3_omni_streaming_helpers.py` |
| Orchestrator stage transition | `tests/engine/test_orchestrator_stage_input_bridge.py` |
| Full-payload accumulation and runtime injection | `tests/worker/test_omni_gpu_model_runner.py` |
| Connector enqueue, retry, receive, and cleanup | `tests/worker/test_omni_connector_mixin.py` |
| Full-payload readiness and timeout | `tests/core/sched/test_omni_scheduling_coordinator.py`, `tests/core/sched/test_omni_scheduler_mixin_timeouts.py` |
| KV scheduler integration | `tests/core/sched/test_omni_ar_scheduler_kv_transfer.py` |
| Model correctness | Matching model-specific offline or online E2E with `async_chunk=false` |
| Performance | Matching benchmark config under `tests/dfx/perf/tests/` |

Useful focused probes for the Qwen3-Omni worked example:

```bash
pytest -q \
  tests/config/test_config_factory.py::TestSentinelDefaultPrecedence::test_async_chunk_dispatches_qwen3_omni_processors
pytest -q tests/model_executor/stage_input_processors/test_qwen3_omni_streaming_helpers.py \
  -k 'thinker2talker_full_payload_packs_complete_tensors or thinker2talker_token_only_preserves_voice_metadata'
pytest -q tests/worker/test_omni_gpu_model_runner.py \
  -k 'full_payload or sync_local_stage_payloads'
pytest -q tests/worker/test_omni_connector_mixin.py \
  -k 'FullPayloadSendWithCustomFunc or TestLocalPayloadCacheLifecycle or TestSendRetry'
pytest -q tests/core/sched/test_omni_scheduling_coordinator.py \
  tests/core/sched/test_omni_scheduler_mixin_timeouts.py
```

The current Qwen helper tests cover terminal-row packing and metadata, but do
not derive scaffold and payload from the same token IDs. Before closing a
Thinker-to-Talker change, add or locate a test that validates every required
payload key and asserts that the placeholder length agrees with the Talker
prefill contract for those IDs.

These probes establish component behavior. Close the task with a fresh
Qwen3-Omni full-model run that explicitly selects `async_chunk=false` and uses
the task's output comparator. Use
`tests/dfx/perf/tests/test_qwen3_omni_no_async_chunk.json` when performance is a
declared gate.

## Search Anchors

| Semantic requirement | Current anchors |
|---|---|
| Stage fields and processor selection | `StagePipelineConfig`, `_select_processor_funcs` |
| Stage planning and adjacency | `StageRuntime`, `_build_logical_stage_init_plans`, `Orchestrator._forward_to_next_stage` |
| Completed-output conversion | `StageEngineCoreClient.process_engine_inputs` |
| Full-payload accumulation and flush | `accumulate_full_payload_output`, `flush_full_payload_outputs` |
| Full-payload background send | `_pending_save_reqs`, `_save_loop`, `_send_single_request` |
| Cumulative replacement | `_FULL_PAYLOAD_REPLACE_KEYS` |
| Consumer readiness gate | `uses_full_payload_input_coordinator`, `_FULL_PAYLOAD_INPUT_STAGES` |
| Receive registration and feedback | `pending_input_registrations`, `stage_recv_req_ids` |
| Runner connector path | `OmniConnectorModelRunnerMixin` |
| KV lifecycle | `OmniKVTransferManager`, `omni_kv_config` |
| Companion expansion and collection | `prompt_expand_func`, `cfg_kv_collect_func` |
| Diffusion submission | `Orchestrator._forward_to_next_stage`, `StagePool.submit_initial` |

Resolve behavior from code and tests before relying on an anchor name.
