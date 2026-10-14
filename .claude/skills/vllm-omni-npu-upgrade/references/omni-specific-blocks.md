# Omni-Specific Code Blocks Reference

This document catalogs omni-specific code blocks in the NPU model runners, making it easier to identify what needs to be preserved during upgrades.

> **IMPORTANT**: This document may not be complete or up-to-date!
>
> - Always grep for `Omni-new` in the GPU implementations (`vllm_omni/worker/`) to find the authoritative list
> - New omni features may be added that are not yet documented here
> - When you discover new omni-specific blocks during an upgrade, please update this document
> - Last verified: Check git history for this file

## OmniNPUModelRunner (npu_model_runner.py)

### load_model - Talker MTP Initialization

```python
def load_model(self, *args, **kwargs) -> None:
    NPUModelRunner.load_model(self, *args, **kwargs)
    # Initialize enable_sp cache to avoid get_current_vllm_config() error
    # in _pad_for_sequence_parallelism during execute_model.
    # This is a workaround for vllm-ascend not passing vllm_config to enable_sp().
    enable_sp(self.vllm_config)
    # TODO move this model specific logic to a separate class
    # TTS model IS the talker (no .talker sub-attr); use getattr to support both Omni and TTS.
    talker_mtp = getattr(self.model, "talker_mtp", None)
    if talker_mtp is not None:
        self.talker_mtp = talker_mtp  # type: ignore[assignment]
        cudagraph_mode = self.compilation_config.cudagraph_mode
        assert cudagraph_mode is not None
        # Only wrap talker_mtp in CUDAGraphWrapper for Omni models that
        # have a separate .talker sub-module.  TTS models' code predictor
        # has internal AR loops / torch.multinomial — not graph-safe.
        has_separate_talker = getattr(self.model, "talker", None) is not None
        if cudagraph_mode.has_full_cudagraphs() and has_separate_talker:
            # NOTE: Use ACLGraphWrapper on NPU, not CUDAGraphWrapper
            self.talker_mtp = ACLGraphWrapper(talker_mtp, self.vllm_config, runtime_mode=CUDAGraphMode.FULL)
        # TTS exposes mtp_hidden_size; Omni uses hf_text_config.hidden_size.
        hidden_size = int(
            getattr(self.model, "mtp_hidden_size", 0) or getattr(self.model_config.hf_text_config, "hidden_size")
        )
        max_batch_size = max(self.max_num_reqs, self.compilation_config.max_cudagraph_capture_size)
        self.talker_mtp_input_ids = self._make_buffer(max_batch_size, dtype=torch.int32)
        self.talker_mtp_inputs_embeds = self._make_buffer(
            max_batch_size, hidden_size, dtype=self.dtype, numpy=False
        )
        self.last_talker_hidden = self._make_buffer(max_batch_size, hidden_size, dtype=self.dtype, numpy=False)
        self.text_step = self._make_buffer(max_batch_size, hidden_size, dtype=self.dtype, numpy=False)
```

### _dummy_run - Talker MTP Dummy Forward

Location: Inside `set_ascend_forward_context` block, before main model forward

```python
# ---------------------------------------Omni-new----------------------------------------------
if getattr(self.model, "talker", None) is not None and hasattr(self.model, "talker_mtp"):
    num_tokens_padded_talker_mtp = num_tokens_padded
    if num_tokens_padded_talker_mtp == self.max_num_tokens:
        num_tokens_padded_talker_mtp = self.talker_mtp_input_ids.gpu.shape[0]
    outputs = self.talker_mtp(
        self.talker_mtp_input_ids.gpu[:num_tokens_padded_talker_mtp],
        self.talker_mtp_inputs_embeds.gpu[:num_tokens_padded_talker_mtp],
        self.last_talker_hidden.gpu[:num_tokens_padded_talker_mtp],
        self.text_step.gpu[:num_tokens_padded_talker_mtp],
    )
    self.compilation_config.cache_dir = None
# ---------------------------------------Omni-new----------------------------------------------
```

### _dummy_run - Extract Multimodal Outputs

Location: After model forward, before dummy_compute_logits

```python
# ---------------------------------------Omni-new----------------------------------------------
hidden_states, multimodal_outputs = self.extract_multimodal_outputs(hidden_states)
# ---------------------------------------Omni-new----------------------------------------------
```

### _model_forward - Omni Output Wrapping

```python
def _model_forward(
    self,
    num_tokens_padded: int,
    input_ids: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    **model_kwargs: dict[str, Any],
):
    """Override to combine NPUModelRunner's signature with OmniGPUModelRunner's logic."""
    # Omni-specific: build and inject extra model kwargs
    model_kwargs_extra = self._build_model_kwargs_extra()

    # Call the model forward (same as NPUModelRunner)
    assert self.model is not None
    model_output = self.model(
        input_ids=input_ids,
        positions=positions,
        intermediate_tensors=intermediate_tensors,
        inputs_embeds=inputs_embeds,
        **model_kwargs,
        **model_kwargs_extra,
    )

    # Omni-specific: wrap output if needed
    if not isinstance(model_output, OmniOutput) and hasattr(self.model, "make_omni_output"):
        model_output = self.model.make_omni_output(model_output, **model_kwargs_extra)

    # Omni-specific: cache model output for later sample_tokens
    self._omni_last_model_output = model_output

    # NPU-specific: update full graph params (keep from vllm-ascend)
    forward_context = get_forward_context()
    # ... NPU graph update logic ...

    # NPU-specific: all-gather for sequence parallelism (keep from vllm-ascend)
    if get_forward_context().sp_enabled and not isinstance(model_output, IntermediateTensors):
        model_output = self._all_gather_hidden_states_and_aux(model_output)

    return model_output
```

---

## NPUARModelRunner (npu_ar_model_runner.py)

### __init__ - KV Transfer Manager

```python
def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.input_ids = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
    # each model stage has their own hidden size
    self.hidden_size = self.model_config.hf_text_config.hidden_size
    self.inputs_embeds = self._make_buffer(self.max_num_tokens, self.hidden_size, dtype=self.dtype, numpy=False)
    # Initialize KV cache manager (preserve vllm_config fallback behavior)
    self.kv_transfer_manager = OmniKVTransferManager.from_vllm_config(self.vllm_config, self.model_config)
```

### execute_model - KV Transfer Before Update States

Location: At the very beginning of execute_model

```python
#  -------------------------------------- Omni-new -------------------------------------------------
# [Omni] Handle KV transfer BEFORE updating states (which removes finished requests)
self.kv_extracted_req_ids = self.kv_transfer_manager.handle_finished_requests_kv_transfer(
    finished_reqs=getattr(scheduler_output, "finished_requests_needing_kv_transfer", {}),
    kv_caches=self.kv_caches,
    block_size=self.cache_config.block_size,
    cache_dtype=str(self.cache_config.cache_dtype),
    request_id_resolver=self._resolve_global_request_id,
)
#  -------------------------------------- Omni-new -------------------------------------------------
```

### execute_model - Custom _update_states Call

Location: Inside synchronize_input_prep context

```python
#  -------------------------------------- Omni-new -------------------------------------------------
self._update_states(scheduler_output)
#  ------------------------------------------------------------------------------------------------
```

### execute_model - Extract Multimodal Outputs

Location: In post process section, after hidden_states assignment

```python
#  -------------------------------------- Omni-new -------------------------------------------------
hidden_states, multimodal_outputs = self.extract_multimodal_outputs(hidden_states)

if multimodal_outputs is not None:
    keys_or_type = (
        list(multimodal_outputs.keys())
        if isinstance(multimodal_outputs, dict)
        else type(multimodal_outputs)
    )
    logger.debug(f"[AR] execute_model: multimodal_outputs keys = {keys_or_type}")
else:
    logger.debug("[AR] execute_model: multimodal_outputs is None")
#  -------------------------------------- Omni-new -------------------------------------------------
```

### execute_model - Compute Logits with sampling_metadata

Location: In both broadcast_pp_output True and False branches

```python
#  -------------------------------------- Omni-new -------------------------------------------------
# Try with sampling_metadata first; fall back to without for models that don't support it
try:
    logits = self.model.compute_logits(
        sample_hidden_states, sampling_metadata=self.input_batch.sampling_metadata
    )
except TypeError:
    logits = self.model.compute_logits(sample_hidden_states)
#  -------------------------------------- Omni-new -------------------------------------------------
```

### sample_tokens - KV Extracted Req IDs

Location: At the beginning of sample_tokens

```python
#  -------------------------------------- Omni-new -------------------------------------------------
kv_extracted_req_ids = getattr(self, "kv_extracted_req_ids", None)
self.kv_extracted_req_ids = None
#  -------------------------------------- Omni-new -------------------------------------------------
```

### sample_tokens - Process Additional Information and Build Output

Location: After bookkeeping sync, replacing the original output construction

```python
#  -------------------------------------- Omni-new -------------------------------------------------
hidden_states_cpu = hidden_states.detach().to("cpu").contiguous()
num_scheduled_tokens_np = getattr(self, "_omni_num_scheduled_tokens_np", None)
if num_scheduled_tokens_np is None:
    req_ids = self.input_batch.req_ids
    num_scheduled_tokens_np = np.array(
        [scheduler_output.num_scheduled_tokens[rid] for rid in req_ids],
        dtype=np.int32,
    )

self._process_additional_information_updates(
    hidden_states, multimodal_outputs, num_scheduled_tokens_np, scheduler_output
)

pooler_output: list[dict[str, object]] = []
for rid in req_ids_output_copy:
    idx = req_id_to_index_output_copy[rid]
    start = int(self.query_start_loc.cpu[idx])
    sched = int(num_scheduled_tokens_np[idx])
    end = start + sched
    hidden_slice = hidden_states_cpu[start:end]
    payload: dict[str, object] = {"hidden": hidden_slice}
    if isinstance(multimodal_outputs, dict) and multimodal_outputs:
        # ... multimodal output slicing logic ...
    pooler_output.append(payload)

model_runner_output = OmniModelRunnerOutput(
    req_ids=req_ids_output_copy,
    req_id_to_index=req_id_to_index_output_copy,
    sampled_token_ids=valid_sampled_token_ids,
    logprobs=logprobs_lists,
    prompt_logprobs_dict=prompt_logprobs_dict,
    pooler_output=(pooler_output if self.vllm_config.model_config.engine_output_type != "text" else None),
    kv_connector_output=kv_connector_output,
)
model_runner_output.kv_extracted_req_ids = kv_extracted_req_ids
#  -------------------------------------- Omni-new -------------------------------------------------
```

---

## NPUGenerationModelRunner (npu_generation_model_runner.py)

### execute_model - Async Chunk Update

Location: Inside prepare input section, before synchronize_input_prep

```python
#  -------------------------------------- Omni-new -------------------------------------------------
if self.model_config.async_chunk and num_scheduled_tokens:
    self._update_request_states(scheduler_output)
#  -------------------------------------- Omni-new -------------------------------------------------
```

### execute_model - Seq Token Counts

Location: After _preprocess call

```python
# [Omni] Pass token counts per request for code2wav output slicing
model_kwargs["seq_token_counts"] = tokens
```

### execute_model - Run Generation Model

Location: Inside forward context

```python
#  -------------------------------------- Omni-new -------------------------------------------------
outputs = self._run_generation_model(
    num_tokens_padded=num_tokens_padded,
    input_ids=input_ids,
    positions=positions,
    intermediate_tensors=intermediate_tensors,
    inputs_embeds=inputs_embeds,
    model_kwargs=model_kwargs,
    logits_indices=logits_indices,
)
_, multimodal_outputs = self.extract_multimodal_outputs(outputs)
# -------------------------------------- Omni-new -------------------------------------------------
```

### sample_tokens - Multimodal Output Processing

The entire sample_tokens method body is omni-specific for generation models:

```python
#  -------------------------------------- Omni-new -------------------------------------------------
pooler_output: list[object] = []
if isinstance(multimodal_outputs, torch.Tensor):
    # ... tensor handling ...
elif isinstance(multimodal_outputs, list):
    # ... list handling ...
elif isinstance(multimodal_outputs, dict):
    # ... dict handling per request ...
else:
    raise RuntimeError("Unsupported diffusion output type")
# [Omni] Copy req_id mappings to avoid async scheduling mutation.
req_ids_output_copy = self.input_batch.req_ids.copy()
req_id_to_index_output_copy = self.input_batch.req_id_to_index.copy()
output = OmniModelRunnerOutput(
    req_ids=req_ids_output_copy,
    req_id_to_index=req_id_to_index_output_copy,
    sampled_token_ids=[],
    logprobs=None,
    prompt_logprobs_dict={},
    pooler_output=pooler_output,
    kv_connector_output=kv_connector_output,
    num_nans_in_logits={},
    ec_connector_output=ec_connector_output if self.supports_mm_inputs else None,
)
#  -------------------------------------- Omni-new -------------------------------------------------
```

### _dummy_run - Model Kwargs Init and Multimodal Extract

Location: Before model forward and after

```python
model_kwargs = self._init_model_kwargs()  # Before forward

# ... forward ...

# -------------------------------------- Omni-new -------------------------------------------------
hidden_states, _ = self.extract_multimodal_outputs(hidden_states)
# -------------------------------------------------------------------------------------------------
```

---

## ExecuteModelState Extension

The `ExecuteModelState` NamedTuple is extended for omni:

```python
class ExecuteModelState(NamedTuple):
    """Ephemeral cached state transferred between execute_model() and
    sample_tokens(), after execute_model() returns None."""

    scheduler_output: SchedulerOutput
    logits: torch.Tensor
    spec_decode_metadata: SpecDecodeMetadata | None
    spec_decode_common_attn_metadata: AscendCommonAttentionMetadata | None
    hidden_states: torch.Tensor
    sample_hidden_states: torch.Tensor
    aux_hidden_states: list[torch.Tensor] | None
    attn_metadata: PerLayerAttnMetadata
    positions: torch.Tensor
    ec_connector_output: ECConnectorOutput | None
    cudagraph_stats: CUDAGraphStat | None
    multimodal_outputs: Any  # <-- Omni extension
```

This extended state must be imported from `npu_ar_model_runner` in `npu_generation_model_runner`.
