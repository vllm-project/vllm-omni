# Pipeline Topology and Stage Hand-off

Source of truth: `StagePipelineConfig` and `PipelineConfig` in
`vllm_omni/config/stage_config.py`. Confirm field names there before use; this
file lists the ones multi-stage omni models normally set.

## Contents
- PipelineConfig fields
- StagePipelineConfig fields
- Processor hook contract
- Skeletons
- Registry entries

## PipelineConfig fields

| Field | Use |
|-------|-----|
| `model_type` | Key in `OMNI_PIPELINES`; matches the HF `model_type` |
| `model_arch` | Default arch for stages that do not set their own |
| `stages` | Tuple of `StagePipelineConfig` |
| `default_deploy_config_name` | File under `vllm_omni/deploy/` used when `--deploy-config` is absent |
| `default_session_mode` | `"turn"` or `"duplex"` |
| `duplex_plugin` | Dotted path to a duplex plugin class, when the model supports duplex |
| `endpoint_restrictions` | `EndpointRestriction(...)` entries for endpoints the model cannot serve |

## StagePipelineConfig fields

| Field | Typical omni value | Notes |
|-------|--------------------|-------|
| `stage_id` | 0, 1, 2 | Must match deploy YAML `stage_id` |
| `model_stage` | `"thinker"`, `"talker"`, `"code2wav"` (or model-specific) | Read by the unified class |
| `execution_type` | `LLM_AR` (thinker, talker), `LLM_GENERATION` (code2wav) | Selects worker / runner family |
| `input_sources` | `()`, `(0,)`, `(1,)` | Upstream stage ids |
| `final_output` / `final_output_type` | thinker `True`/`"text"`, code2wav `True`/`"audio"` | Stages the client sees |
| `owns_tokenizer` | Thinker `True` | One stage per pipeline |
| `requires_multimodal_data` | Thinker `True` | Thinker receives the raw multimodal prompt |
| `hf_config_name` | `"thinker_config"`, `"talker_config"` | Sub-config in the HF `config.json` |
| `engine_output_type` | `"latent"` for thinker/talker, `"audio"` for code2wav | What the engine emits downstream |
| `model_arch` | e.g. `"MiniCPMO45Code2Wav"` | Only when the stage uses a different class |
| `sampling_constraints` | `{"detokenize": False, "stop_token_ids": [...]}` | Enforced stage sampling keys |
| `custom_process_next_stage_input_func` | `f"{_PROC}.<a>2<b>_full_payload"` | Producer hook, `async_chunk` off |
| `async_chunk_process_next_stage_input_func` | `f"{_PROC}.<a>2<b>_async_chunk"` | Producer hook, `async_chunk` on |
| `sync_process_input_func` | `f"{_PROC}.<a>2<b>_token_only"` | Orchestrator placeholder, `async_chunk` off |
| `requires_full_payload_input` | `True` on downstream stages | Non-async path waits for the complete connector payload before scheduling |
| `async_chunk_prewarm_payload_func` | Optional | Lets a receiving stage warm up before chunk 0 (MiniCPM-o code2wav) |
| `retains_state_across_chunks` | `True` when the model keeps per-request state between chunks | Parked request still consumes capacity |
| `supports_native_mrv2_data_plane` | `True` only after MRv2 validation | Takes effect only when deploy sets `model_runner: v2` |

`_select_processor_funcs` (same file) resolves the active pair: with
`async_chunk` on, `async_chunk_process_next_stage_input_func` replaces
`custom_process_next_stage_input_func`; with it off, `sync_process_input_func`
replaces `custom_process_input_func`.

## Processor hook contract

The producer hooks run on the producing stage (scheduler / connector
runtime), not in the orchestrator. Callers:
`distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py`,
`distributed/omni_connectors/model_runner/omni_connector_runtime.py`,
`core/sched/omni_ar_scheduler.py`.

| Hook | Signature (Qwen3-Omni) | Returns |
|------|------------------------|---------|
| full payload | `(transfer_manager, pooling_output, request)` | dict / `OmniPayload`, or `None` |
| async chunk | `(transfer_manager, multimodal_output, request, is_finished=False)` | `OmniPayloadStruct`, or `None` to send nothing this step |
| token only | `(source_outputs, prompt=None, requires_multimodal_data=False, streaming_context=None)` | `list[OmniTokensPrompt]` placeholders |

Rules:

- Read only the `multimodal_outputs` keys your `forward()` writes; keep those
  key names in one place shared by the model and the processor.
- Validate input type first and log the request id on failure; returning
  `None` from a full-payload hook can leave the consumer waiting.
- Key streaming state by `request.external_req_id` and drop it when
  `is_finished` is true.
- Send codec deltas, not cumulative codes, unless the decoder requires the
  window (left context is configured in the connector, not hard-coded).
- Move tensors to CPU or the connector's expected device before returning.
- The token-only placeholder must have the exact length the downstream stage
  will prefill; a wrong length corrupts KV reservation silently.

## Skeletons

```python
# vllm_omni/model_executor/models/<name>/pipeline.py
from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig

_PROC = "vllm_omni.model_executor.stage_input_processors.<name>"

NAME_PIPELINE = PipelineConfig(
    model_type="<hf_model_type>",
    model_arch="<Name>ForConditionalGeneration",
    default_deploy_config_name="<name>.yaml",
    default_session_mode="turn",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="thinker",
            execution_type=StageExecutionType.LLM_AR,
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            hf_config_name="thinker_config",
            engine_output_type="latent",
            custom_process_next_stage_input_func=f"{_PROC}.thinker2talker_full_payload",
            async_chunk_process_next_stage_input_func=f"{_PROC}.thinker2talker_async_chunk",
            sampling_constraints={"detokenize": True},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="talker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(0,),
            hf_config_name="talker_config",
            engine_output_type="latent",
            sync_process_input_func=f"{_PROC}.thinker2talker_token_only",
            custom_process_next_stage_input_func=f"{_PROC}.talker2code2wav_full_payload",
            async_chunk_process_next_stage_input_func=f"{_PROC}.talker2code2wav_async_chunk",
            sampling_constraints={"detokenize": False, "stop_token_ids": [<codec_eos>]},
            requires_full_payload_input=True,
        ),
        StagePipelineConfig(
            stage_id=2,
            model_stage="code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(1,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            requires_full_payload_input=True,
        ),
    ),
)
```

```python
# vllm_omni/model_executor/models/<name>/<name>.py (unified class, sketch)
class NameForConditionalGeneration(nn.Module, SupportsMultiModal):
    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__()
        self.model_stage = getattr(vllm_config.model_config, "model_stage", None) or "thinker"
        if self.model_stage == "thinker":
            self.thinker = NameThinker(vllm_config=vllm_config, prefix=f"{prefix}thinker.")
        elif self.model_stage == "talker":
            self.talker = NameTalker(vllm_config=vllm_config, prefix=f"{prefix}talker.")
        elif self.model_stage == "code2wav":
            self.code2wav = NameCode2Wav(vllm_config=vllm_config, prefix=f"{prefix}code2wav.")
        else:
            raise ValueError(f"Invalid model_stage: {self.model_stage}")

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        ...
        return OmniOutput(
            text_hidden_states=hidden_states,
            multimodal_outputs={"<payload_key>": tensor},
        )
```

For the real control flow (staged vs non-staged runs, MTP, PP, intermediate
tensors) copy from `qwen3_omni/qwen3_omni.py` rather than this sketch.

## Registry entries

```python
# vllm_omni/model_executor/models/registry.py, _OMNI_MODELS
"NameForConditionalGeneration": ("<name>", "<name>", "NameForConditionalGeneration"),
"NameThinkerForConditionalGeneration": ("<name>", "<name>_thinker", "NameThinkerForConditionalGeneration"),
"NameTalkerForConditionalGeneration": ("<name>", "<name>_talker", "NameTalkerForConditionalGeneration"),
"NameCode2Wav": ("<name>", "<name>_code2wav", "NameCode2Wav"),
```

```python
# vllm_omni/config/pipeline_registry.py, OMNI_PIPELINES (alphabetical)
"<hf_model_type>": NAME_PIPELINE,  # or the resolver function
```

Out-of-tree pipelines can use `register_pipeline(...)` from the same module
instead of editing `OMNI_PIPELINES`.
