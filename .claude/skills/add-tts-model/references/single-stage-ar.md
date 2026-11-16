# Single-Stage AR Pattern

When the upstream model cannot be cleanly split into an AR stage and a separate
decoder (e.g. MOSS-TTS-Nano, or any model that bundles AR + codec via an
`inference_stream()` generator), run the whole pipeline inside a single AR
worker that yields audio chunks per request.

This is distinct from VoxCPM2's pattern, which also runs in a single stage but
uses vLLM's native PagedAttention on the base language model with diffusion /
VAE side-computation outside vLLM — see
`plan/voxcpm2_native_ar_design.md` for that variant.

## Implementation

1. **Single model file** — load both AR LM and codec inside
   `modeling_<model>.py`.
2. **Load weights in `load_weights()`**, not `__init__()` — vLLM initializes
   distributed state before any CUDA allocations.
3. **Stream via a per-request generator** stored in `self._stream_gens`:

```python
class YourModelForCausalLM(nn.Module):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        self._lm = None                   # populated in load_weights()
        self._stream_gens: dict = {}      # request_key → generator

    def load_weights(self, weights):
        # Load self._lm here, after vLLM distributed init
        ...

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        runtime_additional_information: list[dict] | None = None,  # one dict per request
        **kwargs,
    ) -> OmniOutput:
        infos = runtime_additional_information or [{}]
        # Skip dummy/profiling calls
        if not runtime_additional_information or all(i.get("_is_dummy") for i in infos):
            self._ar_emit_stop_token = True
            return OmniOutput(...)  # return empty outputs

        outputs, last_flags = [], []
        for info in infos:
            request_key = str(info.get("_omni_req_id", "0"))  # per-request ID from vLLM
            if request_key not in self._stream_gens:
                self._stream_gens[request_key] = self._create_stream_gen(info)
            try:
                chunk, is_last = next(self._stream_gens[request_key])
            except StopIteration:
                chunk, is_last = torch.zeros(0), True
            if is_last:
                del self._stream_gens[request_key]
            outputs.append(chunk)
            last_flags.append(is_last)

        self._ar_emit_stop_token = all(last_flags)
        return OmniOutput(multimodal_outputs={"model_outputs": outputs, ...})

    def _create_stream_gen(self, info: dict):
        """Yield (waveform_tensor, is_last) tuples from inference_stream()."""
        for event in self._lm.inference_stream(...):
            if event["type"] == "audio":
                yield event["waveform"], False
            elif event["type"] == "result":
                # Fallback: some models emit a single "result" event instead of
                # incremental "audio" events — handle both paths
                yield event.get("waveform", torch.zeros(0)), True
                return
        yield torch.zeros(0), True

    def compute_logits(self, hidden_states, sampling_metadata):
        # Emit EOS only after the last chunk so the AR scheduler ends the request
        ...
```

## Key points

- `runtime_additional_information` is the correct parameter name (not
  `**kwargs`) — it carries one dict per request in the batch.
- The request ID is `info.get("_omni_req_id")` — set by vLLM, not by user code.
- Handle both `"audio"` (incremental) and `"result"` (final combined) event
  types from upstream models.

## Pipeline and deploy config

Define the fixed single-stage topology in `pipeline.py`:

```python
from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

YOUR_TTS_PIPELINE = PipelineConfig(
    model_type="your_model_name",
    default_deploy_config_name="your_model_name.yaml",
    model_arch="YourModelForCausalLM",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="your_model_name",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="audio",
            final_output=True,
            final_output_type="audio",
        ),
    ),
)
```

Register it in `vllm_omni/config/pipeline_registry.py`. Keep placement and
runtime sizing in `vllm_omni/deploy/your_model_name.yaml`:

```yaml
async_chunk: false

stages:
  - stage_id: 0
    devices: "0"
    max_num_seqs: 4
    gpu_memory_utilization: 0.8
```

Do not move `execution_type`, output ownership, tokenizer ownership, or model
architecture into the deploy YAML; those are pipeline topology.

## Lint discipline

Only extract variables from `additional_information` that you actually
forward to the model call — unused extractions trip `ruff F841` in
pre-commit.

## Reference implementation

Use `vllm_omni/model_executor/models/moss_tts_nano/pipeline.py` and
`vllm_omni/deploy/moss_tts_nano.yaml` as the in-tree reference.
