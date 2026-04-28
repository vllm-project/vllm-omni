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

## Stage config

Single stage with `worker_type: ar`, `engine_output_type: audio`,
`final_output: true`, `is_comprehension: true`, and `async_chunk: false` at
the top level. Omitting any of these causes silent misclassification in the
serving layer.

## Lint discipline

Only extract variables from `additional_information` that you actually
forward to the model call — unused extractions trip `ruff F841` in
pre-commit.

## Reference implementation

Look for any single-stage AR model under
`vllm_omni/model_executor/models/` — e.g. `moss_tts_nano/` when its
integration lands. If none is in tree yet, follow the skeleton above and
cross-check against the `is_comprehension: true` / `async_chunk: false`
dispatch in `vllm_omni/entrypoints/openai/serving_speech.py`.
