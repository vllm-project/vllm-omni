# Dynin-Omni Online Serving Example

## Installation

Please refer to [README.md](../../../README.md).

## Launch the Server

First, find the `transformers_modules` path:

```bash
python - <<'PY'
from transformers.utils.hub import HF_MODULES_CACHE
print(HF_MODULES_CACHE)
PY
```

Then export it for both `PYTHONPATH` and `HF_MODULES_CACHE`:

```bash
export PYTHONPATH=<transformers_modules_path>:$PYTHONPATH
export HF_MODULES_CACHE=<transformers_modules_path>
```

Run from repository root:

```bash
vllm-omni serve snu-aidas/Dynin-Omni \
  --omni \
  --port 8000 \
  --stage-configs-path "$(pwd)/vllm_omni/model_executor/stage_configs/dynin_omni.yaml"
```

If `vllm-omni` is not in PATH, run:

```bash
PYTHONPATH="$(pwd)" python -m vllm_omni.entrypoints.cli.main serve snu-aidas/Dynin-Omni \
  --omni \
  --port 8000 \
  --stage-configs-path "$(pwd)/vllm_omni/model_executor/stage_configs/dynin_omni.yaml"
```

Wait until the server logs show both `All stages initialized successfully` and
`Application startup complete.` before sending requests.

## Send Requests via Python Client

Move to the example directory:

```bash
cd examples/online_serving/dynin_omni
```

### Text -> Image

```bash
python openai_chat_completion_client_for_multimodal_generation.py \
  --query-type t2i \
  --prompt "A realistic indoor living room with natural daylight."
```

### Image -> Image

```bash
python openai_chat_completion_client_for_multimodal_generation.py \
  --query-type i2i \
  --image-path ../../offline_inference/dynin_omni/data/image/sofa_under_water.jpg \
  --prompt "Transform this surreal underwater setting into a realistic indoor living room while preserving the sofa layout."
```

### Text -> Speech

```bash
python openai_chat_completion_client_for_multimodal_generation.py \
  --query-type t2s \
  --prompt "Hello. This is Dynin-omni."
```

## CLI Arguments

- `--query-type` (`t2i|t2s|i2i`)
- `--model` (default: `snu-aidas/Dynin-Omni`)
- `--host` / `--port` (OpenAI-compatible vLLM endpoint)
- `--prompt` (custom text)
- `--image-path` (required for `i2i`)
- `--modalities` (optional output modalities override)
- `--output-dir` (default: `/tmp/dynin_online_outputs`)

## Notes

- This client currently supports only `t2i`, `t2s`, and `i2i`.
- `t2t` is intentionally not exposed in this online example.
- This example intentionally uses the OpenAI-compatible chat completion endpoint.
- Task routing for non-text outputs relies on Dynin task trigger tokens (`<|t2i|>`, `<|i2i|>`, `<|t2s|>`) injected by the client.
- Outputs are saved under `/tmp/dynin_online_outputs` by default.
- Dynin stage-0 warmup can take a while on first startup; do not send requests before startup completes.
- Dynin itself can execute text-returning tasks such as `t2t`, `s2t`, `i2t`, and `v2t`, but this online serving example currently runs stage-0 in `generation` mode. In that path, the generation worker does not surface the final text as `output.text`, so OpenAI chat responses for those text-output tasks may complete internally but still return empty text.
