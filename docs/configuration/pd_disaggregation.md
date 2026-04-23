# Prefill-Decode (PD) Disaggregation

PD disaggregation splits the Qwen3-Omni thinker into separate prefill and decode
stages so prompt processing and token generation can run on different workers.

After the config refactor, PD is no longer launched from a separate legacy
`stage_configs/*.yaml` file. Instead, it is enabled from the deploy config via
the `pd_disaggregation` section in
[`vllm_omni/deploy/qwen3_omni_moe.yaml`](gh-file:vllm_omni/deploy/qwen3_omni_moe.yaml).

## Current Config-Based Flow

At runtime, the config system does the following when
`pd_disaggregation.enabled: true`:

1. Load the normal 3-stage Qwen3-Omni pipeline + deploy config.
2. Dynamically split the thinker into:
   - stage `0`: thinker prefill
   - stage `1`: thinker decode
3. Shift downstream stages by one index:
   - talker: `1 -> 2`
   - code2wav: `2 -> 3`
4. Inject `is_prefill_only`, `is_decode_only`, and `kv_transfer_config` into the
   resolved runtime stage configs.
5. Reuse the existing PD detection / routing logic in the engine.

So the user-facing deploy file stays single-source, but the resolved runtime
config becomes a 4-stage PD pipeline.

## Requirements

- 3+ GPUs for the common layout:
  - prefill on GPU `0`
  - decode on GPU `1`
  - talker + code2wav on GPU `2`
- A KV connector supported by vLLM, such as `MooncakeConnector`
- Matching `tensor_parallel_size` on the prefill and decode thinker stages

## How to Enable PD

PD is enabled from the existing bundled deploy config:

- `vllm_omni/deploy/qwen3_omni_moe.yaml`

No additional user-facing YAML is required. The intent of the config refactor
is to keep Qwen3-Omni on a single deploy config and switch PD on through the
`pd_disaggregation` section in that file.

Edit `vllm_omni/deploy/qwen3_omni_moe.yaml` and enable / tune:

```yaml
pd_disaggregation:
  enabled: true
  async_chunk: false
  target_stage_id: 0
  stages:
    - role: prefill
      max_num_seqs: 16
      devices: "0"
      tensor_parallel_size: 1
      engine_extras:
        kv_transfer_config:
          kv_connector: "MooncakeConnector"
          kv_role: "kv_producer"
          kv_rank: 0
          kv_parallel_size: 2
          kv_connector_extra_config:
            mooncake_bootstrap_port: 25201
    - role: decode
      max_num_seqs: 64
      devices: "1"
      tensor_parallel_size: 1
      engine_extras:
        kv_transfer_config:
          kv_connector: "MooncakeConnector"
          kv_role: "kv_consumer"
          kv_rank: 1
          kv_parallel_size: 2
          kv_connector_extra_config:
            mooncake_bootstrap_port: 25202
  stage_overrides:
    - stage_id: 1
      devices: "2"
    - stage_id: 2
      devices: "2"
```

Notes:

- `target_stage_id: 0` means the original thinker is the stage being split.
- `async_chunk: false` matches the current PD path.
- The `pd_disaggregation.stage_overrides` block keeps the common 3-GPU layout:
  - original talker (`stage_id: 1`) stays on GPU `2`
  - original code2wav (`stage_id: 2`) stays on GPU `2`
- After PD expansion, these become runtime stage `2` and stage `3`.

## Launching with Config-Based PD

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 \
  --deploy-config vllm_omni/deploy/qwen3_omni_moe.yaml
```

If you edit the bundled deploy file in place, the explicit `--deploy-config`
flag is optional as long as the runtime resolves the default deploy config for
the model.

You can also enable PD from CLI without editing the YAML:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 \
  --deploy-config vllm_omni/deploy/qwen3_omni_moe.yaml \
  --enable-pd-disaggregation
```

To tune the generated prefill/decode runtime stages from CLI, reuse
`--stage-overrides` after PD is enabled. In the resolved 4-stage runtime config,
stage `0` is prefill and stage `1` is decode:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 \
  --deploy-config vllm_omni/deploy/qwen3_omni_moe.yaml \
  --enable-pd-disaggregation \
  --stage-overrides '{"0":{"max_num_seqs":8},"1":{"max_num_seqs":32}}'
```

## Tests

At the moment, the PD-aware tests are these three files:

- `tests/e2e/online_serving/test_qwen3_omni.py`
- `tests/e2e/online_serving/test_qwen3_omni_expansion.py`
- `tests/entrypoints/test_pd_disaggregation.py`

### 1. Online serving E2E

Both online-serving test files include the regular 2-GPU cases and the PD
3-GPU case in the same parametrized suite. The Qwen3-Omni coverage currently
uses these modes:

- `default`: non-PD, 2-GPU layout
- `async_chunk`: non-PD async-chunk path, 2-GPU layout
- `pd_default`: PD disaggregation, 3-GPU layout

No `VLLM_TEST_PD_MODE` environment variable is needed. The tests select the
desired mode directly from the parametrized config path, and all online-serving
Qwen3-Omni cases launch through the stage CLI harness (`use_stage_cli=True`).

Run `test_qwen3_omni.py`:

```bash
pytest -s -v tests/e2e/online_serving/test_qwen3_omni.py \
  -m "advanced_model" --run-level "advanced_model"
```

Run `test_qwen3_omni_expansion.py`:

```bash
pytest -s -v tests/e2e/online_serving/test_qwen3_omni_expansion.py \
  -m "advanced_model" --run-level "advanced_model"
```

Run a single expansion case, for example:

```bash
pytest -s -v tests/e2e/online_serving/test_qwen3_omni_expansion.py \
  -k "test_audio_in_video_002" \
  -m "advanced_model" --run-level "advanced_model"
```

### 2. PD unit / entrypoint coverage

`test_pd_disaggregation.py` does not require the old PD YAML anymore. It builds
a temporary deploy overlay inside the test process only, enables
`pd_disaggregation`, then verifies that the merged runtime config becomes a valid
4-stage PD pipeline. This temporary file is a test helper, not a user-facing
config artifact.

```bash
pytest tests/entrypoints/test_pd_disaggregation.py -q
```

## Operational Notes

- `MooncakeConnector` does not support heterogeneous TP sizes across the PD
  pair. Keep prefill and decode at the same `tensor_parallel_size`.
- If the thinker requires TP=2, both thinker stages must use TP=2 and be given
  separate GPU sets, for example `"0,1"` for prefill and `"2,3"` for decode.
- Choose connector ports and addresses that match your deployment. The values
  shown above are examples only.
