# Performance config schema and runner selection

Keep the model case near its existing perf JSON rather than introducing a new
configuration system. A benchmark row is only collected if its JSON object
matches the runner that the CI step invokes.

## Two runners, one directory

`tests/dfx/perf/tests/*.json` mixes two schemas. The split is by `benchmark_params`
keys, not by marks, `server_type`, or endpoint:

- `benchmark_params[].dataset_name` → `run_benchmark.py` (omni / TTS / OpenAI
  image+video generation via `vllm bench serve --omni`).
- `benchmark_params[].dataset` **without** `dataset_name` → the legacy
  `run_diffusion_benchmark.py` (diffusion client, usually `/v1/chat/completions`
  or custom jsonl).

`is_diffusion_perf_config()` in [tests/dfx/conftest.py](../../../../tests/dfx/conftest.py)
implements this split. `run_benchmark.py` loads every `*.json` under the tests
directory and then filters out diffusion-schema configs, so a legacy case is
neither collected nor run by the new runner. When `--test-config-file` contains
at least one filtered case, it prints what it kept:

```
--test-config-file: loaded N omni/tts/generation case(s); skipped M remaining diffusion case(s) (chat completions / custom jsonl)
```

For a Qwen-Image-2.1-style image or video generation case, use
`run_benchmark.py`. Do not set `server_type` or `task` on those cases.

| | Legacy diffusion runner | Unified runner |
|---|---|---|
| Script | `tests/dfx/perf/scripts/run_diffusion_benchmark.py` | `tests/dfx/perf/scripts/run_benchmark.py` |
| Result dir env | `DIFFUSION_BENCHMARK_DIR` (default `tests/dfx/perf/results`) | **`BENCHMARK_DIR`** (unset → `tests`) |
| Underlying client | `benchmarks/diffusion/diffusion_benchmark_serving.py` subprocess | `vllm bench serve --omni` |
| Parameter key case | kebab-case (`num-prompts`) | **snake_case** (`num_prompts`) |
| Endpoint | implied by `task` | explicit `endpoint` |
| Result file | `diffusion_result_{stem}_{hw}_{ts}.json` | `result_{test_name}_{hw}_{name}_{dataset}_{flow}_{n}_in{}_out{}_{ts}.json` (`{flow}` is the request rate or concurrency, `{n}` the prompt count, and `{hw}` is omitted for `H100`) |

The two directory env vars must not be mixed: a `DIFFUSION_BENCHMARK_DIR` step
run through `run_benchmark.py` writes results to the wrong place, and vice versa.

## Choose the closest existing case

| Output contract | Repository example | Reusable behavior |
|---|---|---|
| Single-GPU image baseline | [test_qwen_image_21_vllm_omni.json](../../../../tests/dfx/perf/tests/test_qwen_image_21_vllm_omni.json) | `run_benchmark.py`, `/v1/images/generations`, `extra_body` size/steps; the same file's `step-execution` case adds `max-num-seqs` for a concurrency sweep |
| Multi-GPU image (Ulysses) | same file, `ulysses-degree: 4` case | `serve_args` parallelism; `cards_4` mark |
| Video I2V | [test_wan22_i2v_vllm_omni.json](../../../../tests/dfx/perf/tests/test_wan22_i2v_vllm_omni.json) | `dataset_name: random-mm`, `/v1/videos`, `extra_body` frames/fps/seed |
| Legacy diffusion | [test_qwen_image_vllm_omni.json](../../../../tests/dfx/perf/tests/test_qwen_image_vllm_omni.json) | `server_type`/`task`/`dataset` kebab-case schema |

Reuse these for lifecycle and interface patterns. Their resolution, steps,
guidance, parallelism, and hardware rows are model-specific.

## Case object schema (unified runner)

```json
{
  "test_name": "test_<model>_single_device",
  "mark": [
    {"hardware_marks": {"res": {"cuda": ["H100", "B200"]}, "num_cards": 1}},
    "full_model",
    "diffusion"
  ],
  "description": "Single-device baseline (no parallelism)",
  "server_params": {
    "model": "Qwen/Qwen-Image-2.1",
    "serve_args": {"vae-use-tiling": true, "enable-diffusion-pipeline-profiler": true}
  },
  "benchmark_params": [
    {
      "name": "512x512_steps20",
      "dataset_name": "random",
      "tokenizer": "gpt2",
      "endpoint": "/v1/images/generations",
      "num_prompts": 10,
      "max_concurrency": 1,
      "random_input_len": 8,
      "random_output_len": 1,
      "percentile_metrics": "e2el",
      "extra_body": {"width": 512, "height": 512, "num_inference_steps": 20}
    }
  ]
}
```

Top-level keys:

| Key | Required | Notes |
|---|---|---|
| `test_name` | Yes | Unique id; becomes the pytest param id prefix `{test_name}-{name}` |
| `mark` | For `-m` selection | Array with **exactly one** `hardware_marks` object plus marker-name strings |
| `server_params` | Yes | `model` + optional `serve_args` dict (server flags, kebab-case inside `serve_args`) |
| `benchmark_params` | Yes | One or more benchmark rows sharing this server |

`mark` rules (enforced by `resolve_pytest_marks`): the field is a list; it must
contain exactly one `{"hardware_marks": {...}}` object and only that key; other
entries must be registered marker names (`full_model`, `diffusion`, `slow`,
`omni`, `tts`, `local_model`, …). `hardware_marks` uses the same shape as
`@hardware_test`; `num_cards` adds `cards_{n}` (including `cards_1` when
omitted).

CUDA — and only CUDA — may list several SKUs on one item:
`res: {"cuda": ["H100", "B200"]}` attaches **both** `H100` and `B200` marks to
that one item, so `-m "H100 and B200 and cards_1"` matches it (this is the exact
selector the Qwen-Image-2.1 nightly step uses). `-m` is boolean AND, so every
listed name must be present on the item: a case declared only for `H100` is not
reachable with `-m "H100 and B200"`. Other platforms take a single SKU
(`res: {"npu": "A3"}`), selected with e.g. `-m "npu and A3 and cards_1"`. A
platform-agnostic `-m "cuda and cards_1"` or `-m "npu and A3 and cards_1"` is
the safe way to target one platform. Different per-platform `num_cards` cannot
share one item (raises `ValueError`) — use `@hardware_test`, or call
`hardware_marks()` once per platform.

## `benchmark_params` keys

| Key | Required | Notes |
|---|---|---|
| `name` | Recommended | Stable suffix for the pytest id and result filename |
| `num_prompts` | Yes | Scalar or sweep array |
| `max_concurrency` | No | Scalar or sweep array; use one of `max_concurrency`/`request_rate` per row |
| `request_rate` | No | QPS sweep; concurrency mode forces `--request-rate inf` |
| `num_warmups` | No | Non-negative integer; runner default is 2 (QPS) or `max(2, concurrency)` |
| `dataset_name` | Yes | `random` (text-only inputs) or `random-mm` (synthetic image/video input) |
| `tokenizer` | Required for `dataset_name: random` | See below |
| `random_input_len` / `random_output_len` | Required for `random` | Construct the synthetic prompt |
| `endpoint` | Yes for generation | `/v1/images/generations`, `/v1/images/edits`, `/v1/videos` |
| `extra_body` | For generation | `{width, height, num_inference_steps, true_cfg_scale, negative_prompt, …}`; video adds `num_frames`, `fps`, `seed` |
| `percentile_metrics` | Recommended | Comma list, e.g. `e2el`; forwarded to `--percentile-metrics` |
| `baseline` | No | Hardware-nested thresholds; see [baseline-and-report.md](baseline-and-report.md) |

Any key not listed as a JSON-only key (see below) is passed to
`vllm bench serve` as `--{key with underscores replaced by hyphens}`
(`arg_name = f"--{key.replace('_', '-')}"`). This is why the JSON must use
**snake_case**: writing `num-prompts` / `max-concurrency` produces empty sweep
lists (see pitfalls).

Keys that are excluded from the generic forwarding (`exclude_keys` in
`run_benchmark.py`): `request_rate`, `baseline`,
`num_prompts`, `max_concurrency`, `num_warmups`, `task`, `enabled`, `eval_phase`,
`trust_remote_code`, `name`, `enable_negative_prompt`, `random_request_config`,
`num_input_images`, `warmup_requests`, `warmup_concurrency`,
`warmup_num_inference_steps`, `expected_duplex_audio_turns_per_session`,
`omniinteract_min_ia_qtf1`, `judge_server_params`, and the OmniInteract aggregate
keys (`_OMNIINTERACT_AGGREGATE_MIN_IA_QTF1`, `_OMNIINTERACT_AGGREGATE_SUBSETS`,
`_OMNIINTERACT_AGGREGATE_GROUP`).

Some excluded keys belong to other runners. `enable_negative_prompt` is consumed
by the legacy diffusion benchmark client, not by the unified runner; copying it
from a legacy row has no effect here. `trust_remote_code` is excluded from the
generic forwarding but is appended explicitly as `--trust-remote-code`.

## Sweep semantics

`num_prompts`, `request_rate`, and `max_concurrency` accept a scalar or an
array. The runner expands a scalar single to match the longest sweep list and
`zip`s them; qps mode emits one `--request-rate/--num-prompts` pair per step and
concurrency mode emits `--max-concurrency/--num-prompts/--request-rate inf`.
Mismatched list lengths raise `ValueError`. A `num_warmups` that is not a
non-negative integer raises `ValueError`. Supplying both `request_rate` and
`max_concurrency` runs the QPS sweep and then the concurrency sweep back to back;
use one per row.

## Required `tokenizer` (contract)

`dataset_name: random` builds a dataset and asserts the tokenizer is not
`None`, so every such row must name a **resolvable** tokenizer. Qwen-Image-2.1's
HF repo is diffusers layout: the root has `model_index.json` but no
`config.json`/`model_type`, so pointing `tokenizer` at the model root fails with
`Unrecognized model in <root>`; its tokenizer lives in the `processor/`
subdirectory.

- **Committed configs keep `"tokenizer": "gpt2"`** (the path is machine-specific
  and absent in CI). Confirm the CI image caches `gpt2`, or the step will fail.
- **Never commit a model-local tokenizer path.** It is not portable to CI or to
  another machine; a path that only resolves on the author's checkout is a config
  bug, not a valid row.

## What the runner asserts (and does not)

`assert_result` checks `result["completed"] == num_prompts` (all requests
succeeded), fixed-length stage workloads when configured, and the OmniInteract
metrics where applicable. It does **not** compare any metric against a threshold.
A row with no `baseline` block still runs; a row with a `baseline` block is not
automatically enforced unless something consumes it (see
[baseline-and-report.md](baseline-and-report.md)).

`extra_body` is JSON-serialized into `--extra-body` without key validation. A
misspelled key, or a key the served model does not support, is silently ignored,
so the row can measure different work than it declares. Verify the effective
workload after the run
([measurement-contract.md](measurement-contract.md#verify-the-effective-workload)).

If the benchmark process produces no result file, `run_benchmark()` writes the
`result_omni_template.json` fallback to the result path and prints
`Benchmark result file not generated, fallback to template`. A result file
therefore does **not** by itself prove the benchmark ran; check its contents.

## Pitfalls

1. **Wrong key case.** snake_case is required. `num-prompts`/`max-concurrency`
   produce empty sweep lists → `zip` yields nothing → **zero benchmark iterations
   run and the test still passes**.
2. **Wrong runner/schema.** A legacy-schema config run through `run_benchmark.py`
   (or the reverse) is skipped, pytest selects 0 tests, and the step exits 5.
   The matching-runner regression test catches this; see
   [ci-wiring.md](ci-wiring.md#validate-collection-routing-and-execution).
3. **Missing `tokenizer`** for `dataset_name: random` → `assert tokenizer is
   not None`. A `processor/`-style tokenizer additionally logs a harmless
   `POST /tokenize 500` / `/tokenize unavailable, skipping alignment` pair; it is
   not the cause of the failure.
4. **Wrong directory env.** `BENCHMARK_DIR` (unified) vs `DIFFUSION_BENCHMARK_DIR`
   (legacy); results land in the wrong place or under `tests/` when
   `BENCHMARK_DIR` is unset.
5. **`-m` boolean AND.** A selector must list only names the target item
   actually carries. `-m "H100 and B200 and cards_1"` matches a CUDA item that
   declares both SKUs, but would match nothing if the case declares only one;
   `-m "cuda and cards_1"` / `-m "npu and A3 and cards_1"` are the safe
   single-platform selectors.
6. **Committing a source workaround.** A workaround that makes a row run must
   stay an env pin, never a source patch; see
   [measurement-contract.md](measurement-contract.md#attention-backend).
7. **Template fallback masks a failed run.** A sweep that produces no result file
   is replaced by `result_omni_template.json`, so a result JSON can exist without
   any benchmark having run. Check the contents and the console log, not the file
   count.
8. **Inert `extra_body` field.** `extra_body` is not validated: a misspelled or
   unsupported key (for example `negative_prompt` without `true_cfg_scale > 1`)
   is silently ignored and the row measures a different workload. Confirm the
   effective steps and guidance after the run.
