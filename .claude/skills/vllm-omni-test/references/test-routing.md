# Test Routing Reference

Use this reference to map testing goals to levels, markers, and runnable commands.

**Repo paths** (`.buildkite/`, `docs/contributing/…`): link with repo-relative paths from this file (e.g. `../../../../.buildkite/test-ready.yml`, `../../../../docs/contributing/ci/CI_5levels.md`).

## Model-centric e2e filename convention (L2–L4)

When generating a **new** test module tied to a **specific model** under `tests/e2e/offline_inference/` or `tests/e2e/online_serving/`:

| Level | Pattern | Example (`Qwen/Qwen2.5-Omni-7B`) |
|-------|---------|----------------------------------|
| **L2**, **L3** | `test_{lowercase_model_slug}.py` | `test_qwen2_5_omni.py` |
| **L4** | `test_{lowercase_model_slug}_expansion.py` | `test_qwen2_5_omni_expansion.py` |

**Slug rules for `{lowercase_model_slug}`** (canonical: [SKILL.md](../SKILL.md) § *Naming: generated test module files*):

1. Start from the HuggingFace-style id (e.g. `Qwen/Qwen2.5-Omni-7B`), but **do not** put the org into the filename: use the **repo segment only** (`Qwen2.5-Omni-7B`), not `Qwen_...` / `qwen_qwen2_5_...`.
2. **Lowercase**; replace `.`, `-`, and whitespace with a single `_` (e.g. `Qwen2.5-Omni-7B` → `qwen2_5_omni`). **Omit** trailing size tokens such as `7b` / `30b` in the basename when a single file covers that model line in the directory (matches `test_qwen2_5_omni.py` in-tree).
3. If two checkpoints in the same folder need separate modules, add a **minimal** disambiguator (e.g. `_7b` vs `_3b`) **only then**.
4. **L1** unit tests are **not** bound to this pattern; use `tests/<area>/test_<feature>.py` as today.

## Model type markers (`omni` / `tts` / `diffusion`)

Every model-centric e2e test must declare **exactly one** type marker:

| Marker | Model family | Typical APIs |
|--------|--------------|--------------|
| `pytest.mark.omni` | Multimodal LLM (Qwen-Omni, …) | `send_omni_request`, `omni_runner`, stage YAML |
| `pytest.mark.tts` | TTS / speech synthesis | `send_audio_speech_request`, `/v1/audio/speech` |
| `pytest.mark.diffusion` | Diffusion generative models | `send_diffusion_request`, `send_video_diffusion_request`, `send_images_generations_http_request`, `OmniDiffusionSamplingParams` |

**Diffusion** uses one marker for all output modalities. Nightly CI splits diffusion by **group**, not by extra markers:

| Nightly group (`test-nightly.yml`) | Diffusion scope | Examples |
|-----------------------------------|-----------------|----------|
| **Diffusion X2I(&A&T) Model Test** | x2**i** / x2**a** / x2**t** — image, audio, text (non-video) | Qwen-Image*, BAGEL, FLUX, SD3, Z-Image, LongCat, DreamZero |
| **Diffusion X2V Model Test** | x2**v** — video only | Wan2.2, HunyuanVideo 1.5, Wan VACE |

PR labels for selective nightly runs: `diffusion-x2iat-test`, `diffusion-x2v-test` (plus `nightly-test` / `NIGHTLY=1`).

## Level and Marker Mapping

| Goal | Suggested Level | Marker baseline | Typical location |
|------|------------------|-----------------|------------------|
| Unit logic, regression on pure code path | L1 | `core_model and cpu` | `tests/<component>/test_*.py` |

### L1 unit tests — mocking (`mocker`, not `unittest.mock`)

L1 modules run in **Simple Test** (`test-ready.yml` / `test-merge.yml`: `-m 'core_model and cpu'`). When isolation requires doubles:

- **Use** the **`mocker`** fixture (`pytest-mock`): `mocker.patch`, `mocker.spy`, `mocker.Mock`, `mocker.MagicMock`, `mocker.AsyncMock`.
- **Use** **`monkeypatch`** for env vars and simple `setattr` / `delattr` without mock objects.
- **Do not use** `unittest.mock` — no `from unittest.mock import ...`, no `@patch`, no `with patch(...)`.

```python
# Good (L1)
def test_handler(mocker):
    mocker.patch("vllm_omni.pkg.fn", return_value=0)

# Bad (L1)
from unittest.mock import patch

@patch("vllm_omni.pkg.fn")
def test_handler(mock_fn): ...
```

E2E (L2+) should not rely on mocks unless documenting a rare exception; prefer real (or lightweight) execution paths.
| Basic integration/e2e | L2 | `core_model` + **one of** `omni` / `tts` / `diffusion` | `tests/e2e/...` |
| Advanced integration | L3 | `advanced_model` + type marker | `tests/e2e/...` |
| Full function / nightly | L4 | `full_model` (nightly) + type marker | **Function:** `tests/e2e/*_expansion.py`; **Perf:** `tests/dfx/perf/tests/*.json`; **Accuracy:** `tests/e2e/accuracy/` |
| Invalid HTTP / param validation | Weekly (dfx) | `pytest.mark.slow` + type marker + `H100` or `L4` | `tests/dfx/reliability/invalid_param_test/test_invalid_*.py` |

## Marker Selection Rules

1. **Level** (pick one):
   - `core_model` — L1/L2 (`test-ready.yml`)
   - `advanced_model` — L3 (`test-merge.yml`)
   - `full_model` — L4 nightly (`test-nightly.yml`); some expansion tests still carry both `advanced_model` and `full_model` during migration
2. **Model type** (pick one): `omni`, `tts`, or `diffusion`
3. **Cross-cutting** when relevant: `parallel`, `cache`, `example`, `benchmark`
4. **Hardware**: `cpu`, `cuda`, `rocm`, `npu`, `L4`, `H100`, `distributed_cuda`, …
5. Multi-card: `@hardware_test(...)` in `tests/helpers/mark.py`

## Command Templates

### Quick local checks

```bash
cd tests
pytest -s -v test_xxxx.py
```

### L1

```bash
cd tests
pytest -s -v -m "core_model and cpu"
```

### L2

```bash
cd tests
pytest -s -v -m "core_model and not cpu" --run-level=core_model
```

### L3 (merge)

```bash
cd tests
pytest -s -v -m "advanced_model" --run-level=advanced_model
```

### L4 (nightly)

**Function** (e2e expansion — default when user asks for “L4 functional cases”):

```bash
cd tests
pytest -s -v e2e/online_serving/test_qwen_image_expansion.py -m "full_model and diffusion and H100" --run-level=full_model
```

**Perf** (dfx benchmark harness — separate pillar; not `omni_server` fixtures):

```bash
cd tests
export DIFFUSION_BENCHMARK_DIR=tests/dfx/perf/results
export DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN
# CI-like: single JSON file (nightly Perf Test steps)
pytest -s -v dfx/perf/scripts/run_diffusion_benchmark.py \
  --test-config-file dfx/perf/tests/test_qwen_image_vllm_omni.json
# Local bulk load: all *.json under tests/dfx/perf/tests/, filter by JSON mark
pytest -sv dfx/perf/scripts/run_diffusion_benchmark.py -m "full_model and diffusion and H100"
pytest -sv dfx/perf/scripts/run_benchmark.py -m "full_model and omni and H100"
```

Broad marker sweep (when no explicit file shard):

```bash
cd tests
pytest -s -v -m "full_model" --run-level=full_model
```

## L4 nightly pillars (Function / Accuracy / Perf)

Nightly **L4** for a model is often **multiple jobs** in `test-nightly.yml`, not a single pytest module:

| Pillar | User says | Deliver | Nightly step |
|--------|-----------|---------|--------------|
| **Function** | “L4 functional cases” / “functional” / `*_expansion.py` | `tests/e2e/.../test_<model>_expansion.py` | `· Function Test with H100/L4` |
| **Perf** | “performance” / “perf” / “benchmark” / “full L4” | `tests/dfx/perf/tests/test_<model>_vllm_omni.json` + runner | `· Perf Test · <Model>` |
| **Accuracy** | “accuracy” / “similarity” | `tests/e2e/accuracy/test_<model>*.py` | `· Accuracy Test` |

**Rule:** “L4 functional cases” → **Function only** by default. State that Perf/Accuracy are separate pillars; add them only when requested.

**Diffusion perf JSON** (`tests/dfx/perf/tests/test_<slug>_vllm_omni.json`): array of **case objects**, each with:

| Field | Required | Notes |
|-------|----------|-------|
| `test_name` | Yes | Unique server/workload id |
| `mark` | Recommended | When present: **`hardware_marks` required** + optional `marks` (`full_model`, `diffusion`, `local_model`, …). Parsed by `resolve_pytest_marks`; enables local `-m` filtering. |
| `server_type` | Diffusion | e.g. `"vllm-omni"` — routes case to `run_diffusion_benchmark.py` |
| `benchmark_endpoint` | Diffusion | e.g. `/v1/videos`, `/v1/images/generations` |
| `server_params` | Yes | `model`, `serve_args`, … |
| `benchmark_params[]` | Yes | Each row: **`name`** (pytest id suffix), workload fields, **`baseline`** |

Copy structure from `test_qwen_image_vllm_omni.json` or `test_cosmos3_vllm_omni.json` (2-GPU mark example).

**Omni perf JSON**: `test_qwen3_omni_*.json` — same `mark` shape with `"marks": ["full_model", "omni"]`; run via `run_benchmark.py`.

**Runner split**: `is_diffusion_perf_config()` → diffusion when `server_type` is set or `"diffusion"` ∈ `mark.marks`; otherwise omni/tts → `run_benchmark.py`.

**Param ids**: `{test_name}-{benchmark_params.name}` (e.g. `test_tts-p0`, not `test_tts-0`).

**Result filenames**: runtime hardware from `get_runtime_resource_label()`; `H100` omitted on default CI pool.

**Diffusion perf nightly step** (under **Diffusion X2I(&A&T)**; copy full `kubernetes` plugins from `· Perf Test · Qwen-Image`):

```yaml
      - label: ":full_moon: Diffusion X2I(&A&T) · Perf Test · <Model-Display-Name>"
        key: nightly-diffusion-x2iat-performance-<slug>
        timeout_in_minutes: 180
        commands:
          - export DIFFUSION_BENCHMARK_DIR=tests/dfx/perf/results
          - export DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN
          - export CACHE_DIT_VERSION=1.3.0
          - |
            set +e
            pytest -s -v tests/dfx/perf/scripts/run_diffusion_benchmark.py --test-config-file tests/dfx/perf/tests/test_<slug>_vllm_omni.json
            EXIT=$$?
            buildkite-agent artifact upload "tests/dfx/perf/results/diffusion_result_*.json"
            buildkite-agent artifact upload "tests/dfx/perf/results/logs/*.log"
            exit $$EXIT
        agents:
          queue: "mithril-h100-pool"
        plugins:
          # … full kubernetes block (often multi-GPU) — copy from Perf Test · Qwen-Image
```

| Model type | Perf runner | Config example |
|------------|-------------|----------------|
| Diffusion X2I/X2V | `dfx/perf/scripts/run_diffusion_benchmark.py` | `test_qwen_image_vllm_omni.json`, `test_cosmos3_vllm_omni.json` |
| TTS | `dfx/perf/scripts/run_benchmark.py` | `test_tts.json`, `test_voxcpm2.json`, `test_higgs_audio_v3.json` |
| Omni | `dfx/perf/scripts/run_benchmark.py` | `test_qwen3_omni_async_chunk.json`, `test_qwen3_omni_no_async_chunk.json`, … |

Do **not** put throughput/latency baselines inside `test_*_expansion.py` — that belongs in the dfx perf JSON + nightly Perf job.

## Invalid parameter validation (weekly / dfx)

When the user’s test plan includes **invalid parameter validation**, **invalid request bodies**, or **HTTP 4xx** validation against a live server:

1. **Do not** author these in `tests/e2e/online_serving/test_*.py` or `*_expansion.py`. **Move** any drafted `test_*` into `tests/dfx/reliability/invalid_param_test/`.
2. **Pick script by route** (extend in-tree file):

| Route family | Script |
|--------------|--------|
| Omni chat / WS video·realtime | `test_invalid_omni_chat.py` |
| `/v1/audio/speech` (+ stream / batch / voices) | `test_invalid_audio_speech.py` |
| Audio diffusion | `test_invalid_audio_diffusion.py` |
| `/v1/images/generations` | `test_invalid_image_generation.py` |
| `/v1/images/edits` | `test_invalid_image_editing.py` |
| `/v1/videos*` | `test_invalid_video_generation.py` |
| Sleep / wakeup / server control | `test_invalid_server_control.py` |

3. **Style:** `pytestmark = [pytest.mark.slow, pytest.mark.<type>]`; `_PARAMS` + `hardware_marks`; `send_*_http_request` with `err_code` + `err_message`; parametrized `body_spec` rows with `id=`; `_minimal_*_json()` helpers; `_SKIP_ISSUE_3649` when tracked in [#3649](https://github.com/vllm-project/vllm-omni/issues/3649).
4. **CI:** [`.buildkite/test-weekly.yml`](../../../../.buildkite/test-weekly.yml) group **Reliability Test - Invalid parameters Test** — **not** ready/merge/nightly.

```bash
# Weekly H100 (diffusion / omni / video invalid-param)
cd tests
pytest -s -v dfx/reliability/invalid_param_test/ -m "slow and H100"

# Weekly L4 (e.g. Qwen3-TTS speech invalid-param)
pytest -s -v dfx/reliability/invalid_param_test/ -m "slow and L4"

# Single script / case
pytest -s -v dfx/reliability/invalid_param_test/test_invalid_image_generation.py::test_images_generations_invalid_requests -m "slow and H100"
```

**Trigger:** `WEEKLY=1` or PR label `weekly-test`. Appending cases to existing scripts usually needs **no YAML edit** (directory sweep). No `source_file_dependencies` on weekly steps.

### Platform-targeted examples

```bash
cd tests
pytest -s -v -m "core_model and distributed_cuda and L4" --run-level=core_model
```

### Concrete e2e paths (common in-tree)

Paths are relative to `tests/` after `cd tests`.

| Scenario | Example command |
|----------|------------------|
| **Omni** offline L2 | `pytest -s -v e2e/offline_inference/test_qwen2_5_omni.py -m "core_model and omni and not cpu" --run-level=core_model` |
| **Omni** offline L2 — one test | `pytest -s -v e2e/offline_inference/test_qwen2_5_omni.py::test_text_to_text -m "core_model and omni and not cpu" --run-level=core_model` |
| **Omni** online L2 | `pytest -s -v e2e/online_serving/test_qwen3_omni.py -m "core_model and omni" --run-level=core_model` |
| **Omni** offline L2 (Qwen3.5-9B VL) | `pytest -s -v e2e/offline_inference/test_qwen3_5_9b.py -m "core_model and omni and not cpu" --run-level=core_model` |
| **TTS** online L2 | `pytest -s -v e2e/online_serving/test_qwen3_tts_base.py -m "core_model and tts" --run-level=core_model` |
| **TTS** online L2 — one test | `pytest -s -v e2e/online_serving/test_qwen3_tts_base.py::test_text_to_audio_001 -m "core_model and tts" --run-level=core_model` |
| **Diffusion X2I** offline L2 | `pytest -s -v e2e/offline_inference/test_t2i_model.py -m "core_model and diffusion and not cpu" --run-level=core_model` |
| **Diffusion X2I** L2 online smoke | `pytest -s -v e2e/online_serving/test_qwen_image.py e2e/online_serving/test_bagel.py -m "core_model and diffusion" --run-level=core_model` |
| **Diffusion X2V** L2 online smoke | `pytest -s -v e2e/online_serving/test_wan22_t2v.py -m "core_model and diffusion" --run-level=core_model` |
| **Diffusion X2I** L4 online expansion (H100) | `pytest -s -v e2e/online_serving/test_qwen_image_expansion.py -m "full_model and diffusion and H100" --run-level=full_model` |
| **Diffusion X2I** L4 perf (nightly) | `pytest -s -v dfx/perf/scripts/run_diffusion_benchmark.py --test-config-file dfx/perf/tests/test_qwen_image_vllm_omni.json` |
| **Diffusion X2I** L4 online expansion (Edit) | `pytest -s -v e2e/online_serving/test_qwen_image_edit_expansion.py -m "full_model and diffusion and H100" --run-level=full_model` |
| **Diffusion X2V** L4 nightly | `pytest -s -v e2e/online_serving/test_wan22_expansion.py -m "full_model and cuda" --run-level=full_model` |
| **Invalid param** weekly H100 | `pytest -s -v dfx/reliability/invalid_param_test/ -m "slow and H100"` |
| **Invalid param** weekly L4 | `pytest -s -v dfx/reliability/invalid_param_test/ -m "slow and L4"` |

### Agent / author completion checklist

When adding or modifying tests, do not stop at “where the file lives” — also deliver:

1. **Local**: `cd tests` + `pytest -s -v <path>` (and `path::test_func` when a single case is enough).
2. **CI-like**: marker string + `--run-level` matching the pipeline (`core_model` / `advanced_model` / `full_model`).
3. **L1 mocks**: `mocker` / `monkeypatch` only; no `unittest.mock`.
4. **API client + assert placement**: **General** validation → implement in `assertions.py`, call **inside** `send_*_request` in `runtime.py`; tests call **`send_*_request` only**. **Special** case validation → `assert_*` in `assertions.py`, called **in the test** after `send_*_request`. Low-level `send_*_http_request` is for negative/dfx tests (`err_code`), not ordinary L2+ success e2e.
5. **Shared assertions**: logic in **`tests/helpers/assertions.py`**; general checks inside `send_*_request`; special checks only in the test. No `_assert_*` in `test_*.py`.
6. **One case → one `test_*`**: function name reflects what is validated; no `if case_id == ...` mega-test merging multiple scenarios.
7. **Fixture scope**: default **`omni_server` + `openai_client`** / **`omni_runner` + `omni_runner_handler`** (module). Use **`omni_server_function` + `openai_client_function`** / **`omni_runner_function` + `omni_runner_handler_function`** only when each `test_*` must spawn a fresh instance. Parametrize name must match fixture (`omni_server` vs `omni_server_function`).
8. **Type marker**: `omni`, `tts`, or `diffusion` on every model e2e module.
9. **Diffusion L4 Function**: wire `*_expansion.py` into **X2I(&A&T)** or **X2V** **Function Test** in **`test-nightly.yml` only** — do not add `test-merge.yml` unless the user also requested L3.
10. **Diffusion L4 Perf** (only when requested): add `tests/dfx/perf/tests/test_<slug>_vllm_omni.json` + **Perf Test · &lt;Model&gt;** step (artifact upload); not part of “L4 functional cases” by default.
11. **E2E Buildkite (L2/L3 only)**: `test-ready.yml` / `test-merge.yml` steps need **`source_file_dependencies`** + full **`agents` + `plugins`**. **L4 nightly** uses explicit file lists or perf scripts in `test-nightly.yml` (no merge job).
12. **Invalid param**: cases in **`tests/dfx/reliability/invalid_param_test/`** (route-matching script), `send_*_http_request` + `err_code`, `pytest.mark.slow` + `H100`/`L4`; CI = **`test-weekly.yml`** only — do not put in e2e or nightly.
13. **Prerequisites**: GPU tier, HF cache/token, and any module `skipif` / platform-only YAML.

## Buildkite pipeline mapping

| Level | Repo file | Model-type grouping |
|-------|-----------|---------------------|
| L1, L2 | [`.buildkite/test-ready.yml`](../../../../.buildkite/test-ready.yml) | Steps prefixed **Omni ·**, **TTS ·**, **Diffusion ·** under **E2E Test** — **`source_file_dependencies` required** |
| L3 | [`.buildkite/test-merge.yml`](../../../../.buildkite/test-merge.yml) | Per-model E2E steps; `-m "advanced_model and …"` — **`source_file_dependencies` required** |
| L4 | [`.buildkite/test-nightly.yml`](../../../../.buildkite/test-nightly.yml) | **Omni / TTS / Diffusion X2I(&A&T) / X2V** — each group may have **Function**, **Accuracy**, **Perf**, **Doc** steps |
| Invalid param (weekly) | [`.buildkite/test-weekly.yml`](../../../../.buildkite/test-weekly.yml) | **Invalid parameters Test · H100** / **· L4** — sweeps `tests/dfx/reliability/invalid_param_test/` |

### `source_file_dependencies` (E2E Test only — ready & merge)

Required on each step inside the **E2E Test** group in `test-ready.yml` and `test-merge.yml`. Typical entries:

| Category | Path pattern |
|----------|----------------|
| Test module | `tests/e2e/online_serving/test_<slug>.py`, `tests/e2e/offline_inference/test_<slug>.py` |
| API client helpers | `tests/helpers/runtime.py` (when the e2e job uses newly added `send_*` methods) |
| Shared assert helpers | `tests/helpers/assertions.py` (when the e2e job uses newly added `assert_*` helpers) |
| AR / omni model | `vllm_omni/model_executor/models/<family>/` |
| Diffusion model | `vllm_omni/diffusion/models/<family>/` |
| Stage processor | `vllm_omni/model_executor/stage_input_processors/<family>.py` |
| Deploy config | `vllm_omni/deploy/<name>.yaml` or `vllm_omni/deploy/ci/<name>.yaml` |

Copy the dependency block from the closest in-tree E2E step for the same model family. Update the list whenever the pytest command or server YAML changes.

### E2E `agents` + `plugins` blocks (required — do not truncate)

When generating or documenting E2E steps, **always include the complete `plugins` section**. Set **`timeout_in_minutes`** on the step for the job deadline; run **`pytest` directly** in `commands` (no `timeout 40m bash -c` wrapper around pytest).

Common patterns in `test-ready.yml` / `test-merge.yml`:

| Queue | Plugin | Required fields |
|-------|--------|-----------------|
| `mithril-h100-pool` | `kubernetes` | `resources.limits.nvidia.com/gpu`, `volumeMounts` (`devshm`, `hf-cache`), `env` (`HF_HOME`, `HF_TOKEN` via `secretKeyRef`), `nodeSelector: gpu-h100-sxm`, `volumes` (`devshm` emptyDir, `hf-cache` hostPath `/mnt/hf-cache`) |
| `gpu_1_queue` / `gpu_4_queue` | `docker#v5.2.0` | `image`, `always-pull`, `propagate-environment`, `shm-size: "8gb"`, `environment: HF_HOME`, `volumes: /fsx/hf_cache` (add `HF_TOKEN` when the step needs hub access) |

**H100 kubernetes template** (copy from **Diffusion · Bagel Test** / **Diffusion · Qwen Image Test** in-tree):

```yaml
  agents:
    queue: "mithril-h100-pool"
  plugins:
    - kubernetes:
        podSpec:
          containers:
            - image: 936637512419.dkr.ecr.us-west-2.amazonaws.com/vllm-ci-pull-through-cache/q9t5s3a7/vllm-ci-test-repo:$BUILDKITE_COMMIT
              resources:
                limits:
                  nvidia.com/gpu: 1
              volumeMounts:
                - name: devshm
                  mountPath: /dev/shm
                - name: hf-cache
                  mountPath: /root/.cache/huggingface
              env:
                - name: HF_HOME
                  value: /root/.cache/huggingface
                - name: HF_TOKEN
                  valueFrom:
                    secretKeyRef:
                      name: hf-token-secret
                      key: token
          nodeSelector:
            node.kubernetes.io/instance-type: gpu-h100-sxm
          volumes:
            - name: devshm
              emptyDir:
                medium: Memory
            - name: hf-cache
              hostPath:
                path: /mnt/hf-cache
                type: DirectoryOrCreate
```

**L4 docker template** (copy from **TTS · Qwen3-TTS Base Test** in `test-merge.yml`):

```yaml
  agents:
    queue: "gpu_1_queue"
  plugins:
    - docker#v5.2.0:
        image: public.ecr.aws/q9t5s3a7/vllm-ci-test-repo:$BUILDKITE_COMMIT
        always-pull: true
        propagate-environment: true
        shm-size: "8gb"
        environment:
          - "HF_HOME=/fsx/hf_cache"
        volumes:
          - "/fsx/hf_cache:/fsx/hf_cache"
```

When extending an existing step (e.g. add `tests/e2e/offline_inference/test_qwen_image.py` to **Diffusion · Qwen Image Test**), update `source_file_dependencies` and `commands` only; **keep the existing `agents` + `plugins` block unchanged** unless the hardware tier changes.

The root [`.buildkite/pipeline.yml`](../../../../.buildkite/pipeline.yml) decides **which** child file is uploaded. To run **L3** on a feature branch, comment out `if: build.branch == "main"` on the merge upload step. To run **L4** without `NIGHTLY=1`, comment out the nightly upload step’s `if: build.env("NIGHTLY") == "1"` and the same `if` on relevant steps inside `test-nightly.yml`. Revert such edits before merging.

Platform-specific pipelines (e.g. AMD) follow the same level → file pairing under `.buildkite/`.

## Diffusion RFC (#1832) Alignment Tips

For **X2I(&A&T)** coverage planning:

- Prioritize high-value feature combinations with minimal case count.
- Split into lightweight L1/L2 validation plus L4 `*_expansion.py` under the **X2I** nightly shards.
- **X2V** models (Wan, HunyuanVideo) stay in the **X2V** group — do not merge into X2I sweeps.

If hardware is insufficient, provide an executable reduced case plus a deferred full CI/nightly plan.
