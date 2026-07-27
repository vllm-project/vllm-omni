---
name: vllm-omni-test
description: Generate and run tests for vllm-project/vllm-omni with CI-aligned levels and markers; wire new tests into Buildkite (test-ready.yml for L1/L2, test-merge.yml for L3, test-nightly.yml for L4). On completion, always provide copy-paste local and CI-like pytest commands plus prerequisites. Use when creating regression tests, adding L1-L4 coverage, selecting pytest markers, or validating fixes from issues/PRs.
---

# vLLM-Omni Test Generator & Runner

## Purpose

Use this skill to generate minimal, stable test cases and run them with the correct marker/level strategy for [vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni).

**Link convention:** Paths such as `.buildkite/` and `docs/contributing/` live at the **vllm-omni repo root**. Markdown links use repo-relative paths from this skill file (e.g. `../../../.buildkite/test-ready.yml`, `../../../docs/contributing/ci/CI_5levels.md`).

Default priorities:

1. Reproducible regression coverage for bug fixes
2. Correct test level and marker selection
3. Low flake, low dependency tests first
4. CI-compatible run commands
5. **Actionable run commands for the human**: whenever you add or change tests, always finish with **copy-paste-ready** `pytest` lines (local: single file and/or single test; CI-like: markers + `--run-level`), plus short **prerequisites** (GPU tier, HF cache, optional `model_prefix`). Do not assume the reader will infer commands from `test-routing.md` alone.

## Inputs

- Issue/PR link and summary
- Changed files or suspected code path
- Whether the user wants local quick validation or CI-equivalent validation
- Hardware constraints (CPU only / CUDA / ROCm / NPU)

## Workflow

### Step 1: Classify Test Goal

- **Bugfix regression**: start from a minimal failing scenario and add assertions that prevent recurrence. Before writing tests, output **`required` / `recommended` / `not_needed`**: **`required`** — stable logic/contract bug that should have been caught; **`recommended`** — environment-sensitive but a small regression still helps; **`not_needed`** — one-off external/config failure or existing tests already cover the path. Prefer the narrowest stable L1 (CPU) case; escalate to L2/L3 only when the bug needs real weights or serving.
- **Feature coverage**: verify new behavior and one negative/boundary case.
- **Perf/benchmark claim**: require benchmark-oriented tests and explicit metrics.

### Step 2: Select Test Level

- **L1**: unit/logic, deterministic, CPU-friendly, fastest feedback.
- **L2**: basic e2e/integration and platform-dependent checks.
- **L3/L4**: advanced model/integration/perf validation.

Use [references/test-routing.md](references/test-routing.md) for level-to-marker and command mapping.

### Step 3: Pick Markers

Always attach markers deliberately:

- **Level**: `core_model` (L1/L2) and/or `advanced_model` (L3) and/or `full_model` (L4 nightly)
- **Model type** (required on model-centric e2e — pick exactly one):
  - `omni` — end-to-end multimodal LLM pipelines (thinker/talker/stages; Qwen-Omni family)
  - `tts` — speech synthesis / TTS-only models (`/v1/audio/speech`, voice clone, etc.)
  - `diffusion` — generative diffusion models (image / audio / text / video from noise)
- **Cross-cutting area** (when relevant): `parallel`, `cache`, `example`, `benchmark`
- **Hardware**: `cpu`, `gpu`, `cuda`, `rocm`, `npu`, `L4`, `H100`, `distributed_cuda`, …
- Optional: `slow`, distributed markers when multi-card is required

**Baseline smoke (L2 + L3):** The simplest e2e case per model — default deploy, minimal request — should usually carry **both** `@pytest.mark.core_model` and `@pytest.mark.advanced_model` on the **same** test function so `test-ready.yml` and `test-merge.yml` share one test. `send_*_request` picks validation depth from `--run-level`. References: `test_voxcpm2_tts.py::test_text_to_audio_001`, `test_qwen3_tts_customvoice.py::test_text_to_audio_001`. Heavier scenarios use **`advanced_model` only**; L4 expansion uses **`full_model`**.

For hardware-aware tests, prefer `@hardware_test(...)` or `hardware_marks(...)` in `tests/helpers/mark.py`.

**Diffusion nightly split** (`test-nightly.yml`): all diffusion tests use `pytest.mark.diffusion`, but CI groups them by **output modality**, not by separate pytest markers:

| Nightly group | Scope | Typical models / paths |
|---------------|--------|-------------------------|
| **Diffusion X2I(&A&T) Model Test** | x2**i** (image), x2**a** (audio), x2**t** (text) and other **non-video** diffusion | Qwen-Image*, BAGEL, FLUX, SD3, Z-Image, LongCat, DreamZero, … — `test_*_expansion.py` under X2I shards; L4 sweep uses `-k "not test_wan and not test_bagel_expansion and not hunyuan"` for L4 |
| **Diffusion X2V Model Test** | x2**v** (video) only | Wan2.2, HunyuanVideo 1.5, Wan VACE, … — e.g. `test_wan22_expansion.py`, `test_hunyuan_video_15_expansion.py` |

Wire new **video** diffusion expansion tests into the **X2V** group; wire **image/audio/text** diffusion expansion tests into **X2I(&A&T)**. Do not place x2v modules in X2I shards (see comment in `test-nightly.yml` above the X2I group).

### Naming: generated test module files (L2–L4, model-centric e2e)

When adding **new** pytest modules whose primary scope is a **specific model** (typical under `tests/e2e/offline_inference/` or `tests/e2e/online_serving/`), use this filename pattern:

| Level | Filename pattern | Example (model `Qwen/Qwen2.5-Omni-7B`) |
|-------|------------------|----------------------------------------|
| **L2**, **L3** | `test_{lowercase_model_slug}.py` | `test_qwen2_5_omni.py` |
| **L4** | `test_{lowercase_model_slug}_expansion.py` | `test_qwen2_5_omni_expansion.py` |

**Slug rules for `{lowercase_model_slug}`:**

1. Start from the HuggingFace-style id (e.g. `Qwen/Qwen2.5-Omni-7B`), but **do not** put the org into the filename: use the **repo segment only** (`Qwen2.5-Omni-7B`), not `Qwen_...` / `qwen_qwen2_5_...`.
2. **Lowercase**; replace `.`, `-`, and whitespace with a single `_` (e.g. `Qwen2.5-Omni-7B` → `qwen2_5_omni`). **Omit** trailing size tokens such as `7b` / `30b` in the basename when a single file covers that model line in the directory (matches `test_qwen2_5_omni.py` in-tree).
3. If two checkpoints in the same folder need separate modules, add a **minimal** disambiguator (e.g. `_7b` vs `_3b`) only then.
4. **L1** unit tests are **not** bound to this pattern; use `tests/<area>/test_<feature>.py` as today.

Routing tables and commands: [references/test-routing.md](references/test-routing.md) § *Model-centric e2e filename convention*.

Existing references: `tests/e2e/offline_inference/test_qwen2_5_omni.py` (L2-style omni), `tests/e2e/offline_inference/test_qwen3_5_9b.py` (L2-style omni, single-stage VL), `tests/e2e/online_serving/test_qwen3_omni_expansion.py` (L4-style omni), `tests/e2e/online_serving/test_qwen_image_edit_expansion.py` / `test_qwen_image_expansion.py` (L4-style diffusion).

### Step 4: Generate Test Case Skeleton

**1. Pick the functional scenario** (then choose directory, fixtures, and markers):

| Scenario | Typical location | Fixtures / runner pattern | Baseline markers & level |
|----------|------------------|---------------------------|---------------------------|
| **Offline inference e2e** | `tests/e2e/offline_inference/` | **Module (default):** `omni_runner` + `omni_runner_handler`. **Function (isolation only):** `omni_runner_function` + `omni_runner_handler_function`. Diffusion/TTS may use `Omni(...).generate` directly | L2: `core_model` + **one of** `omni` / `tts` / `diffusion`; `@hardware_test(...)` when GPU/NPU is required |
| **Online serving e2e** | `tests/e2e/online_serving/` | **Module (default):** `omni_server` + `openai_client`. **Function (isolation only):** `omni_server_function` + `openai_client_function`. Clients: `send_omni_request` (omni), `send_audio_speech_request` (tts), `send_diffusion_request` / `send_video_diffusion_request` / `send_images_generations_request` (diffusion) | Baseline smoke: **`core_model` + `advanced_model`**; heavier paths: `advanced_model` only; L4 expansion: `full_model` |
| **Documentation / runnable examples** | `tests/examples/offline_inference/`, `tests/examples/online_serving/` | **Offline docs (preferred):** extract Python/Bash blocks from the doc README (e.g. `ReadmeSnippet.extract_readme_snippets`), `pytest.mark.parametrize` each snippet, run via `example_runner.run` with a stable `output_subfolder`. **Online docs:** copy client/request scripts into dedicated tests and keep them in sync with the doc page. | Usually **L4**: `advanced_model`, often `example` plus hardware marks matching the nightly docs-example job (see `.buildkite/test-nightly.yml`). Full conventions: [docs/contributing/ci/test_examples/l4_doc_example_tests.inc.md](../../../docs/contributing/ci/test_examples/l4_doc_example_tests.inc.md) (introduced in [PR #1910](https://github.com/vllm-project/vllm-omni/pull/1910): naming, output directory layout, skip rules, avoid trimming `num_inference_steps` without a strong CI reason). |
| **Performance / benchmark** | `tests/dfx/perf/tests/*.json` + `run_*_benchmark.py` | JSON or script-driven server + load config; assert explicit metrics / baselines | L4 Perf: `full_model` + `benchmark`; wire `test-nightly.yml` Perf steps |
| **Invalid parameter / negative HTTP validation** | `tests/dfx/reliability/invalid_param_test/` | Live `omni_server` + low-level `send_*_http_request` with `err_code` / `err_message` | `pytest.mark.slow` + `omni` / `tts` / `diffusion` + `@hardware_marks` (`H100` or `L4`); CI in **`test-weekly.yml`** (not ready/merge/nightly) |

**If the user’s test plan includes invalid parameter validation / invalid params / negative HTTP / 400 validation:** do **not** add those `test_*` functions to `tests/e2e/online_serving/test_*.py` or `*_expansion.py`. **Move or author them** under `tests/dfx/reliability/invalid_param_test/` in the **endpoint-matching script** (see **Invalid parameter validation** below). Success-path e2e and invalid-param dfx tests must stay in separate modules.

**1b. Model type — `omni` vs `tts` vs `diffusion`**

After choosing offline/online/docs/perf, classify the **product under test** and attach **exactly one** model-type marker. All three can live under the same `tests/e2e/` trees; conventions diverge:

| Dimension | **Omni** (`pytest.mark.omni`) | **TTS** (`pytest.mark.tts`) | **Diffusion** (`pytest.mark.diffusion`) |
|-----------|------------------------------|----------------------------|----------------------------------------|
| **What it is** | Multimodal LLM pipeline (thinker/talker/stages; text + vision + audio I/O) | Speech synthesis / voice models | Generative diffusion (noise → image, audio, text, or video) |
| **Examples** | Qwen2.5-Omni, Qwen3-Omni | Qwen3-TTS, VoxCPM2, Higgs-Audio, Step-Audio2 | Qwen-Image, BAGEL, Wan2.2, HunyuanVideo |
| **Offline runner** | `omni_runner` + `omni_runner_handler`, `generate_multimodal` | `Omni(...).generate` or stage YAML + TTS params | `Omni(...).generate` + `OmniDiffusionSamplingParams` |
| **Online client** | `openai_client.send_omni_request` (chat completions, modalities) | `openai_client.send_audio_speech_request` (`/v1/audio/speech`) | `send_diffusion_request` (chat/T2I), `send_video_diffusion_request` (`/v1/videos`, X2V), or `send_images_generations_http_request` / `send_images_edits_http_request` (DALL-E routes) |
| **Typical assertions** | Stage outputs, text/audio keywords via `OmniRunnerHandler` / response handler | WAV bytes, stream chunks, speech endpoint contract | Image/video dimensions, `final_output_type`, `assert_diffusion_response` |
| **Stage / deploy YAML** | Per-model omni stage configs (`ci/qwen3_omni_moe.yaml`, …) | `qwen3_tts.yaml`, `voxcpm2.yaml`, … | Often default serve; parallel/offload YAML for heavy DiT |
| **Nightly group** (`test-nightly.yml`) | **Omni Model Test** — `-m "full_model and omni"` | **TTS Model Test** — `-m "full_model and tts"` | **Diffusion X2I(&A&T)** *or* **Diffusion X2V** (see Step 3 table; same `diffusion` marker, different YAML group / file shard) |
| **L4 pressure** | Expansion per modality/model as needed | Expansion + accuracy/perf in TTS group | X2I: merge feature combos per [#1832](https://github.com/vllm-project/vllm-omni/issues/1832); X2V: separate nightly group |

Do **not** mix fixtures across types (e.g. do not use `omni_runner` layout for a pure diffusion or TTS model without mirroring an in-tree test in that family).

**Diffusion only — X2I(&A&T) vs X2V (nightly routing, not extra markers):**

- **X2I(&A&T)**: image / audio / text generation — Qwen-Image*, FLUX, SD3, Z-Image, BAGEL (expansion), LongCat, audio diffusion, etc.
- **X2V**: video generation only — Wan2.2, HunyuanVideo 1.5, Wan VACE, LTX video similarity paths.

When adding a new `test_*_expansion.py`, place it in the matching **nightly group** step (explicit file list in `test-nightly.yml`), not only by marker expression.

**2. Use the narrowest deterministic skeleton for the scenario**

*L1 unit / logic (CPU-first):*

**Mocking rule (L1 only):** use **pytest** integration — **`mocker`** (`pytest-mock`) or **`monkeypatch`** (built-in). **Do not** import or call **`unittest.mock`** (`patch`, `MagicMock`, `@patch`, `with patch(...)`, etc.) in L1 tests; patches must auto-revert with the test lifecycle.

```python
import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_<scenario_name>(mocker):
    # Prefer mocker.patch / mocker.spy / mocker.Mock — not unittest.mock.patch
    fake_fn = mocker.patch("vllm_omni.some.module.expensive_call", return_value=...)
    # Act
    # Assert
    fake_fn.assert_called_once()


def test_<env_or_attr>(monkeypatch):
    # For simple env / attribute substitution without a Mock object
    monkeypatch.setenv("SOME_FLAG", "1")
    monkeypatch.setattr("vllm_omni.some.module.CONST", 42)
    ...
```

**Avoid in L1:**

```python
# BAD — do not use in L1 unit tests
from unittest.mock import patch, MagicMock

@patch("vllm_omni.some.module.fn")
def test_bad(mock_fn): ...

def test_also_bad():
    with patch("...") as m: ...
```

See **L1 unit test constraints (mocking)** below for the full do/don't list.

*Offline multimodal e2e — **Omni** (representative):*

```python
@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(...)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_<scenario>(omni_runner, omni_runner_handler) -> None:
    request_config = {"prompts": ..., "modalities": [...]}  # optional: images, videos, audios
    omni_runner_handler.send_omni_request(request_config)
```

*Offline generative e2e — **Diffusion** (representative):*

```python
@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(...)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_text_to_image_001(omni_runner_handler) -> None:
    omni_runner_handler.send_diffusion_request({"prompt": "...", "extra_body": {"num_inference_steps": 4, ...}})
```

*Offline TTS e2e — **Qwen3-TTS** (two-stage; representative):*

```python
@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(...)
@pytest.mark.parametrize("omni_runner", tts_server_params, indirect=True)
def test_text_to_audio_001(omni_runner, omni_runner_handler) -> None:
    omni_runner_handler.send_audio_speech_request({
        "input": "...",
        "task_type": "Base",
        "ref_audio": REF_AUDIO_URL,
        "ref_text": REF_TEXT,
    })
```

*Offline TTS e2e — **single-stage** (Coqui XTTS, MOSS-TTS-Nano; representative):*

```python
@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(...)
@pytest.mark.parametrize("omni_runner", tts_server_params, indirect=True)
def test_voice_clone_001(omni_runner_handler) -> None:
    omni_runner_handler.send_single_stage_tts_request({
        "input": "...",
        "language": "en",
        "prompt_audio_path": REF_AUDIO_PATH,
        "response_format": "wav",
        "run_level": "advanced_model",
    })
```

*Online serving e2e — **Omni** (representative):*

```python
@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(...)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_text_to_text_001(omni_server, openai_client) -> None:
    request_config = {"model": omni_server.model, "messages": ..., "modalities": ["text"]}
    openai_client.send_omni_request(request_config)
```

*Online serving e2e — **TTS** (representative):*

```python
@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(...)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_text_to_audio_001(omni_server, openai_client) -> None:
    request_config = {"model": omni_server.model, "input": "...", "response_format": "wav", ...}
    openai_client.send_audio_speech_request(request_config)
```

*Online serving e2e — **Diffusion** X2I (representative):*

```python
@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", _get_default_case(MODEL), indirect=True)
def test_text_to_image_001(omni_server, openai_client) -> None:
    openai_client.send_diffusion_request({...})  # chat completions + extra_body
```

*Online serving e2e — **Diffusion** X2V (representative):*

```python
@pytest.mark.core_model
@pytest.mark.diffusion
def test_text_to_video_001(omni_server, openai_client) -> None:
    openai_client.send_video_diffusion_request({"model": ..., "form_data": {...}})  # /v1/videos
```

*Documentation example tests:* follow the **Preferred Test Strategy** in [l4_doc_example_tests.inc.md](../../../docs/contributing/ci/test_examples/l4_doc_example_tests.inc.md): dynamic extraction for offline READMEs; explicit copied client code for online pages until extraction is justified; use the documented **naming**, **output directory** (page folder + case id), and **skipping** rules (e.g. Gradio-only scripts).

*Performance tests:* add or extend entries under `tests/perf/` (and JSON configs where the project uses them), with **explicit baselines** and the same marker/run-level pairing as the CI step that will execute them.

**3. Cross-cutting rules**

- Reuse existing fixtures for the chosen scenario; do not mix “online client” assumptions into offline `OmniRunner` tests without a clear reason.
- Avoid external network dependency in assertions unless the scenario is explicitly “online serving” or doc examples that require a model hub (then align with CI secrets/cache).
- Keep **one test function = one intent** (one modality combo, one endpoint contract, or one acceleration combo).
- **E2E test function layout**: **one case → one `test_<scenario>` function** with a name that states what is validated (endpoint, size/`n`, server flag, or route). **Do not** merge multiple cases into a single test that branches on `request.node.callspec.id`, `param.id`, or `if case_id == ...`. Use `@pytest.mark.parametrize("omni_server", [...], indirect=True)` per function (usually **one** `OmniServerParams` per test). A loop inside one test is OK only when it serves **that function’s single intent** (e.g. three standard sizes in `test_*_sizes_256_512_1024`).
- **Runtime fixture scope** (`tests/helpers/fixtures/runtime.py`): default **`omni_server` / `omni_runner`** (module) + matching client/handler; use **`omni_server_function` / `omni_runner_function`** only when each `test_*` must start a fresh instance (see below).
- **L1 mocks**: never `unittest.mock`; use `mocker` or `monkeypatch` only (see below).
- **API calls (L2+ e2e, online and offline)**: **reuse** `send_*_request` in `tests/helpers/runtime.py` when it exists; **otherwise add** the helper in `runtime.py` first, then call it from the test. General `assert_*` inside `send_*_request`; special `assert_*` only in the test. See **Runtime send helpers** below — do not call `omni.generate`, raw HTTP, or SDK clients from `test_*.py`.
- **Response assertions**: reusable checks on API bodies / decoded media belong in **`tests/helpers/assertions.py`** — not as private `_assert_*` helpers inside `test_*.py` (see below).
- **Model-specific payloads stay in test modules** — per-model `MODEL`, deploy path, `REF_AUDIO_URL`, `get_prompt()`, `_build_request()`, and inline `request_config` dicts live in `test_{slug}.py` / `test_{slug}_expansion.py` (and offline/L1 siblings). **Do not** create `tests/helpers/{slug}.py` to deduplicate them; a little copy across files is preferred (see `test_glm_tts.py`, `test_cosyvoice3_tts_expansion.py`). `tests/helpers/` is repo-wide harness only (`mark`, `media`, `runtime`, `stage_config`, `assertions`, `fixtures/`).

#### L1 unit test constraints (mocking)

L1 tests (`core_model and cpu`, under `tests/diffusion/`, `tests/engine/`, `tests/model_executor/`, etc.) must follow the repo’s **pytest-mock** convention (`pytest-mock>=3.10.0` in `pyproject.toml` `[project.optional-dependencies] dev`).

| Do | Don't |
|----|-------|
| `def test_foo(mocker):` + `mocker.patch(...)`, `mocker.spy(...)`, `mocker.Mock()`, `mocker.MagicMock()`, `mocker.AsyncMock()` | `from unittest.mock import patch, MagicMock, Mock, AsyncMock` |
| `def test_bar(monkeypatch):` + `monkeypatch.setattr` / `setenv` / `delenv` / `setitem` | `@patch(...)` decorator |
| Let `mocker` auto-stop patches after the test | `with patch(...):` / `patch.object(...)` context managers |
| Mirror neighboring L1 tests in the same directory | `unittest.mock.create_autospec` unless an existing file already documents an exception |

**Rationale:** `mocker` ties patch lifecycle to pytest fixtures (no leaked patches across tests). `unittest.mock` decorators/context managers are easy to compose incorrectly with parametrized or async tests and are inconsistent with in-tree L1 style.

**Minimal patterns:**

```python
def test_returns_cached_config(mocker):
    loader = mocker.patch(
        "vllm_omni.foo.load_yaml",
        return_value={"stages": []},
    )
    result = get_deploy_config("ci/foo.yaml")
    assert result["stages"] == []
    loader.assert_called_once_with("ci/foo.yaml")


def test_skips_when_env_unset(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_FEATURE", raising=False)
    assert should_enable_feature() is False
```

E2E levels (L2+) generally **avoid mocks**; if a rare L2 stub is unavoidable, still prefer `mocker` over `unittest.mock` for consistency.

#### Runtime fixtures — scope (`tests/helpers/fixtures/runtime.py`)

vLLM-Omni e2e tests start a real **OmniServer** (online) or **OmniRunner** (offline). Pick scope by **how often the process must be recreated**, not by test level alone.

| Scope | Online fixtures | Offline fixtures | When to use |
|-------|-----------------|------------------|-------------|
| **Module** (default) | `omni_server` → `openai_client` | `omni_runner` → `omni_runner_handler` | **Default for L2/L3/L4 expansion** — amortize model/server init across `test_*` in the same module. Same `OmniServerParams` / runner config can be reused by multiple tests. |
| **Function** | `omni_server_function` → `openai_client_function` | `omni_runner_function` → `omni_runner_handler_function` | **Only when required** — each `test_*` must get a **clean** server/runner (no shared engine/GPU state). Typical: `tests/dfx/reliability/`, sleep/wakeup, crash/restart, tests that mutate global server state. |

**Rules:**

1. **Default to module scope** (`omni_server` / `omni_runner`) unless the scenario **explicitly** needs a fresh instance per test function.
2. **Indirect parametrize name must match the fixture name**: `@pytest.mark.parametrize("omni_server", ...)` with `omni_server` + `openai_client`; `@pytest.mark.parametrize("omni_server_function", ...)` with `omni_server_function` + `openai_client_function`. Do not mix module fixture with function client (or vice versa).
3. Different `OmniServerParams` per `test_*` (e.g. default vs `--enable-cpu-offload`) is still OK with **module** `omni_server` — pytest parametrizes per test node; only switch to `_function` when isolation between tests matters, not merely because `server_args` differ.
4. One `test_*` with **many** server configs (expansion matrix) → single function + `@pytest.mark.parametrize("omni_server", [...], indirect=True)` + module `omni_server` (see in-tree `test_qwen_image_expansion.py`).

```python
# Default — module-scoped server (L4 expansion)
@pytest.mark.parametrize("omni_server", [pytest.param(OmniServerParams(model=MODEL), marks=H100)], indirect=True)
def test_foo_images_generations_default_1024(omni_server, openai_client) -> None:
    openai_client.send_images_generations_request({...})

# Function-scoped — reliability / per-test clean state only
@pytest.mark.parametrize(
    "omni_server_function",
    [pytest.param(OmniServerParams(model=MODEL), marks=H100)],
    indirect=True,
)
def test_foo_sleep_wakeup_cycle(omni_server_function, openai_client_function) -> None:
    openai_client_function.send_omni_sleep_http_request({...})
    openai_client_function.send_omni_wakeup_http_request({...})
```

#### Runtime send helpers — online **and** offline (`tests/helpers/runtime.py`)

**L2+ e2e (online serving and offline inference) must call APIs through `tests/helpers/runtime.py`.** Fixtures live in `tests/helpers/fixtures/runtime.py`; **send/assert implementation** lives in `tests/helpers/runtime.py` + `tests/helpers/assertions.py`.

| Principle | Action |
|-----------|--------|
| **Reuse first** | Grep `runtime.py` for an existing **`send_*_request`** on `OpenAIClientHandler` (online) or `OmniRunnerHandler` (offline) that matches the endpoint / pipeline shape. |
| **Extend when close** | If an existing `send_*_request` almost fits (e.g. missing one optional field), extend it in `runtime.py` — do not fork logic in the test file. |
| **Add when missing** | No suitable helper → add **`send_<feature>_request`** (high-level: call + general `assert_*`) or **`send_<route>_<verb>_http_request`** (low-level HTTP for negative/dfx) in **`runtime.py` first**, then call it from tests. |
| **Test module owns payload only** | In `test_*.py`: `MODEL`, deploy path, vendored media, `get_prompt()`, and inline **`request_config` dicts** only. **No** `omni.generate(...)`, raw `requests.post`, OpenAI SDK calls, or `_collect_audio()` / `_process_output()` in e2e tests. |

**Online** (`openai_client` from `omni_server`): `OpenAIClientHandler.send_*_request`.

**Offline** (`omni_runner_handler` from `omni_runner`): `OmniRunnerHandler.send_*_request`.

| Do | Don't |
|----|-------|
| Online: `openai_client.send_omni_request`, `send_diffusion_request`, `send_audio_speech_request`, `send_video_diffusion_request`, `send_images_generations_request`, … | `requests.post(f"{base_url}/v1/...", json=…)` or `client.chat.completions.create(...)` inside a test |
| Offline: `omni_runner_handler.send_omni_request`, `send_diffusion_request`, `send_audio_speech_request`, `send_single_stage_tts_request`, `send_single_stage_tts_batch_request`, … | `omni_runner.omni.generate(...)` + hand-rolled tensor/WAV extraction in `test_*.py` |
| Add missing `send_*` to **`runtime.py`** first; bundle general `assert_*` inside the send helper | A one-off `def _post_*` / `def _collect_audio` at the bottom of a test module |
| Mirror naming/style of neighboring `send_*` (docstring, `request_config` dict, optional `run_level`, `err_code` / `err_message` for negative cases) | Different parameter shapes per test file for the same endpoint |

**Workflow when generating online/offline e2e tests:**

1. Decide whether the needed check is **general** (every success call of this `send_*`) or **special** (one case / one parameter combo only). See **General vs special assert placement** below.
2. **Search `runtime.py`** for a matching **`send_*_request`**; if missing, add it (with general `assert_*` inside) before writing the test body.
3. In the test module: build `request_config` → call **`send_*_request` only** for the general contract; call **`assert_*` in the test file only** for special, case-specific checks (import from `assertions.py`).
4. Reserve **low-level** `send_*_http_request` for negative/dfx tests (`err_code` / `err_message`) — not for ordinary L2+ success-path e2e.
5. When wiring Buildkite `source_file_dependencies`, include **`tests/helpers/runtime.py`** and/or **`tests/helpers/assertions.py`** when new helpers were added.

**Exceptions (document in the test docstring why):** models whose offline prompt path cannot go through existing handlers yet (e.g. `test_higgs_audio_v2.py`, `test_voxtral_tts.py` with custom tokenizer compose) may call `omni.generate` directly until a `send_*_request` is added to `runtime.py` — treat as **debt**, not the default for new TTS/diffusion/omni e2e.

**General vs special assert placement:**

| Kind | Definition | Where it lives | Test file calls |
|------|------------|----------------|-----------------|
| **General** | Default success contract for this endpoint/client method — e.g. HTTP 200, `data[]` shape, `n` count, `size` dimensions, decodable image/audio, bundled omni/diffusion fields | Implement in `assertions.py`, invoke **inside** `send_*_request` in `runtime.py` (after low-level send / SDK call, when `err_code` is not set) | **`send_*_request` only** — do **not** repeat the same `assert_*` |
| **Special** | Extra check tied to **this test case only** — e.g. seed byte-identical replay, accuracy/CLIP threshold, perf ceiling, model-specific optional field | Add or reuse `assert_*` in `assertions.py` (never inline in test) | `send_*_request` **then** `assert_<special>(...)` **once** for that case |

Changing `request_config` fields (`size`, `n`, `seed`, server flags) is **not** special validation — the general `assert_*` should read those from `request_config` inside `send_*`.

**`send_*` ↔ `assert_*` pairing (online `OpenAIClientHandler`):**

| High-level `send_*` (prefer in L2+ e2e) | Assert already invoked inside `runtime.py` | Low-level HTTP-only sibling |
|----------------------------------------|--------------------------------------------|-----------------------------|
| `send_omni_request` | `assert_omni_response` | `send_chat_completions_http_request` → `assert_http_error` only |
| `send_diffusion_request` | `assert_diffusion_response` | — |
| `send_audio_speech_request` | `assert_audio_speech_response` | `send_audio_speech_http_request` → `assert_http_error` only |
| `send_video_diffusion_request` | `assert_diffusion_response` | `send_videos_*_http_request` → `assert_http_error` only |
| `send_images_generations_request` *(add when needed)* | **`assert_images_generations_response`** *(general — inside send)* | `send_images_generations_http_request` → `assert_http_error` only |
| `send_images_edits_request` *(add when needed)* | **`assert_images_edits_response`** *(general — inside send)* | `send_images_edits_http_request` → `assert_http_error` only |

**Rule:** Tests call high-level `send_*_request` for the general contract. **Never** call the same bundled `assert_*` again. Call an extra `assert_*` in the test **only** for special, case-specific validation.

**Common `OpenAIClientHandler` entry points** (non-exhaustive — grep `runtime.py` before adding):

| Area | High-level (SDK + assert) | Low-level HTTP (`*_http_request`) |
|------|---------------------------|-----------------------------------|
| Omni chat | `send_omni_request` | `send_chat_completions_http_request` |
| Diffusion T2I (chat route) | `send_diffusion_request` | — |
| Diffusion X2V | `send_video_diffusion_request` | `send_videos_create_http_request`, `send_video_content_http_request`, … |
| DALL-E T2I / edit | `send_images_generations_request`, `send_images_edits_request` | `send_images_generations_http_request`, `send_images_edits_http_request` |
| TTS (online) | `send_audio_speech_request` | `send_audio_speech_http_request`, `send_audio_generate_http_request`, … |
| Ops / meta | — | `send_health_http_request`, `send_models_http_request`, `send_omni_sleep_http_request`, … |

**Common `OmniRunnerHandler` entry points** (offline — grep `runtime.py` before adding):

| Area | High-level (`send_*_request` + assert) | Notes |
|------|----------------------------------------|-------|
| Omni multimodal | `send_omni_request` | `generate_multimodal` path |
| Diffusion offline | `send_diffusion_request` | chat-route / `OmniTextPrompt` offline |
| Qwen-style TTS | `send_audio_speech_request` | two-stage, `generate_multimodal` + `mm_processor_kwargs` |
| Single-stage TTS (Coqui XTTS, MOSS-TTS-Nano, …) | `send_single_stage_tts_request`, `send_single_stage_tts_batch_request` | `prompt` + `additional_information` + `omni.generate` — **do not** duplicate in tests |
| New model family | **Add** `send_<family>_request` here first | Then call from `tests/e2e/offline_inference/test_*.py` |

L1 tests under `tests/entrypoints/` may use **FastAPI `TestClient`** or direct handler calls with mocks; they do **not** need `OpenAIClientHandler` / `OmniRunnerHandler`. **L2+ online and offline e2e** must use `runtime.py` `send_*` helpers.

#### Invalid parameter validation (`tests/dfx/reliability/invalid_param_test/`)

When the user asks for **invalid parameter validation**, **invalid request bodies**, **HTTP 4xx contract tests**, or any case that sends **malformed / out-of-range / mismatched** API payloads against a **live** server:

1. **Do not** place these in `tests/e2e/online_serving/` or `*_expansion.py`. If already drafted there, **move** the `test_*` into the correct `invalid_param_test` script and delete the duplicate from e2e.
2. **Pick the script by HTTP route** (extend an existing file; add a new `test_invalid_<area>.py` only when no in-tree script covers that route family):

| API / area | Script |
|------------|--------|
| Omni chat completions, WebSocket video/realtime paths | `test_invalid_omni_chat.py` |
| `POST /v1/audio/speech`, stream, batch, voices | `test_invalid_audio_speech.py` |
| Audio diffusion endpoints | `test_invalid_audio_diffusion.py` |
| `POST /v1/images/generations` | `test_invalid_image_generation.py` |
| `POST /v1/images/edits` | `test_invalid_image_editing.py` |
| `POST/GET/DELETE /v1/videos*` | `test_invalid_video_generation.py` |
| Sleep / wakeup / server control | `test_invalid_server_control.py` |

3. **Match in-tree style** in the chosen script:

| Element | Convention |
|---------|------------|
| **Module markers** | `pytestmark = [pytest.mark.slow, pytest.mark.<omni or tts or diffusion>]` |
| **Server fixture** | `_PARAMS` / `_QWEN3_TTS_SPEECH`-style list of `pytest.param(OmniServerParams(...), id="...", marks=hardware_marks(...))`; `@pytest.mark.parametrize("omni_server", _PARAMS, indirect=True)` |
| **Hardware** | `hardware_marks(res={"cuda": "H100"})` for heavy diffusion/omni/video; `hardware_marks(res={"cuda": "L4"})` for smaller TTS models (must match weekly `-m "slow and L4"` step) |
| **HTTP client** | **Low-level** `openai_client.send_*_http_request({..., "err_code": 400, "err_message": (...)} )` — **not** `send_*_request` (success path) |
| **Case shape** | Prefer **one `test_*` per route family** + `@pytest.mark.parametrize("body_spec, err_message", [...])` with stable `id=` per case; or dedicated `test_<route>_malformed_json` when not parametrized |
| **Body helpers** | `_minimal_<endpoint>_json(omni_server)` / `_minimal_*_form_data()`; merge overrides with `body.update(body_spec)` |
| **Known gaps** | `pytest.mark.skip(reason="…#3649")` as `_SKIP_ISSUE_3649` when server validation is not yet strict (mirror neighboring cases) |
| **Sections** | Route banner comments (`# ─── POST /v1/images/generations ───`) like existing files |
| **Shared fixtures** | Reuse `tests/dfx/reliability/invalid_param_test/conftest.py` (`tiny_png_bytes`, env defaults) |

4. **Adding a new model** to invalid-param coverage: append a `pytest.param(OmniServerParams(model="...", stage_config_path=..., server_args=...), ...)` entry to the script’s `_PARAMS` list — do **not** create `test_invalid_<model>.py` unless the route is new.

5. **CI — `.buildkite/test-weekly.yml` only** (not `test-ready.yml` / `test-merge.yml` / `test-nightly.yml`):

| Weekly step | Command | When your cases run |
|-------------|---------|---------------------|
| **Invalid parameters Test · H100** | `pytest -s -v tests/dfx/reliability/invalid_param_test/ -m "slow and H100"` | Diffusion / omni / video invalid-param tests with `H100` hardware mark |
| **Invalid parameters Test · L4** | `pytest -s -v tests/dfx/reliability/invalid_param_test/ -m "slow and L4"` | TTS / lighter models marked `L4` |

- **Trigger:** `build.env("WEEKLY") == "1"` or PR label `weekly-test`.
- **Default:** extending an existing `invalid_param_test` script needs **no YAML edit** — the weekly steps already sweep the whole directory.
- **Edit YAML only** when adding a **new hardware queue**, a **new top-level script** that must run in isolation, or a **model-specific weekly shard** (mirror neighboring reliability steps).
- Weekly steps **do not** use `source_file_dependencies`.

**Example — append to `test_invalid_image_generation.py`:**

```python
@pytest.mark.parametrize(
    "body_spec, err_message",
    [
        pytest.param({"seed": -1}, ("seed", "greater_than_equal", "0"), id="seed_negative"),
    ],
)
@pytest.mark.parametrize("omni_server", _PARAMS, indirect=True)
def test_images_generations_invalid_requests(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
    body_spec: dict[str, object],
    err_message: str | tuple[str, ...],
) -> None:
    body = _minimal_images_gen_json(omni_server)
    body.update(body_spec)
    openai_client.send_images_generations_http_request(
        {"json": body, "timeout": 300, "err_code": 400, "err_message": err_message}
    )
```

See [references/test-routing.md](references/test-routing.md) **Invalid parameter / weekly CI**.

#### Assertion helpers (`tests/helpers/assertions.py`)

**Do not** add module-local helpers such as `_assert_images_generations_payload` or `_send_and_assert_*` in e2e test files. Response/media validation belongs in **`tests/helpers/assertions.py`**, grouped by **category**:

| Category | Existing anchors | When to extend vs add |
|----------|------------------|------------------------|
| **Image** (chat diffusion + DALL-E JSON) | `assert_image_diffusion_response`, `assert_image_valid` | DALL-E `/v1/images/generations` JSON → add **`assert_images_generations_response`** beside image helpers (reuses `assert_image_valid`); do not fork decode logic into tests |
| **Video** | `assert_video_diffusion_response`, `assert_video_valid` | Extend these for new video contracts |
| **Audio** | `assert_audio_diffusion_response`, `assert_audio_speech_response`, `assert_audio_valid` | Extend for new TTS/audio endpoints |
| **Omni multimodal** | `assert_omni_response` | Extend for new modality combos |
| **HTTP errors** | `assert_http_error`, `assert_err_message_in_text` | Used inside low-level `send_*_http_request` and negative tests |

| Do | Don't |
|----|-------|
| Put **general** `assert_*` inside `send_*_request` in `runtime.py`; tests call `send_*_request` only | Call `assert_diffusion_response` after `send_diffusion_request` (general assert already bundled) |
| Put **special** `assert_*` in `assertions.py` and call it **in the test** after `send_*_request` when that case needs extra checks | Put general decode/count/size logic in the test because “this case uses `n=4`” (that belongs in general assert reading `request_config`) |
| Extend the matching category function, or add `assert_<endpoint>_response` next to its category | `_send_and_assert_*` or PIL/base64 loops in `test_*.py` |
| Use low-level `send_*_http_request` in dfx/negative tests (`err_code`) | Use low-level HTTP send + manual general assert in every L2+ expansion file |

**Workflow when generating tests:**

1. Implement or reuse **general** `assert_*` in `assertions.py` (by category).
2. Wire it into **`send_*_request`** in `runtime.py` if not already bundled.
3. In the test: `request_config` → `send_*_request`. Add a separate `assert_*` import/call **only** for special case validation.
4. Update `source_file_dependencies` when `runtime.py` or `assertions.py` changes.

**Example — DALL-E `/v1/images/generations`:**

```python
# tests/helpers/assertions.py — general contract (image category)
def assert_images_generations_response(resp_body: dict, request_config: dict, *, run_level: str | None = None) -> None:
    ...  # data[], n, size → assert_image_valid

# tests/helpers/runtime.py — general assert INSIDE send
def send_images_generations_request(self, request_config: dict[str, Any], ...) -> list[HttpResponse]:
    responses = self.send_images_generations_http_request(request_config, ...)
    if request_config.get("err_code") is None:
        body = responses[0].json_body
        assert isinstance(body, dict)
        assert_images_generations_response(body, request_config, run_level=self.run_level)
    return responses

# tests/e2e/.../test_foo_expansion.py — one case per test_* (no case_id branching)
@pytest.mark.parametrize("omni_server", [pytest.param(OmniServerParams(model=MODEL), marks=SINGLE_L4)], indirect=True)
def test_foo_images_generations_default_1024(omni_server, openai_client) -> None:
    openai_client.send_images_generations_request({"json": body, "timeout": 300})

@pytest.mark.parametrize("omni_server", [pytest.param(OmniServerParams(model=MODEL), marks=SINGLE_L4)], indirect=True)
def test_foo_images_generations_sizes_256_512_1024(omni_server, openai_client) -> None:
    for size in ("256x256", "512x512", "1024x1024"):
        openai_client.send_images_generations_request({"json": {**body, "size": size}, "timeout": 300})

# chat diffusion — separate test; send_diffusion_request bundles assert_diffusion_response
@pytest.mark.parametrize("omni_server", [pytest.param(OmniServerParams(model=MODEL), marks=SINGLE_L4)], indirect=True)
def test_foo_chat_completions_t2i_fallback(omni_server, openai_client) -> None:
    openai_client.send_diffusion_request({...})
```

L1 tests under `tests/entrypoints/` may keep **minimal** asserts next to the handler under test when validating a single branch; still prefer `assertions.py` when the same JSON/media check appears in more than one file or level (e.g. L1 protocol test + L4 e2e).

#### Omni Test Writing Guidance (L1-L4 Layering)

When the goal is a general Omni/multimodal test case, prioritize mapping the test to the correct purpose, directory, and resource assumptions aligned with the layers (see [CI_5levels.md](../../../docs/contributing/ci/CI_5levels.md)):

- **L1**: Unit/logic validation on CPU (`core_model and cpu`). Cover input validation, branches, and exception paths (`tests/<component>/test_*.py`). **Mock with `mocker` / `monkeypatch` only — not `unittest.mock`.**
- **L2**: Basic e2e (online/offline basic scenarios). Prefer dummy/lightweight models to validate the end-to-end request-to-output-structure/streaming chain (typically `tests/e2e/online_serving/` and `tests/e2e/offline_inference/`).
- **L3/L4**: Important integration, performance, and accuracy validation. L4 emphasizes “full functional scenarios + performance/stress + runnable doc examples” (typically `*_expansion.py` plus related expansion cases).

Also keep markers consistent with the run level: use `core_model` for L1/L2 and `advanced_model` for L3/L4, and pair with `--run-level` to select the intended CI strategy.

#### Diffusion Test Writing Guidance (L4 Coverage Combinations)

When the task involves diffusion models/features, organize L4 test cases following [`#1832`](https://github.com/vllm-project/vllm-omni/issues/1832): combine multiple diffusion features into as few test cases as possible to fit limited CI GPU resources.

Implementation strategy:

1. If “full L4 is too heavy”: first provide a **reduced local validation case in L1/L2** (so the key assertion and the contract fix point are covered deterministically).
2. Then provide **CI/nightly-ready L3/L4 high-marked cases** (e.g. `advanced_model`) to broaden coverage under resource constraints.

#### L4 nightly sub-pillars — Function vs Accuracy vs Perf

**L4 is not only `test_*_expansion.py`.** In `test-nightly.yml`, each model-type group typically runs **separate jobs**:

| Sub-pillar | Nightly step label pattern | What you add | Typical paths |
|------------|----------------------------|--------------|---------------|
| **Function** | `· Function Test with …` | E2e expansion / feature matrix | `tests/e2e/online_serving/test_<model>_expansion.py` (or offline) |
| **Accuracy** | `· Accuracy Test` | Quality / similarity vs baseline | `tests/e2e/accuracy/test_<model>*.py` |
| **Perf** | `· Perf Test · <Model>` | Throughput / latency / memory benchmarks | `tests/dfx/perf/tests/test_<model>_vllm_omni.json` + runner script |
| **Doc** (optional) | `· Doc Test` | Runnable doc examples | `tests/examples/*/test_text_to_image.py`, … |

**Default when the user asks for “L4 functional cases”:** deliver **Function** pillar only (`*_expansion.py` + Function Test shard in `test-nightly.yml`). **Do not** silently add Perf or Accuracy unless the user also asks for **performance** / **benchmark** / **accuracy** / **full L4** / **full L4 coverage**.

**When the user explicitly asks for L4 perf (or full L4 including perf):**

1. **Do not** put throughput/latency assertions inside `test_*_expansion.py` — perf uses the **dfx benchmark harness**, not `omni_server` e2e fixtures.
2. Add or extend a **JSON config** under `tests/dfx/perf/tests/` (mirror in-tree names: `test_qwen_image_vllm_omni.json`, `test_bagel_vllm_omni.json`).
3. Each JSON entry: `test_name`, `server_params` (`model`, `serve_args`), `benchmark_params[]` with explicit **`baseline`** (`throughput_qps`, `latency_mean`, `peak_memory_mb_mean`, …).
4. Run locally via the matching script (model type → runner):

| Model type | Runner script | Example config |
|------------|---------------|----------------|
| **Diffusion X2I/X2V** | `tests/dfx/perf/scripts/run_diffusion_benchmark.py` | `tests/dfx/perf/tests/test_qwen_image_vllm_omni.json` |
| **TTS** | `tests/dfx/perf/scripts/run_benchmark.py` | `tests/dfx/perf/tests/test_tts.json` (shared) or `test_{slug}.json` (dedicated; use when the model must not join the shared nightly matrix before integration — e.g. `test_voxcpm2.json`, `test_coqui_tts.json`) |
| **Omni** | (see existing Omni Perf steps — `run_benchmark.py` / dedicated omni configs in-tree) | `tests/dfx/perf/tests/test_omni*.json` |

5. Wire **`test-nightly.yml` · Perf Test** step for that model (separate from Function Test): export `DIFFUSION_BENCHMARK_DIR` / `BENCHMARK_DIR`, run pytest on the script with `--test-config-file`, **upload artifacts** (`buildkite-agent artifact upload`), often **multi-GPU H100** for diffusion perf.
6. **One benchmark scenario → one `test_name` block** in JSON (same spirit as one `test_*` per function case). Combine server `serve_args` + workload in one entry; use multiple `benchmark_params` rows for size/step sweeps under the same server config.

**Perf local command (diffusion example):**

```bash
cd tests
export DIFFUSION_BENCHMARK_DIR=tests/dfx/perf/results
export DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN
pytest -s -v dfx/perf/scripts/run_diffusion_benchmark.py \
  --test-config-file dfx/perf/tests/test_<model>_vllm_omni.json
```

**Clarify in the test plan** which L4 pillars you deliver: `Function only` | `Function + Perf` | `Function + Accuracy + Perf`.

See [references/test-routing.md](references/test-routing.md) **L4 nightly pillars** for Buildkite step patterns.

### Step 5: Wire Buildkite (when CI must run the new test)

If the test is not already collected by an existing pipeline command (for example, L1 tests marked `core_model and cpu` are already covered by the **Simple Unit Test** step in ready/merge), update the appropriate pipeline under `.buildkite/`:

| Test level | Edit this file | Typical trigger / intent |
|------------|----------------|---------------------------|
| **L1** and **L2** | [`.buildkite/test-ready.yml`](../../../.buildkite/test-ready.yml) | PR **ready** label; L1 CPU + L2 GPU/basic e2e (steps labeled **Omni ·**, **TTS ·**, **Diffusion ·**) |
| **L3** | [`.buildkite/test-merge.yml`](../../../.buildkite/test-merge.yml) | Post-merge; `advanced_model` integration per model type |
| **L4** | [`.buildkite/test-nightly.yml`](../../../.buildkite/test-nightly.yml) | Nightly; grouped by model type (see below) |
| **Invalid param / reliability (weekly)** | [`.buildkite/test-weekly.yml`](../../../.buildkite/test-weekly.yml) | Weekly; `tests/dfx/reliability/invalid_param_test/` — **not** L1–L4 e2e pipelines |

**Level-specific Buildkite delivery (follow the user’s requested level):**

| User asks for | Document / edit **only** | Do **not** add by default |
|---------------|---------------------------|---------------------------|
| **L4** (`*_expansion.py`, `full_model`) | `test-nightly.yml` — append file to the matching nightly shard (X2I / X2V / Omni / TTS) **or** note it is already collected by an existing `-m` / `-k` sweep | `test-merge.yml` (L3 / `advanced_model`) or `test-ready.yml` (L2) |
| **L3** | `test-merge.yml` with `advanced_model` + `source_file_dependencies` | `test-nightly.yml` unless user also wants nightly |
| **L2** | `test-ready.yml` with `core_model` + `source_file_dependencies` | merge / nightly pipelines |
| **Invalid param** | `test-weekly.yml` — **Invalid parameters Test** group (`-m "slow and H100"` / `-m "slow and L4"`). Usually **no YAML edit** when appending cases to existing `invalid_param_test` scripts | `test-ready.yml`, `test-merge.yml`, `test-nightly.yml` |

`test-nightly.yml` shards use **explicit pytest file paths** in `commands` (and PR labels like `diffusion-x2iat-test`); they generally **do not** use `source_file_dependencies`. Only suggest merge/ready YAML when the requested level is L3/L2 (or the user explicitly asks for multi-level CI).

**`test-nightly.yml` top-level groups** (each uses `full_model` + type marker in pytest `-m`):

| Group | Type marker | Notes |
|-------|-------------|--------|
| **Omni Model Test** | `omni` | Function / doc / accuracy / **perf** for Qwen-Omni family |
| **TTS Model Test** | `tts` | `-m "full_model and L4 and tts"` (and H100 variants when enabled); **Perf Test** uses `run_benchmark.py` + `test_tts.json` |
| **Diffusion X2I(&A&T) Model Test** | `diffusion` | Image/audio/text diffusion; explicit file shards for **Function**; **Perf Test · &lt;Model&gt;** uses `run_diffusion_benchmark.py` + JSON config |
| **Diffusion X2V Model Test** | `diffusion` | Video-only (Wan, HunyuanVideo, …); PR label `diffusion-x2v-test`; separate **Perf** steps when in-tree |

Guidelines when editing YAML:

- **Match markers and `--run-level`** to the test: L1/L2 in `test-ready.yml` use `core_model` + `omni` / `tts` / `diffusion` as appropriate; L3 merge uses `advanced_model`; L4 nightly uses `full_model` with the same type marker.
- **Diffusion L4**: add x2i/x2a/x2t expansion files to an **X2I(&A&T)** step; add x2v expansion files to **X2V** — never rely on a broad `tests/e2e/` sweep alone (shards exist to save GPU and avoid wrong queue).
- **`source_file_dependencies` (required for new E2E jobs in `test-ready.yml` and `test-merge.yml`)**: every step under the **`:card_index_dividers: E2E Test`** group **must** declare `source_file_dependencies` so Buildkite only schedules the job when relevant paths change. Omitting it breaks path-based CI skipping for GPU e2e jobs.
- **Reuse or extend an existing step** when the new test shares the same marker expression, queue, and timeout; otherwise add a new `steps:` entry with the correct `agents.queue`, `timeout_in_minutes`, and docker/kubernetes plugin blocks consistent with neighboring jobs.
- **Job timeout — use `timeout_in_minutes` on the step, not `timeout` before pytest**: when adding or documenting a new E2E job in `test-ready.yml` / `test-merge.yml`, set `timeout_in_minutes` on the step and run `pytest` directly in `commands`. **Do not** wrap pytest in `timeout 40m bash -c "..."` / `timeout 20m bash -c "..."`; Buildkite already enforces the step deadline via `timeout_in_minutes`.
- **Never omit `plugins` in skill examples or PR YAML snippets**: H100 E2E steps on `mithril-h100-pool` use the full `kubernetes` block (`resources`, `volumeMounts`, `env` with `HF_TOKEN` secret, `nodeSelector: gpu-h100-sxm`, `devshm` + `hf-cache` volumes). L4/docker steps on `gpu_1_queue` use the full `docker#v5.2.0` block (`shm-size`, `HF_HOME`, volumes). Do not write `# ... plugins unchanged` — paste the complete block from a sibling step.
- **Platform forks** (e.g. AMD-ready / AMD-merge) live alongside these files; apply the same level → file mapping for those pipelines when the test is platform-specific.

#### `source_file_dependencies` for E2E Test jobs (`test-ready.yml`, `test-merge.yml`)

When adding or splitting a step inside the **E2E Test** group, **always** include `source_file_dependencies` on that step. List paths the job actually depends on — at minimum:

1. **Test module(s)** — every `tests/e2e/...` file the `pytest` command runs (online and offline if both are in one step).
2. **Model implementation** — `vllm_omni/model_executor/models/<family>/` and/or `vllm_omni/diffusion/models/<family>/` as applicable.
3. **Stage / input plumbing** (omni & multi-stage TTS) — `vllm_omni/model_executor/stage_input_processors/<family>.py` when present.
4. **Deploy / stage YAML** — `vllm_omni/deploy/<config>.yaml` or `vllm_omni/deploy/ci/<config>.yaml` referenced by the test’s `OmniServerParams` / `get_deploy_config_path(...)`.
5. **Test helpers** (when changed in the same PR) — `tests/helpers/runtime.py` for new `send_*` client methods; `tests/helpers/assertions.py` for new shared `assert_*` helpers.

Use **repo-relative paths** (no leading `./`). Prefer **directory** entries (`.../models/bagel/`) when the whole package matters; use **file** entries for single test modules and YAML configs. Mirror a neighboring step for the same model family (e.g. **Diffusion · Bagel Test** in `test-ready.yml`).

**Example** (L2 ready — diffusion online smoke, H100 kubernetes):

```yaml
- label: "Diffusion · Bagel Test"
  source_file_dependencies:
    - tests/e2e/online_serving/test_bagel.py
    - vllm_omni/model_executor/models/bagel/
    - vllm_omni/diffusion/models/bagel/
    - vllm_omni/model_executor/stage_input_processors/bagel.py
    - vllm_omni/deploy/bagel.yaml
  timeout_in_minutes: 40
  commands:
    - |
      export VLLM_IMAGE_FETCH_TIMEOUT=60
      pytest -s -v tests/e2e/online_serving/test_bagel.py -m 'core_model' --run-level 'core_model'
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

**Example** (L3 merge — TTS online + offline, L4 docker):

```yaml
- label: "TTS · Qwen3-TTS Base Test"
  source_file_dependencies:
    - tests/e2e/online_serving/test_qwen3_tts_base.py
    - tests/e2e/offline_inference/test_qwen3_tts_base.py
    - vllm_omni/model_executor/models/qwen3_tts/
    - vllm_omni/model_executor/stage_input_processors/qwen3_tts.py
    - vllm_omni/deploy/qwen3_tts.yaml
  timeout_in_minutes: 20
  commands:
    - |
      export VLLM_LOGGING_LEVEL=DEBUG
      export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
      pytest -s -v tests/e2e/online_serving/test_qwen3_tts_base.py tests/e2e/offline_inference/test_qwen3_tts_base.py -m 'advanced_model and cuda' --run-level 'advanced_model'
  agents:
    queue: "gpu_1_queue" # g6.4xlarge instance on AWS, has 1 L4 GPU
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

**Example** (L3 merge — diffusion online + offline, H100 kubernetes):

```yaml
- label: "Diffusion · Qwen Image Test"
  source_file_dependencies:
    - tests/e2e/online_serving/test_qwen_image.py
    - tests/e2e/offline_inference/test_qwen_image.py
    - vllm_omni/diffusion/models/qwen_image/
  timeout_in_minutes: 40
  commands:
    - |
      export VLLM_IMAGE_FETCH_TIMEOUT=60
      pytest -s -v \
        tests/e2e/online_serving/test_qwen_image.py \
        tests/e2e/offline_inference/test_qwen_image.py \
        -m 'advanced_model and cuda' --run-level 'advanced_model'
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

When extending an **existing** E2E step to run an additional test file, **append** that file (and any new model/deploy paths) to `source_file_dependencies` on the same step. When authoring a PR, mention the dependency list in the **Buildkite change** section of your test plan.

#### L3 / L4: validating Buildkite from a feature branch (root `pipeline.yml`)

Production behavior is driven by [`.buildkite/pipeline.yml`](../../../.buildkite/pipeline.yml): it builds the CI image, then uploads **L2** (`test-ready.yml` on non-`main`), **L3** (`test-merge.yml` when `build.branch == "main"`), and **L4** (`test-nightly.yml` when `build.env("NIGHTLY") == "1"`).

When you add or debug **L3** or **L4** tests and need those child pipelines to run **off `main`** or **without** `NIGHTLY=1`, apply **temporary** edits (revert before merge):

1. **Point the upload at the right file (optional shortcut)**  
   In the **Upload Ready Pipeline** step, the command is `buildkite-agent pipeline upload .buildkite/test-ready.yml`. For a one-off run you may change that path to `.buildkite/test-merge.yml` (L3) or `.buildkite/test-nightly.yml` (L4), and adjust the step’s `if:` so it does not conflict with the other upload steps (avoid double-uploading unless intentional).

2. **L3 — merge pipeline**  
   The gate `if: build.branch == "main"` lives on the **Upload Merge Pipeline** step in `pipeline.yml` (not inside `test-merge.yml`). **Comment out that `if` line** so `test-merge.yml` is uploaded on your feature branch. Alternatively rely on step (1) instead of the dedicated merge upload step.

3. **L4 — nightly pipeline**  
   **Comment out** `if: build.env("NIGHTLY") == "1"` on the **Upload Nightly Pipeline** step in `pipeline.yml` so the nightly definition is uploaded without setting the env var.  
   Child steps in [`test-nightly.yml`](../../../.buildkite/test-nightly.yml) often repeat `if: build.env("NIGHTLY") == "1"`; **comment out those lines on the steps you need to run**, otherwise they will still be skipped after upload.

### Step 6: Run Tests

Pick one command path:

- **Quick local regression (preferred first)**: single file or `file.py::test_name`
- **CI-like level run**: marker expression + `--run-level`

Full templates live in [references/test-routing.md](references/test-routing.md). **After authoring tests, always emit concrete commands** (see **Output Format**).

**Typical copy-paste examples** (run from repo `tests/` directory; requires matching vLLM/vllm-omni install and hardware):

| Area | Example |
|------|---------|
| **Omni offline L2** | `pytest -s -v e2e/offline_inference/test_qwen2_5_omni.py -m "core_model and omni and not cpu" --run-level=core_model` |
| **Omni online L2** | `pytest -s -v e2e/online_serving/test_qwen3_omni.py -m "core_model and omni" --run-level=core_model` |
| **TTS online L2** | `pytest -s -v e2e/online_serving/test_qwen3_tts_base.py -m "core_model and tts" --run-level=core_model` |
| **Diffusion X2I L2 online** | `pytest -s -v e2e/online_serving/test_qwen_image.py -m "core_model and diffusion" --run-level=core_model` |
| **Diffusion X2V L2 online** | `pytest -s -v e2e/online_serving/test_wan22_t2v.py -m "core_model and diffusion" --run-level=core_model` |
| **Diffusion X2I L4 nightly** | `pytest -s -v e2e/online_serving/test_qwen_image_expansion.py -m "full_model and diffusion and H100" --run-level=full_model` |
| **Diffusion X2I L4 perf (nightly)** | `pytest -s -v dfx/perf/scripts/run_diffusion_benchmark.py --test-config-file dfx/perf/tests/test_qwen_image_vllm_omni.json` |
| **Diffusion X2V L4 nightly** | `pytest -s -v e2e/online_serving/test_wan22_expansion.py -m "full_model and cuda" --run-level=full_model` |
| **Invalid param (weekly H100)** | `pytest -s -v dfx/reliability/invalid_param_test/ -m "slow and H100"` |
| **Invalid param (weekly L4)** | `pytest -s -v dfx/reliability/invalid_param_test/ -m "slow and L4"` |
| **L1 CPU** | `pytest -s -v -m "core_model and cpu"` |

**Prerequisites to mention when relevant**: GPU model (e.g. L4 vs H100), `HF_HOME` / token for hub weights, module-level `skipif` (NPU/XPU-only gaps), and whether CI already collects the path (e.g. `test_*_expansion.py` glob in `test-nightly.yml`).

### Step 7: Validate Result Quality

Before finishing:

- Is the new assertion directly tied to the bug/feature contract?
- Are API calls made via **`runtime.py` `send_*_request`** (not inline HTTP/SDK in the test file)?
- Is the **general** `assert_*` bundled inside `send_*_request` (not duplicated in the test)?
- Is each scenario a **separate `test_*` function** (no `case_id` / `if param.id` branching across cases)?
- **Fixture scope**: default **`omni_server` / `omni_runner`** (module); **`_function`** variants only when each test must restart the instance?
- Is the test deterministic (no fragile timing/network coupling)?
- Is runtime appropriate for the selected level?
- Are markers and `--run-level` consistent?
- **Invalid-param cases** in `tests/dfx/reliability/invalid_param_test/` (not e2e), using `send_*_http_request` + `err_code`, with `pytest.mark.slow` and correct `H100` / `L4` hardware mark?

## Output Format

When completing a request, return:

1. **Test plan** (level, markers, file target, **L4 pillar(s)** if applicable: Function / Accuracy / Perf / Doc; **or** **Invalid param** pillar with target `invalid_param_test/test_invalid_<area>.py`; module basename for e2e: `test_{slug}.py` / `test_{slug}_expansion.py`)
2. **Generated/updated test file(s)** — model-specific constants and request payloads stay **inside** those test modules (**never** `tests/helpers/{slug}.py`). Extend **`tests/helpers/runtime.py`** / **`tests/helpers/assertions.py`** only when new **repo-wide** `send_*` or shared assert helpers are required (do not leave ad-hoc HTTP or `_assert_*` in test modules). **If L4 Perf was requested**, also list **`tests/dfx/perf/tests/*.json`**. **If invalid-param cases were requested**, list the **`invalid_param_test/` script** and parametrized `test_*` / new `id=` rows — **not** e2e paths.
3. **Buildkite change** — **match the requested level and pillar**:
   - **L4 Function**: `test-nightly.yml` **Function Test** shard (file list / note existing sweep). **No** `test-merge.yml` unless user also asked for L3.
   - **L4 Perf**: `test-nightly.yml` **Perf Test · &lt;Model&gt;** step (new step or extend commands); include env exports + artifact upload pattern from a sibling perf job.
   - **L4 Accuracy**: `test-nightly.yml` **Accuracy Test** step under the same model-type group.
   - **L3**: `test-merge.yml` + full `source_file_dependencies` + `agents` + `plugins`.
   - **L2**: `test-ready.yml` + same E2E block requirements.
   - **Invalid param**: `test-weekly.yml` — note **Invalid parameters Test · H100/L4** group; usually **no YAML change** when only extending existing scripts.
   If the user asked only for **L4 functional cases**, state explicitly that **Perf / Accuracy / Invalid param** were not included (offer to add if needed). If the plan mixes **success e2e** and **invalid param**, split deliverables across e2e vs `invalid_param_test/` and call out both CI files.
4. **Run commands (required)** — always include, in fenced `bash` blocks:
   - **Local — whole file**: `cd tests` then `pytest -s -v <path> …`
   - **Local — single test** (optional but preferred when the change is one function): `pytest -s -v path::test_func …`
   - **CI-like** (when not L1 CPU): the same **marker + `--run-level`** pairing the level uses (see [references/test-routing.md](references/test-routing.md) and **Step 6** table above)
   - **Prerequisites** (one line): e.g. “Requires CUDA L4 + weights cached locally”; “Requires H100 + `HF_TOKEN`”; “Syntax-only validation when vllm is not installed locally”
5. **Result summary** (pass/fail if you executed tests; if not executed, state that explicitly and what the user should run)

## Additional Resources

- Marker and command routing: [references/test-routing.md](references/test-routing.md)
- CI pipelines (vllm-omni): [test-ready.yml](../../../.buildkite/test-ready.yml) (L1/L2), [test-merge.yml](../../../.buildkite/test-merge.yml) (L3), [test-nightly.yml](../../../.buildkite/test-nightly.yml) (L4), [test-weekly.yml](../../../.buildkite/test-weekly.yml) (invalid param / reliability)
- L4 documentation example tests (naming, extraction vs copied scripts, output dirs, skips): [docs/contributing/ci/test_examples/l4_doc_example_tests.inc.md](../../../docs/contributing/ci/test_examples/l4_doc_example_tests.inc.md) — see also [PR #1910](https://github.com/vllm-project/vllm-omni/pull/1910)
