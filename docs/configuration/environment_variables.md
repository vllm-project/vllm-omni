# Environment Variables

Environment variables are process-level configuration shared by the CLI,
deploy YAML, Python APIs, worker processes, and model integrations. They are
therefore documented under **User Guide → Configuration**, rather than under
CLI Reference.

This page describes the public vLLM-Omni environment-variable contract on the
`latest` documentation branch. Use the documentation for your installed
vLLM-Omni release when behavior differs.

!!! important "Public contract"
    Only names in [Public vLLM-Omni variables](#public-vllm-omni-variables) are
    public Omni configuration. A name found in source code is not automatically
    supported: model-specific, benchmark, platform, and internal variables have
    separate ownership and lifecycle rules.

## Naming and implementation boundary

New public variables owned by vLLM-Omni must use the `VLLM_OMNI_` prefix.
Existing public names with older prefixes remain supported for compatibility,
but they do not establish naming precedent for new settings. Inherited vLLM
and third-party names retain the prefix chosen by their owning project.

The source [environment-variable inventory](gh-file:vllm_omni/config/environment_variable_inventory.py)
records ownership, migration disposition, and redaction metadata. Unlike
`vllm.envs`, it is not an executable value resolver: existing consumers still
own parsing, defaults, precedence, and evaluation time. The public tables below
describe those live consumer contracts. Stable model or stage behavior should
move to typed configuration, and request-varying behavior should move to a
request schema instead of adding another environment variable.

## Precedence and evaluation time

There is no repository-wide rule that makes environment variables override (or
lose to) every CLI, YAML, or request option. The tables below state precedence
for each public setting. In general:

- request fields should control behavior that can vary per request;
- typed deploy or stage configuration should control stable model and stage
  behavior;
- environment variables are appropriate for process launch, platform
  integration, emergency diagnostics, and compatibility fallbacks.

An environment variable can be read while a module is imported, when server or
model configuration is constructed, when a worker is spawned, or whenever a
property is accessed. Set import- and startup-time variables before invoking
`vllm serve` or importing vLLM-Omni. Changing the parent shell after workers
start does not update existing worker environments.

The lifecycle labels used below are:

- **Stable** — reviewed public configuration; changes require compatibility
  consideration.
- **Experimental** — public for evaluation, but its contract may still evolve.
- **Diagnostic** — supported for investigation or temporary mitigation, not as
  a normal tuning interface.
- **Deprecated** — compatibility alias; migrate to the named replacement.

## Public vLLM-Omni variables

### Diffusion backend and cache selection

| Name | Type and default | Applies to and read time | Precedence and invalid values | Lifecycle |
| --- | --- | --- | --- | --- |
| `DIFFUSION_ATTENTION_BACKEND` | Backend name or `auto`; default `auto` (platform selection) | Diffusion stages; read when `OmniDiffusionConfig` is constructed | `diffusion_attention_config.default` wins. An unknown backend fails during backend resolution. | Stable fallback |
| `DIFFUSION_CACHE_BACKEND` | `none`, `cache_dit`, `tea_cache`, `mag_cache`, `step_cache`, `stepcache`, or `step_cache_dit`; default `none` | Diffusion runner startup | Explicit `cache_backend` in config wins. Otherwise this name wins over the deprecated alias. Unsupported values raise `ValueError` during runner setup. | Stable fallback |
| `DIFFUSION_CACHE_ADAPTER` | Same values as `DIFFUSION_CACHE_BACKEND`; default `none` | Diffusion runner startup | Used only when neither explicit `cache_backend` nor `DIFFUSION_CACHE_BACKEND` is set. Unsupported values raise `ValueError`. | Deprecated; use `DIFFUSION_CACHE_BACKEND` |
| `OMNI_DIFFUSION_PROMPT_EMBED_CACHE` | Boolean: `1`, `true`, `yes`, `on`, `0`, `false`, `no`, or `off`; default disabled | Each diffusion runner; resolved during model setup | A recognized environment value overrides the explicit enable setting. An unrecognized value is ignored. | Experimental |
| `OMNI_DIFFUSION_PROMPT_EMBED_CACHE_SIZE` | Positive integer; default `32` entries | Each diffusion runner; resolved during model setup | A valid environment value overrides the explicit cache size. A non-integer logs a warning; a non-positive value is ignored. | Experimental |
| `OMNI_DIFFUSION_SESSION_STATE_MANAGER` | Boolean with the same accepted spellings as the prompt cache; default disabled | Experimental diffusion session manager; model setup | A recognized environment value overrides the explicit enable setting. An unrecognized value is ignored. | Experimental |
| `OMNI_DIFFUSION_SESSION_STATE_MANAGER_MAX_SESSIONS` | Positive integer; default `64` | Experimental diffusion session manager; model setup | A valid environment value overrides the explicit maximum. A non-integer logs a warning; a non-positive value is ignored. | Experimental |

Backend names for `DIFFUSION_ATTENTION_BACKEND` are the members of
`DiffusionAttentionBackendEnum`, such as `FLASH_ATTN`, `TORCH_SDPA`,
`SAGE_ATTN`, `FLASHINFER_ATTN`, and `TRTLLM_ATTN`. Platform support still
depends on the installed kernels and model path.

### Serving and runtime

| Name | Type and default | Applies to and read time | Precedence and invalid values | Lifecycle |
| --- | --- | --- | --- | --- |
| `SPEAKER_SAMPLES_DIR` | Filesystem path; default `~/.cache/vllm-omni/speakers` | Speech server; read when speaker storage initializes | Environment-only setting. The directory is created; filesystem errors propagate. | Stable |
| `SPEAKER_MAX_UPLOADED` | Integer; default `1000` | Speech server; read when speaker storage initializes | Environment-only setting. A non-integer logs a warning and uses `1000`; range is not otherwise validated. | Stable |
| `VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT` | Float seconds; default `600` | Diffusion engine async-output wait in `step_streaming`; resolved per call on the request path, not at import | Environment-only setting. A non-float or `<=0` value warns once and uses the default. | Experimental |
| `VLLM_OMNI_EVENT_DRIVEN_ORCH` | `1`, `true`, `yes` or `on` enables; default `0` (off) | Orchestration loop and the serving-side final-output drain; read once when the `Orchestrator` is constructed | Environment-only setting. Values are stripped and case-normalized; any unrecognized value leaves the legacy poll loop selected. | Experimental |
| `VLLM_OMNI_INPUT_WAIT_TIMEOUT_S` | Float seconds; default `600`; `<=0` disables | Full-payload input coordinator, not async-chunk transfer; read when the scheduler module imports in each worker | Environment-only setting. A non-float logs a warning and uses `600`. | Stable operational control |
| `VLLM_OMNI_ORCH_MONITOR_PATH` | Filesystem path; default `<current-working-directory>/vllm_omni_orch_monitor_<timestamp>.json` | Orchestrator monitor enabled by `--enable-orch-monitor`; read when the monitor is created | Environment-only path override. Parent directories are created; write errors are logged. | Diagnostic |
| `VLLM_OMNI_VIDEO_SYNC_TIMEOUT` | Float seconds; default `600` | Synchronous Videos API; read when the API server module imports | Environment-only setting. A non-float raises `ValueError` during import. | Experimental |
| `VLLM_VIDEO_ASYNC_CHUNK` | `on` or `off`; default `on` | Streaming video output; read on attribute access | Environment-only setting. Values are trimmed and case-normalized; an invalid value warns once and uses `on`. | Experimental |
| `VLLM_VIDEO_AUDIO_DELTA_MODE` | `fast` or `slow`; default `fast` | Streaming video audio deltas; read on attribute access | Environment-only setting. Values are trimmed and case-normalized; an invalid value warns once and uses `fast`. | Experimental |

### Server storage

Storage names use Pydantic's nested-settings delimiter (`__`). They are read
when `ServerSettings` is initialized, normally during server import/startup.
Invalid integer values raise a Pydantic validation error.

| Name | Type and default | Meaning | Lifecycle |
| --- | --- | --- | --- |
| `VLLM_OMNI_SERVER_STORAGE__PATH` | Path; default `/tmp/storage` | Directory for completed server files. | Stable |
| `VLLM_OMNI_SERVER_STORAGE__FILE_CONCURRENCY` | Integer; default `4` | Maximum concurrent file operations. No additional range validation is currently applied. | Stable |
| `VLLM_OMNI_SERVER_STORAGE__FILE_TTL` | Integer seconds or unset; default unset | Optional lifetime for locally stored files. | Stable |
| `VLLM_OMNI_SERVER_STORAGE__TTL_SWEEP_INTERVAL` | Integer seconds or unset | Sweep frequency. If file TTL is set and this is unset, the effective default is `300`. | Stable |
| `VLLM_OMNI_STORAGE_PATH` | Path | Deprecated alias for `VLLM_OMNI_SERVER_STORAGE__PATH`. The new name wins when both are set. | Deprecated |
| `VLLM_OMNI_STORAGE_MAX_CONCURRENCY` | Integer | Deprecated alias for `VLLM_OMNI_SERVER_STORAGE__FILE_CONCURRENCY`. The new name wins when both are set. | Deprecated |

The deprecated aliases emit `DeprecationWarning`. They do not affect file TTL
settings.

### Quantization diagnostics and performance

| Name | Type and default | Applies to and read time | Precedence and invalid values | Lifecycle |
| --- | --- | --- | --- | --- |
| `VLLM_OMNI_SKIP_NVFP4_NAN_CLAMP` | Boolean truthy spellings: `1`, `true`, `yes`, `on`; default false | ModelOpt NVFP4 compatibility patch; read when `vllm_omni.patch` imports | Environment-only escape hatch. Any other value means false. Set only to diagnose the upstream NaN-scale issue. | Diagnostic and temporary |
| `VLLM_OMNI_USE_QUACK_FP8` | Boolean truthy spellings: `1`, `true`, `yes`, `on`; unset means hardware auto-detection | FP8 scaled matrix multiplication; evaluated when quack capability is selected | A set value overrides auto-detection. Any non-truthy value forces quack off. If quack cannot load, vLLM-Omni warns and falls back to FlashInfer. | Experimental performance control |

`QUACK_CACHE_DIR` is owned by the external quack library and is not an
Omni-owned variable, even though vLLM-Omni supplies a persistent default when
it is unset.

## Per-stage environment

Deploy configurations can set arbitrary environment keys for one stage:

```yaml
stages:
  - stage_id: 0
    env:
      HCCL_IF_BASE_PORT: 23000
      MY_MODEL_CACHE: /mnt/model-cache
```

The legacy nested form under `runtime.env` is also accepted. vLLM-Omni converts
keys and values to strings, applies them while the stage engine or child process
is launched, logs keys but not values, and restores the parent process's prior
environment afterward. A child process inherits the scoped values. This keeps
sibling stages from permanently changing one another's parent environment.

This scoping has two consequences:

1. A setting read inside stage/model initialization can see the stage value.
2. A setting already cached by an import that occurred before stage launch
   cannot be changed retroactively with stage `env`.

Stage `env` is an escape hatch for process and third-party settings, not a
replacement for typed stage configuration. Do not put request-varying behavior
there. Secret values may reach child processes; logs and diagnostics must emit
their keys only.

## Inherited vLLM variables

vLLM-Omni also reads variables through its aligned vLLM dependency. Refer to
the [vLLM 0.28 environment-variable reference](https://docs.vllm.ai/en/v0.28.0/configuration/env_vars.html)
for their definitions. This includes vLLM launch, cache, logging, plugin, ROCm,
XPU, ModelScope, and FlashInfer workspace settings.

An inherited name does not become Omni-owned because Omni reads or forwards it.
Where behavior differs, the Omni-specific page should document only that
difference. For example, device-visibility values can be further narrowed by a
stage's `devices` setting.

## Platform, external, and secret variables

Hardware runtimes, distributed launchers, and third-party libraries retain
ownership of their variables. Common groups include:

- device visibility (`ASCEND_RT_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, and
  `MUSA_VISIBLE_DEVICES`);
- distributed launch (`MASTER_ADDR`, `MASTER_PORT`, `RANK`, and `WORLD_SIZE`);
- Hugging Face cache and authentication (`HF_HOME`, `HF_TOKEN`, and related
  names);
- accelerator libraries such as NCCL, FlashInfer, MindIE, MORI, and quack.

Consult the owning project's documentation for accepted values and support.
`HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, and `OPENAI_API_KEY` are explicitly marked
for redaction in the reviewed inventory. Never print their values in environment
collection, examples, bug reports, or logs.

## Model-specific variables

The audit found 58 variables read by a single model or pipeline family. They are
not listed as public usage options here because doing so would turn implementation
escape hatches into an accidental compatibility contract.

Every audited model-specific name has a migration disposition in the
[environment-variable inventory](gh-file:vllm_omni/config/environment_variable_inventory.py):

| Disposition | Count | Required outcome |
| --- | ---: | --- |
| Promote | 35 | Move a stable setting into typed stage or model configuration. |
| Request scope | 6 | Move request-varying behavior into a declared request-option schema. |
| External | 0 | Retain only when a supported third-party library owns the contract. |
| Internalize | 11 | Keep a debug or diagnostic switch out of public documentation and configuration. |
| Deprecate/remove | 6 | Remove a compatibility escape hatch that has no continuing contract. |

The disposition is a migration target, not a statement that the existing
environment switch is stable. Promote or request-scope work should land in
model-owner-reviewed follow-up changes with validation and model-page updates.
The family-sized migration packages and their ownership gate are tracked in
[#6232](https://github.com/vllm-project/vllm-omni/issues/6232).

## Environment collection and troubleshooting

Run the repository collector when preparing a bug report:

```bash
python collect_env.py
```

The environment section includes the safe public Omni allowlist, registered
vLLM variables, and selected platform prefixes. It omits names marked for
redaction and names containing common secret terms. Review the output before
sharing it. A key supplied through stage `env` is not collected merely because
it appears in deploy configuration; the same allowlist and prefix rules apply.

If a setting appears ineffective, first check its read time in the tables
above. Restart processes after changing import- or startup-time values, verify
that the variable reaches the intended worker, and check whether an explicit
typed configuration field takes precedence.

## Benchmark and internal variables

The reviewed inventory marks 20 benchmark variables as transitional while
benchmark configuration moves to CLI/request schemas in
[issue #5376](https://github.com/vllm-project/vllm-omni/issues/5376). They are not
part of the serving/runtime contract.

Internal process-coordination and debug variables are deliberately excluded
from this page. The CI scanner resolves string literals and module-level string
constants used by direct `os` access, local wrapper helpers, aliased `os`
imports, and environment membership checks. Server-storage names generated by
Pydantic settings are checked separately against their model fields.

Names assembled from cross-module metadata and arbitrary keys supplied through
stage `env` cannot be inferred reliably. Contributors must classify those names
explicitly. CI fails when a statically resolved name has no classification or
when a reviewed public name is absent from this page.
