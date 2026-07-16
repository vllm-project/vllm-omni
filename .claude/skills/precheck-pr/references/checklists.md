# Pre-Check Checklists

Each item is a concrete check to run against the diff. ✗ = fix before PR. ⚠ = consider fixing.

---

## All PRs: Title Format (check first)

PR title must follow the project convention documented in `docs/contributing/README.md`:

| Prefix | Applies to |
|--------|-----------|
| `[Bugfix]` | Bug fixes |
| `[CI/Build]` | Build or CI improvements |
| `[Doc]` | Documentation changes |
| `[Model]` | New/improved models (include model name) |
| `[Frontend]` | Frontend changes (API server, OmniLLM class, etc.) |
| `[Kernel]` | CUDA/kernel changes |
| `[Core]` | Core logic changes (OmniProcessor, OmniARScheduler, etc.) |
| `[Hardware][Vendor]` | Hardware-specific (e.g., `[Hardware][Ascend]`) |
| `[Misc]` | Other changes (use sparingly) |

**Examples:**
```
✓ [Model] Add MiniCPM-o 4.5 support
✓ [Bugfix] Fix Ovis image text encoder dtype
✓ [Core] Qwen3-Omni performance optimization
✓ [Hardware][Ascend] Fix VoxCPM2 on Ascend
✗ Fix Ovis image text encoder dtype              ← missing prefix
✗ [bugfix] Fix Ovis image text encoder           ← wrong case
✗ [WIP] Add model                                ← WIP in title
```

- [ ] **Title has valid prefix** — ✗ if missing prefix, wrong case, or WIP/Draft in title. For `[Model]` PRs, the title MUST also include the model identifier (e.g., `[Model] Add <ModelName> ...`). ⚠ if missing the model name.

---

## All PRs: Code Quality (run on every PR, quick & full)

Diff-scoped sweep for five fragility patterns that are pervasive in the existing codebase — **only lines the PR adds count**, not the pre-existing backlog. Detection commands, codebase examples, and full severity rules live in [code-quality.md](code-quality.md).

- [ ] **No new `**kwargs` string-lookup plumbing:** the diff adds no `**kwargs` + `kwargs["..."]` / `kwargs.get("...")` / `"..." in kwargs`. ⚠ any new instance; ✗ if a new string key is duplicated across ≥2 files without a shared constant, or unknown keys are silently dropped on a fail-fast path.
- [ ] **No new broad exception swallow:** the diff adds no `except Exception:` / bare `except:`. ⚠ any new instance; ✗ for `except: pass` / `except Exception: return None` / `continue` on a fail-fast path (init, config validation, weight loading, request handling).
- [ ] **No new `Any` / wrong type hints:** the diff adds no `: Any` / `-> Any` / untyped production signature, and no new `SimpleNamespace` in tests faking an object that already has a real typed stub. ⚠ any new instance; ✗ for a *wrong* (actively misleading) annotation or a `SimpleNamespace` test fake of a typed object.
- [ ] **No new hot-path copy:** the diff adds no `.clone()` / `.copy_()` / `copy.deepcopy(...)` inside a per-step loop or per-request path. ⚠ any new instance; ✗ for `.clone()` of a full latent/activations tensor inside an AR or diffusion step loop without a comment, or `deepcopy` of a scheduler / sampling-config on the request path.
- [ ] **No new event-loop blocking:** the diff adds no `time.sleep` / blocking HTTP (`requests`, `urllib`) / `await` inside a held lock in `async def` code. ⚠ any new instance; ✗ for blocking sleep or HTTP on the serving path (`engine/`, `entrypoints/`, async `connectors/`), or a lock held across an `await`. Blocking inside a dedicated worker thread is fine.

Roll the ⚠/✗ counts into the report as a single **Code quality** dimension row.

### Conventions (eyeball — no grep)

Judgment calls that don't grep cleanly — apply them while reading the diff, not as a mechanical check:

- **Right log level** — `info` is for end-user-visible events; routine internal events are `debug`; hot paths are `trace` or removed. Logging is not free.
- **Structured fields over f-strings** — pass dynamic values as structured fields (`logger.info("msg", extra={...})`), not interpolated strings, so they stay greppable and parseable.
- **No reflexive synchronization** — don't add `threading.Lock` / `asyncio.Lock` / `deepcopy` / re-wrap shared state unless there is real concurrent mutation; the GIL covers most Python code. Owners decide their own synchronization — don't pre-lock in a constructor.
- **Context managers for cleanup** — use `with` / `contextlib` for files, locks, CUDA streams, and profilers; not raw `open()` + manual `.close()` or ad-hoc `finally`.
- **No new dependencies without justification** — prefer the stdlib and existing vllm-omni deps; a new top-level dep needs a reason (check `pyproject.toml` / `requirements*.txt`).
- **Naming / interface discipline** — don't rename or change interfaces just for taste in this PR, but do fix names that actively mislead (e.g. `serve` implying a long-running server, `Instance` where multiple instance kinds coexist).

---

## Bug Fix PRs

### Quick

- [ ] **PR body has 4 sections:** what broke (error message), repro steps, root cause, fix description
- [ ] **Fix is minimal:** diff touches only files directly related to the bug. No unrelated refactors or style changes.
- [ ] **Repro is runnable:** the PR body includes a copy-paste command that reproduces the bug
- [ ] **Branch is rebased:** `git merge-base HEAD origin/main` is recent, no merge conflicts

### Full (adds)

- [ ] **Regression test exists:** `git diff --name-only` includes at least one `tests/` file
- [ ] **Regression test reproduces the original error:** the test would fail on main, pass on this branch
- [ ] **Fix matches root cause exactly:** no "fix the symptom + something else" — if the root cause is dtype, the fix is dtype, not dtype + reformatting
- [ ] **No silent failure risk:** no bare `except: pass`, no `try: ... except: return None`, no empty fallback added. See the broad-except pattern in [code-quality.md](code-quality.md#2-broad-exception-swallow) for the full severity rule (⚠ any new broad catch; ✗ for a swallow on a fail-fast path).
- [ ] **Upstream pattern match:** the fix follows the same pattern as existing code for similar cases (grep for analogous `from_pretrained` calls, etc.)
- [ ] **Environment documented:** torch/transformers/vllm versions listed if the bug is version-dependent

---

## Performance PRs

### Quick

- [ ] **Before/after numbers in PR body:** at minimum, one metric (RTF, RPS, latency, VRAM) with main vs PR values
- [ ] **Hardware specified:** GPU model, count, VRAM. "L4" or "H20" is sufficient. "tested on GPU" is not.
- [ ] **Concurrency scaling:** if the optimization claims better throughput, numbers at 2+ concurrency levels
- [ ] **No unexplained regressions:** if any metric goes down at any concurrency level, explain why
- [ ] **Branch is rebased:** `git merge-base HEAD origin/main` is recent, no merge conflicts

### Full (adds)

- [ ] **Software versions:** torch, CUDA, vllm, vllm-omni versions in PR body
- [ ] **Warmup stated:** "1 warmup + 3 measured" or equivalent
- [ ] **Stddev or range:** not just a single best run. "average of 2 rounds" is minimum.
- [ ] **VRAM measured:** peak GPU memory before and after
- [ ] **Benchmark script checked in:** under `tests/` or `examples/`, runnable by reviewer
- [ ] **Exact command line in PR:** not pseudocode, not "run the benchmark"
- [ ] **Pytest summary included:** `36 passed, 3 skipped, 17 warnings in 2897.66s` — proves the whole suite passes
- [ ] **Config changes documented:** if `max_num_seqs`, `gpu_memory_utilization`, or deploy YAML changed, old and new values stated with reasoning

---

## New Model PRs

### Quick

- [ ] **PR body file list matches `git diff --name-only`:** every file path in the description exists in the diff
- [ ] **Registry entries match claimed count:** count `_OMNI_MODELS` or `_DIFFUSION_MODELS` entries added vs claimed in body
- [ ] **Pipeline config `custom_process_input_func` paths resolve:** each referenced module.function exists
- [ ] **`__init__.py` exports match registry:** each registered class appears in `__init__.py` and `__all__`
- [ ] **Shell script modes match README:** `run_curl.sh` supports every mode the README documents
- [ ] **Branch is rebased:** `git merge-base HEAD origin/main` is recent, no merge conflicts

### Full (adds)

- [ ] **Dead code scan (5 patterns):**
  1. Dead `forward()` — defined but never called through the pipeline path
  2. Dead factory/builder functions — instantiated nowhere
  3. Dead wrappers — thin wrappers with no callers
  4. Dead branch guards — `if version > X` where X is always satisfied
  5. Unused parameters — `__init__` args never accessed
- [ ] **Copy-paste detection:** no string constant defined in 3+ files, no cross-module validation duplication, no near-identical shape coercion fns
- [ ] **Import hygiene:** no re-export-only imports, no module-level side effects without docs, no `import os` in function body when already at top. Function-body imports are acceptable when the import is only reachable from a closure or factory return value (i.e., `import` would otherwise execute at module load even though the user may never trigger that path). Either lift to top-of-file or annotate with `# noqa: PLC0415`.
- [ ] **Accuracy:** at least one test compares output against reference; audio = valid WAV at correct sample rate; determinism verified if seed claimed. ⚠ if missing and not explicitly deferred to a follow-up PR or RFC section; — if explicitly deferred (the deferral target SHOULD be linked in the PR body).
- [ ] **Performance:** hardware, software versions, warmup, and metrics (RTF, VRAM, RPS, latency) stated. ⚠ if missing and not explicitly deferred; — if explicitly deferred.
- [ ] **Benchmark settings:** model config (TP, PP, max_model_len, enforce_eager, quant), runtime config (batch, input/output spec), environment versions
- [ ] **Benchmark script checked in + exact command line + pytest summary**

---

## Diffusion Model PRs

Diffusion models live under `vllm_omni/diffusion/`. In addition to the New Model checklist above, verify:

- [ ] **Transformer adapter follows contract:** inherits from `nn.Module`, implements `load_weights()`, uses `vllm_omni.diffusion.attention.layer.Attention`
- [ ] **Pipeline has both offline and online paths:** `forward()` for batch generation, streaming path for online serving
- [ ] **Cache-DiT integration:** at least one acceleration (TeaCache, FBCache, etc.) is wired up. ✗ for new **top-level** pipelines. For **subclass** pipelines that inherit `__call__` from a parent where Cache-DiT is already absent, downgrade to ⚠ and recommend filing a follow-up issue against the parent pipeline family.
- [ ] **Parallelism config:** TP/SP/USP/CFG-Parallel options exposed in pipeline config
- [ ] **Docs table updated:** model added to `docs/models/supported_models.md` with correct metadata
- [ ] **E2E test exists:** at least one test exercises the full generate path

---

## General PRs

### Quick

- [ ] **PR body matches diff:** description doesn't claim changes not in the diff
- [ ] **Branch is rebased:** `git merge-base HEAD origin/main` is recent, no merge conflicts
- [ ] **CI gates passing:** DCO, pre-commit, build — check `gh pr view --json statusCheckRollup` if a PR already exists

### Full (adds)

- [ ] **No new dead code in changed files:** scan for unreachable paths, unused imports
- [ ] **Test coverage:** if core code changed (`engine/`, `stages/`, `connectors/`), tests are added or existing tests cover the path
- [ ] **Import hygiene:** no redundant imports added
- [ ] **No unrelated changes:** diff scope matches PR description
