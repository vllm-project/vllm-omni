# Code-Quality Patterns

Five fragility patterns to sweep on **every** PR, in both quick and full mode. These are pervasive in the existing codebase (hundreds of pre-existing sites) — so the checks are **diff-scoped**: only lines the PR *adds* count. Whole-repo greps light up every PR with the backlog and are useless.

All commands assume:
```bash
BASE="$(git merge-base HEAD origin/main)"
```

**Severity:** ⚠ for any *new* instance introduced by the diff; ✗ for the acute sub-cases defined per pattern. Acute ✗ findings belong in the report's blocking count.

---

## 1. `**kwargs` string-lookup passthrough

### Anti-pattern
```python
def forward(self, **kwargs):
    # BLOCKER-prone: raw dict + string keys duplicated across files
    if "runtime_additional_information" in kwargs and \
       "model_intermediate_buffer" not in kwargs:
        ...
```
Real instances of this exact guard are copy-pasted across `cosyvoice3.py`, `indextts2_talker.py` (×2), `indextts2_s2mel_decoder.py`, `qwen3_tts_talker.py`, `qwen3_omni.py`. A typo in any string key silently no-ops; there is no type check, no IDE completion, and unknown keys are silently dropped.

### Why dangerous
Raw `dict` + `**kwargs` + lists of param-name strings is **fragile by construction**: the compiler can't see a missing or misspelled key, so failures are silent. Each duplicated copy is a new place for the strings to drift.

### Severity
- ⚠ any *new* `**kwargs` signature whose body does `kwargs["..."]` / `kwargs.get("...")` / `"..." in kwargs`.
- ✗ when the diff introduces a **new** string key that is duplicated across ≥2 files without a shared module constant, or when `**kwargs` plumbing **silently drops unknown keys** on a fail-fast path (init, config validation, weight loading).

### Detect (diff-scoped)
```bash
git diff ${BASE}...HEAD -- "*.py" | grep "^+[^+]" \
  | grep -E '\*\*kwargs|kwargs(\.get|\[)|"[a-z_]+"\s+in\s+kwargs'
```

### Fix
Prefer explicit typed params. The repo already has the right primitive: `msgspec.Struct` with `forbid_unknown_fields=True` rejects unknown keys instead of silently dropping them — see `_StructBase` in `vllm_omni/data_entry_keys.py:105` and `MoriPullRequest` in `vllm_omni/distributed/omni_connectors/connectors/mori_transfer_engine_connector.py:81`. A `@dataclass` or `TypedDict` works too.

If `**kwargs` is unavoidable (vLLM base-class compat), centralize the key strings as module-level constants behind **one** typed accessor, instead of repeating the literal across files.

---

## 2. Broad exception swallow

### Anti-pattern
```python
# BLOCKER-prone: catches everything, returns None on any failure
try:
    evt = build_event(...)
    return evt
except Exception:
    return None
```
Real clusters: `vllm_omni/metrics/stats.py:240,274,367,450` and `vllm_omni/metrics/{modality,utils}.py` — all `except Exception: return None`.

### Why dangerous
A generic `except Exception:` is **too flexible**. If we catch an exception type we don't expect, it is almost always an unhandled edge case that needs to be *fixed* — not swallowed into `None`/`pass`/`continue`. Broad catches turn fixable bugs into silent wrong behavior and make debugging impossible.

The repo backs this with ruff `BLE001` (broad-exception catch); it is currently suppressed via `# noqa: BLE001` in `vllm_omni/patch.py:413`. Prefer fixing over suppressing.

### Severity
- ⚠ any *new* `except Exception:` / `except BaseException:` / bare `except:`.
- ✗ for `except:` / `except Exception: pass|return None|continue` on a **fail-fast path** (init, config validation, weight loading, request handling, connector setup). This extends the Bug-Fix "No silent failure risk" ✗ in [checklists.md](checklists.md).

### Detect (diff-scoped)
```bash
# Any new broad catch (⚠ baseline):
git diff ${BASE}...HEAD -- "*.py" | grep "^+[^+]" \
  | grep -E 'except\s*(Exception|BaseException)?\s*:'
# Note: this matches only the immediate-colon swallow forms (`except :`,
# `except Exception:`) — it deliberately excludes specific catches like
# `except ValueError:` and the `except Exception as e:` variant. Eyeball
# the latter; `as e` usually means it is logged, but still verify.
```

### Fix
Catch the **specific** types you actually expect (`ValueError`, `KeyError`, `AttributeError`, `OSError`, `torch.cuda.OutOfMemoryError`, `msgspec.ValidationError`, …). At a genuine top-level / best-effort boundary (metrics, signal handlers), at minimum **log** the exception; never swallow into a bare `pass`/`return None` on a path that should fail loudly.

---

## 3. `Any` / wrong type annotations

### Anti-pattern
```python
# ⚠ Any param + Any return — no contract, no narrowing
def _extract_mm_output(engine_outputs: Any) -> dict[str, Any]:
    ...
```

### Why dangerous (root cause)
`Any` disables the type checker for that position, so typos and wrong shapes pass silently. The repo has 636 `: Any` params, 244 `-> Any` returns, and **1,173 `SimpleNamespace` in `tests/`**. The `Any` leak is largely test-driven: vibe-coded unit tests fake objects with `SimpleNamespace`, which forces the production code consuming them to be typed `Any`, and the `Any` then propagates outward into public signatures.

### Severity
- ⚠ any *new* `: Any` / `-> Any` / fully-untyped signature in production code.
- ✗ for a **wrong** annotation (actively misleading — e.g. `-> bool` that returns `Optional[bool]`), or a *new* `SimpleNamespace` in a test that mimics an object which already has a real typed stub / `@dataclass` / `TypedDict` / `Protocol`.

### Detect (diff-scoped)
```bash
# Any leaking into new signatures:
git diff ${BASE}...HEAD -- "*.py" | grep "^+[^+]" \
  | grep -E ':\s*Any\b|->\s*Any\b'
# SimpleNamespace newly added in tests (the Any-leak source):
git diff ${BASE}...HEAD -- "tests/" | grep "^+[^+]" | grep "SimpleNamespace"
```

### Fix
Replace `Any` with the concrete type, a `Protocol`, or a `Union`. If a type is genuinely dynamic, prefer `object` + `isinstance` narrowing over `Any` — `object` keeps the checker engaged. In tests, replace `SimpleNamespace` with the real class, a small `@dataclass`/`TypedDict`, or a `Protocol` + fake, so the system under test stays fully typed.

---

## 4. Unnecessary copy — `.clone()` / `copy.deepcopy` in hot paths

### Anti-pattern
```python
# Per-step in a diffusion/AR loop: clones the full latent every iteration
for t in timesteps:
    trajectory_latents.append(x_t.clone())   # O(N) GPU alloc + memcpy each step
```
Real clusters: `vllm_omni/diffusion/models/bagel/bagel_transformer.py` calls `x_t.clone()` ~9× across a trajectory loop (lines 1820–2186); the cache hooks `vllm_omni/diffusion/cache/teacache/hook.py:155` and `magcache/hook.py:182` clone `hidden_states` per step. Separately, `copy.deepcopy(self.scheduler)` / `copy.deepcopy(req.sampling_params)` is copy-pasted across diffusion pipelines — `pipeline_qwen_image.py:786`, `pipeline_hunyuan_image3.py:1904`, `pipeline_ltx2*.py` (×3), `pipeline_dreamzero.py:1085`, `diffusion_model_runner.py:398` — a full object-graph deepcopy on the request path. (Contrast `dreamzero/causal_wan_model.py:203`, which carries a `# No .clone() needed.` comment — the aware pattern.)

### Why dangerous
A `.clone()` of a GPU tensor or a `copy.deepcopy` of a config object inside a per-step / per-request path is a hidden allocator: it reads as one Python line but triggers a GPU memcpy or a deep object-graph walk every iteration. Across 50 diffusion steps or hundreds of AR steps it dominates latency and VRAM bandwidth. `deepcopy` additionally ignores any custom device / `__copy__` logic and can drag tensors through CPU.

### Severity
- ⚠ any *new* `.clone()` / `.copy_()` / `copy.deepcopy(...)` inside a per-step loop or per-request path.
- ✗ for `.clone()` of a full latent/activations tensor inside an AR or diffusion step loop without an explanatory comment, or `copy.deepcopy` of a scheduler / sampling-config on the request hot path.

### Detect (diff-scoped)
```bash
git diff ${BASE}...HEAD -- "*.py" | grep "^+[^+]" \
  | grep -E '\.clone\(\)|\.copy_\(|copy\.deepcopy|deepcopy\('
```

### Fix
Tensors: prefer a view (`x[..., :k]`, `torch.narrow`, `as_strided`) or write into a pre-allocated buffer with `dst.copy_(src)` instead of `.clone()`; if the caller no longer needs the original, `move`/`del` it rather than copy. Schedulers / configs: build a fresh lightweight instance per request, or split out the mutable per-request bits so the shared immutable part isn't deepcopied. If a copy is genuinely required (e.g. the source is mutated in place later), keep it and add a one-line comment saying why.

---

## 5. Blocking the asyncio event loop / lock held across `await`

### Anti-pattern
```python
# BLOCKER-prone: blocking sleep / blocking HTTP / lock held across an await,
# all inside an async (event-loop) function
async def handle(self, req):
    with self._lock:
        resp = await self._client.call(req)   # lock held across await -> serializes all requests
    time.sleep(0.1)                           # stalls the single loop for every concurrent request
```

### Why dangerous
vLLM-Omni serves over a single asyncio event loop. Any synchronous blocking work on an `async def` path — `time.sleep`, blocking HTTP (`requests`/`urllib`), heavy CPU work, or a lock held across an `await` — stalls the loop for *every* concurrent request, not just the blocking one. A lock held across `await` silently serializes the whole pipeline; blocking I/O tanks throughput and inflates tail latency.

### Severity
- ⚠ any *new* `time.sleep(`, blocking HTTP call (`requests.get/post`, `urllib.request`, `urlopen`), or `await` inside a `with Lock` / `async with` critical section, in `async def` code.
- ✗ for `time.sleep` or blocking HTTP on the serving path (`engine/`, `entrypoints/`, async `connectors/`), or a lock acquired and held across an `await` on the request path.

### Detect (diff-scoped)
```bash
# New blocking primitives introduced by the diff — then eyeball whether each
# sits in an async function on the loop vs a dedicated worker thread:
git diff ${BASE}...HEAD -- "*.py" | grep "^+[^+]" \
  | grep -E 'time\.sleep|requests\.(get|post|put|delete)|urllib\.request|urlopen'
# Lock acquired with an await potentially inside the critical section (manual eyeball):
git diff ${BASE}...HEAD -- "*.py" | grep "^+[^+]" | grep -E 'async with|\.acquire\(\)'
```
Note: blocking inside a **dedicated worker thread is correct** — the repo already does this (`diffusion/model_loader/hub_prefetch.py`, `distributed/omni_connectors/kv_transfer_manager.py`, the `mooncake`/`mori` connectors poll with `time.sleep` on their own threads). The check targets the *event-loop* path only.

### Fix
Move blocking work off the loop with `asyncio.to_thread(...)` / `run_in_executor`, or use a native async client (`httpx.AsyncClient` / `aiohttp`). Never `await` while holding a lock — drop the lock before the `await` and re-acquire after, or restructure so the critical section contains no `await`.

---

## Running the sweep

Run all five detections against `${BASE}...HEAD`, then for each hit decide ⚠ vs ✗ using the severity rules above. Roll the results into the report as a single **Code quality** dimension row (count of ⚠ and ✗), alongside the type-specific checklist from [checklists.md](checklists.md).
