# Unblock: reference code + env + checkpoints

Unblocks **B1** in [plan-fun-audio-chat-s2s.md](./plan-fun-audio-chat-s2s.md#b1-reference-code--cosyvoice-submodules--env--weights-not-available-as-a-runnable-stack-on-this-box).

**Why this sub-plan exists:** the parent plan needs a runnable reference stack on this box — to (a) run `infer_s2s.py` and capture ground-truth wav/tokens, (b) parity-test every ported module. This box currently has neither a Python env with `transformers`/`vllm` nor the model weights nor the CosyVoice submodule.

---

## done/

### UD1. Clone `FunAudioLLM/Fun-Audio-Chat` reference repo
- **Goal:** get reference modeling/processing/example code readable on-box.
- **Acceptance criteria:** `src/funaudiochat/funaudiochat/modeling_funaudiochat.py` exists (1454 lines).
- **Next action:** n/a.
- **Affected paths:** `src/funaudiochat/**`.

---

## ongoing/

### UO1. Initialize CosyVoice + Matcha-TTS submodules
- **Goal:** populate `src/funaudiochat/third_party/CosyVoice/**` (currently empty because clone was shallow/no-submodules).
- **Acceptance criteria:** `src/funaudiochat/third_party/CosyVoice/cosyvoice/cli/cosyvoice.py` exists; `src/funaudiochat/third_party/CosyVoice/third_party/Matcha-TTS/matcha/**` exists.
- **Dependencies:** UD1.
- **Next action:** `git -C src/funaudiochat submodule update --init --recursive`.
- **Affected paths:** `src/funaudiochat/third_party/**`.

### UO2. Create `uv` venv from a **single tracked dependency spec + runtime contract** with a canonical `uv.lock`
- **Goal:** a reproducible Python env with one input (dep spec) and one lockfile *plus* a tracked runtime contract that pins the interpreter version and the Torch/CUDA build. Any engineer can rebuild `.venv` byte-for-byte and verify the runtime matches via `uv sync --frozen` + a runtime-guard script. No imperative `uv pip install` as a source of truth. Package-level reproducibility is not enough — two machines with the same `uv.lock` can still import different CUDA builds of torch and produce different numerics.
- **Single source of truth — `pyproject.toml` `[project.optional-dependencies] fun-audio-chat-ref`:**
  ```toml
  [project.optional-dependencies]
  fun-audio-chat-ref = [
    # Content mirrored from src/funaudiochat/requirements.txt, pinned to exact versions,
    # plus openai-whisper + hf_transfer + huggingface_hub (also pinned).
    # Hand-curated once here so uv.lock has a deterministic input.
    "openai-whisper==<exact>",
    "librosa==<exact>",
    "torchaudio==<exact>",
    "hf_transfer==<exact>",
    "huggingface_hub==<exact>",
    # ...ref requirements.txt contents, pinned...
  ]
  ```
  (Grep the repo first for existing whisper/librosa/torchaudio pins and match them.)
- **Tracked runtime contract — `tests/e2e/offline_inference/fun_audio_chat_reference/runtime.lock`:**
  ```json
  {
    "python_version": "3.10.14",              // sys.version_info major.minor.micro, exact
    "torch_version":  "2.X.Y+cu121",          // torch.__version__, includes cuXYZ build tag
    "torch_cuda":     "12.1",                 // torch.version.cuda
    "torch_git":      "<40-char sha>",        // torch.version.git_version (build provenance)
    "cudnn_version":  "<int>",                // torch.backends.cudnn.version()
    "cuda_available": true,                    // must be true (we're running on GPU)
    "gpu_name_regex": "A100|H100|L4|L40",     // allowlist of GPUs the baseline may be regenerated on
    "pinned_at":      "<ISO date>"
  }
  ```
- **Acceptance criteria:**
  - `pyproject.toml` has the `fun-audio-chat-ref` optional-extra above with every dep pinned to `==<exact>` (including `torch==<version>+cu<xyz>` so wheel build is pinned, not just version).
  - `pyproject.toml` `requires-python` is pinned to an exact interpreter, e.g. `==3.10.14`, matching `runtime.lock`.
  - `uv.lock` exists and is **tracked**, generated from that `pyproject.toml` alone via `uv lock --extra fun-audio-chat-ref` (the lock is a function of the dep spec; ad-hoc `uv pip install` commands do not feed it).
  - `runtime.lock` exists and is **tracked**. It is populated via an explicit **bootstrap** step (new `scripts/bootstrap_runtime_lock.py`) that:
     1. Verifies the clean-rebuild gate (`uv sync --frozen --extra fun-audio-chat-ref` succeeds with no resolver pass).
     2. Imports torch/sys and writes `runtime.lock` with the schema above.
     3. Prints the content for human inspection and requires the operator to commit it.
    The bootstrap script is the **only** sanctioned path to create/update `runtime.lock`. All other scripts (wrapper, determinism gate, baseline-validity gate) hard-error if `runtime.lock` is missing or does not match the current runtime; they never write to it.
  - Clean-rebuild gate: `rm -rf .venv && uv sync --frozen --extra fun-audio-chat-ref` recreates `.venv` and **no resolver pass runs** (the `--frozen` flag fails if the lock is stale). This is the package-level reproducibility proof.
  - Runtime-match gate: `scripts/check_runtime_contract.py` (tracked) is the **first thing** the reference wrapper (UO4) and every later test run calls; it loads `runtime.lock`, imports torch, and errors if `python_version`, `torch_version`, `torch_cuda`, `torch_git`, or `cudnn_version` do not match, or if the GPU name does not match `gpu_name_regex`. Failure → refuse to run (preserves baseline attribution).
  - `uv run python -c "import torch, transformers, vllm, vllm_omni, whisper, librosa; print('ok')"` prints `ok` from the locked env.
  - `uv run python -c "from funaudiochat.register import register_funaudiochat; register_funaudiochat(); print('ok')"` prints `ok` with `PYTHONPATH` including `src/funaudiochat/`.
  - Submodule SHAs frozen: `src/funaudiochat/third_party/CosyVoice` and `Matcha-TTS` SHAs recorded in `tests/e2e/offline_inference/fun_audio_chat_reference/submodules.lock` (tracked). Reference clone commit pinned via `src/funaudiochat.sha` (tracked one-liner).
- **Dependencies:** UD1.
- **Next action:**
  1. Grep repo for existing pins of whisper/librosa/torchaudio and adopt them.
  2. Curate `[project.optional-dependencies].fun-audio-chat-ref` in `pyproject.toml` with exact `==` pins, including `torch==<version>+cu<xyz>`. No free-floating versions.
  3. Pin `requires-python = "==3.10.14"` (or whatever current interpreter) in `pyproject.toml`.
  4. `uv lock --extra fun-audio-chat-ref` → commit `uv.lock`.
  5. `rm -rf .venv && uv sync --frozen --extra fun-audio-chat-ref` to prove the lock is self-sufficient.
  6. Dump runtime metadata (`python_version`, `torch_version`, `torch_cuda`, `torch_git`, `cudnn_version`, GPU name) into `tests/e2e/offline_inference/fun_audio_chat_reference/runtime.lock`; commit.
  7. Write `scripts/check_runtime_contract.py`; make the UO4 wrapper call it at entry.
  8. Record submodule + reference-clone SHAs; commit those too.
  9. Sanity-imports.
- **Affected paths:** `pyproject.toml` (edit — `fun-audio-chat-ref` optional extra + `requires-python` pin), `uv.lock` (new, tracked), `.venv/` (untracked), `tests/e2e/offline_inference/fun_audio_chat_reference/runtime.lock` (new, tracked), `tests/e2e/offline_inference/fun_audio_chat_reference/submodules.lock` (tracked), `src/funaudiochat.sha` (tracked), `scripts/check_runtime_contract.py` (new, tracked).

### UO3. Download model weights from **immutable** HF revisions recorded before fetch
- **Goal:** checkpoint provenance is fixed at plan time, not at download time. `--revision main` is banned — downloads are driven by SHAs already in `weights.lock`.
- **Two-phase flow:**
  - **Phase A (one-time, pin the snapshot):** on the first run, query HF for the current commit SHA of each repo at the moment of initial download (`huggingface_hub.HfApi().repo_info(repo_id).sha`), write these SHAs into `weights.lock` **before** the actual download, commit `weights.lock`. This is the only moment `main` is ever referenced.
  - **Phase B (download + all later reruns):** all download commands pass `--revision <sha-from-weights.lock>`. If `weights.lock` is absent, the download script **refuses to run** and prints "Phase A not yet complete — run `scripts/pin_fun_audio_chat_weights.py` first." Every engineer and CI run fetches the same immutable snapshot.
- **Acceptance criteria:**
  - `tests/e2e/offline_inference/fun_audio_chat_reference/weights.lock` (tracked) has the schema:
    ```json
    {
      "Fun-Audio-Chat-8B":       {"repo_id": "FunAudioLLM/Fun-Audio-Chat-8B",       "revision": "<40-char sha>", "pinned_at": "<ISO date>"},
      "Fun-CosyVoice3-0.5B-2512":{"repo_id": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512","revision": "<40-char sha>", "pinned_at": "<ISO date>"}
    }
    ```
  - `scripts/download_fun_audio_chat_weights.py` (tracked) reads `weights.lock`, errors if it is missing or has a non-SHA revision, then calls `huggingface_hub.snapshot_download(repo_id, revision=<sha>, local_dir=...)`. No `--revision main` anywhere.
  - After running the download script, `pretrained_models/Fun-Audio-Chat-8B/config.json` and `pretrained_models/Fun-CosyVoice3-0.5B-2512/*.pt` exist.
  - Sanity: `huggingface_hub` `.sha` of the local download matches the recorded SHA.
- **Dependencies:** UO2.
- **Next action:** build `scripts/pin_fun_audio_chat_weights.py` (Phase A) and `scripts/download_fun_audio_chat_weights.py` (Phase B); run Phase A once; commit `weights.lock`; run Phase B.
- **Affected paths:** `scripts/pin_fun_audio_chat_weights.py` (new, tracked — Phase A only), `scripts/download_fun_audio_chat_weights.py` (new, tracked — Phase B), `tests/e2e/offline_inference/fun_audio_chat_reference/weights.lock` (tracked), `pretrained_models/**` (gitignored).

### UO4. Tracked reference smoke wrapper + frozen baseline artifacts
- **Goal:** produce a reproducible reference inference run and **check in its outputs** as the ground-truth baseline the ship-gates in O10 compare against. No untracked path edits. Determinism is gated on a strictly-deterministic artifact with zero volatile fields — volatile metadata lives in a separate non-gated file.
- **Artifact split (volatile vs gated):**
  - **Gated (byte-comparable; determinism check):**
    - `tests/e2e/offline_inference/fun_audio_chat_reference/ck7vv9ag.tokens.json` — `{"text_tokens": [...], "crq_tokens": [...]}`. Sorted keys, no whitespace variation (`json.dumps(..., sort_keys=True, separators=(",",":"))`), newline at EOF.
    - `tests/e2e/offline_inference/fun_audio_chat_reference/ck7vv9ag.wav` — 24 kHz PCM, bit-identical across reruns.
    - `tests/e2e/offline_inference/fun_audio_chat_reference/parity_manifest.json` — strictly deterministic: `{"uv_lock_sha256": "...", "weights_lock_sha256": "...", "submodules_lock_sha256": "...", "refclone_sha": "...", "seed": 42, "sp_gen_kwargs": {...}, "input_audio_sha256": "...", "max_consecutive_crq_run": <int>}`. Zero host info, zero timestamps. Canonical JSON encoding as above. This is the file the determinism gate byte-compares.
  - **Non-gated (for humans / debugging):**
    - `tests/e2e/offline_inference/fun_audio_chat_reference/run_info.json` — *not* under the determinism gate: host info, timestamps, command line, duration, stderr tail. Free to differ across reruns.
- **Acceptance criteria:**
  - A tracked wrapper `scripts/run_reference_infer_s2s.py` runs reference S2S end-to-end. First action: call `scripts/check_runtime_contract.py` (from UO2); refuse to run on runtime mismatch. All params (model_path, audio_path, seed, sp_gen_kwargs) are CLI flags; no local edits to `src/funaudiochat/examples/infer_s2s.py`.
  - Wrapper writes all artifacts above (gated + non-gated) into its `--out-dir` argument (tests use a scratch dir; the tracked-artifact-regeneration command uses the tracked path).
  - **Determinism gate (internal):** `scripts/check_reference_determinism.sh` runs the wrapper twice into two scratch dirs and asserts `sha256sum` equality of `ck7vv9ag.tokens.json`, `ck7vv9ag.wav`, and `parity_manifest.json` between those two runs. `run_info.json` is *not* compared. If the gate fails, seed/greedy is not fully deterministic — fix before proceeding (`use_deterministic_algorithms=True`, `CUBLAS_WORKSPACE_CONFIG=:16:8`, no non-deterministic sdpa kernels).
  - **Baseline-validity gate (mandatory, separate):** `scripts/check_tracked_baseline.sh` runs the wrapper **once into a scratch dir** and asserts `sha256sum` equality between each freshly generated gated artifact and the **tracked** counterpart under `tests/e2e/offline_inference/fun_audio_chat_reference/`. This catches the case where the checked-in baseline is stale, hand-edited, produced from older weights/submodules, or produced on a different runtime. **O10 must not begin until this gate is clean on the current HEAD.** The `tokens.json`/`wav`/`parity_manifest.json` files are treated as generated artifacts — they must always be reproducible from the locks on this HEAD.
  - **Baseline-update path:** the only sanctioned way to update the tracked baseline is running `scripts/regenerate_tracked_baseline.sh` (tracked helper that writes directly into `tests/e2e/offline_inference/fun_audio_chat_reference/` and refuses to run unless `runtime.lock`, `uv.lock`, `weights.lock`, `submodules.lock` are all clean). Manual edits are guarded by a pre-commit check that rejects direct diffs to the gated artifacts without a matching lockfile change.
  - **CI hook:** nightly (or on-change) CI job runs `check_tracked_baseline.sh`; a drift between tracked artifacts and freshly-regenerated ones fails the build and blocks any PR that touches the locks.
  - Whisper sanity: transcription of the reference wav contains at least one Chinese character. (Low-bar smoke, not a parity gate.)
- **Dependencies:** UO1, UO2, UO3.
- **Next action:** build the wrapper + runtime check + three gate scripts; run determinism gate, then baseline-validity gate; commit all tracked artifacts; wire the nightly CI hook in `.buildkite/` (copy conventions from existing e2e tests).
- **Affected paths:** `scripts/run_reference_infer_s2s.py` (new, tracked), `scripts/check_reference_determinism.sh` (new, tracked), `scripts/check_tracked_baseline.sh` (new, tracked), `scripts/regenerate_tracked_baseline.sh` (new, tracked), `tests/e2e/offline_inference/fun_audio_chat_reference/**` (tracked gated baseline artifacts + non-gated run_info.json), `.buildkite/test-*.yml` (add nightly baseline-validity job).

---

## blocked/

None.

---

## Changelog

- 2026-04-21 v1 — initial draft paired with [plan-fun-audio-chat-s2s.md](./plan-fun-audio-chat-s2s.md).
- 2026-04-21 v2 — addressed adversarial-review [medium]: UO2 now mandates a tracked `uv.lock` + pinned submodule SHAs; UO3 records HF weight revisions; UO4 replaced with a tracked wrapper that emits frozen baseline artifacts (`tokens.json` + `wav` + `run_info.json`) and is checked for byte-identical reruns. No untracked reference-code edits.
- 2026-04-21 v3 — addressed adversarial-review v2:
  1. [high] *UO2 lockfile flow unreliable (imperative `pip install` + `pip freeze`)* → UO2 rewritten: single tracked input (`pyproject.toml` `[project.optional-dependencies].fun-audio-chat-ref` with exact `==` pins) → canonical `uv.lock` via `uv lock`. Clean-rebuild gate: `rm -rf .venv && uv sync --frozen --extra fun-audio-chat-ref` must succeed with no resolver pass. Imperative `uv pip install` removed.
  2. [high] *UO3 used `--revision main`* → UO3 split into Phase A (resolve + pin SHAs into `weights.lock`, one-time) and Phase B (download with `--revision <sha-from-weights.lock>`); download script errors out if `weights.lock` missing.
  3. [medium] *UO4 determinism gate contradictory* → UO4 artifact set split into gated (byte-comparable: `tokens.json`, `wav`, `parity_manifest.json` with zero volatile fields) vs non-gated (`run_info.json` with host/time); `scripts/check_reference_determinism.sh` enforces `sha256sum` equality of the gated set only across reruns.
- 2026-04-21 v4 — addressed adversarial-review v3:
  1. [high] *UO2 pins packages but not interpreter/Torch/CUDA runtime* → added tracked `runtime.lock` capturing `python_version`, `torch_version` (incl. `+cuXYZ` build tag), `torch_cuda`, `torch_git`, `cudnn_version`, `gpu_name_regex`; `pyproject.toml` `requires-python = "==3.10.14"`; `torch==<version>+cu<xyz>` pin in the extra; new `scripts/check_runtime_contract.py` gates every reference-wrapper run on a runtime-match.
  2. [high] *UO4 only checked two scratch runs vs each other, never against the tracked baseline* → added a separate `scripts/check_tracked_baseline.sh` gate that asserts the freshly-generated gated artifacts byte-equal the tracked files. O10 may not begin until this is clean. Added `scripts/regenerate_tracked_baseline.sh` as the only sanctioned update path, plus a nightly CI job that fails on drift.
