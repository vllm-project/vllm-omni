# Import dependency declarations

`check-import-dependencies` checks complete Python ASTs without importing the
application, loading weights, contacting a package index, or examining installed
packages. Pre-commit runs it on changed files; GitHub's existing pre-commit job
runs the **same hook** on all files. Install pre-commit normally, then run:

```bash
pre-commit run check-import-dependencies --all-files
```

The standalone equivalent is:

```bash
python tools/pre_commit/check_import_dependencies.py --all-files
```

Only the checker's small dependencies (`packaging`, and `tomli` on Python 3.10)
are needed. Hook environment creation may download these packages, as with other
Python hooks. The check itself is offline. Its isolated regression suite runs
without vLLM or GPU dependencies:

```bash
python -m pytest --confcutdir=tests/tools tests/tools/test_check_import_dependencies.py -q
```

The suite uses `pytest`, `pytest-xdist`, and `pytest-asyncio` from `dev` (the last
two satisfy the repository's pytest configuration), plus the checker dependencies.
The GitHub job installs just these packages; it does not install vLLM-Omni.

## What is checked

Imports in functions, multiline imports, `TYPE_CHECKING`, platform branches, and
`try/except ImportError` are included. Neither a delayed import nor a handler
proves a dependency is optional. Standard-library, explicit repository modules,
relative imports and real sibling modules are excluded. Namespace mappings use
the longest matching module prefix, so `google.protobuf` does not authorize
`google.cloud`. Dynamic imports (`importlib`, strings passed to `__import__`) are
outside this static check's scope.

For scripts that deliberately change `sys.path`, `local_roots` records the
importing paths, repository search directories and reason. The target module
must actually exist in those directories; this is not a package allowlist.

`import_dependencies.json` defines ordered path rules and dependency groups:

- Core code uses common runtime requirements and the separately installed vLLM
  prerequisite contract.
- Platform modules use their corresponding `requirements/{device}.txt`.
- Tests and tools use runtime plus `dev`; examples add `demo`; docs add `docs`.
- Model-specific extras are assigned only to their consuming paths.

The first matching rule wins. A dependency declared only in `dev` cannot satisfy
an import added to core code. When a new optional feature needs another extra,
declare the package in `pyproject.toml`, assign its paths to a group, and document
how users and CI install that group. Do not add arbitrary packages to the vLLM
prerequisite contract: it records reviewed upstream dependencies, with provenance,
because this repository installs vLLM separately. Review it when upgrading vLLM.
It is not a resolver for transitive dependencies.

The parser supports recursive `-r` / `--requirement`, PEP 508 extras, named direct
URLs, line continuations and environment markers. Constraint entries never count
as installation declarations. Unsupported requirement syntax fails the check.
Markers use the group's configured Linux target (x86_64 by default, aarch64 for
NPU), considering Python 3.10–3.13, rather than the developer's host. This checks
whether a declaration exists for a supported Python minor, **not** whether every
version/architecture combination can install or import it. It does not inspect
imports inside external distributions or resolve their requested extras.

Any non-Python change triggers a full scan, covering dependency manifests with
arbitrary names and changes to nested requirement files. `setup.py`, checker
changes, and staged manifest deletions also trigger a full scan. This deliberately
includes some harmless full scans on documentation changes. CI always scans the
whole repository, including when a dependency is deleted without a Python edit.

## Existing debt and exceptions

`import_dependencies_baseline.json` records existing unresolved imports at the
initial rollout. Each entry identifies an exact file, imported name, enclosing
class/function, normalized import statement and occurrence count, plus a reason.
It is not a global package allowlist. A new file, function, statement, or additional
occurrence still fails. Moving an import from examples into runtime code is
checked against the destination's dependency group.

Fix old entries by declaring the dependency in the right installation group,
correcting an import mapping, or removing the import. Full scans reject stale
entries and require their removal/count reduction. Do not automatically regenerate
the baseline to make CI pass. A genuinely optional import with an operational
fallback can have a narrowly scoped exception after review, explaining the
fallback and installation instructions. Type-only and platform-only imports also
require explicit declarations or explained exceptions.

For an inventory (this does **not** update the baseline):

```bash
python tools/pre_commit/check_import_dependencies.py --all-files --report /tmp/import-debt.json
```

## Actual benchmark environment

A declaration gate cannot detect an old CI image, a skipped install step, a broken
shared library, or an incompatible codec. MiniCPM Daily-Omni's CUDA and NPU nightly
steps therefore run this command **before starting the model**:

```bash
python -m vllm_omni.benchmarks.media_preflight
```

It creates a tiny offline MPEG-4/PCM fixture and exercises the benchmark's actual
PyAV frame extraction, vLLM audio loading and JPEG/WAV serialization. Failure is a
nonzero exit, not a skipped test. Current main uses PyAV; the historical `decord`
import from issue #6091 has already been removed. The dataset also checks required
imports before loading samples, so environment failures cannot be swallowed by
its per-sample error handler.

To probe real downloaded media in the same benchmark environment:

```bash
python -m vllm_omni.benchmarks.media_preflight --video /path/to/video.mp4 --audio /path/to/audio.wav
```

Omit `--audio` to decode the video's own audio track. This is a media environment
check, not a model inference or accuracy test.
