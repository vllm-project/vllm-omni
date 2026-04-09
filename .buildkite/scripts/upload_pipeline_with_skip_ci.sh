#!/usr/bin/env bash
# Evaluate docs-only skip-ci and upload continuation steps from the same `.buildkite/pipeline.yml`
# (YAML document after the first `---`). Buildkite `if` is evaluated at upload time.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PIPELINE_YML="${ROOT}/.buildkite/pipeline.yml"

# Prints a single digit to stdout: 1 = skip image CI, 0 = run. Logs go to stderr.
is_docs_only_change() {
  local file_path
  local has_any=0

  while IFS= read -r file_path; do
    [[ -z "${file_path}" ]] && continue
    has_any=1

    if [[ "${file_path}" == docs/* ]]; then
      continue
    fi
    if [[ "${file_path}" == *.md ]]; then
      continue
    fi
    if [[ "${file_path}" == "mkdocs.yaml" ]]; then
      continue
    fi
    return 1
  done

  [[ "${has_any}" -eq 1 ]]
}

resolve_skip_ci() {
  local is_pr_build=0
  local files
  local base_branch base_ref

  if [[ "${BUILDKITE_PULL_REQUEST:-false}" != "false" && -n "${BUILDKITE_PULL_REQUEST:-}" ]]; then
    is_pr_build=1
  fi

  if [[ "${is_pr_build}" -eq 1 ]]; then
    base_branch="${BUILDKITE_PULL_REQUEST_BASE_BRANCH:-main}"
    if ! git rev-parse --verify "origin/${base_branch}" >/dev/null 2>&1; then
      echo "resolve_skip_ci: origin/${base_branch} not found locally; trying fetch" >&2
      git fetch --depth=200 origin "${base_branch}" >/dev/null 2>&1 || true
    fi

    base_ref=""
    if git rev-parse --verify "origin/${base_branch}" >/dev/null 2>&1; then
      base_ref="origin/${base_branch}"
    elif git rev-parse --verify "${base_branch}" >/dev/null 2>&1; then
      base_ref="${base_branch}"
    else
      echo "resolve_skip_ci: cannot resolve PR base ${base_branch}; skip-ci=0" >&2
      echo -n 0
      return 0
    fi

    if ! files="$(git diff --name-only "${base_ref}...${BUILDKITE_COMMIT}" 2>/dev/null)"; then
      echo "resolve_skip_ci: failed to compute PR changed files; skip-ci=0" >&2
      echo -n 0
      return 0
    fi
  elif [[ "${BUILDKITE_BRANCH:-}" == "main" ]]; then
    if ! git rev-parse --verify "${BUILDKITE_COMMIT}^" >/dev/null 2>&1; then
      echo "resolve_skip_ci: commit has no parent on main; skip-ci=0" >&2
      echo -n 0
      return 0
    fi
    if ! files="$(git diff --name-only "${BUILDKITE_COMMIT}^..${BUILDKITE_COMMIT}" 2>/dev/null)"; then
      echo "resolve_skip_ci: failed to compute main changed files; skip-ci=0" >&2
      echo -n 0
      return 0
    fi
  else
    echo "resolve_skip_ci: not PR/main build; skip-ci=0" >&2
    echo -n 0
    return 0
  fi

  if is_docs_only_change <<< "${files}"; then
    echo "resolve_skip_ci: docs-only change detected; skip-ci=1" >&2
    echo -n 1
    return 0
  fi

  echo "resolve_skip_ci: non-doc changes detected; skip-ci=0" >&2
  echo -n 0
}

SKIP_CI="$(resolve_skip_ci)"

if [[ ! -f "${PIPELINE_YML}" ]]; then
  echo "upload_pipeline_with_skip_ci: missing ${PIPELINE_YML}" >&2
  exit 1
fi

export ROOT SKIP_CI PIPELINE_YML
python3 <<'PY' | buildkite-agent pipeline upload
import os
import pathlib

path = pathlib.Path(os.environ["PIPELINE_YML"])
text = path.read_text(encoding="utf-8")
sep = "\n---\n"
if sep not in text:
    raise SystemExit(
        "upload_pipeline_with_skip_ci: .buildkite/pipeline.yml must contain a '\\n---\\n' separator "
        "(document 1 = bootstrap, document 2 = uploaded steps)"
    )
_, continuation = text.split(sep, 1)

skip = os.environ.get("SKIP_CI") == "1"
# When docs-only skip-ci: skip default CI image, but still build for L4 nightly (PR label nightly-test or
# main NIGHTLY=1), otherwise upload-nightly (depends_on image-build) would be skipped too.
nightly_only = (
    '(build.pull_request.labels includes "nightly-test") '
    '|| (build.branch == "main" && build.env("NIGHTLY") == "1")'
)
# Placeholder in pipeline.yml is `if: __IMAGE_BUILD_IF__` (valid YAML); replace value only.
if skip:
    rep = f"'{nightly_only}'"
    ready_rep = "'false'"
    merge_rep = "'false'"
else:
    rep = "'true'"
    ready_rep = "'build.branch != \"main\" && build.pull_request.labels includes \"ready\"'"
    merge_rep = "'build.branch == \"main\" && build.env(\"NIGHTLY\") != \"1\"'"
rendered = (
    continuation
    .replace("__IMAGE_BUILD_IF__", rep)
    .replace("__UPLOAD_READY_IF__", ready_rep)
    .replace("__UPLOAD_MERGE_IF__", merge_rep)
)
print(rendered, end="")
PY
