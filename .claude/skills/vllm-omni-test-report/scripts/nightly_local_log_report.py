#!/usr/bin/env python3
"""
Parse pytest-style logs under logs/nightly_jobs (or --log-dir) and emit HTML or Markdown.

Used for **report type nightly** in vllm-omni-test-report. Discovery rules:
../references/nightly-local-log-layout.md
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, NamedTuple

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from kanban_assets_perf_summary import (  # noqa: E402
    _empty_perf_status_counts,
    build_assets_perf_summary,
)
from kanban_repo_config import KANBAN_REPO_URL  # noqa: E402
from laptop_path_defaults import (  # noqa: E402
    DEFAULT_KANBAN_REPO_ROOT_DISPLAY,
    DEFAULT_LAPTOP_REPO_ROOT_DISPLAY,
    resolve_kanban_repo_root,
    resolve_laptop_repo_root,
)
from local_perf_results import (  # noqa: E402
    collect_local_perf_test_keys,
    local_perf_result_files,
    perf_row_matches_local_test,
    resolve_local_perf_result_dir,
)
from nightly_job_log_discovery import discover_job_logs, read_combined_job_logs  # noqa: E402
from nightly_job_pytest_table import (  # noqa: E402
    ORG,
    PIPELINE,
    collect_nightly_job_log_analyses,
    fetch_nightly_build,
)
from pytest_log_parse import (  # noqa: E402
    extract_pytest_counts,
    extract_pytest_duration_display,
    parse_pytest_log,
)
from report_html_theme import EDITORIAL_THEME_CSS  # noqa: E402
from report_naming import (  # noqa: E402
    default_nightly_html_path,
    nightly_report_title,
    resolve_report_date_iso,
)

_SKILL_DIR = _SCRIPTS.parent


class BkTarget(NamedTuple):
    """A Buildkite org/pipeline pair to query for scheduled nightly builds.

    ``label`` is the human-readable name used as the chapter heading
    (e.g. ``CUDA`` / ``NPU``).
    """

    label: str
    org: str
    pipeline: str
    branch: str = "main"


CUDA_TARGET = BkTarget(label="CUDA", org="vllm", pipeline="vllm-omni")
NPU_TARGET = BkTarget(label="NPU", org="vllm", pipeline="vllm-omni-npu-ci")
ALL_BK_TARGETS: tuple[BkTarget, ...] = (CUDA_TARGET, NPU_TARGET)


def _buildkite_token() -> str | None:
    tok = (os.environ.get("BUILDKITE_API_TOKEN") or os.environ.get("BUILDKITE_TOKEN") or "").strip()
    return tok or None


# Inline SVG paths (24×24, stroke) for HTML report headings — no external assets.
_SVG_CLIPBOARD = (
    '<path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/>'
    '<rect x="8" y="2" width="8" height="4" rx="1" ry="1"/>'
)
_SVG_CLOUD = '<path d="M18 10h-1.26A8 8 0 1 0 9 22h9a5 5 0 1 0 0-12z"/>'
_SVG_SERVER = (
    '<rect x="2" y="2" width="20" height="8" rx="2" ry="2"/>'
    '<rect x="2" y="14" width="20" height="8" rx="2" ry="2"/>'
    '<line x1="6" y1="6" x2="6.01" y2="6"/><line x1="6" y1="18" x2="6.01" y2="18"/>'
)
_SVG_ALERT = (
    '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>'
    '<line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>'
)
_SVG_LIST = '<path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/>'
_SVG_CODE = '<polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/>'
_SVG_MSG = '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>'
_SVG_SPARK = '<path d="m12 3-1.9 5.8L4 10l5.8 1.9L12 18l1.9-5.8L20 10l-6.2-1.9L12 3z"/>'
_SVG_LOG = (
    '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>'
    '<polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/>'
    '<line x1="16" y1="17" x2="8" y2="17"/><line x1="10" y1="9" x2="8" y2="9"/>'
)
# Plus-in-circle (new issue)
_SVG_PLUS_ISSUE = (
    '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="16"/><line x1="8" y1="12" x2="16" y2="12"/>'
)
# Subcollapsible section icons (summary row)
_SVG_CHART_BARS = (
    '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>'
)

VLLM_OMNI_REPO = os.environ.get("VLLM_OMNI_ISSUE_REPO", "https://github.com/vllm-project/vllm-omni").strip().rstrip("/")
VLLM_OMNI_BUG_ISSUE_TEMPLATE = "400-bug-report.yml"
# GitHub issue form field ids — must match explicit `id` in vllm-omni
# `.github/ISSUE_TEMPLATE/400-bug-report.yml` (URL query prefill only works with `id`).
VLLM_OMNI_BUG_ENV_FIELD_ID = "current-environment"
VLLM_OMNI_BUG_ENV_COLLECT_PLACEHOLDER = "Your output of `python collect_env.py` here"
VLLM_OMNI_BUG_ENV_DEFAULT_VALUE = (
    "<details>\n"
    "<summary>The output of <code>python collect_env.py</code></summary>\n\n"
    "```text\n"
    f"{VLLM_OMNI_BUG_ENV_COLLECT_PLACEHOLDER}\n"
    "```\n\n"
    "</details>"
)
VLLM_OMNI_BUG_ENV_CI_REPLACEMENT = "ci env"
VLLM_OMNI_BUG_CODE_VERSION_FIELD_ID = "code-version"
VLLM_OMNI_BUG_DESCRIBE_FIELD_ID = "bug-description"
# GitHub issue labels (exact repo names). Local Submit issue: bug only; CI: bug + ci-failure + high priority.
VLLM_OMNI_BUG_ISSUE_LABELS_CI = ("bug", "ci-failure", "high priority")
VLLM_OMNI_BUG_ISSUE_LABELS_LOCAL = ("bug",)


def _vllm_omni_bug_env_ci_prefill() -> str:
    return VLLM_OMNI_BUG_ENV_DEFAULT_VALUE.replace(
        VLLM_OMNI_BUG_ENV_COLLECT_PLACEHOLDER,
        VLLM_OMNI_BUG_ENV_CI_REPLACEMENT,
    )


# Total raw log bytes (per failing local job) embeddable in HTML; larger logs get a notice + paths only.
FULL_LOG_EMBED_MAX_BYTES = 2 * 1024 * 1024
DEFAULT_KANBAN_ASSETS_DIR = (
    Path(os.environ.get("KANBAN_ASSETS_DIR", "").strip()).resolve()
    if os.environ.get("KANBAN_ASSETS_DIR", "").strip()
    else None
)
DEFAULT_KANBAN_REPO_ROOT = resolve_kanban_repo_root()


@dataclass
class KanbanAssetsConfig:
    assets_dir: Path | None
    repo_root: Path | None
    expected_remote: str | None = None
    expected_branch: str | None = None
    raw_root: Path | None = None
    refresh_from_raw: bool = False
    refresh_note: str | None = None
    refresh_warnings: list[str] = field(default_factory=list)


# Keep this list aligned with https://github.com/hsliuustc0106/vllm-omni-kanban/blob/main/scripts/mkdocs_hooks.py and
# the maintenance note in SKILL.md.
KANBAN_RAW_MODEL_SYNCS: tuple[tuple[str, str], ...] = (
    ("qwen3omni", "qwen3_omni"),
    ("qwen3tts", "qwen3_tts"),
    ("qwen_image", "qwen_image"),
    ("qwen_image_edit", "qwen_image_edit"),
    ("qwen_image_edit_2509", "qwen_image_edit_2509"),
    ("wan22", "wan22"),
)
KANBAN_RAW_PATTERNS = (
    "result_test_*.json",
    "diffusion_result_*.json",
    "benchmark_results_*.json",
)


def _svg_icon(inner: str, *, size: int = 20, extra_class: str = "") -> str:
    c = f"ico {extra_class}".strip()
    return (
        f'<svg class="{c}" width="{size}" height="{size}" viewBox="0 0 24 24" '
        'aria-hidden="true" focusable="false" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        f"{inner}</svg>"
    )


def _heading_html(
    tag: str,
    icon_paths: str,
    label_inner: str,
    *,
    sub: str | None = None,
    klass: str | None = None,
) -> str:
    """`label_inner` / `sub` are trusted HTML fragments (callers must escape)."""
    attrs = ""
    if klass:
        attrs = f' class="{html.escape(klass)}"'
    sub_html = ""
    if sub:
        sub_html = f'<span class="heading-sub">{sub}</span>'
    return (
        f"<{tag}{attrs}>"
        '<span class="heading-row">'
        f'<span class="heading-ico">{_svg_icon(icon_paths, size=22)}</span>'
        f'<span class="heading-text"><span class="heading-label">{label_inner}</span>{sub_html}</span>'
        "</span>"
        f"</{tag}>"
    )


def _table_wrap(table_html: str) -> str:
    return f'<div class="table-scroll">{table_html}</div>'


def _fail_status_cell_html(row_id: str, *, label: str = "Status") -> str:
    """Render a **Status** cell with two interactive buttons for HTML failure tables.

    Initial state shows two buttons:
      * ``Filed`` (Filed as bug) — opens the in-page modal to enter the GitHub
        issue number (and optional note). Replaces the old ``window.prompt()``
        approach so the cell stays editable when the report is embedded via
        iframe (where ``prompt`` is blocked by ``sandbox``).
      * ``Not an issue`` (Not an issue) — opens the modal for the user to fill in an
        optional explanation note before saving.

    State (status + issue number + note) is persisted per ``row_id`` in
    ``localStorage`` so reloads keep the chosen state.
    """
    row_id_attr = html.escape(row_id, quote=True)
    return (
        f'<td class="fail-status-cell" data-row-id="{row_id_attr}" data-status="unset">'
        '<span class="fail-status-buttons">'
        '<button type="button" class="fail-status-btn fail-status-btn--filed" '
        'data-status-action="filed">Filed</button>'
        '<button type="button" class="fail-status-btn fail-status-btn--not-issue" '
        'data-status-action="not-issue">Not an issue</button>'
        "</span></td>"
    )


def _fail_status_modal_html() -> str:
    """In-page modal used by ``_fail_status_submit_script``.

    Replaces the legacy ``window.prompt()`` flow so the cell stays editable when
    the report is embedded via iframe on the kanban Reports page (where
    ``prompt`` is blocked by the sandbox). Single modal handles both flows
    (Filed = enter issue number + note; Not an issue = enter note only).
    """
    return """
<div id="fail-status-modal" class="fail-status-modal" hidden role="dialog"
     aria-modal="true" aria-labelledby="fail-status-modal-title">
  <div class="fail-status-modal-backdrop" data-fsm-close aria-hidden="true"></div>
  <div class="fail-status-modal-panel">
    <header class="fail-status-modal-header">
      <h2 id="fail-status-modal-title" data-fsm-title>Mark status</h2>
      <button type="button" class="fail-status-modal-close"
              data-fsm-close aria-label="Close">&times;</button>
    </header>
    <div class="fail-status-modal-body">
      <label class="fail-status-modal-field" data-fsm-field-issue>
        <span class="fail-status-modal-label">GitHub issue number</span>
        <input type="text" class="fail-status-modal-input"
               data-fsm-input-issue placeholder="e.g. 123" />
        <span class="fail-status-modal-hint">
          ((Leave blank to save note only, no linked issue))
        </span>
      </label>
      <label class="fail-status-modal-field">
        <span class="fail-status-modal-label">Note (optional)</span>
        <textarea class="fail-status-modal-textarea"
                  data-fsm-input-note rows="4"
                  placeholder="e.g. known flaky, fixed in PR #456 / config issue, will track ..."></textarea>
      </label>
    </div>
    <footer class="fail-status-modal-footer">
      <button type="button" class="fail-status-modal-cancel"
              data-fsm-close>Cancel</button>
      <button type="button" class="fail-status-modal-save"
              data-fsm-save>Save</button>
    </footer>
  </div>
</div>
"""


def _fail_status_submit_script() -> str:
    """Client script: handle two-button Status cells in failure-analysis tables.

    Both buttons (``Filed`` / ``Not an issue``) open the in-page modal
    ``#fail-status-modal``. The modal flow replaces the legacy ``window.prompt()``
    so the cell stays editable when the report is embedded via iframe on the
    kanban Reports page (where ``prompt`` is blocked by ``sandbox``).

    Behaviour:
      * Initial: cell shows two buttons (``Filed`` / ``Not an issue``).
      * Click ``Filed`` -> opens the modal pre-filled with the saved issue
        number + note; user can edit the issue number (or leave it blank for
        note-only) and add a note; clicking Save persists ``status=filed``
        along with the entered issue number + note.
      * Click ``Not an issue`` -> opens the modal with the issue number field hidden
        (note-only); clicking Save persists ``status=not-issue`` + note.
      * Click on the displayed status link / pill -> re-opens the modal to edit.
      * Click Reset -> clears localStorage and returns to the two-button state.

    State is keyed by the row's ``data-row-id`` attribute and stored in
    ``localStorage`` so reloading the report keeps the chosen status.
    """
    return """
<script>
(function () {
  function escHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return ({"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"})[c];
    });
  }
  function escAttr(s) {
    return String(s).replace(/"/g, "&quot;");
  }
  function repoIssueUrl(n) {
    return "https://github.com/vllm-project/vllm-omni/issues/" + encodeURIComponent(n);
  }
  function lsKey(rowId) { return "fail-status:" + rowId; }
  function saveStatus(rowId, payload) {
    try {
      if (payload && (payload.status || payload.issue || payload.note)) {
        localStorage.setItem(lsKey(rowId), JSON.stringify(payload));
      } else {
        localStorage.removeItem(lsKey(rowId));
      }
    } catch (e) { /* localStorage unavailable */ }
  }
  function loadStatus(rowId) {
    try {
      var raw = localStorage.getItem(lsKey(rowId));
      return raw ? JSON.parse(raw) : null;
    } catch (e) { return null; }
  }
  function renderUnset(cell) {
    cell.setAttribute("data-status", "unset");
    cell.removeAttribute("data-status-issue");
    cell.removeAttribute("data-status-note");
    cell.innerHTML =
      '<span class="fail-status-buttons">' +
      '<button type="button" class="fail-status-btn fail-status-btn--filed" data-status-action="filed">Filed</button>' +
      '<button type="button" class="fail-status-btn fail-status-btn--not-issue"'
      + ' data-status-action="not-issue">Not an issue</button>' +
      '</span>';
  }
  function renderFiled(cell, issue, note) {
    cell.setAttribute("data-status", "filed");
    cell.setAttribute("data-status-issue", issue || "");
    cell.setAttribute("data-status-note", note || "");
    var issueHtml = issue
      ? '<a class="fail-status-issue-link" href="' + repoIssueUrl(issue) + '"'
        + ' target="_blank" rel="noopener">Filed #' + escHtml(issue) + '</a>'
      : '<span class="fail-status-issue-link">Filed</span>';
    var noteHtml = note ? '<span class="fail-status-note"> — ' + escHtml(note) + '</span>' : '';
    cell.innerHTML =
      '<span class="fail-status-display fail-status-display--filed">' +
      issueHtml + noteHtml +
      '</span>' +
      '<button type="button" class="fail-status-reset" data-status-action="reset">Reset</button>';
  }
  function renderNotIssue(cell, note) {
    cell.setAttribute("data-status", "not-issue");
    cell.setAttribute("data-status-issue", "");
    cell.setAttribute("data-status-note", note || "");
    var noteHtml = note ? '<span class="fail-status-note"> — ' + escHtml(note) + '</span>' : '';
    cell.innerHTML =
      '<span class="fail-status-display fail-status-display--not-issue">Not an issue' + noteHtml + '</span>' +
      '<button type="button" class="fail-status-reset" data-status-action="reset">Reset</button>';
  }

  var modal = document.getElementById("fail-status-modal");
  var modalTitle = modal && modal.querySelector("[data-fsm-title]");
  var modalIssueField = modal && modal.querySelector("[data-fsm-field-issue]");
  var modalIssueInput = modal && modal.querySelector("[data-fsm-input-issue]");
  var modalNoteInput = modal && modal.querySelector("[data-fsm-input-note]");
  var activeCell = null;
  var activeMode = null;  // "filed" | "not-issue"

  function openModal(cell, mode) {
    if (!modal) return;
    activeCell = cell;
    activeMode = mode;
    var saved = loadStatus(cell.getAttribute("data-row-id")) || {};
    if (modalTitle) {
      modalTitle.textContent = mode === "filed" ? "Mark as filed" : "Mark as not an issue";
    }
    if (modalIssueField) {
      // Hide the issue number field entirely for "not-issue" mode.
      modalIssueField.style.display = (mode === "filed") ? "" : "none";
    }
    if (modalIssueInput) {
      modalIssueInput.value = (mode === "filed") ? (saved.issue || "") : "";
    }
    if (modalNoteInput) {
      modalNoteInput.value = saved.note || "";
    }
    modal.hidden = false;
    document.body.classList.add("fail-status-modal-open");
    if (mode === "filed" && modalIssueInput) {
      modalIssueInput.focus();
    } else if (modalNoteInput) {
      modalNoteInput.focus();
    }
  }
  function closeModal() {
    if (!modal) return;
    modal.hidden = true;
    document.body.classList.remove("fail-status-modal-open");
    activeCell = null;
    activeMode = null;
  }
  function commitModal() {
    if (!activeCell) return;
    var issue = activeMode === "filed"
      ? (modalIssueInput ? String(modalIssueInput.value || "").trim() : "")
      : "";
    var note = modalNoteInput ? String(modalNoteInput.value || "").trim() : "";
    var rowId = activeCell.getAttribute("data-row-id");
    if (activeMode === "filed") {
      renderFiled(activeCell, issue, note);
      saveStatus(rowId, {status: "filed", issue: issue, note: note});
    } else if (activeMode === "not-issue") {
      renderNotIssue(activeCell, note);
      saveStatus(rowId, {status: "not-issue", issue: "", note: note});
    }
    closeModal();
  }
  if (modal) {
    modal.addEventListener("click", function (ev) {
      if (ev.target.closest && ev.target.closest("[data-fsm-close]")) {
        ev.preventDefault();
        closeModal();
      } else if (ev.target.closest && ev.target.closest("[data-fsm-save]")) {
        ev.preventDefault();
        commitModal();
      }
    });
  }
  document.addEventListener("keydown", function (ev) {
    if (ev.key === "Escape" && modal && !modal.hidden) {
      ev.preventDefault();
      closeModal();
    } else if (ev.key === "Enter" && (ev.metaKey || ev.ctrlKey) && modal && !modal.hidden) {
      ev.preventDefault();
      commitModal();
    }
  });
  document.addEventListener("click", function (ev) {
    var target = ev.target;
    if (!target || !target.closest) return;
    // Reset button
    var resetBtn = target.closest(".fail-status-reset");
    if (resetBtn) {
      var cell = resetBtn.closest(".fail-status-cell");
      if (!cell) return;
      var rowId = cell.getAttribute("data-row-id");
      saveStatus(rowId, null);
      renderUnset(cell);
      return;
    }
    // Initial two-button row
    var btn = target.closest(".fail-status-btn");
    if (!btn) {
      // Click an existing "filed #N" link — open the modal to edit
      var linkBtn = target.closest(".fail-status-display--filed");
      if (linkBtn) {
        var cell2 = linkBtn.closest(".fail-status-cell");
        if (cell2) {
          ev.preventDefault();
          openModal(cell2, "filed");
        }
        return;
      }
      var notIssueBtn = target.closest(".fail-status-display--not-issue");
      if (notIssueBtn) {
        var cell3 = notIssueBtn.closest(".fail-status-cell");
        if (cell3) {
          ev.preventDefault();
          openModal(cell3, "not-issue");
        }
        return;
      }
      return;
    }
    var cell = btn.closest(".fail-status-cell");
    if (!cell) return;
    var action = btn.getAttribute("data-status-action");
    if (action === "filed") {
      openModal(cell, "filed");
    } else if (action === "not-issue") {
      openModal(cell, "not-issue");
    }
  });
  function restoreSaved() {
    document.querySelectorAll(".fail-status-cell[data-row-id]").forEach(function (cell) {
      var rowId = cell.getAttribute("data-row-id");
      var saved = loadStatus(rowId);
      if (!saved) return;
      if (saved.status === "filed") {
        renderFiled(cell, saved.issue || "", saved.note || "");
      } else if (saved.status === "not-issue") {
        renderNotIssue(cell, saved.note || "");
      }
    });
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", restoreSaved);
  } else {
    restoreSaved();
  }
  restoreSaved();
})();
</script>"""


def _ut_coverage_cell_html(row_id: str, *, initial_value: str = "—") -> str:
    """Render the UT coverage **Result** cell (inline, click-to-edit via modal).

    The cell renders **only the value** (no inline editor buttons, no saved
    hint, no reset) so the row reads the same as every other Result cell in
    the Development Metrics overview snapshot. Clicking the value opens the
    in-page modal ``#ut-coverage-modal`` (same pattern as the failure-analysis
    fail-status modal — works in iframes where ``window.prompt`` is blocked).
    The modal owns Save / Cancel / Reset so the cell stays clean.
    """
    row_id_attr = html.escape(row_id, quote=True)
    init = html.escape(initial_value)
    return (
        '<span class="ut-coverage-cell" data-row-id="' + row_id_attr + '" data-original="' + init + '">'
        '<button type="button" class="ut-coverage-btn" data-ut-action="edit">' + init + "</button>"
        "</span>"
    )


def _ut_coverage_modal_html() -> str:
    """In-page modal used by ``_ut_coverage_submit_script``.

    Single ``<input>`` for the coverage value + footer with Reset / Cancel / Save
    (Reset is moved into the modal so the cell itself stays clean — only the
    value button is rendered there). Same pattern as the fail-status modal in
    failure analysis: works in iframe contexts where ``window.prompt`` is
    blocked by ``sandbox``.
    """
    return """
<div id="ut-coverage-modal" class="ut-coverage-modal" hidden role="dialog"
     aria-modal="true" aria-labelledby="ut-coverage-modal-title">
  <div class="ut-coverage-modal-backdrop" data-ucm-close aria-hidden="true"></div>
  <div class="ut-coverage-modal-panel">
    <header class="ut-coverage-modal-header">
      <h2 id="ut-coverage-modal-title">Edit UT coverage</h2>
      <button type="button" class="ut-coverage-modal-close"
              data-ucm-close aria-label="Close">&times;</button>
    </header>
    <div class="ut-coverage-modal-body">
      <label class="ut-coverage-modal-field">
        <span class="ut-coverage-modal-label">Unit Test coverage</span>
        <input type="text" class="ut-coverage-modal-input"
               data-ucm-input placeholder="e.g. 84.6%" />
        <span class="ut-coverage-modal-hint">
          ((Leave blank to show placeholder "—"; click Reset to clear the saved value))
        </span>
      </label>
    </div>
    <footer class="ut-coverage-modal-footer">
      <button type="button" class="ut-coverage-modal-reset"
              data-ucm-reset>Reset</button>
      <span class="ut-coverage-modal-footer-spacer"></span>
      <button type="button" class="ut-coverage-modal-cancel"
              data-ucm-close>Cancel</button>
      <button type="button" class="ut-coverage-modal-save"
              data-ucm-save>Save</button>
    </footer>
  </div>
</div>
"""


def _ut_coverage_submit_script() -> str:
    """Client script: handle UT coverage cells (edit via in-page modal).

    The cell itself renders **only the value** — no inline Save / Cancel / Reset
    buttons. Clicking the value opens the modal ``#ut-coverage-modal`` which
    owns Save / Cancel / Reset (same pattern as the fail-status modal in
    failure analysis: works in iframes where ``window.prompt`` is blocked by
    the sandbox).

    Behaviour:
      * Click value -> opens the modal pre-filled with the saved value (or
        empty if it is the original placeholder).
      * Click modal Save (or press Enter) -> persist the new value to
        ``localStorage[ut-coverage:<row_id>]`` and update the cell.
      * Click modal Cancel / Esc / backdrop / X -> close without saving.
      * Click modal Reset -> clear localStorage, revert the cell to the
        original placeholder, close the modal.

    State is keyed by the row's ``data-row-id`` attribute and stored in
    ``localStorage`` so reloading the report keeps the saved value.
    """
    return """
<script>
(function () {
  function lsKey(rowId) { return "ut-coverage:" + rowId; }
  function saveValue(rowId, value) {
    try {
      if (value) {
        localStorage.setItem(lsKey(rowId), value);
      } else {
        localStorage.removeItem(lsKey(rowId));
      }
    } catch (e) { /* localStorage unavailable */ }
  }
  function loadValue(rowId) {
    try {
      return localStorage.getItem(lsKey(rowId));
    } catch (e) { return null; }
  }
  function setCellValue(cell, value) {
    var btn = cell.querySelector(".ut-coverage-btn");
    if (btn) btn.textContent = value;
  }

  var modal = document.getElementById("ut-coverage-modal");
  var modalInput = modal && modal.querySelector("[data-ucm-input]");
  var activeCell = null;

  function openModal(cell) {
    if (!modal) return;
    activeCell = cell;
    var rowId = cell.getAttribute("data-row-id");
    var saved = loadValue(rowId) || cell.getAttribute("data-original");
    if (modalInput) {
      modalInput.value = (saved === cell.getAttribute("data-original") || saved === "—") ? "" : saved;
    }
    modal.hidden = false;
    document.body.classList.add("ut-coverage-modal-open");
    if (modalInput) {
      modalInput.focus();
      modalInput.select();
    }
  }
  function closeModal() {
    if (!modal) return;
    modal.hidden = true;
    document.body.classList.remove("ut-coverage-modal-open");
    activeCell = null;
  }
  function commitModal() {
    if (!activeCell) return;
    var v = (modalInput && modalInput.value || "").trim() || "—";
    var rowId = activeCell.getAttribute("data-row-id");
    saveValue(rowId, v === "—" ? "" : v);
    setCellValue(activeCell, v);
    closeModal();
  }
  function resetActive() {
    if (!activeCell) return;
    var rowId = activeCell.getAttribute("data-row-id");
    saveValue(rowId, "");
    setCellValue(activeCell, activeCell.getAttribute("data-original"));
    closeModal();
  }
  if (modal) {
    modal.addEventListener("click", function (ev) {
      if (ev.target.closest && ev.target.closest("[data-ucm-close]")) {
        ev.preventDefault();
        closeModal();
      } else if (ev.target.closest && ev.target.closest("[data-ucm-save]")) {
        ev.preventDefault();
        commitModal();
      } else if (ev.target.closest && ev.target.closest("[data-ucm-reset]")) {
        ev.preventDefault();
        resetActive();
      }
    });
  }
  document.addEventListener("keydown", function (ev) {
    if (ev.key === "Escape" && modal && !modal.hidden) {
      ev.preventDefault();
      closeModal();
    } else if (ev.key === "Enter" && (ev.metaKey || ev.ctrlKey) && modal && !modal.hidden) {
      ev.preventDefault();
      commitModal();
    } else if (ev.key === "Enter" && modal && !modal.hidden && ev.target === modalInput) {
      ev.preventDefault();
      commitModal();
    }
  });
  document.addEventListener("click", function (ev) {
    var target = ev.target;
    if (!target || !target.closest) return;
    var cell = target.closest(".ut-coverage-cell");
    if (!cell) return;
    var action = target.getAttribute("data-ut-action");
    if (action === "edit") {
      openModal(cell);
    }
  });
  function restoreSaved() {
    document.querySelectorAll(".ut-coverage-cell[data-row-id]").forEach(function (cell) {
      var rowId = cell.getAttribute("data-row-id");
      var saved = loadValue(rowId);
      if (saved) {
        setCellValue(cell, saved);
      }
    });
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", restoreSaved);
  } else {
    restoreSaved();
  }
  restoreSaved();
})();
</script>"""


def _details_subcard(
    title: str,
    body_html: str,
    *,
    open_default: bool = False,
    details_class: str = "",
    icon_paths: str | None = None,
) -> str:
    """Collapsible sub-section inside Buildkite Test / Local Test cards."""
    op = " open" if open_default else ""
    extra = f" {details_class.strip()}" if details_class.strip() else ""
    te = html.escape(title)
    if icon_paths:
        ico = _svg_icon(icon_paths, size=18, extra_class="report-subcard-ico")
        label = f'<span class="report-subcard-summary-inner">{ico}<span class="report-subcard-title">{te}</span></span>'
    else:
        label = f'<span class="report-subcard-title">{te}</span>'
    return (
        f'<details class="report-subcard{extra}"{op}>'
        f'<summary class="report-subcard-summary">{label}</summary>'
        f'<div class="report-subcard-body">{body_html}</div>'
        "</details>"
    )


def _github_issue_button_cell() -> str:
    return (
        '<td class="issue-action-cell">'
        '<button type="button" class="btn-github-issue">'
        f"{_svg_icon(_SVG_PLUS_ISSUE, size=17, extra_class='btn-issue-ico')}"
        '<span class="btn-issue-text">Submit issue</span>'
        "</button></td>"
    )


def _github_issue_submit_script() -> str:
    """Client script: Submit issue opens GitHub bug template with all fields prefilled via URL."""
    issue_new = f"{VLLM_OMNI_REPO}/issues/new"
    return f"""
<script>
(function () {{
  var issueBase = {json.dumps(issue_new)};
  var bugTemplate = {json.dumps(VLLM_OMNI_BUG_ISSUE_TEMPLATE)};
  var issueEnvFieldId = {json.dumps(VLLM_OMNI_BUG_ENV_FIELD_ID)};
  var issueEnvCiValue = {json.dumps(_vllm_omni_bug_env_ci_prefill())};
  var issueEnvLocalValue = {json.dumps(VLLM_OMNI_BUG_ENV_DEFAULT_VALUE)};
  var issueCodeVersionFieldId = {json.dumps(VLLM_OMNI_BUG_CODE_VERSION_FIELD_ID)};
  var issueDescribeFieldId = {json.dumps(VLLM_OMNI_BUG_DESCRIBE_FIELD_ID)};
  var issueLabelsCi = {json.dumps(list(VLLM_OMNI_BUG_ISSUE_LABELS_CI))};
  var issueLabelsLocal = {json.dumps(list(VLLM_OMNI_BUG_ISSUE_LABELS_LOCAL))};
  var maxIssueUrlLen = 7800;

  function issueLabelsFor(d) {{
    return d.env === "ci" ? issueLabelsCi : issueLabelsLocal;
  }}

  function applyIssueLabels(u, labels) {{
    if (!labels || !labels.length) return;
    u.searchParams.set("labels", labels.join(","));
  }}

  function finalizeIssueUrl(u) {{
    // URLSearchParams encodes spaces as '+'; GitHub expects '%20' in issue URLs.
    return u.toString().replace(/\\+/g, "%20");
  }}

  function resolveVersionLines(d) {{
    if (d.env === "ci") {{
      var vllmLine = (d.vllmVer && d.vllmVer.trim()) ? d.vllmVer.trim() : "(not found in Buildkite step log)";
      var omniLine;
      if (d.omniVer && d.omniVer.trim()) {{
        omniLine = d.omniVer.trim();
      }} else if (d.buildCommit && d.buildCommit.trim()) {{
        omniLine = d.buildCommit.trim();
      }} else {{
        omniLine = "(not found in Buildkite step log)";
      }}
      return {{ vllmLine: vllmLine, omniLine: omniLine }};
    }}
    return {{ vllmLine: "(pending)", omniLine: "(pending)" }};
  }}

  function buildCodeVersionFieldValue(d) {{
    var ver = resolveVersionLines(d);
    return [
      "<details>",
      "<summary>The commit id or version of vllm</summary>",
      "",
      "```text",
      ver.vllmLine,
      "```",
      "</details>",
      "<details>",
      "<summary>The commit id or version of vllm-omni</summary>",
      "",
      "```text",
      ver.omniLine,
      "```",
      "</details>",
    ].join("\\n");
  }}

  function buildEnvFieldValue(d) {{
    return d.env === "ci" ? issueEnvCiValue : issueEnvLocalValue;
  }}

  function buildDescribeFieldValue(d, excerptOverride) {{
    var kind = d.isErr ? "pytest ERROR" : "pytest FAILED";
    var excerpt = excerptOverride !== undefined ? excerptOverride : d.excerpt;
    var lines = [];
    if (d.bkBuildUrl && d.bkBuildUrl.trim()) {{
      lines.push("**Buildkite build:** " + d.bkBuildUrl.trim());
    }}
    if (d.bkStepUrl && d.bkStepUrl.trim()) {{
      var stepUrl = d.bkStepUrl.trim();
      var stepName = (d.bkStepName && d.bkStepName.trim()) ? d.bkStepName.trim() : "";
      if (stepName) {{
        lines.push("**Buildkite step:** [" + stepName + "](" + stepUrl + ")");
      }} else {{
        lines.push("**Buildkite step:** " + stepUrl);
      }}
    }}
    if (lines.length) lines.push("");
    lines.push("**Failure kind:** " + kind);
    lines.push("**Test node:** `" + d.node + "`");
    lines.push("");
    lines.push("**Log reason:**");
    lines.push((d.reason && d.reason.trim()) ? d.reason.trim() : "(none in report)");
    lines.push("");
    lines.push("**Analysis:**");
    lines.push((d.analysis && d.analysis.trim()) ? d.analysis.trim() : "(none in report)");
    lines.push("");
    lines.push("**Error log excerpt:**");
    lines.push("");
    lines.push("```text");
    lines.push(excerpt || "(empty)");
    lines.push("```");
    lines.push("");
    lines.push("---");
    lines.push(
      "*Generated from a nightly HTML report. Redact secrets before submitting; "
      + "complete the checkboxes on GitHub.*"
    );
    return lines.join("\\n");
  }}

  function issueTitle(d) {{
    var n = d.node.replace(/\\s*\\(ERROR\\)\\s*$/i, "");
    var t = "[Bug]: Nightly / CI failed - " + n;
    return t.length > 220 ? t.slice(0, 217) + "..." : t;
  }}

  function buildIssueUrl(d, opts) {{
    opts = opts || {{}};
    var u = new URL(issueBase);
    u.searchParams.set("template", bugTemplate);
    u.searchParams.set("title", issueTitle(d));
    applyIssueLabels(u, issueLabelsFor(d));
    u.searchParams.set(issueEnvFieldId, buildEnvFieldValue(d));
    u.searchParams.set(issueCodeVersionFieldId, buildCodeVersionFieldValue(d));
    u.searchParams.set(
      issueDescribeFieldId,
      buildDescribeFieldValue(d, opts.excerpt)
    );
    return u;
  }}

  function submitIssue(d) {{
    var url = finalizeIssueUrl(buildIssueUrl(d));
    if (url.length > maxIssueUrlLen) {{
      var over = url.length - maxIssueUrlLen + 220;
      var raw = d.excerpt || "";
      var maxExcerpt = Math.max(400, raw.length - over);
      var truncated =
        raw.slice(0, maxExcerpt) +
        "\\n\\n...(truncated for GitHub URL length; see HTML report for full log)";
      url = finalizeIssueUrl(buildIssueUrl(d, {{ excerpt: truncated }}));
    }}
    if (url.length > maxIssueUrlLen && d.analysis) {{
      var over2 = url.length - maxIssueUrlLen + 120;
      var aRaw = d.analysis || "";
      var maxAnalysis = Math.max(200, aRaw.length - over2);
      d = Object.assign({{}}, d, {{
        analysis: aRaw.slice(0, maxAnalysis) + "\\n\\n...(analysis truncated; see HTML report)",
      }});
      url = finalizeIssueUrl(buildIssueUrl(d));
    }}
    window.open(url, "_blank", "noopener,noreferrer");
  }}

  function gatherRow(btn) {{
    var tr = btn.closest("tr");
    if (!tr || !tr.cells || tr.cells.length < 5) return null;
    var ctx = tr.getAttribute("data-report-context") || "";
    var node = tr.cells[0].innerText.trim();
    var reason = tr.cells[1].innerText.trim();
    var analysis = tr.cells[2].innerText.trim();
    var pre = tr.cells[3].querySelector(".log-excerpt--stored, .log-excerpt");
    var excerpt = pre ? pre.innerText : "";
    var isErr = node.indexOf("(ERROR)") !== -1;
    var env = tr.getAttribute("data-issue-env") || "local";
    var vllmVer = tr.getAttribute("data-vllm-version") || "";
    var omniVer = tr.getAttribute("data-vllm-omni-version") || "";
    var buildCommit = tr.getAttribute("data-build-commit") || "";
    var bkBuildUrl = tr.getAttribute("data-buildkite-build-url") || "";
    var bkStepUrl = tr.getAttribute("data-buildkite-step-url") || "";
    var bkStepName = tr.getAttribute("data-buildkite-step-name") || "";
    return {{
      ctx: ctx, node: node, reason: reason, analysis: analysis, excerpt: excerpt, isErr: isErr,
      env: env, vllmVer: vllmVer, omniVer: omniVer, buildCommit: buildCommit,
      bkBuildUrl: bkBuildUrl, bkStepUrl: bkStepUrl, bkStepName: bkStepName,
    }};
  }}

  document.addEventListener("click", function (ev) {{
    var b = ev.target.closest && ev.target.closest(".btn-github-issue");
    if (b) {{
      ev.preventDefault();
      var d = gatherRow(b);
      if (d) submitIssue(d);
    }}
  }});

  var logExcerptModal = document.getElementById("log-excerpt-modal");
  function openLogExcerptModal(btn) {{
    if (!logExcerptModal) return;
    var id = btn.getAttribute("data-modal-target");
    var stored = id ? document.getElementById(id) : null;
    if (!stored) return;
    var titleEl = logExcerptModal.querySelector("[data-log-modal-title]");
    var bodyPre = logExcerptModal.querySelector("[data-log-modal-pre]");
    if (titleEl) {{
      titleEl.textContent = btn.getAttribute("data-log-title") || "Log excerpt";
    }}
    if (bodyPre) {{
      bodyPre.textContent = stored.textContent;
    }}
    logExcerptModal.hidden = false;
    document.body.classList.add("log-modal-open");
  }}
  function closeLogExcerptModal() {{
    if (!logExcerptModal) return;
    logExcerptModal.hidden = true;
    document.body.classList.remove("log-modal-open");
  }}
  document.addEventListener("click", function (ev) {{
    var openBtn = ev.target.closest && ev.target.closest(".btn-view-log-excerpt");
    if (openBtn) {{
      ev.preventDefault();
      openLogExcerptModal(openBtn);
      return;
    }}
    if (ev.target.closest && ev.target.closest("[data-log-modal-close]")) {{
      ev.preventDefault();
      closeLogExcerptModal();
      return;
    }}
  }});
  document.addEventListener("keydown", function (ev) {{
    if (ev.key === "Escape" && logExcerptModal && !logExcerptModal.hidden) {{
      closeLogExcerptModal();
    }}
  }});

  function applyPerfFilters(scope) {{
    var controls = Array.prototype.slice.call(
      scope.querySelectorAll("select[data-filter-key], input[data-filter-key]")
    );
    var rows = scope.querySelectorAll('tr[data-perf-row="1"]');
    var empty = scope.querySelector("[data-perf-empty]");
    if (!controls.length || !rows.length) return;

    var scalarControls = [];
    var checkboxGroups = {{}};
    controls.forEach(function (control) {{
      var key = control.getAttribute("data-filter-key") || "";
      if (!key) return;
      if (control.type === "checkbox") {{
        if (!checkboxGroups[key]) checkboxGroups[key] = [];
        if (control.checked) checkboxGroups[key].push(control.value || "");
      }} else {{
        scalarControls.push(control);
      }}
    }});

    var visibleCount = 0;
    rows.forEach(function (row) {{
      var ok = true;
      scalarControls.forEach(function (control) {{
        var key = control.getAttribute("data-filter-key") || "";
        var val = (control.value || "").trim();
        var rowVal = row.getAttribute("data-" + key) || "";
        if (!val) return;
        if (control.tagName.toLowerCase() === "input") {{
          if (rowVal.toLowerCase().indexOf(val.toLowerCase()) === -1) ok = false;
        }} else if (rowVal !== val) {{
          ok = false;
        }}
      }});
      Object.keys(checkboxGroups).forEach(function (key) {{
        var vals = checkboxGroups[key] || [];
        if (vals.length && vals.indexOf(row.getAttribute("data-" + key) || "") === -1) {{
          ok = false;
        }}
      }});
      row.hidden = !ok;
      if (ok) visibleCount += 1;
    }});
    if (empty) {{
      empty.hidden = visibleCount !== 0;
    }}
  }}

  document.querySelectorAll("[data-perf-filter-scope]").forEach(function (scope) {{
    scope.querySelectorAll("select[data-filter-key], input[data-filter-key]").forEach(function (control) {{
      var eventName = control.tagName.toLowerCase() === "input" && control.type !== "checkbox" ? "input" : "change";
      control.addEventListener(eventName, function () {{
        applyPerfFilters(scope);
      }});
    }});
    scope.querySelectorAll("[data-focus-expand]").forEach(function (btn) {{
      btn.addEventListener("click", function () {{
        var expanded = scope.classList.toggle("focus-filter-scope--expanded");
        btn.setAttribute("aria-expanded", expanded ? "true" : "false");
        btn.textContent = expanded ? "Collapse table" : "Expand table";
      }});
    }});
    applyPerfFilters(scope);
  }});
}})();
</script>
"""


def _md_cell(s: str) -> str:
    return (s or "").replace("|", "/").replace("\n", " ")


def render_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    cols = len(headers)
    if not headers:
        return ""
    for row in rows:
        if len(row) != cols:
            raise ValueError(f"row has {len(row)} cells, expected {cols}: {row!r}")
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * cols) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_html_table(
    headers: list[str],
    rows: list[list[str]],
    *,
    table_class: str = "",
    row_classes: list[str] | None = None,
    cell_suffixes: list[list[str]] | None = None,
) -> str:
    cls = f' class="{html.escape(table_class)}"' if table_class else ""
    parts = [f"<table{cls}>", "<thead><tr>"]
    for h in headers:
        parts.append(f"<th>{html.escape(h)}</th>")
    parts.append("</tr></thead><tbody>")
    for i, row in enumerate(rows):
        tr_attr = ""
        if row_classes and i < len(row_classes) and (row_classes[i] or "").strip():
            tr_attr = f' class="{html.escape(row_classes[i].strip())}"'
        parts.append(f"<tr{tr_attr}>")
        suffix_row = cell_suffixes[i] if cell_suffixes and i < len(cell_suffixes) else None
        for j, c in enumerate(row):
            suf = ""
            if suffix_row is not None and j < len(suffix_row):
                suf = (suffix_row[j] or "").strip()
            inner = html.escape(c)
            if suf:
                if suf.startswith("↑"):
                    dcls = "perf-delta perf-delta--up"
                elif suf.startswith("↓"):
                    dcls = "perf-delta perf-delta--down"
                else:
                    dcls = "perf-delta"
                inner += f' <span class="{dcls}">{html.escape(suf)}</span>'
            parts.append(f"<td>{inner}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "\n".join(parts)


def _summary_row_kind(info: dict[str, Any] | None) -> str:
    """``ok`` = no failures/errors; ``fail`` = failures; ``unknown`` = could not classify."""
    if not info:
        return "unknown"
    if info.get("failed_nodes") or info.get("error_nodes"):
        return "fail"
    counts = extract_pytest_counts(info.get("summary"))
    if counts["failed"] or counts["error"]:
        return "fail"
    summ = (info.get("summary") or "").strip()
    if not summ:
        return "unknown"
    if re.search(r"\d+\s+(?:passed|failed|skipped|errors?)\b", summ, re.I):
        return "ok"
    return "unknown"


def _summary_row_kind_bk(rec: dict[str, Any]) -> str:
    if rec.get("log_error"):
        return "fail"
    if not rec.get("raw_url"):
        return "unknown"
    return _summary_row_kind(rec.get("info"))


def _job_is_clean(info: dict[str, Any]) -> bool:
    return not info["failed_nodes"] and not info["error_nodes"]


def default_log_dir(repo_root: Path) -> Path:
    return repo_root / "logs" / "nightly_jobs"


def _combined_job_log_disk_bytes(paths: list[Path]) -> int | None:
    """Return total on-disk size of ``paths`` in bytes, or ``None`` if ``stat`` fails."""
    total = 0
    for p in paths:
        try:
            total += p.stat().st_size
        except OSError:
            return None
    return total


# Local Test summary: group jobs by pillar × dimension (folder / job name prefix).
_LOCAL_SUMMARY_PILLARS = ("Omni", "TTS", "Diffusion")
_LOCAL_SUMMARY_DIMS = ("Perf", "Acc", "Function", "doc", "stability")


def _classify_local_nightly_job_strict(job_name: str) -> tuple[str | None, str | None]:
    """
    Prefix form only: ``<pillar>_<dim>`` or ``<dim>_<pillar>`` at start of name
    (after normalizing spaces/hyphens to underscores).
    """
    n = job_name.strip()
    n = re.sub(r"[\s\-]+", "_", n)
    n = re.sub(r"_+", "_", n).lower()
    pillar_key: str | None = None
    dim_key: str | None = None
    m = re.match(
        r"^(omni|tts|diffusion|diff)_(perf|acc|function|doc|stability)(?:_|$)",
        n,
    )
    if m:
        pillar_key, dim_key = m.group(1), m.group(2)
    else:
        m2 = re.match(
            r"^(perf|acc|function|doc|stability)_(omni|tts|diffusion|diff)(?:_|$)",
            n,
        )
        if m2:
            dim_key, pillar_key = m2.group(1), m2.group(2)
    if not pillar_key or not dim_key:
        return (None, None)
    if pillar_key == "omni":
        pillar = "Omni"
    elif pillar_key == "tts":
        pillar = "TTS"
    elif pillar_key in ("diffusion", "diff"):
        pillar = "Diffusion"
    else:
        return (None, None)
    dim_map = {
        "perf": "Perf",
        "acc": "Acc",
        "function": "Function",
        "doc": "doc",
        "stability": "stability",
    }
    dim_label = dim_map.get(dim_key)
    if not dim_label:
        return (None, None)
    return (pillar, dim_label)


def _classify_local_nightly_job_keywords(job_name: str) -> tuple[str | None, str | None]:
    """
    Infer pillar × dim from tokens in the job folder / file stem name
    (e.g. ``full_moon_Diffusion_X2I_A_T_Accuracy_Test`` → Diffusion / Acc).
    """
    name_lower = job_name.strip().lower()

    pillar: str | None = None
    best_pi = len(name_lower) + 1
    for pat, plabel in (
        # Generic / umbrella diffusion tag — must come first so it wins the
        # leftmost-match race when both ``diffusion`` and a sub-model keyword
        # are present.
        ("diffusion", "Diffusion"),
        # Diffusion sub-models whose job names don't carry the ``Diffusion_``
        # prefix (e.g. ``full_moon_HunyuanImage3-DIT_Accuracy_Test``,
        # ``nightly-hunyuan-image3-performance``, ``qwen-image``, ``wan22``,
        # ``bagel``, ``glm-image``, ``longcat``, ``flux``). They all sit under
        # the Diffusion pillar in the Local Test summary.
        (r"hunyuan(?:[_]?image)?", "Diffusion"),
        (r"(?<![a-z0-9])qwen[-_]image(?![a-z0-9])", "Diffusion"),
        (r"(?<![a-z0-9])wan2\.?2?(?![a-z0-9])", "Diffusion"),
        (r"(?<![a-z0-9])wan(?![a-z0-9])", "Diffusion"),
        (r"(?<![a-z0-9])bagel(?![a-z0-9])", "Diffusion"),
        (r"(?<![a-z0-9])glm[-_]?image(?![a-z0-9])", "Diffusion"),
        (r"(?<![a-z0-9])longcat(?![a-z0-9])", "Diffusion"),
        (r"(?<![a-z0-9])flux(?![a-z0-9])", "Diffusion"),
        (r"(?<![a-z0-9])tts(?![a-z0-9])", "TTS"),
        (r"(?<![a-z0-9])omni(?![a-z0-9])", "Omni"),
    ):
        m = re.search(pat, name_lower)
        if m and m.start() < best_pi:
            best_pi = m.start()
            pillar = plabel

    dim: str | None = None
    best_di = len(name_lower) + 1
    # Longer / compound keywords first; leftmost match wins for dim
    for pat, dlabel in (
        (r"accuracy", "Acc"),
        (r"performance", "Perf"),
        (r"functional", "Function"),
        (r"(?<![a-z0-9])function(?![a-z0-9])", "Function"),
        (r"documentation", "doc"),
        (r"(?<![a-z0-9])docs(?![a-z0-9])", "doc"),
        (r"(?<![a-z0-9])doc(?![a-z0-9])", "doc"),
        (r"stability", "stability"),
        (r"(?<![a-z0-9])stable(?![a-z0-9])", "stability"),
        (r"(?<![a-z0-9])perf(?![a-z0-9])", "Perf"),
        (r"(?<![a-z0-9])acc(?![a-z0-9])", "Acc"),
    ):
        m = re.search(pat, name_lower)
        if m and m.start() < best_di:
            best_di = m.start()
            dim = dlabel

    if pillar and dim:
        return (pillar, dim)
    return (None, None)


def _classify_local_nightly_job(job_name: str) -> tuple[str | None, str | None]:
    """
    Map ``job_name`` (directory or flat file stem) to (pillar, dim).

    1. Strict prefix: ``<pillar>_<dim>`` or reverse (see `_classify_local_nightly_job_strict`).
    2. Keyword scan: ``diffusion`` / ``omni`` / ``tts`` and ``accuracy`` / ``perf`` / … anywhere in the name.
    """
    s = _classify_local_nightly_job_strict(job_name)
    if s[0] and s[1]:
        return s
    return _classify_local_nightly_job_keywords(job_name)


def _local_job_rows_with_info(
    groups: list[tuple[str, list[Path]]],
) -> list[tuple[str, list[Path], dict[str, Any]]]:
    out: list[tuple[str, list[Path], dict[str, Any]]] = []
    for job_name, paths in groups:
        text = read_job_text(paths)
        info = parse_pytest_log(text)
        out.append((job_name, paths, info))
    return out


def _render_local_summary_table_html(
    chunk: list[tuple[str, list[Path], dict[str, Any]]],
) -> str:
    summary_rows_loc: list[list[str]] = []
    summary_row_cls_loc: list[str] = []
    for job_name, paths, info in chunk:
        summary_rows_loc.append(_summary_row_for_job(job_name, paths, info))
        summary_row_cls_loc.append(f"summary-row summary-row--{_summary_row_kind(info)}")
    return _table_wrap(
        render_html_table(
            [
                "Job",
                "Total",
                "Passed",
                "Failed",
                "Skipped",
                "Errors",
                "Elapsed time",
            ],
            summary_rows_loc,
            table_class="summary",
            row_classes=summary_row_cls_loc,
        )
    )


def _render_local_summary_grouped_html(
    job_rows: list[tuple[str, list[Path], dict[str, Any]]],
) -> str:
    buckets: dict[str, dict[str, list[tuple[str, list[Path], dict[str, Any]]]]] = defaultdict(lambda: defaultdict(list))
    uncat: list[tuple[str, list[Path], dict[str, Any]]] = []
    for job_name, paths, info in job_rows:
        pillar, dim = _classify_local_nightly_job(job_name)
        if pillar and dim:
            buckets[pillar][dim].append((job_name, paths, info))
        else:
            uncat.append((job_name, paths, info))

    parts: list[str] = []
    for pillar in _LOCAL_SUMMARY_PILLARS:
        dim_map = buckets.get(pillar) or {}
        if not any(dim_map.get(d) for d in _LOCAL_SUMMARY_DIMS):
            continue
        dim_blocks: list[str] = []
        for dim in _LOCAL_SUMMARY_DIMS:
            chunk = dim_map.get(dim) or []
            if not chunk:
                continue
            chunk.sort(key=lambda t: t[0].lower())
            tbl = _render_local_summary_table_html(chunk)
            dim_blocks.append(
                "\n".join(
                    [
                        '<details class="local-summary-dim">',
                        f'<summary class="local-summary-dim-summary">{html.escape(dim)}</summary>',
                        f'<div class="local-summary-dim-body">{tbl}</div>',
                        "</details>",
                    ]
                )
            )
        parts.append(
            "\n".join(
                [
                    '<details class="local-summary-pillar">',
                    f'<summary class="local-summary-pillar-summary">{html.escape(pillar)}</summary>',
                    '<div class="local-summary-pillar-body">',
                    "\n".join(dim_blocks),
                    "</div>",
                    "</details>",
                ]
            )
        )

    if uncat:
        uncat.sort(key=lambda t: t[0].lower())
        parts.append(
            "\n".join(
                [
                    '<details class="local-summary-pillar local-summary-pillar--other">',
                    '<summary class="local-summary-pillar-summary">Other</summary>',
                    '<div class="local-summary-pillar-body">',
                    _render_local_summary_table_html(uncat),
                    "</div>",
                    "</details>",
                ]
            )
        )

    tail_hints = [
        '<p class="hint">If there are failures, click <strong>View full log</strong> in the table '
        "to open logs in a dialog.</p>",
        '<p class="hint summary-legend">Row background: '
        '<strong class="summary-legend--ok">green</strong> = no failures/errors for this job; '
        '<strong class="summary-legend--fail">red</strong> = failures, errors, or log fetch failure; '
        '<strong class="summary-legend--unk">gray</strong> = no pytest result summary detected.</p>',
    ]
    return ("\n".join(parts + tail_hints)) if parts else "\n".join(tail_hints)


def _append_local_summary_grouped_markdown(
    lines: list[str],
    job_rows: list[tuple[str, list[Path], dict[str, Any]]],
) -> None:
    buckets: dict[str, dict[str, list[tuple[str, list[Path], dict[str, Any]]]]] = defaultdict(lambda: defaultdict(list))
    uncat: list[tuple[str, list[Path], dict[str, Any]]] = []
    for job_name, paths, info in job_rows:
        pillar, dim = _classify_local_nightly_job(job_name)
        if pillar and dim:
            buckets[pillar][dim].append((job_name, paths, info))
        else:
            uncat.append((job_name, paths, info))

    for pillar in _LOCAL_SUMMARY_PILLARS:
        dim_map = buckets.get(pillar) or {}
        if not any(dim_map.get(d) for d in _LOCAL_SUMMARY_DIMS):
            continue
        lines.append(f"#### {pillar}")
        lines.append("")
        for dim in _LOCAL_SUMMARY_DIMS:
            chunk = dim_map.get(dim) or []
            if not chunk:
                continue
            chunk.sort(key=lambda t: t[0].lower())
            lines.append(f"##### {dim}")
            lines.append("")
            summary_rows: list[list[str]] = [_summary_row_for_job(n, p, i) for n, p, i in chunk]
            lines.append(
                render_markdown_table(
                    [
                        "Job",
                        "Total",
                        "Passed",
                        "Failed",
                        "Skipped",
                        "Errors",
                        "Elapsed time",
                    ],
                    summary_rows,
                )
            )
            lines.append("")

    if uncat:
        uncat.sort(key=lambda t: t[0].lower())
        lines.append("#### Other")
        lines.append("")
        summary_rows_u = [_summary_row_for_job(n, p, i) for n, p, i in uncat]
        lines.append(
            render_markdown_table(
                [
                    "Job",
                    "Total",
                    "Passed",
                    "Failed",
                    "Skipped",
                    "Errors",
                    "Elapsed time",
                ],
                summary_rows_u,
            )
        )
        lines.append("")


def markdown_local_summary_from_log_dir(log_dir: Path) -> str:
    """
    Markdown block matching the **grouped Summary** under nightly **Local cluster** (pillar × dim tables).

    Used by ``compose_full_report.py`` for **Test Result → H200 / H800 / A100** when
    ``--log-dir-h*`` points at a ``nightly_jobs``-style tree.
    """
    groups = discover_job_logs(log_dir)
    lines: list[str] = [
        f"*Log root:* `{log_dir}` (layout: "
        f"[references/nightly-local-log-layout.md](references/nightly-local-log-layout.md)).",
        "",
        "Same grouping as local **Summary** in nightly HTML/Markdown reports "
        "(Omni / TTS / Diffusion × Perf / Acc / …):",
        "",
    ]
    if not groups:
        lines.append(
            "*No parseable job logs found.* Confirm the directory matches "
            "[references/nightly-local-log-layout.md](references/nightly-local-log-layout.md)."
        )
        lines.append("")
        return "\n".join(lines)

    job_rows = _local_job_rows_with_info(groups)
    _append_local_summary_grouped_markdown(lines, job_rows)
    lines.append(
        "*Per-job failure/error excerpts expand only in the full nightly report; "
        "this section keeps the summary table only.*"
    )
    lines.append("")
    return "\n".join(lines)


def read_job_text(paths: list[Path]) -> str:
    chunks: list[str] = []
    for p in paths:
        try:
            chunks.append(p.read_text(encoding="utf-8-sig", errors="replace"))
        except OSError as e:
            chunks.append(f"\n<<< read error {p}: {e} >>>\n")
    return "\n".join(chunks)


def _read_combined_job_logs(paths: list[Path]) -> str:
    return read_combined_job_logs(paths, include_headers=True)


def _summary_row_for_job(job_name: str, paths: list[Path], info: dict[str, Any]) -> list[str]:
    counts = extract_pytest_counts(info["summary"])
    n_fail = len(info["failed_nodes"])
    n_err = len(info["error_nodes"])

    if info["summary"] is None and not info["failed_nodes"] and not n_err:
        total = ok = bad = skip = errc = ""
    else:
        fc = counts["failed"] if counts["failed"] else n_fail
        ec = counts["error"] if counts["error"] else n_err
        if counts["passed"] or counts["failed"] or counts["skipped"] or counts["error"]:
            total = str(counts["passed"] + counts["failed"] + counts["skipped"] + counts["error"])
            ok = str(counts["passed"])
            bad = str(fc)
            skip = str(counts["skipped"])
            errc = str(ec)
        else:
            total = ""
            ok = "?"
            bad = str(fc)
            skip = str(counts["skipped"]) if counts["skipped"] else "?"
            errc = str(ec)

    dur = extract_pytest_duration_display(info.get("summary"))
    dur_cell = _md_cell(dur) if dur else "—"
    return [
        _md_cell(job_name),
        _md_cell(total),
        _md_cell(ok),
        _md_cell(bad),
        _md_cell(skip),
        _md_cell(errc),
        dur_cell,
    ]


def _summary_row_for_bk_rec(rec: dict[str, Any]) -> list[str]:
    name = rec["name"]
    if not rec.get("raw_url"):
        return [_md_cell(name), "", "", "", "", "", "no log URL"]
    if rec.get("log_error"):
        return [
            _md_cell(name),
            "",
            "",
            "",
            "",
            "",
            _md_cell(f"log fetch: {rec['log_error'][:200]}"),
        ]
    info = rec.get("info")
    if not info:
        return [_md_cell(name), "", "", "", "", "", "—"]
    return _summary_row_for_job(name, [], info)


def _perf_num(v: Any) -> str:
    if isinstance(v, (int, float)):
        s = f"{float(v):.4f}".rstrip("0").rstrip(".")
        return s or "0"
    return "N/A"


def _perf_pct(v: Any) -> str:
    if isinstance(v, (int, float)):
        sign = "+" if float(v) >= 0 else ""
        return f"{sign}{float(v):.2f}%"
    return "N/A"


def _as_num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _fmt_mtime(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    except OSError:
        return ""


def _resolve_kanban_raw_root(kanban_cfg: KanbanAssetsConfig) -> Path | None:
    if kanban_cfg.raw_root is not None:
        return kanban_cfg.raw_root.resolve()
    if kanban_cfg.repo_root is not None:
        return (kanban_cfg.repo_root / "data/buildkite_nightly_raw").resolve()
    return None


def _collect_kanban_raw_files(raw_root: Path | None) -> list[Path]:
    if raw_root is None or not raw_root.is_dir():
        return []
    files: dict[Path, None] = {}
    for pattern in KANBAN_RAW_PATTERNS:
        for path in raw_root.rglob(pattern):
            if path.is_file():
                files[path] = None
    return sorted(files)


def _build_ids_from_raw_files(raw_root: Path | None, paths: list[Path]) -> list[str]:
    if raw_root is None:
        return []
    ids: set[str] = set()
    for path in paths:
        try:
            rel = path.relative_to(raw_root)
        except ValueError:
            continue
        if not rel.parts:
            continue
        head = rel.parts[0]
        if head.isdigit():
            ids.add(head)
    return sorted(ids, key=lambda item: int(item))


def _kanban_raw_assets_diagnostic(
    kanban_cfg: KanbanAssetsConfig,
    summary: dict[str, Any],
) -> dict[str, Any]:
    assets_dir_txt = str(summary.get("assets_dir") or "")
    assets_dir = Path(assets_dir_txt) if assets_dir_txt else None
    history_files = sorted(assets_dir.glob("*_history.json")) if assets_dir and assets_dir.is_dir() else []
    raw_root = _resolve_kanban_raw_root(kanban_cfg)
    raw_files = _collect_kanban_raw_files(raw_root)
    build_ids = _build_ids_from_raw_files(raw_root, raw_files)

    latest_history = max(history_files, key=lambda p: p.stat().st_mtime) if history_files else None
    latest_raw = max(raw_files, key=lambda p: p.stat().st_mtime) if raw_files else None
    recommended = ""
    if kanban_cfg.repo_root:
        recommended = (
            "python scripts/nightly_local_log_report.py --kanban-repo-root "
            f"{kanban_cfg.repo_root} --kanban-refresh-from-raw ..."
        )

    return {
        "raw_root": str(raw_root or ""),
        "raw_exists": bool(raw_root and raw_root.is_dir()),
        "raw_file_count": len(raw_files),
        "raw_build_ids": build_ids[-5:],
        "raw_latest_mtime": _fmt_mtime(latest_raw) if latest_raw else "",
        "history_file_count": len(history_files),
        "history_latest_mtime": _fmt_mtime(latest_history) if latest_history else "",
        "recommended_command": recommended,
    }


def _kanban_repo_dirty(repo: Path) -> tuple[bool | None, str]:
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(repo),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        return None, detail[:500]
    return bool(proc.stdout.strip()), proc.stdout.strip()[:500]


def _kanban_python(repo: Path) -> str:
    candidates = (
        repo / ".venv/bin/python",
        repo / ".venv/Scripts/python.exe",
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def _run_kanban_refresh_from_raw(
    kanban_repo_root: Path | None,
    raw_root: Path | None,
) -> tuple[str | None, list[str]]:
    if kanban_repo_root is None:
        return None, ["kanban raw refresh skipped: --kanban-repo-root is required."]
    repo = kanban_repo_root.resolve()
    if not repo.is_dir():
        return None, [f"kanban raw refresh skipped: repo root not found: {repo}"]
    sync_script = repo / "scripts/sync_buildkite_raw_model_results.py"
    gen_script = repo / "scripts/generate_charts.py"
    missing = [str(p) for p in (sync_script, gen_script) if not p.is_file()]
    if missing:
        return None, ["kanban raw refresh skipped: missing script(s): " + ", ".join(missing)]

    dirty, detail = _kanban_repo_dirty(repo)
    if dirty is None:
        return None, [
            "kanban raw refresh skipped: unable to verify clean git working tree" + (f"; {detail}" if detail else "")
        ]
    if dirty:
        return None, [
            "kanban raw refresh skipped: kanban checkout has uncommitted changes. "
            "Commit or clean that repo before regenerating assets." + (f" Changed entries: {detail}" if detail else "")
        ]

    warnings: list[str] = []
    synced_models = 0
    py = _kanban_python(repo)
    for model_name, model_keywords in KANBAN_RAW_MODEL_SYNCS:
        cmd = [
            py,
            str(sync_script),
            "--model-name",
            model_name,
            "--model-keywords",
            model_keywords,
        ]
        if raw_root is not None:
            cmd.extend(["--raw-root", str(raw_root.resolve())])
        proc = subprocess.run(
            cmd,
            cwd=str(repo),
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            warnings.append(
                f"kanban raw sync failed for {model_name}: exit {proc.returncode}"
                + (f"; {detail[:500]}" if detail else "")
            )
            continue
        synced_models += 1

    proc = subprocess.run(
        [py, str(gen_script)],
        cwd=str(repo),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        warnings.append(
            f"kanban generate_charts.py failed: exit {proc.returncode}" + (f"; {detail[:500]}" if detail else "")
        )
        return None, warnings

    note = f"kanban raw refresh completed: synced {synced_models} model group(s), regenerated chart history assets."
    return note, warnings


_PERF_TABLE_HEADERS = [
    "Type",
    "Config",
    "Test",
    "Metric",
    "latest",
    "baseline",
    "vs baseline",
    "Status",
]
_FOCUS_TABLE_HEADERS = (
    "Source",
    "Model",
    "Type",
    "Config",
    "Test",
    "Metric",
    "latest",
    "baseline",
    "vs baseline",
    "Status",
    "Days failing",
)
_CONSEC_FAIL_DAY_MAX = 5
_CONSEC_FAIL_THRESHOLD_PCT = 6.0
_CONSEC_FAIL_LOOKUP_LIMIT_PER_FILE = 5000


def _metric_direction_hint(metric: str) -> str:
    """Mirror of kanban_assets_perf_summary._metric_direction (local-only).

    Keep this list aligned byte-for-byte with
    kanban_assets_perf_summary.LOWER_BETTER_HINTS / HIGHER_BETTER_HINTS so the
    consecutive-failure lookup computes the same direction verdict the kanban
    baseline comparator uses for PerfRow.status.

    If a metric name shows up in the focus table but is not in this list, both
    local and kanban will treat it as `unknown` → skipped from the streak
    lookup. (We deliberately do NOT add legacy aliases like `itl`,
    `response_time`, `elapsed` — they aren't in kanban's source of truth.)
    """
    m = (metric or "").lower()
    higher_hints = ("throughput", "qps", "tps")
    lower_hints = (
        "latency",
        "ttfp",
        "ttft",
        "tpot",
        "rtl",
        "rtf",
        "memory",
        "e2e",
        "duration",
    )
    if any(h in m for h in higher_hints):
        return "higher_better"
    if any(h in m for h in lower_hints):
        return "lower_better"
    return "unknown"


def _iter_metric_pairs_for_history(rec: dict[str, Any]) -> list[tuple[str, float, float]]:
    """Mirror of kanban_assets_perf_summary._iter_metric_pairs used for the
    consecutive-failure lookup. Kept local so we don't depend on the private
    helper.
    """

    def _to_float(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            txt = value.strip()
            if not txt:
                return None
            try:
                return float(txt)
            except ValueError:
                return None
        return None

    pairs: list[tuple[str, float, float]] = []
    if not isinstance(rec, dict):
        return pairs
    for key, value in rec.items():
        if not isinstance(key, str) or not key.startswith("baseline_"):
            continue
        metric = key[len("baseline_") :]
        baseline = _to_float(value)
        latest = _to_float(rec.get(metric))
        if baseline is None or latest is None:
            continue
        pairs.append((metric, latest, baseline))
    baseline_obj = rec.get("baseline")
    if isinstance(baseline_obj, dict):
        for metric, base_v in baseline_obj.items():
            if not isinstance(metric, str):
                continue
            if any(metric == existing[0] for existing in pairs):
                continue
            baseline = _to_float(base_v)
            latest = _to_float(rec.get(metric))
            if baseline is None or latest is None:
                continue
            pairs.append((metric, latest, baseline))
    return pairs


def _history_records_from_payload(payload: Any) -> list[dict[str, Any]]:
    """Extract the flat record list from a history JSON payload.

    Supports both top-level ``records: [...]`` and group-based
    ``groups: [{records: [...]}]`` shapes (matching the kanban
    ``generate_charts.py`` output).
    """
    if not isinstance(payload, dict):
        return []
    records = payload.get("records")
    if isinstance(records, list):
        return [r for r in records if isinstance(r, dict)]
    groups = payload.get("groups")
    if isinstance(groups, list):
        flat: list[dict[str, Any]] = []
        for grp in groups:
            if not isinstance(grp, dict):
                continue
            rs = grp.get("records")
            if isinstance(rs, list):
                flat.extend(r for r in rs if isinstance(r, dict))
        return flat
    return []


def _compute_history_fail_lookup(
    assets_dir: Path | None,
) -> dict[tuple[str, str, str, str], dict[str, bool]]:
    """Build ``(model_id, test_name, config_key, metric) -> {date: is_fail}``
    by scanning the kanban ``*_history.json`` files under ``assets_dir``.

    ``is_fail`` re-applies ``vs_pct < -CONSEC_FAIL_THRESHOLD_PCT`` (matching
    ``kanban_assets_perf_summary._baseline_compare_status``). When multiple
    records share the same date for the same key, "fail" wins (most pessimistic).
    """
    if assets_dir is None or not assets_dir.is_dir():
        return {}
    lookup: dict[tuple[str, str, str, str], dict[str, bool]] = {}
    for history_path in sorted(assets_dir.glob("*_history.json")):
        try:
            payload = json.loads(history_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        records = _history_records_from_payload(payload)
        per_file = 0
        for rec in records:
            per_file += 1
            if per_file > _CONSEC_FAIL_LOOKUP_LIMIT_PER_FILE:
                break
            date_raw = rec.get("date")
            if not isinstance(date_raw, str):
                continue
            date = date_raw.strip()[:10]
            if len(date) < 10:
                continue
            model_id = rec.get("model_id")
            title = rec.get("title")
            model = str(model_id or title or "")
            if not model:
                continue
            test_name = str(rec.get("test_name") or "")
            config_key = str(rec.get("config_key") or rec.get("source_file") or "")
            for metric, latest, baseline in _iter_metric_pairs_for_history(rec):
                if abs(baseline) < 1e-12:
                    continue
                direction = _metric_direction_hint(metric)
                if direction == "unknown":
                    continue
                raw_pct = (latest - baseline) / baseline * 100.0
                vs_pct = -raw_pct if direction == "lower_better" else raw_pct
                is_fail = vs_pct < -_CONSEC_FAIL_THRESHOLD_PCT
                key = (model, test_name, config_key, metric)
                slot = lookup.setdefault(key, {})
                previous = bool(slot.get(date, False))
                slot[date] = is_fail or previous
    return lookup


def _consec_fail_days_from_history(
    lookup: dict[tuple[str, str, str, str], dict[str, bool]],
    *,
    model: str,
    test: str,
    config_key: str,
    metric: str,
    max_days: int = _CONSEC_FAIL_DAY_MAX,
) -> int:
    """Count how many consecutive days (newest first) the (model, test, config,
    metric) regressed worse than baseline by more than the threshold.

    Returns 0 when the latest dated record for this key is *not* a failure or
    when no history exists for that key.
    """
    by_date = lookup.get((model, test, config_key, metric))
    if not by_date:
        return 0
    sorted_dates_desc = sorted(by_date.keys(), reverse=True)
    if not sorted_dates_desc:
        return 0
    latest = sorted_dates_desc[0]
    if not by_date.get(latest):
        return 0
    try:
        cursor = datetime.strptime(latest, "%Y-%m-%d")
    except ValueError:
        return 0
    count = 0
    cursor_str = latest
    for _ in range(max_days):
        if by_date.get(cursor_str):
            count += 1
            cursor = cursor - timedelta(days=1)
            cursor_str = cursor.strftime("%Y-%m-%d")
        else:
            break
    return count


def _format_consec_fail_days(days: int) -> str:
    """Format consecutive-failure count for table cell display.

    Display rule:
      - 0 days  → "—"  (no streak, ignore)
      - 1 day   → "1 day"
      - 2 days  → "2 days"
      - 3 days  → "3 days"  (boundary: still show the exact count)
      - >= 4 days → "3 days+"  (cap at 3 to keep the table cell compact;
        the underlying ``consec_fail_days`` counter still grows so the CSS
        streak class escalates correctly)
    """
    if days <= 0:
        return "—"
    if days == 1:
        return "1 day"
    if days >= 4:
        return "3 days+"
    return f"{days} days"


def _consec_fail_color_class(days: int) -> str:
    """Map consecutive-failure count to the CSS modifier class.

    Display rule (also mirrored by :func:`_format_consec_fail_days`):
      - 0       → ``""``                 (no streak)
      - 1       → ``regression-streak--1d``  (yellow, single day)
      - 2       → ``regression-streak--2d``  (orange, two days)
      - 3       → ``regression-streak--3d``  (red, exactly three days)
      - >= 4    → ``regression-streak--3d-plus``  (purple, **3 days+** cap)
    """
    if days >= 4:
        return "regression-streak--3d-plus"
    if days >= 3:
        return "regression-streak--3d"
    if days == 2:
        return "regression-streak--2d"
    if days == 1:
        return "regression-streak--1d"
    return ""


def _normalize_focus_key(parts: tuple[str, str, str, str]) -> tuple[str, str, str, str]:
    """Trim whitespace and normalize empty config_key for lookup matching.

    Focus items have ``config == config_view`` (a human-readable label), while
    history records key on ``config_key`` (a normalized slug). We collapse
    whitespace and fall back to an empty string so the lookup hits when these
    roughly agree.
    """
    model, test, config, metric = parts
    return (
        (model or "").strip(),
        (test or "").strip(),
        (config or "").strip(),
        (metric or "").strip(),
    )


def _focus_item_consec_fail_days(
    item: NightlyFocusItem,
    lookup: dict[tuple[str, str, str, str], dict[str, bool]],
) -> int:
    """Compute consecutive-failing-day count for a focus item using the history
    lookup. Tries the (config_key) path first, then falls back to a wildcard
    scan across config_key values when the focus item only carries the
    human-readable config_view.
    """
    key_strict = _normalize_focus_key((item.model, item.test, item.config, item.metric))
    by_date = lookup.get(key_strict)
    if by_date:
        return _consec_fail_days_from_history(
            lookup,
            model=key_strict[0],
            test=key_strict[1],
            config_key=key_strict[2],
            metric=key_strict[3],
        )
    # Fallback: match by (model, test, metric) over any config_key in history.
    best = 0
    target_model, target_test, _, target_metric = key_strict
    for (model, test, config_key, metric), dates in lookup.items():
        if model != target_model or test != target_test or metric != target_metric:
            continue
        count = _consec_fail_days_from_history(
            lookup,
            model=model,
            test=test,
            config_key=config_key,
            metric=metric,
        )
        if count > best:
            best = count
            if best >= _CONSEC_FAIL_DAY_MAX:
                break
    return best


@dataclass
class NightlyFocusItem:
    """One baseline-backed performance row promoted to the top-level nightly focus."""

    source: str
    model: str
    model_type: str
    config: str
    test: str
    metric: str
    latest: Any
    baseline: Any
    vs_baseline_pct: Any
    status: str
    consec_fail_days: int = 0


def _job_kind_counts(kinds: list[str]) -> dict[str, int]:
    out = {"total": len(kinds), "ok": 0, "fail": 0, "unknown": 0}
    for kind in kinds:
        key = kind if kind in ("ok", "fail", "unknown") else "unknown"
        out[key] += 1
    return out


def _buildkite_job_counts(bk_jobs: list[dict[str, Any]] | None) -> dict[str, int]:
    if bk_jobs is None:
        return {"total": 0, "ok": 0, "fail": 0, "unknown": 0}
    return _job_kind_counts([_summary_row_kind_bk(rec) for rec in bk_jobs])


def _local_job_counts(
    job_rows: list[tuple[str, list[Path], dict[str, Any]]],
) -> dict[str, int]:
    return _job_kind_counts([_summary_row_kind(info) for _, _, info in job_rows])


def _perf_counts(summary: dict[str, Any]) -> dict[str, int]:
    counts = _empty_perf_status_counts()
    for key, value in (summary.get("summary") or {}).items():
        if key in counts:
            counts[key] = int(value or 0)
    return counts


def _focus_item_from_perf_row(
    source: str,
    row: Any,
    history_fail_lookup: dict[tuple[str, str, str, str], dict[str, bool]] | None = None,
) -> NightlyFocusItem:
    if isinstance(row, dict):
        get_value = row.get
    else:

        def get_value(key, default=None):
            return getattr(row, key, default)

    out = NightlyFocusItem(
        source=source,
        model=str(get_value("model", "") or "unknown"),
        model_type=str(get_value("model_type", "") or ""),
        config=str(get_value("config_view", "") or ""),
        test=str(get_value("test_name", "") or ""),
        metric=str(get_value("metric", "") or ""),
        latest=get_value("latest"),
        baseline=get_value("baseline"),
        vs_baseline_pct=get_value("vs_baseline_pct"),
        status=str(get_value("status", "") or "n/a"),
    )
    if history_fail_lookup:
        out.consec_fail_days = _focus_item_consec_fail_days(out, history_fail_lookup)
    return out


def _focus_item_sort_key(item: NightlyFocusItem) -> tuple[int, float, str, str]:
    status_rank = {"fail": 0, "normal": 1, "n/a": 2, "pass": 3}
    pct = _as_num(item.vs_baseline_pct)
    if pct is None:
        pct = 999999.0
    return (status_rank.get(item.status, 4), pct, item.source, item.model)


def _focus_perf_items(
    bk_perf_summary: dict[str, Any],
    local_perf_summary: dict[str, Any],
    history_fail_lookup: dict[tuple[str, str, str, str], dict[str, bool]] | None = None,
) -> list[NightlyFocusItem]:
    items: list[NightlyFocusItem] = []
    for row in bk_perf_summary.get("rows", []) or []:
        items.append(_focus_item_from_perf_row("Buildkite", row, history_fail_lookup))
    for row in local_perf_summary.get("rows", []) or []:
        items.append(_focus_item_from_perf_row("Local", row, history_fail_lookup))
    return items


def _select_focus_perf_items(items: list[NightlyFocusItem]) -> tuple[str, list[NightlyFocusItem]]:
    failed = sorted(
        [item for item in items if item.status == "fail"],
        key=_focus_item_sort_key,
    )
    if failed:
        return "fail", failed
    normal = sorted(
        [item for item in items if item.status == "normal"],
        key=_focus_item_sort_key,
    )
    if normal:
        return "normal", normal[:3]
    return "empty", []


def _focus_perf_table_rows(items: list[NightlyFocusItem]) -> list[list[str]]:
    rows: list[list[str]] = []
    for item in items:
        rows.append(
            [
                _md_cell(item.source),
                _md_cell(item.model),
                _md_cell(item.model_type),
                _md_cell(item.config),
                _md_cell(item.test),
                _md_cell(item.metric),
                _perf_num(item.latest),
                _perf_num(item.baseline),
                _perf_pct(item.vs_baseline_pct),
                _md_cell(item.status),
                _format_consec_fail_days(item.consec_fail_days),
            ]
        )
    return rows


def _render_focus_perf_table_html(items: list[NightlyFocusItem]) -> str:
    models = sorted({item.model for item in items if item.model})
    model_checks = []
    for model in models:
        val = html.escape(model, quote=True)
        model_checks.append(
            '<label class="focus-model-check">'
            f'<input type="checkbox" data-filter-key="model" value="{val}">'
            f"<span>{html.escape(model)}</span></label>"
        )
    parts: list[str] = [
        '<div class="perf-filter-scope focus-filter-scope" data-perf-filter-scope="daily-focus-regressions">',
        '<div class="perf-filter-bar focus-filter-bar">',
        '<fieldset class="focus-model-filter"><legend>Model <span>No selection = All</span></legend>',
        "".join(model_checks),
        "</fieldset>",
        '<button type="button" class="focus-expand-btn" data-focus-expand="0" '
        'aria-expanded="false">Expand table</button>',
        "</div>",
        '<div class="table-scroll">',
        '<table class="summary focus-table">',
        "<thead><tr>",
    ]
    for header in _FOCUS_TABLE_HEADERS:
        parts.append(f"<th>{html.escape(header)}</th>")
    parts.append("</tr></thead><tbody>")
    for item in items:
        model = html.escape(item.model, quote=True)
        row_class = "summary-row--fail" if item.status == "fail" else "summary-row--unknown"
        streak_class = _consec_fail_color_class(item.consec_fail_days)
        row_classes = [row_class]
        if streak_class:
            row_classes.append(streak_class)
        row_class_attr = " ".join(row_classes)
        days = item.consec_fail_days
        days_text = _format_consec_fail_days(days)
        days_attr = (
            f' data-consec-days="{days}" data-consec-color="{html.escape(streak_class or "none")}"'
            if streak_class
            else f' data-consec-days="{days}"'
        )
        parts.append(f'<tr class="{row_class_attr}" data-perf-row="1" data-model="{model}"{days_attr}>')
        cells = [
            item.source,
            item.model,
            item.model_type,
            item.config,
            item.test,
            item.metric,
            _perf_num(item.latest),
            _perf_num(item.baseline),
            _perf_pct(item.vs_baseline_pct),
            item.status,
        ]
        for cell in cells:
            parts.append(f"<td>{html.escape(str(cell))}</td>")
        if streak_class:
            title = "Consecutive failing days from *_history.json (today + 2 previous days max)."
            cell_attrs = f' class="{html.escape(streak_class)}" title="{html.escape(title, quote=True)}"'
        else:
            cell_attrs = ""
        parts.append(f"<td{cell_attrs}>{html.escape(days_text)}</td>")
        parts.append("</tr>")
    parts.extend(
        [
            "</tbody></table></div>",
            '<p class="note perf-filter-empty" data-perf-empty hidden>No rows match filters.</p>',
            "</div>",
        ]
    )
    return "\n".join(parts)


def _daily_focus_data(
    *,
    bk_jobs: list[dict[str, Any]] | None,
    local_job_rows: list[tuple[str, list[Path], dict[str, Any]]],
    kanban_cfg: KanbanAssetsConfig,
    log_dir: Path,
) -> dict[str, Any]:
    bk_perf_summary, _ = _buildkite_perf_rows(kanban_cfg, log_dir=log_dir, exclude_local_overlap=True)
    local_perf_summary, _ = _buildkite_perf_rows(kanban_cfg, log_dir=log_dir)
    history_fail_lookup = _compute_history_fail_lookup(kanban_cfg.assets_dir)
    focus_items = _focus_perf_items(bk_perf_summary, local_perf_summary, history_fail_lookup)
    focus_kind, top_items = _select_focus_perf_items(focus_items)
    bk_job_counts = _buildkite_job_counts(bk_jobs)
    local_job_counts = _local_job_counts(local_job_rows)
    bk_perf_counts = _perf_counts(bk_perf_summary)
    local_perf_counts = _perf_counts(local_perf_summary)
    job_fail_count = int(bk_job_counts["fail"]) + int(local_job_counts["fail"])
    perf_fail_count = int(bk_perf_counts["fail"]) + int(local_perf_counts["fail"])
    if job_fail_count or perf_fail_count:
        conclusion = (
            f"Attention needed: {job_fail_count} test failure(s)/anomaly(ies) and "
            f"{perf_fail_count} major performance regression(s) detected."
        )
        severity = "fail"
    elif focus_kind == "normal":
        conclusion = "No major performance regressions; minor fluctuations listed below for observation."
        severity = "normal"
    elif focus_items:
        conclusion = "No major performance regressions detected."
        severity = "ok"
    else:
        conclusion = "No major regressions could be determined; baseline data is missing or empty."
        severity = "unknown"
    return {
        "conclusion": conclusion,
        "severity": severity,
        "focus_kind": focus_kind,
        "top_items": top_items,
        "bk_job_counts": bk_job_counts,
        "local_job_counts": local_job_counts,
        "bk_perf_counts": bk_perf_counts,
        "local_perf_counts": local_perf_counts,
        "bk_perf_status": bk_perf_summary.get("status", ""),
        "local_perf_status": local_perf_summary.get("status", ""),
        "bk_perf_message": bk_perf_summary.get("message", ""),
        "local_perf_message": local_perf_summary.get("message", ""),
    }


def _render_focus_metric_card(title: str, value: str, detail: str, severity: str) -> str:
    return (
        f'<div class="focus-card focus-card--{html.escape(severity)}">'
        f'<div class="focus-card-title">{html.escape(title)}</div>'
        f'<div class="focus-card-value">{html.escape(value)}</div>'
        f'<div class="focus-card-detail">{html.escape(detail)}</div>'
        "</div>"
    )


def _render_daily_focus_html(data: dict[str, Any]) -> str:
    bk_jobs = data["bk_job_counts"]
    local_jobs = data["local_job_counts"]
    bk_perf = data["bk_perf_counts"]
    local_perf = data["local_perf_counts"]
    cards = [
        _render_focus_metric_card(
            "Buildkite jobs",
            f"{int(bk_jobs['fail'])}/{int(bk_jobs['total'])} failed",
            f"ok={int(bk_jobs['ok'])}, unknown={int(bk_jobs['unknown'])}",
            "fail" if bk_jobs["fail"] else "ok",
        ),
        _render_focus_metric_card(
            "Local jobs",
            f"{int(local_jobs['fail'])}/{int(local_jobs['total'])} failed",
            f"ok={int(local_jobs['ok'])}, unknown={int(local_jobs['unknown'])}",
            "fail" if local_jobs["fail"] else "ok",
        ),
        _render_focus_metric_card(
            "Buildkite perf",
            f"{int(bk_perf['fail'])} fail",
            f"pass={int(bk_perf['pass'])}, normal={int(bk_perf['normal'])}, n/a={int(bk_perf['n/a'])}",
            "fail" if bk_perf["fail"] else "ok",
        ),
        _render_focus_metric_card(
            "Local perf",
            f"{int(local_perf['fail'])} fail",
            f"pass={int(local_perf['pass'])}, normal={int(local_perf['normal'])}, n/a={int(local_perf['n/a'])}",
            "fail" if local_perf["fail"] else "ok",
        ),
    ]
    parts: list[str] = [
        f'<section class="panel nightly-focus nightly-focus--{html.escape(str(data["severity"]))}">',
        _heading_html(
            "h2",
            _SVG_SPARK,
            html.escape("Daily focus"),
            sub=html.escape("Performance regressions and test failures"),
        ),
        f'<p class="focus-conclusion">{html.escape(str(data["conclusion"]))}</p>',
        '<div class="focus-card-grid">',
        "\n".join(cards),
        "</div>",
    ]
    top_items: list[NightlyFocusItem] = data.get("top_items") or []
    if top_items:
        label = "All major regressions" if data.get("focus_kind") == "fail" else "Minor fluctuation watchlist"
        parts.extend(
            [
                f'<h3 class="focus-table-title">{html.escape(label)}</h3>',
                _render_focus_perf_table_html(top_items),
            ]
        )
    else:
        notes = []
        if data.get("bk_perf_status") != "ok" and data.get("bk_perf_message"):
            notes.append(f"Buildkite perf: {data['bk_perf_message']}")
        if data.get("local_perf_status") != "ok" and data.get("local_perf_message"):
            notes.append(f"Local perf: {data['local_perf_message']}")
        if notes:
            parts.append(
                '<div class="note"><strong>Data note:</strong><ul>'
                + "".join(f"<li>{html.escape(str(note))}</li>" for note in notes)
                + "</ul></div>"
            )
        else:
            parts.append('<p class="note">No performance regressions require top-level display.</p>')
    parts.append("</section>")
    return "\n".join(parts)


def _append_daily_focus_markdown(lines: list[str], data: dict[str, Any]) -> None:
    lines.append("## Daily focus")
    lines.append("")
    lines.append(f"- **Conclusion:** {_md_cell(str(data['conclusion']))}")
    bk_jobs = data["bk_job_counts"]
    local_jobs = data["local_job_counts"]
    bk_perf = data["bk_perf_counts"]
    local_perf = data["local_perf_counts"]
    lines.append(
        f"- **Test failures:** Buildkite `{int(bk_jobs['fail'])}/{int(bk_jobs['total'])}`, "
        f"Local `{int(local_jobs['fail'])}/{int(local_jobs['total'])}`"
    )
    lines.append(f"- **Performance fail:** Buildkite `{int(bk_perf['fail'])}`, Local `{int(local_perf['fail'])}`")
    lines.append("")
    top_items: list[NightlyFocusItem] = data.get("top_items") or []
    if top_items:
        title = "### All major regressions" if data.get("focus_kind") == "fail" else "### Minor fluctuation watchlist"
        lines.append(title)
        lines.append("")
        lines.append(
            render_markdown_table(
                list(_FOCUS_TABLE_HEADERS),
                _focus_perf_table_rows(top_items),
            )
        )
        lines.append("")
        return
    notes = []
    if data.get("bk_perf_status") != "ok" and data.get("bk_perf_message"):
        notes.append(f"Buildkite perf: {data['bk_perf_message']}")
    if data.get("local_perf_status") != "ok" and data.get("local_perf_message"):
        notes.append(f"Local perf: {data['local_perf_message']}")
    if notes:
        for note in notes:
            lines.append(f"- **Data note:** {_md_cell(str(note))}")
    else:
        lines.append("*No performance regressions require top-level display.*")
    lines.append("")


def _render_perf_model_table_html(table_id: str, rows: list[list[str]]) -> str:
    """Render one model performance table with per-table dropdown filters."""
    tests = sorted({r[2] for r in rows if len(r) > 2 and r[2]})
    metrics = sorted({r[3] for r in rows if len(r) > 3 and r[3]})
    statuses = sorted({r[7] for r in rows if len(r) > 7 and r[7]})

    def _select_html(key: str, label: str, options: list[str]) -> str:
        opts = ['<option value="">All</option>']
        for value in options:
            val = html.escape(value, quote=True)
            txt = html.escape(value)
            opts.append(f'<option value="{val}">{txt}</option>')
        return (
            '<label class="perf-filter-label">'
            f"<span>{html.escape(label)}</span>"
            f'<select class="perf-filter-select" data-filter-key="{html.escape(key, quote=True)}">'
            + "".join(opts)
            + "</select></label>"
        )

    parts: list[str] = [
        f'<div class="perf-filter-scope" data-perf-filter-scope="{html.escape(table_id, quote=True)}">',
        '<div class="perf-filter-bar">',
        _select_html("test", "Test", tests),
        _select_html("metric", "Metric", metrics),
        _select_html("status", "Status", statuses),
        "</div>",
    ]
    parts.append('<div class="table-scroll">')
    parts.append('<table class="summary perf-filter-table">')
    parts.append("<thead><tr>")
    for h in _PERF_TABLE_HEADERS:
        parts.append(f"<th>{html.escape(h)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in rows:
        test_v = html.escape((row[2] if len(row) > 2 else ""), quote=True)
        metric_v = html.escape((row[3] if len(row) > 3 else ""), quote=True)
        status_v = html.escape((row[7] if len(row) > 7 else ""), quote=True)
        parts.append(f'<tr data-perf-row="1" data-test="{test_v}" data-metric="{metric_v}" data-status="{status_v}">')
        for cell in row:
            parts.append(f"<td>{html.escape(cell)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    parts.append('<p class="note perf-filter-empty" data-perf-empty hidden>No rows match filters.</p>')
    parts.append("</div>")
    return "\n".join(parts)


def _grouped_rows_from_summary(summary: dict[str, Any]) -> dict[str, list[list[str]]]:
    grouped_rows: dict[str, list[list[str]]] = {}
    for item in summary.get("rows", []):
        model = _md_cell(str(item.get("model") or "unknown"))
        grouped_rows.setdefault(model, []).append(
            [
                _md_cell(str(item.get("model_type") or "")),
                _md_cell(str(item.get("config_view") or "")),
                _md_cell(str(item.get("test_name") or "")),
                _md_cell(str(item.get("metric") or "")),
                _perf_num(item.get("latest")),
                _perf_num(item.get("baseline")),
                _perf_pct(item.get("vs_baseline_pct")),
                _md_cell(str(item.get("status") or "")),
            ]
        )
    return grouped_rows


def _filter_perf_summary_for_local(
    summary: dict[str, Any],
    *,
    log_dir: Path,
) -> dict[str, Any]:
    result_root = log_dir.resolve()
    resolved_dir = resolve_local_perf_result_dir(result_root)
    perf_files = local_perf_result_files(resolved_dir) if resolved_dir else []
    local_keys = collect_local_perf_test_keys(resolved_dir)

    out = dict(summary)
    scope: dict[str, Any] = {
        "result_root": str(result_root),
        "resolved_dir": str(resolved_dir) if resolved_dir else "",
        "perf_file_count": len(perf_files),
        "test_key_count": len(local_keys),
    }

    if not perf_files:
        out["rows"] = []
        out["status"] = "empty"
        out["message"] = (
            f"No local perf JSON under {result_root}. "
            "Sync logs/nightly_jobs from the cluster before generating the report."
        )
        out["summary"] = {"pass": 0, "normal": 0, "fail": 0, "n/a": 0}
        scope["message"] = (
            f"No local perf JSON under {result_root}; Local performance baseline comparison "
            f"shows only synced log-dir cases."
        )
        out["local_perf_scope"] = scope
        return out

    filtered = [
        row for row in summary.get("rows", []) if isinstance(row, dict) and perf_row_matches_local_test(row, local_keys)
    ]
    stats = {"pass": 0, "normal": 0, "fail": 0, "n/a": 0}
    for row in filtered:
        st = str(row.get("status") or "n/a")
        stats[st] = stats.get(st, 0) + 1

    out["rows"] = filtered
    out["summary"] = stats
    if filtered:
        out["status"] = "ok"
        out["message"] = ""
        scope["message"] = (
            f"Synced {len(perf_files)} perf JSON file(s) → showing {len(filtered)} baseline row(s) "
            f"({len(local_keys)} test key(s))."
        )
    else:
        out["status"] = "empty"
        out["message"] = (
            "Local perf JSON present but no kanban history rows matched. "
            "Run prepare_kanban_before_report.py (manual_* + mkdocs build) then regenerate."
        )
        scope["message"] = (
            f"{len(perf_files)} local perf JSON file(s) present but no kanban history rows matched; "
            "run prepare_kanban_before_report.py first."
        )
    out["local_perf_scope"] = scope
    return out


def _recompute_perf_summary_stats(rows: list[dict[str, Any]]) -> dict[str, int]:
    stats = {"pass": 0, "normal": 0, "fail": 0, "n/a": 0}
    for row in rows:
        st = str(row.get("status") or "n/a")
        stats[st] = stats.get(st, 0) + 1
    return stats


def _filter_perf_summary_exclude_local_overlap(
    summary: dict[str, Any],
    *,
    log_dir: Path,
    local_summary: dict[str, Any],
) -> dict[str, Any]:
    """Drop Buildkite rows for tests already covered in Local performance baseline comparison."""
    if local_summary.get("status") != "ok" or not local_summary.get("rows"):
        return summary

    resolved_dir = resolve_local_perf_result_dir(log_dir.resolve())
    local_keys = collect_local_perf_test_keys(resolved_dir)
    if not local_keys:
        return summary

    original_rows = [row for row in summary.get("rows", []) if isinstance(row, dict)]
    filtered = [row for row in original_rows if not perf_row_matches_local_test(row, local_keys)]

    out = dict(summary)
    out["rows"] = filtered
    out["summary"] = _recompute_perf_summary_stats(filtered)
    if filtered:
        out["status"] = "ok"
        out["message"] = ""
    else:
        out["status"] = "empty"
        out["message"] = "No Buildkite-only baseline rows; cases with local perf JSON are shown under Local Test only."
    return out


def _buildkite_perf_rows(
    kanban_cfg: KanbanAssetsConfig,
    *,
    log_dir: Path | None = None,
    exclude_local_overlap: bool = False,
) -> tuple[dict[str, Any], dict[str, list[list[str]]]]:
    summary = build_assets_perf_summary(
        assets_dir=kanban_cfg.assets_dir,
        kanban_repo_root=kanban_cfg.repo_root,
        expected_remote=kanban_cfg.expected_remote,
        expected_branch=kanban_cfg.expected_branch,
    )
    if kanban_cfg.refresh_note:
        summary.setdefault("warnings", []).append(kanban_cfg.refresh_note)
    for warning in kanban_cfg.refresh_warnings:
        summary.setdefault("warnings", []).append(warning)
    if summary.get("status") != "ok" or kanban_cfg.refresh_warnings:
        summary["raw_fallback"] = _kanban_raw_assets_diagnostic(kanban_cfg, summary)
    if log_dir is not None:
        local_summary = _filter_perf_summary_for_local(summary, log_dir=log_dir)
        if exclude_local_overlap:
            if local_summary.get("rows"):
                summary = _filter_perf_summary_exclude_local_overlap(
                    summary,
                    log_dir=log_dir,
                    local_summary=local_summary,
                )
        else:
            summary = local_summary
    grouped_rows = _grouped_rows_from_summary(summary)
    return summary, grouped_rows


def _kanban_fallback_items(summary: dict[str, Any]) -> list[str]:
    diag = summary.get("raw_fallback") or {}
    if not isinstance(diag, dict):
        return []
    items = [
        "Local nightly_jobs is for pass/fail analysis only; Local performance baseline comparison "
        "shows logs/nightly_jobs synced cases; Buildkite performance baseline comparison "
        "reads all models from kanban docs/assets/charts/*_history.json.",
    ]
    raw_root = str(diag.get("raw_root") or "")
    if raw_root:
        items.append(f"kanban raw root: {raw_root}")
    raw_count = int(diag.get("raw_file_count") or 0)
    if diag.get("raw_exists"):
        items.append(f"raw perf JSON files: {raw_count}")
    else:
        items.append("raw perf JSON root is missing or not a directory.")
    build_ids = diag.get("raw_build_ids") or []
    if build_ids:
        items.append("recent raw build ids: " + ", ".join(str(v) for v in build_ids))
    if diag.get("raw_latest_mtime"):
        items.append(f"latest raw mtime: {diag.get('raw_latest_mtime')}")
    items.append(f"history files: {int(diag.get('history_file_count') or 0)}")
    if diag.get("history_latest_mtime"):
        items.append(f"latest history mtime: {diag.get('history_latest_mtime')}")
    if raw_count:
        items.append(
            "Raw data is present. Re-run with --kanban-refresh-from-raw to invoke kanban sync "
            "+ generate_charts before rendering."
        )
    elif diag.get("recommended_command"):
        items.append(f"Refresh command pattern: {diag.get('recommended_command')}")
    return items


def _render_kanban_fallback_html(summary: dict[str, Any]) -> str:
    items = _kanban_fallback_items(summary)
    if not items:
        return ""
    return (
        '<div class="note"><strong>Raw data fallback diagnostics:</strong><ul>'
        + "".join(f"<li>{html.escape(item)}</li>" for item in items)
        + "</ul></div>"
    )


def _append_kanban_fallback_markdown(lines: list[str], summary: dict[str, Any]) -> None:
    items = _kanban_fallback_items(summary)
    if not items:
        return
    lines.append("- **Raw data fallback diagnostics:**")
    for item in items:
        lines.append(f"  - {_md_cell(item)}")


def _render_buildkite_perf_inner_html(
    kanban_cfg: KanbanAssetsConfig,
    *,
    model_subcard_class: str = "report-subcard--bk-perf-model",
    log_dir: Path | None = None,
    exclude_local_overlap: bool = False,
) -> str:
    summary, grouped_rows = _buildkite_perf_rows(
        kanban_cfg,
        log_dir=log_dir,
        exclude_local_overlap=exclude_local_overlap,
    )
    # Verbose diagnostic lines (Data source / Local filter / History / generated_at /
    # Raw data fallback diagnostics) were removed: the HTML perf comparison card should
    # focus on per-model baseline rows. Diagnostics remain available in
    # `_buildkite_perf_rows` / `_kanban_fallback_items` for tooling.
    parts: list[str] = []
    warnings = summary.get("warnings") or []
    if warnings:
        warn_html = "".join(f"<li>{html.escape(str(w))}</li>" for w in warnings)
        parts.append(f'<div class="note"><strong>Source config notes:</strong><ul>{warn_html}</ul></div>')
    if summary.get("status") != "ok":
        return "\n".join(parts)
    parts.append(
        '<p class="hint">'
        + (
            "Showing baseline comparison in kanban history only for cases with perf JSON under logs/nightly_jobs."
            if log_dir is not None and not exclude_local_overlap
            else (
                "Showing Buildkite-only baseline rows; cases with local perf JSON appear under Local Test only."
                if exclude_local_overlap
                else (
                    f"Showing each model's latest baseline record "
                    f"(freshest = {html.escape(str(summary.get('latest_day') or 'N/A'))}; "
                    "models on different schedules may use different dates)."
                )
            )
        )
        + "</p>"
    )
    stats = summary.get("summary", {})
    parts.append(
        '<p class="meta"><strong>Stats:</strong> '
        f"pass={int(stats.get('pass', 0))}, "
        f"normal={int(stats.get('normal', 0))}, "
        f"fail={int(stats.get('fail', 0))}, "
        f"n/a={int(stats.get('n/a', 0))}</p>"
    )
    for i, model_name in enumerate(sorted(grouped_rows.keys())):
        model_rows = grouped_rows[model_name]
        table_html = _render_perf_model_table_html(f"perf-model-{i}", model_rows)
        parts.append(
            _details_subcard(
                f"{model_name} ({len(model_rows)} rows)",
                table_html,
                open_default=False,
                details_class=model_subcard_class,
                icon_paths=_SVG_LIST,
            )
        )
    return "\n".join(parts)


def _append_local_perf_baseline_markdown(
    lines: list[str],
    kanban_cfg: KanbanAssetsConfig,
    *,
    log_dir: Path,
) -> None:
    lines.append("## Local performance baseline comparison")
    lines.append("")
    summary, grouped_rows = _buildkite_perf_rows(kanban_cfg, log_dir=log_dir)
    _append_buildkite_perf_markdown(lines, summary, grouped_rows)


def _append_buildkite_perf_markdown(
    lines: list[str],
    summary: dict[str, Any],
    grouped_rows: dict[str, list[list[str]]],
    *,
    model_heading_level: int = 4,
) -> None:
    # Verbose diagnostic lines (Data source / Local filter / History / History generated_at /
    # Description / Raw data fallback diagnostics) were removed: the perf comparison block
    # should focus on per-model baseline rows, not on data-source plumbing. Diagnostics
    # remain available in `_buildkite_perf_rows` / `_kanban_fallback_items` for tooling.
    for warning in summary.get("warnings") or []:
        lines.append(f"- **Note:** {_md_cell(str(warning))}")
    if summary.get("status") != "ok":
        lines.append("")
        return
    stats = summary.get("summary", {})
    per_file_days = summary.get("latest_day_per_file") or {}
    per_file_str = ", ".join(f"{k}={v}" for k, v in sorted(per_file_days.items())) if per_file_days else "n/a"
    lines.append(f"- **Latest date per file:** `{summary.get('latest_day')}` (freshest; per-file = {per_file_str})")
    lines.append(
        f"- **Stats:** pass `{int(stats.get('pass', 0))}` / "
        f"normal `{int(stats.get('normal', 0))}` / "
        f"fail `{int(stats.get('fail', 0))}` / n-a `{int(stats.get('n/a', 0))}`"
    )
    lines.append("")
    lines.append("*Grouped by model (Markdown has no collapse).*")
    lines.append("")
    heading_prefix = "#" * model_heading_level
    for model_name in sorted(grouped_rows.keys()):
        lines.append(f"{heading_prefix} {model_name}")
        lines.append("")
        lines.append(
            render_markdown_table(
                _PERF_TABLE_HEADERS,
                grouped_rows[model_name],
            )
        )
        lines.append("")


def _append_buildkite_markdown(
    lines: list[str],
    bk_build: dict[str, Any] | None,
    bk_jobs: list[dict[str, Any]] | None,
    bk_note: str | None,
    kanban_cfg: KanbanAssetsConfig,
    target: BkTarget,
    *,
    log_dir: Path | None = None,
) -> None:
    """Render one collapsible Buildkite chapter (Markdown).

    Each chapter is wrapped in a ``<details>`` block so it can be folded in
    the rendered Markdown viewer. The chapter content mirrors the existing
    Buildkite rendering (build metadata, per-job summary, perf comparison,
    per-step failure analysis) — only ``build_url`` and the chapter label
    change per target.
    """
    lines.append(f"### Buildkite ({target.label}): latest scheduled nightly")
    lines.append("")
    lines.append(
        f"<details><summary><strong>{target.label} — Buildkite ({target.org}/{target.pipeline})</strong></summary>"
    )
    lines.append("")
    if bk_note:
        lines.append(bk_note)
        lines.append("")
        lines.append("### Performance baseline comparison")
        lines.append("")
        summary, grouped_rows = _buildkite_perf_rows(
            kanban_cfg,
            log_dir=log_dir,
            exclude_local_overlap=log_dir is not None,
        )
        _append_buildkite_perf_markdown(lines, summary, grouped_rows)
        lines.append("")
        lines.append("</details>")
        lines.append("")
        return
    if not bk_build or bk_jobs is None:
        lines.append(f"*(Buildkite ({target.label}) section not available.)*")
        lines.append("")
        lines.append("### Performance baseline comparison")
        lines.append("")
        summary, grouped_rows = _buildkite_perf_rows(
            kanban_cfg,
            log_dir=log_dir,
            exclude_local_overlap=log_dir is not None,
        )
        _append_buildkite_perf_markdown(lines, summary, grouped_rows)
        lines.append("")
        lines.append("</details>")
        lines.append("")
        return
    bn = int(bk_build["number"])
    build_url = f"https://buildkite.com/{target.org}/{target.pipeline}/builds/{bn}"
    lines.append(f"- **Build:** [{bn}]({build_url})")
    lines.append(f"- **State:** `{bk_build.get('state') or ''}`")
    lines.append(f"- **Message:** {_md_cell((bk_build.get('message') or '')[:500])}")
    co = (bk_build.get("commit") or "")[:12]
    if co:
        lines.append(f"- **Commit:** `{co}`")
    lines.append("")
    sum_rows = [_summary_row_for_bk_rec(r) for r in bk_jobs]
    lines.append(
        render_markdown_table(
            ["Job", "Total", "Passed", "Failed", "Skipped", "Errors", "Elapsed time"],
            sum_rows,
        )
    )
    lines.append("")
    lines.append("*Failed Buildkite steps only: detailed excerpts below. Passing steps are in the table only.*")
    lines.append("")
    lines.append("### Performance baseline comparison")
    lines.append("")
    summary, grouped_rows = _buildkite_perf_rows(
        kanban_cfg,
        log_dir=log_dir,
        exclude_local_overlap=log_dir is not None,
    )
    _append_buildkite_perf_markdown(lines, summary, grouped_rows)
    for rec in bk_jobs:
        info = rec.get("info")
        if rec.get("log_error"):
            lines.append(f"### Buildkite step: `{_md_cell(rec['name'])}` (log fetch failed)")
            lines.append("")
            lines.append(f"- **Step link:** {rec['step_link']}")
            lines.append(f"- **Error:** {_md_cell(rec['log_error'][:500])}")
            lines.append("")
            continue
        if not info or _job_is_clean(info):
            continue
        lines.append(f"### Buildkite step: `{_md_cell(rec['name'])}`")
        lines.append("")
        lines.append(f"- **Step link:** [{rec['step_link']}]({rec['step_link']})")
        lines.append("")
        fail_rows: list[list[str]] = []
        for node in info["failed_nodes"]:
            fail_rows.append(
                [
                    _md_cell(node),
                    _md_cell(info["failed_reasons"].get(node, "")),
                    _md_cell(info["failure_analyses"].get(node, "")),
                    _excerpt_md_cell(info["failure_excerpts"].get(node, "")),
                    "Submit issue",
                    "Filed / Not an issue",
                ]
            )
        for node in info["error_nodes"]:
            fail_rows.append(
                [
                    _md_cell(node) + " (ERROR)",
                    _md_cell(info["error_reasons"].get(node, "")),
                    _md_cell(info["error_analyses"].get(node, "")),
                    _excerpt_md_cell(info["error_excerpts"].get(node, "")),
                    "Submit issue",
                    "Filed / Not an issue",
                ]
            )
        lines.append("#### Failures & errors")
        lines.append("")
        lines.append(
            render_markdown_table(
                ["Test node", "Log reason", "Analysis", "Excerpt (truncated)", "Submit Issue", "Status"],
                fail_rows,
            )
        )
        lines.append("")
    lines.append("</details>")
    lines.append("")


def _excerpt_md_cell(excerpt: str, limit: int = 900) -> str:
    """Render excerpt in Markdown table cell, preserving line breaks.

    Uses HTML-like line break markers since standard Markdown tables don't support
    multi-line content. The excerpt is truncated if too long, but line breaks
    are preserved as visible separators for readability.
    """
    t = (excerpt or "").strip()
    if not t:
        return _md_cell("—")
    # Truncate if needed but preserve structure
    if len(t) > limit:
        lines = t.splitlines()
        # Truncate by lines first for better readability
        truncated_lines = []
        total_len = 0
        for line in lines:
            if total_len + len(line) + 1 > limit - 3:
                break
            truncated_lines.append(line)
            total_len += len(line) + 1
        if truncated_lines:
            t = "\n".join(truncated_lines) + "\n…"
        else:
            t = t[: limit - 1] + "…"
    # Use explicit line break representation for Markdown table cells
    # Replace newlines with a visible separator that HTML can render
    t = t.replace("\n", "  \n")  # Two spaces + newline = line break in Markdown
    return _md_cell(t)


def emit_report(
    *,
    title: str,
    repo_root: Path,
    log_dir: Path,
    out_fp: Any,
    bk_results: dict[BkTarget, tuple[dict[str, Any] | None, list[dict[str, Any]] | None, str | None]] | None = None,
    kanban_cfg: KanbanAssetsConfig | None = None,
) -> None:
    groups = discover_job_logs(log_dir)
    if kanban_cfg is None:
        kanban_cfg = KanbanAssetsConfig(
            assets_dir=DEFAULT_KANBAN_ASSETS_DIR,
            repo_root=DEFAULT_KANBAN_REPO_ROOT,
        )
    if bk_results is None:
        bk_results = {t: (None, None, None) for t in ALL_BK_TARGETS}

    lines: list[str] = [
        f"# {_md_cell(title)}",
        "",
    ]

    job_rows = _local_job_rows_with_info(groups) if groups else []
    # Daily Focus uses the CUDA (canonical) Buildkite data — that is the
    # pipeline that matches the local H200/H800/A100 runs by default.
    cuda_build, cuda_jobs, _ = bk_results.get(CUDA_TARGET, (None, None, None))
    _append_daily_focus_markdown(
        lines,
        _daily_focus_data(
            bk_jobs=cuda_jobs,
            local_job_rows=job_rows,
            kanban_cfg=kanban_cfg,
            log_dir=log_dir,
        ),
    )

    lines.append("## Buildkite Test")
    lines.append("")
    lines.append("Scheduled nightly — CUDA & NPU chapters. Click a chapter to expand.")
    lines.append("")
    for target in ALL_BK_TARGETS:
        bk_build, bk_jobs, bk_note = bk_results.get(target, (None, None, None))
        _append_buildkite_markdown(
            lines,
            bk_build,
            bk_jobs,
            bk_note,
            kanban_cfg,
            target,
            log_dir=log_dir,
        )

    lines.append("## Local cluster (nightly_jobs)")
    lines.append("")

    if not groups:
        lines.append(
            f"*No job logs found under `{log_dir}`. "
            "Confirm nightly jobs ran, copy logs from the cluster "
            "(vllm-omni-local-test references/nightly-local-log-fetch.md), "
            "and match paths in references/nightly-local-log-layout.md.*"
        )
        lines.append("")
        _append_local_perf_baseline_markdown(lines, kanban_cfg, log_dir=log_dir)
        print("\n".join(lines), file=out_fp)
        return

    lines.append("### Summary")
    lines.append("")
    _append_local_summary_grouped_markdown(lines, job_rows)
    lines.append(
        "*Failed and errored jobs only: detailed excerpts below. Passing jobs appear in the summary table only.*"
    )
    lines.append("")
    _append_local_perf_baseline_markdown(lines, kanban_cfg, log_dir=log_dir)

    for job_name, paths, info in job_rows:
        if _job_is_clean(info):
            continue
        lines.append(f"### Local job: `{_md_cell(job_name)}`")
        lines.append("")
        rel = ", ".join(f"`{p.name}`" for p in paths)
        lines.append(f"- {rel}")
        lines.append("")

        fail_rows: list[list[str]] = []
        for node in info["failed_nodes"]:
            fail_rows.append(
                [
                    _md_cell(node),
                    _md_cell(info["failed_reasons"].get(node, "")),
                    _md_cell(info["failure_analyses"].get(node, "")),
                    _excerpt_md_cell(info["failure_excerpts"].get(node, "")),
                ]
            )
        for node in info["error_nodes"]:
            fail_rows.append(
                [
                    _md_cell(node) + " (ERROR)",
                    _md_cell(info["error_reasons"].get(node, "")),
                    _md_cell(info["error_analyses"].get(node, "")),
                    _excerpt_md_cell(info["failure_excerpts"].get(node, "")),
                ]
            )

        lines.append("#### Failures & errors")
        lines.append("")
        lines.append(
            render_markdown_table(
                ["Test node", "Log reason", "Analysis", "Excerpt (truncated)", "Submit Issue", "Status"],
                fail_rows,
            )
        )
        lines.append("")

    print("\n".join(lines), file=out_fp)


def _excerpt_storage_id(report_context: str, node: str, row_index: int) -> str:
    digest = hashlib.sha1(f"{report_context}\0{node}\0{row_index}".encode()).hexdigest()[:12]
    return f"log-excerpt-{digest}"


def _excerpt_cell_html(
    excerpt: str,
    *,
    storage_id: str,
    title: str,
    max_chars: int = 0,
    button_label: str = "View error log",
) -> str:
    """Render the **Excerpt** cell with a button that opens the in-page log modal.

    By default the **full** log is rendered into the hidden ``<pre>`` store so the
    modal can show the entire failure context (failure reason, stack trace, full
    pytest output). Set ``max_chars`` to a non-zero value if a downstream caller
    really needs to cap the size — left at ``0`` for the failure tables so
    users no longer hit the legacy ``... [truncated]`` cutoff when triaging.
    """
    t = (excerpt or "").strip()
    if not t:
        return '<span class="note">—</span>'
    if max_chars and len(t) > max_chars:
        t = t[:max_chars] + "\n... [truncated]"
    safe_id = html.escape(storage_id)
    safe_title = html.escape(title, quote=True)
    safe_label = html.escape(button_label)
    return (
        '<div class="excerpt-cell-inner">'
        f'<button type="button" class="btn-view-log-excerpt" '
        f'data-modal-target="{safe_id}" data-log-title="{safe_title}">'
        f"{safe_label}</button>"
        f'<pre id="{safe_id}" class="log-excerpt log-excerpt--stored" hidden>'
        f"{html.escape(t)}</pre>"
        "</div>"
    )


def _log_excerpt_modal_html() -> str:
    return """
<div id="log-excerpt-modal" class="log-excerpt-modal" hidden role="dialog" aria-modal="true"
  aria-labelledby="log-excerpt-modal-title">
  <div class="log-excerpt-modal-backdrop" data-log-modal-close aria-hidden="true"></div>
  <div class="log-excerpt-modal-panel">
    <header class="log-excerpt-modal-header">
      <h2 id="log-excerpt-modal-title" data-log-modal-title>Log excerpt</h2>
      <button type="button" class="log-excerpt-modal-close" data-log-modal-close aria-label="Close">&times;</button>
    </header>
    <div class="log-excerpt-modal-body">
      <pre data-log-modal-pre></pre>
    </div>
  </div>
</div>
"""


def _th_labeled(icon_paths: str, text: str, *, col_class: str = "") -> str:
    cls = f' class="{col_class}"' if col_class else ""
    return (
        f'<th scope="col"{cls}>'
        '<span class="th-lbl">'
        f"{_svg_icon(icon_paths, size=16, extra_class='th-ico')}"
        f"<span>{html.escape(text)}</span>"
        "</span></th>"
    )


def _buildkite_build_url(
    build: dict[str, Any] | None,
    target: BkTarget | None = None,
) -> str:
    if not build:
        return ""
    bn = build.get("number")
    if bn is None:
        return ""
    org = target.org if target is not None else ORG
    pipeline = target.pipeline if target is not None else PIPELINE
    return f"https://buildkite.com/{org}/{pipeline}/builds/{int(bn)}"


def _issue_row_data_attrs(
    *,
    issue_env: str = "local",
    issue_vllm_version: str = "",
    issue_vllm_omni_version: str = "",
    issue_build_commit: str = "",
    buildkite_build_url: str = "",
    buildkite_step_url: str = "",
    buildkite_step_name: str = "",
) -> str:
    def aq(s: str) -> str:
        return html.escape(s or "", quote=True)

    return (
        f'data-issue-env="{aq(issue_env)}" '
        f'data-vllm-version="{aq(issue_vllm_version)}" '
        f'data-vllm-omni-version="{aq(issue_vllm_omni_version)}" '
        f'data-build-commit="{aq(issue_build_commit)}" '
        f'data-buildkite-build-url="{aq(buildkite_build_url)}" '
        f'data-buildkite-step-url="{aq(buildkite_step_url)}" '
        f'data-buildkite-step-name="{aq(buildkite_step_name)}"'
    )


def _render_failures_table_html(
    info: dict[str, Any],
    *,
    report_context: str = "",
    issue_env: str = "local",
    issue_vllm_version: str = "",
    issue_vllm_omni_version: str = "",
    issue_build_commit: str = "",
    buildkite_build_url: str = "",
    buildkite_step_url: str = "",
    buildkite_step_name: str = "",
    full_log_text: str = "",  # NEW: complete log for this job/step
) -> str:
    ctx_attr = html.escape(report_context, quote=True)
    row_ex = _issue_row_data_attrs(
        issue_env=issue_env,
        issue_vllm_version=issue_vllm_version,
        issue_vllm_omni_version=issue_vllm_omni_version,
        issue_build_commit=issue_build_commit,
        buildkite_build_url=buildkite_build_url,
        buildkite_step_url=buildkite_step_url,
        buildkite_step_name=buildkite_step_name,
    )
    parts: list[str] = [
        '<table class="fail-table">',
        "<thead><tr>",
        _th_labeled(_SVG_CODE, "Test node"),
        _th_labeled(_SVG_MSG, "Log reason"),
        _th_labeled(_SVG_SPARK, "Analysis"),
        _th_labeled(_SVG_LOG, "Full log", col_class="excerpt-col"),
        _th_labeled(_SVG_PLUS_ISSUE, "GitHub Issue"),
        '<th class="status-col"><span class="th-lbl">Status</span></th>',
        "</tr></thead>",
        "<tbody>",
    ]
    row_index = 0
    for node in info["failed_nodes"]:
        # Use full log instead of excerpt for better debugging
        log_content = full_log_text if full_log_text else info["failure_excerpts"].get(node, "")
        storage_id = _excerpt_storage_id(report_context, node, row_index)
        row_id = f"{report_context}::{node}"
        row_index += 1
        parts.extend(
            [
                f'<tr {row_ex} data-report-context="{ctx_attr}" data-row-id="{html.escape(row_id, quote=True)}">',
                f'<td class="mono">{html.escape(node)}</td>',
                f'<td class="reason">{html.escape(info["failed_reasons"].get(node, ""))}</td>',
                f'<td class="analysis">{html.escape(info["failure_analyses"].get(node, ""))}</td>',
                f'<td class="excerpt-cell">'
                f"{_excerpt_cell_html(log_content, storage_id=storage_id, title=node, button_label='View full log')}"
                f"</td>",
                _github_issue_button_cell(),
                _fail_status_cell_html(row_id),
                "</tr>",
            ]
        )
    for node in info["error_nodes"]:
        label = f"{node} (ERROR)"
        # Use full log instead of excerpt
        log_content = full_log_text if full_log_text else info["error_excerpts"].get(node, "")
        storage_id = _excerpt_storage_id(report_context, label, row_index)
        row_id = f"{report_context}::{label}"
        row_index += 1
        parts.extend(
            [
                f'<tr class="row-error" {row_ex} data-report-context="{ctx_attr}"'
                f' data-row-id="{html.escape(row_id, quote=True)}">',
                f'<td class="mono">{html.escape(node)} (ERROR)</td>',
                f'<td class="reason">{html.escape(info["error_reasons"].get(node, ""))}</td>',
                f'<td class="analysis">{html.escape(info["error_analyses"].get(node, ""))}</td>',
                f'<td class="excerpt-cell">'
                f"{_excerpt_cell_html(log_content, storage_id=storage_id, title=label, button_label='View full log')}"
                f"</td>",
                _github_issue_button_cell(),
                _fail_status_cell_html(row_id),
                "</tr>",
            ]
        )
    parts.append("</tbody></table>")
    return _table_wrap("\n".join(parts))


def _render_buildkite_section_html(
    build: dict[str, Any] | None,
    job_records: list[dict[str, Any]] | None,
    *,
    note: str | None,
    kanban_cfg: KanbanAssetsConfig,
    target: BkTarget,
    log_dir: Path | None = None,
    open_default: bool = False,
) -> str:
    """Render one Buildkite chapter as a collapsible ``<details>`` block.

    Each chapter (CUDA / NPU) is wrapped in its own ``<details>`` element so
    the user can fold or expand the whole chapter independently. Inner
    sub-cards (Summary / Performance / Failure analysis) retain their
    existing per-section collapsibility.
    """
    summary_inner: list[str] = []
    fail_inner = '<p class="note">No data: Buildkite step logs were not loaded.</p>'

    if note:
        summary_inner.append(f'<p class="note">{html.escape(note)}</p>')
    elif build is None or job_records is None:
        summary_inner.append(f'<p class="note">Buildkite ({target.label}) section not available.</p>')
    else:
        bn = int(build["number"])
        build_url = f"https://buildkite.com/{target.org}/{target.pipeline}/builds/{bn}"
        meta_lines: list[str] = [
            '<div class="meta">',
            f'<div><strong>Build:</strong> <a href="{html.escape(build_url)}">#{bn}</a></div>',
            f"<div><strong>State:</strong> {html.escape(str(build.get('state') or ''))}</div>",
            f"<div><strong>Message:</strong> {html.escape((build.get('message') or '')[:500])}</div>",
        ]
        co = (build.get("commit") or "")[:12]
        if co:
            meta_lines.append(f"<div><strong>Commit:</strong> {html.escape(co)}</div>")
        meta_lines.append("</div>")
        summary_inner = ["".join(meta_lines)]
        sum_rows = [_summary_row_for_bk_rec(r) for r in job_records]
        sum_row_cls = [f"summary-row summary-row--{_summary_row_kind_bk(r)}" for r in job_records]
        summary_inner.append(
            _table_wrap(
                render_html_table(
                    ["Job", "Total", "Passed", "Failed", "Skipped", "Errors", "Elapsed time"],
                    sum_rows,
                    table_class="summary",
                    row_classes=sum_row_cls,
                )
            )
        )

        fail_blocks: list[str] = []
        for rec in job_records:
            if rec.get("log_error"):
                bits_bk_err = [
                    '<details class="job-fail-details job-fail-details-bk">',
                    '<summary class="job-fail-details-summary job-fail-details-summary-bk">',
                    _heading_html(
                        "h2",
                        _SVG_ALERT,
                        html.escape(f"Buildkite step: {rec['name']}"),
                    ),
                    '<p class="meta"><strong>Step link:</strong> '
                    f'<a href="{html.escape(rec["step_link"])}">open</a></p>',
                    "</summary>",
                    '<div class="job-fail-details-body">',
                    f'<p class="note"><strong>Log fetch failed:</strong> {html.escape(rec["log_error"][:600])}</p>',
                    "</div></details>",
                ]
                fail_blocks.append("\n".join(bits_bk_err))
                continue
            info = rec.get("info")
            if not info or _job_is_clean(info):
                continue
            bits_bk = [
                '<details class="job-fail-details job-fail-details-bk">',
                '<summary class="job-fail-details-summary job-fail-details-summary-bk">',
                _heading_html(
                    "h2",
                    _SVG_ALERT,
                    html.escape(f"Buildkite step: {rec['name']}"),
                ),
                f'<p class="meta"><strong>Step link:</strong> <a href="{html.escape(rec["step_link"])}">open</a></p>',
                "</summary>",
                '<div class="job-fail-details-body">',
                _heading_html(
                    "h3",
                    _SVG_LIST,
                    html.escape("Failures & errors"),
                    klass="section-failures",
                ),
                _render_failures_table_html(
                    info,
                    report_context=(
                        f"Buildkite ({target.label}) scheduled nightly · build #{bn} · step: {rec['name']}"
                    ),
                    issue_env="ci",
                    issue_vllm_version=(rec.get("ci_versions") or {}).get("vllm", ""),
                    issue_vllm_omni_version=(rec.get("ci_versions") or {}).get("vllm_omni", ""),
                    issue_build_commit=(rec.get("build_commit_short") or ""),
                    buildkite_build_url=build_url,
                    buildkite_step_url=str(rec.get("step_link") or ""),
                    buildkite_step_name=str(rec.get("name") or ""),
                    full_log_text=(rec.get("raw_log") or ""),  # NEW: pass complete step log
                ),
                "</div></details>",
            ]
            fail_blocks.append("\n".join(bits_bk))

        if fail_blocks:
            fail_inner = (
                '<p class="hint">Click each step title to expand or collapse failure details. '
                "Use <strong>View full log</strong> to open the complete step log in a dialog.</p>\n"
                + "\n".join(fail_blocks)
            )
        else:
            fail_inner = '<p class="note">No failed steps currently, or no failure/error excerpts for any job.</p>'

    chapter_heading = _heading_html(
        "h3",
        _SVG_CLOUD,
        html.escape(f"Buildkite ({target.label})"),
        sub=html.escape(f"{target.org}/{target.pipeline} · latest scheduled nightly"),
        klass="bk-chapter-heading",
    )
    sub_cards: list[str] = [
        _details_subcard(
            "Summary (per-job execution)",
            "\n".join(summary_inner),
            open_default=False,
            details_class="report-subcard--bk",
            icon_paths=_SVG_LIST,
        ),
        _details_subcard(
            "Failure analysis",
            fail_inner,
            open_default=False,
            details_class="report-subcard--bk-fail",
            icon_paths=_SVG_ALERT,
        ),
    ]
    # CUDA keeps the kanban-backed Performance baseline comparison.
    # NPU pipeline has no kanban-side perf history yet, so the perf sub-card
    # is omitted there. Toggle by adding/removing this `if` branch.
    if target is not NPU_TARGET:
        sub_cards.insert(
            1,
            _details_subcard(
                "Performance baseline comparison",
                _render_buildkite_perf_inner_html(
                    kanban_cfg,
                    log_dir=log_dir,
                    exclude_local_overlap=log_dir is not None,
                ),
                open_default=False,
                details_class="report-subcard--bk-perf",
                icon_paths=_SVG_CHART_BARS,
            ),
        )
    chapter_body = "\n".join(sub_cards)

    op = " open" if open_default else ""
    return (
        f'<details class="bk-chapter bk-chapter--{target.label.lower()}"{op}>'
        f'<summary class="bk-chapter-summary">{chapter_heading}</summary>'
        f'<div class="bk-chapter-body">'
        f"{chapter_body}"
        f"</div>"
        f"</details>"
    )


def _render_buildkite_note_html(note: str, target: BkTarget | None = None) -> str:
    label = target.label if target is not None else "CUDA"
    inner = f'<p class="note">{html.escape(note)}</p>'
    return "\n".join(
        [
            '<section class="panel nightly-root nightly-root--buildkite">',
            _heading_html(
                "h2",
                _SVG_CLOUD,
                html.escape(f"Buildkite ({label})"),
                sub=html.escape("Latest scheduled nightly (main)"),
            ),
            _details_subcard(
                "Summary (per-job execution)",
                inner,
                open_default=False,
                details_class="report-subcard--bk",
                icon_paths=_SVG_LIST,
            ),
            _details_subcard(
                "Failure analysis",
                '<p class="note">No data: Buildkite step logs were not loaded.</p>',
                open_default=False,
                details_class="report-subcard--bk-fail",
                icon_paths=_SVG_ALERT,
            ),
            "</section>",
        ]
    )


def emit_report_html(
    *,
    title: str,
    repo_root: Path,
    log_dir: Path,
    out_fp: Any,
    bk_results: dict[BkTarget, tuple[dict[str, Any] | None, list[dict[str, Any]] | None, str | None]] | None = None,
    kanban_cfg: KanbanAssetsConfig | None = None,
) -> None:
    groups = discover_job_logs(log_dir)
    if kanban_cfg is None:
        kanban_cfg = KanbanAssetsConfig(
            assets_dir=DEFAULT_KANBAN_ASSETS_DIR,
            repo_root=DEFAULT_KANBAN_REPO_ROOT,
        )
    if bk_results is None:
        bk_results = {t: (None, None, None) for t in ALL_BK_TARGETS}

    css = EDITORIAL_THEME_CSS

    body_parts: list[str] = [
        '<div class="top-bar"><div class="shell top-bar-inner">'
        '<div class="brand">'
        f'<div class="brand-mark">{_svg_icon(_SVG_CLIPBOARD, size=30, extra_class="brand-ico")}</div>'
        '<div class="brand-copy">'
        f"<h1>{html.escape(title)}</h1>"
        '<p class="tagline">Nightly test report</p>'
        "</div></div></div></div>",
        '<div class="shell">',
    ]

    job_rows: list[tuple[str, list[Path], dict[str, Any]]] = []
    if groups:
        job_rows = _local_job_rows_with_info(groups)
    # Daily Focus uses the CUDA (canonical) Buildkite data — that is the
    # pipeline that matches the local H200/H800/A100 runs by default.
    cuda_build, cuda_jobs, _ = bk_results.get(CUDA_TARGET, (None, None, None))
    body_parts.append(
        _render_daily_focus_html(
            _daily_focus_data(
                bk_jobs=cuda_jobs,
                local_job_rows=job_rows,
                kanban_cfg=kanban_cfg,
                log_dir=log_dir,
            )
        )
    )

    body_parts.append(
        '<details class="bk-section panel nightly-root nightly-root--buildkite"><summary class="bk-section-summary">'
    )
    body_parts.append(
        _heading_html(
            "h2",
            _SVG_CLOUD,
            html.escape("Buildkite Test"),
            sub=html.escape("Scheduled nightly — CUDA & NPU chapters (click to expand)"),
        )
    )
    body_parts.append("</summary>")
    body_parts.append('<div class="bk-section-body">')
    for target in ALL_BK_TARGETS:
        bk_build, bk_jobs, bk_note = bk_results.get(target, (None, None, None))
        body_parts.append(
            _render_buildkite_section_html(
                bk_build,
                bk_jobs,
                note=bk_note,
                kanban_cfg=kanban_cfg,
                target=target,
                log_dir=log_dir,
            )
        )
    body_parts.append("</div>")
    body_parts.append("</details>")

    local_chunks: list[str] = [
        '<section class="panel nightly-root nightly-root--local">',
        _heading_html(
            "h2",
            _SVG_SERVER,
            html.escape("Local Test"),
        ),
    ]
    if not groups:
        summary_body = (
            '<p class="note">No job logs found under this directory. Confirm nightly jobs ran, copy logs '
            "(see vllm-omni-local-test references/nightly-local-log-fetch.md), "
            "and match references/nightly-local-log-layout.md.</p>"
        )
    else:
        summary_body = _render_local_summary_grouped_html(job_rows)
    local_chunks.append(
        _details_subcard(
            "Summary",
            summary_body,
            open_default=False,
            details_class="",
            icon_paths=_SVG_LIST,
        )
    )
    local_chunks.append(
        _details_subcard(
            "Performance baseline comparison",
            _render_buildkite_perf_inner_html(
                kanban_cfg,
                model_subcard_class="report-subcard--local-perf-model",
                log_dir=log_dir,
            ),
            open_default=False,
            details_class="report-subcard--local-perf-baseline",
            icon_paths=_SVG_CHART_BARS,
        )
    )

    fail_local_parts: list[str] = []
    # Local failures link back to the canonical CUDA Buildkite build URL
    # (the same CI pipeline the local H200/H800/A100 runs target).
    local_bk_build_url = _buildkite_build_url(cuda_build, CUDA_TARGET)
    if job_rows:
        for job_name, paths, info in job_rows:
            if _job_is_clean(info):
                continue
            log_files = ", ".join(p.name for p in paths)
            bits = [
                '<details class="job-fail-details">',
                '<summary class="job-fail-details-summary">',
                _heading_html(
                    "h2",
                    _SVG_ALERT,
                    html.escape(f"Failed local job: {job_name}"),
                ),
                "</summary>",
                '<div class="job-fail-details-body">',
                _heading_html(
                    "h3",
                    _SVG_LIST,
                    html.escape("Failures & errors"),
                    klass="section-failures",
                ),
                _render_failures_table_html(
                    info,
                    report_context=(f"Local nightly_jobs · job: {job_name} · logs: {log_files}"),
                    buildkite_build_url=local_bk_build_url,
                    full_log_text=read_job_text(paths),
                ),
                "</div></details>",
            ]
            fail_local_parts.append("\n".join(bits))
    if fail_local_parts:
        fail_inner_loc = (
            '<p class="hint">Click each job title to expand or collapse failed test lists. '
            "Use the per-row <strong>View full log</strong> in the failure table to open the "
            "raw job log excerpt.</p>\n" + "\n".join(fail_local_parts)
        )
    else:
        fail_inner_loc = '<p class="note">No failures or errors require itemized analysis.</p>'
    local_chunks.append(
        _details_subcard(
            "Failure analysis",
            fail_inner_loc,
            open_default=False,
            details_class="report-subcard--local-fail",
            icon_paths=_SVG_ALERT,
        )
    )
    local_chunks.append("</section>")
    body_parts.append("\n".join(local_chunks))

    body_parts.append("</div>")
    body_parts.append(_log_excerpt_modal_html())
    body_parts.append(_fail_status_modal_html())
    body_parts.append(_ut_coverage_modal_html())
    doc = _html_document(
        title,
        css,
        "\n".join(body_parts),
        tail=_github_issue_submit_script(),
    )
    print(doc, file=out_fp)


def _html_document(title: str, css: str, body: str, *, tail: str = "") -> str:
    t = html.escape(title)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{t}</title>
<style>
{css}
</style>
</head>
<body>
{body}
{tail}
</body>
</html>
"""


def _resolve_buildkite_target(
    tok: str | None,
    target: BkTarget,
    build_no: int | None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None, str | None]:
    """Return ``(build, job_records, note)`` for one Buildkite target.

    ``note`` is set when the section is skipped or errored; ``build``/``job_records``
    are ``None`` in those cases.
    """
    if not tok:
        return (
            None,
            None,
            (
                f"Buildkite ({target.label}) section skipped: set BUILDKITE_TOKEN or "
                f"BUILDKITE_API_TOKEN to fetch the latest scheduled nightly on "
                f"{target.org}/{target.pipeline}."
            ),
        )
    try:
        bk_build = fetch_nightly_build(
            tok,
            build_no,
            org=target.org,
            pipeline=target.pipeline,
            branch=target.branch,
        )
        bk_jobs = collect_nightly_job_log_analyses(bk_build, tok, org=target.org, pipeline=target.pipeline)
        return bk_build, bk_jobs, None
    except Exception as e:
        return None, None, f"Buildkite ({target.label}) section failed: {e}"


def _resolve_buildkite_for_report(
    include: bool,
    build_no: int | None,
    targets: tuple[BkTarget, ...] = ALL_BK_TARGETS,
) -> dict[BkTarget, tuple[dict[str, Any] | None, list[dict[str, Any]] | None, str | None]]:
    """Resolve the latest scheduled nightly for each target.

    Returns a mapping keyed by target → ``(build, jobs, note)``. When ``include``
    is ``False``, every entry is ``(None, None, None)`` so renderers show the
    section as a no-op.
    """
    if not include:
        return {t: (None, None, None) for t in targets}
    tok = _buildkite_token()
    return {t: _resolve_buildkite_target(tok, t, build_no) for t in targets}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Emit HTML by default (use --html-report or stdout). "
        "Use Markdown only when explicitly requested (--markdown-report / --to-stdout markdown). "
        "Local nightly job logs plus optional Buildkite latest scheduled nightly (needs token).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root: default log dir is <repo-root>/logs/nightly_jobs; "
        f"also shown in the report header (default: $REPO_ROOT or "
        f"{DEFAULT_LAPTOP_REPO_ROOT_DISPLAY}).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Log directory (default: <repo-root>/logs/nightly_jobs; "
        f"repo-root from --repo-root, $REPO_ROOT, or {DEFAULT_LAPTOP_REPO_ROOT_DISPLAY}).",
    )
    parser.add_argument(
        "--report-date",
        default=None,
        metavar="YYYY-MM-DD",
        help="UTC date for default output filename/title (default: today UTC). "
        "Never derived from logs/nightly_jobs_* directory suffixes.",
    )
    parser.add_argument(
        "--html-report",
        type=Path,
        default=None,
        help=(
            "Write HTML report to this file. "
            f"Default when omitted: <skill-dir>/{default_nightly_html_path(_SKILL_DIR).name}"
        ),
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=None,
        help="Write Markdown report to this file (optional).",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print report to stdout instead of the default dated HTML file.",
    )
    parser.add_argument(
        "--to-stdout",
        choices=("html", "markdown"),
        default="html",
        help="Format when using --stdout (default: html).",
    )
    parser.add_argument(
        "--no-buildkite",
        action="store_true",
        help="Do not call Buildkite API (local logs only).",
    )
    parser.add_argument(
        "--buildkite-build",
        type=int,
        default=None,
        help="Pin Buildkite build number (default: latest scheduled nightly on main).",
    )
    parser.add_argument(
        "--kanban-assets-dir",
        type=Path,
        default=DEFAULT_KANBAN_ASSETS_DIR,
        help=f"Path to {KANBAN_REPO_URL} docs/assets/charts for Buildkite performance summary.",
    )
    parser.add_argument(
        "--kanban-repo-root",
        type=Path,
        default=DEFAULT_KANBAN_REPO_ROOT,
        help=(
            f"Local clone root of {KANBAN_REPO_URL}; "
            "resolves docs/assets/charts and enables source validation "
            f"(default: $KANBAN_REPO_ROOT or {DEFAULT_KANBAN_REPO_ROOT_DISPLAY})."
        ),
    )
    parser.add_argument(
        "--kanban-expected-remote",
        default=None,
        help="Expected kanban upstream remote name (for warning only), e.g. upstream.",
    )
    parser.add_argument(
        "--kanban-expected-branch",
        default=None,
        help="Expected kanban branch name (for warning only), e.g. main.",
    )
    parser.add_argument(
        "--kanban-raw-root",
        type=Path,
        default=None,
        help="Optional kanban raw perf artifact root (default: <kanban-repo-root>/data/buildkite_nightly_raw).",
    )
    parser.add_argument(
        "--kanban-refresh-from-raw",
        action="store_true",
        help="Opt-in: run kanban raw sync + generate_charts before reading docs/assets/charts history.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Report title (default: Nightly Buildkite report - <report-date>).",
    )
    args = parser.parse_args()

    if args.html_report and args.markdown_report:
        print("Use only one of --html-report or --markdown-report.", file=sys.stderr)
        sys.exit(2)
    if args.stdout and (args.html_report or args.markdown_report):
        print("Use --stdout without --html-report or --markdown-report.", file=sys.stderr)
        sys.exit(2)

    report_date = resolve_report_date_iso(args.report_date)
    title = args.title or nightly_report_title(report_date)
    if args.html_report is None and args.markdown_report is None and not args.stdout:
        args.html_report = default_nightly_html_path(_SKILL_DIR, report_date)

    if args.repo_root is not None:
        repo = args.repo_root.resolve()
    else:
        repo = resolve_laptop_repo_root()

    log_dir = args.log_dir.resolve() if args.log_dir else default_log_dir(repo)

    bk_results = _resolve_buildkite_for_report(
        include=not args.no_buildkite,
        build_no=args.buildkite_build,
        targets=ALL_BK_TARGETS,
    )
    kanban_cfg = KanbanAssetsConfig(
        assets_dir=args.kanban_assets_dir.resolve() if args.kanban_assets_dir else None,
        repo_root=args.kanban_repo_root.resolve() if args.kanban_repo_root else None,
        expected_remote=(args.kanban_expected_remote or "").strip() or None,
        expected_branch=(args.kanban_expected_branch or "").strip() or None,
        raw_root=args.kanban_raw_root.resolve() if args.kanban_raw_root else None,
        refresh_from_raw=bool(args.kanban_refresh_from_raw),
    )
    if kanban_cfg.refresh_from_raw:
        refresh_note, refresh_warnings = _run_kanban_refresh_from_raw(
            kanban_cfg.repo_root,
            _resolve_kanban_raw_root(kanban_cfg),
        )
        kanban_cfg.refresh_note = refresh_note
        kanban_cfg.refresh_warnings = refresh_warnings

    if args.html_report:
        out = args.html_report
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fp:
            emit_report_html(
                title=title,
                repo_root=repo,
                log_dir=log_dir,
                out_fp=fp,
                bk_results=bk_results,
                kanban_cfg=kanban_cfg,
            )
        print(f"Wrote {out}")
    elif args.markdown_report:
        out = args.markdown_report
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fp:
            emit_report(
                title=title,
                repo_root=repo,
                log_dir=log_dir,
                out_fp=fp,
                bk_results=bk_results,
                kanban_cfg=kanban_cfg,
            )
        print(f"Wrote {out}")
    else:
        if args.to_stdout == "markdown":
            emit_report(
                title=title,
                repo_root=repo,
                log_dir=log_dir,
                out_fp=sys.stdout,
                bk_results=bk_results,
                kanban_cfg=kanban_cfg,
            )
        else:
            emit_report_html(
                title=title,
                repo_root=repo,
                log_dir=log_dir,
                out_fp=sys.stdout,
                bk_results=bk_results,
                kanban_cfg=kanban_cfg,
            )


if __name__ == "__main__":
    main()
