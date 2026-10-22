#!/usr/bin/env bash
# Archive reports DIRECTLY into JiusiServe/omni-reviewbot and push — the sole
# archive target since 2026-10-21 (user decision: STOP pushing report archives
# to vllm-omni-kanban; keep only the omni-review push).
#
# This script lives in the vllm-omni-test-report skill's scripts/ dir (skills
# branch of the vllm-omni-yn repo) and is SHARED by the sibling report skills
# (issue-monitor, pr-monitor, ci-daily, pool-stats), which invoke it as
# ../vllm-omni-test-report/scripts/archive_to_omni_review.sh relative to their
# own skill directory.
#
# Usage:   archive_to_omni_review.sh <src-path> <dest-data-path> [<src> <dest> ...]
# Example: ./scripts/archive_to_omni_review.sh \
#            ./pool-stats-2026-10-21.html data/pool_stats_report/pool-stats-2026-10-21.html
#          ./scripts/archive_to_omni_review.sh \
#            ~/issue-monitor-report-2026-10-21.html data/issue_monitor/issue-monitor-report.html \
#            /home/wy/vllm-omni-kanban/data/local_nightly_raw/manual_20261021 \
#            data/local_nightly_raw/manual_20261021
#
# Mapping:  dest `data/<X>`  ->  omni-reviewbot `kanban/data/vllm_omni/<X>`
#           (same subdir names as the old kanban layout)
# Sources:  any local file or directory (report HTML, manual_* payload dirs).
#           Nothing is read from or written to the vllm-omni-kanban repo.
# Repo:     clone root = $OMNI_REVIEW_REPO_ROOT, else /home/wy/omni-reviewbot,
#           else ~/omni-reviewbot.
# Push:     direct push to main (credentials: /root/.gitconfig github.com
#           helper, yenuo26, WRITE on JiusiServe/omni-reviewbot). Retries with
#           lowSpeedLimit; on non-fast-forward rejection (remote moved ahead,
#           or the initial fetch flaked and the commit landed on a stale base)
#           rebuilds the commit on the fresh origin/main and retries — a plain
#           rebase stalls on same-file conflicts (seen twice on 2026-10-22).
#           A data push with these credentials DOES trigger build-kanban-site
#           (paths: kanban/data/**), so the site rebuilds automatically.
# Identity: commits as yenuo26 <410167048@qq.com> (repo has no local git
#           user config) with the standard Claude co-author trailer (this is
#           a data archive, NOT a fix PR — the yenuo26-only signature rule is
#           scoped to fix PRs).
set -uo pipefail

REVIEW="${OMNI_REVIEW_REPO_ROOT:-/home/wy/omni-reviewbot}"
[ -d "$REVIEW" ] || REVIEW="$HOME/omni-reviewbot"

fail() { echo "ARCHIVE_ERROR: $*" >&2; exit 1; }

[ $# -ge 2 ] && [ $(( $# % 2 )) -eq 0 ] \
  || fail "usage: $0 <src-path> <dest-data-path> [<src> <dest> ...] (dest like data/pool_stats_report/x.html)"

# ---- validate pairs ----------------------------------------------------------
declare -a SRCS=() DESTS=() RELS=()
while [ $# -ge 2 ]; do
  src="$1"; dest="$2"; shift 2
  [ -e "$src" ] || fail "source not found: $src"
  case "$dest" in
    data/*) ;;
    *) fail "dest must start with data/ : $dest" ;;
  esac
  # resolve to an absolute path BEFORE the later `cd $REVIEW` — otherwise a
  # relative src (e.g. ./report.html) would be looked up inside the repo
  src="$(readlink -f "$src")"
  SRCS+=("$src"); DESTS+=("kanban/data/vllm_omni/${dest#data/}"); RELS+=("${dest#data/}")
done

# ---- prepare omni-reviewbot working tree -------------------------------------
cd "$REVIEW" || fail "cannot cd $REVIEW"
# self-heal: a previous run's retry-rebase can stop half-done on a same-file
# conflict and leave the repo unusable; abort it before the clean-tree check
# (only when the stopped rebase is one of OUR archive commits)
if [ -f .git/rebase-merge/done ] && grep -q "chore(data): archive reports" .git/rebase-merge/done 2>/dev/null; then
  git rebase --abort >/dev/null 2>&1 || true
  echo "WARN: aborted a stuck rebase from a previous archive run" >&2
fi
git diff --quiet && git diff --cached --quiet \
  || fail "$REVIEW has uncommitted changes; clean it first"
git fetch origin main >/dev/null 2>&1 \
  || echo "WARN: review fetch failed; continuing on local main" >&2
git merge --ff-only origin/main >/dev/null 2>&1 \
  || echo "WARN: cannot fast-forward to origin/main; push may need a rebase" >&2

# ---- copy --------------------------------------------------------------------
i=0
for rel in "${RELS[@]}"; do
  dst="$REVIEW/${DESTS[$i]}"
  mkdir -p "$(dirname "$dst")"
  rm -rf "$dst"
  cp -a "${SRCS[$i]}" "$dst" || fail "copy failed: ${SRCS[$i]} -> $dst"
  echo "archived: ${SRCS[$i]} -> ${DESTS[$i]}"
  i=$((i+1))
done

git add -- "${DESTS[@]}"
if git diff --cached --quiet; then
  echo "ARCHIVE_NOOP: omni-reviewbot already up to date for: ${RELS[*]}"
  exit 0
fi

TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
git -c user.name="yenuo26" -c user.email="410167048@qq.com" commit -q \
  -m "chore(data): archive reports (UTC $TS)" \
  -m "Archived: ${RELS[*]}" \
  -m "Co-Authored-By: Claude Code <noreply@anthropic.com>"

# ---- push with retry ---------------------------------------------------------
# Rebuild the archive commit on top of the fresh origin/main. The sources are
# still on disk, so abort/reset/re-copy/re-commit is idempotent — and unlike a
# plain `git rebase`, it cannot stall on a same-file conflict.
recommit() {
  git rebase --abort >/dev/null 2>&1 || true
  git fetch origin main >/dev/null 2>&1
  git reset --hard origin/main >/dev/null 2>&1
  local i=0 dst
  for rel in "${RELS[@]}"; do
    dst="$REVIEW/${DESTS[$i]}"
    mkdir -p "$(dirname "$dst")"
    rm -rf "$dst"
    cp -a "${SRCS[$i]}" "$dst" || fail "copy failed: ${SRCS[$i]} -> $dst"
    i=$((i+1))
  done
  git add -- "${DESTS[@]}"
  if git diff --cached --quiet; then
    echo "ARCHIVE_NOOP after rebuild on origin/main: ${RELS[*]}"
    exit 0
  fi
  git -c user.name="yenuo26" -c user.email="410167048@qq.com" commit -q \
    -m "chore(data): archive reports (UTC $(date -u +%Y-%m-%dT%H:%M:%SZ))" \
    -m "Archived: ${RELS[*]}" \
    -m "Co-Authored-By: Claude Code <noreply@anthropic.com>"
}

for try in 1 2 3 4 5 6; do
  if git -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=30 push origin main; then
    echo "ARCHIVE_PUSH_OK (try $try): ${RELS[*]}"
    exit 0
  fi
  recommit
  sleep 10
done
fail "push failed after 6 tries (commit kept locally in $REVIEW - retry later)"
