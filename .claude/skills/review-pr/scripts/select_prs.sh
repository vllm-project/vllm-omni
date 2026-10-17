#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# select_prs.sh — Pick reviewable PRs from vllm-omni for a review session.
#
# Usage: ./select_prs.sh [--days 7] [--limit 5] [--reviewer lishunyang12]
#
# Filters:
#   - Excludes your own PRs (--reviewer)
#   - Excludes WIP / Draft / "Don't merge" PRs
#   - Excludes pure docs/CI-only merged PRs
#   - Excludes PRs you already reviewed
#   - Prioritizes zero-review PRs
#
# Output: JSON array of {number, title, author, createdAt, reviewCount}

set -euo pipefail

REPO="vllm-project/vllm-omni"
DAYS=7
LIMIT=5
REVIEWER=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --days)    DAYS="$2"; shift 2 ;;
    --limit)   LIMIT="$2"; shift 2 ;;
    --reviewer) REVIEWER="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

SINCE=$(date -d "-${DAYS} days" +%Y-%m-%d 2>/dev/null || date -v-${DAYS}d +%Y-%m-%d)

echo "Fetching open PRs from last ${DAYS} days..." >&2

# Fetch all open PRs from the time range
ALL_PRS=$(gh pr list --repo "$REPO" --state open --limit 200 \
  --json number,title,author,createdAt,isDraft \
  --search "created:>=${SINCE}")

# Filter in a single jq pass
FILTERED=$(echo "$ALL_PRS" | jq --arg reviewer "$REVIEWER" '
  [.[] |
    # Skip own PRs
    select(.author.login != $reviewer) |
    # Skip drafts
    select(.isDraft == false) |
    # Skip WIP / Dont merge
    select(.title | test("\\[WIP\\]|\\[Draft\\]|Don.t[ ]+merge|\\[Don.t\\]"; "i") | not)
  ]
')

# For each remaining PR, check review count and whether reviewer already reviewed
echo "Checking review status for $(echo "$FILTERED" | jq length) PRs..." >&2

echo "$FILTERED" | jq -c '.[]' | while read -r pr; do
  NUM=$(echo "$pr" | jq -r '.number')

  # Get review count
  REVIEW_COUNT=$(gh api "repos/${REPO}/pulls/${NUM}/reviews" --jq 'length' 2>/dev/null || echo "0")

  # Check if reviewer already reviewed
  if [ -n "$REVIEWER" ]; then
    ALREADY=$(gh api "repos/${REPO}/pulls/${NUM}/reviews" \
      --jq "[.[] | select(.user.login==\"${REVIEWER}\")] | length" 2>/dev/null || echo "0")
    if [ "$ALREADY" -gt 0 ]; then
      continue
    fi
  fi

  echo "$pr" | jq --argjson rc "$REVIEW_COUNT" '. + {reviewCount: $rc}'
done | jq -s "sort_by(.reviewCount) | .[:${LIMIT}]"
