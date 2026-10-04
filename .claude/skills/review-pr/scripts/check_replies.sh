#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# check_replies.sh — Find unanswered contributor replies to your review comments.
#
# Usage: ./check_replies.sh --reviewer lishunyang12 [--days 14]
#
# Scans PRs you've reviewed for threads where a contributor replied
# but you haven't responded yet. Outputs actionable list.

set -euo pipefail

REPO="vllm-project/vllm-omni"
REVIEWER=""
DAYS=14

while [[ $# -gt 0 ]]; do
  case $1 in
    --reviewer) REVIEWER="$2"; shift 2 ;;
    --days)     DAYS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [ -z "$REVIEWER" ]; then
  echo "Usage: $0 --reviewer <github_login>" >&2
  exit 1
fi

SINCE=$(date -d "-${DAYS} days" +%Y-%m-%d 2>/dev/null || date -v-${DAYS}d +%Y-%m-%d)

echo "Scanning for unanswered replies to ${REVIEWER} (last ${DAYS} days)..." >&2

# Find PRs the reviewer has commented on recently
REVIEWED_PRS=$(gh api "search/issues?q=repo:${REPO}+reviewed-by:${REVIEWER}+is:pr+is:open+updated:>=${SINCE}&per_page=50" \
  --jq '.items[].number' 2>/dev/null)

UNANSWERED=0

for PR_NUM in $REVIEWED_PRS; do
  # Get all review comments on this PR
  COMMENTS=$(gh api "repos/${REPO}/pulls/${PR_NUM}/comments" \
    --jq '[.[] | {id, user: .user.login, body: .body[:100], in_reply_to_id, created_at, path, url: .html_url}]' 2>/dev/null)

  # Find threads: reviewer's comments that got a reply, but reviewer didn't reply back
  # 1. Get reviewer's comment IDs
  REVIEWER_IDS=$(echo "$COMMENTS" | jq -r --arg r "$REVIEWER" '[.[] | select(.user == $r) | .id]')

  # 2. Find replies to reviewer's comments (from others)
  REPLIES=$(echo "$COMMENTS" | jq -c --arg r "$REVIEWER" --argjson rids "$REVIEWER_IDS" '
    [.[] |
      select(.user != $r) |
      select(.in_reply_to_id != null) |
      select([.in_reply_to_id] | inside($rids))
    ]
  ')

  REPLY_COUNT=$(echo "$REPLIES" | jq 'length')
  if [ "$REPLY_COUNT" -eq 0 ]; then
    continue
  fi

  # 3. For each reply, check if reviewer replied after it
  while IFS= read -r reply; do
    THREAD_ID=$(echo "$reply" | jq -r '.in_reply_to_id')
    REPLY_DATE=$(echo "$reply" | jq -r '.created_at')
    REPLY_USER=$(echo "$reply" | jq -r '.user')
    REPLY_URL=$(echo "$reply" | jq -r '.url')

    # Check if reviewer replied after this reply
    FOLLOWUP=$(echo "$COMMENTS" | jq --arg r "$REVIEWER" --arg date "$REPLY_DATE" --argjson tid "$THREAD_ID" '
      [.[] | select(.user == $r) | select(.in_reply_to_id == $tid) | select(.created_at > $date)] | length
    ')

    if [ "$FOLLOWUP" -eq 0 ]; then
      REPLY_BODY=$(echo "$reply" | jq -r '.body')
      echo "PR #${PR_NUM} | @${REPLY_USER} replied | ${REPLY_URL}"
      echo "  > ${REPLY_BODY}"
      echo ""
      UNANSWERED=$((UNANSWERED + 1))
    fi
  done < <(echo "$REPLIES" | jq -c '.[]')
done

if [ "$UNANSWERED" -eq 0 ]; then
  echo "No unanswered replies found." >&2
fi
