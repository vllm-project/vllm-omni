# Requested batch review and helper tools

Use only for a requested batch session, review follow-up, or proposed inline
comments. Commands below run from this skill directory, require `gh`, `jq`,
Python, and Bash, and only read GitHub. Resolve the connected reviewer's login;
do not infer it from examples or repository ownership.

This is generic candidate discovery, not a daily eligibility policy. If the
user's session specifies eligibility labels, a changed head since prior review,
a daily cap, or another selection rule, build that eligible set from current
metadata first. Do not let the legacy selector's own-PR/already-reviewed filters
silently override it: select directly when those filters would discard an
explicitly eligible re-review. User-selected PRs do not need to pass the helper.

1. Check outstanding replies first when the session includes follow-up:
   `bash scripts/check_replies.sh --reviewer <login> --days 14`.
   Reopen each returned thread and inspect its current resolution before
   deciding whether a response is needed. A reply can already be addressed by
   code or another maintainer; do not send an automatic acknowledgement.
2. Select candidates with
   `bash scripts/select_prs.sh --reviewer <login> --days 7 --limit 5`.
   This preserves the existing selector: open recent PRs, excluding own,
   draft/WIP/"don't merge", and already-reviewed PRs, with low-review-count
   candidates first. It does not filter ordinary docs/CI PRs or prove
   mergeability, CI success, or complete search coverage.
3. Freeze and review each selected PR using the normal execution contract.
   Keep CI conclusions separate from conflicts/mergeability. Match depth to
   changed behavior, not candidate rank; no quota requires manufacturing
   findings. Search/review APIs can be limited or fail; inspect raw status when
   coverage matters rather than treating a zero count as a verified clean state.
4. Before separately authorized inline posting, validate the proposed payload:
   `printf '%s' "$REVIEW_JSON" | bash scripts/verify_line_numbers.sh <PR>`.
   Input is an object with a `comments` array containing `path`, `line`,
   optional `side` (default `RIGHT`), and optional same-side `start_line`.
   It checks actual diff hunks, including deletion/rename paths. It does not
   verify the finding or submit it. Re-read base/head before and after its
   live diff fetch; require equality with the reviewed snapshot and use that
   head as `commit_id` for any authorized post. Otherwise refresh the review
   and line mapping first.
5. Preserve a short session record of pinned heads, findings, validation gaps,
   and already-posted comment links so a repeat pass reviews only the delta.

Run helper regression tests without project dependencies or GitHub access:

```bash
python tests/test_helpers.py
```

The helpers are bounded triage aids, not a scheduler. They do not authorize
follow-up comments, review events, code changes, or recurring monitoring.
