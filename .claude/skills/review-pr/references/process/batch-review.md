# Requested batch review and helpers

Run from this skill directory with Bash, Python 3, `gh`, and `jq`. These tools
only read GitHub; they authorize no posting, code changes, or monitoring.
Resolve the connected reviewer's login rather than copying an example.

- For requested follow-up: `bash scripts/check_replies.sh --reviewer <login> --days 14`.
  Reopen returned threads and check resolution, code, and other replies before
  deciding whether a response is needed; never acknowledge automatically.
- For candidate discovery: `bash scripts/select_prs.sh --reviewer <login> --days 7 --limit 5`.
  It selects recent open PRs, excludes own/draft/WIP/"don't merge"/already-reviewed
  PRs, and favors fewer reviews. It does not exclude ordinary docs/CI changes.
  Apply the user's eligibility rules first; select directly for requested PRs or
  eligible re-reviews that these legacy filters would discard.
- Freeze and review each candidate using the normal workflow. APIs have bounded
  coverage and may fail; a zero count does not prove a clean state. Verify raw
  status when coverage matters. Candidate rank proves neither CI nor mergeability.
- Before separately authorized inline posting:
  `printf '%s' "$REVIEW_JSON" | bash scripts/verify_line_numbers.sh <PR>`.
  Input: an object with `comments`, each containing `path`, `line`, optional
  `side` (default `RIGHT`) and same-side `start_line`/`start_side`. It validates
  diff coordinates, including renames/deletions, not findings, and posts nothing.
  Re-read base/head before and after its live fetch; both must equal the reviewed
  snapshot. Otherwise refresh the review and mapping. Post with that `commit_id`.
- Keep pinned heads, findings, validation gaps and posted links for delta reviews.

Offline regression tests: `python tests/test_helpers.py` (requires Git; no project
dependencies or GitHub access).
