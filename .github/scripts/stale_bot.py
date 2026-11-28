# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Close inactive issues and PRs, ignoring activity from automated bots.

Unlike the shared `actions/stale` action, this script decides activity from
each item's timeline with an explicit event whitelist and actor filter:

- Comments, reviews, references, and reopens only count when they come from a
  human account that is not in ``IGNORE_ACTORS``. Bot accounts (GitHub
  ``__typename: Bot``, e.g. mergify or github-actions) and configured
  User-type bot accounts (e.g. the triage bot) never count, so automated
  triage comments cannot keep an item alive.
- Pushed commits always count as activity.
- Label/assignee churn, mentions, subscriptions, and unknown event types are
  ignored, so future event types fail closed: they never keep an item alive.

Lifecycle: an item with no qualifying activity for ``DAYS_BEFORE_STALE`` days
gets the ``stale`` label and a comment; if it then stays quiet for
``DAYS_BEFORE_CLOSE`` more days it is closed as not planned. Any qualifying
activity after the stale label was applied removes the label instead. Only
the 50 most recent timeline events are examined per item, which comfortably
covers the 60-day window; an item too chatty for that window is active by
definition.

Data is collected with GraphQL (50 items per query; the full sweep is ~40
queries) because two REST facts make per-item collection impractical: the
~1.9k open items exceed the 1k REST requests/hour ``GITHUB_TOKEN`` budget in
Actions, and ``cross-referenced`` events do not bump ``updated_at``, so the
timestamp cannot shortcut the timeline check. Mutations (labels, comments,
close) go through REST. Each run processes one rotating slice of the open
items (``number % RUNS_PER_CYCLE == day_of_year % RUNS_PER_CYCLE``) to bound
the mutation burst, completing full coverage every ``RUNS_PER_CYCLE`` days.

Environment:
- ``GITHUB_REPOSITORY``: repository to process (set by GitHub Actions).
- ``DAYS_BEFORE_STALE`` / ``DAYS_BEFORE_CLOSE``: timing knobs (default 60/14).
- ``IGNORE_ACTORS``: comma-separated logins whose events never count
  (default: the triage bot). Bot-type accounts are always ignored.
- ``RUNS_PER_CYCLE``: slice count; coverage completes every N days (default 4).
- ``RUN_SLICE``: force a slice index (0..N-1) instead of the rotating one.
- ``MAX_ITEMS_PER_RUN``: hard cap on items per run (default 600).
- ``DRY_RUN``: set to ``true`` to log the would-be actions without applying.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

STALE_LABEL = "stale"
EXEMPT_LABELS = {"keep-open"}
TIMELINE_WINDOW = 50

STALE_MESSAGE = (
    "This {kind} has been automatically marked as stale because it has not had any activity from "
    "maintainers or contributors within {stale_days} days. Comments from automated bots do not count as "
    "activity. It will be automatically closed if no further activity occurs within {close_days} days. "
    "Leave a comment if you feel this {kind} should remain open. Thank you!"
)
CLOSE_MESSAGE = (
    "This {kind} has been automatically closed due to inactivity (comments from automated bots do not "
    "count as activity). Please feel free to reopen if it is still relevant. Thank you!"
)

_ISSUE_TIMELINE_NODES = """
  __typename
  ... on IssueComment { createdAt author { login __typename } }
  ... on CrossReferencedEvent { createdAt actor { login __typename } }
  ... on ReopenedEvent { createdAt actor { login __typename } }
  ... on ReferencedEvent { createdAt actor { login } }
  ... on LabeledEvent { createdAt label { name } }
"""
_PR_TIMELINE_NODES = """
  __typename
  ... on IssueComment { createdAt author { login __typename } }
  ... on PullRequestReview { createdAt author { login __typename } }
  ... on CrossReferencedEvent { createdAt actor { login __typename } }
  ... on ReopenedEvent { createdAt actor { login __typename } }
  ... on ReferencedEvent { createdAt actor { login } }
  ... on PullRequestCommit { commit { pushedDate committedDate } }
  ... on LabeledEvent { createdAt label { name } }
"""
_GRAPHQL_QUERY = """query($owner: String!, $name: String!, $cursor: String) {{
  repository(owner: $owner, name: $name) {{
    {connection}(first: 50, states: OPEN, after: $cursor) {{
      pageInfo {{ hasNextPage endCursor }}
      nodes {{
        number
        createdAt
        {extra}
        labels(first: 30) {{ nodes {{ name }} }}
        timelineItems(last: {window}) {{
          nodes {{
            {nodes}
          }}
        }}
      }}
    }}
  }}
}}"""


def _gh_rest(endpoint: str, method: str = "GET", payload: dict[str, Any] | None = None) -> Any:
    """Run one gh api REST call and return the decoded JSON response."""
    cmd = ["gh", "api"]
    if method != "GET":
        cmd.extend(["-X", method])
    if payload is not None:
        cmd.extend(["--input", "-"])
    cmd.append(endpoint)
    stdin = json.dumps(payload) if payload is not None else None
    result = subprocess.run(
        cmd, input=stdin, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh api {method} {endpoint} failed: {result.stderr.strip()[:300]}")
    if not result.stdout.strip():
        return None
    return json.loads(result.stdout)


def _gh_graphql(query: str, variables: dict[str, Any]) -> dict[str, Any]:
    """Run one GraphQL query through gh api and return the data object."""
    result = subprocess.run(
        ["gh", "api", "graphql", "--input", "-"],
        input=json.dumps({"query": query, "variables": variables}),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"graphql failed: {result.stderr.strip()[:300]}")
    payload = json.loads(result.stdout)
    if payload.get("errors"):
        raise RuntimeError(f"graphql errors: {json.dumps(payload['errors'])[:300]}")
    return payload["data"]


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _is_human(actor: dict[str, Any] | None, ignore_actors: frozenset[str]) -> bool:
    """True when the event actor represents a human, not an automation."""
    if not actor:
        return False
    if actor.get("__typename") != "User":
        return False
    return actor.get("login", "") not in ignore_actors


def _node_stamp(node: dict[str, Any], ignore_actors: frozenset[str]) -> datetime | None:
    """Timestamp of a timeline node when it counts as real activity."""
    kind = node["__typename"]
    if kind == "PullRequestCommit":
        commit = node.get("commit") or {}
        stamp = commit.get("pushedDate") or commit.get("committedDate")
        return _parse_ts(stamp) if stamp else None
    if kind in ("IssueComment", "PullRequestReview"):
        actor, stamp = node.get("author"), node.get("createdAt")
    elif kind in ("CrossReferencedEvent", "ReopenedEvent", "ReferencedEvent"):
        actor, stamp = node.get("actor"), node.get("createdAt")
    else:
        return None
    if stamp and _is_human(actor, ignore_actors):
        return _parse_ts(stamp)
    return None


def _collect_items(owner: str, name: str, connection: str, nodes: str, extra: str = "") -> list[dict[str, Any]]:
    """Page through one GraphQL connection of open issues or PRs."""
    query = _GRAPHQL_QUERY.format(connection=connection, nodes=nodes, extra=extra, window=TIMELINE_WINDOW)
    items: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        data = _gh_graphql(query, {"owner": owner, "name": name, "cursor": cursor})
        repo = data["repository"][connection]
        items.extend(repo["nodes"])
        if not repo["pageInfo"]["hasNextPage"]:
            return items
        cursor = repo["pageInfo"]["endCursor"]


def _stale_marked_at(timeline: list[dict[str, Any]]) -> datetime | None:
    """When the item was most recently labeled stale, from its timeline."""
    marked: datetime | None = None
    for node in timeline:
        if (
            node.get("__typename") == "LabeledEvent"
            and (node.get("label") or {}).get("name") == STALE_LABEL
            and node.get("createdAt")
        ):
            ts = _parse_ts(node["createdAt"])
            marked = ts if marked is None or ts > marked else marked
    return marked


def _mutate(dry_run: bool, description: str, endpoint: str, method: str, payload: dict[str, Any]) -> None:
    print(f"  -> {'WOULD ' if dry_run else ''}{description}")
    if not dry_run:
        _gh_rest(endpoint, method=method, payload=payload)


def process_item(
    item: dict[str, Any],
    now: datetime,
    days_before_stale: int,
    days_before_close: int,
    ignore_actors: frozenset[str],
    dry_run: bool,
    stats: dict[str, int],
) -> None:
    number = item["number"]
    is_pr = item.get("__typename") == "PullRequest"
    kind = "pull request" if is_pr else "issue"
    is_draft = bool(item.get("isDraft"))
    labels = {label.get("name") for label in (item.get("labels") or {}).get("nodes", [])}
    prefix = f"{kind} #{number}"

    if labels & EXEMPT_LABELS:
        print(f"{prefix}: skipped (exempt label)")
        stats["skipped"] += 1
        return
    if is_draft:
        print(f"{prefix}: skipped (draft)")
        stats["skipped"] += 1
        return

    timeline = (item.get("timelineItems") or {}).get("nodes", [])
    activity = [stamp for node in timeline if (stamp := _node_stamp(node, ignore_actors)) is not None]
    last_activity = max([_parse_ts(item["createdAt"]), *activity])

    endpoint_base = f"repos/{os.environ['GITHUB_REPOSITORY']}/issues/{number}"
    has_stale_label = STALE_LABEL in labels
    if has_stale_label:
        marked_at = _stale_marked_at(timeline)
        if marked_at is None:
            print(
                f"{prefix}: skipped (stale label found but no labeling event in the recent "
                f"{TIMELINE_WINDOW} timeline entries; leaving untouched)"
            )
            stats["skipped"] += 1
            return
        if any(ts > marked_at for ts in activity):
            _mutate(
                dry_run,
                "remove stale label (activity after marking)",
                f"{endpoint_base}/labels/{STALE_LABEL}",
                "DELETE",
                {},
            )
            stats["unstaled"] += 1
        elif now >= marked_at + timedelta(days=days_before_close):
            _mutate(
                dry_run,
                f"close (stale since {marked_at.date()}, no activity since)",
                endpoint_base,
                "PATCH",
                {"state": "closed", "state_reason": "not_planned"},
            )
            _mutate(
                dry_run,
                "post close comment",
                f"{endpoint_base}/comments",
                "POST",
                {"body": CLOSE_MESSAGE.format(kind=kind)},
            )
            stats["closed"] += 1
        else:
            print(f"{prefix}: stale, inside the {days_before_close}-day close window")
            stats["waiting"] += 1
        return

    if now - last_activity > timedelta(days=days_before_stale):
        print(f"{prefix}: no qualifying activity since {last_activity.date()}")
        _mutate(dry_run, "add stale label", f"{endpoint_base}/labels", "POST", {"labels": [STALE_LABEL]})
        stale_body = STALE_MESSAGE.format(kind=kind, stale_days=days_before_stale, close_days=days_before_close)
        _mutate(dry_run, "post stale comment", f"{endpoint_base}/comments", "POST", {"body": stale_body})
        stats["marked"] += 1
    else:
        print(f"{prefix}: active (last qualifying activity {last_activity.date()})")
        stats["active"] += 1


def _select_slice(items: list[dict[str, Any]], now: datetime) -> list[dict[str, Any]]:
    """Pick the rotating slice for this run and apply the per-run cap."""
    runs_per_cycle = int(os.environ.get("RUNS_PER_CYCLE", "4"))
    run_slice = os.environ.get("RUN_SLICE", "auto")
    index = now.timetuple().tm_yday % runs_per_cycle if run_slice == "auto" else int(run_slice) % runs_per_cycle
    selected = sorted(
        (item for item in items if item["number"] % runs_per_cycle == index), key=lambda item: item["number"]
    )
    print(f"slice {index}/{runs_per_cycle}: {len(selected)} of {len(items)} open items")
    max_items = int(os.environ.get("MAX_ITEMS_PER_RUN", "600"))
    if len(selected) > max_items:
        offset = now.timetuple().tm_yday % len(selected)
        selected = (selected[offset:] + selected[:offset])[:max_items]
        print(f"capped to {max_items} items for this run (rotated for fairness)")
    return selected


def main() -> int:
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if not repo:
        print("GITHUB_REPOSITORY is not set", file=sys.stderr)
        return 1
    owner, _, name = repo.partition("/")
    dry_run = os.environ.get("DRY_RUN", "false").lower() == "true"
    days_before_stale = int(os.environ.get("DAYS_BEFORE_STALE", "60"))
    days_before_close = int(os.environ.get("DAYS_BEFORE_CLOSE", "14"))
    ignore_actors = frozenset(
        actor.strip() for actor in os.environ.get("IGNORE_ACTORS", "vllm-omni-review-bot").split(",") if actor.strip()
    )
    print(
        f"stale bot on {repo}: stale_after={days_before_stale}d close_after={days_before_close}d "
        f"ignore_actors={sorted(ignore_actors)} dry_run={dry_run}"
    )

    items = _collect_items(owner, name, "issues", _ISSUE_TIMELINE_NODES)
    items += _collect_items(owner, name, "pullRequests", _PR_TIMELINE_NODES, extra="isDraft")
    stats = {"marked": 0, "unstaled": 0, "closed": 0, "waiting": 0, "active": 0, "skipped": 0, "errors": 0}
    now = datetime.now(timezone.utc)
    for item in _select_slice(items, now):
        try:
            process_item(item, now, days_before_stale, days_before_close, ignore_actors, dry_run, stats)
        except Exception as error:  # noqa: BLE001 - one bad item must not stop the sweep
            print(f"#{item.get('number')}: error, leaving untouched: {error}")
            stats["errors"] += 1
    print(f"done: {json.dumps(stats, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
