# Advisory PR Review Score

Use this score on request after the normal review workflow. Base each rating on
the pinned diff and evidence from that review. Apply the existing blocker and
review verdict rules independently of the score.

| Dimension | Weight | What to assess |
| --- | ---: | --- |
| Scope and intent | 15 | User outcome, boundaries, affected models and entry points, and the match between the diff and the stated goal. |
| Correctness and safety | 25 | Reachable behavior, compatibility, stage and device contracts, lifecycle, failure paths, and security. |
| Validation | 20 | Regression tests for changed paths, the relevance of CI, and the scope of runtime checks actually performed. |
| Claim evidence | 30 | Reproducible performance and quality comparisons, output artifacts, and traceable environment and revision pins. Use [perf-verification](perf-verification.md). |
| Simplicity | 10 | Focused changes, clear ownership, and avoidable duplication, state, copies, or explanatory comments. |

Choose the highest rating supported by the evidence for each applicable dimension.

- **3** — the applicable checks are supported by matching evidence.
- **2** — the main checks are supported, with a bounded gap.
- **1** — some checks are supported, with a material gap or inconsistency.
- **0** — the dimension lacks usable evidence or a material contract violation
  is confirmed.

Use `N/A` when a dimension is outside the change's scope. For example, claim
evidence can be `N/A` for a docs-only change without a performance, quality, or
model support claim. Count each gap in its primary dimension. Identify the
source of each rating, including the difference between contributor evidence
and checks you ran.

The formula is `100 * sum(weight * rating) / (3 * sum(applicable weights))`.
Round to the nearest integer, with halves up. Withhold the total
if the snapshot is stale or a material portion of the diff could not be
reviewed. Report the base and head, total, dimension ratings with evidence,
and the highest-priority remaining action. Use the same authorization and
delivery rules as the normal review.
