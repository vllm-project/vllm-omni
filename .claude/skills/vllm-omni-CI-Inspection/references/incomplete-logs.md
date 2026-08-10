# Incomplete / Truncated Log Detection

Must be flagged separately during nightly batch triage; **never** count as passed.

## Strong Signals (any one suffices)

1. File end has **no** `short test summary info` and **no** pytest closing line of the form `= N passed|M failed|K error`.
2. Last screen still shows `--- Running test: <name>` or long inference/download output.
3. Ends with a `Trying to resume download...` loop and no subsequent summary.
4. Ends immediately after `resource_tracker: There appear to be N leaked` with no summary.

## Weak Signals (judge alongside peer jobs)

| Clue | Notes |
|------|-------|
| Abnormally short line count | Same-batch `full_moon_*_Function_Test` logs are usually thousands of lines; only hundreds is suspicious |
| Only partial PASSED/FAILED lines | `collected N items` but far fewer than N results |
| Common on Buildkite | agent lost, timeout, killed (search log for `🚨`/`canceled`/`agent lost` if present) |

## Report Format

```markdown
| Omni_Function_Test_with_H100 | ⚠️ incomplete | — | 517 lines, no pytest summary, ends mid-test |
```

Recommended action: **fetch complete Buildkite artifact / re-run that step**; do not include in passed-rate summary.
