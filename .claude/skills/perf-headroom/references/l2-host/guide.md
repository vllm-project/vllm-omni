# L2: host, synchronization and submission

## Inputs and detection

Require a coverage-audited timeline, L1 request/phase map and a normal baseline.
Use [Nsys collection](../nsys-collection.md), then
[timing and overlap](../l1-pipeline/timing-and-overlap.md). A long host range or
low GPU activity alone does not establish host starvation.

For material gaps, repeatedly sparse launch regions and request head/tail, identify
the next useful task and its prerequisites. Record previous device completion,
producer readiness, CPU launch submission and next device start when observable.

## Decision tree

1. **Are events complete?** Missing worker/rank/graph nodes -> repair collection.
2. **Was next work submitted?** If yes, inspect stream/event dependencies, queueing
   behind device work, context/resource competition, or transfer completion.
3. **If not submitted, was its input ready?** An unready producer routes to its
   owner/L5. A ready task with delayed submission routes to host root-cause analysis.
4. **What was the responsible CPU thread doing?** Running Python/native code,
   blocked on synchronization/IPC/lock, runnable but not scheduled, or idle awaiting
   admission are different causes. Use [host root causes](host-root-causes.md).
5. **Is a host-device round trip involved?** Follow [synchronization](synchronization.md).
6. **Is repetitive submission exposed?** Check [CUDA Graphs](cuda-graphs.md), existing
   capture and structural fusion (L3), not an unconditional graph recommendation.

## Interpretation and recommendations

Use concrete dependencies and source stacks to distinguish cause from overlap.
A host-preparation interval intersecting device idle is a useful exposure signal,
not proof that deleting that CPU interval would save the whole intersection.
Classify each material region as launch/CPU preparation, synchronization,
producer/transfer readiness, admission/load, resource competition or unresolved.

Report both exposed duration and its denominator: preparation duration, total
request window, or total observed idle time. These answer different questions.
Do not adopt one percentage threshold across models or windows.

If complete traces show little exposed host delay, say L2 leverage is low for
this regime. Continue to L3/L4; continuous activity can still be inefficient or
redundant. Faster kernels can expose previously hidden preparation, so retain
host slack in L6 sensitivity assumptions.

## Output

Deliver a gap/exposure table with interval, owner, prerequisites, stack/source,
classification, counter-explanation and next discriminating collection. Candidate
cards propose specific investigations or changes; they do not implement them.
Unexplained gaps remain unknown, not automatically removable framework overhead.
