# Report and recommendation handoff

## Required report flow

Lead with the exact workload/objective and the scoped conclusion. Then show L0,
L1, L2, L3, L4, L5 and L6 in order, even when some are UNKNOWN or justified N/A.
Each section includes verdict/confidence, measurements with denominators, evidence,
already-good components, opportunities, unknowns and the next discriminating test.
Do not label an unobserved component N/A or an incomplete capture complete.

Use phase/component names from the resolved request path. A useful component row:

| Component / population | Wall / summed / exclusive time and scope | Execution state | Work-volume state | Reference and caveats | Recommendation / missing evidence |
| --- | --- | --- | --- | --- | --- |

Do not add the wall, summed and exclusive columns. They are different views.
L4 needs a shape/population coverage table and interpreted NCU rules, not just
percentages. Include rejected rules/recommendations and why they do not apply.

## Recommendation cards

Use the exact fields in [the card contract](../diagnostics.md#recommendation-card-contract).
Keep one primary owner/layer and cross-reference affected layers. A card answers:

1. What observation needs explanation, and what mechanism does the evidence support?
2. Which source symbol, executed phase, shapes/backend and hardware does it affect?
3. What competing explanation was checked, or remains to be checked?
4. What bounded change should a separate optimization task evaluate, and why?
5. What is the expected local/E2E effect, its type and assumptions? Unknown is valid.
6. What memory/quality/compatibility risk could invalidate the idea?
7. What exact experiment, acceptance criterion and rejection condition would test it?

Do not write "attention low, switch backend". Specify whether the problem is
fallback, packing, shape parallelism, non-matmul work, memory behavior or unresolved;
then justify eligible alternatives. Do not invent a percentage when only direction
is known. The analysis task stops at recommendations and permitted diagnostics.

## Machine and narrative reports

Run `analyze.py assess` using the schema in [evidence.md](../evidence.md). Inspect
the JSON and text output, including missing detail checks and stop blockers.
The machine helper validates arithmetic and declared evidence references; it
does not automatically collect profiles or certify annotation truth.

The helper's L0/L6 text does not replace scientific explanation. Add a narrative
report when needed for source/shape evidence, timing diagrams, hypothesis tests,
NCU interpretation and conditional bound derivations. Keep its evidence IDs and
verdict consistent with the machine output. Preserve richer audit artifacts rather
than flattening everything into a free-form "complete" label.

## Acceptance checklist

- Actual execution identity and unprofiled completion boundary are verified.
- The report distinguishes measurement, inference and recommendation.
- Wall time, kernel sums, device union, exposure and E2E impact are not conflated.
- Every required phase/population/rank has a result or named evidence gap.
- References match semantics/treatment; limited matches remain limited.
- Execution efficiency and avoidable work are reported independently.
- Candidate interactions and replacement costs are accounted for or unknown.
- No empirically measured ceiling is mislabeled a universal lower bound.
- No quality-unaccepted change is called a validated gain.
- A stop proposal is scoped, externally reviewable and never an autonomous stop.

## Durable artifacts

Keep commands, source snapshots, shape/count maps, raw profiles, failed collections,
reference runs and reports in the campaign directory, outside this skill. Record
evaluator revision separately from model revision. Do not include raw prompts,
credentials or model-specific pages in the reusable skill. Casebook summaries can
point to access-controlled run artifacts without copying private data.
