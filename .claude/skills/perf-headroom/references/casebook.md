# Casebook: searchable precedents, not current-run evidence

## Purpose and location

Retain proven diagnostic patterns and rejected hypotheses so another analysis
does not rediscover them. Store model/run-specific records in the campaign or a
separately maintained authorized casebook, not as per-model reference pages in
this skill. Do not create or publish a casebook entry merely because one was read.

## Record format

Each entry has a stable ID, date, layer, symptom tags, mechanism tags, source
revision, hardware/workload constraints and evidence links. Describe:

- observed symptom and the original hypothesis;
- the distinguishing test and counter-evidence;
- confirmed mechanism or unresolved interpretation;
- recommendation considered, eligibility and rejection conditions;
- measured outcome, if one exists, with exact timing boundary and quality status;
- applicability limits, including shape, graph/cache state and topology;
- next revision/workload that requires revalidation.

Label entries `diagnostic_only`, `validated_change`, or `rejected_hypothesis`.
A useful unsuccessful experiment is not a performance gain. No measured outcome
means no invented success story.

## Retrieval procedure

Search first by mechanism/layer (host readback, graph-node coverage, layout copy,
short attention, rank readiness), then refine by relevant shape and environment.
Use a match to choose a distinguishing test, not to transplant its verdict or
numbers. Recheck current source and execution; the same symptom can have a
different cause. Keep contradicting precedents when they prevent overgeneralization.

## Fresh-run and privacy rules

Historical cases may guide collection even in a fresh-only campaign, but their
timings/counters/comparators are historical and cannot populate fresh measurement
fields. Keep lineage to the original run. Never embed remote addresses, passwords,
tokens, private prompts or unrestricted raw artifacts in a reusable entry.

The synthetic [review scenarios](review-scenarios.md) are exercises for skill
behavior, not performance precedents or results for any real model.
