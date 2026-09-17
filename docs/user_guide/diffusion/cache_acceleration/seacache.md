# SeaCache control-CFG consensus for Cosmos3

When SeaCache is enabled, Cosmos3 transfer with control guidance preflights all
active guidance branches before evaluating any of them. A step is cached only
if every branch (and participating parallel rank) votes to cache it. Otherwise
all branches compute and reset their accumulated indicator-distance budgets.

Each branch keeps its own indicator and residual tensors. Consensus shares only
the full/skip decision, aligning full-compute anchor steps and extrapolation
intervals without mixing residuals between positive text, negative text, and
no-control inputs. Indicators still include the actual vision streams supplied
to each branch; the no-control branch does not receive control-video latents.

The preflight uses the same vision tensors passed to the transformer. It does not change FP8 quantization,
the SEA filter, threshold, extrapolation order, guidance formula, or scheduler.
Text-only CFG and calls without SeaCache retain their original behavior.

New requests clear all state. Guidance-interval changes restart active histories
together. Unexpected, duplicate, or unconsumed prepared decisions fail explicitly
instead of silently desynchronizing branches. CFG-parallel voting follows the
same round-robin branch ownership as prediction dispatch, including idle-rank
shape forwards, and reduces the decision once before those forwards.

Consensus does not guarantee an identical skip count or latency to independent
caching: another branch can veto a skip. Measure both the actual schedule and
latency for each workload. Regression coverage is in
`tests/diffusion/cache/test_seacache.py` and
`tests/diffusion/models/cosmos3/test_cosmos3_pipeline.py`.
