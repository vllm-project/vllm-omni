# Historical calibration evidence

The original `*_unrecorded.json` records omitted optimization switches or decoder
versions. They are retained byte for byte for historical method and rejection
tests. Do not add current provenance to old data.

`cuda_nightly_quality_population_within_cohort.json` is the recorded-zero
population before pooling cross-cohort reference comparisons. It retains its
original six within-cohort pixel summaries and six native reference scores per
case. The later version 2 record (now archived as
`cuda_nightly_quality_population_pre_v032.json`) uses every unordered pair of
the same six reference videos. This archive remains unchanged for boundary and derivation
checks; archiving it does not change original hardware pass/fail results.

Archived records are not loaded by the CI runner.
The active nightly records use vLLM 0.32.0 reference cohorts from builds 17633
and 17640. Their frozen training records require a subsequent independent
campaign; historical method controls do not qualify this new reference epoch.

`cuda_nightly_reference_pre_v032.json` and
`cuda_nightly_quality_population_pre_v032.json` preserve the preceding
recorded-zero performance and version 2 pixel/native records byte for byte.
Their SHA256 values are respectively
`0f3286d8d019b4d5c605b653d11c7b37fc147081fa4d41fac9308f00f804cd58` and
`c3289857b3d57db52bee8b08bed1df0b5c1526e21fb74654c81420ea5c893175`.
They retain their original older reference pins, measurements, failed gates and
method controls. Historical tests still exercise those original derivations;
the runner does not load archived records. See the suite's
[current production protocol](../../README.md#current-production-reference-cuda-nightly-calibration-frozen).

`profiles/` preserves all seven earlier profiles byte for byte from
`7d1dfdafbec699b8fb327bd5375a23281dfba2df`. Historical evidence hashes bind to
these snapshots after the active reference pins advance; the active profiles
differ only in `reference_revision`. Do not rewrite archived profile hashes to
make old observations appear to have used the new production reference.

The intermediate `1cc64011cee7ed4860f8a3849511130df747bc50` pin was superseded
before collecting any fresh hardware data: the actual GitHub merge included
newer accepted main changes. The final production snapshot includes upstream
`990d4c1061124bb87ba3b57a8c1095c2799b41aa`; the reference-only protocol and
candidate regression allowances are unchanged.
