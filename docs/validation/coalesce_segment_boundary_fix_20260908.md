# Preserve segment markers during codec coalescing

## Fix

The shared receiver previously merged contiguous codec tensors while keeping
only the first chunk's metadata. Mixed `is_segment_finished` values therefore
lost a later boundary or applied an earlier boundary to later data.

The receiver now compares normalized segment flags with the same scalar
conversion used when committing chunks. A different flag ends the batch and
the next chunk stays in lookahead until the next registration. It is neither
dropped nor consumed twice. Equal flags still coalesce, including the default
PersonaPlex producer's True/True chunks. The existing terminal `finished=True`
barrier, first-frame bypass, batch cap and cancellation fences are unchanged.

This is a shared-contract correction, not evidence that default PersonaPlex
previously lost audio under this condition. No model or KV Append policy was
changed. Missing markers normalize to False, matching existing receiver
semantics; bool and scalar Torch Tensor representations are equivalent.

## CPU validation

New tests live in `tests/distributed/omni_connectors/test_chunk_transfer_adapter.py`:
`test_codec_coalescing_preserves_segment_boundaries` covers 24 variants.

- Against the pre-fix snapshot: 8 failures (the two mixed directions across
  four representation combinations), 16 passes.
- After the fix: 503 passed, 1 deselected in the existing CI-like shared
  connector/SHM/PersonaPlex/close/KV suite, including all new variants.
- Original review-only probes: 5 passed on the fixed snapshot.
- Ruff check/format and `git diff --check` pass.

The mixed-marker tests inspect both receiver registrations, payload identity,
exact tensor contents, frame offset, chunk count, completion marker and read
keys. Checking only whether the first batch contains one frame is insufficient.
Existing L1 sweeps collect these cases; no new CI job is required.

## Reproduction

Prerequisites: Linux and the matching vLLM 0.28.0/Omni environment with pytest
development dependencies. These CPU tests do not need model weights or GPU.
From the repository root:

```bash
# Local targeted regression.
python -m pytest -q tests/distributed/omni_connectors/test_chunk_transfer_adapter.py \
  -k preserves_segment_boundaries

# CI-like connector regression.
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 python -m pytest -q \
  tests/distributed/omni_connectors/test_chunk_transfer_adapter.py \
  -m 'core_model and cpu' --run-level=core_model
```

Raw before/after XML is in workspace `coalesce_segment_fix_20260908/`.
Remote source snapshots are `/root/coalesce-segment-fix.EWjVFl/before` and
`/root/coalesce-segment-fix.EWjVFl/after`, inside existing `vllm-minghui`.

## Receipt and delivery terminology

Receipt retry is bounded by retention/expiry policy, not an indefinite promise
to return every historical receipt. The existing expiry/capacity tests do not
establish a new repeated-append defect and the receipt implementation is not
changed by this patch.

The server's drain watermark represents server-side publication/sends; the
E2E client independently checks received PCM. Neither proves client playback
completion or audio quality. A successful coalescer regression is not an
all-model, performance, or arbitrary-concurrency sign-off.

## H200 default-path check

The fixed source also passes the existing strict PersonaPlex dual-session
client on H200 using the same model/config/input and vLLM 0.28.0 environment.
The run is `coalesce-segment-fix-20260908`; its output is saved alongside the
CPU artifacts. This validates the default same-True producer path separately
from the synthetic mixed-marker regression.

Runtime Python fingerprint (excluding unrelated profiler work):

```text
fd2ec17447e958b85f6f2ad1770ccf761bf6e22dbbe73a4182793c554b7b42a5
```

The run uses the ownership-contained launcher from the previous acceptance,
with `--source-dir /root/coalesce-segment-fix.EWjVFl/after`, physical GPU 1,
and a fresh output label. No thresholds, model policy or dependency versions
were changed to obtain a pass. No GitHub review, commit, push or PR was made.
