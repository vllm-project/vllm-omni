# Qwen3-Omni validation on the published MRV2 candidate

Source: `60125b15` (Python ASTs match measured TTS source `07da1fed`).
Qwen3-Omni-30B-A3B-Instruct BF16, 2 H200 GPUs: thinker on GPU2,
talker and Code2Wav on GPU3, TP1 per stage. Default qwen3_omni_moe deploy
with thinker/talker memory fractions 0.75/0.5. V1 and V2 use the same source,
weights, placement and configuration except runner selection.

The successful V2 follow-up and failed V1 follow-up both set Code2Wav
`enforce_eager=true`; thinker and talker graph configurations remain unchanged.
The original default graph attempt is retained separately.

Workload: 8 short English sentence prompts repeated, 32 warmup requests at c8,
6 mixed text/image/audio smoke requests at c2, then 64 requests each at c8 and
c16. This is a small functional/performance probe, not the SeedTTS workload or
a statistically established capacity comparison. Text temperature=0, text limit
96; talker sampling remains the deployment default. No retries or forced EOS.

| Metric | V1 | V2 | Relative change |
|---|---:|---:|---:|
| c8 audio throughput (audio-s/s) | Invalid baseline | 46.14 | N/A |
| c8 TTFA mean (ms) | Invalid baseline | 190.8 | N/A |
| c8 TTFA P95 (ms) | Invalid baseline | 250.4 | N/A |
| c8 E2E mean (s) | Invalid baseline | 0.649 | N/A |
| c16 audio throughput (audio-s/s) | Invalid baseline | 65.69 | N/A |
| c16 TTFA mean (ms) | Invalid baseline | 290.1 | N/A |
| c16 TTFA P95 (ms) | Invalid baseline | 508.1 | N/A |
| c16 E2E mean (s) | Invalid baseline | 0.857 | N/A |

V1 default graph failed during warmup: Code2Wav returned one model_outputs
entry for a batch of two, triggering the runner's output-axis validation.
With decoder eager, the server returned text and stream termination but only
zero-frame WAV output for all 32 warmup requests. The benchmark initially
rejected the empty WAV immediately; a corrected reader ignored zero-frame
chunks and waited through stream termination, confirming no valid PCM.
V1 therefore has no valid audio-throughput or TTFA baseline. Do not divide by
zero or claim an infinite speedup. The V1 failure root cause is not fixed here.

V2 with corrected PCM reader: 32 warmup + 6 mixed smoke + 128 measured
requests succeeded, with no reported length finish reason. Validation checks
finite decoded audio, positive audio frames, text and terminal event; it does
not prove semantic audio correctness, full codec trajectory equivalence, or
perceptual quality. No WER/MOS evaluation was performed. Playback gaps were
not measured for this Omni probe.

The earlier V2 default-graph run also completed the same number of requests:
c8 46.75 and c16 50.04 audio-s/s. It is not compared against eager V1 as though
configurations matched, nor used to infer a graph performance regression from
one short run.

Ownership samples detected no foreign process on either GPU during the listed
phases. Each short phase has only a few ownership samples; this is not an
exclusive reservation or CPU contention guarantee. All owned services were
cleaned up. Summary/audit data are in omni_comparison.json; full requests and
logs remain in the local experiment workspace.
