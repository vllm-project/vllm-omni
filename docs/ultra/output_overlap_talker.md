# MiniCPM-o 4.5 Talker output overhead

Status: upstream batched codec-sampling pattern manually ported; local static
validation only. Ascend A3 profiling, end-to-end performance, PCM copy-stream
overlap, and complete quality evidence remain pending.

## Baseline audit

The frozen `minicpm-challenge@4105c717` branch does not contain the batched
MiniCPM-o Talker codec-sampling implementation merged into upstream main by
[PR #5792](https://github.com/vllm-project/vllm-omni/pull/5792). Its active
Talker still runs codec projection, repetition penalty, top-p/top-k filtering,
and one `sampled.item()` synchronization independently for every eligible
request row.

The official competition score uses concurrency one, so the upstream
concurrency benefit is not treated as a score claim. At c=1, this port still
removes per-row stop-tensor construction/stacking and reuses the scheduler
`Sampler` object, but any end-to-end effect requires A3 measurement.

## Implemented path

- Collect eligible request rows before codec sampling.
- Run codec projection, bounded batched repetition penalty, top-p/top-k, and
  softmax over one active batch.
- Keep `torch.multinomial` request-local because each request owns its seeded
  generator; request ordering and compaction therefore do not perturb output.
- Replace per-request `.item()` calls with one batched host transfer.
- Build the two-column continue/stop logits in one allocation.
- Reuse the model-owned vLLM `Sampler` instance.

EOS, minimum/maximum-token, incomplete-prefill, native-duplex metadata,
request-local RNG, and codec history semantics remain unchanged.

## Activation and rollback

The batched path defaults on in Python and needs no deploy YAML. Set
`VLLM_OMNI_MINICPMO45_BATCHED_CODEC_SAMPLING=0` and cleanly restart to restore
request-local projection, sampling, and synchronization. Invalid values fail
explicitly.

## Deferred copy-stream work

PCM D2H overlap is not implemented in this PR. It requires an A3 timeline that
shows the copy on the scored critical path plus verified NPU stream/event,
buffer lifetime, abort, and output-builder ownership. Enabling a second stream
without those facts could return incomplete audio or race buffer reuse.

## Promotion gate

- Codec IDs, per-request RNG, EOS/limit behavior, and duplex metadata match.
- A3 timeline shows lower Talker output latency or CPU overhead.
- Chinese Seed-TTS c=1 RTF/TTFP/TTFT gates pass; English compatibility passes.
- Daily-Omni, Video-MME, ASV, and WER complete gates pass.
- c=4/8 request success, tail latency, and HBM remain regression guardrails.
