# Competition Rules Snapshot

Snapshot date: 2026-07-25 (China Standard Time)

Primary source: https://ascend.openbmb.cn/competition

This snapshot supports repository work but is not authoritative when the
official site, announcements, toolkit, or starter kit changes. Recheck the
official material before every scored optimization run.

## Competition Purpose

OpenBMB and the Huawei Ascend ecosystem organize the MiniCPM and Ascend
Inference Optimization and Application Innovation Challenge around MiniCPM-o
4.5. The model combines vision, speech, and text for real-time multimodal
interaction. The competition values not only successful deployment but also
fast first response, low end-to-end latency, concurrency, efficient resources,
stable service, reproducibility, and real application experience.

The repository-oriented default is Track 1, High-Performance Inference
Optimization. Track 2 concerns application demos and has different deliverables
and judging criteria.

## Track 1 Objective

Adapt and optimize MiniCPM-o 4.5 in an Ascend NPU environment. The published
evaluation focus is:

- Model inference adaptation.
- TTFT or first-response speed.
- Single-chunk latency and E2E latency.
- Throughput and concurrent sessions.
- Resource utilization and stability.
- Precision/effect loss control.
- Deployment and reproducibility quality.

Required final materials are reproducible code/configuration, benchmark
scripts, a performance report, and reproduction instructions. The page also
encourages a quantized model, deployment image, monitoring records, and an
optimization analysis document.

## Evaluation Order

The performance track follows a gate-first process:

1. Effect and correctness validation against official tasks or samples.
2. Performance evaluation for solutions that pass the gate.
3. Reproduction and engineering review of code, configuration, scripts, and
   documentation.

Solutions clearly below the effect requirement do not enter performance
ranking. Incomplete or non-reproducible solutions may be invalid.

The public page says the performance result considers TTFT, single-chunk
latency, E2E latency, throughput, concurrent sessions, resource utilization,
and stability. It also evaluates NPU utilization, device/host memory, output
correctness, and preservation of multimodal capability.

As of the snapshot date, the formal scoring formula, metric weights, exact
passing thresholds, and detailed benchmark definitions have not been
published. Do not hardcode weights or construct an unofficial final score.

## Environment and Reproduction

Both tracks must pass reproduction in a unified Ascend environment. The page
states that official hardware, image, drivers, Ascend/CANN version, model
access, test scripts, and package specification will be defined by later
announcements and the starter kit.

Keep complete records of dependency versions, launch commands, configuration,
model loading, inference endpoints, and benchmark execution. The public
workflow calls for environment/function checks, multiple warmup rounds,
multiple formal measurement rounds, result aggregation, and reproduction
review.

The frontend currently links an `oiac-toolkit-v1.0.tar.gz` starter artifact,
but availability and version must be verified rather than assumed. It also
links a Huawei HiDevLab compute-resource application guide. Treat every
download as versioned input and record its URL, date, and checksum.

## Rules That Must Remain Dynamic

Recheck these items instead of copying them permanently into code or reports:

- Submission opening/closing dates and daily submission limits.
- Official hardware model and NPU count/topology.
- Container/image, driver, firmware, CANN, torch-npu, vLLM-Ascend, and
  vLLM-Omni versions.
- Model revision and model acquisition method.
- Benchmark dataset, request schema, workload distribution, concurrency, and
  timeout.
- Exact TTFT/chunk/E2E definitions and aggregation statistics.
- Correctness/effect thresholds and hidden evaluation tasks.
- Scoring formula and metric weights.
- Package size, runtime, network, cache, quantization, and dependency rules.

## Prize Context

The public page lists a total pre-tax prize pool of CNY 406,000 across the two
tracks. Track 1 lists one champion at CNY 90,000, two runners-up at CNY 50,000
each, and three third-place awards at CNY 27,000 each. Awards remain subject to
effect validation, unified-environment reproduction, material completeness,
and the organizer's final review.

## Rule-Refresh Checklist

Before a scored run:

1. Open the official competition and toolkit pages.
2. Check announcements and team notifications for newer material.
3. Download the current starter kit and record its checksum.
4. Diff the current rules against this snapshot.
5. Update the optimization matrix and acceptance gates before benchmarking.
6. Escalate ambiguous scoring or legality questions to contact@openbmb.cn.
