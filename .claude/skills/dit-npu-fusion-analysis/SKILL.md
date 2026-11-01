---
name: dit-npu-fusion-analysis
description: Route Ascend NPU diffusion model and operator optimization to MindIE-SD skills. Use for NPU DiT fusion, profiling, kernel/pattern work and related parallelism, VAE or host bottlenecks. Supply Omni context and load the applicable external skill. CUDA/ROCm tuning, pure AR optimization and runner-version upgrades use their own workflows.
---

# Ascend Diffusion Optimization Router

MindIE-SD owns optimization methods, fusion analysis and NPU acceptance guidance.
This entry keeps its existing name for discovery. Do not maintain a second
operator catalogue or validation protocol in Omni. Routing means reading the
selected instructions in this task, not messaging a task or starting an agent.

## Locate and load

1. Use the task's MindIE-SD checkout, or an existing sibling `MindIE-SD` checkout.
   Record its revision and relevant dirty changes; do not switch or reset its
   branch to route a task.
2. Read its `AGENTS.md` and `.agents/README.md`, then the selected `SKILL.md`
   and required references. Resolve paths against that checkout. Naming a
   skill does not mean its instructions have been loaded.
3. Without a checkout, use the official [dev skill index](https://gitcode.com/Ascend/MindIE-SD/blob/dev/.agents/README.md)
   and [dev tree](https://gitcode.com/Ascend/MindIE-SD/tree/dev/.agents).
   If GitCode is unavailable, use the official [GitHub dev skill index](https://github.com/Ascend/MindIE-SD/blob/dev/.agents/README.md)
   and [GitHub dev tree](https://github.com/Ascend/MindIE-SD/tree/dev/.agents).
   Retrieve only the needed instructions and record their source and revision or access date.
   Follow the actual index if names changed; do not rely on unpublished additions.
4. If required guidance is unavailable, name the missing resource and continue
   useful Omni source/configuration inspection. Do not claim the SD workflow ran
   or recreate it here. Reading guidance requires neither installation nor an NPU.

## Select the applicable entry

Confirm these paths in the selected SD index; they are relative to its root.
A focused operator task need not repeat model onboarding.

| Task | MindIE-SD skill under `.agents/skills/` |
| --- | --- |
| Model optimization with an unknown bottleneck | `model-auto-optimization/SKILL.md` |
| Known bottleneck needing domain selection | `performance-optimization/SKILL.md` |
| Profile collection / diagnosis | `profiling-collect/SKILL.md` / `profiling-analyze/SKILL.md` |
| Fusion scope, benefit or reusable implementation feasibility | `fusion-scope-analyze/SKILL.md` |
| Kernel implementation / tuning | `operator-dev/SKILL.md` |
| Pattern / graph implementation | `pattern-dev/SKILL.md` / `aclgraph-dev/SKILL.md` |
| Framework enablement / missing adapters | `framework-integration/SKILL.md` |
| DiT compute / communication | `dit-perf-opt/SKILL.md` / `dit-parallel-opt/SKILL.md` |
| VAE / host optimization | `vae-opt/SKILL.md` / `host-opt/SKILL.md` |
| MindIE-SD repository development | `dev-workflow/SKILL.md`, then its implementation route |

For numerical/performance acceptance, read `accuracy-gate/SKILL.md` and
`perf-gate/SKILL.md` from that skill root. Those skills own the criteria;
historical evidence applies only within its recorded scope.

## Supply context and preserve ownership

Supply the goal and authorized analysis/implementation scope, Omni revision and
dirty diff, model/checkpoint and active pipeline/symbols, hardware/provider
versions, launch/request commands, workload/dtype/parallel/cache/graph settings,
profiles/baselines, precision constraints and output directory. Keep missing
fields unknown; the selected skill determines how to proceed.

Keep Omni, MindIE-SD and installed provider sources distinct. Model files resolve
against Omni; vendor implementation resolves against its actual provider.
Changes to Omni stay there; MindIE-SD changes follow its development workflow.
External instructions do not expand user authorization or repository scope.

For Omni changes, read [vllm-omni-test](../vllm-omni-test/SKILL.md) for
repository tests/CI and [diffusion-accuracy-test](../diffusion-accuracy-test/SKILL.md)
when adding an Omni accuracy case. Production qualification stays with
[production-add-diffusion-model](../production-add-diffusion-model/SKILL.md).
These checks complement the SD acceptance criteria.

Return the selected skill paths/revision, routing reason, changes, actual checks
and artifacts using the SD workflow's existing output format. Distinguish task
completion from optimization acceptance; identify unread/unexecuted dependencies.
Do not require a parallel Omni evidence schema.
