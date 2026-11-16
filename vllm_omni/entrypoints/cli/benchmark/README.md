# Benchmark CLI

This package owns Omni benchmark subcommands exposed through the CLI.

## What Belongs Here

- CLI parser extensions for benchmark commands.
- Benchmark command dispatch.
- Benchmark-only argument groups.

## What Does Not Belong Here

- Runtime server route helpers.
- OpenAI endpoint implementation details.
- Benchmark execution internals that belong under `vllm_omni.benchmarks`.

This package is not part of the OpenAI endpoint helper migration. It may need
import updates only if CLI/bootstrap ownership changes during #5227.
