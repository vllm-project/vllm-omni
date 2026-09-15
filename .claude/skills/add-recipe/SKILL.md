---
name: add-recipe
description: Add or update an in-repository vLLM-Omni model recipe with verified task, input, output, hardware, command, feature, and validation contracts. Use when creating files under recipes/, restructuring a recipe after review, documenting a newly supported model, or synchronizing recipe claims with support tables and shared feature guides.
---

# Add a vLLM-Omni Recipe

## Source the contract

Read [`recipes/TEMPLATE.md`](../../../recipes/TEMPLATE.md) and
[`recipes/README.md`](../../../recipes/README.md) before editing. Inspect the
canonical model card/repository, implementation, shared examples, tests, and
available local qualification evidence.

Do not infer support from a model name, sibling recipe, registry entry, or
topology validator. Mark unexecuted configurations as configuration-only.

## Route by model family

Read only the contributor and user documentation relevant to the model:

| Family | Contributor guide | User-facing documentation to synchronize |
|---|---|---|
| Diffusion | [`adding a diffusion model`](../add-diffusion-model/SKILL.md) | `docs/user_guide/diffusion_features.md`, applicable diffusion feature guides, and shared image/video/audio examples |
| TTS | [`adding a TTS model`](../add-tts-model/SKILL.md) | `examples/offline_inference/text_to_speech/README.md`, `examples/online_serving/text_to_speech/README.md`, and `docs/serving/speech_api.md` |
| Omni | [`adding an omni model`](../../../docs/contributing/model/adding_omni_model.md) | model-family offline/online example docs, `docs/serving/chat_completions_api.md`, and `docs/user_guide/feature_compatibility.md` |

All families must update `docs/models/supported_models.md` and the matching row
in `recipes/README.md`. Do not add a modality-specific support document when
the repository has no such table; update the closest shared user contract.

## Structure the recipe

Keep the recipe task-oriented and use one hardware-specific file named
`recipes/<vendor>/<model>-<hardware>.md`. A recipe may contain multiple
deployment topologies on that hardware, but each additional accelerator model
or platform requires its own suffixed file.

1. **Summary:** vendor, exact model ID, runtime, modes, named hardware,
   recommended deployment, and maintainer.
2. **Supported model contract:** put task, input, output, and provided-profile
   tables before setup or commands.
3. **References:** link the canonical model source, shared examples, supported
   model table, and feature matrix.
4. **Checkpoint/setup:** pin revisions when possible and state required assets.
5. **Hardware:** document the single accelerator named by the file suffix,
   including every validated device-count/topology profile on it.
6. **Commands:** reuse shared `examples/` entrypoints unless the model contract
   truly requires a dedicated script. Include offline and online paths that
   were validated.
7. **Supported features:** use a compact model-specific topology/status table
   with links to shared feature guides.
8. **Verification:** provide a quick command and exact expected output
   contract.
9. **Qualification evidence:** report correctness, memory, and timing evidence
   with its measurement scope and caveats.

## Separate hardware from software

For every locally validated hardware profile, record:

- accelerator vendor/model and per-device memory;
- number of devices;
- interconnect (`NVLink`, `PCIe`, or the platform equivalent);
- host memory when CPU staging/offload is material;
- whether the profile is runtime-qualified or configuration-only.

Record OS, Python, driver/runtime, framework versions, and the vLLM-Omni
revision in a separate software-environment table. Keep precision, worker
count, and DP/TP/SP/PP sizes with the deployment profile or exact command.

Never generalize a result from one accelerator family to another or combine
NVIDIA, AMD, NPU, or distinct accelerator models in one recipe. Create a
separate hardware-suffixed recipe and `recipes/README.md` row. Distinguish
upstream requirements from hardware exercised by the PR.

## Keep shared documentation concise

Link the applicable diffusion, TTS, omni, serving, or design guide for generic
feature semantics and launch instructions. The recipe should contain only the
model-specific status, valid topology, required flag difference, and evidence
boundary.

Put unsupported combinations in the same feature table. Avoid repeating the
same model-specific prose below the global feature matrix; the recipe is the
detailed source of truth.

## Report evidence precisely

For every performance or memory value, state:

- checkpoint/revision and workload;
- accelerator and device count;
- precision and topology;
- cold/warm scope, step count, and concurrency;
- peak HBM and host-memory metric when measured;
- output/quality guard and sample count;
- monitoring overhead or other known variance.

Call a single bounded run qualification evidence, not a benchmark. Do not add
a benchmark script unless the contribution explicitly requires one.

## Synchronize repository documentation

Update all applicable locations:

- the row in `recipes/README.md`;
- `docs/models/supported_models.md`;
- the family-specific documentation selected in **Route by model family**;
- shared example documentation when commands or flags change.

Keep global support tables compact. Link to the recipe instead of duplicating
its deployment explanation.

## Validate

Run, at minimum:

```bash
pre-commit run --files <changed recipe/docs/skill files>
mkdocs build --strict
git diff --check
```

Remove generated documentation artifacts after validation. Confirm every
recipe link resolves and every claimed profile has matching code or evidence.
