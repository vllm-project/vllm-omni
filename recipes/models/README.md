# vLLM-Omni recipe YAMLs

This directory holds the structured recipes that
[recipes.vllm.ai](https://recipes.vllm.ai) renders for vLLM-Omni models, one
YAML per model at `<hf_org>/<hf_repo>.yaml`. The path mirrors the Hugging Face
ID, so `Qwen/Qwen-Image.yaml` renders at
[recipes.vllm.ai/Qwen/Qwen-Image](https://recipes.vllm.ai/Qwen/Qwen-Image).
Every recipe here carries the `omni` task.

The Markdown recipes one level up (`recipes/<vendor>/*.md`, indexed in
[`recipes/README.md`](../README.md)) are unchanged; this directory adds the
YAML that drives the site's command builder and JSON API.

The site is built by
[vllm-project/recipes](https://github.com/vllm-project/recipes). That
repository owns the schema and the shared vocabulary (hardware profiles,
serving strategies) and pins this directory to a commit.

## Editing a recipe

Start a new recipe from
[`templates/recipe.yaml`](https://github.com/vllm-project/recipes/blob/main/templates/recipe.yaml).
Fields are defined in
[SCHEMA.md](https://github.com/vllm-project/recipes/blob/main/SCHEMA.md); the
`omni` block and the task catalog are described there. Every build checks each
recipe against [`recipe.schema.json`](https://recipes.vllm.ai/recipe.schema.json),
and an unknown or misspelled key is an error. For completion and inline errors
in YAML-aware editors, start the file with
`# yaml-language-server: $schema=https://recipes.vllm.ai/recipe.schema.json`.

To validate a change exactly as this repository's `Recipes` PR check does, run
from a checkout of `vllm-project/recipes`:

```bash
node scripts/validate-source.mjs --source vllm-omni --dir ~/vllm-omni/recipes
```

To preview the site with your changes, add a gitignored `sources.local.yaml`
to that checkout with `overrides: { vllm-omni: ~/vllm-omni/recipes }` and run
`pnpm dev`. This works once vllm-project/recipes reads this directory as its
`vllm-omni` source.
