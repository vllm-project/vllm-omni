# vllm-omni serve

`vllm serve ... --omni` is the main CLI entrypoint for both multi-stage omni models
and diffusion models.

## Stage-Based CLI Quick Start

### Use bundled stage configs

For supported omni models, vLLM-Omni can auto-resolve a bundled stage config when
`--stage-configs-path` is not provided:

```bash
vllm serve Qwen/Qwen2.5-Omni-7B --omni --port 8091
```

### Use a custom stage config

If you need to change device placement, connectors, scheduler settings, or
per-stage defaults, point the CLI at a custom YAML:

```bash
vllm serve Qwen/Qwen2.5-Omni-7B \
  --omni \
  --port 8091 \
  --stage-configs-path /path/to/custom_stage_configs.yaml
```

### Use Ray for distributed execution

For current multi-node or distributed stage deployments, prefer the Ray backend:

```bash
vllm serve Qwen/Qwen2.5-Omni-7B \
  --omni \
  --port 8091 \
  --worker-backend ray \
  --ray-address auto
```

### Legacy `--stage-id` note

`--stage-id` belongs to the legacy stage-per-process flow and should not be used for
new deployments in the current AsyncOmniEngine runtime.

Older setups paired `--stage-id` with `--omni-master-address` and
`--omni-master-port`.

`--headless` is deprecated and not supported in the current AsyncOmniEngine runtime,
so older stage-per-process examples that rely on `--headless` should be treated as
historical reference only.

For the recommended workflow, see
[Stage configs for vLLM-Omni](../configuration/stage_configs.md) and
[Ray-based execution notes](../design/feature/ray_based_execution.md).

## JSON CLI Arguments

--8<-- "docs/cli/json_tip.inc.md"

## Arguments

--8<-- "docs/generated/argparse_omni/omni_serve.inc.md"
