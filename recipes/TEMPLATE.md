# Recipe Title — Hardware

> Example: Qwen3-Omni for speech chat — H100

Name the file `recipes/<vendor>/<model>-<hardware>.md`. Keep exactly one
hardware profile in each recipe; add a separate suffixed recipe when another
accelerator is qualified.

## Summary

- Vendor:
- Model:
- Task:
- Mode:
- Hardware:
- Maintainer:

## When to use this recipe

Briefly describe the concrete scenario this recipe covers.

## Supported model contract

Put user-visible constraints before commands. Use compact tables for:

- supported tasks and shared entrypoints;
- input modalities, counts, duration/geometry limits, and required combinations;
- output duration, geometry, frame/audio rates, and variants;
- the deployment profiles provided on this recipe's named hardware.

Source model specifications from the canonical model card or repository. Do
not copy values from a sibling model or infer support from configuration alone.

## References

- Upstream or canonical docs:
- Related example under `examples/`:
- Related issue or discussion:

## Hardware

- Accelerator model and per-device memory:
- Number of devices:
- Device interconnect (NVLink, PCIe, or other):
- Host memory, when CPU offload or staging is relevant:
- Qualification scope:

Do not add a second accelerator to this file. Create another
`<model>-<hardware>.md` recipe instead. Mention unqualified upstream hardware
only as a limitation, not as a provided profile.

## Software environment

- OS:
- Python:
- Driver / runtime:
- vLLM version:
- vLLM-Omni version or commit:

Keep worker counts, precision, and TP/SP/DP/PP sizes with the command or
deployment profile; they are not software-environment properties.

## Command

Serve commands use `vllm serve <model> --omni`. The `--omni` flag is what
selects the Omni pipeline, so it is required and cannot be dropped.

```bash
# Add the exact command(s) here
```

## Verification

```bash
# Add a quick validation command or expected output here
```

## Notes

- Memory usage:
- Key flags:
- Known limitations:

## Supported features

Use a compact table containing the model-specific status/topology and a link to
the corresponding shared feature guide. Do not repeat the general feature
documentation in the recipe. Record unsupported combinations in the same table
instead of a disconnected restrictions section.
