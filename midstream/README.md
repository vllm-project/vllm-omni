# Midstream

This directory contains midstream-only content for `nm-vllm-omni-ent`. Nothing here exists in the upstream `vllm-project/vllm-omni` repository, so it is safe from upstream rebases and merges.

## Build Pipeline

Omni builds run in [nm-cicd](https://github.com/neuralmagic/nm-cicd) via `omni-pipeline.yml`:

```
accept-sync (wheel + image + partition tests) → OCP model validation
```

The base vLLM wheel is reused via `midstream/vllm-wheels.yml` (`vllm_run_id`).

### Triggering Builds

Use the **Omni release** workflow from the [Actions tab](../../actions/workflows/omni-release.yml):

| Trigger | What happens |
|---------|--------------|
| Push tag `omni-*` | Full pipeline in nm-cicd: accept-sync + OCP validation (`wf_category=RELEASE`) |
| Manual **Omni release** | Dispatch to nm-cicd `omni-pipeline.yml` — choose ref, category, and whether to run OCP validation |

The Ent workflow dispatches once (GitHub App token). The full build+test cycle runs as a **single nm-cicd workflow run** — no cross-repo polling.

**Legacy:** [Midstream Build](../../actions/workflows/midstream-build.yml) (`midstream-build.yml`) is deprecated; use **Omni release** instead.

| Build Step (legacy) | Replacement |
|---------------------|-------------|
| **full-chain** | Omni release (tag or manual with `run_ocp_validation=true`) |
| **omni-wheel** | nm-cicd `accept-sync.yml` with `build_image=false` |
| **docker-image** | nm-cicd `accept-sync.yml` with existing `vllm_run_id` + `omni_run_id` |
| **vllm-wheel** | nm-cicd `build-whl.yml` for `neuralmagic/nm-vllm-ent` |

### Tag-Based Triggers (Full Chain)

Pushing a tag matching `omni-*` runs the **Omni release** workflow, which triggers nm-cicd `omni-pipeline.yml` — accept-sync (wheel + image + partitions) and OCP model validation in one run. The workflow reads `midstream/vllm-version` and looks up the vLLM wheel run ID from `midstream/vllm-wheels.yml`.

```bash
# Tag and push — that's it, full build kicks off
git tag omni-v0.20.0-rc1
git push origin omni-v0.20.0-rc1
```

The tag name is freeform — use whatever makes sense: `omni-v0.20.0`, `omni-doug-feature-foo`, `omni-ricky-demo-2026-05-15`, etc. The vLLM wheel version is determined by the code at the tagged commit, not the tag name.

## Secret: CICD_APP_ID / CICD_APP_PRIVATE_KEY

The **Omni release** workflow uses the org GitHub App (`CICD_APP_ID`, `CICD_APP_PRIVATE_KEY`) to dispatch nm-cicd workflows — the same pattern as `nm-vllm-ent` release.

**Legacy:** `midstream-build.yml` used `CICD_OMNI_PAT`; that path is deprecated.

## vLLM Version Mapping

Two files control which vLLM wheel gets used:

### `midstream/vllm-version`

A single line declaring the vLLM version this omni code is built against:

```
v0.20.0
```

Update this when rebasing to a new upstream version.

### `midstream/vllm-wheels.yml`

Maps vLLM versions to known-good wheel build run IDs:

```yaml
v0.20.0:
  run_id: "25021945246"
  branch: "main"
  note: "reused from deepseek effort, built 2026-05-10"
```

**When to update:**
- **New vLLM version:** after rebasing upstream, update `vllm-version` and add a new entry to `vllm-wheels.yml` once you've built a wheel for it
- **New wheel for existing version:** update the `run_id` for that version entry

**How the workflow uses it:** when `vllm_run_id` is not provided as input (including all tag-push triggers), the workflow reads `vllm-version`, looks up the run ID from `vllm-wheels.yml`, and uses it automatically. If you provide `vllm_run_id` explicitly, it overrides the mapping.

### Default Runner Labels

| Step | Default Label | Override Input |
|------|---------------|----------------|
| vllm-wheel | `k8s-a100-build-13-0` | `build_label_wheel` |
| omni-wheel | `k8s-a100-build-13-0` | `build_label_wheel` |
| docker-image | `ibm-wdc-k8s-h100-dind` | `build_label_image` |

### Manual CLI Alternative

If you have nm-cicd access, you can trigger builds directly:

```bash
# Step 1: vLLM wheel
gh workflow run build-whl.yml --repo neuralmagic/nm-cicd \
  --ref vllm-omni-build \
  -f repo=neuralmagic/nm-vllm-ent \
  -f branch=main \
  -f target_device=cuda \
  -f python=3.12.5 \
  -f build_label=k8s-a100-build-13-0 \
  -f timeout=120 \
  -f partitions_file=neuralmagic/configs/partitions/minimal.yml

# Step 2: vllm-omni wheel (use VLLM_RUN_ID from step 1)
gh workflow run build-whl.yml --repo neuralmagic/nm-cicd \
  --ref vllm-omni-build \
  -f repo=neuralmagic/nm-vllm-omni-ent \
  -f branch=main \
  -f target_device=cuda \
  -f python=3.12.5 \
  -f build_label=k8s-a100-build-13-0 \
  -f timeout=120 \
  -f vllm_run_id=<VLLM_RUN_ID> \
  -f partitions_file=neuralmagic/configs/partitions/minimal.yml

# Step 3: Docker image (use both run IDs)
gh workflow run build-image.yml --repo neuralmagic/nm-cicd \
  --ref vllm-omni-build \
  -f repo=neuralmagic/nm-vllm-omni-ent \
  -f branch=main \
  -f target_device=cuda \
  -f build_label=ibm-wdc-k8s-h100-dind \
  -f run_id=<OMNI_RUN_ID> \
  -f vllm_run_id=<VLLM_RUN_ID>
```

## Workflow Naming Convention

Workflows in `.github/workflows/` are a mix of upstream and midstream:

- **Upstream workflows** (e.g. `build_wheel.yml`, `pre-commit.yml`) — carried forward from `vllm-project/vllm-omni`
- **Midstream workflows** — `omni-release.yml` (trigger), `midstream-build.yml` (deprecated)

See [.github-upstream-policy.md](.github-upstream-policy.md) for rebase guidelines.
