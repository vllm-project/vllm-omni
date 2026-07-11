# Composable parallel strategies

In vLLM-Omni, the `composable_parallel` system layers a declarative *parallel strategy* on top of the existing stage configs. A strategy YAML maps each pipeline stage (by its `model_stage` name) to a stack of axis declarations — one per parallelism dimension — that the engine translates into per-stage parallel sizing (`tensor_parallel_size`, `data_parallel_size`, `pipeline_parallel_size`, expert-parallel toggle, replica count) and, when relevant, an orchestrator-wide load-balancer policy.

Strategies are intentionally narrow: each one only writes the axes it declares, leaving every other engine arg untouched. This makes a strategy file a small, reusable overlay you can swap independently of the deploy YAML (see `configuration/stage_configs.md`).

!!! note
    Composable parallel is **opt-in**. Passing `--strategy-config <path/to/strategy.yaml>` overlays the derived sizing onto the registry-merged stages *before* any CLI overrides are applied. When `--strategy-config` is omitted, stages remain exactly as their deploy YAML produces them.

## Top-level schema reference

A strategy file has a single top-level mapping, `strategies`, keyed by the `model_stage` name of each stage you want to overlay. Each value is a list of axis declarations (see [StrategySpec fields](#strategyspec-fields) below).

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `strategies` | `dict[str, list]` | required | — | Maps `model_stage` name (e.g. `thinker`, `talker`, `code2wav`, `dit`) to a list of axis declarations. The key MUST be the stage name, not the integer `stage_id`. See `vllm_omni/config/composable_parallel/strategy_loader.py`. |

### StrategySpec fields

Each entry in a stage's list becomes one `StrategySpec` (see `vllm_omni/config/composable_parallel/spec.py`). The loader fills in `routing` and `aggregation` from per-kind defaults, so the file stays compact.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `axis` | str | required | — | Mesh-axis kind. One of `tp`, `dp`, `pp`, `ep`, `stage_replica` (wired today) or `sp_ulysses`, `sp_ring`, `cfg`, `vae_pp`, `hsdp`, `stage_pp`, `cp` (reserved; declaring one raises `AxisTranslationError` at translate time). See `MeshAxisKind` in `vllm_omni/config/composable_parallel/spec.py`. |
| `size` | int | required | — | Axis degree. Must be `> 0`. |
| `routing` | str | optional | per-kind default | Stateless routing policy. Accepted only for `dp` / `stage_replica`; other kinds reject it. Values: `random`, `round_robin`, `least_queue`. (`hash` is recognised but currently raises `NotImplementedError`.) See `RouteByStage` in `vllm_omni/config/composable_parallel/routing.py`. |
| `l1_owner` | str | optional | per-kind default | Overrides which layer owns request routing for this axis. `engine` for `tp`/`dp`/`pp`/`ep`, `delegated` for `stage_replica`. Recorded into `shard_extension`. |
| `name` | str | optional | axis kind | Human-readable label for this spec, used in error messages. |

Two additional `StrategySpec` slots — `layer_hook_specs` and `kernel_specs` — let advanced authors bind layer-walker hook IDs (`LayerHookSpec`) or kernel IDs (`KernelSpec`) onto an axis. These are not exposed by the YAML loader today and are intended for programmatic construction. See `vllm_omni/config/composable_parallel/spec.py` for the dataclasses.

### CLI flags

| Flag | Description |
|------|-------------|
| `--strategy-config PATH` | Path to a `strategy.yaml`. Only takes effect on registry-based deploy paths (`--deploy-config` or the bundled default deploy YAML). Its derived sizing is overlaid onto the merged stages *before* CLI overrides. |
| `--stage-configs-path PATH` | **Legacy.** Loads the old `stage_args` YAML schema for un-migrated models. Passing `--strategy-config` together with this path is silently inapplicable — vLLM-Omni emits a warning and the strategy is ignored (see `vllm_omni/entrypoints/utils.py`, lines 372–380). |
| `--omni-lb-policy POLICY` | Orchestrator-wide load-balancer policy. Strategy-derived from any `stage_replica` axis; an explicit CLI value wins and must match the derived one or `AsyncOmniEngine` raises `Conflicting load-balancer policy` at construction. |

!!! warning
    `--strategy-config` cannot be used with `--stage-configs-path` (the legacy deploy path). If your model has not been migrated to the registry-based pipeline yet, the strategy is silently dropped with a warning. Migrate to a registry-based model or use `--deploy-config` to apply a strategy.

## Sub-schemas

### `axis`

The set of recognised axis kinds is the `MeshAxisKind` enum in `vllm_omni/config/composable_parallel/spec.py`:

```yaml
# Translatable today (see `_SUPPORTED_KINDS` in translator.py).
axis: tp              # tensor parallel — sets tensor_parallel_size
axis: dp              # engine data parallel — sets data_parallel_size (vLLM's intra-engine DP LB)
axis: pp              # pipeline parallel — sets pipeline_parallel_size
axis: ep              # dense expert parallel — sets enable_expert_parallel=True; size must equal tp*dp
axis: stage_replica   # omni-coordinator-level replica fan-out — sets num_replicas + omni_lb_policy

# Reserved (raise AxisTranslationError today; declared in the type system for forward compat).
axis: sp_ulysses
axis: sp_ring
axis: cfg
axis: vae_pp
axis: hsdp
axis: stage_pp
axis: cp
```

### `routing`

The full closed hierarchy lives in `vllm_omni/config/composable_parallel/routing.py` (`Broadcast`, `PartitionByHash`, `ShardSequence`, `ShardSpatial`, `RouteByExpert`, `PipelineMicrobatch`, `DuplicateWithCondTag`, `RouteByStage`). For YAML authors only `RouteByStage` is exposed as a `routing:` string, and only for `dp` / `stage_replica` axes (other axes have fixed routing — `tp`/`ep` broadcast, `pp` microbatch). Recognised policies:

```yaml
routing: random         # uniform-random across replicas/DP ranks
routing: round_robin    # cycle through replicas/DP ranks (the default for dp/stage_replica)
routing: least_queue    # send to the replica with the smallest backlog
# routing: hash         # recognised but raises NotImplementedError; no key-stable balancer yet
```

### `aggregation`

Aggregation is filled in automatically by the loader (`_default_aggregation` in `vllm_omni/config/composable_parallel/strategy_loader.py`) from the axis kind, so users do not author it:

| Axis kind | Aggregation pattern |
|-----------|---------------------|
| `tp`, `ep` | `TakeRank` (all ranks agree; pick one) |
| `dp`, ep (in some flows) | `Union` (disjoint replica results) |
| `pp` | `StitchPipeline` |
| `stage_replica` | `FanInByStage` |

Full pattern hierarchy: `TakeRank`, `Union`, `GatherDim`, `AllGather`, `StitchSpatial`, `StitchPipeline`, `Combine`, `FanInByStage` (see `vllm_omni/config/composable_parallel/aggregation.py`).

## Precedence

From highest to lowest, when the same axis is touched by multiple layers:

1. **CLI overrides** (`--stage_N_tensor_parallel_size`, `--stage-overrides`, `--omni-lb-policy`, etc.). A CLI value always wins; a warning is emitted whenever it overrides a strategy-declared axis (see `_reconcile_strategy_with_cli` in `vllm_omni/config/config_factory.py`).
2. **Strategy YAML** (`--strategy-config`). Derived sizing is written onto the merged stages with *conflict-on-explicit* semantics — if the deploy YAML already set a value to something different, application raises `StrategyApplyError`.
3. **Deploy YAML** (`--deploy-config` or the bundled `vllm_omni/deploy/<model_type>.yaml`).
4. **Parser defaults.**

After CLI overrides land, the device-layout guard re-runs (`check_device_layout`, called from `_reconcile_strategy_with_cli`) against the *effective* `tp * dp * pp * num_replicas` and the resolved `devices` string, so a `--devices` or `--tensor-parallel-size` that conflicts with the strategy's world size cannot slip past the pre-spawn check.

## Worked example

### Example 1 — `strategy_tp2.yaml`

Shard the `thinker` stage with TP=2 on top of the bundled Qwen2.5-Omni deploy YAML. Routing is fixed (broadcast) for TP, so no `routing:` field is needed.

```yaml
# examples/offline_inference/qwen2_5_omni/strategy_tp2.yaml
strategies:
  thinker:
    - axis: tp
      size: 2
```

Launch:

```bash
python3 end2end.py --query-type text --num-prompts 6 \
    --strategy-config strategy_tp2.yaml \
    --stage-overrides '{"0": {"devices": "0,1"}, "1": {"devices": "2"}, "2": {"devices": "2"}}'
```

Effective `OmniParallelConfig` per stage after the overlay (see `OmniParallelConfig` in `vllm_omni/config/composable_parallel/translator.py`):

| Stage (`model_stage`) | `tensor_parallel_size` | `data_parallel_size` | `pipeline_parallel_size` | `stage_replica_size` | `omni_lb_policy` |
|-----------------------|------------------------|----------------------|--------------------------|----------------------|------------------|
| `thinker`             | **2** (from strategy)  | 1                    | 1                        | 1                    | —                |
| `talker`              | 1                      | 1                    | 1                        | 1                    | —                |
| `code2wav`            | 1                      | 1                    | 1                        | 1                    | —                |

### Example 2 — `strategy_stage_replica.yaml`

Replicate the `thinker` stage across two independent engines and let omni's StagePool round-robin requests across them.

```yaml
# examples/offline_inference/qwen2_5_omni/strategy_stage_replica.yaml
strategies:
  thinker:
    - axis: stage_replica
      size: 2
      routing: round_robin
```

Launch:

```bash
python3 end2end.py --query-type text --num-prompts 6 \
    --strategy-config strategy_stage_replica.yaml \
    --stage-overrides '{"0": {"devices": "0,1"}, "1": {"devices": "2"}, "2": {"devices": "2"}}'
```

Effective per-stage config and the derived orchestrator knob:

| Stage (`model_stage`) | `tensor_parallel_size` | `stage_replica_size` | `omni_lb_policy` (derived) |
|-----------------------|------------------------|----------------------|----------------------------|
| `thinker`             | 1                      | **2**                | **`round-robin`**          |
| `talker`              | 1                      | 1                    | —                          |
| `code2wav`            | 1                      | 1                    | —                          |

The derived `omni_lb_policy = round-robin` is **not** written into any stage's engine args. Instead it is surfaced on `StrategyApplyResult.omni_lb_policy` and applied once at orchestrator construction by `AsyncOmniEngine._apply_strategy_lb_policy` (see `vllm_omni/engine/async_omni_engine.py`, lines 1107–1134). If you also pass `--omni-lb-policy round-robin` the values must match; passing a different value raises `Conflicting load-balancer policy`.

## Per-field deep-dive

### `name`

Default: the axis kind (e.g. `"tp"`).

Human-readable label that appears in `AxisTranslationError` and `StrategyApplyError` messages. Set it to disambiguate two stages declaring the same axis kind when scanning logs.

### `mesh_axis` (`axis` + `size`)

Default: no default — both fields are required.

`axis` chooses the parallelism dimension; `size` is its degree. Five kinds are wired end-to-end today (`tp`, `dp`, `pp`, `ep`, `stage_replica`); the remaining kinds in `MeshAxisKind` are reserved and raise `AxisTranslationError` at translate time so a typo cannot silently become a no-op (see `_SUPPORTED_KINDS` in `vllm_omni/config/composable_parallel/translator.py`). For `ep`, the size MUST equal `tensor_parallel_size * data_parallel_size`.

### `routing`

Default: per-kind — `RouteByStage(round_robin)` for `dp` and `stage_replica`; `PipelineMicrobatch()` for `pp`; `Broadcast()` for `tp`/`ep` and any other kind.

User-facing string only on `dp` and `stage_replica` axes. The four recognised values are `random`, `round_robin`, `least_queue`, `hash`; `hash` is rejected today with `NotImplementedError`. See `RouteByStage` in `vllm_omni/config/composable_parallel/routing.py`.

### `l1_owner`

Default: per-kind — `engine` for `tp`/`dp`/`pp`/`ep`, `delegated` for `stage_replica`.

Overrides which layer owns request routing along this axis. `engine` means vLLM realizes routing intra-engine; `delegated` means the omni coordinator owns routing across independent replicas. Set this only when overriding the per-kind default in `_DEFAULT_L1_OWNER` (`vllm_omni/config/composable_parallel/translator.py`); invalid values fail fast in `_resolve_l1_owner`.

### `stage_replica` and `omni_lb_policy`

`stage_replica` is *not* a vLLM world dimension; it spins up `size` independent engine replicas of one stage, coordinated by the omni `StagePool` load balancer. The axis writes both:

- The stage's `num_replicas` (a per-stage deploy knob).
- The pipeline-wide `omni_lb_policy` string (an orchestrator-level knob).

Because `omni_lb_policy` is read once at `AsyncOmniEngine` construction — not from per-stage engine args — only one value is allowed across the whole pipeline. If two stages declare conflicting policies, application raises (see `apply_strategy_specs` in `vllm_omni/config/composable_parallel/apply.py`, lines 253–260). To set this knob manually instead, use `--omni-lb-policy` on the CLI; see [the model stage docs](stage_configs.md) for how stage replicas plug into the orchestrator at runtime.

### `model_stage` keys

Default: no default — required for every entry under `strategies:`.

Strategy keys MUST be `model_stage` *names* (strings), the same labels stage configs use for `engine_args.model_stage` (see [`model_stage` in `stage_configs.md`](stage_configs.md#engine_argsmodel_stage)). Integer `stage_id` values are not accepted; the resolver matches on `model_stage` exactly (`_resolve_stage` in `vllm_omni/config/composable_parallel/apply.py`).

## Common pitfalls

!!! warning "Using `stage_id` integers as keys"
    Strategy keys are `model_stage` names (str), not integer stage IDs. A file that uses an integer index will raise at apply time:

    ```text
    StrategyApplyError: strategy role model_stage=0 did not match any stage;
    available stages: 'thinker'(id=0), 'talker'(id=1), 'code2wav'(id=2)
    ```

    Use the `model_stage` name (e.g. `thinker`) instead.

!!! warning "Mixing CLI overrides with a strategy on the same axis"
    A strategy is meant to be the single writer for the axes it declares. If a CLI override (`--stage_0_tensor_parallel_size`, `--stage-overrides '{"0": {...}}'`) sets the same axis, the **CLI value wins** and a warning is emitted from `_reconcile_strategy_with_cli`. The device-layout guard then re-runs against the effective layout, so a CLI override that breaks `tp * dp * pp * num_replicas` will still fail fast at config time.

!!! warning "Passing `--strategy-config` against the legacy `--stage-configs-path`"
    Composable parallel only applies on the registry-based deploy path. If a model still resolves through the legacy `stage_args` YAML (loaded via `--stage-configs-path`), `--strategy-config` is silently inapplicable and emits:

    ```text
    --strategy-config (<path>) was provided but model '<model>' resolves via the
    legacy stage_configs YAML path, which does not support composable-parallel
    strategies; the strategy is ignored. Use a registry-based model to apply it.
    ```

    Migrate the model to the new deploy schema (see `configuration/stage_configs.md`) before applying a strategy.

!!! warning "Conflicting `omni_lb_policy` across stages"
    `omni_lb_policy` is a single pipeline-wide knob. If two `stage_replica` axes in different stages derive different policies, `apply_strategy_specs` raises:

    ```text
    StrategyApplyError: role 'talker' derives omni_lb_policy='least-queue-length'
    but role 'thinker' already derived 'round-robin'. The load-balancer policy is
    pipeline-wide; only one value is allowed.
    ```

    Make the policies match, drop one, or supply the value explicitly via `--omni-lb-policy`.
