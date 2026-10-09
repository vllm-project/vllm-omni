# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Startup attention assignments and shared whole-forward strategy dispatch."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from math import ceil, isfinite

from vllm_omni.diffusion.attention.contracts import (
    AttentionExecutionEnvironment,
    StrategyModelSupport,
    bind_attention_execution,
    validate_strategy_parallel,
)


@dataclass(frozen=True)
class AttentionOperation:
    identity: str
    layer: int
    role: str
    category: str | None
    invariant: bool = False
    component: str = field(kw_only=True)

    def __post_init__(self):
        if not self.component or not self.identity or type(self.layer) is not int or self.layer < 0 or not self.role:
            raise ValueError("Attention operations require component, identity, nonnegative layer and role")


@dataclass(frozen=True)
class Override:
    layers: tuple[int, ...] | None
    role: str | None
    use: str
    component: str | None = None

    def matches(self, operation):
        return (
            (self.component is None or self.component == operation.component)
            and (self.layers is None or operation.layer in self.layers)
            and (self.role is None or self.role in (operation.role, operation.category))
        )


@dataclass(frozen=True)
class Layout:
    default: str
    overrides: tuple[Override, ...]

    def assignment(self, operation):
        matches = [rule.use for rule in self.overrides if rule.matches(operation)]
        if len(matches) > 1:
            raise ValueError(f"Overlapping attention overrides for {operation.identity} ({operation.role})")
        return matches[0] if matches else self.default


def _mapping(value, fields, label):
    if not isinstance(value, Mapping) or value.keys() - fields:
        raise ValueError(f"Invalid {label}; expected fields {sorted(fields)}")
    return value


class AttentionStrategy:
    def __init__(self, presets, layout, layouts, schedule):
        if not presets:
            raise ValueError("An attention strategy requires presets")
        self.presets = deepcopy(dict(presets))
        if layout is not None:
            if layouts is not None or schedule is not None:
                raise ValueError("Use either layout or layouts with schedule")
            raw_layouts = {"static": layout}
        else:
            if not layouts or schedule is None:
                raise ValueError("Named layouts require a schedule")
            raw_layouts = layouts
        if not isinstance(raw_layouts, Mapping):
            raise ValueError("layouts must be a mapping")
        if any(not isinstance(name, str) or not name for name in raw_layouts):
            raise ValueError("Layout names must be nonempty strings")
        self.layouts = {name: self._parse_layout(raw) for name, raw in raw_layouts.items()}
        self.coordinate, self.phases = self._parse_schedule(schedule)
        if self.phases:
            reachable = {name for _, name in self.phases}
            self.layouts = {name: layout for name, layout in self.layouts.items() if name in reachable}

    def _parse_layout(self, raw) -> Layout:
        _mapping(raw, {"default", "overrides"}, "layout")
        default = raw.get("default")
        self._preset(default)
        rules = []
        raw_rules = raw.get("overrides", [])
        if not isinstance(raw_rules, list):
            raise ValueError("overrides must be a list")
        for rule in raw_rules:
            _mapping(rule, {"layers", "attention_role", "use", "component"}, "override")
            self._preset(rule.get("use"))
            layers = rule.get("layers")
            if layers is not None and (
                not isinstance(layers, list)
                or not layers
                or any(type(i) is not int or i < 0 for i in layers)
                or len(set(layers)) != len(layers)
            ):
                raise ValueError("layers must be distinct nonnegative integer indices")
            role = rule.get("attention_role")
            if role is not None and (not isinstance(role, str) or not role):
                raise ValueError("attention_role must be a nonempty string")
            component = rule.get("component")
            if component is not None and (not isinstance(component, str) or not component):
                raise ValueError("component must be a nonempty string")
            rules.append(Override(None if layers is None else tuple(layers), role, rule["use"], component))
        return Layout(default, tuple(rules))

    def _parse_schedule(self, schedule) -> tuple[str | None, tuple[tuple[int | float | None, str], ...]]:
        if schedule is None:
            return None, ()
        _mapping(schedule, {"coordinate", "phases"}, "schedule")
        coordinate = schedule.get("coordinate")
        if coordinate not in ("step_index", "step_fraction"):
            raise ValueError("schedule.coordinate must be step_index or step_fraction")
        phases = schedule.get("phases")
        if not isinstance(phases, list) or not phases:
            raise ValueError("schedule.phases must be a nonempty list")
        parsed = []
        previous = 0
        for index, phase in enumerate(phases):
            _mapping(phase, {"until", "layout"}, "phase")
            if "until" not in phase or phase.get("layout") not in self.layouts:
                raise ValueError("Each phase needs until and a known layout")
            until = phase["until"]
            final = index == len(phases) - 1
            if coordinate == "step_index":
                valid = until is None if final else type(until) is int and until > previous
            else:
                valid = type(until) in (int, float) and isfinite(until) and previous < until <= 1
                valid = valid and (not final or until == 1)
            if not valid:
                raise ValueError("Phase boundaries must increase; end at null (step_index) or 1 (step_fraction)")
            parsed.append((until, phase["layout"]))
            previous = until
        return coordinate, tuple(parsed)

    def _preset(self, name):
        if not isinstance(name, str) or name not in self.presets:
            raise ValueError(f"Unknown attention preset: {name!r}")

    def assignments(self, operation: AttentionOperation):
        specs = tuple(self.presets[layout.assignment(operation)] for layout in self.layouts.values())
        if operation.invariant and any(spec != specs[0] for spec in specs):
            raise ValueError(f"Attention operation {operation.identity!r} must be invariant across reachable layouts")
        return specs

    def validate_inventory(self, operations):
        operations = tuple(operations)
        if not operations:
            raise ValueError("An attention strategy requires a nonempty operation inventory")
        if len({(op.component, op.identity) for op in operations}) != len(operations):
            raise ValueError("Attention operation identities must be unique within each component")
        roles = {r for op in operations for r in (op.role, op.category) if r is not None}
        for layout in self.layouts.values():
            for rule in layout.overrides:
                if rule.role is not None and rule.role not in roles:
                    raise ValueError(f"Unknown attention role: {rule.role}")
                matched = [op for op in operations if rule.matches(op)]
                if not matched or (rule.layers is not None and set(rule.layers) - {op.layer for op in matched}):
                    raise ValueError(f"Unknown layers or empty attention selector: {rule}")
        for operation in operations:
            self.assignments(operation)
        return operations


def validate_strategy_runtime(config):
    attention_config = getattr(config, "diffusion_attention_config", None)
    if getattr(attention_config, "strategy", None) is None:
        return
    validate_strategy_parallel(config)
    unsupported = []
    if config.cache_backend not in (None, "none"):
        unsupported.append(f"cache acceleration ({config.cache_backend})")
    if config.step_execution:
        unsupported.append("step execution")
    if config.streaming_output:
        unsupported.append("streaming output")
    if config.diffusion_kv_mode.value != "dense_legacy":
        unsupported.append(f"KV mode {config.diffusion_kv_mode.value}")
    if config.diffusion_kv_cache_dtype not in (None, "auto", "float"):
        unsupported.append(f"KV quantization ({config.diffusion_kv_cache_dtype})")
    if unsupported:
        raise ValueError("Attention strategies do not support: " + ", ".join(unsupported))


def iter_attention_strategy_runners(pipeline):
    """Yield declared, present components and their possibly unfinalized runner.

    Missing optional components are skipped. A present component without a
    runner yields None so startup validation can distinguish the two cases.
    """
    for name in getattr(pipeline, "attention_strategy_components", ()) or ():
        model = getattr(pipeline, name, None)
        if model is not None:
            yield name, getattr(model, "_attention_strategy_runner", None)


def validate_pipeline_attention_strategy(pipeline, config):
    """Validate all declared components together; never mutate shared configuration."""
    strategy = getattr(getattr(config, "diffusion_attention_config", None), "strategy", None)
    if strategy is None:
        return
    components = getattr(pipeline, "attention_strategy_components", None)
    if not components:
        raise ValueError("Pipeline must declare its attention_strategy_components contract")
    operations = []
    found = set()
    for name, runner in iter_attention_strategy_runners(pipeline):
        if runner is None:
            raise ValueError(f"Pipeline component {name!r} did not finalize an attention strategy")
        owned = {op.component for op in runner.plan.operations}
        if owned != {name} or name in found:
            raise ValueError(f"Pipeline component {name!r} has an inconsistent operation inventory")
        found.add(name)
        operations.extend(runner.plan.operations)
    strategy.validate_inventory(operations)


def finalize_attention_strategy(model):
    # Import lazily: attention construction also uses the pure resolver above.
    from vllm_omni.diffusion.attention.layer import StrategyAttention
    from vllm_omni.diffusion.config import get_current_diffusion_config_or_none

    config = get_current_diffusion_config_or_none()
    attention_config = getattr(config, "diffusion_attention_config", None)

    layers = [m for m in model.modules() if isinstance(m, StrategyAttention)]
    if not layers:
        if getattr(attention_config, "strategy", None) is not None:
            raise ValueError("The model did not register any attention operations for its configured strategy")
        return
    strategy = layers[0]._attention_strategy
    if any(layer._attention_strategy is not strategy for layer in layers):
        raise ValueError("All attention operations in a model must share one strategy")
    support = getattr(model, "attention_strategy_support", None)
    if not isinstance(support, StrategyModelSupport):
        raise ValueError("Transformer must declare an attention_strategy_support contract")
    if support.prepares_inputs and not callable(getattr(model, "prepare_attention_strategy_inputs", None)):
        raise ValueError("Model strategy contract requires an input preparation hook")
    operations = tuple(layer._strategy_operation for layer in layers)
    components = {operation.component for operation in operations}
    if len(components) != 1:
        raise ValueError("A transformer forward must own exactly one attention component")
    if len({operation.identity for operation in operations}) != len(operations):
        raise ValueError("Attention operation identities must be unique within each component")
    if hasattr(model, "_attention_strategy_runner"):
        raise ValueError("Attention strategy is already finalized; rebuild the model to change it")
    plan = ForwardStrategyPlan.from_strategy(strategy, operations=operations)
    if not callable(getattr(model, "forward_with_attention_layout", None)):
        raise ValueError("Strategy models must implement forward_with_attention_layout and dispatch from forward")
    environment = AttentionExecutionEnvironment.for_model(config, support)
    runner = AttentionStrategyRunner(model, plan)
    bind_attention_execution(model, environment)
    model._attention_strategy_runner = runner


@dataclass(frozen=True)
class ForwardStrategyPlan:
    """Immutable startup snapshot; select layouts from logical request progress."""

    layout_names: tuple[str, ...]
    coordinate: str | None
    phases: tuple[tuple[int | float | None, int], ...]
    operations: tuple[AttentionOperation, ...] = ()

    @classmethod
    def from_strategy(cls, strategy, *, operations=()):
        names = tuple(strategy.layouts)
        return cls(
            names, strategy.coordinate, tuple((until, names.index(name)) for until, name in strategy.phases), operations
        )

    def layout_for_step(self, step: int | None, total: int | None) -> int:
        if type(step) is not int or type(total) is not int or not 0 <= step < total:
            raise ValueError("Scheduled attention requires valid logical denoising progress")
        if self.coordinate is None:
            return 0
        for until, layout in self.phases:
            if until is None:
                return layout
            end = ceil(until * total) if self.coordinate == "step_fraction" else until
            if step < end:
                return layout
        raise ValueError("Attention schedule does not cover the requested step")

    def current_layout(self):
        """Called once at the eager transformer boundary, never by compiled attention."""
        if self.coordinate is None:
            return 0
        from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available

        if not is_forward_context_available():
            raise ValueError("Scheduled attention requires logical denoising progress")
        context = get_forward_context()
        return self.layout_for_step(context.denoise_step_idx, context.total_denoise_steps)


class AttentionStrategyRunner:
    """Select immutable layouts outside full-transformer or regional compilation.

    Regional compilation leaves preparation and offloading eager.
    Layouts share model weights and exercise the same compiled decoder blocks.
    """

    def __init__(self, model, plan: ForwardStrategyPlan):
        self.plan = plan
        self._model = model
        self.prepare_inputs = getattr(model, "prepare_attention_strategy_inputs", None)
        forward = model.forward_with_attention_layout
        self._forwards = tuple(partial(forward, attention_layout=layout) for layout in range(len(plan.layout_names)))
        self.compile_granularity: str | None = None
        self.warming = False
        self._warmed_layouts: set[int] = set()

    @property
    def compiled(self) -> bool:
        return self.compile_granularity is not None

    def __getstate__(self):
        if self.compiled:
            raise TypeError(
                "Compiled attention strategies are process-local; save state_dict() and rebuild before loading"
            )
        return self.__dict__

    def compile(self, *, dynamic=True, backend="inductor", recompile_limit=8, granularity="full"):
        import torch

        if self.compiled:
            raise ValueError("Attention strategy forwards are already compiled")
        if granularity not in ("full", "regional"):
            raise ValueError("Attention strategy compilation must be full or regional")
        layerwise_offload = isinstance(self._model, torch.nn.Module) and any(
            getattr(module, "_hook_registry", None) is not None
            and module._hook_registry.get_hook("layerwise_offload") is not None
            for module in self._model.modules()
        )
        if granularity == "full" and (getattr(self._model, "_model_cpu_offload_enabled", False) or layerwise_offload):
            raise ValueError("CPU offloading requires regional compilation or eager execution")
        environment = getattr(self._model, "attention_execution", None)
        if environment is not None:
            if environment.ulysses_degree > 1 and granularity == "full":
                raise ValueError("Ulysses attention strategies require regional compilation or eager execution")
        if type(recompile_limit) is not int or recompile_limit < 1:
            raise ValueError("Attention strategy recompile_limit must be a positive integer")
        if granularity == "regional":
            from vllm_omni.diffusion.compile import regionally_compile

            repeated_blocks = getattr(self._model, "_repeated_blocks", ())
            if not any(type(module).__name__ in repeated_blocks for module in self._model.modules()):
                raise ValueError("Regional attention strategies require declared repeated decoder blocks")
            regionally_compile(
                self._model,
                backend=backend,
                dynamic=dynamic,
                fullgraph=False,
                recompile_limit=recompile_limit,
                isolate_recompiles=True,
            )
        else:
            # Partials share a code object. Isolate each layout's guard budget
            # while keeping Dynamo's aggregate cap intact. Tracing remains lazy.
            self._forwards = tuple(
                torch.compile(
                    forward,
                    backend=backend,
                    fullgraph=False,
                    dynamic=dynamic,
                    recompile_limit=recompile_limit,
                    isolate_recompiles=True,
                )
                for forward in self._forwards
            )
        self.compile_granularity = granularity
        self._warmed_layouts.clear()

    def warmup_status(self):
        return {
            "compiled": self.compiled,
            "compile_granularity": self.compile_granularity,
            "layouts_exercised": tuple(self.plan.layout_names[i] for i in sorted(self._warmed_layouts)),
            "layouts_pending": tuple(
                name for i, name in enumerate(self.plan.layout_names) if i not in self._warmed_layouts
            ),
        }

    def set_warmup(self, active):
        self.warming = active

    def __call__(self, *args, **kwargs):
        if "attention_layout" in kwargs:
            raise ValueError("Attention layout is deployment-owned; request overrides are not supported")
        layout = self.plan.current_layout()
        if self.prepare_inputs is not None:
            args, kwargs = self.prepare_inputs(args, kwargs)
        # Explicit warmup exercises every layout even when a short denoising
        # schedule would never select it. Return only the scheduled result.
        if not self.warming:
            return self._forwards[layout](*args, **kwargs)
        for index, forward in enumerate(self._forwards):
            result = forward(*args, **kwargs)
            if index == layout:
                output = result
            self._warmed_layouts.add(index)
        return output
