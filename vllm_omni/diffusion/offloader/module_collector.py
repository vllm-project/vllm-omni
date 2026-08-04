# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from operator import attrgetter

from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery

logger = init_logger(__name__)


@dataclass
class PipelineModules:
    dits: list[nn.Module]
    dit_names: list[str]
    encoders: list[nn.Module]
    encoder_names: list[str]
    vaes: list[nn.Module]
    resident_modules: list[nn.Module] = field(default_factory=list)
    resident_names: list[str] = field(default_factory=list)

    def outermost_dits(self) -> tuple[list[str], list[nn.Module]]:
        """Discovered DiTs that are not nested inside another discovered DiT.

        Some pipelines declare a DiT and one of its own submodules as separate
        DiTs (e.g. Cosmos3 lists both ``transformer`` and the nested
        ``transformer.language_model``) so that offloading can treat them as
        independent rings. HSDP instead shards each DiT through that DiT's own
        ``_hsdp_shard_conditions``; an inner DiT's blocks are already covered by
        its ancestor's conditions, so sharding it again would double-wrap them
        (and the inner stack may not declare conditions at all). Keeping only the
        outermost DiTs leaves the ancestor's conditions as the single source of
        truth for every nested block. Order follows ``dit_names``.
        """

        def is_nested_in_another_dit(dit: nn.Module) -> bool:
            for other in self.dits:
                if other is dit:
                    continue
                for submodule in other.modules():
                    if submodule is dit:
                        return True
            return False

        outer_names: list[str] = []
        outer_modules: list[nn.Module] = []
        for name, dit in zip(self.dit_names, self.dits):
            if is_nested_in_another_dit(dit):
                continue
            outer_names.append(name)
            outer_modules.append(dit)
        return outer_names, outer_modules


class ModuleDiscovery:
    """Discovers pipeline components via :class:`SupportsComponentDiscovery`."""

    @staticmethod
    def _collect_modules(
        pipeline: nn.Module,
        attr_names: list[str],
        *,
        warn_missing: bool = False,
    ) -> tuple[list[nn.Module], list[str]]:
        """Resolve attribute names to (module, name) pairs, skipping missing.

        Supports dotted paths via :func:`operator.attrgetter`.
        Warns on missing attributes when *warn_missing* is True.
        """
        modules: list[nn.Module] = []
        names: list[str] = []
        seen: set[int] = set()
        for attr in attr_names:
            try:
                module = attrgetter(attr)(pipeline)
            except AttributeError:
                module = None
            if module is None:
                if warn_missing:
                    logger.warning(
                        "Pipeline declares '%s' as offloadable but the attribute does not exist or is None",
                        attr,
                    )
                continue
            if not isinstance(module, nn.Module):
                logger.warning(
                    "Expected '%s' to be nn.Module, got %r",
                    attr,
                    type(module),
                )
                continue
            if id(module) not in seen:
                seen.add(id(module))
                modules.append(module)
                names.append(attr)
        return modules, names

    @staticmethod
    def discover(pipeline: nn.Module) -> PipelineModules:
        """Discover DiT, encoder, and VAE modules from pipeline.

        Args:
            pipeline: Diffusion pipeline model

        Returns:
            PipelineModules with lists of discovered modules and names
        """
        if not isinstance(pipeline, SupportsComponentDiscovery):
            logger.warning(
                "%s does not implement SupportsComponentDiscovery.",
                type(pipeline).__name__,
            )
            return PipelineModules(
                dits=[],
                dit_names=[],
                encoders=[],
                encoder_names=[],
                vaes=[],
            )

        dit_modules, dit_names = ModuleDiscovery._collect_modules(
            pipeline,
            pipeline._dit_modules,
            warn_missing=True,
        )
        encoders, encoder_names = ModuleDiscovery._collect_modules(
            pipeline,
            pipeline._encoder_modules,
            warn_missing=True,
        )
        vaes, _ = ModuleDiscovery._collect_modules(
            pipeline,
            pipeline._vae_modules,
            warn_missing=True,
        )
        residents, resident_names = ModuleDiscovery._collect_modules(
            pipeline,
            pipeline._resident_modules,
            warn_missing=True,
        )

        return PipelineModules(
            dits=dit_modules,
            dit_names=dit_names,
            encoders=encoders,
            encoder_names=encoder_names,
            vaes=vaes,
            resident_modules=residents,
            resident_names=resident_names,
        )
