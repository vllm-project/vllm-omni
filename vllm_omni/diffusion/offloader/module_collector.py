# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from operator import attrgetter

from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.models.interface import SupportsModuleOffload

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


class ModuleDiscovery:
    """Discovers pipeline components for offloading.

    If the pipeline implements :class:`SupportsModuleOffload`,
    its ``_dit_modules``, ``_encoder_modules``, and ``_vae_modules``
    class variables are used directly.  Otherwise, falls back to
    scanning well-known attribute names.
    """

    # Fallback attribute names for pipelines that do not implement
    # SupportsModuleOffload.
    _FALLBACK_DIT_ATTRS = [
        "transformer",
        "transformer_2",
        "dit",
        "sr_dit",
        "language_model",
        "transformer_blocks",
        "model",
    ]
    _FALLBACK_ENCODER_ATTRS = [
        "text_encoder",
        "text_encoder_2",
        "text_encoder_3",
        "image_encoder",
    ]
    _FALLBACK_VAE_ATTRS = [
        "vae",
        "audio_vae",
    ]

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
        declared = isinstance(pipeline, SupportsModuleOffload)
        if declared:
            dit_attrs = pipeline._dit_modules
            enc_attrs = pipeline._encoder_modules
            vae_attrs = pipeline._vae_modules
            res_attrs = pipeline._resident_modules
        else:
            dit_attrs = ModuleDiscovery._FALLBACK_DIT_ATTRS
            enc_attrs = ModuleDiscovery._FALLBACK_ENCODER_ATTRS
            vae_attrs = ModuleDiscovery._FALLBACK_VAE_ATTRS
            res_attrs = []

        dit_modules, dit_names = ModuleDiscovery._collect_modules(pipeline, dit_attrs, warn_missing=declared)
        encoders, encoder_names = ModuleDiscovery._collect_modules(pipeline, enc_attrs, warn_missing=declared)
        vaes, _ = ModuleDiscovery._collect_modules(pipeline, vae_attrs, warn_missing=declared)
        residents, resident_names = ModuleDiscovery._collect_modules(pipeline, res_attrs, warn_missing=declared)

        return PipelineModules(
            dits=dit_modules,
            dit_names=dit_names,
            encoders=encoders,
            encoder_names=encoder_names,
            vaes=vaes,
            resident_modules=residents,
            resident_names=resident_names,
        )
