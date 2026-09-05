# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Layer-wise offloading must honor the component-family selection.

``layerwise_offload_components`` names which families the backend may manage;
a family left out stays fully device-resident. Excluding ``dit`` is the
load-bearing case: it yields the topology where the DiT fits whole on the
device and only the other components are offloaded (the configuration the
sglang-side runs use for TP4xSP2), instead of paying the per-block streaming
cost on the component that dominates step time.
"""

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.offloader.base import OffloadConfig
from vllm_omni.diffusion.offloader.layerwise_backend import LayerWiseOffloadBackend

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


def test_selection_defaults_to_every_family() -> None:
    od = OmniDiffusionConfig(enable_layerwise_offload=True)

    assert od.layerwise_component_selection() == frozenset({"dit", "text_encoder", "vae"}), (
        "None must select every family -- that is the pre-existing behavior"
    )


def test_selection_parses_comma_list() -> None:
    od = OmniDiffusionConfig(enable_layerwise_offload=True, layerwise_offload_components=" text_encoder, vae ")

    assert od.layerwise_component_selection() == frozenset({"text_encoder", "vae"})


@pytest.mark.parametrize("bad", ["ditt", "dit,encoder", "", " , "])
def test_selection_rejects_unknown_or_empty(bad: str) -> None:
    # A typo must fail loudly: silently narrowing the selection would change
    # which components stay resident, and that is a capacity decision.
    with pytest.raises(ValueError, match="layerwise_offload_components"):
        OmniDiffusionConfig(enable_layerwise_offload=True, layerwise_offload_components=bad)


def test_offload_config_carries_selection() -> None:
    od = OmniDiffusionConfig(enable_layerwise_offload=True, layerwise_offload_components="text_encoder,vae")

    cfg = OffloadConfig.from_od_config(od)

    assert cfg.layerwise_components == frozenset({"text_encoder", "vae"})


def test_offload_config_defaults_to_every_family() -> None:
    cfg = OffloadConfig.from_od_config(OmniDiffusionConfig(enable_layerwise_offload=True))

    assert cfg.layerwise_components == frozenset({"dit", "text_encoder", "vae"})


class _Pipeline(nn.Module):
    _dit_modules = ["transformer"]
    _encoder_modules = ["text_encoder"]
    _vae_modules: list[str] = []
    _resident_modules: list[str] = []

    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Linear(2, 2)
        self.text_encoder = nn.Linear(2, 2)


class _Staged(nn.Module):
    """A component whose pipeline loads and releases it around each use."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
        self.norm = nn.Linear(2, 2)
        self.moves: list[str] = []

    def to(self, *args, **kwargs):  # noqa: A003 - mirrors nn.Module.to
        self.moves.append("to")
        return super().to(*args, **kwargs)

    def offload_to_cpu(self) -> None:
        self.moves.append("offload")


def _staged_pipeline(encoder: nn.Module, vae: nn.Module) -> nn.Module:
    """A pipeline that would trigger encoder streaming and VAE parking.

    It declares both components on-demand and declares the encoder's block
    stack, so with the family selected the backend parks them in host memory.
    That is exactly what the selection gate has to suppress.
    """
    from vllm_omni.diffusion.offloader.offload_plan import OffloadPlan

    class Pipeline(nn.Module):
        _dit_modules = ["transformer"]
        _encoder_modules = ["text_encoder"]
        _vae_modules = ["vae"]
        _resident_modules: list[str] = []
        _offload_plan = OffloadPlan(
            encoder_block_attrs={"text_encoder": ("blocks",)},
            on_demand_component_paths=frozenset({"text_encoder", "vae"}),
        )

        def __init__(self) -> None:
            super().__init__()
            self.transformer = nn.Linear(2, 2)
            self.text_encoder = encoder
            self.vae = vae

    return Pipeline()


def _backend(components: str | None) -> LayerWiseOffloadBackend:
    # Bypass __init__: it opens a device copy stream, which does not exist in a
    # CPU-only environment, and the excluded-DiT path never touches it.
    backend = object.__new__(LayerWiseOffloadBackend)
    backend.config = OffloadConfig.from_od_config(
        OmniDiffusionConfig(enable_layerwise_offload=True, layerwise_offload_components=components)
    )
    backend.device = torch.device("cpu")
    backend.enabled = False
    backend._blocks = []
    backend._streamed_encoders = []
    return backend


def test_enable_keeps_excluded_dit_resident_without_hooks() -> None:
    backend = _backend("text_encoder,vae")
    pipeline = _Pipeline()

    backend.enable(pipeline)

    assert backend.enabled is True, "the backend still manages the remaining families"
    assert getattr(pipeline.transformer, "_hook_registry", None) is None, (
        "an excluded DiT must not receive block-streaming hooks"
    )
    assert backend._blocks == [], "no block group may be tracked for an excluded DiT"


def test_serve_cli_registers_flag_and_carries_it_to_default_stage() -> None:
    # The serve parser is hand-written. A dataclass field alone would leave the
    # public flag unregistered and fail before the server starts.
    from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
    from vllm_omni.entrypoints.cli.serve import OmniServeCommand
    from vllm_omni.utils.tracking_parser import TrackingArgumentParser

    parser = TrackingArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand")
    OmniServeCommand().subparser_init(subparsers)

    args = parser.parse_args(["serve", "fake-model", "--omni", "--layerwise-offload-components", "text_encoder,vae"])
    explicit = args.get_explicit_kwargs_dict()
    assert explicit["layerwise_offload_components"] == "text_encoder,vae"

    stage_cfg = AsyncOmniEngine._create_default_diffusion_stage_cfg(
        {"model": "fake-model", "layerwise_offload_components": explicit["layerwise_offload_components"]}
    )
    assert stage_cfg[0]["engine_args"]["layerwise_offload_components"] == "text_encoder,vae"


def test_typed_deploy_chain_carries_selection_to_projection() -> None:
    # Projection silently filters unknown fields, so exercise the complete
    # typed deployment chain rather than checking only the source dataclass.
    from vllm_omni.config.omni_config import _DiffusionConfigProjection, _stage_engine_overrides
    from vllm_omni.config.stage_config import StageDeployConfig

    overrides = _stage_engine_overrides(StageDeployConfig(stage_id=0, layerwise_offload_components="text_encoder,vae"))
    assert overrides["layerwise_offload_components"] == "text_encoder,vae"

    projection = _DiffusionConfigProjection.from_kwargs(**overrides)
    assert projection.layerwise_offload_components == "text_encoder,vae"

    empty = _DiffusionConfigProjection.from_kwargs(**_stage_engine_overrides(StageDeployConfig(stage_id=0)))
    assert empty.layerwise_offload_components is None


def test_enable_leaves_an_excluded_encoder_family_untouched() -> None:
    """ "text_encoder" excluded: no streaming, no parking, just resident.

    With the family selected this same pipeline would have its encoder blocks
    streamed and the encoder parked in host memory (it declares both). Excluding
    the family must produce the behaviour of a backend that does not manage
    encoders at all: a plain placement onto the device.
    """
    encoder, vae = _Staged(), _Staged()
    backend = _backend("dit,vae")

    backend.enable(_staged_pipeline(encoder, vae))

    assert "offload" not in encoder.moves, "an excluded encoder must not be parked in host memory"
    assert encoder.moves == ["to"], f"an excluded encoder must only be placed, got {encoder.moves}"
    assert backend._streamed_encoders == [], "an excluded encoder must not have its blocks streamed"
    assert getattr(encoder, "_omni_layerwise_blocks", None) is None, (
        "an excluded encoder must not carry block-streaming state"
    )
    backend.disable()


def test_enable_leaves_an_excluded_vae_family_untouched() -> None:
    """ "vae" excluded: the declared, stageable VAE is made resident instead.

    With the family selected this VAE is parked (it is declared on-demand and
    exposes ``offload_to_cpu``). Excluding the family must fall back to the
    plain resident placement.
    """
    encoder, vae = _Staged(), _Staged()
    backend = _backend("dit,text_encoder")

    backend.enable(_staged_pipeline(encoder, vae))

    assert "offload" not in vae.moves, "an excluded VAE must not be parked in host memory"
    assert vae.moves == ["to"], f"an excluded VAE must only be placed, got {vae.moves}"
    backend.disable()
