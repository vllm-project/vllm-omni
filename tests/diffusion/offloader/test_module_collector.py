# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for ModuleDiscovery and SupportsModuleOffload."""

from typing import ClassVar

import pytest
from torch import nn

from vllm_omni.diffusion.models.interface import SupportsModuleOffload
from vllm_omni.diffusion.offloader.module_collector import ModuleDiscovery

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]

# NOTE: tests for skipped/warned attributes verify the *behavioral*
# outcome (attribute excluded from results) but do not assert on log
# output.  vllm's logger sets propagate=False, preventing caplog from
# capturing records.  See https://github.com/pytest-dev/pytest/issues/3697


# ---------------------------------------------------------------------------
# Test pipelines
# ---------------------------------------------------------------------------


class FallbackPipeline(nn.Module):
    """Pipeline with standard attribute names (no protocol)."""

    def __init__(self):
        super().__init__()
        self.transformer = nn.Linear(10, 10)
        self.text_encoder = nn.Linear(10, 10)
        self.text_encoder_2 = nn.Linear(10, 10)
        self.vae = nn.Linear(10, 10)


class NonModuleAttrPipeline(nn.Module):
    """Pipeline where an attribute is not an nn.Module (fallback path)."""

    def __init__(self):
        super().__init__()
        self.transformer = nn.Linear(10, 10)
        self.text_encoder = "not_a_module"
        self.vae = nn.Linear(10, 10)


class DuplicateAttrPipeline(nn.Module):
    """Pipeline where two encoder attrs point to the same module."""

    def __init__(self):
        super().__init__()
        self.transformer = nn.Linear(10, 10)
        encoder = nn.Linear(10, 10)
        self.text_encoder = encoder
        self.text_encoder_2 = encoder
        self.vae = nn.Linear(10, 10)


class ProtocolPipeline(nn.Module, SupportsModuleOffload):
    """Pipeline with non-standard names, using the protocol."""

    _dit_modules: ClassVar[list[str]] = ["gen_transformer"]
    _encoder_modules: ClassVar[list[str]] = ["mllm", "vision_model"]
    _vae_modules: ClassVar[list[str]] = ["gen_vae"]

    def __init__(self):
        super().__init__()
        self.gen_transformer = nn.Linear(10, 10)
        self.mllm = nn.Linear(10, 10)
        self.vision_model = nn.Linear(10, 10)
        self.gen_vae = nn.Linear(10, 10)
        # Standard name present but NOT declared — should be ignored
        self.transformer = nn.Linear(10, 10)


class MissingAttrPipeline(nn.Module, SupportsModuleOffload):
    """Pipeline that declares a non-existent attribute."""

    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["nonexistent_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]

    def __init__(self):
        super().__init__()
        self.transformer = nn.Linear(10, 10)
        self.vae = nn.Linear(10, 10)


class MissingIntermediatePipeline(nn.Module, SupportsModuleOffload):
    """Pipeline with dotted path referencing non-existent intermediate."""

    _dit_modules: ClassVar[list[str]] = ["nonexistent.transformer"]
    _encoder_modules: ClassVar[list[str]] = []
    _vae_modules: ClassVar[list[str]] = []

    def __init__(self):
        super().__init__()


class NestedPipeline(nn.Module, SupportsModuleOffload):
    """Pipeline with nested modules accessed via dotted paths."""

    _dit_modules: ClassVar[list[str]] = ["pipe.transformer"]
    _encoder_modules: ClassVar[list[str]] = ["pipe.text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]

    def __init__(self):
        super().__init__()
        self.pipe = nn.Module()
        self.pipe.transformer = nn.Linear(10, 10)
        self.pipe.text_encoder = nn.Linear(10, 10)
        self.vae = nn.Linear(10, 10)


class ResidentPipeline(nn.Module, SupportsModuleOffload):
    """Pipeline with resident modules that must stay on GPU."""

    _dit_modules: ClassVar[list[str]] = ["language_model.model"]
    _encoder_modules: ClassVar[list[str]] = []
    _vae_modules: ClassVar[list[str]] = ["vae"]
    _resident_modules: ClassVar[list[str]] = [
        "bagel.time_embedder",
        "bagel.vae2llm",
    ]

    def __init__(self):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.model = nn.Linear(10, 10)
        self.bagel = nn.Module()
        self.bagel.time_embedder = nn.Linear(10, 10)
        self.bagel.vae2llm = nn.Linear(10, 10)
        self.vae = nn.Linear(10, 10)


class MultiVaePipeline(nn.Module, SupportsModuleOffload):
    """Pipeline with multiple VAEs."""

    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae", "audio_vae"]

    def __init__(self):
        super().__init__()
        self.transformer = nn.Linear(10, 10)
        self.text_encoder = nn.Linear(10, 10)
        self.vae = nn.Linear(10, 10)
        self.audio_vae = nn.Linear(10, 10)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFallbackDiscovery:
    """Test the fallback attribute scan (no SupportsModuleOffload)."""

    def test_discovers_standard_attrs(self):
        pipeline = FallbackPipeline()
        result = ModuleDiscovery.discover(pipeline)

        assert not isinstance(pipeline, SupportsModuleOffload)
        assert result.dit_names == ["transformer"]
        assert result.dits[0] is pipeline.transformer
        assert result.encoder_names == ["text_encoder", "text_encoder_2"]
        assert result.vaes[0] is pipeline.vae
        assert result.resident_modules == []

    def test_deduplicates_encoders(self):
        pipeline = DuplicateAttrPipeline()
        result = ModuleDiscovery.discover(pipeline)

        assert len(result.encoders) == 1
        assert result.encoder_names == ["text_encoder"]

    def test_skips_non_module_attr(self):
        pipeline = NonModuleAttrPipeline()
        result = ModuleDiscovery.discover(pipeline)

        assert len(result.encoders) == 0


class TestProtocolDiscovery:
    """Test discovery via SupportsModuleOffload protocol."""

    def test_discovers_declared_attrs_and_ignores_undeclared(self):
        pipeline = ProtocolPipeline()
        result = ModuleDiscovery.discover(pipeline)

        assert isinstance(pipeline, SupportsModuleOffload)
        assert result.dit_names == ["gen_transformer"]
        assert result.encoder_names == ["mllm", "vision_model"]
        assert len(result.vaes) == 1
        # self.transformer exists but is NOT in _dit_modules
        assert "transformer" not in result.dit_names
        # No _resident_modules declared — defaults to empty
        assert result.resident_modules == []

    def test_skips_missing_attr(self):
        pipeline = MissingAttrPipeline()
        result = ModuleDiscovery.discover(pipeline)

        assert len(result.encoders) == 0

    def test_skips_missing_intermediate(self):
        result = ModuleDiscovery.discover(MissingIntermediatePipeline())

        assert len(result.dits) == 0

    def test_dotted_path_resolves_nested_modules(self):
        pipeline = NestedPipeline()
        result = ModuleDiscovery.discover(pipeline)

        assert result.dit_names == ["pipe.transformer"]
        assert result.dits[0] is pipeline.pipe.transformer
        assert result.encoder_names == ["pipe.text_encoder"]
        assert result.encoders[0] is pipeline.pipe.text_encoder
        assert result.vaes[0] is pipeline.vae

    def test_resident_modules(self):
        pipeline = ResidentPipeline()
        result = ModuleDiscovery.discover(pipeline)

        assert result.resident_names == [
            "bagel.time_embedder",
            "bagel.vae2llm",
        ]
        assert result.resident_modules[0] is pipeline.bagel.time_embedder
        assert result.resident_modules[1] is pipeline.bagel.vae2llm
        assert result.dits[0] is pipeline.language_model.model

    def test_multiple_vaes(self):
        pipeline = MultiVaePipeline()
        result = ModuleDiscovery.discover(pipeline)

        assert len(result.vaes) == 2
        assert result.vaes[0] is pipeline.vae
        assert result.vaes[1] is pipeline.audio_vae
