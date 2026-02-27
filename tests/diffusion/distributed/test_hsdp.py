# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for HSDP (Hybrid Sharded Data Parallel) configuration and utilities.

These tests verify HSDP configuration logic without requiring a distributed environment.
"""

import pytest
import torch.nn as nn

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.distributed.hsdp import HSDPInferenceConfig

pytestmark = [pytest.mark.diffusion, pytest.mark.parallel, pytest.mark.cpu]


class TestHSDPInferenceConfig:
    """Tests for HSDPInferenceConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = HSDPInferenceConfig()
        assert config.enabled is False
        assert config.hsdp_replicate_size == 1
        assert config.hsdp_shard_size == -1

    def test_custom_values(self):
        """Test custom configuration values."""
        config = HSDPInferenceConfig(
            enabled=True,
            hsdp_replicate_size=2,
            hsdp_shard_size=4,
        )
        assert config.enabled is True
        assert config.hsdp_replicate_size == 2
        assert config.hsdp_shard_size == 4


class TestDiffusionParallelConfigHSDP:
    """Tests for HSDP settings in DiffusionParallelConfig."""

    def test_hsdp_disabled_by_default(self):
        """HSDP should be disabled by default."""
        config = DiffusionParallelConfig()
        assert config.use_hsdp is False
        assert config.hsdp_shard_size == -1
        assert config.hsdp_replicate_size == 1

    def test_hsdp_auto_shard_size(self):
        """Test auto-calculation of hsdp_shard_size when use_hsdp=True."""
        # ulysses_degree=4 -> world_size=4
        # hsdp_shard_size should be auto-calculated as 4 // 1 = 4
        config = DiffusionParallelConfig(
            ulysses_degree=4,
            use_hsdp=True,
        )
        assert config.world_size == 4
        assert config.hsdp_shard_size == 4
        assert config.hsdp_replicate_size == 1

    def test_hsdp_auto_shard_size_fails_standalone(self):
        """Test that auto-calculate fails when other parallelism is all 1."""
        # When all other parallelism is 1, cannot auto-calculate
        # User must specify hsdp_shard_size explicitly
        with pytest.raises(ValueError, match="Cannot auto-calculate hsdp_shard_size"):
            DiffusionParallelConfig(
                use_hsdp=True,
                # All other parallelism defaults to 1
            )

    def test_hsdp_standalone_mode(self):
        """Test standalone HSDP (HSDP without other parallelism)."""
        # Standalone HSDP: all other parallelism=1, explicit shard_size
        config = DiffusionParallelConfig(
            use_hsdp=True,
            hsdp_shard_size=4,  # Explicit shard size
            hsdp_replicate_size=1,
        )
        # world_size should be determined by HSDP
        assert config.world_size == 4
        assert config.hsdp_shard_size == 4
        assert config.hsdp_replicate_size == 1

    def test_hsdp_standalone_with_replicate(self):
        """Test standalone HSDP with replication."""
        config = DiffusionParallelConfig(
            use_hsdp=True,
            hsdp_shard_size=4,
            hsdp_replicate_size=2,
        )
        # world_size = shard_size * replicate_size
        assert config.world_size == 8
        assert config.hsdp_shard_size == 4
        assert config.hsdp_replicate_size == 2

    def test_hsdp_with_replicate(self):
        """Test HSDP with replication (hybrid mode) combined with other parallelism."""
        # world_size=8, replicate=2 -> shard_size should be 4
        config = DiffusionParallelConfig(
            ulysses_degree=8,
            use_hsdp=True,
            hsdp_replicate_size=2,
        )
        assert config.world_size == 8
        assert config.hsdp_shard_size == 4
        assert config.hsdp_replicate_size == 2

    def test_hsdp_explicit_shard_size_valid(self):
        """Test explicit hsdp_shard_size that matches world_size."""
        config = DiffusionParallelConfig(
            ulysses_degree=4,
            use_hsdp=True,
            hsdp_shard_size=4,
            hsdp_replicate_size=1,
        )
        assert config.hsdp_shard_size == 4

    def test_hsdp_explicit_shard_size_invalid(self):
        """Test that invalid HSDP dimensions raise an error when combined with other parallelism."""
        with pytest.raises(ValueError, match="HSDP dimensions"):
            DiffusionParallelConfig(
                ulysses_degree=4,  # world_size=4
                use_hsdp=True,
                hsdp_shard_size=3,  # 1 * 3 != 4
                hsdp_replicate_size=1,
            )

    def test_hsdp_replicate_size_exceeds_world_size(self):
        """Test that replicate_size > world_size raises an error."""
        with pytest.raises(ValueError, match="replicate_size.*must evenly divide world_size"):
            DiffusionParallelConfig(
                ulysses_degree=4,  # world_size=4
                use_hsdp=True,
                hsdp_replicate_size=8,  # 8 > 4, invalid
            )

    def test_hsdp_combined_world_size(self):
        """Test that combined HSDP matches other parallelism world_size."""
        config_no_hsdp = DiffusionParallelConfig(ulysses_degree=4)
        config_with_hsdp = DiffusionParallelConfig(
            ulysses_degree=4,
            use_hsdp=True,
            hsdp_shard_size=4,
        )
        # When combined with other parallelism, world_size should match
        assert config_no_hsdp.world_size == config_with_hsdp.world_size == 4

    def test_hsdp_standalone_world_size(self):
        """Test that standalone HSDP determines world_size."""
        config_hsdp = DiffusionParallelConfig(
            use_hsdp=True,
            hsdp_shard_size=8,
        )
        # Standalone HSDP: world_size is determined by HSDP
        assert config_hsdp.world_size == 8

    def test_hsdp_cannot_use_with_tp(self):
        """Test that HSDP and Tensor Parallelism cannot be used together."""
        with pytest.raises(ValueError, match="cannot be used with Tensor Parallelism"):
            DiffusionParallelConfig(
                tensor_parallel_size=2,
                use_hsdp=True,
                hsdp_shard_size=4,
            )

    def test_from_dict_with_hsdp(self):
        """Test creating config from dict with HSDP settings."""
        config = DiffusionParallelConfig.from_dict(
            {
                "ulysses_degree": 4,
                "use_hsdp": True,
                "hsdp_replicate_size": 2,
            }
        )
        assert config.use_hsdp is True
        assert config.hsdp_replicate_size == 2
        assert config.hsdp_shard_size == 2  # auto: 4 // 2


class TestHSDPShardConditions:
    """Tests for _hsdp_shard_conditions matching logic."""

    @staticmethod
    def _is_transformer_block(name: str, module: nn.Module) -> bool:
        """Example shard condition matching transformer blocks."""
        return "blocks" in name and name.split(".")[-1].isdigit()

    def test_condition_matches_blocks(self):
        """Test that condition matches transformer block patterns."""
        cond = self._is_transformer_block
        # Should match
        assert cond("blocks.0", nn.Linear(10, 10)) is True
        assert cond("blocks.15", nn.Linear(10, 10)) is True
        assert cond("transformer.blocks.0", nn.Linear(10, 10)) is True
        # Should not match
        assert cond("blocks", nn.Linear(10, 10)) is False
        assert cond("blocks.norm", nn.Linear(10, 10)) is False
        assert cond("embeddings", nn.Linear(10, 10)) is False

    def test_model_with_shard_conditions(self):
        """Test model with _hsdp_shard_conditions attribute."""

        class MockModel(nn.Module):
            @staticmethod
            def _is_block(name: str, module: nn.Module) -> bool:
                return name.startswith("blocks.") and name.split(".")[-1].isdigit()

            _hsdp_shard_conditions = [_is_block]

            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([nn.Linear(10, 10) for _ in range(2)])

        model = MockModel()
        conditions = getattr(model, "_hsdp_shard_conditions", None)
        assert conditions is not None
        assert len(conditions) == 1

        # Verify conditions work on actual model modules
        matched = []
        for name, module in model.named_modules():
            if any(cond(name, module) for cond in conditions):
                matched.append(name)
        assert "blocks.0" in matched
        assert "blocks.1" in matched
