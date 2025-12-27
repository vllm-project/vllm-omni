# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for diffusion model CPU offloading."""

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.offload import (
    CPUOffloadMixin,
    apply_cpu_offload,
    move_to_device,
    offload_to_cpu,
)


class MockModule(nn.Module):
    """Simple mock module for testing."""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
    
    def forward(self, x):
        return self.linear(x)


class MockConfig:
    """Mock config for testing offload flags."""
    
    def __init__(
        self,
        text_encoder_cpu_offload=False,
        vae_cpu_offload=False,
        image_encoder_cpu_offload=False,
        dit_cpu_offload=False,
        pin_cpu_memory=True,
    ):
        self.text_encoder_cpu_offload = text_encoder_cpu_offload
        self.vae_cpu_offload = vae_cpu_offload
        self.image_encoder_cpu_offload = image_encoder_cpu_offload
        self.dit_cpu_offload = dit_cpu_offload
        self.pin_cpu_memory = pin_cpu_memory


class MockPipeline(nn.Module):
    """Mock diffusion pipeline for testing."""
    
    def __init__(self):
        super().__init__()
        self.text_encoder = MockModule()
        self.vae = MockModule()
        self.transformer = MockModule()


class TestOffloadToCpu:
    """Test offload_to_cpu function."""
    
    def test_offload_none_module(self):
        """Should handle None gracefully."""
        offload_to_cpu(None)  # Should not raise
    
    def test_offload_moves_to_cpu(self):
        """Should move module to CPU."""
        module = MockModule()
        if torch.cuda.is_available():
            module.cuda()
        
        offload_to_cpu(module, pin_memory=False)
        
        device = next(module.parameters()).device
        assert device.type == "cpu"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_offload_pins_memory(self):
        """Should pin memory when requested."""
        module = MockModule().cuda()
        
        offload_to_cpu(module, pin_memory=True)
        
        for p in module.parameters():
            assert p.data.is_pinned()


class TestMoveToDevice:
    """Test move_to_device function."""
    
    def test_move_none_module(self):
        """Should handle None gracefully."""
        move_to_device(None, torch.device("cpu"))  # Should not raise
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_move_to_cuda(self):
        """Should move module to CUDA."""
        module = MockModule()
        
        move_to_device(module, torch.device("cuda"))
        
        device = next(module.parameters()).device
        assert device.type == "cuda"
    
    def test_move_to_cpu(self):
        """Should move module to CPU."""
        module = MockModule()
        if torch.cuda.is_available():
            module.cuda()
        
        move_to_device(module, torch.device("cpu"))
        
        device = next(module.parameters()).device
        assert device.type == "cpu"


class TestApplyCpuOffload:
    """Test apply_cpu_offload function."""
    
    def test_no_offload_when_disabled(self):
        """Should not offload when flags are False."""
        pipeline = MockPipeline()
        if torch.cuda.is_available():
            pipeline.cuda()
            initial_device = "cuda"
        else:
            initial_device = "cpu"
        
        config = MockConfig()  # All flags False
        apply_cpu_offload(pipeline, config)
        
        # All should stay on initial device
        assert next(pipeline.text_encoder.parameters()).device.type == initial_device
        assert next(pipeline.vae.parameters()).device.type == initial_device
        assert next(pipeline.transformer.parameters()).device.type == initial_device
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_offload_text_encoder(self):
        """Should offload only text_encoder when flag is set."""
        pipeline = MockPipeline().cuda()
        
        config = MockConfig(text_encoder_cpu_offload=True)
        apply_cpu_offload(pipeline, config)
        
        assert next(pipeline.text_encoder.parameters()).device.type == "cpu"
        assert next(pipeline.vae.parameters()).device.type == "cuda"
        assert next(pipeline.transformer.parameters()).device.type == "cuda"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_offload_vae(self):
        """Should offload only VAE when flag is set."""
        pipeline = MockPipeline().cuda()
        
        config = MockConfig(vae_cpu_offload=True)
        apply_cpu_offload(pipeline, config)
        
        assert next(pipeline.text_encoder.parameters()).device.type == "cuda"
        assert next(pipeline.vae.parameters()).device.type == "cpu"
        assert next(pipeline.transformer.parameters()).device.type == "cuda"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_offload_multiple(self):
        """Should offload multiple components."""
        pipeline = MockPipeline().cuda()
        
        config = MockConfig(
            text_encoder_cpu_offload=True,
            vae_cpu_offload=True,
        )
        apply_cpu_offload(pipeline, config)
        
        assert next(pipeline.text_encoder.parameters()).device.type == "cpu"
        assert next(pipeline.vae.parameters()).device.type == "cpu"
        assert next(pipeline.transformer.parameters()).device.type == "cuda"


class TestCPUOffloadMixin:
    """Test CPUOffloadMixin class."""
    
    def test_get_execution_device_cpu(self):
        """Should return CPU when no CUDA."""
        
        class TestPipeline(CPUOffloadMixin, nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = MockModule()
        
        pipeline = TestPipeline()
        device = pipeline._get_execution_device()
        assert device.type == "cpu"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_execution_device_cuda(self):
        """Should return CUDA when transformer is on CUDA."""
        
        class TestPipeline(CPUOffloadMixin, nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = MockModule().cuda()
        
        pipeline = TestPipeline()
        device = pipeline._get_execution_device()
        assert device.type == "cuda"
    
    def test_should_offload_false(self):
        """Should return False when no offload flags."""
        
        class TestPipeline(CPUOffloadMixin, nn.Module):
            def __init__(self):
                super().__init__()
                self.od_config = MockConfig()
        
        pipeline = TestPipeline()
        assert not pipeline._should_offload()
    
    def test_should_offload_true(self):
        """Should return True when any offload flag is set."""
        
        class TestPipeline(CPUOffloadMixin, nn.Module):
            def __init__(self):
                super().__init__()
                self.od_config = MockConfig(text_encoder_cpu_offload=True)
        
        pipeline = TestPipeline()
        assert pipeline._should_offload()

