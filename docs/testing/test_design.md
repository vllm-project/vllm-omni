# vLLM-omni Test Design Document

## 1. Testing Strategy Overview

### 1.1 Testing Pyramid
```
                    E2E Tests (10%)
                   /              \
              Integration Tests (30%)
             /                      \
        Unit Tests (60%)
```

### 1.2 Test Categories
- **Unit Tests**: Individual component testing with mocked dependencies
- **Integration Tests**: Component interaction testing with real vLLM integration
- **End-to-End Tests**: Full pipeline testing with real models
- **Performance Tests**: Benchmarking and profiling
- **Compatibility Tests**: vLLM version compatibility validation

## 2. Test Structure

```
tests/
├── __init__.py
├── conftest.py                 # Shared fixtures and configuration
├── unit/                       # Unit tests (60%)
│   ├── __init__.py
│   ├── test_config/           # Configuration testing
│   │   ├── test_stage_config.py
│   │   ├── test_dit_config.py
│   │   └── test_cache_config.py
│   ├── test_core/             # Core component testing
│   │   ├── test_omni_llm.py
│   │   ├── test_async_omni_llm.py
│   │   ├── test_stage_manager.py
│   │   └── test_dit_cache_manager.py
│   ├── test_scheduler/        # Scheduler testing
│   │   ├── test_diffusion_scheduler.py
│   │   └── test_scheduler_interface.py
│   ├── test_executor/         # Executor testing
│   │   ├── test_base_executor.py
│   │   └── test_diffusers_executor.py
│   ├── test_model_executor/   # Model runner testing
│   │   ├── test_ar_model_runner.py
│   │   └── test_dit_model_runner.py
│   ├── test_engine/           # Engine testing
│   │   ├── test_output_processor.py
│   │   └── test_multimodal_processor.py
│   ├── test_worker/           # Worker testing
│   │   └── test_omni_worker.py
│   └── test_utils/            # Utility testing
│       ├── test_multimodal.py
│       └── test_vae.py
├── integration/               # Integration tests (30%)
│   ├── __init__.py
│   ├── test_vllm_integration.py
│   ├── test_stage_processing.py
│   ├── test_cli_integration.py
│   └── test_api_compatibility.py
├── e2e/                       # End-to-end tests (10%)
│   ├── __init__.py
│   ├── test_full_pipeline.py
│   ├── test_multimodal_generation.py
│   └── test_performance.py
├── benchmarks/                # Performance benchmarks
│   ├── __init__.py
│   ├── test_memory_usage.py
│   ├── test_latency.py
│   └── test_throughput.py
└── fixtures/                  # Test data and fixtures
    ├── __init__.py
    ├── sample_models/
    ├── test_images/
    └── test_configs/
```

## 3. Unit Test Specifications

### 3.1 Configuration Tests

#### test_stage_config.py
```python
import pytest
from vllm_omni.config.stage_config import OmniStageConfig, DiTConfig, DiTCacheConfig

class TestOmniStageConfig:
    def test_ar_stage_config_creation(self):
        """Test AR stage configuration creation"""
        config = OmniStageConfig(
            stage_id=0,
            engine_type="AR",
            model_path="test-model",
            input_modalities=["text"],
            output_modalities=["text"]
        )
        assert config.engine_type == "AR"
        assert config.stage_id == 0
    
    def test_dit_stage_config_creation(self):
        """Test DiT stage configuration creation"""
        dit_config = DiTConfig(
            model_type="dit",
            scheduler_type="ddpm",
            num_inference_steps=50
        )
        config = OmniStageConfig(
            stage_id=1,
            engine_type="DiT",
            model_path="test-dit-model",
            input_modalities=["text"],
            output_modalities=["image"],
            dit_config=dit_config
        )
        assert config.engine_type == "DiT"
        assert config.dit_config is not None
    
    def test_invalid_engine_type(self):
        """Test validation of engine type"""
        with pytest.raises(ValueError):
            OmniStageConfig(
                stage_id=0,
                engine_type="INVALID",
                model_path="test-model",
                input_modalities=["text"],
                output_modalities=["text"]
            )
```

#### test_dit_config.py
```python
import pytest
from vllm_omni.config.dit_config import DiTConfig

class TestDiTConfig:
    def test_dit_config_defaults(self):
        """Test DiT configuration with defaults"""
        config = DiTConfig(
            model_type="dit",
            scheduler_type="ddpm",
            num_inference_steps=50
        )
        assert config.use_diffusers is False
        assert config.diffusers_pipeline is None
    
    def test_diffusers_config(self):
        """Test DiT configuration with diffusers"""
        config = DiTConfig(
            model_type="dit",
            scheduler_type="ddpm",
            num_inference_steps=50,
            use_diffusers=True,
            diffusers_pipeline="stable-diffusion"
        )
        assert config.use_diffusers is True
        assert config.diffusers_pipeline == "stable-diffusion"
```

### 3.2 Core Component Tests

#### test_omni_llm.py
```python
import pytest
from unittest.mock import Mock, patch
from vllm_omni.core.omni_llm import OmniLLM
from vllm_omni.config.stage_config import OmniStageConfig

class TestOmniLLM:
    @pytest.fixture
    def mock_stage_configs(self):
        """Create mock stage configurations"""
        return [
            OmniStageConfig(
                stage_id=0,
                engine_type="AR",
                model_path="test-ar-model",
                input_modalities=["text"],
                output_modalities=["text"]
            ),
            OmniStageConfig(
                stage_id=1,
                engine_type="DiT",
                model_path="test-dit-model",
                input_modalities=["text"],
                output_modalities=["image"]
            )
        ]
    
    @patch('vllm_omni.core.omni_llm.LLMEngine')
    def test_omni_llm_initialization(self, mock_llm_engine, mock_stage_configs):
        """Test OmniLLM initialization"""
        omni_llm = OmniLLM(mock_stage_configs)
        assert len(omni_llm.engine_list) == 2
        assert omni_llm.stage_configs == mock_stage_configs
    
    @patch('vllm_omni.core.omni_llm.LLMEngine')
    def test_stage_engine_creation(self, mock_llm_engine, mock_stage_configs):
        """Test stage engine creation"""
        omni_llm = OmniLLM(mock_stage_configs)
        # Verify engines are created for each stage
        assert mock_llm_engine.from_vllm_config.call_count == 2
    
    def test_process_stage_inputs_ar(self, mock_stage_configs):
        """Test AR stage input processing"""
        omni_llm = OmniLLM(mock_stage_configs)
        ar_config = mock_stage_configs[0]
        stage_args = {"prompt": "test prompt"}
        
        result = omni_llm._process_stage_inputs(ar_config, stage_args, None)
        assert "prompt" in result
    
    def test_process_stage_inputs_dit(self, mock_stage_configs):
        """Test DiT stage input processing"""
        omni_llm = OmniLLM(mock_stage_configs)
        dit_config = mock_stage_configs[1]
        stage_args = {"image": "test_image.jpg"}
        
        with patch.object(omni_llm, 'vae') as mock_vae:
            mock_vae.encode.return_value = "encoded_image"
            result = omni_llm._process_stage_inputs(dit_config, stage_args, None)
            assert "encoded_image" in result
```

#### test_async_omni_llm.py
```python
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from vllm_omni.core.omni_llm import AsyncOmniLLM

class TestAsyncOmniLLM:
    @pytest.fixture
    def mock_stage_configs(self):
        """Create mock stage configurations"""
        return [
            OmniStageConfig(
                stage_id=0,
                engine_type="AR",
                model_path="test-ar-model",
                input_modalities=["text"],
                output_modalities=["text"]
            )
        ]
    
    @pytest.mark.asyncio
    async def test_async_generation(self, mock_stage_configs):
        """Test async generation"""
        with patch('vllm_omni.core.omni_llm.AsyncLLM') as mock_async_llm:
            async_omni_llm = AsyncOmniLLM(mock_stage_configs)
            
            # Mock the generate_async method
            mock_async_llm.return_value.generate_async = AsyncMock(return_value=[])
            
            result = await async_omni_llm.generate_async([{"prompt": "test"}])
            assert result == []
```

### 3.3 Scheduler Tests

#### test_diffusion_scheduler.py
```python
import pytest
from unittest.mock import Mock, patch
from vllm_omni.core.sched.diffusion_scheduler import OmniDiffusionScheduler

class TestOmniDiffusionScheduler:
    @pytest.fixture
    def mock_configs(self):
        """Create mock configurations"""
        return {
            'vllm_config': Mock(),
            'kv_cache_config': Mock(),
            'dit_cache_config': Mock(),
            'structured_output_manager': Mock(),
            'mm_registry': Mock(),
            'include_finished_set': False,
            'log_stats': False
        }
    
    def test_scheduler_initialization(self, mock_configs):
        """Test scheduler initialization"""
        scheduler = OmniDiffusionScheduler(**mock_configs)
        assert scheduler.dit_cache_manager is not None
    
    def test_schedule_method(self, mock_configs):
        """Test scheduling method"""
        scheduler = OmniDiffusionScheduler(**mock_configs)
        
        with patch.object(scheduler, 'dit_cache_manager') as mock_cache:
            mock_cache.allocate_cache.return_value = Mock()
            result = scheduler.schedule()
            # Verify scheduling logic
            assert result is not None
```

### 3.4 Model Runner Tests

#### test_ar_model_runner.py
```python
import pytest
import torch
from unittest.mock import Mock, patch
from vllm_omni.model_executor.ar_model_runner import OmniARModelRunner

class TestOmniARModelRunner:
    @pytest.fixture
    def mock_runner(self):
        """Create mock model runner"""
        with patch('vllm_omni.model_executor.ar_model_runner.GPUModelRunner.__init__'):
            runner = OmniARModelRunner(Mock(), Mock(), Mock())
            return runner
    
    def test_execute_model_ar(self, mock_runner):
        """Test AR model execution"""
        mock_scheduler_output = Mock()
        mock_scheduler_output.req_ids = ["req1"]
        mock_scheduler_output.req_id_to_index = {"req1": 0}
        
        with patch.object(mock_runner, 'model') as mock_model:
            mock_model.forward.return_value = (torch.randn(1, 10, 768), None)
            
            result = mock_runner.execute_model(mock_scheduler_output)
            
            assert result.req_ids == ["req1"]
            assert len(result.pooler_output) > 0  # Hidden states
```

#### test_dit_model_runner.py
```python
import pytest
import torch
from unittest.mock import Mock, patch
from vllm_omni.model_executor.dit_model_runner import OmniDiffusionModelRunner

class TestOmniDiffusionModelRunner:
    @pytest.fixture
    def mock_runner(self):
        """Create mock DiT model runner"""
        with patch('vllm_omni.model_executor.dit_model_runner.GPUModelRunner.__init__'):
            runner = OmniDiffusionModelRunner(Mock(), Mock(), Mock())
            return runner
    
    def test_execute_model_dit(self, mock_runner):
        """Test DiT model execution"""
        mock_scheduler_output = Mock()
        mock_scheduler_output.req_ids = ["req1"]
        mock_scheduler_output.req_id_to_index = {"req1": 0}
        
        with patch.object(mock_runner, 'model') as mock_model:
            mock_model.forward.return_value = torch.randn(1, 3, 512, 512)
            
            result = mock_runner.execute_model(mock_scheduler_output)
            
            assert result.req_ids == ["req1"]
            assert len(result.pooler_output) > 0  # DiT output tensors
```

## 4. Integration Tests

### 4.1 vLLM Integration Tests

#### test_vllm_integration.py
```python
import pytest
from vllm_omni.core.omni_llm import OmniLLM
from vllm_omni.config.stage_config import OmniStageConfig

class TestVLLMIntegration:
    @pytest.mark.integration
    def test_vllm_engine_creation(self):
        """Test creation of vLLM engines"""
        stage_configs = [
            OmniStageConfig(
                stage_id=0,
                engine_type="AR",
                model_path="microsoft/DialoGPT-small",  # Small test model
                input_modalities=["text"],
                output_modalities=["text"]
            )
        ]
        
        omni_llm = OmniLLM(stage_configs)
        assert len(omni_llm.engine_list) == 1
        assert omni_llm.engine_list[0] is not None
    
    @pytest.mark.integration
    def test_stage_processing_integration(self):
        """Test integration between stages"""
        # Test with real vLLM components
        pass
```

### 4.2 CLI Integration Tests

#### test_cli_integration.py
```python
import pytest
import subprocess
import sys
from vllm_omni.entrypoints.cli.main import main

class TestCLIIntegration:
    def test_omni_flag_detection(self):
        """Test --omni flag detection"""
        original_argv = sys.argv
        try:
            sys.argv = ["vllm", "serve", "test-model", "--omni"]
            # Test that omni command is triggered
            with patch('vllm_omni.entrypoints.omni.OmniServeCommand') as mock_serve:
                main()
                mock_serve.assert_called_once()
        finally:
            sys.argv = original_argv
    
    def test_forward_to_vllm(self):
        """Test forwarding to vLLM when --omni not present"""
        original_argv = sys.argv
        try:
            sys.argv = ["vllm", "serve", "test-model"]
            with patch('vllm.entrypoints.cli.main.main') as mock_vllm_main:
                main()
                mock_vllm_main.assert_called_once()
        finally:
            sys.argv = original_argv
```

## 5. End-to-End Tests

### 5.1 Full Pipeline Tests

#### test_full_pipeline.py
```python
import pytest
from vllm_omni.core.omni_llm import OmniLLM
from vllm_omni.config.stage_config import OmniStageConfig, DiTConfig

class TestFullPipeline:
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_ar_to_dit_pipeline(self):
        """Test complete AR to DiT pipeline"""
        stage_configs = [
            OmniStageConfig(
                stage_id=0,
                engine_type="AR",
                model_path="microsoft/DialoGPT-small",
                input_modalities=["text"],
                output_modalities=["text"]
            ),
            OmniStageConfig(
                stage_id=1,
                engine_type="DiT",
                model_path="stabilityai/stable-diffusion-2-1",
                input_modalities=["text"],
                output_modalities=["image"],
                dit_config=DiTConfig(
                    model_type="dit",
                    scheduler_type="ddpm",
                    num_inference_steps=10  # Reduced for testing
                )
            )
        ]
        
        omni_llm = OmniLLM(stage_configs)
        
        stage_args = [
            {"prompt": "A beautiful landscape"},
            {"prompt": "A beautiful landscape"}
        ]
        
        result = omni_llm.generate(stage_args)
        assert result is not None
        assert len(result) > 0
```

### 5.2 Multimodal Generation Tests

#### test_multimodal_generation.py
```python
import pytest
from vllm_omni.core.omni_llm import OmniLLM

class TestMultimodalGeneration:
    @pytest.mark.e2e
    def test_text_to_image_generation(self):
        """Test text to image generation"""
        # Test implementation
        pass
    
    @pytest.mark.e2e
    def test_image_to_text_generation(self):
        """Test image to text generation"""
        # Test implementation
        pass
    
    @pytest.mark.e2e
    def test_text_and_image_generation(self):
        """Test combined text and image generation"""
        # Test implementation
        pass
```

## 6. Performance Tests

### 6.1 Memory Usage Tests

#### test_memory_usage.py
```python
import pytest
import psutil
import torch
from vllm_omni.core.omni_llm import OmniLLM

class TestMemoryUsage:
    @pytest.mark.benchmark
    def test_memory_usage_ar_stage(self):
        """Test memory usage for AR stage"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Run AR stage
        # ... test implementation
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Assert memory usage is reasonable
        assert memory_increase < 1024 * 1024 * 1024  # Less than 1GB
    
    @pytest.mark.benchmark
    def test_memory_usage_dit_stage(self):
        """Test memory usage for DiT stage"""
        # Test implementation
        pass
    
    @pytest.mark.benchmark
    def test_cache_memory_management(self):
        """Test DiT cache memory management"""
        # Test implementation
        pass
```

### 6.2 Latency Tests

#### test_latency.py
```python
import pytest
import time
from vllm_omni.core.omni_llm import OmniLLM

class TestLatency:
    @pytest.mark.benchmark
    def test_ar_stage_latency(self):
        """Test AR stage latency"""
        start_time = time.time()
        
        # Run AR stage
        # ... test implementation
        
        end_time = time.time()
        latency = end_time - start_time
        
        # Assert latency is reasonable
        assert latency < 5.0  # Less than 5 seconds
    
    @pytest.mark.benchmark
    def test_dit_stage_latency(self):
        """Test DiT stage latency"""
        # Test implementation
        pass
    
    @pytest.mark.benchmark
    def test_end_to_end_latency(self):
        """Test end-to-end pipeline latency"""
        # Test implementation
        pass
```

## 7. Test Configuration

### 7.1 pytest.ini
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --strict-config
    --cov=vllm_omni
    --cov-report=term-missing
    --cov-report=html
    --cov-report=xml
    --tb=short
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    benchmark: Performance benchmark tests
    slow: Slow running tests
```

### 7.2 conftest.py
```python
import pytest
import torch
from unittest.mock import Mock
from vllm_omni.config.stage_config import OmniStageConfig, DiTConfig

@pytest.fixture(scope="session")
def device():
    """Get available device for testing"""
    return "cuda" if torch.cuda.is_available() else "cpu"

@pytest.fixture
def sample_stage_configs():
    """Sample stage configurations for testing"""
    return [
        OmniStageConfig(
            stage_id=0,
            engine_type="AR",
            model_path="test-ar-model",
            input_modalities=["text"],
            output_modalities=["text"]
        ),
        OmniStageConfig(
            stage_id=1,
            engine_type="DiT",
            model_path="test-dit-model",
            input_modalities=["text"],
            output_modalities=["image"],
            dit_config=DiTConfig(
                model_type="dit",
                scheduler_type="ddpm",
                num_inference_steps=10
            )
        )
    ]

@pytest.fixture
def mock_vllm_config():
    """Mock vLLM configuration"""
    config = Mock()
    config.model = "test-model"
    config.tensor_parallel_size = 1
    config.pipeline_parallel_size = 1
    return config
```

## 8. Test Execution

### 8.1 Running Tests
```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m e2e
pytest -m benchmark

# Run with coverage
pytest --cov=vllm_omni --cov-report=html

# Run specific test file
pytest tests/unit/test_omni_llm.py

# Run with verbose output
pytest -v

# Run in parallel
pytest -n auto
```

### 8.2 Continuous Integration
```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
    - name: Run tests
      run: |
        pytest --cov=vllm_omni --cov-report=xml
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## 9. Test Data Management

### 9.1 Fixtures Directory
```
tests/fixtures/
├── sample_models/
│   ├── ar_model/
│   └── dit_model/
├── test_images/
│   ├── landscape.jpg
│   └── portrait.png
├── test_configs/
│   ├── ar_stage_config.json
│   └── dit_stage_config.json
└── expected_outputs/
    ├── ar_output.json
    └── dit_output.json
```

### 9.2 Test Data Loading
```python
import json
import os
from pathlib import Path

def load_test_config(config_name: str) -> dict:
    """Load test configuration from fixtures"""
    config_path = Path(__file__).parent / "fixtures" / "test_configs" / f"{config_name}.json"
    with open(config_path) as f:
        return json.load(f)

def load_test_image(image_name: str) -> str:
    """Load test image path from fixtures"""
    return str(Path(__file__).parent / "fixtures" / "test_images" / image_name)
```

## 10. Test Quality Metrics

### 10.1 Coverage Targets
- **Overall Coverage**: > 90%
- **Unit Tests**: > 95%
- **Integration Tests**: > 80%
- **Critical Paths**: 100%

### 10.2 Performance Targets
- **Unit Test Execution**: < 30 seconds
- **Integration Test Execution**: < 5 minutes
- **E2E Test Execution**: < 15 minutes
- **Memory Usage**: < 8GB for full test suite

### 10.3 Quality Gates
- All unit tests must pass
- Integration tests must pass
- Coverage threshold must be met
- No critical security vulnerabilities
- Performance regression tests must pass
