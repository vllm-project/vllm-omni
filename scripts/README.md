# vLLM-omni Testing Scripts

This directory contains automated testing scripts for vLLM-omni serving functionality.

## Scripts Overview

### 1. `test_serving.sh` - Comprehensive Test Suite
Full-featured testing script that performs complete validation of the vLLM-omni serving functionality.

**Features:**
- Model existence validation
- Environment setup verification
- Import functionality testing
- Server startup and health checks
- Text generation testing
- Performance benchmarking
- API client integration testing
- Comprehensive logging and error handling

**Usage:**
```bash
# Use default model and port
./scripts/test_serving.sh

# Use specific model and port
./scripts/test_serving.sh ./models/Qwen3-0.6B 8000

# Use HuggingFace model
./scripts/test_serving.sh Qwen/Qwen3-0.6B 8001

# Show help
./scripts/test_serving.sh --help
```

**Test Coverage:**
- ✅ Model loading and server startup
- ✅ Health and info endpoints
- ✅ Text generation functionality
- ✅ Performance metrics
- ✅ API client integration
- ✅ All imports working correctly

### 2. `quick_test.sh` - Fast Validation
Lightweight script for quick validation after making changes.

**Features:**
- Fast import testing
- Basic server startup
- Health endpoint validation
- Simple text generation test
- Retry mechanism for reliability

**Usage:**
```bash
# Use default port (8000)
./scripts/quick_test.sh

# Use specific port
./scripts/quick_test.sh 8001
```

**Test Coverage:**
- ✅ Import functionality
- ✅ Server startup
- ✅ Health endpoint
- ✅ Basic text generation

## Prerequisites

1. **Conda Environment**: Ensure `vllm_omni` environment is activated
2. **Model Available**: Qwen3-0.6B model should be available in `./models/Qwen3-0.6B/`
3. **Dependencies**: All vLLM-omni dependencies should be installed

## Environment Setup

```bash
# Activate conda environment
conda activate vllm_omni

# Verify installation
python -c "import vllm_omni; print('vLLM-omni ready')"
```

## Model Setup

The scripts expect the Qwen3-0.6B model to be available. You can:

1. **Use local model**: Place model in `./models/Qwen3-0.6B/`
2. **Use HuggingFace model**: Pass `Qwen/Qwen3-0.6B` as model path
3. **Download model**: Use the `download_models.py` script

```bash
# Download model using the provided script
python scripts/download_models.py
```

## Output and Logging

### Test Results
- **Success**: Green `[SUCCESS]` messages
- **Info**: Blue `[INFO]` messages  
- **Warnings**: Yellow `[WARNING]` messages
- **Errors**: Red `[ERROR]` messages

### Log Files
- `server.log`: Server startup and runtime logs
- `api_client_test.log`: API client test results
- `simple_usage_test.log`: Simple usage test results

## Troubleshooting

### Common Issues

1. **Server fails to start**
   - Check if port is available
   - Verify model path is correct
   - Check conda environment is activated

2. **Health check fails**
   - Server may need more time to start
   - Check server logs for errors
   - Verify port is not blocked

3. **Import errors**
   - Ensure vllm_omni is properly installed
   - Check conda environment activation
   - Verify all dependencies are installed

4. **Model not found**
   - Verify model path exists
   - Check model directory structure
   - Use absolute paths if needed

### Debug Mode

For detailed debugging, run tests with verbose output:

```bash
# Enable debug mode
export VLLM_OMNI_DEBUG=1
./scripts/test_serving.sh
```

## Integration with CI/CD

These scripts can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Test vLLM-omni
  run: |
    conda activate vllm_omni
    ./scripts/test_serving.sh
```

## Performance Expectations

### Quick Test
- **Duration**: ~30-60 seconds
- **Memory**: ~2-4GB RAM
- **CPU**: Moderate usage during model loading

### Comprehensive Test
- **Duration**: ~2-5 minutes
- **Memory**: ~2-4GB RAM
- **CPU**: Moderate usage during model loading and generation

## Contributing

When adding new tests:

1. **Update test_serving.sh** for comprehensive testing
2. **Update quick_test.sh** for essential functionality
3. **Update this README** with new test descriptions
4. **Test both scripts** before committing changes

## Support

For issues with testing scripts:

1. Check the troubleshooting section above
2. Review log files for detailed error information
3. Ensure all prerequisites are met
4. Test with different ports if port conflicts occur
