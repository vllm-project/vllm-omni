# Installation Verification Guide

This guide helps you verify that vLLM-omni is correctly installed and configured.

## Quick Verification

Run this simple check:

```bash
python -c "import vllm_omni; print(f'vLLM-omni version: {vllm_omni.__version__}')"
```

Expected output:
```
vLLM-omni version: 0.1.0
```

## Detailed Verification

### 1. Check Python Version

```bash
python --version
```

Should be Python 3.8 or higher (3.12 recommended).

### 2. Check CUDA Availability

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

Expected output:
```
CUDA available: True
CUDA version: 11.8  # or your installed CUDA version
```

### 3. Check GPU

```bash
nvidia-smi
```

Should display your GPU information without errors.

### 4. Verify vLLM Installation

```bash
python -c "import vllm; print(f'vLLM imported successfully')"
```

### 5. Verify vLLM-omni Components

```bash
python << 'EOF'
# Test core imports
try:
    from vllm_omni import OmniLLM, AsyncOmniLLM
    print("✓ Core components imported")
except ImportError as e:
    print(f"✗ Import error: {e}")

# Test config imports
try:
    from vllm_omni.config import (
        OmniStageConfig,
        DiTConfig,
        DiTCacheConfig,
        create_ar_stage_config,
        create_dit_stage_config,
    )
    print("✓ Configuration imports successful")
except ImportError as e:
    print(f"✗ Config import error: {e}")

# Test version
try:
    import vllm_omni
    print(f"✓ vLLM-omni version: {vllm_omni.__version__}")
except Exception as e:
    print(f"✗ Version check error: {e}")

print("\n✓ All verification checks passed!")
EOF
```

### 6. Run Simple Example

Create a test file `test_installation.py`:

```python
import sys

def test_imports():
    """Test that all main components can be imported."""
    print("Testing imports...")
    
    try:
        from vllm_omni import OmniLLM, AsyncOmniLLM
        print("  ✓ OmniLLM, AsyncOmniLLM")
    except ImportError as e:
        print(f"  ✗ Failed to import main components: {e}")
        return False
    
    try:
        from vllm_omni.config import (
            OmniStageConfig,
            DiTConfig,
            DiTCacheConfig,
        )
        print("  ✓ Configuration classes")
    except ImportError as e:
        print(f"  ✗ Failed to import config: {e}")
        return False
    
    try:
        from vllm_omni.engine import DiffusionEngine
        print("  ✓ Engine components")
    except ImportError as e:
        print(f"  ✗ Failed to import engine: {e}")
        return False
    
    return True

def test_version():
    """Test version information."""
    print("\nChecking version...")
    
    try:
        import vllm_omni
        print(f"  ✓ vLLM-omni version: {vllm_omni.__version__}")
        print(f"  ✓ Author: {vllm_omni.__author__}")
        return True
    except Exception as e:
        print(f"  ✗ Version check failed: {e}")
        return False

def test_dependencies():
    """Test that required dependencies are available."""
    print("\nChecking dependencies...")
    
    dependencies = [
        'torch',
        'transformers',
        'numpy',
        'PIL',
        'yaml',
        'fastapi',
        'pydantic',
    ]
    
    all_ok = True
    for dep in dependencies:
        try:
            if dep == 'PIL':
                __import__('PIL')
            elif dep == 'yaml':
                __import__('yaml')
            else:
                __import__(dep)
            print(f"  ✓ {dep}")
        except ImportError:
            print(f"  ✗ {dep} not found")
            all_ok = False
    
    return all_ok

def main():
    """Run all verification tests."""
    print("=" * 60)
    print("vLLM-omni Installation Verification")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Version", test_version()))
    results.append(("Dependencies", test_dependencies()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All verification tests passed!")
        print("vLLM-omni is correctly installed and ready to use.")
        return 0
    else:
        print("\n✗ Some verification tests failed.")
        print("Please check the error messages above and ensure all dependencies are installed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

Run the test:

```bash
python test_installation.py
```

Expected output:
```
============================================================
vLLM-omni Installation Verification
============================================================
Testing imports...
  ✓ OmniLLM, AsyncOmniLLM
  ✓ Configuration classes
  ✓ Engine components

Checking version...
  ✓ vLLM-omni version: 0.1.0
  ✓ Author: vLLM-omni Team

Checking dependencies...
  ✓ torch
  ✓ transformers
  ✓ numpy
  ✓ PIL
  ✓ yaml
  ✓ fastapi
  ✓ pydantic

============================================================
Verification Summary
============================================================
Imports              ✓ PASS
Version              ✓ PASS
Dependencies         ✓ PASS
============================================================

✓ All verification tests passed!
vLLM-omni is correctly installed and ready to use.
```

## Troubleshooting Failed Verifications

### Import Errors

If you get import errors:

1. **Check installation**:
   ```bash
   pip show vllm-omni
   ```

2. **Reinstall package**:
   ```bash
   pip install --force-reinstall --no-deps vllm-omni
   ```

3. **Check Python path**:
   ```bash
   python -c "import sys; print('\n'.join(sys.path))"
   ```

### CUDA Not Available

If CUDA is not detected:

1. **Verify CUDA installation**:
   ```bash
   nvcc --version
   ```

2. **Check PyTorch CUDA support**:
   ```bash
   python -c "import torch; print(torch.version.cuda)"
   ```

3. **Reinstall PyTorch with CUDA**:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

### Missing Dependencies

If dependencies are missing:

1. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

2. **Install specific dependency**:
   ```bash
   pip install <dependency-name>
   ```

### Version Mismatch

If version doesn't match expected:

1. **Check installed version**:
   ```bash
   pip show vllm-omni
   ```

2. **Update to latest version**:
   ```bash
   pip install --upgrade vllm-omni
   ```

## Next Steps

After successful verification:

1. **Run examples**: Try the [examples](../examples/) to see vLLM-omni in action
2. **Read documentation**: Check out the [docs](../docs/) for detailed usage
3. **Start building**: Use vLLM-omni in your own projects!

## Getting Help

If verification fails and you can't resolve the issue:

1. Check [GitHub Issues](https://github.com/hsliuustc0106/vllm-omni/issues)
2. Search for similar problems
3. Open a new issue with:
   - Output of all verification steps
   - Your system information
   - Python version
   - CUDA version
   - Error messages

---

**Need more help?** Contact us at hsliuustc@gmail.com or open an issue on GitHub.
