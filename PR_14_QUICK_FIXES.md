# PR #14 - Quick Fix Guide

This document provides ready-to-use code fixes for the critical issues identified in the expert review.

## 🔴 Critical Fixes (Must Address Before Merge)

### 1. Fix Bare Exception Handling (utils.py:113-120)

**Current Code:**
```python
try:
    config = AutoConfig.from_pretrained(args.model)
    if 'Qwen2_5OmniModel' in config.architectures:
        args.legacy_omni_video = False
    else:
        args.legacy_omni_video = True
except:  # ❌ BAD: Catches everything
    args.legacy_omni_video = True
```

**Fixed Code:**
```python
try:
    config = AutoConfig.from_pretrained(args.model)
    if 'Qwen2_5OmniModel' in config.architectures:
        args.legacy_omni_video = False
    else:
        args.legacy_omni_video = True
except (OSError, ValueError, KeyError) as e:
    logger.warning(f"Could not determine model architecture, defaulting to legacy video mode: {e}")
    args.legacy_omni_video = True
```

---

### 2. Fix Unsafe Network Requests (utils.py:90-92)

**Current Code:**
```python
resp = requests.get(video_url)
assert resp.status_code == requests.codes.ok, f"Failed to fetch video from {video_url}, status_code:{resp.status_code}, resp:{resp}"
```

**Fixed Code:**
```python
import logging

logger = logging.getLogger(__name__)

MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500MB limit
REQUEST_TIMEOUT = 30  # 30 seconds

try:
    resp = requests.get(
        video_url, 
        stream=True, 
        timeout=REQUEST_TIMEOUT,
        verify=True
    )
    resp.raise_for_status()  # Better than assert
    
    # Check content length before downloading
    content_length = resp.headers.get('content-length')
    if content_length and int(content_length) > MAX_VIDEO_SIZE:
        raise ValueError(f"Video too large: {content_length} bytes (max: {MAX_VIDEO_SIZE})")
    
    # Write with size check
    downloaded = 0
    for chunk in resp.iter_content(chunk_size=8192):
        downloaded += len(chunk)
        if downloaded > MAX_VIDEO_SIZE:
            raise ValueError(f"Video exceeded size limit during download")
        temp_video_file.write(chunk)
    
    temp_video_file.flush()
    temp_video_file_path = temp_video_file.name
    
except requests.exceptions.RequestException as e:
    logger.error(f"Failed to fetch video from {video_url}: {e}")
    raise
except ValueError as e:
    logger.error(f"Video validation failed: {e}")
    raise
```

---

### 3. Fix Path Traversal Vulnerability (processing_omni.py:97)

**Current Code:**
```python
elif image.startswith("file://"):
    image_obj = Image.open(image[7:])
```

**Fixed Code:**
```python
elif image.startswith("file://"):
    file_path = image[7:]
    
    # Prevent path traversal attacks
    if '..' in file_path:
        raise ValueError(f"Invalid file path: path traversal detected in {file_path}")
    
    # Convert to absolute path and validate
    file_path = os.path.abspath(file_path)
    
    # Ensure file exists and is readable
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")
    
    image_obj = Image.open(file_path)
```

---

### 4. Improve NotImplementedError Message (processing_omni.py:242-244)

**Current Code:**
```python
if 'video_start' in ele or 'video_end' in ele:
    raise NotImplementedError(
        "not support start_pts and end_pts in decord for now.")
```

**Fixed Code:**
```python
if 'video_start' in ele or 'video_end' in ele:
    logger.error(
        "Video trimming (video_start/video_end) is not supported with decord backend.\n"
        "Please use one of these alternatives:\n"
        "  1. Use torchvision backend: add --use-torchvision flag\n"
        "  2. Remove video_start/video_end parameters\n"
        "  3. Set environment variable: FORCE_QWENVL_VIDEO_READER=torchvision"
    )
    raise NotImplementedError(
        "Decord backend does not support video trimming (video_start/video_end). "
        "Use torchvision backend instead or remove trimming parameters."
    )
```

---

### 5. Add File Validation (utils.py:30-39)

**Current Code:**
```python
def resample_wav_to_16khz(input_filepath):
    data, original_sample_rate = sf.read(input_filepath)
    # ...
```

**Fixed Code:**
```python
def resample_wav_to_16khz(input_filepath):
    """Resample audio file to 16kHz.
    
    Args:
        input_filepath: Path to audio file
        
    Returns:
        numpy.ndarray: Resampled audio data at 16kHz
        
    Raises:
        FileNotFoundError: If audio file doesn't exist
        RuntimeError: If file cannot be read or resampled
    """
    if not os.path.exists(input_filepath):
        raise FileNotFoundError(f"Audio file not found: {input_filepath}")
    
    try:
        data, original_sample_rate = sf.read(input_filepath)
        
        # Only use the first channel
        if len(data.shape) > 1:
            data = data[:, 0]
        
        # Resample to 16kHz
        data_resampled = resampy.resample(
            data,
            sr_orig=original_sample_rate,
            sr_new=16000
        )
        return data_resampled
        
    except Exception as e:
        raise RuntimeError(f"Failed to read/resample audio file {input_filepath}: {e}")
```

---

## 🟠 High Priority Improvements

### 6. Cache Processor/Tokenizer (utils.py:110-111)

**Current Code:**
```python
def make_inputs_qwen2_omni(args, messages, ...):
    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
```

**Fixed Code:**
```python
from functools import lru_cache

@lru_cache(maxsize=1)
def get_processor_and_tokenizer(model_name):
    """Cache processor and tokenizer to avoid redundant loading."""
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return processor, tokenizer

def make_inputs_qwen2_omni(args, messages, ...):
    processor, tokenizer = get_processor_and_tokenizer(args.model)
    # ... rest of code
```

---

### 7. Improve Condition Readability (processing_omni.py:170)

**Current Code:**
```python
if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
    raise ValueError(...)
```

**Fixed Code:**
```python
if nframes < FRAME_FACTOR or nframes > total_frames:
    raise ValueError(
        f"Invalid frame count: {nframes}. "
        f"Must be between {FRAME_FACTOR} and {total_frames}. "
        f"Video has {total_frames} frames at {video_fps} fps. "
        f"Try adjusting --fps or --nframes parameters."
    )
```

---

### 8. Fix run.sh to Auto-detect Path

**Current Code:**
```bash
export PYTHONPATH=/path/to/vllm-omni:$PYTHONPATH
```

**Fixed Code:**
```bash
#!/bin/bash

# Auto-detect the vllm-omni root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_OMNI_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Add to PYTHONPATH
export PYTHONPATH="${VLLM_OMNI_ROOT}:$PYTHONPATH"

# Validate that vllm_omni module exists
if [ ! -d "${VLLM_OMNI_ROOT}/vllm_omni" ]; then
    echo "ERROR: vllm_omni module not found at ${VLLM_OMNI_ROOT}/vllm_omni"
    echo "Please run this script from the examples directory or set PYTHONPATH manually"
    exit 1
fi

echo "Using vllm-omni from: ${VLLM_OMNI_ROOT}"

export HF_ENDPOINT=https://hf-mirror.com
python end2end.py --model Qwen/Qwen2.5-Omni-7B \
                  --prompts "Explain the system architecture for a scalable audio generation pipeline. Answer in 15 words." \
                  --voice-type "m02" \
                  --dit-ckpt none \
                  --bigvgan-ckpt none \
                  --output-wav output_audio \
                  --prompt_type text
```

---

### 9. Add Named Constants (end2end.py)

**Add at the top of file:**
```python
# Configuration constants for deterministic generation
DETERMINISTIC_TEMPERATURE = 0.0  # Disable randomness for reproducible results
NUCLEUS_SAMPLING_DISABLED = 1.0  # top_p=1.0 means no nucleus sampling
TOPK_SAMPLING_DISABLED = -1      # top_k=-1 means no top-k filtering
DEFAULT_MAX_TOKENS = 2048
RANDOM_SEED = 42
DEFAULT_REPETITION_PENALTY = 1.1  # Slight penalty to reduce repetition
TALKER_EOS_TOKEN_ID = 8294       # End-of-sequence token for talker stage

# Then use in SamplingParams:
thinker_sampling_params = SamplingParams(
    temperature=DETERMINISTIC_TEMPERATURE,
    top_p=NUCLEUS_SAMPLING_DISABLED,
    top_k=TOPK_SAMPLING_DISABLED,
    max_tokens=DEFAULT_MAX_TOKENS,
    seed=RANDOM_SEED,
    detokenize=True,
    repetition_penalty=DEFAULT_REPETITION_PENALTY,
)
```

---

## 📋 Testing Additions

### 10. Add Basic Unit Tests

Create `tests/test_processing_omni.py`:

```python
import pytest
from examples.offline_inference.qwen_2_5_omni.processing_omni import (
    smart_resize, smart_nframes, round_by_factor, ceil_by_factor, floor_by_factor
)

class TestMathHelpers:
    def test_round_by_factor(self):
        assert round_by_factor(10, 3) == 9
        assert round_by_factor(11, 3) == 12
        assert round_by_factor(12, 3) == 12
    
    def test_ceil_by_factor(self):
        assert ceil_by_factor(10, 3) == 12
        assert ceil_by_factor(12, 3) == 12
    
    def test_floor_by_factor(self):
        assert floor_by_factor(10, 3) == 9
        assert floor_by_factor(12, 3) == 12

class TestSmartResize:
    def test_maintains_aspect_ratio(self):
        h, w = smart_resize(1080, 1920, factor=28)
        original_ratio = 1080 / 1920
        new_ratio = h / w
        assert abs(original_ratio - new_ratio) < 0.1
    
    def test_dimensions_divisible_by_factor(self):
        h, w = smart_resize(1080, 1920, factor=28)
        assert h % 28 == 0
        assert w % 28 == 0
    
    def test_respects_pixel_limits(self):
        h, w = smart_resize(4000, 6000, factor=28, min_pixels=100, max_pixels=1000)
        assert h * w <= 1000
        assert h * w >= 100
    
    def test_extreme_aspect_ratio_raises_error(self):
        with pytest.raises(ValueError, match="aspect ratio"):
            smart_resize(100, 30000, factor=28)  # Ratio > 200

class TestSmartNframes:
    def test_fps_mode(self):
        nframes = smart_nframes(
            {'fps': 2}, 
            total_frames=60, 
            video_fps=30
        )
        assert nframes >= 2  # FRAME_FACTOR
        assert nframes <= 60
        assert nframes % 2 == 0  # Divisible by FRAME_FACTOR
    
    def test_nframes_mode(self):
        nframes = smart_nframes(
            {'nframes': 10}, 
            total_frames=60, 
            video_fps=30
        )
        assert nframes == 10
    
    def test_respects_min_max_frames(self):
        nframes = smart_nframes(
            {'fps': 1, 'min_frames': 20, 'max_frames': 30},
            total_frames=100,
            video_fps=30
        )
        assert 20 <= nframes <= 30
    
    def test_invalid_nframes_raises_error(self):
        with pytest.raises(ValueError):
            smart_nframes({'nframes': 1000}, total_frames=100, video_fps=30)
```

---

## 📖 Documentation Improvements

### 11. Enhanced README.md for the Example

Add to `examples/offline_inference/qwen_2_5_omni/README.md`:

```markdown
## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with at least 16GB VRAM
  - Recommended: V100, A100, RTX 4090, or better
  - Minimum: RTX 3090 (24GB)
- **RAM**: 32GB+ system RAM recommended
- **Disk**: ~50GB free space for model weights and cache

### Software
- **OS**: Linux (Ubuntu 20.04+, CentOS 8+) or macOS
- **CUDA**: 11.8 or higher (for GPU acceleration)
- **Python**: 3.10, 3.11, or 3.12

## Model Information

- **Model Name**: Qwen/Qwen2.5-Omni-7B
- **Download Size**: ~14GB
- **License**: Apache 2.0
- **Architecture**: Thinker-Talker-Codec (3-stage generation)

## Expected Output

Running `bash run.sh` produces:

**Console Output:**
```
Request ID: 0, Text Output: Well, it usually has input modules for data, processing units like neural networks or algorithms, output for generated audio, and scalability through parallel computing or distributed systems.
Request ID: 0, Saved audio to output_audio/output_0.wav
```

**Audio File:**
- Path: `output_audio/output_0.wav`
- Format: WAV, 24kHz sample rate
- Duration: ~8 seconds
- Listen to reference: [output_0.wav](output_audio/output_0.wav)

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce max sequence length
python end2end.py ... --max-model-len 16384
```

### Model Download Fails
```bash
# Use mirror for China users
export HF_ENDPOINT=https://hf-mirror.com

# Or download manually
huggingface-cli download Qwen/Qwen2.5-Omni-7B
```

### Import Errors
```bash
# Verify PYTHONPATH
echo $PYTHONPATH

# Should include path to vllm-omni root directory
```

### Video Processing Errors
```bash
# If decord fails, use torchvision
python end2end.py ... --use-torchvision

# Or set environment variable
export FORCE_QWENVL_VIDEO_READER=torchvision
```
```

---

## ✅ Verification Checklist

After applying fixes, verify:

- [ ] No bare `except:` clauses remain
- [ ] All network requests have timeouts
- [ ] All file operations validate paths and check existence
- [ ] All assertions replaced with proper exception handling
- [ ] Error messages provide actionable guidance
- [ ] Constants extracted and documented
- [ ] Unit tests added and passing
- [ ] Documentation updated with system requirements
- [ ] run.sh auto-detects PYTHONPATH

---

## 🔧 How to Apply These Fixes

1. **Create a new branch:**
   ```bash
   git checkout -b fix/pr14-security-improvements
   ```

2. **Apply fixes in order (Critical → High → Nice-to-have)**

3. **Test each change:**
   ```bash
   # Run unit tests
   pytest tests/

   # Test the example
   cd examples/offline_inference/qwen_2_5_omni
   bash run.sh
   ```

4. **Commit and push:**
   ```bash
   git add -A
   git commit -m "Fix security vulnerabilities and improve error handling"
   git push origin fix/pr14-security-improvements
   ```

---

## Need Help?

If you have questions about any of these fixes:

1. Check the full review: [PR_14_EXPERT_REVIEW.md](PR_14_EXPERT_REVIEW.md)
2. Reference the GitHub review comments
3. Ask in PR discussion thread

Good luck! 🚀
