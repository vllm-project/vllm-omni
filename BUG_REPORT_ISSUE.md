# Bug Report: Missing Setup Documentation for Example Scripts

## Summary
The example scripts in `examples/` require multiple optional dependencies that are not clearly documented, making it difficult for new users to run the examples successfully on first attempt.

## Environment
- Python 3.10+
- vLLM-Omni 0.16.0+
- OS: Linux/Windows/macOS
- GPU: NVIDIA/AMD/CPU

## Steps to Reproduce

1. Clone the repository:
```bash
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
```

2. Install base requirements:
```bash
pip install -e .
```

3. Try to run an example:
```bash
cd examples/offline_inference/qwen3_omni
bash run_single_prompt.sh
```

## Expected Behavior
The example script runs successfully and produces output.

## Actual Behavior
Import errors or missing dependency errors occur:
```
ModuleNotFoundError: No module named 'librosa'
ModuleNotFoundError: No module named 'soundfile'
ModuleNotFoundError: No module named 'PIL'
```

## Root Cause
1. **Missing Documentation**: The examples require optional dependencies (`librosa`, `soundfile`, `imageio`) that are in `pyproject.toml` under `[project.optional-dependencies.dev]` but not clearly referenced in example READMEs
2. **No Quick-Start Guide**: The `examples/offline_inference/qwen3_omni/README.md` doesn't mention installing these dependencies before running
3. **Incomplete Setup Instructions**: Users cannot easily determine which dependencies are needed for each example type

## Solution
Add clear setup instructions to example READMEs that:

1. Document required dependencies per example type:
   - Text examples: basic setup
   - Audio examples: `librosa`, `soundfile`, `resampy`
   - Image examples: `PIL`, `imageio`
   - Video examples: `imageio[ffmpeg]`, `opencv-python`

2. Add a section to each example README:
```markdown
## Required Dependencies

This example requires the following additional packages:

```bash
pip install librosa soundfile resampy
```

## Quick Setup

1. Install vllm-omni with example dependencies:
```bash
pip install -e ".[dev]"
```

2. Run the example:
```bash
python end2end.py --query-type text
```
```

3. Update main README with guidance on running examples

## Impact
- Improves user experience for first-time contributors
- Reduces friction for getting started with vLLM-Omni
- Decreases duplicate issue reports about missing dependencies

## Related Files
- `examples/offline_inference/qwen3_omni/README.md`
- `examples/offline_inference/text_to_audio/README.md`
- `examples/offline_inference/text_to_video/README.md`
- `pyproject.toml` (dev dependencies section)
- Main `README.md`
