## [Model] Add GLM-TTS text-to-speech model support

Adds initial support for [GLM-TTS](https://huggingface.co/zai-org/GLM-TTS) in the X2S pipeline.

### Changes
- `vllm_omni/diffusion/models/glm_tts/` - GLM-TTS model implementation
- `vllm_omni/diffusion/registry.py` - Register GLMTTSPipeline
- `tests/e2e/offline_inference/test_glm_tts_model.py` - Tests

### Implementation
- Flow matching DiT model for speech token → mel-spectrogram
- Pipeline following Stable Audio pattern
- Optional Vocos vocoder for waveform output
- Speaker embedding support

Closes #821
Related: #808
