# Frequently Asked Questions (FAQ)

## General Questions

### What is vLLM-omni?

vLLM-omni is an extension of vLLM that adds support for multi-modal models and non-autoregressive architectures like Diffusion Transformers. It enables inference and serving of models that process text, images, audio, and other modalities.

### How is vLLM-omni different from vLLM?

vLLM focuses on efficient autoregressive text generation. vLLM-omni extends this to:
- Support multi-modal inputs (text, image, audio)
- Handle non-autoregressive architectures (Diffusion models)
- Process multi-stage models (e.g., Thinker-Talker-Codec)
- Generate non-textual outputs (audio, images)

### What models does vLLM-omni support?

Currently supported:
- **Qwen2.5-omni**: Text-to-speech with Thinker-Talker-Codec architecture
- **In development**: Hunyuan-Image 3.0, Qwen-Image

vLLM-omni is designed to be extensible for other multi-modal models.

### Is vLLM-omni production-ready?

vLLM-omni is currently in **alpha** (v0.1.0). It's suitable for:
- Research and experimentation
- Proof-of-concept projects
- Development and testing

For production use, please:
- Thoroughly test with your specific use case
- Monitor performance and stability
- Plan for updates as the project matures

## Installation & Setup

### What are the system requirements?

**Minimum:**
- Python 3.8+
- CUDA-compatible GPU with 8GB+ VRAM
- 20GB free disk space
- Linux OS (Ubuntu 20.04+)

**Recommended:**
- Python 3.12
- NVIDIA GPU with 24GB+ VRAM
- 50GB free disk space
- Ubuntu 22.04

### Why do I need a specific vLLM commit?

vLLM-omni currently depends on specific vLLM features from commit `808a7b69`. We're working on:
- Adapting to stable vLLM v0.11.0
- Reducing version-specific dependencies
- Supporting multiple vLLM versions

### Can I use vLLM-omni on Windows or macOS?

Currently, vLLM-omni is primarily tested on Linux. Windows and macOS support may work but are not officially supported. We recommend:
- **Linux**: Full support (Ubuntu 20.04+)
- **Windows**: Use WSL2 with CUDA support
- **macOS**: Limited/no support (no CUDA)

### Can I use vLLM-omni without a GPU?

vLLM-omni is designed for GPU acceleration and requires CUDA. CPU-only execution is not currently supported, as:
- Model sizes require GPU memory
- Inference would be extremely slow
- vLLM itself requires GPU

## Usage Questions

### How do I load a custom model?

```python
from vllm_omni import OmniLLM

# For HuggingFace models
llm = OmniLLM(model="username/model-name")

# For local models
llm = OmniLLM(model="/path/to/local/model")
```

Ensure your model:
- Has a compatible architecture
- Includes necessary config files
- Is in HuggingFace Transformers format

### How do I process images with text?

```python
from vllm_omni import OmniLLM
from PIL import Image

llm = OmniLLM(model="multimodal-model")
image = Image.open("image.jpg")

outputs = llm.generate(
    prompt="Describe this image",
    image=image
)
```

### Can I use vLLM-omni for real-time applications?

Current limitations:
- Online inference support is in development
- Single request processing (batch support coming)
- Latency depends on model size and complexity

For real-time use:
- Start with smaller models
- Optimize GPU memory usage
- Consider request batching when available

### How do I handle out-of-memory errors?

Reduce memory usage:

```python
llm = OmniLLM(
    model="path/to/model",
    gpu_memory_utilization=0.7,  # Reduce from default 0.9
    max_model_len=2048,          # Reduce context length
    dtype="float16",             # Use FP16 instead of FP32
)
```

Other options:
- Use smaller models
- Process fewer requests simultaneously
- Use tensor parallelism across multiple GPUs

### How do I speed up inference?

Performance tips:

```python
llm = OmniLLM(
    model="path/to/model",
    tensor_parallel_size=2,      # Use multiple GPUs
    dtype="float16",             # Use FP16 precision
    max_num_seqs=32,             # Increase batch size
)
```

Additional:
- Use quantized models when available
- Enable compilation optimizations
- Profile and optimize bottlenecks

## Development Questions

### How can I contribute?

We welcome contributions! See [CONTRIBUTING.md](../CONTRIBUTING.md) for:
- Development setup
- Coding standards
- Pull request process
- Issue reporting

Areas needing help:
- New model support
- Documentation improvements
- Bug fixes
- Performance optimization
- Test coverage

### How do I report a bug?

1. Check [existing issues](https://github.com/hsliuustc0106/vllm-omni/issues)
2. Create a new issue with:
   - Clear description
   - Reproduction steps
   - Error messages
   - System information (Python, CUDA, GPU)
   - Minimal code example

Use the bug report template for best results.

### How do I request a new feature?

1. Check if similar features are requested
2. Open a feature request issue including:
   - Use case and motivation
   - Proposed solution
   - Alternative approaches
   - Impact on existing functionality

### Can I add support for my custom model?

Yes! To add model support:

1. Understand the model architecture
2. Check if similar models are supported
3. Implement model-specific components:
   - Model loader
   - Input processor
   - Output processor
4. Add configuration
5. Write tests
6. Submit PR with documentation

See existing model implementations in `vllm_omni/model_executor/`.

## Technical Questions

### What is the Thinker-Talker-Codec architecture?

A multi-stage processing pipeline:
- **Thinker**: Processes and understands input
- **Talker**: Generates response representation
- **Codec**: Encodes output to target format (e.g., audio)

Used in models like Qwen2.5-omni for text-to-speech.

### What are Diffusion Transformers (DiT)?

Non-autoregressive models that:
- Generate outputs through iterative refinement
- Use diffusion process instead of sequential generation
- Can produce higher quality outputs for some tasks
- Enable parallel generation

vLLM-omni's Diffusion Engine supports DiT models.

### How does multi-modal processing work?

Multi-modal processing pipeline:
1. **Input encoding**: Each modality (text, image, audio) is encoded separately
2. **Feature fusion**: Encoded features are combined
3. **Joint processing**: Unified model processes combined features
4. **Output generation**: Generate desired output format

### What is tensor parallelism?

Splitting model across multiple GPUs:
- Each GPU holds part of model weights
- Computation is distributed
- Enables running larger models
- Improves throughput

Enable with:
```python
llm = OmniLLM(model="path", tensor_parallel_size=4)
```

### How does caching work?

vLLM-omni supports multiple cache types:
- **KV Cache**: For autoregressive generation
- **DiT Cache**: For diffusion models
- Reduces redundant computation
- Improves inference speed

## Troubleshooting

### Why are imports failing?

Common causes:
1. Missing dependencies: `pip install -r requirements.txt`
2. Wrong Python environment: Activate correct venv
3. Installation issue: `pip install -e .` in project directory
4. Python path issue: Check `PYTHONPATH`

### Why is inference slow?

Possible reasons:
1. Large model size - try smaller model
2. CPU bottleneck - check GPU utilization
3. Memory swapping - reduce batch size
4. First run - includes model loading

Profile with:
```bash
nvidia-smi dmon -i 0  # Monitor GPU
```

### Why do I get CUDA errors?

Check:
1. CUDA installation: `nvidia-smi`
2. PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
3. Driver compatibility: Update NVIDIA drivers
4. Memory issues: Reduce model size or batch size

### Model not found error?

Verify:
1. Model path is correct
2. Model files exist (config.json, weights)
3. HuggingFace cache is accessible
4. Internet connection (for downloading)

## Licensing & Legal

### What license is vLLM-omni under?

Apache License 2.0 - permissive open source license allowing:
- Commercial use
- Modification
- Distribution
- Private use

See [LICENSE](../LICENSE) for full terms.

### Can I use vLLM-omni commercially?

Yes! Apache 2.0 allows commercial use. Please:
- Review the license terms
- Ensure model licenses permit your use case
- Follow attribution requirements
- Review dependencies' licenses

### What about model licenses?

vLLM-omni license ≠ model license. Check each model's license:
- Qwen models: Check Qwen license terms
- Other models: Review respective licenses

vLLM-omni is just the inference framework.

## Performance & Scalability

### How many requests can vLLM-omni handle?

Depends on:
- Model size
- GPU memory
- Request complexity
- Hardware specifications

Current state:
- Single request processing
- Batch support in development
- Scale with multiple GPU instances

### Can I deploy vLLM-omni at scale?

Deployment considerations:
- Use container orchestration (Kubernetes)
- Load balancing across multiple instances
- Monitor resource usage
- Plan for model updates
- Implement request queuing

See examples for serving setup.

### What about latency?

Latency factors:
- Model size (larger = slower)
- Input/output length
- Hardware (GPU model)
- Batch size
- Network (for API serving)

Optimize with:
- Smaller models
- GPU optimization
- Request batching
- Caching strategies

## Future Plans

### What's on the roadmap?

Near term (coming soon):
- Stable vLLM v0.11.0 support
- Streaming multi-request support
- Online inference support
- Additional model support

Long term:
- More diffusion models
- Enhanced multi-modal support
- Performance optimizations
- Production features

See [Roadmap in README](../README.md#-roadmap) for details.

### Will there be breaking changes?

As an alpha release (v0.1.x):
- Breaking changes possible
- Documented in CHANGELOG
- Migration guides provided
- Semantic versioning followed

Stable releases (v1.0+):
- Backwards compatibility maintained
- Deprecation warnings
- Clear upgrade paths

### How can I stay updated?

- **Watch** the GitHub repository
- Check [CHANGELOG.md](../CHANGELOG.md)
- Follow release announcements
- Join GitHub Discussions

## Getting Help

### Where can I get help?

1. **Documentation**: Check [docs/](../)
2. **Examples**: Review [examples/](../examples/)
3. **Issues**: Search [GitHub Issues](https://github.com/hsliuustc0106/vllm-omni/issues)
4. **Discussions**: Ask in GitHub Discussions
5. **Email**: hsliuustc@gmail.com

### How do I debug issues?

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Collect information:
- Error messages and stack traces
- System info (`nvidia-smi`, Python version)
- Minimal reproduction code
- Input that triggers issue

### What information should I include in bug reports?

Essential information:
- vLLM-omni version
- vLLM version
- Python version
- CUDA version
- GPU model
- Operating system
- Complete error message
- Minimal reproduction code

See bug report template for full checklist.

---

**Question not answered?** 
- Check [GitHub Discussions](https://github.com/hsliuustc0106/vllm-omni/discussions)
- Open an [issue](https://github.com/hsliuustc0106/vllm-omni/issues)
- Email: hsliuustc@gmail.com
