# Changelog

All notable changes to vLLM-omni will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2024-10-24

### Added
- Initial release of vLLM-omni
- Multi-modal model support (text, image, audio)
- Support for non-autoregressive architectures (Diffusion Transformers)
- Autoregressive Engine for traditional text generation with enhanced KV-caching
- Diffusion Engine for DiT models and iterative generation
- Hybrid Engine for combined AR+DiT processing pipelines
- Qwen2.5-omni model support with offline inference example
- Multi-stage processing architecture
- Modular design extending vLLM core functionality
- Basic API and serving infrastructure
- Example implementations for offline inference
- Documentation for architecture and API design

### Supported Models
- Qwen2.5-omni (Thinker-Talker-Codec structure)

### Known Limitations
- Adaptation to stable vLLM v0.11.0 in progress
- Online inference support under development
- Limited model support (expanding)
- Single request processing (batch processing in development)

[Unreleased]: https://github.com/hsliuustc0106/vllm-omni/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/hsliuustc0106/vllm-omni/releases/tag/v0.1.0
