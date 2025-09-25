# vLLM-omni Product Requirements Document (PRD)

## 1. Product Overview

### 1.1 Product Name
vLLM-omni: Multi-modality models inference and serving with non-autoregressive structures

### 1.2 Product Vision
Extend vLLM beyond traditional text-based, autoregressive generation to support multi-modality models with non-autoregressive structures and non-textual outputs while maintaining vLLM's proven architecture and performance.

### 1.3 Target Users
- AI researchers working with multimodal models
- ML engineers building production inference systems
- Developers integrating DiT (Diffusion Transformer) models
- Organizations requiring efficient multimodal model serving

## 2. Core Requirements

### 2.1 Functional Requirements

#### 2.1.1 Multi-Stage Processing
- **REQ-001**: Support stage-based model processing where each stage can use different engine types (AR/DiT)
- **REQ-002**: Enable sequential processing through multiple stages with data flow between stages
- **REQ-003**: Support both autoregressive (AR) and diffusion (DiT) model stages

#### 2.1.2 vLLM Compatibility
- **REQ-004**: Maintain full compatibility with vLLM V1 architecture (AsyncLLM and EngineCore patterns)
- **REQ-005**: Support existing vLLM CLI commands with `--omni` flag extension
- **REQ-006**: Reuse vLLM's multiprocess worker architecture for scalability

#### 2.1.3 Multimodal Support
- **REQ-007**: Support text, image, and latent space inputs/outputs
- **REQ-008**: Enable image-to-image, text-to-image, and text-to-text generation
- **REQ-009**: Support hidden state passing between AR and DiT stages

#### 2.1.4 CLI and API
- **REQ-010**: Provide CLI command: `vllm serve Qwen/Qwen2.5-Omni-7B --omni --port 8000`
- **REQ-011**: Support both online (AsyncOmniLLM) and offline (OmniLLM) inference modes
- **REQ-012**: Maintain vLLM's existing API compatibility

### 2.2 Non-Functional Requirements

#### 2.2.1 Performance
- **REQ-013**: Maintain vLLM's inference performance for AR stages
- **REQ-014**: Optimize DiT stage processing with caching mechanisms
- **REQ-015**: Support distributed inference across multiple GPUs

#### 2.2.2 Scalability
- **REQ-016**: Support horizontal scaling through vLLM's worker process pattern
- **REQ-017**: Enable efficient memory management for large multimodal models
- **REQ-018**: Support batch processing for multiple requests

#### 2.2.3 Extensibility
- **REQ-019**: Easy integration of new modalities and model architectures
- **REQ-020**: Pluggable scheduler and executor components
- **REQ-021**: Support for future non-autoregressive model types

## 3. Technical Architecture

### 3.1 Core Components

#### 3.1.1 Entry Points
- **OmniServeCommand**: CLI wrapper that intercepts vLLM commands with `--omni` flag
- **OmniLLM**: Offline inference class supporting multi-stage processing
- **AsyncOmniLLM**: Online inference class with asynchronous processing

#### 3.1.2 Stage Management
- **OmniStageConfig**: Configuration for each processing stage
- **Stage Engine List**: Multiple AsyncLLM instances for each stage
- **Stage I/O Management**: Data flow between stages

#### 3.1.3 Engine Components
- **EngineCore**: Reused from vLLM (no changes needed)
- **OmniDiffusionScheduler**: New scheduler for DiT models
- **DiTCacheManager**: Caching system for DiT optimization
- **MultiprocExecutor**: Reused from vLLM for DiT without diffusers
- **DiffusersPipelineExecutor**: New executor for diffusers integration

#### 3.1.4 Model Runners
- **OmniDiffusionModelRunner**: Handles DiT model execution
- **OmniARModelRunner**: Handles AR model execution with hidden state output
- **ModelRunnerOutput**: Extended to support multimodal outputs via pooler_output

#### 3.1.5 Output Processing
- **MultimodalOutputProcessor**: Handles final multimodal output processing
- **RequestState**: Extended to support pooling outputs
- **Output Handlers**: Type-specific output processing

### 3.2 Data Flow
```
API Server → OmniLLM/AsyncOmniLLM → LLMEngine/AsyncLLM → Engine Core
→ Scheduler (AR/DiT) → Executor (AR/DiT) → Worker (AR/DiT)
→ ModelRunner (AR/DiT) → RequestState → OutputProcessor → Final Output
```

## 4. Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
- Package structure and dependencies
- Basic OmniLLM and AsyncOmniLLM classes
- Stage configuration system
- CLI integration

### Phase 2: Core Components (Weeks 3-4)
- DiT scheduler implementation
- Model runners for AR and DiT
- Basic output processing

### Phase 3: Advanced Features (Weeks 5-6)
- Caching system implementation
- Multimodal output processing
- Request state management

### Phase 4: Integration & Testing (Weeks 7-8)
- End-to-end integration
- Comprehensive testing
- Performance optimization
- Documentation

## 5. Success Criteria

### 5.1 Functional Success
- [ ] Successfully run `vllm serve model --omni` command
- [ ] Process multi-stage AR→DiT pipelines
- [ ] Generate multimodal outputs (text + image)
- [ ] Maintain vLLM API compatibility

### 5.2 Performance Success
- [ ] AR stage performance within 5% of native vLLM
- [ ] DiT stage processing with reasonable latency
- [ ] Memory usage comparable to vLLM for equivalent models

### 5.3 Quality Success
- [ ] 90%+ test coverage
- [ ] All integration tests passing
- [ ] Documentation complete and accurate

## 6. Risk Assessment

### 6.1 Technical Risks
- **High**: vLLM API changes breaking compatibility
- **Medium**: DiT model integration complexity
- **Low**: Performance overhead from multi-stage processing

### 6.2 Mitigation Strategies
- Regular vLLM compatibility testing
- Incremental DiT integration with fallback options
- Performance benchmarking at each stage

## 7. Dependencies

### 7.1 External Dependencies
- vLLM >= 0.10.2
- PyTorch >= 2.7
- Transformers >= 4.30.0
- FastAPI, Uvicorn for API serving
- Ray for distributed computing

### 7.2 Optional Dependencies
- xDiT for DiT acceleration
- Cache-DiT for advanced caching
- Diffusers for pipeline-based DiT models

## 8. Future Roadmap

### 8.1 Short-term (3 months)
- Additional DiT model support
- Performance optimizations
- Enhanced caching strategies

### 8.2 Medium-term (6 months)
- Support for video generation models
- Advanced scheduling strategies
- Multi-GPU DiT optimization

### 8.3 Long-term (12 months)
- Custom model architecture support
- Advanced multimodal fusion
- Production deployment tools
