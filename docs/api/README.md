# vLLM-omni API Documentation

This directory contains comprehensive API design documentation for all core modules in vLLM-omni. These templates provide a standardized structure for designing and implementing the core, engine, executor, and worker modules.

## 📋 API Design Templates

### 1. [API Design Template](API_DESIGN_TEMPLATE.md)
**Master template** for creating API documentation for any module in vLLM-omni. Use this as a starting point for new modules.

### 2. [Core Module API](core_module_api.md)
**Core module** provides fundamental scheduling, caching, and resource management functionality.

**Key Components:**
- Request scheduling and prioritization
- DiT cache management for diffusion models
- Resource allocation and coordination
- Inter-module communication

### 3. [Engine Module API](engine_module_api.md)
**Engine module** handles model loading, inference execution, and output processing.

**Key Components:**
- Model loading and initialization
- Inference execution for AR and diffusion models
- Input preprocessing for various modalities
- Output postprocessing and formatting

### 4. [Executor Module API](executor_module_api.md)
**Executor module** coordinates and manages request execution across different engines and workers.

**Key Components:**
- Request routing and load balancing
- Execution pipeline coordination
- Worker management and task distribution
- Error handling and recovery

### 5. [Worker Module API](worker_module_api.md)
**Worker module** provides the actual execution environment for model inference.

**Key Components:**
- Model execution and inference
- GPU resource management
- Request batching and processing
- Performance optimization

## 🎯 How to Use These Templates

### For New Module Development

1. **Start with the Master Template**
   - Copy `API_DESIGN_TEMPLATE.md`
   - Rename to `[module_name]_api.md`
   - Follow the structure exactly

2. **Fill in Module-Specific Details**
   - Update the module overview
   - Define core classes and interfaces
   - Specify public API methods
   - Add configuration options
   - Define error handling strategy
   - Provide usage examples

3. **Review and Validate**
   - Ensure all sections are complete
   - Verify code examples work
   - Check for consistency with other modules
   - Validate configuration options

### For Existing Module Updates

1. **Update API Documentation**
   - Modify existing API files
   - Add new methods and classes
   - Update examples and configuration
   - Maintain backward compatibility notes

2. **Version Control**
   - Track changes in git
   - Use clear commit messages
   - Tag major API changes

## 📚 Template Structure

Each API template follows this standardized structure:

### 1. Module Overview
- **Purpose**: What the module does
- **Responsibilities**: Key responsibilities
- **Dependencies**: Required modules
- **Integration Points**: How it connects with other modules

### 2. Core Classes/Interfaces
- **Base Classes**: Abstract base classes
- **Implementation Classes**: Concrete implementations
- **Data Structures**: Key data models

### 3. Public API Methods
- **Initialization**: Constructor and setup
- **Core Operations**: Main functionality
- **Configuration**: Settings management
- **Lifecycle Management**: Start/stop/cleanup
- **Monitoring**: Status and metrics

### 4. Configuration
- **Configuration Classes**: Dataclasses for settings
- **Required Parameters**: Must-have settings
- **Optional Parameters**: Optional settings with defaults
- **Validation**: Parameter validation rules

### 5. Error Handling
- **Custom Exceptions**: Module-specific errors
- **Error Codes**: Standardized error codes
- **Recovery Strategies**: Error recovery approaches

### 6. Examples
- **Basic Usage**: Simple usage examples
- **Advanced Usage**: Complex scenarios
- **Integration Examples**: Multi-module usage

## 🔧 Implementation Guidelines

### Code Standards
- Use type hints for all methods
- Include comprehensive docstrings
- Follow PEP 8 style guidelines
- Use async/await for I/O operations

### Error Handling
- Define specific exception types
- Provide meaningful error messages
- Include error recovery strategies
- Log errors appropriately

### Configuration
- Use dataclasses for configuration
- Provide sensible defaults
- Include validation methods
- Support environment variables

### Testing
- Write unit tests for all public methods
- Include integration tests
- Test error conditions
- Validate configuration options

## 🚀 Getting Started

### For Developers

1. **Choose a Module**
   - Pick the module you want to work on
   - Read the corresponding API documentation
   - Understand the responsibilities and interfaces

2. **Set Up Development Environment**
   ```bash
   # Clone the repository
   git clone https://github.com/hsliuustc0106/vllm-omni.git
   cd vllm-omni
   
   # Install dependencies
   pip install -r requirements-dev.txt
   
   # Set up pre-commit hooks
   pre-commit install
   ```

3. **Start Implementation**
   - Create the module directory
   - Implement base classes first
   - Add concrete implementations
   - Write tests as you go

### For Contributors

1. **Review API Documentation**
   - Read through all module APIs
   - Understand the overall architecture
   - Identify areas for improvement

2. **Propose Changes**
   - Create issues for API changes
   - Discuss in pull requests
   - Update documentation accordingly

3. **Submit Contributions**
   - Follow the coding standards
   - Include tests for new features
   - Update documentation
   - Submit pull requests

## 📖 Additional Resources

### Architecture Overview
- [vLLM-omni Architecture Design](../architecture/vLLM-omni%20arch%20design%20doc.md)
- [Component Categorization](../../ARCHITECTURE_CATEGORIZATION.md)

### Development Guides
- [Development Setup](../../README.md#development)
- [Testing Guidelines](../../tests/README.md)
- [Contributing Guidelines](../../CONTRIBUTING.md)

### Examples
- [Basic Examples](../../examples/basic/)
- [Advanced Examples](../../examples/advanced/)
- [Multimodal Examples](../../examples/multimodal/)

## 🤝 Contributing

We welcome contributions to improve the API documentation and implementation. Please:

1. **Follow the Template Structure**
   - Use the master template as a guide
   - Maintain consistency across modules
   - Include all required sections

2. **Provide Working Examples**
   - Test all code examples
   - Include both basic and advanced usage
   - Show integration patterns

3. **Keep Documentation Updated**
   - Update docs when APIs change
   - Version control documentation changes
   - Maintain backward compatibility notes

## 📝 Notes

- All API documentation should be kept up-to-date with implementation
- Code examples should be tested and working
- Configuration options should be validated
- Error handling should be comprehensive
- Performance considerations should be documented

For questions or suggestions about the API documentation, please open an issue or start a discussion in the repository.
