# Contributing to vLLM-omni

Thank you for your interest in contributing to vLLM-omni! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Community](#community)

## Code of Conduct

This project adheres to a Code of Conduct. By participating, you are expected to uphold this code. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for details.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/vllm-omni.git
   cd vllm-omni
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/hsliuustc0106/vllm-omni.git
   ```

## Development Setup

### Prerequisites

- Python 3.8 or higher (3.12 recommended)
- CUDA-capable GPU (for testing)
- Git

### Environment Setup

1. **Create a virtual environment**:
   ```bash
   uv venv --python 3.12 --seed
   source .venv/bin/activate
   ```

2. **Install vLLM dependency** (specific commit):
   ```bash
   git clone https://github.com/vllm-project/vllm.git
   cd vllm
   git checkout 808a7b69df479b6b3a16181711cac7ca28a9b941
   VLLM_USE_PRECOMPILED=1 uv pip install --editable .
   cd ..
   ```

3. **Install vllm-omni in development mode**:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks** (optional but recommended):
   ```bash
   pre-commit install
   ```

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

- **Bug fixes**: Submit fixes for identified issues
- **New features**: Propose and implement new functionality
- **Documentation**: Improve or add documentation
- **Tests**: Add or improve test coverage
- **Examples**: Create new usage examples
- **Model support**: Add support for new multi-modal models
- **Performance improvements**: Optimize existing code

### Finding Issues to Work On

- Check the [issue tracker](https://github.com/hsliuustc0106/vllm-omni/issues)
- Look for issues labeled `good first issue` or `help wanted`
- Comment on the issue to let others know you're working on it

## Coding Standards

### Python Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Use [Black](https://black.readthedocs.io/) for code formatting (line length: 88)
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Use type hints where appropriate
- Maximum line length: 88 characters

### Code Quality Tools

Run these before committing:

```bash
# Format code
black .
isort .

# Check code style
flake8 .

# Type checking
mypy vllm_omni
```

### Documentation

- Add docstrings to all public functions, classes, and modules
- Use Google-style docstrings
- Update relevant documentation when adding features
- Include code examples in docstrings where helpful

Example:
```python
def process_multimodal_input(text: str, image: Optional[Image] = None) -> Dict[str, Any]:
    """Process multi-modal input combining text and optional image.
    
    Args:
        text: Input text string to process
        image: Optional image input for multi-modal processing
        
    Returns:
        Dictionary containing processed embeddings and metadata
        
    Raises:
        ValueError: If text is empty or invalid
        
    Example:
        >>> result = process_multimodal_input("Hello world", image=my_image)
        >>> print(result['text_embedding'].shape)
    """
    pass
```

## Testing Guidelines

### Writing Tests

- Write unit tests for all new functionality
- Ensure tests are isolated and reproducible
- Use descriptive test names that explain what is being tested
- Place tests in the `tests/` directory following the project structure

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_engine.py

# Run with coverage
pytest --cov=vllm_omni --cov-report=html

# Run specific test markers
pytest -m unit  # Only unit tests
pytest -m integration  # Only integration tests
```

### Test Categories

Use pytest markers to categorize tests:

- `@pytest.mark.unit`: Fast, isolated unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.benchmark`: Performance benchmarks
- `@pytest.mark.slow`: Long-running tests

## Pull Request Process

### Before Submitting

1. **Update your fork**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**:
   - Write clean, documented code
   - Add or update tests
   - Update documentation as needed

4. **Run tests and checks**:
   ```bash
   black .
   isort .
   flake8 .
   pytest
   ```

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Brief description of changes"
   ```
   
   Commit message guidelines:
   - Use present tense ("Add feature" not "Added feature")
   - Be concise but descriptive
   - Reference issues if applicable (e.g., "Fix #123")

### Submitting the PR

1. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request** on GitHub:
   - Use a clear, descriptive title
   - Fill out the PR template completely
   - Link related issues
   - Describe what changed and why
   - Include screenshots for UI changes
   - Mark as draft if work in progress

3. **Address Review Comments**:
   - Respond to all review comments
   - Make requested changes
   - Re-request review when ready

### PR Review Process

- At least one maintainer approval required
- All CI checks must pass
- Code coverage should not decrease
- Documentation must be updated
- Changes must be backwards compatible (or breaking changes clearly documented)

## Issue Reporting

### Bug Reports

When reporting bugs, include:

- **Description**: Clear description of the issue
- **Environment**: Python version, OS, GPU details, vLLM version
- **Reproduction steps**: Minimal code to reproduce the issue
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Error messages**: Full error messages and stack traces
- **Additional context**: Any other relevant information

Use the bug report template when creating issues.

### Feature Requests

When proposing features, include:

- **Use case**: Why is this feature needed?
- **Proposed solution**: How should it work?
- **Alternatives**: Other solutions you've considered
- **Additional context**: Examples, mockups, or references

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Pull Requests**: Code review and collaboration

### Getting Help

- Check existing [documentation](docs/)
- Search [closed issues](https://github.com/hsliuustc0106/vllm-omni/issues?q=is%3Aissue+is%3Aclosed)
- Ask questions in GitHub Discussions
- Read the [vLLM documentation](https://docs.vllm.ai/) for base framework

### Recognition

Contributors are recognized in:
- GitHub contributors page
- Release notes
- Project documentation

## License

By contributing to vLLM-omni, you agree that your contributions will be licensed under the Apache License 2.0.

---

**Thank you for contributing to vLLM-omni!** Your efforts help make multi-modal AI more accessible to everyone.
