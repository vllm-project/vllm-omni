# Documentation Build Guide

This directory contains the source files for the vLLM-omni documentation.

## Building Documentation Locally

### Prerequisites

Install documentation dependencies:

```bash
pip install -e ".[docs]"
```

### Build and Serve Documentation

From the project root:

```bash
# Serve documentation locally (auto-reload on changes)
# This starts a local web server at http://127.0.0.1:8000
mkdocs serve

# Build static site (generates HTML files in site/ directory)
mkdocs build
```

When using `mkdocs serve`, the documentation will be automatically available at `http://127.0.0.1:8000`. The server will automatically reload when you make changes to the documentation files.

## Auto-generating API Documentation

The documentation automatically extracts docstrings from the code using mkdocstrings. To ensure your code is documented:

1. Add docstrings to all public classes, functions, and methods
2. Use Google or NumPy style docstrings (both are supported)
3. Rebuild the documentation to see changes

Example docstring:

```python
class OmniLLM:
    """Main entry point for vLLM-omni inference.

    This class provides a high-level interface for running multi-modal
    inference with non-autoregressive models.

    Args:
        model: Model name or path
        stage_configs: Optional stage configurations
        **kwargs: Additional arguments passed to the engine

    Example:
        >>> llm = OmniLLM(model="Qwen/Qwen2.5-Omni")
        >>> outputs = llm.generate(prompts="Hello")
    """
```

## Documentation Structure

```
docs/
├── index.md              # Main documentation page
├── getting_started/      # Getting started guides
├── architecture/        # Architecture documentation
├── api/                 # API reference (auto-generated from code)
├── examples/            # Code examples
└── stylesheets/         # Custom CSS
```

## Publishing Documentation

### GitHub Pages (Recommended)

The documentation is automatically deployed to GitHub Pages using GitHub Actions.

1. **Enable GitHub Pages**:
   - Go to repository `Settings` → `Pages`
   - Set `Source` to `GitHub Actions`
   - Save settings

2. **Push changes**:
   ```bash
   git push origin main
   ```

3. **Documentation will be available at**:
   - `https://vllm-project.github.io/vllm-omni/`

The GitHub Actions workflow (`.github/workflows/docs.yml`) will automatically:
- Build the documentation when you push to `main` branch
- Deploy it to GitHub Pages
- Update the documentation whenever you make changes


### Read the Docs (Alternative)

You can also use Read the Docs for hosting:

1. Sign up at https://readthedocs.org/
2. Import the `vllm-project/vllm-omni` repository
3. Read the Docs will automatically build using `.readthedocs.yml`
4. Documentation will be available at: `https://vllm-omni.readthedocs.io/`

## Configuration

The documentation configuration is in `mkdocs.yml` at the project root.

## Tips

- **API Documentation**: API docs are automatically generated using `mkdocs-api-autonav` and `mkdocstrings`
  - No need to manually create API pages - they're generated automatically
  - Use `[module.name.ClassName][]` syntax for cross-references in Summary pages
- **Code Snippets**: Use `--8<-- "path/to/file.py"` for including code snippets
- **Markdown**: Use Markdown for all documentation (no need for RST)
- **Material Theme**: Use Material theme features like:
  - Admonitions: `!!! note`, `!!! warning`, etc.
  - Code blocks with syntax highlighting
  - Tabs for organizing content
  - Math formulas using `pymdownx.arithmatex`

## Troubleshooting

### Documentation not updating

- Make sure you've saved all files
- If using `mkdocs serve`, it should auto-reload
- Check for syntax errors in `mkdocs.yml`

### API links not working

- Ensure class names match exactly (case-sensitive)
- Check that the module is imported correctly
- Run `mkdocs build --strict` to check for errors

### Build errors

- Check Python version (requires 3.9+)
- Ensure all dependencies are installed: `pip install -e ".[docs]"`
- Check `mkdocs.yml` syntax with `mkdocs build --strict`
