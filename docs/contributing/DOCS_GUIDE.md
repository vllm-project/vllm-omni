# Documentation Build Guide

The `docs/` directory contains the source files for the vLLM-Omni documentation.

## Building Documentation Locally

### Prerequisites

Use Python 3.12 to match the Read the Docs build environment. The package supports
Python 3.10 through 3.13, as declared in
[`pyproject.toml`](https://github.com/vllm-project/vllm-omni/blob/main/pyproject.toml).
Follow [Getting Started](README.md#getting-started) to create and activate an
environment, then install documentation dependencies from the repository root:

```bash
uv pip install -e ".[docs]"
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
class Omni:
    """Main entry point for vLLM-Omni inference.

    This class provides a high-level interface for running multi-modal
    inference with non-autoregressive models.

    Args:
        model: Model name or path
        deploy_config: Optional path to a deploy configuration
        **kwargs: Additional arguments passed to the engine

    Example:
        >>> llm = Omni(model="Qwen/Qwen2.5-Omni")
        >>> outputs = llm.generate(prompts="Hello")
    """
```

## Documentation Structure

```text
docs/
├── README.md            # Main documentation page
├── .nav.yml             # Documentation navigation
├── getting_started/     # Getting started guides
├── contributing/        # Contributor guides, including this page
├── design/              # Architecture and design documents
├── api/                 # API reference entry pages
├── examples/            # Examples overview
├── user_guide/examples/  # Task guides and generated model example pages
└── mkdocs/              # Build hooks, theme overrides, CSS, and JavaScript
```

## Naming Model Examples

Offline inference and online serving examples for the same model use the same
directory name and the shared display name in
`examples/model_display_names.yml`. Use these title forms:

- `# <Model>: Offline inference`
- `# <Model>: Online serving`

The documentation generator uses the full title for the page H1 and the shared
display name alone for navigation, and fails the build if a mapped README uses
a different H1. Keep checkpoint identifiers in commands and prose rather than
in the display name, and do not rename an existing example directory solely to
adjust its title because the directory defines its public documentation URL.

## Publishing Documentation

The repository's Read the Docs build is configured in
[`.readthedocs.yml`](https://github.com/vllm-project/vllm-omni/blob/main/.readthedocs.yml).
It uses Python 3.12, installs the package with the `docs` extra, and builds with
`mkdocs.yml`, treating warnings as failures.

Submit documentation changes through a pull request following the
[contribution guide](README.md#pull-requests-code-reviews). Contributors can
preview changes locally with `mkdocs serve`; configuring a hosting service is
not required to contribute.

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

- Check the Python version against `pyproject.toml`; Python 3.12 matches the Read the Docs build
- Ensure all dependencies are installed: `pip install -e ".[docs]"`
- Check `mkdocs.yml` syntax with `mkdocs build --strict`
