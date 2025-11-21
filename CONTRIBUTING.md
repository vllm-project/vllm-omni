# Contributing to vLLM

## Getting Started

Setup environment with uv
```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
```

Pre-commit has been linked to git commit. For run checks locally, please install pre-commit:
```bash
uv pip install pre-commit
```

Then run:
```bash
pre-commit run --show-diff-on-failure --color=always --all-files
```
