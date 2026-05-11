# Contributing to vLLM-Omni

Thank you for your interest in contributing to vLLM-Omni! We welcome contributions of all
kinds: bug reports, feature proposals, documentation improvements, translations, model
ports, performance work, and more.

For the authoritative and most up-to-date guide, see the
[Contributing section of the documentation](https://vllm-omni.readthedocs.io/en/latest/contributing/).
This file is a short quick-reference checklist to help new contributors get oriented.

## Quick Links

- Full contributing guide: <https://vllm-omni.readthedocs.io/en/latest/contributing/>
- Documentation guide: [docs/contributing/DOCS_GUIDE.md](docs/contributing/DOCS_GUIDE.md)
- CI / testing guide: [docs/contributing/ci/test_guide.md](docs/contributing/ci/test_guide.md)
- Adding a model:
  [diffusion](docs/contributing/model/adding_diffusion_model.md) ·
  [omni](docs/contributing/model/adding_omni_model.md) ·
  [tts](docs/contributing/model/adding_tts_model.md)
- Code of Conduct: see the upstream
  [vLLM Code of Conduct](https://github.com/vllm-project/vllm/blob/main/CODE_OF_CONDUCT.md)
- Discussion channels: `#sig-omni` on [vLLM Slack](https://slack.vllm.ai) ·
  [vLLM Forum](https://discuss.vllm.ai)

## Before You Start

1. **Search existing issues and PRs** to avoid duplicating work.
2. For **non-trivial features** or design changes, please open an
   [RFC issue](https://github.com/vllm-project/vllm-omni/issues/new?template=750-RFC.yml)
   first so the community can align on the approach.
3. For **new models**, please open a
   [New Model issue](https://github.com/vllm-project/vllm-omni/issues/new?template=600-new-model.yml)
   so that we can track model coverage and avoid duplicate ports.

## Submitting a Pull Request

1. Fork the repository and create a topic branch from `main`.
2. Make your changes; keep the PR scoped to one logical change.
3. Run the relevant tests locally. See the
   [tests guide](docs/contributing/ci/test_guide.md) for how the 5-level CI works
   and which tests apply to your change.
4. Follow the project's commit style (see the recent `git log` for examples — most
   commits use a `[Tag]` prefix such as `[Bugfix]`, `[Docs]`, `[Model]`, `[CI]`).
5. Fill in the
   [Pull Request Template](.github/PULL_REQUEST_TEMPLATE.md) — in particular, the
   **Test Plan** section is required.
6. Be ready to iterate on review feedback. Maintainers may request changes,
   additional tests, or further benchmarks for performance-sensitive code.

## Reporting Bugs

Please use the appropriate issue template:

- [🐛 Bug report](https://github.com/vllm-project/vllm-omni/issues/new?template=400-bug-report.yml)
- [📚 Documentation issue](https://github.com/vllm-project/vllm-omni/issues/new?template=100-documentation.yml)
- [⚙️ Installation issue](https://github.com/vllm-project/vllm-omni/issues/new?template=200-installation.yml)
- [🚀 Feature request](https://github.com/vllm-project/vllm-omni/issues/new?template=500-feature-request.yml)
- [🆕 New model request](https://github.com/vllm-project/vllm-omni/issues/new?template=600-new-model.yml)
- [⚡ Performance discussion](https://github.com/vllm-project/vllm-omni/issues/new?template=700-performance-discussion.yml)
- [📋 RFC](https://github.com/vllm-project/vllm-omni/issues/new?template=750-RFC.yml)

A good bug report includes a minimal reproduction, the exact command used,
hardware/platform (e.g. CUDA / ROCm / NPU / XPU), `pip freeze` output for the relevant
packages, and the model identifier.

## License

By contributing, you agree that your contributions will be licensed under the
[Apache License, Version 2.0](LICENSE).
