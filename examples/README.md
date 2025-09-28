# vLLM-omni Examples

A curated set of runnable samples demonstrating vLLM-omni usage.

## Directory Overview

- **basic/** – Quick-start scripts for text-only flows.
  - `text_generation.py` – Minimal CPU-friendly text generation with vLLM.
  - `api_client.py` – Calls the FastAPI server using the JSON API.
  - `docker_run.py` – Template for launching the server in Docker.
- **omni/** – Multi-stage AR → DiT pipelines powered by diffusers.
  - `ar_dit_diffusers.py` – Runs Qwen3 (AR) feeding Stable Diffusion 2.1 (DiT).
  - `configs/ar_dit_local.yaml` – Sample plugin config referencing local weights.
- **advanced/** – Placeholder for upcoming complex flows (Ray, benchmarking).

See the subdirectory READMEs for usage instructions.
