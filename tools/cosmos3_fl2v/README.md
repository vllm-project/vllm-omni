# Cosmos3 FL2V (First-Last Boundary to Video) — vLLM-Omni client

First-last boundary conditioning for Cosmos3 on vLLM-Omni: pin a clip to a start
and end boundary (image or short clip) and generate the middle from an event
prompt.

**Full instructions live in the recipe:**
`recipes/cosmos3/Cosmos3-FL2V.md`.

Contents of this directory:

- `fl2v_generate_vllm.py` — HTTP client for `POST /v1/videos/sync`
- `patch_vllm_shm.py` — required upstream vLLM fix (run in the server env)
- `requirements.txt` — client dependencies (no torch)
- `testdata/fl2v_from_cosmos_v2v/` — demo seeds and prompts

Quick check (no GPU): `python fl2v_generate_vllm.py --dry-run`.
