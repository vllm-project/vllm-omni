# Sim-Auto reference fixtures

`reference_visibility/` contains 40 noisy/clean scenarios generated from imaginaire4's
rolling replay metadata and teacher-forcing predicate. The planner matches these Boolean
matrices with exactly-once coverage. This validates the deduplicated policy only.

`reference_parity.json` contains six RGB/joint prompt scenarios (1, 33, 93 frames), framed
token IDs including the four-token sink, and 18 tiny CPU sampler trajectories, covering
RGB/joint seed clocks, condition prefixes and a partial final chunk. The generator calls
the reference prompt augmentors, `tokenize_caption`, rolling prompt framing, chunk schedule,
and distilled sampler with a deterministic test velocity. It does not load model weights.

Regenerate inside the imaginaire4 Cosmos3 environment:

```bash
PYTHONPATH="$PWD/packages/cosmos3:$PWD" python -m projects.cosmos3.interactive.utils.sim_auto_parity_fixtures \
  --tokenizer /exports/cosmos3-nano-sim-auto/text_tokenizer \
  --output <vllm-omni>/tests/diffusion/models/cosmos3_nano_sim_auto/fixtures
```

Run `test_reference_parity.py` with `COSMOS3_SIM_AUTO_TOKENIZER` set to that tokenizer
directory to check real token IDs; without it, only that test skips. File hashes in the
fixture identify the tokenizer assets used. The checked-in tokens used the local
Cosmos3-Nano-Transfer-Auto vocabulary/merges and chat template with `--slow-tokenizer`
(`Qwen2Tokenizer`, Transformers 4.57.1), because its fast `tokenizer.json` was an LFS pointer.
Recheck with the actual Sim-Auto export's fast tokenizer before claiming checkpoint parity.

Generated and checked on CPU on 2026-10-09 (PyTorch version recorded in JSON). Local
validation isolated unrelated GPU imports, executing the production reference function
bodies and real tokenizer/archive preprocessing. Visibility generation substituted a
fail-on-use placeholder for the unused rotary model class. Full pipeline imports,
checkpoint export/load, CUDA noise, GPU latent parity and TP4 were not exercised.
CPU and CUDA RNG streams are different: these CPU values are not CUDA goldens.
