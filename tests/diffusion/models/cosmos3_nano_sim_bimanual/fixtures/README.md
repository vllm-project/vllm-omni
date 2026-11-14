# Bimanual CPU fixture

`unified_action_schema.json` is the action-conditioning payload used
by the CPU tests. It declares 59D normalized unified inputs, scalar or per-row
domain routing, and zero padding to the model's 64D action width.

Run the tests with:

```bash
pytest -q tests/diffusion/models/cosmos3_nano_sim_bimanual
```
