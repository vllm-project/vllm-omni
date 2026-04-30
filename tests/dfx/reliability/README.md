# L5(b) Reliability (`tests/dfx/reliability`)

This directory contains reliability fault-injection tests for key models.
The detailed RFC document is maintained outside this repository (local/internal only).

- Fault helpers: `conftest.py`, `helpers.py`
- Qwen3 tests: `test_reliability_qwen3_omni.py`
- Wan2.2 tests: `test_reliability_wan22.py`

```bash
pytest --collect-only tests/dfx/reliability
pytest -s -v tests/dfx/reliability/test_reliability_qwen3_omni.py -m slow
pytest -s -v tests/dfx/reliability/test_reliability_wan22.py -m slow
```
