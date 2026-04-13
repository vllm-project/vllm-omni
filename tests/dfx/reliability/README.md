# L5(b) Reliability (`tests/dfx/reliability`)

See [RFC-dfx-l5-reliability.md](../../../docs/contributing/rfc/RFC-dfx-l5-reliability.md).

- Fault helpers: `conftest.py`
- Qwen3 tests: `test_reliability_qwen3_omni.py`
- Wan2.2 tests: `test_reliability_wan22.py`

```bash
pytest --collect-only tests/dfx/reliability
pytest -s -v tests/dfx/reliability/test_reliability_qwen3_omni.py -m slow
pytest -s -v tests/dfx/reliability/test_reliability_wan22.py -m slow
```
