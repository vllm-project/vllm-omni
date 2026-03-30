# L5(b) Reliability (`tests/dfx/reliability`)

See [RFC-dfx-l5-reliability.md](../../../docs/contributing/rfc/RFC-dfx-l5-reliability.md).

- Scenarios: `tests/scenarios.json`
- Fault helpers: `scripts/fault_inject.py`
- Pytest entry: `scripts/test_reliability.py`

```bash
pytest --collect-only tests/dfx/reliability
pytest -s -v tests/dfx/reliability/scripts/test_reliability.py -m slow
```
