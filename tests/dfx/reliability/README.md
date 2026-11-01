# L5(b) Reliability (`tests/dfx/reliability`)

This directory contains reliability fault-injection tests for key models.
The detailed RFC document is maintained outside this repository (local/internal only).

- Fault helpers: `conftest.py`, `helpers.py` (`FaultModel`, signals, `scenario_params`)
- Fault steps: `scenarios/`
- Model suites: `test_qwen3_omni.py`, `test_wan22.py`, `test_hunyuan_image.py`, `test_minicpmo_4_5.py`

```bash
pytest --collect-only tests/dfx/reliability
pytest -s -v tests/dfx/reliability/test_qwen3_omni.py -m "slow and cards_2"
pytest -s -v tests/dfx/reliability/test_wan22.py -m "slow and cards_1"
pytest -s -v tests/dfx/reliability/test_hunyuan_image.py -m "slow and cards_4"
pytest -s -v tests/dfx/reliability/test_minicpmo_4_5.py -m "slow and cards_1"
```
