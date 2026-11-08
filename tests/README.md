# Checks

From the repository root:

```bash
python scripts/check.py                             # all assertions
python scripts/check.py --run-dir outputs/single_host # plus saved-result check
```

- `core/`: the default single-host path: analytic formation, numerical damping, Type-I migration, coupled gas/GW, detection and bookkeeping.
- `optional/`: legacy gas routing, torque/gap/trap comparisons and SEVN adapter checks. These do not select the default model.
- `run_tests.py`: simple collector used by `scripts/check.py`; `PYTHONPATH=src python -m pytest -q` is an alternative. Install `.[test]` for optional assertions using pytest helpers. Set `IMRI_SEVN_H_TABLES` and `IMRI_SEVN_HE_TABLES` to run the optional SEVN table check.

The benchmark data under `outputs/damping_benchmark/` is preserved locally; the default run does not read it.
