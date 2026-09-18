"""Automated correctness tests for the canonical scientific modules.

These are *assertions*, not diagnostics: each one has a single right answer and
fails loudly.  Human-facing scans, scientific figures and checkpoint
demonstrations live in ``scripts/`` instead.

The supplied environment has no pytest, so every test here is a plain
``test_*`` function with no fixtures and no third-party test dependency.  They
are collected and run by ``tests/run_tests.py`` and are also discoverable by
pytest if it is ever installed.
"""
