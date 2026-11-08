"""Automated correctness tests for the canonical scientific modules.

These are *assertions*, not diagnostics: each one has a single right answer and
fails loudly.  Human-facing scans, scientific figures and checkpoint
demonstrations live in ``scripts/`` instead.

Core and optional assertions are grouped in separate folders. The no-fixture
functions run via ``python scripts/check.py``, or directly with pytest.
Install the `test` extra for optional tests that import pytest helpers.
"""
