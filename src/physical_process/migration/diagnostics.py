"""Counterfactual diagnostics built from the fixed production torque.

These helpers do not define an alternative migration model and are never used
by production workflows. They construct transparent limiting cases from an
already-computed torque profile so the effect of signed torque zeros can be
measured without changing the gravitational, gap, or thermal physics.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def inward_only_counterfactual_torque(torque: FloatArray) -> FloatArray:
    """Return ``-|Gamma|`` for a trap-free comparison using the same amplitude.

    This is an intentionally nonphysical extreme: every outward cell is
    reflected inward while the local torque magnitude is preserved. It isolates
    the influence of signs and traps; it must not be used as a production
    prescription or described as classical Type-II migration.
    """
    values = np.asarray(torque, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("counterfactual torque must be a one-dimensional profile")
    if not np.isfinite(values).all():
        raise ValueError("counterfactual torque contains nonfinite samples")
    if np.any(values == 0.0):
        raise ValueError(
            "counterfactual torque cannot preserve the magnitude of exact zeros"
        )
    return -np.abs(values)


__all__ = ["inward_only_counterfactual_torque"]
