"""Fixed gap diagnostics for the production migration contract.

The production torque has no selectable gap regime and no Type-I/Type-II
switch. A perturber continuously depletes the local disk according to the
Kanagawa et al. gap-depth fit,

``K = q**2 / (alpha * h**5)`` and ``Sigma_gap / Sigma_0 = 1/(1 + 0.04 K)``.

The Crida et al. parameter is retained only as a diagnostic of a deep gap. It
does not select another torque law. This separation is deliberate: Kanagawa
provides the continuous density reduction used by the torque, whereas Crida
asks whether pressure and viscosity can refill a deep gap.

All quantities in this module are dimensionless. ``q`` is the migrator to
central-mass ratio, ``h = H/R``, and ``alpha`` is the Shakura-Sunyaev
viscosity parameter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

KANAGAWA_GAP_DEPTH_COEFFICIENT = 0.04
KANAGAWA_DEEP_GAP_PARAMETER = 25.0
# Kanagawa et al. calibrated the simple depth fit over approximately this
# range. Larger values are still evaluated for continuity but are explicitly
# marked as extrapolations in every torque profile.
KANAGAWA_CALIBRATED_MAX_PARAMETER = 1.0e4
CRIDA_DEEP_GAP_PARAMETER = 1.0


@dataclass(frozen=True)
class GapDiagnostics:
    """Continuous depletion and deep-gap diagnostics for one radial profile."""

    mass_ratio: float
    alpha: float
    aspect_ratio: FloatArray
    kanagawa_parameter: FloatArray
    gap_depth_fraction: FloatArray
    hill_radius_over_scale_height: FloatArray
    crida_parameter: FloatArray
    deep_gap: BoolArray
    kanagawa_extrapolated: BoolArray


def _validated_inputs(
    mass_ratio: float,
    aspect_ratio: FloatArray,
    alpha: float,
) -> tuple[float, FloatArray, float, BoolArray]:
    if not np.isfinite(mass_ratio) or mass_ratio <= 0.0:
        raise ValueError("mass ratio must be finite and positive")
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and positive")
    local_aspect_ratio = np.asarray(aspect_ratio, dtype=float)
    valid = np.isfinite(local_aspect_ratio) & (local_aspect_ratio > 0.0)
    return float(mass_ratio), local_aspect_ratio, float(alpha), valid


def kanagawa_gap_parameter(
    mass_ratio: float,
    aspect_ratio: FloatArray,
    alpha: float,
) -> FloatArray:
    """Return ``K = q^2 / (alpha h^5)``."""
    mass_ratio, local_aspect_ratio, alpha, valid = _validated_inputs(
        mass_ratio, aspect_ratio, alpha
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        values = mass_ratio**2 / (alpha * local_aspect_ratio**5)
    return np.where(valid, values, np.nan)


def kanagawa_gap_depth_fraction(gap_parameter: FloatArray) -> FloatArray:
    """Return ``Sigma_gap / Sigma_0 = 1 / (1 + 0.04 K)``.

    Invalid or negative ``K`` values are rejected instead of being interpreted
    as an undepleted disk.
    """
    values = np.asarray(gap_parameter, dtype=float)
    if np.any(~np.isfinite(values) | (values < 0.0)):
        raise ValueError("Kanagawa gap parameter must be finite and nonnegative")
    return 1.0 / (1.0 + KANAGAWA_GAP_DEPTH_COEFFICIENT * values)


def hill_radius_over_scale_height(
    mass_ratio: float,
    aspect_ratio: FloatArray,
) -> FloatArray:
    """Return ``R_H/H = (q/3)^(1/3) / h``."""
    if not np.isfinite(mass_ratio) or mass_ratio <= 0.0:
        raise ValueError("mass ratio must be finite and positive")
    local_aspect_ratio = np.asarray(aspect_ratio, dtype=float)
    valid = np.isfinite(local_aspect_ratio) & (local_aspect_ratio > 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.cbrt(mass_ratio / 3.0) / local_aspect_ratio
    return np.where(valid, values, np.nan)


def crida_gap_parameter(
    mass_ratio: float,
    aspect_ratio: FloatArray,
    alpha: float,
) -> FloatArray:
    """Return the Crida pressure-plus-viscosity parameter.

    ``P = 0.75 H/R_H + 50 alpha h^2/q``. ``P <= 1`` labels a deep gap for
    diagnostics; it never switches the production torque to another model.
    """
    mass_ratio, local_aspect_ratio, alpha, valid = _validated_inputs(
        mass_ratio, aspect_ratio, alpha
    )
    hill_ratio = hill_radius_over_scale_height(mass_ratio, local_aspect_ratio)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        values = 0.75 / hill_ratio + 50.0 * alpha * local_aspect_ratio**2 / mass_ratio
    return np.where(valid, values, np.nan)


def gap_diagnostics(
    mass_ratio: float,
    aspect_ratio: FloatArray,
    alpha: float,
) -> GapDiagnostics:
    """Evaluate the fixed production gap law and its validity diagnostics."""
    mass_ratio, local_aspect_ratio, alpha, valid = _validated_inputs(
        mass_ratio, aspect_ratio, alpha
    )
    kanagawa = kanagawa_gap_parameter(mass_ratio, local_aspect_ratio, alpha)
    hill_ratio = hill_radius_over_scale_height(mass_ratio, local_aspect_ratio)
    crida = crida_gap_parameter(mass_ratio, local_aspect_ratio, alpha)
    depth = np.full(local_aspect_ratio.shape, np.nan, dtype=float)
    depth[valid] = kanagawa_gap_depth_fraction(kanagawa[valid])
    return GapDiagnostics(
        mass_ratio=mass_ratio,
        alpha=alpha,
        aspect_ratio=local_aspect_ratio,
        kanagawa_parameter=kanagawa,
        gap_depth_fraction=depth,
        hill_radius_over_scale_height=hill_ratio,
        crida_parameter=crida,
        deep_gap=np.asarray(valid & (crida <= CRIDA_DEEP_GAP_PARAMETER), dtype=bool),
        kanagawa_extrapolated=np.asarray(
            valid & (kanagawa > KANAGAWA_CALIBRATED_MAX_PARAMETER), dtype=bool
        ),
    )


__all__ = [
    "CRIDA_DEEP_GAP_PARAMETER",
    "GapDiagnostics",
    "KANAGAWA_CALIBRATED_MAX_PARAMETER",
    "KANAGAWA_DEEP_GAP_PARAMETER",
    "KANAGAWA_GAP_DEPTH_COEFFICIENT",
    "crida_gap_parameter",
    "gap_diagnostics",
    "hill_radius_over_scale_height",
    "kanagawa_gap_depth_fraction",
    "kanagawa_gap_parameter",
]
