"""Canonical gap-opening criteria for AGN-disk migration.

This module is the single authoritative source for every gap-related
dimensionless quantity and threshold in the project.  Torque construction
(:mod:`.torque`), the Type-I/Type-II switch (:mod:`.evolution`) and every
diagnostic figure read their gap physics from here.

Physical inputs
---------------
``mass_ratio``      q = m_migrator / M_smbh  (dimensionless)
``aspect_ratio``    h = H/R, the local disk aspect ratio (dimensionless)
``alpha``           Shakura-Sunyaev viscosity parameter (dimensionless)

Physical outputs
----------------
Three independent dimensionless gap diagnostics plus the boolean
"this cell has opened a gap" decision for the selected criterion.

The three criteria
------------------
``kanagawa``  **(production default, calibrated)**
    ``K = q^2 / (alpha * h^5)``, gap open when ``K > 25`` -- equivalently, when
    the gap is cleared below half the unperturbed surface density.
    This is the *same* dimensionless parameter that
    :func:`.torque.kanagawa_gap_parameter` uses to attenuate the Type-I torque,
    so the Type-II switch and the Type-I attenuation are now driven by one
    quantity instead of two inconsistent ones.  The threshold is fixed by an
    identity internal to this model: the Type-I gap denominator
    ``1 + K / kappa_norm`` with ``kappa_norm = 25``
    (:class:`.torque.MigrationParameters`) *is* the Kanagawa gap-depth law
    ``Sigma_gap/Sigma_0 = 1/(1 + 0.04 K)``, so ``K = 25`` is exactly where the
    gap reaches half depth and the Type-I torque is halved.

``kii``
    ``K_II = q^2 / (alpha * h^2)``, gap open when ``K_II > 11``.
    The repository's *historical* criterion, retained so that pre-existing
    results can be reproduced with ``--gap-open-condition kii``.

``crida``
    Gap open when ``P < 1`` and ``R_H / H > 1``, with
    ``P = 0.75 / (R_H/H) + 50 * alpha * h^2 / q`` and
    ``R_H / H = (q/3)^(1/3) / h``.

.. warning::

   ``kii`` and ``kanagawa`` differ by a factor ``h^3`` and are therefore *not*
   small perturbations of one another: for ``h ~ 0.01`` they differ by six
   orders of magnitude, and they select very different amounts of the disk for
   the Type-II branch.  The production default was changed from ``kii`` to
   ``kanagawa`` as a deliberate, documented scientific decision; results
   produced before that change are **not** comparable to results produced
   after it.  See ``docs/physics_model.md`` for the measured consequences.

   The ``kanagawa`` threshold (25) is load-bearing and has been **calibrated**
   against the model's own gap-depth law by
   ``scripts/calibrate_gap_threshold.py``; it is not an arbitrary number.  What
   remains a modelling choice is the *depletion fraction* that counts as
   "gapped" -- 25 corresponds to a half-cleared gap.  Use
   :func:`threshold_for_depletion` to re-express any other choice.

Validity
--------
All criteria assume a geometrically thin (``h << 1``), viscously relaxed,
axisymmetric disk and a migrator on a circular, coplanar orbit.  Cells with a
non-finite or non-positive aspect ratio are never reported as gap-open.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

GAP_CRITERION_KII = "kii"
GAP_CRITERION_CRIDA = "crida"
GAP_CRITERION_KANAGAWA = "kanagawa"

#: Selectable gap-opening criteria, most-used first.
GAP_OPENING_CONDITIONS: tuple[str, ...] = (
    GAP_CRITERION_KANAGAWA,
    GAP_CRITERION_KII,
    GAP_CRITERION_CRIDA,
)

#: The production gap criterion.  Changed from ``kii`` to ``kanagawa`` when the
#: model authors resolved the ambiguity documented below; this is the only place
#: the project-wide default is set.
DEFAULT_GAP_CRITERION = GAP_CRITERION_KANAGAWA

#: Aspect-ratio exponent of the historical ``kii`` parameter, q^2/(alpha h^n).
KII_ASPECT_RATIO_EXPONENT = 2.0
#: Gap-open threshold on the historical ``kii`` parameter.
KII_GAP_PARAMETER_THRESHOLD = 11.0

#: Aspect-ratio exponent of the Kanagawa-form parameter used by the pAGN
#: torque attenuation in :mod:`.torque`.
KANAGAWA_ASPECT_RATIO_EXPONENT = 5.0

#: Gap-open threshold on the Kanagawa-form parameter, **calibrated** by
#: ``scripts/calibrate_gap_threshold.py`` (see :func:`depletion_fraction` for the
#: interpretation and ``docs/physics_model.md`` section 3 for the full study).
#:
#: The calibration rests on an identity internal to this model: the Type-I gap
#: denominator ``1 + K / kappa_norm`` with ``kappa_norm = 25`` is *exactly* the
#: Kanagawa gap-depth law ``Sigma_gap / Sigma_0 = 1 / (1 + 0.04 K)`` (verified to
#: 2.2e-16).  A threshold on K is therefore a threshold on residual surface
#: density, and ``K = 25`` is the point at which the gap is half cleared -- the
#: same scale the Type-I torque already uses, so it is fixed by the model rather
#: than chosen freely.
#:
#: The alternative calibration target, placing the switch where the Type-I and
#: Type-II torques have equal magnitude so the transition is continuous, was
#: measured and **rejected**: that crossover spans a factor 2.5e6 in K across the
#: SMBH/migrator mass grid, so no single global threshold can achieve it.
#:
#: Override with ``--gap-threshold`` or ``GapCriterion(kanagawa_threshold=...)``.
KANAGAWA_GAP_PARAMETER_THRESHOLD = 25.0

#: The gap-depth coefficient implied by ``kappa_norm``; ``Sigma_gap/Sigma_0``
#: reaches 1/2 at ``K = 1 / KANAGAWA_GAP_DEPTH_COEFFICIENT``.
KANAGAWA_GAP_DEPTH_COEFFICIENT = 0.04

#: Crida thresholds: a gap requires both a sub-unity P and a Hill radius that
#: exceeds the local scale height.
CRIDA_PARAMETER_THRESHOLD = 1.0
CRIDA_HILL_RATIO_THRESHOLD = 1.0

# Backward-compatible alias for the historical module-level name.
TYPE_II_GAP_PARAMETER_THRESHOLD = KII_GAP_PARAMETER_THRESHOLD


@dataclass(frozen=True)
class GapCriterion:
    """A named gap-opening criterion with explicit, configurable thresholds."""

    name: str = DEFAULT_GAP_CRITERION
    kii_threshold: float = KII_GAP_PARAMETER_THRESHOLD
    kanagawa_threshold: float = KANAGAWA_GAP_PARAMETER_THRESHOLD
    crida_parameter_threshold: float = CRIDA_PARAMETER_THRESHOLD
    crida_hill_ratio_threshold: float = CRIDA_HILL_RATIO_THRESHOLD

    def __post_init__(self) -> None:
        if self.name not in GAP_OPENING_CONDITIONS:
            raise ValueError(
                f"gap-opening condition must be one of {GAP_OPENING_CONDITIONS}"
            )
        for field_name in (
            "kii_threshold",
            "kanagawa_threshold",
            "crida_parameter_threshold",
            "crida_hill_ratio_threshold",
        ):
            value = getattr(self, field_name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kii_threshold": self.kii_threshold,
            "kii_aspect_ratio_exponent": KII_ASPECT_RATIO_EXPONENT,
            "kanagawa_threshold": self.kanagawa_threshold,
            "kanagawa_aspect_ratio_exponent": KANAGAWA_ASPECT_RATIO_EXPONENT,
            "crida_parameter_threshold": self.crida_parameter_threshold,
            "crida_hill_ratio_threshold": self.crida_hill_ratio_threshold,
        }


@dataclass(frozen=True)
class GapDiagnostics:
    """Every gap diagnostic for one radial profile, plus the gap decision.

    All arrays share the shape of the input ``aspect_ratio``.  Quantities are
    dimensionless.
    """

    mass_ratio: float
    alpha: float
    aspect_ratio: FloatArray
    kii_parameter: FloatArray
    kanagawa_parameter: FloatArray
    hill_radius_over_scale_height: FloatArray
    crida_parameter: FloatArray
    criterion: GapCriterion
    gap_open: BoolArray

    @property
    def gap_open_fraction(self) -> float:
        """Fraction of valid cells that have opened a gap."""
        return float(np.mean(self.gap_open)) if self.gap_open.size else 0.0

    def parameter_for(self, name: str) -> FloatArray:
        """Return the dimensionless parameter controlling a named criterion."""
        if name == GAP_CRITERION_KII:
            return self.kii_parameter
        if name == GAP_CRITERION_KANAGAWA:
            return self.kanagawa_parameter
        if name == GAP_CRITERION_CRIDA:
            return self.crida_parameter
        raise ValueError(f"unknown gap criterion: {name}")


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


def kii_gap_parameter(
    mass_ratio: float,
    aspect_ratio: FloatArray,
    alpha: float,
) -> FloatArray:
    """Return the historical gap parameter ``q^2 / (alpha * h^2)``."""
    mass_ratio, local_aspect_ratio, alpha, _ = _validated_inputs(
        mass_ratio, aspect_ratio, alpha
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return mass_ratio**2 / (
            alpha * local_aspect_ratio**KII_ASPECT_RATIO_EXPONENT
        )


def kanagawa_gap_parameter(
    mass_ratio: float,
    aspect_ratio: FloatArray,
    alpha: float,
) -> FloatArray:
    """Return the Kanagawa-form gap parameter ``q^2 / (alpha * h^5)``.

    This is the parameter the pAGN torque model uses for the Type-I gap
    attenuation factor and denominator (see :mod:`.torque`).
    """
    mass_ratio, local_aspect_ratio, alpha, _ = _validated_inputs(
        mass_ratio, aspect_ratio, alpha
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return mass_ratio**2 / (
            alpha * local_aspect_ratio**KANAGAWA_ASPECT_RATIO_EXPONENT
        )


def hill_radius_over_scale_height(
    mass_ratio: float,
    aspect_ratio: FloatArray,
) -> FloatArray:
    """Return ``R_H / H = (q/3)^(1/3) / h``."""
    if not np.isfinite(mass_ratio) or mass_ratio <= 0.0:
        raise ValueError("mass ratio must be finite and positive")
    local_aspect_ratio = np.asarray(aspect_ratio, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.cbrt(mass_ratio / 3.0) / local_aspect_ratio


def crida_gap_parameter(
    mass_ratio: float,
    aspect_ratio: FloatArray,
    alpha: float,
) -> FloatArray:
    """Return ``P = 0.75 / (R_H/H) + 50 * alpha * h^2 / q``."""
    mass_ratio, local_aspect_ratio, alpha, _ = _validated_inputs(
        mass_ratio, aspect_ratio, alpha
    )
    hill_ratio = hill_radius_over_scale_height(mass_ratio, local_aspect_ratio)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return 0.75 / hill_ratio + 50.0 * alpha * local_aspect_ratio**2 / mass_ratio


def depletion_fraction(threshold: float = KANAGAWA_GAP_PARAMETER_THRESHOLD) -> float:
    """Return the residual surface density ``Sigma_gap / Sigma_0`` at ``threshold``.

    This is what a Kanagawa-form gap threshold physically *means* in this model:
    ``K > K*`` is the statement "the gap is cleared to below this fraction".
    The default threshold corresponds to a half-cleared gap.
    """
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold must be finite and positive")
    return 1.0 / (1.0 + KANAGAWA_GAP_DEPTH_COEFFICIENT * float(threshold))


def threshold_for_depletion(fraction: float) -> float:
    """Return the Kanagawa threshold that corresponds to a depletion fraction.

    Inverse of :func:`depletion_fraction`; use it to re-calibrate the criterion
    in physically interpretable units, e.g. ``threshold_for_depletion(0.1)`` for
    a gap cleared to a tenth of the unperturbed surface density.
    """
    if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("depletion fraction must satisfy 0 < f <= 1")
    return (1.0 / float(fraction) - 1.0) / KANAGAWA_GAP_DEPTH_COEFFICIENT


def gap_diagnostics(
    mass_ratio: float,
    aspect_ratio: FloatArray,
    alpha: float,
    criterion: GapCriterion | str | None = None,
) -> GapDiagnostics:
    """Evaluate every gap diagnostic and the gap decision for one profile.

    This is the function that diagnostic scripts and figures must call: it
    guarantees that a plotted criterion is the same one the simulation applies.
    """
    if criterion is None:
        criterion = GapCriterion()
    elif isinstance(criterion, str):
        criterion = GapCriterion(name=criterion)

    mass_ratio, local_aspect_ratio, alpha, valid = _validated_inputs(
        mass_ratio, aspect_ratio, alpha
    )
    kii = kii_gap_parameter(mass_ratio, local_aspect_ratio, alpha)
    kanagawa = kanagawa_gap_parameter(mass_ratio, local_aspect_ratio, alpha)
    hill_ratio = hill_radius_over_scale_height(mass_ratio, local_aspect_ratio)
    crida = crida_gap_parameter(mass_ratio, local_aspect_ratio, alpha)

    if criterion.name == GAP_CRITERION_KII:
        gap_open = valid & (kii > criterion.kii_threshold)
    elif criterion.name == GAP_CRITERION_KANAGAWA:
        gap_open = valid & np.isfinite(kanagawa) & (
            kanagawa > criterion.kanagawa_threshold
        )
    else:
        gap_open = (
            valid
            & np.isfinite(crida)
            & np.isfinite(hill_ratio)
            & (crida < criterion.crida_parameter_threshold)
            & (hill_ratio > criterion.crida_hill_ratio_threshold)
        )

    return GapDiagnostics(
        mass_ratio=mass_ratio,
        alpha=alpha,
        aspect_ratio=local_aspect_ratio,
        kii_parameter=kii,
        kanagawa_parameter=kanagawa,
        hill_radius_over_scale_height=hill_ratio,
        crida_parameter=crida,
        criterion=criterion,
        gap_open=np.asarray(gap_open, dtype=bool),
    )


def gap_opening_mask(
    mass_ratio: float,
    aspect_ratio: FloatArray,
    alpha: float,
    condition: str,
) -> BoolArray:
    """Return the cells that have opened a gap under ``condition``.

    Backward-compatible entry point used by the migration solvers.  Identical
    in behaviour to the historical implementation for ``kii`` and ``crida``.
    """
    return gap_diagnostics(mass_ratio, aspect_ratio, alpha, condition).gap_open


__all__ = [
    "CRIDA_HILL_RATIO_THRESHOLD",
    "CRIDA_PARAMETER_THRESHOLD",
    "GAP_CRITERION_CRIDA",
    "GAP_CRITERION_KANAGAWA",
    "GAP_CRITERION_KII",
    "DEFAULT_GAP_CRITERION",
    "GAP_OPENING_CONDITIONS",
    "GapCriterion",
    "GapDiagnostics",
    "KANAGAWA_ASPECT_RATIO_EXPONENT",
    "KANAGAWA_GAP_DEPTH_COEFFICIENT",
    "KANAGAWA_GAP_PARAMETER_THRESHOLD",
    "KII_ASPECT_RATIO_EXPONENT",
    "KII_GAP_PARAMETER_THRESHOLD",
    "TYPE_II_GAP_PARAMETER_THRESHOLD",
    "crida_gap_parameter",
    "depletion_fraction",
    "gap_diagnostics",
    "gap_opening_mask",
    "hill_radius_over_scale_height",
    "kanagawa_gap_parameter",
    "kii_gap_parameter",
    "threshold_for_depletion",
]
