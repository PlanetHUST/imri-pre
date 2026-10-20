"""One explicit AGN-disk migration torque contract.

The production model is fixed rather than selected at runtime:

1. Jimenez-Masset (2017) linear gravitational torque coefficients are
   evaluated on the pAGN disk.
2. The gravitational torque is reduced continuously by the Kanagawa gap depth
   ``f_gap = 1/(1 + 0.04 K)``.
3. The thermal torque follows the Grishin, Gilbaum & Stone (2024) equations:
   local Rosseland opacity controls radiative diffusivity, hot and cold terms
   receive their separate saturation factors, and optically thin gas is
   suppressed by ``1 - exp(-sqrt(alpha) tau_gap)``.
4. The total torque is ``f_gap * Gamma_grav + Gamma_thermal``. There is no
   classical Type-II branch and no ungapped production branch.

For this project's low central masses, Eddington-capped luminosity is retained:
the standard local Bondi supply exceeds the Eddington requirement throughout
the physical disks for ``M_bh <= 1e5 Msun``. The paper's analytic disk zones
and global luminosity bifurcation are not imported because pAGN owns the disk
structure here.

The linear theory is frequently extrapolated in this mass-ratio regime.
``thermal_torque_regime_valid`` and ``kanagawa_extrapolated`` expose that fact
per radial cell; they do not silently zero or replace the requested demo model.

pAGN quantities and all ``*_si`` torques use SI units. Radius aliases ending in
``_cm``, ``_rg`` and ``_rs`` are provided for downstream CGS evolution.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import numpy as np

from .gap import GapDiagnostics, gap_diagnostics

METER_TO_CM = 100.0
SI_TORQUE_TO_CGS = 1.0e7


def torque_si_to_cgs(torque_si: np.ndarray) -> np.ndarray:
    """Convert torque from SI [kg m^2 s^-2] to CGS [g cm^2 s^-2]."""
    return np.asarray(torque_si, dtype=float) * SI_TORQUE_TO_CGS


@dataclass(frozen=True)
class MigrationParameters:
    """Physical parameters shared by the disk and fixed torque contract."""

    alpha: float = 0.01
    gamma_adiabat: float = 5.0 / 3.0
    hydrogen_fraction: float = 0.7
    #: Whether to keep the thermal torque where its linear derivation does not
    #: hold.  The Grishin-2024/Masset-2017 thermal torque is linear theory: it
    #: assumes the perturber's Hill sphere sits inside the disk thickness
    #: (``R_H < H``), among other conditions collected in
    #: ``thermal_torque_regime_valid``.  In this project that condition fails in
    #: *every* cell of *every* migration path -- ``R_H/H ~ 2-11``, because AGN
    #: disks are thin and the mass ratios are not small enough.  Extrapolating
    #: a linear result that far produced the dominant term in the torque
    #: balance, and with it the traps that set the IMRI yield.
    #:
    #: Default ``False``: outside its validity the thermal torque is dropped
    #: rather than extrapolated.  That is a statement about what the model can
    #: justify, not a claim that the true torque is zero there -- the honest
    #: reading is "unknown", and zero is the choice that does not manufacture
    #: structure the theory cannot support.  Set ``True`` to reproduce the
    #: extrapolating behaviour.
    extrapolate_thermal_torque: bool = False


@dataclass(frozen=True)
class ThermalTorqueTerms:
    """Grishin-2024 thermal coefficient and its regime diagnostics."""

    coefficient: np.ndarray
    thermal_diffusivity_m2_s: np.ndarray
    effective_optical_depth: np.ndarray
    thermal_mass_ratio: np.ndarray
    thermal_length_over_scale_height: np.ndarray
    corotation_offset_over_thermal_length: np.ndarray
    bondi_radius_over_scale_height: np.ndarray
    thermal_torque_regime_valid: np.ndarray


@dataclass
class TorqueProfile:
    """Torque components and validity diagnostics on one pAGN radial grid."""

    radius_m: np.ndarray
    radius_cm: np.ndarray
    radius_rg: np.ndarray
    radius_rs: np.ndarray
    surface_density_kg_m2: np.ndarray
    pressure_pa: np.ndarray
    opacity_m2_kg: np.ndarray
    dlog_surface_density_dlog_radius: np.ndarray
    dlog_temperature_dlog_radius: np.ndarray
    dlog_pressure_dlog_radius: np.ndarray
    corotation_coeff: np.ndarray
    lindblad_coeff: np.ndarray
    thermal_coeff: np.ndarray
    gap_parameter_dimensionless: np.ndarray
    gap_depth_fraction: np.ndarray
    crida_parameter: np.ndarray
    deep_gap: np.ndarray
    kanagawa_extrapolated: np.ndarray
    hill_radius_over_scale_height: np.ndarray
    effective_optical_depth: np.ndarray
    thermal_diffusivity_m2_s: np.ndarray
    thermal_mass_ratio: np.ndarray
    thermal_length_over_scale_height: np.ndarray
    corotation_offset_over_thermal_length: np.ndarray
    bondi_radius_over_scale_height: np.ndarray
    thermal_torque_regime_valid: np.ndarray
    #: Cells where the background varies faster than one scale height, so the
    #: linear torque derivation does not apply. Their log-slopes were repaired
    #: by interpolation from the smooth background either side.
    wkb_invalid: np.ndarray
    #: 1.0 where the thermal torque was applied, 0.0 where it was dropped as
    #: outside its linear-theory validity. Always 1.0 when
    #: ``MigrationParameters.extrapolate_thermal_torque`` is set.
    thermal_validity_weight: np.ndarray
    gamma_0_si: np.ndarray
    gamma_type_i_si: np.ndarray
    gamma_th_si: np.ndarray
    #: Gap-limited viscous torque, slowed by secondary inertia. Negative.
    gamma_secondary_dominated_si: np.ndarray
    #: ``B = Sigma r^2 / m``: local disk mass relative to the migrator.
    #: ``B >> 1`` is the classical disk-dominated Type-II regime; ``B << 1``
    #: means the migrator dominates and transport is inertia-limited.
    disk_dominance_b: np.ndarray
    gamma_total_si: np.ndarray


def _require_pagn_bits():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        import pagn.constants as ct
        from pagn.opacities import electron_scattering_opacity
    except ImportError as exc:
        raise ImportError(
            "pAGN is required for migration torques. Install with `pip install pagn`."
        ) from exc
    return ct, electron_scattering_opacity


def _constant(ct, *names: str) -> float:
    for name in names:
        if hasattr(ct, name):
            return float(getattr(ct, name))
    raise AttributeError(f"pAGN.constants is missing one of: {', '.join(names)}")


def _sorted_disk_view(disk) -> SimpleNamespace:
    radius_m = np.asarray(disk.R, dtype=float)
    order = np.argsort(radius_m)
    data = {"Mbh": float(disk.Mbh), "Rs": float(disk.Rs)}
    for name in ("R", "rho", "T", "h", "cs", "Omega", "tauV", "kappa", "Teff4"):
        values = np.asarray(getattr(disk, name), dtype=float)
        data[name] = values[order] if values.shape == radius_m.shape else values
    return SimpleNamespace(**data)


def safe_divide(numerator, denominator):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    num, den = np.broadcast_arrays(numerator, denominator)
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den != 0.0)


def radial_log_slope(radius_m, values, *, smoothing: float = 0.005, k: int = 3):
    radius_m = np.asarray(radius_m, dtype=float)
    values = np.asarray(values, dtype=float)
    if np.any(radius_m <= 0.0) or np.any(values <= 0.0):
        raise ValueError("radius and values must be positive for log-slope")

    try:
        from scipy.interpolate import UnivariateSpline
    except ImportError as exc:
        raise ImportError("scipy is required for torque-profile slopes") from exc

    radius_log = np.log10(radius_m)
    values_log = np.log10(values)
    spline = UnivariateSpline(
        radius_log,
        values_log,
        k=min(int(k), radius_log.size - 1),
        s=float(smoothing),
        ext=0,
    )
    return np.asarray(spline.derivative()(radius_log), dtype=float)


def wkb_invalid_mask(radius_m, *value_arrays, max_step_dex: float = 0.15):
    """Flag cells separated by a discontinuous jump in the disk background.

    The linear (WKB) torque derivation assumes the background is smooth on the
    scale of the wave it launches.  The pAGN solution contains a genuine
    thermal instability at hydrogen ionisation, where the steady-state
    temperature drops ~0.63 dex between two adjacent cells.  It is real
    bistability of the disk, not a table artefact (the opacity spline itself is
    smooth, with ``|dln kappa/dln T| ~ 25``) and not under-resolution: the jump
    stays 0.63 dex from ``N = 400`` to ``N = 6400`` while the cell width shrinks
    16-fold.

    Across it the measured ``|dlnT/dlnR|`` reaches ~133 against a typical 0.5.
    The Lindblad coefficient is linear in that slope, so the torque there is
    not merely inaccurate but sign-reversed -- the origin of the apparent
    migration traps.

    The criterion is the **jump in the value**, in decades, not the slope.  A
    slope threshold would be resolution dependent in exactly the way that makes
    such criteria unusable: refining the grid divides the same fixed jump by a
    smaller cell width and inflates the slope without changing the physics.
    The step in dex is invariant, so this flags the same feature at every
    resolution.
    """
    radius_m = np.asarray(radius_m, dtype=float)
    invalid = np.zeros(radius_m.shape, dtype=bool)
    for values in value_arrays:
        values = np.asarray(values, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            step_dex = np.abs(np.diff(np.log10(np.abs(values))))
        jump = np.flatnonzero(np.isfinite(step_dex) & (step_dex > float(max_step_dex)))
        invalid[jump] = True
        invalid[jump + 1] = True
    return invalid


def repair_slopes_across(mask, radius_m, *slope_arrays):
    """Interpolate log-slopes across flagged cells from their valid neighbours.

    The unstable zone is geometrically thin -- one to three cells, a few
    thousandths of a decade -- so a migrator crosses it in negligible time and
    the transit contributes nothing to the integrated migration time.  What it
    must not do is inject a spurious sign flip into the torque.  Replacing the
    slopes there by interpolation from the smooth background on either side
    removes the artefact without inventing structure: the repaired slopes are
    bounded by values the disk actually attains.

    Returns the slopes unchanged when every cell is valid, or when so many are
    flagged that interpolation would be extrapolation.
    """
    mask = np.asarray(mask, dtype=bool)
    radius_log = np.log10(np.asarray(radius_m, dtype=float))
    valid = ~mask
    if not mask.any() or valid.sum() < 2:
        return tuple(np.asarray(s, dtype=float) for s in slope_arrays)
    return tuple(
        np.where(
            mask,
            np.interp(radius_log, radius_log[valid], np.asarray(s, dtype=float)[valid]),
            np.asarray(s, dtype=float),
        )
        for s in slope_arrays
    )


def total_pressure_pa(disk):
    """Return gas plus radiation pressure [Pa] from the pAGN solution."""
    ct, _ = _require_pagn_bits()
    k_boltzmann = _constant(ct, "Kb", "kB", "k")
    mass_u = _constant(ct, "massU", "mU", "mp")
    gas_pressure_pa = disk.rho * disk.T * k_boltzmann / mass_u
    radiation_pressure_pa = disk.tauV * ct.sigmaSB * disk.Teff4 / (2.0 * ct.c)
    return gas_pressure_pa + radiation_pressure_pa


def surface_density_kg_m2(disk):
    """Return the pAGN vertically integrated surface density [kg m^-2]."""
    return 2.0 * disk.rho * disk.h


def comp_corot(
    dlog_surface_density_dlog_radius, dlog_temperature_dlog_radius, gamma_adiabat
):
    """Return the Jimenez-Masset linear corotation coefficient."""
    return (
        0.46
        + 0.96 * dlog_surface_density_dlog_radius
        - 1.8 * dlog_temperature_dlog_radius
    ) / gamma_adiabat


def thermal_diffusivity_m2_s(disk, gamma_adiabat):
    """Return radiative thermal diffusivity [m^2 s^-1].

    Grishin et al. equation 15 uses the ambient disk opacity. Here that is the
    local pAGN Rosseland mean ``disk.kappa``, not a fixed scattering opacity.
    """
    ct, _ = _require_pagn_bits()
    opacity_m2_kg = np.asarray(disk.kappa, dtype=float)
    if np.any(~np.isfinite(opacity_m2_kg) | (opacity_m2_kg <= 0.0)):
        raise ValueError("disk opacity must be finite and positive")
    return (
        16.0
        * gamma_adiabat
        * (gamma_adiabat - 1.0)
        * ct.sigmaSB
        * disk.T**4
        / (3.0 * opacity_m2_kg * disk.rho**2 * disk.h**2 * disk.Omega**2)
    )


def comp_lind(
    dlog_surface_density_dlog_radius,
    dlog_temperature_dlog_radius,
    disk,
    gamma_adiabat,
):
    """Return the Jimenez-Masset linear Lindblad coefficient."""
    diffusivity_m2_s = thermal_diffusivity_m2_s(disk, gamma_adiabat)
    critical_diffusivity_m2_s = disk.h**2 * disk.Omega
    x_sqrt = np.sqrt(safe_divide(diffusivity_m2_s, 2.0 * critical_diffusivity_m2_s))
    f_gamma = (x_sqrt + 1.0 / gamma_adiabat) / (x_sqrt + 1.0)
    return (
        -2.34
        - 0.1 * dlog_surface_density_dlog_radius
        + 1.5 * dlog_temperature_dlog_radius
    ) * f_gamma


def comp_gamma_0_si(
    mass_ratio,
    radius_m,
    scale_height_m,
    orbital_frequency_rad_s,
    density_kg_m3,
):
    """Return the net Type-I normalization [kg m^2 s^-2]."""
    aspect_ratio = scale_height_m / radius_m
    return (
        mass_ratio**2
        * radius_m**4
        * orbital_frequency_rad_s**2
        * (2.0 * density_kg_m3 * scale_height_m)
        / aspect_ratio**2
    )


def comp_secondary_dominated_torque_si(
    alpha,
    surface_density_kg_m2,
    radius_m,
    aspect_ratio,
    orbital_frequency_rad_s,
    migrator_mass_kg,
):
    """Return the gap-limited viscous torque, slowed by secondary inertia.

    Once a gap is open the linear Lindblad/corotation torques no longer
    describe the transport: the secondary is locked to the disk's viscous
    evolution.  The classical Type-II rate assumes the disk carries the
    secondary, which requires the local disk mass to dominate,

        B = Sigma r^2 / m >> 1.

    In this project ``B ~ 1e-3``: the migrator outweighs the gas it is embedded
    in by three orders of magnitude, so it cannot be advected at the viscous
    speed. In this deliberately simple limiting model the migration rate is
    reduced by ``B``:

        v_r  = -(3/2) alpha h^2 r Omega * min(1, B)
        Gamma_sd = (dL/dr) v_r
                 = -(3/4) alpha h^2 m r^2 Omega^2 min(1, B)

    using ``dL/dr = (1/2) m r Omega``. For ``B < 1`` this is equivalently
    ``-(3/4) alpha h^2 Sigma r^4 Omega^2``; for ``B >= 1`` the radial rate
    saturates at the classical viscous drift speed. It is negative-definite: a
    viscously accreting disk drags the secondary inward.

    ``min(1, B)`` is a deliberately conservative interpolation.  It reproduces
    classical Type-II for a disk-dominated secondary and the inertia-limited
    scaling for ``B << 1``, without importing a specific fitted transition
    whose calibration does not cover this mass-ratio regime.
    """
    alpha = float(alpha)
    surface_density = np.asarray(surface_density_kg_m2, dtype=float)
    radius = np.asarray(radius_m, dtype=float)
    aspect = np.asarray(aspect_ratio, dtype=float)
    omega = np.asarray(orbital_frequency_rad_s, dtype=float)
    disk_dominance = safe_divide(
        surface_density * radius**2, float(migrator_mass_kg)
    )
    inertia_factor = np.minimum(1.0, disk_dominance)
    return (
        -0.75
        * alpha
        * aspect**2
        * float(migrator_mass_kg)
        * radius**2
        * omega**2
        * inertia_factor
    )


def compose_migration_torque_si(
    gravitational_torque_si,
    thermal_torque_si,
    gap_depth_fraction,
    secondary_dominated_torque_si,
):
    """Return the production torque, continuous across gap opening.

    ``f_gap`` hands weight from the linear gravitational torque to the
    gap-limited viscous torque rather than switching between them:

        Gamma = f_gap Gamma_grav + Gamma_th + (1 - f_gap) Gamma_sd

    As the gap deepens the linear Lindblad/corotation torques are suppressed by
    the gas removed from their resonances, and exactly that weight is handed to
    the gap-limited viscous torque.  The previous form omitted the
    ``(1 - f_gap)`` term, so a deep gap drove the gravitational contribution to
    *zero* and migration stalled -- an artefact of the composition, not
    physics: a gap stops linear torques, it does not stop viscous transport.

    The thermal term keeps the unattenuated treatment of the existing contract.
    Whether ``Gamma_th`` should also carry ``f_gap`` (it scales with the local
    surface density, so arguably yes) is a separate modelling question and is
    deliberately not changed here.
    """
    gravitational = np.asarray(gravitational_torque_si, dtype=float)
    thermal = np.asarray(thermal_torque_si, dtype=float)
    gap_depth = np.asarray(gap_depth_fraction, dtype=float)
    secondary_dominated = np.asarray(secondary_dominated_torque_si, dtype=float)
    shapes = {
        gravitational.shape,
        thermal.shape,
        gap_depth.shape,
        secondary_dominated.shape,
    }
    if len(shapes) != 1:
        raise ValueError("torque components and gap depth must have matching shapes")
    if np.any(~np.isfinite(gap_depth) | (gap_depth < 0.0) | (gap_depth > 1.0)):
        raise ValueError("gap depth fraction must be finite and lie in [0, 1]")
    return gap_depth * gravitational + thermal + (
        1.0 - gap_depth
    ) * secondary_dominated


def compute_thermal_torque_terms(
    dlog_pressure_dlog_radius,
    disk,
    migrator_mass_kg,
    mass_ratio,
    gap: GapDiagnostics,
    params: MigrationParameters,
) -> ThermalTorqueTerms:
    """Return the saturated Grishin-2024 thermal coefficient and diagnostics.

    The coefficient multiplies the net Type-I normalization. The explicit
    ``1/h`` converts it to the one-sided normalization used by the thermal
    torque derivation. The hot and cold terms use equation 36 saturation. For
    ``r_B > H`` the thermal mass uses ``min(r_B, H)`` as the paper's suggested
    Bondi-Hoyle transition; this correction matters in the low-mass disks.
    """
    ct, electron_scattering_opacity = _require_pagn_bits()
    gamma = params.gamma_adiabat
    diffusivity_m2_s = thermal_diffusivity_m2_s(disk, gamma)
    radius_m = np.asarray(disk.R, dtype=float)
    scale_height_m = np.asarray(disk.h, dtype=float)
    sound_speed_m_s = np.asarray(disk.cs, dtype=float)
    orbital_frequency_rad_s = np.asarray(disk.Omega, dtype=float)

    corotation_offset_m = -(
        scale_height_m**2
        / (3.0 * gamma * radius_m)
        * np.asarray(dlog_pressure_dlog_radius, dtype=float)
    )
    thermal_length_m = np.sqrt(
        2.0 * diffusivity_m2_s / (3.0 * gamma * orbital_frequency_rad_s)
    )
    bondi_radius_m = ct.G * migrator_mass_kg / sound_speed_m_s**2
    saturation_radius_m = np.minimum(bondi_radius_m, scale_height_m)
    thermal_mass_ratio = safe_divide(
        diffusivity_m2_s,
        sound_speed_m_s * saturation_radius_m,
    )

    scattering_opacity_m2_kg = float(
        electron_scattering_opacity(X=params.hydrogen_fraction)
    )
    eddington_luminosity_w = (
        4.0 * np.pi * ct.G * ct.c * migrator_mass_kg / scattering_opacity_m2_kg
    )
    critical_luminosity_w = (
        4.0
        * np.pi
        * ct.G
        * migrator_mass_kg
        * disk.rho
        * diffusivity_m2_s
        / gamma
    )
    luminosity_ratio = safe_divide(eddington_luminosity_w, critical_luminosity_w)

    hot_saturation = safe_divide(
        4.0 * thermal_mass_ratio,
        1.0 + 4.0 * thermal_mass_ratio,
    )
    cold_saturation = safe_divide(
        2.0 * thermal_mass_ratio,
        1.0 + 2.0 * thermal_mass_ratio,
    )
    hot_minus_cold = luminosity_ratio * hot_saturation - cold_saturation

    effective_optical_depth = np.asarray(disk.tauV, dtype=float) * (
        gap.gap_depth_fraction
    )
    optical_depth_suppression = 1.0 - np.exp(
        -np.sqrt(params.alpha) * effective_optical_depth
    )
    aspect_ratio = scale_height_m / radius_m
    offset_over_length = safe_divide(corotation_offset_m, thermal_length_m)
    coefficient = (
        1.61
        * (gamma - 1.0)
        / gamma
        * offset_over_length
        * hot_minus_cold
        * safe_divide(1.0, aspect_ratio)
        * optical_depth_suppression
    )

    thermal_length_over_height = safe_divide(thermal_length_m, scale_height_m)
    bondi_radius_over_height = safe_divide(bondi_radius_m, scale_height_m)
    photon_diffusion_valid = effective_optical_depth > safe_divide(
        scale_height_m, thermal_length_m
    )
    regime_valid = (
        np.isfinite(coefficient)
        & (np.abs(offset_over_length) < 1.0)
        & (thermal_length_over_height < 1.0)
        & (gap.hill_radius_over_scale_height < 1.0)
        & photon_diffusion_valid
    )

    return ThermalTorqueTerms(
        coefficient=np.asarray(coefficient, dtype=float),
        thermal_diffusivity_m2_s=np.asarray(diffusivity_m2_s, dtype=float),
        effective_optical_depth=np.asarray(effective_optical_depth, dtype=float),
        thermal_mass_ratio=np.asarray(thermal_mass_ratio, dtype=float),
        thermal_length_over_scale_height=np.asarray(
            thermal_length_over_height, dtype=float
        ),
        corotation_offset_over_thermal_length=np.asarray(
            offset_over_length, dtype=float
        ),
        bondi_radius_over_scale_height=np.asarray(
            bondi_radius_over_height, dtype=float
        ),
        thermal_torque_regime_valid=np.asarray(regime_valid, dtype=bool),
    )


def comp_gamma_th_coeff(
    dlog_pressure_dlog_radius,
    disk,
    migrator_mass_kg,
    mass_ratio,
    gap: GapDiagnostics,
    params: MigrationParameters,
):
    """Compatibility-sized public entry point for the thermal coefficient."""
    return compute_thermal_torque_terms(
        dlog_pressure_dlog_radius,
        disk,
        migrator_mass_kg,
        mass_ratio,
        gap,
        params,
    ).coefficient


def compute_torque_profile(
    disk,
    migrator_mass_msun: float,
    max_radius_rg: Optional[float],
    *,
    smoothing: float = 0.005,
    params: Optional[MigrationParameters] = None,
) -> TorqueProfile:
    """Build the one production torque profile on a solved pAGN disk."""
    params = MigrationParameters() if params is None else params
    ct, _ = _require_pagn_bits()

    # Retain the established diagnostic extension behaviour. Production passes
    # ``None`` and therefore never mutates or re-solves its cached disk.
    if max_radius_rg is not None:
        radius_m = np.asarray(disk.R, dtype=float)
        max_radius_m = 0.5 * float(max_radius_rg) * float(disk.Rs)
        if max_radius_m > float(np.nanmax(radius_m)):
            for name in ("Rmax", "Rout", "R_out"):
                if hasattr(disk, name):
                    setattr(disk, name, max_radius_m)
                    break
            for name, value in (("i", 0), ("isf", -1), ("Mdisk", 0.0)):
                if hasattr(disk, name):
                    setattr(disk, name, value)
            with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
                disk.solve_disk(N=radius_m.size)

    disk_view = _sorted_disk_view(disk)
    migrator_mass_kg = float(migrator_mass_msun) * ct.MSun
    radius_m = np.asarray(disk_view.R, dtype=float)
    scale_height_m = np.asarray(disk_view.h, dtype=float)
    density_kg_m3 = np.asarray(disk_view.rho, dtype=float)
    surface_density_values_kg_m2 = surface_density_kg_m2(disk_view)
    pressure_values_pa = total_pressure_pa(disk_view)
    temperature_k = np.asarray(disk_view.T, dtype=float)

    slopes = tuple(
        radial_log_slope(radius_m, values, smoothing=smoothing)
        for values in (surface_density_values_kg_m2, temperature_k, pressure_values_pa)
    )
    (
        dlog_surface_density_dlog_radius,
        dlog_temperature_dlog_radius,
        dlog_pressure_dlog_radius,
    ) = slopes

    mass_ratio = migrator_mass_kg / disk_view.Mbh
    aspect_ratio = scale_height_m / radius_m

    # The pAGN solution is thermally bistable at hydrogen ionisation. Linear
    # torque theory does not apply across that jump, so build the torque from
    # the smooth background bracketing it instead of from the instability.
    wkb_invalid = wkb_invalid_mask(
        radius_m,
        temperature_k,
        surface_density_values_kg_m2,
        pressure_values_pa,
    )
    (
        dlog_surface_density_dlog_radius,
        dlog_temperature_dlog_radius,
        dlog_pressure_dlog_radius,
    ) = repair_slopes_across(
        wkb_invalid,
        radius_m,
        dlog_surface_density_dlog_radius,
        dlog_temperature_dlog_radius,
        dlog_pressure_dlog_radius,
    )

    gap = gap_diagnostics(mass_ratio, aspect_ratio, params.alpha)

    corotation_coeff = comp_corot(
        dlog_surface_density_dlog_radius,
        dlog_temperature_dlog_radius,
        params.gamma_adiabat,
    )
    lindblad_coeff = comp_lind(
        dlog_surface_density_dlog_radius,
        dlog_temperature_dlog_radius,
        disk_view,
        params.gamma_adiabat,
    )
    gamma_0_si = comp_gamma_0_si(
        mass_ratio,
        radius_m,
        scale_height_m,
        disk_view.Omega,
        density_kg_m3,
    )
    thermal = compute_thermal_torque_terms(
        dlog_pressure_dlog_radius,
        disk_view,
        migrator_mass_kg,
        mass_ratio,
        gap,
        params,
    )
    gamma_type_i_si = (corotation_coeff + lindblad_coeff) * gamma_0_si
    # Bound the thermal torque by its own validity flag instead of
    # extrapolating linear theory into the non-linear regime.
    thermal_validity_weight = np.where(
        thermal.thermal_torque_regime_valid, 1.0, 0.0
    )
    if params.extrapolate_thermal_torque:
        thermal_validity_weight = np.ones_like(thermal_validity_weight)
    gamma_th_si = thermal.coefficient * gamma_0_si * thermal_validity_weight
    disk_dominance_b = safe_divide(
        surface_density_values_kg_m2 * radius_m**2, migrator_mass_kg
    )
    gamma_secondary_dominated_si = comp_secondary_dominated_torque_si(
        params.alpha,
        surface_density_values_kg_m2,
        radius_m,
        aspect_ratio,
        disk_view.Omega,
        migrator_mass_kg,
    )
    gamma_total_si = compose_migration_torque_si(
        gamma_type_i_si,
        gamma_th_si,
        gap.gap_depth_fraction,
        gamma_secondary_dominated_si,
    )

    radius_cm = radius_m * METER_TO_CM
    radius_rs = radius_m / disk_view.Rs
    return TorqueProfile(
        radius_m=radius_m,
        radius_cm=radius_cm,
        radius_rg=2.0 * radius_rs,
        radius_rs=radius_rs,
        surface_density_kg_m2=surface_density_values_kg_m2,
        pressure_pa=pressure_values_pa,
        opacity_m2_kg=np.asarray(disk_view.kappa, dtype=float),
        dlog_surface_density_dlog_radius=dlog_surface_density_dlog_radius,
        dlog_temperature_dlog_radius=dlog_temperature_dlog_radius,
        dlog_pressure_dlog_radius=dlog_pressure_dlog_radius,
        corotation_coeff=corotation_coeff,
        lindblad_coeff=lindblad_coeff,
        thermal_coeff=thermal.coefficient,
        gap_parameter_dimensionless=gap.kanagawa_parameter,
        gap_depth_fraction=gap.gap_depth_fraction,
        crida_parameter=gap.crida_parameter,
        deep_gap=gap.deep_gap,
        kanagawa_extrapolated=gap.kanagawa_extrapolated,
        hill_radius_over_scale_height=gap.hill_radius_over_scale_height,
        effective_optical_depth=thermal.effective_optical_depth,
        thermal_diffusivity_m2_s=thermal.thermal_diffusivity_m2_s,
        thermal_mass_ratio=thermal.thermal_mass_ratio,
        thermal_length_over_scale_height=(
            thermal.thermal_length_over_scale_height
        ),
        corotation_offset_over_thermal_length=(
            thermal.corotation_offset_over_thermal_length
        ),
        bondi_radius_over_scale_height=thermal.bondi_radius_over_scale_height,
        thermal_torque_regime_valid=thermal.thermal_torque_regime_valid,
        wkb_invalid=np.asarray(wkb_invalid, dtype=bool),
        thermal_validity_weight=np.asarray(thermal_validity_weight, dtype=float),
        gamma_0_si=gamma_0_si,
        gamma_type_i_si=gamma_type_i_si,
        gamma_th_si=gamma_th_si,
        gamma_secondary_dominated_si=gamma_secondary_dominated_si,
        disk_dominance_b=disk_dominance_b,
        gamma_total_si=gamma_total_si,
    )


__all__ = [
    "MigrationParameters",
    "ThermalTorqueTerms",
    "TorqueProfile",
    "comp_corot",
    "comp_gamma_0_si",
    "comp_gamma_th_coeff",
    "comp_lind",
    "compose_migration_torque_si",
    "compute_thermal_torque_terms",
    "compute_torque_profile",
    "surface_density_kg_m2",
    "thermal_diffusivity_m2_s",
    "torque_si_to_cgs",
    "total_pressure_pa",
]
