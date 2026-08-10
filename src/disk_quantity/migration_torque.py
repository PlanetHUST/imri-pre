"""pAGN-style migration torque profile and torque-trap finder."""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import numpy as np

METER_TO_CM = 100.0
SI_TORQUE_TO_CGS = 1.0e7
DEFAULT_MIN_TRAP_SEPARATION_CELLS = 1.0
STABLE_TRAP_KIND = 1
ANTI_TRAP_KIND = -1


def torque_si_to_cgs(torque_si: np.ndarray) -> np.ndarray:
    """Convert torque from pAGN SI units [kg m^2 s^-2] to CGS."""
    return np.asarray(torque_si, dtype=float) * SI_TORQUE_TO_CGS


@dataclass(frozen=True)
class MigrationParameters:
    alpha: float = 0.01
    gamma_adiabat: float = 5.0 / 3.0
    hydrogen_fraction: float = 0.7
    kappa_t: float = 20.0
    kappa_norm: float = 25.0


@dataclass
class TorqueProfile:
    """Torque profile arrays.

    pAGN solves in SI. Fields ending in ``_si`` are torques [kg m^2 s^-2].
    Radius aliases ending in ``_cm``, ``_rg``, and ``_rs`` are for downstream
    CGS migration code and plotting. ``gamma_th_si`` is the raw thermal torque
    component; multiply it by ``gap_attenuation_factor`` when composing the
    pAGN Type-I plus thermal torque.
    """

    radius_m: np.ndarray
    radius_cm: np.ndarray
    radius_rg: np.ndarray
    radius_rs: np.ndarray
    surface_density_kg_m2: np.ndarray
    pressure_pa: np.ndarray
    dlog_surface_density_dlog_radius: np.ndarray
    dlog_temperature_dlog_radius: np.ndarray
    dlog_pressure_dlog_radius: np.ndarray
    corotation_coeff: np.ndarray
    lindblad_coeff: np.ndarray
    thermal_coeff: np.ndarray
    gap_parameter_dimensionless: np.ndarray
    gap_attenuation_factor: np.ndarray
    type_i_gap_denominator: np.ndarray
    gamma_0_si: np.ndarray
    gamma_i_si: np.ndarray
    gamma_th_si: np.ndarray
    gamma_type_ii_si: np.ndarray


@dataclass(frozen=True)
class TorqueTrapRadii:
    """Torque zero radii separated by stable and anti-trap behavior."""

    stable_radius_cm: np.ndarray
    anti_trap_radius_cm: np.ndarray


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
            return getattr(ct, name)
    raise AttributeError(f"pagn.constants is missing one of: {', '.join(names)}")


def _sorted_disk_view(disk) -> SimpleNamespace:
    radius_m = np.asarray(disk.R, dtype=float)
    order = np.argsort(radius_m)
    data = {"Mbh": disk.Mbh, "Rs": disk.Rs}
    for name in ["R", "rho", "T", "h", "cs", "Omega", "tauV", "kappa", "Teff4"]:
        arr = np.asarray(getattr(disk, name), dtype=float)
        if arr.shape == radius_m.shape:
            arr = arr[order]
        data[name] = arr
    return SimpleNamespace(**data)


def safe_divide(numerator, denominator):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    num, den = np.broadcast_arrays(numerator, denominator)
    out = np.zeros_like(num, dtype=float)
    return np.divide(num, den, out=out, where=den != 0.0)


def radial_log_slope(radius_m, values, *, smoothing: float = 0.005, k: int = 3):
    radius_m = np.asarray(radius_m, dtype=float)
    values = np.asarray(values, dtype=float)
    if np.any(radius_m <= 0.0) or np.any(values <= 0.0):
        raise ValueError("radius and values must be positive for log-slope")

    radius_log = np.log10(radius_m)
    values_log = np.log10(values)

    try:
        from scipy.interpolate import UnivariateSpline
    except ImportError as exc:
        raise ImportError(
            "scipy is required for smoothed torque-profile slopes. Install scipy "
            "or install pAGN with its scientific Python dependencies."
        ) from exc

    spline_order = min(int(k), radius_log.size - 1)
    spline = UnivariateSpline(
        radius_log, values_log, k=spline_order, s=float(smoothing), ext=0
    )
    return np.asarray(spline.derivative()(radius_log), dtype=float)


def total_pressure_pa(disk):
    ct, _ = _require_pagn_bits()
    k_boltzmann = _constant(ct, "Kb", "kB", "k")
    mass_u = _constant(ct, "massU", "mU", "mp")
    density_kg_m3 = disk.rho
    temperature_k = disk.T
    optical_depth = disk.tauV
    effective_temperature_fourth_k4 = disk.Teff4
    gas_pressure_pa = density_kg_m3 * temperature_k * k_boltzmann / mass_u
    radiation_pressure_pa = (
        optical_depth * ct.sigmaSB * effective_temperature_fourth_k4 / (2.0 * ct.c)
    )
    return gas_pressure_pa + radiation_pressure_pa


def surface_density_kg_m2(disk):
    density_kg_m3 = disk.rho
    scale_height_m = disk.h
    return 2.0 * density_kg_m3 * scale_height_m


def comp_corot(
    dlog_surface_density_dlog_radius,
    dlog_temperature_dlog_radius,
    gamma_adiabat,
):
    return (
        0.46
        + 0.96 * dlog_surface_density_dlog_radius
        - 1.8 * dlog_temperature_dlog_radius
    ) / gamma_adiabat


def thermal_diffusivity_m2_s(disk, gamma_adiabat):
    ct, _ = _require_pagn_bits()
    temperature_k = disk.T
    opacity_m2_kg = disk.kappa
    density_kg_m3 = disk.rho
    scale_height_m = disk.h
    orbital_frequency_rad_s = disk.Omega
    return (
        16.0
        * gamma_adiabat
        * (gamma_adiabat - 1.0)
        * ct.sigmaSB
        * temperature_k**4
        / (
            3.0
            * opacity_m2_kg
            * density_kg_m3**2
            * scale_height_m**2
            * orbital_frequency_rad_s**2
        )
    )


def comp_lind(
    dlog_surface_density_dlog_radius,
    dlog_temperature_dlog_radius,
    disk,
    gamma_adiabat,
):
    thermal_diffusivity_values_m2_s = thermal_diffusivity_m2_s(disk, gamma_adiabat)
    scale_height_m = disk.h
    orbital_frequency_rad_s = disk.Omega
    critical_diffusivity_m2_s = scale_height_m**2 * orbital_frequency_rad_s
    x_sq = np.sqrt(
        safe_divide(thermal_diffusivity_values_m2_s, 2.0 * critical_diffusivity_m2_s)
    )
    f_gamma = (x_sq + 1.0 / gamma_adiabat) / (x_sq + 1.0)
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
    """Return Gamma_0 in SI torque units [kg m^2 s^-2]."""
    surface_density_kg_m2_values = 2.0 * density_kg_m3 * scale_height_m
    aspect_ratio = scale_height_m / radius_m
    return (
        mass_ratio**2
        * radius_m**4
        * orbital_frequency_rad_s**2
        * surface_density_kg_m2_values
        / aspect_ratio**2
    )


def comp_gap_parameter_dimensionless(mass_ratio, aspect_ratio, alpha):
    """Return the pAGN gap-opening parameter K = q^2 / ((h/R)^5 alpha)."""
    return mass_ratio**2 / (aspect_ratio**5 * alpha)


def comp_gamma_th_coeff(
    dlog_pressure_dlog_radius,
    disk,
    migrator_mass_kg,
    params,
):
    """Return the dimensionless thermal torque coefficient."""
    ct, electron_scattering_opacity = _require_pagn_bits()
    gamma_adiabat = params.gamma_adiabat

    thermal_diffusivity_values_m2_s = thermal_diffusivity_m2_s(disk, gamma_adiabat)
    sound_speed_m_s = disk.cs
    radius_m = disk.R
    scale_height_m = disk.h
    orbital_frequency_rad_s = disk.Omega
    density_kg_m3 = disk.rho

    corotation_radius_m = np.abs(dlog_pressure_dlog_radius) * sound_speed_m_s**2 / (
        3.0 * gamma_adiabat * radius_m * orbital_frequency_rad_s**2
    )

    electron_scattering_opacity_m2_kg = electron_scattering_opacity(
        X=params.hydrogen_fraction
    )

    eddington_luminosity_w = (
        4.0 * np.pi * ct.G * ct.c * migrator_mass_kg
        / electron_scattering_opacity_m2_kg
    )
    critical_luminosity_w = (
        4.0
        * np.pi
        * ct.G
        * migrator_mass_kg
        * density_kg_m3
        * thermal_diffusivity_values_m2_s
        / gamma_adiabat
    )

    thermal_length_m = np.sqrt(
        2.0
        * thermal_diffusivity_values_m2_s
        / (3.0 * gamma_adiabat * orbital_frequency_rad_s)
    )

    bondi_radius_m = ct.G * migrator_mass_kg / sound_speed_m_s**2

    thermal_saturation_parameter = thermal_diffusivity_values_m2_s / (
        sound_speed_m_s * np.minimum(scale_height_m, bondi_radius_m)
    )

    unsaturated = safe_divide(eddington_luminosity_w, critical_luminosity_w) * (
        4.0
        * thermal_saturation_parameter
        / (1.0 + 4.0 * thermal_saturation_parameter)
    ) - (
        2.0
        * thermal_saturation_parameter
        / (1.0 + 2.0 * thermal_saturation_parameter)
    )
    saturated = safe_divide(eddington_luminosity_w, critical_luminosity_w) - 1.0

    prefactor = (
        1.61
        * (gamma_adiabat - 1.0)
        * safe_divide(corotation_radius_m, gamma_adiabat * thermal_length_m)
        * (radius_m / scale_height_m)
    )

    return np.where(
        thermal_saturation_parameter < 1.0,
        prefactor * unsaturated,
        prefactor * saturated,
    )


def comp_gamma_type_ii_si(
    alpha,
    migrator_mass_kg,
    aspect_ratio,
    radius_m,
    orbital_frequency_rad_s,
):
    """Return Type-II migration torque in SI units [kg m^2 s^-2]."""
    return (
        -alpha
        * migrator_mass_kg
        * aspect_ratio**2
        * radius_m**2
        * orbital_frequency_rad_s**2
    )


def compute_torque_profile(
    disk,
    *,
    migrator_mass_msun: float = 10.0,
    smoothing: float = 0.005,
    params: Optional[MigrationParameters] = None,
) -> TorqueProfile:
    if params is None:
        params = MigrationParameters()

    ct, _ = _require_pagn_bits()
    disk_view = _sorted_disk_view(disk)
    migrator_mass_kg = float(migrator_mass_msun) * ct.MSun
    radius_m = disk_view.R
    scale_height_m = disk_view.h
    density_kg_m3 = disk_view.rho
    temperature_k = disk_view.T
    orbital_frequency_rad_s = disk_view.Omega

    surface_density_values_kg_m2 = surface_density_kg_m2(disk_view)
    pressure_values_pa = total_pressure_pa(disk_view)

    dlog_surface_density_dlog_radius = radial_log_slope(
        radius_m,
        surface_density_values_kg_m2,
        smoothing=smoothing,
    )
    dlog_temperature_dlog_radius = radial_log_slope(
        radius_m,
        temperature_k,
        smoothing=smoothing,
    )
    dlog_pressure_dlog_radius = radial_log_slope(
        radius_m,
        pressure_values_pa,
        smoothing=smoothing,
    )

    mass_ratio = migrator_mass_kg / disk_view.Mbh
    aspect_ratio = scale_height_m / radius_m
    gap_parameter_dimensionless = comp_gap_parameter_dimensionless(
        mass_ratio,
        aspect_ratio,
        params.alpha,
    )
    gap_attenuation_factor = np.exp(-gap_parameter_dimensionless / params.kappa_t)
    type_i_gap_denominator = 1.0 + gap_parameter_dimensionless / params.kappa_norm

    corot = comp_corot(
        dlog_surface_density_dlog_radius,
        dlog_temperature_dlog_radius,
        params.gamma_adiabat,
    )

    lind = comp_lind(
        dlog_surface_density_dlog_radius,
        dlog_temperature_dlog_radius,
        disk_view,
        params.gamma_adiabat,
    )

    gamma_0_si = comp_gamma_0_si(
        mass_ratio,
        radius_m,
        scale_height_m,
        orbital_frequency_rad_s,
        density_kg_m3,
    )

    gamma_th_coeff = comp_gamma_th_coeff(
        dlog_pressure_dlog_radius,
        disk_view,
        migrator_mass_kg,
        params,
    )

    gamma_i_si = (
        (corot * gap_attenuation_factor + lind)
        / type_i_gap_denominator
        * gamma_0_si
    )

    gamma_th_si = gamma_0_si * gamma_th_coeff

    gamma_type_ii_si = comp_gamma_type_ii_si(
        params.alpha,
        migrator_mass_kg,
        aspect_ratio,
        radius_m,
        orbital_frequency_rad_s,
    )

    radius_cm = radius_m * METER_TO_CM
    radius_rs = radius_m / disk_view.Rs
    radius_rg = radius_rs * 2.0

    return TorqueProfile(
        radius_m=radius_m,
        radius_cm=radius_cm,
        radius_rg=radius_rg,
        radius_rs=radius_rs,
        surface_density_kg_m2=surface_density_values_kg_m2,
        pressure_pa=pressure_values_pa,
        dlog_surface_density_dlog_radius=dlog_surface_density_dlog_radius,
        dlog_temperature_dlog_radius=dlog_temperature_dlog_radius,
        dlog_pressure_dlog_radius=dlog_pressure_dlog_radius,
        corotation_coeff=corot,
        lindblad_coeff=lind,
        thermal_coeff=gamma_th_coeff,
        gap_parameter_dimensionless=gap_parameter_dimensionless,
        gap_attenuation_factor=gap_attenuation_factor,
        type_i_gap_denominator=type_i_gap_denominator,
        gamma_0_si=gamma_0_si,
        gamma_i_si=gamma_i_si,
        gamma_th_si=gamma_th_si,
        gamma_type_ii_si=gamma_type_ii_si,
    )


#############################################


def classify_torque_traps(
    radius_cm: np.ndarray,
    torque_values: np.ndarray,
) -> TorqueTrapRadii:
    """Return trap and anti-trap radii [cm]."""
    radius_cm = np.asarray(radius_cm, dtype=float)
    values = np.asarray(torque_values, dtype=float)

    finite = np.isfinite(radius_cm) & np.isfinite(values) & (values != 0.0)
    radius_cm = radius_cm[finite]
    values = values[finite]

    if values.size < 2:
        empty = np.array([], dtype=float)
        return TorqueTrapRadii(stable_radius_cm=empty, anti_trap_radius_cm=empty)

    order = np.argsort(radius_cm)
    radius_cm = radius_cm[order]
    values = values[order]

    root_radius_cm = []
    crossing_kind = []

    # check in sign change
    for index in np.where(np.sign(values[1:]) != np.sign(values[:-1]))[0]:
        inner_radius_cm, outer_radius_cm = radius_cm[index], radius_cm[index + 1]
        inner_torque_value, outer_torque_value = values[index], values[index + 1]

        root_cm = inner_radius_cm - inner_torque_value * (
            outer_radius_cm - inner_radius_cm
        ) / (outer_torque_value - inner_torque_value)

        if inner_torque_value > 0.0 and outer_torque_value < 0.0:
            kind = STABLE_TRAP_KIND
        elif inner_torque_value < 0.0 and outer_torque_value > 0.0:
            kind = ANTI_TRAP_KIND
        else:
            continue

        root_radius_cm.append(root_cm)
        crossing_kind.append(kind)

    if not root_radius_cm:
        empty = np.array([], dtype=float)
        return TorqueTrapRadii(stable_radius_cm=empty, anti_trap_radius_cm=empty)

    root_radius_cm_array = np.asarray(root_radius_cm, dtype=float)
    crossing_kind_array = np.asarray(crossing_kind, dtype=int)

    return TorqueTrapRadii(
        stable_radius_cm=root_radius_cm_array[
            (crossing_kind_array == STABLE_TRAP_KIND)
        ],
        anti_trap_radius_cm=root_radius_cm_array[
            (crossing_kind_array == ANTI_TRAP_KIND)
        ],
    )


def find_traps_locations(
    torque_profile: TorqueProfile,
) -> tuple[np.ndarray, np.ndarray]:

    trap_radii = classify_torque_traps(
        torque_profile.radius_cm,
        torque_profile.gamma_i_si
        + torque_profile.gap_attenuation_factor * torque_profile.gamma_th_si,
    )
    return trap_radii.stable_radius_cm, trap_radii.anti_trap_radius_cm
