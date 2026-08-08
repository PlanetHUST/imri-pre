"""pAGN-style migration torque profile and torque-trap finder."""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import numpy as np

METER_TO_CM = 100.0
DEFAULT_MIN_TRAP_SEPARATION_CELLS = 1.0
STABLE_TRAP_KIND = 1
ANTI_TRAP_KIND = -1


@dataclass(frozen=True)
class MigrationParameters:
    alpha: float = 0.01
    gamma_adiabat: float = 5.0 / 3.0
    hydrogen_fraction: float = 0.7
    kappa_t: float = 20.0
    kappa_norm: float = 25.0
    include_type_ii: bool = True


@dataclass
class TorqueProfile:
    radius: np.ndarray
    radius_cm: np.ndarray
    radius_rg: np.ndarray
    radius_rs: np.ndarray
    surface_density: np.ndarray
    pressure: np.ndarray
    dsigma_dlogr: np.ndarray
    dtemp_dlogr: np.ndarray
    dpressure_dlogr: np.ndarray
    corotation_coeff: np.ndarray
    lindblad_coeff: np.ndarray
    thermal_coeff: np.ndarray
    gap_parameter: np.ndarray
    gamma_0: np.ndarray
    gamma_i: np.ndarray
    gamma_th: np.ndarray
    gamma_type_ii: np.ndarray
    gamma_tot: np.ndarray
    tau_mig: np.ndarray
    tau_e: np.ndarray
    tau_i: np.ndarray


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
    radius = np.asarray(disk.R, dtype=float)
    order = np.argsort(radius)
    data = {"Mbh": disk.Mbh, "Rs": disk.Rs}
    for name in ["R", "rho", "T", "h", "cs", "Omega", "tauV", "kappa", "Teff4"]:
        arr = np.asarray(getattr(disk, name), dtype=float)
        if arr.shape == radius.shape:
            arr = arr[order]
        data[name] = arr
    return SimpleNamespace(**data)


def safe_divide(numerator, denominator):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    num, den = np.broadcast_arrays(numerator, denominator)
    out = np.zeros_like(num, dtype=float)
    return np.divide(num, den, out=out, where=den != 0.0)


def radial_log_slope(radius, values, *, smoothing: float = 0.005, k: int = 3):
    radius = np.asarray(radius, dtype=float)
    values = np.asarray(values, dtype=float)
    if np.any(radius <= 0.0) or np.any(values <= 0.0):
        raise ValueError("radius and values must be positive for log-slope")

    radius_log = np.log10(radius)
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


def total_pressure(disk):
    ct, _ = _require_pagn_bits()
    k_boltzmann = _constant(ct, "Kb", "kB", "k")
    mass_u = _constant(ct, "massU", "mU", "mp")
    gas_pressure = disk.rho * disk.T * k_boltzmann / mass_u
    radiation_pressure = disk.tauV * ct.sigmaSB * disk.Teff4 / (2.0 * ct.c)
    return gas_pressure + radiation_pressure


def surface_density(disk):
    return 2.0 * disk.rho * disk.h


def comp_corot(dsigma_dlogr, dtemp_dlogr, gamma_adiabat):
    return (0.46 + 0.96 * dsigma_dlogr - 1.8 * dtemp_dlogr) / gamma_adiabat


def thermal_diffusivity(disk, gamma_adiabat):
    ct, _ = _require_pagn_bits()
    return (
        16.0
        * gamma_adiabat
        * (gamma_adiabat - 1.0)
        * ct.sigmaSB
        * disk.T**4
        / (3.0 * disk.kappa * disk.rho**2 * disk.h**2 * disk.Omega**2)
    )


def comp_lind(dsigma_dlogr, dtemp_dlogr, disk, gamma_adiabat):
    chi = thermal_diffusivity(disk, gamma_adiabat)
    chi_c = disk.h**2 * disk.Omega
    x_sq = np.sqrt(safe_divide(chi, 2.0 * chi_c))
    f_gamma = (x_sq + 1.0 / gamma_adiabat) / (x_sq + 1.0)
    return (-2.34 - 0.1 * dsigma_dlogr + 1.5 * dtemp_dlogr) * f_gamma


def comp_kappa(q, h_over_r, alpha):
    return q**2 / (h_over_r**5 * alpha)


def comp_gamma_0(q, radius, scale_height, omega, rho):
    # return normalization torque Gamma_0 = (q/h)^2 * Sigma * r^4 * Omega^2
    sigma = 2.0 * rho * scale_height
    h_over_r = scale_height / radius
    return q**2 * radius**4 * omega**2 * sigma / (h_over_r**2)


def comp_gamma_th(dpressure_dlogr, disk, migrator_mass, params):
    # return the thermal torque coefficient
    ct, electron_scattering_opacity = _require_pagn_bits()
    gamma = params.gamma_adiabat  # adibatic index

    chi = thermal_diffusivity(disk, gamma)

    x_c = (
        np.abs(dpressure_dlogr) * disk.cs**2 / (3.0 * gamma * disk.R * disk.Omega**2)
    )  # corotational radius of the embedded object

    k_es = electron_scattering_opacity(X=params.hydrogen_fraction)

    l_edd = (
        4.0 * np.pi * ct.G * ct.c * migrator_mass / k_es
    )  # Eddington luminosity of the migrator
    l_crit = (
        4.0 * np.pi * ct.G * migrator_mass * disk.rho * chi / gamma
    )  # critical luminosity for saturation of the thermal torque

    lambda_c = np.sqrt(
        2.0 * chi / (3.0 * gamma * disk.Omega)
    )  # typical length scale for thermal diffusion in the corotation region

    bondi_radius = ct.G * migrator_mass / disk.cs**2
    mu_th = chi / (disk.cs * np.minimum(disk.h, bondi_radius))

    unsaturated = safe_divide(l_edd, l_crit) * (4.0 * mu_th / (1.0 + 4.0 * mu_th)) - (
        2.0 * mu_th / (1.0 + 2.0 * mu_th)
    )
    saturated = safe_divide(l_edd, l_crit) - 1.0

    prefactor = (
        1.61 * (gamma - 1.0) * safe_divide(x_c, gamma * lambda_c) * (disk.R / disk.h)
    )

    return np.where(mu_th < 1.0, prefactor * unsaturated, prefactor * saturated)


def comp_gamma_type_ii(alpha, migrator_mass, h_over_r, radius, omega):
    """Return inward Type-II (viscous-limited) migration torque [J].

    Gamma_type,ii = alpha * m * h^2 * R^2 * Omega^2

    pAGN stores mass in kg, radius in metres, and Omega in s^-1, so the
    returned torque uses the same SI units as the other torque terms. The
    migration integrator treats negative torque as inward angular-momentum
    loss, so this returns the signed inward torque.
    """
    return -alpha * migrator_mass * h_over_r**2 * radius**2 * omega**2


def compute_torque_profile(
    disk,
    *,
    migrator_mass_msun: float = 10.0,
    z: float = 0.0,
    smoothing: float = 0.005,
    params: Optional[MigrationParameters] = None,
) -> TorqueProfile:
    if params is None:
        params = MigrationParameters()

    ct, _ = _require_pagn_bits()
    disk_view = _sorted_disk_view(disk)
    migrator_mass = float(migrator_mass_msun) * ct.MSun

    sigma = surface_density(disk_view)
    pressure = total_pressure(disk_view)

    dsigma_dlogr = radial_log_slope(disk_view.R, sigma, smoothing=smoothing)
    dtemp_dlogr = radial_log_slope(disk_view.R, disk_view.T, smoothing=smoothing)
    dpressure_dlogr = radial_log_slope(disk_view.R, pressure, smoothing=smoothing)

    q = migrator_mass / disk_view.Mbh  # mass ratio of migrator to central black hole
    h_over_r = disk_view.h / disk_view.R  # aspect ratio of the disk at each radius

    dep_factor = np.exp(
        -0.5 * (float(z) / disk_view.h) ** 2
    )  # vertical dependence factor for the torque, assuming a Gaussian vertical density profile

    # compute coefficients for torques
    corot = comp_corot(dsigma_dlogr, dtemp_dlogr, params.gamma_adiabat)
    lind = comp_lind(dsigma_dlogr, dtemp_dlogr, disk_view, params.gamma_adiabat)
    kappa_gap = comp_kappa(q, h_over_r, params.alpha)
    gamma_0 = comp_gamma_0(q, disk_view.R, disk_view.h, disk_view.Omega, disk_view.rho)
    gamma_th_coeff = comp_gamma_th(dpressure_dlogr, disk_view, migrator_mass, params)

    gap_exp = np.exp(-kappa_gap / params.kappa_t)
    gamma_i = (
        (corot * gap_exp + lind) / (1.0 + kappa_gap / params.kappa_norm) * gamma_0
    )  # Lindblad torque
    gamma_th = gamma_0 * gamma_th_coeff  # thermal torque
    if params.include_type_ii:
        gamma_type_ii = comp_gamma_type_ii(
            params.alpha, migrator_mass, h_over_r, disk_view.R, disk_view.Omega
        )
    else:
        gamma_type_ii = np.zeros_like(gamma_i, dtype=float)
    gamma_tot = gamma_i + gap_exp * gamma_th + gamma_type_ii

    orbital_angular_momentum = migrator_mass * disk_view.Omega * disk_view.R**2

    tau_mig = np.full_like(gamma_tot, np.inf, dtype=float)
    finite_mask = (gamma_tot != 0.0) & np.isfinite(gamma_tot) & (dep_factor > 0.0)
    tau_mig[finite_mask] = orbital_angular_momentum[finite_mask] / (
        2.0 * gamma_tot[finite_mask] * dep_factor[finite_mask]
    )  # differential migration timescale, accounting for vertical dependence

    tau_e = np.abs(tau_mig) * h_over_r**2 / 0.780
    tau_i = np.abs(tau_mig) * h_over_r**2 / 0.544

    radius_cm = disk_view.R * METER_TO_CM
    radius_rs = disk_view.R / disk_view.Rs
    radius_rg = radius_rs * 2.0

    return TorqueProfile(
        radius=disk_view.R,
        radius_cm=radius_cm,
        radius_rg=radius_rg,
        radius_rs=radius_rs,
        surface_density=sigma,
        pressure=pressure,
        dsigma_dlogr=dsigma_dlogr,
        dtemp_dlogr=dtemp_dlogr,
        dpressure_dlogr=dpressure_dlogr,
        corotation_coeff=corot,
        lindblad_coeff=lind,
        thermal_coeff=gamma_th_coeff,
        gap_parameter=kappa_gap,
        gamma_0=gamma_0,
        gamma_i=gamma_i,
        gamma_th=gamma_th,
        gamma_type_ii=gamma_type_ii,
        gamma_tot=gamma_tot,
        tau_mig=tau_mig,
        tau_e=tau_e,
        tau_i=tau_i,
    )


def _resolved_crossing_mask(
    root_radius: np.ndarray,
    crossing_kind: np.ndarray,
    local_width: np.ndarray,
    min_separation_cells: float,
) -> np.ndarray:
    keep = np.ones(root_radius.size, dtype=bool)
    if min_separation_cells <= 0.0 or root_radius.size < 2:
        return keep

    group = [0]

    def flush_group() -> None:
        if len(group) < 2:
            return
        group_kind = crossing_kind[np.asarray(group, dtype=int)]
        if np.unique(group_kind).size < 2:
            return
        for group_index in group:
            keep[group_index] = False

    for crossing_index in range(1, root_radius.size):
        previous_index = group[-1]
        separation = root_radius[crossing_index] - root_radius[previous_index]
        local_resolution = min(
            local_width[crossing_index],
            local_width[previous_index],
        )
        if separation < min_separation_cells * local_resolution:
            group.append(crossing_index)
            continue
        flush_group()
        group = [crossing_index]
    flush_group()
    return keep


def classify_torque_traps(
    radius: np.ndarray,
    gamma_tot: np.ndarray,
    *,
    min_separation_cells: float = DEFAULT_MIN_TRAP_SEPARATION_CELLS,
) -> TorqueTrapRadii:
    """Return stable and anti-trap torque zero radii [cm].

    Radius is sorted outward. A stable trap has positive torque just inside the
    zero and negative torque just outside it, so objects are pushed toward the
    zero from both sides. An anti-trap has the opposite sign ordering. Adjacent
    opposite-type roots closer than ``min_separation_cells`` local grid
    intervals are treated as unresolved numerical sign wiggles and discarded.
    """
    radius = np.asarray(radius, dtype=float)
    values = np.asarray(gamma_tot, dtype=float)
    min_separation_cells = float(min_separation_cells)

    finite = np.isfinite(radius) & np.isfinite(values) & (values != 0.0)
    radius = radius[finite]
    values = values[finite]
    if values.size < 2:
        empty = np.array([], dtype=float)
        return TorqueTrapRadii(stable_radius_cm=empty, anti_trap_radius_cm=empty)

    order = np.argsort(radius)
    radius = radius[order]
    values = values[order]

    root_radius = []
    crossing_kind = []
    local_width = []
    for index in np.where(np.sign(values[1:]) != np.sign(values[:-1]))[0]:
        inner_radius, outer_radius = radius[index], radius[index + 1]
        inner_torque, outer_torque = values[index], values[index + 1]
        root = inner_radius - inner_torque * (
            outer_radius - inner_radius
        ) / (outer_torque - inner_torque)
        if inner_torque > 0.0 and outer_torque < 0.0:
            kind = STABLE_TRAP_KIND
        elif inner_torque < 0.0 and outer_torque > 0.0:
            kind = ANTI_TRAP_KIND
        else:
            continue
        root_radius.append(root)
        crossing_kind.append(kind)
        local_width.append(outer_radius - inner_radius)

    if not root_radius:
        empty = np.array([], dtype=float)
        return TorqueTrapRadii(stable_radius_cm=empty, anti_trap_radius_cm=empty)

    root_radius_array = np.asarray(root_radius, dtype=float)
    crossing_kind_array = np.asarray(crossing_kind, dtype=int)
    local_width_array = np.asarray(local_width, dtype=float)
    keep = _resolved_crossing_mask(
        root_radius_array,
        crossing_kind_array,
        local_width_array,
        min_separation_cells,
    )

    return TorqueTrapRadii(
        stable_radius_cm=root_radius_array[
            keep & (crossing_kind_array == STABLE_TRAP_KIND)
        ],
        anti_trap_radius_cm=root_radius_array[
            keep & (crossing_kind_array == ANTI_TRAP_KIND)
        ],
    )


def find_stable_traps(
    torque_profile: TorqueProfile,
    *,
    min_separation_cells: float = DEFAULT_MIN_TRAP_SEPARATION_CELLS,
) -> np.ndarray:
    """Return stable trap radii [cm], where migration flows toward the zero."""
    trap_radii = classify_torque_traps(
        torque_profile.radius_cm,
        torque_profile.gamma_tot,
        min_separation_cells=min_separation_cells,
    )
    return trap_radii.stable_radius_cm


def find_anti_traps(
    torque_profile: TorqueProfile,
    *,
    min_separation_cells: float = DEFAULT_MIN_TRAP_SEPARATION_CELLS,
) -> np.ndarray:
    """Return anti-trap radii [cm], where migration flows away from the zero."""
    return classify_torque_traps(
        torque_profile.radius_cm,
        torque_profile.gamma_tot,
        min_separation_cells=min_separation_cells,
    ).anti_trap_radius_cm


def classify_torque_profile_traps(
    torque_profile: TorqueProfile,
    *,
    min_separation_cells: float = DEFAULT_MIN_TRAP_SEPARATION_CELLS,
) -> TorqueTrapRadii:
    """Return stable and anti-trap radii [cm] for a torque profile."""
    return classify_torque_traps(
        torque_profile.radius_cm,
        torque_profile.gamma_tot,
        min_separation_cells=min_separation_cells,
    )
