"""One-dimensional AGN disk structure model.

Inputs use SMBH mass in solar masses and accretion rate in g/s. The solver
uses cgs internally and returns radius [cm], temperature [K], surface density
[g/cm^2], sound speed [cm/s], and density [g/cm^3].
"""

import numpy as np
import numpy.typing as npt

if __package__ and __package__.startswith("src."):
    from src.physical_process import constants as const
    from src.disk_quantity.opacity import findopacity
else:  # pragma: no cover - supports PYTHONPATH=src legacy imports.
    import physical_process.constants as const
    from disk_quantity.opacity import findopacity

FloatArray = npt.NDArray[np.float64]
DiskPointWithOpacity = tuple[float, float, float, float, float, float, float, float]
DiskPoint = tuple[float, float, float, float, float, float, float]
DiskProfiles = tuple[
    float,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
]

STEFAN_BOLTZMANN_CGS = 5.67e-5
BOLTZMANN_CONSTANT_CGS = 1.38e-16
PROTON_MASS_G = 1.67e-24
ELECTRON_SCATTERING_OPACITY_CM2_G = 0.34
TOOMRE_Q_MIN = 1.0
TEMPERATURE_RELATIVE_TOLERANCE = 1.0e-3
SURFACE_DENSITY_RELATIVE_TOLERANCE = 1.0e-3
DISK_VISCOSITY_ALPHA = 0.1
DISK_INNER_RADIUS_RG = 6.0
DISK_OUTER_RADIUS_PC = 1
DISK_RADIAL_GRID_SIZE = 1000
INITIAL_TEMPERATURE_K = 1.0e7
INITIAL_SURFACE_DENSITY_G_CM2 = 1.0e3
INITIAL_TOOMRE_Q = 10.0
OPACITY_TEMPERATURE_SCALE_K = 1.0e6
MAX_OUTER_SOLVER_ITERATIONS = 100

KRAMERS_OPACITY = 0
TABULATED_OPACITY = 1
PIECEWISE_OPACITY = 2
OPACITY_MODE = TABULATED_OPACITY


def find_closest_index(r_list: FloatArray, r: float) -> int:
    """Return the index of the radius [cm] closest to ``r`` [cm]."""
    radius_grid_cm = np.asarray(r_list, dtype=float)
    if radius_grid_cm.size == 0:
        raise ValueError("radius grid must not be empty")
    return int(np.argmin(np.abs(radius_grid_cm - r)))


def _opacity_cm2_g(temperature_k: float, density_g_cm3: float) -> float:
    """Return Rosseland mean opacity [cm^2/g] for the configured model."""
    if OPACITY_MODE == KRAMERS_OPACITY:
        absorption_opacity_cm2_g = 6.4e22 * density_g_cm3 * temperature_k ** (-3.5)
        return absorption_opacity_cm2_g + ELECTRON_SCATTERING_OPACITY_CM2_G
    if OPACITY_MODE == TABULATED_OPACITY:
        log_r_parameter = np.log10(
            density_g_cm3 / (temperature_k / OPACITY_TEMPERATURE_SCALE_K) ** 3
        )
        log_temperature_k = np.log10(temperature_k)
        return float(10 ** findopacity(log_temperature_k, log_r_parameter))
    if OPACITY_MODE == PIECEWISE_OPACITY:
        return fun_kappa(temperature_k, density_g_cm3)
    raise ValueError(f"unknown opacity mode: {OPACITY_MODE}")


def fun_disk_outer(
    r: float, Tc: float, Mbh_sun: float, Mdot: float
) -> DiskPointWithOpacity:
    """Solve one self-gravitating outer-disk point.

    Legacy parameter names are retained. Units are ``r`` [cm], ``Tc`` [K],
    ``Mbh_sun`` [Msun], and ``Mdot`` [g/s].
    """
    toomre_q = TOOMRE_Q_MIN
    orbital_frequency_rad_s = np.sqrt(const.G * Mbh_sun * const.msun / r**3)
    density_g_cm3 = orbital_frequency_rad_s**2 / (2 * np.pi * const.G * toomre_q)
    scale_height_cm = (
        Mdot
        * orbital_frequency_rad_s
        / (
            3
            * np.pi
            * DISK_VISCOSITY_ALPHA
            * 2
            * density_g_cm3
            * orbital_frequency_rad_s**2
        )
    ) ** (1.0 / 3.0)
    surface_density_g_cm2 = 2 * density_g_cm3 * scale_height_cm
    opacity_cm2_g = _opacity_cm2_g(Tc, density_g_cm3)
    optical_depth = opacity_cm2_g * surface_density_g_cm2 / 2
    effective_temperature_k = (
        Tc**4 / (3.0 / 8.0 * optical_depth + 1.0 / 2.0 + 1.0 / (4.0 * optical_depth))
    ) ** 0.25
    sound_speed_cm_s = scale_height_cm * orbital_frequency_rad_s
    gas_pressure_cgs = density_g_cm3 * BOLTZMANN_CONSTANT_CGS * Tc / PROTON_MASS_G
    radiation_pressure_cgs = (
        optical_depth / 2 * STEFAN_BOLTZMANN_CGS * effective_temperature_k**4 / const.c
    )
    pressure_sound_speed_cm_s = np.sqrt(
        (radiation_pressure_cgs + gas_pressure_cgs) / density_g_cm3
    )
    relative_error = sound_speed_cm_s / pressure_sound_speed_cm_s - 1
    return (
        float(relative_error),
        float(surface_density_g_cm2),
        toomre_q,
        float(optical_depth),
        float(scale_height_cm / r),
        float(sound_speed_cm_s),
        float(density_g_cm3),
        opacity_cm2_g,
    )


def fun_disk_in(
    r: float,
    Tc: float,
    Sigma: float,
    Mbh_sun: float,
    Mdot: float,
) -> DiskPointWithOpacity:
    """Solve one non-self-gravitating inner-disk point in cgs units."""
    orbital_frequency_rad_s = np.sqrt(const.G * Mbh_sun * const.msun / r**3)
    effective_temperature_k = (
        3 / (8 * np.pi) * Mdot * orbital_frequency_rad_s**2 / STEFAN_BOLTZMANN_CGS
    ) ** 0.25
    quadratic_a = 3.0 / 8.0
    quadratic_b = 0.5 - Tc**4 / effective_temperature_k**4
    quadratic_c = 0.25
    discriminant = quadratic_b**2 - 4 * quadratic_a * quadratic_c

    if discriminant < 0:
        return 0.0, Sigma, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0

    optical_depth = (-quadratic_b + np.sqrt(discriminant)) / (2 * quadratic_a)
    relative_error = 1.0
    surface_density_g_cm2 = Sigma
    scale_height_cm = np.nan
    density_g_cm3 = np.nan
    opacity_cm2_g = np.nan
    while abs(relative_error) > SURFACE_DENSITY_RELATIVE_TOLERANCE:
        surface_density_step_g_cm2 = surface_density_g_cm2 / 1.0e4
        scale_height_cm = np.sqrt(
            Mdot
            / (
                orbital_frequency_rad_s
                * 3
                * np.pi
                * DISK_VISCOSITY_ALPHA
                * surface_density_g_cm2
            )
        )
        density_g_cm3 = surface_density_g_cm2 / (2 * scale_height_cm)
        opacity_cm2_g = _opacity_cm2_g(Tc, density_g_cm3)
        relative_error = 0.5 * opacity_cm2_g * surface_density_g_cm2 / optical_depth - 1
        if relative_error > 0:
            surface_density_g_cm2 -= surface_density_step_g_cm2
        else:
            surface_density_g_cm2 += surface_density_step_g_cm2

    gas_pressure_cgs = density_g_cm3 * BOLTZMANN_CONSTANT_CGS * Tc / PROTON_MASS_G
    radiation_pressure_cgs = (
        optical_depth / 2 * STEFAN_BOLTZMANN_CGS * effective_temperature_k**4 / const.c
    )
    sound_speed_cm_s = np.sqrt(
        (radiation_pressure_cgs + gas_pressure_cgs) / density_g_cm3
    )
    relative_error = sound_speed_cm_s / scale_height_cm / orbital_frequency_rad_s - 1
    toomre_q = orbital_frequency_rad_s**2 / (2 * np.pi * const.G * density_g_cm3)
    return (
        float(relative_error),
        float(surface_density_g_cm2),
        float(toomre_q),
        float(optical_depth),
        float(scale_height_cm / r),
        float(sound_speed_cm_s),
        float(density_g_cm3),
        float(opacity_cm2_g),
    )


def fun_kappa(Tc: float, rho: float) -> float:
    """Return piecewise analytic opacity [cm^2/g].

    Legacy inputs are central temperature ``Tc`` [K] and density ``rho``
    [g/cm^3].
    """
    if Tc <= 166.810:
        coefficient, density_power, temperature_power = 2e-4, 0.0, 2.0
    elif Tc <= 202.677:
        coefficient, density_power, temperature_power = 2e16, 0.0, -7.0
    elif Tc <= 2286.77 * rho ** (2 / 49):
        coefficient, density_power, temperature_power = 0.1, 0.0, 0.5
    elif Tc <= 2029.76 * rho ** (1 / 81):
        coefficient, density_power, temperature_power = 2e81, 1.0, -24.0
    elif Tc <= 1e4 * rho ** (1 / 21):
        coefficient, density_power, temperature_power = 1e-8, 2 / 3, 3.0
    elif Tc <= 31195.2 * rho ** (4 / 75):
        coefficient, density_power, temperature_power = 1e-36, 1 / 3, 10.0
    elif Tc <= 1.79393e8 * rho ** (2 / 5):
        coefficient, density_power, temperature_power = 1.5e20, 1.0, -2.5
    else:
        coefficient, density_power, temperature_power = 0.348, 0.0, 0.0
    return float(coefficient * rho**density_power * Tc**temperature_power)


def fun_disk(
    r: float,
    Tc: float,
    Q: float,
    Sigma: float,
    Mbh_sun: float,
    Mdot: float,
) -> DiskPoint:
    """Iterate one disk radius to temperature convergence.

    Legacy inputs use radius [cm], temperature [K], surface density [g/cm^2],
    SMBH mass [Msun], and accretion rate [g/s].
    """
    temperature_k = Tc
    toomre_q = Q
    surface_density_g_cm2 = Sigma
    relative_error = 1.0
    outer_iterations = 0
    optical_depth = np.nan
    aspect_ratio = np.nan
    sound_speed_cm_s = np.nan
    density_g_cm3 = np.nan

    while abs(relative_error) >= TEMPERATURE_RELATIVE_TOLERANCE:
        temperature_step_k = temperature_k / 1.0e4
        if toomre_q <= TOOMRE_Q_MIN:
            temperature_step_k = temperature_k / 1.0e2
            (
                relative_error,
                surface_density_g_cm2,
                toomre_q,
                optical_depth,
                aspect_ratio,
                sound_speed_cm_s,
                density_g_cm3,
                _opacity_cm2_g_value,
            ) = fun_disk_outer(r, temperature_k, Mbh_sun, Mdot)
            temperature_k += (
                -temperature_step_k if relative_error < 0 else temperature_step_k
            )
            outer_iterations += 1
            if outer_iterations > MAX_OUTER_SOLVER_ITERATIONS:
                break
        else:
            (
                relative_error,
                surface_density_g_cm2,
                toomre_q,
                optical_depth,
                aspect_ratio,
                sound_speed_cm_s,
                density_g_cm3,
                _opacity_cm2_g_value,
            ) = fun_disk_in(
                r,
                temperature_k,
                surface_density_g_cm2,
                Mbh_sun,
                Mdot,
            )
            temperature_k += (
                -temperature_step_k if relative_error > 0 else temperature_step_k
            )
            if toomre_q <= TOOMRE_Q_MIN:
                (
                    relative_error,
                    surface_density_g_cm2,
                    toomre_q,
                    optical_depth,
                    aspect_ratio,
                    sound_speed_cm_s,
                    density_g_cm3,
                    _opacity_cm2_g_value,
                ) = fun_disk_outer(r, temperature_k, Mbh_sun, Mdot)

    return (
        float(temperature_k),
        float(toomre_q),
        float(surface_density_g_cm2),
        float(optical_depth),
        float(aspect_ratio),
        float(sound_speed_cm_s),
        float(density_g_cm3),
    )


def radial_structure(Mbh_sun: float, Mdot: float) -> DiskProfiles:
    """Calculate the radial AGN disk structure.

    Args:
        Mbh_sun: SMBH mass [Msun].
        Mdot: Mass accretion rate [g/s].

    Returns:
        Stable outer radius [cm], followed by arrays of radius [cm],
        temperature [K], Toomre Q, surface density [g/cm^2], optical depth,
        aspect ratio, sound speed [cm/s], and density [g/cm^3].
    """
    if Mbh_sun <= 0.0 or Mdot <= 0.0:
        raise ValueError("SMBH mass and accretion rate must be positive")
    smbh_mass_g = Mbh_sun * const.msun
    gravitational_radius_cm = const.G * smbh_mass_g / const.c**2
    minimum_radius_cm = DISK_INNER_RADIUS_RG * gravitational_radius_cm
    maximum_radius_cm = DISK_OUTER_RADIUS_PC * const.parsec
    radius_cm = np.logspace(
        np.log10(minimum_radius_cm),
        np.log10(maximum_radius_cm),
        DISK_RADIAL_GRID_SIZE,
    )

    temperature_k = np.zeros(DISK_RADIAL_GRID_SIZE)
    toomre_q = np.zeros(DISK_RADIAL_GRID_SIZE)
    surface_density_g_cm2 = np.zeros(DISK_RADIAL_GRID_SIZE)
    optical_depth = np.zeros(DISK_RADIAL_GRID_SIZE)
    aspect_ratio = np.zeros(DISK_RADIAL_GRID_SIZE)
    sound_speed_cm_s = np.zeros(DISK_RADIAL_GRID_SIZE)
    density_g_cm3 = np.zeros(DISK_RADIAL_GRID_SIZE)

    current_temperature_k = INITIAL_TEMPERATURE_K
    current_surface_density_g_cm2 = INITIAL_SURFACE_DENSITY_G_CM2
    current_toomre_q = INITIAL_TOOMRE_Q
    stable_outer_index = 0
    disk_is_stable = True

    for index, local_radius_cm in enumerate(radius_cm):
        (
            current_temperature_k,
            current_toomre_q,
            current_surface_density_g_cm2,
            local_optical_depth,
            local_aspect_ratio,
            local_sound_speed_cm_s,
            local_density_g_cm3,
        ) = fun_disk(
            float(local_radius_cm),
            current_temperature_k,
            current_toomre_q,
            current_surface_density_g_cm2,
            Mbh_sun,
            Mdot,
        )
        temperature_k[index] = current_temperature_k
        toomre_q[index] = current_toomre_q
        surface_density_g_cm2[index] = current_surface_density_g_cm2
        optical_depth[index] = local_optical_depth
        aspect_ratio[index] = local_aspect_ratio
        sound_speed_cm_s[index] = local_sound_speed_cm_s
        density_g_cm3[index] = local_density_g_cm3

        if disk_is_stable and current_toomre_q > TOOMRE_Q_MIN:
            stable_outer_index = index
        else:
            disk_is_stable = False

    return (
        float(radius_cm[stable_outer_index]),
        radius_cm,
        temperature_k,
        toomre_q,
        surface_density_g_cm2,
        optical_depth,
        aspect_ratio,
        sound_speed_cm_s,
        density_g_cm3,
    )


if __name__ == "__main__":
    smbh_mass_msun = 1.0e5
    accretion_rate_g_s = smbh_mass_msun * 1.44e18
    outer_radius_cm, *_profiles = radial_structure(smbh_mass_msun, accretion_rate_g_s)
    gravitational_radius_cm = const.G * smbh_mass_msun * const.msun / const.c**2
    print("#" * 10)
    print(
        f"the outer radius of the disk is {outer_radius_cm:.3e} cm = "
        f"{outer_radius_cm/const.parsec:.3e} pc = "
        f"{outer_radius_cm/gravitational_radius_cm:.3e} rg"
    )
    print("#" * 10)
