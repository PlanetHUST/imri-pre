"""NSC orbital sampling and relaxation helpers."""

import numpy as np
import numpy.typing as npt

try:
    import physical_process.constants as const
except ImportError:
    import constants as const

FloatArray = npt.NDArray[np.float64]

MIN_INCLINATION_RAD = np.pi / 18.0
MAX_INCLINATION_RAD = 17.0 * np.pi / 18.0
VELOCITY_DISPERSION_NORMALIZATION_CM_S = 2.3e5
VELOCITY_DISPERSION_MASS_EXPONENT = 1.0 / 4.38
RELAXATION_COEFFICIENT = 0.34
COULOMB_LOGARITHM = 10.0
RELAXATION_REFERENCE_STELLAR_MASS_G = const.msun


def initial_star(
    star_count: int,
    smbh_mass_msun: float,
    minimum_radius_cm: float,
    max_radius_cm: float,
    gamma: float = 7.0 / 4.0,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Sample orbital elements.

    Args:
        star_count: Number of stellar orbits.
        smbh_mass_msun: SMBH mass [Msun].
        minimum_radius_cm: Minimum semi-major axis [cm].
        max_radius_cm: Maximum semi-major axis [cm].
        gamma: Number-density power-law index.

    Returns:
        Semi-major axis [cm], eccentricity, inclination [rad], and
        ``cos(omega)``. The historical angular distributions are preserved.
    """
    if star_count < 0:
        raise ValueError("number of stars must be nonnegative")
    if smbh_mass_msun <= 0.0 or minimum_radius_cm <= 0.0 or max_radius_cm <= 0.0:
        raise ValueError("SMBH mass and orbital radii must be positive")

    volume_integral_exponent = 2.0 - gamma
    inverse_cdf_exponent = volume_integral_exponent + 1.0
    if max_radius_cm < minimum_radius_cm:
        raise ValueError("maximum radius must exceed the inner orbital radius")

    uniform_radius = np.random.random(star_count)
    semi_major_axis_cm = (
        minimum_radius_cm**inverse_cdf_exponent
        + uniform_radius
        * (
            max_radius_cm**inverse_cdf_exponent
            - minimum_radius_cm**inverse_cdf_exponent
        )
    ) ** (1.0 / inverse_cdf_exponent)
    eccentricity = np.sqrt(np.random.random(star_count))
    inclination_rad = np.random.uniform(
        MIN_INCLINATION_RAD,
        MAX_INCLINATION_RAD,
        star_count,
    )
    cos_omega = np.random.random(star_count)
    return semi_major_axis_cm, eccentricity, inclination_rad, cos_omega


def estimate_nsc_radius_cm(
    nsc_mass_msun: float,
    reference_radius_pc: float,
    reference_mass_msun: float,
    mass_slope: float,
) -> float:
    """Return the configured NSC mass-radius relation in centimeters."""
    nsc_radius_pc = (
        reference_radius_pc * (nsc_mass_msun / reference_mass_msun) ** mass_slope
    )
    return float(nsc_radius_pc * const.parsec)


def disk_crossing_mask(
    semi_major_axis_cm: FloatArray,
    eccentricity: FloatArray,
    cos_omega: FloatArray,
    disk_inner_radius_cm: float,
    disk_outer_radius_cm: float,
    use_disk_inner_radius: bool,
) -> npt.NDArray[np.bool_]:
    """Return objects whose orbital nodes intersect the selected disk annulus."""
    node_radius_1_cm, node_radius_2_cm = orbital_node_radii_cm(
        semi_major_axis_cm,
        eccentricity,
        cos_omega,
    )
    node_1_in_disk, node_2_in_disk = orbital_nodes_in_disk_annulus(
        node_radius_1_cm,
        node_radius_2_cm,
        disk_inner_radius_cm,
        disk_outer_radius_cm,
        use_disk_inner_radius,
    )
    return np.asarray(node_1_in_disk | node_2_in_disk, dtype=bool)


def orbital_node_radii_cm(
    semi_major_axis_cm: float | FloatArray,
    eccentricity: float | FloatArray,
    cos_omega: float | FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Return the two disk-plane orbital-node radii [cm]."""
    semi_major_axis = np.asarray(semi_major_axis_cm, dtype=float)
    eccentricity_values = np.asarray(eccentricity, dtype=float)
    cos_omega_values = np.asarray(cos_omega, dtype=float)
    semi_latus_rectum_cm = semi_major_axis * (1.0 - eccentricity_values**2)
    return (
        np.asarray(
            semi_latus_rectum_cm / (1.0 - eccentricity_values * cos_omega_values),
            dtype=float,
        ),
        np.asarray(
            semi_latus_rectum_cm / (1.0 + eccentricity_values * cos_omega_values),
            dtype=float,
        ),
    )


def orbital_nodes_in_disk_annulus(
    node_radius_1_cm: float | FloatArray,
    node_radius_2_cm: float | FloatArray,
    disk_inner_radius_cm: float,
    disk_outer_radius_cm: float,
    use_disk_inner_radius: bool,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """Return per-node membership in the selected physical disk annulus."""
    if disk_inner_radius_cm <= 0.0 or disk_outer_radius_cm <= 0.0:
        raise ValueError("disk radii must be positive")
    if disk_outer_radius_cm < disk_inner_radius_cm:
        raise ValueError("disk outer radius must not be smaller than its inner radius")

    def node_membership(radius_cm: float | FloatArray) -> npt.NDArray[np.bool_]:
        radius = np.asarray(radius_cm, dtype=float)
        within_outer_edge = np.isfinite(radius) & (radius <= disk_outer_radius_cm)
        if not use_disk_inner_radius:
            return np.asarray(within_outer_edge & (radius > 0.0), dtype=bool)
        return np.asarray(
            within_outer_edge & (radius >= disk_inner_radius_cm),
            dtype=bool,
        )

    return (
        node_membership(node_radius_1_cm),
        node_membership(node_radius_2_cm),
    )


def velocity_disp(smbh_mass_msun: float) -> float:
    """Return NSC velocity dispersion [cm/s] for SMBH mass [Msun]."""
    if smbh_mass_msun <= 0.0:
        raise ValueError("SMBH mass must be positive")
    return float(
        VELOCITY_DISPERSION_NORMALIZATION_CM_S
        * smbh_mass_msun**VELOCITY_DISPERSION_MASS_EXPONENT
    )


def number_R_MSC(
    smbh_mass_msun: float,
    stellar_mass_g: float,
    radius_cm: float,
    gamma: float = 1.3,
) -> float:
    """Return NSC stellar number density [cm^-3].

    ``smbh_mass_msun`` is in solar masses, stellar mass is in grams, and
    radius is in centimeters.
    """
    if smbh_mass_msun <= 0.0 or stellar_mass_g <= 0.0 or radius_cm <= 0.0:
        raise ValueError("masses and radius must be positive")
    smbh_mass_g = smbh_mass_msun * const.msun
    influence_radius_cm = const.G * smbh_mass_g / velocity_disp(smbh_mass_msun) ** 2
    return float(
        (3.0 - gamma)
        * smbh_mass_g
        / (2.0 * np.pi * stellar_mass_g)
        * influence_radius_cm ** (-3.0)
        * (radius_cm / influence_radius_cm) ** (-gamma)
    )


def relaxaion_timescale_s(smbh_mass_msun: float, radius_cm: float) -> float:
    """Return the two-body relaxation time [s] at radius ``radius_cm``."""
    number_density_cm3 = number_R_MSC(
        smbh_mass_msun,
        RELAXATION_REFERENCE_STELLAR_MASS_G,
        radius_cm,
    )
    relaxation_time_s = (
        RELAXATION_COEFFICIENT
        * velocity_disp(smbh_mass_msun) ** 3
        / (
            const.G**2
            * RELAXATION_REFERENCE_STELLAR_MASS_G**2
            * number_density_cm3
            * COULOMB_LOGARITHM
        )
    )
    return float(relaxation_time_s)
