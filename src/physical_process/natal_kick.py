"""Keplerian natal-kick transform, migrated from adv-damp (MIT).

The legacy transform constants are retained for numerical parity with existing
SEVN runs; they differ slightly from the project-wide damping constants.
"""

import numpy as np
import numpy.typing as npt

from .damp import OrbitalState

FloatVector = npt.NDArray[np.float64]
KICK_GRAVITATIONAL_CONSTANT_CGS = 6.67430e-8
KICK_SOLAR_MASS_G = 1.988409870698051e33


def true_anomaly_from_mean_anomaly(
    mean_anomaly_rad: float,
    eccentricity: float,
    tolerance: float = 1.0e-12,
    max_iterations: int = 60,
) -> float:
    """Solve Kepler's equation ``M = E - e sin E`` and return the true anomaly.

    Newton-Raphson starts from ``E_0 = M`` with at most 60 iterations. Sampling a *mean* anomaly uniformly is the time-weighted way to pick
    where in its orbit a star happens to be when it collapses; sampling the true
    anomaly uniformly would over-weight periapsis.
    """
    if not 0.0 <= eccentricity < 1.0:
        raise ValueError("eccentricity must satisfy 0 <= e < 1")
    eccentric_anomaly = float(mean_anomaly_rad)
    for _ in range(max_iterations):
        residual = (
            eccentric_anomaly
            - eccentricity * np.sin(eccentric_anomaly)
            - mean_anomaly_rad
        )
        derivative = 1.0 - eccentricity * np.cos(eccentric_anomaly)
        step = residual / derivative
        eccentric_anomaly -= step
        if abs(step) < tolerance:
            break
    half = eccentric_anomaly / 2.0
    return float(
        2.0
        * np.arctan2(
            np.sqrt(1.0 + eccentricity) * np.sin(half),
            np.sqrt(1.0 - eccentricity) * np.cos(half),
        )
    )


def state_vectors(
    state: OrbitalState,
    smbh_mass_g: float,
    true_anomaly_rad: float,
    argument_of_periapsis_rad: float,
) -> tuple[FloatVector, FloatVector]:
    """Return Cartesian ``(position_cm, velocity_cm_s)``.

    The longitude of the ascending node is taken to be zero and the disk lies
    in the ``z = 0`` plane.
    """
    mu = KICK_GRAVITATIONAL_CONSTANT_CGS * smbh_mass_g
    e = state.eccentricity
    semi_latus_rectum = state.semi_major_axis_cm * (1.0 - e**2)
    radius = semi_latus_rectum / (1.0 + e * np.cos(true_anomaly_rad))

    position_perifocal = np.array(
        [radius * np.cos(true_anomaly_rad), radius * np.sin(true_anomaly_rad), 0.0]
    )
    velocity_perifocal = np.sqrt(mu / semi_latus_rectum) * np.array(
        [-np.sin(true_anomaly_rad), e + np.cos(true_anomaly_rad), 0.0]
    )

    cos_w, sin_w = np.cos(argument_of_periapsis_rad), np.sin(argument_of_periapsis_rad)
    cos_i, sin_i = state.cos_inclination, state.sin_inclination
    rotate_periapsis = np.array(
        [[cos_w, -sin_w, 0.0], [sin_w, cos_w, 0.0], [0.0, 0.0, 1.0]]
    )
    rotate_inclination = np.array(
        [[1.0, 0.0, 0.0], [0.0, cos_i, -sin_i], [0.0, sin_i, cos_i]]
    )
    rotation = rotate_inclination @ rotate_periapsis
    return rotation @ position_perifocal, rotation @ velocity_perifocal


def elements_from_state_vectors(
    position_cm: FloatVector,
    velocity_cm_s: FloatVector,
    smbh_mass_g: float,
    time_s: float = 0.0,
) -> OrbitalState | None:
    """Recover osculating elements, or ``None`` if the orbit is unbound.

    Returning ``None`` for an unbound orbit is how a natal kick that ejects the
    remnant from the nuclear star cluster is represented.
    """
    mu = KICK_GRAVITATIONAL_CONSTANT_CGS * smbh_mass_g
    radius = float(np.linalg.norm(position_cm))
    speed_squared = float(velocity_cm_s @ velocity_cm_s)
    if radius <= 0.0:
        return None

    specific_energy = 0.5 * speed_squared - mu / radius
    if specific_energy >= 0.0:
        return None
    semi_major_axis = -mu / (2.0 * specific_energy)

    angular_momentum = np.cross(position_cm, velocity_cm_s)
    angular_momentum_norm = float(np.linalg.norm(angular_momentum))
    if angular_momentum_norm <= 0.0:
        return None

    eccentricity_vector = (
        (speed_squared - mu / radius) * position_cm
        - float(position_cm @ velocity_cm_s) * velocity_cm_s
    ) / mu
    eccentricity = float(np.linalg.norm(eccentricity_vector))
    if eccentricity >= 1.0:
        return None

    inclination = float(
        np.arccos(np.clip(angular_momentum[2] / angular_momentum_norm, -1.0, 1.0))
    )

    # Node vector n = z_hat x h points along the line of nodes.
    node_vector = np.array(
        [-angular_momentum[1], angular_momentum[0], 0.0], dtype=float
    )
    node_norm = float(np.linalg.norm(node_vector))
    if node_norm <= 0.0 or eccentricity <= 0.0:
        # Equatorial or circular: the argument of periapsis is undefined;
        # preserve the original convention cos(omega) = 1.
        cos_argument_of_periapsis = 1.0
    else:
        cos_argument_of_periapsis = float(
            np.clip(
                (node_vector @ eccentricity_vector) / (node_norm * eccentricity),
                -1.0,
                1.0,
            )
        )

    return OrbitalState(
        semi_major_axis_cm=float(semi_major_axis),
        eccentricity=eccentricity,
        inclination_rad=inclination,
        cos_omega=cos_argument_of_periapsis,
        time_s=float(time_s),
    )


def apply_orbital_kick(
    state: OrbitalState,
    kick_velocity_cm_s: tuple[float, float, float],
    smbh_mass_msun: float,
    rng: np.random.Generator,
) -> OrbitalState | None:
    """Apply an instantaneous velocity kick at a time-weighted orbital phase."""
    if all(component == 0.0 for component in kick_velocity_cm_s):
        return state
    smbh_mass_g = smbh_mass_msun * KICK_SOLAR_MASS_G
    argument_of_periapsis = float(np.arccos(np.clip(state.cos_omega, -1.0, 1.0)))
    mean_anomaly = float(rng.uniform(0.0, 2.0 * np.pi))
    true_anomaly = true_anomaly_from_mean_anomaly(mean_anomaly, state.eccentricity)
    position_cm, velocity_cm_s = state_vectors(
        state, smbh_mass_g, true_anomaly, argument_of_periapsis
    )
    kicked_velocity = velocity_cm_s + np.asarray(kick_velocity_cm_s, dtype=float)
    return elements_from_state_vectors(
        position_cm, kicked_velocity, smbh_mass_g, time_s=state.time_s
    )
