"""SEVN endpoint and local kick checks through the production adapter."""

from importlib.util import find_spec
import os
from pathlib import Path
from unittest import SkipTest

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from physical_process.damp import OrbitalState
from physical_process import constants as const
from physical_process.gw_frequency_radius import (
    optimistic_eccentric_band_time_myr,
    peters_time_to_lisa_band_myr,
)
from optional_models.sevn import apply_sevn_kick, evolve_stellar_endpoint


H_TABLE = Path(os.environ.get("IMRI_SEVN_H_TABLES", ""))
HE_TABLE = Path(os.environ.get("IMRI_SEVN_HE_TABLES", ""))
HAS_SEVN = (
    find_spec("sevnpy") is not None
    and bool(os.environ.get("IMRI_SEVN_H_TABLES"))
    and bool(os.environ.get("IMRI_SEVN_HE_TABLES"))
    and H_TABLE.is_dir() and HE_TABLE.is_dir()
)


def test_eccentric_postkick_screen_uses_pericenter_and_peters_evolution() -> None:
    rg_cm = const.G * 1.0e5 * const.msun / const.c**2
    axis_cm = 1000.0 * rg_cm
    # The high-e orbit already exceeds the generous pericenter-frequency
    # screening boundary, although its circular n=2 frequency is below it.
    assert optimistic_eccentric_band_time_myr(
        1.0e5, 30.0, axis_cm, 0.9, 0.001
    ) == 0.0
    assert peters_time_to_lisa_band_myr(
        1.0e5, 30.0, axis_cm, 0.001, 0.9, 0.0
    ) > 0.0
    # At zero eccentricity the screen reduces to the circular Peters result
    # for a threshold divided by the chosen harmonic factor.
    assert optimistic_eccentric_band_time_myr(
        1.0e5, 30.0, axis_cm, 0.0, 0.001
    ) == pytest.approx(
        peters_time_to_lisa_band_myr(
            1.0e5, 30.0, axis_cm, 0.001 / 4.0, 0.0, 0.0
        )
    )


def test_eccentric_screen_time_matches_direct_peters_ode() -> None:
    primary_mass_g = 1.0e5 * const.msun
    secondary_mass_g = 30.0 * const.msun
    rg_cm = const.G * primary_mass_g / const.c**2
    axis0_cm = 3000.0 * rg_cm
    e0 = 0.5
    beta = (
        (64.0 / 5.0) * const.G**3 * primary_mass_g * secondary_mass_g
        * (primary_mass_g + secondary_mass_g) / const.c**5
    )

    def peters_rate(_time_myr: float, state: np.ndarray) -> tuple[float, float]:
        axis_cm, e = state
        axis_rate_cm_s = -beta / axis_cm**3 * (
            1.0 + 73.0 / 24.0 * e**2 + 37.0 / 96.0 * e**4
        ) / (1.0 - e**2) ** 3.5
        e_rate_s = -(19.0 / 12.0) * beta * e / axis_cm**4 * (
            1.0 + 121.0 / 304.0 * e**2
        ) / (1.0 - e**2) ** 2.5
        return axis_rate_cm_s * const.Myr, e_rate_s * const.Myr

    def reaches_proxy(_time_myr: float, state: np.ndarray) -> float:
        axis_cm, e = state
        orbital_hz = np.sqrt(
            const.G * (primary_mass_g + secondary_mass_g) / axis_cm**3
        ) / (2.0 * np.pi)
        return 8.0 * orbital_hz * np.sqrt(1.0 + e) / (1.0 - e) ** 1.5 - 0.001

    reaches_proxy.terminal = True
    reaches_proxy.direction = 1
    direct = solve_ivp(
        peters_rate, (0.0, 100.0), (axis0_cm, e0),
        events=reaches_proxy, rtol=1.0e-10, atol=1.0e-12,
    )
    assert direct.success and len(direct.t_events[0]) == 1
    assert optimistic_eccentric_band_time_myr(
        1.0e5, 30.0, axis0_cm, e0, 0.001
    ) == pytest.approx(direct.t_events[0][0], rel=2.0e-5)


def test_sevn_pair_instability_is_physical_zero_remnant() -> None:
    if not HAS_SEVN:
        raise SkipTest("optional SEVN backend/tables not configured")
    endpoint = evolve_stellar_endpoint(
        source_id=249,
        zams_mass_msun=156.93437,
        metallicity_zsun=1.0e-5,
        task_seed=2688385916,
        hydrogen_tables=H_TABLE,
        helium_tables=HE_TABLE,
    )
    assert endpoint.remnant_type == -1
    assert endpoint.remnant_mass_msun == 0.0
    assert not endpoint.is_black_hole
    assert endpoint.presupernova_mass_msun > 0.0
    assert 0 <= endpoint.presupernova_phase <= 6


def test_same_sevn_kick_seed_gives_same_bound_orbit() -> None:
    initial = OrbitalState(1.0e15, 0.05, 0.04, 0.3, 2.0e13)
    first = apply_sevn_kick(
        initial, kick_speed_km_s=20.0, smbh_mass_msun=1.0e5, seed=812
    )
    second = apply_sevn_kick(
        initial, kick_speed_km_s=20.0, smbh_mass_msun=1.0e5, seed=812
    )
    assert first == second
    assert first is not None
    assert first != initial
    assert np.isfinite(first.semi_major_axis_cm)
    assert apply_sevn_kick(
        initial, kick_speed_km_s=0.0, smbh_mass_msun=1.0e5, seed=812
    ) == initial


def test_kick_cartesian_orbital_cases() -> None:
    from physical_process.natal_kick import (
        KICK_GRAVITATIONAL_CONSTANT_CGS,
        KICK_SOLAR_MASS_G,
        elements_from_state_vectors,
    )

    mass_g = 1.0e5 * KICK_SOLAR_MASS_G
    radius_cm = 1.0e15
    circular_speed = np.sqrt(KICK_GRAVITATIONAL_CONSTANT_CGS * mass_g / radius_cm)
    position = np.array([radius_cm, 0.0, 0.0])

    def orbit(vx: float, vy: float, vz: float) -> OrbitalState | None:
        return elements_from_state_vectors(
            position, circular_speed * np.array([vx, vy, vz]), mass_g
        )

    prograde = orbit(0.0, 1.1, 0.0)
    retrograde_impulse = orbit(0.0, 0.9, 0.0)
    radial = orbit(0.1, 1.0, 0.0)
    vertical = orbit(0.0, 1.0, 0.1)
    assert prograde is not None and retrograde_impulse is not None
    assert radial is not None and vertical is not None
    assert prograde.semi_major_axis_cm == pytest.approx(radius_cm / 0.79)
    assert prograde.eccentricity == pytest.approx(0.21)
    assert retrograde_impulse.semi_major_axis_cm == pytest.approx(radius_cm / 1.19)
    assert retrograde_impulse.eccentricity == pytest.approx(0.19)
    assert radial.eccentricity == pytest.approx(0.1)
    assert radial.inclination_rad == pytest.approx(0.0)
    assert vertical.inclination_rad == pytest.approx(np.arctan(0.1))
    assert orbit(0.0, 1.5, 0.0) is None
