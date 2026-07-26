"""
damp.py - Orbital damping calculations for objects in AGN disks.
Now accepts precomputed disk radial profiles (r_list, sigma_list, h_list)
to avoid repeated recomputation.
"""

import numpy as np
import numpy.typing as npt
import constants as const
from dataclasses import dataclass
from typing import TypedDict
from helper_function import find_closest_index
from general_class import AGN_properties

FloatArray = npt.NDArray[np.float64]

MIN_EMBEDDED_INCLINATION_DEG = 8.0
DEGREES_PER_RADIAN = 180.0 / np.pi
INITIAL_TIME_STEP_FRACTION = 0.01
MIN_TIME_STEP_FRACTION = 1.0e-6
MAX_TIME_STEP_FRACTION = 0.1
MAX_ACCEPTED_RELATIVE_CHANGE = 0.01
GROW_TIME_STEP_RELATIVE_CHANGE = 0.001
SHRINK_TIME_STEP_FACTOR = 0.5
GROW_TIME_STEP_FACTOR = 2.0
STELLAR_MASS_RADIUS_EXPONENT = 0.8


class TimescaleData(TypedDict):
    """Computed orbital, disk, and drag quantities in cgs units."""

    disk: "DiskProperties"
    drag: "DragProperties"
    crossing_time_s: tuple[float, float]
    orbital_period_s: float
    orbital_velocity_cm_s: tuple[float, float]
    orbital_radius_cm: tuple[float, float]


# ----------------------------------------------------------------------
# Helper functions for closest‑index lookup (same as before)
def h_gas(
    radius_cm: float, aspect_ratio: FloatArray, radius_grid_cm: FloatArray
) -> float:
    """Return dimensionless disk aspect ratio at ``radius_cm``."""
    index = find_closest_index(radius_grid_cm, radius_cm)
    return float(aspect_ratio[index])


def sigma_gas(
    radius_cm: float,
    surface_density_g_cm2: FloatArray,
    radius_grid_cm: FloatArray,
) -> float:
    """Return disk surface density [g/cm^2] at ``radius_cm``."""
    index = find_closest_index(radius_grid_cm, radius_cm)
    return float(surface_density_g_cm2[index])


# ----------------------------------------------------------------------
@dataclass(frozen=True)
class OrbitalState:
    """Immutable orbital state.

    Legacy compact field names are retained: ``a`` [cm], ``e`` dimensionless,
    ``i`` [rad], ``cosw`` dimensionless, and ``t`` [s].
    """

    a: float  # semi‑major axis [cm]
    e: float  # eccentricity
    i: float  # inclination [rad]
    cosw: float  # cosine of argument of periapsis
    t: float = 0.0  # time [s]

    @property
    def sini(self) -> float:
        return np.sin(self.i)

    @property
    def cosi(self) -> float:
        return np.cos(self.i)

    def update(self, **kwargs: float) -> "OrbitalState":
        """Return a new state with updated fields."""
        data = dict(self.__dict__)
        data.update(kwargs)
        return OrbitalState(**data)


# ----------------------------------------------------------------------
@dataclass
class DiskProperties:
    """
    Disk properties at the two orbital extremal points (periapsis/apoapsis).
    The static method `from_agn` now uses precomputed lists.
    """

    r: tuple[float, float]  # radii at positions 1 and 2 [cm]
    H: float  # disk scale height at average radius [cm]
    h: float  # h = H/r (dimensionless)
    sigma: tuple[float, float]  # surface densities [g/cm^2]
    rho: tuple[float, float]  # volume densities [g/cm^3]

    @classmethod
    def from_agn(
        cls,
        radius_1_cm: float,
        radius_2_cm: float,
        surface_density_g_cm2: FloatArray,
        aspect_ratio: FloatArray,
        radius_grid_cm: FloatArray,
    ) -> "DiskProperties":
        """Create disk properties from precomputed radial profiles."""
        average_radius_cm = 0.5 * (radius_1_cm + radius_2_cm)
        local_aspect_ratio = h_gas(
            average_radius_cm, aspect_ratio, radius_grid_cm
        )
        scale_height_cm = local_aspect_ratio * average_radius_cm
        sigma_1_g_cm2 = sigma_gas(
            radius_1_cm, surface_density_g_cm2, radius_grid_cm
        )
        sigma_2_g_cm2 = sigma_gas(
            radius_2_cm, surface_density_g_cm2, radius_grid_cm
        )
        density_1_g_cm3 = sigma_1_g_cm2 / scale_height_cm
        density_2_g_cm3 = sigma_2_g_cm2 / scale_height_cm
        return cls(
            r=(radius_1_cm, radius_2_cm),
            H=scale_height_cm,
            h=local_aspect_ratio,
            sigma=(sigma_1_g_cm2, sigma_2_g_cm2),
            rho=(density_1_g_cm3, density_2_g_cm3),
        )


# ----------------------------------------------------------------------
@dataclass
class DragProperties:
    """Drag-related quantities at the two orbital positions."""

    rdrag: tuple[float, float]  # effective drag radii [cm]
    vrel: tuple[float, float]  # relative velocities [cm/s]
    tau: tuple[float, float]  # drag timescales [s]

    @classmethod
    def compute(
        cls,
        mass_g: float,
        state: OrbitalState,
        disk: DiskProperties,
        agn_properties: AGN_properties,
        obj_type: str,
    ) -> "DragProperties":
        """Compute drag properties using orbital state and disk properties."""
        smbh_mass_g = agn_properties.M_SMBH * const.msun

        velocity_prefactor_cm_s = np.sqrt(
            const.G * smbh_mass_g / state.a
        ) / np.sqrt(1 - state.e**2)

        relative_velocity_1_cm_s = velocity_prefactor_cm_s * np.sqrt(
            2
            - 3 * state.e * state.cosw
            + state.e**2
            - 2 * (1 - state.e * state.cosw) ** 1.5 * state.cosi
        )
        relative_velocity_2_cm_s = velocity_prefactor_cm_s * np.sqrt(
            2
            + 3 * state.e * state.cosw
            + state.e**2
            - 2 * (1 + state.e * state.cosw) ** 1.5 * state.cosi
        )

        # Geometric radius (Bondi for BHs, stellar radius for stars)
        if obj_type in ["BH", "BH_test"]:
            geometric_radius_cm = 0.0
        elif obj_type in ["Star", "Star_test"]:
            geometric_radius_cm = const.Rsun * (
                mass_g / const.msun
            ) ** STELLAR_MASS_RADIUS_EXPONENT
        else:
            raise ValueError(f"Unknown object type: {obj_type}")

        drag_radius_1_cm = float(
            np.maximum(
                geometric_radius_cm,
                2 * const.G * mass_g / relative_velocity_1_cm_s**2,
            )
        )
        drag_radius_2_cm = float(
            np.maximum(
                geometric_radius_cm,
                2 * const.G * mass_g / relative_velocity_2_cm_s**2,
            )
        )
        drag_time_1_s = (
            mass_g
            / (
                np.pi
                * drag_radius_1_cm**2
                * disk.rho[0]
                * relative_velocity_1_cm_s
            )
        )
        drag_time_2_s = (
            mass_g
            / (
                np.pi
                * drag_radius_2_cm**2
                * disk.rho[1]
                * relative_velocity_2_cm_s
            )
        )
        return cls(
            rdrag=(drag_radius_1_cm, drag_radius_2_cm),
            vrel=(
                float(relative_velocity_1_cm_s),
                float(relative_velocity_2_cm_s),
            ),
            tau=(float(drag_time_1_s), float(drag_time_2_s)),
        )


# ----------------------------------------------------------------------
class TimescaleCalculator:
    """
    Computes orbital and disk timescales.
    Now stores precomputed disk lists to avoid recomputation.
    """

    def __init__(
        self,
        agn_properties: AGN_properties,
        radius_grid_cm: FloatArray,
        surface_density_g_cm2: FloatArray,
        aspect_ratio: FloatArray,
    ) -> None:
        self.agn_properties = agn_properties
        self.smbh_mass_g = agn_properties.M_SMBH * const.msun
        self.radius_grid_cm = radius_grid_cm
        self.surface_density_g_cm2 = surface_density_g_cm2
        self.aspect_ratio = aspect_ratio

    def compute_orbital_positions(
        self, state: OrbitalState
    ) -> tuple[float, float]:
        """Return the two extremal radii (periapsis‑like and apoapsis‑like)."""
        radius_1_cm = (
            state.a * (1 - state.e**2) / (1 - state.e * state.cosw)
        )
        radius_2_cm = (
            state.a * (1 - state.e**2) / (1 + state.e * state.cosw)
        )
        return radius_1_cm, radius_2_cm

    def compute_orbital_velocities(
        self,
        state: OrbitalState,
        radius_1_cm: float,
        radius_2_cm: float,
    ) -> tuple[float, float]:
        """Orbital speeds at the two positions."""
        velocity_1_cm_s = np.sqrt(
            const.G
            * self.smbh_mass_g
            * (2 / radius_1_cm - 1 / state.a)
        )
        velocity_2_cm_s = np.sqrt(
            const.G
            * self.smbh_mass_g
            * (2 / radius_2_cm - 1 / state.a)
        )
        return float(velocity_1_cm_s), float(velocity_2_cm_s)

    def orbital_period(self, state: OrbitalState) -> float:
        """Keplerian orbital period [s]."""
        return float(
            2
            * np.pi
            * np.sqrt(state.a**3 / const.G / self.smbh_mass_g)
        )

    def compute_crossing_times(
        self,
        scale_height_cm: float,
        velocity_1_cm_s: float,
        velocity_2_cm_s: float,
        state: OrbitalState,
    ) -> tuple[float, float]:
        """Disk crossing times (time to pass through the disk thickness)."""
        crossing_time_1_s = (
            2 * scale_height_cm / (velocity_1_cm_s * state.sini)
        )
        crossing_time_2_s = (
            2 * scale_height_cm / (velocity_2_cm_s * state.sini)
        )
        return float(crossing_time_1_s), float(crossing_time_2_s)

    def compute_all_timescales(
        self, mass_g: float, state: OrbitalState, obj_type: str
    ) -> TimescaleData:
        """
        Compute all relevant timescales and disk/drag properties.
        Uses the stored disk lists (no recomputation of radial structure).
        """
        # Orbital positions
        radius_1_cm, radius_2_cm = self.compute_orbital_positions(state)

        # Disk properties at those radii (using precomputed lists)
        disk_prop = DiskProperties.from_agn(
            radius_1_cm,
            radius_2_cm,
            self.surface_density_g_cm2,
            self.aspect_ratio,
            self.radius_grid_cm,
        )

        # Orbital velocities
        velocity_1_cm_s, velocity_2_cm_s = (
            self.compute_orbital_velocities(
                state, radius_1_cm, radius_2_cm
            )
        )

        # Drag properties
        drag = DragProperties.compute(
            mass_g,
            state,
            disk_prop,
            self.agn_properties,
            obj_type,
        )

        # Crossing times
        crossing_time_1_s, crossing_time_2_s = (
            self.compute_crossing_times(
                disk_prop.H,
                velocity_1_cm_s,
                velocity_2_cm_s,
                state,
            )
        )

        # Orbital period
        orbital_period_s = self.orbital_period(state)

        return {
            "disk": disk_prop,
            "drag": drag,
            "crossing_time_s": (
                crossing_time_1_s,
                crossing_time_2_s,
            ),
            "orbital_period_s": orbital_period_s,
            "orbital_velocity_cm_s": (
                velocity_1_cm_s,
                velocity_2_cm_s,
            ),
            "orbital_radius_cm": (radius_1_cm, radius_2_cm),
        }


# ----------------------------------------------------------------------
class DampingCalculator:
    """
    Main class for damping calculations.
    Accepts precomputed disk lists and passes them to TimescaleCalculator.
    """

    def __init__(
        self,
        AGN: AGN_properties,
        verbose: bool,
        r_list: FloatArray,
        sigma_list: FloatArray,
        h_list: FloatArray,
    ) -> None:
        """Initialize from AGN state and disk grids.

        Legacy keyword names are retained: ``r_list`` [cm], ``sigma_list``
        [g/cm^2], and dimensionless ``h_list``.
        """
        self.AGN = AGN
        self.verbose = verbose
        self.timescale_calc = TimescaleCalculator(
            AGN, r_list, sigma_list, h_list
        )
        self._setup_constants()

    def _setup_constants(self) -> None:
        self.smbh_mass_g = self.AGN.M_SMBH * const.msun
        self.minimum_inclination_rad = np.deg2rad(
            MIN_EMBEDDED_INCLINATION_DEG
        )

    def damping_semi_analysis(
        self, mass_g: float, state: OrbitalState, obj_type: str
    ) -> float:
        """Return the analytical orbital damping estimate [s]."""
        timescales = self.timescale_calc.compute_all_timescales(
            mass_g, state, obj_type
        )
        crossing_time_1_s, crossing_time_2_s = timescales[
            "crossing_time_s"
        ]
        orbital_period_s = timescales["orbital_period_s"]
        drag_time_1_s, drag_time_2_s = timescales["drag"].tau

        if (
            drag_time_1_s <= 0
            or drag_time_2_s <= 0
            or orbital_period_s <= 0
        ):
            return np.inf

        damping_time_s = (
            crossing_time_1_s / (drag_time_1_s * orbital_period_s)
            + crossing_time_2_s
            / (drag_time_2_s * orbital_period_s)
        ) ** (-1)
        if "test" in obj_type and self.verbose:
            self._print_debug_info(
                state, mass_g, timescales, damping_time_s
            )
        return float(damping_time_s)

    def _print_debug_info(
        self,
        state: OrbitalState,
        mass_g: float,
        timescales: TimescaleData,
        damping_time_s: float,
    ) -> None:
        """Print detailed debug information."""
        print("#" * 50)
        print(f"Myr: {const.Myr:.3e} s")
        print(f"rg: {self.AGN.rg:.3e} cm")
        print(
            f"torb: {timescales['orbital_period_s']/const.Myr:.2e} Myr"
        )
        print(f"tdrag: {damping_time_s/const.Myr:.2e} Myr")
        print(f"mass: {mass_g/const.msun:.2e} Msun")
        print("\n=== Orbital Parameters ===")
        print(f"a: {state.a/self.AGN.rg:.2e} rg")
        print(f"e: {state.e:.3f}")
        print(f"i: {np.degrees(state.i):.1f}°")
        print(f"cosw: {state.cosw:.3f}")
        drag = timescales["drag"]
        print(f"rdrag: {drag.rdrag[0]:3e}, {drag.rdrag[1]:3e} cm")
        print("\n=== Disk Properties ===")
        disk = timescales["disk"]
        print(f"H: {disk.H/self.AGN.rg:.2e} rg")
        print(f"sigma1: {disk.sigma[0]:.2e} g/cm²")
        print(f"sigma2: {disk.sigma[1]:.2e} g/cm²")
        print(f"rhod1: {disk.rho[0]:.2e} g/cm³")
        print(f"rhod2: {disk.rho[1]:.2e} g/cm³")
        print("\n=== Velocities ===")
        print(f"vrel1: {drag.vrel[0]/1e5:.2e} km/s")
        print(f"vrel2: {drag.vrel[1]/1e5:.2e} km/s")
        print("\n=== Timescales ===")
        crossing_time_1_s, crossing_time_2_s = timescales[
            "crossing_time_s"
        ]
        print(f"tcross1: {crossing_time_1_s/const.Myr:.2e} Myr")
        print(f"tcross2: {crossing_time_2_s/const.Myr:.2e} Myr")
        print(f"tau1: {drag.tau[0]/const.Myr:.2e} Myr")
        print(f"tau2: {drag.tau[1]/const.Myr:.2e} Myr")

    def compute_ode_coefficients(
        self,
        state: OrbitalState,
        drag_time_1_s: float,
        drag_time_2_s: float,
        crossing_time_1_s: float,
        crossing_time_2_s: float,
    ) -> tuple[float, float, float, float, float, float]:
        """Compute ODE coefficients for da/dt, de/dt, di/dt."""
        # Position 1 (periapsis‑like)
        da1 = (
            -state.a
            / drag_time_1_s
            * 2
            / (1 - state.e**2)
            * (
                1
                - 2 * state.e * state.cosw
                + state.e**2
                - (1 - state.e * state.cosw) ** 1.5 * state.cosi
            )
            * crossing_time_1_s
        )
        de1 = (
            -1
            / drag_time_1_s
            * (
                2 * state.e
                - 2 * state.cosw
                - (state.e + state.e * state.cosw**2 - 2 * state.cosw)
                / np.sqrt(1 - state.e * state.cosw)
                * state.cosi
            )
            * crossing_time_1_s
        )
        di1 = (
            -1
            / drag_time_1_s
            * state.sini
            / np.sqrt(1 - state.e * state.cosw)
            * crossing_time_1_s
        )

        # Position 2 (apoapsis‑like)
        da2 = (
            -state.a
            / drag_time_2_s
            * 2
            / (1 - state.e**2)
            * (
                1
                - 2 * state.e * state.cosw
                + state.e**2
                - (1 + state.e * state.cosw) ** 1.5 * state.cosi
            )
            * crossing_time_2_s
        )
        de2 = (
            -1
            / drag_time_2_s
            * (
                2 * state.e
                + 2 * state.cosw
                - (state.e + state.e * state.cosw**2 - 2 * state.cosw)
                / np.sqrt(1 + state.e * state.cosw)
                * state.cosi
            )
            * crossing_time_2_s
        )
        di2 = (
            -1
            / drag_time_2_s
            * state.sini
            / np.sqrt(1 + state.e * state.cosw)
            * crossing_time_2_s
        )

        return da1, da2, di1, di2, de1, de2

    def adaptive_time_step(
        self,
        state: OrbitalState,
        delta_semi_major_axis_cm: float,
        delta_inclination_rad: float,
        tolerance: float = 3e-2,
    ) -> float:
        """Return factor to multiply dt (1 = keep, >1 increase, <1 decrease)."""
        relative_axis_change = (
            np.abs(delta_semi_major_axis_cm / state.a)
            if state.a > 0
            else 0
        )
        relative_inclination_change = (
            np.abs(delta_inclination_rad / state.i) if state.i > 0 else 0
        )
        if (
            relative_axis_change < 0.1 * tolerance
            and relative_inclination_change < 0.1 * tolerance
        ):
            return 2.0
        if (
            relative_axis_change > tolerance
            or relative_inclination_change > tolerance
        ):
            return 0.1
        return 1.0

    def damping_simulation(
        self,
        mass_g: float,
        initial_state: OrbitalState,
        formation_time_myr: float,
        obj_type: str,
        type_flag: int,
        max_iter: int = 50000,
    ) -> tuple[float, float, int]:
        """
        Numerical integration of orbital damping.
        Returns: (final_time [s], final_semi_major_axis [cm], type_flag)
        """
        if mass_g <= 0.0:
            raise ValueError("damped object mass must be positive")
        if formation_time_myr < 0.0:
            raise ValueError("formation time must be nonnegative")
        state = initial_state
        formation_time_s = formation_time_myr * const.Myr
        duty_cycle_s = self.AGN.AGNlifetime * const.Myr

        time_step_fraction = INITIAL_TIME_STEP_FRACTION

        iteration = 0

        while (
            state.i > self.minimum_inclination_rad
            and iteration < max_iter
        ):
            iteration += 1

            if state.e < 0:
                state = state.update(e=0.0)
            if state.e >= 1.0:
                if self.verbose:
                    print("Warning: Orbit became unbound (e >= 1)")
                break

            timescales = self.timescale_calc.compute_all_timescales(
                mass_g, state, obj_type
            )
            drag_time_1_s, drag_time_2_s = timescales["drag"].tau
            crossing_time_1_s, crossing_time_2_s = timescales[
                "crossing_time_s"
            ]
            orbital_period_s = timescales["orbital_period_s"]

            time_step_s = (
                np.minimum(
                    drag_time_1_s / crossing_time_1_s,
                    drag_time_2_s / crossing_time_2_s,
                )
                * orbital_period_s
                * time_step_fraction
            )

            # Compute derivatives
            da1, da2, di1, di2, de1, de2 = self.compute_ode_coefficients(
                state,
                drag_time_1_s,
                drag_time_2_s,
                crossing_time_1_s,
                crossing_time_2_s,
            )
            delta_axis_per_orbit_cm = da1 + da2
            delta_eccentricity_per_orbit = de1 + de2
            delta_inclination_per_orbit_rad = di1 + di2

            new_semi_major_axis_cm = (
                state.a
                + delta_axis_per_orbit_cm
                * time_step_s
                / orbital_period_s
            )
            new_eccentricity = max(
                state.e
                + delta_eccentricity_per_orbit
                * time_step_s
                / orbital_period_s,
                0.0,
            )
            new_inclination_rad = (
                state.i
                + delta_inclination_per_orbit_rad
                * time_step_s
                / orbital_period_s
            )

            # Adaptive step control
            relative_axis_change = (
                abs((new_semi_major_axis_cm - state.a) / state.a)
                if state.a > 0
                else 0
            )
            relative_inclination_change = (
                abs((new_inclination_rad - state.i) / state.i)
                if state.i > self.minimum_inclination_rad
                else 0
            )
            maximum_relative_change = max(
                relative_axis_change, relative_inclination_change
            )
            if maximum_relative_change > MAX_ACCEPTED_RELATIVE_CHANGE:
                time_step_fraction *= SHRINK_TIME_STEP_FACTOR
                time_step_fraction = max(
                    time_step_fraction, MIN_TIME_STEP_FRACTION
                )
                continue  # redo iteration with smaller step
            if maximum_relative_change < GROW_TIME_STEP_RELATIVE_CHANGE:
                time_step_fraction *= GROW_TIME_STEP_FACTOR
                time_step_fraction = min(
                    time_step_fraction, MAX_TIME_STEP_FRACTION
                )

            state = state.update(
                a=float(new_semi_major_axis_cm),
                e=float(new_eccentricity),
                i=float(new_inclination_rad),
                t=float(state.t + time_step_s),
            )

            # Star → BH transition handling
            result = self._handle_type_transition(
                state,
                mass_g,
                formation_time_s,
                duty_cycle_s,
                obj_type,
            )
            if result is not None:
                return result

            if iteration >= max_iter:
                if self.verbose:
                    print(f"Warning: Maximum iterations ({max_iter}) reached")
                break

        return self._determine_final_state(
            state,
            formation_time_s,
            duty_cycle_s,
            obj_type,
        )

    def _handle_type_transition(
        self,
        state: OrbitalState,
        mass_g: float,
        formation_time_s: float,
        duty_cycle_s: float,
        obj_type: str,
    ) -> tuple[float, float, int] | None:
        """Return (t_BH, a_final, new_flag) if transition occurs, else None."""
        if "test" in obj_type:
            return None
        if obj_type == "BH":
            if state.t < duty_cycle_s:
                return None
            return (0.0, 0.0, 2)  # BH outside disk
        if obj_type == "Star":
            if state.t <= formation_time_s:
                return None
            black_hole_embedding_time_s = (
                state.t
                + self.damping_semi_analysis(mass_g, state, "BH")
            )
            if black_hole_embedding_time_s < duty_cycle_s:
                return (
                    black_hole_embedding_time_s,
                    state.a,
                    1,
                )
            return (0.0, 0.0, 2)
        return None

    def _determine_final_state(
        self,
        state: OrbitalState,
        formation_time_s: float,
        duty_cycle_s: float,
        obj_type: str,
    ) -> tuple[float, float, int]:
        """Return final time [s], semi-major axis [cm], and object flag."""
        if "test" in obj_type:
            return state.t, state.a, -1
        if state.t > duty_cycle_s:
            return 0.0, 0.0, 2
        if state.t <= formation_time_s:
            final_type_flag = 0
        else:
            final_type_flag = 1
        return state.t, state.a, final_type_flag
