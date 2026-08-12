"""
damp.py - Orbital damping calculations for objects in AGN disks.
Now accepts precomputed disk radial profiles with unit-explicit names
to avoid repeated recomputation.
"""

import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from enum import Enum
from typing import TypedDict

try:
    from physical_process.agn import AGNProperties
    import physical_process.constants as const
except ImportError:
    from agn import AGNProperties
    import constants as const

try:
    from physical_process.disk_migration import find_closest_index
except ImportError:
    from disk_migration import find_closest_index

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


class DampingStatus(str, Enum):
    """Physical or numerical outcome of one orbital-damping calculation."""

    EMBEDDED_STAR = "embedded_star"
    EMBEDDED_BH = "embedded_bh"
    OUTSIDE_DISK = "outside_disk"
    CALCULATION_ERROR = "calculation_error"
    TEST = "test"


class TimescaleData(TypedDict):
    """Computed orbital, disk, and drag quantities in cgs units."""

    disk: "DiskProperties"
    drag: "DragProperties"
    crossing_time_s: tuple[float, float]
    orbital_period_s: float
    orbital_velocity_cm_s: tuple[float, float]
    orbital_radius_cm: tuple[float, float]


# ----------------------------------------------------------------------
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

    Names are explicit about units for dimensional quantities.
    """

    semi_major_axis_cm: float
    eccentricity: float
    inclination_rad: float
    cos_omega: float
    time_s: float = 0.0

    @property
    def sin_inclination(self) -> float:
        return np.sin(self.inclination_rad)

    @property
    def cos_inclination(self) -> float:
        return np.cos(self.inclination_rad)

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

    radius_cm: tuple[float, float]
    scale_height_cm: float
    aspect_ratio: float
    surface_density_g_cm2: tuple[float, float]
    density_g_cm3: tuple[float, float]

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
            radius_cm=(radius_1_cm, radius_2_cm),
            scale_height_cm=scale_height_cm,
            aspect_ratio=local_aspect_ratio,
            surface_density_g_cm2=(sigma_1_g_cm2, sigma_2_g_cm2),
            density_g_cm3=(density_1_g_cm3, density_2_g_cm3),
        )


# ----------------------------------------------------------------------
@dataclass
class DragProperties:
    """Drag-related quantities at the two orbital positions."""

    drag_radius_cm: tuple[float, float]
    relative_velocity_cm_s: tuple[float, float]
    drag_time_s: tuple[float, float]

    @classmethod
    def compute(
        cls,
        mass_g: float,
        state: OrbitalState,
        disk: DiskProperties,
        agn_properties: AGNProperties,
        obj_type: str,
    ) -> "DragProperties":
        """Compute drag properties using orbital state and disk properties."""
        smbh_mass_g = agn_properties.smbh_mass_msun * const.msun

        velocity_prefactor_cm_s = np.sqrt(
            const.G * smbh_mass_g / state.semi_major_axis_cm
        ) / np.sqrt(1 - state.eccentricity**2)

        relative_velocity_1_cm_s = velocity_prefactor_cm_s * np.sqrt(
            2
            - 3 * state.eccentricity * state.cos_omega
            + state.eccentricity**2
            - 2 * (1 - state.eccentricity * state.cos_omega) ** 1.5 * state.cos_inclination
        )
        relative_velocity_2_cm_s = velocity_prefactor_cm_s * np.sqrt(
            2
            + 3 * state.eccentricity * state.cos_omega
            + state.eccentricity**2
            - 2 * (1 + state.eccentricity * state.cos_omega) ** 1.5 * state.cos_inclination
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
                * disk.density_g_cm3[0]
                * relative_velocity_1_cm_s
            )
        )
        drag_time_2_s = (
            mass_g
            / (
                np.pi
                * drag_radius_2_cm**2
                * disk.density_g_cm3[1]
                * relative_velocity_2_cm_s
            )
        )
        return cls(
            drag_radius_cm=(drag_radius_1_cm, drag_radius_2_cm),
            relative_velocity_cm_s=(
                float(relative_velocity_1_cm_s),
                float(relative_velocity_2_cm_s),
            ),
            drag_time_s=(float(drag_time_1_s), float(drag_time_2_s)),
        )


# ----------------------------------------------------------------------
class TimescaleCalculator:
    """
    Computes orbital and disk timescales.
    Now stores precomputed disk lists to avoid recomputation.
    """

    def __init__(
        self,
        agn_properties: AGNProperties,
        radius_grid_cm: FloatArray,
        surface_density_g_cm2: FloatArray,
        aspect_ratio: FloatArray,
    ) -> None:
        self.agn_properties = agn_properties
        self.smbh_mass_g = agn_properties.smbh_mass_msun * const.msun
        self.radius_grid_cm = radius_grid_cm
        self.surface_density_g_cm2 = surface_density_g_cm2
        self.aspect_ratio = aspect_ratio

    def compute_orbital_positions(
        self, state: OrbitalState
    ) -> tuple[float, float]:
        """Return the two extremal radii (periapsis‑like and apoapsis‑like)."""
        radius_1_cm = (
            state.semi_major_axis_cm * (1 - state.eccentricity**2) / (1 - state.eccentricity * state.cos_omega)
        )
        radius_2_cm = (
            state.semi_major_axis_cm * (1 - state.eccentricity**2) / (1 + state.eccentricity * state.cos_omega)
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
            * (2 / radius_1_cm - 1 / state.semi_major_axis_cm)
        )
        velocity_2_cm_s = np.sqrt(
            const.G
            * self.smbh_mass_g
            * (2 / radius_2_cm - 1 / state.semi_major_axis_cm)
        )
        return float(velocity_1_cm_s), float(velocity_2_cm_s)

    def orbital_period(self, state: OrbitalState) -> float:
        """Keplerian orbital period [s]."""
        return float(
            2
            * np.pi
            * np.sqrt(state.semi_major_axis_cm**3 / const.G / self.smbh_mass_g)
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
            2 * scale_height_cm / (velocity_1_cm_s * state.sin_inclination)
        )
        crossing_time_2_s = (
            2 * scale_height_cm / (velocity_2_cm_s * state.sin_inclination)
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
                disk_prop.scale_height_cm,
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
        agn_properties: AGNProperties,
        verbose: bool,
        radius_grid_cm: FloatArray,
        surface_density_g_cm2: FloatArray,
        aspect_ratio: FloatArray,
    ) -> None:
        """Initialize from AGN state and disk grids."""
        self.agn_properties = agn_properties
        self.verbose = verbose
        self.timescale_calc = TimescaleCalculator(
            agn_properties,
            radius_grid_cm,
            surface_density_g_cm2,
            aspect_ratio,
        )
        self._setup_constants()

    def _setup_constants(self) -> None:
        self.smbh_mass_g = self.agn_properties.smbh_mass_msun * const.msun
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
        drag_time_1_s, drag_time_2_s = timescales["drag"].drag_time_s

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
        gravitational_radius_cm = self.agn_properties.gravitational_radius_cm
        print(f"rg: {gravitational_radius_cm:.3e} cm")
        print(
            f"torb: {timescales['orbital_period_s']/const.Myr:.2e} Myr"
        )
        print(f"tdrag: {damping_time_s/const.Myr:.2e} Myr")
        print(f"mass: {mass_g/const.msun:.2e} Msun")
        print("\n=== Orbital Parameters ===")
        print(f"a: {state.semi_major_axis_cm/gravitational_radius_cm:.2e} rg")
        print(f"e: {state.eccentricity:.3f}")
        print(f"i: {np.degrees(state.inclination_rad):.1f}°")
        print(f"cosw: {state.cos_omega:.3f}")
        drag = timescales["drag"]
        print(
            "rdrag: "
            f"{drag.drag_radius_cm[0]:3e}, {drag.drag_radius_cm[1]:3e} cm"
        )
        print("\n=== Disk Properties ===")
        disk = timescales["disk"]
        print(f"H: {disk.scale_height_cm/gravitational_radius_cm:.2e} rg")
        print(f"sigma1: {disk.surface_density_g_cm2[0]:.2e} g/cm²")
        print(f"sigma2: {disk.surface_density_g_cm2[1]:.2e} g/cm²")
        print(f"rhod1: {disk.density_g_cm3[0]:.2e} g/cm³")
        print(f"rhod2: {disk.density_g_cm3[1]:.2e} g/cm³")
        print("\n=== Velocities ===")
        print(f"vrel1: {drag.relative_velocity_cm_s[0]/1e5:.2e} km/s")
        print(f"vrel2: {drag.relative_velocity_cm_s[1]/1e5:.2e} km/s")
        print("\n=== Timescales ===")
        crossing_time_1_s, crossing_time_2_s = timescales[
            "crossing_time_s"
        ]
        print(f"tcross1: {crossing_time_1_s/const.Myr:.2e} Myr")
        print(f"tcross2: {crossing_time_2_s/const.Myr:.2e} Myr")
        print(f"tau1: {drag.drag_time_s[0]/const.Myr:.2e} Myr")
        print(f"tau2: {drag.drag_time_s[1]/const.Myr:.2e} Myr")

    def compute_ode_coefficients(
        self,
        state: OrbitalState,
        drag_time_1_s: float,
        drag_time_2_s: float,
        crossing_time_1_s: float,
        crossing_time_2_s: float,
    ) -> tuple[float, float, float, float, float, float]:
        """Compute per-orbit ODE coefficients for a, e, and inclination."""
        # Position 1 (periapsis‑like)
        delta_axis_1_cm_per_orbit = (
            -state.semi_major_axis_cm
            / drag_time_1_s
            * 2
            / (1 - state.eccentricity**2)
            * (
                1
                - 2 * state.eccentricity * state.cos_omega
                + state.eccentricity**2
                - (1 - state.eccentricity * state.cos_omega) ** 1.5
                * state.cos_inclination
            )
            * crossing_time_1_s
        )
        delta_eccentricity_1_per_orbit = (
            -1
            / drag_time_1_s
            * (
                2 * state.eccentricity
                - 2 * state.cos_omega
                - (
                    state.eccentricity
                    + state.eccentricity * state.cos_omega**2
                    - 2 * state.cos_omega
                )
                / np.sqrt(1 - state.eccentricity * state.cos_omega)
                * state.cos_inclination
            )
            * crossing_time_1_s
        )
        delta_inclination_1_rad_per_orbit = (
            -1
            / drag_time_1_s
            * state.sin_inclination
            / np.sqrt(1 - state.eccentricity * state.cos_omega)
            * crossing_time_1_s
        )

        # Position 2 (apoapsis‑like)
        delta_axis_2_cm_per_orbit = (
            -state.semi_major_axis_cm
            / drag_time_2_s
            * 2
            / (1 - state.eccentricity**2)
            * (
                1
                - 2 * state.eccentricity * state.cos_omega
                + state.eccentricity**2
                - (1 + state.eccentricity * state.cos_omega) ** 1.5
                * state.cos_inclination
            )
            * crossing_time_2_s
        )
        delta_eccentricity_2_per_orbit = (
            -1
            / drag_time_2_s
            * (
                2 * state.eccentricity
                + 2 * state.cos_omega
                - (
                    state.eccentricity
                    + state.eccentricity * state.cos_omega**2
                    - 2 * state.cos_omega
                )
                / np.sqrt(1 + state.eccentricity * state.cos_omega)
                * state.cos_inclination
            )
            * crossing_time_2_s
        )
        delta_inclination_2_rad_per_orbit = (
            -1
            / drag_time_2_s
            * state.sin_inclination
            / np.sqrt(1 + state.eccentricity * state.cos_omega)
            * crossing_time_2_s
        )

        return (
            delta_axis_1_cm_per_orbit,
            delta_axis_2_cm_per_orbit,
            delta_inclination_1_rad_per_orbit,
            delta_inclination_2_rad_per_orbit,
            delta_eccentricity_1_per_orbit,
            delta_eccentricity_2_per_orbit,
        )

    def adaptive_time_step(
        self,
        state: OrbitalState,
        delta_semi_major_axis_cm: float,
        delta_inclination_rad: float,
        tolerance: float = 3e-2,
    ) -> float:
        """Return factor to multiply dt (1 = keep, >1 increase, <1 decrease)."""
        relative_axis_change = (
            np.abs(delta_semi_major_axis_cm / state.semi_major_axis_cm)
            if state.semi_major_axis_cm > 0
            else 0
        )
        relative_inclination_change = (
            np.abs(delta_inclination_rad / state.inclination_rad)
            if state.inclination_rad > 0
            else 0
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
        max_iter: int = 50000,
    ) -> tuple[float, float, DampingStatus]:
        """
        Numerical integration of orbital damping.
        Returns: (final_time [s], final_semi_major_axis [cm], status)
        """
        if mass_g <= 0.0:
            raise ValueError("damped object mass must be positive")
        if formation_time_myr < 0.0:
            raise ValueError("formation time must be nonnegative")
        state = initial_state
        formation_time_s = formation_time_myr * const.Myr
        duty_cycle_s = self.agn_properties.agn_lifetime_myr * const.Myr

        time_step_fraction = INITIAL_TIME_STEP_FRACTION

        iteration = 0

        while (
            state.inclination_rad > self.minimum_inclination_rad
            and iteration < max_iter
        ):
            iteration += 1

            if state.eccentricity < 0:
                state = state.update(eccentricity=0.0)
            if state.eccentricity >= 1.0:
                if self.verbose:
                    print("Warning: Orbit became unbound (e >= 1)")
                break

            timescales = self.timescale_calc.compute_all_timescales(
                mass_g, state, obj_type
            )
            drag_time_1_s, drag_time_2_s = timescales["drag"].drag_time_s
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
            (
                delta_axis_1_cm_per_orbit,
                delta_axis_2_cm_per_orbit,
                delta_inclination_1_rad_per_orbit,
                delta_inclination_2_rad_per_orbit,
                delta_eccentricity_1_per_orbit,
                delta_eccentricity_2_per_orbit,
            ) = self.compute_ode_coefficients(
                state,
                drag_time_1_s,
                drag_time_2_s,
                crossing_time_1_s,
                crossing_time_2_s,
            )
            delta_axis_per_orbit_cm = (
                delta_axis_1_cm_per_orbit + delta_axis_2_cm_per_orbit
            )
            delta_eccentricity_per_orbit = (
                delta_eccentricity_1_per_orbit
                + delta_eccentricity_2_per_orbit
            )
            delta_inclination_per_orbit_rad = (
                delta_inclination_1_rad_per_orbit
                + delta_inclination_2_rad_per_orbit
            )

            new_semi_major_axis_cm = (
                state.semi_major_axis_cm
                + delta_axis_per_orbit_cm
                * time_step_s
                / orbital_period_s
            )
            new_eccentricity = max(
                state.eccentricity
                + delta_eccentricity_per_orbit
                * time_step_s
                / orbital_period_s,
                0.0,
            )
            new_inclination_rad = (
                state.inclination_rad
                + delta_inclination_per_orbit_rad
                * time_step_s
                / orbital_period_s
            )

            # Adaptive step control
            relative_axis_change = (
                abs(
                    (new_semi_major_axis_cm - state.semi_major_axis_cm)
                    / state.semi_major_axis_cm
                )
                if state.semi_major_axis_cm > 0
                else 0
            )
            relative_inclination_change = (
                abs(
                    (new_inclination_rad - state.inclination_rad)
                    / state.inclination_rad
                )
                if state.inclination_rad > self.minimum_inclination_rad
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
                semi_major_axis_cm=float(new_semi_major_axis_cm),
                eccentricity=float(new_eccentricity),
                inclination_rad=float(new_inclination_rad),
                time_s=float(state.time_s + time_step_s),
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
    ) -> tuple[float, float, DampingStatus] | None:
        """Return a completed transition outcome, or ``None`` to continue."""
        if "test" in obj_type:
            return None
        if obj_type == "BH":
            if state.time_s < duty_cycle_s:
                return None
            return (0.0, 0.0, DampingStatus.OUTSIDE_DISK)
        if obj_type == "Star":
            if state.time_s <= formation_time_s:
                return None
            black_hole_embedding_time_s = (
                state.time_s
                + self.damping_semi_analysis(mass_g, state, "BH")
            )
            if black_hole_embedding_time_s < duty_cycle_s:
                return (
                    black_hole_embedding_time_s,
                    state.semi_major_axis_cm,
                    DampingStatus.EMBEDDED_BH,
                )
            return (0.0, 0.0, DampingStatus.OUTSIDE_DISK)
        return None

    def _determine_final_state(
        self,
        state: OrbitalState,
        formation_time_s: float,
        duty_cycle_s: float,
        obj_type: str,
    ) -> tuple[float, float, DampingStatus]:
        """Return final time [s], semi-major axis [cm], and damping status."""
        if "test" in obj_type:
            return state.time_s, state.semi_major_axis_cm, DampingStatus.TEST
        if state.time_s > duty_cycle_s:
            return 0.0, 0.0, DampingStatus.OUTSIDE_DISK
        if state.time_s <= formation_time_s:
            final_status = DampingStatus.EMBEDDED_STAR
        else:
            final_status = DampingStatus.EMBEDDED_BH
        return state.time_s, state.semi_major_axis_cm, final_status
