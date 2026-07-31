"""State containers used by the AGN and compact-object simulations."""

from pathlib import Path

import numpy as np
import numpy.typing as npt

try:
    import physical_process.constants as const
except ImportError:
    import constants as const

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int_]

AGN_VISCOSITY_ALPHA = 0.01
MIGRATION_TRAP_RADIUS_RG = 1300.0
VELOCITY_DISPERSION_REFERENCE_CM_S = 2.0e7
VELOCITY_DISPERSION_REFERENCE_MASS_MSUN = 1.0e8
VELOCITY_DISPERSION_MASS_SLOPE = 0.2


class general_flags:
    """General simulation switches and paths."""

    def __init__(self) -> None:
        self.spinrms: float = 0.1
        self.Z_list: str = "0.02"
        self.in_dir: Path = Path("input/input_data.txt")
        self.ngng: bool = False
        self.out_met: Path = Path("output")


class BBH_properties:
    """Binary black-hole population arrays with unit-explicit names."""

    def __init__(self, num_binaries: int) -> None:
        self.num_binaries = num_binaries
        self.object_id: IntArray = np.zeros(num_binaries, dtype=int)
        self.primary_mass_msun: FloatArray = np.zeros(num_binaries, dtype=float)
        self.secondary_mass_msun: FloatArray = np.zeros(num_binaries, dtype=float)
        self.semi_major_axis_rsun: FloatArray = np.zeros(num_binaries, dtype=float)
        self.eccentricity: FloatArray = np.zeros(num_binaries, dtype=float)
        self.second_semi_major_axis_rsun: FloatArray = np.zeros(
            num_binaries, dtype=float
        )
        self.second_eccentricity: FloatArray = np.zeros(num_binaries, dtype=float)
        self.eccentricity_10hz: FloatArray = np.zeros(num_binaries, dtype=float)
        self.eccentricity_0p1hz: FloatArray = np.zeros(num_binaries, dtype=float)
        self.primary_spin: FloatArray = np.zeros(num_binaries, dtype=float)
        self.secondary_spin: FloatArray = np.zeros(num_binaries, dtype=float)
        self.primary_spin_angle_rad: FloatArray = np.zeros(num_binaries, dtype=float)
        self.secondary_spin_angle_rad: FloatArray = np.zeros(num_binaries, dtype=float)
        self.separation_cm: FloatArray = np.zeros(num_binaries, dtype=float)
        self.merger_mass_msun: FloatArray = np.zeros(num_binaries, dtype=float)
        self.merger_spin: FloatArray = np.zeros(num_binaries, dtype=float)
        self.kick_velocity_cm_s: FloatArray = np.zeros(num_binaries, dtype=float)
        self.min_secondary_mass_msun: float = 3.0
        self.trap_reach_time_myr: FloatArray = np.zeros(num_binaries, dtype=float)


class stellar_properties:
    """Stellar population arrays with units documented on each attribute."""

    def __init__(self, num_stars: int) -> None:
        self.num_stars = num_stars
        self.object_id: IntArray = np.zeros(num_stars, dtype=int)
        self.mass_msun: FloatArray = np.zeros(num_stars, dtype=float)
        self.remnant_mass_msun: FloatArray = np.zeros(num_stars, dtype=float)
        self.semi_major_axis_cm: FloatArray = np.zeros(num_stars, dtype=float)
        self.eccentricity: FloatArray = np.zeros(num_stars, dtype=float)
        self.inclination_rad: FloatArray = np.zeros(num_stars, dtype=float)
        self.cos_omega: FloatArray = np.zeros(num_stars, dtype=float)

    def init_Stellar(self) -> None:
        """Compatibility placeholder for the historical initialization API."""


class AGN_properties:
    """AGN scalar properties.

    Attribute names carry their physical units where applicable.
    """

    def __init__(self) -> None:
        self.agn_lifetime_myr: float = 0.0
        self.escape_velocity_cm_s: float = 0.0
        self.smbh_mass_msun: float = 0.0
        self.gravitational_radius_cm: float = 0.0
        self.viscosity_alpha: float = 0.0
        self.migration_trap_radius_cm: float = 0.0
        self.average_stellar_mass_msun: float = 0.0
        self.velocity_dispersion_cm_s: float = 0.0

    def init_AGN(self, smbh_mass_msun: float, duty_cycle_myr: float) -> None:
        """Initialize an AGN from SMBH mass [Msun] and lifetime [Myr]."""
        if smbh_mass_msun <= 0.0:
            raise ValueError("SMBH mass must be positive")
        if duty_cycle_myr < 0.0:
            raise ValueError("AGN duty cycle must be nonnegative")

        self.smbh_mass_msun = float(smbh_mass_msun)
        self.agn_lifetime_myr = float(duty_cycle_myr)
        self.viscosity_alpha = AGN_VISCOSITY_ALPHA
        self.gravitational_radius_cm = (
            const.G * self.smbh_mass_msun * const.msun / const.c**2
        )
        self.migration_trap_radius_cm = (
            MIGRATION_TRAP_RADIUS_RG * self.gravitational_radius_cm
        )
        self.velocity_dispersion_cm_s = VELOCITY_DISPERSION_REFERENCE_CM_S * (
            self.smbh_mass_msun / VELOCITY_DISPERSION_REFERENCE_MASS_MSUN
        ) ** VELOCITY_DISPERSION_MASS_SLOPE


class merger_flags:
    """Integer flags used by the compact-object merger model."""

    def __init__(self, N: int) -> None:
        self.N = N
        self.merger_flag: IntArray = np.zeros(N, dtype=int)
        self.trap_reached_flag: IntArray = np.zeros(N, dtype=int)
        self.GWescape_flag: IntArray = np.zeros(N, dtype=int)
        self.limited_num_flag: IntArray = np.zeros(N, dtype=int)
        self.continue_ng_flag: IntArray = np.zeros(N, dtype=int)
        self.model_err_flag: IntArray = np.zeros(N, dtype=int)
        self.hard: IntArray = np.zeros(N, dtype=int)


class damp_flag:
    """Damping state: 0=star in disk, 1=BH in disk, 2=outside disk."""

    def __init__(self, N: int) -> None:
        self.type_flag: IntArray = np.zeros(N, dtype=int)
