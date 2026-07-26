"""State containers used by the AGN and compact-object simulations."""

from pathlib import Path

import numpy as np
import numpy.typing as npt

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
    """Binary black-hole population arrays with legacy attribute names."""

    def __init__(self, numBH: int) -> None:
        self.numBH = numBH
        self.idi: IntArray = np.zeros(numBH, dtype=int)
        self.m1: FloatArray = np.zeros(numBH, dtype=float)  # Msun
        self.m2: FloatArray = np.zeros(numBH, dtype=float)  # Msun
        self.sma: FloatArray = np.zeros(numBH, dtype=float)  # Rsun
        self.ecc: FloatArray = np.zeros(numBH, dtype=float)
        self.sma2: FloatArray = np.zeros(numBH, dtype=float)  # Rsun
        self.ecc2: FloatArray = np.zeros(numBH, dtype=float)
        self.ecc10: FloatArray = np.zeros(numBH, dtype=float)
        self.ecc0p1: FloatArray = np.zeros(numBH, dtype=float)
        self.chi1: FloatArray = np.zeros(numBH, dtype=float)
        self.chi2: FloatArray = np.zeros(numBH, dtype=float)
        self.th1: FloatArray = np.zeros(numBH, dtype=float)  # rad
        self.th2: FloatArray = np.zeros(numBH, dtype=float)  # rad
        self.A: FloatArray = np.zeros(numBH, dtype=float)  # cm
        self.mmerg: FloatArray = np.zeros(numBH, dtype=float)  # Msun
        self.chimerg: FloatArray = np.zeros(numBH, dtype=float)
        self.vk: FloatArray = np.zeros(numBH, dtype=float)  # cm/s
        self.m2min: float = 3.0  # Msun
        self.trapreach_time: FloatArray = np.zeros(numBH, dtype=float)  # Myr


class stellar_properties:
    """Stellar population arrays with units documented on each attribute."""

    def __init__(self, numBH: int) -> None:
        self.numBH = numBH
        self.idi: IntArray = np.zeros(numBH, dtype=int)
        self.m: FloatArray = np.zeros(numBH, dtype=float)  # Msun
        self.mBH: FloatArray = np.zeros(numBH, dtype=float)  # Msun
        self.a: FloatArray = np.zeros(numBH, dtype=float)  # cm
        self.ecc: FloatArray = np.zeros(numBH, dtype=float)
        self.i: FloatArray = np.zeros(numBH, dtype=float)  # rad
        self.cosw: FloatArray = np.zeros(numBH, dtype=float)

    def init_Stellar(self) -> None:
        """Compatibility placeholder for the historical initialization API."""


class AGN_properties:
    """AGN scalar properties.

    ``M_SMBH`` is in solar masses, ``AGNlifetime`` is in Myr, radii are in
    centimeters, and ``sigma`` is in cm/s. Legacy names are retained because
    the damping API consumes them directly.
    """

    def __init__(self) -> None:
        self.AGNlifetime: float = 0.0
        self.vesc: float = 0.0
        self.M_SMBH: float = 0.0
        self.rg: float = 0.0
        self.alpha: float = 0.0
        self.rtrap: float = 0.0
        self.maverage: float = 0.0
        self.sigma: float = 0.0

    def init_AGN(self, MSMBH: float, dutycycle: float) -> None:
        """Initialize an AGN from SMBH mass [Msun] and lifetime [Myr]."""
        if MSMBH <= 0.0:
            raise ValueError("SMBH mass must be positive")
        if dutycycle < 0.0:
            raise ValueError("AGN duty cycle must be nonnegative")

        self.M_SMBH = float(MSMBH)
        self.AGNlifetime = float(dutycycle)
        self.alpha = AGN_VISCOSITY_ALPHA
        self.rg = const.G * self.M_SMBH * const.msun / const.c**2
        self.rtrap = MIGRATION_TRAP_RADIUS_RG * self.rg
        self.sigma = VELOCITY_DISPERSION_REFERENCE_CM_S * (
            self.M_SMBH / VELOCITY_DISPERSION_REFERENCE_MASS_MSUN
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
