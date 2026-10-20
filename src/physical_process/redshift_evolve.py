"""Cosmic age and redshift in the flat matter-plus-Lambda detector cosmology."""

from dataclasses import dataclass

import numpy as np

from .constants import MEGAYEAR_S, MPC_M


@dataclass(frozen=True)
class RedshiftEvolution:
    """Map source-frame elapsed Myr to redshift using flat Lambda-CDM.

    The same ``H0`` and ``Omega_m`` also set the LISA luminosity distance.
    Radiation is omitted in both calculations.
    """

    hubble_constant_km_s_mpc: float
    omega_m: float

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.hubble_constant_km_s_mpc)
            or self.hubble_constant_km_s_mpc <= 0.0
        ):
            raise ValueError("Hubble constant must be finite and positive")
        if not np.isfinite(self.omega_m) or not 0.0 < self.omega_m < 1.0:
            raise ValueError("Omega_m must satisfy 0 < Omega_m < 1")

    @property
    def hubble_constant_per_myr(self) -> float:
        """Return H0 in inverse Myr, derived from the shared cosmology."""
        return self.hubble_constant_km_s_mpc * 1000.0 / MPC_M * MEGAYEAR_S

    def redshift_to_cosmic_time(self, redshift: float) -> float:
        """Return cosmic age [Myr] at redshift ``z >= 0``."""
        if not np.isfinite(redshift) or redshift < 0.0:
            raise ValueError("redshift must be finite and nonnegative")
        omega_lambda = 1.0 - self.omega_m
        return float(
            2.0
            / (3.0 * self.hubble_constant_per_myr * np.sqrt(omega_lambda))
            * np.arcsinh(
                np.sqrt(omega_lambda / self.omega_m)
                * (1.0 + redshift) ** (-1.5)
            )
        )

    def cosmic_time_to_redshift(self, cosmic_time_myr: float) -> float:
        """Invert cosmic age [Myr], clamping future times to ``z=0``."""
        if not np.isfinite(cosmic_time_myr) or cosmic_time_myr <= 0.0:
            raise ValueError("cosmic time must be finite and positive")
        if cosmic_time_myr >= self.redshift_to_cosmic_time(0.0):
            return 0.0
        omega_lambda = 1.0 - self.omega_m
        argument = (
            1.5
            * self.hubble_constant_per_myr
            * np.sqrt(omega_lambda)
            * cosmic_time_myr
        )
        return float(
            (np.sqrt(omega_lambda / self.omega_m) / np.sinh(argument)) ** (2.0 / 3.0)
            - 1.0
        )

    def redshift_evlove_with_time(
        self,
        initial_redshift: float,
        evolve_time_myr: float,
    ) -> float:
        """Return redshift after source-frame elapsed time [Myr]."""
        if not np.isfinite(evolve_time_myr) or evolve_time_myr < 0.0:
            raise ValueError("evolved time must be finite and nonnegative")

        initial_age_myr = self.redshift_to_cosmic_time(initial_redshift)
        return self.cosmic_time_to_redshift(initial_age_myr + evolve_time_myr)

    def remaining_time_to_zero_redshift_myr(self, redshift: float) -> float:
        """Return source-frame Myr until the same cosmology reaches ``z=0``."""
        return max(
            0.0,
            self.redshift_to_cosmic_time(0.0)
            - self.redshift_to_cosmic_time(redshift),
        )
