"""Shared flat-LambdaCDM time, redshift, and distance relations."""

from dataclasses import dataclass
from math import asinh, sinh, sqrt

import numpy as np
from scipy.integrate import quad

from physical_process.constants import MEGAYEAR_S, MPC_M, SPEED_OF_LIGHT_M_S

COSMOLOGY_QUADRATURE_RELATIVE_TOLERANCE = 1.0e-11


@dataclass(frozen=True)
class FlatLambdaCDM:
    """Flat matter-plus-Lambda cosmology with source times measured in Myr."""

    hubble_constant_km_s_mpc: float
    omega_m: float

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.hubble_constant_km_s_mpc)
            or self.hubble_constant_km_s_mpc <= 0.0
        ):
            raise ValueError("H0 must be finite and positive")
        if not np.isfinite(self.omega_m) or not 0.0 < self.omega_m < 1.0:
            raise ValueError("Omega_m must satisfy 0 < Omega_m < 1")

    @property
    def omega_lambda(self) -> float:
        return 1.0 - self.omega_m

    @property
    def hubble_constant_per_myr(self) -> float:
        """Return H0 converted from km/s/Mpc to Myr^-1."""
        return (
            self.hubble_constant_km_s_mpc
            * 1000.0
            / MPC_M
            * MEGAYEAR_S
        )

    def cosmic_age_myr(self, redshift: float) -> float:
        """Return cosmic age at ``redshift`` [Myr]."""
        self._validate_redshift(redshift)
        omega_lambda = self.omega_lambda
        return float(
            2.0
            / (3.0 * self.hubble_constant_per_myr * sqrt(omega_lambda))
            * asinh(
                sqrt(omega_lambda / self.omega_m)
                * (1.0 + redshift) ** (-1.5)
            )
        )

    def redshift_at_cosmic_age_myr(self, age_myr: float) -> float:
        """Return redshift at a cosmic age within the modeled history."""
        age_at_z0_myr = self.cosmic_age_myr(0.0)
        if (
            not np.isfinite(age_myr)
            or age_myr <= 0.0
            or age_myr > age_at_z0_myr
        ):
            raise ValueError("cosmic age must satisfy 0 < age <= age(z=0)")
        omega_lambda = self.omega_lambda
        expansion = sinh(
            1.5
            * self.hubble_constant_per_myr
            * sqrt(omega_lambda)
            * age_myr
        )
        return float(
            (sqrt(omega_lambda / self.omega_m) / expansion) ** (2.0 / 3.0)
            - 1.0
        )

    def evolve_redshift(
        self,
        initial_redshift: float,
        elapsed_source_myr: float,
    ) -> float:
        """Return redshift after positive source-frame elapsed time [Myr]."""
        self._validate_redshift(initial_redshift)
        if not np.isfinite(elapsed_source_myr) or elapsed_source_myr < 0.0:
            raise ValueError("elapsed source time must be finite and nonnegative")
        if elapsed_source_myr == 0.0:
            return float(initial_redshift)
        final_age_myr = self.cosmic_age_myr(initial_redshift) + elapsed_source_myr
        if final_age_myr >= self.cosmic_age_myr(0.0):
            return 0.0
        return max(0.0, self.redshift_at_cosmic_age_myr(final_age_myr))

    def remaining_time_to_z0_myr(self, redshift: float) -> float:
        """Return source-frame time remaining until redshift zero [Myr]."""
        self._validate_redshift(redshift)
        return float(self.cosmic_age_myr(0.0) - self.cosmic_age_myr(redshift))

    def luminosity_distance_mpc(self, redshift: float) -> float:
        """Return flat matter-plus-Lambda luminosity distance [Mpc]."""
        self._validate_redshift(redshift)
        if redshift == 0.0:
            return 0.0

        def inverse_expansion_rate(integration_redshift: float) -> float:
            return 1.0 / sqrt(
                self.omega_m * (1.0 + integration_redshift) ** 3
                + self.omega_lambda
            )

        comoving_integral, _ = quad(
            inverse_expansion_rate,
            0.0,
            redshift,
            epsabs=0.0,
            epsrel=COSMOLOGY_QUADRATURE_RELATIVE_TOLERANCE,
            limit=200,
        )
        speed_of_light_km_s = SPEED_OF_LIGHT_M_S / 1000.0
        return float(
            (1.0 + redshift)
            * speed_of_light_km_s
            / self.hubble_constant_km_s_mpc
            * comoving_integral
        )

    @staticmethod
    def _validate_redshift(redshift: float) -> None:
        if not np.isfinite(redshift) or redshift < 0.0:
            raise ValueError("redshift must be finite and nonnegative")
