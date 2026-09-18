"""Redshift, transport-time, and cosmic-time relations for IMRI workflows."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RedshiftEvolution:
    """Evolve cosmic redshift with ``(1+z) = (H0 t)**(-2/3)``."""

    hubble_constant_per_myr: float

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.hubble_constant_per_myr)
            or self.hubble_constant_per_myr <= 0.0
        ):
            raise ValueError("Hubble constant must be positive")

    def redshift_to_cosmic_time(self, redshift: float) -> float:
        cosmic_time_myr = (1 + redshift) ** (-3 / 2) / self.hubble_constant_per_myr
        return cosmic_time_myr

    def cosmic_time_to_redshift(self, cosmic_time_myr: float) -> float:
        redshift = (self.hubble_constant_per_myr * cosmic_time_myr) ** (-2 / 3) - 1
        return redshift

    def redshift_evlove_with_time(
        self,
        initial_redshift: float,
        evolve_time_myr: float,
    ) -> float:
        """Return redshift after ``evolve_time_myr`` of source-frame evolution."""

        if not np.isfinite(initial_redshift) or initial_redshift < 0.0:
            raise ValueError("initial redshift must be nonnegative")
        if not np.isfinite(evolve_time_myr) or evolve_time_myr < 0.0:
            raise ValueError("evolved time must be nonnegative")

        time_initial_myr = self.redshift_to_cosmic_time(initial_redshift)
        time_final_myr = time_initial_myr + evolve_time_myr
        redshift = self.cosmic_time_to_redshift(time_final_myr)

        return float(max(0.0, redshift))

    def remaining_time_to_zero_redshift_myr(self, redshift: float) -> float:
        """Return source-frame time [Myr] until this model reaches z=0."""

        if not np.isfinite(redshift) or redshift < 0.0:
            raise ValueError("redshift must be nonnegative")

        time_myr = self.redshift_to_cosmic_time(redshift)

        return 1 / self.hubble_constant_per_myr - time_myr
