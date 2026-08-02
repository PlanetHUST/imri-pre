"""Linear redshift evolution helpers for simplified cosmic IMRI workflows."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RedshiftEvolutionState:
    """System redshift at a source-frame elapsed time."""

    elapsed_time_myr: float
    redshift: float


@dataclass(frozen=True)
class LinearRedshiftEvolution:
    """Evolve redshift with the prompt's linear ``z = H0 * (t0 - t)`` model."""

    hubble_constant_per_myr: float

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.hubble_constant_per_myr)
            or self.hubble_constant_per_myr <= 0.0
        ):
            raise ValueError("Hubble constant must be positive")

    def redshift_after_time(
        self,
        initial_redshift: float,
        elapsed_time_myr: float,
    ) -> float:
        """Return redshift after ``elapsed_time_myr`` of source-frame evolution."""
        if not np.isfinite(initial_redshift) or initial_redshift < 0.0:
            raise ValueError("initial redshift must be nonnegative")
        if not np.isfinite(elapsed_time_myr) or elapsed_time_myr < 0.0:
            raise ValueError("elapsed time must be nonnegative")
        redshift = initial_redshift - self.hubble_constant_per_myr * elapsed_time_myr
        return float(max(0.0, redshift))

    def state_after_time(
        self,
        initial_redshift: float,
        elapsed_time_myr: float,
    ) -> RedshiftEvolutionState:
        """Return a compact redshift-state payload."""
        return RedshiftEvolutionState(
            elapsed_time_myr=float(elapsed_time_myr),
            redshift=self.redshift_after_time(initial_redshift, elapsed_time_myr),
        )

    def remaining_time_to_zero_redshift_myr(self, redshift: float) -> float:
        """Return source-frame time [Myr] until this model reaches z=0."""
        if not np.isfinite(redshift) or redshift < 0.0:
            raise ValueError("redshift must be nonnegative")
        return float(redshift / self.hubble_constant_per_myr)
