"""Simplified cosmic SMBH-merger interface for IMRI detectability."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CosmicMergeModel:
    """One-event SMBH merger after a source-frame post-gas delay."""

    merge_time_myr: float
    smbh_mass_growth_factor: float = 2.0
    radius_growth_factor: float = 2.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.merge_time_myr) and self.merge_time_myr != np.inf:
            raise ValueError("cosmic merge time must be finite or infinity")
        if self.merge_time_myr <= 0.0:
            raise ValueError("cosmic merge time must be positive")
        if (
            not np.isfinite(self.smbh_mass_growth_factor)
            or self.smbh_mass_growth_factor <= 0.0
        ):
            raise ValueError("SMBH mass growth factor must be positive")
        if (
            not np.isfinite(self.radius_growth_factor)
            or self.radius_growth_factor <= 0.0
        ):
            raise ValueError("radius growth factor must be positive")

    @property
    def is_active(self) -> bool:
        """Return whether a finite merger event should be applied."""
        return bool(np.isfinite(self.merge_time_myr))

    def post_merge_smbh_mass_msun(self, smbh_mass_msun: float) -> float:
        """Return the central SMBH mass [Msun] after the simplified merger."""
        return float(smbh_mass_msun * self.smbh_mass_growth_factor)

    def post_merge_radius_cm(self, orbital_radius_cm: float) -> float:
        """Return the IMRI orbital radius [cm] after the simplified merger."""
        return float(orbital_radius_cm * self.radius_growth_factor)
