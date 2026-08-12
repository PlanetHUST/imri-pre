"""Active galactic nucleus state used by orbital damping and migration."""

from dataclasses import dataclass

from . import constants as const


@dataclass(frozen=True)
class AGNProperties:
    """Unit-explicit AGN values consumed by the active IMRI workflow."""

    smbh_mass_msun: float
    agn_lifetime_myr: float
    gravitational_radius_cm: float

    @classmethod
    def from_mass(
        cls,
        smbh_mass_msun: float,
        duty_cycle_myr: float,
    ) -> "AGNProperties":
        if smbh_mass_msun <= 0.0:
            raise ValueError("SMBH mass must be positive")
        if duty_cycle_myr < 0.0:
            raise ValueError("AGN duty cycle must be nonnegative")
        return cls(
            smbh_mass_msun=float(smbh_mass_msun),
            agn_lifetime_myr=float(duty_cycle_myr),
            gravitational_radius_cm=(
                const.G * float(smbh_mass_msun) * const.msun / const.c**2
            ),
        )
