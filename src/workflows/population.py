"""Stellar population, orbital sampling, and AGN-disk selection stage."""

from dataclasses import dataclass

import numpy as np

import physical_process.constants as const
import physical_process.nsc_orbital as nsc_orbital
import physical_process.sampling as stellar_sampling

from .config import IMRIDynConfig, ORBITAL_SAMPLING_MIN_RADIUS_RG, YEARS_PER_MYR
from .models import DiskProfiles, GeometricPopulation, StellarPopulation


@dataclass(frozen=True)
class PopulationSelection:
    """Initial and disk-crossing populations for one realization."""

    initial: StellarPopulation
    disk_crossing: GeometricPopulation
    mean_relaxation_time_myr: float
    selected_indices: np.ndarray

    @property
    def total_count(self) -> int:
        return len(self.initial["mass_msun"])

    @property
    def disk_crossing_count(self) -> int:
        return len(self.disk_crossing["mass_msun"])

    @property
    def geometric_fraction(self) -> float:
        return (
            self.disk_crossing_count / self.total_count if self.total_count > 0 else 0.0
        )


def sample_disk_crossing_population(
    m_smbh_msun: float,
    m_nsc_msun: float,
    disk_data: DiskProfiles,
    config: IMRIDynConfig,
) -> PopulationSelection:
    """Sample one NSC population and select orbits crossing the AGN disk."""
    mass_msun, lifetime_yr = stellar_sampling.sample_efficient(
        m_nsc_msun,
        Z=config.metallicity,
        popiii_max_mass_msun=config.popiii_max_mass_msun,
        popiii_imf_slope=config.popiii_imf_slope,
    )
    nsc_radius_cm = nsc_orbital.estimate_nsc_radius_cm(
        m_nsc_msun,
        config.nsc_radius_pc_at_reference_mass,
        config.nsc_radius_reference_mass_msun,
        config.nsc_radius_mass_slope,
    )
    semi_major_axis_cm, eccentricity, inclination_rad, cos_omega = (
        nsc_orbital.initial_star(
            star_count=len(mass_msun),
            smbh_mass_msun=m_smbh_msun,
            minimum_radius_cm=(
                ORBITAL_SAMPLING_MIN_RADIUS_RG
                * const.G
                * m_smbh_msun
                * const.msun
                / const.c**2
            ),
            max_radius_cm=nsc_radius_cm,
            gamma=config.initial_stellar_density_gamma,
        )
    )
    initial: StellarPopulation = {
        "mass_msun": np.asarray(mass_msun, dtype=float),
        "formation_time_myr": np.asarray(lifetime_yr / YEARS_PER_MYR, dtype=float),
        "semi_major_axis_cm": np.asarray(semi_major_axis_cm, dtype=float),
        "eccentricity": np.asarray(eccentricity, dtype=float),
        "inclination_rad": np.asarray(inclination_rad, dtype=float),
        "cos_omega": np.asarray(cos_omega, dtype=float),
        "nsc_radius_cm": nsc_radius_cm,
    }
    crossing_mask = nsc_orbital.disk_crossing_mask(
        initial["semi_major_axis_cm"],
        initial["eccentricity"],
        initial["cos_omega"],
        disk_data["inner_radius_cm"],
        disk_data["outer_radius_cm"],
        config.use_disk_inner_radius,
    )
    selected_indices = np.where(crossing_mask)[0]
    disk_crossing: GeometricPopulation = {
        "mass_msun": initial["mass_msun"][selected_indices],
        "formation_time_myr": initial["formation_time_myr"][selected_indices],
        "semi_major_axis_cm": initial["semi_major_axis_cm"][selected_indices],
        "eccentricity": initial["eccentricity"][selected_indices],
        "inclination_rad": initial["inclination_rad"][selected_indices],
        "cos_omega": initial["cos_omega"][selected_indices],
    }
    if selected_indices.size == 0:
        mean_relaxation_time_myr = np.nan
    else:
        relaxation_time_myr = (
            np.asarray(
                [
                    nsc_orbital.relaxaion_timescale_s(m_smbh_msun, orbit_radius_cm)
                    for orbit_radius_cm in disk_crossing["semi_major_axis_cm"]
                ],
                dtype=float,
            )
            / const.Myr
        )
        mean_relaxation_time_myr = float(np.mean(relaxation_time_myr))

    return PopulationSelection(
        initial=initial,
        disk_crossing=disk_crossing,
        mean_relaxation_time_myr=mean_relaxation_time_myr,
        selected_indices=selected_indices,
    )
