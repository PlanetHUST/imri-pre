"""Configuration and constants for the dynamical IMRI workflow."""

from pathlib import Path
from typing import Sequence

import numpy as np
import numpy.typing as npt

from physical_process.migration import (
    DEFAULT_GAP_CRITERION,
    GAP_OPENING_CONDITIONS,
    GapCriterion,
)

FloatArray = npt.NDArray[np.float64]

SECONDS_PER_MINUTE = 60.0
YEARS_PER_MYR = 1.0e6

DISK_INNER_RADIUS_RG = 6.0
ORBITAL_SAMPLING_MIN_RADIUS_RG = 6.0
PAGN_ALPHA = 0.1
DISK_ACCRETION_RATE_G_S_PER_MSUN = 1.44e18

IMRI_AVERAGED_RESULTS_FILENAME = "imri_averaged_results.csv"
IMRI_EVENTS_FILENAME = "imri_events.csv"
IMRI_REPEAT_RESULTS_FILENAME = "imri_repeat_results.csv"
METADATA_FILENAME = "simulation_metadata.json"
IMRI_LISA_MIN_GW_FREQUENCY_HZ = 1.0e-3
IMRI_LISA_MAX_GW_FREQUENCY_HZ = 1.0e-1
IMRI_MAX_GW_TIME_MYR = 13.6e3
IMRI_GW_ECCENTRICITY = 0.0
IMRI_REDSHIFT = 15
IMRI_HUBBLE_CONSTANT_PER_MYR = 1.0 / IMRI_MAX_GW_TIME_MYR
IMRI_SNR_THRESHOLD = 20.0
IMRI_LISA_MISSION_DURATION_YEARS = 4.0
IMRI_DETECTOR_HUBBLE_CONSTANT_KM_S_MPC = 67.4
IMRI_DETECTOR_OMEGA_M = 0.315
IMRI_COSMIC_MERGE_TIME_MYR = 1.0e3
IMRI_COSMIC_MERGE_SMBH_MASS_FACTOR = 2.0
IMRI_COSMIC_MERGE_RADIUS_FACTOR = 2.0
# Set in physical_process.migration.gap; never hard-coded here.
DEFAULT_GAP_OPENING_CONDITION = DEFAULT_GAP_CRITERION
TORQUE_WITH_TRAP = "torque_with_trap"
SEMI_WITHOUT_TRAP = "semi_without_trap"
DISK_MODELS = (TORQUE_WITH_TRAP, SEMI_WITHOUT_TRAP)
DEFAULT_DISK_MODEL = TORQUE_WITH_TRAP


class IMRIDynConfig:
    """Configuration for the dynamical IMRI simulation.

    Mass inputs are in solar masses, times are in Myr, and radii ending in
    ``_pc``, ``_cm``, or ``_rg`` are measured in parsecs, centimeters, or
    gravitational radii.
    """

    def __init__(
        self,
        *,
        # Sampling and runtime controls
        smbh_masses_msun: Sequence[float],
        nsc_masses_msun: Sequence[float],
        num_repeats: int,
        duty_cycle_myr: float,
        # Output location
        output_dir: str | Path,
        # Stellar and NSC model parameters
        metallicity: float,
        initial_stellar_density_gamma: float,
        # Disk-crossing and migration controls
        use_disk_inner_radius: bool,
        consider_type_ii_migration: bool,
        disk_model: str,
        # Simplified IMRI controls
        lisa_min_gw_frequency_hz: float,
        lisa_max_gw_frequency_hz: float,
        max_gw_time_myr: float,
        gw_eccentricity: float,
        redshift: float,
        hubble_constant_per_myr: float,
        snr_threshold: float,
        lisa_mission_duration_years: float,
        detector_hubble_constant_km_s_mpc: float,
        detector_omega_m: float,
        cosmic_merge_time_myr: float,
        cosmic_merge_smbh_mass_factor: float,
        cosmic_merge_radius_factor: float,
        # Execution controls
        damping_cpu_count: int | None,
        checkpoint_grid_interval: int,
        print_debug_results: bool,
        base_seed: int | None = None,
        # Internal NSC model parameters
        nsc_radius_pc_at_reference_mass: float = 3.31,
        nsc_radius_mass_slope: float = 0.321,
        nsc_radius_reference_mass_msun: float = 3.6e6,
        gap_opening_condition: str = DEFAULT_GAP_OPENING_CONDITION,
        gap_parameter_threshold: float | None = None,
    ) -> None:
        if len(smbh_masses_msun) == 0 or len(nsc_masses_msun) == 0:
            raise ValueError("SMBH and NSC mass vectors must not be empty")
        if any(mass <= 0.0 for mass in (*smbh_masses_msun, *nsc_masses_msun)):
            raise ValueError("SMBH and NSC masses must be positive")
        if num_repeats <= 0:
            raise ValueError("repeat count must be positive")
        if duty_cycle_myr <= 0.0:
            raise ValueError("AGN duty cycle must be positive")
        if not 0.0 < initial_stellar_density_gamma < 3.0:
            raise ValueError("initial stellar density gamma must satisfy 0 < gamma < 3")
        if checkpoint_grid_interval <= 0:
            raise ValueError("checkpoint interval must be positive")
        if base_seed is not None:
            if isinstance(base_seed, bool) or not isinstance(
                base_seed, (int, np.integer)
            ):
                raise TypeError("base seed must be an integer or None")
            if base_seed < 0:
                raise ValueError("base seed must be nonnegative")
        if not 0.0 < lisa_min_gw_frequency_hz < lisa_max_gw_frequency_hz:
            raise ValueError("LISA frequencies must satisfy 0 < f_min < f_max")
        if max_gw_time_myr <= 0.0:
            raise ValueError("maximum GW time must be positive")
        if not 0.0 <= gw_eccentricity < 1.0:
            raise ValueError("GW eccentricity must satisfy 0 <= e < 1")
        if not np.isfinite(redshift) or redshift < 0.0:
            raise ValueError("redshift must be nonnegative")
        if not np.isfinite(hubble_constant_per_myr) or hubble_constant_per_myr <= 0.0:
            raise ValueError("Hubble constant must be positive")
        if not np.isfinite(snr_threshold) or snr_threshold <= 0.0:
            raise ValueError("SNR threshold must be positive")
        if lisa_mission_duration_years not in (0.5, 1.0, 2.0, 4.0):
            raise ValueError("LISA mission duration must be 0.5, 1, 2, or 4 years")
        if (
            not np.isfinite(detector_hubble_constant_km_s_mpc)
            or detector_hubble_constant_km_s_mpc <= 0.0
        ):
            raise ValueError("detector H0 must be finite and positive")
        if not np.isfinite(detector_omega_m) or not 0.0 < detector_omega_m < 1.0:
            raise ValueError("detector Omega_m must satisfy 0 < Omega_m < 1")
        if not np.isfinite(cosmic_merge_time_myr) and cosmic_merge_time_myr != np.inf:
            raise ValueError("cosmic merge time must be finite or infinity")
        if cosmic_merge_time_myr <= 0.0:
            raise ValueError("cosmic merge time must be positive")
        if (
            not np.isfinite(cosmic_merge_smbh_mass_factor)
            or cosmic_merge_smbh_mass_factor <= 0.0
        ):
            raise ValueError("cosmic merger SMBH mass factor must be positive")
        if (
            not np.isfinite(cosmic_merge_radius_factor)
            or cosmic_merge_radius_factor <= 0.0
        ):
            raise ValueError("cosmic merger radius factor must be positive")
        if gap_parameter_threshold is not None and (
            not np.isfinite(gap_parameter_threshold) or gap_parameter_threshold <= 0.0
        ):
            raise ValueError("gap parameter threshold must be finite and positive")
        if gap_opening_condition not in GAP_OPENING_CONDITIONS:
            raise ValueError(
                f"gap-opening condition must be one of {GAP_OPENING_CONDITIONS}"
            )
        if disk_model not in DISK_MODELS:
            raise ValueError(f"disk model must be one of {DISK_MODELS}")

        self.smbh_masses_msun = np.asarray(smbh_masses_msun, dtype=float)
        self.nsc_masses_msun = np.asarray(nsc_masses_msun, dtype=float)
        self.num_repeats = num_repeats
        self.duty_cycle_myr = duty_cycle_myr
        self.output_dir = Path(output_dir)

        self.metallicity = metallicity
        self.initial_stellar_density_gamma = initial_stellar_density_gamma
        self.nsc_radius_pc_at_reference_mass = nsc_radius_pc_at_reference_mass
        self.nsc_radius_mass_slope = nsc_radius_mass_slope
        self.nsc_radius_reference_mass_msun = nsc_radius_reference_mass_msun

        self.use_disk_inner_radius = use_disk_inner_radius
        self.consider_type_ii_migration = consider_type_ii_migration
        self.disk_model = disk_model
        self.gap_opening_condition = gap_opening_condition
        self.gap_parameter_threshold = (
            float(gap_parameter_threshold)
            if gap_parameter_threshold is not None
            else None
        )
        self.lisa_min_gw_frequency_hz = lisa_min_gw_frequency_hz
        self.lisa_max_gw_frequency_hz = lisa_max_gw_frequency_hz
        self.max_gw_time_myr = max_gw_time_myr
        self.gw_eccentricity = gw_eccentricity
        self.redshift = redshift
        self.hubble_constant_per_myr = hubble_constant_per_myr
        self.snr_threshold = snr_threshold
        self.lisa_mission_duration_years = lisa_mission_duration_years
        self.detector_hubble_constant_km_s_mpc = (
            detector_hubble_constant_km_s_mpc
        )
        self.detector_omega_m = detector_omega_m
        self.cosmic_merge_time_myr = cosmic_merge_time_myr
        self.cosmic_merge_smbh_mass_factor = cosmic_merge_smbh_mass_factor
        self.cosmic_merge_radius_factor = cosmic_merge_radius_factor

        self.damping_cpu_count = damping_cpu_count
        self.checkpoint_grid_interval = checkpoint_grid_interval
        self.print_debug_results = print_debug_results
        self.base_seed = int(base_seed) if base_seed is not None else None

    @property
    def gap_criterion(self) -> GapCriterion:
        """Return the resolved gap criterion, including any threshold override.

        ``gap_parameter_threshold`` overrides the threshold belonging to the
        *selected* criterion only; the others keep their canonical defaults.
        """
        if self.gap_parameter_threshold is None:
            return GapCriterion(name=self.gap_opening_condition)
        field = {
            "kii": "kii_threshold",
            "kanagawa": "kanagawa_threshold",
            "crida": "crida_parameter_threshold",
        }[self.gap_opening_condition]
        return GapCriterion(
            name=self.gap_opening_condition,
            **{field: self.gap_parameter_threshold},
        )

    def get_param_grid(self) -> tuple[FloatArray, FloatArray]:
        """Return the explicitly selected SMBH and NSC masses [Msun]."""
        return self.smbh_masses_msun.copy(), self.nsc_masses_msun.copy()
