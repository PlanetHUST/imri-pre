"""Configuration and constants for the dynamical IMRI workflow."""

from pathlib import Path
from typing import Sequence

import numpy as np
import numpy.typing as npt

from lisa.snr import DEFAULT_SNR_WINDOW_PLACEMENT, SNR_WINDOW_PLACEMENTS
from physical_process.population_models import POPULATION_MODELS, LEGACY_POPIII, PAN_NSC, PAN_PAPER, XUE_BH

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
IMRI_REPLENISHMENT_LEDGER_FILENAME = "imri_replenishment_ledger.csv"
IMRI_STELLAR_LEDGER_FILENAME = "imri_stellar_ledger.csv"
IMRI_SOURCE_TRANSITIONS_FILENAME = "imri_source_transitions.csv"
IMRI_WORKFLOW_CHECKPOINTS_FILENAME = "imri_workflow_checkpoints.csv"
METADATA_FILENAME = "simulation_metadata.json"
IMRI_LISA_MIN_GW_FREQUENCY_HZ = 1.0e-3
IMRI_LISA_MAX_GW_FREQUENCY_HZ = 1.0e-1
IMRI_MAX_GW_TIME_MYR = 13.6e3
IMRI_GW_ECCENTRICITY = 0.0
IMRI_REDSHIFT = 15
IMRI_SNR_THRESHOLD = 20.0
IMRI_LISA_MISSION_DURATION_YEARS = 4.0
IMRI_SNR_WINDOW_PLACEMENT = DEFAULT_SNR_WINDOW_PLACEMENT
IMRI_DETECTOR_HUBBLE_CONSTANT_KM_S_MPC = 67.4
IMRI_DETECTOR_OMEGA_M = 0.315
IMRI_COSMIC_MERGE_TIME_MYR = np.inf
IMRI_COSMIC_MERGE_SMBH_MASS_FACTOR = 2.0
IMRI_COSMIC_MERGE_RADIUS_FACTOR = 2.0
SEMI_TYPE_I = "semi_type_i"
TORQUE_WITH_TRAP = "torque_with_trap"
SEMI_WITHOUT_TRAP = "semi_without_trap"
DISK_MODELS = (SEMI_TYPE_I, TORQUE_WITH_TRAP, SEMI_WITHOUT_TRAP)
DEFAULT_DISK_MODEL = SEMI_TYPE_I
GAP_OPENING_CONDITIONS = ("kii", "crida")  # optional Type-I/II comparison
DEFAULT_GAP_OPENING_CONDITION = "kii"
MIGRATION_CONTRACT = "jm17_grishin2024_continuous_kanagawa"
SEMI_ANALYTIC_MIGRATION_CONTRACT = "integrated_local_semi_analytic_type_i_ii"
MIGRATION_CONTRACTS = {
    SEMI_TYPE_I: "integrated_local_semi_analytic_type_i",
    TORQUE_WITH_TRAP: MIGRATION_CONTRACT,
    SEMI_WITHOUT_TRAP: SEMI_ANALYTIC_MIGRATION_CONTRACT,
}


def migration_contract_name(disk_model: str) -> str:
    """Return the contract label for a migration scenario.

    The production CLI constructs only Type-I semi-analytic migration.
    Torque/trap and Type-I/II scenarios are optional programmatic comparisons.
    """
    try:
        return MIGRATION_CONTRACTS[disk_model]
    except KeyError as error:
        raise ValueError(
            f"disk model must be one of {DISK_MODELS}"
        ) from error


class IMRIDynConfig:
    """Configuration for the dynamical IMRI simulation.

    Mass inputs are in solar masses, times are in Myr, and radii ending in
    ``_pc``, ``_cm``, or ``_rg`` are measured in parsecs, centimeters, or
    gravitational radii. Both CLI and programmatic defaults use the original
    analytic IMF sampler. Published population presets require explicit selection.
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
        # The CLI uses Type-I; alternative scenarios are programmatic only.
        disk_model: str = DEFAULT_DISK_MODEL,
        gap_opening_condition: str = DEFAULT_GAP_OPENING_CONDITION,
        consider_type_ii_migration: bool = True,
        # Simplified IMRI controls
        lisa_min_gw_frequency_hz: float,
        lisa_max_gw_frequency_hz: float,
        max_gw_time_myr: float,
        gw_eccentricity: float,
        redshift: float,
        snr_threshold: float,
        lisa_mission_duration_years: float,
        snr_window_placement: str,
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
        consumption_replenishment: bool = True,
        stellar_evolution_model: str = "fixed_mass",
        natal_kick_model: str = "none",
        sevn_hydrogen_tables: str | Path | None = None,
        sevn_helium_tables: str | Path | None = None,
        population_model: str = LEGACY_POPIII,
        popiii_min_mass_msun: float = 13.6,
        popiii_max_mass_msun: float = 197.0,
        popiii_imf_slope: float = 1.77,
        # Independent disk inputs; None preserves the historical Mdot(M) / Q=1 edge.
        disk_alpha: float = PAGN_ALPHA,
        disk_accretion_rate_g_s: float | None = None,
        disk_outer_radius_cm: float | None = None,
        # Internal NSC model parameters
        nsc_radius_pc_at_reference_mass: float = 3.31,
        nsc_radius_mass_slope: float = 0.321,
        nsc_radius_reference_mass_msun: float = 3.6e6,
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
        if not isinstance(consumption_replenishment, bool):
            raise TypeError("consumption_replenishment must be a bool")
        if stellar_evolution_model not in ("fixed_mass", "sevn"):
            raise ValueError("stellar evolution model must be fixed_mass or sevn")
        if natal_kick_model not in ("none", "sevn"):
            raise ValueError("natal kick model must be none or sevn")
        if natal_kick_model == "sevn" and stellar_evolution_model != "sevn":
            raise ValueError("SEVN kicks require SEVN stellar evolution")
        if stellar_evolution_model == "sevn" and (
            sevn_hydrogen_tables is None or sevn_helium_tables is None
        ):
            raise ValueError("SEVN requires both hydrogen and helium table paths")
        if population_model not in POPULATION_MODELS:
            raise ValueError(f"population model must be one of {POPULATION_MODELS}")
        if population_model in (PAN_PAPER, XUE_BH) and len(set(nsc_masses_msun)) > 1:
            raise ValueError("This paper-normalized population has no independent NSC-mass grid")
        if population_model != LEGACY_POPIII and stellar_evolution_model != "fixed_mass":
            raise ValueError("pre-existing published populations currently require fixed_mass, not SEVN ZAMS evolution")
        if population_model in (PAN_NSC, PAN_PAPER) and initial_stellar_density_gamma < 1:
            raise ValueError("Pan Kepler-tracer inversion requires gamma >= 1")
        if population_model == XUE_BH and not np.isclose(initial_stellar_density_gamma, 2):
            raise ValueError("Xue dN/dr proportional to r^0 corresponds to spatial gamma=2")
        if not np.isfinite(popiii_min_mass_msun) or popiii_min_mass_msun <= 0:
            raise ValueError("Pop III minimum mass must be finite and positive")
        if not np.isfinite(popiii_max_mass_msun) or popiii_max_mass_msun <= popiii_min_mass_msun:
            raise ValueError("Pop III maximum mass must be finite and exceed minimum mass")
        if not np.isfinite(popiii_imf_slope):
            raise ValueError("Pop III IMF slope must be finite")
        if disk_model not in DISK_MODELS:
            raise ValueError(f"disk model must be one of {DISK_MODELS}")
        if gap_opening_condition not in GAP_OPENING_CONDITIONS:
            raise ValueError(
                "gap-opening condition must be one of "
                f"{GAP_OPENING_CONDITIONS}"
            )
        if not isinstance(consider_type_ii_migration, bool):
            raise TypeError("consider_type_ii_migration must be a bool")
        if not 0.0 < lisa_min_gw_frequency_hz < lisa_max_gw_frequency_hz:
            raise ValueError("LISA frequencies must satisfy 0 < f_min < f_max")
        if max_gw_time_myr <= 0.0:
            raise ValueError("maximum GW time must be positive")
        if gw_eccentricity != 0.0:
            raise ValueError("the coupled circular orbit requires gw_eccentricity=0")
        if not np.isfinite(redshift) or redshift < 0.0:
            raise ValueError("redshift must be nonnegative")
        if not np.isfinite(snr_threshold) or snr_threshold <= 0.0:
            raise ValueError("SNR threshold must be positive")
        if lisa_mission_duration_years not in (0.5, 1.0, 2.0, 4.0):
            raise ValueError("LISA mission duration must be 0.5, 1, 2, or 4 years")
        if snr_window_placement not in SNR_WINDOW_PLACEMENTS:
            raise ValueError(
                f"SNR window placement must be one of {SNR_WINDOW_PLACEMENTS}"
            )
        if (
            not np.isfinite(detector_hubble_constant_km_s_mpc)
            or detector_hubble_constant_km_s_mpc <= 0.0
        ):
            raise ValueError("detector H0 must be finite and positive")
        if not np.isfinite(detector_omega_m) or not 0.0 < detector_omega_m < 1.0:
            raise ValueError("detector Omega_m must satisfy 0 < Omega_m < 1")
        if cosmic_merge_time_myr != np.inf:
            raise ValueError(
                "the coupled fixed-mass orbit requires cosmic merger disabled "
                "(cosmic_merge_time_myr=inf)"
            )
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
        for name, value in (
            ("disk_alpha", disk_alpha),
            ("disk_accretion_rate_g_s", disk_accretion_rate_g_s),
            ("disk_outer_radius_cm", disk_outer_radius_cm),
            ("nsc_radius_pc_at_reference_mass", nsc_radius_pc_at_reference_mass),
            ("nsc_radius_reference_mass_msun", nsc_radius_reference_mass_msun),
        ):
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(metallicity) or metallicity < 0:
            raise ValueError("metallicity must be finite and nonnegative")
        if not np.isfinite(nsc_radius_mass_slope):
            raise ValueError("NSC radius mass slope must be finite")
        self.disk_alpha = float(disk_alpha)
        self.disk_accretion_rate_g_s = disk_accretion_rate_g_s
        self.disk_outer_radius_cm = disk_outer_radius_cm
        self.smbh_masses_msun = np.asarray(smbh_masses_msun, dtype=float)
        self.nsc_masses_msun = np.asarray(nsc_masses_msun, dtype=float)
        self.num_repeats = num_repeats
        self.duty_cycle_myr = duty_cycle_myr
        self.output_dir = Path(output_dir)

        self.population_model = population_model
        self.popiii_min_mass_msun = float(popiii_min_mass_msun)
        self.popiii_max_mass_msun = float(popiii_max_mass_msun)
        self.popiii_imf_slope = float(popiii_imf_slope)
        self.metallicity = metallicity
        self.initial_stellar_density_gamma = initial_stellar_density_gamma
        self.nsc_radius_pc_at_reference_mass = nsc_radius_pc_at_reference_mass
        self.nsc_radius_mass_slope = nsc_radius_mass_slope
        self.nsc_radius_reference_mass_msun = nsc_radius_reference_mass_msun

        self.use_disk_inner_radius = use_disk_inner_radius
        # Type-I semi-analytic is the default; other models are opt-in.
        self.disk_model = disk_model
        self.gap_opening_condition = gap_opening_condition
        self.consider_type_ii_migration = consider_type_ii_migration
        self.lisa_min_gw_frequency_hz = lisa_min_gw_frequency_hz
        self.lisa_max_gw_frequency_hz = lisa_max_gw_frequency_hz
        self.max_gw_time_myr = max_gw_time_myr
        self.gw_eccentricity = gw_eccentricity
        self.redshift = redshift
        self.snr_threshold = snr_threshold
        self.lisa_mission_duration_years = lisa_mission_duration_years
        self.snr_window_placement = snr_window_placement
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
        self.consumption_replenishment = consumption_replenishment
        self.stellar_evolution_model = stellar_evolution_model
        self.natal_kick_model = natal_kick_model
        self.sevn_hydrogen_tables = (
            Path(sevn_hydrogen_tables).expanduser().resolve()
            if sevn_hydrogen_tables is not None else None
        )
        self.sevn_helium_tables = (
            Path(sevn_helium_tables).expanduser().resolve()
            if sevn_helium_tables is not None else None
        )

    def get_param_grid(self) -> tuple[FloatArray, FloatArray]:
        """Return the explicitly selected SMBH and NSC masses [Msun]."""
        return self.smbh_masses_msun.copy(), self.nsc_masses_msun.copy()
