"""Command line for the default single-host IMRI simulation.

Analytical Type-I is default; signed torque comparison modes are explicit CLI options.
The default CLI uses the original analytic IMF and formation-time prescription.
Published component presets are opt-in only.
"""

import argparse
import time
from pathlib import Path
from typing import Sequence

from lisa.snr import SNR_WINDOW_PLACEMENTS
from physical_process.population_models import POPULATION_MODELS, LEGACY_POPIII, PAN_NSC, PAN_PAPER, XUE_BH
from .config import (
    IMRIDynConfig, DEFAULT_DISK_MODEL, TORQUE_WITH_TRAP, SECONDS_PER_MINUTE,
    IMRI_GW_ECCENTRICITY, IMRI_LISA_MAX_GW_FREQUENCY_HZ,
    IMRI_LISA_MIN_GW_FREQUENCY_HZ, IMRI_MAX_GW_TIME_MYR,
    IMRI_DETECTOR_HUBBLE_CONSTANT_KM_S_MPC, IMRI_DETECTOR_OMEGA_M,
    IMRI_LISA_MISSION_DURATION_YEARS, IMRI_REDSHIFT, IMRI_SNR_THRESHOLD,
    IMRI_SNR_WINDOW_PLACEMENT, IMRI_COSMIC_MERGE_RADIUS_FACTOR,
    IMRI_COSMIC_MERGE_SMBH_MASS_FACTOR, IMRI_COSMIC_MERGE_TIME_MYR,
)
from .workflow import run_simulation

DEFAULT_NUM_REPEATS = 1
DEFAULT_OUTPUT_DIR = Path("outputs/single_host")
DEFAULT_DAMPING_CPU_COUNT = None
DEFAULT_DUTY_CYCLE_MYR = 10.0
DEFAULT_SMBH_MASSES_MSUN = (1e3, 1e4, 1e5, 1e6)
DEFAULT_NSC_MASSES_MSUN = (2e5,)
DEFAULT_METALLICITY = 1e-5
DEFAULT_INITIAL_STELLAR_DENSITY_GAMMA = 7.0 / 4.0  # Legacy fallback; published presets resolve below.
DEFAULT_CHECKPOINT_GRID_INTERVAL = 5


def _parse_mass_vector(value: str) -> list[float]:
    items = value.strip().removeprefix("[").removesuffix("]").split(",")
    try:
        masses = [float(item.strip()) for item in items if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("mass vectors must contain comma-separated numbers") from error
    if not masses:
        raise argparse.ArgumentTypeError("mass vectors must not be empty")
    return masses


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simulate one NSC/AGN host and its LISA IMRIs")
    parser.add_argument("num_repeats", nargs="?", type=int, default=DEFAULT_NUM_REPEATS)
    parser.add_argument("output_dir", nargs="?", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("damping_cpu_count", nargs="?", type=int, default=DEFAULT_DAMPING_CPU_COUNT)
    parser.add_argument("--duty-cycle-myr", type=float, default=DEFAULT_DUTY_CYCLE_MYR)
    parser.add_argument("--smbh-masses", type=_parse_mass_vector, default=DEFAULT_SMBH_MASSES_MSUN)
    parser.add_argument("--nsc-masses", type=_parse_mass_vector, default=DEFAULT_NSC_MASSES_MSUN)
    parser.add_argument("--migration-model", choices=(DEFAULT_DISK_MODEL, TORQUE_WITH_TRAP), default=DEFAULT_DISK_MODEL)
    parser.add_argument("--torque-no-gap-transport", action="store_true",
                        help="Torque comparison: use ungapped Type-I + validity-masked thermal only")
    parser.add_argument("--disk-alpha", type=float, default=0.1)
    parser.add_argument("--disk-accretion-rate-g-s", type=float, default=None)
    parser.add_argument("--disk-outer-radius-cm", type=float, default=None)
    parser.add_argument("--nsc-radius-pc-at-reference-mass", type=float, default=3.31)
    parser.add_argument("--nsc-radius-mass-slope", type=float, default=0.321)
    parser.add_argument("--metallicity", type=float, default=DEFAULT_METALLICITY)
    parser.add_argument("--population-model", choices=POPULATION_MODELS, default=None,
                        help="Default: legacy_popiii (original IMF); published component presets are opt-in")
    parser.add_argument("--popiii-min-mass-msun", type=float, default=13.6)
    parser.add_argument("--popiii-max-mass-msun", type=float, default=197.0)
    parser.add_argument("--popiii-imf-slope", type=float, default=1.77)
    parser.add_argument("--initial-stellar-density-gamma", type=float, default=None,
                        help="Legacy: 1.75; Pan fiducial: 1.5; Xue radial proxy: 2")
    parser.add_argument("--ignore-disk-inner-radius", action="store_true")
    parser.add_argument("--lisa-min-gw-frequency-hz", type=float, default=IMRI_LISA_MIN_GW_FREQUENCY_HZ)
    parser.add_argument("--lisa-max-gw-frequency-hz", type=float, default=IMRI_LISA_MAX_GW_FREQUENCY_HZ)
    parser.add_argument("--max-gw-time-myr", type=float, default=IMRI_MAX_GW_TIME_MYR)
    parser.add_argument("--gw-eccentricity", type=float, default=IMRI_GW_ECCENTRICITY)
    parser.add_argument("--redshift", "--red-shift", dest="redshift", type=float, default=IMRI_REDSHIFT)
    parser.add_argument("--snr-threshold", type=float, default=IMRI_SNR_THRESHOLD)
    optional = parser.add_argument_group("optional SEVN backend (not used by default)")
    optional.add_argument("--stellar-evolution-model", choices=("fixed_mass", "sevn"), default="fixed_mass")
    optional.add_argument("--natal-kick-model", choices=("none", "sevn"), default="none")
    optional.add_argument("--sevn-hydrogen-tables", type=Path)
    optional.add_argument("--sevn-helium-tables", type=Path)
    parser.add_argument("--lisa-mission-duration-years", type=float, choices=(0.5, 1.0, 2.0, 4.0), default=IMRI_LISA_MISSION_DURATION_YEARS)
    parser.add_argument("--snr-window-placement", choices=SNR_WINDOW_PLACEMENTS, default=IMRI_SNR_WINDOW_PLACEMENT)
    parser.add_argument("--detector-hubble-constant-km-s-mpc", type=float, default=IMRI_DETECTOR_HUBBLE_CONSTANT_KM_S_MPC)
    parser.add_argument("--detector-omega-m", type=float, default=IMRI_DETECTOR_OMEGA_M)
    parser.add_argument("--cosmic-merge-time-myr", type=float, default=IMRI_COSMIC_MERGE_TIME_MYR)
    parser.add_argument("--cosmic-merge-smbh-mass-factor", type=float, default=IMRI_COSMIC_MERGE_SMBH_MASS_FACTOR)
    parser.add_argument("--cosmic-merge-radius-factor", type=float, default=IMRI_COSMIC_MERGE_RADIUS_FACTOR)
    parser.add_argument("--checkpoint-interval", type=int, default=DEFAULT_CHECKPOINT_GRID_INTERVAL)
    parser.add_argument("--debug-results", action="store_true")
    parser.add_argument("--base-seed", type=int, default=None)
    parser.add_argument("--no-consumption-replenishment", action="store_true")
    return parser


def build_config_from_arguments(arguments: argparse.Namespace) -> IMRIDynConfig:
    custom_imf = (arguments.popiii_min_mass_msun, arguments.popiii_max_mass_msun,
                  arguments.popiii_imf_slope) != (13.6, 197.0, 1.77)
    model = arguments.population_model or LEGACY_POPIII
    if model != LEGACY_POPIII and custom_imf:
        raise ValueError("Published component populations do not use Pop III IMF controls; select legacy_popiii for those experiments")
    return IMRIDynConfig(
        smbh_masses_msun=arguments.smbh_masses, nsc_masses_msun=arguments.nsc_masses,
        num_repeats=arguments.num_repeats, duty_cycle_myr=arguments.duty_cycle_myr,
        output_dir=arguments.output_dir, metallicity=arguments.metallicity,
        disk_alpha=arguments.disk_alpha,
        disk_accretion_rate_g_s=arguments.disk_accretion_rate_g_s,
        disk_outer_radius_cm=arguments.disk_outer_radius_cm,
        nsc_radius_pc_at_reference_mass=arguments.nsc_radius_pc_at_reference_mass,
        nsc_radius_mass_slope=arguments.nsc_radius_mass_slope,
        population_model=model,
        popiii_min_mass_msun=arguments.popiii_min_mass_msun,
        popiii_max_mass_msun=arguments.popiii_max_mass_msun,
        popiii_imf_slope=arguments.popiii_imf_slope,
        initial_stellar_density_gamma=(arguments.initial_stellar_density_gamma
            if arguments.initial_stellar_density_gamma is not None else
            1.5 if model in (PAN_NSC, PAN_PAPER) else
            2.0 if model == XUE_BH else DEFAULT_INITIAL_STELLAR_DENSITY_GAMMA),
        use_disk_inner_radius=not arguments.ignore_disk_inner_radius,
        disk_model=arguments.migration_model,
        torque_gap_transport=not arguments.torque_no_gap_transport,
        lisa_min_gw_frequency_hz=arguments.lisa_min_gw_frequency_hz,
        lisa_max_gw_frequency_hz=arguments.lisa_max_gw_frequency_hz,
        max_gw_time_myr=arguments.max_gw_time_myr,
        gw_eccentricity=arguments.gw_eccentricity, redshift=arguments.redshift,
        snr_threshold=arguments.snr_threshold,
        lisa_mission_duration_years=arguments.lisa_mission_duration_years,
        snr_window_placement=arguments.snr_window_placement,
        detector_hubble_constant_km_s_mpc=arguments.detector_hubble_constant_km_s_mpc,
        detector_omega_m=arguments.detector_omega_m,
        cosmic_merge_time_myr=arguments.cosmic_merge_time_myr,
        cosmic_merge_smbh_mass_factor=arguments.cosmic_merge_smbh_mass_factor,
        cosmic_merge_radius_factor=arguments.cosmic_merge_radius_factor,
        damping_cpu_count=arguments.damping_cpu_count,
        checkpoint_grid_interval=arguments.checkpoint_interval,
        print_debug_results=arguments.debug_results, base_seed=arguments.base_seed,
        consumption_replenishment=not arguments.no_consumption_replenishment,
        stellar_evolution_model=arguments.stellar_evolution_model,
        natal_kick_model=arguments.natal_kick_model,
        sevn_hydrogen_tables=arguments.sevn_hydrogen_tables,
        sevn_helium_tables=arguments.sevn_helium_tables,
    )


def run_cli(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        config = build_config_from_arguments(arguments)
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    start = time.time()
    run_simulation(config)
    elapsed = time.time() - start
    print(f"Total runtime: {elapsed:.2f} seconds = {elapsed/SECONDS_PER_MINUTE:.2f} minutes")
