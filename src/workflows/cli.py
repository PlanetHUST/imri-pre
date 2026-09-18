"""Command-line interface for the dynamical IMRI workflow."""

import argparse
import time
from pathlib import Path
from typing import Sequence

from physical_process.disk_migration import (
    DEFAULT_MIGRATION_MODEL,
    GAP_OPENING_CONDITIONS,
    MIGRATION_MODELS,
)

from .config import (
    DEFAULT_BASE_SEED,
    DEFAULT_DISK_MODEL,
    DEFAULT_GAP_OPENING_CONDITION,
    DISK_MODELS,
    IMRI_GW_ECCENTRICITY,
    IMRI_LISA_MAX_GW_FREQUENCY_HZ,
    IMRI_LISA_MIN_GW_FREQUENCY_HZ,
    IMRI_MAX_GW_TIME_MYR,
    IMRI_DETECTOR_HUBBLE_CONSTANT_KM_S_MPC,
    IMRI_DETECTOR_OMEGA_M,
    IMRI_LISA_MISSION_DURATION_YEARS,
    IMRI_REDSHIFT,
    IMRI_SNR_THRESHOLD,
    IMRI_SNR_WINDOW_PLACEMENT,
    IMRIDynConfig,
    SECONDS_PER_MINUTE,
)
from .lisa_snr import SNR_WINDOW_PLACEMENTS
from .run_figures import generate_production_figures
from .workflow import run_simulation

DEFAULT_NUM_REPEATS = 1
DEFAULT_OUTPUT_DIR = Path("output/dyn_results")
DEFAULT_DAMPING_CPU_COUNT = None
DEFAULT_DUTY_CYCLE_MYR = 10.0
DEFAULT_SMBH_MASSES_MSUN = (1e3, 1e4, 1e5, 1e6)
DEFAULT_NSC_MASSES_MSUN = (2e5,)
DEFAULT_METALLICITY = 1e-5
DEFAULT_INITIAL_STELLAR_DENSITY_GAMMA = 7.0 / 4.0
DEFAULT_CHECKPOINT_GRID_INTERVAL = 5


def _parse_mass_vector(value: str) -> list[float]:
    """Parse a comma-separated mass vector, optionally enclosed in brackets."""
    items = value.strip().removeprefix("[").removesuffix("]").split(",")
    try:
        masses = [float(item.strip()) for item in items if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "mass vectors must contain comma-separated numbers"
        ) from error
    if not masses:
        raise argparse.ArgumentTypeError("mass vectors must not be empty")
    return masses


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Run the dynamical NSC-AGN IMRI-rate simulation."
    )
    parser.add_argument(
        "num_repeats",
        nargs="?",
        type=int,
        default=DEFAULT_NUM_REPEATS,
        help="Monte Carlo repeats per grid point",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="output directory",
    )
    parser.add_argument(
        "damping_cpu_count",
        nargs="?",
        type=int,
        default=DEFAULT_DAMPING_CPU_COUNT,
        help="worker count; scheduler/local CPU count is used when omitted",
    )
    parser.add_argument(
        "--duty-cycle-myr",
        type=float,
        default=DEFAULT_DUTY_CYCLE_MYR,
    )
    parser.add_argument(
        "--smbh-masses",
        "--smbh-mass-grid",
        dest="smbh_masses",
        type=_parse_mass_vector,
        default=DEFAULT_SMBH_MASSES_MSUN,
        metavar="M1,M2,...",
        help="comma-separated SMBH mass values in Msun",
    )
    parser.add_argument(
        "--nsc-masses",
        "--nsc-mass-grid",
        dest="nsc_masses",
        type=_parse_mass_vector,
        default=DEFAULT_NSC_MASSES_MSUN,
        metavar="M1,M2,...",
        help="comma-separated NSC mass values in Msun",
    )
    parser.add_argument("--metallicity", type=float, default=DEFAULT_METALLICITY)

    parser.add_argument(
        "--initial-stellar-density-gamma",
        type=float,
        default=DEFAULT_INITIAL_STELLAR_DENSITY_GAMMA,
        help="initial stellar radial number-density power-law index",
    )
    parser.add_argument(
        "--ignore-disk-inner-radius",
        action="store_true",
        help="select disk-crossing objects using only the disk outer radius",
    )
    parser.add_argument(
        "--no-type-ii-migration",
        action="store_true",
        help="legacy alias that forces the ungapped migration model",
    )
    parser.add_argument(
        "--migration-model",
        choices=MIGRATION_MODELS,
        default=DEFAULT_MIGRATION_MODEL,
        help="torque composition model (default: continuous_gap)",
    )
    parser.add_argument(
        "--gap-open-condition",
        choices=GAP_OPENING_CONDITIONS,
        default=DEFAULT_GAP_OPENING_CONDITION,
        help=(
            "deep-gap diagnostic; selects the hard Type-II branch only for "
            "classical_type2"
        ),
    )
    parser.add_argument(
        "--disk-model",
        choices=DISK_MODELS,
        default=DEFAULT_DISK_MODEL,
        help=(
            "gas-migration scenario; this does not select the underlying "
            "Sirko/Thompson disk structure"
        ),
    )
    parser.add_argument(
        "--lisa-min-gw-frequency-hz",
        type=float,
        default=IMRI_LISA_MIN_GW_FREQUENCY_HZ,
        help="lower GW frequency defining LISA-band entry",
    )
    parser.add_argument(
        "--lisa-max-gw-frequency-hz",
        type=float,
        default=IMRI_LISA_MAX_GW_FREQUENCY_HZ,
        help="upper GW frequency recorded for the LISA band",
    )
    parser.add_argument(
        "--max-gw-time-myr",
        type=float,
        default=IMRI_MAX_GW_TIME_MYR,
        help="maximum Peters GW time from the IMRI threshold to LISA entry",
    )
    parser.add_argument(
        "--gw-eccentricity",
        type=float,
        default=IMRI_GW_ECCENTRICITY,
        help="fixed eccentricity used by the simplified Peters threshold",
    )
    parser.add_argument(
        "--redshift",
        "--red-shift",
        dest="redshift",
        type=float,
        default=IMRI_REDSHIFT,
        help="source redshift used for LISA frequency and time-budget conversion",
    )
    parser.add_argument(
        "--snr-threshold",
        type=float,
        default=IMRI_SNR_THRESHOLD,
        help="physical sky-averaged LISA SNR detection threshold",
    )
    parser.add_argument(
        "--lisa-mission-duration-years",
        type=float,
        choices=(0.5, 1.0, 2.0, 4.0),
        default=IMRI_LISA_MISSION_DURATION_YEARS,
        help="observer-frame mission duration selecting the RCL foreground curve",
    )
    parser.add_argument(
        "--snr-window-placement",
        choices=SNR_WINDOW_PLACEMENTS,
        default=IMRI_SNR_WINDOW_PLACEMENT,
        help=(
            "when the mission window is assumed to occur: 'optimal' ends it at "
            "the observed ISCO (merger-rate convention), 'band_entry' starts it "
            "at the observed LISA-band crossing (legacy, faintest epoch)"
        ),
    )
    parser.add_argument(
        "--detector-hubble-constant-km-s-mpc",
        type=float,
        default=IMRI_DETECTOR_HUBBLE_CONSTANT_KM_S_MPC,
        help="flat-LambdaCDM H0 used for cosmic time and luminosity distance",
    )
    parser.add_argument(
        "--detector-omega-m",
        type=float,
        default=IMRI_DETECTOR_OMEGA_M,
        help="flat-LambdaCDM matter density used for time and distance",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=DEFAULT_CHECKPOINT_GRID_INTERVAL,
        help="save partial CSV results every N grid points",
    )
    parser.add_argument(
        "--debug-results",
        action="store_true",
        help="print detailed per-realization counters",
    )
    parser.add_argument(
        "--seed",
        "--base-seed",
        dest="base_seed",
        type=int,
        default=DEFAULT_BASE_SEED,
        help=(
            "root seed for reproducible, paired Monte Carlo tasks "
            f"(default {DEFAULT_BASE_SEED})"
        ),
    )
    parser.add_argument(
        "--n-repeats",
        dest="n_repeats",
        type=int,
        default=None,
        help="override the positional repeat count",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir_flag",
        type=Path,
        default=None,
        help="override the positional output directory",
    )
    parser.add_argument(
        "--workers",
        dest="workers",
        type=int,
        default=None,
        help="override the positional damping worker count",
    )
    parser.add_argument(
        "--plots",
        dest="plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="generate production figures for the run (default: enabled)",
    )
    return parser


def run_cli(argv: Sequence[str] | None = None) -> None:
    """Parse command-line arguments and run the IMRI simulation."""
    arguments = build_parser().parse_args(argv)
    num_repeats = (
        arguments.n_repeats
        if arguments.n_repeats is not None
        else arguments.num_repeats
    )
    output_dir = (
        arguments.output_dir_flag
        if arguments.output_dir_flag is not None
        else arguments.output_dir
    )
    damping_cpu_count = (
        arguments.workers
        if arguments.workers is not None
        else arguments.damping_cpu_count
    )
    config = IMRIDynConfig(
        smbh_masses_msun=arguments.smbh_masses,
        nsc_masses_msun=arguments.nsc_masses,
        num_repeats=num_repeats,
        duty_cycle_myr=arguments.duty_cycle_myr,
        output_dir=output_dir,
        metallicity=arguments.metallicity,
        initial_stellar_density_gamma=arguments.initial_stellar_density_gamma,
        use_disk_inner_radius=not arguments.ignore_disk_inner_radius,
        consider_type_ii_migration=not arguments.no_type_ii_migration,
        disk_model=arguments.disk_model,
        gap_opening_condition=arguments.gap_open_condition,
        migration_model=arguments.migration_model,
        lisa_min_gw_frequency_hz=arguments.lisa_min_gw_frequency_hz,
        lisa_max_gw_frequency_hz=arguments.lisa_max_gw_frequency_hz,
        max_gw_time_myr=arguments.max_gw_time_myr,
        gw_eccentricity=arguments.gw_eccentricity,
        redshift=arguments.redshift,
        snr_threshold=arguments.snr_threshold,
        lisa_mission_duration_years=arguments.lisa_mission_duration_years,
        snr_window_placement=arguments.snr_window_placement,
        detector_hubble_constant_km_s_mpc=(
            arguments.detector_hubble_constant_km_s_mpc
        ),
        detector_omega_m=arguments.detector_omega_m,
        damping_cpu_count=damping_cpu_count,
        checkpoint_grid_interval=arguments.checkpoint_interval,
        print_debug_results=arguments.debug_results,
        base_seed=arguments.base_seed,
    )
    start_time_s = time.time()
    run_simulation(config)
    runtime_s = time.time() - start_time_s
    if arguments.plots:
        figures = generate_production_figures(
            config.output_dir,
            selected_nsc_mass_msun=float(config.nsc_masses_msun[0]),
        )
        print(f"Production figures written under: {config.output_dir / 'figures'}")
        for name, paths in figures.items():
            if isinstance(paths, list):
                for path in paths:
                    print(f"  figure[{name}]: {path}")
            else:
                print(f"  data[{name}]: {paths}")
    print(
        f"Total runtime: {runtime_s:.2f} seconds = "
        f"{runtime_s/SECONDS_PER_MINUTE:.2f} minutes"
    )
