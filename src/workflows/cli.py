"""Command-line interface for the dynamical IMRI workflow."""

import argparse
import time
from pathlib import Path
from typing import Sequence

from .config import (
    IMRI_GW_ECCENTRICITY,
    IMRI_LISA_MAX_GW_FREQUENCY_HZ,
    IMRI_LISA_MIN_GW_FREQUENCY_HZ,
    IMRI_MAX_GW_TIME_MYR,
    IMRIDynConfig,
    SECONDS_PER_MINUTE,
)
from .workflow import run_simulation


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
        default=1,
        help="Monte Carlo repeats per grid point",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=Path("./imri_dyn_results"),
        help="output directory",
    )
    parser.add_argument(
        "damping_cpu_count",
        nargs="?",
        type=int,
        default=None,
        help="worker count; scheduler/local CPU count is used when omitted",
    )
    parser.add_argument("--duty-cycle-myr", type=float, default=10.0)
    parser.add_argument(
        "--smbh-masses",
        type=_parse_mass_vector,
        default=[3e3, 1e4, 3e4],
        metavar="M1,M2,...",
        help="comma-separated SMBH mass values in Msun",
    )
    parser.add_argument(
        "--nsc-masses",
        type=_parse_mass_vector,
        default=[2e5],
        metavar="M1,M2,...",
        help="comma-separated NSC mass values in Msun",
    )
    parser.add_argument("--metallicity", type=float, default=1e-5)
    parser.add_argument(
        "--initial-stellar-density-gamma",
        type=float,
        default=7.0 / 4.0,
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
        help="force embedded compact objects to use type-I migration",
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
        "--checkpoint-interval",
        type=int,
        default=5,
        help="save partial CSV results every N grid points",
    )
    parser.add_argument(
        "--debug-results",
        action="store_true",
        help="print detailed per-realization counters",
    )
    return parser


def run_cli(argv: Sequence[str] | None = None) -> None:
    """Parse command-line arguments and run the IMRI simulation."""
    arguments = build_parser().parse_args(argv)
    config = IMRIDynConfig(
        smbh_masses_msun=arguments.smbh_masses,
        nsc_masses_msun=arguments.nsc_masses,
        num_repeats=arguments.num_repeats,
        duty_cycle_myr=arguments.duty_cycle_myr,
        output_dir=arguments.output_dir,
        metallicity=arguments.metallicity,
        initial_stellar_density_gamma=arguments.initial_stellar_density_gamma,
        use_disk_inner_radius=not arguments.ignore_disk_inner_radius,
        consider_type_ii_migration=not arguments.no_type_ii_migration,
        lisa_min_gw_frequency_hz=arguments.lisa_min_gw_frequency_hz,
        lisa_max_gw_frequency_hz=arguments.lisa_max_gw_frequency_hz,
        max_gw_time_myr=arguments.max_gw_time_myr,
        gw_eccentricity=arguments.gw_eccentricity,
        damping_cpu_count=arguments.damping_cpu_count,
        checkpoint_grid_interval=arguments.checkpoint_interval,
        print_debug_results=arguments.debug_results,
    )
    start_time_s = time.time()
    run_simulation(config)
    runtime_s = time.time() - start_time_s
    print(
        f"Total runtime: {runtime_s:.2f} seconds = "
        f"{runtime_s/SECONDS_PER_MINUTE:.2f} minutes"
    )
