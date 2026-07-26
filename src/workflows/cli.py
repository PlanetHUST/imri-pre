"""Command-line interface for the TDE-rate workflow."""

import argparse
import time
from pathlib import Path
from typing import Sequence

from .config import (
    AVERAGED_RESULTS_FILENAME,
    SECONDS_PER_MINUTE,
    TIMESCALES_FILENAME,
    VISUALIZATION_DIRECTORY,
    TDEDynConfig,
)
from .outputs import (
    ORBITAL_PARAMS_FILENAME,
    create_orbital_param_pdfs_from_csv,
    create_timescale_histograms_from_csv,
    create_visualizations_from_csv,
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
        description="Run the dynamical NSC-AGN TDE-rate simulation."
    )
    parser.add_argument(
        "num_repeats",
        nargs="?",
        type=int,
        default=1,
        help="Monte Carlo repeats per grid point (default: 3)",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=Path("./tde_dyn_results"),
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
    parser.add_argument("--metallicity", type=float, default=0.01)
    parser.add_argument(
        "--initial-stellar-density-gamma",
        type=float,
        default=7.0 / 4.0,
        help="initial stellar radial number-density power-law index",
    )
    parser.add_argument("--max-copies-per-object", type=int, default=30)
    parser.add_argument(
        "--no-resupply",
        action="store_true",
        help="disable relaxation-driven resupply copies",
    )
    parser.add_argument(
        "--resample-copy-orbits",
        action="store_true",
        help="sample new orbits instead of reusing the primary orbit",
    )
    parser.add_argument(
        "--direct-tde",
        action="store_true",
        help="count stars ending inside the configured TDE radius",
    )
    parser.add_argument(
        "--no-type-ii-migration",
        action="store_true",
        help="force all embedded stars to use type-I migration",
    )
    parser.add_argument("--tde-radius-rg", type=float, default=10.0)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="save partial CSV results every N grid points",
    )
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="skip plot generation",
    )
    parser.add_argument(
        "--intermediate-plots",
        action="store_true",
        help="write repeat-averaged initial/disk-crossing diagnostic figures",
    )
    parser.add_argument(
        "--no-debug-results",
        action="store_true",
        help="suppress detailed per-realization counters",
    )
    parser.add_argument(
        "--plot-only",
        nargs="?",
        const=Path("./tde_dyn_results"),
        type=Path,
        metavar="PATH",
        help=(
            "skip simulation and regenerate all figures from an output "
            "directory (default: ./tde_dyn_results)"
        ),
    )
    return parser


def run_cli(argv: Sequence[str] | None = None) -> None:
    """Parse command-line arguments and run the simulation."""
    arguments = build_parser().parse_args(argv)
    if arguments.plot_only is not None:
        output_dir = arguments.plot_only
        if output_dir.suffix.lower() == ".csv":
            output_dir = output_dir.parent
        visualization_dir = output_dir / VISUALIZATION_DIRECTORY
        visualization_dir.mkdir(parents=True, exist_ok=True)
        create_visualizations_from_csv(
            output_dir / AVERAGED_RESULTS_FILENAME,
            visualization_dir,
        )
        create_timescale_histograms_from_csv(
            output_dir / TIMESCALES_FILENAME,
            visualization_dir,
        )
        create_orbital_param_pdfs_from_csv(
            output_dir / ORBITAL_PARAMS_FILENAME,
            visualization_dir,
        )
        return

    config = TDEDynConfig(
        smbh_masses_msun=arguments.smbh_masses,
        nsc_masses_msun=arguments.nsc_masses,
        num_repeats=arguments.num_repeats,
        duty_cycle_myr=arguments.duty_cycle_myr,
        output_dir=arguments.output_dir,
        metallicity=arguments.metallicity,
        initial_stellar_density_gamma=arguments.initial_stellar_density_gamma,
        dyn_resupply=not arguments.no_resupply,
        max_copies_per_object=arguments.max_copies_per_object,
        reuse_same_orbit_for_copies=(not arguments.resample_copy_orbits),
        tde_radius_rg=arguments.tde_radius_rg,
        direct_tde=arguments.direct_tde,
        consider_type_ii_migration=not arguments.no_type_ii_migration,
        damping_cpu_count=arguments.damping_cpu_count,
        checkpoint_grid_interval=arguments.checkpoint_interval,
        create_visualizations=not arguments.no_visualizations,
        create_intermediate_plots=arguments.intermediate_plots,
        print_debug_results=not arguments.no_debug_results,
    )
    start_time_s = time.time()
    run_simulation(config)
    runtime_s = time.time() - start_time_s
    print(
        f"Total runtime: {runtime_s:.2f} seconds = "
        f"{runtime_s/SECONDS_PER_MINUTE:.2f} minutes"
    )
