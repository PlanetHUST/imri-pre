"""Command-line interface for the dynamical IMRI workflow.

Two execution modes share one parser:

* **single-halo mode** (the original behaviour) - every option describes one
  local simulation grid and ``run_simulation`` is called exactly as before;
* **merger-tree batch mode** - selected with ``--tree``; the same options form
  the *base* configuration, and one single-halo job is executed per selected
  merger-tree host through ``workflows.tree_batch``.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

from lisa.snr import SNR_WINDOW_PLACEMENTS

from .config import (
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
    IMRI_COSMIC_MERGE_RADIUS_FACTOR,
    IMRI_COSMIC_MERGE_SMBH_MASS_FACTOR,
    IMRI_COSMIC_MERGE_TIME_MYR,
    IMRIDynConfig,
    SECONDS_PER_MINUTE,
)
from .host_mapping import (
    DEFAULT_DOMAIN_SMBH_MASS_MAX_MSUN,
    DEFAULT_DOMAIN_SMBH_MASS_MIN_MSUN,
    DEFAULT_NSC_STELLAR_MASS_FRACTION,
    DEFAULT_SMBH_STELLAR_MASS_FRACTION,
    METALLICITY_SOURCES,
    METALLICITY_SOURCE_CONFIG,
    NSC_MASS_SOURCES,
    NSC_MASS_SOURCE_CONFIG,
    SMBH_MASS_SOURCES,
    SMBH_MASS_SOURCE_TREE,
    HostMappingPolicy,
)
from .merger_tree import DEFAULT_CHUNK_SIZE, HostSelection, MergerTreeError
from .tree_batch import (
    BatchExecutionError,
    BatchSettings,
    DEFAULT_MASTER_SEED,
    DEFAULT_WORKERS,
    run_tree_batch,
)
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
        type=_parse_mass_vector,
        default=DEFAULT_SMBH_MASSES_MSUN,
        metavar="M1,M2,...",
        help="comma-separated SMBH mass values in Msun",
    )
    parser.add_argument(
        "--nsc-masses",
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
        help=(
            "legacy recorded GW-time parameter; the active limit is the "
            "cosmological z=0 age"
        ),
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
        help="flat-LambdaCDM H0 shared by cosmic time and luminosity distance",
    )
    parser.add_argument(
        "--detector-omega-m",
        type=float,
        default=IMRI_DETECTOR_OMEGA_M,
        help="flat-LambdaCDM matter density shared by cosmic time and distance",
    )
    parser.add_argument(
        "--cosmic-merge-time-myr",
        type=float,
        default=IMRI_COSMIC_MERGE_TIME_MYR,
        help="source-frame delay after gas evolution for one simplified SMBH merger",
    )
    parser.add_argument(
        "--cosmic-merge-smbh-mass-factor",
        type=float,
        default=IMRI_COSMIC_MERGE_SMBH_MASS_FACTOR,
        help="central-SMBH mass multiplier applied by the simplified merger",
    )
    parser.add_argument(
        "--cosmic-merge-radius-factor",
        type=float,
        default=IMRI_COSMIC_MERGE_RADIUS_FACTOR,
        help="orbital-radius multiplier applied by the simplified merger",
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
        "--base-seed",
        type=int,
        default=None,
        help="optional root seed for reproducible, paired Monte Carlo tasks",
    )
    _add_tree_arguments(parser)
    return parser


def _parse_id_list(value: str) -> tuple[float, ...]:
    """Parse a comma-separated list of node/branch identities."""
    items = [item.strip() for item in value.split(",") if item.strip()]
    try:
        identities = tuple(float(item) for item in items)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "node/branch lists must contain comma-separated numbers"
        ) from error
    if not identities:
        raise argparse.ArgumentTypeError("node/branch lists must not be empty")
    return identities


def _add_tree_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the merger-tree batch options (ignored unless --tree is given)."""
    group = parser.add_argument_group(
        "merger-tree batch mode",
        "Supplying --tree switches to batch mode: every option above becomes the "
        "base configuration and one single-halo job runs per selected host.",
    )
    group.add_argument(
        "--tree",
        type=Path,
        default=None,
        metavar="CSV",
        help="merger-tree CSV; enables batch mode",
    )
    group.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="DIR",
        help="batch output root (overrides the positional output_dir)",
    )
    group.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="number of parallel host jobs; each job additionally uses "
        "damping_cpu_count processes (default 1 in batch mode)",
    )
    group.add_argument(
        "--master-seed",
        type=int,
        default=DEFAULT_MASTER_SEED,
        help="root seed from which every host base seed is derived",
    )
    group.add_argument(
        "--tree-chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="rows per CSV chunk (engineering parameter, not physical)",
    )
    group.add_argument(
        "--node", type=_parse_id_list, default=None, metavar="ID[,ID...]"
    )
    group.add_argument(
        "--branch", type=_parse_id_list, default=None, metavar="ID[,ID...]"
    )
    group.add_argument("--z-min", type=float, default=None)
    group.add_argument("--z-max", type=float, default=None)
    group.add_argument(
        "--mbh-min", type=float, default=None, help="tree MBH lower bound [Msun]"
    )
    group.add_argument(
        "--mbh-max", type=float, default=None, help="tree MBH upper bound [Msun]"
    )
    group.add_argument(
        "--mhalo-min", type=float, default=None, help="tree m lower bound [Msun]"
    )
    group.add_argument(
        "--mhalo-max", type=float, default=None, help="tree m upper bound [Msun]"
    )
    group.add_argument(
        "--max-hosts",
        type=int,
        default=None,
        help="stop after this many selected hosts (file order)",
    )
    group.add_argument(
        "--smbh-mass-source",
        choices=SMBH_MASS_SOURCES,
        default=SMBH_MASS_SOURCE_TREE,
        help="where each host's SMBH mass comes from: the tree MBH column, the "
        "single --smbh-masses value, or an explicit fraction of the tree Mstar",
    )
    group.add_argument(
        "--nsc-mass-source",
        choices=NSC_MASS_SOURCES,
        default=NSC_MASS_SOURCE_CONFIG,
        help="where each host's NSC mass comes from: the single --nsc-masses "
        "value or an explicit fraction of the tree Mstar",
    )
    group.add_argument(
        "--metallicity-source",
        choices=METALLICITY_SOURCES,
        default=METALLICITY_SOURCE_CONFIG,
        help="where each host's metallicity comes from: --metallicity or the "
        "tree Z mass fraction converted to solar units",
    )
    group.add_argument(
        "--smbh-stellar-mass-fraction",
        type=float,
        default=DEFAULT_SMBH_STELLAR_MASS_FRACTION,
        help="M_smbh = fraction * Mstar when --smbh-mass-source stellar-scaling",
    )
    group.add_argument(
        "--nsc-stellar-mass-fraction",
        type=float,
        default=DEFAULT_NSC_STELLAR_MASS_FRACTION,
        help="M_nsc = fraction * Mstar when --nsc-mass-source stellar-scaling",
    )
    group.add_argument(
        "--domain-smbh-min-msun",
        type=float,
        default=DEFAULT_DOMAIN_SMBH_MASS_MIN_MSUN,
        help="local-model SMBH mass lower bound; hosts below are not run",
    )
    group.add_argument(
        "--domain-smbh-max-msun",
        type=float,
        default=DEFAULT_DOMAIN_SMBH_MASS_MAX_MSUN,
        help="local-model SMBH mass upper bound; hosts above are not run",
    )
    group.add_argument(
        "--no-batch-figures",
        action="store_true",
        help="skip the two batch interface figures",
    )


TREE_ONLY_OPTIONS = (
    "output",
    "node",
    "branch",
    "z_min",
    "z_max",
    "mbh_min",
    "mbh_max",
    "mhalo_min",
    "mhalo_max",
    "max_hosts",
)


def build_config_from_arguments(arguments: argparse.Namespace) -> IMRIDynConfig:
    """Build the (base) configuration from parsed CLI arguments."""
    return IMRIDynConfig(
        smbh_masses_msun=arguments.smbh_masses,
        nsc_masses_msun=arguments.nsc_masses,
        num_repeats=arguments.num_repeats,
        duty_cycle_myr=arguments.duty_cycle_myr,
        output_dir=arguments.output_dir,
        metallicity=arguments.metallicity,
        initial_stellar_density_gamma=arguments.initial_stellar_density_gamma,
        use_disk_inner_radius=not arguments.ignore_disk_inner_radius,
        lisa_min_gw_frequency_hz=arguments.lisa_min_gw_frequency_hz,
        lisa_max_gw_frequency_hz=arguments.lisa_max_gw_frequency_hz,
        max_gw_time_myr=arguments.max_gw_time_myr,
        gw_eccentricity=arguments.gw_eccentricity,
        redshift=arguments.redshift,
        snr_threshold=arguments.snr_threshold,
        lisa_mission_duration_years=arguments.lisa_mission_duration_years,
        snr_window_placement=arguments.snr_window_placement,
        detector_hubble_constant_km_s_mpc=(arguments.detector_hubble_constant_km_s_mpc),
        detector_omega_m=arguments.detector_omega_m,
        cosmic_merge_time_myr=arguments.cosmic_merge_time_myr,
        cosmic_merge_smbh_mass_factor=arguments.cosmic_merge_smbh_mass_factor,
        cosmic_merge_radius_factor=arguments.cosmic_merge_radius_factor,
        damping_cpu_count=arguments.damping_cpu_count,
        checkpoint_grid_interval=arguments.checkpoint_interval,
        print_debug_results=arguments.debug_results,
        base_seed=arguments.base_seed,
    )


def build_host_selection(arguments: argparse.Namespace) -> HostSelection:
    """Translate CLI selectors into a declarative host filter."""
    if arguments.max_hosts is not None and arguments.max_hosts <= 0:
        raise argparse.ArgumentTypeError("--max-hosts must be positive")
    return HostSelection(
        nodes=arguments.node,
        branches=arguments.branch,
        redshift_min=arguments.z_min,
        redshift_max=arguments.z_max,
        smbh_mass_min_msun=arguments.mbh_min,
        smbh_mass_max_msun=arguments.mbh_max,
        halo_mass_min_msun=arguments.mhalo_min,
        halo_mass_max_msun=arguments.mhalo_max,
        max_hosts=arguments.max_hosts,
    )


def build_host_mapping_policy(arguments: argparse.Namespace) -> HostMappingPolicy:
    """Translate CLI mapping options into an explicit mapping policy."""
    return HostMappingPolicy(
        smbh_mass_source=arguments.smbh_mass_source,
        nsc_mass_source=arguments.nsc_mass_source,
        metallicity_source=arguments.metallicity_source,
        smbh_stellar_mass_fraction=arguments.smbh_stellar_mass_fraction,
        nsc_stellar_mass_fraction=arguments.nsc_stellar_mass_fraction,
        domain_smbh_mass_min_msun=arguments.domain_smbh_min_msun,
        domain_smbh_mass_max_msun=arguments.domain_smbh_max_msun,
    )


def _run_tree_mode(
    parser: argparse.ArgumentParser, arguments: argparse.Namespace
) -> None:
    if arguments.workers < 1:
        parser.error("--workers must be a positive integer")
    if arguments.master_seed < 0:
        parser.error("--master-seed must be nonnegative")
    if arguments.tree_chunk_size < 1:
        parser.error("--tree-chunk-size must be a positive integer")
    if not arguments.tree.is_file():
        parser.error(f"--tree file not found: {arguments.tree}")

    base_config = build_config_from_arguments(arguments)
    try:
        selection = build_host_selection(arguments)
        policy = build_host_mapping_policy(arguments)
    except (argparse.ArgumentTypeError, ValueError) as error:
        parser.error(str(error))
    output_root = (
        arguments.output if arguments.output is not None else arguments.output_dir
    )
    settings = BatchSettings(
        workers=arguments.workers,
        chunk_size=arguments.tree_chunk_size,
        master_seed=arguments.master_seed,
        make_figures=not arguments.no_batch_figures,
    )
    start_time_s = time.time()
    try:
        outcome = run_tree_batch(
            arguments.tree,
            base_config,
            output_root,
            selection=selection,
            policy=policy,
            settings=settings,
            config_source="command line: " + " ".join(sys.argv),
        )
    except (MergerTreeError, BatchExecutionError) as error:
        parser.error(str(error))
    runtime_s = time.time() - start_time_s
    print(
        f"Total runtime: {runtime_s:.2f} seconds = "
        f"{runtime_s/SECONDS_PER_MINUTE:.2f} minutes; "
        f"status counts: {outcome.status_counts}"
    )


def run_cli(argv: Sequence[str] | None = None) -> None:
    """Parse command-line arguments and run the IMRI simulation."""
    parser = build_parser()
    arguments = parser.parse_args(argv)

    if arguments.tree is not None:
        _run_tree_mode(parser, arguments)
        return

    used_tree_options = [
        f"--{name.replace('_', '-')}"
        for name in TREE_ONLY_OPTIONS
        if getattr(arguments, name) is not None
    ]
    if used_tree_options:
        parser.error(
            "these options require --tree (merger-tree batch mode): "
            + ", ".join(used_tree_options)
        )

    config = build_config_from_arguments(arguments)
    start_time_s = time.time()
    run_simulation(config)
    runtime_s = time.time() - start_time_s
    print(
        f"Total runtime: {runtime_s:.2f} seconds = "
        f"{runtime_s/SECONDS_PER_MINUTE:.2f} minutes"
    )
