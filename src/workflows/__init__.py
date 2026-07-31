"""Public API for the dynamical IMRI-rate simulation."""

from .cli import build_parser, run_cli
from .config import (
    DISK_ACCRETION_RATE_G_S_PER_MSUN,
    DISK_INNER_RADIUS_RG,
    IMRI_AVERAGED_RESULTS_FILENAME,
    IMRI_EVENTS_FILENAME,
    IMRI_GW_ECCENTRICITY,
    IMRI_HUBBLE_CONSTANT_PER_MYR,
    IMRI_LISA_MAX_GW_FREQUENCY_HZ,
    IMRI_LISA_MIN_GW_FREQUENCY_HZ,
    IMRI_MAX_GW_TIME_MYR,
    IMRI_REDSHIFT,
    METADATA_FILENAME,
    SECONDS_PER_MINUTE,
    YEARS_PER_MYR,
    IMRIDynConfig,
)
from .engine import IMRIDynEngine
from .models import (
    BoolArray,
    DampingResult,
    DampingTask,
    DiskProfiles,
    FloatArray,
    GeometricPopulation,
    IMRIEventRecord,
    IMRIDynAveragedResult,
    IMRIDynSingleResult,
    IntermediatePopulation,
    StellarPopulation,
)
from .outputs import save_events, save_metadata, save_results
from .runtime import set_verbose, vprint
from .single_work import IMRISingleWorkflow
from .workers import (
    get_hpc_cpu_count,
    init_damping_worker,
    process_damping_task,
    run_damping_tasks_parallel,
)
from .workflow import debug_print_result, main, print_header, run_simulation

__all__ = [
    "BoolArray",
    "DISK_ACCRETION_RATE_G_S_PER_MSUN",
    "DISK_INNER_RADIUS_RG",
    "DampingResult",
    "DampingTask",
    "DiskProfiles",
    "FloatArray",
    "GeometricPopulation",
    "IMRI_AVERAGED_RESULTS_FILENAME",
    "IMRI_EVENTS_FILENAME",
    "IMRI_GW_ECCENTRICITY",
    "IMRI_HUBBLE_CONSTANT_PER_MYR",
    "IMRI_LISA_MAX_GW_FREQUENCY_HZ",
    "IMRI_LISA_MIN_GW_FREQUENCY_HZ",
    "IMRI_MAX_GW_TIME_MYR",
    "IMRI_REDSHIFT",
    "IMRIEventRecord",
    "IMRIDynAveragedResult",
    "IMRIDynConfig",
    "IMRIDynEngine",
    "IMRIDynSingleResult",
    "IMRISingleWorkflow",
    "IntermediatePopulation",
    "METADATA_FILENAME",
    "SECONDS_PER_MINUTE",
    "StellarPopulation",
    "YEARS_PER_MYR",
    "build_parser",
    "debug_print_result",
    "get_hpc_cpu_count",
    "init_damping_worker",
    "main",
    "print_header",
    "process_damping_task",
    "run_cli",
    "run_damping_tasks_parallel",
    "run_simulation",
    "save_events",
    "save_metadata",
    "save_results",
    "set_verbose",
    "vprint",
]
