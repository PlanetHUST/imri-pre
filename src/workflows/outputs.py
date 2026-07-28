"""Persistence helpers for dynamical IMRI simulations."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .config import (
    IMRI_AVERAGED_RESULTS_FILENAME,
    IMRI_EVENTS_FILENAME,
    METADATA_FILENAME,
    IMRIDynConfig,
)
from .models import IMRIEventRecord, IMRIDynAveragedResult

IMRI_RESULT_COLUMNS = (
    "m_smbh_msun",
    "m_nsc_msun",
    "num_repeats",
    "f_geo_mean",
    "disk_crossing_objects_mean",
    "embedded_bh_count_mean",
    "bh_candidate_count_mean",
    "imri_count_mean",
    "imri_count_min",
    "imri_count_max",
    "imri_mass_mean_msun",
    "imri_rate_mean_per_myr",
    "imri_mass_rate_mean_msun_per_myr",
    "mean_relaxation_time_geo_mean_myr",
)
IMRI_EVENT_COLUMNS = tuple(IMRIEventRecord.__annotations__)


def save_results(
    averaged_results: Sequence[IMRIDynAveragedResult],
    output_dir: Path,
) -> pd.DataFrame:
    """Save averaged IMRI results with units encoded in column names."""
    frame = pd.DataFrame(
        [result.to_dict() for result in averaged_results],
        columns=IMRI_RESULT_COLUMNS,
    )
    path = output_dir / IMRI_AVERAGED_RESULTS_FILENAME
    frame.to_csv(path, index=False, float_format="%.6e")
    print(f"saved: {path}")
    return frame


def save_events(
    averaged_results: Sequence[IMRIDynAveragedResult],
    output_dir: Path,
) -> pd.DataFrame:
    """Save one row per embedded BH evaluated for IMRI delivery."""
    records: list[IMRIEventRecord] = []
    for averaged_result in averaged_results:
        for result in averaged_result.results:
            records.extend(result.imri_events)
    frame = pd.DataFrame.from_records(records, columns=IMRI_EVENT_COLUMNS)
    path = output_dir / IMRI_EVENTS_FILENAME
    frame.to_csv(path, index=False, float_format="%.6e")
    print(f"saved: {path}")
    return frame


def save_metadata(
    output_dir: Path,
    config: IMRIDynConfig,
    runtime_stats: dict[str, float | int | str],
) -> None:
    """Save simulation configuration and runtime statistics."""
    metadata: dict[str, Any] = {
        "simulation_info": {
            "date": datetime.now().isoformat(),
            "timestamp_s_since_epoch": time.time(),
            "description": "Dynamical IMRI rate simulation with parallel damping",
        },
        "parameters": {
            "smbh_masses_msun": config.smbh_masses_msun.tolist(),
            "nsc_masses_msun": config.nsc_masses_msun.tolist(),
            "num_repeats": config.num_repeats,
            "total_simulations": (
                len(config.smbh_masses_msun)
                * len(config.nsc_masses_msun)
                * config.num_repeats
            ),
            "duty_cycle_myr": config.duty_cycle_myr,
            "metallicity": config.metallicity,
            "initial_stellar_density_gamma": config.initial_stellar_density_gamma,
            "nsc_radius_pc_at_reference_mass": (
                config.nsc_radius_pc_at_reference_mass
            ),
            "nsc_radius_reference_mass_msun": (
                config.nsc_radius_reference_mass_msun
            ),
            "nsc_radius_mass_slope": config.nsc_radius_mass_slope,
            "use_disk_inner_radius": config.use_disk_inner_radius,
            "consider_type_ii_migration": config.consider_type_ii_migration,
            "lisa_min_gw_frequency_hz": config.lisa_min_gw_frequency_hz,
            "lisa_max_gw_frequency_hz": config.lisa_max_gw_frequency_hz,
            "max_gw_time_myr": config.max_gw_time_myr,
            "gw_eccentricity": config.gw_eccentricity,
            "damping_cpu_count": config.damping_cpu_count,
            "checkpoint_grid_interval": config.checkpoint_grid_interval,
            "print_debug_results": config.print_debug_results,
        },
        "runtime_statistics": runtime_stats,
    }
    path = output_dir / METADATA_FILENAME
    with path.open("w") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, default=str)
    print(f"saved: {path}")
