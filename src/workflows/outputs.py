"""Persistence helpers for dynamical IMRI simulations."""

import json
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
import pandas as pd

import physical_process.constants as const
from .config import (
    IMRI_AVERAGED_RESULTS_FILENAME,
    IMRI_EVENTS_FILENAME,
    METADATA_FILENAME,
    IMRIDynConfig,
)
from .models import IMRIEventRecord, IMRIDynAveragedResult, IMRIDynSingleResult

IMRI_RESULT_COLUMN_MAP = [
    ("m_smbh_msun", "m_smbh_msun"),
    ("m_nsc_msun", "m_nsc_msun"),
    ("num_repeats", "n_repeat"),
    ("gravitational_radius_cm", "rg_cm"),
    ("disk_outer_radius_mpc", "r_disk_out_mpc"),
    ("disk_outer_radius_rg", "r_disk_out_rg"),
    ("f_geo_mean", "f_geo"),
    ("disk_crossing_objects_mean", "n_disk_cross"),
    ("embedded_star_count_mean", "n_emb_star"),
    ("embedded_bh_count_mean", "n_emb_bh"),
    ("migration_formed_bh_count_mean", "n_mfbh"),
    ("bh_candidate_count_mean", "n_bh_candi"),
    ("imri_candidate_count_mean", "n_imri"),
    ("imri_candidate_count_min", "n_imri_min"),
    ("imri_candidate_count_max", "n_imri_max"),
    ("imri_candidate_mass_mean_msun", "m_imri_msun"),
    ("imri_candidate_rate_mean_per_myr", "rate_imri_per_myr"),
    ("no_detectable_imri_count_mean", "n_nondet"),
    ("detectable_imri_rate_mean_per_myr", "rate_det_per_myr"),
    ("detectable_imri_count_mean", "n_det"),
    ("detectable_imri_count_min", "n_det_min"),
    ("detectable_imri_count_max", "n_det_max"),
    ("torque_migration_evaluated_count_mean", "n_mig_eval"),
    ("finite_torque_migration_count_mean", "n_mig_finite"),
    ("trap_bound_migration_count_mean", "n_mig_trap"),
    ("torque_migration_error_count_mean", "n_mig_error"),
    ("trap_bound_mass_mean_msun", "m_mig_trap_msun"),
]
IMRI_RESULT_COLUMNS = [
    output_name for _source_name, output_name in IMRI_RESULT_COLUMN_MAP
]

IMRI_EVENT_COLUMNS = [
    "m_smbh_msun",
    "m_nsc_msun",
    "repeat",
    "redshift",
    "detection_redshift",
    "object_index",
    "mass_msun",
    "bh_origin",
    "event_rate_class",
    "detect_rate_class",
    "formation_time_myr",
    "damping_time_myr",
    "bh_embedding_time_myr",
    "r_ini_cm",
    "r_ini_rg",
    "r_gas_fin_cm",
    "r_gas_fin_rg",
    "r_entry_cm",
    "r_entry_rg",
    "r_det_entry_cm",
    "r_det_entry_rg",
    "f_ini_src_hz",
    "f_ini_obs_hz",
    "f_gas_fin_src_hz",
    "f_gas_fin_obs_hz",
    "f_entry_src_hz",
    "f_entry_obs_hz",
    "gas_elapsed_time_myr",
    "gas_final_time_myr",
    "migration_time_to_entry_myr",
    "migration_status",
    "blocking_radius_rg",
    "gw_time_to_entry_myr",
    "gw_time_after_gas_myr",
    "detection_time_myr",
    "final_smbh_mass_msun",
    "cosmic_merge_applied",
    "bh_candidate",
    "is_imri_candidate",
    "is_lisa_detectable",
    "snr_proxy",
    "event_rate_reason",
    "detect_rate_reason",
]
EVENT_HIDDEN_COLUMNS = {
    "r_ini_cm",
    "r_gas_fin_cm",
    "r_entry_cm",
    "r_det_entry_cm",
}
EVENT_FREQUENCY_COLUMNS = {
    "f_ini_src_hz": "f_ini_src",
    "f_ini_obs_hz": "f_ini_obs",
    "f_gas_fin_src_hz": "f_gas_fin_src",
    "f_gas_fin_obs_hz": "f_gas_fin_obs",
    "f_entry_src_hz": "f_entry_src",
    "f_entry_obs_hz": "f_entry_obs",
}
FREQUENCY_UNITS_HZ = [
    ("hz", 1.0),
    ("milli_hz", 1.0e-3),
    ("micro_hz", 1.0e-6),
    ("nano_hz", 1.0e-9),
    ("pico_hz", 1.0e-12),
]


def _mean_disk_outer_radius_rg_pc(
    averaged_result: IMRIDynAveragedResult,
) -> tuple[float, float]:

    radii_rg: list[float] = []
    for result in averaged_result.results:
        if result.disk_outer_radius_cm is None:
            continue
        result_rg_cm = result.gravitational_radius_cm
        if result_rg_cm is None:
            continue
        if result_rg_cm > 0.0 and np.isfinite(result.disk_outer_radius_cm):
            radii_rg.append(float(result.disk_outer_radius_cm / result_rg_cm))
    if not radii_rg:
        return np.nan, np.nan
    mean_radius_rg = float(np.mean(radii_rg))
    rg_values = [
        result.gravitational_radius_cm
        for result in averaged_result.results
        if result.gravitational_radius_cm is not None
    ]
    rg_cm = float(np.mean(rg_values)) if rg_values else np.nan
    return mean_radius_rg, float(mean_radius_rg * rg_cm)


def _result_output_record(
    averaged_result: IMRIDynAveragedResult,
) -> dict[str, float | int]:
    record = dict(averaged_result.to_dict())
    rg_values = [
        result.gravitational_radius_cm
        for result in averaged_result.results
        if result.gravitational_radius_cm is not None
    ]
    record["gravitational_radius_cm"] = (
        float(np.mean(rg_values)) if rg_values else np.nan
    )
    rout_rg, rout_cm = _mean_disk_outer_radius_rg_pc(averaged_result)
    record["disk_outer_radius_rg"] = rout_rg
    record["disk_outer_radius_mpc"] = rout_cm / (1e-3 * const.parsec)
    return {
        output_name: record[source_name]
        for source_name, output_name in IMRI_RESULT_COLUMN_MAP
    }


def _choose_frequency_unit(values_hz: pd.Series) -> tuple[str, float]:
    numeric = np.asarray(pd.to_numeric(values_hz, errors="coerce"), dtype=float)
    finite_nonzero = np.abs(numeric[np.isfinite(numeric) & (numeric != 0.0)])
    if finite_nonzero.size == 0:
        return "hz", 1.0

    reference_hz = float(np.median(finite_nonzero))
    for unit_name, scale_hz in FREQUENCY_UNITS_HZ:
        if reference_hz / scale_hz >= 1.0:
            return unit_name, scale_hz
    return FREQUENCY_UNITS_HZ[-1]


def _format_event_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output_frame = pd.DataFrame(index=frame.index)

    for column_name in IMRI_EVENT_COLUMNS:
        if column_name in EVENT_HIDDEN_COLUMNS:
            continue
        source_series = cast(pd.Series, frame[column_name])
        if column_name in EVENT_FREQUENCY_COLUMNS:
            base_name = EVENT_FREQUENCY_COLUMNS[column_name]
            values_hz = pd.Series(
                np.asarray(
                    pd.to_numeric(source_series, errors="coerce"),
                    dtype=float,
                ),
                index=frame.index,
            )
            unit_name, output_scale_hz = _choose_frequency_unit(values_hz)
            output_name = f"{base_name}_{unit_name}"
            output_frame[output_name] = values_hz / output_scale_hz
            continue
        output_frame[column_name] = source_series

    return output_frame


def save_results(
    averaged_results: Sequence[IMRIDynAveragedResult],
    output_dir: Path,
) -> pd.DataFrame:
    """Save averaged IMRI results with units encoded in column names."""
    frame = pd.DataFrame.from_records(
        [_result_output_record(result) for result in averaged_results],
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
    """Save one row per embedded BH or delayed-BH object evaluated for IMRI."""
    records: list[IMRIEventRecord] = []
    for averaged_result in averaged_results:
        for result in averaged_result.results:
            records.extend(result.imri_events)
    raw_frame = pd.DataFrame.from_records(records, columns=IMRI_EVENT_COLUMNS)
    frame = _format_event_frame(raw_frame)
    path = output_dir / IMRI_EVENTS_FILENAME
    frame.to_csv(path, index=False, float_format="%.6e")
    print(f"saved: {path}")
    return frame


def save_migration_diagnostics(
    result: IMRIDynSingleResult,
    output_dir: Path,
    config: IMRIDynConfig,
) -> Path:
    """Save strict per-repeat torque migration diagnostics as JSON."""
    def json_float(value: float | None) -> float | None:
        if value is None or not np.isfinite(value):
            return None
        return float(value)

    intermediate_dir = output_dir / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    smbh_token = f"{result.m_smbh_msun:.6e}".replace("+", "")
    nsc_token = f"{result.m_nsc_msun:.6e}".replace("+", "")
    path = intermediate_dir / (
        f"migration_smbh_{smbh_token}_nsc_{nsc_token}_"
        f"repeat_{result.repeat_idx:04d}.json"
    )

    trap_bound_objects = [
        {
            "object_index": event["object_index"],
            "mass_msun": json_float(event["mass_msun"]),
            "r_ini_rg": json_float(event["r_ini_rg"]),
            "r_entry_rg": json_float(event["r_entry_rg"]),
            "migration_time_to_entry_myr": None,
            "migration_status": event["migration_status"],
            "blocking_radius_rg": json_float(event["blocking_radius_rg"]),
        }
        for event in result.imri_events
        if event["migration_status"] == "trap_bound"
    ]
    if (
        result.torque_migration_evaluated_count
        != result.finite_torque_migration_count
        + result.trap_bound_migration_count
        + result.torque_migration_error_count
    ):
        raise ValueError("torque migration counters do not sum to evaluated count")
    if result.trap_bound_migration_count != len(trap_bound_objects):
        raise ValueError("trap-bound counter does not match event records")
    if result.torque_migration_error_count != len(result.torque_errors):
        raise ValueError("torque error counter does not match error records")

    payload = {
        "m_smbh_msun": json_float(result.m_smbh_msun),
        "m_nsc_msun": json_float(result.m_nsc_msun),
        "repeat": result.repeat_idx,
        "gap_opening_condition": config.gap_opening_condition,
        "consider_type_ii_migration": config.consider_type_ii_migration,
        "torque_migration_evaluated_count": (
            result.torque_migration_evaluated_count
        ),
        "finite_torque_migration_count": result.finite_torque_migration_count,
        "trap_bound_migration_count": result.trap_bound_migration_count,
        "torque_migration_error_count": result.torque_migration_error_count,
        "trap_bound_mass_msun": json_float(result.trap_bound_mass_msun),
        "trap_bound_objects": trap_bound_objects,
        "torque_errors": [
            {
                "object_index": error["object_index"],
                "mass_msun": json_float(error["mass_msun"]),
                "r_ini_rg": json_float(error["r_ini_rg"]),
                "r_entry_rg": json_float(error["r_entry_rg"]),
                "error": error["error"],
            }
            for error in result.torque_errors
        ],
    }
    with path.open("w") as diagnostics_file:
        json.dump(payload, diagnostics_file, indent=2, allow_nan=False)
    print(f"saved: {path}")
    return path


def save_metadata(
    output_dir: Path,
    metadata: dict[str, Any],
) -> None:
    """Persist an assembled metadata payload as JSON."""
    path = output_dir / METADATA_FILENAME
    with path.open("w") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, default=str)
    print(f"saved: {path}")
