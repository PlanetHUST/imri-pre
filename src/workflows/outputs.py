"""Persistence helpers for dynamical IMRI simulations."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
import pandas as pd

import physical_process.constants as const
from .config import (
    IMRI_AVERAGED_RESULTS_FILENAME,
    IMRI_EVENTS_FILENAME,
    IMRI_REPEAT_RESULTS_FILENAME,
    METADATA_FILENAME,
    SEMI_WITHOUT_TRAP,
    IMRIDynConfig,
)
from .models import IMRIEventRecord, IMRIDynAveragedResult, IMRIDynSingleResult
from .lisa_snr import (
    SNR_STATUS_EVALUATED,
    SNR_STATUS_MODEL_UNAVAILABLE,
    SNR_STATUS_NO_OVERLAP,
    SNR_STATUS_NUMERICAL_FAILURE,
    SNR_STATUS_REQUIRED_INPUT_UNAVAILABLE,
)

IMRI_RESULT_COLUMN_MAP = [
    ("m_smbh_msun", "m_smbh_msun"),
    ("m_nsc_msun", "m_nsc_msun"),
    ("disk_model", "disk_model"),
    ("num_repeats", "n_repeat"),
    ("rate_evaluation_status", "rate_status"),
    ("detector_rate_evaluation_status", "detector_rate_status"),
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
    ("detector_evaluated_imri_count_mean", "n_detector_eval"),
    ("detector_unavailable_imri_count_mean", "n_detector_unavailable"),
    ("detector_below_threshold_imri_count_mean", "n_snr_below"),
    ("detector_no_overlap_imri_count_mean", "n_no_overlap"),
    (
        "detector_required_input_unavailable_imri_count_mean",
        "n_detector_input_unavailable",
    ),
    ("detector_model_unavailable_imri_count_mean", "n_detector_model_unavailable"),
    ("detector_numerical_failure_imri_count_mean", "n_detector_numerical_failure"),
    ("band_entry_evaluated_count_mean", "n_band_eval"),
    ("entry_in_migration_domain_count_mean", "n_entry_in_domain"),
    ("entry_handoff_count_mean", "n_entry_handoff"),
    ("band_entry_inside_isco_count_mean", "n_entry_in_ISCO"),
    ("profile_error_count_mean", "n_profile_error"),
    ("migration_evaluated_count_mean", "n_mig_eval"),
    ("finite_migration_count_mean", "n_mig_finite"),
    ("trap_bound_migration_count_mean", "n_mig_trap"),
    ("migration_error_count_mean", "n_mig_error"),
    ("trap_bound_mass_mean_msun", "m_mig_trap_msun"),
]
IMRI_RESULT_COLUMNS = [
    output_name for _source_name, output_name in IMRI_RESULT_COLUMN_MAP
]

IMRI_EVENT_COLUMNS = [
    "event_id",
    "m_smbh_msun",
    "m_nsc_msun",
    "repeat",
    "disk_model",
    "redshift",
    "detection_redshift",
    "object_index",
    "mass_msun",
    "mass_ratio_secondary_to_primary",
    "bh_origin",
    "band_entry_status",
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
    "r_entry_formal_cm",
    "r_entry_formal_rg",
    "r_isco_cm",
    "r_isco_rg",
    "r_disk_inner_cm",
    "r_disk_inner_rg",
    "r_migration_min_cm",
    "r_migration_min_rg",
    "r_handoff_cm",
    "r_handoff_rg",
    "source_entry_frequency_hz",
    "r_det_entry_cm",
    "r_det_entry_rg",
    # "f_ini_src_hz",
    # "f_ini_obs_hz",
    # "f_gas_fin_src_hz",
    # "f_gas_fin_obs_hz",
    "f_entry_src_hz",
    "f_entry_obs_hz",
    "snr_input_observed_frequency_hz",
    "gas_elapsed_time_myr",
    "gas_final_time_myr",
    "migration_time_to_entry_myr",
    "migration_time_to_handoff_myr",
    "gw_time_handoff_to_entry_myr",
    "gas_stop_reason",
    "migration_evaluated",
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
    "lisa_snr",
    "snr_status",
    "chirp_mass_source_msun",
    "chirp_mass_detector_msun",
    "luminosity_distance_mpc",
    "snr_f_low_hz",
    "snr_f_high_hz",
    "f_isco_observer_hz",
    "mission_duration_years",
    "snr_observation_duration_years",
    "snr_observation_case",
    "snr_threshold",
    "monte_carlo_event_weight",
    "intrinsic_rate_contribution_per_myr",
    "detected_rate_contribution_per_myr",
    "snr_proxy",
    "event_rate_reason",
    "detect_rate_reason",
]

IMRI_REPEAT_RESULT_COLUMNS = [
    "redshift",
    "disk_model",
    "m_smbh_msun",
    "m_nsc_msun",
    "repeat",
    "base_seed",
    "task_seed",
    "rate_status",
    "detector_rate_status",
    "duty_cycle_myr",
    "n_sampled",
    "n_disk_cross",
    "f_geo",
    "n_emb_star",
    "n_emb_bh",
    "n_mfbh",
    "n_outside_or_failed",
    "n_bh_candi",
    "n_no_imri",
    "n_imri",
    "rate_imri_per_myr",
    "n_det",
    "n_nondet",
    "n_detector_eval",
    "n_detector_unavailable",
    "n_snr_below",
    "n_no_overlap",
    "n_detector_input_unavailable",
    "n_detector_model_unavailable",
    "n_detector_numerical_failure",
    "rate_det_per_myr",
    "n_band_eval",
    "n_entry_in_domain",
    "n_entry_handoff",
    "n_entry_inside_isco",
    "n_profile_error",
    "n_unavailable",
    "n_mig_eval",
    "n_mig_finite",
    "n_mig_trap",
    "n_mig_error",
]

EVENT_HIDDEN_COLUMNS = {
    "r_ini_cm",
    "r_gas_fin_cm",
    "r_entry_cm",
    "r_entry_formal_cm",
    "r_isco_cm",
    "r_disk_inner_cm",
    "r_migration_min_cm",
    "r_handoff_cm",
}

EVENT_FREQUENCY_COLUMNS = {
    "f_ini_src_hz": "f_ini_src",
    "f_ini_obs_hz": "f_ini_obs",
    "f_gas_fin_src_hz": "f_gas_fin_src",
    "f_gas_fin_obs_hz": "f_gas_fin_obs",
    "f_entry_src_hz": "f_entry_src",
    "f_entry_obs_hz": "f_entry_obs",
    "snr_input_observed_frequency_hz": "snr_input_observed_frequency",
    "snr_f_low_hz": "snr_f_low",
    "snr_f_high_hz": "snr_f_high",
    "f_isco_observer_hz": "f_isco_obs",
}

FREQUENCY_UNITS_HZ = [
    ("hz", 1.0),
    ("milli_hz", 1.0e-3),
    ("micro_hz", 1.0e-6),
    ("nano_hz", 1.0e-9),
    ("pico_hz", 1.0e-12),
]


@dataclass(frozen=True)
class _PhysicalEntrySummary:
    """Output-layer counts for systems that reach the physical LISA entry."""

    count: int
    mass_msun: float
    detectable_count: int
    detectable_mass_msun: float
    nondetectable_count: int
    detector_evaluated_count: int
    detector_unavailable_count: int
    below_threshold_count: int
    no_overlap_count: int
    required_input_unavailable_count: int
    model_unavailable_count: int
    numerical_failure_count: int


def _boolean_state(value: Any) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return None


def _physical_entry_summary(result: IMRIDynSingleResult) -> _PhysicalEntrySummary:
    entries = [
        event
        for event in result.imri_events
        if _boolean_state(event["is_imri_candidate"]) is True
        and np.isfinite(float(event["detection_time_myr"]))
    ]
    detectable_count = 0
    detectable_mass_msun = 0.0
    nondetectable_count = 0
    detector_unavailable_count = 0
    below_threshold_count = 0
    no_overlap_count = 0
    required_input_unavailable_count = 0
    model_unavailable_count = 0
    numerical_failure_count = 0
    for event in entries:
        detector_state = _boolean_state(event["is_lisa_detectable"])
        snr_status = str(event["snr_status"])
        if detector_state is True:
            if snr_status != SNR_STATUS_EVALUATED:
                raise ValueError("detectable physical entry has unavailable SNR")
            detectable_count += 1
            detectable_mass_msun += float(event["mass_msun"])
        elif detector_state is False:
            nondetectable_count += 1
            if snr_status == SNR_STATUS_EVALUATED:
                below_threshold_count += 1
            elif snr_status == SNR_STATUS_NO_OVERLAP:
                no_overlap_count += 1
            else:
                raise ValueError("physical nondetection has unavailable SNR")
        else:
            detector_unavailable_count += 1
            if snr_status == SNR_STATUS_REQUIRED_INPUT_UNAVAILABLE:
                required_input_unavailable_count += 1
            elif snr_status == SNR_STATUS_MODEL_UNAVAILABLE:
                model_unavailable_count += 1
            elif snr_status == SNR_STATUS_NUMERICAL_FAILURE:
                numerical_failure_count += 1
            else:
                raise ValueError(
                    "detector-unavailable physical entry has unknown SNR status"
                )

    detector_evaluated_count = detectable_count + nondetectable_count
    if len(entries) != detector_evaluated_count + detector_unavailable_count:
        raise ValueError("physical entries do not conserve detector outcomes")
    if nondetectable_count != below_threshold_count + no_overlap_count:
        raise ValueError("physical nondetections do not conserve SNR outcomes")
    if detector_unavailable_count != (
        required_input_unavailable_count
        + model_unavailable_count
        + numerical_failure_count
    ):
        raise ValueError("detector-unavailable entries do not conserve outcomes")
    return _PhysicalEntrySummary(
        count=len(entries),
        mass_msun=float(sum(float(event["mass_msun"]) for event in entries)),
        detectable_count=detectable_count,
        detectable_mass_msun=detectable_mass_msun,
        nondetectable_count=nondetectable_count,
        detector_evaluated_count=detector_evaluated_count,
        detector_unavailable_count=detector_unavailable_count,
        below_threshold_count=below_threshold_count,
        no_overlap_count=no_overlap_count,
        required_input_unavailable_count=required_input_unavailable_count,
        model_unavailable_count=model_unavailable_count,
        numerical_failure_count=numerical_failure_count,
    )


def _physical_rate_from_core_denominator(
    result: IMRIDynSingleResult,
    count: int,
) -> float:
    """Recover the configured duty denominator without changing the core model."""
    if _rate_evaluation_status(result) != "complete":
        return np.nan
    if count == 0:
        return 0.0
    if result.imri_count <= 0 or not np.isfinite(result.imri_rate_per_myr):
        raise ValueError("cannot recover duty-cycle denominator for physical entries")
    duty_cycle_myr = result.imri_count / result.imri_rate_per_myr
    if not np.isfinite(duty_cycle_myr) or duty_cycle_myr <= 0.0:
        raise ValueError("recovered duty-cycle denominator is invalid")
    return float(count / duty_cycle_myr)


def _core_duty_cycle_myr(result: IMRIDynSingleResult) -> float | None:
    if result.imri_count == 0:
        return None
    if not np.isfinite(result.imri_rate_per_myr) or result.imri_rate_per_myr <= 0.0:
        return None
    duty_cycle_myr = result.imri_count / result.imri_rate_per_myr
    return float(duty_cycle_myr) if duty_cycle_myr > 0.0 else None


def _detector_status_for_summary(
    result: IMRIDynSingleResult,
    summary: _PhysicalEntrySummary,
) -> str:
    if _rate_evaluation_status(result) != "complete":
        return "not_evaluated_intrinsic_incomplete"
    if summary.detector_unavailable_count:
        return "incomplete_detector_unavailable"
    return "complete"


def _serialized_event_record(
    result: IMRIDynSingleResult,
    event: IMRIEventRecord,
    physical_summary: _PhysicalEntrySummary,
) -> dict[str, Any]:
    """Return an event copy with contributions aligned to physical entry."""
    record = dict(event)
    if _rate_evaluation_status(result) != "complete":
        record["intrinsic_rate_contribution_per_myr"] = np.nan
        record["detected_rate_contribution_per_myr"] = np.nan
        return record
    candidate = _boolean_state(event["is_imri_candidate"]) is True
    physical_entry = candidate and np.isfinite(float(event["detection_time_myr"]))
    duty_cycle_myr = _core_duty_cycle_myr(result)
    if physical_entry and duty_cycle_myr is None:
        raise ValueError("physical event is missing its AGN duty-cycle denominator")
    record["intrinsic_rate_contribution_per_myr"] = (
        1.0 / duty_cycle_myr if physical_entry and duty_cycle_myr else 0.0
    )

    detector_status = _detector_status_for_summary(result, physical_summary)
    detector_state = _boolean_state(event["is_lisa_detectable"])
    if detector_status != "complete" and physical_entry:
        record["detected_rate_contribution_per_myr"] = np.nan
    else:
        record["detected_rate_contribution_per_myr"] = (
            1.0
            / duty_cycle_myr
            if physical_entry and detector_state is True and duty_cycle_myr
            else 0.0
        )
    return record


def _validate_serialized_contributions(
    result: IMRIDynSingleResult,
    records: Sequence[dict[str, Any]],
    physical_summary: _PhysicalEntrySummary,
) -> None:
    intrinsic_total = float(
        sum(
            float(record["intrinsic_rate_contribution_per_myr"])
            for record in records
            if np.isfinite(float(record["intrinsic_rate_contribution_per_myr"]))
        )
    )
    detected_total = float(
        sum(
            float(record["detected_rate_contribution_per_myr"])
            for record in records
            if np.isfinite(float(record["detected_rate_contribution_per_myr"]))
        )
    )
    if detected_total > intrinsic_total + 1.0e-12:
        raise ValueError("detected contributions exceed intrinsic contributions")
    if _rate_evaluation_status(result) == "complete":
        expected_intrinsic = _physical_rate_from_core_denominator(
            result, physical_summary.count
        )
        if not np.isclose(
            intrinsic_total, expected_intrinsic, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError("physical-entry contributions disagree with repeat rate")
    if _detector_status_for_summary(result, physical_summary) == "complete":
        expected_detected = _physical_rate_from_core_denominator(
            result, physical_summary.detectable_count
        )
        if not np.isclose(
            detected_total, expected_detected, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError("detected contributions disagree with detector rate")


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
) -> dict[str, float | int | str]:
    record = dict(averaged_result.to_dict())
    summaries = [_physical_entry_summary(result) for result in averaged_result.results]
    physical_rate_statuses = [
        _rate_evaluation_status(result) for result in averaged_result.results
    ]
    rate_complete = all(status == "complete" for status in physical_rate_statuses)
    if (
        record["rate_evaluation_status"] == "complete"
        and not rate_complete
    ):
        record["rate_evaluation_status"] = "incomplete_physical_entry_unavailable"
    detector_complete = rate_complete and all(
        summary.detector_unavailable_count == 0 for summary in summaries
    )

    # The model's legacy ``imri_candidate_*`` keys count eligibility for
    # post-gas propagation. Override those mapped values at the persistence
    # boundary so the established n_imri columns mean finite physical entry.
    if rate_complete:
        physical_counts = np.asarray([summary.count for summary in summaries])
        physical_masses = np.asarray([summary.mass_msun for summary in summaries])
        physical_rates = np.asarray(
            [
                _physical_rate_from_core_denominator(result, summary.count)
                for result, summary in zip(
                    averaged_result.results, summaries, strict=True
                )
            ]
        )
        record["imri_candidate_count_mean"] = float(np.mean(physical_counts))
        record["imri_candidate_count_min"] = float(np.min(physical_counts))
        record["imri_candidate_count_max"] = float(np.max(physical_counts))
        record["imri_candidate_mass_mean_msun"] = float(np.mean(physical_masses))
        record["imri_candidate_rate_mean_per_myr"] = float(
            np.mean(physical_rates)
        )
    else:
        for name in (
            "imri_candidate_count_mean",
            "imri_candidate_count_min",
            "imri_candidate_count_max",
            "imri_candidate_mass_mean_msun",
            "imri_candidate_rate_mean_per_myr",
        ):
            record[name] = np.nan

    record["detector_rate_evaluation_status"] = (
        "complete"
        if detector_complete
        else "not_evaluated_intrinsic_incomplete"
        if not rate_complete
        else "incomplete_detector_unavailable"
    )
    record["detector_evaluated_imri_count_mean"] = float(
        np.mean([summary.detector_evaluated_count for summary in summaries])
    )
    record["detector_unavailable_imri_count_mean"] = float(
        np.mean([summary.detector_unavailable_count for summary in summaries])
    )
    record["detector_below_threshold_imri_count_mean"] = float(
        np.mean([summary.below_threshold_count for summary in summaries])
    )
    record["detector_no_overlap_imri_count_mean"] = float(
        np.mean([summary.no_overlap_count for summary in summaries])
    )
    record["detector_required_input_unavailable_imri_count_mean"] = float(
        np.mean([summary.required_input_unavailable_count for summary in summaries])
    )
    record["detector_model_unavailable_imri_count_mean"] = float(
        np.mean([summary.model_unavailable_count for summary in summaries])
    )
    record["detector_numerical_failure_imri_count_mean"] = float(
        np.mean([summary.numerical_failure_count for summary in summaries])
    )
    if detector_complete:
        detectable_counts = np.asarray(
            [summary.detectable_count for summary in summaries]
        )
        detectable_rates = np.asarray(
            [
                _physical_rate_from_core_denominator(
                    result, summary.detectable_count
                )
                for result, summary in zip(
                    averaged_result.results, summaries, strict=True
                )
            ]
        )
        record["detectable_imri_count_mean"] = float(
            np.mean(detectable_counts)
        )
        record["detectable_imri_count_min"] = float(np.min(detectable_counts))
        record["detectable_imri_count_max"] = float(np.max(detectable_counts))
        record["detectable_imri_mass_mean_msun"] = float(
            np.mean([summary.detectable_mass_msun for summary in summaries])
        )
        record["detectable_imri_rate_mean_per_myr"] = float(
            np.mean(detectable_rates)
        )
        record["no_detectable_imri_count_mean"] = float(
            np.mean([summary.nondetectable_count for summary in summaries])
        )
    else:
        for name in (
            "detectable_imri_count_mean",
            "detectable_imri_count_min",
            "detectable_imri_count_max",
            "detectable_imri_mass_mean_msun",
            "detectable_imri_rate_mean_per_myr",
            "no_detectable_imri_count_mean",
        ):
            record[name] = np.nan

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


def _validate_event_output(frame: pd.DataFrame) -> None:
    """Validate physical-entry fields and per-event contribution ordering."""
    candidate = np.asarray(
        cast(pd.Series, frame["is_imri_candidate"]).map(_boolean_state).eq(True),
        dtype=bool,
    )
    detection_time = np.asarray(
        pd.to_numeric(
            cast(pd.Series, frame["detection_time_myr"]), errors="coerce"
        ),
        dtype=float,
    )
    physical_entry = candidate & np.isfinite(detection_time)
    for column in (
        "detection_redshift",
        "r_det_entry_cm",
        "r_det_entry_rg",
        "f_entry_src_hz",
        "f_entry_obs_hz",
    ):
        values = np.asarray(
            pd.to_numeric(
                cast(pd.Series, frame.loc[physical_entry, column]), errors="coerce"
            ),
            dtype=float,
        )
        if not np.isfinite(values).all():
            raise ValueError(f"physical entries require finite {column}")

    intrinsic = np.asarray(
        pd.to_numeric(
            cast(pd.Series, frame["intrinsic_rate_contribution_per_myr"]),
            errors="coerce",
        ),
        dtype=float,
    )
    detected = np.asarray(
        pd.to_numeric(
            cast(pd.Series, frame["detected_rate_contribution_per_myr"]),
            errors="coerce",
        ),
        dtype=float,
    )
    comparable = np.isfinite(intrinsic) & np.isfinite(detected)
    if np.any(detected[comparable] > intrinsic[comparable] + 1.0e-12):
        raise ValueError("detected event contribution exceeds intrinsic contribution")


def _validate_rate_output(frame: pd.DataFrame) -> None:
    """Validate physical/detector counts without converting unavailable rates."""
    intrinsic = np.asarray(
        pd.to_numeric(cast(pd.Series, frame["n_imri"]), errors="coerce"),
        dtype=float,
    )
    detected = np.asarray(
        pd.to_numeric(cast(pd.Series, frame["n_det"]), errors="coerce"),
        dtype=float,
    )
    comparable = np.isfinite(intrinsic) & np.isfinite(detected)
    if np.any(detected[comparable] > intrinsic[comparable]):
        raise ValueError("detectable IMRI count exceeds physical-entry count")

    intrinsic_rate = np.asarray(
        pd.to_numeric(
            cast(pd.Series, frame["rate_imri_per_myr"]), errors="coerce"
        ),
        dtype=float,
    )
    detector_rate = np.asarray(
        pd.to_numeric(cast(pd.Series, frame["rate_det_per_myr"]), errors="coerce"),
        dtype=float,
    )
    comparable_rates = np.isfinite(intrinsic_rate) & np.isfinite(detector_rate)
    if np.any(
        detector_rate[comparable_rates]
        > intrinsic_rate[comparable_rates] + 1.0e-12
    ):
        raise ValueError("detectable IMRI rate exceeds physical-entry rate")


def save_results(
    averaged_results: Sequence[IMRIDynAveragedResult],
    output_dir: Path,
) -> pd.DataFrame:
    """Save averaged IMRI results with units encoded in column names."""
    frame = pd.DataFrame.from_records(
        [_result_output_record(result) for result in averaged_results],
        columns=IMRI_RESULT_COLUMNS,
    )
    _validate_rate_output(frame)
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
            physical_summary = _physical_entry_summary(result)
            result_records = [
                cast(
                    IMRIEventRecord,
                    _serialized_event_record(result, event, physical_summary),
                )
                for event in result.imri_events
            ]
            _validate_serialized_contributions(
                result,
                cast(Sequence[dict[str, Any]], result_records),
                physical_summary,
            )
            records.extend(result_records)
    raw_frame = pd.DataFrame.from_records(records, columns=IMRI_EVENT_COLUMNS)
    _validate_event_output(raw_frame)
    frame = _format_event_frame(raw_frame)
    path = output_dir / IMRI_EVENTS_FILENAME
    frame.to_csv(path, index=False, float_format="%.6e")
    print(f"saved: {path}")
    return frame


def _rate_evaluation_status(result: IMRIDynSingleResult) -> str:
    if result.band_entry_inside_isco_count and result.profile_error_count:
        return "not_evaluated_band_entry_and_profile_error"
    if result.band_entry_inside_isco_count:
        return "not_evaluated_band_entry_inside_isco"
    if result.profile_error_count:
        return "incomplete_profile_error"
    physical_entry_unavailable = any(
        _boolean_state(event["is_imri_candidate"]) is True
        and not np.isfinite(float(event["detection_time_myr"]))
        and _boolean_state(event["is_lisa_detectable"]) is None
        for event in result.imri_events
    )
    if physical_entry_unavailable:
        return "incomplete_physical_entry_unavailable"
    return "complete"


def _detector_rate_evaluation_status(result: IMRIDynSingleResult) -> str:
    return _detector_status_for_summary(result, _physical_entry_summary(result))


def _repeat_output_record(
    result: IMRIDynSingleResult,
    config: IMRIDynConfig,
) -> dict[str, float | int | str | None]:
    physical = _physical_entry_summary(result)
    rate_status = _rate_evaluation_status(result)
    detector_rate_status = _detector_status_for_summary(result, physical)
    intrinsic_rate = (
        physical.count / config.duty_cycle_myr
        if rate_status == "complete"
        else np.nan
    )
    detector_rate = (
        physical.detectable_count / config.duty_cycle_myr
        if detector_rate_status == "complete"
        else np.nan
    )
    return {
        "redshift": config.redshift,
        "disk_model": result.disk_model,
        "m_smbh_msun": result.m_smbh_msun,
        "m_nsc_msun": result.m_nsc_msun,
        "repeat": result.repeat_idx,
        "base_seed": config.base_seed,
        "task_seed": result.task_seed,
        "rate_status": rate_status,
        "detector_rate_status": detector_rate_status,
        "duty_cycle_myr": config.duty_cycle_myr,
        "n_sampled": result.total_sampled_objects,
        "n_disk_cross": result.disk_crossing_objects,
        "f_geo": result.f_geo,
        "n_emb_star": result.embedded_star_count,
        "n_emb_bh": result.embedded_bh_count,
        "n_mfbh": result.migration_formed_bh_count,
        "n_outside_or_failed": result.outside_or_failed_count,
        "n_bh_candi": result.bh_candidate_count,
        "n_no_imri": result.no_imri_count + result.imri_count - physical.count,
        "n_imri": physical.count,
        "rate_imri_per_myr": intrinsic_rate,
        "n_det": physical.detectable_count,
        "n_nondet": physical.nondetectable_count,
        "n_detector_eval": physical.detector_evaluated_count,
        "n_detector_unavailable": physical.detector_unavailable_count,
        "n_snr_below": physical.below_threshold_count,
        "n_no_overlap": physical.no_overlap_count,
        "n_detector_input_unavailable": physical.required_input_unavailable_count,
        "n_detector_model_unavailable": physical.model_unavailable_count,
        "n_detector_numerical_failure": physical.numerical_failure_count,
        "rate_det_per_myr": detector_rate,
        "n_band_eval": result.band_entry_evaluated_count,
        "n_entry_in_domain": result.entry_in_migration_domain_count,
        "n_entry_handoff": result.entry_handoff_count,
        "n_entry_inside_isco": result.band_entry_inside_isco_count,
        "n_profile_error": result.profile_error_count,
        "n_unavailable": (
            result.band_entry_inside_isco_count + result.profile_error_count
        ),
        "n_mig_eval": result.migration_evaluated_count,
        "n_mig_finite": result.finite_migration_count,
        "n_mig_trap": result.trap_bound_migration_count,
        "n_mig_error": result.migration_error_count,
    }


def save_repeat_results(
    averaged_results: Sequence[IMRIDynAveragedResult],
    output_dir: Path,
    config: IMRIDynConfig,
) -> pd.DataFrame:
    """Save one row for every normally collected Monte Carlo repeat."""
    records = [
        _repeat_output_record(result, config)
        for averaged_result in averaged_results
        for result in averaged_result.results
    ]
    frame = pd.DataFrame.from_records(records, columns=IMRI_REPEAT_RESULT_COLUMNS)
    for column in ("base_seed", "task_seed"):
        frame[column] = pd.array(
            cast(pd.Series, frame[column]),
            dtype="Int64",
        )
    _validate_rate_output(frame)
    path = output_dir / IMRI_REPEAT_RESULTS_FILENAME
    frame.to_csv(path, index=False, float_format="%.6e")
    print(f"saved: {path}")
    return frame


def save_migration_diagnostics(
    result: IMRIDynSingleResult,
    output_dir: Path,
    config: IMRIDynConfig,
) -> Path:
    """Save strict per-repeat migration diagnostics as JSON."""

    if result.disk_model != config.disk_model:
        raise ValueError("result disk model disagrees with workflow configuration")
    physical = _physical_entry_summary(result)
    rate_status = _rate_evaluation_status(result)
    detector_rate_status = _detector_status_for_summary(result, physical)

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

    band_entry_objects = [
        {
            "object_index": event["object_index"],
            "mass_msun": json_float(event["mass_msun"]),
            "band_entry_status": event["band_entry_status"],
            "source_entry_frequency_hz": json_float(event["source_entry_frequency_hz"]),
            "r_entry_formal_rg": json_float(event["r_entry_formal_rg"]),
            "r_isco_rg": json_float(event["r_isco_rg"]),
            "r_disk_inner_rg": json_float(event["r_disk_inner_rg"]),
            "r_migration_min_rg": json_float(event["r_migration_min_rg"]),
            "r_handoff_rg": json_float(event["r_handoff_rg"]),
            "migration_time_to_handoff_myr": json_float(
                event["migration_time_to_handoff_myr"]
            ),
            "gw_time_handoff_to_entry_myr": json_float(
                event["gw_time_handoff_to_entry_myr"]
            ),
            "gas_stop_reason": event["gas_stop_reason"],
            "migration_evaluated": event["migration_evaluated"],
        }
        for event in result.imri_events
    ]

    recorded_band_status_count = (
        result.entry_in_migration_domain_count
        + result.entry_handoff_count
        + result.band_entry_inside_isco_count
        + result.profile_error_count
    )
    if result.band_entry_evaluated_count != recorded_band_status_count:
        raise ValueError("band-entry counters do not conserve evaluated objects")
    if result.migration_evaluated_count != (
        result.entry_in_migration_domain_count
        + result.entry_handoff_count
        + result.profile_error_count
    ):
        raise ValueError("migration-applicable counters do not sum to evaluated count")

    if (
        result.migration_evaluated_count
        != result.finite_migration_count
        + result.trap_bound_migration_count
        + result.migration_error_count
    ):
        raise ValueError("migration counters do not sum to evaluated count")
    if result.trap_bound_migration_count != len(trap_bound_objects):
        raise ValueError("trap-bound counter does not match event records")
    if (
        result.disk_model == SEMI_WITHOUT_TRAP
        and result.trap_bound_migration_count != 0
    ):
        raise ValueError("semi-analytic migration cannot produce trap-bound outcomes")
    if result.migration_error_count != len(result.migration_errors):
        raise ValueError("migration error counter does not match error records")
    if result.profile_error_count != result.migration_error_count:
        raise ValueError("profile-error counter does not match migration errors")

    if result.imri_count != (
        result.detector_evaluated_imri_count
        + result.detector_unavailable_imri_count
    ):
        raise ValueError("intrinsic IMRI candidates do not conserve detector outcomes")
    if result.detector_evaluated_imri_count != (
        result.detectable_imri_count + result.no_detectable_imri_count
    ):
        raise ValueError("detector-evaluated IMRIs do not conserve classifications")
    if result.no_detectable_imri_count != (
        result.detector_below_threshold_imri_count
        + result.detector_no_overlap_imri_count
    ):
        raise ValueError("physical nondetections do not conserve SNR outcomes")
    if result.detector_unavailable_imri_count != (
        result.detector_required_input_unavailable_imri_count
        + result.detector_model_unavailable_imri_count
        + result.detector_numerical_failure_imri_count
    ):
        raise ValueError("detector-unavailable IMRIs do not conserve failure outcomes")
    if (
        np.isfinite(result.imri_rate_per_myr)
        and np.isfinite(result.detectable_imri_rate_per_myr)
        and result.detectable_imri_rate_per_myr > result.imri_rate_per_myr
    ):
        raise ValueError("detectable IMRI rate exceeds intrinsic IMRI rate")

    intrinsic_contribution_per_myr = sum(
        event["intrinsic_rate_contribution_per_myr"]
        for event in result.imri_events
        if np.isfinite(event["intrinsic_rate_contribution_per_myr"])
    )
    detected_contribution_per_myr = sum(
        event["detected_rate_contribution_per_myr"]
        for event in result.imri_events
        if np.isfinite(event["detected_rate_contribution_per_myr"])
    )
    expected_intrinsic_contribution_per_myr = (
        result.imri_count / config.duty_cycle_myr
    )
    expected_detected_contribution_per_myr = (
        result.detectable_imri_count / config.duty_cycle_myr
    )
    if not np.isclose(
        intrinsic_contribution_per_myr,
        expected_intrinsic_contribution_per_myr,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("event intrinsic-rate contributions do not match the repeat")
    if not np.isclose(
        detected_contribution_per_myr,
        expected_detected_contribution_per_myr,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("event detected-rate contributions do not match the repeat")

    event_status_counts = {
        status: sum(
            event["band_entry_status"] == status for event in result.imri_events
        )
        for status in (
            "entry_in_migration_domain",
            "entry_below_migration_domain_but_above_isco",
            "band_entry_inside_isco",
        )
    }
    if event_status_counts["entry_in_migration_domain"] != (
        result.entry_in_migration_domain_count
    ):
        raise ValueError("in-domain counter does not match event records")
    if event_status_counts["entry_below_migration_domain_but_above_isco"] != (
        result.entry_handoff_count
    ):
        raise ValueError("handoff counter does not match event records")
    if event_status_counts["band_entry_inside_isco"] != (
        result.band_entry_inside_isco_count
    ):
        raise ValueError("inside-ISCO counter does not match event records")

    payload = {
        "m_smbh_msun": json_float(result.m_smbh_msun),
        "m_nsc_msun": json_float(result.m_nsc_msun),
        "repeat": result.repeat_idx,
        "disk_model": result.disk_model,
        "redshift": json_float(config.redshift),
        "base_seed": config.base_seed,
        "task_seed": result.task_seed,
        "duty_cycle_myr": json_float(config.duty_cycle_myr),
        "gap_opening_condition": config.gap_opening_condition,
        "consider_type_ii_migration": config.consider_type_ii_migration,
        "rate_evaluation_status": rate_status,
        "detector_rate_evaluation_status": detector_rate_status,
        "physical_entry_imri_count": physical.count,
        "gw_propagation_eligible_count": result.imri_count,
        "band_entry_evaluated_count": result.band_entry_evaluated_count,
        "entry_in_migration_domain_count": result.entry_in_migration_domain_count,
        "entry_handoff_count": result.entry_handoff_count,
        "band_entry_inside_isco_count": result.band_entry_inside_isco_count,
        "profile_error_count": result.profile_error_count,
        "detector_evaluated_imri_count": physical.detector_evaluated_count,
        "detector_unavailable_imri_count": physical.detector_unavailable_count,
        "detector_below_threshold_imri_count": (
            physical.below_threshold_count
        ),
        "detector_no_overlap_imri_count": physical.no_overlap_count,
        "detector_required_input_unavailable_imri_count": (
            physical.required_input_unavailable_count
        ),
        "detector_model_unavailable_imri_count": (
            physical.model_unavailable_count
        ),
        "detector_numerical_failure_imri_count": (
            physical.numerical_failure_count
        ),
        "migration_evaluated_count": result.migration_evaluated_count,
        "finite_migration_count": result.finite_migration_count,
        "trap_bound_migration_count": result.trap_bound_migration_count,
        "migration_error_count": result.migration_error_count,
        "trap_bound_mass_msun": json_float(result.trap_bound_mass_msun),
        "imri_rate_per_myr": json_float(
            physical.count / config.duty_cycle_myr
            if rate_status == "complete"
            else np.nan
        ),
        "detectable_imri_rate_per_myr": json_float(
            physical.detectable_count / config.duty_cycle_myr
            if detector_rate_status == "complete"
            else np.nan
        ),
        "gw_propagation_eligible_rate_per_myr": json_float(result.imri_rate_per_myr),
        "band_entry_objects": band_entry_objects,
        "trap_bound_objects": trap_bound_objects,
        "migration_errors": [
            {
                "object_index": error["object_index"],
                "mass_msun": json_float(error["mass_msun"]),
                "r_ini_rg": json_float(error["r_ini_rg"]),
                "r_entry_rg": json_float(error["r_entry_rg"]),
                "r_isco_rg": json_float(error["r_isco_rg"]),
                "band_entry_status": error["band_entry_status"],
                "error": error["error"],
            }
            for error in result.migration_errors
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

    def json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return json_safe(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    path = output_dir / METADATA_FILENAME
    with path.open("w") as metadata_file:
        json.dump(json_safe(metadata), metadata_file, indent=2, allow_nan=False)
    print(f"saved: {path}")
