# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportOperatorIssue=false, reportIndexIssue=false, reportCallIssue=false, reportGeneralTypeIssues=false
"""Build compact scholar-facing IMRI rate tables and figures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import (
    DISK_MODELS,
    IMRI_AVERAGED_RESULTS_FILENAME,
    IMRI_EVENTS_FILENAME,
    IMRI_REPEAT_RESULTS_FILENAME,
    METADATA_FILENAME,
)

GROUP_COLUMNS = ["disk_model", "redshift", "m_smbh_msun", "m_nsc_msun"]
REPEAT_KEY_COLUMNS = [*GROUP_COLUMNS, "repeat"]
EVENT_REQUIRED_COLUMNS = [
    *REPEAT_KEY_COLUMNS,
    "gas_final_time_myr",
    "is_imri_candidate",
]
REPEAT_REQUIRED_COLUMNS = [
    *REPEAT_KEY_COLUMNS,
    "rate_status",
    "duty_cycle_myr",
    "n_imri",
    "rate_imri_per_myr",
    "n_det",
    "rate_det_per_myr",
]
TIME_SERIES_COLUMNS = [
    *GROUP_COLUMNS,
    "time_myr",
    "n_repeats_valid",
    "cumulative_rate_mean_per_myr",
    "cumulative_rate_p16_per_myr",
    "cumulative_rate_p84_per_myr",
]
RATE_SUMMARY_COLUMNS = [
    *GROUP_COLUMNS,
    "duty_cycle_myr",
    "n_repeats_total",
    "n_repeats_valid",
    "mean_imri_count",
    "imri_rate_mean_per_myr",
    "imri_rate_p16_per_myr",
    "imri_rate_p84_per_myr",
    "proxy_detectable_rate_mean_per_myr",
    "result_state",
]
STATUS_FRACTION_COLUMNS = [
    "disk_model",
    "redshift",
    "m_nsc_msun",
    "variable",
    "category",
    "count",
    "denominator",
    "fraction",
    "result_state",
]
STATUS_FRACTION_SMBH_MASS_COLUMNS = [
    "disk_model",
    "m_smbh_msun",
    "m_nsc_msun",
    "variable",
    "category",
    "count",
    "denominator",
    "fraction",
    "result_state",
]
BAND_ENTRY_COUNTER_CATEGORIES = {
    "n_entry_in_domain": "entry_in_migration_domain",
    "n_entry_handoff": "entry_below_migration_domain_but_above_isco",
    "n_entry_inside_isco": "band_entry_inside_isco",
    "n_profile_error": "profile_error",
}
STATUS_VARIABLE_ORDER = (
    "band_entry_status",
    "transport_status",
    "event_outcome",
    "detectability_outcome",
)
BAND_ENTRY_ORDER = tuple(BAND_ENTRY_COUNTER_CATEGORIES.values())
TRANSPORT_ORDER = (
    "already_inside_lisa_entry",
    "entry_reached",
    "gas_to_gw_handoff",
    "gw_dominates_at_start",
    "gw_dominates_after_gas",
    "duty_cycle_limited",
    "damping_misses_duty_cycle",
    "migration_trap_bound",
    "invalid_radius",
)
EVENT_OUTCOME_ORDER = ("imri candi", "no-IMRI", "not evaluated")
DETECTABILITY_ORDER = ("imri data", "non-detectable", "not evaluated")
STATUS_CATEGORY_ORDER = {
    "band_entry_status": BAND_ENTRY_ORDER,
    "transport_status": TRANSPORT_ORDER,
    "event_outcome": EVENT_OUTCOME_ORDER,
    "detectability_outcome": DETECTABILITY_ORDER,
}
TRANSPORT_STATUS_BY_GAS_STOP_REASON = {
    category: category for category in TRANSPORT_ORDER
}
# Keep invalid-radius evaluations visible in this denominator as an explicit
# not-evaluable outcome; inside-ISCO and blank/null reasons remain excluded.
TRANSPORT_EXCLUDED_REASONS = {"band_entry_inside_isco"}
STATUS_VARIABLE_LABELS = {
    "band_entry_status": "Band-entry geometry",
    "transport_status": "Gas/GW transport outcome\n(including invalid states)",
    "event_outcome": "IMRI event outcome",
    "detectability_outcome": "Detectability outcome",
}
STATUS_CATEGORY_LABELS = {
    "band_entry_status": {
        "entry_in_migration_domain": "In migration domain",
        "entry_below_migration_domain_but_above_isco": "GW handoff required",
        "band_entry_inside_isco": "Inside ISCO",
        "profile_error": "Profile unavailable",
    },
    "transport_status": {
        "already_inside_lisa_entry": "Already at/inside entry",
        "entry_reached": "Gas reaches entry",
        "gas_to_gw_handoff": "Gas reaches GW handoff",
        "gw_dominates_at_start": "GW faster at start",
        "gw_dominates_after_gas": "Gas→GW timescale crossover",
        "duty_cycle_limited": "AGN lifetime exhausted",
        "damping_misses_duty_cycle": "No gas time after embedding",
        "migration_trap_bound": "Migration trapped",
        "invalid_radius": "Transport not evaluable",
    },
    "event_outcome": {
        "imri candi": "IMRI candidate",
        "no-IMRI": "No IMRI",
        "not evaluated": "Not evaluated",
    },
    "detectability_outcome": {
        "imri data": "Detectable",
        "non-detectable": "Non-detectable",
        "not evaluated": "Not evaluated",
    },
}
STATUS_CATEGORY_COLORS = {
    "band_entry_status": {
        "entry_in_migration_domain": "#0072B2",
        "entry_below_migration_domain_but_above_isco": "#009E73",
        "band_entry_inside_isco": "#D55E00",
        "profile_error": "#7F7F7F",
    },
    "transport_status": {
        "already_inside_lisa_entry": "#56B4E9",
        "entry_reached": "#009E73",
        "gas_to_gw_handoff": "#0072B2",
        "gw_dominates_at_start": "#CC79A7",
        "gw_dominates_after_gas": "#9467BD",
        "duty_cycle_limited": "#E69F00",
        "damping_misses_duty_cycle": "#F0E442",
        "migration_trap_bound": "#D55E00",
        "invalid_radius": "#7F7F7F",
    },
    "event_outcome": {
        "imri candi": "#009E73",
        "no-IMRI": "#D55E00",
        "not evaluated": "#7F7F7F",
    },
    "detectability_outcome": {
        "imri data": "#0072B2",
        "non-detectable": "#E69F00",
        "not evaluated": "#7F7F7F",
    },
}

MIN_REPEATS_FOR_BAND = 10
FIGURE_SIZE = (7.0, 4.5)
OKABE_ITO_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
)


@dataclass(frozen=True)
class RunResults:
    """Persisted tables and metadata for one scalar-redshift workflow run."""

    events: pd.DataFrame
    repeats: pd.DataFrame
    averaged: pd.DataFrame
    metadata: dict[str, Any]


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _true_mask(series: pd.Series) -> pd.Series:
    return series.eq(True) | series.astype(str).str.lower().eq("true")  # noqa: E712


def validate_repeat_results(repeats: pd.DataFrame) -> None:
    """Validate repeat identity, rate denominators, and availability semantics."""
    _require_columns(repeats, REPEAT_REQUIRED_COLUMNS, "repeat results")
    if repeats.duplicated(REPEAT_KEY_COLUMNS).any():
        duplicates = repeats.loc[
            repeats.duplicated(REPEAT_KEY_COLUMNS, keep=False), REPEAT_KEY_COLUMNS
        ]
        raise ValueError(f"repeat results contain duplicate keys:\n{duplicates}")

    duty_cycle = pd.to_numeric(repeats["duty_cycle_myr"], errors="coerce")
    if duty_cycle.isna().any() or (duty_cycle <= 0.0).any():
        raise ValueError("repeat duty_cycle_myr values must be finite and positive")

    complete = repeats["rate_status"].eq("complete")
    for count_column, rate_column in (
        ("n_imri", "rate_imri_per_myr"),
        ("n_det", "rate_det_per_myr"),
    ):
        count = pd.to_numeric(repeats[count_column], errors="coerce")
        rate = pd.to_numeric(repeats[rate_column], errors="coerce")
        expected = count / duty_cycle
        if rate[complete].isna().any() or not np.allclose(
            rate[complete], expected[complete], rtol=5.0e-6, atol=5.0e-12
        ):
            raise ValueError(
                f"complete repeat {rate_column} does not equal "
                f"{count_column}/duty_cycle_myr"
            )
        if rate[~complete].notna().any():
            raise ValueError(f"unavailable repeat {rate_column} must remain NaN")


def validate_event_keys(events: pd.DataFrame, repeats: pd.DataFrame) -> None:
    """Require every persisted event to resolve to one repeat result."""
    _require_columns(events, EVENT_REQUIRED_COLUMNS, "event results")
    event_keys = events[REPEAT_KEY_COLUMNS].drop_duplicates()
    repeat_keys = repeats[REPEAT_KEY_COLUMNS].drop_duplicates()
    joined = event_keys.merge(
        repeat_keys, on=REPEAT_KEY_COLUMNS, how="left", indicator=True
    )
    missing = joined["_merge"].ne("both")
    if missing.any():
        raise ValueError(
            "event rows do not resolve to repeat results:\n"
            f"{joined.loc[missing, REPEAT_KEY_COLUMNS]}"
        )


def load_run_results(results_dir: str | Path) -> RunResults:
    """Load and validate one completed scalar-redshift workflow output."""
    root = Path(results_dir)
    required_paths = {
        "events": root / IMRI_EVENTS_FILENAME,
        "repeats": root / IMRI_REPEAT_RESULTS_FILENAME,
        "averaged": root / IMRI_AVERAGED_RESULTS_FILENAME,
        "metadata": root / METADATA_FILENAME,
    }
    missing = [str(path) for path in required_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "result directory is incomplete or predates the visualization schema; "
            f"rerun the workflow to create: {missing}"
        )

    events = pd.read_csv(required_paths["events"])
    repeats = pd.read_csv(required_paths["repeats"])
    averaged = pd.read_csv(required_paths["averaged"])
    with required_paths["metadata"].open() as metadata_file:
        metadata = json.load(metadata_file)

    validate_repeat_results(repeats)
    validate_event_keys(events, repeats)
    redshifts = pd.to_numeric(repeats["redshift"], errors="coerce").dropna().unique()
    if len(redshifts) != 1:
        raise ValueError("one workflow result directory must contain one redshift")
    disk_models = repeats["disk_model"].dropna().astype(str).unique()
    if len(disk_models) != 1:
        raise ValueError("one workflow result directory must contain one disk model")
    metadata_redshift = float(metadata["parameters"]["redshift"])
    if not np.isclose(float(redshifts[0]), metadata_redshift):
        raise ValueError("repeat-table redshift disagrees with simulation metadata")
    metadata_disk_model = str(metadata["parameters"]["disk_model"])
    if str(disk_models[0]) != metadata_disk_model:
        raise ValueError("repeat-table disk model disagrees with simulation metadata")
    return RunResults(events, repeats, averaged, metadata)


def _statistics(values: np.ndarray) -> tuple[float, float, float]:
    if values.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(values))
    if values.size < 2:
        return mean, np.nan, np.nan
    lower, upper = np.quantile(values, [0.16, 0.84])
    return mean, float(lower), float(upper)


def _event_group(
    events: pd.DataFrame,
    disk_model: str,
    redshift: float,
    m_smbh_msun: float,
    m_nsc_msun: float,
) -> pd.DataFrame:
    mask = (
        events["disk_model"].astype(str).eq(disk_model)
        & np.isclose(events["redshift"].astype(float), redshift)
        & np.isclose(events["m_smbh_msun"].astype(float), m_smbh_msun)
        & np.isclose(events["m_nsc_msun"].astype(float), m_nsc_msun)
    )
    return events.loc[mask]


def build_agn_time_plot_data(
    events: pd.DataFrame,
    repeats: pd.DataFrame,
) -> pd.DataFrame:
    """Build compact cumulative candidate-rate time series."""
    validate_repeat_results(repeats)
    validate_event_keys(events, repeats)
    records: list[dict[str, float | int | str]] = []

    for group_key, repeat_group in repeats.groupby(GROUP_COLUMNS, sort=True):
        disk_model = str(group_key[0])
        redshift, m_smbh_msun, m_nsc_msun = (float(value) for value in group_key[1:])
        duties = pd.to_numeric(repeat_group["duty_cycle_myr"]).unique()
        if len(duties) != 1:
            raise ValueError("duty cycle must be constant within a plotted grid bin")
        duty_cycle_myr = float(duties[0])
        valid_repeats = repeat_group[repeat_group["rate_status"].eq("complete")]
        valid_ids = valid_repeats["repeat"].astype(int).to_numpy()

        event_group = _event_group(
            events, disk_model, redshift, m_smbh_msun, m_nsc_msun
        )
        candidate_events = event_group[
            _true_mask(event_group["is_imri_candidate"])
            & event_group["repeat"].isin(valid_ids)
        ].copy()
        candidate_events["_candidate_time"] = pd.to_numeric(
            candidate_events["gas_final_time_myr"], errors="coerce"
        )
        finite_candidates = candidate_events[
            np.isfinite(candidate_events["_candidate_time"])
            & candidate_events["_candidate_time"].between(
                0.0, duty_cycle_myr, inclusive="both"
            )
        ]
        time_grid = np.unique(
            np.concatenate(
                (
                    np.asarray([0.0, duty_cycle_myr]),
                    finite_candidates["_candidate_time"].to_numpy(dtype=float),
                )
            )
        )

        for time_myr in time_grid:
            repeat_rates = np.asarray(
                [
                    float(
                        (
                            finite_candidates["repeat"].eq(repeat_index)
                            & finite_candidates["_candidate_time"].le(time_myr)
                        ).sum()
                    )
                    / duty_cycle_myr
                    for repeat_index in valid_ids
                ],
                dtype=float,
            )
            mean, lower, upper = _statistics(repeat_rates)
            records.append(
                {
                    "disk_model": disk_model,
                    "redshift": redshift,
                    "m_smbh_msun": m_smbh_msun,
                    "m_nsc_msun": m_nsc_msun,
                    "time_myr": float(time_myr),
                    "n_repeats_valid": len(valid_ids),
                    "cumulative_rate_mean_per_myr": mean,
                    "cumulative_rate_p16_per_myr": lower,
                    "cumulative_rate_p84_per_myr": upper,
                }
            )
    return pd.DataFrame.from_records(records, columns=TIME_SERIES_COLUMNS)


def build_rate_summary_data(repeats: pd.DataFrame) -> pd.DataFrame:
    """Build one compact scholar-facing row per model/redshift/mass bin."""
    validate_repeat_results(repeats)
    records: list[dict[str, float | int | str]] = []

    for group_key, group in repeats.groupby(GROUP_COLUMNS, sort=True):
        disk_model = str(group_key[0])
        redshift, m_smbh_msun, m_nsc_msun = (float(value) for value in group_key[1:])
        duties = pd.to_numeric(group["duty_cycle_myr"]).unique()
        if len(duties) != 1:
            raise ValueError("duty cycle must be constant within a plotted grid bin")

        valid = group[group["rate_status"].eq("complete")]
        n_total = len(group)
        n_valid = len(valid)
        result_state = (
            "valid" if n_valid == n_total else "partial" if n_valid else "unavailable"
        )
        if n_valid:
            mean_count = float(pd.to_numeric(valid["n_imri"]).mean())
            rate_values = pd.to_numeric(valid["rate_imri_per_myr"]).to_numpy(
                dtype=float
            )
            mean_rate, lower_rate, upper_rate = _statistics(rate_values)
            proxy_rate = float(pd.to_numeric(valid["rate_det_per_myr"]).mean())
        else:
            mean_count = mean_rate = lower_rate = upper_rate = proxy_rate = np.nan

        records.append(
            {
                "disk_model": disk_model,
                "redshift": redshift,
                "m_smbh_msun": m_smbh_msun,
                "m_nsc_msun": m_nsc_msun,
                "duty_cycle_myr": float(duties[0]),
                "n_repeats_total": n_total,
                "n_repeats_valid": n_valid,
                "mean_imri_count": mean_count,
                "imri_rate_mean_per_myr": mean_rate,
                "imri_rate_p16_per_myr": lower_rate,
                "imri_rate_p84_per_myr": upper_rate,
                "proxy_detectable_rate_mean_per_myr": proxy_rate,
                "result_state": result_state,
            }
        )
    return pd.DataFrame.from_records(records, columns=RATE_SUMMARY_COLUMNS)


def shared_rate_axis_limits(summary_data: pd.DataFrame) -> tuple[float, float]:
    """Return one nonnegative rate range suitable for paired scenario figures."""
    _require_columns(summary_data, RATE_SUMMARY_COLUMNS, "rate summary")
    values = summary_data[
        [
            "imri_rate_mean_per_myr",
            "imri_rate_p16_per_myr",
            "imri_rate_p84_per_myr",
        ]
    ].apply(pd.to_numeric, errors="coerce")
    finite = values.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    upper = float(np.max(finite)) if finite.size else 0.0
    return 0.0, (1.05 * upper if upper > 0.0 else 1.0)


def _status_fraction_columns(scan_column: str) -> list[str]:
    if scan_column == "redshift":
        return STATUS_FRACTION_COLUMNS
    if scan_column == "m_smbh_msun":
        return STATUS_FRACTION_SMBH_MASS_COLUMNS
    raise ValueError(f"unsupported status-fraction scan column: {scan_column}")


def validate_status_fraction_data(
    status_data: pd.DataFrame,
    scan_column: str = "redshift",
) -> None:
    """Validate the four-layer taxonomy, denominators, and conservation."""
    _require_columns(
        status_data, _status_fraction_columns(scan_column), "status fractions"
    )
    expected_variables = set(STATUS_VARIABLE_ORDER)
    unknown_variables = sorted(
        set(status_data["variable"].astype(str)).difference(expected_variables)
    )
    if unknown_variables:
        raise ValueError(
            f"status fractions contain unknown variables: {unknown_variables}"
        )
    for variable in STATUS_VARIABLE_ORDER:
        expected_categories = set(STATUS_CATEGORY_ORDER[variable])
        if set(STATUS_CATEGORY_LABELS[variable]) != expected_categories:
            raise ValueError(f"status labels are incomplete for {variable}")
        if set(STATUS_CATEGORY_COLORS[variable]) != expected_categories:
            raise ValueError(f"status colors are incomplete for {variable}")

    key_columns = [
        "disk_model",
        scan_column,
        "m_nsc_msun",
        "variable",
        "category",
    ]
    if status_data.duplicated(key_columns).any():
        raise ValueError("status fractions contain duplicate category keys")

    count = pd.to_numeric(status_data["count"], errors="coerce")
    denominator = pd.to_numeric(status_data["denominator"], errors="coerce")
    fraction = pd.to_numeric(status_data["fraction"], errors="coerce")
    if (
        not np.isfinite(count.to_numpy(dtype=float)).all()
        or not np.isfinite(denominator.to_numpy(dtype=float)).all()
    ):
        raise ValueError("status counts and denominators must be finite")
    if (count < 0.0).any() or (denominator < 0.0).any():
        raise ValueError("status counts and denominators must be nonnegative")

    valid = status_data["result_state"].eq("valid")
    unavailable = status_data["result_state"].eq("unavailable")
    if not (valid | unavailable).all():
        raise ValueError("status fraction state must be valid or unavailable")
    if (
        (denominator[valid] <= 0.0).any()
        or not np.isfinite(fraction[valid].to_numpy(dtype=float)).all()
        or (fraction[valid] < 0.0).any()
        or (fraction[valid] > 1.0).any()
    ):
        raise ValueError("valid status fractions require a positive denominator")
    if (denominator[unavailable] != 0.0).any() or fraction[unavailable].notna().any():
        raise ValueError("unavailable fractions require zero denominator and NaN")

    group_columns = ["disk_model", scan_column, "m_nsc_msun", "variable"]
    scan_group_columns = ["disk_model", scan_column, "m_nsc_msun"]
    for scan_key, scan_group in status_data.groupby(scan_group_columns, sort=False):
        variables = set(scan_group["variable"].astype(str))
        if variables != expected_variables:
            raise ValueError(
                f"status bin {scan_key} does not contain all four variables"
            )

    for group_key, group in status_data.groupby(group_columns, sort=False):
        variable = str(group_key[-1])
        observed_categories = set(group["category"].astype(str))
        expected_categories = set(STATUS_CATEGORY_ORDER[variable])
        if observed_categories != expected_categories:
            unexpected = sorted(observed_categories.difference(expected_categories))
            missing = sorted(expected_categories.difference(observed_categories))
            raise ValueError(
                f"invalid categories for {variable}: "
                f"unexpected={unexpected}, missing={missing}"
            )
        group_denominators = pd.to_numeric(group["denominator"]).unique()
        if len(group_denominators) != 1:
            raise ValueError("status denominator varies within one scan bin")
        group_denominator = float(group_denominators[0])
        group_count = float(pd.to_numeric(group["count"]).sum())
        if group_denominator == 0.0:
            if group_count != 0.0 or not group["result_state"].eq(
                "unavailable"
            ).all():
                raise ValueError("unavailable status bin has a nonzero count")
            continue
        if not group["result_state"].eq("valid").all():
            raise ValueError("positive status denominator requires valid rows")
        if not np.isclose(group_count, group_denominator, rtol=0.0, atol=1.0e-12):
            raise ValueError("status counts do not conserve the denominator")
        group_fraction = float(pd.to_numeric(group["fraction"]).sum())
        # Status tables are persisted with ten significant digits.
        if not np.isclose(group_fraction, 1.0, rtol=0.0, atol=1.0e-9):
            raise ValueError("status fractions do not sum to one")


def _normalize_transport_status(gas_stop_reason: pd.Series) -> pd.Series:
    """Return strict plot-layer transport keys while excluding non-transport rows."""
    raw = gas_stop_reason.astype("string").str.strip()
    defined = raw.notna() & raw.ne("")
    observed = {str(value) for value in raw[defined].unique()}
    allowed = set(TRANSPORT_STATUS_BY_GAS_STOP_REASON) | TRANSPORT_EXCLUDED_REASONS
    unknown = sorted(observed.difference(allowed))
    if unknown:
        raise ValueError(f"unknown gas_stop_reason categories: {unknown}")
    return raw.map(TRANSPORT_STATUS_BY_GAS_STOP_REASON).astype("string")


def _validate_class_values(
    values: pd.Series,
    *,
    source_column: str,
    expected_categories: Sequence[str],
) -> pd.Series:
    raw = values.astype("string").str.strip()
    missing = raw.isna() | raw.eq("")
    observed = {str(value) for value in raw[~missing].unique()}
    unknown = sorted(observed.difference(expected_categories))
    if missing.any():
        unknown.append("<blank/null>")
    if unknown:
        raise ValueError(f"unknown {source_column} categories: {unknown}")
    return raw


def _fraction_records(
    *,
    disk_model: str,
    scan_column: str,
    scan_value: float,
    nsc_mass_msun: float,
    variable: str,
    category_counts: dict[str, float],
) -> list[dict[str, float | str]]:
    categories = STATUS_CATEGORY_ORDER[variable]
    denominator = float(sum(category_counts[category] for category in categories))
    state = "valid" if denominator > 0.0 else "unavailable"
    return [
        {
            "disk_model": disk_model,
            scan_column: scan_value,
            "m_nsc_msun": nsc_mass_msun,
            "variable": variable,
            "category": category,
            "count": category_counts[category],
            "denominator": denominator,
            "fraction": (
                category_counts[category] / denominator
                if denominator > 0.0
                else np.nan
            ),
            "result_state": state,
        }
        for category in categories
    ]


def _build_status_fraction_data(
    events: pd.DataFrame,
    repeats: pd.DataFrame,
    scan_column: str,
) -> pd.DataFrame:
    validate_repeat_results(repeats)
    validate_event_keys(events, repeats)
    _require_columns(
        repeats,
        [*GROUP_COLUMNS, "n_band_eval", *BAND_ENTRY_COUNTER_CATEGORIES],
        "repeat results",
    )
    _require_columns(
        events,
        [
            *GROUP_COLUMNS,
            "gas_stop_reason",
            "event_rate_class",
            "detect_rate_class",
        ],
        "event results",
    )

    output_columns = _status_fraction_columns(scan_column)
    scan_columns = ["disk_model", scan_column, "m_nsc_msun"]
    records: list[dict[str, float | str]] = []
    expected_bins = repeats[scan_columns].drop_duplicates().sort_values(scan_columns)
    normalized_events = events.copy()
    normalized_events["_transport_status"] = _normalize_transport_status(
        normalized_events["gas_stop_reason"]
    )
    normalized_events["_event_outcome"] = _validate_class_values(
        normalized_events["event_rate_class"],
        source_column="event_rate_class",
        expected_categories=EVENT_OUTCOME_ORDER,
    )
    normalized_events["_detectability_outcome"] = _validate_class_values(
        normalized_events["detect_rate_class"],
        source_column="detect_rate_class",
        expected_categories=DETECTABILITY_ORDER,
    )

    for _, expected in expected_bins.iterrows():
        disk_model = str(expected["disk_model"])
        scan_value = float(expected[scan_column])
        nsc_mass_msun = float(expected["m_nsc_msun"])
        repeat_mask = (
            repeats["disk_model"].astype(str).eq(disk_model)
            & np.isclose(repeats[scan_column].astype(float), scan_value)
            & np.isclose(repeats["m_nsc_msun"].astype(float), nsc_mass_msun)
        )
        repeat_group = repeats.loc[repeat_mask]
        band_denominator = float(pd.to_numeric(repeat_group["n_band_eval"]).sum())
        band_counts = {
            category: float(pd.to_numeric(repeat_group[column]).sum())
            for column, category in BAND_ENTRY_COUNTER_CATEGORIES.items()
        }
        if not np.isclose(
            sum(band_counts.values()), band_denominator, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError("band-entry counters do not sum to n_band_eval")
        records.extend(
            _fraction_records(
                disk_model=disk_model,
                scan_column=scan_column,
                scan_value=scan_value,
                nsc_mass_msun=nsc_mass_msun,
                variable="band_entry_status",
                category_counts=band_counts,
            )
        )

        event_mask = (
            normalized_events["disk_model"].astype(str).eq(disk_model)
            & np.isclose(normalized_events[scan_column].astype(float), scan_value)
            & np.isclose(
                normalized_events["m_nsc_msun"].astype(float), nsc_mass_msun
            )
        )
        event_group = normalized_events.loc[event_mask]
        for variable, source_column in (
            ("transport_status", "_transport_status"),
            ("event_outcome", "_event_outcome"),
            ("detectability_outcome", "_detectability_outcome"),
        ):
            source = event_group[source_column].dropna().astype(str)
            category_counts = {
                category: float(source.eq(category).sum())
                for category in STATUS_CATEGORY_ORDER[variable]
            }
            records.extend(
                _fraction_records(
                    disk_model=disk_model,
                    scan_column=scan_column,
                    scan_value=scan_value,
                    nsc_mass_msun=nsc_mass_msun,
                    variable=variable,
                    category_counts=category_counts,
                )
            )

    frame = pd.DataFrame.from_records(records, columns=output_columns)
    validate_status_fraction_data(frame, scan_column)
    return frame


def build_status_fraction_data(
    events: pd.DataFrame,
    repeats: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate four diagnostic layers across the configured SMBH-mass grid."""
    return _build_status_fraction_data(events, repeats, "redshift")


def build_status_fraction_by_smbh_mass_data(
    events: pd.DataFrame,
    repeats: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate four diagnostic layers across the requested redshift grid."""
    return _build_status_fraction_data(events, repeats, "m_smbh_msun")


def validate_agn_time_endpoints(
    time_data: pd.DataFrame,
    repeats: pd.DataFrame,
    averaged: pd.DataFrame,
) -> None:
    """Check complete cumulative endpoints against persisted averaged rates."""
    _require_columns(time_data, TIME_SERIES_COLUMNS, "time-series results")
    _require_columns(
        averaged,
        [
            "disk_model",
            "m_smbh_msun",
            "m_nsc_msun",
            "rate_status",
            "rate_imri_per_myr",
        ],
        "averaged results",
    )
    for group_key, repeat_group in repeats.groupby(GROUP_COLUMNS, sort=True):
        if not repeat_group["rate_status"].eq("complete").all():
            continue
        disk_model = str(group_key[0])
        redshift, m_smbh_msun, m_nsc_msun = (float(value) for value in group_key[1:])
        duty_cycle_myr = float(repeat_group["duty_cycle_myr"].iloc[0])
        endpoint = time_data[
            time_data["disk_model"].astype(str).eq(disk_model)
            & np.isclose(time_data["redshift"], redshift)
            & np.isclose(time_data["m_smbh_msun"], m_smbh_msun)
            & np.isclose(time_data["m_nsc_msun"], m_nsc_msun)
            & np.isclose(time_data["time_myr"], duty_cycle_myr)
        ]
        reference = averaged[
            averaged["disk_model"].astype(str).eq(disk_model)
            & np.isclose(averaged["m_smbh_msun"], m_smbh_msun)
            & np.isclose(averaged["m_nsc_msun"], m_nsc_msun)
        ]
        if len(endpoint) != 1 or len(reference) != 1:
            raise ValueError("could not resolve one endpoint and averaged-rate row")
        actual = float(endpoint["cumulative_rate_mean_per_myr"].iloc[0])
        expected = float(reference["rate_imri_per_myr"].iloc[0])
        if not np.isclose(actual, expected, rtol=5.0e-6, atol=5.0e-12):
            raise ValueError(
                "candidate time-curve endpoint disagrees with averaged rate: "
                f"disk_model={disk_model}, SMBH={m_smbh_msun}, "
                f"NSC={m_nsc_msun}, "
                f"curve={actual}, averaged={expected}"
            )


def save_plot_data(frame: pd.DataFrame, path: str | Path) -> Path:
    """Persist compact, human-readable values supplied to the presentation layer."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False, float_format="%.10g")
    return output_path


def _sorted_unique_values(values: Sequence[float]) -> list[float]:
    return sorted({float(value) for value in values})


def _resolve_nsc_mass(frame: pd.DataFrame, selected: float | None) -> float:
    masses = sorted({float(value) for value in frame["m_nsc_msun"].unique()})
    if selected is None:
        if len(masses) != 1:
            raise ValueError(
                "select one NSC mass for scholar-facing figures; "
                f"available values are {masses}"
            )
        return masses[0]
    matches = [mass for mass in masses if np.isclose(mass, selected)]
    if len(matches) != 1:
        raise ValueError(f"selected NSC mass {selected} is not available: {masses}")
    return matches[0]


def _single_disk_model(frame: pd.DataFrame) -> str:
    _require_columns(frame, ["disk_model"], "plot data")
    disk_models = frame["disk_model"].dropna().astype(str).unique()
    if len(disk_models) != 1:
        raise ValueError(
            "one figure set requires exactly one disk model; "
            f"found {sorted(disk_models)}"
        )
    return str(disk_models[0])


def _disk_model_label(disk_model: str) -> str:
    labels = {
        "torque_with_trap": "torque with traps",
        "semi_without_trap": "semi-analytic without traps",
    }
    return labels.get(disk_model, disk_model.replace("_", " "))


def select_time_series_data(
    time_data: pd.DataFrame,
    selected_nsc_mass_msun: float | None = None,
) -> pd.DataFrame:
    """Select one NSC mass while retaining every configured SMBH curve."""
    _require_columns(time_data, TIME_SERIES_COLUMNS, "time-series results")
    _single_disk_model(time_data)
    nsc_mass = _resolve_nsc_mass(time_data, selected_nsc_mass_msun)
    nsc_data = time_data[np.isclose(time_data["m_nsc_msun"], nsc_mass)]
    masses = _sorted_unique_values(nsc_data["m_smbh_msun"].unique())
    selected = nsc_data[nsc_data["m_smbh_msun"].isin(masses)].copy()
    return selected.sort_values(["m_smbh_msun", "time_myr"]).reset_index(drop=True)


def _mass_label(mass_msun: float) -> str:
    exponent = int(np.floor(np.log10(mass_msun)))
    coefficient = mass_msun / 10.0**exponent
    value = (
        rf"10^{{{exponent}}}"
        if np.isclose(coefficient, 1.0)
        else rf"{coefficient:.2g}\times10^{{{exponent}}}"
    )
    return rf"${value}\,M_\odot$"


def _caption_mass_label(mass_msun: float) -> str:
    return _mass_label(mass_msun).replace(chr(92) * 2, chr(92))


def _color_map(values: Sequence[float]) -> dict[float, Any]:
    selected = _sorted_unique_values(values)
    if len(selected) <= len(OKABE_ITO_COLORS):
        colors: Sequence[Any] = OKABE_ITO_COLORS[: len(selected)]
    else:
        color_map = plt.get_cmap("tab20")
        colors = [
            color_map(index / max(1, len(selected) - 1))
            for index in range(len(selected))
        ]
    return dict(zip(selected, colors, strict=True))


def _new_figure(title: str) -> tuple[Any, Any]:
    fig, axis = plt.subplots(figsize=FIGURE_SIZE)
    axis.set_title(title, fontsize=13, pad=10)
    # axis.spines["top"].set_visible(False)
    # axis.spines["right"].set_visible(False)
    axis.yaxis.grid(True, color="0.88", linewidth=0.8)
    axis.xaxis.grid(False)
    axis.tick_params(labelsize=11)
    return fig, axis


def _save_figure(fig: Any, output_stem: Path) -> list[Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf")]
    for path in paths:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return paths


def _plot_status_fractions(
    status_data: pd.DataFrame,
    output_stem: str | Path,
    selected_nsc_mass_msun: float | None = None,
    scan_column: str = "redshift",
) -> list[Path]:
    validate_status_fraction_data(status_data, scan_column)
    nsc_mass = _resolve_nsc_mass(status_data, selected_nsc_mass_msun)
    selected = status_data[np.isclose(status_data["m_nsc_msun"], nsc_mass)]
    available_models = set(selected["disk_model"].astype(str))
    disk_models = [model for model in DISK_MODELS if model in available_models]
    disk_models.extend(sorted(available_models.difference(disk_models)))
    if not disk_models:
        raise ValueError("status fraction figure has no disk models")

    fig, axes = plt.subplots(
        len(STATUS_VARIABLE_ORDER),
        len(disk_models),
        figsize=(5.6 * len(disk_models), 12.4),
        sharey="row",
        squeeze=False,
    )
    for column_index, disk_model in enumerate(disk_models):
        for row_index, variable in enumerate(STATUS_VARIABLE_ORDER):
            axis = axes[row_index, column_index]
            panel = selected[
                selected["disk_model"].astype(str).eq(disk_model)
                & selected["variable"].eq(variable)
            ]
            scan_values = sorted({float(value) for value in panel[scan_column]})
            if scan_column == "redshift":
                x_values = np.asarray(scan_values, dtype=float)
                if x_values.size > 1:
                    bar_width = 0.65 * float(np.min(np.diff(x_values)))
                else:
                    bar_width = 0.6
                tick_labels = [f"{value:g}" for value in scan_values]
            else:
                x_values = np.arange(len(scan_values), dtype=float)
                bar_width = 0.72
                tick_labels = [_mass_label(value) for value in scan_values]
            bottom = np.zeros(x_values.shape, dtype=float)
            for category in STATUS_CATEGORY_ORDER[variable]:
                fractions = []
                for scan_value in scan_values:
                    row = panel[
                        np.isclose(panel[scan_column], scan_value)
                        & panel["category"].astype(str).eq(category)
                        & panel["result_state"].eq("valid")
                    ]
                    fractions.append(
                        float(row["fraction"].iloc[0]) if len(row) == 1 else 0.0
                    )
                values = np.asarray(fractions, dtype=float)
                axis.bar(
                    x_values,
                    values,
                    width=bar_width,
                    bottom=bottom,
                    color=STATUS_CATEGORY_COLORS[variable][category],
                    edgecolor="white",
                    linewidth=0.5,
                    label=STATUS_CATEGORY_LABELS[variable][category],
                )
                bottom += values

            unavailable_values = sorted(
                {
                    float(value)
                    for value in panel.loc[
                        panel["result_state"].eq("unavailable"), scan_column
                    ]
                }
            )
            if unavailable_values:
                unavailable_positions = (
                    unavailable_values
                    if scan_column == "redshift"
                    else [scan_values.index(value) for value in unavailable_values]
                )
                axis.scatter(
                    unavailable_positions,
                    np.full(len(unavailable_values), 0.5),
                    marker="x",
                    color="black",
                    linewidths=1.7,
                    s=55,
                    label="Unavailable bin",
                    zorder=5,
                )
            axis.set_ylim(0.0, 1.0)
            axis.set_xticks(x_values)
            axis.set_xticklabels(
                tick_labels,
                rotation=20 if scan_column == "m_smbh_msun" else 0,
                ha="right" if scan_column == "m_smbh_msun" else "center",
            )
            axis.yaxis.grid(True, color="0.9", linewidth=0.8)
            axis.set_axisbelow(True)
            if row_index == 0:
                axis.set_title(_disk_model_label(disk_model), fontsize=12)
            if column_index == 0:
                axis.set_ylabel(STATUS_VARIABLE_LABELS[variable])
            if row_index == len(STATUS_VARIABLE_ORDER) - 1:
                axis.set_xlabel(
                    "Redshift $z$" if scan_column == "redshift" else "SMBH mass"
                )
            if column_index == len(disk_models) - 1:
                axis.legend(
                    frameon=False,
                    fontsize=8,
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                )

    dimension_label = "redshift" if scan_column == "redshift" else "SMBH mass"
    fig.suptitle(
        f"IMRI classification fractions across {dimension_label}\n"
        + rf"NSC mass $={_mass_label(nsc_mass).strip('$')}$",
        fontsize=13,
    )
    aggregation_note = (
        "Each redshift bar aggregates configured SMBH masses."
        if scan_column == "redshift"
        else "Each SMBH-mass bar aggregates requested redshifts."
    )
    fig.text(
        0.01,
        0.01,
        "Denominators: geometry = attempted band-entry classifications; transport = "
        "persisted transport outcomes, including invalid states but excluding "
        "inside-ISCO geometry and blank/null reasons; event and detectability = all "
        "persisted event rows.\n"
        + aggregation_note
        + " Status fractions are diagnostic and may include outcomes from repeats "
        "excluded from rate estimation.",
        fontsize=8.0,
        color="0.3",
    )
    fig.tight_layout(rect=(0.0, 0.065, 1.0, 0.95))
    return _save_figure(fig, Path(output_stem))


def plot_status_fractions(
    status_data: pd.DataFrame,
    output_stem: str | Path,
    selected_nsc_mass_msun: float | None = None,
) -> list[Path]:
    """Plot 100%-stacked status fractions by redshift and disk model."""
    return _plot_status_fractions(
        status_data,
        output_stem,
        selected_nsc_mass_msun,
        "redshift",
    )


def plot_status_fractions_vs_smbh_mass(
    status_data: pd.DataFrame,
    output_stem: str | Path,
    selected_nsc_mass_msun: float | None = None,
) -> list[Path]:
    """Plot 100%-stacked status fractions by SMBH mass and disk model."""
    return _plot_status_fractions(
        status_data,
        output_stem,
        selected_nsc_mass_msun,
        "m_smbh_msun",
    )


def plot_agn_time(
    time_data: pd.DataFrame,
    output_stem: str | Path,
    rate_ylim: tuple[float, float] | None = None,
) -> list[Path]:
    """Plot one scholar-facing cumulative intrinsic-rate panel."""
    selected = select_time_series_data(time_data)
    disk_model = _single_disk_model(selected)
    redshifts = sorted(selected["redshift"].unique())
    if len(redshifts) != 1:
        raise ValueError("AGN-time figure requires one scalar-redshift run")

    fig, axis = _new_figure(f"IMRI formation: {_disk_model_label(disk_model)}")
    colors = _color_map(selected["m_smbh_msun"].unique())
    unavailable: list[str] = []
    for mass, color in colors.items():
        curve = selected[np.isclose(selected["m_smbh_msun"], mass)].sort_values(
            "time_myr"
        )
        y = curve["cumulative_rate_mean_per_myr"].to_numpy(dtype=float)
        if not np.isfinite(y).any():
            unavailable.append(_mass_label(mass))
            continue
        x = curve["time_myr"].to_numpy(dtype=float)
        axis.step(
            x,
            y,
            where="post",
            color=color,
            linewidth=2.1,
            marker="o",
            markersize=4.5,
            label=_mass_label(mass),
            clip_on=False,
        )
        valid_band = (
            curve["n_repeats_valid"].to_numpy(dtype=int) >= MIN_REPEATS_FOR_BAND
        )
        if valid_band.any():
            axis.fill_between(
                x,
                curve["cumulative_rate_p16_per_myr"].to_numpy(dtype=float),
                curve["cumulative_rate_p84_per_myr"].to_numpy(dtype=float),
                where=valid_band,
                step="post",
                color=color,
                alpha=0.14,
            )

    duty_cycle = float(selected["time_myr"].max())
    axis.set_xlim(0.0, duty_cycle)
    # axis.set_ylim(bottom=0.0)
    axis.set_xlabel("AGN time [Myr]", fontsize=12)
    axis.set_ylabel("Cumulative IMRI rate [Myr$^{-1}$]", fontsize=12)
    if rate_ylim is not None:
        axis.set_ylim(*rate_ylim)
    # axis.set_yscale("log")
    if axis.lines:
        axis.legend(frameon=False, fontsize=10, loc="best")
    if unavailable:
        axis.text(
            0.98,
            0.04,
            "Unavailable: " + ", ".join(unavailable),
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="0.35",
        )
    fig.tight_layout()
    return _save_figure(fig, Path(output_stem))


def _plot_rate_dependence(
    summary_data: pd.DataFrame,
    output_stem: Path,
    *,
    x_column: str,
    curve_column: str,
    curve_label: Any,
    title: str,
    x_label: str,
    log_x: bool,
    selected_nsc_mass_msun: float | None,
    rate_ylim: tuple[float, float] | None,
) -> list[Path]:
    _require_columns(summary_data, RATE_SUMMARY_COLUMNS, "rate summary")
    disk_model = _single_disk_model(summary_data)
    nsc_mass = _resolve_nsc_mass(summary_data, selected_nsc_mass_msun)
    nsc_data = summary_data[np.isclose(summary_data["m_nsc_msun"], nsc_mass)]
    colors = _color_map(nsc_data[curve_column].unique())

    fig, axis = _new_figure(f"{title}: {_disk_model_label(disk_model)}")
    has_unavailable = False
    has_partial = False
    unavailable_positions: list[float] = []
    unavailable_curves: list[str] = []
    line_styles = ("-", "--", ":", "-.")
    for curve_index, (curve_value, color) in enumerate(colors.items()):
        curve = nsc_data[np.isclose(nsc_data[curve_column], curve_value)].sort_values(
            x_column
        )
        x = curve[x_column].to_numpy(dtype=float)
        y = curve["imri_rate_mean_per_myr"].to_numpy(dtype=float)
        finite = np.isfinite(y)
        if finite.any():
            axis.plot(
                x,
                y,
                color=color,
                linestyle=line_styles[curve_index % len(line_styles)],
                linewidth=2.1,
                marker="o",
                markersize=6.0,
                label=curve_label(float(curve_value)),
                clip_on=False,
            )
        else:
            unavailable_curves.append(curve_label(float(curve_value)))

        valid_band = (
            curve["n_repeats_valid"].to_numpy(dtype=int) >= MIN_REPEATS_FOR_BAND
        ) & np.isfinite(y)
        if valid_band.any():
            axis.fill_between(
                x,
                curve["imri_rate_p16_per_myr"].to_numpy(dtype=float),
                curve["imri_rate_p84_per_myr"].to_numpy(dtype=float),
                where=valid_band,
                color=color,
                alpha=0.14,
            )

        partial = curve["result_state"].eq("partial").to_numpy()
        if partial.any():
            has_partial = True
            axis.scatter(
                x[partial],
                y[partial],
                facecolors="white",
                edgecolors=color,
                linewidths=1.5,
                s=48,
                zorder=4,
                clip_on=False,
            )
        unavailable = curve["result_state"].eq("unavailable").to_numpy()
        if unavailable.any():
            has_unavailable = True
            unavailable_positions.extend(float(value) for value in x[unavailable])

    if unavailable_positions:
        positions = sorted(set(unavailable_positions))
        axis.scatter(
            positions,
            np.zeros(len(positions)),
            marker="x",
            color="0.35",
            linewidths=1.7,
            s=55,
            zorder=4,
            clip_on=False,
        )

    axis.set_xlabel(x_label, fontsize=12)
    axis.set_ylabel("IMRI rate [Myr$^{-1}$]", fontsize=12)
    if rate_ylim is not None:
        axis.set_ylim(*rate_ylim)
    # axis.set_ylim(bottom=0.0)
    if log_x:
        axis.set_xscale("log")
    # axis.set_yscale("log")
    axis.legend(frameon=False, fontsize=10, loc="best")
    notes: list[str] = []
    if has_partial:
        notes.append("open marker: partial repeats")
    if has_unavailable:
        unavailable_note = "x: unavailable"
        if unavailable_curves:
            unavailable_note += " (" + ", ".join(unavailable_curves) + ")"
        notes.append(unavailable_note)
    if notes:
        axis.text(
            0.98,
            0.04,
            "; ".join(notes),
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="0.35",
        )
    fig.tight_layout()
    return _save_figure(fig, output_stem)


def plot_rate_vs_redshift(
    summary_data: pd.DataFrame,
    output_stem: str | Path,
    selected_nsc_mass_msun: float | None = None,
    rate_ylim: tuple[float, float] | None = None,
) -> list[Path]:
    """Plot one intrinsic-rate panel versus redshift."""
    return _plot_rate_dependence(
        summary_data,
        Path(output_stem),
        x_column="redshift",
        curve_column="m_smbh_msun",
        curve_label=_mass_label,
        title="IMRI rate versus redshift",
        x_label="Redshift $z$",
        log_x=False,
        selected_nsc_mass_msun=selected_nsc_mass_msun,
        rate_ylim=rate_ylim,
    )


def plot_rate_vs_smbh_mass(
    summary_data: pd.DataFrame,
    output_stem: str | Path,
    selected_nsc_mass_msun: float | None = None,
    rate_ylim: tuple[float, float] | None = None,
) -> list[Path]:
    """Plot one intrinsic-rate panel versus SMBH mass."""
    return _plot_rate_dependence(
        summary_data,
        Path(output_stem),
        x_column="m_smbh_msun",
        curve_column="redshift",
        curve_label=lambda redshift: f"$z={redshift:g}$",
        title="IMRI rate versus SMBH mass",
        x_label=r"SMBH mass [$M_\odot$]",
        log_x=True,
        selected_nsc_mass_msun=selected_nsc_mass_msun,
        rate_ylim=rate_ylim,
    )


def write_figure_captions(
    summary_data: pd.DataFrame,
    output_path: str | Path,
    *,
    time_redshift: float | None,
    include_redshift: bool,
    selected_nsc_mass_msun: float | None = None,
) -> Path:
    """Write concise captions for the figures produced from this dataset."""
    disk_model = _single_disk_model(summary_data)
    scenario_text = _disk_model_label(disk_model)
    nsc_mass = _resolve_nsc_mass(summary_data, selected_nsc_mass_msun)
    selected = summary_data[np.isclose(summary_data["m_nsc_msun"], nsc_mass)]
    duties = sorted({float(value) for value in selected["duty_cycle_myr"].unique()})
    if len(duties) != 1:
        raise ValueError("figure captions require one AGN duty-cycle duration")
    repeat_counts = sorted(
        {int(value) for value in selected["n_repeats_total"].unique()}
    )
    repeat_text = (
        str(repeat_counts[0])
        if len(repeat_counts) == 1
        else f"{repeat_counts[0]}--{repeat_counts[-1]}"
    )
    band_text = (
        "Shading marks the 16th--84th percentile where at least "
        f"{MIN_REPEATS_FOR_BAND} valid repeats are available."
    )
    if selected["n_repeats_valid"].max() < MIN_REPEATS_FOR_BAND:
        band_text = (
            "This is an exploratory result with fewer than "
            f"{MIN_REPEATS_FOR_BAND} valid repeats; no uncertainty band is shown."
        )

    captions = ["# Figure captions", "", f"Migration scenario: {scenario_text}.", ""]
    if time_redshift is not None:
        captions.extend(
            [
                "## IMRI production through one AGN episode",
                "",
                (
                    "Cumulative intrinsic IMRI candidate rate under the "
                    f"{scenario_text} migration scenario during a "
                    f"{duties[0]:g} Myr AGN episode at redshift "
                    f"$z={time_redshift:g}$ for an NSC mass of "
                    f"${_caption_mass_label(nsc_mass).strip('$')}$. Curves distinguish "
                    f"SMBH mass and summarize {repeat_text} Monte Carlo repeats. "
                    f"{band_text}"
                ),
                "",
            ]
        )
    if include_redshift:
        captions.extend(
            [
                "## IMRI rate versus redshift",
                "",
                (
                    "Final intrinsic IMRI candidate rate under the "
                    f"{scenario_text} migration scenario as a function of redshift "
                    f"for an NSC mass of ${_caption_mass_label(nsc_mass).strip('$')}$. "
                    f"Colors distinguish SMBH mass; each bin uses {repeat_text} "
                    f"Monte Carlo repeats and a {duties[0]:g} Myr rate denominator. "
                    f"{band_text} A cross marks an unavailable result, not zero."
                ),
                "",
            ]
        )
    captions.extend(
        [
            "## IMRI rate versus SMBH mass",
            "",
            (
                "Final intrinsic IMRI candidate rate under the "
                f"{scenario_text} migration scenario as a function of SMBH mass "
                f"for an NSC mass of ${_caption_mass_label(nsc_mass).strip('$')}$. "
                f"Colors distinguish redshift; each bin uses {repeat_text} Monte "
                f"Carlo repeats and a {duties[0]:g} Myr rate denominator. "
                f"{band_text} A cross marks an unavailable result, not zero."
            ),
            "",
            (
                "The proxy-detectable rate is retained in `imri_rate_summary.csv` "
                "as secondary information and is not a cosmological detector rate."
            ),
            "",
        ]
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(captions))
    return path


def generate_run_figures(
    results_dir: str | Path,
    output_dir: str | Path | None = None,
    selected_nsc_mass_msun: float | None = None,
) -> dict[str, list[Path] | Path]:
    """Generate the time and mass figures supported by one scalar-redshift run."""
    run = load_run_results(results_dir)
    figure_dir = (
        Path(output_dir) if output_dir is not None else Path(results_dir) / "figures"
    )
    data_dir = figure_dir / "data"

    time_data = build_agn_time_plot_data(run.events, run.repeats)
    validate_agn_time_endpoints(time_data, run.repeats, run.averaged)
    selected_time = select_time_series_data(time_data, selected_nsc_mass_msun)
    summary_data = build_rate_summary_data(run.repeats)
    time_path = save_plot_data(selected_time, data_dir / "imri_rate_time_series.csv")
    summary_path = save_plot_data(summary_data, data_dir / "imri_rate_summary.csv")
    redshift = float(run.repeats["redshift"].iloc[0])
    captions = write_figure_captions(
        summary_data,
        figure_dir / "figure_captions.md",
        time_redshift=redshift,
        include_redshift=False,
        selected_nsc_mass_msun=selected_nsc_mass_msun,
    )
    return {
        "time_series_data": time_path,
        "rate_summary_data": summary_path,
        "agn_time_figures": plot_agn_time(
            selected_time, figure_dir / "imri_rate_vs_agn_time"
        ),
        "smbh_mass_figures": plot_rate_vs_smbh_mass(
            summary_data,
            figure_dir / "imri_rate_vs_smbh_mass",
            selected_nsc_mass_msun,
        ),
        "figure_captions": captions,
    }


def generate_scan_figures(
    repeats: pd.DataFrame,
    output_dir: str | Path,
    reference_run: RunResults,
    selected_nsc_mass_msun: float | None = None,
    rate_ylim: tuple[float, float] | None = None,
) -> dict[str, list[Path] | Path]:
    """Generate one coherent three-figure set for one model's redshift scan."""
    figure_dir = Path(output_dir)
    disk_model = _single_disk_model(repeats)
    reference_disk_model = _single_disk_model(reference_run.repeats)
    if disk_model != reference_disk_model:
        raise ValueError("scan repeats and reference run use different disk models")
    data_dir = figure_dir / "data"
    summary_data = build_rate_summary_data(repeats)
    time_data = build_agn_time_plot_data(reference_run.events, reference_run.repeats)
    validate_agn_time_endpoints(
        time_data, reference_run.repeats, reference_run.averaged
    )
    selected_time = select_time_series_data(time_data, selected_nsc_mass_msun)
    time_path = save_plot_data(selected_time, data_dir / "imri_rate_time_series.csv")
    summary_path = save_plot_data(summary_data, data_dir / "imri_rate_summary.csv")
    time_redshift = float(reference_run.repeats["redshift"].iloc[0])
    captions = write_figure_captions(
        summary_data,
        figure_dir / "figure_captions.md",
        time_redshift=time_redshift,
        include_redshift=True,
        selected_nsc_mass_msun=selected_nsc_mass_msun,
    )
    return {
        "time_series_data": time_path,
        "rate_summary_data": summary_path,
        "agn_time_figures": plot_agn_time(
            selected_time, figure_dir / "imri_rate_vs_agn_time", rate_ylim
        ),
        "redshift_figures": plot_rate_vs_redshift(
            summary_data,
            figure_dir / "imri_rate_vs_redshift",
            selected_nsc_mass_msun,
            rate_ylim,
        ),
        "smbh_mass_figures": plot_rate_vs_smbh_mass(
            summary_data,
            figure_dir / "imri_rate_vs_smbh_mass",
            selected_nsc_mass_msun,
            rate_ylim,
        ),
        "figure_captions": captions,
    }
