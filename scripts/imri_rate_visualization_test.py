# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportGeneralTypeIssues=false
"""Deterministic numerical and presentation harness for IMRI rate figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence
from unittest.mock import patch

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
for path in (
    SOURCE_ROOT / "physical_process",
    SOURCE_ROOT / "disk_quantity",
    SOURCE_ROOT,
    PROJECT_ROOT,
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.workflows.rate_plots import (
    BAND_ENTRY_ORDER,
    GROUP_COLUMNS,
    RATE_SUMMARY_COLUMNS,
    STATUS_CATEGORY_COLORS,
    STATUS_CATEGORY_LABELS,
    STATUS_CATEGORY_ORDER,
    STATUS_FRACTION_COLUMNS,
    STATUS_FRACTION_SMBH_MASS_COLUMNS,
    STATUS_VARIABLE_LABELS,
    STATUS_VARIABLE_ORDER,
    TIME_SERIES_COLUMNS,
    TRANSPORT_ORDER,
    RunResults,
    build_agn_time_plot_data,
    build_rate_summary_data,
    build_status_fraction_by_smbh_mass_data,
    build_status_fraction_data,
    generate_scan_figures,
    plot_agn_time,
    plot_rate_vs_redshift,
    plot_status_fractions,
    plot_status_fractions_vs_smbh_mass,
    save_plot_data,
    select_time_series_data,
    shared_rate_axis_limits,
    validate_agn_time_endpoints,
    validate_event_keys,
    validate_repeat_results,
    validate_status_fraction_data,
)
from src.workflows.config import SEMI_WITHOUT_TRAP, TORQUE_WITH_TRAP
from src.workflows.workflow import derive_task_seed

T_AGN_MYR = 10.0
BASE_SEED = 12345
EXPECTED_TASK_SEEDS = (2688385916, 3546910048)
REDSHIFTS = (10, 12, 15)
SMBH_MASSES_MSUN = (1.0e3, 1.0e4, 1.0e5, 1.0e6)
NSC_MASS_MSUN = 2.0e5
REPEATS = (0, 1)
EXPECTED_TIME_GRID_MYR = (0.0, 2.0, 8.0, T_AGN_MYR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate deterministic IMRI rate aggregation and figures."
    )
    parser.add_argument("--output-dir", type=Path, default="figs/")
    return parser


def _repeat_record(
    *,
    redshift: float,
    mass: float,
    repeat: int,
    status: str,
    n_imri: int,
    n_det: int,
    disk_model: str = TORQUE_WITH_TRAP,
) -> dict[str, Any]:
    mass_index = SMBH_MASSES_MSUN.index(mass)
    complete = status == "complete"
    is_inside_isco = status == "not_evaluated_band_entry_inside_isco"
    is_profile_error = status == "incomplete_profile_error"
    return {
        "disk_model": disk_model,
        "redshift": redshift,
        "m_smbh_msun": mass,
        "m_nsc_msun": NSC_MASS_MSUN,
        "repeat": repeat,
        "base_seed": BASE_SEED,
        "task_seed": derive_task_seed(BASE_SEED, mass_index, 0, repeat),
        "rate_status": status,
        "duty_cycle_myr": T_AGN_MYR,
        "n_sampled": 100 + mass_index,
        "n_imri": n_imri,
        "rate_imri_per_myr": n_imri / T_AGN_MYR if complete else np.nan,
        "n_det": n_det,
        "rate_det_per_myr": n_det / T_AGN_MYR if complete else np.nan,
        "n_band_eval": 1,
        "n_entry_in_domain": int(not is_inside_isco and not is_profile_error),
        "n_entry_handoff": 0,
        "n_entry_inside_isco": int(is_inside_isco),
        "n_profile_error": int(is_profile_error),
    }


def _event_record(
    *,
    redshift: float,
    mass: float,
    repeat: int,
    object_index: int,
    gas_final_time_myr: float,
    detection_time_myr: float,
    is_imri_candidate: bool | None,
    migration_status: str,
    disk_model: str = TORQUE_WITH_TRAP,
) -> dict[str, Any]:
    gas_stop_reason = {
        "finite": "entry_reached",
        "trap_bound": "migration_trap_bound",
        "not_evaluated": "band_entry_inside_isco",
        "profile_error": "",
    }[migration_status]
    band_entry_status = {
        "not_evaluated": "band_entry_inside_isco",
        "profile_error": "profile_error",
    }.get(migration_status, "entry_in_migration_domain")
    event_rate_class = {
        True: "imri candi",
        False: "no-IMRI",
        None: "not evaluated",
    }[is_imri_candidate]
    is_lisa_detectable = (
        None
        if is_imri_candidate is None
        else bool(is_imri_candidate and np.isfinite(detection_time_myr))
    )
    detect_rate_class = {
        True: "imri data",
        False: "non-detectable",
        None: "not evaluated",
    }[is_lisa_detectable]
    return {
        "disk_model": disk_model,
        "redshift": redshift,
        "m_smbh_msun": mass,
        "m_nsc_msun": NSC_MASS_MSUN,
        "repeat": repeat,
        "object_index": object_index,
        "gas_final_time_myr": gas_final_time_myr,
        "detection_time_myr": detection_time_myr,
        "is_imri_candidate": is_imri_candidate,
        "migration_status": migration_status,
        "band_entry_status": band_entry_status,
        "gas_stop_reason": gas_stop_reason,
        "event_rate_class": event_rate_class,
        "detect_rate_class": detect_rate_class,
        "is_lisa_detectable": is_lisa_detectable,
    }


def synthetic_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return paired complete, valid-zero, trap, and unavailable cases."""
    repeat_records: list[dict[str, Any]] = []
    event_records: list[dict[str, Any]] = []
    for redshift in REDSHIFTS:
        detectable_count = 2 if redshift == REDSHIFTS[0] else 1
        repeat_records.extend(
            [
                _repeat_record(
                    redshift=redshift,
                    mass=1.0e3,
                    repeat=0,
                    status="complete",
                    n_imri=2,
                    n_det=detectable_count,
                ),
                _repeat_record(
                    redshift=redshift,
                    mass=1.0e3,
                    repeat=1,
                    status="complete",
                    n_imri=0,
                    n_det=0,
                ),
            ]
        )
        event_records.extend(
            [
                _event_record(
                    redshift=redshift,
                    mass=1.0e3,
                    repeat=0,
                    object_index=0,
                    gas_final_time_myr=2.0,
                    detection_time_myr=5.0,
                    is_imri_candidate=True,
                    migration_status="finite",
                ),
                _event_record(
                    redshift=redshift,
                    mass=1.0e3,
                    repeat=0,
                    object_index=1,
                    gas_final_time_myr=8.0,
                    detection_time_myr=12.0,
                    is_imri_candidate=True,
                    migration_status="finite",
                ),
            ]
        )

        for repeat in REPEATS:
            repeat_records.append(
                _repeat_record(
                    redshift=redshift,
                    mass=1.0e4,
                    repeat=repeat,
                    status="complete",
                    n_imri=0,
                    n_det=0,
                )
            )
            event_records.append(
                _event_record(
                    redshift=redshift,
                    mass=1.0e4,
                    repeat=repeat,
                    object_index=repeat,
                    gas_final_time_myr=T_AGN_MYR,
                    detection_time_myr=np.inf,
                    is_imri_candidate=False,
                    migration_status="trap_bound",
                )
            )

            repeat_records.append(
                _repeat_record(
                    redshift=redshift,
                    mass=1.0e5,
                    repeat=repeat,
                    status="not_evaluated_band_entry_inside_isco",
                    n_imri=0,
                    n_det=0,
                )
            )
            event_records.append(
                _event_record(
                    redshift=redshift,
                    mass=1.0e5,
                    repeat=repeat,
                    object_index=repeat,
                    gas_final_time_myr=np.nan,
                    detection_time_myr=np.inf,
                    is_imri_candidate=None,
                    migration_status="not_evaluated",
                )
            )

            repeat_records.append(
                _repeat_record(
                    redshift=redshift,
                    mass=1.0e6,
                    repeat=repeat,
                    status="incomplete_profile_error",
                    n_imri=0,
                    n_det=0,
                )
            )
            event_records.append(
                _event_record(
                    redshift=redshift,
                    mass=1.0e6,
                    repeat=repeat,
                    object_index=repeat,
                    gas_final_time_myr=np.nan,
                    detection_time_myr=np.inf,
                    is_imri_candidate=None,
                    migration_status="profile_error",
                )
            )

    torque_repeats = pd.DataFrame.from_records(repeat_records)
    torque_events = pd.DataFrame.from_records(event_records)
    semi_repeats = torque_repeats.copy()
    semi_repeats["disk_model"] = SEMI_WITHOUT_TRAP
    semi_changed = np.isclose(semi_repeats["m_smbh_msun"], 1.0e3) & semi_repeats[
        "repeat"
    ].eq(0)
    semi_repeats.loc[semi_changed, "n_imri"] = 1
    semi_repeats.loc[semi_changed, "rate_imri_per_myr"] = 1.0 / T_AGN_MYR
    semi_repeats.loc[semi_changed, "n_det"] = 1
    semi_repeats.loc[semi_changed, "rate_det_per_myr"] = 1.0 / T_AGN_MYR
    semi_handoff = np.isclose(semi_repeats["m_smbh_msun"], 1.0e4)
    semi_repeats.loc[semi_handoff, "n_entry_in_domain"] = 0
    semi_repeats.loc[semi_handoff, "n_entry_handoff"] = 1

    semi_events = torque_events.copy()
    semi_events["disk_model"] = SEMI_WITHOUT_TRAP
    semi_second_candidate = np.isclose(semi_events["m_smbh_msun"], 1.0e3) & semi_events[
        "object_index"
    ].eq(1)
    semi_events.loc[semi_second_candidate, "is_imri_candidate"] = False
    semi_events.loc[semi_second_candidate, "event_rate_class"] = "no-IMRI"
    semi_events.loc[semi_second_candidate, "is_lisa_detectable"] = False
    semi_events.loc[semi_second_candidate, "detect_rate_class"] = "non-detectable"
    semi_events.loc[
        semi_events["migration_status"].eq("trap_bound"), "migration_status"
    ] = "finite"
    semi_events.loc[
        semi_events["gas_stop_reason"].eq("migration_trap_bound"), "gas_stop_reason"
    ] = "entry_reached"

    repeats = pd.concat([torque_repeats, semi_repeats], ignore_index=True)
    events = pd.concat([torque_events, semi_events], ignore_index=True)
    first_redshift = repeats[np.isclose(repeats["redshift"], REDSHIFTS[0])]
    averaged_records: list[dict[str, Any]] = []
    for (disk_model, mass), group in first_redshift.groupby(
        ["disk_model", "m_smbh_msun"], sort=True
    ):
        complete = group["rate_status"].eq("complete").all()
        averaged_records.append(
            {
                "disk_model": str(disk_model),
                "m_smbh_msun": float(mass),
                "m_nsc_msun": NSC_MASS_MSUN,
                "rate_status": (
                    "complete" if complete else str(group["rate_status"].iloc[0])
                ),
                "rate_imri_per_myr": (
                    float(group["rate_imri_per_myr"].mean()) if complete else np.nan
                ),
            }
        )
    return events, repeats, pd.DataFrame.from_records(averaged_records)


def _assert_monotone(time_data: pd.DataFrame) -> None:
    for _, curve in time_data.groupby(GROUP_COLUMNS):
        finite = (
            curve.sort_values("time_myr")["cumulative_rate_mean_per_myr"]
            .dropna()
            .to_numpy()
        )
        if finite.size and not np.all(np.diff(finite) >= -1.0e-14):
            raise AssertionError("cumulative rate curve is not monotone")


def _assert_round_trip(frame: pd.DataFrame, path: Path) -> None:
    loaded = pd.read_csv(path, float_precision="round_trip")
    pd.testing.assert_frame_equal(
        loaded,
        frame.reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        rtol=5.0e-10,
        atol=5.0e-12,
    )


def run_harness(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    events, repeats, averaged = synthetic_tables()
    events.to_csv(output_dir / "synthetic_events.csv", index=False)
    repeats.to_csv(output_dir / "synthetic_repeats.csv", index=False)

    validate_repeat_results(repeats)
    validate_event_keys(events, repeats)
    torque_mask = repeats["disk_model"].eq(TORQUE_WITH_TRAP)
    first_repeat_mask = torque_mask & np.isclose(repeats["redshift"], REDSHIFTS[0])
    first_repeats = repeats[first_repeat_mask]
    first_events = events[
        events["disk_model"].eq(TORQUE_WITH_TRAP)
        & np.isclose(events["redshift"], REDSHIFTS[0])
    ]
    first_averaged = averaged[averaged["disk_model"].eq(TORQUE_WITH_TRAP)]
    time_data = build_agn_time_plot_data(first_events, first_repeats)
    selected_time = select_time_series_data(time_data)
    summary_data = build_rate_summary_data(repeats)
    status_data = build_status_fraction_data(events, repeats)
    mass_status_data = build_status_fraction_by_smbh_mass_data(events, repeats)
    validate_status_fraction_data(status_data)
    validate_status_fraction_data(mass_status_data, "m_smbh_msun")
    validate_agn_time_endpoints(time_data, first_repeats, first_averaged)
    _assert_monotone(time_data)

    if list(selected_time.columns) != TIME_SERIES_COLUMNS:
        raise AssertionError("time-series table is not compact or ordered")
    if list(summary_data.columns) != RATE_SUMMARY_COLUMNS:
        raise AssertionError("rate-summary table is not compact or ordered")
    if list(status_data.columns) != STATUS_FRACTION_COLUMNS:
        raise AssertionError("status-fraction table is not compact or ordered")
    if list(mass_status_data.columns) != STATUS_FRACTION_SMBH_MASS_COLUMNS:
        raise AssertionError("SMBH-mass status table is not compact or ordered")
    for frame, scan_column in (
        (status_data, "redshift"),
        (mass_status_data, "m_smbh_msun"),
    ):
        for _, scan_group in frame.groupby(
            ["disk_model", scan_column, "m_nsc_msun"], sort=False
        ):
            if tuple(scan_group["variable"].drop_duplicates()) != STATUS_VARIABLE_ORDER:
                raise AssertionError("status variables do not use the scientific order")
            for variable in STATUS_VARIABLE_ORDER:
                categories = tuple(
                    scan_group.loc[
                        scan_group["variable"].eq(variable), "category"
                    ].astype(str)
                )
                if categories != STATUS_CATEGORY_ORDER[variable]:
                    raise AssertionError(
                        f"status categories are not stable for {variable}"
                    )
                if set(STATUS_CATEGORY_LABELS[variable]) != set(categories):
                    raise AssertionError(
                        f"presentation labels are incomplete: {variable}"
                    )
                if set(STATUS_CATEGORY_COLORS[variable]) != set(categories):
                    raise AssertionError(f"semantic colors are incomplete: {variable}")

    status_groups = status_data[status_data["result_state"].eq("valid")].groupby(
        ["disk_model", "redshift", "m_nsc_msun", "variable"]
    )["fraction"]
    if not np.allclose(status_groups.sum(), 1.0):
        raise AssertionError("status fractions do not conserve probability")
    mass_status_groups = mass_status_data[
        mass_status_data["result_state"].eq("valid")
    ].groupby(["disk_model", "m_smbh_msun", "m_nsc_msun", "variable"])["fraction"]
    if not np.allclose(mass_status_groups.sum(), 1.0):
        raise AssertionError("SMBH-mass status fractions do not conserve probability")
    if set(mass_status_data["m_smbh_msun"]) != set(SMBH_MASSES_MSUN):
        raise AssertionError("SMBH-mass status bins do not match the configured masses")
    semi_mass_traps = mass_status_data[
        mass_status_data["disk_model"].eq(SEMI_WITHOUT_TRAP)
        & mass_status_data["variable"].eq("transport_status")
        & mass_status_data["category"].eq("migration_trap_bound")
    ]
    valid_semi_mass_traps = semi_mass_traps[
        semi_mass_traps["result_state"].eq("valid")
    ]
    if (
        semi_mass_traps.empty
        or not np.allclose(semi_mass_traps["count"], 0.0)
        or not np.allclose(valid_semi_mass_traps["fraction"], 0.0)
    ):
        raise AssertionError("SMBH-mass status fractions pooled the disk models")
    unavailable_mass = mass_status_data[
        np.isclose(mass_status_data["m_smbh_msun"], 1.0e6)
        & mass_status_data["variable"].eq("transport_status")
    ]
    if not unavailable_mass["result_state"].eq("unavailable").all():
        raise AssertionError("unavailable SMBH-mass status bins became valid zeros")
    if unavailable_mass["fraction"].notna().any():
        raise AssertionError("unavailable SMBH-mass status fractions became zero")
    torque_status = status_data[
        status_data["disk_model"].eq(TORQUE_WITH_TRAP)
        & np.isclose(status_data["redshift"], REDSHIFTS[0])
    ]
    profile_fraction = torque_status[
        torque_status["variable"].eq("band_entry_status")
        & torque_status["category"].eq("profile_error")
    ]["fraction"].iloc[0]
    if not np.isclose(profile_fraction, 0.25):
        raise AssertionError("band-entry profile errors use the wrong denominator")
    torque_trap_fraction = torque_status[
        torque_status["variable"].eq("transport_status")
        & torque_status["category"].eq("migration_trap_bound")
    ]["fraction"].iloc[0]
    if not np.isclose(torque_trap_fraction, 0.5):
        raise AssertionError("transport outcomes use the wrong eligible denominator")
    torque_transport = torque_status[
        torque_status["variable"].eq("transport_status")
    ]
    if set(torque_transport["category"]) != set(TRANSPORT_ORDER):
        raise AssertionError("the transport taxonomy is incomplete")
    if torque_transport["category"].eq("band_entry_inside_isco").any():
        raise AssertionError("inside-ISCO geometry leaked into transport outcomes")
    if not np.allclose(torque_transport["denominator"], 4.0):
        raise AssertionError("inside-ISCO/blank rows entered the transport denominator")
    semi_trap_rows = status_data.loc[
        status_data["disk_model"].eq(SEMI_WITHOUT_TRAP)
        & status_data["variable"].eq("transport_status")
        & status_data["category"].eq("migration_trap_bound")
    ]
    if semi_trap_rows.empty or not np.allclose(semi_trap_rows["fraction"], 0.0):
        raise AssertionError("status fractions pooled the two disk models")

    for variable, category, expected_count in (
        ("event_outcome", "not evaluated", 4.0),
        ("detectability_outcome", "not evaluated", 4.0),
    ):
        rows = torque_status[
            torque_status["variable"].eq(variable)
            & torque_status["category"].eq(category)
        ]
        if len(rows) != 1 or not np.isclose(rows["count"].iloc[0], expected_count):
            raise AssertionError(f"{variable} did not preserve not evaluated")

    geometry_repeats = repeats.iloc[[0]].copy()
    geometry_repeats.loc[:, "n_band_eval"] = 7
    geometry_repeats.loc[:, "n_entry_in_domain"] = 2
    geometry_repeats.loc[:, "n_entry_handoff"] = 3
    geometry_repeats.loc[:, "n_entry_inside_isco"] = 1
    geometry_repeats.loc[:, "n_profile_error"] = 1
    geometry_status = build_status_fraction_data(
        events.iloc[0:0].copy(), geometry_repeats
    )
    geometry_rows = geometry_status[
        geometry_status["variable"].eq("band_entry_status")
    ]
    if tuple(geometry_rows["category"]) != BAND_ENTRY_ORDER:
        raise AssertionError("geometry categories changed order")
    if not np.allclose(geometry_rows["denominator"], 7.0):
        raise AssertionError("geometry denominator does not use n_band_eval")
    if not np.isclose(geometry_rows["fraction"].sum(), 1.0):
        raise AssertionError("geometry fractions do not conserve seven evaluations")

    invalid_transport_events = events.copy()
    invalid_transport_events.loc[
        invalid_transport_events.index[0], "gas_stop_reason"
    ] = "invalid_radius"
    invalid_transport = build_status_fraction_data(invalid_transport_events, repeats)
    invalid_rows = invalid_transport[
        invalid_transport["variable"].eq("transport_status")
        & invalid_transport["category"].eq("invalid_radius")
    ]
    if invalid_rows["count"].sum() != 1.0:
        raise AssertionError("invalid transport state was not kept explicit")

    unknown_events = events.copy()
    unknown_events.loc[unknown_events.index[0], "gas_stop_reason"] = "mystery_stop"
    try:
        build_status_fraction_data(unknown_events, repeats)
    except ValueError as exc:
        if "mystery_stop" not in str(exc):
            raise AssertionError(
                "unknown transport error omitted the raw label"
            ) from exc
    else:
        raise AssertionError("unknown transport label was silently accepted")

    redshift_geometry = torque_status[
        torque_status["variable"].eq("band_entry_status")
    ]
    if not np.allclose(redshift_geometry["denominator"], 8.0):
        raise AssertionError("redshift status builder did not aggregate SMBH masses")
    mass_geometry = mass_status_data[
        mass_status_data["disk_model"].eq(TORQUE_WITH_TRAP)
        & np.isclose(mass_status_data["m_smbh_msun"], 1.0e3)
        & mass_status_data["variable"].eq("band_entry_status")
    ]
    if not np.allclose(mass_geometry["denominator"], 6.0):
        raise AssertionError("SMBH-mass status builder did not aggregate redshifts")
    empty_repeats = repeats.iloc[[0]].copy()
    for column in (
        "n_band_eval",
        "n_entry_in_domain",
        "n_entry_handoff",
        "n_entry_inside_isco",
        "n_profile_error",
    ):
        empty_repeats[column] = 0
    empty_status = build_status_fraction_data(events.iloc[0:0].copy(), empty_repeats)
    if not empty_status["result_state"].eq("unavailable").all():
        raise AssertionError("zero status denominators were converted to valid zeros")
    if empty_status["fraction"].notna().any():
        raise AssertionError("unavailable status fractions were converted to zero")

    candidate = selected_time[
        np.isclose(selected_time["m_smbh_msun"], 1.0e3)
    ].sort_values("time_myr")
    if tuple(candidate["time_myr"]) != EXPECTED_TIME_GRID_MYR:
        raise AssertionError("candidate-production time grid changed")
    if (candidate["time_myr"] > T_AGN_MYR).any():
        raise AssertionError("later proxy-entry time leaked into the production clock")
    endpoint = candidate[np.isclose(candidate["time_myr"], T_AGN_MYR)].iloc[0]
    if not np.isclose(endpoint["cumulative_rate_mean_per_myr"], 0.1):
        raise AssertionError("zero-event repeat is missing from endpoint mean")
    if not np.isclose(endpoint["cumulative_rate_p16_per_myr"], 0.032):
        raise AssertionError("16th percentile is incorrect")
    if not np.isclose(endpoint["cumulative_rate_p84_per_myr"], 0.168):
        raise AssertionError("84th percentile is incorrect")

    valid_zero = summary_data[np.isclose(summary_data["m_smbh_msun"], 1.0e4)]
    if not valid_zero["result_state"].eq("valid").all():
        raise AssertionError("valid zero was not retained as valid")
    if not np.allclose(valid_zero["imri_rate_mean_per_myr"], 0.0):
        raise AssertionError("valid zero rate changed")
    unavailable = summary_data[summary_data["m_smbh_msun"].isin((1.0e5, 1.0e6))]
    if not unavailable["result_state"].eq("unavailable").all():
        raise AssertionError("unavailable states were conflated")
    if unavailable["imri_rate_mean_per_myr"].notna().any():
        raise AssertionError("unavailable rates were converted to zero")

    proxy_z1 = summary_data[
        summary_data["disk_model"].eq(TORQUE_WITH_TRAP)
        & np.isclose(summary_data["redshift"], REDSHIFTS[0])
        & np.isclose(summary_data["m_smbh_msun"], 1.0e3)
    ]["proxy_detectable_rate_mean_per_myr"].iloc[0]
    proxy_z2 = summary_data[
        summary_data["disk_model"].eq(TORQUE_WITH_TRAP)
        & np.isclose(summary_data["redshift"], REDSHIFTS[1])
        & np.isclose(summary_data["m_smbh_msun"], 1.0e3)
    ]["proxy_detectable_rate_mean_per_myr"].iloc[0]
    if not np.isclose(proxy_z1, 0.1) or not np.isclose(proxy_z2, 0.05):
        raise AssertionError("secondary proxy rate summary changed")

    actual_seeds = (
        derive_task_seed(BASE_SEED, 0, 0, 0),
        derive_task_seed(BASE_SEED, 0, 0, 1),
    )
    if actual_seeds != EXPECTED_TASK_SEEDS:
        raise AssertionError("stable task-seed derivation changed")
    paired = repeats.groupby(["m_smbh_msun", "m_nsc_msun", "repeat"])[
        ["task_seed", "n_sampled"]
    ].nunique()
    if not paired.eq(1).all().all():
        raise AssertionError("matching redshift/model tasks are not paired")

    paired_rates = summary_data[
        np.isclose(summary_data["redshift"], REDSHIFTS[0])
        & np.isclose(summary_data["m_smbh_msun"], 1.0e3)
    ].set_index("disk_model")["imri_rate_mean_per_myr"]
    if not np.isclose(paired_rates[TORQUE_WITH_TRAP], 0.1):
        raise AssertionError("torque scenario rate changed")
    if not np.isclose(paired_rates[SEMI_WITHOUT_TRAP], 0.05):
        raise AssertionError("disk models were pooled during aggregation")

    original_savefig = Figure.savefig
    figure_metrics: list[tuple[int, int, int, int, str, tuple[float, float]]] = []

    def capture_figure(fig: Figure, path: Any, *args: Any, **kwargs: Any) -> Any:
        if Path(path).suffix == ".png":
            axis = fig.axes[0]
            legend = axis.get_legend()
            legend_text = (
                " ".join(text.get_text() for text in legend.get_texts())
                if legend is not None
                else ""
            )
            figure_metrics.append(
                (
                    len(fig.axes),
                    len(axis.lines),
                    len(legend.get_texts()) if legend is not None else 0,
                    sum(
                        isinstance(collection, PolyCollection)
                        for collection in axis.collections
                    ),
                    legend_text,
                    tuple(float(value) for value in axis.get_ylim()),
                )
            )
        return original_savefig(fig, path, *args, **kwargs)

    outputs_by_model: dict[str, dict[str, Any]] = {}
    paired_rate_ylim = shared_rate_axis_limits(summary_data)
    with patch.object(Figure, "savefig", capture_figure):
        for disk_model in (TORQUE_WITH_TRAP, SEMI_WITHOUT_TRAP):
            model_repeats = repeats[repeats["disk_model"].eq(disk_model)]
            model_events = events[
                events["disk_model"].eq(disk_model)
                & np.isclose(events["redshift"], REDSHIFTS[0])
            ]
            model_first_repeats = model_repeats[
                np.isclose(model_repeats["redshift"], REDSHIFTS[0])
            ]
            model_averaged = averaged[averaged["disk_model"].eq(disk_model)]
            reference_run = RunResults(
                model_events, model_first_repeats, model_averaged, {}
            )
            outputs_by_model[disk_model] = generate_scan_figures(
                model_repeats,
                figures_dir / disk_model,
                reference_run,
                rate_ylim=paired_rate_ylim,
            )

    torque_outputs = outputs_by_model[TORQUE_WITH_TRAP]
    _assert_round_trip(selected_time, Path(torque_outputs["time_series_data"]))
    _assert_round_trip(
        summary_data[summary_data["disk_model"].eq(TORQUE_WITH_TRAP)].reset_index(
            drop=True
        ),
        Path(torque_outputs["rate_summary_data"]),
    )
    if len(figure_metrics) != 6:
        raise AssertionError("the paired output is not exactly six figures")
    for axes, curves, legend_entries, bands, legend_text, _ylim in figure_metrics:
        if axes != 1:
            raise AssertionError("a main figure contains more than one axis")
        if curves > max(len(SMBH_MASSES_MSUN), len(REDSHIFTS)):
            raise AssertionError("a main figure contains too many curves")
        if bands != 0:
            raise AssertionError("an uncertainty band was drawn with only two repeats")
        if "proxy" in legend_text.lower():
            raise AssertionError("proxy detectability leaked into a main figure")
    if len({metric[-1] for metric in figure_metrics}) != 1:
        raise AssertionError("paired scenario figures do not share rate-axis limits")

    five_masses = (1.0e3, 5.0e3, 1.0e4, 5.0e4, 1.0e5)
    five_mass_time_data = pd.DataFrame(
        [
            {
                "disk_model": TORQUE_WITH_TRAP,
                "redshift": REDSHIFTS[0],
                "m_smbh_msun": mass,
                "m_nsc_msun": NSC_MASS_MSUN,
                "time_myr": time_myr,
                "n_repeats_valid": 1,
                "cumulative_rate_mean_per_myr": mass_index * time_myr / T_AGN_MYR,
                "cumulative_rate_p16_per_myr": np.nan,
                "cumulative_rate_p84_per_myr": np.nan,
            }
            for mass_index, mass in enumerate(five_masses, start=1)
            for time_myr in (0.0, T_AGN_MYR)
        ],
        columns=TIME_SERIES_COLUMNS,
    )
    five_mass_rate_data = pd.DataFrame(
        [
            {
                "disk_model": TORQUE_WITH_TRAP,
                "redshift": redshift,
                "m_smbh_msun": mass,
                "m_nsc_msun": NSC_MASS_MSUN,
                "duty_cycle_myr": T_AGN_MYR,
                "n_repeats_total": 1,
                "n_repeats_valid": 1,
                "mean_imri_count": float(mass_index),
                "imri_rate_mean_per_myr": mass_index / T_AGN_MYR,
                "imri_rate_p16_per_myr": np.nan,
                "imri_rate_p84_per_myr": np.nan,
                "proxy_detectable_rate_mean_per_myr": 0.0,
                "result_state": "valid",
            }
            for mass_index, mass in enumerate(five_masses, start=1)
            for redshift in REDSHIFTS[:2]
        ],
        columns=RATE_SUMMARY_COLUMNS,
    )
    five_mass_metrics: list[tuple[int, int]] = []

    def capture_five_mass_figure(
        fig: Figure, path: Any, *args: Any, **kwargs: Any
    ) -> Any:
        if Path(path).suffix == ".png":
            axis = fig.axes[0]
            legend = axis.get_legend()
            five_mass_metrics.append(
                (
                    len(axis.lines),
                    len(legend.get_texts()) if legend is not None else 0,
                )
            )
        return original_savefig(fig, path, *args, **kwargs)

    with patch.object(Figure, "savefig", capture_five_mass_figure):
        plot_agn_time(
            five_mass_time_data,
            figures_dir / "five_mass_agn_time",
        )
        plot_rate_vs_redshift(
            five_mass_rate_data,
            figures_dir / "five_mass_rate_vs_redshift",
        )
    if five_mass_metrics != [(5, 5), (5, 5)]:
        raise AssertionError(
            "five configured SMBH masses were not all plotted: " f"{five_mass_metrics}"
        )

    status_data_path = save_plot_data(
        status_data, figures_dir / "data" / "imri_status_fraction_summary.csv"
    )
    mass_status_data_path = save_plot_data(
        mass_status_data,
        figures_dir / "data" / "imri_status_fraction_by_smbh_mass_summary.csv",
    )
    status_figure_metrics: list[tuple[int, tuple[str, ...], str, str]] = []

    def capture_status_figure(
        fig: Figure, path: Any, *args: Any, **kwargs: Any
    ) -> Any:
        if Path(path).suffix == ".png":
            y_labels = tuple(
                axis.get_ylabel() for axis in fig.axes if axis.get_ylabel()
            )
            legend_text = " ".join(
                text.get_text()
                for axis in fig.axes
                for legend in [axis.get_legend()]
                if legend is not None
                for text in legend.get_texts()
            )
            footer_text = " ".join(text.get_text() for text in fig.texts)
            status_figure_metrics.append(
                (len(fig.axes), y_labels, legend_text, footer_text)
            )
        return original_savefig(fig, path, *args, **kwargs)

    with patch.object(Figure, "savefig", capture_status_figure):
        status_figures = plot_status_fractions(
            status_data,
            figures_dir / "imri_status_fractions_vs_redshift",
        )
        mass_status_figures = plot_status_fractions_vs_smbh_mass(
            mass_status_data,
            figures_dir / "imri_status_fractions_vs_smbh_mass",
        )
    if len(status_figure_metrics) != 2:
        raise AssertionError("both layered status figures were not generated")
    for axes, y_labels, legend_text, footer_text in status_figure_metrics:
        if axes != 2 * len(STATUS_VARIABLE_ORDER):
            raise AssertionError("status figure does not have four taxonomy rows")
        if y_labels != tuple(STATUS_VARIABLE_LABELS.values()):
            raise AssertionError("status figure row labels are not scholar-facing")
        if "GW faster at start" not in legend_text or "gas_stop_reason" in legend_text:
            raise AssertionError("status figure legend did not use explicit labels")
        if "Denominators:" not in footer_text or "aggregates" not in footer_text:
            raise AssertionError(
                "status figure omitted denominator/aggregation wording"
            )
    _assert_round_trip(status_data, status_data_path)
    _assert_round_trip(mass_status_data, mass_status_data_path)
    validate_status_fraction_data(pd.read_csv(status_data_path))
    validate_status_fraction_data(
        pd.read_csv(mass_status_data_path), "m_smbh_msun"
    )

    figure_paths: list[Path] = []
    for outputs in outputs_by_model.values():
        for key in ("agn_time_figures", "redshift_figures", "smbh_mass_figures"):
            figure_paths.extend(Path(path) for path in outputs[key])
    figure_paths.extend(Path(path) for path in status_figures)
    figure_paths.extend(Path(path) for path in mass_status_figures)
    for path in figure_paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise AssertionError(f"missing or empty figure: {path}")
        if path.suffix == ".png" and mpimg.imread(path).size == 0:
            raise AssertionError(f"unreadable PNG figure: {path}")

    for disk_model, outputs in outputs_by_model.items():
        captions = Path(outputs["figure_captions"]).read_text()
        if captions.count("## ") != 3 or "exploratory" not in captions.lower():
            raise AssertionError("scholar-facing captions are incomplete")
        if disk_model.replace("_", " ").split()[0] not in captions.lower():
            raise AssertionError("migration scenario is absent from captions")
        if "trap_bound" in captions or "profile_error" in captions:
            raise AssertionError("diagnostic terminology leaked into captions")

    statuses = set(repeats["rate_status"])
    if "not_evaluated_band_entry_inside_isco" not in statuses:
        raise AssertionError("inside-ISCO source status was lost")
    if "trap_bound" not in set(events["migration_status"]):
        raise AssertionError("physical trap source status was lost")
    print(f"PASS: scholar-facing IMRI visualization harness ({output_dir})")


def main(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    run_harness(arguments.output_dir)


if __name__ == "__main__":
    main()
