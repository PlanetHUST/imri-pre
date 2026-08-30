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
    GROUP_COLUMNS,
    RATE_SUMMARY_COLUMNS,
    TIME_SERIES_COLUMNS,
    RunResults,
    build_agn_time_plot_data,
    build_rate_summary_data,
    generate_scan_figures,
    select_time_series_data,
    shared_rate_axis_limits,
    validate_agn_time_endpoints,
    validate_event_keys,
    validate_repeat_results,
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

    semi_events = torque_events.copy()
    semi_events["disk_model"] = SEMI_WITHOUT_TRAP
    semi_second_candidate = np.isclose(semi_events["m_smbh_msun"], 1.0e3) & semi_events[
        "object_index"
    ].eq(1)
    semi_events.loc[semi_second_candidate, "is_imri_candidate"] = False
    semi_events.loc[
        semi_events["migration_status"].eq("trap_bound"), "migration_status"
    ] = "finite"

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
    validate_agn_time_endpoints(time_data, first_repeats, first_averaged)
    _assert_monotone(time_data)

    if list(selected_time.columns) != TIME_SERIES_COLUMNS:
        raise AssertionError("time-series table is not compact or ordered")
    if list(summary_data.columns) != RATE_SUMMARY_COLUMNS:
        raise AssertionError("rate-summary table is not compact or ordered")

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
    figure_metrics: list[tuple[int, int, int, int, str]] = []

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
    for axes, curves, legend_entries, bands, legend_text in figure_metrics:
        if axes != 1:
            raise AssertionError("a main figure contains more than one axis")
        if curves > 4 or legend_entries > 4:
            raise AssertionError("a main figure contains too many curves")
        if bands != 0:
            raise AssertionError("an uncertainty band was drawn with only two repeats")
        if "proxy" in legend_text.lower():
            raise AssertionError("proxy detectability leaked into a main figure")

    figure_paths: list[Path] = []
    for outputs in outputs_by_model.values():
        for key in ("agn_time_figures", "redshift_figures", "smbh_mass_figures"):
            figure_paths.extend(Path(path) for path in outputs[key])
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
