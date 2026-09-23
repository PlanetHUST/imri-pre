# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportOperatorIssue=false, reportIndexIssue=false, reportCallIssue=false, reportGeneralTypeIssues=false
"""Production figures describing one persisted IMRI run.

Figures are generated strictly from persisted outputs after ``run_simulation``
has completed.  Plotting never reruns physics and never advances the production
random number state.  Representative candidate subsets are selected by
deterministic sorting rather than by resampling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from disk_quantity.pagn_disk_adapter import build_pagn_disk_profile

from .config import DISK_ACCRETION_RATE_G_S_PER_MSUN, PAGN_ALPHA
from lisa.snr import (
    SNR_STATUS_EVALUATED,
    SNR_STATUS_MODEL_UNAVAILABLE,
    SNR_STATUS_NO_OVERLAP,
    SNR_STATUS_NUMERICAL_FAILURE,
    SNR_STATUS_REQUIRED_INPUT_UNAVAILABLE,
    fourier_inspiral_amplitude_hz_inverse,
    lisa_effective_strain_psd_hz,
)
from viz.rate_plots import (
    STATUS_CATEGORY_COLORS,
    STATUS_CATEGORY_LABELS,
    STATUS_CATEGORY_ORDER,
    TRANSPORT_ORDER,
    RunResults,
    generate_run_figures,
    load_run_results,
    new_figure,
    save_figure,
    save_plot_data,
)

FREQUENCY_UNIT_SCALES = (
    ("hz", 1.0),
    ("milli_hz", 1.0e-3),
    ("micro_hz", 1.0e-6),
    ("nano_hz", 1.0e-9),
    ("pico_hz", 1.0e-12),
)
MAX_REPRESENTATIVE_TRACKS = 25
SNR_DENSITY_FACTOR = 5.0**0.5
CANDIDATE_TRACK_LINEWIDTH = 1.4
CANDIDATE_TRACK_ALPHA = 0.8
PRIMARY_MASS_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
CANDIDATE_TRACK_LINESTYLES = ("-", "--", "-.", ":")
CAPTION_FILENAME = "caption.md"
DETECTOR_STATUS_ORDER = (
    "detectable",
    "lisa_snr_below_threshold",
    "no_lisa_frequency_overlap",
    "required_input_unavailable",
    "model_unavailable",
    "numerical_failure",
    "not_evaluated_upstream",
    "not_applicable_not_imri_candidate",
)
DETECTOR_STATUS_LABELS = {
    "detectable": "LISA detectable",
    "lisa_snr_below_threshold": "Evaluated below threshold",
    "no_lisa_frequency_overlap": "No frequency overlap",
    "required_input_unavailable": "Required input unavailable",
    "model_unavailable": "Model unavailable",
    "numerical_failure": "Numerical failure",
    "not_evaluated_upstream": "Not evaluated upstream",
    "not_applicable_not_imri_candidate": "Not an intrinsic candidate",
}
DETECTOR_STATUS_COLORS = {
    "detectable": "#0072B2",
    "lisa_snr_below_threshold": "#E69F00",
    "no_lisa_frequency_overlap": "#CC79A7",
    "required_input_unavailable": "#7F7F7F",
    "model_unavailable": "#9467BD",
    "numerical_failure": "#D55E00",
    "not_evaluated_upstream": "#BBBBBB",
    "not_applicable_not_imri_candidate": "#DDDDDD",
}
POPULATION_COLUMNS = (
    "n_sampled",
    "n_disk_cross",
    "n_embedded",
    "n_gw_eligible",
    "n_imri",
    "n_det",
)
POPULATION_LABELS = {
    "n_sampled": "sampled",
    "n_disk_cross": "disk crossing",
    "n_embedded": "embedded",
    "n_gw_eligible": "gw propagation eligible",
    "n_imri": "physical LISA entry",
    "n_det": "LISA detectable",
}
MIGRATION_TIMESCALE_HISTOGRAM_COLUMNS = (
    "m_smbh_msun",
    "bin_left_myr",
    "bin_right_myr",
    "imri_candidate_count",
)
MIGRATION_MASS_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">")
MIGRATION_ROUTE_ORDER = ("entry_route", "gw_handoff_route")
MIGRATION_ROUTE_LABELS = {
    "entry_route": "Entry route",
    "gw_handoff_route": "GW-handoff route",
}
MIGRATION_ROUTE_MARKERS = {
    "entry_route": "o",
    "gw_handoff_route": "^",
}
MIGRATION_TIMESCALE_SCATTER_COLUMNS = (
    "event_id",
    "m_smbh_msun",
    "r_ini_rg",
    "band_entry_status",
    "migration_route",
    "migration_status",
    "migration_timescale_myr",
)
MIGRATION_TIMESCALE_HISTOGRAM_BIN_COUNT = 24

plt.rcParams.update(
    {
        "font.size": 18,
        "legend.fontsize": 18,
        "axes.titlesize": 20,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
    }
)


def _write_caption(
    path: Path,
    title: str,
    entries: Sequence[tuple[str, str]],
) -> Path:
    """Write removed in-figure notes as a companion caption file."""
    lines = [f"# {title}", ""]
    for figure_stem, note in entries:
        if not note:
            continue
        lines.extend([f"## `{figure_stem}`", "", f"- {note}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


def _boolean_series(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.lower()
    result = pd.Series(pd.NA, index=series.index, dtype="boolean")
    result[text.eq("true")] = True
    result[text.eq("false")] = False
    return result


def _frequency_column(frame: pd.DataFrame, base_name: str) -> tuple[str, float]:
    for unit_name, scale_hz in FREQUENCY_UNIT_SCALES:
        column = f"{base_name}_{unit_name}"
        if column in frame.columns:
            return column, scale_hz
    if base_name in frame.columns:
        return base_name, 1.0
    raise KeyError(
        f"neither {base_name!r} nor a unit-suffixed variant is present in the table"
    )


def _frequency_values(frame: pd.DataFrame, base_name: str) -> np.ndarray:
    column, scale_hz = _frequency_column(frame, base_name)
    values = np.asarray(pd.to_numeric(frame[column], errors="coerce"), dtype=float)
    return values * scale_hz


def _disk_profiles_frame(run: RunResults) -> pd.DataFrame:
    masses = run.metadata["parameters"]["smbh_masses_msun"]
    records: list[dict[str, float]] = []
    for mass in masses:
        profile = build_pagn_disk_profile(
            float(mass),
            PAGN_ALPHA,
            accretion_rate_g_s=DISK_ACCRETION_RATE_G_S_PER_MSUN * float(mass),
        )
        radius_rg = np.asarray(profile.radius_cm, dtype=float) / float(
            profile.gravitational_radius_cm
        )
        hold = {
            "m_smbh_msun": float(mass),
            "r_out_over_rg": float(
                profile.outer_radius_cm / profile.gravitational_radius_cm
            ),
        }
        for radius, sigma, aspect, q_value in zip(
            radius_rg,
            np.asarray(profile.surface_density_g_cm2, dtype=float),
            np.asarray(profile.aspect_ratio, dtype=float),
            np.asarray(profile.toomre_q, dtype=float),
            strict=True,
        ):
            records.append(
                {
                    **hold,
                    "r_over_rg": float(radius),
                    "surface_density_g_cm2": float(sigma),
                    "aspect_ratio": float(aspect),
                    "toomre_q": float(q_value),
                }
            )
    return pd.DataFrame.from_records(records)


def _figure_disk(run: RunResults, figure_root: Path) -> dict[str, list[Path] | Path]:
    frame = _disk_profiles_frame(run)
    data_path = save_plot_data(
        frame, figure_root / "disk" / "data" / "disk_profiles.csv"
    )
    masses = sorted({float(value) for value in frame["m_smbh_msun"].unique()})
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, max(len(masses), 1)))

    fig, axes = new_figure(
        "pAGN disk profiles with the Q=1 outer boundary",
        nrows=3,
        ncols=1,
        figsize=(7.0, 11.0),
    )
    panel_specs = (
        ("surface_density_g_cm2", r"$\Sigma\;[\mathrm{g\,cm^{-2}}]$"),
        ("aspect_ratio", r"$H/R$"),
        ("toomre_q", r"$Q$"),
    )
    for panel_index, (column, ylabel) in enumerate(panel_specs):
        axis = axes[panel_index, 0]
        for mass, color in zip(masses, colors, strict=False):
            selected = frame[np.isclose(frame["m_smbh_msun"], mass)]
            radius = selected["r_over_rg"].to_numpy(dtype=float)
            values = selected[column].to_numpy(dtype=float)
            mask = (
                np.isfinite(radius)
                & np.isfinite(values)
                & (radius > 0.0)
                & (values > 0.0)
            )
            if not mask.any():
                continue
            axis.plot(
                radius[mask],
                values[mask],
                color=color,
                linewidth=2.0,
                label=rf"$M_\bullet={mass:.1e}\,M_\odot$",
            )
            axis.axvline(
                selected["r_out_over_rg"].iloc[0],
                color=color,
                linestyle=":",
                linewidth=1.2,
                alpha=0.8,
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_ylabel(ylabel)
        axis.set_xlabel(r"$R/r_g$")
        axis.yaxis.grid(True, color="0.88", linewidth=0.8)
        if column == "toomre_q":
            axis.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
        axis.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    figures = save_figure(fig, figure_root / "disk" / "disk_profiles")
    caption_path = _write_caption(
        figure_root / "disk" / CAPTION_FILENAME,
        "pAGN disk figure captions",
        [
            (
                "disk_profiles",
                "The dashed horizontal line marks the Toomre threshold Q = 1; the "
                "dotted vertical lines mark the physical Q=1 outer boundary "
                "R_Q=1 for each SMBH mass.",
            ),
        ],
    )
    return {
        "disk_profile_data": data_path,
        "disk_caption": caption_path,
        "disk_profile_figures": figures,
    }


def _population_frame(run: RunResults) -> pd.DataFrame:
    repeats = run.repeats.copy()
    repeats["n_embedded"] = pd.to_numeric(
        repeats["n_emb_star"], errors="coerce"
    ) + pd.to_numeric(repeats["n_emb_bh"], errors="coerce")
    candidates = _boolean_series(run.events["is_imri_candidate"]).eq(True)
    repeat_keys = repeats[["m_smbh_msun", "repeat"]].drop_duplicates()
    eligible_counts = (
        run.events.loc[candidates]
        .groupby(["m_smbh_msun", "repeat"], sort=True)
        .size()
        .rename("n_gw_eligible")
        .reset_index()
    )
    eligible = repeat_keys.merge(
        eligible_counts, on=["m_smbh_msun", "repeat"], how="left"
    )
    eligible["n_gw_eligible"] = eligible["n_gw_eligible"].fillna(0.0)
    eligible_mean = (
        eligible.groupby("m_smbh_msun", sort=True)["n_gw_eligible"].mean().reset_index()
    )
    grouped = repeats.groupby("m_smbh_msun", sort=True).mean(numeric_only=True)
    grouped = grouped.reset_index()
    frame = grouped[["m_smbh_msun"]].copy()
    for column in ("n_sampled", "n_disk_cross", "n_embedded", "n_imri", "n_det"):
        frame[column] = pd.to_numeric(grouped[column], errors="coerce")
    frame = frame.merge(eligible_mean, on="m_smbh_msun", how="left")
    frame["n_gw_eligible"] = frame["n_gw_eligible"].fillna(0.0)
    return pd.DataFrame(frame[["m_smbh_msun", *POPULATION_COLUMNS]])


def _figure_population(
    run: RunResults, figure_root: Path
) -> dict[str, list[Path] | Path]:
    frame = _population_frame(run)
    data_path = save_plot_data(
        frame, figure_root / "population" / "data" / "outcome_stages.csv"
    )
    masses = frame["m_smbh_msun"].to_numpy(dtype=float)
    positions = np.arange(len(POPULATION_COLUMNS), dtype=float)
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, max(len(masses), 1)))

    fig, axis = new_figure("Outcome stages per SMBH mass")
    width = 0.8 / max(len(masses), 1)
    for mass_index, (mass, color) in enumerate(zip(masses, colors, strict=False)):
        values = (
            frame.loc[np.isclose(frame["m_smbh_msun"], mass), list(POPULATION_COLUMNS)]
            .to_numpy(dtype=float)
            .reshape(-1)
        )
        axis.bar(
            positions + (mass_index - (len(masses) - 1) / 2.0) * width,
            values,
            width=width,
            color=color,
            label=rf"$M_\bullet={mass:.1e}\,M_\odot$",
        )
    axis.set_yscale("symlog", linthresh=1.0)
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [POPULATION_LABELS[column] for column in POPULATION_COLUMNS],
        rotation=20,
        ha="right",
    )
    axis.set_ylabel("objects per repeat [mean]")
    axis.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    figures = save_figure(fig, figure_root / "population" / "outcome_stages")
    caption_path = _write_caption(
        figure_root / "population" / CAPTION_FILENAME,
        "Outcome-stage figure captions",
        [
            (
                "outcome_stages",
                "Counts are averaged per Monte Carlo repeat (not summed) for each "
                "configured SMBH mass. Stages are shown as separate bars and are "
                "not drawn as a strict funnel unless the persisted counters are "
                "genuinely nested.",
            ),
        ],
    )
    return {
        "population_data": data_path,
        "population_caption": caption_path,
        "population_figures": figures,
    }


def _migration_frame(run: RunResults) -> pd.DataFrame:
    events = run.events.copy()
    events["transport_status"] = events["gas_stop_reason"].astype("string").str.strip()
    columns = [
        "event_id",
        "m_smbh_msun",
        "mass_msun",
        "r_ini_rg",
        "r_gas_fin_rg",
        "r_entry_rg",
        "r_entry_formal_rg",
        "r_isco_rg",
        "r_migration_min_rg",
        "r_handoff_rg",
        "gas_stop_reason",
        "transport_status",
        "migration_status",
        "migration_evaluated",
        "is_imri_candidate",
    ]
    return pd.DataFrame(events[columns])


def _imri_candidate_migration_timescale_and_validity(
    events: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the persisted route-specific time, route, and histogram validity."""
    required_event_columns = {
        "m_smbh_msun",
        "band_entry_status",
        "migration_time_to_entry_myr",
        "migration_time_to_handoff_myr",
        "migration_evaluated",
        "migration_status",
        "is_imri_candidate",
    }
    missing = sorted(required_event_columns.difference(events.columns))
    if missing:
        raise ValueError(f"event results are missing migration columns: {missing}")

    entry_time_myr = pd.to_numeric(
        events["migration_time_to_entry_myr"], errors="coerce"
    ).to_numpy(dtype=float)
    handoff_time_myr = pd.to_numeric(
        events["migration_time_to_handoff_myr"], errors="coerce"
    ).to_numpy(dtype=float)
    handoff_required = (
        events["band_entry_status"]
        .astype(str)
        .eq("entry_below_migration_domain_but_above_isco")
    )
    migration_route = np.where(
        handoff_required.to_numpy(dtype=bool),
        "gw_handoff_route",
        "entry_route",
    )
    migration_timescale_myr = np.where(
        handoff_required.to_numpy(dtype=bool),
        handoff_time_myr,
        entry_time_myr,
    )
    valid = (
        _boolean_series(events["is_imri_candidate"])
        .eq(True)
        .fillna(False)
        .to_numpy(dtype=bool)
        & _boolean_series(events["migration_evaluated"])
        .eq(True)
        .fillna(False)
        .to_numpy(dtype=bool)
        & events["migration_status"]
        .astype(str)
        .isin(("already_at_target", "finite"))
        .to_numpy(dtype=bool)
        & np.isfinite(migration_timescale_myr)
        & (migration_timescale_myr >= 0.0)
    )
    return migration_timescale_myr, migration_route, valid


def build_imri_candidate_migration_timescale_scatter(run: RunResults) -> pd.DataFrame:
    """Return valid candidate migration times against their persisted start radii."""
    required_event_columns = {"event_id", "r_ini_rg"}
    missing = sorted(required_event_columns.difference(run.events.columns))
    if missing:
        raise ValueError(f"event results are missing migration columns: {missing}")

    events = run.events
    migration_timescale_myr, migration_route, valid = (
        _imri_candidate_migration_timescale_and_validity(events)
    )
    selected = events.loc[
        valid,
        [
            "event_id",
            "m_smbh_msun",
            "r_ini_rg",
            "band_entry_status",
            "migration_status",
        ],
    ].copy()
    selected["migration_route"] = migration_route[valid]
    selected["migration_timescale_myr"] = migration_timescale_myr[valid]
    return pd.DataFrame(selected[list(MIGRATION_TIMESCALE_SCATTER_COLUMNS)])


def build_imri_candidate_migration_histogram(run: RunResults) -> pd.DataFrame:
    """Bin valid route-specific migration times for intrinsic IMRI candidates."""
    events = run.events
    migration_timescale_myr, _, valid = (
        _imri_candidate_migration_timescale_and_validity(events)
    )
    event_masses_msun = pd.to_numeric(events["m_smbh_msun"], errors="coerce").to_numpy(
        dtype=float
    )

    candidate_times_myr = migration_timescale_myr[valid]
    positive_times_myr = candidate_times_myr[candidate_times_myr > 0.0]
    if positive_times_myr.size:
        minimum_positive_myr = float(positive_times_myr.min())
        maximum_myr = float(positive_times_myr.max())
        if np.isclose(minimum_positive_myr, maximum_myr):
            minimum_positive_myr /= np.sqrt(10.0)
            maximum_myr *= np.sqrt(10.0)
        positive_edges_myr = np.geomspace(
            minimum_positive_myr,
            maximum_myr,
            MIGRATION_TIMESCALE_HISTOGRAM_BIN_COUNT + 1,
        )
        bin_edges_myr = (
            np.concatenate(([0.0], positive_edges_myr))
            if np.any(candidate_times_myr == 0.0)
            else positive_edges_myr
        )
    elif candidate_times_myr.size:
        bin_edges_myr = np.asarray([0.0, 1.0], dtype=float)
    else:
        return pd.DataFrame(columns=MIGRATION_TIMESCALE_HISTOGRAM_COLUMNS)

    records: list[dict[str, float | int | str]] = []
    smbh_masses_msun = np.sort(np.unique(event_masses_msun[valid]))
    for smbh_mass_msun in smbh_masses_msun:
        counts, _ = np.histogram(
            migration_timescale_myr[
                valid & np.isclose(event_masses_msun, smbh_mass_msun)
            ],
            bins=bin_edges_myr,
        )
        records.extend(
            {
                "m_smbh_msun": float(smbh_mass_msun),
                "bin_left_myr": float(bin_edges_myr[index]),
                "bin_right_myr": float(bin_edges_myr[index + 1]),
                "imri_candidate_count": int(count),
            }
            for index, count in enumerate(counts)
        )
    return pd.DataFrame.from_records(
        records, columns=MIGRATION_TIMESCALE_HISTOGRAM_COLUMNS
    )


def _figure_migration(
    run: RunResults, figure_root: Path
) -> dict[str, list[Path] | Path]:
    frame = _migration_frame(run)
    data_path = save_plot_data(
        frame, figure_root / "migration" / "data" / "migration_radii.csv"
    )
    fig, radius_axes = new_figure(
        "Gas transport outcomes and initial to gas-final radius",
        nrows=1,
        ncols=1,
        figsize=(11.5, 6.2),
    )
    radius_axis = radius_axes[0, 0]

    initial = pd.to_numeric(frame["r_ini_rg"], errors="coerce").to_numpy(dtype=float)
    final = pd.to_numeric(frame["r_gas_fin_rg"], errors="coerce").to_numpy(dtype=float)
    event_masses_msun = pd.to_numeric(
        frame["m_smbh_msun"], errors="coerce"
    ).to_numpy(
        dtype=float
    )
    valid = np.isfinite(initial) & np.isfinite(final) & (initial > 0.0) & (final > 0.0)
    statuses = frame["transport_status"].astype(str).to_numpy()
    masses = sorted(
        {float(value) for value in event_masses_msun[valid] if np.isfinite(value)}
    )
    mass_markers = {
        mass: (
            MIGRATION_MASS_MARKERS[index]
            if index < len(MIGRATION_MASS_MARKERS)
            else f"${index + 1}$"
        )
        for index, mass in enumerate(masses)
    }

    for category in TRANSPORT_ORDER:
        for mass in masses:
            mask = (
                valid
                & (statuses == category)
                & np.isclose(event_masses_msun, mass)
            )
            if not mask.any():
                continue
            radius_axis.scatter(
                initial[mask],
                final[mask],
                s=22,
                marker=mass_markers[mass],
                color=STATUS_CATEGORY_COLORS["transport_status"][category],
            )
        unavailable_mass = (
            valid & ~np.isfinite(event_masses_msun) & (statuses == category)
        )
        if unavailable_mass.any():
            radius_axis.scatter(
                initial[unavailable_mass],
                final[unavailable_mass],
                s=22,
                marker="x",
                color=STATUS_CATEGORY_COLORS["transport_status"][category],
            )

    if valid.any():
        limits = (
            min(initial[valid].min(), final[valid].min()),
            max(initial[valid].max(), final[valid].max()),
        )
        radius_axis.plot(limits, limits, color="0.4", linewidth=1.0, linestyle="--")
        radius_axis.set_xlim(*limits)
        radius_axis.set_ylim(*limits)
    radius_axis.set_xscale("log")
    radius_axis.set_yscale("log")
    radius_axis.set_xlabel(r"initial radius $r_{\rm ini}/r_g$")
    radius_axis.set_ylabel(r"gas-final radius $r_{\rm gas,fin}/r_g$")
    transport_handles = [
        Line2D(
            [],
            [],
            color=STATUS_CATEGORY_COLORS["transport_status"][category],
            linestyle="none",
            marker="o",
            markersize=6,
            label=STATUS_CATEGORY_LABELS["transport_status"][category],
        )
        for category in TRANSPORT_ORDER
    ]
    mass_handles = [
        Line2D(
            [],
            [],
            color="0.35",
            linestyle="none",
            marker=mass_markers[mass],
            markersize=6,
            label=rf"$M_\bullet={mass:.0e}\,M_\odot$",
        )
        for mass in masses
    ]
    if np.any(valid & ~np.isfinite(event_masses_msun)):
        mass_handles.append(
            Line2D(
                [],
                [],
                color="0.35",
                linestyle="none",
                marker="x",
                markersize=6,
                label="SMBH mass unavailable",
            )
        )
    transport_legend = radius_axis.legend(
        handles=transport_handles,
        title="Transport outcome (color)",
        frameon=False,
        fontsize=7,
        title_fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    radius_axis.add_artist(transport_legend)
    if mass_handles:
        radius_axis.legend(
            handles=mass_handles,
            title="SMBH mass (marker)",
            frameon=False,
            fontsize=7,
            title_fontsize=8,
            loc="lower left",
            bbox_to_anchor=(1.02, 0.0),
            borderaxespad=0.0,
        )
    fig.tight_layout()
    figures = save_figure(fig, figure_root / "migration" / "migration_radii")

    status_counts = pd.DataFrame(
        {
            "transport_status": list(TRANSPORT_ORDER),
            "count": [
                int(frame["transport_status"].astype(str).eq(category).sum())
                for category in TRANSPORT_ORDER
            ],
        }
    )
    status_data_path = save_plot_data(
        status_counts,
        figure_root / "migration" / "data" / "transport_status_counts.csv",
    )
    status_fig, status_axes = new_figure(
        "Gas transport-status counts",
        nrows=1,
        ncols=1,
        figsize=(9.0, 5.4),
    )
    status_axis = status_axes[0, 0]
    positions = np.arange(len(TRANSPORT_ORDER), dtype=float)
    status_axis.bar(
        positions,
        status_counts["count"].to_numpy(dtype=int),
        color=[STATUS_CATEGORY_COLORS["transport_status"][c] for c in TRANSPORT_ORDER],
    )
    status_axis.set_yscale("symlog", linthresh=1.0)
    status_axis.set_xticks(positions)
    status_axis.set_xticklabels(
        [STATUS_CATEGORY_LABELS["transport_status"][c] for c in TRANSPORT_ORDER],
        rotation=30,
        ha="right",
        fontsize=8,
    )
    status_axis.set_ylabel("persisted events")
    status_axis.set_title("Transport status", fontsize=12)
    status_fig.tight_layout()
    status_figures = save_figure(
        status_fig,
        figure_root / "migration" / "migration_transport_status",
    )

    timescale_scatter = build_imri_candidate_migration_timescale_scatter(run)
    timescale_scatter_data_path = save_plot_data(
        timescale_scatter,
        figure_root
        / "migration"
        / "data"
        / "migration_timescale_vs_initial_radius.csv",
    )
    timescale_scatter_fig, timescale_scatter_axes = new_figure(
        "IMRI-candidate migration timescale vs initial radius",
        nrows=1,
        ncols=1,
        figsize=(9.2, 5.6),
    )
    timescale_scatter_axis = timescale_scatter_axes[0, 0]
    scatter_initial_rg = pd.to_numeric(
        timescale_scatter["r_ini_rg"], errors="coerce"
    ).to_numpy(dtype=float)
    scatter_timescale_myr = pd.to_numeric(
        timescale_scatter["migration_timescale_myr"], errors="coerce"
    ).to_numpy(dtype=float)
    scatter_masses_msun = pd.to_numeric(
        timescale_scatter["m_smbh_msun"], errors="coerce"
    ).to_numpy(dtype=float)
    scatter_routes = timescale_scatter["migration_route"].astype(str).to_numpy()
    scatter_valid = (
        np.isfinite(scatter_initial_rg)
        & (scatter_initial_rg > 0.0)
        & np.isfinite(scatter_timescale_myr)
        & (scatter_timescale_myr >= 0.0)
        & np.isfinite(scatter_masses_msun)
    )
    scatter_masses = sorted(
        {float(value) for value in scatter_masses_msun[scatter_valid]}
    )
    scatter_colors = plt.get_cmap("tab10")(
        np.arange(len(scatter_masses), dtype=int) % 10
    )
    mass_colors = {
        mass: color
        for mass, color in zip(scatter_masses, scatter_colors, strict=True)
    }
    for mass in scatter_masses:
        for route in MIGRATION_ROUTE_ORDER:
            mask = (
                scatter_valid
                & np.isclose(scatter_masses_msun, mass)
                & (scatter_routes == route)
            )
            if not mask.any():
                continue
            timescale_scatter_axis.scatter(
                scatter_initial_rg[mask],
                scatter_timescale_myr[mask],
                s=28,
                marker=MIGRATION_ROUTE_MARKERS[route],
                color=mass_colors[mass],
                alpha=0.8,
            )
    timescale_scatter_axis.set_xscale("log")
    if scatter_valid.any():
        positive_timescale_myr = scatter_timescale_myr[
            scatter_valid & (scatter_timescale_myr > 0.0)
        ]
        if np.any(scatter_timescale_myr[scatter_valid] == 0.0):
            linear_threshold_myr = (
                float(positive_timescale_myr.min()) / 10.0
                if positive_timescale_myr.size
                else 1.0
            )
            timescale_scatter_axis.set_yscale(
                "symlog", linthresh=linear_threshold_myr
            )
        else:
            timescale_scatter_axis.set_yscale("log")
    else:
        timescale_scatter_axis.text(
            0.5,
            0.5,
            "No valid migration-timescale points",
            transform=timescale_scatter_axis.transAxes,
            ha="center",
            va="center",
        )
    timescale_scatter_axis.set_xlabel(r"initial radius $r_{\rm ini}/r_g$")
    timescale_scatter_axis.set_ylabel("migration timescale to gas target [Myr]")
    mass_color_handles = [
        Line2D(
            [],
            [],
            color=mass_colors[mass],
            linestyle="none",
            marker="o",
            markersize=6,
            label=rf"$M_\bullet={mass:.0e}\,M_\odot$",
        )
        for mass in scatter_masses
    ]
    present_routes = [
        route
        for route in MIGRATION_ROUTE_ORDER
        if np.any(scatter_valid & (scatter_routes == route))
    ]
    route_handles = [
        Line2D(
            [],
            [],
            color="0.35",
            linestyle="none",
            marker=MIGRATION_ROUTE_MARKERS[route],
            markersize=6,
            label=MIGRATION_ROUTE_LABELS[route],
        )
        for route in present_routes
    ]
    if mass_color_handles:
        mass_legend = timescale_scatter_axis.legend(
            handles=mass_color_handles,
            title="SMBH mass (color)",
            frameon=False,
            fontsize=8,
            title_fontsize=9,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
        )
        timescale_scatter_axis.add_artist(mass_legend)
    if route_handles:
        timescale_scatter_axis.legend(
            handles=route_handles,
            title="Gas target (marker)",
            frameon=False,
            fontsize=8,
            title_fontsize=9,
            loc="lower left",
            bbox_to_anchor=(1.02, 0.0),
            borderaxespad=0.0,
        )
    timescale_scatter_fig.tight_layout()
    timescale_scatter_figures = save_figure(
        timescale_scatter_fig,
        figure_root / "migration" / "migration_timescale_vs_initial_radius",
    )

    timescale_histogram = build_imri_candidate_migration_histogram(run)
    timescale_data_path = save_plot_data(
        timescale_histogram,
        figure_root / "migration" / "data" / "migration_timescale_histogram.csv",
    )
    timescale_fig, timescale_axis = new_figure(
        "IMRI-candidate gas-migration timescales"
    )
    if not timescale_histogram.empty:
        smbh_masses_msun = np.sort(timescale_histogram["m_smbh_msun"].unique())
        colors = plt.get_cmap("tab10")(np.arange(len(smbh_masses_msun), dtype=int) % 10)
        for smbh_mass_msun, color in zip(smbh_masses_msun, colors, strict=True):
            selected = timescale_histogram[
                np.isclose(timescale_histogram["m_smbh_msun"], smbh_mass_msun)
            ]
            bin_edges_myr = np.concatenate(
                (
                    selected["bin_left_myr"].to_numpy(dtype=float),
                    [float(selected["bin_right_myr"].iloc[-1])],
                )
            )
            counts = selected["imri_candidate_count"].to_numpy(dtype=int)
            timescale_axis.stairs(
                counts,
                bin_edges_myr,
                color=color,
                linewidth=2.0,
                label=(
                    rf"$M_\bullet={smbh_mass_msun:.0e}\,M_\odot$ "
                    f"($n={int(counts.sum())}$)"
                ),
            )
        all_edges_myr = np.concatenate(
            (
                timescale_histogram["bin_left_myr"].unique(),
                timescale_histogram["bin_right_myr"].unique(),
            )
        )
        positive_edges_myr = all_edges_myr[all_edges_myr > 0.0]
        if np.any(all_edges_myr == 0.0) and positive_edges_myr.size:
            timescale_axis.set_xscale(
                "symlog", linthresh=float(positive_edges_myr.min()) / 10.0
            )
        elif positive_edges_myr.size:
            timescale_axis.set_xscale("log")
        timescale_axis.legend(frameon=False, fontsize=8, loc="best")
    else:
        timescale_axis.text(
            0.5,
            0.5,
            "No valid migration timescales",
            transform=timescale_axis.transAxes,
            ha="center",
            va="center",
        )
    timescale_axis.set_xlabel("migration timescale to gas target [Myr]")
    timescale_axis.set_ylabel("IMRI candidates per bin")
    timescale_axis.yaxis.grid(True, color="0.88", linewidth=0.8)
    timescale_fig.tight_layout()
    timescale_figures = save_figure(
        timescale_fig,
        figure_root / "migration" / "migration_timescale_histogram",
    )
    caption_path = _write_caption(
        figure_root / "migration" / CAPTION_FILENAME,
        "Migration figure captions",
        [
            (
                "migration_radii",
                "Color encodes the transport outcome and marker shape encodes "
                "SMBH mass. The dashed diagonal marks "
                "$r_{\\rm gas,fin}=r_{\\rm ini}$; horizontal branches correspond "
                "to common physical stopping radii.",
            ),
            (
                "migration_transport_status",
                "Counts retain every transport-outcome category in the configured "
                "order. Inside-ISCO and blank rows are excluded from this transport "
                "summary; unavailable or failure states remain distinct and are "
                "never converted to zero-valued physical outcomes.",
            ),
            (
                "migration_timescale_vs_initial_radius",
                "Points use exactly the same intrinsic-candidate validity logic and "
                "route-specific migration timescale as the histogram. Color "
                "distinguishes SMBH mass; marker shape distinguishes formal-entry "
                "routes from GW-handoff routes. Valid zero times remain visible on "
                "a symmetric-logarithmic axis when present.",
            ),
            (
                "migration_timescale_histogram",
                "Shared time bins contain only intrinsic IMRI candidates with "
                "valid model-predicted gas-migration times. The target is "
                "formal entry for in-domain routes and the handoff radius for "
                "GW-handoff routes. The legend reports the candidate count for "
                "each SMBH mass; non-candidates, trapped paths, failures, and "
                "not-evaluated rows are excluded.",
            ),
        ],
    )
    return {
        "migration_data": data_path,
        "migration_status_data": status_data_path,
        "migration_timescale_scatter_data": timescale_scatter_data_path,
        "migration_timescale_histogram_data": timescale_data_path,
        "migration_caption": caption_path,
        "migration_figures": figures,
        "migration_status_figures": status_figures,
        "migration_timescale_scatter_figures": timescale_scatter_figures,
        "migration_timescale_histogram_figures": timescale_figures,
    }


def _entry_frame(run: RunResults) -> pd.DataFrame:
    events = run.events
    columns = [
        "event_id",
        "band_entry_status",
        "r_entry_formal_rg",
        "r_isco_rg",
        "r_migration_min_rg",
        "r_handoff_rg",
        "r_entry_rg",
        "source_entry_frequency_hz",
        "is_imri_candidate",
    ]
    return pd.DataFrame(events[columns])


def _figure_entry(run: RunResults, figure_root: Path) -> dict[str, list[Path] | Path]:
    frame = _entry_frame(run)
    data_path = save_plot_data(
        frame, figure_root / "entry" / "data" / "entry_boundaries.csv"
    )
    numeric = frame.apply(
        lambda column: (
            pd.to_numeric(column, errors="coerce")
            if column.name.startswith("r_")
            else column
        )
    )
    formal = numeric["r_entry_formal_rg"].to_numpy(dtype=float)
    isco = numeric["r_isco_rg"].to_numpy(dtype=float)
    migration_min = numeric["r_migration_min_rg"].to_numpy(dtype=float)

    status_counts = pd.DataFrame(
        {
            "band_entry_status": list(STATUS_CATEGORY_ORDER["band_entry_status"]),
            "count": [
                int(pd.to_numeric(run.repeats[column], errors="coerce").sum())
                for column, _category in (
                    ("n_entry_in_domain", "entry_in_migration_domain"),
                    (
                        "n_entry_handoff",
                        "entry_below_migration_domain_but_above_isco",
                    ),
                    ("n_entry_inside_isco", "band_entry_inside_isco"),
                    ("n_profile_error", "profile_error"),
                )
            ],
        }
    )
    status_data_path = save_plot_data(
        status_counts, figure_root / "entry" / "data" / "entry_status_counts.csv"
    )

    fig, axes = new_figure(
        "Physical entry geometry: formal boundary, ISCO, migration floor, and status",
        nrows=1,
        ncols=3,
        figsize=(18.0, 5.2),
    )
    boundary_axis = axes[0, 0]
    for category in STATUS_CATEGORY_ORDER["band_entry_status"]:
        mask = frame["band_entry_status"].astype(str).eq(category).to_numpy()
        if not mask.any():
            continue
        boundary_axis.scatter(
            formal[mask],
            isco[mask],
            s=24,
            color=STATUS_CATEGORY_COLORS["band_entry_status"][category],
            label=STATUS_CATEGORY_LABELS["band_entry_status"][category],
        )
    finite = np.isfinite(formal) & np.isfinite(isco) & (formal > 0) & (isco > 0)
    if finite.any():
        limits = (
            min(formal[finite].min(), isco[finite].min()),
            max(formal[finite].max(), isco[finite].max()),
        )
        boundary_axis.plot(limits, limits, color="0.4", linestyle="--", linewidth=1.0)
        boundary_axis.set_xlim(*limits)
        boundary_axis.set_ylim(*limits)
    boundary_axis.set_xscale("log")
    boundary_axis.set_yscale("log")
    boundary_axis.set_xlabel(r"formal entry $r_{\rm entry,formal}/r_g$")
    boundary_axis.set_ylabel(r"ISCO $r_{\rm ISCO}/r_g$")
    boundary_axis.legend(frameon=False, fontsize=7, loc="best")

    floor_axis = axes[0, 1]
    floor_axis.scatter(
        migration_min,
        formal,
        s=24,
        color="#0072B2",
        label="formal entry vs migration floor",
    )
    floor_axis.set_xscale("log")
    floor_axis.set_yscale("log")
    floor_axis.set_xlabel(r"migration lower boundary $r_{\rm migration,min}/r_g$")
    floor_axis.set_ylabel(r"formal entry $r_{\rm entry,formal}/r_g$")
    floor_axis.legend(frameon=False, fontsize=8, loc="best")

    status_axis = axes[0, 2]
    positions = np.arange(len(status_counts), dtype=float)
    status_axis.bar(
        positions,
        status_counts["count"].to_numpy(dtype=float),
        color=[
            STATUS_CATEGORY_COLORS["band_entry_status"][status]
            for status in status_counts["band_entry_status"]
        ],
    )
    status_axis.set_xticks(positions)
    status_axis.set_xticklabels(
        [
            STATUS_CATEGORY_LABELS["band_entry_status"][status]
            for status in status_counts["band_entry_status"]
        ],
        rotation=30,
        ha="right",
        fontsize=8,
    )
    status_axis.set_ylabel("evaluated band entries")
    status_axis.set_yscale("symlog", linthresh=1.0)
    status_axis.set_title("Band-entry status", fontsize=12)
    fig.tight_layout()
    figures = save_figure(fig, figure_root / "entry" / "entry_boundaries")
    caption_path = _write_caption(
        figure_root / "entry" / CAPTION_FILENAME,
        "Physical-entry figure captions",
        [
            (
                "entry_boundaries",
                "Inside-ISCO and profile-error band-entry outcomes remain distinct "
                "and are never plotted as zero.",
            ),
        ],
    )
    return {
        "entry_data": data_path,
        "entry_status_data": status_data_path,
        "entry_caption": caption_path,
        "entry_figures": figures,
    }


def _candidate_events(run: RunResults) -> pd.DataFrame:
    events = run.events.copy()
    candidates = _boolean_series(events["is_imri_candidate"]).eq(True)
    detection_time = pd.to_numeric(events["detection_time_myr"], errors="coerce")
    physical = candidates & np.isfinite(detection_time)
    return events.loc[physical].copy()


def _balanced_representative_candidates(
    candidates: pd.DataFrame,
    *,
    max_tracks: int,
) -> pd.DataFrame:
    """Select the same number of lowest-redshift candidates per primary mass."""
    if candidates.empty or max_tracks <= 0:
        return candidates.iloc[0:0].copy()

    ordered = candidates.assign(
        _primary_mass_msun=pd.to_numeric(
            candidates["final_smbh_mass_msun"], errors="coerce"
        )
    )
    ordered = ordered.loc[np.isfinite(ordered["_primary_mass_msun"])].sort_values(
        by=["_primary_mass_msun", "detection_redshift", "mass_msun", "event_id"],
        kind="stable",
    )
    primary_mass_count = ordered["_primary_mass_msun"].nunique()
    if primary_mass_count == 0:
        return candidates.iloc[0:0].copy()

    tracks_per_mass = max(1, max_tracks // primary_mass_count)
    selected = (
        ordered.groupby("_primary_mass_msun", sort=True, group_keys=False)
        .head(tracks_per_mass)
        .head(max_tracks)
    )
    return selected.drop(columns="_primary_mass_msun")


def _candidate_tracks(
    candidates: pd.DataFrame,
    *,
    default_mission_duration_years: float,
    detector_min_hz: float,
    detector_max_hz: float,
    max_tracks: int = MAX_REPRESENTATIVE_TRACKS,
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(
            columns=[
                "candidate_id",
                "m1_source_msun",
                "frequency_observer_hz",
                "h_c_eff",
                "h_n",
                "in_detector_band",
                "in_snr_window",
            ]
        )
    ordered = _balanced_representative_candidates(
        candidates,
        max_tracks=max_tracks,
    )
    f_low_all = _frequency_values(ordered, "snr_f_low")
    f_high_all = _frequency_values(ordered, "snr_f_high")
    distance = pd.to_numeric(
        ordered["luminosity_distance_mpc"], errors="coerce"
    ).to_numpy(dtype=float)
    chirp = pd.to_numeric(
        ordered["chirp_mass_detector_msun"], errors="coerce"
    ).to_numpy(dtype=float)
    mission = pd.to_numeric(
        ordered["mission_duration_years"], errors="coerce"
    ).to_numpy(dtype=float)
    primary_mass_msun = pd.to_numeric(
        ordered["final_smbh_mass_msun"], errors="coerce"
    ).to_numpy(dtype=float)
    mission = np.where(
        np.isfinite(mission) & (mission > 0.0), mission, default_mission_duration_years
    )
    records: list[dict[str, Any]] = []
    for index in range(len(ordered)):
        if (
            not np.isfinite(f_low_all[index])
            or not np.isfinite(f_high_all[index])
            or not np.isfinite(distance[index])
            or not np.isfinite(chirp[index])
            or f_high_all[index] <= f_low_all[index]
        ):
            continue
        grid = np.geomspace(f_low_all[index], f_high_all[index], 120)
        event_id = str(ordered["event_id"].iloc[index])
        for frequency in grid:
            amplitude = fourier_inspiral_amplitude_hz_inverse(
                float(frequency), float(chirp[index]), float(distance[index])
            )
            noise = float(
                lisa_effective_strain_psd_hz(float(frequency), float(mission[index]))
            )
            characteristic = (4.0 / SNR_DENSITY_FACTOR) * float(frequency) * amplitude
            records.append(
                {
                    "candidate_id": event_id,
                    "m1_source_msun": float(primary_mass_msun[index]),
                    "frequency_observer_hz": float(frequency),
                    "h_c_eff": characteristic,
                    "h_n": float(np.sqrt(float(frequency) * noise)),
                    "in_detector_band": bool(
                        detector_min_hz <= frequency <= detector_max_hz
                    ),
                    "in_snr_window": bool(
                        f_low_all[index] <= frequency <= f_high_all[index]
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def _noise_curve(
    mission_duration_years: float,
    detector_min_hz: float,
    detector_max_hz: float,
) -> pd.DataFrame:
    grid = np.geomspace(detector_min_hz, detector_max_hz, 400)
    noise = np.asarray(
        [
            np.sqrt(
                frequency
                * lisa_effective_strain_psd_hz(frequency, mission_duration_years)
            )
            for frequency in grid
        ],
        dtype=float,
    )
    return pd.DataFrame({"frequency_observer_hz": grid, "h_n": noise})


def _lisa_candidate_summary(
    run: RunResults,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    redshift = (
        pd.to_numeric(candidates["detection_redshift"], errors="coerce")
        if not candidates.empty
        else pd.Series(dtype=float)
    )
    return pd.DataFrame(
        {
            "event_id": candidates["event_id"] if not candidates.empty else [],
            "m1_source_msun": (
                pd.to_numeric(candidates["final_smbh_mass_msun"], errors="coerce")
                if not candidates.empty
                else []
            ),
            "m2_source_msun": (
                pd.to_numeric(candidates["mass_msun"], errors="coerce")
                if not candidates.empty
                else []
            ),
            "detection_redshift": redshift,
            "luminosity_distance_mpc": (
                pd.to_numeric(candidates["luminosity_distance_mpc"], errors="coerce")
                if not candidates.empty
                else []
            ),
            "lisa_snr": (
                pd.to_numeric(candidates["lisa_snr"], errors="coerce")
                if not candidates.empty
                else []
            ),
            "snr_status": (candidates["snr_status"] if not candidates.empty else []),
            "is_lisa_detectable": (
                _boolean_series(candidates["is_lisa_detectable"])
                if not candidates.empty
                else []
            ),
        }
    )


def _detector_status_counts(run: RunResults) -> pd.DataFrame:
    events = run.events
    candidate = _boolean_series(events["is_imri_candidate"])
    detection_time = pd.to_numeric(events["detection_time_myr"], errors="coerce")
    snr_status = events["snr_status"].astype("string").str.strip()
    physical = candidate.eq(True) & np.isfinite(detection_time)
    categories = pd.Series(
        "not_applicable_not_imri_candidate", index=events.index, dtype="object"
    )
    categories[candidate.eq(True) & ~physical] = "not_evaluated_upstream"
    categories[physical & snr_status.eq(SNR_STATUS_EVALUATED)] = (
        "lisa_snr_below_threshold"
    )
    categories[physical & snr_status.eq(SNR_STATUS_NO_OVERLAP)] = (
        "no_lisa_frequency_overlap"
    )
    categories[physical & snr_status.eq(SNR_STATUS_REQUIRED_INPUT_UNAVAILABLE)] = (
        "required_input_unavailable"
    )
    categories[physical & snr_status.eq(SNR_STATUS_MODEL_UNAVAILABLE)] = (
        "model_unavailable"
    )
    categories[physical & snr_status.eq(SNR_STATUS_NUMERICAL_FAILURE)] = (
        "numerical_failure"
    )
    detectable = physical & _boolean_series(events["is_lisa_detectable"]).eq(True)
    categories[detectable] = "detectable"
    counts = categories.value_counts()
    return pd.DataFrame(
        {
            "detector_status": list(DETECTOR_STATUS_ORDER),
            "count": [int(counts.get(status, 0)) for status in DETECTOR_STATUS_ORDER],
        }
    )


def _legend_panel_figure(
    title: str,
    *,
    main_ratio: float = 4.0,
    legend_ratio: float = 1.35,
    figsize: tuple[float, float] = (11.0, 5.4),
) -> tuple[Any, Any, Any]:
    """Build a main axes plus a dedicated legend-only panel.

    Keeping the legend out of the data box prevents the legend marker glyphs
    (for example scatter dots) from being mistaken for real data points.
    """
    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(main_ratio, legend_ratio),
        wspace=0.04,
    )
    axis = fig.add_subplot(grid[0, 0])
    legend_axis = fig.add_subplot(grid[0, 1])
    legend_axis.axis("off")
    fig.suptitle(title, fontsize=13)
    axis.yaxis.grid(True, color="0.88", linewidth=0.8)
    axis.xaxis.grid(False)
    axis.tick_params(labelsize=11)
    return fig, axis, legend_axis


def _render_legend_panel(
    axis: Any,
    legend_axis: Any,
    *,
    fontsize: float = 9.0,
    handles: Sequence[Any] | None = None,
    labels: Sequence[str] | None = None,
) -> None:
    """Move the main axes legend into the dedicated upper-right panel."""
    if handles is None:
        handles, labels = axis.get_legend_handles_labels()
    if not handles:
        return
    if labels is None:
        labels = [handle.get_label() for handle in handles]
    legend_axis.legend(
        handles,
        labels,
        loc="upper right",
        frameon=False,
        fontsize=fontsize,
    )


PRIMARY_MASS_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">")


def _primary_mass_styles(
    primary_mass_msun: np.ndarray,
) -> dict[float, tuple[str, str]]:
    """Assign deterministic marker and color pairs to primary masses."""
    masses = sorted({float(value) for value in primary_mass_msun})
    return {
        mass: (
            PRIMARY_MASS_MARKERS[index % len(PRIMARY_MASS_MARKERS)],
            PRIMARY_MASS_COLORS[index % len(PRIMARY_MASS_COLORS)],
        )
        for index, mass in enumerate(masses)
    }


def _scatter_primary_mass(
    axis: Any,
    x: np.ndarray,
    y: np.ndarray,
    primary_mass_msun: np.ndarray,
    detectable: np.ndarray,
    *,
    size: float = 34.0,
) -> dict[float, tuple[str, str]]:
    """Scatter points: marker/color encode mass; fill encodes detectability.

    Filled markers are detector-detectable and open markers are evaluated but
    below threshold, so detectability is not conflated with primary mass.
    """
    styles = _primary_mass_styles(primary_mass_msun)
    for mass, (marker, color) in styles.items():
        mass_mask = np.isclose(primary_mass_msun, mass)
        for is_detectable in (True, False):
            mask = mass_mask & (detectable == is_detectable)
            if not mask.any():
                continue
            axis.scatter(
                x[mask],
                y[mask],
                marker=marker,
                s=size,
                facecolors=color if is_detectable else "white",
                edgecolors=color,
                linewidths=0.8 if is_detectable else 1.3,
                zorder=3,
            )
    return styles


def _primary_mass_legend_handles(
    styles: dict[float, tuple[str, str]],
    *,
    include_detectability: bool = True,
) -> list[Any]:
    from matplotlib.lines import Line2D

    handles: list[Any] = [
        Line2D(
            [],
            [],
            marker=marker,
            linestyle="None",
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=7,
            label=rf"$M_\bullet={mass:.0e}\,M_\odot$",
        )
        for mass, (marker, color) in styles.items()
    ]
    if include_detectability:
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="None",
                markerfacecolor="0.25",
                markeredgecolor="0.25",
                markersize=7,
                label="detectable (filled)",
            )
        )
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="None",
                markerfacecolor="white",
                markeredgecolor="0.25",
                markersize=7,
                label="below threshold (open)",
            )
        )
    return handles


def _figure_lisa(run: RunResults, figure_root: Path) -> dict[str, list[Path] | Path]:
    parameters = run.metadata["parameters"]
    mission_duration_years = float(parameters["lisa_mission_duration_years"])
    detector_min_hz = float(parameters["lisa_min_gw_frequency_hz"])
    detector_max_hz = float(parameters["lisa_max_gw_frequency_hz"])
    snr_threshold = float(parameters["snr_threshold"])
    lisa_dir = figure_root / "lisa"
    candidates = _candidate_events(run)
    outputs: dict[str, list[Path] | Path] = {}

    tracks = _candidate_tracks(
        candidates,
        default_mission_duration_years=mission_duration_years,
        detector_min_hz=detector_min_hz,
        detector_max_hz=detector_max_hz,
    )
    noise = _noise_curve(mission_duration_years, detector_min_hz, detector_max_hz)
    outputs["lisa_candidate_track_data"] = save_plot_data(
        tracks, lisa_dir / "data" / "characteristic_strain_candidates.csv"
    )
    outputs["lisa_noise_data"] = save_plot_data(
        noise, lisa_dir / "data" / "characteristic_noise.csv"
    )
    fig, axis, legend_axis = _legend_panel_figure(
        "LISA characteristic strain of intrinsic IMRI candidates"
    )
    axis.plot(
        noise["frequency_observer_hz"],
        noise["h_n"],
        color="black",
        linewidth=2.0,
        label=r"LISA noise $h_n=\sqrt{fS_n}$",
    )
    if not tracks.empty:
        primary_masses = sorted(
            pd.to_numeric(tracks["m1_source_msun"], errors="coerce").dropna().unique()
        )
        for mass_index, primary_mass_msun in enumerate(primary_masses):
            mass_tracks = tracks[
                np.isclose(tracks["m1_source_msun"], primary_mass_msun)
            ]
            color = PRIMARY_MASS_COLORS[mass_index % len(PRIMARY_MASS_COLORS)]
            linestyle = CANDIDATE_TRACK_LINESTYLES[
                mass_index % len(CANDIDATE_TRACK_LINESTYLES)
            ]
            for track_index, (_, track) in enumerate(
                mass_tracks.groupby("candidate_id", sort=True)
            ):
                axis.plot(
                    track["frequency_observer_hz"],
                    track["h_c_eff"],
                    color=color,
                    linestyle=linestyle,
                    linewidth=CANDIDATE_TRACK_LINEWIDTH,
                    alpha=CANDIDATE_TRACK_ALPHA,
                    label=(
                        rf"$M_\bullet={primary_mass_msun:.0e}\,M_\odot$ candidates"
                        if track_index == 0
                        else None
                    ),
                )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("observer-frame frequency [Hz]")
    axis.set_ylabel(r"characteristic strain")
    _render_legend_panel(axis, legend_axis)
    outputs["lisa_characteristic_strain_figures"] = save_figure(
        fig, lisa_dir / "characteristic_strain_candidates"
    )

    summary = _lisa_candidate_summary(run, candidates)
    outputs["lisa_candidate_summary_data"] = save_plot_data(
        summary, lisa_dir / "data" / "candidate_summary.csv"
    )
    snr_values = pd.to_numeric(summary["lisa_snr"], errors="coerce").to_numpy(
        dtype=float
    )
    redshift = pd.to_numeric(summary["detection_redshift"], errors="coerce").to_numpy(
        dtype=float
    )
    m1 = pd.to_numeric(summary["m1_source_msun"], errors="coerce").to_numpy(dtype=float)
    detectable = summary["is_lisa_detectable"].fillna(False).astype(bool).to_numpy()
    fig, axis, legend_axis = _legend_panel_figure(
        "Integrated LISA SNR versus detection redshift by primary mass"
    )
    if len(summary):
        styles = _scatter_primary_mass(axis, redshift, snr_values, m1, detectable)
        from matplotlib.lines import Line2D

        handles = _primary_mass_legend_handles(styles)
        handles.append(
            Line2D(
                [],
                [],
                color="black",
                linestyle="--",
                label=rf"threshold $\rho={snr_threshold:g}$",
            )
        )
        _render_legend_panel(axis, legend_axis, handles=handles)
    axis.axhline(snr_threshold, color="black", linewidth=1.0, linestyle="--")
    axis.set_xlabel("detection redshift")
    axis.set_ylabel(r"integrated LISA SNR $\rho$")
    outputs["lisa_snr_redshift_figures"] = save_figure(
        fig, lisa_dir / "snr_vs_detection_redshift"
    )

    m2 = pd.to_numeric(summary["m2_source_msun"], errors="coerce").to_numpy(dtype=float)
    fig, axis, legend_axis = _legend_panel_figure(
        "IMRI mass versus detection redshift by primary mass"
    )
    if len(summary):
        styles = _scatter_primary_mass(axis, redshift, m2, m1, detectable)
        _render_legend_panel(
            axis,
            legend_axis,
            handles=_primary_mass_legend_handles(styles),
        )
    axis.set_xlabel("detection redshift")
    axis.set_ylabel(r"migrator mass $m_2$ [$M_\odot$]")
    axis.set_yscale("log")
    outputs["lisa_mass_redshift_figures"] = save_figure(
        fig, lisa_dir / "mass_redshift_detectability"
    )

    status_counts = _detector_status_counts(run)
    outputs["lisa_status_data"] = save_plot_data(
        status_counts, lisa_dir / "data" / "detection_status_summary.csv"
    )
    fig, axis = new_figure("Detector classification status summary")
    positions = np.arange(len(status_counts), dtype=float)
    axis.bar(
        positions,
        status_counts["count"].to_numpy(dtype=float),
        color=[
            DETECTOR_STATUS_COLORS[status]
            for status in status_counts["detector_status"]
        ],
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [DETECTOR_STATUS_LABELS[status] for status in status_counts["detector_status"]],
        rotation=30,
        ha="right",
        fontsize=8,
    )
    axis.set_ylabel("events")
    axis.set_yscale("symlog", linthresh=1.0)
    outputs["lisa_status_figures"] = save_figure(
        fig, lisa_dir / "detection_status_summary"
    )
    outputs["lisa_caption"] = _write_caption(
        lisa_dir / CAPTION_FILENAME,
        "LISA candidate figure captions",
        [
            (
                "characteristic_strain_candidates",
                "The representative subset uses the same number of lowest-detection-"
                "redshift candidates per available primary-mass bin; color and line "
                "style encode primary mass. Formal detectability uses the integrated "
                "production SNR threshold, not visual crossing of the "
                "characteristic-strain curves.",
            ),
            (
                "snr_vs_detection_redshift",
                "One marker and color pair is assigned to each primary (SMBH) "
                "mass because the primary mass sets the observed ISCO and strongly "
                "influences the SNR. Filled markers are "
                "detector-detectable and open markers are evaluated below "
                "threshold. Points reflect the persisted production SNR; entries "
                "with non-finite or unavailable SNR are omitted rather than "
                "plotted as zero.",
            ),
            (
                "mass_redshift_detectability",
                "Marker type and color encode primary (SMBH) mass; fill encodes "
                "persisted detectability (filled = detectable, open = below "
                "threshold). Grid coverage is sparse, so "
                "apparent gaps are sampling, not absence of sources.",
            ),
            (
                "detection_status_summary",
                "Unavailable, no-overlap, and below-threshold states remain "
                "distinct and are never plotted as physical zeros.",
            ),
        ],
    )
    return outputs


def generate_production_figures(
    results_dir: str | Path,
    output_dir: str | Path | None = None,
    selected_nsc_mass_msun: float | None = None,
) -> dict[str, list[Path] | Path]:
    """Generate the production figure set for one persisted scalar-redshift run."""
    run = load_run_results(results_dir)
    figure_root = (
        Path(output_dir) if output_dir is not None else Path(results_dir) / "figures"
    )
    outputs: dict[str, list[Path] | Path] = {}
    outputs.update(
        {
            f"rates_{name}": value
            for name, value in generate_run_figures(
                results_dir,
                figure_root / "rates",
                selected_nsc_mass_msun,
            ).items()
        }
    )
    outputs.update(_figure_disk(run, figure_root))
    outputs.update(_figure_population(run, figure_root))
    outputs.update(_figure_migration(run, figure_root))
    outputs.update(_figure_entry(run, figure_root))
    outputs.update(_figure_lisa(run, figure_root))
    return outputs
