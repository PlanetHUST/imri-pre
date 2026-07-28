"""Persistence and visualization for TDE-rate simulations."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .config import (
    AVERAGED_RESULTS_FILENAME,
    COMBINED_TIMESCALE_HISTOGRAM_FILENAME,
    COPY_TIMESCALE_HISTOGRAM_FILENAME,
    METADATA_FILENAME,
    PRIMARY_TIMESCALE_HISTOGRAM_FILENAME,
    RATE_HEATMAP_FILENAME,
    RATE_VS_SMBH_PLOT_FILENAME,
    TIMESCALES_FILENAME,
)
from .config import TDEDynConfig
from .models import (
    TDEOrbitalParams,
    TDEDynAveragedResult,
    TDEDynSingleResult,
)

RESULT_COLUMNS = (
    "m_smbh_msun",
    "m_nsc_msun",
    "tde_mass_primary_mean_msun",
    "tde_mass_total_mean_msun",
    "tde_mass_rate_total_mean_msun_per_myr",
    "tde_mass_rate_total_min_msun_per_myr",
    "tde_mass_rate_total_max_msun_per_myr",
    "tde_star_count_total_mean",
)
TDE_EVENT_COLUMNS = (
    "m_smbh_msun",
    "m_nsc_msun",
    "repeat",
    "task_kind",
    "event_time_myr",
    "event_mass_msun",
    "cumulative_tde_mass_msun",
)
TDE_MASS_RATE_VS_TIME_PLOT_FILENAME = "tde_mass_rate_vs_time.png"
TDE_MASS_RATE_PRIMARY_PLOT_FILENAME = "tde_mass_rate_vs_time_primary.png"
TDE_MASS_RATE_COPIED_PLOT_FILENAME = "tde_mass_rate_vs_time_copied.png"
ARTICLE_PRESENTATION_ONLY = True
ARTICLE_FIGURE_FILENAMES = {
    COMBINED_TIMESCALE_HISTOGRAM_FILENAME,
    "tde_hist_mass.png",
}

PRESENTATION_LABEL_FONTSIZE = 24
PRESENTATION_TICK_FONTSIZE = 20
PRESENTATION_LEGEND_FONTSIZE = 18
PRESENTATION_LINE_WIDTH = 3.0
PRESENTATION_REFERENCE_LINE_WIDTH = 2.4
PRESENTATION_SPINE_LINE_WIDTH = 1.3
PRESENTATION_TICK_WIDTH = 1.2
PRESENTATION_GRID_LINE_WIDTH = 0.8
PRESENTATION_SAVE_DPI = 300
PRESENTATION_TIGHT_PAD = 1.5
PRESENTATION_FONT_FAMILY = "Arial"
PRESENTATION_SERIF_FONTS = ["Times New Roman"]
PRESENTATION_MATH_FONTSET = "stix"

EDDINGTON_LUMINOSITY_ERG_S_PER_MSUN = 1.26e38
ACCRETION_EFFICIENCY = 0.1
SPEED_OF_LIGHT_CM_S = 2.99792458e10
GRAMS_PER_MSUN = 1.98847e33
SECONDS_PER_YEAR = 365.25 * 24.0 * 60.0 * 60.0
ORBITAL_PARAMS_FILENAME = "orbital_params.csv"
ORBITAL_PARAM_COLUMNS = (
    "m_smbh_msun",
    "m_nsc_msun",
    "repeat",
    "task_kind",
    "event_time_myr",
    "semi_major_axis_cm",
    "eccentricity",
    "inclination_rad",
    "mass_msun",
)
ORBITAL_PLOT_SPECS = {
    "semi_major_axis_cm": (
        "tde_pdf_sma.png",
        "tde_hist_sma.png",
        r"Initial $a\,[{\rm cm}]$",
        None,
    ),
    "eccentricity": (
        "tde_pdf_ecc.png",
        "tde_hist_ecc.png",
        r"Initial eccentricity $e$",
        (1e-4, 1.0),
    ),
    "inclination_rad": (
        "tde_pdf_inc.png",
        "tde_hist_inc.png",
        r"Initial inclination $i\,[{\rm rad}]$",
        (1e-5, 1e1),
    ),
    "mass_msun": (
        "tde_pdf_mass.png",
        "tde_hist_mass.png",
        r"Initial stellar mass $m_\star\,[M_\odot]$",
        (1e-1, 1e2),
    ),
}
INTERMEDIATE_DIRECTORY = "intermediate"
INTERMEDIATE_POPULATION_SPECS = {
    "mass_msun": (
        "mass",
        r"Initial stellar mass $m_\star\,[M_\odot]$",
        True,
    ),
    "formation_time_myr": (
        "formation_time",
        r"Formation time $[{\rm Myr}]$",
        True,
    ),
    "semi_major_axis_cm": (
        "sma",
        r"Initial $a\,[{\rm cm}]$",
        True,
    ),
    "eccentricity": (
        "ecc",
        r"Initial eccentricity $e$",
        True,
    ),
    "inclination_rad": (
        "inc",
        r"Initial inclination $i\,[{\rm rad}]$",
        True,
    ),
}
INTERMEDIATE_HEATMAP_SPECS = (
    (
        "mass_formation_time",
        "mass_msun",
        "formation_time_myr",
        r"Initial stellar mass $m_\star\,[M_\odot]$",
        r"Formation time $[{\rm Myr}]$",
    ),
    (
        "sma_ecc",
        "semi_major_axis_cm",
        "eccentricity",
        r"Initial $a\,[{\rm cm}]$",
        r"Initial eccentricity $e$",
    ),
)


def _is_article_figure(filename: str) -> bool:
    return filename in ARTICLE_FIGURE_FILENAMES


def _should_write_figure(filename: str) -> bool:
    return (not ARTICLE_PRESENTATION_ONLY) or _is_article_figure(filename)


def _apply_presentation_font_style() -> None:
    if not ARTICLE_PRESENTATION_ONLY:
        return
    # plt.rcParams.update(
    #     {
    #         "text.usetex": True,
    #         "font.family": PRESENTATION_FONT_FAMILY,
    #         "font.serif": PRESENTATION_SERIF_FONTS,
    #         "mathtext.fontset": PRESENTATION_MATH_FONTSET,
    #         # "axes.labelweight": "bold",
    #         # "font.weight": "bold",
    #     }
    # )
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "Arial",
            "font.size": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 14,
            "axes.titlesize": 16,
        }
    )


def _apply_presentation_axes(axis: Any) -> None:
    axis.tick_params(
        axis="both",
        which="major",
        labelsize=PRESENTATION_TICK_FONTSIZE,
        width=PRESENTATION_TICK_WIDTH,
    )
    axis.tick_params(
        axis="both",
        which="minor",
        width=PRESENTATION_TICK_WIDTH,
    )
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(PRESENTATION_SPINE_LINE_WIDTH)


def _hide_irrelevant_article_figures(output_dir: Path) -> None:
    if not ARTICLE_PRESENTATION_ONLY or not output_dir.exists():
        return
    hidden_dir = output_dir / "hidden"
    for path in sorted(output_dir.glob("*.png")):
        if _is_article_figure(path.name):
            continue
        hidden_dir.mkdir(parents=True, exist_ok=True)
        destination = hidden_dir / path.name
        counter = 1
        while destination.exists():
            destination = hidden_dir / f"{path.stem}_{counter}{path.suffix}"
            counter += 1
        path.replace(destination)
        print(f"hidden: {path} -> {destination}")


def save_results(
    averaged_results: Sequence[TDEDynAveragedResult],
    output_dir: Path,
) -> pd.DataFrame:
    """Save averaged results with units encoded in column names."""
    frame = pd.DataFrame(
        [result.to_dict() for result in averaged_results],
        columns=RESULT_COLUMNS,
    )
    path = output_dir / AVERAGED_RESULTS_FILENAME
    frame.to_csv(path, index=False, float_format="%.6e")
    print(f"saved: {path}")
    return frame


def save_timescales(
    averaged_results: Sequence[TDEDynAveragedResult],
    output_dir: Path,
) -> pd.DataFrame:
    """Save one row per TDE event, ordered by occurrence time."""
    records: list[dict[str, float | int | str]] = []
    for averaged_result in averaged_results:
        for result in averaged_result.results:
            events = [
                (event_time, event_mass, task_kind)
                for times, masses, task_kind in (
                    (
                        result.tde_times_primary_myr,
                        result.tde_masses_primary_msun,
                        "primary",
                    ),
                    (
                        result.tde_times_copies_myr,
                        result.tde_masses_copies_msun,
                        "copy",
                    ),
                )
                for event_time, event_mass in zip(times, masses, strict=True)
            ]
            cumulative_mass_msun = 0.0
            for event_time, event_mass, task_kind in sorted(events):
                cumulative_mass_msun += event_mass
                records.append(
                    {
                        "m_smbh_msun": averaged_result.m_smbh_msun,
                        "m_nsc_msun": averaged_result.m_nsc_msun,
                        "repeat": result.repeat_idx,
                        "task_kind": task_kind,
                        "event_time_myr": event_time,
                        "event_mass_msun": event_mass,
                        "cumulative_tde_mass_msun": cumulative_mass_msun,
                    }
                )
    frame = pd.DataFrame.from_records(records, columns=TDE_EVENT_COLUMNS)
    path = output_dir / TIMESCALES_FILENAME
    frame.to_csv(path, index=False, float_format="%.6e")
    print(f"saved: {path}")
    return frame


def save_orbital_params(
    averaged_results: Sequence[TDEDynAveragedResult],
    output_dir: Path,
    *,
    append: bool = False,
) -> pd.DataFrame:
    """Save initial properties for stars that become TDEs."""
    records: list[dict[str, float | int | str]] = []
    for averaged_result in averaged_results:
        for result in averaged_result.results:
            base = {
                "m_smbh_msun": averaged_result.m_smbh_msun,
                "m_nsc_msun": averaged_result.m_nsc_msun,
                "repeat": result.repeat_idx,
            }
            for event_time_myr, orbital_params in _iter_tde_orbital_params(result):
                records.append(
                    {
                        **base,
                        "event_time_myr": event_time_myr,
                        **orbital_params,
                    }
                )
    frame = pd.DataFrame.from_records(records, columns=ORBITAL_PARAM_COLUMNS)
    path = output_dir / ORBITAL_PARAMS_FILENAME
    write_header = not append or not path.exists()
    frame.to_csv(
        path,
        index=False,
        float_format="%.6e",
        mode="a" if append else "w",
        header=write_header,
    )
    print(f"saved: {path}")
    return frame


def _iter_tde_orbital_params(
    result: TDEDynSingleResult,
) -> list[tuple[float, TDEOrbitalParams]]:
    primary = zip(
        result.tde_times_primary_myr,
        (
            params
            for params in result.tde_orbital_params
            if params["task_kind"] == "primary"
        ),
        strict=False,
    )
    copies = zip(
        result.tde_times_copies_myr,
        (
            params
            for params in result.tde_orbital_params
            if params["task_kind"] == "copy"
        ),
        strict=False,
    )
    return [*primary, *copies]


def save_intermediate_population_plots(
    repeat_results: Sequence[TDEDynSingleResult],
    output_dir: Path,
    *,
    m_smbh_msun: float,
    m_nsc_msun: float,
) -> None:
    """Save repeat-averaged sampled-population diagnostics for one grid point."""
    intermediate_dir = output_dir / INTERMEDIATE_DIRECTORY
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    tag = f"smbh_{m_smbh_msun:.1e}_nsc_{m_nsc_msun:.1e}".replace("+", "")
    for column, (
        short_name,
        xlabel,
        use_log_x,
    ) in INTERMEDIATE_POPULATION_SPECS.items():
        path = intermediate_dir / f"{short_name}_{tag}.png"
        _plot_intermediate_distribution(
            _collect_intermediate_values(repeat_results, "initial", column),
            _collect_intermediate_values(repeat_results, "disk_crossing", column),
            xlabel,
            path,
            use_log_x=use_log_x,
        )
    for short_name, x_column, y_column, xlabel, ylabel in INTERMEDIATE_HEATMAP_SPECS:
        initial_x = _collect_intermediate_values(repeat_results, "initial", x_column)
        initial_y = _collect_intermediate_values(repeat_results, "initial", y_column)
        disk_crossing_x = _collect_intermediate_values(
            repeat_results,
            "disk_crossing",
            x_column,
        )
        disk_crossing_y = _collect_intermediate_values(
            repeat_results,
            "disk_crossing",
            y_column,
        )
        x_bins = _shared_intermediate_bins(initial_x, disk_crossing_x)
        y_bins = _shared_intermediate_bins(initial_y, disk_crossing_y)
        _plot_intermediate_heatmap(
            initial_x,
            initial_y,
            xlabel,
            ylabel,
            intermediate_dir / f"{short_name}_initial_{tag}.png",
            x_bins=x_bins,
            y_bins=y_bins,
        )
        _plot_intermediate_heatmap(
            disk_crossing_x,
            disk_crossing_y,
            xlabel,
            ylabel,
            intermediate_dir / f"{short_name}_disk_crossing_{tag}.png",
            x_bins=x_bins,
            y_bins=y_bins,
        )


def _collect_intermediate_values(
    repeat_results: Sequence[TDEDynSingleResult],
    subset: str,
    column: str,
) -> list[np.ndarray]:
    values_by_repeat: list[np.ndarray] = []
    for result in repeat_results:
        population = (
            result.intermediate_initial_population
            if subset == "initial"
            else result.intermediate_disk_crossing_population
        )
        if population is None:
            values_by_repeat.append(np.asarray([], dtype=float))
            continue
        values_by_repeat.append(np.asarray(population[column], dtype=float))
    return values_by_repeat


def _plot_intermediate_distribution(
    initial_values_by_repeat: Sequence[np.ndarray],
    disk_crossing_values_by_repeat: Sequence[np.ndarray],
    xlabel: str,
    output_path: Path,
    *,
    use_log_x: bool,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    initial_values = _finite_intermediate_values(initial_values_by_repeat, use_log_x)
    disk_crossing_values = _finite_intermediate_values(
        disk_crossing_values_by_repeat,
        use_log_x,
    )
    combined_values = (
        np.concatenate([initial_values, disk_crossing_values])
        if initial_values.size or disk_crossing_values.size
        else np.asarray([], dtype=float)
    )
    if combined_values.size:
        bins = _intermediate_bins(combined_values, use_log_x, bins=30)
        _plot_mean_step_histogram(
            axis,
            initial_values_by_repeat,
            bins,
            "all sampled",
            use_log_x=use_log_x,
        )
        _plot_mean_step_histogram(
            axis,
            disk_crossing_values_by_repeat,
            bins,
            "disk crossing",
            use_log_x=use_log_x,
        )
        if use_log_x:
            axis.set_xscale("log")
        axis.set_yscale("log")
        axis.legend(frameon=False)
    else:
        axis.text(
            0.5,
            0.5,
            "No positive finite data",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Mean number of stars per bin")
    # axis.set_title("Initial population diagnostics")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"saved: {output_path}")


def _plot_intermediate_heatmap(
    x_values_by_repeat: Sequence[np.ndarray],
    y_values_by_repeat: Sequence[np.ndarray],
    xlabel: str,
    ylabel: str,
    output_path: Path,
    *,
    x_bins: np.ndarray,
    y_bins: np.ndarray,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    finite_pairs = [
        _finite_intermediate_pair(x_values, y_values)
        for x_values, y_values in zip(
            x_values_by_repeat,
            y_values_by_repeat,
            strict=True,
        )
    ]
    if x_bins.size and y_bins.size:
        histograms = []
        for x_values, y_values in finite_pairs:
            if x_values.size == 0:
                histograms.append(np.zeros((len(x_bins) - 1, len(y_bins) - 1)))
                continue
            histogram, _, _ = np.histogram2d(
                x_values,
                y_values,
                bins=(x_bins, y_bins),
            )
            histograms.append(histogram)
        mean_histogram = np.mean(histograms, axis=0)
        mesh = axis.pcolormesh(
            x_bins,
            y_bins,
            mean_histogram.T,
            shading="auto",
        )
        figure.colorbar(mesh, ax=axis, label="Mean number of stars per bin")
        axis.set_xscale("log")
        axis.set_yscale("log")
    else:
        axis.text(
            0.5,
            0.5,
            "No positive finite data",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    # axis.set_title("Initial population diagnostics")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"saved: {output_path}")


def _finite_intermediate_values(
    values_by_repeat: Sequence[np.ndarray],
    positive_only: bool,
) -> np.ndarray:
    finite_values = []
    for values in values_by_repeat:
        selected = values[np.isfinite(values)]
        if positive_only:
            selected = selected[selected > 0.0]
        finite_values.append(selected)
    if not finite_values:
        return np.asarray([], dtype=float)
    return np.concatenate(finite_values)


def _finite_intermediate_pair(
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask = (
        np.isfinite(x_values)
        & np.isfinite(y_values)
        & (x_values > 0.0)
        & (y_values > 0.0)
    )
    return x_values[mask], y_values[mask]


def _intermediate_bins(
    values: np.ndarray,
    use_log_x: bool,
    *,
    bins: int,
) -> np.ndarray:
    value_min = float(values.min())
    value_max = float(values.max())
    if value_min == value_max:
        value_min /= 1.1
        value_max *= 1.1
    if use_log_x:
        return np.geomspace(value_min, value_max, bins + 1)
    return np.linspace(value_min, value_max, bins + 1)


def _shared_intermediate_bins(
    first_values_by_repeat: Sequence[np.ndarray],
    second_values_by_repeat: Sequence[np.ndarray],
) -> np.ndarray:
    first_values = _finite_intermediate_values(first_values_by_repeat, True)
    second_values = _finite_intermediate_values(second_values_by_repeat, True)
    combined_values = (
        np.concatenate([first_values, second_values])
        if first_values.size or second_values.size
        else np.asarray([], dtype=float)
    )
    if not combined_values.size:
        return np.asarray([], dtype=float)
    return _intermediate_bins(combined_values, True, bins=30)


def _plot_mean_step_histogram(
    axis: Any,
    values_by_repeat: Sequence[np.ndarray],
    bins: np.ndarray,
    label: str,
    *,
    use_log_x: bool,
) -> None:
    histograms = []
    for values in values_by_repeat:
        selected = values[np.isfinite(values)]
        if use_log_x:
            selected = selected[selected > 0.0]
        histogram, _ = np.histogram(selected, bins=bins)
        histograms.append(histogram)
    if not histograms:
        return
    mean_histogram = np.mean(histograms, axis=0)
    step_x = np.repeat(bins, 2)[1:-1]
    step_y = np.repeat(mean_histogram, 2)
    axis.plot(
        step_x,
        step_y,
        drawstyle="steps-post",
        linewidth=1.6,
        label=label,
    )


def create_orbital_param_pdfs_from_csv(
    orbital_params_csv: Path,
    output_dir: Path,
) -> None:
    """Plot repeat-median PDFs of disrupted stars' initial properties."""
    _apply_presentation_font_style()
    _hide_irrelevant_article_figures(output_dir)
    frame = pd.read_csv(orbital_params_csv)
    if "stage" in frame.columns:
        frame = frame[frame["stage"] == "tde"]
    metadata_path = orbital_params_csv.parent / METADATA_FILENAME
    num_repeats = 0
    if metadata_path.exists():
        with metadata_path.open() as metadata_file:
            parameters = json.load(metadata_file).get("parameters", {})
        num_repeats = int(parameters.get("num_repeats", 0))
    if num_repeats <= 0 and not frame.empty:
        num_repeats = int(frame["repeat"].max()) + 1
    output_dir.mkdir(parents=True, exist_ok=True)
    for column, (
        pdf_filename,
        hist_filename,
        xlabel,
        value_range,
    ) in ORBITAL_PLOT_SPECS.items():
        if _should_write_figure(pdf_filename):
            _create_orbital_param_plot(
                frame,
                column,
                xlabel,
                value_range,
                num_repeats,
                output_dir / pdf_filename,
                normalized_per_log_bin=False,
            )
        if _should_write_figure(hist_filename):
            _create_orbital_param_plot(
                frame,
                column,
                xlabel,
                value_range,
                num_repeats,
                output_dir / hist_filename,
                normalized_per_log_bin=True,
            )


def _create_orbital_param_plot(
    frame: pd.DataFrame,
    column: str,
    xlabel: str,
    value_range: tuple[float, float] | None,
    num_repeats: int,
    output_path: Path,
    *,
    normalized_per_log_bin: bool,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 6))
    positive = frame.loc[np.isfinite(frame[column]) & (frame[column] > 0.0)]
    plotted = False
    bin_edges: np.ndarray | None = None
    bin_centers: np.ndarray | None = None
    if not positive.empty:
        values = positive[column].to_numpy(dtype=float)
        value_min, value_max = value_range or (values.min(), values.max())
        if value_min == value_max:
            value_min /= 10.0**0.05
            value_max *= 10.0**0.05
        bin_edges = np.geomspace(
            value_min, value_max, 21 if normalized_per_log_bin else 31
        )
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    for (m_smbh_msun, m_nsc_msun), grid_frame in positive.groupby(
        ["m_smbh_msun", "m_nsc_msun"]
    ):
        if bin_edges is None or bin_centers is None:
            continue
        label = (
            rf"$M_{{\rm BH}}={m_smbh_msun:.1e}, " rf"M_{{\rm NSC}}={m_nsc_msun:.1e}$"
        )
        if normalized_per_log_bin:
            histogram, _ = np.histogram(
                grid_frame[column].to_numpy(dtype=float),
                bins=bin_edges,
                density=False,
            )
            total = histogram.sum()
            normalized_count_per_log_bin = (
                histogram / (total * np.diff(np.log(bin_edges)))
                if total
                else np.zeros(len(bin_edges) - 1)
            )
            axis.stairs(
                normalized_count_per_log_bin,
                bin_edges,
                linewidth=PRESENTATION_LINE_WIDTH,
                label=label,
            )
            plotted = True
            continue
        repeat_pdfs = []
        for repeat in range(num_repeats):
            repeat_values = grid_frame.loc[
                grid_frame["repeat"] == repeat, column
            ].to_numpy(dtype=float)
            histogram, _ = np.histogram(repeat_values, bins=bin_edges, density=False)
            total = histogram.sum()
            repeat_pdfs.append(
                (
                    histogram / (total * np.diff(bin_edges))
                    if total
                    else np.zeros(len(bin_edges) - 1)
                )
            )
        pdfs = np.asarray(repeat_pdfs)
        median = np.median(pdfs, axis=0)
        lower = np.quantile(pdfs, 0.16, axis=0)
        upper = np.quantile(pdfs, 0.84, axis=0)
        (line,) = axis.plot(
            bin_centers,
            median,
            linewidth=PRESENTATION_LINE_WIDTH,
            label=label,
        )
        axis.fill_between(
            bin_centers, lower, upper, color=line.get_color(), alpha=0.2, linewidth=0.0
        )
        plotted = True
    if column == "mass_msun":
        _plot_kroupa_reference(
            axis,
            value_min=1e-1,
            value_max=1e2,
            normalized_per_log_bin=normalized_per_log_bin,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    if value_range is not None:
        axis.set_xlim(*value_range)
    axis.set_ylim([1e-4, 1e0])
    axis.set_xlabel(xlabel, fontsize=PRESENTATION_LABEL_FONTSIZE)
    axis.set_ylabel(
        "Normalized number" if normalized_per_log_bin else "Probability density",
        fontsize=PRESENTATION_LABEL_FONTSIZE,
    )
    # axis.set_title("Initial properties of tidally disrupted stars")
    axis.grid(
        True,
        which="both",
        alpha=0.3,
        linestyle=":",
        linewidth=PRESENTATION_GRID_LINE_WIDTH,
    )
    _apply_presentation_axes(axis)
    if plotted:
        axis.legend(frameon=True, fontsize=PRESENTATION_LEGEND_FONTSIZE)
    else:
        axis.text(
            0.5,
            0.5,
            "No positive TDE data",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    figure.tight_layout(pad=PRESENTATION_TIGHT_PAD)
    figure.savefig(output_path, dpi=PRESENTATION_SAVE_DPI, bbox_inches="tight")
    plt.close(figure)
    print(f"saved: {output_path}")


def _plot_kroupa_reference(
    axis: Any,
    value_min: float,
    value_max: float,
    *,
    normalized_per_log_bin: bool,
) -> None:
    """Overlay the canonical broken-power-law Kroupa initial mass function."""
    mass = np.geomspace(value_min, value_max, 400)
    # d_number_d_mass = np.where(
    #     mass < 0.5,
    #     mass**-1.3,
    #     0.5 * mass**-2.3,
    # )
    d_number_d_mass = mass**-2.3
    if normalized_per_log_bin:
        reference = mass * d_number_d_mass
        reference /= _trapezoidal_integral(reference, np.log(mass))
    else:
        reference = d_number_d_mass / _trapezoidal_integral(d_number_d_mass, mass)
    axis.plot(
        mass,
        reference,
        "k--",
        linewidth=PRESENTATION_REFERENCE_LINE_WIDTH,
        label="Kroupa IMF",
    )


def _trapezoidal_integral(values: np.ndarray, coordinates: np.ndarray) -> float:
    """Integrate without depending on NumPy's version-specific aliases."""
    return float(np.sum(0.5 * (values[:-1] + values[1:]) * np.diff(coordinates)))


def print_timescale_statistics(timescales_csv: Path) -> pd.DataFrame:
    """Print TDE event statistics read from ``timescales.csv``."""
    frame = pd.read_csv(timescales_csv)
    print("\nTDE event statistics")
    rows = []
    for task_kind in ("primary", "copy", "total"):
        selected = (
            frame if task_kind == "total" else frame[frame["task_kind"] == task_kind]
        )
        row = {
            "task_kind": task_kind,
            "event_count": len(selected),
            "tde_mass_msun": selected["event_mass_msun"].sum(),
        }
        rows.append(row)
        print(
            f"  {task_kind}: events={row['event_count']}, "
            f"mass={row['tde_mass_msun']:.3e} Msun"
        )
    return pd.DataFrame(rows)


def create_timescale_histograms_from_csv(
    timescales_csv: Path,
    output_dir: Path,
) -> None:
    """Create median cumulative-mass profiles across repeated runs."""
    _apply_presentation_font_style()
    _hide_irrelevant_article_figures(output_dir)
    frame = pd.read_csv(timescales_csv)
    duty_cycle_myr, num_repeats = _read_output_sampling(
        timescales_csv,
        frame,
    )
    mass_grid = _read_output_mass_grid(timescales_csv, frame)
    time_grid_myr = np.linspace(0.0, duty_cycle_myr, 201)

    def create_figure(
        selected: pd.DataFrame,
        title: str,
        filename: str,
        show_mass_rate: bool = False,
    ) -> None:
        if not _should_write_figure(filename):
            return
        figure, axis = plt.subplots(figsize=(10, 6))
        plotted = False
        has_positive_mass = False
        positive_values: list[np.ndarray] = []
        for grid_point in mass_grid.itertuples(index=False):
            grid_events = selected[
                (selected["m_smbh_msun"] == grid_point.m_smbh_msun)
                & (selected["m_nsc_msun"] == grid_point.m_nsc_msun)
            ]
            median_mass, lower_mass, upper_mass = _cumulative_mass_statistics(
                grid_events,
                time_grid_myr,
                num_repeats,
            )
            if show_mass_rate:
                elapsed_time_yr = time_grid_myr * 1.0e6
                median_mass = np.divide(
                    median_mass,
                    elapsed_time_yr,
                    out=np.zeros_like(median_mass),
                    where=elapsed_time_yr > 0.0,
                )
                lower_mass = np.divide(
                    lower_mass,
                    elapsed_time_yr,
                    out=np.zeros_like(lower_mass),
                    where=elapsed_time_yr > 0.0,
                )
                upper_mass = np.divide(
                    upper_mass,
                    elapsed_time_yr,
                    out=np.zeros_like(upper_mass),
                    where=elapsed_time_yr > 0.0,
                )
                eddington_rate_msun_per_yr = (
                    EDDINGTON_LUMINOSITY_ERG_S_PER_MSUN
                    * grid_point.m_smbh_msun
                    / (ACCRETION_EFFICIENCY * SPEED_OF_LIGHT_CM_S**2)
                    * SECONDS_PER_YEAR
                    / GRAMS_PER_MSUN
                )
                endpoint_eddington_ratio = median_mass[-1] / eddington_rate_msun_per_yr
            (line,) = axis.plot(
                time_grid_myr,
                median_mass,
                linewidth=PRESENTATION_LINE_WIDTH,
                label=(
                    (
                        f"{grid_point.m_smbh_msun:.0e}, "
                        f"{grid_point.m_nsc_msun:.0e}: "
                        rf"${endpoint_eddington_ratio:.2g}\,\dot{{M}}_{{\rm Edd}}$"
                    )
                    if show_mass_rate
                    else (
                        rf"$M_{{\rm BH}}={grid_point.m_smbh_msun:.1e}$, "
                        rf"$M_{{\rm NSC}}={grid_point.m_nsc_msun:.1e}$"
                    )
                ),
            )
            if not show_mass_rate:
                axis.fill_between(
                    time_grid_myr,
                    lower_mass,
                    upper_mass,
                    color=line.get_color(),
                    alpha=0.2,
                    linewidth=0.0,
                )
            plotted = True
            has_positive_mass = has_positive_mass or bool(np.any(upper_mass > 0.0))
            positive_values.append(median_mass[median_mass > 0.0])

        if not plotted:
            axis.text(
                0.5,
                0.5,
                "No positive finite values",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        axis.set_xlabel(
            r"TDE event time $t\,[{\rm Myr}]$",
            fontsize=PRESENTATION_LABEL_FONTSIZE,
        )
        axis.set_ylabel(
            (
                r"TDE mass rate $M_{\rm TDE}(<t)/t$ " r"$[M_\odot\,{\rm yr}^{-1}]$"
                if show_mass_rate
                else r"Cumulative TDE mass $[M_\odot]$"
            ),
            fontsize=PRESENTATION_LABEL_FONTSIZE,
        )
        axis.set_xlim(0.0, duty_cycle_myr)
        if has_positive_mass:
            axis.set_yscale("log")
        if show_mass_rate and positive_values:
            positive_rates = np.concatenate(positive_values)
            if positive_rates.size:
                axis.set_ylim(
                    positive_rates.min() / 1.5,
                    positive_rates.max() * 1.5,
                )
        # axis.set_title(title, fontsize=18, pad=14)
        axis.grid(
            True,
            which="both",
            alpha=0.5,
            linestyle=":",
            linewidth=PRESENTATION_GRID_LINE_WIDTH,
        )
        _apply_presentation_axes(axis)
        # if plotted:
        #     axis.legend(
        #         loc="lower right",
        #         frameon=True,
        #         fontsize=PRESENTATION_LEGEND_FONTSIZE,
        #         ncols=2 if show_mass_rate else 1,
        #         title=(
        #             (
        #                 r"$M_{\rm BH}, M_{\rm NSC}\ [M_\odot]$; "
        #                 r"rate at $t_{\rm duty}$"
        #             )
        #             if show_mass_rate
        #             else None
        #         ),
        #     )
        figure.tight_layout(pad=PRESENTATION_TIGHT_PAD)
        path = output_dir / filename
        figure.savefig(path, dpi=PRESENTATION_SAVE_DPI, bbox_inches="tight")
        plt.close(figure)
        print(f"saved: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    create_figure(
        frame.loc[frame["task_kind"] == "primary"],
        "Cumulative TDE mass: primary stars",
        PRIMARY_TIMESCALE_HISTOGRAM_FILENAME,
    )
    create_figure(
        frame.loc[frame["task_kind"] == "copy"],
        "Cumulative TDE mass: copied stars",
        COPY_TIMESCALE_HISTOGRAM_FILENAME,
    )
    create_figure(
        frame.loc[frame["task_kind"].isin(("primary", "copy"))],
        "Cumulative TDE mass: primary and copied stars",
        COMBINED_TIMESCALE_HISTOGRAM_FILENAME,
    )
    create_figure(
        frame.loc[frame["task_kind"].isin(("primary", "copy"))],
        r"Time-dependent TDE mass rate $M_{\rm TDE}(<t)/t$",
        TDE_MASS_RATE_VS_TIME_PLOT_FILENAME,
        show_mass_rate=True,
    )
    create_figure(
        frame.loc[frame["task_kind"] == "primary"],
        r"Primary-star TDE mass rate $M_{\rm TDE}(<t)/t$",
        TDE_MASS_RATE_PRIMARY_PLOT_FILENAME,
        show_mass_rate=True,
    )
    create_figure(
        frame.loc[frame["task_kind"] == "copy"],
        r"Copied-star TDE mass rate $M_{\rm TDE}(<t)/t$",
        TDE_MASS_RATE_COPIED_PLOT_FILENAME,
        show_mass_rate=True,
    )


def _read_output_sampling(
    timescales_csv: Path,
    frame: pd.DataFrame,
) -> tuple[float, int]:
    metadata_path = timescales_csv.parent / METADATA_FILENAME
    if metadata_path.exists():
        with metadata_path.open() as metadata_file:
            parameters = json.load(metadata_file).get("parameters", {})
        duty_cycle_myr = float(parameters.get("duty_cycle_myr", 0.0))
        num_repeats = int(parameters.get("num_repeats", 0))
    else:
        duty_cycle_myr = 0.0
        num_repeats = 0
    if duty_cycle_myr <= 0.0:
        maximum_event_time_myr = (
            float(frame["event_time_myr"].max())
            if not frame.empty and "event_time_myr" in frame
            else 0.0
        )
        duty_cycle_myr = max(maximum_event_time_myr, 1.0)
    if num_repeats <= 0:
        num_repeats = (
            int(frame["repeat"].max()) + 1
            if not frame.empty and "repeat" in frame
            else 1
        )
    return duty_cycle_myr, num_repeats


def _read_output_mass_grid(
    timescales_csv: Path,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    averaged_results_path = timescales_csv.parent / AVERAGED_RESULTS_FILENAME
    if averaged_results_path.exists():
        averaged_results = pd.read_csv(
            averaged_results_path,
            usecols=["m_smbh_msun", "m_nsc_msun"],
        )
        return averaged_results.drop_duplicates()
    required_columns = {"m_smbh_msun", "m_nsc_msun"}
    if not required_columns.issubset(frame.columns):
        return pd.DataFrame(columns=sorted(required_columns))
    return frame[["m_smbh_msun", "m_nsc_msun"]].drop_duplicates()


def _cumulative_mass_statistics(
    events: pd.DataFrame,
    time_grid_myr: np.ndarray,
    num_repeats: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cumulative_curves = np.zeros((num_repeats, time_grid_myr.size))
    for repeat, repeat_events in events.groupby("repeat"):
        repeat_index = int(repeat)
        if repeat_index < 0 or repeat_index >= num_repeats:
            continue
        repeat_events = repeat_events.sort_values("event_time_myr")
        event_times = repeat_events["event_time_myr"].to_numpy(dtype=float)
        cumulative_mass = repeat_events["event_mass_msun"].cumsum().to_numpy()
        event_indices = np.searchsorted(event_times, time_grid_myr, side="right")
        has_event = event_indices > 0
        cumulative_curves[repeat_index, has_event] = cumulative_mass[
            event_indices[has_event] - 1
        ]
    return (
        np.median(cumulative_curves, axis=0),
        np.quantile(cumulative_curves, 0.16, axis=0),
        np.quantile(cumulative_curves, 0.84, axis=0),
    )


def save_metadata(
    output_dir: Path,
    config: TDEDynConfig,
    runtime_stats: dict[str, float | int | str],
) -> None:
    """Save simulation configuration and runtime statistics."""
    metadata: dict[str, Any] = {
        "simulation_info": {
            "date": datetime.now().isoformat(),
            "timestamp_s_since_epoch": time.time(),
            "description": (
                "Dynamical TDE/IMRI simulation with unified parallel damping"
            ),
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
            "nsc_radius_pc_at_reference_mass": (config.nsc_radius_pc_at_reference_mass),
            "nsc_radius_reference_mass_msun": (config.nsc_radius_reference_mass_msun),
            "nsc_radius_mass_slope": config.nsc_radius_mass_slope,
            "dyn_resupply": config.dyn_resupply,
            "max_copies_per_object": config.max_copies_per_object,
            "reuse_same_orbit_for_copies": (config.reuse_same_orbit_for_copies),
            "tde_radius_rg": config.tde_radius_rg,
            "direct_tde": config.direct_tde,
            "consider_type_ii_migration": config.consider_type_ii_migration,
            "imri_lisa_min_gw_frequency_hz": (
                config.imri_lisa_min_gw_frequency_hz
            ),
            "imri_lisa_max_gw_frequency_hz": (
                config.imri_lisa_max_gw_frequency_hz
            ),
            "imri_max_gw_time_myr": config.imri_max_gw_time_myr,
            "imri_gw_eccentricity": config.imri_gw_eccentricity,
            "damping_cpu_count": config.damping_cpu_count,
            "checkpoint_grid_interval": (config.checkpoint_grid_interval),
            "create_visualizations": config.create_visualizations,
            "create_intermediate_plots": config.create_intermediate_plots,
            "print_debug_results": config.print_debug_results,
        },
        "runtime_statistics": runtime_stats,
    }
    path = output_dir / METADATA_FILENAME
    with path.open("w") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, default=str)
    print(f"saved: {path}")


def create_visualizations_from_csv(csv_file: Path, output_dir: Path) -> None:
    """Create result-oriented TDE-rate plots."""
    if ARTICLE_PRESENTATION_ONLY:
        _hide_irrelevant_article_figures(output_dir)
        print("rate-grid visualizations skipped in article-presentation mode")
        return
    frame = pd.read_csv(csv_file)
    if frame.empty:
        print("No data to visualize")
        return

    rate_column = "tde_mass_rate_total_mean_msun_per_myr"
    frame = frame.groupby(["m_smbh_msun", "m_nsc_msun"], as_index=False)[
        rate_column
    ].mean()

    smbh_masses_msun = np.sort(frame["m_smbh_msun"].unique())
    nsc_masses_msun = np.sort(frame["m_nsc_msun"].unique())
    rate_grid = frame.pivot(
        index="m_smbh_msun",
        columns="m_nsc_msun",
        values=rate_column,
    ).values

    figure, axis = plt.subplots(figsize=(10, 8))
    for nsc_mass_msun in nsc_masses_msun:
        subset = frame[frame["m_nsc_msun"] == nsc_mass_msun].sort_values("m_smbh_msun")
        axis.plot(
            subset["m_smbh_msun"],
            subset[rate_column],
            "o-",
            label=f"M_NSC={nsc_mass_msun:.1e}",
        )
    axis.set_xscale("log")
    if np.any(frame[rate_column] > 0.0):
        axis.set_yscale("log")
    axis.set_xlabel(r"$M_{\rm SMBH}\,[M_\odot]$")
    axis.set_ylabel(r"TDE mass rate $[M_\odot/{\rm Myr}]$")
    # axis.set_title("Dynamical TDE mass rate vs SMBH mass")
    axis.grid(True, alpha=0.3, linestyle="--")
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        output_dir / RATE_VS_SMBH_PLOT_FILENAME,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.pcolormesh(
        smbh_masses_msun,
        nsc_masses_msun,
        rate_grid.T,
        shading="auto",
        norm="log" if np.any(rate_grid > 0.0) else None,
        cmap="viridis",
    )
    _configure_heatmap_axes(axis, smbh_masses_msun, nsc_masses_msun)
    # axis.set_title(r"Dynamical TDE mass rate $[M_\odot/{\rm Myr}]$")
    figure.colorbar(image)
    figure.tight_layout()
    figure.savefig(
        output_dir / RATE_HEATMAP_FILENAME,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"visualizations saved to: {output_dir}")


def _configure_heatmap_axes(
    axis: Any,
    smbh_masses_msun: np.ndarray,
    nsc_masses_msun: np.ndarray,
) -> None:
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"$M_{\rm SMBH}\,[M_\odot]$")
    axis.set_ylabel(r"$M_{\rm NSC}\,[M_\odot]$")
    if smbh_masses_msun.size > 1:
        axis.set_xlim(smbh_masses_msun.min(), smbh_masses_msun.max())
    if nsc_masses_msun.size > 1:
        axis.set_ylim(nsc_masses_msun.min(), nsc_masses_msun.max())
