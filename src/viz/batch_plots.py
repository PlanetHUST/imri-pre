"""Interface-validation figures for one merger-tree batch run.

Only two figures are produced, because only two answer an interface question:

1. ``selected_hosts_domain.png`` - which merger-tree hosts were actually
   handed to the local simulation (resolved SMBH mass versus redshift, coloured
   by execution status);
2. ``batch_status.png`` - how many jobs ended in each execution state, keeping
   "evaluated" separate from "could not be evaluated".

No cosmic-rate quantity is plotted.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from workflows.single_halo import JOB_STATUSES, STATUS_SUCCESS  # noqa: E402

FIGURES_DIRNAME = "figures"

# Categorical slots in fixed order (blue, orange, aqua, yellow, magenta, green,
# violet, red); status colours never double as series colours.
SERIES_COLORS = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)
SUCCESS_COLOR = "#2a78d6"
NOT_EVALUATED_COLOR = "#9a9891"
TEXT_COLOR = "#0b0b0b"
MUTED_COLOR = "#52514e"
GRID_COLOR = "#e4e2dc"


def _style_axes(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(GRID_COLOR)
    axis.spines["bottom"].set_color(GRID_COLOR)
    axis.tick_params(colors=MUTED_COLOR, labelsize=9)
    axis.grid(True, color=GRID_COLOR, linewidth=0.6)
    axis.set_axisbelow(True)


def plot_selected_hosts_domain(job_summary: pd.DataFrame, path: Path) -> Path:
    """Scatter the resolved SMBH mass versus redshift for every job."""
    figure, axis = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    redshift = pd.to_numeric(job_summary["redshift"], errors="coerce")
    smbh = pd.to_numeric(job_summary["smbh_mass_msun"], errors="coerce")
    status = job_summary["status"].astype(str)

    evaluated = status == STATUS_SUCCESS
    for mask, color, label in (
        (~evaluated, NOT_EVALUATED_COLOR, "not evaluated"),
        (evaluated, SUCCESS_COLOR, "evaluated (success)"),
    ):
        if mask.any():
            axis.scatter(
                redshift[mask],
                smbh[mask],
                s=28,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                label=f"{label} (n={int(mask.sum())})",
            )
    finite_smbh = smbh[np.isfinite(smbh) & (smbh > 0)]
    if len(finite_smbh) and finite_smbh.max() / finite_smbh.min() > 10.0:
        axis.set_yscale("log")
    else:
        axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useOffset=False)
    axis.ticklabel_format(axis="x", useOffset=False)
    axis.set_xlabel("host redshift z (tree column z)", color=MUTED_COLOR)
    axis.set_ylabel("resolved SMBH mass [Msun]", color=MUTED_COLOR)
    axis.set_title(
        "Merger-tree hosts handed to the single-halo kernel",
        color=TEXT_COLOR,
        fontsize=11,
        loc="left",
    )
    _style_axes(axis)
    axis.legend(frameon=False, fontsize=8, loc="best")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_batch_status(status_counts: dict[str, int], path: Path) -> Path:
    """Horizontal bar chart of job execution states."""
    labels = [status for status in JOB_STATUSES if status_counts.get(status, 0) > 0]
    if not labels:
        labels = [STATUS_SUCCESS]
    values = [status_counts.get(status, 0) for status in labels]
    colors = [
        SUCCESS_COLOR if status == STATUS_SUCCESS else NOT_EVALUATED_COLOR
        for status in labels
    ]
    figure, axis = plt.subplots(figsize=(6.4, 0.6 + 0.45 * len(labels)), dpi=150)
    positions = np.arange(len(labels))
    axis.barh(positions, values, color=colors, height=0.55)
    for position, value in zip(positions, values):
        axis.text(
            value, position, f" {value}", va="center", fontsize=9, color=TEXT_COLOR
        )
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=9, color=TEXT_COLOR)
    axis.invert_yaxis()
    axis.set_xlabel("jobs", color=MUTED_COLOR)
    axis.set_title(
        "Batch execution status (blue = physics evaluated)",
        color=TEXT_COLOR,
        fontsize=11,
        loc="left",
    )
    _style_axes(axis)
    axis.grid(True, axis="x", color=GRID_COLOR, linewidth=0.6)
    axis.grid(False, axis="y")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path


def write_batch_figures(
    output_root: Path,
    job_summary: pd.DataFrame,
    status_counts: dict[str, int],
) -> list[Path]:
    """Write the two batch interface figures and return their paths."""
    figures_dir = Path(output_root) / FIGURES_DIRNAME
    figures_dir.mkdir(parents=True, exist_ok=True)
    written = []
    if len(job_summary.index):
        written.append(
            plot_selected_hosts_domain(
                job_summary, figures_dir / "selected_hosts_domain.png"
            )
        )
    written.append(plot_batch_status(status_counts, figures_dir / "batch_status.png"))
    return written


__all__ = ["plot_batch_status", "plot_selected_hosts_domain", "write_batch_figures"]
