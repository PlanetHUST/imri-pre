"""Disk-subsystem diagnostic: pAGN profiles and the physical Q=1 boundary.

This harness calls the production disk model directly.  It never runs the full
workflow, and it uses only controlled SMBH masses.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

from disk_quantity.pagn_disk_adapter import build_pagn_disk_profile
from workflows.config import DISK_ACCRETION_RATE_G_S_PER_MSUN, PAGN_ALPHA
from workflows.rate_plots import new_figure, save_figure, save_plot_data

SMBH_MASSES_MSUN = (1.0e5, 1.0e6, 1.0e7)
OUTPUT_DIR = Path("diagnostics/disk")
TOOMRE_BOUNDARY = 1.0
Q1_TOLERANCE = 0.05


def _profile_frame(mass_msun: float) -> tuple[pd.DataFrame, float, float]:
    profile = build_pagn_disk_profile(
        mass_msun,
        PAGN_ALPHA,
        accretion_rate_g_s=DISK_ACCRETION_RATE_G_S_PER_MSUN * mass_msun,
    )
    radius_cm = np.asarray(profile.radius_cm, dtype=float)
    sigma = np.asarray(profile.surface_density_g_cm2, dtype=float)
    aspect = np.asarray(profile.aspect_ratio, dtype=float)
    q_values = np.asarray(profile.toomre_q, dtype=float)
    rg_cm = float(profile.gravitational_radius_cm)

    finite = np.isfinite(radius_cm) & np.isfinite(sigma) & np.isfinite(aspect) & np.isfinite(q_values)
    assert finite.any(), f"mass {mass_msun:g}: no finite disk samples"
    positive = finite & (radius_cm > 0.0) & (sigma > 0.0) & (aspect > 0.0) & (q_values > 0.0)
    assert positive.all(), (
        f"mass {mass_msun:g}: disk profiles must be finite and positive "
        "(surface density, aspect ratio, Toomre Q)"
    )
    interior = radius_cm <= float(profile.outer_radius_cm) * (1.0 + 1.0e-12)
    assert q_values[interior].max() > TOOMRE_BOUNDARY, (
        f"mass {mass_msun:g}: no Q > 1 disk interior"
    )
    assert q_values[interior].min() <= TOOMRE_BOUNDARY * (1.0 + Q1_TOLERANCE), (
        f"mass {mass_msun:g}: outer boundary does not reach Q = 1"
    )

    frame = pd.DataFrame(
        {
            "m_smbh_msun": mass_msun,
            "r_cm": radius_cm,
            "r_over_rg": radius_cm / rg_cm,
            "surface_density_g_cm2": sigma,
            "aspect_ratio": aspect,
            "toomre_q": q_values,
        }
    )
    return frame, float(profile.outer_radius_cm / rg_cm), float(profile.outer_radius_cm)


def _plot(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    masses = sorted({float(value) for value in frame["m_smbh_msun"].unique()})
    colors = list(plt.get_cmap("tab10")(np.linspace(0.0, 1.0, max(len(masses), 1))))
    fig, axes = new_figure(
        "pAGN disk profiles and the Q=1 outer boundary",
        nrows=3,
        ncols=1,
        figsize=(7.0, 11.0),
    )
    panels = (
        ("surface_density_g_cm2", r"$\Sigma\;[\mathrm{g\,cm^{-2}}]$"),
        ("aspect_ratio", r"$H/R$"),
        ("toomre_q", r"$Q$"),
    )
    for row, (column, label) in enumerate(panels):
        axis = axes[row, 0]
        for mass, color in zip(masses, colors, strict=False):
            selected = frame[np.isclose(frame["m_smbh_msun"], mass)]
            axis.plot(
                selected["r_over_rg"],
                selected[column],
                color=color,
                linewidth=2.0,
                label=rf"$M_\bullet={mass:.1e}\,M_\odot$",
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_ylabel(label)
        axis.set_xlabel(r"$R/r_g$")
        axis.yaxis.grid(True, color="0.9", linewidth=0.8)
        if column == "toomre_q":
            axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
        axis.legend(frameon=False, fontsize=8, loc="best")
    return save_figure(fig, output_dir / "disk_profiles")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="diagnostic figure and data directory",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    frames = []
    print("Disk subsystem diagnostic")
    print("  mass_msun     r_out_rg      r_out_cm")
    for mass_msun in SMBH_MASSES_MSUN:
        frame, outer_rg, outer_cm = _profile_frame(mass_msun)
        frames.append(frame)
        print(f"  {mass_msun:11.6e}  {outer_rg:12.6e}  {outer_cm:12.6e}")
    combined = pd.concat(frames, ignore_index=True)
    data_path = save_plot_data(combined, arguments.output_dir / "data" / "disk_profiles.csv")
    figures = _plot(combined, arguments.output_dir)
    print(f"  data: {data_path}")
    for figure in figures:
        print(f"  figure: {figure}")
    print("Disk diagnostic passed.")


if __name__ == "__main__":
    main()
