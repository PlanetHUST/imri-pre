"""Calibrate the gap-opening threshold of the production criterion.

The production criterion is ``kanagawa``: a cell has opened a gap when

    K = q^2 / (alpha h^5)  >  K_threshold

This script establishes what ``K_threshold`` *means* in this model, what value
is defensible, and how sensitive the science is to it.  It uses only code and
data inside this repository - no external reference is consulted.

What it does
------------
1. **Identifies the criterion family.**  Verifies that the project's own Type-I
   gap denominator ``1 + K/kappa_norm`` is algebraically identical to the
   Kanagawa gap-depth law ``Sigma_gap/Sigma_0 = 1/(1 + 0.04 K)``.  That makes
   the threshold a statement about *residual surface density*, and lets any
   threshold be quoted as a physically interpretable depletion fraction.

2. **Tests the obvious alternative and rejects it.**  A switch could instead be
   placed where the two torque branches have equal magnitude, making the
   transition continuous.  The script measures that crossover across the
   SMBH/migrator mass grid; it spans several orders of magnitude in K, so no
   single global threshold can achieve continuity.  Continuity is therefore not
   an available calibration target, and the size of the residual jump is
   reported instead.

3. **Scans the threshold** over the depletion fractions a gap criterion could
   reasonably use, recording for each: gap-open fraction, surviving
   outward-torque cells, migration traps, and the torque discontinuity at the
   switch.

4. **Runs the end-to-end simulation** at each candidate threshold and records
   the physical outcome (migration status, IMRI candidates, local rate).

Outputs
-------
``output/round2/calibration_gap_threshold/``
    ``depth_identity.json``      the gap-depth identity check
    ``torque_crossover.csv``     where the branches actually cross
    ``threshold_scan.csv``       structure versus threshold
    ``end_to_end_scan.csv``      physical outcome versus threshold
    ``calibration.json``         the recommendation and its basis
    ``gap_threshold_calibration.png``

Run with::

    python scripts/calibrate_gap_threshold.py
    python scripts/calibrate_gap_threshold.py --quick   # skip the end-to-end scan
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from disk_quantity.pagn_disk_adapter import build_pagn_disk_profile  # noqa: E402
from physical_process.migration import (  # noqa: E402
    GapCriterion,
    MigrationParameters,
    classify_torque_traps,
    compute_torque_profile,
    gap_diagnostics,
    kanagawa_gap_parameter,
    torque_si_to_cgs,
)
from workflows.cli import build_config_from_arguments, build_parser  # noqa: E402
from workflows.config import DISK_ACCRETION_RATE_G_S_PER_MSUN, PAGN_ALPHA  # noqa: E402
from workflows.single_halo import STATUS_SUCCESS, run_single_halo  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "output" / "round2" / "calibration_gap_threshold"

SMBH_GRID_MSUN = (1.0e3, 1.0e4, 1.0e5, 1.0e6)
MIGRATOR_GRID_MSUN = (10.0, 30.0, 100.0)
CALIBRATION_MIGRATOR_MSUN = 10.0

#: Candidate thresholds, quoted by the residual surface density they imply.
#: K = (1/f - 1) / 0.04 for a depletion fraction f = Sigma_gap / Sigma_0.
CANDIDATE_DEPLETION_FRACTIONS = (0.9, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01)

# End-to-end scan configuration (cheap Pop III host, matched seed).
E2E_SMBH_MSUN = 1.0e5
E2E_NSC_MSUN = 5.0e4
E2E_REPEATS = 2
E2E_SEED = 777

TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e2dc"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7")


def depletion_to_threshold(fraction: float, kappa_norm: float) -> float:
    """Return the K at which the gap-depth law reaches ``fraction``."""
    return (1.0 / fraction - 1.0) * kappa_norm


def threshold_to_depletion(threshold: float, kappa_norm: float) -> float:
    """Return the residual surface-density fraction implied by ``threshold``."""
    return 1.0 / (1.0 + threshold / kappa_norm)


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _profiles(migrator_mass_msun: float) -> dict[float, Any]:
    cache: dict[float, Any] = {}
    for smbh_mass_msun in SMBH_GRID_MSUN:
        disk = build_pagn_disk_profile(
            smbh_mass_msun, PAGN_ALPHA,
            accretion_rate_g_s=DISK_ACCRETION_RATE_G_S_PER_MSUN * smbh_mass_msun)
        torque = compute_torque_profile(
            disk.disk, migrator_mass_msun=migrator_mass_msun,
            max_radius_rg=None, params=MigrationParameters(alpha=PAGN_ALPHA))
        aspect = np.interp(torque.radius_cm, disk.radius_cm, disk.aspect_ratio)
        cache[smbh_mass_msun] = (disk, torque, aspect)
    return cache


# ------------------------------------------------------------------ step 1
def check_depth_identity() -> dict[str, Any]:
    """The Type-I denominator *is* the Kanagawa gap-depth law."""
    params = MigrationParameters()
    K = np.logspace(-2, 4, 400)
    denominator = 1.0 + K / params.kappa_norm
    kanagawa_depth_denominator = 1.0 + 0.04 * K
    # 0.04 and 1/25 are the same number but not the same float, so compare to
    # floating-point precision rather than bit-exactly.
    deviation = float(np.max(np.abs(denominator / kanagawa_depth_denominator - 1.0)))
    identical = deviation < 1.0e-15
    payload = {
        "kappa_norm": params.kappa_norm,
        "kappa_t": params.kappa_t,
        "type_i_denominator_equals_kanagawa_gap_depth": identical,
        "max_relative_deviation": deviation,
        "implied_law": "Sigma_gap / Sigma_0 = 1 / (1 + 0.04 K)",
        "interpretation": (
            "The Type-I torque is scaled by the residual surface density, so a "
            "threshold on K is exactly a threshold on how far the gap has been "
            "cleared.  A threshold should therefore be quoted as a depletion "
            "fraction, not as a bare number."
        ),
        "threshold_for_depletion": {
            f"{f:g}": depletion_to_threshold(f, params.kappa_norm)
            for f in CANDIDATE_DEPLETION_FRACTIONS
        },
        "corotation_attenuation_at_threshold": {
            f"{f:g}": float(np.exp(-depletion_to_threshold(f, params.kappa_norm)
                                   / params.kappa_t))
            for f in CANDIDATE_DEPLETION_FRACTIONS
        },
    }
    print(f"  gap-depth identity holds: {identical} "
          f"(max relative deviation {deviation:.3g})")
    return payload


# ------------------------------------------------------------------ step 2
def measure_torque_crossover() -> pd.DataFrame:
    """Where do the Type-I and Type-II branches have equal magnitude?"""
    rows = []
    for migrator_mass_msun in MIGRATOR_GRID_MSUN:
        cache = _profiles(migrator_mass_msun)
        for smbh_mass_msun, (_disk, torque, aspect) in cache.items():
            mass_ratio = migrator_mass_msun / smbh_mass_msun
            K = kanagawa_gap_parameter(mass_ratio, aspect, PAGN_ALPHA)
            type_i = (torque.gamma_type_i_si
                      + (1.0 - torque.gap_attenuation_factor) * torque.gamma_th_si)
            ratio = np.abs(type_i) / np.abs(torque.gamma_type_ii_si)
            usable = np.isfinite(ratio) & (ratio > 0) & np.isfinite(K) & (K > 0)
            crossings = np.where(np.diff(np.sign(np.log(ratio[usable]))) != 0)[0]
            for index in crossings:
                rows.append({
                    "smbh_mass_msun": smbh_mass_msun,
                    "migrator_mass_msun": migrator_mass_msun,
                    "crossover_K": float(K[usable][index]),
                    "crossover_radius_rg": float(torque.radius_rg[usable][index]),
                })
    frame = pd.DataFrame(rows)
    if len(frame.index):
        span = frame["crossover_K"].max() / frame["crossover_K"].min()
        print(f"  torque-crossover K spans {span:.3g}x "
              f"({frame['crossover_K'].min():.3g} to {frame['crossover_K'].max():.3g})")
    return frame


# ------------------------------------------------------------------ step 3
def scan_threshold_structure() -> pd.DataFrame:
    """Gap fraction, surviving outward torque and traps versus threshold."""
    params = MigrationParameters()
    cache = _profiles(CALIBRATION_MIGRATOR_MSUN)
    rows = []
    for fraction in CANDIDATE_DEPLETION_FRACTIONS:
        threshold = depletion_to_threshold(fraction, params.kappa_norm)
        criterion = GapCriterion(name="kanagawa", kanagawa_threshold=threshold)
        for smbh_mass_msun, (_disk, torque, aspect) in cache.items():
            mass_ratio = CALIBRATION_MIGRATOR_MSUN / smbh_mass_msun
            diagnostics = gap_diagnostics(mass_ratio, aspect, PAGN_ALPHA, criterion)
            type_i = (torque.gamma_type_i_si
                      + (1.0 - torque.gap_attenuation_factor) * torque.gamma_th_si)
            selected = np.where(diagnostics.gap_open,
                                torque.gamma_type_ii_si, type_i)
            selected_cgs = torque_si_to_cgs(selected)
            traps = classify_torque_traps(torque.radius_rg, selected_cgs,
                                          min_separation_cells=1.0)
            switch = np.where(np.diff(diagnostics.gap_open.astype(int)) != 0)[0]
            jumps = [abs(torque.gamma_type_ii_si[i] - type_i[i]) / abs(type_i[i])
                     for i in switch if type_i[i] != 0.0]
            rows.append({
                "depletion_fraction": fraction,
                "threshold_K": threshold,
                "smbh_mass_msun": smbh_mass_msun,
                "gap_open_fraction": diagnostics.gap_open_fraction,
                "outward_torque_cells": int(np.count_nonzero(selected_cgs > 0)),
                "stable_traps": int(traps.stable_radius_rg.size),
                "anti_traps": int(traps.anti_trap_radius_rg.size),
                "switch_cells": int(switch.size),
                "max_relative_torque_jump": max(jumps) if jumps else np.nan,
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ step 4
def scan_end_to_end() -> pd.DataFrame:
    """Run the real simulation at each candidate threshold."""
    params = MigrationParameters()
    parser = build_parser()
    rows = []
    for fraction in CANDIDATE_DEPLETION_FRACTIONS:
        threshold = depletion_to_threshold(fraction, params.kappa_norm)
        run_dir = OUTPUT_DIR / "runs" / f"f{fraction:g}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        config = build_config_from_arguments(parser.parse_args([
            str(E2E_REPEATS), str(run_dir), "2",
            "--smbh-masses", repr(E2E_SMBH_MSUN),
            "--nsc-masses", repr(E2E_NSC_MSUN),
            "--base-seed", str(E2E_SEED),
            "--gap-threshold", repr(threshold),
        ]))
        result = run_single_halo(config, host=None, seed=E2E_SEED, output_dir=run_dir)
        row: dict[str, Any] = {
            "depletion_fraction": fraction,
            "threshold_K": threshold,
            "status": result.status,
        }
        if result.status == STATUS_SUCCESS:
            row.update({k: result.summary.get(k) for k in
                        ("n_disk_cross", "n_emb_bh", "n_bh_candi", "n_mig_eval",
                         "n_mig_finite", "n_mig_trap", "n_imri",
                         "rate_imri_per_myr", "n_det")})
        rows.append(row)
        print(f"  f={fraction:<5g} K={threshold:>9.4g}  "
              f"trap={row.get('n_mig_trap')} finite={row.get('n_mig_finite')} "
              f"imri={row.get('n_imri')} rate={row.get('rate_imri_per_myr')}")
    return pd.DataFrame(rows)


def plot_calibration(structure: pd.DataFrame, end_to_end: pd.DataFrame,
                     crossover: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12.6, 3.9), dpi=150)

    for index, (smbh_mass_msun, group) in enumerate(structure.groupby("smbh_mass_msun")):
        ordered = group.sort_values("threshold_K")
        axes[0].semilogx(ordered["threshold_K"], 100 * ordered["gap_open_fraction"],
                         marker="o", markersize=3.5, linewidth=1.6,
                         color=SERIES[index % len(SERIES)],
                         label=rf"$M_\bullet={smbh_mass_msun:.0e}$")
    axes[0].set_ylim(-3, 103)
    axes[0].set_xlabel(r"threshold $K_\ast$", color=MUTED, fontsize=9)
    axes[0].set_ylabel("disk in Type-II regime [%]", color=MUTED, fontsize=9)
    axes[0].set_title("Type-II extent versus threshold", color=TEXT, fontsize=10, loc="left")
    axes[0].legend(frameon=False, fontsize=7)

    for index, (smbh_mass_msun, group) in enumerate(structure.groupby("smbh_mass_msun")):
        ordered = group.sort_values("threshold_K")
        axes[1].semilogx(ordered["threshold_K"], ordered["stable_traps"],
                         marker="s", markersize=4, linewidth=1.6,
                         color=SERIES[index % len(SERIES)])
    axes[1].set_xlabel(r"threshold $K_\ast$", color=MUTED, fontsize=9)
    axes[1].set_ylabel("stable migration traps", color=MUTED, fontsize=9)
    axes[1].set_title("Trap survival versus threshold", color=TEXT, fontsize=10, loc="left")

    if len(end_to_end.index):
        ordered = end_to_end.sort_values("threshold_K")
        axes[2].semilogx(ordered["threshold_K"], ordered["n_mig_trap"],
                         marker="o", markersize=4, linewidth=1.8,
                         color=SERIES[1], label="trap-bound")
        axes[2].semilogx(ordered["threshold_K"], ordered["n_imri"],
                         marker="s", markersize=4, linewidth=1.8,
                         color=SERIES[0], label="IMRI candidates")
    axes[2].set_xlabel(r"threshold $K_\ast$", color=MUTED, fontsize=9)
    axes[2].set_ylabel("objects per realization", color=MUTED, fontsize=9)
    axes[2].set_title(rf"End-to-end, $M_\bullet=${E2E_SMBH_MSUN:.0e}$\,M_\odot$",
                      color=TEXT, fontsize=10, loc="left")
    axes[2].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(True, color=GRID, linewidth=0.6)
        axis.set_axisbelow(True)
        axis.tick_params(colors=MUTED, labelsize=8)
        axis.axvline(25.0, color=MUTED, linestyle="--", linewidth=1.0)
    figure.suptitle("Gap-threshold calibration (dashed line: K = 25, the 50 % depletion point)",
                    fontsize=11, x=0.01, ha="left")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true",
                        help="skip the end-to-end simulation scan")
    arguments = parser.parse_args(argv)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Criterion family")
    identity = check_depth_identity()
    with (OUTPUT_DIR / "depth_identity.json").open("w") as handle:
        json.dump(_safe(identity), handle, indent=2)

    print("2. Torque-continuity alternative")
    crossover = measure_torque_crossover()
    crossover.to_csv(OUTPUT_DIR / "torque_crossover.csv", index=False)

    print("3. Threshold scan (structure)")
    structure = scan_threshold_structure()
    structure.to_csv(OUTPUT_DIR / "threshold_scan.csv", index=False)

    end_to_end = pd.DataFrame()
    if not arguments.quick:
        print("4. Threshold scan (end-to-end simulation)")
        end_to_end = scan_end_to_end()
        end_to_end.to_csv(OUTPUT_DIR / "end_to_end_scan.csv", index=False)

    plot_calibration(structure, end_to_end, crossover,
                     OUTPUT_DIR / "gap_threshold_calibration.png")

    continuity_available = bool(
        len(crossover.index)
        and crossover["crossover_K"].max() / crossover["crossover_K"].min() < 3.0
    )
    calibration = {
        "criterion": "kanagawa",
        "criterion_family": "gap depth (residual surface density)",
        "basis": identity["implied_law"],
        "threshold_is_a_depletion_fraction": True,
        "continuity_calibration_available": continuity_available,
        "continuity_note": (
            "Rejected: the |Gamma_I| = |Gamma_II| crossover spans "
            f"{crossover['crossover_K'].max() / crossover['crossover_K'].min():.3g}x "
            "in K across the SMBH/migrator grid, so no single global threshold "
            "makes the Type-I -> Type-II switch continuous."
            if len(crossover.index) else "no crossover found"
        ),
        "recommended_threshold_K": 25.0,
        "recommended_depletion_fraction": 0.5,
        "recommendation_basis": (
            "K = 25 is the point at which this model's own gap-depth law, "
            "Sigma_gap/Sigma_0 = 1/(1+0.04K), reaches one half.  It is the only "
            "value in the family that is fixed by the model itself rather than "
            "chosen: it is the same kappa_norm the Type-I torque already uses.  "
            "No external calibration was possible offline."
        ),
        "residual_judgement": (
            "Whether 'gap open' should mean 50 per cent or 10 per cent depletion "
            "is a modelling choice this repository cannot settle.  The sensitivity "
            "tables record what changes if it is revised."
        ),
        "how_to_change": "--gap-threshold K   (or GapCriterion(kanagawa_threshold=K))",
    }
    with (OUTPUT_DIR / "calibration.json").open("w") as handle:
        json.dump(_safe(calibration), handle, indent=2)

    print("\nRecommendation:")
    print(f"  K_threshold = {calibration['recommended_threshold_K']} "
          f"(depletion fraction {calibration['recommended_depletion_fraction']})")
    print(f"  basis: {identity['implied_law']}")
    print(f"  continuity alternative: {'available' if continuity_available else 'REJECTED'}")
    print(f"  => {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
