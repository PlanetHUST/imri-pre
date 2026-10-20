"""Validate the fixed migration contract on real pAGN disk solutions.

This diagnostic reports, rather than hides, cells where the Grishin-2024
linear thermal theory is excluded or the simple Kanagawa depth fit is being
extrapolated. It also compares signed production transport with the explicit
inward-only no-trap counterfactual. It writes one compact row per central mass.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from physical_process import constants as const  # noqa: E402
from disk_quantity.pagn_disk_adapter import build_pagn_disk_profile  # noqa: E402
from physical_process.migration import (  # noqa: E402
    MigrationParameters,
    classify_torque_traps,
    compose_migration_torque_si,
    compute_torque_profile,
    inward_only_counterfactual_torque,
    torque_migration_outcome,
    torque_si_to_cgs,
)
from workflows.config import (  # noqa: E402
    DISK_ACCRETION_RATE_G_S_PER_MSUN,
    MIGRATION_CONTRACT,
    PAGN_ALPHA,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smbh-masses",
        type=float,
        nargs="+",
        default=(1.0e3, 1.0e4, 1.0e5, 1.0e6),
        help="central masses [Msun]",
    )
    parser.add_argument(
        "--migrator-mass",
        type=float,
        default=10.0,
        help="embedded-object mass [Msun]",
    )
    parser.add_argument("--resolution", type=int, default=800)
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="destination for the persisted validation table",
    )
    return parser


def _finite_median(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if finite.size else float("nan")


def validate_mass(
    smbh_mass_msun: float,
    migrator_mass_msun: float,
    resolution: int,
) -> dict[str, float | int | str]:
    profile = build_pagn_disk_profile(
        smbh_mass_msun,
        PAGN_ALPHA,
        n_resolution=resolution,
        accretion_rate_g_s=DISK_ACCRETION_RATE_G_S_PER_MSUN * smbh_mass_msun,
        suppress_pagn_output=True,
    )
    torque = compute_torque_profile(
        profile.disk,
        migrator_mass_msun,
        None,
        params=MigrationParameters(alpha=PAGN_ALPHA),
    )
    recomposed = compose_migration_torque_si(
        torque.gamma_type_i_si,
        torque.gamma_th_si,
        torque.gap_depth_fraction,
        torque.gamma_secondary_dominated_si,
    )
    if not np.array_equal(torque.gamma_total_si, recomposed):
        raise AssertionError("production total torque does not match its contract")
    if not np.isfinite(torque.gamma_total_si).all():
        raise AssertionError("production total torque contains non-finite values")
    if not ((torque.gap_depth_fraction > 0.0).all()
            and (torque.gap_depth_fraction <= 1.0).all()):
        raise AssertionError("Kanagawa gap depth is outside (0, 1]")

    traps = classify_torque_traps(torque.radius_rg, torque.gamma_total_si)
    production_torque_cgs = torque_si_to_cgs(torque.gamma_total_si)
    no_trap_torque_cgs = inward_only_counterfactual_torque(
        production_torque_cgs
    )
    path_arguments = (
        torque.radius_cm,
        migrator_mass_msun * const.msun,
        float(torque.radius_cm[-2]),
        float(torque.radius_cm[1]),
        smbh_mass_msun,
    )
    production_outcome = torque_migration_outcome(
        path_arguments[0],
        production_torque_cgs,
        *path_arguments[1:],
    )
    no_trap_outcome = torque_migration_outcome(
        path_arguments[0],
        no_trap_torque_cgs,
        *path_arguments[1:],
    )
    if no_trap_outcome.status != "finite":
        raise AssertionError("inward-only counterfactual must have finite delivery")
    return {
        "migration_contract": MIGRATION_CONTRACT,
        "smbh_mass_msun": float(smbh_mass_msun),
        "migrator_mass_msun": float(migrator_mass_msun),
        "radial_cells": int(torque.radius_rg.size),
        "finite_total_fraction": float(np.isfinite(torque.gamma_total_si).mean()),
        "outward_torque_fraction": float((torque.gamma_total_si > 0.0).mean()),
        "stable_traps": int(traps.stable_radius_rg.size),
        "outermost_stable_trap_rg": float(traps.outermost_stable_radius_rg),
        "production_path_status": production_outcome.status,
        "production_path_time_myr": float(production_outcome.time_myr),
        "no_trap_path_time_myr": float(no_trap_outcome.time_myr),
        "deep_gap_fraction": float(torque.deep_gap.mean()),
        "kanagawa_extrapolated_fraction": float(
            torque.kanagawa_extrapolated.mean()
        ),
        "thermal_regime_valid_fraction": float(
            torque.thermal_torque_regime_valid.mean()
        ),
        "thermal_applied_fraction": float(torque.thermal_validity_weight.mean()),
        "wkb_invalid_fraction": float(torque.wkb_invalid.mean()),
        "median_disk_dominance_b": _finite_median(torque.disk_dominance_b),
        "median_gap_depth_fraction": _finite_median(torque.gap_depth_fraction),
        "median_hill_over_height": _finite_median(
            torque.hill_radius_over_scale_height
        ),
        "median_bondi_over_height": _finite_median(
            torque.bondi_radius_over_scale_height
        ),
        "median_thermal_length_over_height": _finite_median(
            torque.thermal_length_over_scale_height
        ),
        "median_effective_optical_depth": _finite_median(
            torque.effective_optical_depth
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.migrator_mass <= 0.0:
        raise ValueError("migrator mass must be positive")
    if arguments.resolution < 4:
        raise ValueError("resolution must be at least four")
    if any(mass <= 0.0 for mass in arguments.smbh_masses):
        raise ValueError("SMBH masses must be positive")

    table = pd.DataFrame.from_records(
        validate_mass(mass, arguments.migrator_mass, arguments.resolution)
        for mass in arguments.smbh_masses
    )
    arguments.output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(arguments.output_csv, index=False, float_format="%.17g")
    print(table.to_string(index=False))
    print(f"saved: {arguments.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
