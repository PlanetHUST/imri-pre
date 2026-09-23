"""Locate a simulated embedded-BH population in the fixed torque contract.

The report does not compare selectable migration prescriptions. For each SMBH
mass it measures how much of every object's gas-migration path lies outside the
calibrated Kanagawa or linear Grishin-2024 regimes, then derives traps directly
from the same total torque used by the workflow.

Usage: ``python scripts/analyse_population_torque.py RUN_DIR``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from disk_quantity.pagn_disk_adapter import build_pagn_disk_profile  # noqa: E402
from physical_process.migration import (  # noqa: E402
    MigrationParameters,
    classify_torque_traps,
    compute_torque_profile,
)
from workflows.config import (  # noqa: E402
    DISK_ACCRETION_RATE_G_S_PER_MSUN,
    MIGRATION_CONTRACT,
    PAGN_ALPHA,
)

RESOLUTION = 800


def _disk(smbh_mass_msun: float):
    return build_pagn_disk_profile(
        smbh_mass_msun,
        PAGN_ALPHA,
        n_resolution=RESOLUTION,
        accretion_rate_g_s=DISK_ACCRETION_RATE_G_S_PER_MSUN * smbh_mass_msun,
        suppress_pagn_output=True,
    )


def population_summary(events: pd.DataFrame) -> None:
    print("\n=== 1. Embedded / migrating population ===\n")
    print(f'{"M_smbh":>8} {"N":>5} | {"mass [Msun]":^27} | {"r_ini [rg]":^31}')
    print(
        f'{"":>8} {"":>5} | {"min":>8}{"med":>9}{"max":>10} '
        f'| {"min":>10}{"med":>10}{"max":>11}'
    )
    for smbh_mass, group in events.groupby("m_smbh_msun"):
        print(
            f"{smbh_mass:8.0e} {len(group):5d} | "
            f"{group.mass_msun.min():8.1f}{group.mass_msun.median():9.1f}"
            f"{group.mass_msun.max():10.1f} | "
            f"{group.r_ini_rg.min():10.2e}{group.r_ini_rg.median():10.2e}"
            f"{group.r_ini_rg.max():11.2e}"
        )


def validity_occupancy(events: pd.DataFrame) -> None:
    """Summarize regime flags on each object's actual radial path."""
    print("\n=== 2. Mean fraction of each gas path in flagged regimes ===\n")
    print(
        f'{"M_smbh":>8} | {"deep gap":>10}{"K>1e4":>10}'
        f'{"thermal invalid":>18}'
    )
    for smbh_mass, group in events.groupby("m_smbh_msun"):
        profile = _disk(float(smbh_mass))
        deep_gap: list[float] = []
        extrapolated: list[float] = []
        thermal_invalid: list[float] = []
        for row in group.itertuples(index=False):
            torque = compute_torque_profile(
                profile.disk,
                float(row.mass_msun),
                None,
                params=MigrationParameters(alpha=PAGN_ALPHA),
            )
            inner = min(float(row.r_gas_fin_rg), float(row.r_ini_rg))
            outer = max(float(row.r_gas_fin_rg), float(row.r_ini_rg))
            on_path = (torque.radius_rg >= inner) & (torque.radius_rg <= outer)
            if not on_path.any():
                continue
            deep_gap.append(float(torque.deep_gap[on_path].mean()))
            extrapolated.append(float(torque.kanagawa_extrapolated[on_path].mean()))
            thermal_invalid.append(
                float((~torque.thermal_torque_regime_valid[on_path]).mean())
            )
        print(
            f"{smbh_mass:8.0e} | {np.mean(deep_gap):10.3f}"
            f"{np.mean(extrapolated):10.3f}{np.mean(thermal_invalid):18.3f}"
        )


def trap_structure(events: pd.DataFrame) -> None:
    """Report signed-torque structure at each population's median mass."""
    print("\n=== 3. Fixed-contract torque structure at median mass ===\n")
    print(
        f'{"M_smbh":>8} {"m_med":>8} | {"outward":>9}{"traps":>7} '
        f'| {"stable trap radii [rg]":>28}'
    )
    for smbh_mass, group in events.groupby("m_smbh_msun"):
        median_mass = float(group.mass_msun.median())
        torque = compute_torque_profile(
            _disk(float(smbh_mass)).disk,
            median_mass,
            None,
            params=MigrationParameters(alpha=PAGN_ALPHA),
        )
        traps = classify_torque_traps(torque.radius_rg, torque.gamma_total_si)
        radii = np.array2string(
            np.round(traps.stable_radius_rg, 1), max_line_width=200
        )
        print(
            f"{smbh_mass:8.0e} {median_mass:8.1f} | "
            f"{int((torque.gamma_total_si > 0).sum()):9d}"
            f"{len(traps.stable_radius_rg):7d} | {radii:>28}"
        )


def main(argv: list[str]) -> int:
    run_dir = Path(argv[1]) if len(argv) > 1 else Path("output/_pop")
    events = pd.read_csv(run_dir / "imri_events.csv")
    print(f"run: {run_dir}   objects: {len(events)}")
    print(f"migration contract: {MIGRATION_CONTRACT}")
    population_summary(events)
    validity_occupancy(events)
    trap_structure(events)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
