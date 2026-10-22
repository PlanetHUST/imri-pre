"""Run seed-paired reduced torque/trap and semi/no-trap coupled-orbit checks.

The same production workflow owns both runs. This script compares only the
upstream embedded-object ledger, so differing entries are attributable to the
gas prescription rather than different sampled objects.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from workflows.cli import build_config_from_arguments, build_parser  # noqa: E402
from workflows.config import SEMI_WITHOUT_TRAP, TORQUE_WITH_TRAP  # noqa: E402
from workflows.workflow import run_simulation  # noqa: E402

UPSTREAM_COLUMNS = (
    "repeat",
    "object_index",
    "mass_msun",
    "formation_time_myr",
    "damping_time_myr",
    "r_ini_rg",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--smbh-mass", type=float, default=1.0e5)
    parser.add_argument("--nsc-mass", type=float, default=5.0e4)
    parser.add_argument("--duty-cycle-myr", type=float, default=30.0)
    parser.add_argument("--redshift", type=float, default=15.0)
    parser.add_argument("--base-seed", type=int, default=424242)
    args = parser.parse_args()

    ledgers: dict[str, pd.DataFrame] = {}
    for scenario in (TORQUE_WITH_TRAP, SEMI_WITHOUT_TRAP):
        output_dir = args.output_root / scenario
        main_args = build_parser().parse_args(
            [
                "1",
                str(output_dir),
                "1",
                "--smbh-masses",
                str(args.smbh_mass),
                "--nsc-masses",
                str(args.nsc_mass),
                "--duty-cycle-myr",
                str(args.duty_cycle_myr),
                "--redshift",
                str(args.redshift),
                "--base-seed",
                str(args.base_seed),
            ]
        )
        config = build_config_from_arguments(main_args, disk_model=scenario)
        run_simulation(config)
        events = pd.read_csv(output_dir / "imri_events.csv")
        repeats = pd.read_csv(output_dir / "imri_repeat_results.csv")
        with (output_dir / "simulation_metadata.json").open() as stream:
            metadata = json.load(
                stream,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-strict JSON constant: {value}")
                ),
            )
        if events.empty:
            raise AssertionError(f"{scenario}: no embedded objects to compare")
        if metadata["parameters"]["orbit_model"] != "circular_coupled_gas_gw_v1":
            raise AssertionError(f"{scenario}: metadata orbit provenance mismatch")
        if not events["disk_model"].eq(scenario).all():
            raise AssertionError(f"{scenario}: scenario provenance mismatch")
        if not events["orbit_model"].eq("circular_coupled_gas_gw_v1").all():
            raise AssertionError(f"{scenario}: orbit provenance mismatch")
        if int(events["is_imri_candidate"].eq(True).sum()) != int(
            repeats["n_imri"].sum()
        ):
            raise AssertionError(f"{scenario}: event/repeat IMRI counts disagree")
        entries = events.loc[events["orbit_stop_reason"].eq("band_entry")]
        if not np.allclose(
            entries["band_entry_time_myr"],
            entries["orbit_final_time_myr"],
            rtol=0.0,
            atol=1.0e-8,
        ):
            raise AssertionError(f"{scenario}: entry/final orbit clocks disagree")
        candidate_entries = entries.loc[entries["is_imri_candidate"].eq(True)]
        gas_active_candidates = candidate_entries.loc[
            candidate_entries["gas_active_at_entry"].eq(True)
        ]
        if not gas_active_candidates["snr_status"].eq("evaluated").all():
            raise AssertionError(f"{scenario}: gas-active SNR was not evaluated")
        if not np.isfinite(
            gas_active_candidates["lisa_snr"].to_numpy(dtype=float)
        ).all():
            raise AssertionError(f"{scenario}: gas-active SNR is not finite")
        if not gas_active_candidates["snr_observation_case"].str.startswith(
            "coupled_"
        ).all():
            raise AssertionError(f"{scenario}: gas-active SNR provenance is wrong")
        if not np.allclose(
            candidate_entries["detection_redshift"],
            candidate_entries["band_entry_redshift"],
            rtol=1.0e-6,
        ):
            raise AssertionError(f"{scenario}: detector/orbit epochs disagree")
        if not np.allclose(
            entries["f_entry_obs_micro_hz"].to_numpy(dtype=float),
            config.lisa_min_gw_frequency_hz * 1.0e6,
            rtol=1.0e-6,
        ):
            raise AssertionError(f"{scenario}: moving-band frequency mismatch")
        if not np.isclose(
            repeats["rate_imri_per_myr"].iloc[0],
            repeats["n_imri"].iloc[0] / args.duty_cycle_myr,
            rtol=1.0e-6,
        ):
            raise AssertionError(f"{scenario}: intrinsic rate denominator mismatch")
        if int(candidate_entries["is_lisa_detectable"].eq(True).sum()) != int(
            repeats["n_det"].sum()
        ):
            raise AssertionError(f"{scenario}: detector event/repeat counts disagree")
        detector_rate = repeats["rate_det_per_myr"].iloc[0]
        if np.isfinite(detector_rate) and not np.isclose(
            detector_rate, repeats["n_det"].iloc[0] / args.duty_cycle_myr,
            rtol=1.0e-6,
            atol=1.0e-12,
        ):
            raise AssertionError(f"{scenario}: detector rate denominator mismatch")
        ledgers[scenario] = events.sort_values(
            ["repeat", "object_index"]
        ).reset_index(drop=True)
        print(
            f"{scenario}: embedded={len(events)}, "
            f"IMRI={int(repeats['n_imri'].sum())}, "
            f"detectable={int(repeats['n_det'].sum())}"
        )

    left = ledgers[TORQUE_WITH_TRAP]
    right = ledgers[SEMI_WITHOUT_TRAP]
    if len(left) != len(right):
        raise AssertionError("scenario runs sampled different embedded-object counts")
    for column in UPSTREAM_COLUMNS:
        if not np.allclose(
            left[column].to_numpy(dtype=float),
            right[column].to_numpy(dtype=float),
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        ):
            raise AssertionError(f"upstream ledger differs at {column}")
    comparison = left[["repeat", "object_index", "mass_msun"]].copy()
    for name, frame in (("torque", left), ("semi", right)):
        for column in (
            "orbit_stop_reason",
            "band_entry_time_myr",
            "gas_active_at_entry",
            "disk_off_reason",
            "is_imri_candidate",
            "snr_status",
            "lisa_snr",
            "snr_observation_case",
        ):
            comparison[f"{name}_{column}"] = frame[column]
    comparison["semi_minus_torque_entry_myr"] = (
        right["band_entry_time_myr"] - left["band_entry_time_myr"]
    )
    comparison_path = args.output_root / "paired_entry_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    print("paired upstream ledgers: exact match")
    print(f"saved: {comparison_path}")


if __name__ == "__main__":
    main()
