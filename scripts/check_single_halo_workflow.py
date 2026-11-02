"""Check persisted source transitions and stage checkpoints for one run.

Usage: python scripts/check_single_halo_workflow.py output/dyn_results
This reads saved artifacts only; it never reruns the physical simulation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


STAGES = [
    "population_selected",
    "primary_damping_complete",
    "primary_sources_complete",
    "consumption_cycle_complete",
    "rates_finalized",
]
KEY = ["m_smbh_msun", "m_nsc_msun", "repeat"]


def check_run(output_dir: Path) -> int:
    """Assert stage, source, donor, and rate conservation for saved repeats."""
    checkpoints = pd.read_csv(output_dir / "imri_workflow_checkpoints.csv")
    transitions = pd.read_csv(output_dir / "imri_source_transitions.csv")
    repeats = pd.read_csv(output_dir / "imri_repeat_results.csv")
    donor_path = output_dir / "imri_replenishment_ledger.csv"
    donors = pd.read_csv(donor_path)
    stellar_path = output_dir / "imri_stellar_ledger.csv"
    stellar = pd.read_csv(stellar_path) if stellar_path.exists() else None

    checked = 0
    for key, group in checkpoints.groupby(KEY, sort=False):
        if group["stage"].tolist() != STAGES:
            raise AssertionError(f"{key}: incomplete or reordered stage checkpoints")
        final = group.iloc[-1]
        matching = repeats.loc[
            (repeats[KEY[0]] == key[0])
            & (repeats[KEY[1]] == key[1])
            & (repeats[KEY[2]] == key[2])
        ]
        if len(matching) != 1:
            raise AssertionError(f"{key}: expected one repeat output row")
        repeat = matching.iloc[0]
        for control in ("disk_model", "redshift", "duty_cycle_myr"):
            same_as_repeat = (
                final[control] == repeat[control]
                if control == "disk_model"
                else np.isclose(final[control], repeat[control], rtol=1.0e-6)
            )
            if group[control].nunique() != 1 or not same_as_repeat:
                raise AssertionError(f"{key}: {control} changes across checkpoints")
        for checkpoint_name, repeat_name in (
            ("n_sampled", "n_sampled"),
            ("n_disk_cross", "n_disk_cross"),
            ("n_isco_consumed", "n_isco_consumed"),
            ("n_replenished", "n_replenished"),
            ("n_source_remaining", "n_source_remaining"),
            ("n_stellar_unavailable", "n_stellar_unavailable"),
            ("n_entry_inside_isco", "n_entry_inside_isco"),
            ("n_profile_error", "n_profile_error"),
            ("n_replenishment_unavailable", "n_replenishment_unavailable"),
            ("n_detector_unavailable", "n_detector_unavailable"),
        ):
            if int(final[checkpoint_name]) != int(repeat[repeat_name]):
                raise AssertionError(f"{key}: {checkpoint_name} differs from repeat")
        if int(final["n_primary_damped"]) != int(final["n_disk_cross"]):
            raise AssertionError(f"{key}: a primary lacks a damping result")
        if int(final["n_sources_processed"]) != int(final["n_disk_cross"] + final["n_replenished"]):
            raise AssertionError(f"{key}: processed source count does not balance")

        source_rows = transitions.loc[
            (transitions[KEY[0]] == key[0])
            & (transitions[KEY[1]] == key[1])
            & (transitions[KEY[2]] == key[2])
        ]
        if (source_rows["time_myr"] < source_rows["injection_time_myr"]).any():
            raise AssertionError(f"{key}: a source action precedes injection")
        for _, source_path in source_rows.groupby("object_index", sort=False):
            if source_path["stage"].tolist() not in (
                ["damping", "entry_detection"],
                ["damping", "stellar_handoff", "entry_detection"],
            ):
                raise AssertionError(f"{key}: a source has an invalid action path")
        for stage in ("damping", "entry_detection"):
            at_stage = source_rows.loc[source_rows["stage"] == stage]
            if len(at_stage) != int(final["n_sources_processed"]):
                raise AssertionError(f"{key}: {stage} rows do not balance")
            if at_stage["object_index"].duplicated().any():
                raise AssertionError(f"{key}: duplicate object in {stage} trace")
        damped = source_rows.loc[source_rows["stage"] == "damping"]
        if damped["source_id"].duplicated().any():
            raise AssertionError(f"{key}: NSC source was used twice")
        if int((damped["task_kind"] == "primary").sum()) != int(final["n_disk_cross"]):
            raise AssertionError(f"{key}: primary source count does not balance")
        if int((damped["task_kind"] == "replacement").sum()) != int(final["n_replenished"]):
            raise AssertionError(f"{key}: donor source count does not balance")
        if stellar is not None:
            stellar_rows = stellar.loc[
                (stellar[KEY[0]] == key[0])
                & (stellar[KEY[1]] == key[1])
                & (stellar[KEY[2]] == key[2])
            ]
            if len(stellar_rows) != int(final["n_stellar_rows"]):
                raise AssertionError(f"{key}: stellar endpoint rows do not balance")
        elif int(final["n_stellar_rows"]) != 0:
            raise AssertionError(f"{key}: stellar endpoint file is missing")

        donor_rows = donors.loc[
            (donors[KEY[0]] == key[0])
            & (donors[KEY[1]] == key[1])
            & (donors[KEY[2]] == key[2])
        ]
        if len(donor_rows) != int(final["n_isco_consumed"]):
            raise AssertionError(f"{key}: ISCO consumption rows do not balance")
        if not donor_rows["consumption_time_myr"].is_monotonic_increasing:
            raise AssertionError(f"{key}: ISCO agenda is out of time order")
        injected = donor_rows.loc[donor_rows["status"] == "injected"]
        if len(injected) != int(final["n_replenished"]):
            raise AssertionError(f"{key}: donor injection rows do not balance")
        if injected["source_id"].duplicated().any():
            raise AssertionError(f"{key}: donor used more than once")

        for status_name, checkpoint_name, repeat_name in (
            ("rate_status", "rate_imri_per_myr", "rate_imri_per_myr"),
            ("detector_rate_status", "rate_det_per_myr", "rate_det_per_myr"),
        ):
            checkpoint_rate = final[checkpoint_name]
            repeat_rate = repeat[repeat_name]
            if repeat[status_name] == "complete":
                # Repeat CSV has six significant digits; checkpoint has nine.
                if not np.isclose(checkpoint_rate, repeat_rate, rtol=1.0e-6, atol=5.0e-7):
                    raise AssertionError(f"{key}: {checkpoint_name} differs from repeat")
                if not np.isclose(
                    checkpoint_rate,
                    final["n_imri" if status_name == "rate_status" else "n_det"]
                    / final["duty_cycle_myr"],
                    rtol=0.0,
                    atol=5.0e-10,
                ):
                    raise AssertionError(f"{key}: {checkpoint_name} has a wrong denominator")
            elif not (pd.isna(checkpoint_rate) and pd.isna(repeat_rate)):
                raise AssertionError(f"{key}: unavailable rate was treated as zero")
        checked += 1
    if checked != len(repeats):
        raise AssertionError("checkpoint groups do not match repeat rows")
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    count = check_run(args.output_dir)
    print(f"PASS: {count} single-halo repeat(s) have balanced workflow checkpoints")


if __name__ == "__main__":
    main()
