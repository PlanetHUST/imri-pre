"""Report the saved single-halo baseline and paired SEVN scenario results.

This reads persisted artifacts only; it does not run any simulation or draw RNG.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


STAGES = (
    ("Sampled NSC objects", "n_sampled"),
    ("Disk crossing", "n_disk_cross"),
    ("Embedded stars", "n_emb_star"),
    ("Embedded BHs", "n_emb_bh"),
    ("BHs formed during migration", "n_mfbh"),
    ("Intrinsic IMRIs", "n_imri"),
    ("Detector evaluated", "n_detector_eval"),
    ("Detector unavailable", "n_detector_unavailable"),
    ("Stellar model unavailable", "n_stellar_unavailable"),
    ("Detections at configured threshold", "n_det"),
    ("Replacement donors", "n_replenished"),
)


def _load(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    repeats = pd.read_csv(directory / "imri_repeat_results.csv")
    events = pd.read_csv(directory / "imri_events.csv")
    metadata = json.loads((directory / "simulation_metadata.json").read_text())
    ledger_path = directory / "imri_stellar_ledger.csv"
    ledger = pd.read_csv(ledger_path) if ledger_path.exists() else pd.DataFrame()
    return repeats, events, metadata, ledger


def _mean(frame: pd.DataFrame, column: str) -> str:
    if column not in frame:
        return "—"
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any():
        return "unavailable"
    return f"{values.mean():.4g}"


def _snr_cut(events: pd.DataFrame, threshold: float) -> int:
    mask = (
        events["is_imri_candidate"].astype(str).str.lower().eq("true")
        & events["snr_status"].eq("evaluated")
    )
    return int((pd.to_numeric(events.loc[mask, "lisa_snr"]) >= threshold).sum())


def _primary_keys(events: pd.DataFrame, *, candidate_only: bool) -> set[tuple[int, int]]:
    primary = events[events["task_kind"].eq("primary")]
    if candidate_only:
        primary = primary[
            primary["is_imri_candidate"].astype(str).str.lower().eq("true")
        ]
    return set(zip(primary["repeat"].astype(int), primary["source_id"].astype(int)))


def build_report(scenarios: list[tuple[str, Path]]) -> str:
    data = [(name, directory, *_load(directory)) for name, directory in scenarios]
    reference_repeats, _, reference_meta, _ = data[0][2:]
    reference_params = reference_meta["parameters"]
    parameter_keys = (
        "smbh_masses_msun", "nsc_masses_msun", "duty_cycle_myr",
        "metallicity", "redshift", "disk_model", "base_seed",
    )
    for name, _, repeats, _, metadata, _ in data[1:]:
        for key in parameter_keys:
            if metadata["parameters"][key] != reference_params[key]:
                raise ValueError(f"{name} differs in controlled parameter {key}")
        if len(repeats) != len(reference_repeats):
            raise ValueError(f"{name} has a different repeat count")
        for key in ("repeat", "task_seed", "n_sampled", "n_disk_cross"):
            if not repeats[key].reset_index(drop=True).equals(
                reference_repeats[key].reset_index(drop=True)
            ):
                raise ValueError(f"{name} does not preserve paired {key}")
    names = [name for name, *_ in data]
    lines = [
        "# Single-halo stellar-integration baseline and matched scenarios", "",
        "The denominator is one configured AGN episode per repeat in one modeled halo;",
        "these are not cosmological volumetric rates. All stage counts below are",
        "means per episode. An unavailable rate is not a physical zero.", "",
        f"Control: `{data[0][1]}`. Repeats: {len(reference_repeats)}; ",
        f"SMBH: {reference_params['smbh_masses_msun']} Msun; ",
        f"NSC: {reference_params['nsc_masses_msun']} Msun; ",
        f"redshift: {reference_params['redshift']}; ",
        f"duty: {reference_params['duty_cycle_myr']} Myr; ",
        f"metallicity: {reference_params['metallicity']} Zsun; ",
        f"base seed: {reference_params['base_seed']}.", "",
        "| Measure | " + " | ".join(names) + " |",
        "| --- | " + " | ".join("---:" for _ in names) + " |",
    ]
    for label, column in STAGES:
        lines.append(
            "| " + label + " | "
            + " | ".join(_mean(repeats, column) for _, _, repeats, _, _, _ in data)
            + " |"
        )
    for label, column in (
        ("Intrinsic rate [halo Myr]^-1", "rate_imri_per_myr"),
        ("Detected rate [halo Myr]^-1", "rate_det_per_myr"),
    ):
        lines.append(
            "| " + label + " | "
            + " | ".join(_mean(repeats, column) for _, _, repeats, _, _, _ in data)
            + " |"
        )
    for threshold in (15.0, 20.0):
        lines.append(
            f"| Evaluated SNR ≥ {threshold:g}, total | "
            + " | ".join(
                str(_snr_cut(events, threshold))
                for _, _, _, events, _, _ in data
            )
            + " |"
        )
    if len(data) > 1:
        control_events = data[0][3]
        lines += ["", "## Matched primary-source dependency", ""]
        for name, _, _, events, _, _ in data[1:]:
            control_rows = _primary_keys(control_events, candidate_only=False)
            changed_rows = _primary_keys(events, candidate_only=False)
            control_imri = _primary_keys(control_events, candidate_only=True)
            changed_imri = _primary_keys(events, candidate_only=True)
            lines += [
                f"- {name}: {len(control_rows & changed_rows)} primary event IDs "
                f"remain shared; {len(control_rows - changed_rows)} disappear and "
                f"{len(changed_rows - control_rows)} appear at the embedding/event "
                "stage. Among intrinsic IMRIs, "
                f"{len(control_imri & changed_imri)} remain shared, "
                f"{len(control_imri - changed_imri)} disappear, and "
                f"{len(changed_imri - control_imri)} appear."
            ]
        lines += [
            "", "Replacement donor chains are conditional on ISCO consumption; "
            "their membership need not remain paired after a stellar-model change.",
        ]
        lines += ["", "Paired per-episode differences (mean ± standard error):", ""]
        for name, _, repeats, _, _, _ in data[1:]:
            for label, column in (
                ("intrinsic IMRIs", "n_imri"), ("detections", "n_det")
            ):
                difference = (
                    repeats[column].to_numpy(dtype=float)
                    - reference_repeats[column].to_numpy(dtype=float)
                )
                standard_error = (
                    difference.std(ddof=1) / len(difference) ** 0.5
                    if len(difference) > 1 else 0.0
                )
                lines.append(
                    f"- {name} minus control, {label}: "
                    f"{difference.mean():+.3g} ± {standard_error:.3g}."
                )
        zero_scenarios = [
            name for name, _, repeats, _, _, _ in data
            if int(repeats["n_det"].sum()) == 0
        ]
        if zero_scenarios:
            upper_per_episode = -math.log(0.05) / len(reference_repeats)
            lines += [
                "", f"Zero detections in {len(reference_repeats)} episodes "
                f"permit {upper_per_episode:.3g} detections per episode at "
                "the one-sided 95% Poisson upper limit for "
                f"{', '.join(zero_scenarios)}. The observed zero is not a "
                "precise detection-rate constraint.",
            ]
        if len(data) >= 3:
            no_kick_repeats = data[1][2]
            kick_repeats = data[2][2]
            difference = (
                kick_repeats["n_imri"].to_numpy(dtype=float)
                - no_kick_repeats["n_imri"].to_numpy(dtype=float)
            )
            standard_error = (
                difference.std(ddof=1) / len(difference) ** 0.5
                if len(difference) > 1 else 0.0
            )
            lines.append(
                f"SEVN kick minus SEVN no-kick, intrinsic IMRIs per episode: "
                f"{difference.mean():+.3g} ± {standard_error:.3g}."
            )
    if len(data) >= 3 and not data[1][5].empty and not data[2][5].empty:
        left = data[1][5]
        right = data[2][5]
        left = left[left["task_kind"].eq("primary")]
        right = right[right["task_kind"].eq("primary")]
        paired = left.merge(
            right, on=["repeat", "source_id"], suffixes=("_no_kick", "_kick"),
            validate="one_to_one",
        )
        if len(paired) != len(left) or len(paired) != len(right):
            raise ValueError("SEVN primary source IDs differ between kick modes")
        for quantity in (
            "zams_mass_msun", "presupernova_mass_msun",
            "presupernova_phase", "collapse_time_myr", "remnant_mass_msun",
            "remnant_type", "kick_speed_km_s", "sevn_seed",
        ):
            a = pd.to_numeric(paired[f"{quantity}_no_kick"])
            b = pd.to_numeric(paired[f"{quantity}_kick"])
            if not a.equals(b):
                raise ValueError(f"kick modes changed paired SEVN {quantity}")
        lines += [
            "", f"The two SEVN modes retain the same {len(paired)} primary "
            "source IDs, ZAMS masses, collapse times, remnant types/masses, "
            "SEVN kick speeds, and SEVN seeds. Their orbital outcomes differ "
            "only after the kick switch.",
        ]
    lines += ["", "## Status and dependency checks", ""]
    for name, directory, repeats, _, metadata, ledger in data:
        lines += [f"### {name}", "", f"Artifact directory: `{directory}`.", ""]
        lines += [
            f"Rate status by repeat: {repeats['rate_status'].value_counts().to_dict()}.",
            "Detector status by repeat: "
            f"{repeats['detector_rate_status'].value_counts().to_dict()}.",
        ]
        if not ledger.empty:
            stellar_meta = metadata.get("stellar_model", {})
            lines += [
                f"Stellar outcomes: {ledger['stellar_outcome'].value_counts().to_dict()}.",
                f"Damping outcomes: {ledger['damping_status'].value_counts().to_dict()}.",
                f"Nonzero kicks applied: {int(ledger['kick_applied'].sum())}.",
                f"H tables: `{metadata['parameters']['sevn_hydrogen_tables']}`; "
                f"He tables: `{metadata['parameters']['sevn_helium_tables']}`.",
                f"SEVNpy version: {stellar_meta.get('sevnpy_version', 'unavailable')}; "
                f"Kick interface: {stellar_meta.get('kick_orbit_interface', 'not_used')}.",
                f"H/He table SHA256: "
                f"{stellar_meta.get('hydrogen_tables_sha256', 'unavailable')} / "
                f"{stellar_meta.get('helium_tables_sha256', 'unavailable')}.",
            ]
            no_entry = ledger[
                ledger["stellar_outcome"].eq("postkick_no_band_before_z0")
            ]
            if not no_entry.empty:
                proxy_times = pd.to_numeric(
                    no_entry["postkick_gw_to_band_z0_myr"], errors="raise"
                )
                lines += [
                    f"Non-embedded kicked remnants screened out: {len(no_entry)}; "
                    f"shortest Peters time to 4x pericenter-frequency proxy "
                    f"at z=0: {proxy_times.min():.3g} Myr. This is a "
                    "screening convention, not an eccentric SNR calculation."
                ]
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("scenario", nargs="+", help="NAME=ARTIFACT_DIRECTORY")
    args = parser.parse_args()
    scenarios = []
    for item in args.scenario:
        name, separator, directory = item.partition("=")
        if not separator or not name or not directory:
            parser.error("each scenario must be NAME=ARTIFACT_DIRECTORY")
        scenarios.append((name, Path(directory)))
    report = build_report(scenarios)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
