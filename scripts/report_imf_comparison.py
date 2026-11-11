#!/usr/bin/env python3
"""Summarize the fixed-mass IMF benchmark, its control, and population caveats."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def power_integral(low, high, exponent):
    return np.log(high / low) if exponent == -1 else (high**(exponent + 1) - low**(exponent + 1)) / (exponent + 1)


def imf_expectation(low, high, slope=1.77, cluster_mass=3e6, tail=258.46068825):
    number = power_integral(low, high, -slope)
    mass = power_integral(low, high, 1 - slope)
    tail_count = cluster_mass / mass * power_integral(max(tail, low), high, -slope) if high > max(tail, low) else 0
    return dict(minimum_msun=low, maximum_msun=high, slope=slope,
                mean_mass_msun=mass / number, expected_objects=cluster_mass * number / mass,
                expected_objects_above_258_msun=tail_count)


def diagnostics(frame):
    rows = []
    for _, point in frame.iterrows():
        path = Path(point.run_dir)
        repeat = pd.read_csv(path / "imri_repeat_results.csv")
        events = pd.read_csv(path / "imri_events.csv")
        candidates = events[events.is_imri_candidate.astype(str).str.lower().eq("true")]
        detected = candidates[candidates.is_lisa_detectable.astype(str).str.lower().eq("true")]
        rows.append(dict(z=point.z, M=point.M, M_NSC=point.M_NSC,
            sampled_objects_mean=repeat.n_sampled.mean(), crossers_mean=repeat.n_disk_cross.mean(),
            embedded_stars_mean=repeat.n_emb_star.mean(), embedded_bhs_mean=repeat.n_emb_bh.mean(),
            migration_formed_bhs_mean=repeat.n_mfbh.mean(),
            imri_candidate_evaluations=len(candidates), detections=len(detected),
            imri_mass_min=candidates.mass_msun.min(), imri_mass_max=candidates.mass_msun.max(),
            max_snr=candidates.lisa_snr.max(),
            candidates_below_original_min=int((candidates.mass_msun < 13.6).sum()),
            candidates_above_original_max=int((candidates.mass_msun > 197).sum()),
            detections_above_original_max=int((detected.mass_msun > 197).sum()),
            entries_after_duty_cycle=int((candidates.detection_time_myr > repeat.duty_cycle_myr.iloc[0]).sum()),
            minimum_detected_secondary=detected.mass_msun.min(), maximum_detected_secondary=detected.mass_msun.max()))
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("outputs/core/high_redshift_rates"))
    parser.add_argument("--broad", type=Path, default=Path("outputs/archive/imf/imf_1_400_rates"))
    parser.add_argument("--upper-control", type=Path, default=Path("outputs/archive/imf/imf_13p6_400_reference"))
    args = parser.parse_args(argv)
    old = pd.read_csv(args.baseline / "host_rates.csv")
    broad = pd.read_csv(args.broad / "host_rates.csv")
    control = pd.read_csv(args.upper_control / "host_rates.csv")
    keys = ["z", "M", "M_NSC"]
    comparison = old.merge(broad, on=keys, suffixes=("_default", "_broad"), validate="one_to_one")
    if len(comparison) != 13:
        raise ValueError("IMF comparison requires all 13 configurations")
    for rate in ("damping", "event", "detect"):
        comparison[f"{rate}_change"] = comparison[f"{rate}_rate_broad"] - comparison[f"{rate}_rate_default"]
    comparison.to_csv(args.broad / "comparison_with_default.csv", index=False)
    details = diagnostics(broad)
    details.to_csv(args.broad / "imf_population_diagnostics.csv", index=False)
    expectations = pd.DataFrame([imf_expectation(*bounds) for bounds in ((13.6, 197), (13.6, 400), (1, 400))])
    expectations.to_csv(args.broad / "analytic_imf_mass_budget.csv", index=False)
    select = lambda frame: frame[(frame.z == 10) & (frame.M == 1e5) & (frame.M_NSC == 3e6)].iloc[0]
    reference_old, reference_control, reference_new = map(select, (old, control, broad))
    labels = ("Default 13.6–197", "Upper-only 13.6–400", "Broad 1–400")
    lines = ["# IMF interval dependency: fixed-mass benchmark", "",
        "## Controlled setup", "",
        "The requested 1–400 Msun interval replaces 13.6–197 Msun at unchanged power-law slope 1.77 and unchanged total NSC mass. The z=8–15, SMBH=1e4–1e6 Msun, NSC=1e6–1e7 Msun grid, reference parameters, disk, lifetime fit, migration/damping prescription, three-repeat seeds and detector selection remain fixed. The original outputs are retained.", "",
        "Changing both bounds is NOT an isolated upper-cutoff experiment. A supplementary reference-only 13.6–400 Msun run isolates the upper-bound change at z=10, M_bullet=1e5, M_NSC=3e6 Msun.", "",
        "## Fixed total mass changes the number budget", "",
        "Analytic expectations for dN/dm proportional to m^-1.77 and total mass 3e6 Msun:", "",
        "| Interval [Msun] | Mean mass [Msun] | Expected total objects | Expected initial objects above 258.46 Msun |", "|---|---:|---:|---:|"]
    for _, row in expectations.iterrows():
        lines.append(f"| {row.minimum_msun:g}–{row.maximum_msun:g} | {row.mean_mass_msun:.3f} | {row.expected_objects:,.0f} | {row.expected_objects_above_258_msun:,.0f} |")
    lines.extend(["", "The last column is an initial whole-cluster number, NOT a crossing, embedding or detection count. Lowering the minimum adds numerous small objects but redistributes the same mass budget; it can dilute the high-mass tail relative to an upper-only extension.", "",
        "## Measured reference rates [Myr^-1]", "",
        "| Model interval [Msun] | Damping | IMRI entry | Detectable |", "|---|---:|---:|---:|"])
    for label, row in zip(labels, (reference_old, reference_control, reference_new)):
        lines.append(f"| {label} | {row.damping_rate:.4g} ± {row.damping_sem:.3g} | {row.event_rate:.4g} ± {row.event_sem:.3g} | {row.detect_rate:.4g} ± {row.detect_sem:.3g} |")
    lines.extend(["", "Error bars are Monte Carlo SEM with three repeats, not confidence intervals. Shared base seeds do not give object-by-object pairing after changing population sizes. Do not interpret small differences as resolved trends.", "",
        "## Redshift comparison", "",
        "| z | Default IMRI | Broad IMRI | Default detectable | Broad detectable | Broad maximum SNR |", "|---:|---:|---:|---:|---:|---:|"])
    for _, row in comparison[(comparison.M == 1e5) & (comparison.M_NSC == 3e6)].sort_values("z").iterrows():
        detail = details[(details.z == row.z) & (details.M == row.M) & (details.M_NSC == row.M_NSC)].iloc[0]
        lines.append(f"| {row.z:g} | {row.event_rate_default:.4g} | {row.event_rate_broad:.4g} | {row.detect_rate_default:.4g} | {row.detect_rate_broad:.4g} | {detail.max_snr:.4g} |")
    lines.extend(["", "## Endpoint and interpretation flags", "",
        f"- The broad calculation contains {int(details.candidates_below_original_min.sum())} candidate evaluations below the original 13.6-Msun lower bound and {int(details.entries_after_duty_cycle.sum())} entries after the AGN duty interval. These are explicitly flagged in imf_population_diagnostics.csv. The unchanged analytic lifetime-to-BH rule is not a calibrated low-mass remnant model; these candidates must not be interpreted as validated low-mass BH formation.",
        f"- There are {int(details.detections.sum())} positive detection evaluations across the distinct configurations, of which {int(details.detections_above_original_max.sum())} have secondaries above 197 Msun. Repeated configurations/paired redshifts are not independent astrophysical objects. See the per-configuration diagnostic table for masses, SNRs and counts.",
        "- Positive detections, if present, test the fixed-mass detector mechanism: raising the allowed secondary-mass tail can cross the unchanged threshold 20. They do NOT establish that 1–400 Msun stars leave equal-mass BHs.",
        "- The retained lifetime polynomial is extrapolated outside 13.6–197 Msun. Winds, NS/WD outcomes, pair-instability disruption, realistic remnant masses and old-BH populations are not introduced. The previously identified eccentric post-alignment handoff and stellar-disruption/refilling limitations remain unchanged.",
        "- A zero sample detection count is not an upper confidence bound. The high-MBH endpoint remains unevaluated where the selected band begins beyond ISCO; gaps are preserved.", "",
        "## What the cited papers actually assume", "",
        "**Pan & Yang (2021), Formation Rate of Extreme Mass Ratio Inspirals in Active Galactic Nuclei**, Physical Review D 103, 103018, DOI: https://doi.org/10.1103/PhysRevD.103.103018 ; arXiv:2101.09146. The local PDF (v6) was checked: Sec. II.C.3, p. 4 uses two components, 1-Msun stars and 10-Msun BHs, with total stellar mass 20 M_bullet and BH relative abundance delta=1e-3. It does not prescribe a 1–400 Msun stellar IMF.",
        "**Xue, Bartos, Haiman & Tagawa (2026), Recoil-regulated extreme mass-ratio inspirals in AGN disks**, https://arxiv.org/abs/2605.29305 (local PDF v1). Sec. II.A and Table I, p. 3 specify pre-existing BHs with a 5–15 Msun power-law mass distribution, slope 2.35, initial number M_bullet/(20 Msun). Larger secondary masses emerge through accretion and mergers. The inner boundary corresponds to f_GW=1e-4 Hz, unlike the retained 1-mHz observed cutoff of this benchmark. Its quoted cosmological rates are not the same observable/normalization as this experiment's single-system Myr^-1 rates.",
        "The other local Pan et al. (2021) paper, **Wet Extreme Mass Ratio Inspirals May Be More Common For Spaceborne Gravitational Wave Detection**, https://arxiv.org/abs/2104.01208 (v2), was also checked. Its detection population in Sec. V uses 10-Msun BHs and SMBH spin a=0.98, not a 1–400 Msun stellar IMF.",
        "**Detector models also differ:** Xue et al. Sec. II.F uses FastEMRIWaveforms with Kerr spin a=0.9, e=0 and aligned orbits, followed by sky averaging. This benchmark retains its nonspinning quadrupole model and original band. Spin/waveform and population-growth differences should be tested separately rather than attributed entirely to an IMF upper cutoff.",
        "None of these checked papers justifies calling this a literature-calibrated 1–400 Msun stellar-IMF model.", "",
        "## Suggestions", "",
        "1. Use the 1–400 figure as an explicit mass-interval sensitivity benchmark only. Keep the upper-only control when discussing whether a change comes from the high-mass tail or the low-mass normalization.",
        "2. For a realistic stellar IMF, adopt and validate a metallicity-dependent remnant mapping over the full mass interval, including NS/WD and pair-instability outcomes, instead of making all exhausted stars equal-mass BHs.",
        "3. For a Pan/Xue-inspired comparison, initialize separate stellar and pre-existing-BH components with an explicit BH abundance, then model BH growth if a high-mass tail is needed. Do not emulate hierarchical growth merely by increasing the initial stellar mass cutoff.",
        "4. Resolve/validate the post-alignment eccentric phase and stellar consumption/refill physics before treating any of these rate curves as predictive astrophysical rates.", "",
        "Figure: rate_dependence.png/pdf/svg. Main-text material: ppt_notes.md plus this report. Full point comparison: comparison_with_default.csv. Original simulation exports are not rescaled or overwritten.",
    ])
    (args.broad / "imf_comparison_report.md").write_text("\n".join(lines) + "\n")
    print(args.broad / "imf_comparison_report.md")


if __name__ == "__main__":
    main()
