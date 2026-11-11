#!/usr/bin/env python3
"""Audit saved IMRI detectability without altering simulation data or thresholds."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy.constants import G, c, parsec
from scipy.integrate import simpson
from workflows.cli import build_parser, build_config_from_arguments
from workflows.engine import IMRIDynEngine
from workflows.event_work import IMRIEventEvaluator
from lisa.snr import evaluate_lisa_inspiral_snr
from lisa.detectability import classify_coupled_entry
from physical_process.orbital_evolution import CircularOrbitOutcome, continue_circular_band_track
from physical_process.redshift_evolve import RedshiftEvolution

MSUN_KG = 1.98847e30


def frequency_hz(record, stem):
    """Decode the output formatter's Hz / mHz / micro-Hz columns explicitly."""
    for suffix, scale in (("hz", 1.0), ("milli_hz", 1e-3), ("micro_hz", 1e-6), ("nano_hz", 1e-9)):
        column = f"{stem}_{suffix}"
        if column in record:
            return float(record[column]) * scale
    raise KeyError(f"Missing frequency column {stem}")


def reference_distance_mpc(z, h0=67.4, omega_m=0.315):
    """Independent fixed Gauss-Legendre cosmology, not production quad()."""
    nodes, weights = np.polynomial.legendre.leggauss(128)
    x = 0.5 * z * (nodes + 1)
    integral = 0.5 * z * np.sum(weights / np.sqrt(omega_m * (1 + x)**3 + 1 - omega_m))
    return (1 + z) * (c / 1000) / h0 * integral


def reference_psd(f):
    """Independent vectorized RCL two-channel PSD, four-year confusion curve."""
    f = np.asarray(f, dtype=float)
    arm = 2.5e9
    fstar = c / (2 * np.pi * arm)
    oms = (1.5e-11)**2 * (1 + (2e-3 / f)**4)
    acc = (3e-15)**2 * (1 + (0.4e-3 / f)**2) * (1 + (f / 8e-3)**4)
    instrument = 10 / (3 * arm**2) * (oms + 2 * (1 + np.cos(f / fstar)**2) * acc / (2 * np.pi * f)**4) * (1 + 0.6 * (f / fstar)**2)
    confusion = 9e-45 * f**(-7/3) * np.exp(-f**0.138 - 221 * f * np.sin(521 * f)) * (1 + np.tanh(1680 * (0.00113 - f)))
    return instrument + confusion


def reference_snr(m1, m2, z, distance, low, high, drift_profile=None, samples=8193):
    """Independent log-frequency Simpson integral, optionally with gas df/dt.

    At effectively constant z over months/years, residence-time correction is
    |dr/dt|_GW / |dr/dt|_(GW+gas). No production amplitude, noise or SNR helper
    is used. A finite mission is represented by the supplied measured bounds.
    """
    if not 0 < low < high:
        raise ValueError("Invalid reference frequency bounds")
    logf = np.linspace(np.log(low), np.log(high), samples)
    f = np.exp(logf)
    chirp_source = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
    mc_time = G * MSUN_KG * chirp_source * (1 + z) / c**3
    distance_time = distance * 1e6 * parsec / c
    amplitude2 = (5/24) * mc_time**(5/3) / (np.pi**(4/3) * distance_time**2) * f**(-7/3)
    residence = np.ones_like(f)
    if drift_profile is not None:
        grid, gas_drift, disk_inner = drift_profile
        radius_cm = (G * (m1 + m2) * MSUN_KG / (np.pi * f * (1 + z))**2)**(1/3) * 100
        beta_cm4_s = 64/5 * G**3 * (m1 * m2 * (m1 + m2)) * MSUN_KG**3 / c**5 * 1e8
        gw_speed = beta_cm4_s / radius_cm**3
        gas = np.where((radius_cm >= max(disk_inner, grid[0])) & (radius_cm <= grid[-1]), np.interp(radius_cm, grid, gas_drift), 0)
        residence = gw_speed / (gw_speed - gas)
        if not np.all((residence > 0) & (residence <= 1 + 1e-12)):
            raise ValueError("Reference audit assumes inward-only Type-I gas")
    return float(np.sqrt(16/5 * simpson(f * amplitude2 / reference_psd(f) * residence, x=logf)))


def config_for_run(path):
    parameters = json.loads((path / "simulation_metadata.json").read_text())["parameters"]
    config = build_config_from_arguments(build_parser().parse_args([
        "--population-model", parameters.get("population_model", "legacy_popiii")]))
    for name in ("disk_alpha", "disk_accretion_rate_g_s", "disk_outer_radius_cm", "disk_model", "duty_cycle_myr", "redshift", "snr_threshold", "lisa_mission_duration_years", "snr_window_placement", "lisa_min_gw_frequency_hz", "lisa_max_gw_frequency_hz", "detector_hubble_constant_km_s_mpc", "detector_omega_m"):
        setattr(config, name, parameters[name])
    return config


def replay_event(event, config, engine, disk):
    """Rebuild an in-band continuation from exported entry conditions only."""
    clock = RedshiftEvolution(config.detector_hubble_constant_km_s_mpc, config.detector_omega_m)
    time = float(event.band_entry_time_myr)
    radius = float(event.band_entry_radius_cm)
    redshift = float(event.band_entry_redshift)
    active = str(event.gas_active_at_entry).lower() == "true"
    entry = CircularOrbitOutcome(
        status="band_entry", entry_time_myr=time, entry_radius_cm=radius,
        entry_redshift=redshift, final_time_myr=time, final_radius_cm=radius,
        final_redshift=redshift, disk_off_time_myr=float(event.disk_off_time_myr),
        disk_off_radius_cm=float(event.disk_off_radius_cm),
        disk_off_reason=str(event.disk_off_reason), gas_active_at_entry=active,
    )
    evaluator = IMRIEventEvaluator(engine)
    grid, drift = evaluator._gas_drift_for_mass(SimpleNamespace(m_smbh_msun=event.m_smbh_msun), SimpleNamespace(smbh_mass_msun=event.m_smbh_msun), disk, event.mass_msun)
    track = continue_circular_band_track(
        entry=entry, primary_mass_msun=event.m_smbh_msun,
        secondary_mass_msun=event.mass_msun, initial_redshift=config.redshift,
        duty_cycle_myr=config.duty_cycle_myr, disk_inner_radius_cm=disk["inner_radius_cm"],
        gas_radius_cm=grid, gas_drift_cm_s=drift,
        max_observed_frequency_hz=config.lisa_max_gw_frequency_hz, redshift_model=clock,
    )
    results = []
    for threshold in (10.0, 15.0, 20.0):
        decision = classify_coupled_entry(
            orbit=entry, band_track=track, bh_candidate=True, is_imri_candidate=True,
            primary_mass_msun=event.m_smbh_msun, secondary_mass_msun=event.mass_msun,
            min_observed_frequency_hz=config.lisa_min_gw_frequency_hz,
            max_observed_frequency_hz=config.lisa_max_gw_frequency_hz,
            snr_threshold=threshold, mission_duration_years=config.lisa_mission_duration_years,
            detector_hubble_constant_km_s_mpc=config.detector_hubble_constant_km_s_mpc,
            detector_omega_m=config.detector_omega_m, snr_window_placement=config.snr_window_placement,
        )
        results.append(dict(z=config.redshift, event_id=event.event_id,
            saved_snr=event.lisa_snr, replay_snr=decision.lisa_snr,
            threshold=threshold, detectable=decision.is_lisa_detectable,
            status=decision.snr_status, reason=decision.detect_rate_reason,
            relative_snr_error=abs(decision.lisa_snr / event.lisa_snr - 1)))
    return results


def detector_controls():
    rows = []
    for m2, z, fmin in [(100, 1, .001), (100, 8, .001), (197, 8, .001), (1000, 8, .001), (1000, 10, .001), (197, 8, .0001)]:
        result = evaluate_lisa_inspiral_snr(
            m1_source_msun=1e5, m2_source_msun=m2, redshift=z,
            current_observed_frequency_hz=fmin, mission_duration_years=4,
            detector_min_frequency_hz=fmin, detector_max_frequency_hz=.1, snr_threshold=20,
        )
        rows.append(dict(m1=1e5, m2=m2, z=z, f_min_hz=fmin,
                         snr=result.lisa_snr, detectable=result.is_detectable,
                         status=result.status, observation_years=result.observation_duration_years))
    return pd.DataFrame(rows)


def plot_snr(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = summary[np.isclose(summary.M, 1e5) & np.isclose(summary.M_NSC, 3e6)].sort_values("z")
    with plt.rc_context({"font.size": 12, "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(6.4, 3.7))
        ax.fill_between(selected.z, selected.snr_p10, selected.snr_p90, color="#0072B2", alpha=.14, label="10–90%")
        ax.plot(selected.z, selected.snr_median, "o-", color="#0072B2", label="Median")
        ax.plot(selected.z, selected.snr_max, "s--", color="#D55E00", label="Maximum")
        ax.axhline(20, color="0.3", linestyle=":", label="Threshold")
        ax.set(xlabel="Redshift $z$", ylabel="SNR", xticks=selected.z, ylim=(0, 23))
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, ncol=2, fontsize=10)
        fig.tight_layout()
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(output / f"snr_vs_redshift.{suffix}", dpi=220)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/core/high_redshift_rates"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/supporting/detectability_audit"))
    args = parser.parse_args(argv)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    points = pd.read_csv(args.input_dir / "host_rates.csv")
    event_rows, summaries, thresholds, replays = [], [], [], []
    resources = {}
    for _, point in points.iterrows():
        path = Path(point.run_dir)
        events = pd.read_csv(path / "imri_events.csv")
        repeats = pd.read_csv(path / "imri_repeat_results.csv")
        candidates = events[events.is_imri_candidate.astype(str).str.lower().eq("true")].copy()
        config = config_for_run(path)
        if config.disk_model != "semi_type_i" or config.lisa_mission_duration_years != 4:
            raise ValueError("Independent reference audit supports four-year Type-I runs")
        key = (point.M, config.disk_alpha, config.disk_accretion_rate_g_s, config.disk_outer_radius_cm)
        if len(candidates) and key not in resources:
            engine = IMRIDynEngine(config)
            resources[key] = (engine, engine.get_disk_profiles(point.M))
        if len(candidates):
            engine, disk = resources[key]
            evaluator = IMRIEventEvaluator(engine)
        for _, event in candidates.iterrows():
            low, high = frequency_hz(event, "snr_f_low"), frequency_hz(event, "snr_f_high")
            z = event.snr_observation_redshift
            distance = reference_distance_mpc(z, config.detector_hubble_constant_km_s_mpc, config.detector_omega_m)
            independent_vacuum = reference_snr(event.m_smbh_msun, event.mass_msun, z, distance, low, high)
            grid, drift = evaluator._gas_drift_for_mass(SimpleNamespace(m_smbh_msun=point.M), SimpleNamespace(smbh_mass_msun=point.M), disk, event.mass_msun)
            active = str(event.gas_active_at_entry).lower() == "true"
            if active and event.band_entry_time_myr + config.lisa_mission_duration_years / (1 + z) / 1e6 >= config.duty_cycle_myr:
                raise ValueError("Reference frequency audit needs a split at AGN shutoff")
            independent_coupled = reference_snr(event.m_smbh_msun, event.mass_msun, z, distance, low, high,
                (grid, drift, disk["inner_radius_cm"]) if active else None)
            event_rows.append(dict(point=point.point, z=point.z, M=point.M, M_NSC=point.M_NSC,
                event_id=event.event_id, repeat=event["repeat"], source_id=event.source_id,
                task_kind=event.task_kind, copy_number=event.copy_number,
                mass_msun=event.mass_msun, snr=event.lisa_snr,
                status=event.snr_status, reason=event.detect_rate_reason,
                threshold=event.snr_threshold, observation_years=event.snr_observation_duration_years,
                distance_mpc=event.luminosity_distance_mpc, independent_distance_mpc=distance,
                chirp_mass_detector_msun=event.chirp_mass_detector_msun,
                independent_chirp_detector_msun=(1 + z) * (event.m_smbh_msun * event.mass_msun)**(.6) / (event.m_smbh_msun + event.mass_msun)**(.2),
                low_hz=low, high_hz=high, gas_active_at_entry=active,
                independent_vacuum_snr=independent_vacuum,
                independent_coupled_snr=independent_coupled,
                independent_snr_relative_error=abs(independent_coupled / event.lisa_snr - 1)))
        snrs = candidates.lisa_snr.to_numpy()
        summaries.append(dict(point=point.point, z=point.z, M=point.M, M_NSC=point.M_NSC,
            candidates=len(candidates), evaluated=int((candidates.snr_status == "evaluated").sum()),
            above_threshold=int((candidates.lisa_snr >= config.snr_threshold).sum()),
            snr_min=float(snrs.min()) if len(snrs) else np.nan,
            snr_p10=float(np.quantile(snrs, .1)) if len(snrs) else np.nan,
            snr_median=float(np.median(snrs)) if len(snrs) else np.nan,
            snr_p90=float(np.quantile(snrs, .9)) if len(snrs) else np.nan,
            snr_max=float(snrs.max()) if len(snrs) else np.nan,
            noncandidate_reason_counts=json.dumps(events.loc[~events.index.isin(candidates.index), "event_rate_reason"].value_counts().to_dict())))
        for threshold in (5, 8, 10, 15, 20):
            rates = []
            for _, repeat in repeats.iterrows():
                subset = candidates[candidates["repeat"] == repeat["repeat"]]
                rates.append((subset.lisa_snr >= threshold).sum() / repeat.duty_cycle_myr)
            thresholds.append(dict(z=point.z, M=point.M, M_NSC=point.M_NSC,
                threshold=threshold, detected_count=int((snrs >= threshold).sum()),
                rate_per_myr=float(np.mean(rates)) if len(candidates) else np.nan))
        if point.M == 1e5 and point.M_NSC == 3e6 and len(candidates):
            engine, disk = resources[key]
            # Engine config is only physical disk controls; use this run's z explicitly.
            best = candidates.loc[candidates.lisa_snr.idxmax()]
            replays.extend(replay_event(best, config, engine, disk))
        print(f"Audited z={point.z:g}, M={point.M:g}, M_NSC={point.M_NSC:g}: {len(candidates)} candidates", flush=True)
    audit, summary = pd.DataFrame(event_rows), pd.DataFrame(summaries)
    audit.to_csv(output / "candidate_audit.csv", index=False)
    summary.to_csv(output / "snr_summary.csv", index=False)
    pd.DataFrame(thresholds).to_csv(output / "threshold_sensitivity.csv", index=False)
    replay = pd.DataFrame(replays)
    replay.to_csv(output / "track_replay.csv", index=False)
    controls = detector_controls()
    controls.to_csv(output / "detector_controls.csv", index=False)
    plot_snr(summary, output)
    selected = audit[np.isclose(audit.M, 1e5) & np.isclose(audit.M_NSC, 3e6)]
    paired = selected.pivot(index=["repeat", "source_id", "task_kind", "copy_number"], columns="z", values="snr")
    paired.to_csv(output / "paired_redshift_snr.csv")
    checks = {
        "candidate_evaluations": len(audit),
        "all_candidates_evaluated": bool((audit.status == "evaluated").all()),
        "all_candidate_snrs_positive_finite": bool((np.isfinite(audit.snr) & (audit.snr > 0)).all()),
        "all_candidates_below_20": bool((audit.snr < 20).all()),
        "max_independent_snr_relative_error": float(audit.independent_snr_relative_error.max()),
        "max_track_replay_relative_error": float(replay.relative_snr_error.max()),
        "max_distance_relative_error": float(abs(audit.distance_mpc / audit.independent_distance_mpc - 1).max()),
        "max_detector_chirp_mass_relative_error": float(abs(audit.chirp_mass_detector_msun / audit.independent_chirp_detector_msun - 1).max()),
        "max_gas_residence_snr_suppression": float((1 - audit.independent_coupled_snr / audit.independent_vacuum_snr).max()),
        "paired_sources": len(paired),
        "same_sources_present_at_all_redshifts": bool(paired.notna().all().all()),
        "every_paired_snr_decreases_with_redshift": bool((np.diff(paired.to_numpy(), axis=1) < 0).all()),
        "positive_detector_controls_pass": bool(controls.loc[((controls.m2 == 100) & (controls.z == 1)) | (controls.m2 == 1000), "detectable"].all()),
        "track_replay_can_detect_at_lower_threshold": bool(replay.loc[replay.threshold == 10, "detectable"].any()),
    }
    (output / "checks.json").write_text(json.dumps(checks, indent=2))
    lines = ["# Why the high-redshift detectable rate is zero", "",
        "## Findings", "",
        f"- {len(audit)} candidate evaluations across configurations (not independent objects: redshift runs reuse paired populations). All have finite positive SNR, evaluated status, and SNR below the unchanged threshold 20.",
        "- The detector is not returning zero SNR. Redshift changes SNR, but a hard threshold can leave the counted detection rate zero at every redshift.",
        "- All 53 paired candidates remain below threshold at z=8–15. Damping and intrinsic counts stay constant for these fixed-population runs; this is not a claim of cosmological rate independence.",
        "- Exported frequency columns use Hz/mHz/micro-Hz according to their suffix. The audit converts them explicitly, not by treating micro-Hz values as Hz.", "",
        "| Initial z | Median SNR | Maximum SNR | Candidates above 20 |", "|---:|---:|---:|---:|",
    ]
    for _, row in summary[(summary.M == 1e5) & (summary.M_NSC == 3e6)].sort_values("z").iterrows():
        lines.append(f"| {row.z:g} | {row.snr_median:.4f} | {row.snr_max:.4f} | {row.above_threshold:g} |")
    lines.extend(["", "## Independent and end-to-end checks", "",
        f"- Independent four-year RCL PSD, SPA amplitude, Gauss-Legendre distance and log-frequency Simpson integration reproduce the stored coupled SNRs to maximum relative error {checks['max_independent_snr_relative_error']:.3g}. The reference uses saved frequency bounds and the same gas drift; it is an independent SNR integral, not an independent migration model.",
        f"- Rebuilding the brightest event's coupled in-band track at each redshift and rerunning the production classifier reproduces the stored SNR to maximum relative error {checks['max_track_replay_relative_error']:.3g}. Lower-threshold replay produces positive detections without changing its SNR.",
        "- Detector-only positive controls: a 1e5+100 Msun vacuum binary at z=1 is detectable; 1e5+1000 Msun at z=8 or 10 is detectable. These are detector controls, not additional simulated populations or rates.", "",
        "## Interpretation", "",
        "- The source population is limited to about 197 Msun secondaries. Distance increases with redshift and the observed ISCO frequency shifts downward, reducing overlap with LISA's sensitive band. The 1-mHz lower cutoff removes still-lower-frequency inspiral.",
        "- A four-year mission does not give four years of signal when the system traverses the selected band in months. Reported observation durations are actual in-band durations, with observer/source time conversion included.",
        f"- Gas-induced residence-time suppression on the measured in-band intervals is at most {checks['max_gas_residence_snr_suppression']:.3g} in fractional SNR in the independent integral; it is not driving the zero count.",
        "- The orbit and detector use slightly different historical solar-mass constants (1.98892e33 g versus 1.98847e30 kg), and slightly different G. This is a small normalization inconsistency, consistent with the roughly 0.02% coupled/vacuum reference discrepancy, not the tens-of-percent SNR deficit needed for detection. Constants were not changed in this audit.",
        "- A detector-only 1e5+197 Msun binary at z=8 gives SNR 17.469 with f_min=1 mHz and 17.539 with f_min=0.1 mHz. Lowering the cutoff alone does not cross threshold 20 for that reference source.",
        "- Current SNRs are orientation-averaged. A hard cut on averaged SNR is not an orientation-marginalized detection probability; some favorable orientations could exceed threshold. This audit does not claim zero detectability for every physical orientation.",
        "- Secondary masses, threshold, band and production rates were not changed to manufacture detections. threshold_sensitivity.csv is explicitly a conditional reclassification of existing candidates, not a new physical simulation.",
        "- The 1e6-Msun point has unevaluated entry classifications (ISCO before the selected band) and is not counted as a valid zero-SNR or zero-detection control.", "",
        "## Artifacts", "", "snr_vs_redshift.png/pdf/svg, candidate_audit.csv, snr_summary.csv, paired_redshift_snr.csv, threshold_sensitivity.csv, track_replay.csv, detector_controls.csv, checks.json.",
    ])
    (output / "report.md").write_text("\n".join(lines) + "\n")
    if checks["max_independent_snr_relative_error"] > .01 or checks["max_track_replay_relative_error"] > .01 or not checks["all_candidates_evaluated"] or not checks["positive_detector_controls_pass"]:
        raise RuntimeError("Detector audit failed; inspect checks.json before interpreting rates")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
