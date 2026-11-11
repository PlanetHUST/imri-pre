#!/usr/bin/env python3
"""Run existing four physical scenarios across both host benchmarks and z=8–15.

No new physics. Reuses compatible saved runs; caches new full simulations.
Per-scenario plots include all evaluated candidates, not only detections.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd

from compare_low_mass_redshift import REDSHIFTS, plot, scatter_population, snr_records
from compare_physical_scenarios import CASES, config_from_saved_run, entry_diagnostics, helpers, primary_records
from workflows.workflow import run_simulation


def scenario_config(plan, workers):
    config = config_from_saved_run(Path(plan["baseline_run"]), Path(plan["target"]), workers)
    case = next(case for case in CASES if case[0] == plan["case"])
    config.disk_model, config.torque_gap_transport, config.consumption_replenishment = case[2:]
    return config


def complete(path, config):
    if not helpers.compatible_run(path, config):
        return False
    repeats = pd.read_csv(path / "imri_repeat_results.csv")
    return len(repeats) == config.num_repeats and repeats["repeat"].nunique() == config.num_repeats


def evaluate_plan(plan, workers):
    config = scenario_config(plan, workers)
    candidates = [Path(plan["target"])]
    if plan["case"] == "baseline":
        candidates.insert(0, Path(plan["baseline_run"]))
    elif plan["M"] == 1e5 and plan["z"] == 10:
        candidates.insert(0, Path("outputs/core/physical_scenario_rates") / plan["case"])
    source = next((path for path in candidates if complete(path, config)), None)
    reused = source is not None
    if source is None:
        source = Path(plan["target"])
        source.mkdir(parents=True, exist_ok=True)
        with (source / "run.log").open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            run_simulation(config)
        if not complete(source, config):
            raise RuntimeError(f"Incomplete simulation: {source}")
    pd.testing.assert_frame_equal(primary_records(source), primary_records(Path(plan["baseline_run"])),
                                  check_exact=False, rtol=1e-7, atol=1e-9)
    return dict(plan, run_dir=str(source), reused=reused, status="completed")


def collect(plans, output):
    """Write summaries only for completed runs; unknowns never become zeros."""
    all_rows, all_snrs = [], []
    for case, *_ in CASES:
        done = sorted([p for p in plans if p["case"] == case and p["status"] == "completed"],
                      key=lambda p: (p["M"], p["z"]))
        if not done:
            continue
        rows, repeats, snrs, candidates = [], [], [], []
        for plan in done:
            source = Path(plan["run_dir"])
            summary, repeat = helpers.summarize_run(source)
            row = dict(case=case, M=plan["M"], z=plan["z"], M_NSC=3e6, **summary)
            row.update(entry_diagnostics(source, case))
            rows.append(row)
            repeat["case"], repeat["M"], repeat["initial_z"] = case, plan["M"], plan["z"]
            repeat["run_dir"] = str(source)
            repeats.append(repeat)
            s, events = snr_records(source, plan["z"], plan["M"])
            s["case"] = case
            snrs.append(s)
            events["case"] = case
            candidates.append(events)
        folder = output / case
        folder.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(folder / "host_rates.csv", index=False)
        pd.concat(repeats, ignore_index=True).to_csv(folder / "repeat_rates.csv", index=False)
        pd.DataFrame(snrs).to_csv(folder / "snr_summary.csv", index=False)
        pd.concat(candidates, ignore_index=True).to_csv(folder / "candidate_snr.csv", index=False)
        all_rows.extend(rows)
        all_snrs.extend(snrs)
    pd.DataFrame(all_rows).to_csv(output / "scenario_rates.csv", index=False)
    pd.DataFrame(all_snrs).to_csv(output / "scenario_snr_summary.csv", index=False)


def plot_overview(output):
    """Shared scales across all prescriptions; linear below the SNR threshold."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = [scatter_population(pd.read_csv(output / case / "candidate_snr.csv")) for case, *_ in CASES]
    maximum = max(22., max(float(data.lisa_snr.max()) for data in datasets) * 1.15)
    with plt.rc_context({"font.size": 11, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(4, 2, figsize=(10, 12), sharey=True)
        for row in range(1, 4):
            axes[row, 0].sharex(axes[0, 0])
        for row in range(3):
            axes[row, 0].tick_params(labelbottom=False)
        for row, ((case, label, *_), data) in enumerate(zip(CASES, datasets)):
            for mass, color, marker, legend in ((1e4, "#0072B2", "o", r"$M_\bullet=10^4M_\odot$"),
                                               (1e5, "#D55E00", "^", r"$M_\bullet=10^5M_\odot$")):
                selected = data[np.isclose(data.M, mass)]
                for ax, column in zip(axes[row], ("mass_msun", "snr_observation_redshift")):
                    ax.scatter(selected[column], selected.lisa_snr, color=color, marker=marker,
                               s=14, alpha=.4, linewidths=0, label=legend)
            for ax in axes[row]:
                ax.axhline(20, color=".3", linestyle="--", linewidth=1, label="SNR 20")
                ax.set_yscale("symlog", linthresh=20, linscale=1.)
                ax.set_ylim(0, maximum)
                ax.set_yticks([tick for tick in (0, 10, 20, 50, 100, 200, 500, 1000) if tick < maximum])
                ax.set_yticklabels([str(tick) for tick in (0, 10, 20, 50, 100, 200, 500, 1000) if tick < maximum])
                ax.grid(alpha=.18)
                ax.spines[["top", "right"]].set_visible(False)
            axes[row, 0].set_ylabel(label.replace("\n", " ") + "\nSNR")
            if case == "torque_gap_viscous":
                axes[row, 1].set_xscale("log")
                axes[row, 1].set_xlabel(r"Observation redshift $z_{\rm obs}$ (log scale)")
        axes[-1, 0].set_xlabel(r"Secondary mass $m$ [$M_\odot$]")
        axes[-1, 1].set_xlabel(r"Observation redshift $z_{\rm obs}$")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
        manifest_path = output / "experiment.json"
        pending = []
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            pending = [p for p in manifest["runs"] if p["status"] != "completed"]
        if pending:
            fig.text(.5, .955, f"PRELIMINARY: {40-len(pending)}/40 configurations complete; missing runs omitted",
                     ha="center", fontsize=10, color=".35")
        fig.tight_layout(rect=(0, 0, 1, .94 if pending else .96))
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(output / f"snr_population_scenarios.{suffix}", dpi=220)
        plt.close(fig)


def write_report(output, manifest):
    rates = pd.read_csv(output / "scenario_rates.csv")
    lines = ["# Four-scenario candidate populations", "",
             "Both central masses (1e4 and 1e5 Msun), total NSC mass 3e6 Msun, initial z=8,10,12,14,15; three seeded repeats per configuration. Baselines keep their own mass-scaled fueling and calibrated disk edges. Every alternative copies its corresponding baseline disk, population, detector, duration and seed; only migration/refill changes. No src physics was changed.", "",
             "Baseline: analytical inward Type-I with refill. Torque I + thermal: signed ungapped gravitational + validity-masked thermal torque. Gap/viscous: existing continuous gap/viscous prescription. No refill: baseline migration with finite-object donor replacement disabled. Torque branches retain their existing implicit Radau solver.", "",
             "## Completion", "",
             f"{sum(p['status']=='completed' for p in manifest['runs'])}/40 host/redshift/scenario configurations completed. Reuse is recorded per run in experiment.json; incomplete tasks are not zero-yield simulations.", "",
             "## Figures", "",
             "`snr_population_scenarios.png/pdf/svg` compares all four scenarios on a common SNR scale: linear up to 20 and logarithmic above 20 to show loud delayed outliers without hiding the subthreshold population. The gap/viscous observation-redshift axis is logarithmic; other redshift axes are linear. Each individual figure names its model once in the title. Each scenario folder also contains `snr_population_scatter.png/pdf/svg`: SNR versus secondary mass (left), actual observation redshift (right); blue circles=1e4 Msun, orange triangles=1e5 Msun. All finite evaluated candidates are shown, including subthreshold sources. The dashed threshold remains 20. `rate_dependence` and `snr_vs_redshift` retain the complementary rate/distribution comparisons.", "",
             "| Scenario | Scatter figure |", "|---|---|"]
    for case, label, *_ in CASES:
        lines.append(f"| {label.replace(chr(10), ' ')} | [{case}]({case}/snr_population_scatter.png) |")
    lines += ["", "## z=10 comparison", "",
              "Rates are counts per source-frame AGN Myr per host; scatter is three-repeat Monte Carlo SEM, not physical uncertainty. Damping includes stars and BHs. Total yields retain delayed entries; in-episode detectable rate is separate.", "",
              "| Scenario | Central mass | IMRI rate | Detectable total | Detectable within AGN | Delayed detections | Max SNR |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in rates[np.isclose(rates.z, 10)].sort_values(["case", "M"]).iterrows():
        lines.append(f"| {r['case']} | {r.M:g} | {r.event_rate:.4g} ± {r.event_sem:.3g} | {r.detect_rate:.4g} | {r.detect_rate_within_agn:.4g} | {r.detections_after_agn:g} | {r.max_snr:.4g} |")
    if not rates.empty and int(rates.detections.sum()) > 0:
        lines += ["", "## Delayed detections", "",
                  f"The completed configurations contain {int(rates.detections.sum())} threshold-passing candidate evaluations, of which {int(rates.detections_after_agn.sum())} occur after their AGN episodes. Counts across paired redshifts/scenarios are not independent sources."]
        if rates.detect_rate_within_agn.eq(0).all():
            lines.append("Every completed configuration has zero sampled within-AGN detectable rate. The loud tail therefore does not establish detections during the initial z=8–15 AGN episodes.")
    validation_path = output / "snr_validation.json"
    if validation_path.exists():
        checks = json.loads(validation_path.read_text())["checks"]
        lines += ["", "## Brightest delayed-source SNR check", "",
                  f"Independent distance/noise/frequency integration reproduces the brightest z=10 gap/viscous SNRs with maximum relative error {max(check['relative_error'] for check in checks):.3g} (snr_validation.json). This validates the numerical SNR, not the fixed-mass long-term host evolution."]
        for check in checks:
            lines.append(f"- Central mass {check['central_mass_msun']:g} Msun: secondary {check['source_mass_msun']:.3f} Msun, actual observation z={check['observation_redshift']:.5f}, SNR={check['stored_snr']:.3f}. A loud source need not be the most massive secondary: a very late, nearby entry can dominate its SNR.")
    lines += ["", "## Selection and physical limitations", "",
              "Loud late events must not be called high-z episode detections: inspect band_entry_time_myr and snr_observation_redshift in candidate_snr.csv, plus the within-AGN and delayed counts in scenario_rates.csv. Fixed-mass Gyr continuations are conditional yields, not validated host/SMBH evolution. Initial z is not the plotted actual observation z.",
              "Paired-redshift and paired-prescription evaluations are not independent binaries. Do not add counts across scenarios as a physical population, or interpret counts in scatter panels as normalized probability densities. The original fixed-mass/remnant mapping, stellar-disruption and post-alignment handoff limitations remain. Primary damping records are checked against each exact baseline. Thermal/gap validity and cluster-potential approximations are not repaired by this comparison.",
              "SNR 20 on an orientation-averaged waveform is not an orientation-marginalized detection probability. Unavailable statuses stay unavailable; no thresholds, masses, forces or source physics were adjusted to manufacture detections.", "",
              "## Reproduce", "", "```bash", "python scripts/compare_scenario_populations.py --jobs 4 --workers 1",
              "python scripts/compare_scenario_populations.py --plot-only", "```"]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=Path("outputs/core/low_mass_redshift"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/core/scenario_populations"))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1 or args.workers < 1:
        parser.error("jobs and workers must be positive")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        manifest = json.loads((output / "experiment.json").read_text())
        if manifest["status"] != "completed":
            raise ValueError("Plot-only requires all 40 configurations completed")
    else:
        baseline = pd.read_csv(args.baseline_dir / "host_rates.csv")
        plans = []
        for case, *_ in CASES:
            for mass in (1e4, 1e5):
                for z in REDSHIFTS:
                    row = baseline[np.isclose(baseline.M, mass) & np.isclose(baseline.z, z)]
                    if len(row) != 1:
                        raise ValueError(f"Missing unique baseline M={mass}, z={z}")
                    plans.append(dict(case=case, M=mass, z=z, baseline_run=str(row.iloc[0].run_dir),
                        target=str(output / case / "simulations" / f"m_{mass:g}_z_{z:g}"), status="pending"))
        manifest = dict(status="running", baseline_index=str(args.baseline_dir / "host_rates.csv"),
                        repeats=3, runs=plans, jobs=args.jobs, damping_workers_per_job=args.workers)
        def save():
            (output / "experiment.json").write_text(json.dumps(manifest, indent=2) + "\n")
        save()
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(evaluate_plan, plan, args.workers): i for i, plan in enumerate(plans)}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    plans[index] = future.result()
                    print(f"{sum(p['status']=='completed' for p in plans)}/40: {plans[index]}", flush=True)
                except Exception as error:
                    plans[index].update(status="failed", error=repr(error))
                    print(f"FAILED: {plans[index]}", flush=True)
                manifest["runs"] = plans
                save()
                collect(plans, output)
                write_report(output, manifest)
        manifest["status"] = "completed" if all(p["status"] == "completed" for p in plans) else "incomplete"
        save()
        if manifest["status"] != "completed":
            raise SystemExit("Some runs failed; inspect experiment.json; no failed runs plotted as zero.")
    collect(manifest["runs"], output)
    for case, *_ in CASES:
        plot(output / case)
    plot_overview(output)
    write_report(output, manifest)
    print(f"Results: {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
