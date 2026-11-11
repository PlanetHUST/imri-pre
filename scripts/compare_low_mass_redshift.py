#!/usr/bin/env python3
"""Additional central-mass host benchmarks: redshift-dependent rates and SNRs.

Copies the archived 1e5/3e6 benchmark controls, but rescales absolute Mdot
with central mass and recalibrates the disk edge. Does not modify src physics.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from compare_physical_scenarios import config_from_benchmark, helpers
from workflows.config import DISK_ACCRETION_RATE_G_S_PER_MSUN
from workflows.engine import IMRIDynEngine
from workflows.workflow import run_simulation

REDSHIFTS = (8., 10., 12., 14., 15.)
MODEL_LABELS = {
    "baseline": "Analytical Type-I + refill",
    "torque_i_thermal": "Signed Type-I + thermal",
    "torque_gap_viscous": "Gap/viscous + thermal",
    "no_replenishment": "Analytical Type-I, no refill",
}


def model_label(output):
    return MODEL_LABELS.get(output.name, MODEL_LABELS["baseline"])


def snr_records(run_dir, z, mass):
    events = pd.read_csv(run_dir / "imri_events.csv")
    selected = events[events.is_imri_candidate.astype(str).str.lower().eq("true")].copy()
    evaluated = selected[selected.snr_status.eq("evaluated") & np.isfinite(selected.lisa_snr)]
    values = evaluated.lisa_snr.to_numpy(dtype=float)
    summary = dict(z=z, M=mass, candidates=len(selected), evaluated=len(evaluated),
                   unavailable_events=int(events.event_rate_class.astype(str).str.lower().str.replace("_", " ", regex=False).eq("not evaluated").sum()) if "event_rate_class" in events else 0,
                   detections=int(selected.is_lisa_detectable.astype(str).str.lower().eq("true").sum()),
                   snr_p10=float(np.quantile(values, .1)) if len(values) else np.nan,
                   snr_median=float(np.median(values)) if len(values) else np.nan,
                   snr_p90=float(np.quantile(values, .9)) if len(values) else np.nan,
                   snr_max=float(np.max(values)) if len(values) else np.nan,
                   entries_after_agn=int((selected.band_entry_time_myr > 10).sum()),
                   detections_after_agn=int(((selected.band_entry_time_myr > 10) & selected.is_lisa_detectable.astype(str).str.lower().eq("true")).sum()),
                   run_dir=str(run_dir))
    columns = [name for name in ("repeat", "source_id", "event_id", "mass_msun", "band_entry_time_myr",
               "snr_observation_redshift", "lisa_snr", "snr_status", "is_lisa_detectable",
               "snr_observation_duration_years", "snr_f_low", "snr_f_high") if name in selected]
    selected = selected[columns]
    selected.insert(0, "M", mass)
    selected.insert(0, "initial_z", z)
    return summary, selected


def scatter_population(frame):
    """All finite evaluated candidates in the two band-accessible benchmarks."""
    required = {"M", "mass_msun", "snr_observation_redshift", "lisa_snr", "snr_status"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing scatter columns: {sorted(missing)}")
    mask = (np.isclose(frame.M, 1e4) | np.isclose(frame.M, 1e5)) & frame.snr_status.eq("evaluated")
    for column in ("mass_msun", "snr_observation_redshift", "lisa_snr"):
        mask &= np.isfinite(frame[column])
    mask &= (frame.mass_msun > 0) & (frame.lisa_snr > 0) & (frame.snr_observation_redshift >= 0)
    return frame.loc[mask].copy()


def plot_candidate_properties(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = scatter_population(pd.read_csv(output / "candidate_snr.csv"))
    if data.empty:
        raise ValueError("No finite evaluated candidates for the population scatter")
    styles = ((1e4, "#0072B2", "o", r"$M_\bullet=10^4\,M_\odot$"),
              (1e5, "#D55E00", "^", r"$M_\bullet=10^5\,M_\odot$"))
    counts = []
    with plt.rc_context({"font.size": 12, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for mass, color, marker, label in styles:
            selected = data[np.isclose(data.M, mass)]
            if selected.empty:
                continue
            identities = [column for column in ("repeat", "source_id") if column in selected]
            counts.append(dict(central_mass_msun=mass, candidate_evaluations=len(selected),
                paired_source_groups=len(selected[identities].drop_duplicates()) if len(identities) == 2 else None,
                above_snr_20=int((selected.lisa_snr >= 20).sum()),
                secondary_mass_min_msun=float(selected.mass_msun.min()),
                secondary_mass_max_msun=float(selected.mass_msun.max()),
                observation_redshift_min=float(selected.snr_observation_redshift.min()),
                observation_redshift_max=float(selected.snr_observation_redshift.max())))
            for ax, column in zip(axes, ("mass_msun", "snr_observation_redshift")):
                ax.scatter(selected[column], selected.lisa_snr, s=25, marker=marker,
                           color=color, alpha=.45, edgecolors="white", linewidths=.25,
                           label=label, rasterized=False)
        for ax in axes:
            ax.axhline(20, color="0.3", linestyle="--", linewidth=1.2, label="SNR threshold 20")
            ax.grid(alpha=.18)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            maximum = max(22., float(data.lisa_snr.max()) * 1.1)
            ax.set_ylim(0, maximum)
            if maximum > 40:
                ax.set_yscale("symlog", linthresh=20, linscale=1.)
                ticks = [tick for tick in (0, 10, 20, 50, 100, 200, 500, 1000, 2000) if tick < maximum]
                ax.set_yticks(ticks, labels=[str(tick) for tick in ticks])
        axes[0].set(xlabel=r"Secondary mass $m$ [$M_\odot$]", ylabel="Orientation-averaged SNR")
        axes[1].set_xlabel(r"Observation redshift $z_{\rm obs}$")
        if output.name == "torque_gap_viscous":
            axes[1].set_xscale("log")
            axes[1].set_xlabel(r"Observation redshift $z_{\rm obs}$ (log scale)")
        fig.suptitle(model_label(output), y=.995, fontsize=14)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .94), ncol=3, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, .82), w_pad=1.5)
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(output / f"snr_population_scatter.{suffix}", dpi=220)
        plt.close(fig)
    (output / "snr_population_scatter_summary.json").write_text(json.dumps(dict(
        source="candidate_snr.csv", selection="All finite evaluated candidates; not threshold-selected detections",
        redshift="snr_observation_redshift, not initial host redshift",
        model=model_label(output), redshift_scale="log" if output.name == "torque_gap_viscous" else "linear",
        pairing="Five initial-redshift runs reuse paired populations; points are not independent binaries",
        benchmarks=counts), indent=2) + "\n")
    return data


def plot(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rates = pd.read_csv(output / "host_rates.csv")
    snrs = pd.read_csv(output / "snr_summary.csv")
    styles = ((1e4, "#0072B2", "o", r"$10^4\,M_\odot$"),
              (1e5, "#D55E00", "s", r"$10^5\,M_\odot$"),
              (1e6, "#009E73", "^", r"$10^6\,M_\odot$"))
    styles = [style for style in styles if np.isclose(rates.M, style[0]).any()]
    with plt.rc_context({"font.size": 12, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
        maximum = 0.1
        for ax, rate, title in zip(axes, ("damping", "event", "detect"),
                                   ("Embedding", "IMRI band entry", "Detectable")):
            for mass, color, marker, label in styles:
                data = rates[np.isclose(rates.M, mass)].sort_values("z")
                y = data[f"{rate}_rate"].to_numpy(float)
                error = data[f"{rate}_sem"].to_numpy(float)
                if not np.isfinite(y).any():
                    ax.text(.03, .94, label + ": N/A (band beyond ISCO)", color=color,
                            transform=ax.transAxes, va="top", fontsize=9)
                    continue
                maximum = max(maximum, float(np.nanmax(y + error)))
                ax.errorbar(data.z, y, yerr=np.vstack((np.minimum(y, error), error)),
                            color=color, marker=marker, capsize=2, label=label)
            ax.set(title=title, xlabel="Initial redshift", xticks=REDSHIFTS)
            ax.set_yscale("symlog", linthresh=.1, linscale=.6)
            ax.grid(axis="y", alpha=.2)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set(ylabel=r"Rate [Myr$^{-1}$]", ylim=(0, maximum * 1.25))
        axes[0].legend(frameon=False, title=r"Central BH mass")
        fig.suptitle(model_label(output), y=.995, fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, .91))
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(output / f"rate_dependence.{suffix}", dpi=220)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
        for mass, color, marker, label in styles:
            data = snrs[np.isclose(snrs.M, mass)].sort_values("z")
            if not np.isfinite(data.snr_max).any():
                axes[0].text(.03, .94, label + ": SNR N/A (band beyond ISCO)",
                             color=color, transform=axes[0].transAxes, va="top", fontsize=9)
                axes[1].plot([], [], color=color, marker=marker, label=label + " (N/A)")
                continue
            axes[0].plot(data.z, data.snr_median, color=color, marker=marker, label=label)
            axes[0].fill_between(data.z.to_numpy(), data.snr_p10.to_numpy(), data.snr_p90.to_numpy(), color=color, alpha=.15)
            axes[1].plot(data.z, data.snr_max, color=color, marker=marker, label=label)
        for ax, title in zip(axes, ("Median and 10–90% candidate range", "Maximum candidate SNR")):
            ax.axhline(20, color="0.3", linestyle="--", linewidth=1, label="Threshold 20")
            ax.set(title=title, xlabel="Initial redshift", xticks=REDSHIFTS, ylim=(0, None))
            ax.grid(alpha=.2)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("Orientation-averaged SNR")
        axes[1].legend(frameon=False)
        if output.name == "torque_gap_viscous":
            for ax in axes:
                ax.set_xscale("log")
                ax.set_xticks(REDSHIFTS, labels=[f"{z:g}" for z in REDSHIFTS])
                ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
                ax.set_xlabel("Initial redshift (log scale)")
        fig.suptitle(model_label(output), y=.995, fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, .89))
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(output / f"snr_vs_redshift.{suffix}", dpi=220)
        plt.close(fig)
    if (output / "candidate_snr.csv").exists():
        plot_candidate_properties(output)


def write_report(output):
    manifest = json.loads((output / "experiment.json").read_text())
    rates = pd.read_csv(output / "host_rates.csv")
    snrs = pd.read_csv(output / "snr_summary.csv")
    mass = manifest["central_mass_msun"]
    lines = ["# Central-mass benchmarks: redshift-dependent rates and SNR", "",
             "## Design", "",
             f"New central mass {mass:g} Msun; total NSC mass 3e6 Msun; three repeats at each initial z=8,10,12,14,15, paired base seed 20261107. Existing comparison runs are reused unchanged.",
             f"New absolute Mdot={manifest['new_disk']['Mdot_g_s']:.6g} g/s; newly calibrated outer edge={manifest['new_disk']['outer_radius_cm']:.9g} cm.",
             "Population, NSC radius, 10-Myr duty time, analytical Type-I migration/refill, and detector controls are copied from the archived reference. Stellar IMF 13.6–197 Msun, slope 1.77; fixed-mass/lifetime prescription. LISA: four years, 1–100 mHz, averaged SNR threshold 20.", "",
             "This is an additional disk/host benchmark at the standard mass-scaled fueling rate, NOT a fixed-disk mass-only derivative. The old mass-axis scan used the 1e5-Msun reference disk edge and absolute Mdot, so its other-mass points are not reused here.", "",
             "## Measured results", "",
             "Rates are counts / 10 source-frame Myr per host; errors are three-repeat sampling SEM, not physical uncertainty. Damping includes stars and BHs; IMRI counts first BH band entry, not ISCO mergers. Initial z differs from actual entry/observation z. All finite delayed entries are retained; delay counts are shown separately.", "",
             "| Central mass [Msun] | Initial z | Embedding | IMRI entry | Detectable | Median SNR | Max SNR | Delayed entries / detections |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, row in rates.sort_values(["M", "z"]).iterrows():
        s = snrs[np.isclose(snrs.M, row.M) & np.isclose(snrs.z, row.z)].iloc[0]
        measured = [f"{row[name+'_rate']:.4g} ± {row[name+'_sem']:.3g}" if np.isfinite(row[name+'_rate']) else "N/A" for name in ("damping", "event", "detect")]
        median = f"{s.snr_median:.4g}" if np.isfinite(s.snr_median) else "N/A"
        maximum = f"{s.snr_max:.4g}" if np.isfinite(s.snr_max) else "N/A"
        delays = f"{s.entries_after_agn:g} / {s.detections_after_agn:g}" if row.event_status == "complete" else "N/A"
        lines.append(f"| {row.M:g} | {row.z:g} | " + " | ".join(measured) + f" | {median} | {maximum} | {delays} |")
    lines += ["", "## Band availability and statuses", "",
              "For a nonspinning 1e6-Msun central BH, observed ISCO frequency is approximately 0.49 mHz at z=8 and 0.28 mHz at z=15: below the retained 1-mHz lower edge. The workflow flags such entries `isco_before_entry`; the summaries preserve these as unavailable entry/detection rates and SNR, not evaluated zeros. Damping can still be measured. No lower-frequency band or Kerr waveform was silently introduced.",
              "`host_rates.csv` retains event/detector statuses and unavailable reasons; `snr_summary.csv` records unavailable-event counts. Zero candidate counts in an unavailable case do not establish a measured zero physical rate.", "",
              "## Interpretation boundaries", "",
              "The high NSC/central-BH mass ratio requires checking the cluster contribution to the orbital potential; the engine remains Keplerian. At 1e4 Msun, the secondary mass ratio can reach about 0.02. Linear Type-I applicability/gaps and the circular nonspinning quadrupole approximation need separate validation; this experiment does not repair them.",
              "Known post-alignment capture, finite-star tidal-disruption and remnant-mapping limitations remain. Zero selected detections are not a zero upper limit or an orientation-marginalized probability. Paired-redshift evaluations are not independent binaries. No cosmic host abundance or observer-rate normalization is introduced.", "",
              "## Products", "", "- `rate_dependence.png/pdf/svg`: shared-scale redshift rate comparison by central mass.",
              "- `snr_vs_redshift.png/pdf/svg`: candidate SNR distribution summaries and maxima.",
              "- `host_rates.csv`, `repeat_rates.csv`, `snr_summary.csv`, `candidate_snr.csv`: measured summaries and event-level evidence.",
              "- `simulations/z_*/`: full new simulations; old runs remain in the original core benchmark folder.",
              f"- `experiment.json`: exact settings and provenance. Replot with `python scripts/compare_low_mass_redshift.py --plot-only --output-dir {output}`."]
    if (output / "snr_population_scatter_summary.json").exists():
        summary = json.loads((output / "snr_population_scatter_summary.json").read_text())
        lines += ["", "## Candidate-population scatter", "",
                  "`snr_population_scatter.png/pdf/svg` plots SNR against secondary mass (left) and actual SNR-observation redshift (right). Blue circles denote 1e4-Msun central BHs; orange triangles denote 1e5-Msun central BHs. The dashed line marks the unchanged threshold 20. All finite evaluated candidates are shown, including subthreshold sources; this is not an observed detected-population sample.",
                  "The horizontal redshift is `snr_observation_redshift`, not the initial z=8,10,12,14,15 label. No mass/redshift jitter or threshold change is applied. Several SNRs at a given mass can be the same source evaluated at different redshifts. The mass panel mixes those redshifts; isolate initial-redshift groups in the CSV before interpreting a mass-only scaling. The mass values retain the benchmark's fixed-mass/remnant assumptions."]
        for item in summary["benchmarks"]:
            lines.append(f"- M_bullet={item['central_mass_msun']:g} Msun: {item['candidate_evaluations']} evaluated points, {item['paired_source_groups']} (repeat, source_id) groups reused across redshift, {item['above_snr_20']} above SNR 20.")
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=Path("outputs/core/high_redshift_rates"))
    parser.add_argument("--mass", type=float, choices=(1e4, 1e6), default=1e4)
    parser.add_argument("--comparison-dir", type=Path, help="Optional existing multi-mass summary to include unchanged")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plot-only", action="store_true")
    mode.add_argument("--scatter-only", action="store_true", help="Plot candidate mass/redshift against SNR without rerunning simulations")
    args = parser.parse_args()
    output = args.output_dir or Path("outputs/core/low_mass_redshift" if args.mass == 1e4 else "outputs/core/high_mass_redshift")
    output.mkdir(parents=True, exist_ok=True)
    if args.scatter_only:
        plot_candidate_properties(output)
        write_report(output)
        return
    if args.plot_only:
        plot(output)
        write_report(output)
        return
    reference = args.baseline_dir / "simulations/point_005"
    base = config_from_benchmark(reference, output, args.workers)
    base.smbh_masses_msun = np.array([args.mass])
    base.disk_accretion_rate_g_s = DISK_ACCRETION_RATE_G_S_PER_MSUN * args.mass
    base.disk_outer_radius_cm = None
    disk = IMRIDynEngine(base).get_disk_profiles(args.mass)
    base.disk_outer_radius_cm = float(disk["outer_radius_cm"])
    manifest = dict(status="running", benchmark_run=str(reference), central_mass_msun=args.mass,
                    nsc_mass_msun=3e6, repeats=base.num_repeats, seed=base.base_seed, redshifts=REDSHIFTS,
                    design="Host benchmarks; standard mass-scaled Mdot and newly calibrated Q-boundary for new mass",
                    new_disk=dict(Mdot_g_s=base.disk_accretion_rate_g_s, outer_radius_cm=base.disk_outer_radius_cm),
                    comparison_source=str((args.comparison_dir or args.baseline_dir) / "host_rates.csv"))
    (output / "experiment.json").write_text(json.dumps(manifest, indent=2) + "\n")
    old = pd.read_csv(manifest["comparison_source"])
    rows, repeats, snrs, candidates = [], [], [], []
    for z in REDSHIFTS:
        config = copy.deepcopy(base)
        config.redshift = z
        config.output_dir = output / "simulations" / f"z_{z:g}"
        path = config.output_dir
        if not helpers.compatible_run(path, config):
            path.mkdir(parents=True, exist_ok=True)
            print(f"Running new benchmark z={z:g}", flush=True)
            with (path / "run.log").open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                run_simulation(config)
        selected = old[~np.isclose(old.M, args.mass) & np.isclose(old.M_NSC, 3e6) & np.isclose(old.z, z)]
        if selected.empty or selected.M.duplicated().any() or not np.isclose(selected.M, 1e5).any():
            raise ValueError(f"Missing or duplicate comparison hosts at z={z}")
        sources = [(args.mass, path)] + [(float(row.M), Path(row.run_dir)) for _, row in selected.sort_values("M").iterrows()]
        for mass, source in sources:
            summary, repeat = helpers.summarize_run(source)
            rows.append(dict(z=z, M=mass, M_NSC=3e6, **summary))
            repeat["initial_z"], repeat["M"] = z, mass
            repeat["run_dir"] = str(source)
            repeats.append(repeat)
            summary_snr, events = snr_records(source, z, mass)
            snrs.append(summary_snr)
            candidates.append(events)
        pd.DataFrame(rows).to_csv(output / "host_rates.csv", index=False)
        pd.concat(repeats, ignore_index=True).to_csv(output / "repeat_rates.csv", index=False)
        pd.DataFrame(snrs).to_csv(output / "snr_summary.csv", index=False)
        pd.concat(candidates, ignore_index=True).to_csv(output / "candidate_snr.csv", index=False)
        print(f"Completed z={z:g}: {rows[-len(sources)]}", flush=True)
    manifest["status"] = "completed"
    (output / "experiment.json").write_text(json.dumps(manifest, indent=2) + "\n")
    plot(output)
    write_report(output)
    print(f"Results: {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
