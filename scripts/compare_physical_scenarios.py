#!/usr/bin/env python3
"""Four approved one-change-at-a-time rate cases; hydro damping is pending."""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from workflows.cli import build_parser, build_config_from_arguments
from workflows.config import IMRIDynConfig, SEMI_TYPE_I, TORQUE_WITH_TRAP
from workflows.engine import IMRIDynEngine
from workflows.workflow import run_simulation

spec = importlib.util.spec_from_file_location("_scenario_host_helpers", Path(__file__).with_name("compare_host_rates.py"))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)

CASES = (
    ("baseline", "Baseline", SEMI_TYPE_I, True, True),
    ("torque_i_thermal", "Torque I\n+ thermal", TORQUE_WITH_TRAP, False, True),
    ("torque_gap_viscous", "Torque +\ngap/viscous", TORQUE_WITH_TRAP, True, True),
    ("no_replenishment", "No refill", SEMI_TYPE_I, True, False),
)


def config_from_benchmark(run_dir, output_dir, workers):
    parameters = json.loads((run_dir / "simulation_metadata.json").read_text())["parameters"]
    base = build_config_from_arguments(build_parser().parse_args(["--population-model", "legacy_popiii"]))
    kwargs = {}
    for name in inspect.signature(IMRIDynConfig).parameters:
        value = parameters.get(name, getattr(base, name))
        if name == "consumption_replenishment":
            value = parameters.get(name, {"enabled": True})["enabled"]
        if name == "cosmic_merge_time_myr" and value is None:
            value = np.inf
        kwargs[name] = value
    kwargs.update(output_dir=output_dir, damping_cpu_count=workers)
    config = IMRIDynConfig(**kwargs)
    if (config.population_model != "legacy_popiii" or config.popiii_min_mass_msun != 13.6
            or config.popiii_max_mass_msun != 197 or config.popiii_imf_slope != 1.77):
        raise ValueError("Approved benchmark is the original 13.6–197 Msun IMF, slope 1.77")
    if not (np.array_equal(config.smbh_masses_msun, [1e5]) and np.array_equal(config.nsc_masses_msun, [3e6])
            and config.redshift == 10 and config.num_repeats == 3 and config.base_seed is not None):
        raise ValueError("Approved reference: z=10, M=1e5, M_NSC=3e6, three seeded repeats")
    return config


def primary_records(path):
    transitions = pd.read_csv(path / "imri_source_transitions.csv")
    selected = transitions[(transitions.stage == "damping") & (transitions.task_kind == "primary")]
    return selected[["repeat", "source_id", "mass_msun", "status", "time_myr"]].sort_values(["repeat", "source_id"]).reset_index(drop=True)


def plot(frame, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter
    if frame.case.tolist() != [case[0] for case in CASES]:
        raise ValueError("Plot requires all four completed cases in the approved order")
    rates = ("damping", "event", "detect")
    ymax = max(.1, max((frame[f"{rate}_rate"] + frame[f"{rate}_sem"].fillna(0)).max() for rate in rates) * 1.2)
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    with plt.rc_context({"font.size": 11, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
        for ax, rate, title in zip(axes, rates, ("Damping", "IMRI", "Detectable")):
            for i, (_, row) in enumerate(frame.iterrows()):
                mean, sem = row[f"{rate}_rate"], row[f"{rate}_sem"]
                if np.isfinite(mean):
                    ax.errorbar(i, mean, yerr=[[min(mean, sem)], [sem]], fmt="o", color=colors[i],
                                markersize=7, capsize=4, linewidth=1.4, clip_on=False)
                else:
                    ax.text(i, .03, "N/A", transform=ax.get_xaxis_transform(), ha="center", fontsize=10)
            ax.set(title=title, xticks=range(4), xticklabels=[case[1] for case in CASES], xlim=(-.4, 3.4), ylim=(0, ymax))
            ax.set_yscale("symlog", linthresh=.1, linscale=.6)
            ax.yaxis.set_major_locator(FixedLocator([0] + [b*10.**p for p in range(-1, 6) for b in (1, 3) if b*10.**p < ymax]))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:g}"))
            ax.tick_params(axis="x", labelsize=9)
            ax.grid(axis="y", color=".9", linewidth=.6)
            ax.spines[["right", "top"]].set_visible(False)
        axes[0].set_ylabel(r"Rate [Myr$^{-1}$]")
        fig.tight_layout(w_pad=1.4)
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(output / f"physical_scenario_rates.{suffix}", dpi=220)
        plt.close(fig)


def torque_diagnostics(config, output):
    """Small representative-mass grid; cell fractions are not event coverage."""
    engine = IMRIDynEngine(config)
    disk = engine.get_disk_profiles(1e5)
    rows = []
    for gap in (False, True):
        config.torque_gap_transport = gap
        for mass in (13.6, 25., 50., 100., 197.):
            profile = engine.get_torque_profile(disk, 1e5, mass)
            rows.append(dict(gap_transport=gap, secondary_mass_msun=mass,
                cells=len(profile.radius_cm), thermal_valid_fraction=float(np.mean(profile.thermal_torque_regime_valid)),
                thermal_applied_fraction=float(np.mean(profile.thermal_validity_weight > 0)),
                positive_total_torque_fraction=float(np.mean(profile.gamma_total_si > 0)),
                min_gap_depth=float(np.min(profile.gap_depth_fraction)),
                total_torque_min_si=float(np.min(profile.gamma_total_si)), total_torque_max_si=float(np.max(profile.gamma_total_si))))
    pd.DataFrame(rows).to_csv(output / "representative_torque_profiles.csv", index=False)


def entry_diagnostics(path, case):
    events = pd.read_csv(path / "imri_events.csv")
    repeats = pd.read_csv(path / "imri_repeat_results.csv")
    candidates = events[events.is_imri_candidate.astype(str).str.lower().eq("true")]
    detected = candidates[candidates.is_lisa_detectable.astype(str).str.lower().eq("true")]
    duration = float(repeats.duty_cycle_myr.iloc[0])
    normalization = len(repeats) * duration
    return dict(case=case, candidate_evaluations=len(candidates),
        event_rate_within_agn=float((candidates.detection_time_myr <= duration).sum() / normalization),
        detect_rate_within_agn=float((detected.detection_time_myr <= duration).sum() / normalization),
        detected_entry_redshift_min=detected.detection_redshift.min(),
        detected_entry_redshift_max=detected.detection_redshift.max(),
        detection_time_min_myr=detected.detection_time_myr.min(),
        detection_time_max_myr=detected.detection_time_myr.max(),
        entries_after_agn=int((candidates.detection_time_myr > duration).sum()),
        detections=len(detected), detections_after_agn=int((detected.detection_time_myr > duration).sum()),
        entry_redshift_min=candidates.detection_redshift.min(), entry_redshift_max=candidates.detection_redshift.max(),
        entry_time_median_myr=candidates.detection_time_myr.median(), max_snr=candidates.lisa_snr.max(),
        migration_formed_bhs_mean=repeats.n_mfbh.mean(),
        replacement_injections_mean=repeats.n_replenished.mean())


def write_notes(frame, output, manifest):
    lines = ["# Physical-scenario comparison: PPT main text", "",
        "## Fixed benchmark", "",
        "z=10, M_bullet=1e5 Msun, M_NSC=3e6 Msun, original analytic IMF 13.6–197 Msun with slope 1.77 and density index 1.75. Three repeats use the original paired seeds. The disk, absolute accretion rate, disk edge, NSC radius, numerical damping, AGN lifetime and detector settings are copied from the saved benchmark metadata, not replaced by CLI defaults.",
        f"Source benchmark: `{manifest['benchmark_run']}`; seed {manifest['seed']}. The original files are unchanged.", "",
        "## Cases", "",
        "1. Baseline: analytical Type-I migration, numerical two-node damping, finite-object replenishment enabled.",
        "2. Torque I + thermal: Gamma=Gamma_I+Gamma_th. No gap attenuation or viscous term; thermal optical depth also uses the unperturbed disk (no gap factor). Signed torques are retained, so outward drift/trapping can occur. This ungapped counterfactual can enter a regime where a linear Type-I description is not physically justified; it is a controlled prescription comparison, not a claim that no gap forms.",
        "3. Torque + gap/viscous: Gamma=f_gap*Gamma_I+Gamma_th+(1-f_gap)*Gamma_viscous, with the existing continuous Kanagawa/inertia-limited transport. This is NOT a sharp classical Type-II switch.",
        "4. No refill: identical analytical Type-I physics to baseline, but no donor replacement after consumption.",
        "Torque cases use implicit Radau integration; the benchmark and no-refill cases retain DOP853. Both solve the same signed gas+GW ODE, tolerances and boundary/event conditions. The small local restoring time near a stable torque trap is not an NSC two-body relaxation time; it makes the ODE stiff for an explicit integrator. Nonstiff entry/ISCO consistency and a stiff restoring-trap test check this solver-only change.",
        "Thermal torque is included only where the existing validity mask permits it; extrapolation stays off in both torque cases. Suppressed cells indicate an uncalibrated thermal contribution, not proof that the physical thermal torque vanishes. representative_torque_profiles.csv logs a small unweighted radial-cell/mass grid, not event-weighted applicability or proof that paths are valid.", "",
        "## What supply and damping mean here", "",
        "Supply ON/OFF refers only to the existing finite NSC-object donor pool. It does not toggle external gas inflow, SMBH fueling, the disk profile or the accretion rate. Gas inflow from outside the model is not implemented.",
        "Damping counts all successful stellar and BH embeddings, including replacements, divided by source-frame AGN duration. IMRI counts finite physical band entries, not all ISCO mergers. Detectable counts the SNR-selected subset. Units are simply Myr^-1 in the simulated system; no host-number/volume normalization or observer-time conversion is applied.",
        "Finite entries after AGN shutdown are retained exactly as in the benchmark and normalized by the original AGN duration. These yields are not an instantaneous event rate at the initial redshift. scenario_entry_diagnostics.csv separates delayed entries and detections and records their actual entry redshifts; late events must not be interpreted as detections at z=10.",
        "All primary numerical-damping records are checked against the same benchmark. Thus differences in TOTAL damping rates between cases arise from changed replacement histories, not from a changed damping law. Primary-only rates are saved separately.",
        "Hydrodynamically calibrated damping and its final-radius closure are explicitly PENDING, not silently substituted or extrapolated.", "",
        "## Rates", "",
        "| Scenario | Damping | IMRI | Detectable | Primary damping | Status (intrinsic / detector) |", "|---|---:|---:|---:|---:|---|"]
    for _, row in frame.iterrows():
        lines.append(f"| {row.case} | {row.damping_rate:.4g} ± {row.damping_sem:.3g} | {row.event_rate:.4g} ± {row.event_sem:.3g} | {row.detect_rate:.4g} ± {row.detect_sem:.3g} | {row.primary_damping_rate:.4g} | {row.event_status} / {row.detect_status} |")
    lines.extend(["", "Figure points are means; bars are Monte Carlo SEM, not confidence intervals or convergence evidence. The shared symlog y-axis is linear below 0.1 Myr^-1 and includes genuine zeros. N/A denotes incomplete/unevaluated results, never substituted zeros. In particular a zero detection sample is not a zero upper confidence limit.",
        "Known post-alignment eccentric-damping handoff, finite-star disruption, stellar/remnant mapping and Kepler-potential limitations remain unchanged. Torque-case star trajectories likewise do not add new finite-star physics. Do not attribute a joint gap/viscous comparison solely to a Type-II timescale or interpret three repeats as statistically resolved astrophysical constraints."])
    diagnostics_path = output / "scenario_entry_diagnostics.csv"
    if diagnostics_path.exists():
        diagnostics = pd.read_csv(diagnostics_path)
        lines.extend(["", "## Delay and BH-formation diagnostics", "",
            "Counts below combine the three realizations; they are not independent model-population samples. Entry redshift is distinct from initial host redshift.", "",
            "| Scenario | Physical entries | Entries after AGN | Detections | Detections after AGN | Entry z range | Mean migration-formed BHs |", "|---|---:|---:|---:|---:|---|---:|"])
        late_notes = []
        for _, row in diagnostics.iterrows():
            lines.append(f"| {row.case} | {row.candidate_evaluations:g} | {row.entries_after_agn:g} | {row.detections:g} | {row.detections_after_agn:g} | {row.entry_redshift_min:.4g}–{row.entry_redshift_max:.4g} | {row.migration_formed_bhs_mean:.4g} |")
            if row.detections_after_agn:
                late_notes.append(f"\n**{row.case}: {row.detections_after_agn:g} detections are after disk shutdown.** Detection times span {row.detection_time_min_myr:.4g}–{row.detection_time_max_myr:.4g} Myr after initial z=10, and actual detected-entry redshifts span {row.detected_entry_redshift_min:.4g}–{row.detected_entry_redshift_max:.4g}. These are not detections during the high-z AGN episode.\n")
        lines.extend(late_notes)
        lines.append("\nThe within-AGN detected rates are saved separately in scenario_entry_diagnostics.csv. Long post-AGN propagation retains fixed component masses and the benchmark's disabled cosmic-merger model; Gyr-delayed events are not a calibrated prediction of real host evolution.")
    (output / "ppt_notes.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-run", type=Path, default=Path("outputs/high_redshift_rates/simulations/point_005"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/physical_scenario_rates"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args(argv)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        manifest = json.loads((output / "experiment.json").read_text())
        frame = pd.read_csv(output / "scenario_rates.csv")
        pd.DataFrame([entry_diagnostics(Path(row.run_dir), row.case) for _, row in frame.iterrows()]).to_csv(output / "scenario_entry_diagnostics.csv", index=False)
        plot(frame, output)
        write_notes(frame, output, manifest)
        return
    base = config_from_benchmark(args.benchmark_run, output, args.workers)
    manifest = dict(benchmark_run=str(args.benchmark_run), seed=base.base_seed, repeats=base.num_repeats,
        status="running",
        cases=[dict(name=name, disk_model=model, torque_gap_transport=gap, replenishment=refill,
                    orbit_integrator="Radau" if model == TORQUE_WITH_TRAP else "DOP853") for name, _, model, gap, refill in CASES],
        hydro_damping="pending_user_validation", primary_damping_identical=None)
    (output / "experiment.json").write_text(json.dumps(manifest, indent=2))
    original_primary = primary_records(args.benchmark_run)
    rows, repeat_frames, entry_rows = [], [], []
    for name, label, model, gap, refill in CASES:
        config = config_from_benchmark(args.benchmark_run, output / name, args.workers)
        config.disk_model, config.torque_gap_transport, config.consumption_replenishment = model, gap, refill
        source = config.output_dir
        if name == "baseline" and helpers.compatible_run(args.benchmark_run, config):
            source = args.benchmark_run
        elif not helpers.compatible_run(source, config):
            source.mkdir(parents=True, exist_ok=True)
            print(f"Running {name}", flush=True)
            with (source / "run.log").open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                run_simulation(config)
        summary, repeats = helpers.summarize_run(source)
        pd.testing.assert_frame_equal(primary_records(source), original_primary, check_exact=False, rtol=1e-7, atol=1e-9)
        rows.append(dict(case=name, **summary))
        entry_rows.append(entry_diagnostics(source, name))
        pd.DataFrame(entry_rows).to_csv(output / "scenario_entry_diagnostics.csv", index=False)
        repeats["case"] = name
        repeat_frames.append(repeats)
        pd.DataFrame(rows).to_csv(output / "scenario_rates.csv", index=False)
        pd.concat(repeat_frames, ignore_index=True).to_csv(output / "scenario_repeats.csv", index=False)
        print(f"{name}: damping={summary['damping_rate']:.4g}, IMRI={summary['event_rate']:.4g}, detectable={summary['detect_rate']:.4g}", flush=True)
    manifest["primary_damping_identical"] = True
    manifest["status"] = "completed"
    (output / "experiment.json").write_text(json.dumps(manifest, indent=2))
    frame = pd.DataFrame(rows)
    plot(frame, output)
    write_notes(frame, output, manifest)
    torque_diagnostics(config_from_benchmark(args.benchmark_run, output, args.workers), output)
    print(f"Figure: {output / 'physical_scenario_rates.png'}", flush=True)


if __name__ == "__main__":
    main()
