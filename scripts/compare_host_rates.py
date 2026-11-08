#!/usr/bin/env python3
"""One clean three-panel figure: rates versus redshift, SMBH mass and NSC mass."""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

# Use a unique module name: the experiment and a regression test share a basename.
_experiment_spec = importlib.util.spec_from_file_location(
    "_imri_initial_parameter_experiment", Path(__file__).with_name("test_parameter_dependence.py")
)
_experiment = importlib.util.module_from_spec(_experiment_spec)
_experiment_spec.loader.exec_module(_experiment)
apply_values = _experiment.apply_values
reference_values = _experiment.reference_values
from workflows.cli import build_config_from_arguments, build_parser
from workflows.workflow import run_simulation
from physical_process.population_models import (
    POPULATION_MODELS, LEGACY_POPIII, PAN_NSC, PAN_PAPER, XUE_BH,
    population_description, pan_scale_radius_cm,
)
from physical_process import constants as const

REFERENCE_Z = 10.0
REFERENCE_M = 1e5
REFERENCE_NSC = 3e6
REDSHIFTS = (8.0, 10.0, 12.0, 14.0, 15.0)
SMBH_MASSES = (1e4, 3e4, 1e5, 3e5, 1e6)
NSC_MASSES = (1e6, 2e6, 3e6, 5e6, 1e7)
RATES = ("damping", "event", "detect")


def comparison_design():
    """Vary one parameter per panel; hold the other two at the reference."""
    return {
        "redshift": {
            "axis": "z", "grid": REDSHIFTS, "scale": 1.0,
            "xlabel": r"Redshift $z$", "title": "Low to high redshift",
            "curves": [dict(M=REFERENCE_M, M_NSC=REFERENCE_NSC)],
        },
        "smbh_mass": {
            "axis": "M", "grid": SMBH_MASSES, "scale": 1.0,
            "xlabel": r"$M_\bullet$ [$M_\odot$]", "title": "Low to high SMBH mass",
            "curves": [dict(z=REFERENCE_Z, M_NSC=REFERENCE_NSC)],
        },
        "nsc_mass": {
            "axis": "M_NSC", "grid": NSC_MASSES, "scale": 1.0,
            "xlabel": r"$M_{\rm NSC}$ [$M_\odot$]", "title": "Low to high NSC mass",
            "curves": [dict(z=REFERENCE_Z, M=REFERENCE_M)],
        },
    }


def points_for_curve(spec, curve):
    return [dict(curve, **{spec["axis"]: value}) for value in spec["grid"]]


def point_key(point):
    return tuple(float(point[name]) for name in ("z", "M", "M_NSC"))


def unique_points():
    points = {}
    for spec in comparison_design().values():
        for curve in spec["curves"]:
            for point in points_for_curve(spec, curve):
                points[point_key(point)] = point
    return [points[key] for key in sorted(points)]


def summarize_run(path):
    """Count SUCCESSFUL embeddings, not damping tasks or inverse timescales."""
    repeats = pd.read_csv(path / "imri_repeat_results.csv")
    events = pd.read_csv(path / "imri_events.csv")
    # Core aggregates can report complete/zero even when individual entries are
    # explicitly unevaluated (e.g. ISCO reached before the selected band).
    # Preserve those unknown classifications rather than manufacturing zeros.
    repeats["core_rate_status"] = repeats.rate_status
    repeats["core_detector_rate_status"] = repeats.detector_rate_status
    event_class = events.event_rate_class.astype(str).str.lower().str.replace("_", " ", regex=False)
    unavailable = events[event_class == "not evaluated"]
    for repeat_index in unavailable["repeat"].unique():
        mask = repeats["repeat"] == repeat_index
        repeats.loc[mask, ["rate_imri_per_myr", "rate_det_per_myr"]] = np.nan
        repeats.loc[mask, "rate_status"] = "incomplete_event_classification"
        repeats.loc[mask, "detector_rate_status"] = "not_evaluated_intrinsic_incomplete"
    transitions = pd.read_csv(path / "imri_source_transitions.csv")
    damping = transitions[transitions.stage == "damping"]
    successes = damping[damping.status.isin(("embedded_star", "embedded_bh"))]
    errors = damping[damping.status.isin(("calculation_error", "stellar_unavailable", "stellar_evolution_unavailable"))]
    # Also retain primary-only damping to diagnose replenishment feedback in z.
    primary = successes[successes.task_kind == "primary"].groupby("repeat").size()
    counts = successes.groupby("repeat").size()
    repeats["rate_damping_per_myr"] = (repeats.n_emb_star + repeats.n_emb_bh) / repeats.duty_cycle_myr
    for index, repeat in repeats.iterrows():
        if int(counts.get(repeat["repeat"], 0)) != int(repeat.n_emb_star + repeat.n_emb_bh):
            raise ValueError(f"Embedding count mismatch in {path}")
        if (errors["repeat"] == repeat["repeat"]).any() or repeat.n_stellar_unavailable > 0:
            repeats.loc[index, "rate_damping_per_myr"] = np.nan
    repeats["rate_primary_damping_per_myr"] = [primary.get(index, 0) / duration for index, duration in zip(repeats["repeat"], repeats.duty_cycle_myr)]
    columns = {"damping": "rate_damping_per_myr", "event": "rate_imri_per_myr", "detect": "rate_det_per_myr", "primary_damping": "rate_primary_damping_per_myr"}
    result = {"repeats": len(repeats), "run_dir": str(path),
              "event_unavailable_reasons": ";".join(sorted(unavailable.event_rate_reason.dropna().unique()))}
    for name, column in columns.items():
        values = repeats[column].to_numpy(dtype=float)
        complete = np.isfinite(values).all()
        result[f"{name}_rate"] = float(np.mean(values)) if complete else np.nan
        result[f"{name}_sem"] = float(np.std(values, ddof=1) / np.sqrt(len(values))) if complete and len(values) > 1 else np.nan
    result["event_status"] = "complete" if (repeats.rate_status == "complete").all() else "incomplete"
    result["detect_status"] = "complete" if (repeats.detector_rate_status == "complete").all() else "incomplete"
    return result, repeats


def select_panel(frame, spec):
    records = []
    for point in points_for_curve(spec, spec["curves"][0]):
        mask = np.ones(len(frame), dtype=bool)
        for key, value in point.items():
            mask &= np.isclose(frame[key], value, rtol=1e-10, atol=1e-12)
        selected = frame[mask]
        if len(selected) != 1:
            raise ValueError(f"Expected one measured point for {point}; got {len(selected)}")
        records.append(selected.iloc[0])
    return pd.DataFrame(records)


def plot_comparisons(frame, output, yscale="symlog"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, LogFormatterSciNotation, NullLocator

    panels = [select_panel(frame, spec) for spec in comparison_design().values()]
    upper = []
    for panel in panels:
        for name in RATES:
            upper.extend((panel[f"{name}_rate"] + panel[f"{name}_sem"].fillna(0)).dropna().tolist())
    ymax = max(max(upper, default=0.1) * 1.15, 0.1)
    styles = [("#0072B2", "o", "-", "Damping"),
              ("#D55E00", "s", "--", "IMRI"),
              ("#009E73", "^", ":", "Detectable")]
    with plt.rc_context({"font.size": 12, "axes.labelsize": 14, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
        for ax, spec, selected in zip(axes, comparison_design().values(), panels):
            x = selected[spec["axis"]] / spec["scale"]
            for rate, (color, marker, linestyle, label) in zip(RATES, styles):
                y = selected[f"{rate}_rate"].to_numpy(dtype=float)
                sem = selected[f"{rate}_sem"].to_numpy(dtype=float)
                # Do not turn unavailable values into zero; lines only connect samples.
                ax.errorbar(x, y, yerr=np.vstack((np.minimum(y, sem), sem)),
                            color=color, marker=marker, linestyle=linestyle,
                            markersize=5, linewidth=1.8, elinewidth=0.8,
                            capsize=2, label=label)
            ax.set_xlabel(spec["xlabel"])
            ax.set_ylim(0, ymax)
            if spec["axis"] == "z":
                ax.set_xticks(spec["grid"])
                ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:g}"))
            else:
                ax.set_xscale("log")
                endpoints = (spec["grid"][0], spec["grid"][-1])
                ax.xaxis.set_major_locator(FixedLocator(
                    [endpoints[0], REFERENCE_M, endpoints[-1]] if spec["axis"] == "M" else [endpoints[0], REFERENCE_NSC, endpoints[-1]]
                ))
                ax.xaxis.set_major_formatter(LogFormatterSciNotation(labelOnlyBase=False, minor_thresholds=(np.inf, np.inf)))
                ax.xaxis.set_minor_locator(NullLocator())
            if yscale == "symlog":
                ax.set_yscale("symlog", linthresh=0.1, linscale=0.6)
                ticks = [0.0] + [base * 10.0**power for power in range(-1, 6) for base in (1, 3) if base * 10.0**power <= ymax]
                ax.yaxis.set_major_locator(FixedLocator(ticks))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:g}"))
            ax.grid(axis="y", color="0.9", linewidth=0.6)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=11)
        axes[0].set_ylabel(r"Rate [Myr$^{-1}$]")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend([handle.lines[0] for handle in handles], labels,
                   loc="upper center", bbox_to_anchor=(0.5, 1.0),
                   ncol=3, frameon=False, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=1.5)
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(output / f"rate_dependence.{suffix}", dpi=220)
        plt.close(fig)
    return (0.0, ymax)


def write_ppt_notes(frame, output, manifest):
    reference = manifest["reference"]
    population_model = manifest.get("population_model", LEGACY_POPIII)
    legacy = population_model == LEGACY_POPIII
    baseline = frame[np.isclose(frame.z, REFERENCE_Z) & np.isclose(frame.M, REFERENCE_M) & np.isclose(frame.M_NSC, REFERENCE_NSC)].iloc[0]
    lines = [
        "# Rate-dependence figure: PPT main-text notes", "",
        "## Figure caption", "",
        "Three panels show damping, intrinsic IMRI-entry, and detectable IMRI-entry rates as a function of redshift, SMBH mass, and NSC mass. Only the horizontal parameter changes in each panel. All three panels use the same vertical scale and limits.", "",
        "## Reference and fixed inputs", "",
        f"- Reference: z={REFERENCE_Z:g}, M_bullet={REFERENCE_M:g} Msun, M_NSC={REFERENCE_NSC:g} Msun. Scans: z=8–15, M_bullet=1e4–1e6 Msun, M_NSC=1e6–1e7 Msun.",
        "- The large early-Universe NSC masses are an uncertain scenario range, not a calibrated host-population prediction.",
        f"- AGN duration: 10 Myr; {manifest['repeats']} paired-seed Monte Carlo repeats; seed {manifest['seed']}.",
        (f"- Fixed physical NSC radius: {reference['R_NSC']:.6g} pc; density slope: {reference['gamma_n']:.6g}." if legacy else
         f"- Pan Tremaine spatial profile: gamma={reference['gamma_n']:.6g}, reference r_a={pan_scale_radius_cm(REFERENCE_M) / const.parsec:.6g} pc, with r_a tracking the published M-sigma relation as M_bullet varies. The empirical NSC radius input is unused."),
        f"- Fixed alpha={reference['alpha']:.6g}, absolute Mdot={reference['Mdot']:.6g} g/s, physical outer disk radius={reference['r_out']:.6g} cm.",
        (f"- Legacy Pop III prescription; IMF slope {manifest.get('imf', {}).get('slope', 1.77):g}, mass range {manifest.get('imf', {}).get('minimum_msun', 13.6):g}–{manifest.get('imf', {}).get('maximum_msun', 197):g} Msun, Z=1e-5 Zsun." if legacy else
         "- Pan components: 1-Msun representative long-lived stars plus pre-existing 10-Msun BHs, N_BH/N_star=1e-3. M_NSC is total component mass in this conditional NSC-mass experiment; it is not the paper's fixed M_star=20*M_bullet normalization."),
        ("- Non-default legacy IMF bounds extrapolate the lifetime fit; remnant endpoints and mass loss are not calibrated." if legacy else
         "- Full Tremaine spatial tracer shape is sampled through an isotropic Eddington inversion in the Kepler potential, not by assigning a spatial-radius CDF directly to semimajor axis. Cluster self-gravity and Pan's Fokker-Planck evolution are not reproduced. There is no automatic stellar-to-BH lifetime conversion for representative 1-Msun stars."),
        "- LISA: 4-year mission, SNR threshold 20, observed band 1e-3 to 1e-1 Hz.", "",
        "## Rate and uncertainty definitions", "",
        "- Rates are counts divided by the 10-Myr source-frame AGN duration in this one simulated system. Units are Myr^-1; there is no host-number or volume normalization and no observer-time (1+z) conversion.",
        "- Damping counts successful stellar plus BH embeddings, including replacements; it is not the number of attempted damping tasks or inverse individual damping time.",
        "- IMRI counts physical first entry into the specified observed GW band, not ISCO mergers. Detectable counts the subset passing the existing SNR selection.",
        "- Points show repeat means; error bars are Monte Carlo SEM, clipped at zero for display. Three repeats are exploratory; zero events are not a zero upper confidence limit. Lines are guides through samples, not fits.",
        "- Missing/incomplete evaluations remain NaN rather than being replaced by zero. Per-event classification flags are checked independently: at the high-mass endpoint, the core aggregate exports zero/complete even though its events are marked not evaluated (isco_before_entry). This report preserves those event-level unknowns and records the original core statuses separately.",
        f"- Display: {manifest.get('yscale', 'symlog')} shared vertical axis; symlog is linear below 0.1 Myr^-1 and logarithmic above. Definitions and settings are deliberately outside the figure.", "",
        "## Interpretation cautions", "",
        "- The redshift scan does not evolve the NSC or disk population cosmologically. It changes the observed/source band mapping, cosmic-time horizon and detectability; the model's replenishment eligibility may also feed back on total embedding counts. Primary-only damping is saved separately.",
        ("- Mass scans hold absolute Mdot, disk radius and NSC radius fixed." if legacy else
         "- Mass scans hold absolute Mdot and disk outer edge fixed, but the Pan population scale follows the published M-sigma relation. Thus this is a population-prescription change, not an IMF-slope-only comparison."),
        "- Across the two-decade SMBH-mass scan, fixed absolute Mdot makes the Eddington-scaled rate vary inversely with mass: at 1e4 Msun it is ten times its reference value. The low-mass end can leave the calibrated thin-disk regime; do not treat these conditional solver outputs as validated super-Eddington predictions.",
        "- A fixed outer disk radius need not remain at Q=1 as the SMBH mass changes. The boundary experiment is phenomenological.",
        "- At M_bullet=1e6 Msun and z=8–15, the observed ISCO frequency is below the retained 1-mHz lower band edge. The existing model flags these intrinsic/detector evaluations as unavailable; their curves have gaps rather than fabricated zero rates. Damping is still evaluated. A lower-frequency-band study would be a separate experiment.", "",
        "## Measured rates", "",
        f"Reference mean ± SEM: damping {baseline.damping_rate:.4g} ± {baseline.damping_sem:.3g}, IMRI {baseline.event_rate:.4g} ± {baseline.event_sem:.3g}, detectable {baseline.detect_rate:.4g} ± {baseline.detect_sem:.3g} Myr^-1.", "",
        "| z | M_bullet [Msun] | M_NSC [Msun] | Damping [Myr^-1] | IMRI [Myr^-1] | Detectable [Myr^-1] |", "|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in frame.iterrows():
        lines.append(f"| {row.z:g} | {row.M:g} | {row.M_NSC:g} | {row.damping_rate:.4g} | {row.event_rate:.4g} | {row.detect_rate:.4g} |")
    redshift = select_panel(frame, comparison_design()["redshift"])
    nsc = select_panel(frame, comparison_design()["nsc_mass"])
    smbh = select_panel(frame, comparison_design()["smbh_mass"])
    finite_smbh = smbh[np.isfinite(smbh.event_rate)]
    lines.extend([
        "", "## Main-text takeaways", "",
        f"- In this controlled sample, the detectable rate changes from {redshift.detect_rate.iloc[0]:.4g} Myr^-1 at z={redshift.z.iloc[0]:g} to {redshift.detect_rate.iloc[-1]:.4g} Myr^-1 at z={redshift.z.iloc[-1]:g}. A zero sampled rate does not establish a strict detectability cutoff.",
        "- Damping and intrinsic-entry redshift responses should be read as results for a fixed population, not as a prediction of cosmological AGN-population evolution.",
        "- Intrinsic and detectable curves can coincide at low redshift because every intrinsic event in the finite sample passes the SNR selection; they are still separate quantities.",
        f"- Across M_NSC={nsc.M_NSC.iloc[0]:g}–{nsc.M_NSC.iloc[-1]:g} Msun, the sampled damping rate changes from {nsc.damping_rate.iloc[0]:.4g} to {nsc.damping_rate.iloc[-1]:.4g} Myr^-1 and intrinsic-entry rate from {nsc.event_rate.iloc[0]:.4g} to {nsc.event_rate.iloc[-1]:.4g} Myr^-1.",
        "- NSC-mass and SMBH-mass curves are exploratory finite-sampling comparisons; do not treat point-to-point wiggles as resolved physical trends.",
    ])
    if len(finite_smbh) > 1:
        lines.append(f"- Over evaluated SMBH masses {finite_smbh.M.iloc[0]:g}–{finite_smbh.M.iloc[-1]:g} Msun, the sampled intrinsic-entry rate changes from {finite_smbh.event_rate.iloc[0]:.4g} to {finite_smbh.event_rate.iloc[-1]:.4g} Myr^-1. Check the disk-regime and observed-band caveats before interpreting this response.")
    (output / "ppt_notes.md").write_text("\n".join(lines) + "\n")


def compatible_run(path, config):
    metadata = path / "simulation_metadata.json"
    if not metadata.exists():
        return False
    parameters = json.loads(metadata.read_text())["parameters"]
    fields = ("smbh_masses_msun", "nsc_masses_msun", "num_repeats", "base_seed", "redshift", "duty_cycle_myr", "disk_model", "population_model", "metallicity", "popiii_min_mass_msun", "popiii_max_mass_msun", "popiii_imf_slope", "initial_stellar_density_gamma", "nsc_radius_pc_at_reference_mass", "nsc_radius_mass_slope", "nsc_radius_reference_mass_msun", "disk_alpha", "disk_accretion_rate_g_s", "disk_outer_radius_cm", "stellar_evolution_model", "natal_kick_model", "use_disk_inner_radius", "lisa_min_gw_frequency_hz", "lisa_max_gw_frequency_hz", "snr_threshold")
    for field in fields:
        value = getattr(config, field)
        if isinstance(value, np.ndarray):
            value = value.tolist()
        saved_value = parameters.get(field, 13.6 if field == "popiii_min_mass_msun" else LEGACY_POPIII if field == "population_model" else None)
        if saved_value != value:
            return False
    if config.population_model != LEGACY_POPIII and parameters.get("population_prescription") != population_description(config.population_model):
        return False
    if parameters.get("consumption_replenishment", {}).get("enabled") != config.consumption_replenishment:
        return False
    return all((path / filename).exists() for filename in ("imri_repeat_results.csv", "imri_source_transitions.csv", "imri_events.csv"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--population-model", choices=POPULATION_MODELS, default=None)
    parser.add_argument("--popiii-min-mass-msun", type=float, default=13.6)
    parser.add_argument("--popiii-max-mass-msun", type=float, default=197.0)
    parser.add_argument("--reuse-dir", type=Path, default=Path("outputs/parameter_dependence"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--yscale", choices=("linear", "symlog"), default="symlog")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--reference-only", action="store_true", help="Run only the shared reference, for an inexpensive IMF control")
    args = parser.parse_args(argv)
    custom_imf = (args.popiii_min_mass_msun, args.popiii_max_mass_msun) != (13.6, 197.0)
    model = args.population_model or LEGACY_POPIII
    if model != LEGACY_POPIII and custom_imf:
        parser.error("Published population presets do not use Pop III IMF bounds; select legacy_popiii explicitly for an ad hoc IMF test")
    if model in (PAN_PAPER, XUE_BH) and not (args.reference_only or args.plot_only):
        parser.error("This paper normalization has no independent M_NSC; use --reference-only, not a fictitious NSC-mass panel")
    output = args.output_dir or Path(
        f"outputs/{model}_rates" if model != LEGACY_POPIII else
        "outputs/high_redshift_rates" if not custom_imf else
        f"outputs/imf_{args.popiii_min_mass_msun:g}_{args.popiii_max_mass_msun:g}_rates"
    )
    output.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        print(f"Population preset: {model}", flush=True)
        if args.population_model is None and custom_imf:
            print("Custom IMF bounds select the archived legacy population, not a published component preset.", flush=True)
    if args.plot_only:
        frame = pd.read_csv(output / "host_rates.csv")
        manifest = json.loads((output / "experiment.json").read_text())
        manifest["yscale"] = args.yscale
        manifest["y_limits"] = plot_comparisons(frame, output, args.yscale)
        (output / "experiment.json").write_text(json.dumps(manifest, indent=2))
        write_ppt_notes(frame, output, manifest)
        return
    cli = build_parser().parse_args([
        str(args.repeats), str(output), str(args.workers),
        "--smbh-masses", str(REFERENCE_M), "--nsc-masses", str(REFERENCE_NSC),
        "--base-seed", str(args.seed), "--population-model", model,
        "--popiii-min-mass-msun", str(args.popiii_min_mass_msun),
        "--popiii-max-mass-msun", str(args.popiii_max_mass_msun),
    ])
    reference = reference_values(build_config_from_arguments(cli))
    points = ([dict(z=REFERENCE_Z, M=REFERENCE_M, M_NSC=REFERENCE_NSC)]
              if args.reference_only else unique_points())
    manifest = {
        "reference": reference, "seed": args.seed, "repeats": args.repeats, "yscale": args.yscale,
        "design": "reference_only" if args.reference_only else comparison_design(), "points": points,
        "reference_redshift": REFERENCE_Z,
        "population_model": model,
        "population_prescription": population_description(model),
        "imf": {"minimum_msun": args.popiii_min_mass_msun,
                "maximum_msun": args.popiii_max_mass_msun, "slope": 1.77,
                "mass_budget": "same total NSC mass, newly sampled population",
                "interpretation": "legacy-only parameters; unused by published component presets",
                "used": model == LEGACY_POPIII},
        "scenario": "early_Universe_high_redshift_uncertain_massive_NSC",
        "nsc_mass_prior": "1e6–1e7 Msun is an exploratory uncertain range, not an empirically calibrated prior",
        "fixed_inputs": ("Absolute Mdot, disk edge, NSC radius, alpha, IMF, Z, duration, detector and migration fixed" if model == LEGACY_POPIII else
                         "Absolute Mdot, disk edge, alpha, duration, detector and migration fixed; published population radial scale tracks M_bullet"),
        "damping_rate": "Successful embeddings (stars + BHs, primaries + replacements) / AGN source-frame Myr; not attempted damping tasks or 1/t_damp.",
        "event_rate": "Physical first observed-band IMRI entries / source-frame AGN Myr, not ISCO merger rate.",
        "detect_rate": "Subset of physical entries passing the existing LISA SNR selection / source-frame AGN Myr.",
        "caveats": ["No 1/(1+z) observer-time conversion is applied.", "No cosmological evolution of the fixed NSC/disk population is introduced; z affects band, cosmic-time horizon, SNR and possibly the existing replenishment eligibility.", "Primary-only damping is saved to distinguish direct damping from replenishment feedback.", "Fixed physical outer radius is phenomenological: it need not remain at Q=1 when M changes.", "Fixed absolute Mdot makes its Eddington-scaled value ten times the reference at M=1e4 Msun; thin-disk validity at that endpoint is not guaranteed.", "At high z and M=1e6 Msun the 1-mHz observed band starts beyond ISCO; existing unevaluated rate flags are preserved as gaps.", "All plots share one global y range; symlog retains true zeros. Error bars are sampling SEM, not confidence limits.", "Lines connect sampled means; no interpolation-based new rates or significance claims."],
    }
    (output / "experiment.json").write_text(json.dumps(manifest, indent=2))
    candidates = sorted(args.reuse_dir.glob("semi_type_i/point_*"))
    rows, repeat_tables = [], []
    for index, point in enumerate(points):
        config = build_config_from_arguments(cli)
        values = dict(reference, M=point["M"], M_NSC=point["M_NSC"])
        apply_values(config, values)
        config.redshift = point["z"]
        target = output / "simulations" / f"point_{index:03d}"
        config.output_dir = target
        source = next((path for path in [target, *candidates] if compatible_run(path, config)), None)
        print(f"{index + 1}/{len(points)}: {point}" + (f" [reuse {source}]" if source else ""), flush=True)
        if source is None:
            target.mkdir(parents=True, exist_ok=True)
            with (target / "run.log").open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                run_simulation(config)
            source = target
        summary, repeats = summarize_run(source)
        rows.append(dict(point=index, **point, **summary))
        for key, value in dict(point=index, **point).items():
            repeats[key] = value
        repeat_tables.append(repeats)
        pd.DataFrame(rows).to_csv(output / "host_rates.csv", index=False)
        pd.concat(repeat_tables, ignore_index=True).to_csv(output / "repeat_rates.csv", index=False)
    if not args.reference_only:
        manifest["y_limits"] = plot_comparisons(pd.DataFrame(rows), output, args.yscale)
        write_ppt_notes(pd.DataFrame(rows), output, manifest)
    (output / "experiment.json").write_text(json.dumps(manifest, indent=2))
    print(f"Reference rates saved in {output / 'host_rates.csv'}" if args.reference_only
          else f"Three-panel figure saved in {output / 'rate_dependence.png'}")


if __name__ == "__main__":
    main()
