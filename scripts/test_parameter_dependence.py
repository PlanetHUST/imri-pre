#!/usr/bin/env python3
"""Controlled +/-50% initial-condition experiment (not a unit-test shortcut)."""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from workflows.cli import build_config_from_arguments, build_parser
from workflows.config import DISK_ACCRETION_RATE_G_S_PER_MSUN, DISK_MODELS
from workflows.engine import IMRIDynEngine
from workflows.workflow import run_simulation

PARAMETERS = ("alpha", "M", "Mdot", "r_out", "gamma_n", "gamma_m", "m_max", "Z", "R_NSC", "M_NSC")
LABELS = (r"$\alpha$", r"$M_\bullet$", r"$\dot M$", r"$r_{\rm out}$", r"$\gamma_n$", r"$\gamma_m$", r"$m_{\max}$", r"$Z$", r"$R_{\rm NSC}$", r"$M_{\rm NSC}$")
FACTORS = (0.5, 1.0, 1.5)


def reference_values(config):
    """Calibrate dimensional radii once, using the unmodified default Q=1 disk."""
    mass = float(config.smbh_masses_msun[0])
    nsc_mass = float(config.nsc_masses_msun[0])
    edge = IMRIDynEngine(config).get_disk_profiles(mass)["outer_radius_cm"]
    radius_pc = config.nsc_radius_pc_at_reference_mass * (nsc_mass / config.nsc_radius_reference_mass_msun) ** config.nsc_radius_mass_slope
    return dict(zip(PARAMETERS, (
        config.disk_alpha, mass, DISK_ACCRETION_RATE_G_S_PER_MSUN * mass,
        edge, config.initial_stellar_density_gamma, config.popiii_imf_slope,
        config.popiii_max_mass_msun, config.metallicity, radius_pc, nsc_mass,
    )))


def apply_values(config, values):
    """Map independent physical inputs onto workflow controls; hold others fixed."""
    config.disk_alpha = values["alpha"]
    config.smbh_masses_msun = np.array([values["M"]])
    config.disk_accretion_rate_g_s = values["Mdot"]
    config.disk_outer_radius_cm = values["r_out"]
    config.initial_stellar_density_gamma = values["gamma_n"]
    config.popiii_imf_slope = values["gamma_m"]
    config.popiii_max_mass_msun = values["m_max"]
    config.metallicity = values["Z"]
    config.nsc_masses_msun = np.array([values["M_NSC"]])
    # Compensate the empirical mass-radius relation so M_NSC is independent.
    config.nsc_radius_pc_at_reference_mass = values["R_NSC"] / (values["M_NSC"] / config.nsc_radius_reference_mass_msun) ** config.nsc_radius_mass_slope


def design():
    """21 OAT points plus four off-axis points for a measured 3x3 phase plane."""
    points = {tuple([1.0] * len(PARAMETERS))}
    for index in range(len(PARAMETERS)):
        for factor in (0.5, 1.5):
            factors = [1.0] * len(PARAMETERS)
            factors[index] = factor
            points.add(tuple(factors))
    for alpha in FACTORS:
        for mdot in FACTORS:
            factors = [1.0] * len(PARAMETERS)
            factors[0], factors[2] = alpha, mdot
            points.add(tuple(factors))
    return sorted(points)


def plot_results(frame, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = frame.model.unique()
    fig, axes = plt.subplots(len(models), 2, figsize=(13, 4.5 * len(models)), squeeze=False, constrained_layout=True)
    for row_index, model in enumerate(models):
        data = frame[frame.model == model]
        sensitivity = np.full((len(PARAMETERS), len(FACTORS)), np.nan)
        errors = np.full_like(sensitivity, np.nan)
        for i, parameter in enumerate(PARAMETERS):
            other = [f"factor_{p}" for p in PARAMETERS if p != parameter]
            selected = data[(data[other] == 1.0).all(axis=1)]
            for j, factor in enumerate(FACTORS):
                cell = selected.loc[selected[f"factor_{parameter}"] == factor].iloc[0]
                sensitivity[i, j] = cell["rate_imri_per_myr"]
                errors[i, j] = cell["intrinsic_rate_sem"]
        ax = axes[row_index, 0]
        image = ax.imshow(np.ma.masked_invalid(sensitivity), aspect="auto", origin="upper", cmap="viridis")
        ax.set(xticks=range(3), xticklabels=["-50%", "default", "+50%"], yticks=range(10), yticklabels=LABELS, title=f"{model}: one parameter at a time")
        for i in range(10):
            for j in range(3):
                value = sensitivity[i, j]
                label = f"{value:.2g}" if np.isfinite(value) else "N/A"
                if np.isfinite(value) and np.isfinite(errors[i, j]):
                    label += f" ± {errors[i, j]:.1g}"
                color = "black" if np.isfinite(value) and image.norm(value) > 0.65 else "white"
                ax.text(j, i, label, ha="center", va="center", color=color, fontsize=9)
        fig.colorbar(image, ax=ax, label=r"Intrinsic rate [host$^{-1}$ Myr$^{-1}$]")
        other = [f"factor_{p}" for p in PARAMETERS if p not in ("alpha", "Mdot")]
        plane = data[(data[other] == 1.0).all(axis=1)].pivot(index="factor_Mdot", columns="factor_alpha", values="rate_imri_per_myr").reindex(index=FACTORS, columns=FACTORS)
        ax = axes[row_index, 1]
        image = ax.imshow(np.ma.masked_invalid(plane.to_numpy()), origin="lower", extent=(0.25, 1.75, 0.25, 1.75), cmap="viridis", aspect="auto")
        ax.set(xticks=FACTORS, yticks=FACTORS, xlabel=r"$\alpha/\alpha_0$", ylabel=r"$\dot M/\dot M_0$", title="Measured phase-space slice (other inputs fixed)")
        for i, mdot in enumerate(FACTORS):
            for j, alpha in enumerate(FACTORS):
                value = plane.iloc[i, j]
                color = "black" if np.isfinite(value) and image.norm(value) > 0.65 else "white"
                ax.text(alpha, mdot, f"{value:.2g}" if np.isfinite(value) else "N/A", ha="center", va="center", color=color)
        fig.colorbar(image, ax=ax, label=r"Intrinsic rate [host$^{-1}$ Myr$^{-1}$]")
    manifest = json.loads((output / "experiment.json").read_text())
    reference = manifest["baseline"]
    fig.suptitle(
        "AGN-disk IMRI sensitivity: +/-50% independent inputs; OAT labels: mean ± MC SEM\n"
        f"Reference: M={reference['M']:.2g} Msun, M_NSC={reference['M_NSC']:.2g} Msun, "
        f"Z={reference['Z']:.2g} Zsun; {manifest['repeats']} repeats; z=15, AGN=10 Myr"
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"imri_parameter_phase_space.{suffix}", dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/supporting/parameter_dependence"))
    parser.add_argument("--smbh-mass", type=float, default=1e5, help="Reference member of default SMBH grid")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--models", nargs="+", choices=DISK_MODELS, default=["semi_type_i"])
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args(argv)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        plot_results(pd.read_csv(output / "parameter_dependence.csv"), output)
        return
    base_args = build_parser().parse_args([str(args.repeats), str(output), str(args.workers), "--smbh-masses", str(args.smbh_mass), "--base-seed", str(args.seed), "--population-model", "legacy_popiii"])
    base = build_config_from_arguments(base_args)
    baseline = reference_values(base)
    manifest = {
        "baseline": baseline, "units": {"M": "Msun", "Mdot": "g/s", "r_out": "cm", "R_NSC": "pc", "M_NSC": "Msun", "Z": "Zsun", "m_max": "Msun"},
        "factors": FACTORS, "seed": args.seed, "repeats": args.repeats, "models": args.models,
        "design": "one-at-a-time plus alpha-Mdot 3x3; independent dimensional inputs",
        "rate": "intrinsic observed-band first-entry count / source-frame AGN duty Myr / host; not volumetric or observer-time rate",
        "caveats": ["Reference M=1e5 is one member of the CLI default mass grid; all other physical defaults retained.", "Explicit r_out is calibrated to the default Q=1 edge; +50% can include Q<1 gas, a phenomenological boundary experiment.", "Z only switches Pop III/II at 1e-4 Zsun; +/-50% about 1e-5 stays Pop III and must have no effect.", "gamma_m and m_max control the Pop III IMF, not a remnant IMF; default fixed-mass backend has no metallicity-dependent remnant evolution.", "Paired seeds reduce noise but population-size changes do not give object-by-object matching.", "Three repeats give exploratory Monte Carlo errors, not converged astrophysical constraints."],
    }
    (output / "experiment.json").write_text(json.dumps(manifest, indent=2))
    rows = []
    for model in args.models:
        for index, factors in enumerate(design()):
            config = build_config_from_arguments(base_args)
            values = {name: baseline[name] * factor for name, factor in zip(PARAMETERS, factors)}
            apply_values(config, values)
            config.disk_model = model
            config.output_dir = output / model / f"point_{index:03d}"
            config.output_dir.mkdir(parents=True, exist_ok=True)
            print(f"{model} {index + 1}/{len(design())}: {factors}", flush=True)
            with (config.output_dir / "run.log").open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                result = run_simulation(config).iloc[0]
            repeat_frame = pd.read_csv(config.output_dir / "imri_repeat_results.csv")
            rates = repeat_frame["rate_imri_per_myr"].to_numpy(dtype=float)
            row = dict(model=model, point=index, **values)
            row.update({f"factor_{name}": factor for name, factor in zip(PARAMETERS, factors)})
            for column in ("rate_status", "detector_rate_status", "rate_imri_per_myr", "rate_det_per_myr", "n_imri", "n_det"):
                row[column] = result[column]
            row["intrinsic_rate_sem"] = float(np.std(rates, ddof=1) / np.sqrt(len(rates))) if len(rates) > 1 and np.isfinite(rates).all() else np.nan
            rows.append(row)
            pd.DataFrame(rows).to_csv(output / "parameter_dependence.csv", index=False)
    plot_results(pd.DataFrame(rows), output)
    print(f"Results and plots: {output}")


if __name__ == "__main__":
    main()
