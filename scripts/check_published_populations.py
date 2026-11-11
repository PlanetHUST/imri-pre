#!/usr/bin/env python3
"""Check component normalization and radial-law definitions against Pan/Xue.

This is a population census, not a new capture-rate simulation.
"""
import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from workflows.cli import build_parser, build_config_from_arguments
from workflows.engine import IMRIDynEngine
from workflows.population import sample_disk_crossing_population
from physical_process import constants as const
from physical_process.population_models import (
    LEGACY_POPIII, PAN_NSC, PAN_PAPER, XUE_BH,
    pan_orbits, pan_scale_radius_cm, tremaine_spatial_cdf, population_description,
)


def check_radial_cdf(count=120000):
    """Draw phases independently to validate actual r, not just a."""
    state = np.random.get_state()
    try:
        np.random.seed(32021)
        minimum = 6 * const.G * 1e5 * const.msun / const.c**2
        a, e, inclination, omega, scale = pan_orbits(count, 1e5, minimum, 1.5)
        mean_anomaly = np.random.uniform(0, 2*np.pi, count)
        eccentric_anomaly = mean_anomaly.copy()
        for _ in range(60):
            step = (eccentric_anomaly - e*np.sin(eccentric_anomaly) - mean_anomaly) / (1 - e*np.cos(eccentric_anomaly))
            eccentric_anomaly -= np.clip(step, -1, 1)
        if np.max(abs(eccentric_anomaly - e*np.sin(eccentric_anomaly) - mean_anomaly)) > 1e-9:
            raise RuntimeError("Kepler phase validation did not converge")
        radius = a * (1 - e*np.cos(eccentric_anomaly))
        uniform = np.sort(tremaine_spatial_cdf(radius, scale, 1.5))
        deviation = float(np.max(abs(uniform - (np.arange(count) + .5) / count)))
        return dict(samples=count, spatial_cdf_max_deviation=deviation,
                    mean_cos_inclination=float(np.cos(inclination).mean()),
                    mean_eccentricity_squared=float((e*e).mean()),
                    pan_scale_radius_pc=scale / const.parsec)
    finally:
        np.random.set_state(state)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/archive/populations/published_population_check"))
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = build_config_from_arguments(build_parser().parse_args(["--smbh-masses", "1e5", "--nsc-masses", "3e6"]))
    disk = IMRIDynEngine(base).get_disk_profiles(1e5)
    rows = []
    for label, model, low, high in (
        ("Legacy 13.6–197", LEGACY_POPIII, 13.6, 197),
        ("Ad hoc 1–400", LEGACY_POPIII, 1, 400),
        ("Pan conditional M_NSC", PAN_NSC, 13.6, 197),
        ("Pan paper normalization", PAN_PAPER, 13.6, 197),
        ("Xue BH mass/radial proxy", XUE_BH, 13.6, 197),
    ):
        config = build_config_from_arguments(build_parser().parse_args([
            "--smbh-masses", "1e5", "--nsc-masses", "3e6", "--population-model", model,
            "--popiii-min-mass-msun", str(low), "--popiii-max-mass-msun", str(high),
        ]))
        np.random.seed(2893371290)
        pop = sample_disk_crossing_population(1e5, 3e6, disk, config)
        mass = pop.initial["mass_msun"]
        clocks = pop.initial["formation_time_myr"]
        rows.append(dict(label=label, model=model, objects=len(mass),
            mean_mass_msun=float(mass.mean()), actual_sample_mass_msun=float(mass.sum()),
            stars_at_agn_start=int((clocks > 0).sum()), preexisting_bhs=int((clocks == 0).sum()),
            disk_crossers=pop.disk_crossing_count, fraction_crossing=pop.geometric_fraction,
            crossing_preexisting_bhs=int((pop.disk_crossing["formation_time_myr"] == 0).sum()),
            gamma_parameter=config.initial_stellar_density_gamma,
            nsc_input_used=population_description(model)["nsc_mass_input_used"],
            population_characteristic_radius_pc=pop.initial["nsc_radius_cm"] / const.parsec))
        print(rows[-1], flush=True)
        del pop, mass, clocks
        gc.collect()
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "population_normalization.csv", index=False)
    radial = check_radial_cdf()
    if radial["spatial_cdf_max_deviation"] > .01:
        raise RuntimeError("Tremaine spatial-profile check failed")
    (args.output_dir / "radial_checks.json").write_text(json.dumps(radial, indent=2))
    (args.output_dir / "prescriptions.json").write_text(json.dumps({model: population_description(model) for model in (LEGACY_POPIII, PAN_NSC, PAN_PAPER, XUE_BH)}, indent=2))
    lines = ["# Published-population normalization and radial-law check", "",
        "Reference M_bullet=1e5 Msun; M_NSC input=3e6 Msun. This is a seeded initial-population census, NOT newly computed capture or event rates.", "",
        "| Population | Objects | Mean mass [Msun] | Actual component mass [Msun] | Existing BHs | Disk crossers |", "|---|---:|---:|---:|---:|---:|"]
    for _, row in frame.iterrows():
        lines.append(f"| {row.label} | {row.objects:,.0f} | {row.mean_mass_msun:.6g} | {row.actual_sample_mass_msun:,.3f} | {row.preexisting_bhs:,.0f} | {row.disk_crossers:,.0f} |")
    lines.extend(["", "## The radial exponents are NOT interchangeable", "",
        "- Legacy: p(a) proportional to a^(2-gamma), gamma=1.75, so dN/da proportional to a^0.25. The angular sampler was also non-isotropic (uniform i on 10–170 degrees and uniform cos omega on 0–1).",
        "- Pan & Yang 2021 Eq. (3): n(r) proportional to r^(-gamma)*(r+r_a)^(gamma-4), fiducial gamma=1.5, r_a=4 r_h. It is a BROKEN spatial density: inner slope -1.5, outer slope -4. Its shell number density is dN/dr proportional to r^2*n(r), not n(r) itself. It is not a global a^0.5 law.",
        "- Pan's full spatial tracer shape is now sampled using Eddington inversion in the Kepler potential with isotropic orbital angles and thermal eccentricities. The spatial-radius CDF is not incorrectly assigned directly to a. The check below draws random orbital phases and compares actual r to the analytic Tremaine CDF.",
        f"- With {radial['samples']} validation draws, the maximum spatial-CDF deviation is {radial['spatial_cdf_max_deviation']:.6g}; r_a={radial['pan_scale_radius_pc']:.6g} pc at the reference SMBH mass.",
        "- Xue et al. 2026: dN/dr proportional to r^(gamma_rho), gamma_rho=0. If converted to a spherical spatial density, n(r) is proportional to r^-2, NOT constant density. The implemented comparison uses a circular-radius proxy (r=a), so the radial number law is explicit. Its isotropic orientations are NOT Xue's Gaussian-velocity prescription; it is only a mass/count/radial control.", "",
        "## What is aligned and what is not", "",
        "- Pan component masses and number ratio are implemented: 1-Msun representative stars, pre-existing 10-Msun BHs, N_BH/N_star=1e-3. Stars have no automatic BH-birth clock; initial BHs use formation time zero.",
        "- pan2021_nsc preserves M_NSC as TOTAL stellar+BH mass so the requested NSC-mass dependency can be tested. This is a conditional population variant, not the published fixed M_star=20 M_bullet normalization.",
        "- pan2021_paper implements M_star=20 M_bullet and its associated BH abundance. The M_NSC input is unused; it cannot produce an independent M_NSC-dependence panel.",
        "- xue2026_bh uses 5–15 Msun BHs, dN/dm proportional to m^-2.35, and N_BH=floor(M_bullet/20 Msun). M_NSC input is unused, and it is not a complete stellar population. It cannot be normalized to all of M_NSC or called a 1–400 Msun stellar IMF.",
        "- Pan's cluster contribution to the gravitational potential and Fokker-Planck evolution are not reproduced; the orbit engine is still Keplerian. Xue's accretion, mergers, recoil and published kinematics are not reproduced. Disk and detector differences remain. None of the presets claims a complete reproduction of the published event rates.", "",
        "The published-default switch has been rolled back: host comparisons default to the original legacy_popiii IMF; published presets require explicit selection. Historical Pop III / 1–400 rates remain archived and are not rescaled or relabeled. Fresh rate simulations are required with the new population; old cache entries are rejected by population-model identity.", "",
        "Sources checked locally: Pan & Yang, PRD 103, 103018 (2021), arXiv:2101.09146v6, Eq. (3), Eq. (19), Sec. II.C.3; Xue et al., arXiv:2605.29305v1 (2026), Sec. II.A and Table I.",
    ])
    pilot_rows = []
    for model in (PAN_NSC, PAN_PAPER, XUE_BH):
        result_path = Path("outputs/archive/populations/published_population_validation") / model / "host_rates.csv"
        if result_path.exists():
            pilot = pd.read_csv(result_path).iloc[0]
            stock = frame[frame.model == model].iloc[0]
            pilot_rows.append(dict(model=model, actual_initial_component_mass_msun=stock.actual_sample_mass_msun,
                nsc_mass_input_used=bool(stock.nsc_input_used), repeats=int(pilot.repeats),
                damping_rate=pilot.damping_rate, event_rate=pilot.event_rate,
                detect_rate=pilot.detect_rate, primary_damping_rate=pilot.primary_damping_rate,
                run_dir=pilot.run_dir))
    if pilot_rows:
        pd.DataFrame(pilot_rows).to_csv(args.output_dir / "reference_pilots.csv", index=False)
        lines.extend(["", "## Fresh reference smoke runs (not converged rate estimates)", "",
            "These are one-realization, 10-Myr end-to-end runs at M_bullet=1e5 Msun and z=10. Their component masses and normalization differ; they are NOT isolated IMF changes or reproductions of published rates.", "",
            "| Preset | Actual initial component mass [Msun] | Damping [Myr^-1] | IMRI [Myr^-1] | Detectable [Myr^-1] |", "|---|---:|---:|---:|---:|"])
        for row in pilot_rows:
            lines.append(f"| {row['model']} | {row['actual_initial_component_mass_msun']:,.3f} | {row['damping_rate']:.4g} | {row['event_rate']:.4g} | {row['detect_rate']:.4g} |")
        lines.extend(["", "The revised Pan NSC population supplies many more stellar embeddings, but its small initial BH number fraction does not imply a similarly high BH-entry count. The BH-only Xue proxy has a BH-only donor pool and circular starting orbits; repeated eligible refills can dominate its total over its primary-only damping rate. Its published velocity model, recoil, mergers, growth and detector waveform are absent, so this is NOT Xue's rate prediction.",
            "The archived 13-point IMF figures have not been relabeled or multiplied by a number-count ratio. A full new rate grid and convergence study remain necessary; the known eccentric-handoff and stellar-disruption limitations are not repaired by population replacement."])
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
