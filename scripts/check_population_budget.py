#!/usr/bin/env python3
"""Trace population -> disk crossers -> embeddings -> BH entries, without changing rates."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from check_detect_process import config_for_run
from workflows.engine import IMRIDynEngine
from workflows.population import sample_disk_crossing_population
from workflows.workers import build_damping_tasks, run_damping_tasks_parallel
from physical_process import constants as const
from physical_process.agn import AGNProperties
from physical_process.damp import DampingCalculator, OrbitalState, STELLAR_MASS_RADIUS_EXPONENT
from physical_process.nsc_orbital import disk_crossing_mask
from physical_process.orbital_evolution import peters_beta_cm4_s
from lisa.snr import evaluate_lisa_inspiral_snr

SUCCESS = ("embedded_star", "embedded_bh")


def population_config(path):
    config = config_for_run(path)
    parameters = json.loads((path / "simulation_metadata.json").read_text())["parameters"]
    for key in ("metallicity", "popiii_min_mass_msun", "popiii_max_mass_msun", "popiii_imf_slope", "initial_stellar_density_gamma", "nsc_radius_pc_at_reference_mass", "nsc_radius_mass_slope", "nsc_radius_reference_mass_msun", "use_disk_inner_radius"):
        setattr(config, key, parameters.get(key, 13.6) if key == "popiii_min_mass_msun" else parameters[key])
    return config


def census_run(path):
    repeats = pd.read_csv(path / "imri_repeat_results.csv")
    transitions = pd.read_csv(path / "imri_source_transitions.csv")
    events = pd.read_csv(path / "imri_events.csv")
    damping = transitions[transitions.stage == "damping"]
    rows = []
    for _, repeat in repeats.iterrows():
        d = damping[damping["repeat"] == repeat["repeat"]]
        primary = d[d.task_kind == "primary"]
        replacement = d[d.task_kind == "replacement"]
        e = events[events["repeat"] == repeat["repeat"]]
        primary_success = int(primary.status.isin(SUCCESS).sum())
        if len(primary) != repeat.n_disk_cross:
            raise ValueError("Primary damping count does not equal geometric selection")
        if len(replacement) != repeat.n_replenished:
            raise ValueError("Replacement task count mismatch")
        if int(d.status.isin(SUCCESS).sum()) != repeat.n_emb_star + repeat.n_emb_bh:
            raise ValueError("Embedding count mismatch")
        row = dict(repeat=int(repeat["repeat"]), task_seed=int(repeat.task_seed),
            z=repeat.redshift, M=repeat.m_smbh_msun, M_NSC=repeat.m_nsc_msun,
            duty_myr=repeat.duty_cycle_myr, sampled=int(repeat.n_sampled),
            initial_crossers=len(primary), crossing_fraction=len(primary) / repeat.n_sampled,
            primary_embedded=primary_success,
            primary_embedded_star=int((primary.status == "embedded_star").sum()),
            primary_embedded_bh=int((primary.status == "embedded_bh").sum()),
            primary_incomplete=int((primary.status == "damping_incomplete").sum()),
            primary_incomplete_before_deadline=int(((primary.status == "damping_incomplete") & (primary.time_myr < repeat.duty_cycle_myr)).sum()),
            primary_outside=int((primary.status == "outside_disk").sum()),
            primary_errors=int(primary.status.isin(("calculation_error", "stellar_unavailable")).sum()),
            replacements=len(replacement), replacement_embedded=int(replacement.status.isin(SUCCESS).sum()),
            total_embedded=int(d.status.isin(SUCCESS).sum()),
            embedded_star=int(repeat.n_emb_star), embedded_bh=int(repeat.n_emb_bh),
            migration_formed_bh=int(repeat.n_mfbh),
            no_bh_at_band=int((e.event_rate_reason == "bh_not_formed_by_band_entry").sum()),
            imri_entries=int(repeat.n_imri), detected=int(repeat.n_det),
            damping_rate=(repeat.n_emb_star + repeat.n_emb_bh) / repeat.duty_cycle_myr,
            imri_rate=repeat.rate_imri_per_myr, detected_rate=repeat.rate_det_per_myr,
            geometric_selection_averaged_over_duration=len(primary) / repeat.duty_cycle_myr,
            success_given_crossing=primary_success / len(primary) if len(primary) else np.nan,
            run_dir=str(path))
        rows.append(row)
    return pd.DataFrame(rows), damping, events


def replay_primary_damping(path, repeat_table, workers):
    config = population_config(path)
    engine = IMRIDynEngine(config)
    m1, mnsc = float(repeat_table.M.iloc[0]), float(repeat_table.M_NSC.iloc[0])
    disk = engine.get_disk_profiles(m1)
    agn = AGNProperties.from_mass(m1, config.duty_cycle_myr)
    calculator = DampingCalculator(agn, verbose=False,
        radius_grid_cm=disk["radius_cm"], surface_density_g_cm2=disk["surface_density_g_cm2"],
        aspect_ratio=disk["aspect_ratio"], disk_inner_radius_cm=disk["inner_radius_cm"],
        disk_outer_radius_cm=disk["outer_radius_cm"], use_disk_inner_radius=config.use_disk_inner_radius,
        density_g_cm3=disk["density_g_cm3"])
    stored = pd.read_csv(path / "imri_source_transitions.csv")
    stored = stored[(stored.stage == "damping") & (stored.task_kind == "primary")]
    replay_rows, geometry_rows, stock_rows = [], [], []
    for _, repeat in repeat_table.iterrows():
        seed = int(repeat.task_seed)
        state = np.random.get_state()
        try:
            np.random.seed(seed)
            population = sample_disk_crossing_population(m1, mnsc, disk, config)
        finally:
            np.random.set_state(state)
        if population.total_count != repeat.sampled or population.disk_crossing_count != repeat.initial_crossers:
            raise ValueError("Seeded population reconstruction does not match saved simulation")
        initial = population.initial
        stock_rows.append(dict(repeat=int(repeat["repeat"]), objects=population.total_count,
            sampled_mass_msun=float(initial["mass_msun"].sum()),
            mean_mass_msun=float(initial["mass_msun"].mean()),
            min_mass_msun=float(initial["mass_msun"].min()), max_mass_msun=float(initial["mass_msun"].max()),
            collapse_within_agn=int((initial["formation_time_myr"] <= config.duty_cycle_myr).sum()),
            initially_inside_outer_edge=int((initial["semi_major_axis_cm"] <= disk["outer_radius_cm"]).sum()),
            nsc_radius_pc=initial["nsc_radius_cm"] / const.parsec,
            disk_outer_radius_pc=disk["outer_radius_cm"] / const.parsec))
        for factor in (.5, 1, 2, 10):
            crossing = disk_crossing_mask(initial["semi_major_axis_cm"], initial["eccentricity"], initial["cos_omega"], disk["inner_radius_cm"], disk["outer_radius_cm"] * factor, config.use_disk_inner_radius)
            geometry_rows.append(dict(repeat=int(repeat["repeat"]), outer_edge_factor=factor,
                crossing_objects=int(crossing.sum()), crossing_fraction=float(crossing.mean())))
        tasks = build_damping_tasks(population.disk_crossing, population.selected_indices, seed)
        results = run_damping_tasks_parallel(tasks, agn, disk["radius_cm"], disk["surface_density_g_cm2"], disk["aspect_ratio"], disk["inner_radius_cm"], disk["outer_radius_cm"], config.use_disk_inner_radius, workers, disk["density_g_cm3"], config)
        saved = stored[stored["repeat"] == repeat["repeat"]].set_index("source_id")
        for result in results:
            original = saved.loc[result["source_id"]]
            if result["status"].value != original.status or not np.isclose(result["final_time_myr"], original.time_myr, rtol=1e-7, atol=1e-9):
                raise ValueError("Replayed primary damping does not match saved result")
            orbital = OrbitalState(
                semi_major_axis_cm=result["final_semi_major_axis_cm"], eccentricity=result["final_eccentricity"],
                inclination_rad=result["final_inclination_rad"], cos_omega=result["final_cos_omega"],
                time_s=result["final_time_s"])
            criteria = calculator.embedding_criteria(orbital)
            replay_rows.append(dict(repeat=int(repeat["repeat"]), source_id=result["source_id"],
                status=result["status"].value, mass_msun=result["mass_msun"],
                initial_a_pc=result["initial_semi_major_axis_cm"] / const.parsec,
                final_a_pc=result["final_semi_major_axis_cm"] / const.parsec,
                final_e=result["final_eccentricity"], final_i_rad=result["final_inclination_rad"],
                low_inclination=criteria.low_inclination, sufficiently_circular=criteria.sufficiently_circular,
                radially_contained=criteria.radially_contained, all_embedding_predicates=criteria.eligible,
                final_time_myr=result["final_time_myr"], collapse_time_myr=result["formation_time_myr"],
                before_agn_end=result["final_time_myr"] <= config.duty_cycle_myr))
        print(f"Reproduced primary population and {len(results)} damping outcomes, repeat={int(repeat['repeat'])}", flush=True)
    return pd.DataFrame(replay_rows), pd.DataFrame(stock_rows), pd.DataFrame(geometry_rows)


def stellar_timing(events, damping):
    keys = ["repeat", "source_id", "task_kind", "copy_number"]
    stars = events[events.event_rate_reason == "bh_not_formed_by_band_entry"].copy()
    joined = stars.merge(damping[keys + ["time_myr"]], on=keys, validate="one_to_one")
    joined["migration_to_entry_myr"] = joined.band_entry_time_myr - joined.time_myr
    joined["collapse_after_entry_myr"] = joined.formation_time_myr - joined.band_entry_time_myr
    beta = np.array([peters_beta_cm4_s(m1, m2) for m1, m2 in zip(joined.m_smbh_msun, joined.mass_msun)])
    isco = 6 * const.G * (joined.m_smbh_msun + joined.mass_msun) * const.msun / const.c**2
    joined["point_mass_vacuum_isco_delay_myr"] = np.maximum(0, joined.band_entry_radius_cm**4 - isco**4) / (4 * beta * const.Myr)
    # Diagnostic only: finite stars are not physically valid compact objects at ISCO.
    radius_star = const.Rsun * joined.mass_msun**STELLAR_MASS_RADIUS_EXPONENT
    joined["newtonian_tidal_radius_cm"] = radius_star * (joined.m_smbh_msun / joined.mass_msun)**(1/3)
    joined["tidal_radius_over_band_radius"] = joined.newtonian_tidal_radius_cm / joined.band_entry_radius_cm
    return joined


def critical_secondary_masses():
    rows = []
    for z in (8, 10, 15):
        def snr(m2):
            return evaluate_lisa_inspiral_snr(m1_source_msun=1e5, m2_source_msun=m2,
                redshift=z, current_observed_frequency_hz=.001, mission_duration_years=4,
                detector_min_frequency_hz=.001, detector_max_frequency_hz=.1, snr_threshold=20).lisa_snr
        rows.append(dict(z=z, snr_at_population_cap=snr(197), secondary_mass_for_snr20=brentq(lambda m: snr(m) - 20, 197, 1e4)))
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/high_redshift_rates"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/population_budget_audit"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    points = pd.read_csv(args.input_dir / "host_rates.csv")
    tables = []
    for _, point in points.iterrows():
        census, damping, events = census_run(Path(point.run_dir))
        tables.append(census)
        if point.z == 10 and point.M == 1e5 and point.M_NSC == 3e6:
            reference = census
            reference_path = Path(point.run_dir)
            timing = stellar_timing(events, damping)
    all_repeats = pd.concat(tables, ignore_index=True)
    all_repeats.to_csv(args.output_dir / "population_census.csv", index=False)
    replay, stock, geometry = replay_primary_damping(reference_path, reference, args.workers)
    replay.to_csv(args.output_dir / "primary_damping_replay.csv", index=False)
    stock.to_csv(args.output_dir / "reconstructed_reference_stock.csv", index=False)
    geometry.to_csv(args.output_dir / "geometry_only_sensitivity.csv", index=False)
    timing.to_csv(args.output_dir / "embedded_stellar_timing.csv", index=False)
    controls = critical_secondary_masses()
    controls.to_csv(args.output_dir / "secondary_mass_snr_controls.csv", index=False)
    mean = reference.mean(numeric_only=True)
    early = replay[(replay.status == "damping_incomplete") & replay.before_agn_end]
    checks = dict(reference_primary_outcomes=len(replay), all_primary_outcomes_reproduced=True,
        initial_mean_objects=float(stock.objects.mean()), mean_initial_mass_msun=float(stock.mean_mass_msun.mean()),
        nsc_radius_pc=float(stock.nsc_radius_pc.iloc[0]), disk_outer_radius_pc=float(stock.disk_outer_radius_pc.iloc[0]),
        initial_crossing_fraction=float(mean.crossing_fraction), primary_success_fraction_given_crossing=float(mean.success_given_crossing),
        early_incomplete_cases=len(early), early_incomplete_low_inclination=int(early.low_inclination.sum()),
        early_incomplete_not_circular=int((~early.sufficiently_circular).sum()),
        early_incomplete_not_contained=int((~early.radially_contained).sum()),
        embedded_stars_missing_bh_at_entry=len(timing),
        stars_could_collapse_before_point_mass_vacuum_isco=int((timing.collapse_after_entry_myr <= timing.point_mass_vacuum_isco_delay_myr).sum()),
        stars_tidal_radius_outside_band=int((timing.tidal_radius_over_band_radius > 1).sum()),
        no_migration_formed_bh_in_reference=bool((reference.migration_formed_bh == 0).all()))
    (args.output_dir / "checks.json").write_text(json.dumps(checks, indent=2))
    lines = ["# Population supply and low embedding-rate audit", "",
        "## Count budget: reference z=10, M_bullet=1e5, M_NSC=3e6 Msun", "",
        "All counts below are means per 10-Myr realization, not instantaneous rates.", "",
        "| Stage | Mean objects |", "|---|---:|"]
    for name, column in [("Sampled initial objects", "sampled"), ("Initial disk-crossing objects", "initial_crossers"),
        ("Primary fully embedded objects", "primary_embedded"), ("Primary embedded stars", "primary_embedded_star"),
        ("Primary embedded BHs", "primary_embedded_bh"), ("Primary damping incomplete", "primary_incomplete"),
        ("Primary incomplete before AGN deadline", "primary_incomplete_before_deadline"),
        ("Replacement tasks", "replacements"), ("Replacement successes", "replacement_embedded"),
        ("All successful embeddings", "total_embedded"), ("All embedded stars", "embedded_star"),
        ("All embedded BHs", "embedded_bh"), ("BHs formed during migration", "migration_formed_bh"),
        ("Physical BH IMRI entries", "imri_entries"), ("Detectable entries", "detected")]:
        lines.append(f"| {name} | {mean[column]:.3f} |")
    lines.extend(["", "## Where supply is restricted", "",
        f"1. **Population model, not millions of low-mass stars plus old BHs.** The selected Z=1e-5 Zsun invokes the top-heavy 13.6–197 Msun IMF. Mean sampled mass is {stock.mean_mass_msun.mean():.3f} Msun, giving only {stock.objects.mean():.1f} objects for 3e6 Msun. There are no stars below 13.6 Msun and no independently initialized old-BH component in these runs.",
        "2. **Coeval young initialization.** Analytic stellar lifetimes are measured from AGN start; the default worker starts objects as stars and switches drag to the BH branch after the assigned lifetime. No initial age distribution is provided. In the fixed-mass prescription, sampled masses remain unchanged; there is no calibrated remnant-mass mapping or a mass-dependent BH/NS/PISN endpoint filter. This is not equivalent to a realistic mixture in which only selected progenitors become BHs.",
        f"3. **Small physical disk.** R_out={stock.disk_outer_radius_pc.iloc[0]:.6g} pc versus R_NSC={stock.nsc_radius_pc.iloc[0]:.6g} pc. Only {100 * mean.crossing_fraction:.3f}% of initial objects have a node inside the annulus. Almost all NSC objects are not initial damping tasks. Angular distributions and the node selection are the existing historical prescription.",
        f"4. **Complete capture is stricter than crossing or inclination damping.** Primary success among crossers is {100 * mean.success_given_crossing:.2f}%. Full handoff requires low inclination, e<=0.01 and radial containment before the 10-Myr AGN end. Counting repeated disk-plane passages would be a different rate, not a rate of newly captured objects.",
        f"5. **The plotted rate is a lifetime average.** {mean.total_embedded:.3f} completed embeddings / 10 Myr = {mean.damping_rate:.3f} Myr^-1, including {mean.embedded_star:.3f} stars as well as {mean.embedded_bh:.3f} BHs. Stars were not omitted from the damping curve. The initial geometric stock divided by duration is {mean.geometric_selection_averaged_over_duration:.3f} Myr^-1, but is not a steady-state influx prediction.",
        "6. **Limited replenishment.** Only in-model BH IMRI candidates reaching ISCO trigger unique donor replacement on a channel orbit. Relaxation times are diagnostic, not a continuous angular-momentum-diffusion inflow; there is no external stellar/BH supply. Stellar noncandidate entries do not trigger replacement. Supply in the broader scenario can therefore differ substantially from this initialization and refill prescription.", "",
        "## Replayed damping outcomes: a numerical/model handoff caveat", "",
        f"The exact seeds reproduce all {len(replay)} primary outcomes across the three reference repeats. Of the incomplete outcomes, {len(early)} stop before 10 Myr; {int(early.low_inclination.sum())} are already inclination-damped, {int((~early.sufficiently_circular).sum())} fail circularity, and {int((~early.radially_contained).sum())} fail radial containment (categories can overlap).",
        "The evolution loop currently stops when inclination alone is damped; the final full embedding predicate can then reject a still-eccentric or radially uncontained object without continuing its evolution through the remaining AGN time. Thus not every 'damping_incomplete' is evidence that all gas damping is physically too slow. This is a handoff/evolution limitation requiring separate validation, not a basis to relabel failed objects as captures.",
        "The timestep also can overshoot the AGN deadline; these late states remain rejected. This audit reproduces existing outcomes and does not claim a corrected capture rate or silently alter integration.", "",
        "## Why embedded stars do not become BH candidates here", "",
        f"All {len(timing)} reference embedded-star events classified no-IMRI reach the selected band before their assigned collapse time; the migration-formed-BH branch is not disabled, but timing does not select it in these runs. Median time from embedding to band entry is {timing.migration_to_entry_myr.median():.6g} Myr; median remaining stellar lifetime at entry is {timing.collapse_after_entry_myr.median():.6g} Myr.",
        f"As a point-mass diagnostic only, their vacuum source time from band entry to ISCO ranges from {timing.point_mass_vacuum_isco_delay_myr.min()*1e6:.4g} to {timing.point_mass_vacuum_isco_delay_myr.max()*1e6:.4g} years; none would collapse before that extrapolated endpoint. Inward gas would not make this residence longer.",
        f"**Missing stellar-endpoint physics:** using the same R_star proportional to M^0.8 as the drag model, the Newtonian tidal radius exceeds the nominal band-entry radius for {checks['stars_tidal_radius_outside_band']}/{len(timing)} such stellar events. A finite star cannot simply be interpreted as a compact point-mass inspiral through that region. Tidal disruption, stellar survival/consumption, and associated refilling need explicit modeling. Do not turn these hypothetical stellar entries/ISCO times into BH IMRIs or an additive corrected replenishment rate.", "",
        "## Candidate counts versus SNR", "",
        "Fewer events reduce the number of opportunities to observe a loud system; they do not reduce a given binary's SNR. SNRs do not add across unrelated IMRI candidates when selecting individual detections. More candidates can sample a louder tail only if the allowed mass/orientation distribution contains that tail.",
        "In detector-only fixed-z vacuum controls for M_bullet=1e5 Msun (same four-year, orientation-averaged model and 1-mHz cutoff):", "",
        "| z | SNR at 197-Msun secondary | Secondary mass for SNR=20 [Msun] |", "|---:|---:|---:|"])
    for _, row in controls.iterrows():
        lines.append(f"| {row.z:g} | {row.snr_at_population_cap:.3f} | {row.secondary_mass_for_snr20:.1f} |")
    lines.extend(["", "These controls are not additional simulated populations or rates. They show why increasing the count while retaining the same capped mass distribution does not automatically restore detections. The actual entry redshift evolves slightly, so the table is a fixed-z diagnostic, not an exact bound on every event.", "",
        "## Recommended next steps", "",
        "First decide the physical population: coeval young Pop III cluster, or an NSC with a birth-age distribution, low-mass stars and an existing BH reservoir. Validate the IMF mass budget and remnant mapping rather than multiplying all object counts into BH candidates. Then validate post-alignment eccentricity/radial damping and stellar disruption/consumption/refill handling. Only after those choices should full rates and SNR selection be rerun. Do not rescale the current rates or modify the detection threshold to force detections.", "",
        "geometry_only_sensitivity.csv varies only the selection annulus on the same sampled orbits. These are supply counts, not re-solved disk models or predicted capture rates; large extensions can violate the Q=1 boundary.",
    ])
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
