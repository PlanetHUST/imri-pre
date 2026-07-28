"""Top-level grid workflow for TDE-rate calculations."""

import time
from pathlib import Path

import numpy as np
import pandas as pd

import physical_process.general_class as agn
import physical_process.helper_function as helpers

from .config import (
    AVERAGED_RESULTS_FILENAME,
    IMRI_AVERAGED_RESULTS_FILENAME,
    IMRI_EVENTS_FILENAME,
    SECONDS_PER_MINUTE,
    TIMESCALES_FILENAME,
    VISUALIZATION_DIRECTORY,
    TDEDynConfig,
)
from .engine import TDEDynEngine
from .models import (
    DampingResult,
    DiskProfiles,
    TDEDynAveragedResult,
    TDEDynSingleResult,
)
from .outputs import (
    create_timescale_histograms_from_csv,
    create_orbital_param_pdfs_from_csv,
    create_visualizations_from_csv,
    print_timescale_statistics,
    save_intermediate_population_plots,
    save_metadata,
    save_orbital_params,
    save_results,
    save_timescales,
)

IMRI_EVENT_COLUMNS = (
    "m_smbh_msun",
    "m_nsc_msun",
    "repeat",
    "task_kind",
    "object_index",
    "mass_msun",
    "formation_time_myr",
    "damping_time_myr",
    "bh_embedding_time_myr",
    "start_radius_cm",
    "start_radius_rg",
    "lisa_entry_radius_cm",
    "lisa_entry_radius_rg",
    "inner_radius_cm",
    "inner_radius_rg",
    "start_gw_frequency_hz",
    "inner_gw_frequency_hz",
    "gw_time_to_lisa_myr",
    "migration_time_myr",
    "arrival_time_myr",
    "bh_candidate",
    "counted_as_imri",
    "outcome",
)
IMRI_AVERAGED_COLUMNS = (
    "m_smbh_msun",
    "m_nsc_msun",
    "num_repeats",
    "embedded_bh_count_mean",
    "bh_candidate_count_mean",
    "imri_count_mean",
    "imri_count_min",
    "imri_count_max",
    "imri_mass_mean_msun",
    "imri_rate_mean_per_myr",
    "imri_mass_rate_mean_msun_per_myr",
)


def print_header(text: str, width: int = 72, char: str = "=") -> None:
    """Print a centered workflow heading."""
    print(f"\n{char * width}")
    print(f"{text:^{width}}")
    print(f"{char * width}")


def debug_print_result(result: TDEDynSingleResult) -> None:
    """Print detailed counters for one realization."""
    print("\n========== DEBUG INFO ==========")
    print("\n--- Embedded ---")
    print(f"Stars: {result.embedded_stars_primary}")
    print(f"BH: {result.embedded_bh_primary}")
    print(f"Outside: {result.outside_primary}")
    print("\n--- TDE Counts ---")
    print(f"Primary Direct: {result.direct_tde_primary_count}")
    print(f"Primary Migration: {result.migration_tde_primary_count}")
    print(f"Copy Direct: {result.direct_tde_copy_count}")
    print(f"Copy Migration: {result.migration_tde_copy_count}")
    print("\n--- Mass --- first 5 TDEs ---")
    print(f"Primary masses: {result.tde_masses_primary_msun[:5]}")
    print(f"Copy masses: {result.tde_masses_copies_msun[:5]}")
    print("\n--- Times (Myr) first 5 TDEs ---")
    print(f"Primary times: {result.tde_times_primary_myr[:5]}")
    print(f"Copy times: {result.tde_times_copies_myr[:5]}")
    print("================================\n")


def run_imri_demo(config: TDEDynConfig | None = None) -> pd.DataFrame:
    """Run the simplified primary-BH IMRI demo workflow."""
    if config is None:
        config = TDEDynConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print_header("IMRI DEMO WORKFLOW", 72, "#")
    smbh_masses_msun, nsc_masses_msun = config.get_param_grid()
    grid_size = len(smbh_masses_msun) * len(nsc_masses_msun)
    print(f"Grid size: {len(smbh_masses_msun)} x {len(nsc_masses_msun)} = {grid_size}")
    print(f"Repeats per point: {config.num_repeats}")
    print(f"Duty cycle: {config.duty_cycle_myr:.3e} Myr")
    print(
        "LISA GW band: "
        f"{config.imri_lisa_min_gw_frequency_hz:.3e} to "
        f"{config.imri_lisa_max_gw_frequency_hz:.3e} Hz"
    )
    print(f"Maximum GW time: {config.imri_max_gw_time_myr:.3e} Myr")
    print(f"Type-II migration: {config.consider_type_ii_migration}")
    print(f"Output directory: {config.output_dir}")

    engine = TDEDynEngine(config)
    event_records: list[dict[str, object]] = []
    averaged_records: list[dict[str, float | int]] = []
    simulation_count = 0
    start_time_s = time.time()

    for smbh_index, smbh_mass_msun in enumerate(smbh_masses_msun):
        for nsc_index, nsc_mass_msun in enumerate(nsc_masses_msun):
            grid_index = smbh_index * len(nsc_masses_msun) + nsc_index + 1
            print_header(
                f"Grid Point {grid_index}/{grid_size}: "
                f"M_SMBH={smbh_mass_msun:.2e}, "
                f"M_NSC={nsc_mass_msun:.2e}",
                72,
                "-",
            )
            repeat_summaries: list[dict[str, float]] = []
            repeat_results: list[TDEDynSingleResult] = []
            for repeat_index in range(config.num_repeats):
                print(f"Repeat {repeat_index + 1}/{config.num_repeats}")
                (
                    result,
                    agn_properties,
                    disk_data,
                    _geo_pool,
                    _relaxation_time_myr_geo,
                    primary_results,
                    _cpu_count,
                ) = engine.run_primary_damping_realization(
                    float(smbh_mass_msun),
                    float(nsc_mass_msun),
                    repeat_index,
                )
                events = _classify_imri_demo_events(
                    config,
                    result,
                    agn_properties,
                    disk_data,
                    primary_results,
                )
                event_records.extend(events)
                summary = _summarize_imri_repeat(events)
                repeat_summaries.append(summary)
                repeat_results.append(result)
                simulation_count += 1
                _print_imri_repeat_summary(summary)

            if config.create_intermediate_plots:
                save_intermediate_population_plots(
                    repeat_results,
                    config.output_dir,
                    m_smbh_msun=float(smbh_mass_msun),
                    m_nsc_msun=float(nsc_mass_msun),
                )
            averaged_records.append(
                _average_imri_repeats(
                    float(smbh_mass_msun),
                    float(nsc_mass_msun),
                    repeat_summaries,
                    config.duty_cycle_myr,
                )
            )
            if (
                grid_index % config.checkpoint_grid_interval == 0
                or grid_index == grid_size
            ):
                _save_imri_demo_outputs(
                    event_records,
                    averaged_records,
                    config.output_dir,
                )

    runtime_s = time.time() - start_time_s
    print_header("IMRI DEMO COMPLETE", 72, "#")
    print(f"Total simulations: {simulation_count}")
    print(
        f"Runtime: {runtime_s:.2f} s "
        f"({runtime_s/SECONDS_PER_MINUTE:.2f} min)"
    )
    frame = _save_imri_demo_outputs(
        event_records,
        averaged_records,
        config.output_dir,
    )
    save_metadata(
        config.output_dir,
        config,
        {
            "workflow": "imri_demo",
            "total_simulations": simulation_count,
            "total_runtime_seconds": runtime_s,
            "total_runtime_minutes": runtime_s / SECONDS_PER_MINUTE,
            "average_time_per_simulation_seconds": (
                runtime_s / simulation_count
                if simulation_count > 0
                else np.nan
            ),
        },
    )
    return frame


def _classify_imri_demo_events(
    config: TDEDynConfig,
    result: TDEDynSingleResult,
    agn_properties: agn.AGN_properties,
    disk_data: DiskProfiles,
    primary_results: list[DampingResult],
) -> list[dict[str, object]]:
    """Classify primary embedded BHs as simplified IMRI candidates."""
    records: list[dict[str, object]] = []
    rg_cm = disk_data["gravitational_radius_cm"]
    for damping_result in primary_results:
        if damping_result["obj_type"] != 1:
            continue
        start_radius_cm = damping_result["final_semi_major_axis_cm"]
        bh_candidate = bool(
            np.isfinite(damping_result["formation_time_myr"])
            and np.isfinite(damping_result["damping_time_myr"])
            and damping_result["formation_time_myr"]
            < damping_result["damping_time_myr"]
        )
        (
            lisa_entry_radius_cm,
            inner_radius_cm,
            gw_time_to_lisa_myr,
            migration_time_myr,
            arrival_time_myr,
            counted_as_imri,
            outcome,
        ) = _evaluate_imri_delivery(
            config,
            result,
            agn_properties,
            disk_data,
            damping_result,
            start_radius_cm,
            bh_candidate,
        )
        total_mass_msun = result.m_smbh_msun + damping_result["mass_msun"]
        records.append(
            {
                "m_smbh_msun": result.m_smbh_msun,
                "m_nsc_msun": result.m_nsc_msun,
                "repeat": result.repeat_idx,
                "task_kind": damping_result["task_kind"],
                "object_index": damping_result["idx"],
                "mass_msun": damping_result["mass_msun"],
                "formation_time_myr": damping_result["formation_time_myr"],
                "damping_time_myr": damping_result["damping_time_myr"],
                "bh_embedding_time_myr": damping_result["final_time_myr"],
                "start_radius_cm": start_radius_cm,
                "start_radius_rg": start_radius_cm / rg_cm,
                "lisa_entry_radius_cm": lisa_entry_radius_cm,
                "lisa_entry_radius_rg": lisa_entry_radius_cm / rg_cm,
                "inner_radius_cm": inner_radius_cm,
                "inner_radius_rg": inner_radius_cm / rg_cm,
                "start_gw_frequency_hz": helpers.gw_frequency_hz(
                    total_mass_msun,
                    start_radius_cm,
                ),
                "inner_gw_frequency_hz": (
                    helpers.gw_frequency_hz(total_mass_msun, inner_radius_cm)
                    if np.isfinite(inner_radius_cm)
                    else np.nan
                ),
                "gw_time_to_lisa_myr": gw_time_to_lisa_myr,
                "migration_time_myr": migration_time_myr,
                "arrival_time_myr": arrival_time_myr,
                "bh_candidate": bh_candidate,
                "counted_as_imri": counted_as_imri,
                "outcome": outcome,
            }
        )
    return records


def _evaluate_imri_delivery(
    config: TDEDynConfig,
    result: TDEDynSingleResult,
    agn_properties: agn.AGN_properties,
    disk_data: DiskProfiles,
    damping_result: DampingResult,
    start_radius_cm: float,
    bh_candidate: bool,
) -> tuple[float, float, float, float, float, bool, str]:
    compact_mass_msun = damping_result["mass_msun"]
    lisa_entry_radius_cm = helpers.lisa_entry_radius_cm(
        result.m_smbh_msun,
        compact_mass_msun,
        config.imri_lisa_min_gw_frequency_hz,
    )
    inner_radius_cm = helpers.imri_lisa_hubble_threshold_radius_cm(
        result.m_smbh_msun,
        compact_mass_msun,
        config.imri_max_gw_time_myr,
        config.imri_lisa_min_gw_frequency_hz,
        config.imri_gw_eccentricity,
    )
    gw_time_to_lisa_myr = (
        helpers.peters_time_to_lisa_band_myr(
            result.m_smbh_msun,
            compact_mass_msun,
            inner_radius_cm,
            config.imri_lisa_min_gw_frequency_hz,
            config.imri_gw_eccentricity,
        )
        if np.isfinite(inner_radius_cm)
        else np.inf
    )
    migration_time_myr = _semi_analytic_migration_time_to_radius_myr(
        config,
        agn_properties,
        disk_data,
        damping_result["mass_g"],
        start_radius_cm,
        inner_radius_cm,
    )
    arrival_time_myr = damping_result["final_time_myr"] + migration_time_myr
    reaches_lisa_in_time = bool(
        np.isfinite(gw_time_to_lisa_myr)
        and gw_time_to_lisa_myr
        <= config.imri_max_gw_time_myr * (1.0 + 1.0e-10)
    )
    reaches_inner_before_duty = bool(
        np.isfinite(arrival_time_myr)
        and arrival_time_myr <= config.duty_cycle_myr
    )
    counted_as_imri = bool(
        bh_candidate
        and reaches_lisa_in_time
        and reaches_inner_before_duty
    )
    if not bh_candidate:
        outcome = "not_bh_before_damping"
    elif not reaches_lisa_in_time:
        outcome = "no_lisa_entry_within_gw_time"
    elif not np.isfinite(migration_time_myr):
        outcome = "migration_unavailable"
    elif not reaches_inner_before_duty:
        outcome = "misses_duty_cycle"
    else:
        outcome = "imri"
    return (
        lisa_entry_radius_cm,
        inner_radius_cm,
        gw_time_to_lisa_myr,
        migration_time_myr,
        arrival_time_myr,
        counted_as_imri,
        outcome,
    )


def _semi_analytic_migration_time_to_radius_myr(
    config: TDEDynConfig,
    agn_properties: agn.AGN_properties,
    disk_data: DiskProfiles,
    mass_g: float,
    start_radius_cm: float,
    target_radius_cm: float,
) -> float:
    """Approximate migration time [Myr] using the local semi-analytic timescale."""
    if not np.isfinite(target_radius_cm) or start_radius_cm <= 0.0:
        return np.inf
    if start_radius_cm <= target_radius_cm:
        return 0.0
    local_time_myr = helpers.t_migration(
        agn_properties,
        mass_g,
        start_radius_cm,
        h_list=disk_data["aspect_ratio"],
        sigma_list=disk_data["surface_density_g_cm2"],
        r_list=disk_data["radius_cm"],
        consider_type_ii=config.consider_type_ii_migration,
    )
    if not np.isfinite(local_time_myr) or local_time_myr <= 0.0:
        return np.inf
    return float(local_time_myr * abs(np.log(start_radius_cm / target_radius_cm)))


def _summarize_imri_repeat(events: list[dict[str, object]]) -> dict[str, float]:
    imri_events = [event for event in events if event["counted_as_imri"]]
    candidate_events = [event for event in events if event["bh_candidate"]]
    imri_mass_msun = float(sum(float(event["mass_msun"]) for event in imri_events))
    return {
        "embedded_bh_count": float(len(events)),
        "bh_candidate_count": float(len(candidate_events)),
        "imri_count": float(len(imri_events)),
        "imri_mass_msun": imri_mass_msun,
    }


def _average_imri_repeats(
    m_smbh_msun: float,
    m_nsc_msun: float,
    repeat_summaries: list[dict[str, float]],
    duty_cycle_myr: float,
) -> dict[str, float | int]:
    def values(name: str) -> np.ndarray:
        return np.asarray([summary[name] for summary in repeat_summaries], dtype=float)

    imri_count = values("imri_count")
    imri_mass = values("imri_mass_msun")
    imri_count_mean = float(np.mean(imri_count))
    imri_mass_mean = float(np.mean(imri_mass))
    return {
        "m_smbh_msun": m_smbh_msun,
        "m_nsc_msun": m_nsc_msun,
        "num_repeats": len(repeat_summaries),
        "embedded_bh_count_mean": float(np.mean(values("embedded_bh_count"))),
        "bh_candidate_count_mean": float(np.mean(values("bh_candidate_count"))),
        "imri_count_mean": imri_count_mean,
        "imri_count_min": float(np.min(imri_count)),
        "imri_count_max": float(np.max(imri_count)),
        "imri_mass_mean_msun": imri_mass_mean,
        "imri_rate_mean_per_myr": imri_count_mean / duty_cycle_myr,
        "imri_mass_rate_mean_msun_per_myr": imri_mass_mean / duty_cycle_myr,
    }


def _print_imri_repeat_summary(summary: dict[str, float]) -> None:
    print(f"  embedded BHs         : {summary['embedded_bh_count']:.3e}")
    print(f"  BH candidates        : {summary['bh_candidate_count']:.3e}")
    print(f"  IMRI count           : {summary['imri_count']:.3e}")
    print(f"  IMRI mass            : {summary['imri_mass_msun']:.3e} Msun")


def _save_imri_demo_outputs(
    event_records: list[dict[str, object]],
    averaged_records: list[dict[str, float | int]],
    output_dir: Path,
) -> pd.DataFrame:
    event_frame = pd.DataFrame.from_records(event_records, columns=IMRI_EVENT_COLUMNS)
    event_path = output_dir / IMRI_EVENTS_FILENAME
    event_frame.to_csv(event_path, index=False, float_format="%.6e")
    print(f"saved: {event_path}")

    averaged_frame = pd.DataFrame.from_records(
        averaged_records,
        columns=IMRI_AVERAGED_COLUMNS,
    )
    averaged_path = output_dir / IMRI_AVERAGED_RESULTS_FILENAME
    averaged_frame.to_csv(averaged_path, index=False, float_format="%.6e")
    print(f"saved: {averaged_path}")
    return averaged_frame


def run_simulation(config: TDEDynConfig | None = None) -> pd.DataFrame:
    """Run the configured mass grid and return averaged results."""
    if config is None:
        config = TDEDynConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print_header("TDE DYN SIMULATION", 72, "#")
    smbh_masses_msun, nsc_masses_msun = config.get_param_grid()
    grid_size = len(smbh_masses_msun) * len(nsc_masses_msun)
    print(f"Grid size: {len(smbh_masses_msun)} x {len(nsc_masses_msun)} = {grid_size}")
    print(f"Repeats per point: {config.num_repeats}")
    print(f"Dynamic resupply: {config.dyn_resupply}")
    print(f"Direct TDE flag: {config.direct_tde}")
    print(f"Type-II migration: {config.consider_type_ii_migration}")
    print(f"Output directory: {config.output_dir}")
    print(
        "Damping CPU count: "
        f"{config.damping_cpu_count if config.damping_cpu_count is not None else 'auto'}"
    )

    engine = TDEDynEngine(config)
    averaged_results: list[TDEDynAveragedResult] = []
    orbital_params_written = False
    simulation_count = 0
    start_time_s = time.time()

    for smbh_index, smbh_mass_msun in enumerate(smbh_masses_msun):
        for nsc_index, nsc_mass_msun in enumerate(nsc_masses_msun):
            grid_index = smbh_index * len(nsc_masses_msun) + nsc_index + 1
            print_header(
                f"Grid Point {grid_index}/{grid_size}: "
                f"M_SMBH={smbh_mass_msun:.2e}, "
                f"M_NSC={nsc_mass_msun:.2e}",
                72,
                "-",
            )
            repeat_results: list[TDEDynSingleResult] = []
            for repeat_index in range(config.num_repeats):
                print(
                    f"Repeat {repeat_index + 1}/{config.num_repeats}"
                )
                result = engine.run_single_simulation(
                    float(smbh_mass_msun),
                    float(nsc_mass_msun),
                    repeat_index,
                )
                _print_result_summary(result)
                if config.print_debug_results:
                    debug_print_result(result)
                repeat_results.append(result)
                simulation_count += 1

            averaged_result = TDEDynAveragedResult(
                float(smbh_mass_msun),
                float(nsc_mass_msun),
                repeat_results,
            )
            averaged_results.append(averaged_result)
            if config.create_intermediate_plots:
                save_intermediate_population_plots(
                    repeat_results,
                    config.output_dir,
                    m_smbh_msun=float(smbh_mass_msun),
                    m_nsc_msun=float(nsc_mass_msun),
                )
            save_orbital_params(
                [averaged_result],
                config.output_dir,
                append=orbital_params_written,
            )
            orbital_params_written = True
            for result in repeat_results:
                result.intermediate_initial_population = None
                result.intermediate_disk_crossing_population = None
                result.timescale_records = []
                result.tde_orbital_params = []
            if (
                grid_index % config.checkpoint_grid_interval == 0
                or grid_index == grid_size
            ):
                save_results(averaged_results, config.output_dir)
                save_timescales(averaged_results, config.output_dir)

    runtime_s = time.time() - start_time_s
    print_header("SIMULATION COMPLETE", 72, "#")
    print(f"Total simulations: {simulation_count}")
    print(
        f"Runtime: {runtime_s:.2f} s "
        f"({runtime_s/SECONDS_PER_MINUTE:.2f} min)"
    )
    frame = save_results(averaged_results, config.output_dir)
    save_timescales(averaged_results, config.output_dir)
    orbital_params_csv = config.output_dir / "orbital_params.csv"
    timescales_csv = config.output_dir / TIMESCALES_FILENAME
    print_timescale_statistics(timescales_csv)
    save_metadata(
        config.output_dir,
        config,
        {
            "total_simulations": simulation_count,
            "total_runtime_seconds": runtime_s,
            "total_runtime_minutes": runtime_s
            / SECONDS_PER_MINUTE,
            "average_time_per_simulation_seconds": (
                runtime_s / simulation_count
                if simulation_count > 0
                else np.nan
            ),
        },
    )
    if config.create_visualizations:
        visualization_dir = (
            config.output_dir / VISUALIZATION_DIRECTORY
        )
        visualization_dir.mkdir(parents=True, exist_ok=True)
        create_visualizations_from_csv(
            config.output_dir / AVERAGED_RESULTS_FILENAME,
            visualization_dir,
        )
        create_timescale_histograms_from_csv(
            timescales_csv,
            visualization_dir,
        )
        create_orbital_param_pdfs_from_csv(
            orbital_params_csv,
            visualization_dir,
        )
    return frame


def _print_result_summary(result: TDEDynSingleResult) -> None:
    print(f"  total stars          : {result.total_stars:.3e}")
    print(
        f"  geo-selected stars   : "
        f"{result.geo_selected_stars:.3e}"
    )
    print(f"  f_geo                : {result.f_geo:.3e}")
    print(
        f"  primary TDE mass     : "
        f"{result.tde_mass_primary_msun:.3e} Msun"
    )
    print(
        f"  copy TDE mass        : "
        f"{result.copy_tde_mass_msun:.3e} Msun"
    )
    print(
        f"  total TDE mass       : "
        f"{result.tde_mass_total_msun:.3e} Msun"
    )
    print(
        f"  total TDE mass rate  : "
        f"{result.tde_mass_rate_total_msun_per_myr:.3e} Msun/Myr"
    )
    print(
        f"  copies generated     : "
        f"{result.num_copies_generated:.3e}"
    )
    print(
        f"  total TDE count      : "
        f"{result.tde_star_count_total:.3e}"
    )


# Historical public function name.
main = run_simulation
