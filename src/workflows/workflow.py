"""Top-level grid workflow for TDE-rate calculations."""

import time

import numpy as np
import pandas as pd

from .config import (
    AVERAGED_RESULTS_FILENAME,
    SECONDS_PER_MINUTE,
    TIMESCALES_FILENAME,
    VISUALIZATION_DIRECTORY,
    TDEDynConfig,
)
from .engine import TDEDynEngine
from .models import TDEDynAveragedResult, TDEDynSingleResult
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
