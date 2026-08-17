"""Top-level grid workflow for dynamical IMRI-rate calculations."""

import time

import numpy as np
import pandas as pd

import physical_process.gw_frequency_radius as gw_helpers

from .config import IMRIDynConfig, SECONDS_PER_MINUTE
from .engine import IMRIDynEngine
from .models import IMRIDynAveragedResult, IMRIDynSingleResult
from .metadata import build_metadata
from .outputs import (
    save_events,
    save_metadata,
    save_migration_diagnostics,
    save_results,
)
from .reporting import debug_print_result, print_header, print_result_summary
from .single_work import IMRISingleWorkflow


def run_simulation(config: IMRIDynConfig) -> pd.DataFrame:
    """Run the configured IMRI mass grid and return averaged results."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print_header("IMRI DYN SIMULATION", 72, "#")
    smbh_masses_msun, nsc_masses_msun = config.get_param_grid()
    grid_size = len(smbh_masses_msun) * len(nsc_masses_msun)
    print(f"Grid size: {len(smbh_masses_msun)} x {len(nsc_masses_msun)} = {grid_size}")
    print(f"Repeats per point: {config.num_repeats}")
    print(f"Duty cycle: {config.duty_cycle_myr:.3e} Myr")
    print(f"Metallicity: {config.metallicity:.3e} Zsun")
    print(f"Type-II migration: {config.consider_type_ii_migration}")
    print(f"gap_opening_condition: {config.gap_opening_condition}")
    print(f"Redshift: {config.redshift:.3e}")

    source_lisa_min_hz = gw_helpers.source_frame_gw_frequency_hz(
        config.lisa_min_gw_frequency_hz,
        config.redshift,
    )
    source_lisa_max_hz = gw_helpers.source_frame_gw_frequency_hz(
        config.lisa_max_gw_frequency_hz,
        config.redshift,
    )

    print(
        "Observed LISA GW band: "
        f"{config.lisa_min_gw_frequency_hz:.3e} to "
        f"{config.lisa_max_gw_frequency_hz:.3e} Hz"
    )
    print(
        "Source-frame LISA GW band: "
        f"{source_lisa_min_hz:.3e} to "
        f"{source_lisa_max_hz:.3e} Hz"
    )
    print("Cosmic merge time: " f"{config.cosmic_merge_time_myr:.3e} Myr")
    print(f"Output directory: {config.output_dir}")
    print(
        "Damping CPU count: "
        f"{config.damping_cpu_count if config.damping_cpu_count is not None else 'auto'}"
    )

    single_work = IMRISingleWorkflow(IMRIDynEngine(config))
    averaged_results: list[IMRIDynAveragedResult] = []
    simulation_count = 0
    start_time_s = time.time()

    for smbh_index, smbh_mass_msun in enumerate(smbh_masses_msun):
        for nsc_index, nsc_mass_msun in enumerate(nsc_masses_msun):
            grid_index = smbh_index * len(nsc_masses_msun) + nsc_index + 1
            print_header(
                f"Grid Point {grid_index}/{grid_size}: "
                f"smbh_mass_msun={smbh_mass_msun:.2e}, "
                f"nsc_mass_msun={nsc_mass_msun:.2e}",
                72,
                "-",
            )
            repeat_results: list[IMRIDynSingleResult] = []
            for repeat_index in range(config.num_repeats):
                print(f"Repeat {repeat_index + 1}/{config.num_repeats}")
                result = single_work.run_single_simulation(
                    float(smbh_mass_msun),
                    float(nsc_mass_msun),
                    repeat_index,
                )
                print_result_summary(result)
                save_migration_diagnostics(result, config.output_dir, config)

                if config.print_debug_results:
                    debug_print_result(result)
                repeat_results.append(result)
                simulation_count += 1

            averaged_result = IMRIDynAveragedResult(
                float(smbh_mass_msun),
                float(nsc_mass_msun),
                repeat_results,
            )
            averaged_results.append(averaged_result)
            if (
                grid_index % config.checkpoint_grid_interval == 0
                or grid_index == grid_size
            ):
                save_results(averaged_results, config.output_dir)
                save_events(averaged_results, config.output_dir)

    runtime_s = time.time() - start_time_s
    print_header("SIMULATION COMPLETE", 72, "#")
    print(f"Total simulations: {simulation_count}")
    print(f"Runtime: {runtime_s:.2f} s " f"({runtime_s/SECONDS_PER_MINUTE:.2f} min)")
    frame = save_results(averaged_results, config.output_dir)
    save_events(averaged_results, config.output_dir)
    save_metadata(
        config.output_dir,
        build_metadata(
            config,
            {
                "workflow": "imri",
                "total_simulations": simulation_count,
                "total_runtime_seconds": runtime_s,
                "total_runtime_minutes": runtime_s / SECONDS_PER_MINUTE,
                "average_time_per_simulation_seconds": (
                    runtime_s / simulation_count if simulation_count > 0 else np.nan
                ),
            },
        ),
    )
    return frame
