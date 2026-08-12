"""Top-level grid workflow for dynamical IMRI-rate calculations."""

import time

import numpy as np
import pandas as pd

import physical_process.gw_frequency_radius as gw_helpers

from .config import IMRIDynConfig, SECONDS_PER_MINUTE
from .engine import IMRIDynEngine
from .models import IMRIDynAveragedResult, IMRIDynSingleResult
from .outputs import (
    save_events,
    save_metadata,
    save_migration_diagnostics,
    save_results,
)
from .single_work import IMRISingleWorkflow


def print_header(text: str, width: int = 72, char: str = "=") -> None:
    """Print a centered workflow heading."""
    print(f"\n{char * width}")
    print(f"{text:^{width}}")
    print(f"{char * width}")


def debug_print_result(result: IMRIDynSingleResult) -> None:
    """Print detailed counters for one realization."""
    print("\n========== DEBUG INFO ==========")
    print(f"Sampled objects: {result.total_sampled_objects}")
    print(f"Disk crossing: {result.disk_crossing_objects}")
    print(f"Embedded stars: {result.embedded_star_count}")
    print(f"Embedded BHs: {result.embedded_bh_count}")
    print(f"Migration-formed BHs: {result.migration_formed_bh_count}")
    print(f"BH candidates: {result.bh_candidate_count}")
    print(f"IMRI candidates: {result.imri_count}")
    print(f"Detectable IMRIs: {result.detectable_imri_count}")
    print(f"IMRI candidate mass: {result.imri_mass_msun:.3e} Msun")
    print(f"Detectable IMRI mass: {result.detectable_imri_mass_msun:.3e} Msun")
    print("================================\n")


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
    redshift_transport_time_myr = gw_helpers.redshift_transport_time_myr(
        config.redshift,
        config.hubble_constant_per_myr,
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
    print(
        "Redshift transport time: "
        f"{redshift_transport_time_myr:.3e} Myr"
    )
    print(
        "Cosmic merge time: "
        f"{config.cosmic_merge_time_myr:.3e} Myr"
    )
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
                _print_result_summary(result)
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
    )
    return frame


def _print_result_summary(result: IMRIDynSingleResult) -> None:
    print(f"  total sampled objects : {result.total_sampled_objects:.3e}")
    print(f"  disk-crossing objects : {result.disk_crossing_objects:.3e}")
    print(f"  f_geo                 : {result.f_geo:.3e}")
    print(f"  embedded stars        : {result.embedded_star_count:.3e}")
    print(f"  embedded BHs          : {result.embedded_bh_count:.3e}")
    print(f"  migration-formed BHs  : {result.migration_formed_bh_count:.3e}")
    print(f"  BH candidates         : {result.bh_candidate_count:.3e}")
    print(f"  IMRI candidates       : {result.imri_count:.3e}")
    print(f"  no-IMRI objects       : {result.no_imri_count:.3e}")
    print(f"  detectable IMRIs      : {result.detectable_imri_count:.3e}")
    print(f"  non-detectable IMRIs  : {result.no_detectable_imri_count:.3e}")
    print(
        "  Torque migration      : "
        f"evaluated={result.torque_migration_evaluated_count}, "
        f"finite={result.finite_torque_migration_count}, "
        f"trap_bound={result.trap_bound_migration_count}, "
        f"errors={result.torque_migration_error_count}"
    )
    print(f"  Trap-bound mass       : {result.trap_bound_mass_msun:.6e} Msun")
    print(f"  IMRI candidate rate   : {result.imri_rate_per_myr:.3e} /Myr")
    print(f"  detectable IMRI rate  : {result.detectable_imri_rate_per_myr:.3e} /Myr")
    print(
        f"  candidate mass rate   : "
        f"{result.imri_mass_rate_msun_per_myr:.3e} Msun/Myr"
    )
    print(
        f"  detectable mass rate  : "
        f"{result.detectable_imri_mass_rate_msun_per_myr:.3e} Msun/Myr"
    )


main = run_simulation
