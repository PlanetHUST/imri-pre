"""Run the comparison-only integrated semi-analytic migration scenario.

This script is **not** a production entry point. It assembles
``IMRIDynConfig(disk_model="semi_without_trap")`` programmatically so the
archived local Type-I/Type-II integrated-timescale model can be compared with
the default signed-torque scenario on identical Monte Carlo seeds.

The three scenarios mirror the torque comparison runs:

===============  ========  ==============
scenario         redshift  duty cycle
===============  ========  ==============
z15_duty10       15        10 Myr
z10_duty10       10        10 Myr
z15_duty100      15        100 Myr
===============  ========  ==============
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from workflows.config import (  # noqa: E402
    IMRI_COSMIC_MERGE_RADIUS_FACTOR,
    IMRI_COSMIC_MERGE_SMBH_MASS_FACTOR,
    IMRI_COSMIC_MERGE_TIME_MYR,
    IMRI_DETECTOR_HUBBLE_CONSTANT_KM_S_MPC,
    IMRI_DETECTOR_OMEGA_M,
    IMRI_GW_ECCENTRICITY,
    IMRI_LISA_MAX_GW_FREQUENCY_HZ,
    IMRI_LISA_MIN_GW_FREQUENCY_HZ,
    IMRI_LISA_MISSION_DURATION_YEARS,
    IMRI_MAX_GW_TIME_MYR,
    IMRI_SNR_THRESHOLD,
    IMRI_SNR_WINDOW_PLACEMENT,
    SEMI_WITHOUT_TRAP,
    IMRIDynConfig,
    migration_contract_name,
)
from workflows.workflow import run_simulation  # noqa: E402

DEFAULT_REPEATS = 20
DEFAULT_CPU_COUNT = 8
DEFAULT_BASE_SEED = 12345
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output"
SMBH_MASS_MSUN = 1.0e5
NSC_MASS_MSUN = 2.0e5
METALLICITY_ZSUN = 1.0e-5
INITIAL_STELLAR_DENSITY_GAMMA = 7.0 / 4.0

SCENARIOS = {
    "z15_duty10": (15.0, 10.0),
    "z10_duty10": (10.0, 10.0),
    "z15_duty100": (15.0, 100.0),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scenarios",
        nargs="*",
        choices=tuple(SCENARIOS),
        default=None,
        help="scenario names to run (default: all three)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="Monte Carlo repeats per scenario",
    )
    parser.add_argument(
        "--cpu-count",
        type=int,
        default=DEFAULT_CPU_COUNT,
        help="damping worker count per repeat",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=DEFAULT_BASE_SEED,
        help="root seed; identical seeds pair this run with the torque run",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="directory under which the scenario output directories are created",
    )
    parser.add_argument(
        "--gap-opening-condition",
        choices=("kii", "crida"),
        default="kii",
        help="local Type-II switch used by the comparison scenario",
    )
    parser.add_argument(
        "--no-type-ii-migration",
        action="store_true",
        help="use the local Type-I timescale everywhere",
    )
    return parser


def build_config(
    output_dir: Path,
    redshift: float,
    duty_cycle_myr: float,
    repeats: int,
    cpu_count: int,
    base_seed: int,
    gap_opening_condition: str,
    consider_type_ii_migration: bool,
) -> IMRIDynConfig:
    return IMRIDynConfig(
        smbh_masses_msun=[SMBH_MASS_MSUN],
        nsc_masses_msun=[NSC_MASS_MSUN],
        num_repeats=repeats,
        duty_cycle_myr=duty_cycle_myr,
        output_dir=output_dir,
        metallicity=METALLICITY_ZSUN,
        initial_stellar_density_gamma=INITIAL_STELLAR_DENSITY_GAMMA,
        use_disk_inner_radius=True,
        disk_model=SEMI_WITHOUT_TRAP,
        gap_opening_condition=gap_opening_condition,
        consider_type_ii_migration=consider_type_ii_migration,
        lisa_min_gw_frequency_hz=IMRI_LISA_MIN_GW_FREQUENCY_HZ,
        lisa_max_gw_frequency_hz=IMRI_LISA_MAX_GW_FREQUENCY_HZ,
        max_gw_time_myr=IMRI_MAX_GW_TIME_MYR,
        gw_eccentricity=IMRI_GW_ECCENTRICITY,
        redshift=redshift,
        snr_threshold=IMRI_SNR_THRESHOLD,
        lisa_mission_duration_years=IMRI_LISA_MISSION_DURATION_YEARS,
        snr_window_placement=IMRI_SNR_WINDOW_PLACEMENT,
        detector_hubble_constant_km_s_mpc=(
            IMRI_DETECTOR_HUBBLE_CONSTANT_KM_S_MPC
        ),
        detector_omega_m=IMRI_DETECTOR_OMEGA_M,
        cosmic_merge_time_myr=IMRI_COSMIC_MERGE_TIME_MYR,
        cosmic_merge_smbh_mass_factor=IMRI_COSMIC_MERGE_SMBH_MASS_FACTOR,
        cosmic_merge_radius_factor=IMRI_COSMIC_MERGE_RADIUS_FACTOR,
        damping_cpu_count=cpu_count,
        checkpoint_grid_interval=1,
        print_debug_results=False,
        base_seed=base_seed,
    )


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if arguments.cpu_count <= 0:
        raise ValueError("--cpu-count must be positive")

    scenarios = arguments.scenarios or tuple(SCENARIOS)
    consider_type_ii = not arguments.no_type_ii_migration
    print("=" * 78)
    print("COMPARISON-ONLY INTEGRATED SEMI-ANALYTIC MIGRATION")
    print("=" * 78)
    print(f"disk_model: {SEMI_WITHOUT_TRAP}")
    print(f"migration_contract: {migration_contract_name(SEMI_WITHOUT_TRAP)}")
    print(
        "type-I time: M_smbh^2*h^2/(2*m*Sigma*R^2*Omega); "
        "type-II time: 1/(2*alpha*h^2*Omega)"
    )
    print("path time: integral t_local(R) dlnR; always inward; no traps")
    print(
        f"gap-opening condition: {arguments.gap_opening_condition}; "
        f"type-II enabled: {consider_type_ii}"
    )
    print(
        f"SMBH={SMBH_MASS_MSUN:.3e} Msun, NSC={NSC_MASS_MSUN:.3e} Msun, "
        f"metallicity={METALLICITY_ZSUN:.3e} Zsun, "
        f"gamma={INITIAL_STELLAR_DENSITY_GAMMA:g}"
    )
    print(
        f"repeats={arguments.repeats}, base_seed={arguments.base_seed}, "
        f"damping_cpu_count={arguments.cpu_count}"
    )

    total_start_s = time.time()
    for name in scenarios:
        redshift, duty_cycle_myr = SCENARIOS[name]
        output_dir = arguments.output_root / (
            f"single_host_example_semi_{name}_r{arguments.repeats}"
        )
        print("\n" + "-" * 78)
        print(
            f"scenario={name}: redshift={redshift:g}, "
            f"duty_cycle_myr={duty_cycle_myr:g}, output={output_dir}"
        )
        print("-" * 78)
        config = build_config(
            output_dir,
            redshift,
            duty_cycle_myr,
            arguments.repeats,
            arguments.cpu_count,
            arguments.base_seed,
            arguments.gap_opening_condition,
            consider_type_ii,
        )
        scenario_start_s = time.time()
        run_simulation(config)
        print(
            f"scenario {name} runtime: "
            f"{time.time() - scenario_start_s:.2f} s"
        )

    print("\n" + "=" * 78)
    print(f"all scenarios runtime: {time.time() - total_start_s:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
