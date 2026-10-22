"""Run scalar-redshift IMRI workflows for the fixed migration contract."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

# Defaults used by ``python scripts/redshift_scan.py`` with no arguments.
# Masses are in Msun, times are in Myr, and frequencies are in Hz.
DEFAULT_CONFIG_JSON: Path | None = None
DEFAULT_REDSHIFTS = (10, 13, 15)
DEFAULT_BASE_SEED = 12345
DEFAULT_OUTPUT_DIR = Path("output/redshift_scan_results")
DEFAULT_NSC_MASS_MSUN: float | None = None
DEFAULT_CONFIG_TEMPLATE: dict[str, Any] = {
    "smbh_masses_msun": (1.0e3, 5e3, 1.0e4, 5e4, 1e5),
    "nsc_masses_msun": (2.0e5,),
    "num_repeats": 10,
    "duty_cycle_myr": 10.0,
    "metallicity": 1.0e-5,
    "initial_stellar_density_gamma": 7.0 / 4.0,
    "use_disk_inner_radius": True,
    "lisa_min_gw_frequency_hz": 1.0e-3,
    "lisa_max_gw_frequency_hz": 1.0e-1,
    "max_gw_time_myr": 13.6e3,
    "gw_eccentricity": 0.0,
    "snr_threshold": 20.0,
    "lisa_mission_duration_years": 4.0,
    "snr_window_placement": "optimal",
    "detector_hubble_constant_km_s_mpc": 67.4,
    "detector_omega_m": 0.315,
    "cosmic_merge_time_myr": np.inf,
    "cosmic_merge_smbh_mass_factor": 2.0,
    "cosmic_merge_radius_factor": 2.0,
    "damping_cpu_count": None,
    "checkpoint_grid_interval": 5,
    "print_debug_results": False,
    "nsc_radius_pc_at_reference_mass": 3.31,
    "nsc_radius_mass_slope": 0.321,
    "nsc_radius_reference_mass_msun": 3.6e6,
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from workflows.config import IMRIDynConfig, MIGRATION_CONTRACT
from viz.rate_plots import (
    STATUS_VARIABLE_ORDER,
    RunResults,
    build_rate_summary_data,
    build_status_fraction_by_smbh_mass_data,
    build_status_fraction_data,
    generate_scan_figures,
    load_run_results,
    plot_status_fractions,
    plot_status_fractions_vs_smbh_mass,
    save_plot_data,
    shared_rate_axis_limits,
    write_status_fraction_captions,
)
from workflows.workflow import run_simulation

CONFIG_TEMPLATE_FIELDS = (
    "smbh_masses_msun",
    "nsc_masses_msun",
    "num_repeats",
    "duty_cycle_myr",
    "metallicity",
    "initial_stellar_density_gamma",
    "use_disk_inner_radius",
    "lisa_min_gw_frequency_hz",
    "lisa_max_gw_frequency_hz",
    "max_gw_time_myr",
    "gw_eccentricity",
    "snr_threshold",
    "lisa_mission_duration_years",
    "snr_window_placement",
    "detector_hubble_constant_km_s_mpc",
    "detector_omega_m",
    "cosmic_merge_time_myr",
    "cosmic_merge_smbh_mass_factor",
    "cosmic_merge_radius_factor",
    "damping_cpu_count",
    "checkpoint_grid_interval",
    "print_debug_results",
    "nsc_radius_pc_at_reference_mass",
    "nsc_radius_mass_slope",
    "nsc_radius_reference_mass_msun",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run complete scalar-redshift IMRI workflows with paired task seeds."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config-json",
        type=Path,
        default=DEFAULT_CONFIG_JSON,
        help=(
            "optional JSON object with every CONFIG_TEMPLATE_FIELDS value, or an "
            "existing simulation_metadata.json whose parameters contain those "
            "values; defaults to DEFAULT_CONFIG_TEMPLATE"
        ),
    )
    parser.add_argument(
        "--redshifts",
        type=float,
        nargs="+",
        default=DEFAULT_REDSHIFTS,
        help="ordered scalar redshifts; matching grid tasks use paired seeds",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=DEFAULT_BASE_SEED,
        help="nonnegative root seed shared by every redshift run",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="scan directory; replacing a nonempty directory requires confirmation",
    )
    parser.add_argument(
        "--nsc-mass-msun",
        type=float,
        default=DEFAULT_NSC_MASS_MSUN,
        help="NSC mass to present when the configuration contains several",
    )
    return parser


def load_config_template(path: Path | None) -> dict[str, Any]:
    """Load the explicit fixed configuration shared by all redshift runs."""
    if path is None:
        return dict(DEFAULT_CONFIG_TEMPLATE)
    with path.open() as config_file:
        payload = json.load(config_file)
    is_metadata = "parameters" in payload
    source = payload.get("parameters", payload)
    if not isinstance(source, dict):
        raise ValueError("config JSON must contain an object")
    if "hubble_constant_per_myr" in source:
        warnings.warn(
            "ignoring obsolete hubble_constant_per_myr; cosmic time now uses "
            "detector_hubble_constant_km_s_mpc and detector_omega_m",
            stacklevel=2,
        )
    historical_detector_defaults = {
        "lisa_mission_duration_years": 4.0,
        "snr_window_placement": "optimal",
        "detector_hubble_constant_km_s_mpc": 67.4,
        "detector_omega_m": 0.315,
    }
    missing_detector_fields = [
        field for field in historical_detector_defaults if field not in source
    ]
    if missing_detector_fields and is_metadata:
        source = dict(source)
        for field in missing_detector_fields:
            source[field] = historical_detector_defaults[field]
        warnings.warn(
            "historical simulation metadata has no physical detector settings; "
            f"explicitly inferring defaults for {missing_detector_fields}",
            stacklevel=2,
        )
    missing = [field for field in CONFIG_TEMPLATE_FIELDS if field not in source]
    if missing:
        raise ValueError(f"config JSON is missing explicit fields: {missing}")
    template = {field: source[field] for field in CONFIG_TEMPLATE_FIELDS}
    if template["cosmic_merge_time_myr"] is None:
        template["cosmic_merge_time_myr"] = np.inf
    return template


def _redshift_token(index: int, redshift: float) -> str:
    value = f"{redshift:.12g}".replace("-", "m").replace("+", "p").replace(".", "p")
    return f"z{index:03d}_{value}"


def _json_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_paths(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _output_dir_has_entries(output_dir: Path) -> bool:
    if not output_dir.exists():
        return False
    if not output_dir.is_dir():
        raise NotADirectoryError(f"scan output path is not a directory: {output_dir}")
    return next(output_dir.iterdir(), None) is not None


def _validate_replacement_target(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    project_root = PROJECT_ROOT.resolve()
    if output_dir.is_symlink():
        raise ValueError(
            f"refusing to replace a symbolic-link output path: {output_dir}"
        )
    if (
        len(resolved.parts) < 3
        or resolved == project_root
        or resolved in project_root.parents
        or (resolved / ".git").exists()
    ):
        raise ValueError(f"refusing to replace unsafe output directory: {resolved}")


def _confirm_output_replacement(output_dir: Path) -> bool:
    _validate_replacement_target(output_dir)
    prompt = (
        f"Scan output directory is not empty: {output_dir}\n"
        "Permanently delete it and run a new scan? [yes/no]: "
    )
    while True:
        try:
            answer = input(prompt).strip().lower()
        except EOFError:
            return False
        if answer == "yes":
            return True
        if answer == "no":
            return False
        print("Please enter 'yes' or 'no'.")


def _prepare_output_dir(output_dir: Path, *, overwrite_existing: bool) -> None:
    if _output_dir_has_entries(output_dir):
        if not overwrite_existing:
            raise FileExistsError(f"scan output directory is not empty: {output_dir}")
        _validate_replacement_target(output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def run_scan(
    *,
    config_json: Path | None,
    redshifts: Sequence[float],
    base_seed: int,
    output_dir: Path,
    selected_nsc_mass_msun: float | None = None,
    overwrite_existing: bool = False,
) -> dict[str, Any]:
    """Run a fixed-contract redshift scan and return strict metadata."""
    redshift_values = [float(value) for value in redshifts]
    if not redshift_values:
        raise ValueError("at least one redshift is required")
    if any(not np.isfinite(value) or value < 0.0 for value in redshift_values):
        raise ValueError("redshifts must be finite and nonnegative")
    if len(set(redshift_values)) != len(redshift_values):
        raise ValueError("redshift values must be unique")
    if base_seed < 0:
        raise ValueError("base seed must be nonnegative")
    _prepare_output_dir(output_dir, overwrite_existing=overwrite_existing)

    template = load_config_template(config_json)
    repeat_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    run_records: list[dict[str, Any]] = []
    reference_run: RunResults | None = None
    for index, redshift in enumerate(redshift_values):
        run_dir = output_dir / _redshift_token(index, redshift)
        config = IMRIDynConfig(
            **template,
            output_dir=run_dir,
            redshift=redshift,
            base_seed=base_seed,
        )
        run_simulation(config)
        run = load_run_results(run_dir)
        if reference_run is None:
            reference_run = run

        repeat_frame = run.repeats.copy()
        repeat_frame["run_output_dir"] = str(run_dir)
        repeat_frames.append(repeat_frame)
        event_frames.append(run.events.copy())
        run_records.append(
            {
                "redshift": redshift,
                "directory_token": run_dir.name,
                "output_dir": run_dir,
            }
        )

    combined_repeats = pd.concat(repeat_frames, ignore_index=True)
    summary_path = output_dir / "redshift_scan_summary.csv"
    combined_repeats.to_csv(summary_path, index=False, float_format="%.17g")
    rate_summary = build_rate_summary_data(combined_repeats)
    rate_summary_path = save_plot_data(
        rate_summary, output_dir / "figures" / "data" / "imri_rate_summary.csv"
    )
    rate_ylim = shared_rate_axis_limits(rate_summary)
    combined_events = pd.concat(event_frames, ignore_index=True)
    status_fractions = build_status_fraction_data(combined_events, combined_repeats)
    status_fraction_path = save_plot_data(
        status_fractions,
        output_dir / "figures" / "data" / "imri_status_fraction_summary.csv",
    )
    status_fraction_figures = plot_status_fractions(
        status_fractions,
        output_dir / "figures" / "imri_status_fractions_vs_redshift",
        selected_nsc_mass_msun,
    )
    mass_status_fractions = build_status_fraction_by_smbh_mass_data(
        combined_events, combined_repeats
    )
    mass_status_fraction_path = save_plot_data(
        mass_status_fractions,
        output_dir
        / "figures"
        / "data"
        / "imri_status_fraction_by_smbh_mass_summary.csv",
    )
    mass_status_fraction_figures = plot_status_fractions_vs_smbh_mass(
        mass_status_fractions,
        output_dir / "figures" / "imri_status_fractions_vs_smbh_mass",
        selected_nsc_mass_msun,
    )
    status_fraction_caption_path = write_status_fraction_captions(
        status_fractions,
        mass_status_fractions,
        output_dir / "figures" / "figure_captions.md",
        selected_nsc_mass_msun=selected_nsc_mass_msun,
    )

    if reference_run is None:  # Guarded above; keeps the type contract explicit.
        raise RuntimeError("redshift scan produced no reference run")
    scan_figures = generate_scan_figures(
        combined_repeats,
        output_dir / "figures",
        reference_run,
        selected_nsc_mass_msun,
        rate_ylim,
    )
    metadata: dict[str, Any] = {
        "description": "Fixed-contract redshift IMRI scan",
        "config_json": config_json.resolve() if config_json is not None else None,
        "config_template": template,
        "redshifts": redshift_values,
        "migration_contract": MIGRATION_CONTRACT,
        "base_seed": base_seed,
        "presentation_nsc_mass_msun": selected_nsc_mass_msun,
        "shared_rate_axis_limits_per_myr": rate_ylim,
        "task_seed_policy": (
            "numpy_seed_sequence(base_seed,smbh_index,nsc_index,repeat_index);"
            "redshift_excluded"
        ),
        "rate_normalization": "AGN-duty-cycle-normalized production yield",
        "gas_time_coordinate": "gas_final_time_myr",
        "physical_entry_time_coordinate": "detection_time_myr",
        "duty_cycle_behavior": (
            "gas termination followed by GW-only evolution when eligible"
        ),
        "lisa_entry_boundary": "evolving with redshift",
        "runs": run_records,
        "redshift_scan_summary": summary_path,
        "redshift_rate_summary": rate_summary_path,
        "status_fraction_summary": status_fraction_path,
        "status_fraction_figures": status_fraction_figures,
        "status_fraction_smbh_mass_summary": mass_status_fraction_path,
        "status_fraction_smbh_mass_figures": mass_status_fraction_figures,
        "status_fraction_figure_captions": status_fraction_caption_path,
        "status_fraction_semantics": {
            "variables": list(STATUS_VARIABLE_ORDER),
            "redshift_aggregation": (
                "each redshift bar aggregates the configured SMBH-mass grid"
            ),
            "smbh_mass_aggregation": (
                "each SMBH-mass bar aggregates the requested redshift grid"
            ),
            "rate_completeness": (
                "diagnostic status fractions may include evaluations from repeats "
                "excluded from rate estimation"
            ),
        },
        "scan_figures": scan_figures,
    }
    metadata_path = output_dir / "redshift_scan_metadata.json"
    metadata["redshift_scan_metadata"] = metadata_path
    with metadata_path.open("w") as metadata_file:
        json.dump(_json_paths(metadata), metadata_file, indent=2, allow_nan=False)
    return metadata


def main(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    overwrite_existing = False
    if _output_dir_has_entries(arguments.output_dir):
        overwrite_existing = _confirm_output_replacement(arguments.output_dir)
        if not overwrite_existing:
            print("Scan cancelled; existing output was not changed.")
            return
    outputs = run_scan(
        config_json=arguments.config_json,
        redshifts=arguments.redshifts,
        base_seed=arguments.base_seed,
        output_dir=arguments.output_dir,
        selected_nsc_mass_msun=arguments.nsc_mass_msun,
        overwrite_existing=overwrite_existing,
    )
    print(f"saved scan summary: {outputs['redshift_scan_summary']}")
    print(f"saved status fractions: {outputs['status_fraction_summary']}")
    print(
        "saved SMBH-mass status fractions: "
        f"{outputs['status_fraction_smbh_mass_summary']}"
    )
    print(f"saved figure captions: {outputs['status_fraction_figure_captions']}")
    print(f"saved scan metadata: {outputs['redshift_scan_metadata']}")


if __name__ == "__main__":
    main()
