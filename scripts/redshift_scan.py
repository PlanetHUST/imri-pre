"""Run paired scalar-redshift IMRI workflows and build comparison figures."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any, Sequence, cast

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
    "consider_type_ii_migration": True,
    "migration_model": "continuous_gap",
    "disk_model": "torque_with_trap",
    "lisa_min_gw_frequency_hz": 1.0e-3,
    "lisa_max_gw_frequency_hz": 1.0e-1,
    "max_gw_time_myr": 13.6e3,
    "gw_eccentricity": 0.0,
    "snr_threshold": 20.0,
    "lisa_mission_duration_years": 4.0,
    "snr_window_placement": "optimal",
    "detector_hubble_constant_km_s_mpc": 67.4,
    "detector_omega_m": 0.315,
    "damping_cpu_count": None,
    "checkpoint_grid_interval": 5,
    "print_debug_results": False,
    "nsc_radius_pc_at_reference_mass": 3.31,
    "nsc_radius_mass_slope": 0.321,
    "nsc_radius_reference_mass_msun": 3.6e6,
    "gap_opening_condition": "kii",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from workflows.config import (
    DEFAULT_DISK_MODEL,
    DISK_MODELS,
    SEMI_WITHOUT_TRAP,
    TORQUE_WITH_TRAP,
    IMRIDynConfig,
)
from physical_process.disk_migration import DEFAULT_MIGRATION_MODEL, MIGRATION_MODELS
from workflows.rate_plots import (
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
    "consider_type_ii_migration",
    "migration_model",
    "disk_model",
    "lisa_min_gw_frequency_hz",
    "lisa_max_gw_frequency_hz",
    "max_gw_time_myr",
    "gw_eccentricity",
    "snr_threshold",
    "lisa_mission_duration_years",
    "snr_window_placement",
    "detector_hubble_constant_km_s_mpc",
    "detector_omega_m",
    "damping_cpu_count",
    "checkpoint_grid_interval",
    "print_debug_results",
    "nsc_radius_pc_at_reference_mass",
    "nsc_radius_mass_slope",
    "nsc_radius_reference_mass_msun",
    "gap_opening_condition",
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
        "--disk-models",
        choices=DISK_MODELS,
        nargs="+",
        default=DISK_MODELS,
        help="ordered migration scenarios; defaults to the paired comparison",
    )
    parser.add_argument(
        "--migration-model",
        choices=MIGRATION_MODELS,
        default=DEFAULT_MIGRATION_MODEL,
        help="torque composition model shared by every scan run",
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
    if "disk_model" not in source and is_metadata:
        source = dict(source)
        source["disk_model"] = DEFAULT_DISK_MODEL
        warnings.warn(
            "historical simulation metadata has no disk_model; explicitly "
            f"inferring {DEFAULT_DISK_MODEL}",
            stacklevel=2,
        )
    if "migration_model" not in source:
        source = dict(source)
        source["migration_model"] = DEFAULT_MIGRATION_MODEL
        warnings.warn(
            "configuration has no migration_model; explicitly inferring "
            f"{DEFAULT_MIGRATION_MODEL}",
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
    legacy_fields = sorted(
        field
        for field in (
            "hubble_constant_per_myr",
            "cosmic_merge_time_myr",
            "cosmic_merge_smbh_mass_factor",
            "cosmic_merge_radius_factor",
        )
        if field in source
    )
    if legacy_fields:
        warnings.warn(
            "ignoring legacy single-AGN-incompatible configuration fields: "
            f"{legacy_fields}",
            stacklevel=2,
        )
    missing = [field for field in CONFIG_TEMPLATE_FIELDS if field not in source]
    if missing:
        raise ValueError(f"config JSON is missing explicit fields: {missing}")
    template = {field: source[field] for field in CONFIG_TEMPLATE_FIELDS}
    return template


def _redshift_token(index: int, redshift: float) -> str:
    value = f"{redshift:.12g}".replace("-", "m").replace("+", "p").replace(".", "p")
    return f"z{index:03d}_{value}"


def build_paired_rate_comparison(rate_summary: pd.DataFrame) -> pd.DataFrame:
    """Return model-separated rates and a guarded semi-minus-torque comparison."""
    key_columns = ["redshift", "m_smbh_msun", "m_nsc_msun"]
    required = [
        "disk_model",
        *key_columns,
        "imri_rate_mean_per_myr",
        "result_state",
    ]
    missing = sorted(set(required).difference(rate_summary.columns))
    if missing:
        raise ValueError(f"rate summary is missing comparison columns: {missing}")

    records: list[dict[str, Any]] = []
    for key, group in rate_summary.groupby(key_columns, sort=True, dropna=False):
        if not isinstance(key, tuple) or len(key) != len(key_columns):
            raise ValueError(f"unexpected comparison group key: {key!r}")
        redshift, smbh_mass_msun, nsc_mass_msun = (float(value) for value in key)
        by_model = {str(row["disk_model"]): row for _, row in group.iterrows()}
        torque = by_model.get(TORQUE_WITH_TRAP)
        semi = by_model.get(SEMI_WITHOUT_TRAP)
        torque_rate = (
            float(torque["imri_rate_mean_per_myr"]) if torque is not None else np.nan
        )
        semi_rate = (
            float(semi["imri_rate_mean_per_myr"]) if semi is not None else np.nan
        )
        torque_state = str(torque["result_state"]) if torque is not None else "missing"
        semi_state = str(semi["result_state"]) if semi is not None else "missing"
        both_finite = np.isfinite(torque_rate) and np.isfinite(semi_rate)
        difference = semi_rate - torque_rate if both_finite else np.nan
        ratio = semi_rate / torque_rate if both_finite and torque_rate > 0.0 else np.nan

        if torque is None or semi is None:
            comparison_status = "missing_scenario"
        elif not both_finite:
            unavailable = [
                name
                for name, value in (
                    (TORQUE_WITH_TRAP, torque_rate),
                    (SEMI_WITHOUT_TRAP, semi_rate),
                )
                if not np.isfinite(value)
            ]
            comparison_status = "unavailable:" + ",".join(unavailable)
        elif torque_rate <= 0.0:
            comparison_status = "ratio_undefined_nonpositive_torque_rate"
        elif torque_state == "valid" and semi_state == "valid":
            comparison_status = "valid"
        else:
            comparison_status = "partial_repeats"

        records.append(
            {
                "redshift": redshift,
                "m_smbh_msun": smbh_mass_msun,
                "m_nsc_msun": nsc_mass_msun,
                "torque_with_trap_rate_per_myr": torque_rate,
                "torque_with_trap_result_state": torque_state,
                "semi_without_trap_rate_per_myr": semi_rate,
                "semi_without_trap_result_state": semi_state,
                "semi_minus_torque_rate_per_myr": difference,
                "semi_over_torque_rate_ratio": ratio,
                "comparison_status": comparison_status,
            }
        )
    return pd.DataFrame.from_records(records)


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
    disk_models: Sequence[str] = DISK_MODELS,
    base_seed: int,
    output_dir: Path,
    selected_nsc_mass_msun: float | None = None,
    overwrite_existing: bool = False,
    migration_model: str = DEFAULT_MIGRATION_MODEL,
) -> dict[str, Any]:
    """Run a paired scan and return the strict metadata payload."""
    redshift_values = [float(value) for value in redshifts]
    if not redshift_values:
        raise ValueError("at least one redshift is required")
    if any(not np.isfinite(value) or value < 0.0 for value in redshift_values):
        raise ValueError("redshifts must be finite and nonnegative")
    if len(set(redshift_values)) != len(redshift_values):
        raise ValueError("redshift values must be unique")
    if base_seed < 0:
        raise ValueError("base seed must be nonnegative")
    model_values = [str(value) for value in disk_models]
    if not model_values:
        raise ValueError("at least one disk model is required")
    if len(set(model_values)) != len(model_values):
        raise ValueError("disk model values must be unique")
    invalid_models = sorted(set(model_values).difference(DISK_MODELS))
    if invalid_models:
        raise ValueError(f"unsupported disk models: {invalid_models}")
    if migration_model not in MIGRATION_MODELS:
        raise ValueError(f"migration model must be one of {MIGRATION_MODELS}")
    _prepare_output_dir(output_dir, overwrite_existing=overwrite_existing)

    template = load_config_template(config_json)
    repeat_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    run_records: list[dict[str, Any]] = []
    reference_runs: dict[str, RunResults] = {}
    for disk_model in model_values:
        for index, redshift in enumerate(redshift_values):
            run_dir = output_dir / disk_model / _redshift_token(index, redshift)
            config_values = dict(template)
            config_values["disk_model"] = disk_model
            config_values["migration_model"] = migration_model
            config = IMRIDynConfig(
                **config_values,
                output_dir=run_dir,
                redshift=redshift,
                base_seed=base_seed,
            )
            run_simulation(config)
            run = load_run_results(run_dir)
            reference_runs.setdefault(disk_model, run)

            repeat_frame = run.repeats.copy()
            repeat_frame["run_output_dir"] = str(run_dir)
            repeat_frames.append(repeat_frame)
            event_frames.append(run.events.copy())
            run_records.append(
                {
                    "disk_model": disk_model,
                    "redshift": redshift,
                    "directory_token": run_dir.name,
                    "output_dir": run_dir,
                }
            )

    combined_repeats = pd.concat(repeat_frames, ignore_index=True)
    summary_path = output_dir / "disk_model_scan_summary.csv"
    combined_repeats.to_csv(summary_path, index=False, float_format="%.17g")
    rate_summary = build_rate_summary_data(combined_repeats)
    rate_summary_path = save_plot_data(
        rate_summary, output_dir / "figures" / "data" / "imri_rate_summary.csv"
    )
    comparison = build_paired_rate_comparison(rate_summary)
    comparison_path = output_dir / "disk_model_rate_comparison.csv"
    comparison.to_csv(comparison_path, index=False, float_format="%.17g")
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

    scan_figures: dict[str, Any] = {}
    for disk_model in model_values:
        model_repeats = cast(
            pd.DataFrame,
            combined_repeats.loc[
                combined_repeats["disk_model"].astype(str).eq(disk_model)
            ].copy(),
        )
        scan_figures[disk_model] = generate_scan_figures(
            model_repeats,
            output_dir / "figures" / disk_model,
            reference_runs[disk_model],
            selected_nsc_mass_msun,
            rate_ylim,
        )
    metadata: dict[str, Any] = {
        "description": "Paired disk-migration-scenario and redshift IMRI scan",
        "config_json": config_json.resolve() if config_json is not None else None,
        "config_template": template,
        "redshifts": redshift_values,
        "disk_models": model_values,
        "base_seed": base_seed,
        "presentation_nsc_mass_msun": selected_nsc_mass_msun,
        "shared_rate_axis_limits_per_myr": rate_ylim,
        "task_seed_policy": (
            "numpy_seed_sequence(base_seed,smbh_index,nsc_index,repeat_index);"
            "redshift_and_disk_model_excluded"
        ),
        "rate_normalization": "AGN-duty-cycle-normalized production yield",
        "gas_time_coordinate": "gas_final_time_myr",
        "migration_model": migration_model,
        "physical_entry_time_coordinate": "detection_time_myr",
        "duty_cycle_behavior": (
            "gas termination followed by GW-only evolution when eligible"
        ),
        "lisa_entry_boundary": "evolving with redshift",
        "runs": run_records,
        "disk_model_scan_summary": summary_path,
        "disk_model_rate_summary": rate_summary_path,
        "disk_model_rate_comparison": comparison_path,
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
    metadata_path = output_dir / "disk_model_scan_metadata.json"
    metadata["disk_model_scan_metadata"] = metadata_path
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
        disk_models=arguments.disk_models,
        base_seed=arguments.base_seed,
        output_dir=arguments.output_dir,
        selected_nsc_mass_msun=arguments.nsc_mass_msun,
        overwrite_existing=overwrite_existing,
        migration_model=arguments.migration_model,
    )
    print(f"saved scan summary: {outputs['disk_model_scan_summary']}")
    print(f"saved comparison: {outputs['disk_model_rate_comparison']}")
    print(f"saved status fractions: {outputs['status_fraction_summary']}")
    print(
        "saved SMBH-mass status fractions: "
        f"{outputs['status_fraction_smbh_mass_summary']}"
    )
    print(f"saved figure captions: {outputs['status_fraction_figure_captions']}")
    print(f"saved scan metadata: {outputs['disk_model_scan_metadata']}")


if __name__ == "__main__":
    main()
