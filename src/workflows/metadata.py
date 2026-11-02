"""Simulation metadata assembly for the IMRI workflow."""

import time
import hashlib
from importlib.metadata import version
from datetime import datetime
from pathlib import Path
from typing import Any

import physical_process.gw_frequency_radius as gw_helpers
from physical_process.sampling import SOLAR_METALLICITY
from physical_process.redshift_evolve import RedshiftEvolution
from physical_process.migration import (
    CRIDA_DEEP_GAP_PARAMETER,
    KANAGAWA_CALIBRATED_MAX_PARAMETER,
    KANAGAWA_GAP_DEPTH_COEFFICIENT,
)

from .config import (
    IMRIDynConfig,
    MIGRATION_CONTRACT,
    PAGN_ALPHA,
    SEMI_WITHOUT_TRAP,
    migration_contract_name,
)
from lisa.snr import (
    COUPLED_TRACK_SNR_RELATIVE_TOLERANCE,
    COSMOLOGY_QUADRATURE_RELATIVE_TOLERANCE,
    LISA_AVERAGING_CONVENTION,
    LISA_ARM_LENGTH_M,
    LISA_COSMOLOGY_MODEL,
    LISA_ISCO_MODEL,
    LISA_SENSITIVITY_MODEL,
    LISA_WAVEFORM_MODEL,
    SNR_QUADRATURE_METHOD,
    SNR_QUADRATURE_RELATIVE_TOLERANCE,
)


def _tree_sha256(directory: Path, suffixes: tuple[str, ...] | None = None) -> str:
    """Fingerprint names and bytes, independent of the directory path."""
    digest = hashlib.sha256()
    for path in sorted(
        item for item in directory.rglob("*")
        if item.is_file() and (suffixes is None or item.suffix in suffixes)
    ):
        digest.update(str(path.relative_to(directory)).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _stellar_model_metadata(config: IMRIDynConfig) -> dict[str, Any]:
    details: dict[str, Any] = {
        "model": config.stellar_evolution_model,
        "kick_model": config.natal_kick_model,
        "precollapse_mass_model": "fixed_zams_mass",
        "metallicity_absolute_z": config.metallicity * SOLAR_METALLICITY,
    }
    if config.stellar_evolution_model == "sevn":
        from sevnpy.sevn import SEVNmanager

        if config.sevn_hydrogen_tables is None or config.sevn_helium_tables is None:
            raise ValueError("SEVN table paths are required for metadata")
        defaults = SEVNmanager.get_sevnParamsDefault()
        details.update(
            sevnpy_version=version("sevnpy"),
            hydrogen_tables=str(config.sevn_hydrogen_tables),
            helium_tables=str(config.sevn_helium_tables),
            hydrogen_tables_sha256=_tree_sha256(config.sevn_hydrogen_tables),
            helium_tables_sha256=_tree_sha256(config.sevn_helium_tables),
            sevn_supernova_parameters={
                key: value for key, value in defaults.items() if key.startswith("sn_")
            },
        )
    if config.natal_kick_model == "sevn":
        kick_source = Path(__file__).resolve().parents[1] / "physical_process" / "natal_kick.py"
        details["kick_orbit_interface"] = "physical_process.natal_kick.apply_orbital_kick"
        details["kick_orbit_sha256"] = hashlib.sha256(kick_source.read_bytes()).hexdigest()
        details["kick_direction"] = "object_seeded_isotropic_orbital_frame"
    return details


def _migration_contract_metadata(config: IMRIDynConfig) -> dict[str, Any]:
    """Describe the active migration scenario and its equations."""
    if config.disk_model == SEMI_WITHOUT_TRAP:
        return {
            "name": migration_contract_name(SEMI_WITHOUT_TRAP),
            "scenario": "comparison_only_not_production",
            "type_i_timescale": (
                "t_I(R)=M_smbh^2*h^2/(2*m*Sigma*R^2*Omega)"
            ),
            "type_ii_timescale": "t_II(R)=1/(2*alpha*h^2*Omega)",
            "integration": "t_mig=integral t_local(R) dlnR",
            "gap_opening_condition": config.gap_opening_condition,
            "type_ii_kanagawa_threshold": 11.0,
            "consider_type_ii_migration": config.consider_type_ii_migration,
            "alpha": PAGN_ALPHA,
            "direction": "inward_only",
            "traps": "not_represented",
            "classical_type_ii_branch": config.consider_type_ii_migration,
            "disk_profile": "pagn_sirko_thompson",
            "omega_source": "project_keplerian_from_smbh_mass",
        }

    return {
        "name": MIGRATION_CONTRACT,
        "gravitational_torque": "jimenez_masset_2017_linear",
        "thermal_torque": "grishin_gilbaum_stone_2024_saturated",
        "thermal_validity_policy": "exclude_outside_linear_regime",
        "thermal_extrapolation_enabled": False,
        "gap_depth": "kanagawa_1_over_1_plus_0p04K",
        "gap_depth_coefficient": KANAGAWA_GAP_DEPTH_COEFFICIENT,
        "kanagawa_calibrated_max_parameter": (
            KANAGAWA_CALIBRATED_MAX_PARAMETER
        ),
        "deep_gap_diagnostic": "crida_parameter_le_1",
        "crida_deep_gap_parameter": CRIDA_DEEP_GAP_PARAMETER,
        "gap_transport": (
            "continuous_inertia_limited_viscous_drift_min_1_B"
        ),
        "disk_dominance_parameter": "B=Sigma*r^2/migrator_mass",
        "wkb_jump_policy": "interpolate_log_slopes_across_flagged_cells",
        "classical_type_ii_branch": False,
        "embedded_bh_luminosity": "eddington_capped",
    }


def build_metadata(
    config: IMRIDynConfig,
    runtime_stats: dict[str, float | int | str],
) -> dict[str, Any]:
    """Return configuration, derived model quantities, and runtime statistics."""
    cosmic_clock = RedshiftEvolution(
        config.detector_hubble_constant_km_s_mpc,
        config.detector_omega_m,
    )
    return {
        "simulation_info": {
            "date": datetime.now().isoformat(),
            "timestamp_s_since_epoch": time.time(),
            "description": "Dynamical IMRI rate simulation with parallel damping",
        },
        "parameters": {
            "smbh_masses_msun": config.smbh_masses_msun.tolist(),
            "nsc_masses_msun": config.nsc_masses_msun.tolist(),
            "num_repeats": config.num_repeats,
            "total_simulations": (
                len(config.smbh_masses_msun)
                * len(config.nsc_masses_msun)
                * config.num_repeats
            ),
            "duty_cycle_myr": config.duty_cycle_myr,
            "consumption_replenishment": {
                "enabled": config.consumption_replenishment,
                "refill_trigger": "first_isco_after_physical_imri_entry",
                "refill_horizon": "agn_duty_cycle",
                "donor_pool": "initial_nsc_objects_not_selected_as_primary_disk_crossers",
                "donor_policy": "unique_uniform_without_replacement",
                "channel_orbit": "initial_primary_orbit",
                "relaxation_time_role": "diagnostic_only",
                "external_inflow": False,
            },
            "metallicity": config.metallicity,
            "popiii_max_mass_msun": config.popiii_max_mass_msun,
            "popiii_imf_slope": config.popiii_imf_slope,
            "metallicity_absolute_z": config.metallicity * SOLAR_METALLICITY,
            "stellar_evolution_model": config.stellar_evolution_model,
            "natal_kick_model": config.natal_kick_model,
            "sevn_hydrogen_tables": (
                str(config.sevn_hydrogen_tables)
                if config.sevn_hydrogen_tables is not None else None
            ),
            "sevn_helium_tables": (
                str(config.sevn_helium_tables)
                if config.sevn_helium_tables is not None else None
            ),
            "initial_stellar_density_gamma": config.initial_stellar_density_gamma,
            "nsc_radius_pc_at_reference_mass": config.nsc_radius_pc_at_reference_mass,
            "nsc_radius_reference_mass_msun": config.nsc_radius_reference_mass_msun,
            "nsc_radius_mass_slope": config.nsc_radius_mass_slope,
            "use_disk_inner_radius": config.use_disk_inner_radius,
            "disk_model": config.disk_model,
            "disk_migration_scenario": config.disk_model,
            "gap_opening_condition": config.gap_opening_condition,
            "consider_type_ii_migration": config.consider_type_ii_migration,
            "migration_contract": _migration_contract_metadata(config),
            "orbit_model": "circular_coupled_gas_gw_v1",
            "orbit_assumptions": {
                "eccentricity_at_embedding": "projected_to_zero_after_e_le_0.01_damping",
                "component_masses": (
                    "fixed" if config.stellar_evolution_model == "fixed_mass"
                    else "sevn_event_piecewise"
                ),
                "disk_background": "static_until_duty_cycle_end",
                "gas_and_gw": "simultaneous_signed_radial_drift",
                "band_entry": "first_observed_frequency_crossing",
                "gas_active_snr": "first_order_circular_coupled_track",
            },
            "legacy_route_columns": (
                "initial_redshift_geometry_and_evaluation_compatibility_only;"
                "not_a_gas_gw_crossover_or_coupled_trap_census"
            ),
            "redshift": config.redshift,
            "cosmic_time_model": "flat_lcdm_matter_lambda",
            "cosmic_time_origin": "source_redshift_at_agn_start",
            "cosmic_time_hubble_constant_per_myr": (
                cosmic_clock.hubble_constant_per_myr
            ),
            "cosmic_age_at_initial_redshift_myr": (
                cosmic_clock.redshift_to_cosmic_time(config.redshift)
            ),
            "cosmic_age_at_z0_myr": cosmic_clock.redshift_to_cosmic_time(0.0),
            "lisa_min_gw_frequency_hz": config.lisa_min_gw_frequency_hz,
            "source_lisa_min_gw_frequency_hz": gw_helpers.source_frame_gw_frequency_hz(
                config.lisa_min_gw_frequency_hz,
                config.redshift,
            ),
            "lisa_max_gw_frequency_hz": config.lisa_max_gw_frequency_hz,
            "source_lisa_max_gw_frequency_hz": gw_helpers.source_frame_gw_frequency_hz(
                config.lisa_max_gw_frequency_hz,
                config.redshift,
            ),
            "snr_threshold": config.snr_threshold,
            "lisa_mission_duration_years": config.lisa_mission_duration_years,
            "snr_window_placement": config.snr_window_placement,
            "detector_hubble_constant_km_s_mpc": (
                config.detector_hubble_constant_km_s_mpc
            ),
            "detector_omega_m": config.detector_omega_m,
            "cosmic_merge_time_myr": config.cosmic_merge_time_myr,
            "cosmic_merge_time_origin": "disabled_in_coupled_orbit",
            "cosmic_merge_smbh_mass_factor": config.cosmic_merge_smbh_mass_factor,
            "cosmic_merge_radius_factor": config.cosmic_merge_radius_factor,
            "max_gw_time_myr": config.max_gw_time_myr,
            "gw_eccentricity": config.gw_eccentricity,
            "damping_cpu_count": config.damping_cpu_count,
            "checkpoint_grid_interval": config.checkpoint_grid_interval,
            "print_debug_results": config.print_debug_results,
            "base_seed": config.base_seed,
            "task_seed_policy": (
                "numpy_seed_sequence(base_seed,smbh_index,nsc_index,repeat_index);"
                "redshift_excluded"
                if config.base_seed is not None
                else None
            ),
        },
        "stellar_model": _stellar_model_metadata(config),
        "detector_model": {
            "sensitivity_model": LISA_SENSITIVITY_MODEL,
            "waveform_model": LISA_WAVEFORM_MODEL,
            "averaging_convention": LISA_AVERAGING_CONVENTION,
            "cosmology_model": LISA_COSMOLOGY_MODEL,
            "cosmology_flat": True,
            "cosmology_omega_lambda": 1.0 - config.detector_omega_m,
            "cosmology_radiation_included": False,
            "cosmology_quadrature_relative_tolerance": (
                COSMOLOGY_QUADRATURE_RELATIVE_TOLERANCE
            ),
            "lisa_arm_length_m": LISA_ARM_LENGTH_M,
            "isco_model": LISA_ISCO_MODEL,
            "snr_quadrature_method": SNR_QUADRATURE_METHOD,
            "snr_quadrature_relative_tolerance": (
                SNR_QUADRATURE_RELATIVE_TOLERANCE
            ),
            "coupled_track_snr_model": (
                "quadrupole_spa_amplitude_actual_in_band_residence_time"
            ),
            "coupled_track_snr_relative_tolerance": (
                COUPLED_TRACK_SNR_RELATIVE_TOLERANCE
            ),
            "environmental_phase_template": False,
            "frequency_frame": "observer",
            "mass_inputs_frame": "source",
            "noise_quantity": "effective_strain_psd_hz_inverse",
            "snr_statistic": "lisa_snr",
            "snr_proxy_compatibility": "deprecated_alias_of_lisa_snr",
        },
        "runtime_statistics": runtime_stats,
    }
