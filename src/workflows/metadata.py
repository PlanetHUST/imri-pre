"""Simulation metadata assembly for the IMRI workflow."""

import time
from datetime import datetime
from typing import Any

import physical_process.gw_frequency_radius as gw_helpers

from .config import IMRIDynConfig
from lisa.snr import (
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


def build_metadata(
    config: IMRIDynConfig,
    runtime_stats: dict[str, float | int | str],
) -> dict[str, Any]:
    """Return configuration, derived model quantities, and runtime statistics."""
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
            "metallicity": config.metallicity,
            "initial_stellar_density_gamma": config.initial_stellar_density_gamma,
            "nsc_radius_pc_at_reference_mass": config.nsc_radius_pc_at_reference_mass,
            "nsc_radius_reference_mass_msun": config.nsc_radius_reference_mass_msun,
            "nsc_radius_mass_slope": config.nsc_radius_mass_slope,
            "use_disk_inner_radius": config.use_disk_inner_radius,
            "disk_model": config.disk_model,
            "consider_type_ii_migration": config.consider_type_ii_migration,
            "gap_opening_condition": config.gap_opening_condition,
            "gap_criterion": config.gap_criterion.to_dict(),
            "redshift": config.redshift,
            "hubble_constant_per_myr": config.hubble_constant_per_myr,
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
            "frequency_frame": "observer",
            "mass_inputs_frame": "source",
            "noise_quantity": "effective_strain_psd_hz_inverse",
            "snr_statistic": "lisa_snr",
            "snr_proxy_compatibility": "deprecated_alias_of_lisa_snr",
        },
        "runtime_statistics": runtime_stats,
    }
