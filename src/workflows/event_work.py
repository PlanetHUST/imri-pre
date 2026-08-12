"""Evaluate damped objects and assemble their IMRI event records."""

from typing import Sequence

import numpy as np

from disk_quantity.migration_torque import TorqueProfile, torque_si_to_cgs
from physical_process import constants as const
from physical_process import disk_migration
from physical_process import gw_frequency_radius as gw_helpers
from physical_process.agn import AGNProperties
from physical_process.damp import DampingStatus

from .config import IMRIDynConfig
from .detect_rate import classify_detectability
from .engine import IMRIDynEngine, TorqueProfileError
from .event_rate import (
    BH_ORIGIN_MIGRATION_FORMED,
    GAS_FINAL_RADIUS_SAMPLES,
    classify_event_rate,
)
from .models import (
    DampingResult,
    DiskProfiles,
    FloatArray,
    IMRIEventRecord,
    IMRIDynSingleResult,
    TorqueErrorRecord,
)
from .redshift_evolve import RedshiftEvolution
from .runtime import vprint


def _safe_gw_frequency_hz(total_mass_msun: float, radius_cm: float) -> float:
    if not np.isfinite(radius_cm) or radius_cm <= 0.0:
        return np.nan
    return gw_helpers.gw_frequency_hz(total_mass_msun, radius_cm)


def _safe_observed_gw_frequency_hz(
    source_frequency_hz: float,
    redshift: float,
) -> float:
    if not np.isfinite(source_frequency_hz) or source_frequency_hz <= 0.0:
        return np.nan
    return gw_helpers.observed_frame_gw_frequency_hz(source_frequency_hz, redshift)


class IMRIEventEvaluator:
    """Classify damped objects and update one realization's event totals."""

    def __init__(self, engine: IMRIDynEngine) -> None:
        self.engine = engine
        self.config: IMRIDynConfig = engine.config
        self.duty_cycle_myr = engine.duty_cycle_myr

    def classify_imri_candidates(
        self,
        result: IMRIDynSingleResult,
        agn_properties: AGNProperties,
        disk_data: DiskProfiles,
        damping_results: Sequence[DampingResult],
    ) -> None:
        """Classify embedded BHs and delayed-BH embedded stars as IMRI candidates."""
        rg_cm = disk_data["gravitational_radius_cm"]
        for damping_result in damping_results:
            if damping_result["status"] is DampingStatus.EMBEDDED_STAR:
                result.embedded_star_count += 1
            elif damping_result["status"] is DampingStatus.EMBEDDED_BH:
                result.embedded_bh_count += 1
            else:
                result.outside_or_failed_count += 1
                continue

            result.torque_migration_evaluated_count += 1
            try:
                event = self._imri_event_record(
                    result,
                    agn_properties,
                    disk_data,
                    damping_result,
                    rg_cm,
                )
            except (disk_migration.TorqueMigrationError, TorqueProfileError) as exc:
                r_entry_cm = gw_helpers.lisa_entry_radius_cm(
                    result.m_smbh_msun,
                    damping_result["mass_msun"],
                    self.config.lisa_min_gw_frequency_hz,
                    self.config.redshift,
                )
                torque_error: TorqueErrorRecord = {
                    "object_index": damping_result["idx"],
                    "mass_msun": damping_result["mass_msun"],
                    "r_ini_rg": damping_result["final_semi_major_axis_cm"] / rg_cm,
                    "r_entry_rg": r_entry_cm / rg_cm,
                    "error": str(exc),
                }
                result.torque_errors.append(torque_error)
                result.torque_migration_error_count += 1
                vprint(
                    "Torque migration error: "
                    f"object_index={damping_result['idx']}, "
                    f"mass_msun={damping_result['mass_msun']:.6e}, error={exc}"
                )
                continue

            result.imri_events.append(event)
            if event["migration_status"] == "trap_bound":
                result.trap_bound_migration_count += 1
                result.trap_bound_mass_msun += event["mass_msun"]
                vprint(
                    "Trap-bound migration: "
                    f"object_index={event['object_index']}, "
                    f"mass_msun={event['mass_msun']:.6e}, "
                    f"r_ini_rg={event['r_ini_rg']:.6e}, "
                    f"r_entry_rg={event['r_entry_rg']:.6e}, "
                    f"blocking_radius_rg={event['blocking_radius_rg']}"
                )
            else:
                result.finite_torque_migration_count += 1
            if event["bh_candidate"]:
                result.bh_candidate_count += 1
            if event["bh_origin"] == BH_ORIGIN_MIGRATION_FORMED:
                result.migration_formed_bh_count += 1
            if event["is_imri_candidate"]:
                result.imri_count += 1
                result.imri_mass_msun += event["mass_msun"]
                if event["is_lisa_detectable"]:
                    result.detectable_imri_count += 1
                    result.detectable_imri_mass_msun += event["mass_msun"]
                else:
                    result.no_detectable_imri_count += 1
            else:
                result.no_imri_count += 1

        result.imri_rate_per_myr = result.imri_count / self.duty_cycle_myr
        result.imri_mass_rate_msun_per_myr = (
            result.imri_mass_msun / self.duty_cycle_myr
        )
        result.detectable_imri_rate_per_myr = (
            result.detectable_imri_count / self.duty_cycle_myr
        )
        result.detectable_imri_mass_rate_msun_per_myr = (
            result.detectable_imri_mass_msun / self.duty_cycle_myr
        )

    def _imri_event_record(
        self,
        result: IMRIDynSingleResult,
        agn_properties: AGNProperties,
        disk_data: DiskProfiles,
        damping_result: DampingResult,
        rg_cm: float,
    ) -> IMRIEventRecord:
        redshift = self.config.redshift
        migrator_mass_msun = damping_result["mass_msun"]
        r_ini_cm = damping_result["final_semi_major_axis_cm"]
        total_mass_msun = result.m_smbh_msun + migrator_mass_msun
        formation_time_myr = damping_result["formation_time_myr"]
        damping_time_myr = damping_result["damping_time_myr"]

        r_entry_cm = gw_helpers.lisa_entry_radius_cm(
            result.m_smbh_msun,
            migrator_mass_msun,
            self.config.lisa_min_gw_frequency_hz,
            redshift,
        )
        vprint(f"r_ini is {r_ini_cm:.3e} cm, r_entry is {r_entry_cm:.3e} cm")

        migration_outcomes: dict[
            tuple[float, float], disk_migration.TorqueMigrationOutcome
        ] = {}

        def migration_outcome_call(
            start_radius_cm: float, target_radius_cm: float
        ) -> disk_migration.TorqueMigrationOutcome:
            key = (float(start_radius_cm), float(target_radius_cm))
            if key not in migration_outcomes:
                migration_outcomes[key] = self.torque_migration_outcome_to_radius(
                    agn_properties,
                    disk_data,
                    damping_result["mass_g"],
                    start_radius_cm,
                    target_radius_cm,
                )
            return migration_outcomes[key]

        def migration_time_myr_call(
            start_radius_cm: float, target_radius_cm: float
        ) -> float:
            return migration_outcome_call(start_radius_cm, target_radius_cm).time_myr

        initial_migration_outcome = migration_outcome_call(r_ini_cm, r_entry_cm)

        def gw_time_to_entry_myr_call(radius_cm: float) -> float:
            return gw_helpers.peters_time_to_lisa_band_myr(
                result.m_smbh_msun,
                migrator_mass_msun,
                radius_cm,
                self.config.lisa_min_gw_frequency_hz,
                self.config.gw_eccentricity,
                redshift,
            )

        event_decision = classify_event_rate(
            object_status=damping_result["status"],
            formation_time_myr=formation_time_myr,
            damping_time_myr=damping_time_myr,
            embed_time_myr=damping_result["final_time_myr"],
            r_ini_cm=r_ini_cm,
            r_entry_cm=r_entry_cm,
            migration_time_myr_call=migration_time_myr_call,
            gw_time_to_entry_myr_call=gw_time_to_entry_myr_call,
            duty_cycle_myr=self.config.duty_cycle_myr,
            num_radius_samples=GAS_FINAL_RADIUS_SAMPLES,
        )
        vprint(
            "total_migration_time is "
            f"{event_decision.migration_time_to_entry_myr:.3e} myr"
        )
        vprint(f"total_gw_time is {event_decision.gw_time_to_entry_myr:.3e} myr")
        vprint(
            f"stop reason is {event_decision.gas_stop_reason}, "
            f"r_gas_fin is {event_decision.r_gas_fin_cm:.3e} cm"
        )

        redshift_at_gas_fin = RedshiftEvolution(
            self.config.hubble_constant_per_myr
        ).redshift_after_time(redshift, event_decision.gas_final_time_myr)

        f_ini_source_hz = _safe_gw_frequency_hz(total_mass_msun, r_ini_cm)
        f_gas_fin_source_hz = _safe_gw_frequency_hz(
            total_mass_msun, event_decision.r_gas_fin_cm
        )
        f_entry_source_hz = _safe_gw_frequency_hz(total_mass_msun, r_entry_cm)
        f_ini_observed_hz = _safe_observed_gw_frequency_hz(
            f_ini_source_hz, redshift
        )
        f_gas_fin_observed_hz = _safe_observed_gw_frequency_hz(
            f_gas_fin_source_hz, redshift_at_gas_fin
        )

        detect_decision = classify_detectability(
            is_imri_candidate=event_decision.is_imri_candidate,
            primary_mass_msun=result.m_smbh_msun,
            secondary_mass_msun=migrator_mass_msun,
            r_gas_fin_cm=event_decision.r_gas_fin_cm,
            gas_final_time_myr=event_decision.gas_final_time_myr,
            initial_redshift=redshift,
            hubble_constant_per_myr=self.config.hubble_constant_per_myr,
            min_observed_frequency_hz=self.config.lisa_min_gw_frequency_hz,
            max_observed_frequency_hz=self.config.lisa_max_gw_frequency_hz,
            eccentricity=self.config.gw_eccentricity,
            cosmic_merge_time_myr=self.config.cosmic_merge_time_myr,
            cosmic_merge_smbh_mass_factor=self.config.cosmic_merge_smbh_mass_factor,
            cosmic_merge_radius_factor=self.config.cosmic_merge_radius_factor,
            snr_threshold=self.config.snr_threshold,
        )

        f_entry_output_source_hz = (
            detect_decision.source_frequency_hz
            if np.isfinite(detect_decision.source_frequency_hz)
            else f_entry_source_hz
        )
        f_entry_output_observed_hz = (
            detect_decision.observed_frequency_hz
            if np.isfinite(detect_decision.observed_frequency_hz)
            else _safe_observed_gw_frequency_hz(f_entry_source_hz, redshift)
        )

        return {
            "m_smbh_msun": result.m_smbh_msun,
            "m_nsc_msun": result.m_nsc_msun,
            "repeat": result.repeat_idx,
            "redshift": redshift,
            "detection_redshift": detect_decision.detection_redshift,
            "object_index": damping_result["idx"],
            "mass_msun": migrator_mass_msun,
            "bh_origin": event_decision.bh_origin,
            "event_rate_class": event_decision.event_rate_class,
            "detect_rate_class": detect_decision.detect_rate_class,
            "formation_time_myr": formation_time_myr,
            "damping_time_myr": damping_time_myr,
            "bh_embedding_time_myr": event_decision.bh_embedding_time_myr,
            "r_ini_cm": r_ini_cm,
            "r_ini_rg": r_ini_cm / rg_cm,
            "r_gas_fin_cm": event_decision.r_gas_fin_cm,
            "r_gas_fin_rg": event_decision.r_gas_fin_cm / rg_cm,
            "r_entry_cm": r_entry_cm,
            "r_entry_rg": r_entry_cm / rg_cm,
            "r_det_entry_cm": detect_decision.final_entry_radius_cm,
            "r_det_entry_rg": detect_decision.final_entry_radius_cm / rg_cm,
            "f_ini_src_hz": f_ini_source_hz,
            "f_ini_obs_hz": f_ini_observed_hz,
            "f_gas_fin_src_hz": f_gas_fin_source_hz,
            "f_gas_fin_obs_hz": f_gas_fin_observed_hz,
            "f_entry_src_hz": f_entry_output_source_hz,
            "f_entry_obs_hz": f_entry_output_observed_hz,
            "gas_elapsed_time_myr": event_decision.gas_elapsed_time_myr,
            "gas_final_time_myr": event_decision.gas_final_time_myr,
            "migration_time_to_entry_myr": event_decision.migration_time_to_entry_myr,
            "migration_status": initial_migration_outcome.status,
            "blocking_radius_rg": (
                initial_migration_outcome.blocking_radius_cm / rg_cm
                if initial_migration_outcome.blocking_radius_cm is not None
                else None
            ),
            "gw_time_to_entry_myr": event_decision.gw_time_to_entry_myr,
            "gw_time_after_gas_myr": detect_decision.gw_time_after_gas_myr,
            "detection_time_myr": detect_decision.detection_time_myr,
            "final_smbh_mass_msun": detect_decision.final_smbh_mass_msun,
            "cosmic_merge_applied": detect_decision.cosmic_merge_applied,
            "bh_candidate": event_decision.bh_candidate,
            "is_imri_candidate": event_decision.is_imri_candidate,
            "is_lisa_detectable": detect_decision.is_lisa_detectable,
            "snr_proxy": detect_decision.snr_proxy,
            "event_rate_reason": event_decision.event_rate_reason,
            "detect_rate_reason": detect_decision.detect_rate_reason,
        }

    def torque_migration_outcome_to_radius(
        self,
        agn_properties: AGNProperties,
        disk_data: DiskProfiles,
        mass_g: float,
        start_radius_cm: float,
        target_radius_cm: float,
    ) -> disk_migration.TorqueMigrationOutcome:
        """Return the signed torque migration outcome for one radial path."""
        if start_radius_cm <= target_radius_cm:
            return disk_migration.torque_migration_outcome(
                np.asarray([], dtype=float),
                np.asarray([], dtype=float),
                mass_g,
                start_radius_cm,
                target_radius_cm,
                agn_properties.smbh_mass_msun,
            )

        migrator_mass_msun = mass_g / const.msun
        torque_profile = self.engine.get_torque_profile(
            disk_data,
            agn_properties.smbh_mass_msun,
            migrator_mass_msun,
        )
        torque_cgs = self._migration_torque_cgs(
            torque_profile,
            agn_properties,
            disk_data,
            mass_g,
        )
        return disk_migration.torque_migration_outcome(
            np.asarray(torque_profile.radius_cm, dtype=float),
            torque_cgs,
            mass_g,
            start_radius_cm,
            target_radius_cm,
            agn_properties.smbh_mass_msun,
        )

    def _migration_torque_cgs(
        self,
        torque_profile: TorqueProfile,
        agn_properties: AGNProperties,
        disk_data: DiskProfiles,
        mass_g: float,
    ) -> FloatArray:
        """Compose the selected migration torque and convert SI to CGS."""
        torque_radius_cm = np.asarray(torque_profile.radius_cm, dtype=float)
        disk_radius_cm = np.asarray(disk_data["radius_cm"], dtype=float)
        aspect_ratio = np.asarray(disk_data["aspect_ratio"], dtype=float)
        order = np.argsort(disk_radius_cm)
        local_aspect_ratio = np.interp(
            torque_radius_cm,
            disk_radius_cm[order],
            aspect_ratio[order],
        )
        if "torque_alpha" not in disk_data:
            raise TorqueProfileError("cached pAGN disk has no torque alpha")
        torque_alpha = float(disk_data["torque_alpha"])
        mass_ratio = mass_g / (agn_properties.smbh_mass_msun * const.msun)
        selection = disk_migration.select_migration_torque_si(
            torque_profile,
            mass_ratio,
            local_aspect_ratio,
            torque_alpha,
            self.config.gap_opening_condition,
            self.config.consider_type_ii_migration,
        )
        return torque_si_to_cgs(selection.torque_si)
