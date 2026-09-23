"""Evaluate damped objects and assemble their IMRI event records."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from physical_process.migration import TorqueProfile, torque_si_to_cgs
from physical_process import constants as const
from physical_process import migration as disk_migration
from physical_process import gw_frequency_radius as gw_helpers
from physical_process.agn import AGNProperties
from physical_process.damp import DampingStatus

from .config import IMRIDynConfig
from lisa.detectability import (
    classify_detectability,
    solve_time_to_observed_lisa_entry,
    unsupported_band_entry_decision,
)
from .engine import IMRIDynEngine, MigrationProfileError, TorqueProfileError
from .event_rate import (
    BH_ORIGIN_MIGRATION_FORMED,
    EVENT_RATE_NO_IMRI,
    GAS_FINAL_RADIUS_SAMPLES,
    classify_event_rate,
    classify_unsupported_band_entry,
    is_physical_imri_entry,
)
from lisa.snr import (
    SNR_STATUS_EVALUATED,
    SNR_STATUS_MODEL_UNAVAILABLE,
    SNR_STATUS_NO_OVERLAP,
    SNR_STATUS_NUMERICAL_FAILURE,
    SNR_STATUS_REQUIRED_INPUT_UNAVAILABLE,
)
from .models import (
    DampingResult,
    DiskProfiles,
    FloatArray,
    IMRIEventRecord,
    IMRIDynSingleResult,
    MigrationErrorRecord,
)
from physical_process.redshift_evolve import RedshiftEvolution
from .runtime import vprint

BAND_ENTRY_IN_MIGRATION_DOMAIN = "entry_in_migration_domain"
BAND_ENTRY_HANDOFF = "entry_below_migration_domain_but_above_isco"
BAND_ENTRY_INSIDE_ISCO = "band_entry_inside_isco"
BAND_ENTRY_PROFILE_ERROR = "profile_error"


@dataclass(frozen=True)
class BandEntryRoute:
    """Resolved gas target for a valid circular entry outside ISCO."""

    status: str
    disk_inner_radius_cm: float
    migration_min_radius_cm: float
    handoff_radius_cm: float


def resolve_band_entry_route(
    boundary: gw_helpers.CircularBandBoundary,
    disk_inner_radius_cm: float,
    migration_radius_cm: FloatArray,
) -> BandEntryRoute:
    """Resolve an in-domain entry or lower-bound gas-to-GW handoff."""

    radius_cm = np.asarray(migration_radius_cm, dtype=float)
    if (
        radius_cm.ndim != 1
        or radius_cm.size < 2
        or not np.all(np.isfinite(radius_cm) & (radius_cm > 0.0))
    ):
        raise MigrationProfileError(
            "migration profile has no valid one-dimensional radius domain"
        )
    if not np.isfinite(disk_inner_radius_cm) or disk_inner_radius_cm <= 0.0:
        raise MigrationProfileError("disk profile has no finite positive inner radius")
    if boundary.status == BAND_ENTRY_INSIDE_ISCO:
        raise ValueError("inside-ISCO entry must be handled before migration routing")

    migration_min_radius_cm = float(np.min(radius_cm))
    lower_domain_radius_cm = max(disk_inner_radius_cm, migration_min_radius_cm)
    lower_tolerance_cm = max(
        np.spacing(lower_domain_radius_cm),
        lower_domain_radius_cm * 1.0e-12,
    )
    if boundary.formal_entry_radius_cm < lower_domain_radius_cm - lower_tolerance_cm:
        return BandEntryRoute(
            status=BAND_ENTRY_HANDOFF,
            disk_inner_radius_cm=disk_inner_radius_cm,
            migration_min_radius_cm=migration_min_radius_cm,
            handoff_radius_cm=lower_domain_radius_cm,
        )
    return BandEntryRoute(
        status=BAND_ENTRY_IN_MIGRATION_DOMAIN,
        disk_inner_radius_cm=disk_inner_radius_cm,
        migration_min_radius_cm=migration_min_radius_cm,
        handoff_radius_cm=boundary.formal_entry_radius_cm,
    )


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

            result.band_entry_evaluated_count += 1
            boundary = gw_helpers.circular_band_boundary(
                result.m_smbh_msun,
                damping_result["mass_msun"],
                self.config.lisa_min_gw_frequency_hz,
                self.config.redshift,
            )

            try:
                event = self._imri_event_record(
                    result,
                    agn_properties,
                    disk_data,
                    damping_result,
                    rg_cm,
                    boundary,
                )

            except (disk_migration.MigrationError, MigrationProfileError) as exc:
                migration_error: MigrationErrorRecord = {
                    "object_index": damping_result["idx"],
                    "mass_msun": damping_result["mass_msun"],
                    "r_ini_rg": damping_result["final_semi_major_axis_cm"] / rg_cm,
                    "r_entry_rg": boundary.formal_entry_radius_cm / rg_cm,
                    "r_isco_rg": boundary.isco_radius_cm / rg_cm,
                    "band_entry_status": BAND_ENTRY_PROFILE_ERROR,
                    "error": str(exc),
                }
                result.migration_errors.append(migration_error)
                result.migration_evaluated_count += 1
                result.migration_error_count += 1
                result.profile_error_count += 1
                # vprint(
                #     "Migration error: "
                #     f"object_index={damping_result['idx']}, "
                #     f"mass_msun={damping_result['mass_msun']:.6e}, error={exc}"
                # )
                continue

            result.imri_events.append(event)
            if event["band_entry_status"] == BAND_ENTRY_INSIDE_ISCO:
                result.band_entry_inside_isco_count += 1
            else:
                result.migration_evaluated_count += 1
                if event["band_entry_status"] == BAND_ENTRY_HANDOFF:
                    result.entry_handoff_count += 1
                else:
                    result.entry_in_migration_domain_count += 1

                if event["migration_status"] == "trap_bound":
                    result.trap_bound_migration_count += 1
                    result.trap_bound_mass_msun += event["mass_msun"]
                elif event["migration_status"] == "outward_blocked":
                    result.outward_blocked_migration_count += 1
                    result.outward_blocked_mass_msun += event["mass_msun"]
                # vprint(
                #     "Trap-bound migration: "
                #     f"object_index={event['object_index']}, "
                #     f"mass_msun={event['mass_msun']:.6e}, "
                #     f"r_ini_rg={event['r_ini_rg']:.6e}, "
                #     f"r_entry_rg={event['r_entry_rg']:.6e}, "
                #     f"blocking_radius_rg={event['blocking_radius_rg']}"
                # )
                elif event["migration_status"] in ("finite", "already_at_target"):
                    result.finite_migration_count += 1
                else:
                    raise ValueError(
                        "unknown migration status: "
                        f"{event['migration_status']}"
                    )

            if event["bh_candidate"]:
                result.bh_candidate_count += 1
            if event["bh_origin"] == BH_ORIGIN_MIGRATION_FORMED:
                result.migration_formed_bh_count += 1
            if event["is_imri_candidate"] is True:
                result.imri_count += 1
                result.imri_mass_msun += event["mass_msun"]
                if event["is_lisa_detectable"] is True:
                    if event["snr_status"] != SNR_STATUS_EVALUATED:
                        raise RuntimeError("detectable event has a non-evaluated SNR status")
                    result.detector_evaluated_imri_count += 1
                    result.detectable_imri_count += 1
                    result.detectable_imri_mass_msun += event["mass_msun"]
                elif event["is_lisa_detectable"] is False:
                    result.detector_evaluated_imri_count += 1
                    result.no_detectable_imri_count += 1
                    if event["snr_status"] == SNR_STATUS_EVALUATED:
                        result.detector_below_threshold_imri_count += 1
                    elif event["snr_status"] == SNR_STATUS_NO_OVERLAP:
                        result.detector_no_overlap_imri_count += 1
                    else:
                        raise RuntimeError(
                            "physical nondetection has an unavailable SNR status"
                        )
                else:
                    result.detector_unavailable_imri_count += 1
                    if event["snr_status"] == SNR_STATUS_REQUIRED_INPUT_UNAVAILABLE:
                        result.detector_required_input_unavailable_imri_count += 1
                    elif event["snr_status"] == SNR_STATUS_MODEL_UNAVAILABLE:
                        result.detector_model_unavailable_imri_count += 1
                    elif event["snr_status"] == SNR_STATUS_NUMERICAL_FAILURE:
                        result.detector_numerical_failure_imri_count += 1
                    else:
                        raise RuntimeError(
                            "detector-unavailable event has an unrecognized SNR status"
                        )
            elif event["is_imri_candidate"] is False:
                result.no_imri_count += 1

        if result.band_entry_inside_isco_count > 0 or result.profile_error_count > 0:
            result.imri_rate_per_myr = np.nan
            result.imri_mass_rate_msun_per_myr = np.nan
        else:
            result.imri_rate_per_myr = result.imri_count / self.duty_cycle_myr
            result.imri_mass_rate_msun_per_myr = (
                result.imri_mass_msun / self.duty_cycle_myr
            )

        detector_rate_unavailable = (
            result.band_entry_inside_isco_count > 0
            or result.profile_error_count > 0
            or result.detector_unavailable_imri_count > 0
        )
        if detector_rate_unavailable:
            result.detectable_imri_rate_per_myr = np.nan
            result.detectable_imri_mass_rate_msun_per_myr = np.nan
        else:
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
        boundary: gw_helpers.CircularBandBoundary,
    ) -> IMRIEventRecord:
        redshift = self.config.redshift
        redshift_model = RedshiftEvolution(self.config.hubble_constant_per_myr)
        migrator_mass_msun = damping_result["mass_msun"]
        r_ini_cm = damping_result["final_semi_major_axis_cm"]
        total_mass_msun = result.m_smbh_msun + migrator_mass_msun
        formation_time_myr = damping_result["formation_time_myr"]
        damping_time_myr = damping_result["damping_time_myr"]
        r_entry_formal_cm = boundary.formal_entry_radius_cm
        r_isco_cm = boundary.isco_radius_cm
        disk_inner_radius_cm = float(disk_data["inner_radius_cm"])
        migration_min_radius_cm = np.nan
        handoff_radius_cm = np.nan
        r_entry_cm = np.nan
        migration_evaluated = False
        migration_status = "not_applicable"
        blocking_radius_cm: float | None = None

        if boundary.status == BAND_ENTRY_INSIDE_ISCO:
            band_entry_status = BAND_ENTRY_INSIDE_ISCO
            event_decision = classify_unsupported_band_entry(
                object_status=damping_result["status"],
                embed_time_myr=damping_result["final_time_myr"],
                r_ini_cm=r_ini_cm,
            )
            detect_decision = unsupported_band_entry_decision(
                primary_mass_msun=result.m_smbh_msun,
                current_radius_cm=r_ini_cm,
                mission_duration_years=self.config.lisa_mission_duration_years,
            )
        else:
            torque_profile = self.engine.get_torque_profile(
                disk_data,
                agn_properties.smbh_mass_msun,
                migrator_mass_msun,
            )
            migration_radius_cm = np.asarray(
                torque_profile.radius_cm,
                dtype=float,
            )
            route = resolve_band_entry_route(
                boundary,
                disk_inner_radius_cm,
                migration_radius_cm,
            )
            band_entry_status = route.status
            migration_min_radius_cm = route.migration_min_radius_cm
            handoff_radius_cm = route.handoff_radius_cm
            r_entry_cm = r_entry_formal_cm
            migration_evaluated = True

            migration_outcomes: dict[
                tuple[float, float], disk_migration.MigrationOutcome
            ] = {}

            def migration_outcome_call(
                start_radius_cm: float, target_radius_cm: float
            ) -> disk_migration.MigrationOutcome:
                key = (float(start_radius_cm), float(target_radius_cm))
                if key not in migration_outcomes:
                    migration_outcomes[key] = (
                        self.torque_migration_outcome_to_radius(
                            agn_properties,
                            disk_data,
                            damping_result["mass_g"],
                            start_radius_cm,
                            target_radius_cm,
                            torque_profile=torque_profile,
                        )
                    )
                return migration_outcomes[key]

            def migration_time_myr_call(
                start_radius_cm: float, target_radius_cm: float
            ) -> float:
                return migration_outcome_call(
                    start_radius_cm, target_radius_cm
                ).time_myr

            initial_migration_outcome = migration_outcome_call(
                r_ini_cm,
                handoff_radius_cm,
            )
            migration_status = initial_migration_outcome.status
            blocking_radius_cm = initial_migration_outcome.blocking_radius_cm

            def gw_time_to_entry_myr_call(
                radius_cm: float,
                absolute_time_myr: float,
            ) -> float:
                current_redshift = redshift_model.redshift_evlove_with_time(
                    redshift,
                    absolute_time_myr,
                )
                entry_outcome = solve_time_to_observed_lisa_entry(
                    primary_mass_msun=result.m_smbh_msun,
                    secondary_mass_msun=migrator_mass_msun,
                    initial_radius_cm=radius_cm,
                    initial_redshift=redshift,
                    start_time_myr=absolute_time_myr,
                    redshift_model=redshift_model,
                    min_observed_frequency_hz=(
                        self.config.lisa_min_gw_frequency_hz
                    ),
                    eccentricity=self.config.gw_eccentricity,
                    max_elapsed_time_myr=(
                        redshift_model.remaining_time_to_zero_redshift_myr(
                            current_redshift
                        )
                    ),
                )
                return entry_outcome.time_to_entry_myr

            event_decision = classify_event_rate(
                object_status=damping_result["status"],
                formation_time_myr=formation_time_myr,
                damping_time_myr=damping_time_myr,
                embed_time_myr=damping_result["final_time_myr"],
                r_ini_cm=r_ini_cm,
                r_gas_target_cm=handoff_radius_cm,
                r_entry_cm=r_entry_cm,
                handoff_required=band_entry_status == BAND_ENTRY_HANDOFF,
                migration_time_myr_call=migration_time_myr_call,
                gw_time_to_entry_myr_call=gw_time_to_entry_myr_call,
                duty_cycle_myr=self.config.duty_cycle_myr,
                num_radius_samples=GAS_FINAL_RADIUS_SAMPLES,
            )
            if event_decision.is_imri_candidate is None:
                raise RuntimeError("valid band entry was not classified")
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
                cosmic_merge_smbh_mass_factor=(
                    self.config.cosmic_merge_smbh_mass_factor
                ),
                cosmic_merge_radius_factor=self.config.cosmic_merge_radius_factor,
                snr_threshold=self.config.snr_threshold,
                mission_duration_years=self.config.lisa_mission_duration_years,
                snr_window_placement=self.config.snr_window_placement,
                detector_hubble_constant_km_s_mpc=(
                    self.config.detector_hubble_constant_km_s_mpc
                ),
                detector_omega_m=self.config.detector_omega_m,
            )
        # vprint(
        #     "total_migration_time is "
        #     f"{event_decision.migration_time_to_entry_myr:.3e} myr"
        # )
        # vprint(f"total_gw_time is {event_decision.gw_time_to_entry_myr:.3e} myr")
        # vprint(
        #     f"stop reason is {event_decision.gas_stop_reason}, "
        #     f"r_gas_fin is {event_decision.r_gas_fin_cm:.3e} cm"
        # )

        redshift_at_gas_fin = redshift_model.redshift_evlove_with_time(
            redshift, event_decision.gas_final_time_myr
        )

        f_ini_source_hz = _safe_gw_frequency_hz(total_mass_msun, r_ini_cm)
        f_gas_fin_source_hz = _safe_gw_frequency_hz(
            total_mass_msun, event_decision.r_gas_fin_cm
        )
        f_entry_source_hz = _safe_gw_frequency_hz(total_mass_msun, r_entry_cm)
        f_ini_observed_hz = _safe_observed_gw_frequency_hz(f_ini_source_hz, redshift)
        f_gas_fin_observed_hz = _safe_observed_gw_frequency_hz(
            f_gas_fin_source_hz, redshift_at_gas_fin
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

        event_id = (
            f"{self.config.disk_model}:z={redshift:.12g}:"
            f"m_smbh={result.m_smbh_msun:.12g}:m_nsc={result.m_nsc_msun:.12g}:"
            f"repeat={result.repeat_idx}:object={damping_result['idx']}"
        )
        # One definition of a counted candidate, shared with the averaged CSV:
        # the gas stage produced a candidate *and* the object actually reaches
        # the LISA band in finite time. Applying it here keeps the per-event
        # table, the in-model tally and the headline count in agreement.
        counted_imri_candidate = is_physical_imri_entry(
            event_decision.is_imri_candidate,
            float(detect_decision.detection_time_myr),
        )
        counted_event_rate_class = (
            event_decision.event_rate_class
            if counted_imri_candidate is None or counted_imri_candidate
            else EVENT_RATE_NO_IMRI
        )
        intrinsic_rate_contribution_per_myr = (
            1.0 / self.duty_cycle_myr
            if counted_imri_candidate is True
            else 0.0
            if counted_imri_candidate is False
            else np.nan
        )
        detected_rate_contribution_per_myr = (
            1.0 / self.duty_cycle_myr
            if detect_decision.is_lisa_detectable is True
            else 0.0
            if (
                counted_imri_candidate is True
                and detect_decision.is_lisa_detectable is False
            )
            else np.nan
        )

        return {
            "event_id": event_id,
            "m_smbh_msun": result.m_smbh_msun,
            "m_nsc_msun": result.m_nsc_msun,
            "repeat": result.repeat_idx,
            "disk_model": self.config.disk_model,
            "redshift": redshift,
            "detection_redshift": detect_decision.detection_redshift,
            "object_index": damping_result["idx"],
            "mass_msun": migrator_mass_msun,
            "mass_ratio_secondary_to_primary": (
                migrator_mass_msun / detect_decision.final_smbh_mass_msun
            ),
            "bh_origin": event_decision.bh_origin,
            "band_entry_status": band_entry_status,
            "event_rate_class": counted_event_rate_class,
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
            "r_entry_formal_cm": r_entry_formal_cm,
            "r_entry_formal_rg": r_entry_formal_cm / rg_cm,
            "r_isco_cm": r_isco_cm,
            "r_isco_rg": r_isco_cm / rg_cm,
            "r_disk_inner_cm": disk_inner_radius_cm,
            "r_disk_inner_rg": disk_inner_radius_cm / rg_cm,
            "r_migration_min_cm": migration_min_radius_cm,
            "r_migration_min_rg": migration_min_radius_cm / rg_cm,
            "r_handoff_cm": handoff_radius_cm,
            "r_handoff_rg": handoff_radius_cm / rg_cm,
            "r_det_entry_cm": detect_decision.final_entry_radius_cm,
            "r_det_entry_rg": detect_decision.final_entry_radius_cm / rg_cm,
            "source_entry_frequency_hz": boundary.source_entry_frequency_hz,
            "f_ini_src_hz": f_ini_source_hz,
            "f_ini_obs_hz": f_ini_observed_hz,
            "f_gas_fin_src_hz": f_gas_fin_source_hz,
            "f_gas_fin_obs_hz": f_gas_fin_observed_hz,
            "f_entry_src_hz": f_entry_output_source_hz,
            "f_entry_obs_hz": f_entry_output_observed_hz,
            "snr_input_observed_frequency_hz": (
                detect_decision.observed_frequency_hz
            ),
            "gas_elapsed_time_myr": event_decision.gas_elapsed_time_myr,
            "gas_final_time_myr": event_decision.gas_final_time_myr,
            "migration_time_to_entry_myr": event_decision.migration_time_to_entry_myr,
            "migration_time_to_handoff_myr": (
                event_decision.migration_time_to_gas_target_myr
            ),
            "gw_time_handoff_to_entry_myr": (
                event_decision.gw_time_gas_target_to_entry_myr
            ),
            "gas_stop_reason": event_decision.gas_stop_reason,
            "migration_evaluated": migration_evaluated,
            "migration_status": migration_status,
            "blocking_radius_rg": (
                blocking_radius_cm / rg_cm if blocking_radius_cm is not None else None
            ),
            "gw_time_to_entry_myr": event_decision.gw_time_to_entry_myr,
            "gw_time_after_gas_myr": detect_decision.gw_time_after_gas_myr,
            "detection_time_myr": detect_decision.detection_time_myr,
            "final_smbh_mass_msun": detect_decision.final_smbh_mass_msun,
            "cosmic_merge_applied": detect_decision.cosmic_merge_applied,
            "bh_candidate": event_decision.bh_candidate,
            "is_imri_candidate": counted_imri_candidate,
            "is_lisa_detectable": detect_decision.is_lisa_detectable,
            "lisa_snr": detect_decision.lisa_snr,
            "snr_status": detect_decision.snr_status,
            "chirp_mass_source_msun": detect_decision.chirp_mass_source_msun,
            "chirp_mass_detector_msun": detect_decision.chirp_mass_detector_msun,
            "luminosity_distance_mpc": detect_decision.luminosity_distance_mpc,
            "snr_f_low_hz": detect_decision.snr_f_low_hz,
            "snr_f_high_hz": detect_decision.snr_f_high_hz,
            "f_isco_observer_hz": detect_decision.f_isco_observer_hz,
            "mission_duration_years": detect_decision.mission_duration_years,
            "snr_observation_duration_years": (
                detect_decision.snr_observation_duration_years
            ),
            "snr_observation_case": detect_decision.snr_observation_case,
            "snr_window_placement": detect_decision.snr_window_placement,
            "snr_threshold": self.config.snr_threshold,
            "monte_carlo_event_weight": 1.0,
            "intrinsic_rate_contribution_per_myr": (
                intrinsic_rate_contribution_per_myr
            ),
            "detected_rate_contribution_per_myr": (
                detected_rate_contribution_per_myr
            ),
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
        torque_profile: TorqueProfile | None = None,
    ) -> disk_migration.MigrationOutcome:
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
        if torque_profile is None:
            torque_profile = self.engine.get_torque_profile(
                disk_data,
                agn_properties.smbh_mass_msun,
                migrator_mass_msun,
            )
        torque_cgs = self._migration_torque_cgs(torque_profile)
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
    ) -> FloatArray:
        """Convert the already-composed production torque from SI to CGS."""
        return torque_si_to_cgs(torque_profile.gamma_total_si)
