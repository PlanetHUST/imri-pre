"""LISA detectability classification for IMRI event candidates."""

from dataclasses import dataclass

import numpy as np

import physical_process.gw_frequency_radius as gw_helpers

from .cosmic_merge import CosmicMergeModel
from .lisa_snr import evaluate_lisa_inspiral_snr
from .redshift_evolve import RedshiftEvolution

DETECTION_CLASS_DETECTABLE = "imri data"
DETECTION_CLASS_NOT_DETECTABLE = "non-detectable"
DETECTION_CLASS_NOT_EVALUATED = "not evaluated"


@dataclass(frozen=True)
class DetectRateDecision:
    """Classification result for the physical detectability stage."""

    detect_rate_class: str
    is_lisa_detectable: bool | None
    snr_proxy: float
    source_frequency_hz: float
    observed_frequency_hz: float
    detection_time_myr: float
    detection_redshift: float
    final_smbh_mass_msun: float
    final_radius_cm: float
    final_entry_radius_cm: float
    gw_time_after_gas_myr: float
    cosmic_merge_applied: bool
    detect_rate_reason: str
    lisa_snr: float = np.nan
    snr_status: str = "not_evaluated"
    chirp_mass_source_msun: float = np.nan
    chirp_mass_detector_msun: float = np.nan
    luminosity_distance_mpc: float = np.nan
    snr_f_low_hz: float = np.nan
    snr_f_high_hz: float = np.nan
    f_isco_observer_hz: float = np.nan
    mission_duration_years: float = np.nan
    snr_observation_duration_years: float = np.nan
    snr_observation_case: str = "not_evaluated"


def unsupported_band_entry_decision(
    *,
    primary_mass_msun: float,
    current_radius_cm: float,
    mission_duration_years: float = np.nan,
) -> DetectRateDecision:
    """Return an explicit unevaluated detector result for entry inside ISCO."""

    return DetectRateDecision(
        detect_rate_class=DETECTION_CLASS_NOT_EVALUATED,
        is_lisa_detectable=None,
        snr_proxy=np.nan,
        source_frequency_hz=np.nan,
        observed_frequency_hz=np.nan,
        detection_time_myr=np.nan,
        detection_redshift=np.nan,
        final_smbh_mass_msun=primary_mass_msun,
        final_radius_cm=current_radius_cm,
        final_entry_radius_cm=np.nan,
        gw_time_after_gas_myr=np.nan,
        cosmic_merge_applied=False,
        detect_rate_reason="band_entry_inside_isco",
        mission_duration_years=mission_duration_years,
    )


def classify_detectability(
    *,
    is_imri_candidate: bool,
    primary_mass_msun: float,
    secondary_mass_msun: float,
    r_gas_fin_cm: float,
    gas_final_time_myr: float,
    initial_redshift: float,
    hubble_constant_per_myr: float,
    min_observed_frequency_hz: float,
    max_observed_frequency_hz: float,
    eccentricity: float,
    cosmic_merge_time_myr: float,
    cosmic_merge_smbh_mass_factor: float,
    cosmic_merge_radius_factor: float,
    snr_threshold: float,
    mission_duration_years: float,
    detector_hubble_constant_km_s_mpc: float,
    detector_omega_m: float,
) -> DetectRateDecision:
    """Classify one IMRI candidate after gas delivery and cosmic evolution."""
    if not is_imri_candidate:
        return DetectRateDecision(
            detect_rate_class=DETECTION_CLASS_NOT_EVALUATED,
            is_lisa_detectable=None,
            snr_proxy=np.nan,
            source_frequency_hz=np.nan,
            observed_frequency_hz=np.nan,
            detection_time_myr=np.inf,
            detection_redshift=np.nan,
            final_smbh_mass_msun=primary_mass_msun,
            final_radius_cm=r_gas_fin_cm,
            final_entry_radius_cm=np.nan,
            gw_time_after_gas_myr=np.inf,
            cosmic_merge_applied=False,
            detect_rate_reason="not_imri_candidate",
            mission_duration_years=mission_duration_years,
        )

    if not np.isfinite(r_gas_fin_cm) or r_gas_fin_cm <= 0.0:
        return DetectRateDecision(
            detect_rate_class=DETECTION_CLASS_NOT_EVALUATED,
            is_lisa_detectable=None,
            snr_proxy=np.nan,
            source_frequency_hz=np.nan,
            observed_frequency_hz=np.nan,
            detection_time_myr=np.inf,
            detection_redshift=np.nan,
            final_smbh_mass_msun=primary_mass_msun,
            final_radius_cm=r_gas_fin_cm,
            final_entry_radius_cm=np.nan,
            gw_time_after_gas_myr=np.inf,
            cosmic_merge_applied=False,
            detect_rate_reason="required_input_unavailable_invalid_gas_final_radius",
            mission_duration_years=mission_duration_years,
            snr_status="required_input_unavailable",
        )

    redshift_model = RedshiftEvolution(hubble_constant_per_myr)
    merge_model = CosmicMergeModel(
        merge_time_myr=cosmic_merge_time_myr,
        smbh_mass_growth_factor=cosmic_merge_smbh_mass_factor,
        radius_growth_factor=cosmic_merge_radius_factor,
    )
    current_time_myr = float(max(0.0, gas_final_time_myr))
    current_redshift = redshift_model.redshift_evlove_with_time(
        initial_redshift,
        current_time_myr,
    )
    current_smbh_mass_msun = float(primary_mass_msun)
    current_radius_cm = float(r_gas_fin_cm)
    cosmic_merge_applied = False

    entry_radius_cm = gw_helpers.lisa_entry_radius_cm(
        current_smbh_mass_msun,
        secondary_mass_msun,
        min_observed_frequency_hz,
        current_redshift,
    )
    if not np.isfinite(entry_radius_cm):
        return DetectRateDecision(
            detect_rate_class=DETECTION_CLASS_NOT_EVALUATED,
            is_lisa_detectable=None,
            snr_proxy=np.nan,
            source_frequency_hz=np.nan,
            observed_frequency_hz=np.nan,
            detection_time_myr=np.inf,
            detection_redshift=current_redshift,
            final_smbh_mass_msun=current_smbh_mass_msun,
            final_radius_cm=current_radius_cm,
            final_entry_radius_cm=entry_radius_cm,
            gw_time_after_gas_myr=np.inf,
            cosmic_merge_applied=cosmic_merge_applied,
            detect_rate_reason="required_input_unavailable_invalid_frequency",
            mission_duration_years=mission_duration_years,
            snr_status="required_input_unavailable",
        )

    gw_time_after_gas_myr = gw_helpers.peters_time_to_lisa_band_myr(
        current_smbh_mass_msun,
        secondary_mass_msun,
        current_radius_cm,
        min_observed_frequency_hz,
        eccentricity,
        current_redshift,
    )
    if not np.isfinite(gw_time_after_gas_myr):
        return DetectRateDecision(
            detect_rate_class=DETECTION_CLASS_NOT_EVALUATED,
            is_lisa_detectable=None,
            snr_proxy=np.nan,
            source_frequency_hz=np.nan,
            observed_frequency_hz=np.nan,
            detection_time_myr=np.inf,
            detection_redshift=current_redshift,
            final_smbh_mass_msun=current_smbh_mass_msun,
            final_radius_cm=current_radius_cm,
            final_entry_radius_cm=entry_radius_cm,
            gw_time_after_gas_myr=gw_time_after_gas_myr,
            cosmic_merge_applied=cosmic_merge_applied,
            detect_rate_reason="snr_model_unavailable_gw_evolution",
            mission_duration_years=mission_duration_years,
            snr_status="model_unavailable",
        )

    remaining_cosmic_time_myr = redshift_model.remaining_time_to_zero_redshift_myr(
        current_redshift,
    )
    time_to_merge_myr = merge_model.merge_time_myr
    if merge_model.is_active and time_to_merge_myr < gw_time_after_gas_myr:
        if time_to_merge_myr >= remaining_cosmic_time_myr:
            return _not_detected_due_to_time(
                primary_mass_msun=current_smbh_mass_msun,
                radius_cm=current_radius_cm,
                entry_radius_cm=entry_radius_cm,
                gw_time_after_gas_myr=gw_time_after_gas_myr,
                detection_redshift=current_redshift,
                cosmic_merge_applied=False,
                reason="misses_cosmic_time_before_merge",
                mission_duration_years=mission_duration_years,
            )
        radius_at_merge_cm = gw_helpers.peters_radius_after_time_cm(
            current_smbh_mass_msun,
            secondary_mass_msun,
            current_radius_cm,
            time_to_merge_myr,
            eccentricity,
            entry_radius_cm,
        )
        current_time_myr += time_to_merge_myr
        current_redshift = redshift_model.redshift_evlove_with_time(
            initial_redshift,
            current_time_myr,
        )
        current_smbh_mass_msun = merge_model.post_merge_smbh_mass_msun(
            current_smbh_mass_msun,
        )
        current_radius_cm = merge_model.post_merge_radius_cm(radius_at_merge_cm)
        cosmic_merge_applied = True
        entry_radius_cm = gw_helpers.lisa_entry_radius_cm(
            current_smbh_mass_msun,
            secondary_mass_msun,
            min_observed_frequency_hz,
            current_redshift,
        )
        gw_time_after_gas_myr = gw_helpers.peters_time_to_lisa_band_myr(
            current_smbh_mass_msun,
            secondary_mass_msun,
            current_radius_cm,
            min_observed_frequency_hz,
            eccentricity,
            current_redshift,
        )
        remaining_cosmic_time_myr = redshift_model.remaining_time_to_zero_redshift_myr(
            current_redshift
        )

    if gw_time_after_gas_myr > remaining_cosmic_time_myr:
        return _not_detected_due_to_time(
            primary_mass_msun=current_smbh_mass_msun,
            radius_cm=current_radius_cm,
            entry_radius_cm=entry_radius_cm,
            gw_time_after_gas_myr=gw_time_after_gas_myr,
            detection_redshift=current_redshift,
            cosmic_merge_applied=cosmic_merge_applied,
            reason="misses_cosmic_time_budget",
            mission_duration_years=mission_duration_years,
        )

    detection_time_myr = current_time_myr + gw_time_after_gas_myr
    detection_redshift = redshift_model.redshift_evlove_with_time(
        initial_redshift,
        detection_time_myr,
    )
    final_entry_radius_cm = gw_helpers.lisa_entry_radius_cm(
        current_smbh_mass_msun,
        secondary_mass_msun,
        min_observed_frequency_hz,
        detection_redshift,
    )
    final_radius_cm = (
        current_radius_cm
        if current_radius_cm <= final_entry_radius_cm
        else final_entry_radius_cm
    )
    source_frequency_hz = gw_helpers.gw_frequency_hz(
        current_smbh_mass_msun + secondary_mass_msun,
        final_radius_cm,
    )
    observed_frequency_hz = gw_helpers.observed_frame_gw_frequency_hz(
        source_frequency_hz,
        detection_redshift,
    )
    snr_result = evaluate_lisa_inspiral_snr(
        m1_source_msun=current_smbh_mass_msun,
        m2_source_msun=secondary_mass_msun,
        redshift=detection_redshift,
        current_observed_frequency_hz=observed_frequency_hz,
        mission_duration_years=mission_duration_years,
        detector_min_frequency_hz=min_observed_frequency_hz,
        detector_max_frequency_hz=max_observed_frequency_hz,
        snr_threshold=snr_threshold,
        hubble_constant_km_s_mpc=detector_hubble_constant_km_s_mpc,
        omega_m=detector_omega_m,
        eccentricity=eccentricity,
    )
    detect_rate_class = (
        DETECTION_CLASS_DETECTABLE
        if snr_result.is_detectable is True
        else DETECTION_CLASS_NOT_DETECTABLE
        if snr_result.is_detectable is False
        else DETECTION_CLASS_NOT_EVALUATED
    )
    return DetectRateDecision(
        detect_rate_class=detect_rate_class,
        is_lisa_detectable=snr_result.is_detectable,
        # Deprecated compatibility alias; the heuristic proxy is no longer run.
        snr_proxy=snr_result.lisa_snr,
        source_frequency_hz=source_frequency_hz,
        observed_frequency_hz=observed_frequency_hz,
        detection_time_myr=detection_time_myr,
        detection_redshift=detection_redshift,
        final_smbh_mass_msun=current_smbh_mass_msun,
        final_radius_cm=final_radius_cm,
        final_entry_radius_cm=final_entry_radius_cm,
        gw_time_after_gas_myr=gw_time_after_gas_myr,
        cosmic_merge_applied=cosmic_merge_applied,
        detect_rate_reason=snr_result.reason,
        lisa_snr=snr_result.lisa_snr,
        snr_status=snr_result.status,
        chirp_mass_source_msun=snr_result.chirp_mass_source_msun,
        chirp_mass_detector_msun=snr_result.chirp_mass_detector_msun,
        luminosity_distance_mpc=snr_result.luminosity_distance_mpc,
        snr_f_low_hz=snr_result.f_low_hz,
        snr_f_high_hz=snr_result.f_high_hz,
        f_isco_observer_hz=snr_result.f_isco_observer_hz,
        mission_duration_years=snr_result.mission_duration_years,
        snr_observation_duration_years=snr_result.observation_duration_years,
        snr_observation_case=snr_result.observation_case,
    )


def _not_detected_due_to_time(
    *,
    primary_mass_msun: float,
    radius_cm: float,
    entry_radius_cm: float,
    gw_time_after_gas_myr: float,
    detection_redshift: float,
    cosmic_merge_applied: bool,
    reason: str,
    mission_duration_years: float,
) -> DetectRateDecision:
    return DetectRateDecision(
        detect_rate_class=DETECTION_CLASS_NOT_DETECTABLE,
        is_lisa_detectable=False,
        snr_proxy=0.0,
        source_frequency_hz=np.nan,
        observed_frequency_hz=np.nan,
        detection_time_myr=np.inf,
        detection_redshift=detection_redshift,
        final_smbh_mass_msun=primary_mass_msun,
        final_radius_cm=radius_cm,
        final_entry_radius_cm=entry_radius_cm,
        gw_time_after_gas_myr=gw_time_after_gas_myr,
        cosmic_merge_applied=cosmic_merge_applied,
        detect_rate_reason=f"no_lisa_frequency_overlap:{reason}",
        lisa_snr=0.0,
        snr_status="no_physical_lisa_overlap",
        mission_duration_years=mission_duration_years,
        snr_observation_duration_years=0.0,
        snr_observation_case="no_lisa_frequency_overlap",
    )
