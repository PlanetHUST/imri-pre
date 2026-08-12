"""Simple LISA detectability classification for IMRI event candidates."""

from dataclasses import dataclass

import numpy as np

import physical_process.gw_frequency_radius as gw_helpers

from .cosmic_merge import CosmicMergeModel
from .redshift_evolve import RedshiftEvolution

DETECTION_CLASS_DETECTABLE = "imri data"
DETECTION_CLASS_NOT_DETECTABLE = "non-detectable"


@dataclass(frozen=True)
class DetectRateDecision:
    """Classification result for the simplified detectability stage."""

    detect_rate_class: str
    is_lisa_detectable: bool
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


def single_frequency_snr_proxy(
    *,
    observed_frequency_hz: float,
    min_observed_frequency_hz: float,
    max_observed_frequency_hz: float,
    snr_threshold: float,
) -> float:
    """Return a minimal SNR proxy for the current single-frequency model."""
    in_band = (
        np.isfinite(observed_frequency_hz)
        and min_observed_frequency_hz
        <= observed_frequency_hz
        <= max_observed_frequency_hz
    )
    return float(snr_threshold if in_band else 0.0)


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
) -> DetectRateDecision:
    """Classify one IMRI candidate after gas delivery and cosmic evolution."""
    if not is_imri_candidate:
        return DetectRateDecision(
            detect_rate_class=DETECTION_CLASS_NOT_DETECTABLE,
            is_lisa_detectable=False,
            snr_proxy=0.0,
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
        )

    if not np.isfinite(r_gas_fin_cm) or r_gas_fin_cm <= 0.0:
        return DetectRateDecision(
            detect_rate_class=DETECTION_CLASS_NOT_DETECTABLE,
            is_lisa_detectable=False,
            snr_proxy=0.0,
            source_frequency_hz=np.nan,
            observed_frequency_hz=np.nan,
            detection_time_myr=np.inf,
            detection_redshift=np.nan,
            final_smbh_mass_msun=primary_mass_msun,
            final_radius_cm=r_gas_fin_cm,
            final_entry_radius_cm=np.nan,
            gw_time_after_gas_myr=np.inf,
            cosmic_merge_applied=False,
            detect_rate_reason="invalid_gas_final_radius",
        )

    redshift_model = RedshiftEvolution(hubble_constant_per_myr)
    merge_model = CosmicMergeModel(
        merge_time_myr=cosmic_merge_time_myr,
        smbh_mass_growth_factor=cosmic_merge_smbh_mass_factor,
        radius_growth_factor=cosmic_merge_radius_factor,
    )
    current_time_myr = float(max(0.0, gas_final_time_myr))
    current_redshift = redshift_model.redshift_after_time(
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
            detect_rate_class=DETECTION_CLASS_NOT_DETECTABLE,
            is_lisa_detectable=False,
            snr_proxy=0.0,
            source_frequency_hz=np.nan,
            observed_frequency_hz=np.nan,
            detection_time_myr=np.inf,
            detection_redshift=current_redshift,
            final_smbh_mass_msun=current_smbh_mass_msun,
            final_radius_cm=current_radius_cm,
            final_entry_radius_cm=entry_radius_cm,
            gw_time_after_gas_myr=np.inf,
            cosmic_merge_applied=cosmic_merge_applied,
            detect_rate_reason="invalid_frequency",
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
            detect_rate_class=DETECTION_CLASS_NOT_DETECTABLE,
            is_lisa_detectable=False,
            snr_proxy=0.0,
            source_frequency_hz=np.nan,
            observed_frequency_hz=np.nan,
            detection_time_myr=np.inf,
            detection_redshift=current_redshift,
            final_smbh_mass_msun=current_smbh_mass_msun,
            final_radius_cm=current_radius_cm,
            final_entry_radius_cm=entry_radius_cm,
            gw_time_after_gas_myr=gw_time_after_gas_myr,
            cosmic_merge_applied=cosmic_merge_applied,
            detect_rate_reason="gw_evolution_unavailable",
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
        current_redshift = redshift_model.redshift_after_time(
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
        )

    detection_time_myr = current_time_myr + gw_time_after_gas_myr
    detection_redshift = redshift_model.redshift_after_time(
        initial_redshift,
        detection_time_myr,
    )
    final_entry_radius_cm = gw_helpers.lisa_entry_radius_cm(
        current_smbh_mass_msun,
        secondary_mass_msun,
        min_observed_frequency_hz,
        detection_redshift,
    )
    final_radius_cm = final_entry_radius_cm
    source_frequency_hz = gw_helpers.gw_frequency_hz(
        current_smbh_mass_msun + secondary_mass_msun,
        final_radius_cm,
    )
    observed_frequency_hz = gw_helpers.observed_frame_gw_frequency_hz(
        source_frequency_hz,
        detection_redshift,
    )
    snr_proxy = single_frequency_snr_proxy(
        observed_frequency_hz=observed_frequency_hz,
        min_observed_frequency_hz=min_observed_frequency_hz,
        max_observed_frequency_hz=max_observed_frequency_hz,
        snr_threshold=snr_threshold,
    )
    if snr_proxy >= snr_threshold:
        return DetectRateDecision(
            detect_rate_class=DETECTION_CLASS_DETECTABLE,
            is_lisa_detectable=True,
            snr_proxy=snr_proxy,
            source_frequency_hz=source_frequency_hz,
            observed_frequency_hz=observed_frequency_hz,
            detection_time_myr=detection_time_myr,
            detection_redshift=detection_redshift,
            final_smbh_mass_msun=current_smbh_mass_msun,
            final_radius_cm=final_radius_cm,
            final_entry_radius_cm=final_entry_radius_cm,
            gw_time_after_gas_myr=gw_time_after_gas_myr,
            cosmic_merge_applied=cosmic_merge_applied,
            detect_rate_reason=DETECTION_CLASS_DETECTABLE,
        )

    return DetectRateDecision(
        detect_rate_class=DETECTION_CLASS_NOT_DETECTABLE,
        is_lisa_detectable=False,
        snr_proxy=snr_proxy,
        source_frequency_hz=source_frequency_hz,
        observed_frequency_hz=observed_frequency_hz,
        detection_time_myr=detection_time_myr,
        detection_redshift=detection_redshift,
        final_smbh_mass_msun=current_smbh_mass_msun,
        final_radius_cm=final_radius_cm,
        final_entry_radius_cm=final_entry_radius_cm,
        gw_time_after_gas_myr=gw_time_after_gas_myr,
        cosmic_merge_applied=cosmic_merge_applied,
        detect_rate_reason="outside_lisa_band",
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
        detect_rate_reason=reason,
    )
