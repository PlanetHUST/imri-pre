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


@dataclass(frozen=True)
class ObservedLisaEntryOutcome:
    """Time and orbital state from one moving-boundary GW evolution segment."""

    status: str
    time_to_entry_myr: float
    evaluated_elapsed_time_myr: float
    state_time_myr: float
    state_redshift: float
    state_radius_cm: float
    entry_radius_cm: float
    source_frequency_hz: float
    observed_frequency_hz: float


def solve_time_to_observed_lisa_entry(
    *,
    primary_mass_msun: float,
    secondary_mass_msun: float,
    initial_radius_cm: float,
    initial_redshift: float,
    start_time_myr: float,
    redshift_model: RedshiftEvolution,
    min_observed_frequency_hz: float,
    eccentricity: float,
    max_elapsed_time_myr: float,
) -> ObservedLisaEntryOutcome:
    """Solve the first Peters-evolved crossing of the observed LISA boundary."""

    if not np.isfinite(start_time_myr) or start_time_myr < 0.0:
        raise ValueError("GW segment start time must be finite and nonnegative")
    if not np.isfinite(max_elapsed_time_myr) or max_elapsed_time_myr < 0.0:
        raise ValueError("GW segment time limit must be finite and nonnegative")
    if not np.isfinite(initial_radius_cm) or initial_radius_cm <= 0.0:
        raise ValueError("GW segment initial radius must be finite and positive")

    start_redshift = redshift_model.redshift_evlove_with_time(
        initial_redshift,
        start_time_myr,
    )
    start_boundary = gw_helpers.circular_band_boundary(
        primary_mass_msun,
        secondary_mass_msun,
        min_observed_frequency_hz,
        start_redshift,
    )
    isco_radius_cm = start_boundary.isco_radius_cm
    start_source_frequency_hz = gw_helpers.gw_frequency_hz(
        primary_mass_msun + secondary_mass_msun,
        initial_radius_cm,
    )
    start_observed_frequency_hz = gw_helpers.observed_frame_gw_frequency_hz(
        start_source_frequency_hz,
        start_redshift,
    )

    if initial_radius_cm <= isco_radius_cm:
        return ObservedLisaEntryOutcome(
            status="invalid_orbit_at_or_inside_isco",
            time_to_entry_myr=np.inf,
            evaluated_elapsed_time_myr=0.0,
            state_time_myr=start_time_myr,
            state_redshift=start_redshift,
            state_radius_cm=initial_radius_cm,
            entry_radius_cm=start_boundary.formal_entry_radius_cm,
            source_frequency_hz=start_source_frequency_hz,
            observed_frequency_hz=start_observed_frequency_hz,
        )

    if start_observed_frequency_hz >= min_observed_frequency_hz:
        return ObservedLisaEntryOutcome(
            status="already_inside_band",
            time_to_entry_myr=0.0,
            evaluated_elapsed_time_myr=0.0,
            state_time_myr=start_time_myr,
            state_redshift=start_redshift,
            state_radius_cm=initial_radius_cm,
            entry_radius_cm=start_boundary.formal_entry_radius_cm,
            source_frequency_hz=start_source_frequency_hz,
            observed_frequency_hz=start_observed_frequency_hz,
        )

    start_merger_time_myr = gw_helpers.peters_merger_time_myr(
        primary_mass_msun,
        secondary_mass_msun,
        initial_radius_cm,
        eccentricity,
    )
    isco_merger_time_myr = gw_helpers.peters_merger_time_myr(
        primary_mass_msun,
        secondary_mass_msun,
        isco_radius_cm,
        eccentricity,
    )
    time_to_isco_myr = max(0.0, start_merger_time_myr - isco_merger_time_myr)
    remaining_cosmic_time_myr = redshift_model.remaining_time_to_zero_redshift_myr(
        start_redshift
    )
    elapsed_limit_myr = min(
        max_elapsed_time_myr,
        remaining_cosmic_time_myr,
        time_to_isco_myr,
    )

    def fixed_boundary_time_myr(elapsed_time_myr: float) -> float:
        redshift = redshift_model.redshift_evlove_with_time(
            initial_redshift,
            start_time_myr + elapsed_time_myr,
        )
        return gw_helpers.peters_time_to_lisa_band_myr(
            primary_mass_msun,
            secondary_mass_msun,
            initial_radius_cm,
            min_observed_frequency_hz,
            eccentricity,
            redshift,
        )

    time_to_fixed_end_boundary_myr = fixed_boundary_time_myr(elapsed_limit_myr)
    entry_reached = (
        np.isfinite(time_to_fixed_end_boundary_myr)
        and elapsed_limit_myr >= time_to_fixed_end_boundary_myr
    )
    if entry_reached:
        lower_time_myr = 0.0
        upper_time_myr = elapsed_limit_myr
        for _ in range(80):
            midpoint_time_myr = 0.5 * (lower_time_myr + upper_time_myr)
            midpoint_fixed_time_myr = fixed_boundary_time_myr(midpoint_time_myr)
            if (
                np.isfinite(midpoint_fixed_time_myr)
                and midpoint_time_myr >= midpoint_fixed_time_myr
            ):
                upper_time_myr = midpoint_time_myr
            else:
                lower_time_myr = midpoint_time_myr

        time_to_entry_myr = upper_time_myr
        arrival_time_myr = start_time_myr + time_to_entry_myr
        arrival_redshift = redshift_model.redshift_evlove_with_time(
            initial_redshift,
            arrival_time_myr,
        )
        arrival_entry_radius_cm = gw_helpers.lisa_entry_radius_cm(
            primary_mass_msun,
            secondary_mass_msun,
            min_observed_frequency_hz,
            arrival_redshift,
        )
        if not np.isfinite(arrival_entry_radius_cm):
            raise RuntimeError("moving LISA entry root is inside ISCO")
        arrival_radius_cm = arrival_entry_radius_cm
        source_frequency_hz = gw_helpers.gw_frequency_hz(
            primary_mass_msun + secondary_mass_msun,
            arrival_radius_cm,
        )
        observed_frequency_hz = gw_helpers.observed_frame_gw_frequency_hz(
            source_frequency_hz,
            arrival_redshift,
        )
        return ObservedLisaEntryOutcome(
            status="entry_reached",
            time_to_entry_myr=time_to_entry_myr,
            evaluated_elapsed_time_myr=time_to_entry_myr,
            state_time_myr=arrival_time_myr,
            state_redshift=arrival_redshift,
            state_radius_cm=arrival_radius_cm,
            entry_radius_cm=arrival_entry_radius_cm,
            source_frequency_hz=source_frequency_hz,
            observed_frequency_hz=observed_frequency_hz,
        )

    final_time_myr = start_time_myr + elapsed_limit_myr
    final_redshift = redshift_model.redshift_evlove_with_time(
        initial_redshift,
        final_time_myr,
    )
    final_radius_cm = gw_helpers.peters_radius_after_time_cm(
        primary_mass_msun,
        secondary_mass_msun,
        initial_radius_cm,
        elapsed_limit_myr,
        eccentricity,
        isco_radius_cm,
    )
    final_boundary = gw_helpers.circular_band_boundary(
        primary_mass_msun,
        secondary_mass_msun,
        min_observed_frequency_hz,
        final_redshift,
    )
    final_source_frequency_hz = gw_helpers.gw_frequency_hz(
        primary_mass_msun + secondary_mass_msun,
        final_radius_cm,
    )
    final_observed_frequency_hz = gw_helpers.observed_frame_gw_frequency_hz(
        final_source_frequency_hz,
        final_redshift,
    )
    reached_isco = np.isclose(
        elapsed_limit_myr,
        time_to_isco_myr,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    return ObservedLisaEntryOutcome(
        status=(
            "entry_not_reached_before_isco" if reached_isco else "time_limit_reached"
        ),
        time_to_entry_myr=np.inf,
        evaluated_elapsed_time_myr=elapsed_limit_myr,
        state_time_myr=final_time_myr,
        state_redshift=final_redshift,
        state_radius_cm=final_radius_cm,
        entry_radius_cm=final_boundary.formal_entry_radius_cm,
        source_frequency_hz=final_source_frequency_hz,
        observed_frequency_hz=final_observed_frequency_hz,
    )


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

    if not np.isfinite(gas_final_time_myr) or gas_final_time_myr < 0.0:
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
            detect_rate_reason="required_input_unavailable_invalid_gas_final_time",
            mission_duration_years=mission_duration_years,
            snr_status="required_input_unavailable",
        )

    redshift_model = RedshiftEvolution(hubble_constant_per_myr)
    merge_model = CosmicMergeModel(
        merge_time_myr=cosmic_merge_time_myr,
        smbh_mass_growth_factor=cosmic_merge_smbh_mass_factor,
        radius_growth_factor=cosmic_merge_radius_factor,
    )
    current_time_myr = float(gas_final_time_myr)
    current_redshift = redshift_model.redshift_evlove_with_time(
        initial_redshift,
        current_time_myr,
    )
    current_smbh_mass_msun = float(primary_mass_msun)
    current_radius_cm = float(r_gas_fin_cm)
    cosmic_merge_applied = False
    remaining_cosmic_time_myr = redshift_model.remaining_time_to_zero_redshift_myr(
        current_redshift
    )
    time_to_merge_myr = merge_model.merge_time_myr
    initial_segment_limit_myr = min(
        remaining_cosmic_time_myr,
        time_to_merge_myr if merge_model.is_active else remaining_cosmic_time_myr,
    )
    entry_outcome = solve_time_to_observed_lisa_entry(
        primary_mass_msun=current_smbh_mass_msun,
        secondary_mass_msun=secondary_mass_msun,
        initial_radius_cm=current_radius_cm,
        initial_redshift=initial_redshift,
        start_time_myr=current_time_myr,
        redshift_model=redshift_model,
        min_observed_frequency_hz=min_observed_frequency_hz,
        eccentricity=eccentricity,
        max_elapsed_time_myr=initial_segment_limit_myr,
    )

    entry_statuses = {"already_inside_band", "entry_reached"}
    if entry_outcome.status not in entry_statuses:
        if entry_outcome.status == "entry_not_reached_before_isco":
            return _gw_evolution_unavailable(
                primary_mass_msun=current_smbh_mass_msun,
                outcome=entry_outcome,
                cosmic_merge_applied=False,
                mission_duration_years=mission_duration_years,
            )

        merger_occurs_first = (
            merge_model.is_active and time_to_merge_myr < remaining_cosmic_time_myr
        )
        if not merger_occurs_first:
            return _not_detected_due_to_time(
                primary_mass_msun=current_smbh_mass_msun,
                radius_cm=entry_outcome.state_radius_cm,
                entry_radius_cm=entry_outcome.entry_radius_cm,
                gw_time_after_gas_myr=np.inf,
                detection_redshift=entry_outcome.state_redshift,
                cosmic_merge_applied=False,
                reason=(
                    "misses_cosmic_time_before_merge"
                    if merge_model.is_active
                    else "misses_cosmic_time_budget"
                ),
                mission_duration_years=mission_duration_years,
            )

        pre_merge_boundary = gw_helpers.circular_band_boundary(
            current_smbh_mass_msun,
            secondary_mass_msun,
            min_observed_frequency_hz,
            current_redshift,
        )
        radius_at_merge_cm = gw_helpers.peters_radius_after_time_cm(
            current_smbh_mass_msun,
            secondary_mass_msun,
            current_radius_cm,
            time_to_merge_myr,
            eccentricity,
            pre_merge_boundary.isco_radius_cm,
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
        if (
            not np.isfinite(current_smbh_mass_msun)
            or current_smbh_mass_msun <= 0.0
            or not np.isfinite(current_radius_cm)
            or current_radius_cm <= 0.0
            or not np.isfinite(current_redshift)
            or current_redshift < 0.0
        ):
            raise RuntimeError("cosmic merger produced an invalid orbital state")
        cosmic_merge_applied = True
        remaining_cosmic_time_myr = (
            redshift_model.remaining_time_to_zero_redshift_myr(current_redshift)
        )
        entry_outcome = solve_time_to_observed_lisa_entry(
            primary_mass_msun=current_smbh_mass_msun,
            secondary_mass_msun=secondary_mass_msun,
            initial_radius_cm=current_radius_cm,
            initial_redshift=initial_redshift,
            start_time_myr=current_time_myr,
            redshift_model=redshift_model,
            min_observed_frequency_hz=min_observed_frequency_hz,
            eccentricity=eccentricity,
            max_elapsed_time_myr=remaining_cosmic_time_myr,
        )
        if entry_outcome.status not in entry_statuses:
            if entry_outcome.status == "entry_not_reached_before_isco":
                return _gw_evolution_unavailable(
                    primary_mass_msun=current_smbh_mass_msun,
                    outcome=entry_outcome,
                    cosmic_merge_applied=True,
                    mission_duration_years=mission_duration_years,
                )
            return _not_detected_due_to_time(
                primary_mass_msun=current_smbh_mass_msun,
                radius_cm=entry_outcome.state_radius_cm,
                entry_radius_cm=entry_outcome.entry_radius_cm,
                gw_time_after_gas_myr=np.inf,
                detection_redshift=entry_outcome.state_redshift,
                cosmic_merge_applied=True,
                reason="misses_cosmic_time_budget",
                mission_duration_years=mission_duration_years,
            )

    detection_time_myr = entry_outcome.state_time_myr
    detection_redshift = entry_outcome.state_redshift
    final_entry_radius_cm = entry_outcome.entry_radius_cm
    final_radius_cm = entry_outcome.state_radius_cm
    source_frequency_hz = entry_outcome.source_frequency_hz
    observed_frequency_hz = entry_outcome.observed_frequency_hz
    gw_time_after_gas_myr = detection_time_myr - gas_final_time_myr
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


def _gw_evolution_unavailable(
    *,
    primary_mass_msun: float,
    outcome: ObservedLisaEntryOutcome,
    cosmic_merge_applied: bool,
    mission_duration_years: float,
) -> DetectRateDecision:
    """Return an explicit invalid-regime result for a GW segment ending at ISCO."""

    return DetectRateDecision(
        detect_rate_class=DETECTION_CLASS_NOT_EVALUATED,
        is_lisa_detectable=None,
        snr_proxy=np.nan,
        source_frequency_hz=outcome.source_frequency_hz,
        observed_frequency_hz=outcome.observed_frequency_hz,
        detection_time_myr=np.inf,
        detection_redshift=outcome.state_redshift,
        final_smbh_mass_msun=primary_mass_msun,
        final_radius_cm=outcome.state_radius_cm,
        final_entry_radius_cm=outcome.entry_radius_cm,
        gw_time_after_gas_myr=np.inf,
        cosmic_merge_applied=cosmic_merge_applied,
        detect_rate_reason="required_input_unavailable_invalid_frequency",
        mission_duration_years=mission_duration_years,
        snr_status="required_input_unavailable",
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
