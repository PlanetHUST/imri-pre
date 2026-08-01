"""Simple LISA detectability classification for IMRI event candidates."""

from dataclasses import dataclass

import numpy as np

import physical_process.gw_frequency_radius as gw_helpers

DETECTION_CLASS_DETECTABLE = "imri data"
DETECTION_CLASS_NOT_DETECTABLE = "non-detectable"
DEFAULT_SNR_THRESHOLD = 20.0


@dataclass(frozen=True)
class DetectRateDecision:
    """Classification result for the simplified detectability stage."""

    detect_rate_class: str
    is_lisa_detectable: bool
    snr_proxy: float
    observed_frequency_hz: float
    detect_rate_reason: str


def single_frequency_snr_proxy(
    *,
    observed_frequency_hz: float,
    min_observed_frequency_hz: float,
    max_observed_frequency_hz: float,
    snr_threshold: float = DEFAULT_SNR_THRESHOLD,
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
    source_frequency_hz: float,
    redshift: float,
    min_observed_frequency_hz: float,
    max_observed_frequency_hz: float,
    snr_threshold: float = DEFAULT_SNR_THRESHOLD,
) -> DetectRateDecision:
    """Classify one IMRI candidate with a single-frequency LISA-band proxy."""
    if not is_imri_candidate:
        return DetectRateDecision(
            detect_rate_class=DETECTION_CLASS_NOT_DETECTABLE,
            is_lisa_detectable=False,
            snr_proxy=0.0,
            observed_frequency_hz=np.nan,
            detect_rate_reason="not_imri_candidate",
        )

    if not np.isfinite(source_frequency_hz) or source_frequency_hz <= 0.0:
        return DetectRateDecision(
            detect_rate_class=DETECTION_CLASS_NOT_DETECTABLE,
            is_lisa_detectable=False,
            snr_proxy=0.0,
            observed_frequency_hz=np.nan,
            detect_rate_reason="invalid_frequency",
        )

    observed_frequency_hz = gw_helpers.observed_frame_gw_frequency_hz(
        source_frequency_hz,
        redshift,
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
            observed_frequency_hz=observed_frequency_hz,
            detect_rate_reason=DETECTION_CLASS_DETECTABLE,
        )

    return DetectRateDecision(
        detect_rate_class=DETECTION_CLASS_NOT_DETECTABLE,
        is_lisa_detectable=False,
        snr_proxy=snr_proxy,
        observed_frequency_hz=observed_frequency_hz,
        detect_rate_reason="outside_lisa_band",
    )
