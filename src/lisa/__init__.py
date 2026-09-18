"""LISA detector subsystem: does the instrument see an already-formed source?

Scientific boundary
-------------------
This package is strictly **downstream** of event formation.  It answers only::

    IMRI source parameters (masses, redshift, entry frequency)
        -> GW amplitude and observation window
        -> LISA noise / sensitivity
        -> SNR
        -> detectability decision

It never decides whether an object *becomes* an IMRI; that is the job of
:mod:`workflows.event_rate` and :mod:`workflows.event_work`.  Conversely those
modules contain no sensitivity, noise or SNR formula of their own.

Modules
-------
``snr``
    Robson, Cornish & Liu (2019) analytic two-channel sky-averaged sensitivity,
    a circular non-spinning quadrupole SPA inspiral amplitude, flat-LambdaCDM
    luminosity distance, the mission observation window and the integrated SNR.
``detectability``
    Time-to-LISA-band-entry solving under gas/GW evolution, and the SNR
    threshold decision that turns an SNR into a detectability class.

Redshift convention
-------------------
Component masses enter :func:`evaluate_lisa_inspiral_snr` in the **source**
frame and are redshifted to the detector frame exactly once, inside
:mod:`.snr`.  Frequencies crossing the public API are **observer** frame.
Callers must not pre-redshift either quantity.
"""

from .detectability import (
    DETECTION_CLASS_DETECTABLE,
    DETECTION_CLASS_NOT_DETECTABLE,
    DETECTION_CLASS_NOT_EVALUATED,
    classify_detectability,
    solve_time_to_observed_lisa_entry,
    unsupported_band_entry_decision,
)
from .snr import (
    COSMOLOGY_QUADRATURE_RELATIVE_TOLERANCE,
    LISA_ARM_LENGTH_M,
    LISA_AVERAGING_CONVENTION,
    LISA_COSMOLOGY_MODEL,
    LISA_ISCO_MODEL,
    LISA_SENSITIVITY_MODEL,
    LISA_WAVEFORM_MODEL,
    SNR_QUADRATURE_METHOD,
    SNR_QUADRATURE_RELATIVE_TOLERANCE,
    SNR_STATUS_EVALUATED,
    SNR_STATUS_MODEL_UNAVAILABLE,
    SNR_STATUS_NO_OVERLAP,
    SNR_STATUS_NUMERICAL_FAILURE,
    SNR_STATUS_REQUIRED_INPUT_UNAVAILABLE,
    SUPPORTED_MISSION_DURATIONS_YEARS,
    LISAObservationWindow,
    LISASNRResult,
    evaluate_lisa_inspiral_snr,
    integrated_lisa_snr,
    lisa_effective_strain_psd_hz,
    luminosity_distance_mpc,
    mission_observation_window,
    schwarzschild_isco_gw_frequency_source_hz,
    source_chirp_mass_msun,
)

__all__ = [
    "COSMOLOGY_QUADRATURE_RELATIVE_TOLERANCE",
    "DETECTION_CLASS_DETECTABLE",
    "DETECTION_CLASS_NOT_DETECTABLE",
    "DETECTION_CLASS_NOT_EVALUATED",
    "LISA_ARM_LENGTH_M",
    "LISA_AVERAGING_CONVENTION",
    "LISA_COSMOLOGY_MODEL",
    "LISA_ISCO_MODEL",
    "LISA_SENSITIVITY_MODEL",
    "LISA_WAVEFORM_MODEL",
    "LISAObservationWindow",
    "LISASNRResult",
    "SNR_QUADRATURE_METHOD",
    "SNR_QUADRATURE_RELATIVE_TOLERANCE",
    "SNR_STATUS_EVALUATED",
    "SNR_STATUS_MODEL_UNAVAILABLE",
    "SNR_STATUS_NO_OVERLAP",
    "SNR_STATUS_NUMERICAL_FAILURE",
    "SNR_STATUS_REQUIRED_INPUT_UNAVAILABLE",
    "SUPPORTED_MISSION_DURATIONS_YEARS",
    "classify_detectability",
    "evaluate_lisa_inspiral_snr",
    "integrated_lisa_snr",
    "lisa_effective_strain_psd_hz",
    "luminosity_distance_mpc",
    "mission_observation_window",
    "schwarzschild_isco_gw_frequency_source_hz",
    "solve_time_to_observed_lisa_entry",
    "source_chirp_mass_msun",
    "unsupported_band_entry_decision",
]
