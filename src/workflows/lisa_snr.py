"""First-order physical LISA inspiral signal-to-noise calculations.

The detector model follows the two-channel, sky-averaged analytic sensitivity
of Robson, Cornish & Liu (2019).  Frequencies and mission times are observer
frame.  Component masses passed to the public evaluation function are source
frame and are redshifted exactly once inside this module.
"""

from dataclasses import dataclass
from math import cos, exp, isclose, log, pi, sin, sqrt, tanh

import numpy as np
from scipy.integrate import quad

SPEED_OF_LIGHT_M_S = 2.99792458e8
GRAVITATIONAL_CONSTANT_SI = 6.67430e-11
SOLAR_MASS_KG = 1.98847e30
MPC_M = 3.085677581491367e22
JULIAN_YEAR_S = 365.25 * 24.0 * 3600.0

LISA_ARM_LENGTH_M = 2.5e9
LISA_TRANSFER_FREQUENCY_HZ = SPEED_OF_LIGHT_M_S / (2.0 * pi * LISA_ARM_LENGTH_M)
LISA_CONFUSION_AMPLITUDE_HZ_INV = 9.0e-45

LISA_SENSITIVITY_MODEL = "robson_cornish_liu_2019_analytic_two_channel"
LISA_WAVEFORM_MODEL = "circular_nonspinning_quadrupole_spa_inspiral"
LISA_AVERAGING_CONVENTION = "rcl_sky_polarization_psd_plus_inclination_16_over_5"
LISA_COSMOLOGY_MODEL = "flat_lcdm_matter_lambda"
LISA_ISCO_MODEL = "schwarzschild_isco_6gm_over_c2"
SNR_QUADRATURE_METHOD = "scipy_quad_log_frequency"
SNR_QUADRATURE_RELATIVE_TOLERANCE = 1.0e-8
COSMOLOGY_QUADRATURE_RELATIVE_TOLERANCE = 1.0e-11

DEFAULT_DETECTOR_H0_KM_S_MPC = 67.4
DEFAULT_DETECTOR_OMEGA_M = 0.315

# RCL Table 1: (alpha, beta, kappa, gamma, f_k [Hz]).
_CONFUSION_PARAMETERS = {
    0.5: (0.133, 243.0, 482.0, 917.0, 0.00258),
    1.0: (0.171, 292.0, 1020.0, 1680.0, 0.00215),
    2.0: (0.165, 299.0, 611.0, 1340.0, 0.00173),
    4.0: (0.138, -221.0, 521.0, 1680.0, 0.00113),
}
SUPPORTED_MISSION_DURATIONS_YEARS = tuple(_CONFUSION_PARAMETERS)

SNR_STATUS_EVALUATED = "evaluated"
SNR_STATUS_NO_OVERLAP = "no_physical_lisa_overlap"
SNR_STATUS_REQUIRED_INPUT_UNAVAILABLE = "required_input_unavailable"
SNR_STATUS_MODEL_UNAVAILABLE = "model_unavailable"
SNR_STATUS_NUMERICAL_FAILURE = "numerical_failure"


@dataclass(frozen=True)
class LISAObservationWindow:
    """Observer-frame frequency interval sampled during one LISA mission."""

    status: str
    reason: str
    observation_case: str
    f_low_hz: float
    f_high_hz: float
    f_upper_available_hz: float
    time_to_band_years: float
    observation_duration_years: float


@dataclass(frozen=True)
class LISASNRResult:
    """Auditable result of one first-order LISA inspiral calculation."""

    status: str
    reason: str
    is_detectable: bool | None
    lisa_snr: float
    m1_source_msun: float
    m2_source_msun: float
    redshift: float
    chirp_mass_source_msun: float
    chirp_mass_detector_msun: float
    luminosity_distance_mpc: float
    f_isco_source_hz: float
    f_isco_observer_hz: float
    current_observed_frequency_hz: float
    f_low_hz: float
    f_high_hz: float
    mission_duration_years: float
    observation_duration_years: float
    observation_case: str
    snr_threshold: float


def source_chirp_mass_msun(m1_source_msun: float, m2_source_msun: float) -> float:
    """Return the source-frame chirp mass in solar masses."""
    if (
        not np.isfinite(m1_source_msun)
        or not np.isfinite(m2_source_msun)
        or m1_source_msun <= 0.0
        or m2_source_msun <= 0.0
    ):
        raise ValueError("source component masses must be finite and positive")
    total_mass_msun = m1_source_msun + m2_source_msun
    return float(
        (m1_source_msun * m2_source_msun) ** (3.0 / 5.0)
        / total_mass_msun ** (1.0 / 5.0)
    )


def luminosity_distance_mpc(
    redshift: float,
    *,
    hubble_constant_km_s_mpc: float = DEFAULT_DETECTOR_H0_KM_S_MPC,
    omega_m: float = DEFAULT_DETECTOR_OMEGA_M,
) -> float:
    """Return flat matter-plus-Lambda luminosity distance in Mpc."""
    if not np.isfinite(redshift) or redshift < 0.0:
        raise ValueError("redshift must be finite and nonnegative")
    if (
        not np.isfinite(hubble_constant_km_s_mpc)
        or hubble_constant_km_s_mpc <= 0.0
    ):
        raise ValueError("detector H0 must be finite and positive")
    if not np.isfinite(omega_m) or not 0.0 < omega_m < 1.0:
        raise ValueError("Omega_m must satisfy 0 < Omega_m < 1")
    if redshift == 0.0:
        return 0.0

    omega_lambda = 1.0 - omega_m

    def inverse_expansion_rate(integration_redshift: float) -> float:
        return 1.0 / sqrt(
            omega_m * (1.0 + integration_redshift) ** 3 + omega_lambda
        )

    comoving_integral, _ = quad(
        inverse_expansion_rate,
        0.0,
        redshift,
        epsabs=0.0,
        epsrel=COSMOLOGY_QUADRATURE_RELATIVE_TOLERANCE,
        limit=200,
    )
    speed_of_light_km_s = SPEED_OF_LIGHT_M_S / 1000.0
    return float(
        (1.0 + redshift)
        * speed_of_light_km_s
        / hubble_constant_km_s_mpc
        * comoving_integral
    )


def schwarzschild_isco_gw_frequency_source_hz(
    total_mass_source_msun: float,
) -> float:
    """Return dominant-quadrupole GW frequency at Schwarzschild ISCO."""
    if not np.isfinite(total_mass_source_msun) or total_mass_source_msun <= 0.0:
        raise ValueError("source total mass must be finite and positive")
    total_mass_kg = total_mass_source_msun * SOLAR_MASS_KG
    return float(
        SPEED_OF_LIGHT_M_S**3
        / (6.0 ** (3.0 / 2.0) * pi * GRAVITATIONAL_CONSTANT_SI * total_mass_kg)
    )


def lisa_optical_metrology_displacement_psd_m2_hz(
    observed_frequency_hz: float,
) -> float:
    """Return RCL single-link optical-metrology displacement PSD [m^2/Hz]."""
    _require_positive_frequency(observed_frequency_hz)
    return float(
        (1.5e-11) ** 2 * (1.0 + (2.0e-3 / observed_frequency_hz) ** 4)
    )


def lisa_acceleration_psd_m2_s4_hz(observed_frequency_hz: float) -> float:
    """Return RCL single-test-mass acceleration PSD [m^2 s^-4/Hz]."""
    _require_positive_frequency(observed_frequency_hz)
    return float(
        (3.0e-15) ** 2
        * (1.0 + (0.4e-3 / observed_frequency_hz) ** 2)
        * (1.0 + (observed_frequency_hz / 8.0e-3) ** 4)
    )


def lisa_instrument_strain_psd_hz(observed_frequency_hz: float) -> float:
    """Return RCL two-channel effective instrument strain PSD [1/Hz]."""
    p_oms_m2_hz = lisa_optical_metrology_displacement_psd_m2_hz(
        observed_frequency_hz
    )
    p_acc_m2_s4_hz = lisa_acceleration_psd_m2_s4_hz(observed_frequency_hz)
    acceleration_displacement_psd_m2_hz = (
        2.0
        * (1.0 + cos(observed_frequency_hz / LISA_TRANSFER_FREQUENCY_HZ) ** 2)
        * p_acc_m2_s4_hz
        / (2.0 * pi * observed_frequency_hz) ** 4
    )
    finite_arm_response = 1.0 + 0.6 * (
        observed_frequency_hz / LISA_TRANSFER_FREQUENCY_HZ
    ) ** 2
    return float(
        10.0
        / (3.0 * LISA_ARM_LENGTH_M**2)
        * (p_oms_m2_hz + acceleration_displacement_psd_m2_hz)
        * finite_arm_response
    )


def lisa_confusion_strain_psd_hz(
    observed_frequency_hz: float,
    mission_duration_years: float,
) -> float:
    """Return RCL unresolved Galactic-binary strain PSD [1/Hz]."""
    _require_positive_frequency(observed_frequency_hz)
    parameters = _confusion_parameters(mission_duration_years)
    alpha, beta, kappa, gamma, knee_frequency_hz = parameters
    exponent = -observed_frequency_hz**alpha + beta * observed_frequency_hz * sin(
        kappa * observed_frequency_hz
    )
    return float(
        LISA_CONFUSION_AMPLITUDE_HZ_INV
        * observed_frequency_hz ** (-7.0 / 3.0)
        * exp(exponent)
        * (1.0 + tanh(gamma * (knee_frequency_hz - observed_frequency_hz)))
    )


def lisa_effective_strain_psd_hz(
    observed_frequency_hz: float,
    mission_duration_years: float,
) -> float:
    """Return total RCL effective strain noise PSD [1/Hz], never ASD."""
    total_psd_hz = lisa_instrument_strain_psd_hz(
        observed_frequency_hz
    ) + lisa_confusion_strain_psd_hz(
        observed_frequency_hz, mission_duration_years
    )
    if not np.isfinite(total_psd_hz) or total_psd_hz <= 0.0:
        raise ArithmeticError("LISA effective strain PSD is not finite and positive")
    return float(total_psd_hz)


def fourier_inspiral_amplitude_hz_inverse(
    observed_frequency_hz: float,
    chirp_mass_detector_msun: float,
    luminosity_distance_mpc_value: float,
) -> float:
    """Return the uninclined SPA amplitude A(f) in Hz^-1.

    The RCL inclination average is applied later through ``16/5`` in the SNR
    squared.  The sky and polarization response is already in the effective
    RCL strain PSD.
    """
    _require_positive_frequency(observed_frequency_hz)
    if (
        not np.isfinite(chirp_mass_detector_msun)
        or chirp_mass_detector_msun <= 0.0
    ):
        raise ValueError("detector-frame chirp mass must be finite and positive")
    if (
        not np.isfinite(luminosity_distance_mpc_value)
        or luminosity_distance_mpc_value <= 0.0
    ):
        raise ValueError("luminosity distance must be finite and positive")
    chirp_mass_time_s = (
        GRAVITATIONAL_CONSTANT_SI
        * chirp_mass_detector_msun
        * SOLAR_MASS_KG
        / SPEED_OF_LIGHT_M_S**3
    )
    luminosity_distance_time_s = (
        luminosity_distance_mpc_value * MPC_M / SPEED_OF_LIGHT_M_S
    )
    return float(
        sqrt(5.0 / 24.0)
        * chirp_mass_time_s ** (5.0 / 6.0)
        / (pi ** (2.0 / 3.0) * luminosity_distance_time_s)
        * observed_frequency_hz ** (-7.0 / 6.0)
    )


def inspiral_time_to_coalescence_s(
    observed_frequency_hz: float,
    chirp_mass_detector_msun: float,
) -> float:
    """Return leading-order observer-frame time to formal coalescence [s]."""
    _require_positive_frequency(observed_frequency_hz)
    if (
        not np.isfinite(chirp_mass_detector_msun)
        or chirp_mass_detector_msun <= 0.0
    ):
        raise ValueError("detector-frame chirp mass must be finite and positive")
    chirp_mass_time_s = (
        GRAVITATIONAL_CONSTANT_SI
        * chirp_mass_detector_msun
        * SOLAR_MASS_KG
        / SPEED_OF_LIGHT_M_S**3
    )
    return float(
        5.0
        / 256.0
        * chirp_mass_time_s ** (-5.0 / 3.0)
        * (pi * observed_frequency_hz) ** (-8.0 / 3.0)
    )


def observed_frequency_from_time_to_coalescence_hz(
    time_to_coalescence_s: float,
    chirp_mass_detector_msun: float,
) -> float:
    """Invert the leading-order observer-frame inspiral time relation."""
    if not np.isfinite(time_to_coalescence_s) or time_to_coalescence_s <= 0.0:
        raise ValueError("time to coalescence must be finite and positive")
    if (
        not np.isfinite(chirp_mass_detector_msun)
        or chirp_mass_detector_msun <= 0.0
    ):
        raise ValueError("detector-frame chirp mass must be finite and positive")
    chirp_mass_time_s = (
        GRAVITATIONAL_CONSTANT_SI
        * chirp_mass_detector_msun
        * SOLAR_MASS_KG
        / SPEED_OF_LIGHT_M_S**3
    )
    return float(
        (
            5.0
            / (256.0 * time_to_coalescence_s)
            * chirp_mass_time_s ** (-5.0 / 3.0)
        )
        ** (3.0 / 8.0)
        / pi
    )


def mission_observation_window(
    *,
    current_observed_frequency_hz: float,
    f_isco_observer_hz: float,
    chirp_mass_detector_msun: float,
    detector_min_frequency_hz: float,
    detector_max_frequency_hz: float,
    mission_duration_years: float,
) -> LISAObservationWindow:
    """Return the physical inspiral interval overlapping one LISA mission."""
    _require_positive_frequency(current_observed_frequency_hz)
    _require_positive_frequency(f_isco_observer_hz)
    _require_positive_frequency(detector_min_frequency_hz)
    _require_positive_frequency(detector_max_frequency_hz)
    _confusion_parameters(mission_duration_years)
    if detector_min_frequency_hz >= detector_max_frequency_hz:
        raise ValueError("detector frequencies must satisfy f_min < f_max")

    f_upper_available_hz = min(
        detector_max_frequency_hz, f_isco_observer_hz
    )
    if f_upper_available_hz <= detector_min_frequency_hz:
        return _no_overlap_window(
            "observed_isco_at_or_below_detector_minimum",
            f_upper_available_hz,
        )
    if current_observed_frequency_hz >= f_upper_available_hz:
        return _no_overlap_window(
            "source_already_beyond_inspiral_observation_interval",
            f_upper_available_hz,
        )

    mission_duration_s = mission_duration_years * JULIAN_YEAR_S
    if current_observed_frequency_hz < detector_min_frequency_hz:
        time_to_band_s = inspiral_time_to_coalescence_s(
            current_observed_frequency_hz, chirp_mass_detector_msun
        ) - inspiral_time_to_coalescence_s(
            detector_min_frequency_hz, chirp_mass_detector_msun
        )
        if time_to_band_s >= mission_duration_s:
            return _no_overlap_window(
                "source_does_not_enter_band_during_mission",
                f_upper_available_hz,
                time_to_band_years=time_to_band_s / JULIAN_YEAR_S,
            )
        f_low_hz = detector_min_frequency_hz
        remaining_observation_s = mission_duration_s - time_to_band_s
        start_case = "enters_band_during_mission"
    else:
        time_to_band_s = 0.0
        f_low_hz = current_observed_frequency_hz
        remaining_observation_s = mission_duration_s
        start_case = "already_in_band"

    time_to_upper_s = inspiral_time_to_coalescence_s(
        f_low_hz, chirp_mass_detector_msun
    ) - inspiral_time_to_coalescence_s(
        f_upper_available_hz, chirp_mass_detector_msun
    )
    if remaining_observation_s >= time_to_upper_s:
        f_high_hz = f_upper_available_hz
        observation_duration_s = time_to_upper_s
        if f_isco_observer_hz <= detector_max_frequency_hz:
            end_case = "reaches_isco_during_mission"
        else:
            end_case = "reaches_detector_max_during_mission"
    else:
        target_coalescence_time_s = inspiral_time_to_coalescence_s(
            f_low_hz, chirp_mass_detector_msun
        ) - remaining_observation_s
        f_high_hz = min(
            f_upper_available_hz,
            observed_frequency_from_time_to_coalescence_hz(
                target_coalescence_time_s, chirp_mass_detector_msun
            ),
        )
        observation_duration_s = remaining_observation_s
        end_case = "mission_limited_slow_inspiral"

    if not np.isfinite(f_high_hz) or f_high_hz <= f_low_hz:
        raise ArithmeticError("mission frequency interval is not positive")
    return LISAObservationWindow(
        status=SNR_STATUS_EVALUATED,
        reason="lisa_frequency_overlap",
        observation_case=f"{start_case}:{end_case}",
        f_low_hz=float(f_low_hz),
        f_high_hz=float(f_high_hz),
        f_upper_available_hz=float(f_upper_available_hz),
        time_to_band_years=float(time_to_band_s / JULIAN_YEAR_S),
        observation_duration_years=float(observation_duration_s / JULIAN_YEAR_S),
    )


def integrated_lisa_snr(
    *,
    chirp_mass_detector_msun: float,
    luminosity_distance_mpc_value: float,
    f_low_hz: float,
    f_high_hz: float,
    mission_duration_years: float,
) -> float:
    """Return RCL sky/orientation-averaged inspiral SNR.

    RCL put the sky and polarization response into the effective strain PSD.
    Averaging the binary inclination gives ``rho^2 = (16/5) int A^2/S_n df``.
    The integral is performed in log-frequency for numerical stability.
    """
    _require_positive_frequency(f_low_hz)
    _require_positive_frequency(f_high_hz)
    _confusion_parameters(mission_duration_years)
    if f_high_hz <= f_low_hz:
        raise ValueError("SNR bounds must satisfy f_low < f_high")

    def log_frequency_integrand(log_frequency: float) -> float:
        frequency_hz = exp(log_frequency)
        amplitude_hz_inverse = fourier_inspiral_amplitude_hz_inverse(
            frequency_hz,
            chirp_mass_detector_msun,
            luminosity_distance_mpc_value,
        )
        noise_psd_hz = lisa_effective_strain_psd_hz(
            frequency_hz, mission_duration_years
        )
        return frequency_hz * amplitude_hz_inverse**2 / noise_psd_hz

    integral, _ = quad(
        log_frequency_integrand,
        log(f_low_hz),
        log(f_high_hz),
        epsabs=0.0,
        epsrel=SNR_QUADRATURE_RELATIVE_TOLERANCE,
        limit=300,
    )
    snr_squared = 16.0 / 5.0 * integral
    if not np.isfinite(snr_squared) or snr_squared < 0.0:
        raise ArithmeticError("integrated LISA SNR squared is invalid")
    return float(sqrt(snr_squared))


def evaluate_lisa_inspiral_snr(
    *,
    m1_source_msun: float,
    m2_source_msun: float,
    redshift: float,
    current_observed_frequency_hz: float,
    mission_duration_years: float,
    detector_min_frequency_hz: float,
    detector_max_frequency_hz: float,
    snr_threshold: float,
    hubble_constant_km_s_mpc: float = DEFAULT_DETECTOR_H0_KM_S_MPC,
    omega_m: float = DEFAULT_DETECTOR_OMEGA_M,
    eccentricity: float = 0.0,
) -> LISASNRResult:
    """Evaluate the complete first-order physical LISA SNR chain."""
    base = {
        "m1_source_msun": float(m1_source_msun),
        "m2_source_msun": float(m2_source_msun),
        "redshift": float(redshift),
        "current_observed_frequency_hz": float(current_observed_frequency_hz),
        "mission_duration_years": float(mission_duration_years),
        "snr_threshold": float(snr_threshold),
    }
    unavailable = {
        "chirp_mass_source_msun": np.nan,
        "chirp_mass_detector_msun": np.nan,
        "luminosity_distance_mpc": np.nan,
        "f_isco_source_hz": np.nan,
        "f_isco_observer_hz": np.nan,
        "f_low_hz": np.nan,
        "f_high_hz": np.nan,
        "observation_duration_years": np.nan,
        "observation_case": "not_evaluated",
    }
    if not np.isfinite(eccentricity) or eccentricity != 0.0:
        return LISASNRResult(
            status=SNR_STATUS_MODEL_UNAVAILABLE,
            reason="snr_model_unavailable_nonzero_eccentricity",
            is_detectable=None,
            lisa_snr=np.nan,
            **base,
            **unavailable,
        )
    try:
        _confusion_parameters(mission_duration_years)
    except ValueError:
        return LISASNRResult(
            status=SNR_STATUS_MODEL_UNAVAILABLE,
            reason="snr_model_unavailable_mission_duration",
            is_detectable=None,
            lisa_snr=np.nan,
            **base,
            **unavailable,
        )
    if (
        not np.isfinite(redshift)
        or redshift <= 0.0
        or not np.isfinite(snr_threshold)
        or snr_threshold <= 0.0
    ):
        return LISASNRResult(
            status=SNR_STATUS_REQUIRED_INPUT_UNAVAILABLE,
            reason="required_detector_input_unavailable",
            is_detectable=None,
            lisa_snr=np.nan,
            **base,
            **unavailable,
        )

    try:
        chirp_mass_source_msun = source_chirp_mass_msun(
            m1_source_msun, m2_source_msun
        )
        chirp_mass_detector_msun = (1.0 + redshift) * chirp_mass_source_msun
        distance_mpc = luminosity_distance_mpc(
            redshift,
            hubble_constant_km_s_mpc=hubble_constant_km_s_mpc,
            omega_m=omega_m,
        )
        f_isco_source_hz = schwarzschild_isco_gw_frequency_source_hz(
            m1_source_msun + m2_source_msun
        )
        f_isco_observer_hz = f_isco_source_hz / (1.0 + redshift)
        window = mission_observation_window(
            current_observed_frequency_hz=current_observed_frequency_hz,
            f_isco_observer_hz=f_isco_observer_hz,
            chirp_mass_detector_msun=chirp_mass_detector_msun,
            detector_min_frequency_hz=detector_min_frequency_hz,
            detector_max_frequency_hz=detector_max_frequency_hz,
            mission_duration_years=mission_duration_years,
        )
        common = {
            **base,
            "chirp_mass_source_msun": chirp_mass_source_msun,
            "chirp_mass_detector_msun": chirp_mass_detector_msun,
            "luminosity_distance_mpc": distance_mpc,
            "f_isco_source_hz": f_isco_source_hz,
            "f_isco_observer_hz": f_isco_observer_hz,
            "f_low_hz": window.f_low_hz,
            "f_high_hz": window.f_high_hz,
            "observation_duration_years": window.observation_duration_years,
            "observation_case": window.observation_case,
        }
        if window.status == SNR_STATUS_NO_OVERLAP:
            return LISASNRResult(
                status=window.status,
                reason="no_lisa_frequency_overlap",
                is_detectable=False,
                lisa_snr=0.0,
                **common,
            )
        snr = integrated_lisa_snr(
            chirp_mass_detector_msun=chirp_mass_detector_msun,
            luminosity_distance_mpc_value=distance_mpc,
            f_low_hz=window.f_low_hz,
            f_high_hz=window.f_high_hz,
            mission_duration_years=mission_duration_years,
        )
    except ValueError:
        return LISASNRResult(
            status=SNR_STATUS_REQUIRED_INPUT_UNAVAILABLE,
            reason="required_detector_input_unavailable",
            is_detectable=None,
            lisa_snr=np.nan,
            **base,
            **unavailable,
        )
    except (ArithmeticError, OverflowError):
        return LISASNRResult(
            status=SNR_STATUS_NUMERICAL_FAILURE,
            reason="snr_numerical_failure",
            is_detectable=None,
            lisa_snr=np.nan,
            **base,
            **unavailable,
        )

    is_detectable = snr >= snr_threshold
    return LISASNRResult(
        status=SNR_STATUS_EVALUATED,
        reason=(
            "lisa_snr_above_threshold"
            if is_detectable
            else "lisa_snr_below_threshold"
        ),
        is_detectable=is_detectable,
        lisa_snr=snr,
        **common,
    )


def _confusion_parameters(
    mission_duration_years: float,
) -> tuple[float, float, float, float, float]:
    if not np.isfinite(mission_duration_years):
        raise ValueError("mission duration must be finite")
    for supported_years, parameters in _CONFUSION_PARAMETERS.items():
        if isclose(mission_duration_years, supported_years, rel_tol=0.0, abs_tol=1e-12):
            return parameters
    raise ValueError(
        "mission duration must be one of "
        f"{SUPPORTED_MISSION_DURATIONS_YEARS} years"
    )


def _require_positive_frequency(observed_frequency_hz: float) -> None:
    if not np.isfinite(observed_frequency_hz) or observed_frequency_hz <= 0.0:
        raise ValueError("observed frequency must be finite and positive")


def _no_overlap_window(
    reason: str,
    f_upper_available_hz: float,
    *,
    time_to_band_years: float = np.inf,
) -> LISAObservationWindow:
    return LISAObservationWindow(
        status=SNR_STATUS_NO_OVERLAP,
        reason=reason,
        observation_case="no_lisa_frequency_overlap",
        f_low_hz=np.nan,
        f_high_hz=np.nan,
        f_upper_available_hz=float(f_upper_available_hz),
        time_to_band_years=float(time_to_band_years),
        observation_duration_years=0.0,
    )
