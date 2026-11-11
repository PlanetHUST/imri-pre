"""Independent detector normalization and nonzero-detection regression checks."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from lisa.snr import (
    luminosity_distance_mpc, lisa_effective_strain_psd_hz,
    evaluate_lisa_inspiral_snr, evaluate_lisa_coupled_track_snr,
)
from physical_process import constants as const
from physical_process.orbital_evolution import CircularBandTrack, peters_beta_cm4_s

spec = importlib.util.spec_from_file_location("detector_audit_script", Path(__file__).resolve().parents[2] / "scripts/check_detect_process.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def evaluate(m2=100, z=8, threshold=20, fmin=.001):
    return evaluate_lisa_inspiral_snr(
        m1_source_msun=1e5, m2_source_msun=m2, redshift=z,
        current_observed_frequency_hz=fmin, mission_duration_years=4,
        detector_min_frequency_hz=fmin, detector_max_frequency_hz=.1,
        snr_threshold=threshold,
    )


def test_reference_distance_and_noise_agree_with_production():
    for z in (.1, 1, 8, 15):
        assert audit.reference_distance_mpc(z) == pytest.approx(luminosity_distance_mpc(z), rel=1e-10)
    frequencies = np.geomspace(1e-4, .1, 31)
    production = [lisa_effective_strain_psd_hz(float(f), 4) for f in frequencies]
    np.testing.assert_allclose(audit.reference_psd(frequencies), production, rtol=1e-12)


def test_independent_snr_integral_and_inverse_distance_scaling():
    result = evaluate(m2=197, z=8)
    reference = audit.reference_snr(1e5, 197, 8, result.luminosity_distance_mpc, result.f_low_hz, result.f_high_hz)
    assert reference == pytest.approx(result.lisa_snr, rel=1e-6)
    farther = audit.reference_snr(1e5, 197, 8, 2 * result.luminosity_distance_mpc, result.f_low_hz, result.f_high_hz)
    assert farther == pytest.approx(reference / 2, rel=1e-12)


def test_detector_can_return_positive_detections():
    nearby = evaluate(m2=100, z=1)
    massive = evaluate(m2=1000, z=8)
    distant = evaluate(m2=100, z=8)
    assert nearby.status == massive.status == distant.status == "evaluated"
    assert nearby.is_detectable is True
    assert nearby.lisa_snr == pytest.approx(58.070693, rel=1e-6)
    assert massive.is_detectable is True
    assert distant.is_detectable is False
    assert 0 < distant.lisa_snr < 20


def test_redshift_changes_snr_even_when_all_counted_detections_are_zero():
    results = [evaluate(m2=197, z=z) for z in (8, 10, 12, 14, 15)]
    assert all(result.status == "evaluated" for result in results)
    assert all(result.is_detectable is False for result in results)
    assert np.all(np.diff([result.lisa_snr for result in results]) < 0)
    assert results[0].lisa_snr > 2 * results[-1].lisa_snr
    for z, result in zip((8, 10, 12, 14, 15), results):
        assert result.chirp_mass_detector_msun == pytest.approx((1 + z) * result.chirp_mass_source_msun)
        assert result.f_isco_observer_hz == pytest.approx(result.f_isco_source_hz / (1 + z))


def test_threshold_reclassification_does_not_change_snr_or_window():
    strict, relaxed = evaluate(m2=197, threshold=20), evaluate(m2=197, threshold=15)
    assert strict.is_detectable is False
    assert relaxed.is_detectable is True
    assert strict.lisa_snr == relaxed.lisa_snr
    assert strict.f_low_hz == relaxed.f_low_hz
    assert strict.f_high_hz == relaxed.f_high_hz
    assert 0 < strict.observation_duration_years < strict.mission_duration_years


def test_coupled_track_vacuum_limit_can_detect_at_relaxed_threshold():
    m1, m2, z = 1e5, 197, 8
    total = m1 + m2
    r_entry = (const.G * total * const.msun / (np.pi * .001 * (1 + z))**2)**(1/3)
    r_isco = 6 * const.G * total * const.msun / const.c**2
    beta = peters_beta_cm4_s(m1, m2)
    end = (r_entry**4 - r_isco**4) / (4 * beta * const.Myr)
    class ConstantClock:
        def redshift_evlove_with_time(self, initial_redshift, time_myr):
            return initial_redshift
    track = CircularBandTrack(
        status="isco", entry_time_myr=0, entry_radius_cm=r_entry,
        end_time_myr=end, end_radius_cm=r_isco, end_redshift=z,
        gas_end_time_myr=0, gas_end_radius_cm=r_entry, beta_cm4_s=beta,
        gas_log_radius_solution=None, initial_redshift=z, total_mass_msun=total,
        redshift_model=ConstantClock(),
    )
    results = [evaluate_lisa_coupled_track_snr(
        track=track, m1_source_msun=m1, m2_source_msun=m2,
        mission_duration_years=4, detector_min_frequency_hz=.001,
        detector_max_frequency_hz=.1, snr_threshold=threshold,
    ) for threshold in (15, 20)]
    assert all(result.status == "evaluated" for result in results)
    assert results[0].is_detectable is True
    assert results[1].is_detectable is False
    assert results[0].lisa_snr == pytest.approx(evaluate(m2=197).lisa_snr, rel=5e-4)
    # At identical frequency bounds, explicitly account for the historical
    # CGS/SI solar-mass normalization mismatch rather than hiding a factor error.
    result = results[0]
    independent = audit.reference_snr(m1, m2, z, result.luminosity_distance_mpc,
                                      result.f_low_hz, result.f_high_hz)
    constants_correction = (const.GRAVITATIONAL_CONSTANT_SI * const.SOLAR_MASS_KG /
                            (const.G * const.msun * 1e-6))**(5/6)
    assert result.lisa_snr == pytest.approx(independent * constants_correction, rel=1e-6)


def test_frequency_export_units_are_explicit():
    assert audit.frequency_hz({"snr_f_low_micro_hz": 1000}, "snr_f_low") == .001
    assert audit.frequency_hz({"snr_f_low_milli_hz": 1}, "snr_f_low") == .001
    assert audit.frequency_hz({"snr_f_low_hz": .001}, "snr_f_low") == .001
    with pytest.raises(KeyError):
        audit.frequency_hz({}, "snr_f_low")
