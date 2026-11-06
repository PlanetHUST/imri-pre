"""Independent checks of comparison physics and first-passage semantics."""
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from physical_process import constants as c
from physical_process.damping_models import (
    DampingInput, XueDampingAdapter, RepoDampingAdapter, OutcomeStatus,
    REFERENCE_RADIUS_CM, REFERENCE_H, REFERENCE_SIGMA, SCATTERING,
    make_local_repo_calculator, orbital_invariants,
)


@pytest.fixture
def source():
    r = REFERENCE_RADIUS_CM
    i = np.deg2rad(8.)
    vk = np.sqrt(c.G * 1.e7 * c.msun / r)
    return DampingInput(0., r, i, 0., 25*c.msun, 1.e7*c.msun,
        REFERENCE_SIGMA, REFERENCE_H*r, 1.19e6, 2*vk*np.sin(i/2), 10*c.Myr)


def test_published_coefficient_cadence_and_sign(source):
    x = source
    rh = x.radius_cm * (x.compact_object_mass_g/(3*x.smbh_mass_g))**(1/3)
    mh = 2*np.pi*rh**2*x.surface_density_g_cm2/x.compact_object_mass_g
    decrement = x.inclination_rad * 10**(0.67*np.log10(mh)
        -2.64*np.log10(max(np.sin(x.inclination_rad)/REFERENCE_H, 4.6))+1.80)
    period = 2*np.pi*np.sqrt(x.radius_cm**3/(c.G*x.smbh_mass_g))
    rate = XueDampingAdapter().inclination_damping_rate(x).value
    assert rate == pytest.approx(-2*decrement/period, rel=1.e-12)
    occupancy = XueDampingAdapter.effective_crossing_probability(x).value
    assert occupancy == pytest.approx(2/np.pi*np.arcsin(REFERENCE_H/np.sin(x.inclination_rad)))
    assert rate != pytest.approx(-2*decrement/period*occupancy, rel=1.e-3, abs=0.)


def test_low_i_exponential_and_density_power(source):
    model = XueDampingAdapter()
    x = replace(source, inclination_rad=2*REFERENCE_H)
    rate = model.inclination_damping_rate(x).value
    # Below knee the fitted fractional loss is constant, hence exact exponential.
    expected = np.log(2)*x.inclination_rad/abs(rate)
    assert model.damping_time(x).value == pytest.approx(expected, rel=1.e-10)
    y = replace(x, surface_density_g_cm2=.5*x.surface_density_g_cm2)
    assert model.damping_time(y).value / model.damping_time(x).value == pytest.approx(2**.67)


@pytest.mark.parametrize('field,value', [('eccentricity', .1), ('inclination_rad', np.pi/2),
    ('compact_object_mass_g', 30*c.msun), ('smbh_mass_g', 4e6*c.msun),
    ('surface_density_g_cm2', 1.), ('surface_density_g_cm2', 2*REFERENCE_SIGMA)])
def test_uncalibrated_never_becomes_zero(source, field, value):
    model = XueDampingAdapter()
    x = replace(source, **{field: value})
    assert not model.validity_mask(x).applicable
    assert model.inclination_damping_rate(x).value is None
    o = model.evolve(x)
    assert o.status == OutcomeStatus.OUT_OF_RANGE
    assert o.t_embed_myr is None and o.delta_energy is None


def test_unavailable_scattering(source):
    for model in (XueDampingAdapter(SCATTERING), RepoDampingAdapter(
            make_local_repo_calculator(source), SCATTERING)):
        o = model.evolve(source)
        assert o.status == OutcomeStatus.UNAVAILABLE
        assert o.effective_scattering_di_dt_rad_s is None
        assert o.effective_scattering_delta_energy is None
        assert o.effective_scattering_delta_angular_momentum is None


def test_lifetime_boundary_initial_embedding_and_missing_work(source):
    model = XueDampingAdapter()
    duration = model.damping_time(source).value
    for factor, expected in ((.001, OutcomeStatus.CENSORED), (.999, OutcomeStatus.CENSORED),
                             (1., OutcomeStatus.EMBEDDED), (100., OutcomeStatus.EMBEDDED)):
        x = replace(source, disk_lifetime_s=duration*factor)
        o = model.evolve(x, keep_history=True)
        assert o.status == expected
        assert o.final_state.time_s <= x.disk_lifetime_s*(1+1e-12)
        assert o.delta_energy is None and o.delta_angular_momentum is None
        assert all(a.inclination_rad >= b.inclination_rad for a,b in zip(o.history,o.history[1:]))
    x = replace(source, inclination_rad=REFERENCE_H/2)
    assert model.evolve(x).t_embed_myr == 0.
    assert model.damping_time(x).value == 0.


def test_repo_clips_shutdown_and_retains_orbital_losses(source):
    x = replace(source, disk_lifetime_s=1.)
    o = RepoDampingAdapter(make_local_repo_calculator(x)).evolve(x)
    assert o.status == OutcomeStatus.CENSORED
    assert o.final_state.time_s == 1.
    assert o.t_embed_myr is None
    o = RepoDampingAdapter(make_local_repo_calculator(source)).evolve(source, keep_history=True)
    assert o.status == OutcomeStatus.EMBEDDED
    assert o.inclination_at_embed_rad == pytest.approx(REFERENCE_H)
    assert o.r_embed_cm < source.radius_cm
    assert o.delta_energy < 0 and o.delta_angular_momentum < 0
    ei, li = orbital_invariants(source.state, source)
    ef, lf = orbital_invariants(o.final_state, source)
    assert o.delta_energy == ef-ei
    assert o.delta_angular_momentum == lf-li


def test_iteration_limit_is_not_physical_censoring(source):
    o = RepoDampingAdapter(make_local_repo_calculator(source)).evolve(source, max_iter=1)
    assert o.status == OutcomeStatus.INVALID
    assert o.censor_time_myr is None


def test_repo_timestep_convergence(source):
    model = RepoDampingAdapter(make_local_repo_calculator(source))
    coarse = model.evolve(source)
    with patch('physical_process.damp.MAX_ACCEPTED_RELATIVE_CHANGE', .0025):
        fine = model.evolve(source)
    assert fine.status == coarse.status == OutcomeStatus.EMBEDDED
    assert abs(fine.t_embed_myr/coarse.t_embed_myr-1) < .02
    assert abs(fine.r_embed_cm/coarse.r_embed_cm-1) < .001


def test_reference_sound_speed_and_inviscid_inputs(source):
    model = XueDampingAdapter()
    assert model.validity_mask(replace(source, viscosity_alpha=0.)).applicable
    assert not model.validity_mask(replace(source, sound_speed_cm_s=1.e9)).applicable


def test_fixture_vertical_closure(source):
    calc = make_local_repo_calculator(source).timescale_calc
    np.testing.assert_allclose(2*calc.density_g_cm3*calc.aspect_ratio*calc.radius_grid_cm,
                               calc.surface_density_g_cm2)


def test_invalid_and_zero_available_time(source):
    model = XueDampingAdapter()
    assert model.evolve(replace(source, radius_cm=float('nan'))).status == OutcomeStatus.INVALID
    o = model.evolve(replace(source, disk_lifetime_s=0.))
    assert o.status == OutcomeStatus.CENSORED
    assert o.final_state.time_s == 0.
