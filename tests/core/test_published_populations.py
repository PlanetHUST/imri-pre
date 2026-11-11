"""Published component budgets and spatial-versus-orbital radial semantics."""
from contextlib import redirect_stderr
from io import StringIO
from types import SimpleNamespace
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from physical_process import constants as const
from physical_process.population_models import (
    PAN_NSC, PAN_PAPER, XUE_BH, LEGACY_POPIII,
    pan_counts, published_masses, pan_orbits, xue_orbits,
    tremaine_spatial_cdf, population_description,
)
from physical_process.damp import DampingCalculator, DampingStatus
from workflows.cli import build_parser, build_config_from_arguments
from workflows.metadata import build_metadata
from workflows.population import sample_disk_crossing_population


def config(model, *extra):
    return build_config_from_arguments(build_parser().parse_args(["--population-model", model, *extra]))


def test_cli_default_restores_original_imf_and_published_presets_are_explicit():
    default = build_config_from_arguments(build_parser().parse_args([]))
    assert default.population_model == LEGACY_POPIII
    assert default.initial_stellar_density_gamma == 1.75
    legacy = build_config_from_arguments(build_parser().parse_args([
        "--popiii-min-mass-msun", "1", "--popiii-max-mass-msun", "400"]))
    assert legacy.population_model == LEGACY_POPIII
    assert legacy.initial_stellar_density_gamma == 1.75
    with pytest.raises(ValueError, match="do not use Pop III"):
        config(PAN_NSC, "--popiii-max-mass-msun", "400")
    with pytest.raises(ValueError, match="no independent NSC-mass grid"):
        config(PAN_PAPER, "--nsc-masses", "1e6,1e7")


def test_pan_total_mass_and_number_ratio_are_not_a_continuous_imf():
    nstar, nbh = pan_counts(1e5, 3e6)
    assert (nstar, nbh) == (2970300, 2970)
    assert nstar + 10 * nbh == 3e6
    assert nbh / nstar == pytest.approx(1e-3, abs=2e-7)
    assert 3e6 / (nstar + nbh) == pytest.approx(1.0089901018)
    np.random.seed(1)
    mass, clocks = published_masses(PAN_NSC, 100, 1010)
    assert set(mass) == {1, 10}
    assert (clocks[mass == 10] == 0).all()
    assert np.isinf(clocks[mass == 1]).all()
    assert mass.sum() == 1010


def test_pan_paper_normalization_and_xue_do_not_depend_on_nsc_input():
    assert pan_counts(1e5, 1e6, paper_normalization=True) == (2000000, 2000)
    assert pan_counts(1e5, 1e7, paper_normalization=True) == (2000000, 2000)
    np.random.seed(4)
    mass, clocks = published_masses(XUE_BH, 1e5, 3e6)
    assert len(mass) == 5000
    assert mass.min() >= 5 and mass.max() <= 15
    assert (clocks == 0).all()
    assert mass.sum() < 75000  # This is a BH component, not all of M_NSC.
    exponent = 1 - 2.35
    transformed = (mass**exponent - 5**exponent) / (15**exponent - 5**exponent)
    assert np.max(abs(np.sort(transformed) - (np.arange(5000) + .5) / 5000)) < .025
    assert not population_description(XUE_BH)["nsc_mass_input_used"]


def test_pan_orbits_recover_full_spatial_tremaine_cdf_not_an_a_cdf():
    np.random.seed(17)
    n = 60000
    minimum = 6 * const.G * 1e5 * const.msun / const.c**2
    a, e, inc, cosw, scale = pan_orbits(n, 1e5, minimum, 1.5)
    mean = np.random.uniform(0, 2*np.pi, n)
    anomaly = mean.copy()
    for _ in range(60):
        step = (anomaly - e*np.sin(anomaly) - mean) / (1 - e*np.cos(anomaly))
        anomaly -= np.clip(step, -1, 1)
    r = a * (1 - e*np.cos(anomaly))
    cdf = np.sort(tremaine_spatial_cdf(r, scale))
    assert np.max(abs(cdf - (np.arange(n) + .5) / n)) < .012
    assert abs(np.cos(inc).mean()) < .015
    assert (e*e).mean() == pytest.approx(.5, abs=.01)
    assert cosw.mean() == pytest.approx(0, abs=.015)
    assert cosw.std() == pytest.approx(np.sqrt(.5), abs=.015)
    assert a.min() >= minimum


def test_tremaine_is_broken_profile_and_xue_zero_index_is_shell_density():
    gamma = 1.5
    def density(r):
        return r**(-gamma) * (r + 1)**(gamma - 4)
    inner = np.log(density(2e-6)/density(1e-6)) / np.log(2)
    outer = np.log(density(2e6)/density(1e6)) / np.log(2)
    assert inner == pytest.approx(-1.5, abs=1e-4)
    assert outer == pytest.approx(-4, abs=1e-4)
    assert config(PAN_NSC).initial_stellar_density_gamma == 1.5
    assert config(XUE_BH).initial_stellar_density_gamma == 2
    assert config(LEGACY_POPIII).initial_stellar_density_gamma == 1.75
    np.random.seed(6)
    a, e, _, _, maximum = xue_orbits(50000, 1e5)
    assert (e == 0).all()  # Explicit circular-radius proxy: r=a.
    assert a.mean() == pytest.approx(maximum / 2, rel=.02)
    assert "NOT a uniform spatial density" in population_description(XUE_BH)["equivalent_spherical_density"]


def test_initial_bhs_and_stars_reach_workflow_with_correct_clocks():
    cfg = config(PAN_NSC)
    pop = sample_disk_crossing_population(1e5, 1010,
        {"inner_radius_cm": 1e12, "outer_radius_cm": 1e16}, cfg)
    assert np.isnan(pop.mean_relaxation_time_myr)
    masses, clocks = pop.initial["mass_msun"], pop.initial["formation_time_myr"]
    assert masses.sum() == 1010
    assert (clocks[masses == 10] == 0).all()
    assert np.isinf(clocks[masses == 1]).all()
    assert DampingCalculator._object_type_at_time("Star", 0, 0) == "BH"
    assert DampingCalculator._object_type_at_time("Star", 10 * const.Myr, np.inf) == "Star"
    metadata = build_metadata(cfg, {})["parameters"]
    assert metadata["population_model"] == PAN_NSC
    assert metadata["population_prescription"]["bh_to_star_number_ratio"] == .001
    with pytest.raises(ValueError, match="require fixed_mass"):
        config(PAN_NSC, "--stellar-evolution-model", "sevn", "--sevn-hydrogen-tables", ".", "--sevn-helium-tables", ".")


def test_preexisting_bh_records_are_not_mislabeled_as_zams_stars():
    from workflows.workers import damp_task_with_calculator
    cfg = config(PAN_NSC)
    class Calculator:
        def damping_simulation(self, mass_g, state, formation_time_myr, object_type):
            return SimpleNamespace(final_state=state, status=(DampingStatus.EMBEDDED_BH
                if formation_time_myr == 0 else DampingStatus.EMBEDDED_STAR))
    task = dict(semi_major_axis_cm=1e15, eccentricity=0, inclination_rad=.001,
                cos_omega=.5, state_time_s=0, task_kind="primary", idx=0,
                mass_g=10*const.msun, formation_time_myr=0)
    bh = damp_task_with_calculator(task, Calculator(), cfg)
    assert bh["status"] is DampingStatus.EMBEDDED_BH
    assert np.isnan(bh["zams_mass_msun"])
    assert bh["remnant_mass_msun"] == 10
    assert bh["stellar_outcome"] == "preexisting_bh"
    task.update(mass_g=const.msun, formation_time_myr=np.inf)
    star = damp_task_with_calculator(task, Calculator(), cfg)
    assert star["status"] is DampingStatus.EMBEDDED_STAR
    assert star["zams_mass_msun"] == 1
    assert np.isnan(star["remnant_mass_msun"])
    assert star["stellar_outcome"] == "representative_long_lived_star"


def test_paper_normalized_model_cannot_fake_an_independent_nsc_panel():
    spec = importlib.util.spec_from_file_location("published_comparison_driver", Path(__file__).resolve().parents[2] / "scripts/compare_host_rates.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    with redirect_stderr(StringIO()), pytest.raises(SystemExit):
        driver.main(["--population-model", XUE_BH])
    with redirect_stderr(StringIO()), pytest.raises(SystemExit):
        driver.main(["--population-model", PAN_PAPER])
