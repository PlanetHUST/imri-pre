"""Population IMF controls and their reproducible baseline."""
import numpy as np
import pytest
from physical_process.sampling import sample_salpeter, sample_efficient
from workflows.cli import build_parser, build_config_from_arguments
from workflows.host_mapping import config_with_overrides


def test_flat_log_imf_uses_uniform_logarithmic_quantiles():
    np.random.seed(92)
    u = np.random.rand(100)
    np.random.seed(92)
    sample = sample_salpeter(100, 13.6, 600., 1.)
    assert np.allclose(np.log(sample / 13.6) / np.log(600. / 13.6), u)


def test_default_sampling_preserves_historical_random_draws():
    np.random.seed(123)
    u = np.random.rand(100)
    old = (u * (197.**(-0.77) - 13.6**(-0.77)) + 13.6**(-0.77))**(-1/0.77)
    np.random.seed(123)
    assert np.array_equal(sample_salpeter(100, 13.6, 197., 1.77), old)


def test_imf_options_are_preserved_by_host_config_and_obey_bounds():
    config = build_config_from_arguments(build_parser().parse_args([
        '--popiii-max-mass-msun', '600', '--popiii-imf-slope', '1']))
    clone = config_with_overrides(config)
    assert clone.popiii_max_mass_msun == 600
    assert clone.popiii_imf_slope == 1
    np.random.seed(10)
    masses, _ = sample_efficient(2e5, 1e-5, popiii_max_mass_msun=600, popiii_imf_slope=1)
    assert masses.max() <= 600 and masses.max() > 245
    assert masses.sum() >= 2e5 and masses[:-1].sum() < 2e5
    with pytest.raises(ValueError):
        config_with_overrides(config, popiii_max_mass_msun=13.6)


def test_popiii_controls_do_not_change_popii_sampling():
    np.random.seed(77)
    baseline = sample_efficient(1000., 1e-3)
    np.random.seed(77)
    changed = sample_efficient(1000., 1e-3, popiii_max_mass_msun=600, popiii_imf_slope=1)
    assert all(np.array_equal(a, b) for a, b in zip(baseline, changed))
