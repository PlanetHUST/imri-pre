"""Default single-host contract: original IMF and inward Type-I migration."""

from types import SimpleNamespace
from contextlib import redirect_stderr
from io import StringIO

import numpy as np
from physical_process import constants as const
from physical_process.migration.type_i import type_i_local_time_s
from workflows.cli import build_config_from_arguments, build_parser
from workflows.config import SEMI_TYPE_I
from workflows.event_work import IMRIEventEvaluator


def test_default_cli_selects_type_i_and_no_sevn():
    parser = build_parser()
    config = build_config_from_arguments(parser.parse_args([]))
    assert config.disk_model == SEMI_TYPE_I
    assert config.stellar_evolution_model == "fixed_mass"
    assert config.natal_kick_model == "none"
    assert not any(action.dest == "tree" for action in parser._actions)


def test_type_i_drift_ignores_the_optional_type_ii_flag():
    radius = np.array([1e12, 2e12, 3e12])
    sigma = np.full(3, 1e4)
    aspect = np.full(3, 0.05)
    mass_msun = 100.0
    config = SimpleNamespace(
        disk_model=SEMI_TYPE_I, gap_opening_condition="kii",
        consider_type_ii_migration=True,
    )
    evaluator = IMRIEventEvaluator(SimpleNamespace(config=config, duty_cycle_myr=10.0))
    disk = {
        "radius_cm": radius, "surface_density_g_cm2": sigma,
        "aspect_ratio": aspect, "torque_alpha": 0.1,
    }
    returned_radius, drift = evaluator._gas_drift_for_mass(
        SimpleNamespace(m_smbh_msun=1e5),
        SimpleNamespace(smbh_mass_msun=1e5), disk, mass_msun,
    )
    type_i_time = type_i_local_time_s(
        radius, sigma, aspect, mass_msun * const.msun, 1e5,
    )
    assert np.array_equal(returned_radius, radius)
    np.testing.assert_allclose(drift, -radius / type_i_time)
    assert np.all(drift < 0)


def test_tree_flag_is_rejected():
    try:
        with redirect_stderr(StringIO()):
            build_parser().parse_args(["--tree", "anything.csv"])
    except SystemExit as error:
        assert error.code != 0
    else:
        raise AssertionError("tree mode must not be available")
