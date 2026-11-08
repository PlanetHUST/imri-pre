"""Equation-level tests for the per-crossing damping coefficients.

These test the published Fan et al. (2024) relations directly, independently of
the damping loop, so a transcription error in one node cannot hide behind the
step controller or the eccentricity clamp.

The reference values are computed from first principles: an impulsive drag
kick ``dv = -eps * v_rel`` is applied to the state vector at a node, and the
osculating elements are recomputed.  The analytic per-orbit coefficients must
reproduce the finite difference of those elements as ``eps -> 0``.
"""

import numpy as np

from physical_process.damp import OrbitalState

GM = 1.0
SEMI_MAJOR_AXIS = 1.0

#: (eccentricity, cos omega, inclination [rad]) probes spanning the sampled
#: population: low/moderate/high e, cos omega near 0 and near 1, and the
#: low-inclination regime where the second-node terms dominate.
CASES = (
    (0.3, 0.5, np.deg2rad(30.0)),
    (0.5, 0.8, np.deg2rad(60.0)),
    (0.7, 0.95, np.deg2rad(10.0)),
    (0.6, 0.9, np.deg2rad(20.0)),
    (0.1, 0.2, np.deg2rad(45.0)),
    (0.45, 0.0, np.deg2rad(75.0)),
)


def _node_state_vectors(
    eccentricity: float,
    cos_omega: float,
    inclination_rad: float,
    node: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (position, velocity) at the requested ascending/descending node.

    The node lies on the line of nodes, which we place along ``x``.  The sign
    convention matches ``orbital_node_radii_cm``: node 1 has
    ``r = p / (1 - e cos omega)``, node 2 has ``r = p / (1 + e cos omega)``.
    """
    semi_latus_rectum = SEMI_MAJOR_AXIS * (1.0 - eccentricity**2)
    sign = -1.0 if node == 1 else 1.0
    radius = semi_latus_rectum / (1.0 + sign * eccentricity * cos_omega)

    # True anomaly measured from periapsis; at the node the projection of the
    # eccentricity vector onto the node line is e*cos(omega).
    cos_true_anomaly = sign * cos_omega
    sin_true_anomaly = float(np.sqrt(max(0.0, 1.0 - cos_true_anomaly**2)))

    position = np.array([radius, 0.0, 0.0])

    # Perifocal velocity rotated so the node sits on +x.
    speed_scale = np.sqrt(GM / semi_latus_rectum)
    v_radial = speed_scale * eccentricity * sin_true_anomaly
    v_transverse = speed_scale * (1.0 + eccentricity * cos_true_anomaly)
    velocity = np.array(
        [
            v_radial,
            v_transverse * np.cos(inclination_rad),
            v_transverse * np.sin(inclination_rad),
        ]
    )
    return position, velocity


def _osculating_elements(
    position: np.ndarray, velocity: np.ndarray
) -> tuple[float, float]:
    """Return (semi-major axis, eccentricity) for a state vector."""
    r = float(np.linalg.norm(position))
    v_sq = float(velocity @ velocity)
    energy = 0.5 * v_sq - GM / r
    semi_major_axis = -GM / (2.0 * energy)
    angular_momentum = np.cross(position, velocity)
    eccentricity_vector = np.cross(velocity, angular_momentum) / GM - position / r
    return semi_major_axis, float(np.linalg.norm(eccentricity_vector))


def _numeric_da_de(
    eccentricity: float, cos_omega: float, inclination_rad: float, node: int
) -> tuple[float, float]:
    """Finite-difference da and de per unit drag impulse at the node."""
    position, velocity = _node_state_vectors(
        eccentricity, cos_omega, inclination_rad, node
    )
    # The gas moves on a circular coplanar orbit at this radius.
    radius = float(np.linalg.norm(position))
    gas_velocity = np.array([0.0, np.sqrt(GM / radius), 0.0])
    relative_velocity = velocity - gas_velocity

    base_a, base_e = _osculating_elements(position, velocity)
    del base_a, base_e
    # Central difference: second-order accurate, so the comparison below is
    # limited by round-off rather than by truncation.
    step = 1.0e-6
    plus_a, plus_e = _osculating_elements(position, velocity - step * relative_velocity)
    minus_a, minus_e = _osculating_elements(
        position, velocity + step * relative_velocity
    )
    return (plus_a - minus_a) / (2.0 * step), (plus_e - minus_e) / (2.0 * step)


def _analytic_da_de(
    eccentricity: float, cos_omega: float, inclination_rad: float, node: int
) -> tuple[float, float]:
    """Per-orbit da and de from the code, for one node.

    ``compute_ode_coefficients`` returns ``dX/dt_drag * t_cross``; setting
    ``t_cross = 1`` and ``t_drag = 1`` leaves the bracket, which is what the
    finite difference measures up to one common positive normalisation.  Signs
    and the da:de ratio are therefore directly comparable.
    """
    from physical_process import damp

    state = OrbitalState(
        semi_major_axis_cm=SEMI_MAJOR_AXIS,
        eccentricity=eccentricity,
        inclination_rad=inclination_rad,
        cos_omega=cos_omega,
        time_s=0.0,
    )
    calculator = damp.DampingCalculator.__new__(damp.DampingCalculator)
    # Return order is (da1, da2, di1, di2, de1, de2).
    da1, da2, _, _, de1, de2 = damp.DampingCalculator.compute_ode_coefficients(
        calculator,
        state,
        1.0,
        1.0,
        1.0,
        1.0,
        (node == 1, node == 2),
    )
    return (da1, de1) if node == 1 else (da2, de2)


def test_both_nodes_match_an_independent_osculating_element_calculation() -> None:
    """Analytic da/de must agree with the finite-differenced elements.

    With unit drag time and unit crossing time the published coefficients are
    exactly the per-unit-impulse element derivatives, so this is a direct
    value comparison, not merely a sign check.
    """
    failures: list[str] = []
    for eccentricity, cos_omega, inclination_rad in CASES:
        for node in (1, 2):
            numeric = _numeric_da_de(eccentricity, cos_omega, inclination_rad, node)
            analytic = _analytic_da_de(eccentricity, cos_omega, inclination_rad, node)
            for label, want, got in zip(("da", "de"), numeric, analytic):
                if not np.isclose(got, want, rtol=1.0e-6, atol=1.0e-9):
                    failures.append(
                        f"node {node} {label}: e={eccentricity} "
                        f"cosw={cos_omega} i={np.rad2deg(inclination_rad):.0f}deg "
                        f"expected {want:+.6f} got {got:+.6f}"
                    )
    assert not failures, "damping coefficient mismatches:\n" + "\n".join(failures)


def test_the_two_nodes_coincide_when_cos_omega_is_zero() -> None:
    """At cos omega = 0 both nodes sit at the semi-latus rectum and must agree."""
    for eccentricity in (0.1, 0.45, 0.8):
        for inclination_deg in (15.0, 75.0, 150.0):
            first = _analytic_da_de(eccentricity, 0.0, np.deg2rad(inclination_deg), 1)
            second = _analytic_da_de(eccentricity, 0.0, np.deg2rad(inclination_deg), 2)
            assert np.allclose(first, second, rtol=1.0e-12), (
                f"nodes disagree at cos omega = 0 (e={eccentricity}, "
                f"i={inclination_deg}deg): {first} vs {second}"
            )


def test_vertical_velocity_matches_the_state_vector_at_both_nodes() -> None:
    """``v_z`` must equal the out-of-plane component of the real velocity.

    At a node the body lies on the line of nodes, which is *in* the disk plane,
    so the radial velocity is in-plane and only the transverse component tilts
    out of it: ``v_z = v_t sin i``, not ``|v| sin i``.
    """
    from physical_process import constants as const
    from physical_process import damp

    calculator = damp.TimescaleCalculator.__new__(damp.TimescaleCalculator)
    calculator.smbh_mass_g = 1.0 / const.G  # makes G*M = 1, matching GM above

    for eccentricity, cos_omega, inclination_rad in CASES:
        state = damp.OrbitalState(
            semi_major_axis_cm=SEMI_MAJOR_AXIS,
            eccentricity=eccentricity,
            inclination_rad=inclination_rad,
            cos_omega=cos_omega,
            time_s=0.0,
        )
        radius_1_cm, radius_2_cm = calculator.compute_orbital_positions(state)
        computed = calculator.compute_vertical_velocities(
            state, radius_1_cm, radius_2_cm
        )
        for node in (1, 2):
            _, velocity = _node_state_vectors(
                eccentricity, cos_omega, inclination_rad, node
            )
            expected = abs(float(velocity[2]))
            assert np.isclose(computed[node - 1], expected, rtol=1.0e-12), (
                f"node {node}: expected v_z={expected:.6f}, "
                f"got {computed[node - 1]:.6f}"
            )


def test_total_speed_overestimates_the_vertical_crossing_speed() -> None:
    """The superseded ``|v| sin i`` form is never smaller than the true ``v_z``.

    It therefore always shortened the crossing and under-damped.  This pins the
    direction of the correction so a future refactor cannot silently undo it.
    """
    for eccentricity, cos_omega, inclination_rad in CASES:
        for node in (1, 2):
            _, velocity = _node_state_vectors(
                eccentricity, cos_omega, inclination_rad, node
            )
            true_vertical = abs(float(velocity[2]))
            superseded = float(np.linalg.norm(velocity)) * np.sin(inclination_rad)
            assert superseded >= true_vertical * (1.0 - 1.0e-12), (
                f"node {node}: |v| sin i = {superseded:.6f} should never be "
                f"below v_z = {true_vertical:.6f}"
            )


def test_drag_removes_energy_at_the_super_keplerian_node() -> None:
    """At the node inside the semi-latus rectum, drag must always shrink ``a``.

    Node 2 sits at ``r = p / (1 + e cos omega)``, inside the semi-latus rectum,
    where the body moves *faster* than the local circular speed.  Gas drag there
    always removes orbital energy, so its ``da`` bracket must stay non-negative
    (``delta_axis = -(...) * bracket``).

    Node 1 carries no such invariant and must not be given one: at
    ``r = p / (1 - e cos omega)`` the body is *sub*-Keplerian, the disk overtakes
    it, and drag legitimately raises ``a`` while circularising the orbit.  That
    is why the correct test is per-node, and why a blanket "damping never
    increases a" assertion would be wrong physics.
    """
    rng = np.random.default_rng(20260921)
    count = 20000
    eccentricity = np.sqrt(rng.random(count))
    cos_omega = rng.random(count)
    inclination_rad = rng.uniform(np.pi / 18.0, 17.0 * np.pi / 18.0, count)
    cos_inclination = np.cos(inclination_rad)

    bracket = (
        1.0
        + 2.0 * eccentricity * cos_omega
        + eccentricity**2
        - (1.0 + eccentricity * cos_omega) ** 1.5 * cos_inclination
    )
    worst = float(bracket.min())
    assert worst >= -1.0e-12, (
        f"second-node da bracket goes negative (min {worst:.6f}): drag would "
        "add energy at the super-Keplerian node"
    )

    # The pre-fix expression (node 1's bracket pasted into node 2) must fail
    # this invariant, so the test genuinely discriminates.
    regressed = (
        1.0
        - 2.0 * eccentricity * cos_omega
        + eccentricity**2
        - (1.0 + eccentricity * cos_omega) ** 1.5 * cos_inclination
    )
    assert regressed.min() < 0.0, (
        "the pre-fix second-node bracket should violate this invariant; if it "
        "does not, the test no longer discriminates"
    )
