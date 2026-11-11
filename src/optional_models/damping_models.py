"""Comparison adapters; the production solver in damp.py remains unchanged.

CGS inputs, radians, absolute seconds since disk turn-on. Radius means Keplerian
semimajor axis. Energy and L are TOTAL test-particle orbital quantities (erg,
g cm^2/s), not specific quantities; delta = final - initial, L is the magnitude.
The Xue adapter is an inclination-only, fixed-reference-radius calculation.
Its total orbital work/torque are unavailable, never reported as evaluated zero.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq

from physical_process import constants as c
from physical_process.agn import AGNProperties
from physical_process.damp import DampingCalculator, DampingStatus, OrbitalState

REPO_MODEL = "repo_two_node_direct_v1"
XUE_MODEL = "xue2026_inclination_fixed_reference_v1"
XUE_CALIBRATION = "xue2605.29305_eq2-3_whitehead2505.23899_eq42"
DIRECT = "direct_disk_damping"
SCATTERING = "direct_plus_effective_scattering"
PARSEC_CM = 3.085677581491367e18
# Rowan et al. 2505.23739 Sec 2.6: these dimensional values are rounded there.
REFERENCE_RADIUS_CM = 0.01 * PARSEC_CM
REFERENCE_MASS_MSUN = 25.0
REFERENCE_SMBH_MSUN = 1.e7
REFERENCE_H = 0.0057
REFERENCE_SIGMA = 1.6e4  # 1.6e5 kg/m^2 -> g/cm^2
XUE_A, XUE_B, XUE_C, XUE_KNEE = 0.67, -2.64, 1.80, 4.6


class OutcomeStatus(str, Enum):
    EMBEDDED = "embedded"
    CENSORED = "agn_shutdown_censored"
    TERMINATED = "physical_termination"
    ALIGNED_ECCENTRIC = "aligned_but_not_circular"
    OUT_OF_RANGE = "out_of_calibration_range"
    INVALID = "invalid_numerical_result"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class DampingInput:
    time_s: float
    radius_cm: float
    inclination_rad: float
    eccentricity: float
    compact_object_mass_g: float
    smbh_mass_g: float
    surface_density_g_cm2: float
    scale_height_cm: float
    sound_speed_cm_s: float
    relative_velocity_cm_s: float
    disk_lifetime_s: float
    source_weight: float = 1.0
    cos_omega: float = 0.5
    source_id: str = "0"
    host_id: str = "host0"
    source_channel: str = "matched_compact_object_fixture"
    seed: int = 0
    realization: int = 0
    disk_model_id: str = "published_local_state_fixture"
    viscosity_alpha: float = 0.1

    @property
    def aspect_ratio(self) -> float:
        return self.scale_height_cm / self.radius_cm

    @property
    def state(self) -> OrbitalState:
        return OrbitalState(self.radius_cm, self.eccentricity,
                            self.inclination_rad, self.cos_omega, self.time_s)

    def invalid_flags(self) -> tuple[str, ...]:
        numeric = (self.time_s, self.radius_cm, self.inclination_rad,
                   self.eccentricity, self.compact_object_mass_g, self.smbh_mass_g,
                   self.surface_density_g_cm2, self.scale_height_cm,
                   self.sound_speed_cm_s, self.relative_velocity_cm_s,
                   self.disk_lifetime_s, self.source_weight, self.cos_omega,
                   self.viscosity_alpha)
        if not np.all(np.isfinite(numeric)):
            return ("nonfinite_input",)
        if (min(self.radius_cm, self.compact_object_mass_g, self.smbh_mass_g,
                self.scale_height_cm, self.sound_speed_cm_s) <= 0
            or min(self.surface_density_g_cm2, self.relative_velocity_cm_s,
                   self.source_weight, self.time_s, self.disk_lifetime_s, self.viscosity_alpha) < 0
            or not 0 <= self.eccentricity < 1
            or not 0 <= self.inclination_rad <= np.pi
            or not -1 <= self.cos_omega <= 1
            or self.time_s > self.disk_lifetime_s):
            return ("invalid_input_domain",)
        return ()


@dataclass(frozen=True)
class Validity:
    applicable: bool
    flags: tuple[str, ...] = ()
    coverage: str = "unavailable"


@dataclass(frozen=True)
class LocalValue:
    value: float | None
    status: str
    units: str
    validity_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class DampingOutcome:
    status: OutcomeStatus
    t_embed_myr: float | None
    r_embed_cm: float | None
    inclination_at_embed_rad: float | None
    eccentricity_at_embed: float | None
    delta_energy: float | None
    delta_angular_momentum: float | None
    model_id: str
    calibration_id: str
    validity_flags: tuple[str, ...]
    completeness_flags: tuple[str, ...]
    final_state: OrbitalState
    history: tuple[OrbitalState, ...] = ()
    native_status: str = ""
    censor_time_myr: float | None = None
    configuration: str = DIRECT
    # Separate contribution: None means not calibrated, not physically zero.
    effective_scattering_di_dt_rad_s: float | None = None
    effective_scattering_delta_energy: float | None = None
    effective_scattering_delta_angular_momentum: float | None = None


def orbital_invariants(state: OrbitalState, x: DampingInput) -> tuple[float, float]:
    """SMBH-dominated Newtonian total orbital energy and |L|."""
    energy = -c.G * x.smbh_mass_g * x.compact_object_mass_g / (2 * state.semi_major_axis_cm)
    angular = x.compact_object_mass_g * np.sqrt(
        c.G * x.smbh_mass_g * state.semi_major_axis_cm * (1 - state.eccentricity**2))
    return float(energy), float(angular)


def _empty(x: DampingInput, model: str, calibration: str, status: OutcomeStatus,
           flags: tuple[str, ...], configuration: str = DIRECT) -> DampingOutcome:
    return DampingOutcome(status, None, None, None, None, None, None, model,
                          calibration, flags, ("not_evaluated",), x.state,
                          configuration=configuration)


def _configuration_flags(configuration: str) -> tuple[str, ...]:
    if configuration not in (DIRECT, SCATTERING):
        raise ValueError(f"unknown damping configuration: {configuration}")
    return (("effective_scattering_not_calibrated_in_cited_papers",)
            if configuration == SCATTERING else ())


class RepoDampingAdapter:
    model_id = REPO_MODEL
    calibration_id = "repository_native_two_node_equations"

    def __init__(self, calculator: DampingCalculator, configuration: str = DIRECT):
        self.calculator = calculator
        self.configuration = configuration
        self.config_flags = _configuration_flags(configuration)

    def validity_mask(self, x: DampingInput) -> Validity:
        flags = x.invalid_flags() + self.config_flags
        if not np.isclose(x.smbh_mass_g / c.msun,
                          self.calculator.agn_properties.smbh_mass_msun, rtol=1.e-12):
            flags += ("host_mass_mismatch",)
        return Validity(not flags, flags, "native_solver_no_hydro_calibration_claim")

    def inclination_damping_rate(self, x: DampingInput) -> LocalValue:
        validity = self.validity_mask(x)
        if not validity.applicable:
            return LocalValue(None, "unavailable", "rad/s", validity.flags)
        if self.calculator.inclination_is_damped(x.state):
            return LocalValue(0., "evaluated_inside_embedding_boundary", "rad/s")
        try:
            ts = self.calculator.timescale_calc.compute_all_timescales(
                x.compact_object_mass_g, x.state, "BH")
            if ts is None:
                return LocalValue(0., "evaluated_no_active_nodes", "rad/s")
            coeff = self.calculator.compute_ode_coefficients(
                x.state, *ts["drag"].drag_time_s, *ts["crossing_time_s"], ts["active_node_mask"])
            rate = float((coeff[2] + coeff[3]) / ts["orbital_period_s"])
            return LocalValue(rate if np.isfinite(rate) else None,
                              "evaluated" if np.isfinite(rate) else "invalid_numerical_result", "rad/s")
        except (ValueError, FloatingPointError, ZeroDivisionError):
            return LocalValue(None, "invalid_numerical_result", "rad/s")

    def evolve(self, x: DampingInput, *, max_iter: int = 50000,
               keep_history: bool = False) -> DampingOutcome:
        validity = self.validity_mask(x)
        if not validity.applicable:
            return _empty(x, self.model_id, self.calibration_id,
                          OutcomeStatus.UNAVAILABLE if self.config_flags else OutcomeStatus.INVALID,
                          validity.flags, self.configuration)
        # A comparison-only AGN wrapper prevents the native adaptive Euler loop
        # overshooting shutdown. No modification of the production integrator.
        calc = self.calculator
        deadline = min(x.disk_lifetime_s, calc.agn_properties.agn_lifetime_myr * c.Myr)
        states: list[OrbitalState] = []
        try:
            native = calc.damping_simulation(x.compact_object_mass_g, x.state, 0., "BH",
                                            max_iter=max_iter, history=states,
                                            stop_time_s=deadline)
            final = native.final_state
            flags: tuple[str, ...] = ()
            if len(states) > 1 and calc.inclination_is_damped(final):
                before = states[-2]
                def interpolated(f: float) -> OrbitalState:
                    return OrbitalState(*(getattr(before, key) + f * (getattr(final, key) - getattr(before, key))
                                          for key in ("semi_major_axis_cm", "eccentricity", "inclination_rad", "cos_omega", "time_s")))
                def boundary(f: float) -> float:
                    s = interpolated(f)
                    return s.inclination_rad - calc.inclination_limit_rad(s)
                if boundary(0.) > 0. and boundary(1.) <= 0.:
                    crossing = interpolated(brentq(boundary, 0., 1., xtol=1.e-12))
                    final = crossing.update(inclination_rad=calc.inclination_limit_rad(crossing))
                    states[-1] = final
                    flags += ("first_passage_interpolated_native_euler_step",)
            if calc.embedding_criteria(final).eligible and final.time_s <= deadline:
                status = OutcomeStatus.EMBEDDED
            elif native.status == DampingStatus.CALCULATION_ERROR:
                status = OutcomeStatus.INVALID
            elif native.status == DampingStatus.OUTSIDE_DISK or final.eccentricity >= 1:
                status = OutcomeStatus.TERMINATED
            elif calc.inclination_is_damped(final) and final.eccentricity > .01:
                status = OutcomeStatus.ALIGNED_ECCENTRIC
            elif final.time_s >= deadline:
                status = OutcomeStatus.CENSORED
            else:
                status = OutcomeStatus.INVALID
                flags += ("integration_stopped_before_boundary",)
            ei, li = orbital_invariants(x.state, x)
            ef, lf = orbital_invariants(final, x)
            embedded = status == OutcomeStatus.EMBEDDED
            # Terminal losses for censored cases remain explicitly labeled in raw output.
            return DampingOutcome(status, final.time_s / c.Myr if embedded else None,
                final.semi_major_axis_cm if embedded else None,
                final.inclination_rad if embedded else None, final.eccentricity if embedded else None,
                ef - ei, lf - li, self.model_id, self.calibration_id, flags,
                ("full_native_orbit", "losses_at_embedding" if embedded else "losses_at_last_evaluated_state"),
                final, tuple(states) if keep_history else (), native.status.value,
                (deadline / c.Myr if status == OutcomeStatus.CENSORED else
                 final.time_s / c.Myr if status == OutcomeStatus.ALIGNED_ECCENTRIC else None),
                self.configuration)
        except (ValueError, FloatingPointError, ZeroDivisionError) as exc:
            return _empty(x, self.model_id, self.calibration_id, OutcomeStatus.INVALID,
                          (f"{type(exc).__name__}:{exc}",), self.configuration)

    def damping_time(self, x: DampingInput) -> LocalValue:
        outcome = self.evolve(x)
        value = ((outcome.t_embed_myr - x.time_s / c.Myr) * c.Myr
                 if outcome.t_embed_myr is not None else None)
        return LocalValue(value, outcome.status.value, "s", outcome.validity_flags)


class XueDampingAdapter:
    """Xue Eq(2)-(3), using the base-10 convention of Whitehead Eq(42).

    Restricted application of the combined fit: Rowan Sec2.6 reference local
    state, Sigma in [0.25,2] Sigma0, m=25 Msun, M=1e7 Msun, circular,
    prograde, i<=15deg. Interpolation between density suites and continuation
    through the Whitehead low-i branch are flagged modeling assumptions, not
    direct hydrodynamic coverage of every joint point. Other local states are
    unavailable. Rowan evolves inviscid gas: no alpha-dependent correction is
    calibrated. Equality tolerances only absorb floating-point roundoff.
    """
    model_id = XUE_MODEL
    calibration_id = XUE_CALIBRATION

    def __init__(self, configuration: str = DIRECT):
        self.configuration = configuration
        self.config_flags = _configuration_flags(configuration)

    def validity_mask(self, x: DampingInput) -> Validity:
        invalid = x.invalid_flags()
        if invalid or self.config_flags:
            return Validity(False, invalid + self.config_flags)
        flags = []
        checks = ((x.compact_object_mass_g / c.msun, REFERENCE_MASS_MSUN, "mass_not_in_verified_hydro_subset"),
                  (x.smbh_mass_g / c.msun, REFERENCE_SMBH_MSUN, "host_not_in_verified_hydro_subset"),
                  (x.radius_cm, REFERENCE_RADIUS_CM, "radius_not_in_verified_hydro_subset"),
                  (x.aspect_ratio, REFERENCE_H, "aspect_ratio_not_in_verified_hydro_subset"),
                  (x.sound_speed_cm_s, 1.19e6, "sound_speed_not_in_verified_reference_state"))
        for value, reference, flag in checks:
            if not np.isclose(value, reference, rtol=1.e-10, atol=0.):
                flags.append(flag)
        if x.eccentricity != 0.:
            flags.append("eccentric_orbit_not_calibrated")
        if not 0 <= x.inclination_rad <= np.deg2rad(15.):
            flags.append("inclination_not_in_prograde_hydro_range")
        if not .25 * REFERENCE_SIGMA <= x.surface_density_g_cm2 <= 2 * REFERENCE_SIGMA:
            flags.append("surface_density_not_in_verified_hydro_subset")
        # Whitehead Fig5 labels the low-i fitted environments: mtilde =
        # .005,.012,.023,.031,.058,.061,.169,.199 (the .115 stall is excluded).
        # Restrict the ENTIRE path to the overlap with this branch, since an
        # embedding prediction must traverse it; do not extrapolate to .341.
        if not flags and not .005 <= self.hill_mass_ratio(x) <= .199:
            flags.append("hill_mass_outside_verified_low_i_branch_span_for_embedding")
        if flags:
            return Validity(False, tuple(flags), "outside_verified_subset_not_necessarily_outside_all_published_data")
        advisory = ("rounded_published_reference_state", "combined_fit_low_i_transfer_assumption",
                    "continuous_density_interpolation_not_individually_simulated",
                    "rowan_inviscid_no_alpha_response_calibrated",
                    "fixed_reference_radius_no_calibrated_orbital_closure")
        return Validity(True, advisory, "conditional_combined_fit_application_not_full_joint_hydro_coverage")

    @staticmethod
    def hill_mass_ratio(x: DampingInput) -> float:
        rh = x.radius_cm * (x.compact_object_mass_g / (3 * x.smbh_mass_g))**(1 / 3)
        return float(2 * np.pi * rh**2 * x.surface_density_g_cm2 / x.compact_object_mass_g)

    @staticmethod
    def _fractional_loss(x: DampingInput, inclination: float) -> float:
        scaled = max(np.sin(inclination) / x.aspect_ratio, XUE_KNEE)
        return float(10**XUE_C * XueDampingAdapter.hill_mass_ratio(x)**XUE_A * scaled**XUE_B)

    @staticmethod
    def effective_crossing_probability(x: DampingInput) -> LocalValue:
        """Xue Eq8 time occupancy, NOT a scattering probability or damping factor."""
        if x.invalid_flags():
            return LocalValue(None, "invalid_input", "dimensionless", x.invalid_flags())
        sine = abs(np.sin(x.inclination_rad))
        value = 1. if sine <= x.aspect_ratio else 2 / np.pi * np.arcsin(x.aspect_ratio / sine)
        return LocalValue(float(value), "evaluated_geometric_occupancy_only", "dimensionless")

    def inclination_damping_rate(self, x: DampingInput) -> LocalValue:
        validity = self.validity_mask(x)
        if not validity.applicable:
            return LocalValue(None, "out_of_range" if not self.config_flags else "unavailable",
                              "rad/s", validity.flags)
        if x.inclination_rad <= x.aspect_ratio:
            return LocalValue(0., "evaluated_inside_embedding_boundary", "rad/s", validity.flags)
        half_period = np.pi * np.sqrt(x.radius_cm**3 / (c.G * x.smbh_mass_g))
        value = -x.inclination_rad * self._fractional_loss(x, x.inclination_rad) / half_period
        return LocalValue(float(value), "evaluated", "rad/s", validity.flags)

    def damping_time(self, x: DampingInput) -> LocalValue:
        """Uncensored first-passage duration at the fixed local reference state."""
        validity = self.validity_mask(x)
        if not validity.applicable:
            return LocalValue(None, "unavailable" if self.config_flags else "out_of_range", "s", validity.flags)
        if x.inclination_rad <= x.aspect_ratio:
            return LocalValue(0., "evaluated", "s", validity.flags)
        half_period = np.pi * np.sqrt(x.radius_cm**3 / (c.G * x.smbh_mass_g))
        lower, upper = np.log(x.aspect_ratio), np.log(x.inclination_rad)
        knee = np.arcsin(XUE_KNEE * x.aspect_ratio)
        points = [np.log(knee)] if x.aspect_ratio < knee < x.inclination_rad else None
        duration = half_period * quad(lambda log_i: 1 / self._fractional_loss(x, np.exp(log_i)),
                                      lower, upper, points=points, epsabs=0., epsrel=1.e-9)[0]
        return LocalValue(float(duration), "evaluated", "s", validity.flags)

    def evolve(self, x: DampingInput, *, keep_history: bool = False, **_) -> DampingOutcome:
        validity = self.validity_mask(x)
        if not validity.applicable:
            status = (OutcomeStatus.INVALID if x.invalid_flags() else
                      OutcomeStatus.UNAVAILABLE if self.config_flags else OutcomeStatus.OUT_OF_RANGE)
            return _empty(x, self.model_id, self.calibration_id, status, validity.flags, self.configuration)
        duration = self.damping_time(x).value
        assert duration is not None
        available = x.disk_lifetime_s - x.time_s
        embedded = duration <= available * (1 + 1.e-12)
        stop = min(duration, available)
        if duration == 0.:
            states = (x.state,)
        else:
            half_period = np.pi * np.sqrt(x.radius_cm**3 / (c.G * x.smbh_mass_g))
            def rhs(_, log_i):
                return [-duration / half_period * self._fractional_loss(x, np.exp(log_i[0]))]
            sol = solve_ivp(rhs, (0., stop / duration), [np.log(x.inclination_rad)],
                            rtol=2.e-9, atol=1.e-11, max_step=.02, dense_output=True)
            if not sol.success:
                return _empty(x, self.model_id, self.calibration_id, OutcomeStatus.INVALID, (sol.message,))
            ts = np.linspace(0., stop / duration, 81 if keep_history else 2)
            states = tuple(x.state.update(time_s=x.time_s + float(t * duration),
                                         inclination_rad=float(np.exp(sol.sol(t)[0]))) for t in ts)
            if embedded:
                states = (*states[:-1], states[-1].update(inclination_rad=x.aspect_ratio))
        final = states[-1]
        return DampingOutcome(OutcomeStatus.EMBEDDED if embedded else OutcomeStatus.CENSORED,
            final.time_s / c.Myr if embedded else None, x.radius_cm if embedded else None,
            final.inclination_rad if embedded else None, x.eccentricity if embedded else None,
            None, None, self.model_id, self.calibration_id, validity.flags,
            ("inclination_only", "radius_held_fixed_reference_assumption", "eccentricity_fixed_zero_calibration",
             "total_energy_and_angular_momentum_unavailable"),
            final, states if keep_history else (), "inclination_fit",
            x.disk_lifetime_s / c.Myr if not embedded else None, self.configuration)


def make_local_repo_calculator(x: DampingInput) -> DampingCalculator:
    """Uniform local-state TEST FIXTURE, not a new production disk family.

    Sigma=2 rho H uses the repository one-zone convention at every grid point.
    Sigma and H/R are constant, so rho scales as 1/r. This controlled fixture
    isolates drag laws; it is not a new production disk model. The separate
    pAGN audit uses solved radial profiles and their solved midplane density.
    """
    radii = np.unique(np.r_[np.geomspace(x.radius_cm * 1.e-3, x.radius_cm * 10, 256), x.radius_cm])
    return DampingCalculator(AGNProperties.from_mass(x.smbh_mass_g / c.msun,
        x.disk_lifetime_s / c.Myr), False, radii,
        np.full_like(radii, x.surface_density_g_cm2), np.full_like(radii, x.aspect_ratio),
        float(radii[0]), float(radii[-1]), density_g_cm3=(
            x.surface_density_g_cm2 / (2 * x.aspect_ratio * radii)))
