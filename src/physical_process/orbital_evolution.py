"""One circular radial evolution under simultaneous gas and GW losses.

The disk and component masses are static. The gas drift is supplied by one
explicit migration scenario; this module adds the circular Peters drift and
finds the first observed lower-band crossing without a gas/GW crossover rule.
"""

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from . import constants as const
from .gw_frequency_radius import gw_frequency_hz
from .redshift_evolve import RedshiftEvolution

ISCO_RADIUS_RG = 6.0
ISCO_CONSUMPTION_FREQUENCY_MARGIN = 1.01
ORBIT_RELATIVE_TOLERANCE = 1.0e-9
ORBIT_LOG_RADIUS_ABSOLUTE_TOLERANCE = 1.0e-11


@dataclass(frozen=True)
class CircularOrbitOutcome:
    """First-band-entry result; times are since the host's initial redshift."""

    status: str
    entry_time_myr: float
    entry_radius_cm: float
    entry_redshift: float
    final_time_myr: float
    final_radius_cm: float
    final_redshift: float
    disk_off_time_myr: float
    disk_off_radius_cm: float
    disk_off_reason: str
    gas_active_at_entry: bool | None


@dataclass(frozen=True)
class CircularBandTrack:
    """Continuous post-entry orbit segment, including any physical disk-off."""

    status: str
    entry_time_myr: float
    entry_radius_cm: float
    end_time_myr: float
    end_radius_cm: float
    end_redshift: float
    gas_end_time_myr: float
    gas_end_radius_cm: float
    beta_cm4_s: float
    gas_log_radius_solution: Callable[[float], np.ndarray] | None
    initial_redshift: float
    total_mass_msun: float
    redshift_model: RedshiftEvolution

    def radius_cm_at(self, time_myr: float) -> float:
        """Evaluate the same coupled trajectory within its saved time domain."""
        if not self.entry_time_myr <= time_myr <= self.end_time_myr:
            raise ValueError("time lies outside the in-band orbit track")
        if (
            self.gas_log_radius_solution is not None
            and time_myr <= self.gas_end_time_myr
        ):
            return float(np.exp(self.gas_log_radius_solution(time_myr)[0]))
        fourth_power = self.gas_end_radius_cm**4 - 4.0 * self.beta_cm4_s * (
            time_myr - self.gas_end_time_myr
        ) * const.Myr
        return float(max(0.0, fourth_power) ** 0.25)

    def observed_frequency_hz_at(self, time_myr: float) -> float:
        redshift = self.redshift_model.redshift_evlove_with_time(
            self.initial_redshift, time_myr
        )
        return gw_frequency_hz(self.total_mass_msun, self.radius_cm_at(time_myr)) / (
            1.0 + redshift
        )


def _coupled_radial_drift_cm_s(
    radius_cm: float,
    beta_cm4_s: float,
    gas_radius_cm: np.ndarray,
    gas_drift_cm_s: np.ndarray,
) -> float:
    """Single signed circular law shared by entry and in-band continuation."""
    return float(np.interp(radius_cm, gas_radius_cm, gas_drift_cm_s)) - (
        beta_cm4_s / radius_cm**3
    )


def peters_beta_cm4_s(
    primary_mass_msun: float, secondary_mass_msun: float
) -> float:
    """Return circular Peters ``beta`` in cm^4/s."""
    primary_mass_g = primary_mass_msun * const.msun
    secondary_mass_g = secondary_mass_msun * const.msun
    if not np.isfinite(primary_mass_g) or not np.isfinite(secondary_mass_g):
        raise ValueError("binary masses must be finite")
    if primary_mass_g <= 0.0 or secondary_mass_g <= 0.0:
        raise ValueError("binary masses must be positive")
    return float(
        64.0
        / 5.0
        * const.G**3
        * primary_mass_g
        * secondary_mass_g
        * (primary_mass_g + secondary_mass_g)
        / const.c**5
    )


def evolve_circular_orbit(
    *,
    primary_mass_msun: float,
    secondary_mass_msun: float,
    initial_radius_cm: float,
    initial_time_myr: float,
    initial_redshift: float,
    duty_cycle_myr: float,
    disk_inner_radius_cm: float,
    gas_radius_cm: np.ndarray,
    gas_drift_cm_s: np.ndarray,
    min_observed_frequency_hz: float,
    redshift_model: RedshiftEvolution,
    stop_time_myr: float | None = None,
    solver_method: str = "DOP853",
) -> CircularOrbitOutcome:
    """Evolve the circular orbit to first band entry, ISCO, or the z=0 limit.

    ``gas_drift_cm_s`` is the *signed* gas-only ``dr/dt`` on a sorted radial
    profile. The two contributions add while the object is inside the disk
    and the AGN is active; after gas ceases, the Peters radius is analytic.
    """
    if solver_method not in ("DOP853", "Radau"):
        raise ValueError("orbit solver_method must be DOP853 or Radau")
    radius_grid = np.asarray(gas_radius_cm, dtype=float)
    gas_drift = np.asarray(gas_drift_cm_s, dtype=float)
    if (
        radius_grid.ndim != 1
        or gas_drift.ndim != 1
        or radius_grid.size < 2
        or radius_grid.shape != gas_drift.shape
        or not np.all(np.isfinite(radius_grid) & (radius_grid > 0.0))
        or not np.all(np.diff(radius_grid) > 0.0)
        or not np.all(np.isfinite(gas_drift))
    ):
        raise ValueError("gas drift requires finite, ordered radial samples")
    if not np.isfinite(initial_radius_cm) or initial_radius_cm <= 0.0:
        raise ValueError("initial radius must be finite and positive")
    if not np.isfinite(initial_time_myr) or initial_time_myr < 0.0:
        raise ValueError("initial time must be finite and nonnegative")
    if not np.isfinite(duty_cycle_myr) or duty_cycle_myr <= 0.0:
        raise ValueError("AGN duty cycle must be finite and positive")
    if not np.isfinite(disk_inner_radius_cm) or disk_inner_radius_cm <= 0.0:
        raise ValueError("disk inner radius must be finite and positive")
    if not np.isfinite(min_observed_frequency_hz) or min_observed_frequency_hz <= 0:
        raise ValueError("minimum observed frequency must be positive")
    if stop_time_myr is not None and (
        not np.isfinite(stop_time_myr) or stop_time_myr < initial_time_myr
    ):
        raise ValueError("event stop time must be finite and no earlier than orbit start")

    total_mass_msun = primary_mass_msun + secondary_mass_msun
    beta_cm4_s = peters_beta_cm4_s(primary_mass_msun, secondary_mass_msun)
    isco_radius_cm = (
        ISCO_RADIUS_RG * const.G * total_mass_msun * const.msun / const.c**2
    )
    gas_inner_cm = max(disk_inner_radius_cm, float(radius_grid[0]))
    gas_outer_cm = float(radius_grid[-1])
    cosmic_limit_myr = redshift_model.remaining_time_to_zero_redshift_myr(
        initial_redshift
    )
    event_limit_myr = (
        min(cosmic_limit_myr, stop_time_myr)
        if stop_time_myr is not None else cosmic_limit_myr
    )

    def redshift_at(time_myr: float) -> float:
        return redshift_model.redshift_evlove_with_time(initial_redshift, time_myr)

    def observed_frequency_hz(radius_cm: float, time_myr: float) -> float:
        return gw_frequency_hz(total_mass_msun, radius_cm) / (
            1.0 + redshift_at(time_myr)
        )

    def outcome(
        status: str,
        time_myr: float,
        radius_cm: float,
        disk_off_time_myr: float,
        disk_off_radius_cm: float,
        gas_active_at_entry: bool | None = None,
        disk_off_reason: str = "not_reached",
    ) -> CircularOrbitOutcome:
        reached_entry = status in {"band_entry", "already_inside_band"}
        return CircularOrbitOutcome(
            status=status,
            entry_time_myr=time_myr if reached_entry else np.inf,
            entry_radius_cm=radius_cm if reached_entry else np.nan,
            entry_redshift=redshift_at(time_myr) if reached_entry else np.nan,
            final_time_myr=time_myr,
            final_radius_cm=radius_cm,
            final_redshift=redshift_at(time_myr),
            disk_off_time_myr=disk_off_time_myr,
            disk_off_radius_cm=disk_off_radius_cm,
            disk_off_reason=disk_off_reason,
            gas_active_at_entry=gas_active_at_entry if reached_entry else None,
        )

    if initial_radius_cm <= isco_radius_cm:
        return outcome(
            "initial_orbit_at_or_inside_isco",
            initial_time_myr,
            initial_radius_cm,
            np.nan,
            np.nan,
        )
    if initial_time_myr >= cosmic_limit_myr:
        return outcome(
            "cosmic_time_limit",
            initial_time_myr,
            initial_radius_cm,
            np.nan,
            np.nan,
        )
    if stop_time_myr is not None and initial_time_myr >= stop_time_myr:
        return outcome(
            "stopped_at_event", initial_time_myr, initial_radius_cm,
            np.nan, np.nan,
        )

    if observed_frequency_hz(initial_radius_cm, initial_time_myr) >= (
        min_observed_frequency_hz
    ):
        initially_in_gas = (
            initial_time_myr < duty_cycle_myr
            and gas_inner_cm <= initial_radius_cm <= gas_outer_cm
        )
        return outcome(
            "already_inside_band",
            initial_time_myr,
            initial_radius_cm,
            np.nan if initially_in_gas else initial_time_myr,
            np.nan if initially_in_gas else initial_radius_cm,
            initially_in_gas,
            "gas_active_at_entry" if initially_in_gas else "already_outside_gas",
        )

    # A GW-driven object may start beyond the tabulated disk and enter its
    # outer edge while the AGN is still on. Do not extrapolate gas there.
    if (
        initial_radius_cm > gas_outer_cm > isco_radius_cm
        and initial_time_myr < duty_cycle_myr
    ):
        time_at_outer_myr = initial_time_myr + (
            initial_radius_cm**4 - gas_outer_cm**4
        ) / (4.0 * beta_cm4_s * const.Myr)
        if time_at_outer_myr < min(duty_cycle_myr, event_limit_myr):
            def outer_vacuum_radius_cm(time_myr: float) -> float:
                fourth_power = initial_radius_cm**4 - 4.0 * beta_cm4_s * (
                    time_myr - initial_time_myr
                ) * const.Myr
                return float(max(gas_outer_cm**4, fourth_power) ** 0.25)

            if observed_frequency_hz(gas_outer_cm, time_at_outer_myr) >= (
                min_observed_frequency_hz
            ):
                entry_time_myr = brentq(
                    lambda time_myr: (
                        observed_frequency_hz(
                            outer_vacuum_radius_cm(time_myr), time_myr
                        )
                        / min_observed_frequency_hz
                        - 1.0
                    ),
                    initial_time_myr,
                    time_at_outer_myr,
                    xtol=1.0e-15,
                )
                return outcome(
                    "band_entry",
                    entry_time_myr,
                    outer_vacuum_radius_cm(entry_time_myr),
                    initial_time_myr,
                    initial_radius_cm,
                    False,
                    "entry_before_disk_contact",
                )
            return evolve_circular_orbit(
                primary_mass_msun=primary_mass_msun,
                secondary_mass_msun=secondary_mass_msun,
                initial_radius_cm=gas_outer_cm,
                initial_time_myr=time_at_outer_myr,
                initial_redshift=initial_redshift,
                duty_cycle_myr=duty_cycle_myr,
                disk_inner_radius_cm=disk_inner_radius_cm,
                gas_radius_cm=radius_grid,
                gas_drift_cm_s=gas_drift,
                min_observed_frequency_hz=min_observed_frequency_hz,
                redshift_model=redshift_model,
                stop_time_myr=stop_time_myr,
                solver_method=solver_method,
            )

    gas_active = (
        initial_time_myr < duty_cycle_myr
        and gas_inner_cm <= initial_radius_cm <= gas_outer_cm
    )
    current_time_myr = float(initial_time_myr)
    current_radius_cm = float(initial_radius_cm)
    disk_off_reason = (
        "agn_inactive_at_embedding"
        if initial_time_myr >= duty_cycle_myr
        else "inside_gas_inner_edge"
        if initial_radius_cm < gas_inner_cm
        else "outside_gas_outer_edge"
    )
    if gas_active:
        gas_deadline_myr = min(duty_cycle_myr, event_limit_myr)
        net_initial_cm_s = _coupled_radial_drift_cm_s(
            current_radius_cm, beta_cm4_s, radius_grid, gas_drift
        )
        local_gw_speed_cm_s = beta_cm4_s / current_radius_cm**3
        if abs(net_initial_cm_s) <= 1.0e-14 * local_gw_speed_cm_s:
            if observed_frequency_hz(current_radius_cm, gas_deadline_myr) >= (
                min_observed_frequency_hz
            ):
                entry_time_myr = brentq(
                    lambda candidate_myr: (
                        observed_frequency_hz(current_radius_cm, candidate_myr)
                        / min_observed_frequency_hz
                        - 1.0
                    ),
                    current_time_myr,
                    gas_deadline_myr,
                    xtol=1.0e-15,
                )
                return outcome(
                    "band_entry",
                    entry_time_myr,
                    current_radius_cm,
                    np.nan,
                    np.nan,
                    True,
                    "gas_active_at_entry",
                )
            current_time_myr = gas_deadline_myr
            if stop_time_myr is not None and current_time_myr >= stop_time_myr:
                return outcome(
                    "stopped_at_event", current_time_myr, current_radius_cm,
                    np.nan, np.nan, disk_off_reason="event_during_gas",
                )
            if current_time_myr >= cosmic_limit_myr:
                return outcome(
                    "cosmic_time_limit",
                    current_time_myr,
                    current_radius_cm,
                    np.nan,
                    np.nan,
                    disk_off_reason="gas_active_at_cosmic_limit",
                )
            disk_off_reason = "agn_duty_end"
            gas_active = False

    if gas_active:
        gas_end_time_myr = min(duty_cycle_myr, event_limit_myr)
        safe_log_min = np.log(0.5 * min(isco_radius_cm, gas_inner_cm))
        safe_log_max = np.log(2.0 * max(initial_radius_cm, gas_outer_cm))

        def stage_radius_cm(state: np.ndarray) -> float:
            # Runge-Kutta trial stages may overshoot a terminal boundary;
            # evaluation there must remain finite until its root is located.
            return float(np.exp(np.clip(state[0], safe_log_min, safe_log_max)))

        def log_radius_rate(time_myr: float, state: np.ndarray) -> list[float]:
            radius_cm = stage_radius_cm(state)
            return [
                _coupled_radial_drift_cm_s(
                    radius_cm, beta_cm4_s, radius_grid, gas_drift
                ) * const.Myr / radius_cm
            ]

        def band_event(time_myr: float, state: np.ndarray) -> float:
            return (
                observed_frequency_hz(stage_radius_cm(state), time_myr)
                / min_observed_frequency_hz
                - 1.0
            )

        def isco_event(_time_myr: float, state: np.ndarray) -> float:
            return state[0] - np.log(isco_radius_cm)

        def inner_edge_event(_time_myr: float, state: np.ndarray) -> float:
            return state[0] - np.log(gas_inner_cm)

        def outer_edge_event(_time_myr: float, state: np.ndarray) -> float:
            return state[0] - np.log(gas_outer_cm)

        band_event.terminal = True
        band_event.direction = 1
        isco_event.terminal = True
        isco_event.direction = -1
        inner_edge_event.terminal = True
        inner_edge_event.direction = -1
        outer_edge_event.terminal = True
        outer_edge_event.direction = 1
        events = [band_event, isco_event, inner_edge_event, outer_edge_event]
        solution = solve_ivp(
            log_radius_rate,
            (current_time_myr, gas_end_time_myr),
            [np.log(current_radius_cm)],
            events=events,
            method=solver_method,
            rtol=ORBIT_RELATIVE_TOLERANCE,
            atol=ORBIT_LOG_RADIUS_ABSOLUTE_TOLERANCE,
            max_step=max((gas_end_time_myr - current_time_myr) / 128.0, 1.0e-12),
        )
        if not solution.success:
            raise ArithmeticError(f"coupled orbit integration failed: {solution.message}")
        current_time_myr = float(solution.t[-1])
        current_radius_cm = float(np.exp(solution.y[0, -1]))
        for index, event_times in enumerate(solution.t_events):
            if event_times.size == 0:
                continue
            if index == 0:
                return outcome(
                    "band_entry",
                    current_time_myr,
                    current_radius_cm,
                    np.nan,
                    np.nan,
                    True,
                    "gas_active_at_entry",
                )
            if index == 1:
                return outcome(
                    "isco_before_entry",
                    current_time_myr,
                    current_radius_cm,
                    np.nan,
                    np.nan,
                    disk_off_reason="gas_active_at_isco",
                )
            disk_off_reason = (
                "inner_profile_edge" if index == 2 else "outer_profile_edge"
            )
            break
        if current_time_myr >= cosmic_limit_myr:
            return outcome(
                "cosmic_time_limit",
                current_time_myr,
                current_radius_cm,
                np.nan,
                np.nan,
                disk_off_reason="gas_active_at_cosmic_limit",
            )
        if stop_time_myr is not None and current_time_myr >= stop_time_myr:
            return outcome(
                "stopped_at_event", current_time_myr, current_radius_cm,
                np.nan, np.nan, disk_off_reason="event_during_gas",
            )
        if not any(event_times.size for event_times in solution.t_events):
            disk_off_reason = "agn_duty_end"

    disk_off_time_myr = current_time_myr
    disk_off_radius_cm = current_radius_cm
    time_to_isco_myr = max(
        0.0,
        (current_radius_cm**4 - isco_radius_cm**4)
        / (4.0 * beta_cm4_s * const.Myr),
    )
    final_time_myr = min(
        event_limit_myr, current_time_myr + time_to_isco_myr
    )

    def vacuum_radius_cm(time_myr: float) -> float:
        fourth_power = max(
            isco_radius_cm**4,
            current_radius_cm**4
            - 4.0 * beta_cm4_s * const.Myr * (time_myr - current_time_myr),
        )
        return float(fourth_power**0.25)

    final_radius_cm = vacuum_radius_cm(final_time_myr)
    if observed_frequency_hz(final_radius_cm, final_time_myr) >= (
        min_observed_frequency_hz
    ):
        entry_time_myr = brentq(
            lambda time_myr: (
                observed_frequency_hz(vacuum_radius_cm(time_myr), time_myr)
                / min_observed_frequency_hz
                - 1.0
            ),
            current_time_myr,
            final_time_myr,
            xtol=1.0e-15,
        )
        return outcome(
            "band_entry",
            entry_time_myr,
            vacuum_radius_cm(entry_time_myr),
            disk_off_time_myr,
            disk_off_radius_cm,
            False,
            disk_off_reason,
        )
    if stop_time_myr is not None and final_time_myr >= stop_time_myr:
        return outcome(
            "stopped_at_event", final_time_myr, final_radius_cm,
            disk_off_time_myr, disk_off_radius_cm,
            disk_off_reason=disk_off_reason,
        )
    final_status = (
        "isco_before_entry"
        if current_time_myr + time_to_isco_myr <= cosmic_limit_myr
        else "cosmic_time_limit"
    )
    return outcome(
        final_status,
        final_time_myr,
        final_radius_cm,
        disk_off_time_myr,
        disk_off_radius_cm,
        disk_off_reason=disk_off_reason,
    )


def continue_circular_band_track(
    *,
    entry: CircularOrbitOutcome,
    primary_mass_msun: float,
    secondary_mass_msun: float,
    initial_redshift: float,
    duty_cycle_myr: float,
    disk_inner_radius_cm: float,
    gas_radius_cm: np.ndarray,
    gas_drift_cm_s: np.ndarray,
    max_observed_frequency_hz: float,
    redshift_model: RedshiftEvolution,
    solver_method: str = "DOP853",
) -> CircularBandTrack:
    """Continue the entry orbit to detector maximum, ISCO, or cosmic horizon.

    This is a continuation of the same signed radial law, not a gas/GW route
    comparison. The returned dense gas solution and exact vacuum branch let
    the detector integrate a mission window without inventing a new radius.
    """
    if solver_method not in ("DOP853", "Radau"):
        raise ValueError("orbit solver_method must be DOP853 or Radau")
    if entry.status not in {"band_entry", "already_inside_band"}:
        raise ValueError("band continuation requires a finite physical entry")
    if not np.isfinite(max_observed_frequency_hz) or max_observed_frequency_hz <= 0:
        raise ValueError("detector maximum frequency must be positive")
    radius_grid = np.asarray(gas_radius_cm, dtype=float)
    gas_drift = np.asarray(gas_drift_cm_s, dtype=float)
    if (
        radius_grid.ndim != 1
        or gas_drift.shape != radius_grid.shape
        or radius_grid.size < 2
        or not np.all(np.isfinite(radius_grid) & (radius_grid > 0.0))
        or not np.all(np.diff(radius_grid) > 0.0)
        or not np.all(np.isfinite(gas_drift))
    ):
        raise ValueError("in-band gas drift requires finite ordered samples")
    total_mass_msun = primary_mass_msun + secondary_mass_msun
    beta_cm4_s = peters_beta_cm4_s(primary_mass_msun, secondary_mass_msun)
    isco_cm = ISCO_RADIUS_RG * const.G * total_mass_msun * const.msun / const.c**2
    cosmic_limit_myr = redshift_model.remaining_time_to_zero_redshift_myr(
        initial_redshift
    )
    time_myr = float(entry.entry_time_myr)
    radius_cm = float(entry.entry_radius_cm)
    gas_end_time_myr = time_myr
    gas_end_radius_cm = radius_cm
    gas_solution: Callable[[float], np.ndarray] | None = None

    def redshift_at(t_myr: float) -> float:
        return redshift_model.redshift_evlove_with_time(initial_redshift, t_myr)

    def observed_frequency(radius_value_cm: float, t_myr: float) -> float:
        return gw_frequency_hz(total_mass_msun, radius_value_cm) / (
            1.0 + redshift_at(t_myr)
        )

    def result(status: str, end_t_myr: float, end_r_cm: float) -> CircularBandTrack:
        return CircularBandTrack(
            status=status,
            entry_time_myr=entry.entry_time_myr,
            entry_radius_cm=entry.entry_radius_cm,
            end_time_myr=end_t_myr,
            end_radius_cm=end_r_cm,
            end_redshift=redshift_at(end_t_myr),
            gas_end_time_myr=gas_end_time_myr,
            gas_end_radius_cm=gas_end_radius_cm,
            beta_cm4_s=beta_cm4_s,
            gas_log_radius_solution=gas_solution,
            initial_redshift=initial_redshift,
            total_mass_msun=total_mass_msun,
            redshift_model=redshift_model,
        )

    if radius_cm <= isco_cm:
        return result("isco_at_entry", time_myr, radius_cm)
    if observed_frequency(radius_cm, time_myr) >= max_observed_frequency_hz:
        return result("detector_max_at_entry", time_myr, radius_cm)

    gas_active = bool(
        entry.gas_active_at_entry
        and time_myr < duty_cycle_myr
        and max(disk_inner_radius_cm, float(radius_grid[0]))
        <= radius_cm <= float(radius_grid[-1])
    )
    if gas_active:
        gas_deadline_myr = min(duty_cycle_myr, cosmic_limit_myr)
        net_initial_cm_s = _coupled_radial_drift_cm_s(
            radius_cm, beta_cm4_s, radius_grid, gas_drift
        )
        local_gw_speed_cm_s = beta_cm4_s / radius_cm**3
        if abs(net_initial_cm_s) <= 1.0e-14 * local_gw_speed_cm_s:
            # Exact net-torque balance is a stationary circular orbit while
            # the disk remains on. Jump over a potentially cosmological
            # plateau without taking millions of near-zero ODE steps.
            fixed_radius_cm = radius_cm
            gas_solution = lambda _time_myr: np.array(
                [np.log(fixed_radius_cm)]
            )
            if observed_frequency(radius_cm, gas_deadline_myr) >= (
                max_observed_frequency_hz
            ):
                time_myr = brentq(
                    lambda candidate_myr: (
                        observed_frequency(radius_cm, candidate_myr)
                        / max_observed_frequency_hz
                        - 1.0
                    ),
                    time_myr,
                    gas_deadline_myr,
                    xtol=1.0e-15,
                )
                gas_end_time_myr = time_myr
                return result("detector_max", time_myr, radius_cm)
            time_myr = gas_deadline_myr
            gas_end_time_myr = time_myr
            if time_myr >= cosmic_limit_myr:
                return result("cosmic_time_limit", time_myr, radius_cm)
            gas_active = False

    if gas_active:
        gas_deadline_myr = min(duty_cycle_myr, cosmic_limit_myr)
        safe_log_min = np.log(0.5 * min(isco_cm, radius_grid[0]))
        safe_log_max = np.log(2.0 * max(radius_cm, radius_grid[-1]))

        def stage_radius(state: np.ndarray) -> float:
            return float(np.exp(np.clip(state[0], safe_log_min, safe_log_max)))

        def log_rate(_t_myr: float, state: np.ndarray) -> list[float]:
            r_cm = stage_radius(state)
            return [
                _coupled_radial_drift_cm_s(
                    r_cm, beta_cm4_s, radius_grid, gas_drift
                ) * const.Myr / r_cm
            ]

        def maximum_event(t_myr: float, state: np.ndarray) -> float:
            return (
                observed_frequency(stage_radius(state), t_myr)
                / max_observed_frequency_hz
                - 1.0
            )

        def isco_event(_t_myr: float, state: np.ndarray) -> float:
            return state[0] - np.log(isco_cm)

        def inner_event(_t_myr: float, state: np.ndarray) -> float:
            return state[0] - np.log(max(disk_inner_radius_cm, radius_grid[0]))

        def outer_event(_t_myr: float, state: np.ndarray) -> float:
            return state[0] - np.log(radius_grid[-1])

        maximum_event.terminal = True
        maximum_event.direction = 1
        isco_event.terminal = True
        isco_event.direction = -1
        inner_event.terminal = True
        inner_event.direction = -1
        outer_event.terminal = True
        outer_event.direction = 1
        solution = solve_ivp(
            log_rate,
            (time_myr, gas_deadline_myr),
            [np.log(radius_cm)],
            events=[maximum_event, isco_event, inner_event, outer_event],
            dense_output=True,
            method=solver_method,
            rtol=ORBIT_RELATIVE_TOLERANCE,
            atol=ORBIT_LOG_RADIUS_ABSOLUTE_TOLERANCE,
            max_step=max((gas_deadline_myr - time_myr) / 128.0, 1.0e-12),
        )
        if not solution.success or solution.sol is None:
            raise ArithmeticError("coupled in-band orbit integration failed")
        gas_solution = solution.sol
        time_myr = float(solution.t[-1])
        radius_cm = float(np.exp(solution.y[0, -1]))
        gas_end_time_myr = time_myr
        gas_end_radius_cm = radius_cm
        if solution.t_events[0].size:
            return result("detector_max", time_myr, radius_cm)
        if solution.t_events[1].size:
            return result("isco", time_myr, radius_cm)
        if time_myr >= cosmic_limit_myr:
            return result("cosmic_time_limit", time_myr, radius_cm)

    fourth_power_rate = 4.0 * beta_cm4_s * const.Myr
    to_isco_myr = max(0.0, (radius_cm**4 - isco_cm**4) / fourth_power_rate)
    final_time_myr = min(cosmic_limit_myr, time_myr + to_isco_myr)

    def vacuum_radius(t_myr: float) -> float:
        fourth_power = radius_cm**4 - fourth_power_rate * (t_myr - time_myr)
        return float(max(isco_cm**4, fourth_power) ** 0.25)

    final_radius_cm = vacuum_radius(final_time_myr)
    if observed_frequency(final_radius_cm, final_time_myr) >= (
        max_observed_frequency_hz
    ):
        end_time_myr = brentq(
            lambda t_myr: (
                observed_frequency(vacuum_radius(t_myr), t_myr)
                / max_observed_frequency_hz - 1.0
            ),
            time_myr,
            final_time_myr,
            xtol=1.0e-15,
        )
        return result("detector_max", end_time_myr, vacuum_radius(end_time_myr))
    return result(
        "isco" if time_myr + to_isco_myr <= cosmic_limit_myr else "cosmic_time_limit",
        final_time_myr,
        final_radius_cm,
    )


def continue_to_isco(
    *,
    entry: CircularOrbitOutcome,
    primary_mass_msun: float,
    secondary_mass_msun: float,
    initial_redshift: float,
    duty_cycle_myr: float,
    disk_inner_radius_cm: float,
    gas_radius_cm: np.ndarray,
    gas_drift_cm_s: np.ndarray,
    redshift_model: RedshiftEvolution,
    solver_method: str = "DOP853",
) -> CircularBandTrack:
    """Continue the same coupled orbit through the detector band to ISCO."""
    total_mass_msun = primary_mass_msun + secondary_mass_msun
    isco_radius_cm = (
        ISCO_RADIUS_RG * const.G * total_mass_msun * const.msun / const.c**2
    )
    unreachable_observed_max_hz = (
        ISCO_CONSUMPTION_FREQUENCY_MARGIN
        * gw_frequency_hz(total_mass_msun, isco_radius_cm)
    )
    return continue_circular_band_track(
        entry=entry,
        primary_mass_msun=primary_mass_msun,
        secondary_mass_msun=secondary_mass_msun,
        initial_redshift=initial_redshift,
        duty_cycle_myr=duty_cycle_myr,
        disk_inner_radius_cm=disk_inner_radius_cm,
        gas_radius_cm=gas_radius_cm,
        gas_drift_cm_s=gas_drift_cm_s,
        max_observed_frequency_hz=unreachable_observed_max_hz,
        redshift_model=redshift_model,
        solver_method=solver_method,
    )
