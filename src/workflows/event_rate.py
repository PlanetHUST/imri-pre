"""Event-rate classification for gas-assisted IMRI formation."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from physical_process.damp import DampingStatus

BH_ORIGIN_EMBEDDED = "ebh"
BH_ORIGIN_MIGRATION_FORMED = "mfbh"
BH_ORIGIN_NOT_FORMED = "no_bh"

EVENT_RATE_IMRI_CANDIDATE = "imri candi"
EVENT_RATE_NO_IMRI = "no-IMRI"
EVENT_RATE_NOT_EVALUATED = "not evaluated"
GAS_FINAL_RADIUS_SAMPLES = 96


@dataclass(frozen=True)
class EventRateDecision:
    """Classification result for the event-rate stage."""

    bh_origin: str
    bh_candidate: bool
    is_imri_candidate: bool | None
    bh_embedding_time_myr: float
    gas_elapsed_time_myr: float
    gas_final_time_myr: float
    migration_time_to_entry_myr: float
    migration_time_to_gas_target_myr: float
    gw_time_to_entry_myr: float
    gw_time_gas_target_to_entry_myr: float
    r_gas_fin_cm: float
    gas_stop_reason: str
    event_rate_class: str
    event_rate_reason: str


MigrationTimeFunction = Callable[[float, float], float]
GwTimeFunction = Callable[[float, float], float]


def _safe_migration_time_myr(
    migration_time_myr: MigrationTimeFunction,
    start_radius_cm: float,
    target_radius_cm: float,
) -> float:
    value = float(migration_time_myr(start_radius_cm, target_radius_cm))
    return value if np.isfinite(value) and value >= 0.0 else np.inf


def _safe_gw_time_myr(
    gw_time_to_entry_myr: GwTimeFunction,
    radius_cm: float,
    absolute_time_myr: float,
) -> float:
    try:
        value = float(gw_time_to_entry_myr(radius_cm, absolute_time_myr))
    except (ArithmeticError, ValueError, OverflowError):
        return np.inf
    return value if np.isfinite(value) and value >= 0.0 else np.inf


def _refine_first_true_radius_cm(
    *,
    outer_radius_cm: float,
    inner_radius_cm: float,
    condition_at_radius: Callable[[float], bool],
) -> float:
    """Refine a bracketed inward transition and return its first true radius."""

    outer_cm = float(outer_radius_cm)
    inner_cm = float(inner_radius_cm)
    for _ in range(48):
        midpoint_cm = float(np.sqrt(outer_cm * inner_cm))
        if condition_at_radius(midpoint_cm):
            inner_cm = midpoint_cm
        else:
            outer_cm = midpoint_cm
    return inner_cm


def solve_gas_final_radius_cm(
    *,
    r_ini_cm: float,
    r_gas_target_cm: float,
    r_entry_cm: float,
    handoff_required: bool,
    gas_start_time_myr: float,
    available_gas_time_myr: float,
    migration_time_myr_call: MigrationTimeFunction,
    gw_time_to_entry_myr: GwTimeFunction,
    num_radius_samples: int,
) -> tuple[float, float, float, float, float, str]:
    """Return the final gas-delivered radius before GW or duty-cycle stopping.

    The gas stage moves inward from ``r_ini_cm`` until one of three conditions is reached:
    1. its in-domain target, either LISA entry or a gas-to-GW handoff;
    2. exhaustion of the AGN duty-cycle time;
    3. available after damping, or the point where GW-only evolution to entry is faster than continued gas migration.
    """
    if (
        not np.isfinite(r_ini_cm)
        or not np.isfinite(r_gas_target_cm)
        or not np.isfinite(r_entry_cm)
        or r_ini_cm <= 0.0
        or r_gas_target_cm <= 0.0
        or r_entry_cm <= 0.0
    ):
        return np.nan, np.inf, np.inf, np.inf, np.inf, "invalid_radius"

    total_migration_time_myr = _safe_migration_time_myr(
        migration_time_myr_call,
        r_ini_cm,
        r_gas_target_cm,
    )
    total_gw_time_myr = _safe_gw_time_myr(
        gw_time_to_entry_myr,
        r_ini_cm,
        gas_start_time_myr,
    )
    gas_target_time_myr = gas_start_time_myr + (
        total_migration_time_myr
        if np.isfinite(total_migration_time_myr)
        else 0.0
    )
    gw_time_gas_target_to_entry_myr = _safe_gw_time_myr(
        gw_time_to_entry_myr,
        r_gas_target_cm,
        gas_target_time_myr,
    )
    target_reached_reason = "gas_to_gw_handoff" if handoff_required else "entry_reached"

    if r_ini_cm <= r_gas_target_cm:
        stop_reason = (
            "already_inside_lisa_entry"
            if r_ini_cm <= r_entry_cm
            else target_reached_reason
        )
        return (
            r_ini_cm,
            0.0,
            0.0,
            total_gw_time_myr,
            gw_time_gas_target_to_entry_myr,
            stop_reason,
        )

    if not np.isfinite(available_gas_time_myr) or available_gas_time_myr <= 0.0:
        return (
            r_ini_cm,
            0.0,
            total_migration_time_myr,
            total_gw_time_myr,
            gw_time_gas_target_to_entry_myr,
            "damping_misses_duty_cycle",
        )

    if not np.isfinite(total_migration_time_myr):
        return (
            r_ini_cm,
            0.0,
            total_migration_time_myr,
            total_gw_time_myr,
            gw_time_gas_target_to_entry_myr,
            "migration_trap_bound",
        )

    gas_assisted_time_to_entry_myr = (
        total_migration_time_myr + gw_time_gas_target_to_entry_myr
    )
    if (
        np.isfinite(total_gw_time_myr)
        and gas_assisted_time_to_entry_myr >= total_gw_time_myr
    ):
        return (
            r_ini_cm,
            0.0,
            total_migration_time_myr,
            total_gw_time_myr,
            gw_time_gas_target_to_entry_myr,
            "gw_dominates_at_start",
        )

    sample_count = max(2, int(num_radius_samples))
    radius_samples_cm = np.geomspace(r_ini_cm, r_gas_target_cm, sample_count)

    def gas_is_not_faster(radius_cm: float) -> bool:
        gas_elapsed_time_myr = _safe_migration_time_myr(
            migration_time_myr_call,
            r_ini_cm,
            radius_cm,
        )
        remaining_migration_time_myr = _safe_migration_time_myr(
            migration_time_myr_call,
            radius_cm,
            r_gas_target_cm,
        )
        absolute_time_myr = gas_start_time_myr + gas_elapsed_time_myr
        direct_gw_time_myr = _safe_gw_time_myr(
            gw_time_to_entry_myr,
            radius_cm,
            absolute_time_myr,
        )
        target_gw_time_myr = _safe_gw_time_myr(
            gw_time_to_entry_myr,
            r_gas_target_cm,
            absolute_time_myr + remaining_migration_time_myr,
        )
        return bool(
            np.isfinite(direct_gw_time_myr)
            and remaining_migration_time_myr + target_gw_time_myr
            >= direct_gw_time_myr
        )

    previous_radius_cm = r_ini_cm
    previous_elapsed_time_myr = 0.0

    for sample_index, radius_sample_cm in enumerate(radius_samples_cm[1:], start=1):
        radius_sample_cm = float(radius_sample_cm)
        evolved_gas_time_myr = _safe_migration_time_myr(
            migration_time_myr_call,
            r_ini_cm,
            radius_sample_cm,
        )
        if not np.isfinite(evolved_gas_time_myr):
            return (
                previous_radius_cm,
                previous_elapsed_time_myr,
                total_migration_time_myr,
                total_gw_time_myr,
                gw_time_gas_target_to_entry_myr,
                "migration_trap_bound",
            )

        target_reached = sample_index == sample_count - 1
        target_reachable = total_migration_time_myr <= available_gas_time_myr

        if target_reached and target_reachable:
            return (
                r_gas_target_cm,
                total_migration_time_myr,
                total_migration_time_myr,
                total_gw_time_myr,
                gw_time_gas_target_to_entry_myr,
                target_reached_reason,
            )

        if evolved_gas_time_myr >= available_gas_time_myr:
            duty_radius_cm = _refine_first_true_radius_cm(
                outer_radius_cm=previous_radius_cm,
                inner_radius_cm=radius_sample_cm,
                condition_at_radius=lambda radius_cm: _safe_migration_time_myr(
                    migration_time_myr_call,
                    r_ini_cm,
                    radius_cm,
                )
                >= available_gas_time_myr,
            )
            if gas_is_not_faster(duty_radius_cm):
                crossover_radius_cm = _refine_first_true_radius_cm(
                    outer_radius_cm=previous_radius_cm,
                    inner_radius_cm=duty_radius_cm,
                    condition_at_radius=gas_is_not_faster,
                )
                crossover_elapsed_time_myr = _safe_migration_time_myr(
                    migration_time_myr_call,
                    r_ini_cm,
                    crossover_radius_cm,
                )
                return (
                    crossover_radius_cm,
                    crossover_elapsed_time_myr,
                    total_migration_time_myr,
                    total_gw_time_myr,
                    gw_time_gas_target_to_entry_myr,
                    "gw_dominates_after_gas",
                )
            return (
                duty_radius_cm,
                available_gas_time_myr,
                total_migration_time_myr,
                total_gw_time_myr,
                gw_time_gas_target_to_entry_myr,
                "duty_cycle_limited",
            )

        if gas_is_not_faster(radius_sample_cm):
            crossover_radius_cm = _refine_first_true_radius_cm(
                outer_radius_cm=previous_radius_cm,
                inner_radius_cm=radius_sample_cm,
                condition_at_radius=gas_is_not_faster,
            )
            crossover_elapsed_time_myr = _safe_migration_time_myr(
                migration_time_myr_call,
                r_ini_cm,
                crossover_radius_cm,
            )
            return (
                crossover_radius_cm,
                crossover_elapsed_time_myr,
                total_migration_time_myr,
                total_gw_time_myr,
                gw_time_gas_target_to_entry_myr,
                "gw_dominates_after_gas",
            )

        previous_radius_cm = radius_sample_cm
        previous_elapsed_time_myr = evolved_gas_time_myr

    raise RuntimeError("gas-delivery radius search did not reach a terminal state")


def classify_event_rate(
    *,
    object_status: DampingStatus,
    formation_time_myr: float,
    damping_time_myr: float,
    embed_time_myr: float,
    r_ini_cm: float,
    r_gas_target_cm: float,
    r_entry_cm: float,
    handoff_required: bool,
    migration_time_myr_call: MigrationTimeFunction,
    gw_time_to_entry_myr_call: GwTimeFunction,
    duty_cycle_myr: float,
    num_radius_samples: int,
) -> EventRateDecision:
    """Classify whether gas migration produces an IMRI event candidate."""

    available_gas_time_myr = duty_cycle_myr - embed_time_myr

    (
        r_gas_fin_cm,
        gas_elapsed_time_myr,
        migration_time_to_entry_myr,
        gw_time_to_entry_myr_at_start,
        gw_time_gas_target_to_entry_myr,
        gas_stop_reason,
    ) = solve_gas_final_radius_cm(
        r_ini_cm=r_ini_cm,
        r_gas_target_cm=r_gas_target_cm,
        r_entry_cm=r_entry_cm,
        handoff_required=handoff_required,
        gas_start_time_myr=embed_time_myr,
        available_gas_time_myr=available_gas_time_myr,
        migration_time_myr_call=migration_time_myr_call,
        gw_time_to_entry_myr=gw_time_to_entry_myr_call,
        num_radius_samples=num_radius_samples,
    )

    gas_final_time_myr = embed_time_myr + gas_elapsed_time_myr
    migration_time_to_gas_target_myr = migration_time_to_entry_myr
    if handoff_required:
        migration_time_to_entry_myr = np.nan
    embedded_bh = object_status is DampingStatus.EMBEDDED_BH

    migration_formed_bh = bool(
        object_status is DampingStatus.EMBEDDED_STAR
        and np.isfinite(formation_time_myr)
        and np.isfinite(damping_time_myr)
        and formation_time_myr < damping_time_myr + gas_elapsed_time_myr
    )

    bh_candidate = bool(embedded_bh or migration_formed_bh)

    if embedded_bh:
        bh_origin = BH_ORIGIN_EMBEDDED
        bh_embedding_time_myr = embed_time_myr
    elif migration_formed_bh:
        bh_origin = BH_ORIGIN_MIGRATION_FORMED
        bh_embedding_time_myr = formation_time_myr
    else:
        bh_origin = BH_ORIGIN_NOT_FORMED
        bh_embedding_time_myr = np.nan

    if not bh_candidate:
        return EventRateDecision(
            bh_origin=bh_origin,
            bh_candidate=False,
            is_imri_candidate=False,
            bh_embedding_time_myr=bh_embedding_time_myr,
            gas_elapsed_time_myr=gas_elapsed_time_myr,
            gas_final_time_myr=gas_final_time_myr,
            migration_time_to_entry_myr=migration_time_to_entry_myr,
            migration_time_to_gas_target_myr=migration_time_to_gas_target_myr,
            gw_time_to_entry_myr=gw_time_to_entry_myr_at_start,
            gw_time_gas_target_to_entry_myr=gw_time_gas_target_to_entry_myr,
            r_gas_fin_cm=r_gas_fin_cm,
            gas_stop_reason=gas_stop_reason,
            event_rate_class=EVENT_RATE_NO_IMRI,
            event_rate_reason="not_bh_before_gas_final",
        )

    candidate_reasons = {
        "already_inside_lisa_entry",
        "entry_reached",
        "gw_dominates_at_start",
        "gw_dominates_after_gas",
        "gas_to_gw_handoff",
        "duty_cycle_limited",
    }
    is_imri_candidate = gas_stop_reason in candidate_reasons

    return EventRateDecision(
        bh_origin=bh_origin,
        bh_candidate=True,
        is_imri_candidate=is_imri_candidate,
        bh_embedding_time_myr=bh_embedding_time_myr,
        gas_elapsed_time_myr=gas_elapsed_time_myr,
        gas_final_time_myr=gas_final_time_myr,
        migration_time_to_entry_myr=migration_time_to_entry_myr,
        migration_time_to_gas_target_myr=migration_time_to_gas_target_myr,
        gw_time_to_entry_myr=gw_time_to_entry_myr_at_start,
        gw_time_gas_target_to_entry_myr=gw_time_gas_target_to_entry_myr,
        r_gas_fin_cm=r_gas_fin_cm,
        gas_stop_reason=gas_stop_reason,
        event_rate_class=(
            EVENT_RATE_IMRI_CANDIDATE if is_imri_candidate else EVENT_RATE_NO_IMRI
        ),
        event_rate_reason=gas_stop_reason,
    )


def classify_unsupported_band_entry(
    *,
    object_status: DampingStatus,
    embed_time_myr: float,
    r_ini_cm: float,
) -> EventRateDecision:
    """Retain formation bookkeeping when circular band entry is inside ISCO."""

    embedded_bh = object_status is DampingStatus.EMBEDDED_BH
    return EventRateDecision(
        bh_origin=BH_ORIGIN_EMBEDDED if embedded_bh else BH_ORIGIN_NOT_FORMED,
        bh_candidate=embedded_bh,
        is_imri_candidate=None,
        bh_embedding_time_myr=embed_time_myr if embedded_bh else np.nan,
        gas_elapsed_time_myr=0.0,
        gas_final_time_myr=embed_time_myr,
        migration_time_to_entry_myr=np.nan,
        migration_time_to_gas_target_myr=np.nan,
        gw_time_to_entry_myr=np.nan,
        gw_time_gas_target_to_entry_myr=np.nan,
        r_gas_fin_cm=r_ini_cm,
        gas_stop_reason="band_entry_inside_isco",
        event_rate_class=EVENT_RATE_NOT_EVALUATED,
        event_rate_reason="band_entry_inside_isco",
    )
