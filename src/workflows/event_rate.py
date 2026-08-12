"""Event-rate classification for gas-assisted IMRI formation."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

BH_ORIGIN_EMBEDDED = "ebh"
BH_ORIGIN_MIGRATION_FORMED = "mfbh"
BH_ORIGIN_NOT_FORMED = "no_bh"

EVENT_RATE_IMRI_CANDIDATE = "imri candi"
EVENT_RATE_NO_IMRI = "no-IMRI"


@dataclass(frozen=True)
class EventRateDecision:
    """Classification result for the event-rate stage."""

    bh_origin: str
    bh_candidate: bool
    is_imri_candidate: bool
    bh_embedding_time_myr: float
    gas_elapsed_time_myr: float
    gas_final_time_myr: float
    migration_time_to_entry_myr: float
    gw_time_to_entry_myr: float
    r_gas_fin_cm: float
    event_rate_class: str
    event_rate_reason: str


MigrationTimeFunction = Callable[[float, float], float]
GwTimeFunction = Callable[[float], float]


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
) -> float:
    try:
        value = float(gw_time_to_entry_myr(radius_cm))
    except (ArithmeticError, ValueError, OverflowError):
        return np.inf
    return value if np.isfinite(value) and value >= 0.0 else np.inf


def solve_gas_final_radius_cm(
    *,
    r_ini_cm: float,
    r_entry_cm: float,
    available_gas_time_myr: float,
    migration_time_myr_call: MigrationTimeFunction,
    gw_time_to_entry_myr: GwTimeFunction,
    num_radius_samples: int = 96,
) -> tuple[float, float, float, float, str]:
    """Return the final gas-delivered radius before GW or duty-cycle stopping.

    The gas stage moves inward from ``r_ini_cm`` until one of three conditions is reached:
    1. LISA-band entry;
    2. exhaustion of the AGN duty-cycle time;
    3. available after damping, or the point where GW-only evolution to entry is faster than continued gas migration.
    """
    if (
        not np.isfinite(r_ini_cm)
        or not np.isfinite(r_entry_cm)
        or r_ini_cm <= 0.0
        or r_entry_cm <= 0.0
    ):
        return np.nan, np.inf, np.inf, np.inf, "invalid_radius"

    total_migration_time_myr = _safe_migration_time_myr(
        migration_time_myr_call,
        r_ini_cm,
        r_entry_cm,
    )
    print(f"total_migration_time is {total_migration_time_myr:.3e} myr")

    total_gw_time_myr = _safe_gw_time_myr(gw_time_to_entry_myr, r_ini_cm)

    print(f"total_gw_time is {total_gw_time_myr:.3e} myr")

    if r_ini_cm <= r_entry_cm:
        return r_ini_cm, 0.0, 0.0, 0.0, "already_inside_lisa_entry"

    if not np.isfinite(available_gas_time_myr) or available_gas_time_myr <= 0.0:
        return (
            r_ini_cm,
            0.0,
            total_migration_time_myr,
            total_gw_time_myr,
            "damping_misses_duty_cycle",
        )

    if not np.isfinite(total_migration_time_myr):
        return (
            r_ini_cm,
            0.0,
            total_migration_time_myr,
            total_gw_time_myr,
            "migration_trap_bound",
        )

    # gw dominates at start
    if total_migration_time_myr >= total_gw_time_myr:
        return (
            r_ini_cm,
            0.0,
            total_migration_time_myr,
            total_gw_time_myr,
            "gw_dominates_at_start",
        )

    # migration towards the r_entry
    if total_migration_time_myr <= available_gas_time_myr:
        return (
            r_entry_cm,
            total_migration_time_myr,
            total_migration_time_myr,
            total_gw_time_myr,
            "entry_reached",
        )

    # total_migration_time_myr between available_gas_time_myr and total_gw_time_myr, the most common case
    # calculate details
    sample_count = int(num_radius_samples)
    radius_samples_cm = np.geomspace(r_ini_cm, r_entry_cm, sample_count)

    # initilization parameters
    r_gas_fin_cm = r_ini_cm
    gas_elapsed_time_myr = 0.0
    stop_reason = "duty_cycle_limited"

    for radius_sample_cm in radius_samples_cm[1:]:
        evloved_embed_myr = _safe_migration_time_myr(
            migration_time_myr_call,
            r_ini_cm,
            float(radius_sample_cm),
        )
        remaining_migration_myr = _safe_migration_time_myr(
            migration_time_myr_call,
            float(radius_sample_cm),
            r_entry_cm,
        )
        remaining_gw_myr = _safe_gw_time_myr(
            gw_time_to_entry_myr, float(radius_sample_cm)
        )

        if not np.isfinite(evloved_embed_myr):
            stop_reason = "migration_trap_bound"
            break

        if evloved_embed_myr > available_gas_time_myr:
            stop_reason = "duty_cycle_limited"
            break

        r_gas_fin_cm = float(radius_sample_cm)
        gas_elapsed_time_myr = evloved_embed_myr

        if not np.isfinite(remaining_migration_myr):
            stop_reason = "migration_trap_bound"
            break

        if remaining_migration_myr >= remaining_gw_myr:
            stop_reason = "gw_dominates_after_gas"
            break

        if radius_sample_cm <= r_entry_cm:
            stop_reason = "entry_reached"
            break

    return (
        r_gas_fin_cm,
        gas_elapsed_time_myr,
        total_migration_time_myr,
        total_gw_time_myr,
        stop_reason,
    )


def classify_event_rate(
    *,
    object_type: int,
    formation_time_myr: float,
    damping_time_myr: float,
    embed_time_myr: float,
    r_ini_cm: float,
    r_entry_cm: float,
    migration_time_myr_call: MigrationTimeFunction,
    gw_time_to_entry_myr_call: GwTimeFunction,
    duty_cycle_myr: float,
    num_radius_samples: int = 96,
) -> EventRateDecision:
    """Classify whether gas migration produces an IMRI event candidate."""

    available_gas_time_myr = duty_cycle_myr - embed_time_myr

    (
        r_gas_fin_cm,
        gas_elapsed_time_myr,
        migration_time_to_entry_myr,
        gw_time_to_entry_myr_at_start,
        gas_stop_reason,
    ) = solve_gas_final_radius_cm(
        r_ini_cm=r_ini_cm,
        r_entry_cm=r_entry_cm,
        available_gas_time_myr=available_gas_time_myr,
        migration_time_myr_call=migration_time_myr_call,
        gw_time_to_entry_myr=gw_time_to_entry_myr_call,
        num_radius_samples=num_radius_samples,
    )

    print(f"stop reason is {gas_stop_reason}, r_gas_fin is {r_gas_fin_cm:.3e} cm")

    gas_final_time_myr = embed_time_myr + gas_elapsed_time_myr
    embedded_bh = object_type == 1

    migration_formed_bh = bool(
        object_type == 0
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
            gw_time_to_entry_myr=gw_time_to_entry_myr_at_start,
            r_gas_fin_cm=r_gas_fin_cm,
            event_rate_class=EVENT_RATE_NO_IMRI,
            event_rate_reason="not_bh_before_gas_final",
        )

    candidate_reasons = {
        "already_inside_lisa_entry",
        "entry_reached",
        "gw_dominates_at_start",
        "gw_dominates_after_gas",
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
        gw_time_to_entry_myr=gw_time_to_entry_myr_at_start,
        r_gas_fin_cm=r_gas_fin_cm,
        event_rate_class=(
            EVENT_RATE_IMRI_CANDIDATE if is_imri_candidate else EVENT_RATE_NO_IMRI
        ),
        event_rate_reason=gas_stop_reason,
    )
